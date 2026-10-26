#include "castorbind.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <random>
#include <variant>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "r_rng.h"

// R's standalone math library (vendor/r-source/src/nmath)
extern "C" {
double unif_rand(void);
double norm_rand(void);
double exp_rand(void);
double rbinom(double n, double p);
}

// casters for the shim's own result types; with pybind11/stl.h, these let
// py::cast convert a whole Rcpp::castorbind_detail::Value
namespace pybind11::detail {

template <> struct type_caster<Rcpp::castorbind_detail::MatrixValue> {
  PYBIND11_TYPE_CASTER(Rcpp::castorbind_detail::MatrixValue,
                       const_name("numpy.ndarray"));
  bool load(handle, bool) { return false; }
  static handle cast(const Rcpp::castorbind_detail::MatrixValue &m,
                     return_value_policy, handle) {
    // column-major data -> C-contiguous (nrow, ncol) array
    array_t<double> result(
        {pybind11::ssize_t(m.nrow), pybind11::ssize_t(m.ncol)});
    auto view = result.mutable_unchecked<2>();
    for (long c = 0; c < m.ncol; ++c)
      for (long r = 0; r < m.nrow; ++r) view(r, c) = m.data[r + c * m.nrow];
    return result.release();
  }
};

// R list -> dict, or list if it has no names
template <> struct type_caster<std::shared_ptr<const Rcpp::List>> {
  PYBIND11_TYPE_CASTER(std::shared_ptr<const Rcpp::List>,
                       const_name("dict | list"));
  bool load(handle, bool) { return false; }
  static handle cast(const std::shared_ptr<const Rcpp::List> &list,
                     return_value_policy, handle) {
    const auto &names = list->names();
    if (std::all_of(names.begin(), names.end(),
                    [](const std::string &name) { return name.empty(); })) {
      pybind11::list result;
      for (std::size_t i = 0; i < list->size(); ++i)
        result.append(pybind11::cast((*list)[i].value));
      return result.release();
    }
    pybind11::dict result;
    for (std::size_t i = 0; i < list->size(); ++i) {
      object key = names[i].empty() ? object(int_(i)) : object(str(names[i]));
      result[key] = pybind11::cast((*list)[i].value);
    }
    return result.release();
  }
};

} // namespace pybind11::detail

namespace castorbind::_impl {

using Rcpp::castorbind_detail::Value;

// set by init_module
unsigned long _main_thread = 0;

[[noreturn]] void _type_error(const char *name, const char *expected) {
  throw py::type_error(std::string("argument '") + name + "': expected " +
                       expected);
}

template <class T, class E = T>
std::vector<E> _array_to_vector(py::handle h, const char *name) {
  if (h.is_none()) return {};
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a) {
    PyErr_Clear();
    _type_error(name, "an array-like of numbers");
  }
  return std::vector<E>(a.data(), a.data() + a.size());
}

py::array _as_array(py::handle h, const char *name) {
  py::array a = py::array::ensure(h);
  if (!a) {
    PyErr_Clear();
    _type_error(name, "an array-like");
  }
  return a;
}

// R silently truncates doubles passed as integers; reject them instead
void _check_integral(const py::array &a, const char *name) {
  if (a.dtype().kind() != 'f') return;
  for (double x : _array_to_vector<double>(a, name))
    if (!std::isfinite(x) || std::trunc(x) != x || x < -0x1p63 ||
        x >= 0x1p63)
      _type_error(name, "integers");
}

Value _value_from_py(py::handle h);

Rcpp::List _list_from_py(py::handle h) {
  Rcpp::List result;
  if (py::isinstance<py::dict>(h)) {
    for (auto item : py::reinterpret_borrow<py::dict>(h))
      result.push_back(Rcpp::RObject(_value_from_py(item.second)),
                       py::str(item.first).cast<std::string>());
  } else {
    for (auto item : py::reinterpret_borrow<py::sequence>(h))
      result.push_back(Rcpp::RObject(_value_from_py(item)));
  }
  return result;
}

// a list element -> what R would have passed for it. castor's list
// arguments hold numbers, numeric vectors and nested lists, so nothing else
// (e.g., strings or complex numbers) is supported
Value _value_from_py(py::handle h) {
  if (h.is_none()) return std::monostate{};
  if (py::isinstance<py::dict>(h))
    return std::make_shared<const Rcpp::List>(_list_from_py(h));
  if (py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h)) {
    // any non-scalar item (list, dict, numpy array, ...) makes this a list
    for (auto item : py::reinterpret_borrow<py::sequence>(h))
      if (py::isinstance<py::dict>(item) ||
          (py::isinstance<py::sequence>(item) && !py::isinstance<py::str>(item)))
        return std::make_shared<const Rcpp::List>(_list_from_py(h));
  }
  constexpr const char *name = "list element";
  py::array a = _as_array(h, name);
  const bool scalar = a.ndim() == 0;
  switch (a.dtype().kind()) {
  case 'b':
    if (scalar) return h.cast<bool>();
    return _array_to_vector<bool>(a, name);
  case 'i':
  case 'u':
    if (scalar) return h.cast<long>();
    return _array_to_vector<long long, long>(a, name);
  case 'f':
    if (scalar) return h.cast<double>();
    return _array_to_vector<double>(a, name);
  case 'c':
    PyErr_SetString(PyExc_NotImplementedError,
                    "castorbind: complex numbers in list arguments are not "
                    "supported");
    throw py::error_already_set();
  default:
    throw py::type_error("castorbind: cannot pass " +
                         std::string(py::repr(h)).substr(0, 80) +
                         " to castor; list elements must be numbers, numeric "
                         "arrays, or nested lists or dicts");
  }
}

template <class T>
using _matrix_for = Rcpp::Matrix<std::is_same_v<T, double>
                                     ? Rcpp::castorbind_detail::REALSXP
                                     : Rcpp::castorbind_detail::INTSXP>;

template <class T>
_matrix_for<T> _matrix_from_py(py::handle h, const char *name) {
  if constexpr (std::is_integral_v<T>) _check_integral(_as_array(h, name), name);
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a || a.ndim() != 2) {
    PyErr_Clear();
    _type_error(name, "a 2-D array");
  }
  const long nrow = long(a.shape(0)), ncol = long(a.shape(1));
  _matrix_for<T> result(nrow, ncol);
  auto view = a.template unchecked<2>();
  for (long r = 0; r < nrow; ++r)
    for (long c = 0; c < ncol; ++c) result(r, c) = view(r, c);
  return result;
}

// run f() with the castor mutex held and the GIL released
template <class F> auto _locked(F &&f) {
  py::gil_scoped_release release;
  std::lock_guard<std::mutex> lock(castor_mutex());
  return f();
}

template <class F> std::vector<double> _draw(long n, F &&f) {
  if (n < 0) throw py::value_error("n must be non-negative");
  return _locked([&] {
    std::vector<double> result(static_cast<std::size_t>(n));
    for (auto &x : result) x = f();
    return result;
  });
}

} // namespace castorbind::_impl

namespace Rcpp {

// castorbind::call redirects this to sys.stdout
std::ostream Rcout(nullptr);

void checkUserInterrupt() {
  // Python only runs signal handlers on the main thread, and taking the GIL
  // can stall behind other threads, so check rarely and only there
  if (PyThread_get_thread_ident() != castorbind::_impl::_main_thread) return;
  const auto now = std::chrono::steady_clock::now();
  static auto last_check = now;
  if (now - last_check < std::chrono::milliseconds(100)) return;
  last_check = now;
  pybind11::gil_scoped_acquire gil;
  if (PyErr_CheckSignals() != 0) throw castorbind::Interrupted();
}

} // namespace Rcpp

namespace castorbind {

using namespace _impl;

std::mutex &castor_mutex() {
  static std::mutex mutex;
  return mutex;
}

py::object to_py(const Rcpp::RObject &x) { return py::cast(x.value); }

std::vector<double> doubles_from_py(py::handle h, const char *name) {
  return _array_to_vector<double>(h, name);
}

std::vector<long> longs_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  py::array a = _as_array(h, name);
  _check_integral(a, name);
  return _array_to_vector<long long, long>(a, name);
}

Rcpp::LogicalVector logicals_from_py(py::handle h, const char *name) {
  return Rcpp::LogicalVector(_array_to_vector<bool>(h, name));
}

Rcpp::ComplexVector complexes_from_py(py::handle h, const char *name) {
  Rcpp::ComplexVector result;
  for (const auto &c : _array_to_vector<std::complex<double>>(h, name))
    result.push_back(Rcomplex{c.real(), c.imag()});
  return result;
}

Rcpp::NumericMatrix numeric_matrix_from_py(py::handle h, const char *name) {
  return _matrix_from_py<double>(h, name);
}

Rcpp::IntegerMatrix integer_matrix_from_py(py::handle h, const char *name) {
  return _matrix_from_py<int>(h, name);
}

Rcpp::List list_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  if (!py::isinstance<py::dict>(h) && !py::isinstance<py::list>(h) &&
      !py::isinstance<py::tuple>(h))
    _type_error(name, "a dict, list or tuple");
  return _list_from_py(h);
}

void init_module(py::module_ &m) {
  py::register_local_exception<Rcpp::castorbind_detail::missing_name>(
      m, "MissingNameError", PyExc_KeyError);

  _main_thread = py::module_::import("threading")
                     .attr("main_thread")()
                     .attr("ident")
                     .cast<unsigned long>();

  // R seeds from the clock on first use; use the OS instead
  castorbind_set_seed(int(std::random_device{}()));

  m.def(
      "set_seed",
      [](int seed) {
        if (seed == INT_MIN)
          throw py::value_error("seed -2**31 is R's NA, not a valid seed");
        _locked([=] { castorbind_set_seed(seed); });
      },
      py::arg("seed").noconvert(),  // no floats etc.
      "Seed R's Mersenne-Twister RNG, giving the same stream as "
      "set.seed(seed) in R.");

  m.def(
      "get_rng_state",
      [] {
        return _locked([] {
          std::vector<int> state(CASTORBIND_RNG_STATE_LEN);
          castorbind_get_rng_state(state.data());
          return state;
        });
      },
      "Return the RNG state as 625 ints, laid out like .Random.seed[-1] "
      "in R.");

  m.def(
      "set_rng_state",
      [](const std::vector<long long> &values) {
        if (values.size() != CASTORBIND_RNG_STATE_LEN)
          throw py::value_error("state must have 625 entries");
        // state[0] is mti, the position in the Mersenne-Twister state
        if (values[0] < 0 || values[0] > CASTORBIND_RNG_STATE_LEN)
          throw py::value_error("state[0] must be in [0, 625]");
        std::vector<int> state;
        for (long long x : values) {
          if (x < INT_MIN || x > UINT_MAX)
            throw py::value_error("state entries must fit in 32 bits");
          state.push_back(int(static_cast<unsigned int>(x)));
        }
        if (_locked([&] { return castorbind_set_rng_state(state.data()); }))
          throw py::value_error("invalid RNG state: all zero");
      },
      py::arg("state").noconvert(),
      "Restore an RNG state returned by get_rng_state(), or taken from "
      ".Random.seed[-1] in R.");

  // R-identical draws, for testing the RNG against R itself
  m.def(
      "_runif", [](long n) { return _draw(n, unif_rand); }, py::arg("n"),
      "Same as runif(n) in R.");
  m.def(
      "_rnorm", [](long n) { return _draw(n, norm_rand); }, py::arg("n"),
      "Same as rnorm(n) in R.");
  m.def(
      "_rexp", [](long n) { return _draw(n, exp_rand); }, py::arg("n"),
      "Same as rexp(n) in R.");
  m.def(
      "_rbinom",
      [](long n, double size, double prob) {
        return _draw(n, [=] { return rbinom(size, prob); });
      },
      py::arg("n"), py::arg("size"), py::arg("prob"),
      "Same as rbinom(n, size, prob) in R.");
}

} // namespace castorbind
