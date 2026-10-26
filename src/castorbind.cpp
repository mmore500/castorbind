#include "castorbind.h"

#include <algorithm>
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
    array_t<double, array::f_style> fortran(
        {pybind11::ssize_t(m.nrow), pybind11::ssize_t(m.ncol)},
        m.data.data());
    return module_::import("numpy")
        .attr("ascontiguousarray")(fortran)
        .release();
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

[[noreturn]] void _type_error(const char *name, const char *expected) {
  throw py::type_error(std::string("argument '") + name + "': expected " +
                       expected);
}

template <class T> std::vector<T> _to_vector(py::handle h) {
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a) throw py::error_already_set();
  return std::vector<T>(a.data(), a.data() + a.size());
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

bool _is_nested(py::handle h) {
  for (auto item : py::reinterpret_borrow<py::sequence>(h))
    if (!py::isinstance<py::str>(item) && py::isinstance<py::sequence>(item))
      return true;
  return false;
}

// python value -> what R would have passed for it: an atomic vector or
// scalar where numpy sees a homogeneous array, else a list
Value _value_from_py(py::handle h) {
  if (h.is_none()) return std::monostate{};
  if (py::isinstance<py::str>(h)) return h.cast<std::string>();
  if (py::isinstance<py::dict>(h) ||
      ((py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h)) &&
       _is_nested(h)))
    return std::make_shared<const Rcpp::List>(_list_from_py(h));
  py::array a = py::module_::import("numpy").attr("asarray")(h);
  const bool scalar = a.ndim() == 0;
  switch (a.dtype().kind()) {
  case 'b':
    return scalar ? Value(h.cast<bool>()) : Value(_to_vector<bool>(a));
  case 'i':
  case 'u':
    return scalar ? Value(h.cast<long>()) : Value(_to_vector<long>(a));
  case 'f':
    return scalar ? Value(h.cast<double>()) : Value(_to_vector<double>(a));
  case 'c':
    if (scalar) return h.cast<std::complex<double>>();
    return _to_vector<std::complex<double>>(a);
  case 'U': return h.cast<std::vector<std::string>>();
  default:
    throw py::type_error("castorbind: cannot convert " +
                         std::string(py::str(h.get_type())) + " for castor");
  }
}

template <class T>
std::vector<T> _array_to_vector(py::handle h, const char *name) {
  if (h.is_none()) return {};
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a) _type_error(name, "an array-like of numbers");
  return std::vector<T>(a.data(), a.data() + a.size());
}

template <class T>
using _matrix_for = Rcpp::Matrix<std::is_same_v<T, double>
                                     ? Rcpp::castorbind_detail::REALSXP
                                     : Rcpp::castorbind_detail::INTSXP>;

template <class T>
_matrix_for<T> _matrix_from_py(py::handle h, const char *name) {
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a || a.ndim() != 2) _type_error(name, "a 2-D array");
  const long nrow = long(a.shape(0)), ncol = long(a.shape(1));
  _matrix_for<T> result(nrow, ncol);
  auto view = a.template unchecked<2>();
  for (long r = 0; r < nrow; ++r)
    for (long c = 0; c < ncol; ++c) result(r, c) = view(r, c);
  return result;
}

template <class F> py::list _draw(long n, F &&f) {
  if (n < 0) throw py::value_error("n must be non-negative");
  std::vector<double> result;
  {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(castor_mutex());
    for (long i = 0; i < n; ++i) result.push_back(f());
  }
  return py::cast(result);
}

} // namespace castorbind::_impl

namespace Rcpp {

// castorbind::call redirects this to sys.stdout
std::ostream Rcout(nullptr);

void checkUserInterrupt() {
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

long long_from_py(py::handle h, const char *name) {
  if (PyFloat_Check(h.ptr())) {
    const double x = h.cast<double>();
    if (std::trunc(x) != x) _type_error(name, "an integer");
    return long(x);
  }
  try {
    return h.cast<long>();
  } catch (const py::cast_error &) {
    _type_error(name, "an integer");
  }
}

double double_from_py(py::handle h, const char *name) {
  try {
    return h.cast<double>();
  } catch (const py::cast_error &) {
    _type_error(name, "a number");
  }
}

bool bool_from_py(py::handle h, const char *name) {
  try {
    return h.cast<bool>();
  } catch (const py::cast_error &) {
    _type_error(name, "a bool");
  }
}

std::string string_from_py(py::handle h, const char *name) {
  if (!py::isinstance<py::str>(h)) _type_error(name, "a str");
  return h.cast<std::string>();
}

std::vector<double> doubles_from_py(py::handle h, const char *name) {
  return _array_to_vector<double>(h, name);
}

std::vector<long> longs_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  py::array a = py::module_::import("numpy").attr("asarray")(h);
  if (a.dtype().kind() == 'f') {
    // R would silently truncate doubles; insist on integral values instead
    for (double x : _array_to_vector<double>(a, name))
      if (std::trunc(x) != x && !std::isnan(x))
        _type_error(name, "an array-like of integers");
  }
  auto values = _array_to_vector<long long>(a, name);
  return std::vector<long>(values.begin(), values.end());
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

Rcpp::StringVector strings_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  if (py::isinstance<py::str>(h)) return {h.cast<std::string>()};
  try {
    return Rcpp::StringVector(h.cast<std::vector<std::string>>());
  } catch (const py::cast_error &) {
    _type_error(name, "a sequence of str");
  }
}

Rcpp::NumericMatrix numeric_matrix_from_py(py::handle h, const char *name) {
  return _matrix_from_py<double>(h, name);
}

Rcpp::IntegerMatrix integer_matrix_from_py(py::handle h, const char *name) {
  return _matrix_from_py<int>(h, name);
}

Rcpp::List list_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  if (!py::isinstance<py::dict>(h) && !py::isinstance<py::sequence>(h))
    _type_error(name, "a dict or sequence");
  return _list_from_py(h);
}

void init_module(py::module_ &m) {
  py::register_local_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const std::invalid_argument &e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::out_of_range &e) {
      PyErr_SetString(PyExc_KeyError, e.what());
    }
  });

  // R seeds from the clock on first use; use the OS instead
  castorbind_set_seed(int(std::random_device{}()));

  m.def(
      "set_seed",
      [](py::handle seed) {
        const long s = long_from_py(seed, "seed");
        if (s <= INT_MIN || s > INT_MAX)
          throw py::value_error("seed must be a 32-bit signed integer, "
                                "excluding -2**31 (R's NA)");
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(castor_mutex());
        castorbind_set_seed(int(s));
      },
      py::arg("seed"),
      "Seed R's Mersenne-Twister RNG, giving the same stream as "
      "set.seed(seed) in R.");

  m.def(
      "get_rng_state",
      [] {
        std::vector<int> state(CASTORBIND_RNG_STATE_LEN);
        {
          py::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(castor_mutex());
          castorbind_get_rng_state(state.data());
        }
        return state;
      },
      "Return the RNG state as 625 ints, laid out like .Random.seed[-1] "
      "in R.");

  m.def(
      "set_rng_state",
      [](py::handle state_) {
        const std::vector<long> values = longs_from_py(state_, "state");
        if (values.size() != CASTORBIND_RNG_STATE_LEN)
          throw py::value_error("state must have 625 entries");
        std::vector<int> state;
        for (long x : values) {
          if (x < INT_MIN || x > long(UINT_MAX))
            throw py::value_error("state entries must fit in 32 bits");
          state.push_back(int(static_cast<unsigned int>(x)));
        }
        int status;
        {
          py::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(castor_mutex());
          status = castorbind_set_rng_state(state.data());
        }
        if (status != 0) throw py::value_error("invalid RNG state: all zero");
      },
      py::arg("state"),
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
