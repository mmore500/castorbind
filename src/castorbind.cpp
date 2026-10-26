#include "castorbind.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <complex>
#include <variant>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// R's standalone math library (vendor/r-source/src/nmath)
extern "C" {
double exp_rand(void);
double rbinom(double n, double p);
}

// numpy's C interface to its bit generators (numpy/random/bitgen.h)
struct bitgen_t {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
  uint64_t (*next_raw)(void *st);
};

namespace {
bitgen_t *mt19937 = nullptr; // set by init_module
} // namespace

// R's unif_rand() for Mersenne-Twister (src/main/RNG.c), on numpy's
// MT19937, which is the same generator; nmath draws everything through it
extern "C" double unif_rand(void) {
  constexpr double i2_32m1 = 2.328306437080797e-10; // 1/(2^32 - 1)
  const double x = mt19937->next_uint32(mt19937->state) * 2.3283064365386963e-10;
  // as R's fixup(): never return 0 or 1
  if (x <= 0.0) return 0.5 * i2_32m1;
  if (1.0 - x <= 0.0) return 1.0 - 0.5 * i2_32m1;
  return x;
}

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

// castor result -> python: numeric and logical vectors and matrices become
// numpy arrays, lists become dicts (or lists, if unnamed)
py::object _to_py(const Value &v) {
  return std::visit(
      [](const auto &x) -> py::object {
        using X = std::decay_t<decltype(x)>;
        using Rcpp::castorbind_detail::is_std_vector;
        if constexpr (std::is_same_v<X, std::vector<bool>>) {
          py::array_t<bool> result(py::ssize_t(x.size()));
          auto view = result.mutable_unchecked<1>();
          for (std::size_t i = 0; i < x.size(); ++i) view(i) = x[i];
          return std::move(result);
        } else if constexpr (is_std_vector<X>::value) {
          if constexpr (std::is_arithmetic_v<typename X::value_type>)
            return py::array_t<typename X::value_type>(x.size(), x.data());
          else
            return py::cast(x); // strings
        } else if constexpr (std::is_same_v<X, Rcpp::castorbind_detail::
                                                   MatrixValue>) {
          // column-major data -> C-contiguous (nrow, ncol) array
          py::array_t<double> result({py::ssize_t(x.nrow), py::ssize_t(x.ncol)});
          auto view = result.mutable_unchecked<2>();
          for (long c = 0; c < x.ncol; ++c)
            for (long r = 0; r < x.nrow; ++r) view(r, c) = x.data[r + c * x.nrow];
          return std::move(result);
        } else if constexpr (std::is_same_v<X,
                                            std::shared_ptr<const Rcpp::List>>) {
          const auto &names = x->names();
          if (std::all_of(names.begin(), names.end(),
                          [](const std::string &n) { return n.empty(); })) {
            py::list result;
            for (std::size_t i = 0; i < x->size(); ++i)
              result.append(_to_py((*x)[i].value));
            return std::move(result);
          }
          py::dict result;
          for (std::size_t i = 0; i < x->size(); ++i) {
            py::object key = names[i].empty() ? py::object(py::int_(i))
                                              : py::object(py::str(names[i]));
            result[key] = _to_py((*x)[i].value);
          }
          return std::move(result);
        } else {
          return py::cast(x); // None, scalars, strings
        }
      },
      v);
}

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

template <class F> std::vector<double> _draw(long n, F &&f) {
  if (n < 0) throw py::value_error("n must be non-negative");
  std::vector<double> result(static_cast<std::size_t>(n));
  const auto lock = lock_castor();
  py::gil_scoped_release release;
  for (auto &x : result) x = f();
  return result;
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

std::unique_lock<std::mutex> lock_castor() {
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
  // never wait while holding the GIL: the holder may need it (e.g., for
  // checkUserInterrupt)
  py::gil_scoped_release release;
  lock.lock();
  return lock;
}

py::object to_py(const Rcpp::RObject &x) { return _to_py(x.value); }

template <> std::vector<double> from_py(py::handle h, const char *name) {
  return _array_to_vector<double>(h, name);
}

template <> std::vector<long> from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  py::array a = _as_array(h, name);
  _check_integral(a, name);
  return _array_to_vector<long long, long>(a, name);
}

template <> Rcpp::LogicalVector from_py(py::handle h, const char *name) {
  return Rcpp::LogicalVector(_array_to_vector<bool>(h, name));
}

template <> Rcpp::ComplexVector from_py(py::handle h, const char *name) {
  std::vector<Rcomplex> result;
  for (const auto &c : _array_to_vector<std::complex<double>>(h, name))
    result.push_back(Rcomplex{c.real(), c.imag()});
  return Rcpp::ComplexVector(result);
}

template <> Rcpp::NumericMatrix from_py(py::handle h, const char *name) {
  return _matrix_from_py<double>(h, name);
}

template <> Rcpp::IntegerMatrix from_py(py::handle h, const char *name) {
  return _matrix_from_py<int>(h, name);
}

template <> Rcpp::List from_py(py::handle h, const char *name) {
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

  // R's RNG state, as numpy's MT19937 (seeded from the OS, where R would
  // use the clock); castorbind/_rng.py implements set.seed() etc. on it
  py::object bit_generator =
      py::module_::import("numpy.random").attr("MT19937")();
  m.attr("_bit_generator") = bit_generator;
  mt19937 = static_cast<bitgen_t *>(
      bit_generator.attr("capsule").cast<py::capsule>().get_pointer());

  m.def(
      "_with_castor_lock",
      [](py::function f) {
        const auto lock = lock_castor();
        return f();
      },
      "Call f() while no castor function is running.");

  // R-identical draws, for testing the RNG against R itself
  m.def(
      "_runif", [](long n) { return _draw(n, unif_rand); }, py::arg("n"),
      "Same as runif(n) in R.");
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
