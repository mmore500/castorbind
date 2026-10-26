#include "castorbind.h"

#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <streambuf>
#include <variant>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "r_rng.h"

// R's standalone math library (vendor/rmath)
extern "C" {
double unif_rand(void);
double norm_rand(void);
double exp_rand(void);
double rbinom(double n, double p);
}

namespace {

namespace py = pybind11;

// line-buffered stream that writes to sys.stdout
class PyStdoutBuf : public std::streambuf {
public:
  void flush_to_python() {
    if (buffer_.empty()) return;
    py::gil_scoped_acquire gil;
    try {
      py::object out = py::module_::import("sys").attr("stdout");
      if (!out.is_none()) out.attr("write")(buffer_);
    } catch (py::error_already_set &e) {
      e.discard_as_unraisable("castorbind: writing castor output");
    }
    buffer_.clear();
  }

protected:
  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      buffer_.push_back(traits_type::to_char_type(c));
      if (c == '\n') flush_to_python();
    }
    return traits_type::not_eof(c);
  }
  std::streamsize xsputn(const char *s, std::streamsize n) override {
    buffer_.append(s, std::size_t(n));
    if (std::memchr(s, '\n', std::size_t(n)) != nullptr) flush_to_python();
    return n;
  }
  int sync() override {
    flush_to_python();
    return 0;
  }

private:
  std::string buffer_;
};

PyStdoutBuf rcout_buf;

[[noreturn]] void type_error(const char *name, const char *expected) {
  throw py::type_error(std::string("argument '") + name + "': expected " +
                       expected);
}

using Rcpp::castorbind_detail::Value;

template <class T> std::vector<T> as_std_vector(const py::array &a) {
  auto c = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(a);
  return std::vector<T>(c.data(), c.data() + c.size());
}

Value value_from_py(py::handle h);

Rcpp::List list_from_dict_or_sequence(py::handle h) {
  Rcpp::List result;
  if (py::isinstance<py::dict>(h)) {
    for (auto item : py::reinterpret_borrow<py::dict>(h))
      result.push_back(Rcpp::RObject(value_from_py(item.second)),
                       py::str(item.first).cast<std::string>());
  } else {
    for (auto item : py::reinterpret_borrow<py::sequence>(h))
      result.push_back(Rcpp::RObject(value_from_py(item)));
  }
  return result;
}

// python value -> what R would have passed for it
Value value_from_py(py::handle h) {
  if (h.is_none()) return std::monostate{};
  if (py::isinstance<py::bool_>(h)) return h.cast<bool>();
  if (py::isinstance<py::str>(h)) return h.cast<std::string>();
  if (py::isinstance<py::dict>(h))
    return std::make_shared<const Rcpp::List>(list_from_dict_or_sequence(h));
  if (py::isinstance<py::array>(h) || py::isinstance<py::list>(h) ||
      py::isinstance<py::tuple>(h)) {
    py::array a = py::module_::import("numpy").attr("asarray")(h);
    switch (a.dtype().kind()) {
    case 'b': return as_std_vector<bool>(a);
    case 'i':
    case 'u': return as_std_vector<long>(a);
    case 'f': return as_std_vector<double>(a);
    case 'c': return as_std_vector<std::complex<double>>(a);
    case 'U': return h.cast<std::vector<std::string>>();
    default:
      return std::make_shared<const Rcpp::List>(list_from_dict_or_sequence(h));
    }
  }
  // numpy scalars and other numbers
  py::object np_value = py::module_::import("numpy").attr("asarray")(h);
  switch (py::array(np_value).dtype().kind()) {
  case 'b': return h.cast<bool>();
  case 'i':
  case 'u': return h.cast<long>();
  case 'f': return h.cast<double>();
  case 'c': return h.cast<std::complex<double>>();
  default:
    throw py::type_error("castorbind: cannot convert " +
                         std::string(py::str(h.get_type())) +
                         " for castor");
  }
}

py::object value_to_py(const Value &v) {
  return std::visit(
      [](const auto &x) -> py::object {
        using X = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<X, std::monostate>) {
          return py::none();
        } else if constexpr (std::is_same_v<X, Rcpp::castorbind_detail::
                                                   MatrixValue>) {
          // column-major data -> (nrow, ncol) array
          py::array_t<double, py::array::f_style> result(
              {py::ssize_t(x.nrow), py::ssize_t(x.ncol)});
          std::copy(x.data.begin(), x.data.end(), result.mutable_data());
          return py::module_::import("numpy").attr("ascontiguousarray")(
              result);
        } else if constexpr (std::is_same_v<X,
                                            std::shared_ptr<const Rcpp::List>>) {
          const Rcpp::List &list = *x;
          const auto &names = list.names();
          bool any_named = false;
          for (const auto &name : names) any_named |= !name.empty();
          if (!any_named) {
            py::list result;
            for (std::size_t i = 0; i < list.size(); ++i)
              result.append(value_to_py(list[i].value));
            return std::move(result);
          }
          py::dict result;
          for (std::size_t i = 0; i < list.size(); ++i) {
            py::object key = names[i].empty() ? py::object(py::int_(i))
                                              : py::object(py::str(names[i]));
            result[key] = value_to_py(list[i].value);
          }
          return std::move(result);
        } else {
          return py::cast(x);
        }
      },
      v);
}

template <class T> std::vector<T> array_to_vector(py::handle h,
                                                  const char *name) {
  if (h.is_none()) return {};
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a) type_error(name, "an array-like of numbers");
  return std::vector<T>(a.data(), a.data() + a.size());
}

template <class T>
using matrix_for = Rcpp::Matrix<std::is_same_v<T, double>
                                    ? Rcpp::castorbind_detail::REALSXP
                                    : Rcpp::castorbind_detail::INTSXP>;

template <class T> matrix_for<T> matrix_from_py(py::handle h,
                                                const char *name) {
  auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
  if (!a || a.ndim() != 2) type_error(name, "a 2-D array");
  const long nrow = long(a.shape(0)), ncol = long(a.shape(1));
  matrix_for<T> result(nrow, ncol);
  auto view = a.template unchecked<2>();
  for (long r = 0; r < nrow; ++r)
    for (long c = 0; c < ncol; ++c) result(r, c) = view(r, c);
  return result;
}

} // namespace

namespace Rcpp {

std::ostream Rcout(&rcout_buf);

void checkUserInterrupt() {
  pybind11::gil_scoped_acquire gil;
  if (PyErr_CheckSignals() != 0) throw castorbind::Interrupted();
}

} // namespace Rcpp

namespace castorbind {

std::mutex &castor_mutex() {
  static std::mutex mutex;
  return mutex;
}

void flush_rcout() {
  Rcpp::Rcout.flush();
  rcout_buf.flush_to_python();
}

py::object to_py(const Rcpp::RObject &x) { return value_to_py(x.value); }

long long_from_py(py::handle h, const char *name) {
  if (PyFloat_Check(h.ptr())) {
    const double x = h.cast<double>();
    if (std::trunc(x) != x) type_error(name, "an integer");
    return long(x);
  }
  try {
    return h.cast<long>();
  } catch (const py::cast_error &) {
    type_error(name, "an integer");
  }
}

double double_from_py(py::handle h, const char *name) {
  try {
    return h.cast<double>();
  } catch (const py::cast_error &) {
    type_error(name, "a number");
  }
}

bool bool_from_py(py::handle h, const char *name) {
  try {
    return h.cast<bool>();
  } catch (const py::cast_error &) {
    type_error(name, "a bool");
  }
}

std::string string_from_py(py::handle h, const char *name) {
  if (!py::isinstance<py::str>(h)) type_error(name, "a str");
  return h.cast<std::string>();
}

std::vector<double> doubles_from_py(py::handle h, const char *name) {
  return array_to_vector<double>(h, name);
}

std::vector<long> longs_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  py::array a = py::module_::import("numpy").attr("asarray")(h);
  if (a.dtype().kind() == 'f') {
    // R would silently truncate doubles; insist on integral values instead
    auto values = array_to_vector<double>(a, name);
    for (double x : values)
      if (std::trunc(x) != x && !std::isnan(x))
        type_error(name, "an array-like of integers");
  }
  auto values = array_to_vector<long long>(a, name);
  return std::vector<long>(values.begin(), values.end());
}

Rcpp::LogicalVector logicals_from_py(py::handle h, const char *name) {
  return Rcpp::LogicalVector(array_to_vector<bool>(h, name));
}

Rcpp::ComplexVector complexes_from_py(py::handle h, const char *name) {
  Rcpp::ComplexVector result;
  for (const auto &c : array_to_vector<std::complex<double>>(h, name))
    result.push_back(Rcomplex{c.real(), c.imag()});
  return result;
}

Rcpp::StringVector strings_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  if (py::isinstance<py::str>(h)) return {h.cast<std::string>()};
  try {
    return Rcpp::StringVector(h.cast<std::vector<std::string>>());
  } catch (const py::cast_error &) {
    type_error(name, "a sequence of str");
  }
}

Rcpp::NumericMatrix numeric_matrix_from_py(py::handle h, const char *name) {
  return matrix_from_py<double>(h, name);
}

Rcpp::IntegerMatrix integer_matrix_from_py(py::handle h, const char *name) {
  return matrix_from_py<int>(h, name);
}

Rcpp::List list_from_py(py::handle h, const char *name) {
  if (h.is_none()) return {};
  if (!py::isinstance<py::dict>(h) && !py::isinstance<py::sequence>(h))
    type_error(name, "a dict or sequence");
  return list_from_dict_or_sequence(h);
}

namespace {

template <class F> py::list draw(long n, F &&f) {
  if (n < 0) throw py::value_error("n must be non-negative");
  std::vector<double> result;
  {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(castor_mutex());
    for (long i = 0; i < n; ++i) result.push_back(f());
  }
  return py::cast(result);
}

} // namespace

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
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(castor_mutex());
        if (castorbind_set_rng_state(state.data()) != 0) {
          py::gil_scoped_acquire gil;
          throw py::value_error("invalid RNG state: all zero");
        }
      },
      py::arg("state"),
      "Restore an RNG state returned by get_rng_state(), or taken from "
      ".Random.seed[-1] in R.");

  // R-identical draws, for testing the RNG against R itself
  m.def(
      "_runif", [](long n) { return draw(n, unif_rand); }, py::arg("n"),
      "Same as runif(n) in R.");
  m.def(
      "_rnorm", [](long n) { return draw(n, norm_rand); }, py::arg("n"),
      "Same as rnorm(n) in R.");
  m.def(
      "_rexp", [](long n) { return draw(n, exp_rand); }, py::arg("n"),
      "Same as rexp(n) in R.");
  m.def(
      "_rbinom",
      [](long n, double size, double prob) {
        return draw(n, [=] { return rbinom(size, prob); });
      },
      py::arg("n"), py::arg("size"), py::arg("prob"),
      "Same as rbinom(n, size, prob) in R.");
}

} // namespace castorbind
