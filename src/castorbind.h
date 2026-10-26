// Conversions between Python and the Rcpp shim, used by the generated
// bindings (_core.cpp, written by src/gen_bindings.py).
#pragma once

#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include <Rcpp.h>

namespace castorbind {

namespace py = pybind11;

// thrown by Rcpp::checkUserInterrupt when a Python signal handler raised;
// the Python exception stays set and is re-raised once the GIL is back
struct Interrupted {};

// castor and R's RNG keep global state, so calls into them are serialized;
// never wait for this while holding the GIL
std::mutex &castor_mutex();

py::object to_py(const Rcpp::RObject &x);

long long_from_py(py::handle h, const char *name);
double double_from_py(py::handle h, const char *name);
bool bool_from_py(py::handle h, const char *name);
std::string string_from_py(py::handle h, const char *name);
std::vector<double> doubles_from_py(py::handle h, const char *name);
std::vector<long> longs_from_py(py::handle h, const char *name);
Rcpp::LogicalVector logicals_from_py(py::handle h, const char *name);
Rcpp::ComplexVector complexes_from_py(py::handle h, const char *name);
Rcpp::StringVector strings_from_py(py::handle h, const char *name);
Rcpp::NumericMatrix numeric_matrix_from_py(py::handle h, const char *name);
Rcpp::IntegerMatrix integer_matrix_from_py(py::handle h, const char *name);
Rcpp::List list_from_py(py::handle h, const char *name);

// convert a Python argument to the C++ type a castor function takes
template <class T> T from_py(py::handle h, const char *name) {
  if constexpr (std::is_same_v<T, bool>) {
    return bool_from_py(h, name);
  } else if constexpr (std::is_integral_v<T>) {
    const long x = long_from_py(h, name);
    if (x < std::numeric_limits<T>::min() || x > std::numeric_limits<T>::max())
      throw py::value_error(std::string("argument '") + name +
                            "': out of range");
    return static_cast<T>(x);
  } else if constexpr (std::is_floating_point_v<T>) {
    return double_from_py(h, name);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return string_from_py(h, name);
  } else if constexpr (std::is_same_v<T, std::vector<double>>) {
    return doubles_from_py(h, name);
  } else if constexpr (std::is_same_v<T, std::vector<long>>) {
    return longs_from_py(h, name);
  } else if constexpr (std::is_same_v<T, Rcpp::LogicalVector>) {
    return logicals_from_py(h, name);
  } else if constexpr (std::is_same_v<T, Rcpp::ComplexVector>) {
    return complexes_from_py(h, name);
  } else if constexpr (std::is_same_v<T, Rcpp::StringVector>) {
    return strings_from_py(h, name);
  } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    return strings_from_py(h, name).std_vector();
  } else if constexpr (std::is_same_v<T, Rcpp::NumericMatrix>) {
    return numeric_matrix_from_py(h, name);
  } else if constexpr (std::is_same_v<T, Rcpp::IntegerMatrix>) {
    return integer_matrix_from_py(h, name);
  } else if constexpr (std::is_same_v<T, Rcpp::List>) {
    return list_from_py(h, name);
  } else {
    static_assert(Rcpp::castorbind_detail::always_false<T>,
                  "castorbind: unsupported argument type");
  }
}

// run a castor function without the GIL, then convert its result
template <class F> py::object call(F &&f) {
  std::unique_lock<std::mutex> lock(castor_mutex(), std::defer_lock);
  {
    py::gil_scoped_release release;
    lock.lock();
  }
  py::scoped_ostream_redirect redirect(Rcpp::Rcout);
  Rcpp::RObject result;
  try {
    py::gil_scoped_release release;
    result = Rcpp::wrap(f());
  } catch (const Interrupted &) {
    throw py::error_already_set();
  }
  return to_py(result);
}

// RNG functions, castor_version etc.
void init_module(py::module_ &m);

} // namespace castorbind
