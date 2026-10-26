// Conversions between Python and the Rcpp shim, used by the generated
// bindings (_core.cpp, written by src/gen_bindings.py).
#pragma once

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Rcpp.h>

namespace castorbind {

namespace py = pybind11;

// thrown by Rcpp::checkUserInterrupt when a Python signal handler raised;
// the Python exception stays set and is re-raised once the GIL is back
struct Interrupted {};

// castor and R's RNG keep global state, so calls into them are serialized;
// this waits for the lock without holding the GIL
std::unique_lock<std::mutex> lock_castor();

py::object to_py(const Rcpp::RObject &x);

// convert a Python array or list argument to the C++ type a castor
// function takes (pybind11 converts scalar and string arguments itself)
template <class T> T from_py(py::handle h, const char *name);
template <> std::vector<double> from_py(py::handle h, const char *name);
template <> std::vector<long> from_py(py::handle h, const char *name);
template <> Rcpp::LogicalVector from_py(py::handle h, const char *name);
template <> Rcpp::ComplexVector from_py(py::handle h, const char *name);
template <> Rcpp::NumericMatrix from_py(py::handle h, const char *name);
template <> Rcpp::IntegerMatrix from_py(py::handle h, const char *name);
template <> Rcpp::List from_py(py::handle h, const char *name);

// run a castor function without the GIL, then convert its result
template <class F> py::object call(F &&f) {
  const auto lock = lock_castor();
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
