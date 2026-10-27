// Stand-in for the parts of Rcpp that castor uses, so castor's C++ compiles
// unmodified without R. Everything here is plain C++: results are held as
// C++ values and converted to Python once per call, at the binding boundary
// (src/castorbind.cpp), so castor itself runs without holding the GIL.
#pragma once

// headers that castor relies on real Rcpp/R to include
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// R's boolean type (R_ext/Boolean.h)
typedef enum { FALSE = 0, TRUE } Rboolean;

// R's complex type; castor reads its .r and .i members
typedef struct {
  double r;
  double i;
} Rcomplex;

namespace Rcpp {

class List;
class RObject;

namespace castorbind_detail {

// R type codes, as in Rinternals.h
constexpr int LGLSXP = 10;
constexpr int INTSXP = 13;
constexpr int REALSXP = 14;
constexpr int CPLXSXP = 15;
constexpr int STRSXP = 16;

template <int RTYPE> struct storage;
template <> struct storage<LGLSXP> { using type = int; };
template <> struct storage<INTSXP> { using type = int; };
template <> struct storage<REALSXP> { using type = double; };
template <> struct storage<CPLXSXP> { using type = Rcomplex; };
template <> struct storage<STRSXP> { using type = std::string; };

// a named list element is missing (a KeyError in Python)
struct missing_name : std::out_of_range {
  using std::out_of_range::out_of_range;
};

// column-major, like R
struct MatrixValue {
  long nrow = 0;
  long ncol = 0;
  std::vector<double> data;
};

// everything castor can hand back, after normalization by to_value()
using Value = std::variant<
    std::monostate, bool, long, double, std::string,
    std::vector<bool>, std::vector<long>, std::vector<double>,
    std::vector<std::string>,
    MatrixValue, std::shared_ptr<const List>>;

template <class T> Value to_value(const T &x);

template <class T> struct is_std_vector : std::false_type {};
template <class T, class A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template <class T> inline constexpr bool always_false = false;

// element-wise conversion of any numeric value to std::vector<E>
template <class E> std::vector<E> numeric_as(const Value &v) {
  return std::visit(
      [](const auto &x) -> std::vector<E> {
        using X = std::decay_t<decltype(x)>;
        if constexpr (std::is_arithmetic_v<X>) {
          return {static_cast<E>(x)};
        } else if constexpr (is_std_vector<X>::value) {
          if constexpr (std::is_arithmetic_v<typename X::value_type>)
            return std::vector<E>(x.begin(), x.end());
        }
        throw std::invalid_argument("castorbind: expected a numeric value");
      },
      v);
}

} // namespace castorbind_detail

// type-erased value, as returned by wrap() or read from a List
class RObject {
public:
  RObject() = default;
  RObject(castorbind_detail::Value v) : value(std::move(v)) {}
  castorbind_detail::Value value;
};

template <class T> RObject wrap(const T &x) {
  return RObject(castorbind_detail::to_value(x));
}

template <class T> T as(const RObject &x);

template <class T, class U> T as(const U &x) {
  return as<T>(RObject(castorbind_detail::to_value(x)));
}

template <int RTYPE> class Vector {
public:
  using stored_type = typename castorbind_detail::storage<RTYPE>::type;
  using value_type = stored_type;
  using iterator = typename std::vector<stored_type>::iterator;
  using const_iterator = typename std::vector<stored_type>::const_iterator;

  Vector() = default;
  explicit Vector(std::size_t n) : data_(n) {}
  template <class U>
  Vector(const std::vector<U> &x) : data_(x.begin(), x.end()) {}
  Vector(const RObject &x);

  std::size_t size() const { return data_.size(); }
  stored_type &operator[](std::size_t i) { return data_[i]; }
  const stored_type &operator[](std::size_t i) const { return data_[i]; }
  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  const std::vector<stored_type> &std_vector() const { return data_; }

private:
  std::vector<stored_type> data_;
};

template <int RTYPE> class Matrix {
public:
  using stored_type = typename castorbind_detail::storage<RTYPE>::type;

  Matrix() = default;
  Matrix(long nrow, long ncol)
      : nrow_(nrow), ncol_(ncol), data_(std::size_t(nrow * ncol)) {}

  long nrow() const { return nrow_; }
  long ncol() const { return ncol_; }
  stored_type &operator()(long r, long c) { return data_[r + c * nrow_]; }
  const stored_type &operator()(long r, long c) const {
    return data_[r + c * nrow_];
  }
  const std::vector<stored_type> &column_major() const { return data_; }

private:
  long nrow_ = 0;
  long ncol_ = 0;
  std::vector<stored_type> data_;
};

using NumericVector = Vector<castorbind_detail::REALSXP>;
using IntegerVector = Vector<castorbind_detail::INTSXP>;
using LogicalVector = Vector<castorbind_detail::LGLSXP>;
using ComplexVector = Vector<castorbind_detail::CPLXSXP>;
using StringVector = Vector<castorbind_detail::STRSXP>;
using NumericMatrix = Matrix<castorbind_detail::REALSXP>;
using IntegerMatrix = Matrix<castorbind_detail::INTSXP>;

struct NamedArg {
  std::string name;
  RObject value;
};

// Rcpp::Named("key") = value
class Named {
public:
  explicit Named(std::string name) : name_(std::move(name)) {}
  template <class T> NamedArg operator=(const T &x) const {
    return {name_, wrap(x)};
  }

private:
  std::string name_;
};

// R list; names are "" for unnamed entries
class List {
public:
  List() = default;
  List(const RObject &x);
  // Rcpp's implicit List(size); castor only reaches it in error paths
  // (`return NAN_D;`), which R rejects as an invalid length
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  List(const T &) {
    throw std::invalid_argument("castorbind: invalid list length");
  }

  template <class... Args> static List create(const Args &...args) {
    List result;
    (result.append(args), ...);
    return result;
  }

  std::size_t size() const { return values_.size(); }
  const std::vector<std::string> &names() const { return names_; }

  const RObject &operator[](std::size_t i) const { return values_.at(i); }
  const RObject &operator[](const std::string &name) const {
    for (std::size_t i = 0; i < names_.size(); ++i)
      if (names_[i] == name) return values_[i];
    throw castorbind_detail::missing_name("castorbind: no list element named '" +
                                          name + "'");
  }
  const RObject &operator[](const char *name) const {
    return (*this)[std::string(name)];
  }

  template <class T> void push_back(const T &x) { append(x); }
  template <class T> void push_back(const T &x, const std::string &name) {
    values_.push_back(wrap(x));
    names_.push_back(name);
  }

private:
  void append(const NamedArg &x) {
    values_.push_back(x.value);
    names_.push_back(x.name);
  }
  template <class T> void append(const T &x) {
    values_.push_back(wrap(x));
    names_.emplace_back();
  }

  std::vector<RObject> values_;
  std::vector<std::string> names_;
};

namespace castorbind_detail {

template <class T> Value to_value(const T &x) {
  if constexpr (std::is_same_v<T, Value>) {
    return x;
  } else if constexpr (std::is_same_v<T, RObject>) {
    return x.value;
  } else if constexpr (std::is_same_v<T, List>) {
    return std::make_shared<const List>(x);
  } else if constexpr (std::is_same_v<T, bool>) {
    return x;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    return static_cast<long>(x);
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<double>(x);
  } else if constexpr (std::is_convertible_v<T, std::string>) {
    return std::string(x);
  } else if constexpr (is_std_vector<T>::value) {
    using E = typename T::value_type;
    if constexpr (std::is_same_v<E, bool>) {
      return std::vector<bool>(x.begin(), x.end());
    } else if constexpr (std::is_integral_v<E>) {
      return std::vector<long>(x.begin(), x.end());
    } else if constexpr (std::is_floating_point_v<E>) {
      return std::vector<double>(x.begin(), x.end());
    } else if constexpr (std::is_convertible_v<E, std::string>) {
      return std::vector<std::string>(x.begin(), x.end());
    } else {
      // e.g. vector<vector<long>>, which Rcpp wraps as an unnamed list
      List result;
      for (const auto &item : x) result.push_back(item);
      return std::make_shared<const List>(std::move(result));
    }
  } else if constexpr (std::is_same_v<T, LogicalVector>) {
    return std::vector<bool>(x.begin(), x.end());
  } else if constexpr (std::is_same_v<T, NumericVector> ||
                       std::is_same_v<T, IntegerVector> ||
                       std::is_same_v<T, StringVector>) {
    return to_value(x.std_vector());
  } else if constexpr (std::is_same_v<T, NumericMatrix> ||
                       std::is_same_v<T, IntegerMatrix>) {
    const auto &data = x.column_major();
    return MatrixValue{x.nrow(), x.ncol(),
                       std::vector<double>(data.begin(), data.end())};
  } else {
    static_assert(always_false<T>, "castorbind Rcpp shim: unsupported type");
  }
}

} // namespace castorbind_detail

template <class T> T as(const RObject &x) {
  using namespace castorbind_detail;
  if constexpr (std::is_arithmetic_v<T>) {
    const auto values = numeric_as<T>(x.value);
    if (values.size() != 1)
      throw std::invalid_argument("castorbind: expected a scalar");
    return values[0];
  } else if constexpr (std::is_same_v<T, List>) {
    return List(x);
  } else if constexpr (is_std_vector<T>::value) {
    return numeric_as<typename T::value_type>(x.value);
  } else {
    static_assert(always_false<T>, "castorbind Rcpp shim: unsupported as<>");
  }
}

template <int RTYPE> Vector<RTYPE>::Vector(const RObject &x) {
  static_assert(std::is_arithmetic_v<stored_type>,
                "castorbind Rcpp shim: only numeric vectors from RObject");
  data_ = castorbind_detail::numeric_as<stored_type>(x.value);
}

inline List::List(const RObject &x) {
  if (std::holds_alternative<std::monostate>(x.value)) return;
  const auto *list = std::get_if<std::shared_ptr<const List>>(&x.value);
  if (list == nullptr)
    throw std::invalid_argument("castorbind: expected a list");
  *this = **list;
}

// defined in src/castorbind.cpp: raise pending Python signals (e.g. Ctrl-C)
void checkUserInterrupt();

// defined in src/castorbind.cpp: line-buffered, written to sys.stdout
extern std::ostream Rcout;

} // namespace Rcpp

// R's standalone math library (vendor/r-source/src/nmath), linked in place of R
namespace castorbind_rmath {
extern "C" {
double runif(double a, double b);
double rexp(double scale);
double rbinom(double n, double p);
}
} // namespace castorbind_rmath

namespace R {
inline double runif(double a, double b) {
  return castorbind_rmath::runif(a, b);
}
inline double rexp(double scale) { return castorbind_rmath::rexp(scale); }
inline double rbinom(double n, double p) {
  return castorbind_rmath::rbinom(n, p);
}
} // namespace R
