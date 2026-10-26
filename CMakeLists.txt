cmake_minimum_required(VERSION 3.15...3.30)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# R's usual flags (-O2, no -ffast-math) keep floating-point behaviour close
# to CRAN builds; stop GCC/clang fusing multiply-adds (e.g., on ARM), which
# would change the last bits relative to x86
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-ffp-contract=off)
endif()

set(CASTOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/castor)
set(RSOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/r-source)
if(NOT EXISTS ${CASTOR_DIR}/src/RcppExports.cpp
   OR NOT EXISTS ${RSOURCE_DIR}/src/nmath/nmath.h)
  message(FATAL_ERROR
    "vendor/ submodules are empty; run `git submodule update --init`")
endif()

# R's standalone math library, as a static library so only the routines
# castor calls end up in the wheel
file(STRINGS ${RSOURCE_DIR}/VERSION PACKAGE_VERSION LIMIT_COUNT 1)
set(RMATH_HAVE_WORKING_LOG1P "# define HAVE_WORKING_LOG1P 1")
configure_file(
  ${RSOURCE_DIR}/src/include/Rmath.h0.in
  ${CMAKE_CURRENT_BINARY_DIR}/rmath/Rmath.h @ONLY)
file(GLOB NMATH_SOURCES ${RSOURCE_DIR}/src/nmath/*.c)
add_library(rmath STATIC ${NMATH_SOURCES})
target_compile_definitions(rmath PRIVATE MATHLIB_STANDALONE HAVE_CONFIG_H)
target_include_directories(rmath PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rmath_config
  ${CMAKE_CURRENT_BINARY_DIR}/rmath
  ${RSOURCE_DIR}/src/include
  ${RSOURCE_DIR}/src/nmath)
if(NOT MSVC)
  target_compile_options(rmath PRIVATE -w)  # vendored, unmodified
endif()

# bindings, generated from castor's RcppExports
set(CORE_CPP ${CMAKE_CURRENT_BINARY_DIR}/_core.cpp)
add_custom_command(
  OUTPUT ${CORE_CPP}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_bindings.py
    --castor ${CASTOR_DIR} --cpp-out ${CORE_CPP}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gen_bindings.py
    ${CASTOR_DIR}/src/RcppExports.cpp
    ${CASTOR_DIR}/R/RcppExports.R
    ${CASTOR_DIR}/DESCRIPTION
  COMMENT "Generating castor bindings")

set(CASTOR_SOURCES
  ${CASTOR_DIR}/src/phylogenetics_cpp_routines.cpp
  ${CASTOR_DIR}/src/STMathExpression.cpp)
if(NOT MSVC)
  set_source_files_properties(${CASTOR_SOURCES} PROPERTIES COMPILE_OPTIONS -w)
endif()

pybind11_add_module(_core MODULE NO_EXTRAS
  ${CORE_CPP}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/castorbind.cpp
  ${CASTOR_SOURCES})
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shim
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(_core PRIVATE rmath)

# pybind11 hides C++ symbols only; hide C ones too, so a process that also
# loads R (e.g., via reticulate) can't interpose its own unif_rand etc.
set_target_properties(rmath _core PROPERTIES
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

install(TARGETS _core LIBRARY DESTINATION castorbind)
