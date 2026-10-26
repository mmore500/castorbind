#!/usr/bin/env python3
"""Generate castorbind's pybind11 module and type stubs from castor.

Rcpp writes one clean prototype per exported function into castor's
src/RcppExports.cpp, and the matching R wrapper (with any default argument
values) into R/RcppExports.R. Both are parsed here, rather than castor's
heavily commented source.

Usage:
    python3 src/gen_bindings.py [--castor vendor/castor]
        [--cpp-out _core.cpp] [--pyi-out python/castorbind/legacy/__init__.pyi]
        [--check]
"""

import argparse
import dataclasses
import keyword
import pathlib
import re
import sys
import typing

HERE = pathlib.Path(__file__).resolve().parent

# C++ argument type (minus const/&) -> python stub type
ARG_STUB_TYPES = {
    "bool": "bool",
    "int": "int",
    "long": "int",
    "double": "float",
    "std::string": "str",
    "std::vector<double>": "ArrayLike",
    "std::vector<long>": "ArrayLike",
    "LogicalVector": "ArrayLike",
    "ComplexVector": "ArrayLike",
    "NumericMatrix": "ArrayLike",
    "IntegerMatrix": "ArrayLike",
    "StringVector": "list[str] | tuple[str, ...]",
    "std::vector<std::string>": "list[str] | tuple[str, ...]",
    "Rcpp::List": "dict[str, typing.Any] | list[typing.Any] | tuple[typing.Any, ...]",
}

# C++ argument type (minus const/&) -> type the binding takes, letting
# pybind11 convert it; other arguments go through castorbind::from_py
NATIVE_ARG_TYPES = {
    "bool": "bool",
    "int": "int",
    "long": "long",
    "double": "double",
    "std::string": "std::string",
    "StringVector": "std::vector<std::string>",
    "std::vector<std::string>": "std::vector<std::string>",
}

# C++ return type -> python stub type
RETURN_STUB_TYPES = {
    "Rcpp::List": "dict[str, typing.Any]",
    "NumericVector": "list[float]",
    "IntegerVector": "list[int]",
    "NumericMatrix": "np.ndarray",
    "bool": "bool",
    "long": "int",
    "double": "float",
    "std::string": "str",
    "std::vector<double>": "list[float]",
    "std::vector<long>": "list[int]",
    "std::vector<std::vector<long> >": "list[list[int]]",
}

# R literal default -> (C++ expression, python stub expression)
R_LITERALS = {"TRUE": ("true", "True"), "FALSE": ("false", "False")}


@dataclasses.dataclass
class Param:
    cpp_type: str  # as declared, e.g. "const std::vector<double>&"
    name: str
    default: typing.Optional[str] = None  # R literal

    @property
    def base_type(self) -> str:
        base = re.sub(r"\bconst\b|&", "", self.cpp_type)
        return re.sub(r"\s+", " ", base).strip()

    @property
    def py_name(self) -> str:
        return self.name + "_" if keyword.iskeyword(self.name) else self.name


@dataclasses.dataclass
class Export:
    return_type: str
    cpp_name: str  # e.g. get_distances_from_root_CPP
    params: typing.List[Param]

    @property
    def py_name(self) -> str:
        return re.sub(r"_CPP$", "", self.cpp_name)

    @property
    def prototype(self) -> str:
        params = ", ".join(f"{p.cpp_type} {p.name}" for p in self.params)
        return f"{self.return_type} {self.cpp_name}({params});"


def split_params(text: str) -> typing.List[str]:
    """Split on commas that are not inside template brackets."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def parse_rcpp_exports(path: pathlib.Path) -> typing.List[Export]:
    lines = path.read_text().splitlines()
    exports = []
    for header, prototype in zip(lines, lines[1:]):
        match = re.fullmatch(r"// (\w+_CPP)", header)
        if not match:
            continue
        name = match.group(1)
        proto = re.fullmatch(rf"(.+?)\s+{name}\((.*)\);", prototype.strip())
        if not proto:
            raise ValueError(f"unparseable prototype for {name}: {prototype}")
        params = []
        for param in split_params(proto.group(2)):
            cpp_type, param_name = re.fullmatch(
                r"(.+?)\s*(\w+)", param
            ).groups()
            params.append(Param(cpp_type.strip(), param_name))
        exports.append(Export(proto.group(1).strip(), name, params))
    return exports


def parse_r_defaults(
    path: pathlib.Path,
) -> typing.Dict[str, typing.Dict[str, str]]:
    defaults = {}
    pattern = re.compile(r"^(\w+) <- function\((.*)\) \{$", re.M)
    for name, args in pattern.findall(path.read_text()):
        defaults[name] = {
            arg.split("=")[0].strip(): arg.split("=", 1)[1].strip()
            for arg in args.split(",")
            if "=" in arg
        }
    return defaults


def translate_default(r_literal: str) -> typing.Tuple[str, str]:
    """R literal -> (C++ expression, python expression)."""
    if r_literal in R_LITERALS:
        return R_LITERALS[r_literal]
    if re.fullmatch(r"-?\d+L", r_literal):
        return r_literal[:-1] + "L", r_literal[:-1]
    if re.fullmatch(r"-?\d+(\.\d*)?", r_literal):  # R numeric: a double
        value = repr(float(r_literal))
        return value, value
    raise NotImplementedError(f"unsupported default value: {r_literal}")


def read_castor_version(castor: pathlib.Path) -> str:
    description = (castor / "DESCRIPTION").read_text()
    return re.search(r"^Version:\s*(\S+)", description, re.M).group(1)


def load_exports(castor: pathlib.Path) -> typing.List[Export]:
    exports = parse_rcpp_exports(castor / "src" / "RcppExports.cpp")
    defaults = parse_r_defaults(castor / "R" / "RcppExports.R")
    for export in exports:
        r_defaults = defaults.get(export.cpp_name, {})
        for param in export.params:
            param.default = r_defaults.get(param.name)
        unknown_types = {
            p.base_type for p in export.params
        } - ARG_STUB_TYPES.keys()
        if unknown_types:
            raise ValueError(f"{export.cpp_name}: unsupported {unknown_types}")
        if export.return_type not in RETURN_STUB_TYPES:
            raise ValueError(
                f"{export.cpp_name}: unsupported return {export.return_type}"
            )
    return exports


def generate_cpp(exports: typing.List[Export], castor_version: str) -> str:
    out = [
        "// Generated by src/gen_bindings.py from castor's RcppExports;",
        "// do not edit by hand.",
        '#include "castorbind.h"',
        "",
        "using namespace Rcpp;",
        "",
        "// castor's exported functions, as declared in RcppExports.cpp",
    ]
    out += [export.prototype for export in exports]
    out += [
        "",
        "PYBIND11_MODULE(_core, m) {",
        "  namespace py = pybind11;",
        '  m.doc() = "Bindings for the C++ core of the castor R package.";',
        f'  m.attr("_castor_version") = "{castor_version}";',
        "  castorbind::init_module(m);",
    ]
    for export in exports:
        signature = ", ".join(
            f"{NATIVE_ARG_TYPES.get(p.base_type, 'py::object')} {p.py_name}"
            for p in export.params
        )
        out.append(f'  m.def("{export.py_name}", []({signature}) {{')
        args = []
        for param in export.params:
            if param.base_type in NATIVE_ARG_TYPES:
                args.append(param.py_name)
                continue
            out.append(
                f"    auto a_{param.name} = castorbind::from_py<"
                f'{param.base_type}>({param.py_name}, "{param.py_name}");'
            )
            args.append(f"a_{param.name}")
        out.append(
            f"    return castorbind::call([&] {{ return {export.cpp_name}("
            f"{', '.join(args)}); }});"
        )
        py_args = []
        for param in export.params:
            py_arg = f'py::arg("{param.py_name}")'
            if param.base_type in ("bool", "int", "long"):
                # no None or numbers for bools; no floats etc. for ints
                py_arg += ".noconvert()"
            if param.default is not None:
                cpp_default, _ = translate_default(param.default)
                py_arg += f" = {cpp_default}"
            py_args.append(py_arg)
        doc = f"Binding for castor's {export.cpp_name}."
        out.append(f'  }}, {", ".join(py_args + [repr_cpp(doc)])});')
    out.append("}")
    return "\n".join(out) + "\n"


def repr_cpp(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generate_pyi(exports: typing.List[Export]) -> str:
    out = [
        "# Generated by src/gen_bindings.py from castor's RcppExports;",
        "# do not edit by hand.",
        "import typing",
        "",
        "import numpy as np",
        "from numpy.typing import ArrayLike",
        "",
        "def set_seed(seed: int) -> None: ...",
        "def get_rng_state() -> list[int]: ...",
        "def set_rng_state(state: list[int] | tuple[int, ...]) -> None: ...",
    ]
    for export in exports:
        params = []
        for param in export.params:
            stub = f"{param.py_name}: {ARG_STUB_TYPES[param.base_type]}"
            if param.default is not None:
                stub += f" = {translate_default(param.default)[1]}"
            params.append(stub)
        out.append(
            f"def {export.py_name}({', '.join(params)}) -> "
            f"{RETURN_STUB_TYPES[export.return_type]}: ..."
        )
    return "\n".join(out) + "\n"


def write_or_check(path: pathlib.Path, content: str, check: bool) -> bool:
    if check:
        if not path.exists() or path.read_text() != content:
            print(f"{path} is out of date; rerun {__file__}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--castor", type=pathlib.Path, default=HERE.parent / "vendor/castor"
    )
    parser.add_argument("--cpp-out", type=pathlib.Path)
    parser.add_argument(
        "--pyi-out",
        type=pathlib.Path,
        help="defaults to python/castorbind/legacy/__init__.pyi, unless "
        "--cpp-out is given",
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if outputs are stale"
    )
    args = parser.parse_args()
    if args.cpp_out is None and args.pyi_out is None:
        args.pyi_out = HERE.parent / "python/castorbind/legacy/__init__.pyi"

    exports = load_exports(args.castor)
    castor_version = read_castor_version(args.castor)
    ok = True
    if args.cpp_out is not None:
        cpp = generate_cpp(exports, castor_version)
        ok &= write_or_check(args.cpp_out, cpp, args.check)
    if args.pyi_out is not None:
        pyi = generate_pyi(exports)
        ok &= write_or_check(args.pyi_out, pyi, args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
