# castorbind

[
![PyPI](https://img.shields.io/pypi/v/castorbind.svg)
](https://pypi.python.org/pypi/castorbind)
[
![Python versions](https://img.shields.io/pypi/pyversions/castorbind.svg)
](https://pypi.python.org/pypi/castorbind)
[
![CI](https://github.com/mmore500/castorbind/actions/workflows/ci.yaml/badge.svg)
](https://github.com/mmore500/castorbind/actions)
[
![License](https://img.shields.io/github/license/mmore500/castorbind.svg)
](https://github.com/mmore500/castorbind/blob/master/LICENSE)
[
![GitHub stars](https://img.shields.io/github/stars/mmore500/castorbind.svg?style=round-square&logo=github&label=Stars&logoColor=white)](https://github.com/mmore500/castorbind)

Python bindings for the C++ core of [castor](https://cran.r-project.org/package=castor), the R package for efficient phylogenetics on large trees (tree simulation, ancestral state reconstruction, and fitting models of trait evolution and diversification), with no R needed at build time or at runtime.

- Free software: GPL-2.0-or-later
- Repository: <https://github.com/mmore500/castorbind>

castor is by Stilianos Louca, and full credit for it is due to its original author.
castorbind is not an official port.

castorbind wraps every C++ function castor exports, under its castor name minus the `_CPP` suffix, taking castor's own flat, 0-based inputs and returning dicts.
R's random number generator is included, so outputs match castor in R bit for bit, including seeded simulations: `set_seed(n)` gives the same random stream as `set.seed(n)` in R.
castor's R-level functions are not included.

castorbind is versioned independently of castor; `castorbind.castor_version` gives the version of castor it wraps.

## Usage

```python3
import castorbind as cb

r = cb.read_Newick_string(
    input="((a:1,b:1):1,c:2);",
    underscores_as_blanks=False,
    interpret_quotes=True,
    look_for_edge_names=False,
    look_for_edge_numbers=False,
)
depths = cb.get_distances_from_root(
    Ntips=r["Ntips"],
    Nnodes=r["Nnodes"],
    Nedges=len(r["tree_edge"]) // 2,
    tree_edge=r["tree_edge"],
    edge_length=r["edge_lengths"],
)
```

- Trees are counts plus a row-major, 0-based `tree_edge` array of length 2 × Nedges (a `(Nedges, 2)` array also works). Tips are 0 to Ntips−1 and internal nodes follow.
- Results are dicts with castor's own keys (`success`, `error`, ...); failures are reported through `success=False`, not raised.
- Arrays go in as numpy arrays or lists. Vectors come back as lists and matrices as 2-D numpy arrays.

## Installation

```bash
python3 -m pip install castorbind
```

Building from source needs a C++17 compiler and CMake:

```bash
git clone --recursive https://github.com/mmore500/castorbind
python3 -m pip install ./castorbind
```

## AI Use

Substantial portions of castorbind were prepared using agentic coding tools.

## Citing

If castorbind is useful in your research, please cite castor:

> Louca, S., & Doebeli, M. (2017). Efficient comparative phylogenetics on large trees. Bioinformatics, 34(6), 1053–1055. <https://doi.org/10.1093/bioinformatics/btx701>

```bibtex
@article{louca2017efficient,
  title = {Efficient comparative phylogenetics on large trees},
  volume = {34},
  ISSN = {1367-4811},
  url = {http://dx.doi.org/10.1093/bioinformatics/btx701},
  DOI = {10.1093/bioinformatics/btx701},
  number = {6},
  journal = {Bioinformatics},
  publisher = {Oxford University Press (OUP)},
  author = {Louca, Stilianos and Doebeli, Michael},
  editor = {Valencia, Alfonso},
  year = {2017},
  month = oct,
  pages = {1053--1055}
}
```
