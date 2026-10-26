# vendor

Third-party sources, kept unmodified so a diff against upstream stays empty.
Any fixes live in `src/` instead.

- `castor/`: git submodule of <https://github.com/cran/castor>, pinned to CRAN tag `1.8.7`. GPL (>= 2).
- `rmath/`: from R 4.5.3 (<https://github.com/wch/r-source>, tag `tags/R-4-5-3`, commit `fe4249004a6976acb6e747ee831a6a56b2f3f122`). GPL (>= 2); see `rmath/COPYING`. Contains
  - `src/nmath/`: R's math library, in full,
  - `src/include/Rmath.h0.in`: template for `Rmath.h`, filled in by `CMakeLists.txt`,
  - `src/include/R_ext/{Boolean,Print,RS,Random}.h`: the R headers nmath includes,
  - `VERSION`, `COPYING`.

To update `rmath/`, replace these files with the same paths from a newer R release, and update the tag and commit above.
