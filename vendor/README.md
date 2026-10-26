# vendor

Third-party sources, as git submodules kept unmodified so a diff against upstream stays empty.
Any fixes live in `src/` instead.

- `castor/`: <https://github.com/cran/castor>, pinned to CRAN tag `1.8.7`. GPL (>= 2).
- `r-source/`: <https://github.com/wch/r-source> (a mirror of R's sources), pinned to tag `tags/R-4-5-3`, as a shallow submodule. GPL (>= 2). Only R's math library (`src/nmath`) and a few headers are built, and only those go in the sdist.

To upgrade, check out a new tag in the submodule and commit the bump.
