#!/usr/bin/env python3
"""Install genuine R and castor from conda-forge, for tests that compare
castorbind against R on the fly.

Usage:
    python3 tests/install_r_env.py [PREFIX]

Then point the tests at it:
    export CASTORBIND_RSCRIPT=PREFIX/bin/Rscript

Requires py-rattler (pip install "py-rattler==0.28.0").
"""

import asyncio
import pathlib
import re
import sys

import rattler

DESCRIPTION = (
    pathlib.Path(__file__).resolve().parents[1]
    / "vendor"
    / "castor"
    / "DESCRIPTION"
)


def castor_version() -> str:
    """Version of castor that castorbind wraps."""
    match = re.search(r"^Version:\s*(\S+)$", DESCRIPTION.read_text(), re.M)
    return match.group(1)


async def install(prefix: pathlib.Path) -> None:
    records = await rattler.solve(
        ["conda-forge"],
        specs=["r-base>=4.4", f"r-castor=={castor_version()}"],
        virtual_packages=rattler.VirtualPackage.detect(),
    )
    await rattler.install(records, target_prefix=prefix)


def main() -> int:
    prefix = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".r-env")
    asyncio.run(install(prefix.resolve()))
    print(prefix.resolve() / "bin" / "Rscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
