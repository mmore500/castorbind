import pathlib
import re

import numpy as np
import pytest

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"


def test_readme_usage_example():
    if not README.exists():
        pytest.skip("README.md not available")
    (example,) = re.findall(r"```python3\n(.*?)```", README.read_text(), re.S)
    namespace = {}
    exec(example, namespace)
    assert namespace["r"]["success"]
    np.testing.assert_equal(namespace["depths"][:3], [2.0, 2.0, 2.0])
