import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run(
        [sys.executable, str(ROOT / "sample_data" / "make_sample_data.py"), str(out)],
        check=True, capture_output=True,
    )
    return out
