import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def fx(tmp_path):
    """Writable copy of the fixtures so negative controls can corrupt files."""
    dst = tmp_path / "fx"
    shutil.copytree(FIX, dst)
    return dst


@pytest.fixture
def db(tmp_path):
    return tmp_path / "db"
