import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc

import json


def test_version_and_built():
    print(hpc.__version__)
    print(hpc.__built_json__)

    j = json.loads(hpc.__built_json__)
    print(j)
