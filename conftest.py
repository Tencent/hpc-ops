import pytest
import os
from pathlib import Path

import sys
from typing import Any, Dict, List
import subprocess
import types
import tempfile
import torch

tempfile.tempdir = "/dev/shm"


def save_data(file_name, module_name, func_name, ret, args, kwargs):
    data = {
        "module_name": module_name,
        "func_name": func_name,
        "ret": ret,
        "args": args,
        "kwargs": kwargs,
    }

    torch.save(data, file_name)


def dump_test_py(file_name, test_before_file, test_after_file, pypath):

    text = """ 
import sys
import os
from pathlib import Path

sys.path.insert(0, "{}")

import torch
import hpc

din = torch.load("{}")
dout = torch.load("{}")

func = getattr(hpc, din['func_name'])
args = din['args']
kwargs = din['kwargs']
gt = dout['ret']

s = func(*args, **kwargs)

# test output
def assert_equal(my, gt):
  if isinstance(my, torch.Tensor):
    assert torch.equal(my.byte(), gt.byte())
  elif isinstance(my, tuple):
    for i, e in enumerate(my):
      assert_equal(my[i], gt[i])
  elif isinstance(my, dict):
    for k in my.keys():
      assert_equal(my[k], gt[k])
  else:
    assert my == gt

assert_equal(s, gt)
assert_equal(args, dout['args'])
assert_equal(kwargs, dout['kwargs'])

""".format(
        pypath, test_before_file, test_after_file
    )

    with open(file_name, "w") as fp:
        fp.write(text)


def sanitizer_check(file_name, check):
    cmd = f'PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool={check} --require-cuda-init=no --kernel-name regex="hpc.+" python3 {file_name}'
    print(cmd)
    try:
        output = subprocess.check_output(cmd, shell=True)
        text = output.decode("utf-8")
        print(text)
    except subprocess.CalledProcessError as e:
        raise e


class TraceHook(object):
    def __init__(self, checks, module_name):
        self.checks_ = checks
        self.module_name = module_name

    def _wrap_func(self, module, func_name):
        if not hasattr(module, func_name):
            return False

        org_func = getattr(module, func_name)

        def wrapped(*args, **kwargs):
            fd, tmp_py_file = tempfile.mkstemp(prefix="tmp_hpc_" + func_name + "_", suffix=".py")
            os.close(fd)
            tmp_before_invoke_file = tmp_py_file.replace(".py", "_before_invoke.pth")
            tmp_after_invoke_file = tmp_py_file.replace(".py", "_after_invoke.pth")

            save_data(tmp_before_invoke_file, "hpc", func_name, None, args, kwargs)
            ret = org_func(*args, **kwargs)
            save_data(tmp_after_invoke_file, "hpc", func_name, ret, args, kwargs)

            pypath = os.path.realpath(list(Path(__file__).parent.glob("./build/lib.*/"))[0])
            dump_test_py(tmp_py_file, tmp_before_invoke_file, tmp_after_invoke_file, pypath)
            print(tmp_py_file)

            for check in self.checks_:
                print(f"{check}...")
                sanitizer_check(tmp_py_file, check)

            os.unlink(tmp_before_invoke_file)
            os.unlink(tmp_after_invoke_file)
            os.unlink(tmp_py_file)

            return ret

        if len(self.checks_) > 0:
            setattr(module, func_name, wrapped)

        return True

    def hook(self):
        sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("./build/lib.*/"))[0]))
        module = __import__(self.module_name)

        dirs = dir(module)
        for d in dirs:
            if d.endswith("fake") or d.startswith("_"):
                continue
            e = getattr(module, d)
            if not isinstance(e, types.FunctionType):
                continue
            if not callable(e):
                continue
            self._wrap_func(module, d)


def get_checks():
    checks = os.getenv("SANITIZER_CHECK")
    if not checks:
        return []
    checks = [e.strip() for e in checks.split(",")]

    return checks


# enable compute-sanitizer by set environment
# eg.
# export SANITIZER_CHECK=memcheck,synccheck,racecheck
# to enable memcheck etc.


# ---------------------------------------------------------------------------
# Auto-skip multi-GPU tests when running on a single GPU.
#
# Detection: if a test file contains `set_device` (indicating multi-GPU usage),
# the entire file will be skipped.
#
# Control via environment variable SKIP_MULTI_GPU:
#   export SKIP_MULTI_GPU=0   -> do NOT skip (force run multi-GPU tests)
#   Any other value or unset  -> skip multi-GPU tests
# ---------------------------------------------------------------------------


def _should_skip_multi_gpu() -> bool:
    """Determine whether multi-GPU tests should be skipped."""
    env_val = os.getenv("SKIP_MULTI_GPU")
    if env_val is not None and env_val.strip() == "0":
        return False
    return True


def _file_contains_set_device(filepath: str) -> bool:
    """Check if a test file contains 'set_device' (multi-GPU indicator)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return "set_device" in f.read()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip multi-GPU test files based on set_device detection."""
    if not _should_skip_multi_gpu():
        return

    skip_marker = pytest.mark.skip(
        reason="Skipped: multi-GPU test (set_device detected), " "set SKIP_MULTI_GPU=0 to force run"
    )

    for item in items:
        test_file = str(item.fspath)
        if _file_contains_set_device(test_file):
            item.add_marker(skip_marker)


def pytest_configure(config):
    checks = get_checks()
    hooker = TraceHook(checks, "hpc")
    hooker.hook()
