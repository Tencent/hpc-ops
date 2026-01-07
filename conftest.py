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


def dump_test_py(file_name, test_data_file, pypath):

    text = """ 
import sys
import os
from pathlib import Path

sys.path.insert(0, "{}")

import torch
import hpc

d = torch.load("{}")

func = getattr(hpc, d['func_name'])
args = d['args']
kwargs = d['kwargs']
gt = d['ret']

s = func(*args, **kwargs)

if isinstance(s, torch.Tensor):
  assert torch.allclose(s.view(torch.int8), gt.view(torch.int8))
elif isinstance(s, tuple):
  for i, e in enumerate(s):
    assert torch.allclose(s[i].view(torch.int8), gt[i].view(torch.int8))
else:
  print('=== type :', type(s))
  assert False
""".format(
        pypath, test_data_file
    )

    with open(file_name, "w") as fp:
        fp.write(text)


def sanitizer_check(file_name, check):
    cmd = f"compute-sanitizer --tool={check} --require-cuda-init=no python3 {file_name}"
    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    text = output.decode("utf-8")
    print(text)


class TraceHook(object):
    def __init__(self, checks, module_name):
        self.checks_ = checks
        self.module_name = module_name

    def _wrap_func(self, module, func_name):
        if not hasattr(module, func_name):
            return False

        org_func = getattr(module, func_name)

        def wrapped(*args, **kwargs):
            ret = org_func(*args, **kwargs)

            # save data
            tmp_data_file = tempfile.mktemp(prefix="tmp_hpc_" + func_name + "_", suffix=".pth")
            tmp_py_file = tmp_data_file.replace(".pth", ".py")
            pypath = os.path.realpath(list(Path(__file__).parent.glob("./build/lib.*/"))[0])

            save_data(tmp_data_file, "hpc", func_name, ret, args, kwargs)
            dump_test_py(tmp_py_file, tmp_data_file, pypath)

            for check in self.checks_:
                print(f"{check}...")
                sanitizer_check(tmp_py_file, check)

            os.unlink(tmp_data_file)
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


def pytest_configure(config):
    checks = get_checks()
    hooker = TraceHook(checks, "hpc")
    hooker.hook()
