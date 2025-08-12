PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") setup.py
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
CUH_FILES=$(shell find src -name "*.cuh")
CUH_FILES=$(shell find src -name "*.h")

CSRC_FILES=$(CC_FILES) $(CU_FILES) $(CUH_FILES) $(H_FILES)

TORCH_PATH=$(shell python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')

all:
	python3 setup.py build

cmake:
	cmake -S . -B build -DCMAKE_PREFIX_PATH=$(TORCH_PATH) ..
	cmake --build build --parallel

wheel:
	python3 setup.py bdist_wheel

doc:
	python3 tools/generate_docs.py
	python3 -m mkdocs build

format:
	python3 -m black --line-length 100 $(PY_FILES)
	clang-format --style=file -i $(CSRC_FILES)
	python3 -m cpplint --quiet $(CSRC_FILES)

format-check:
	python3 -m black --check --line-length 100 $(PY_FILES)
	clang-format --style=file --dry-run -Werror $(CSRC_FILES)
	python3 -m cpplint $(CSRC_FILES)

test:
	python3 -m pytest tests

clean:
	rm -rf build dist hpc_ops.egg-info hpc.egg-info site
