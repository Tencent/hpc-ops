PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") $(shell find ./ -maxdepth 1 -name "*.py")
PY_TEST=$(shell find tests -name "test_*.py")
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
H_FILES=$(shell find src -name "*.h")
CUH_FILES=$(shell find src -name "*.cuh")

CSRC_FILES=$(CC_FILES) $(CU_FILES) $(CUH_FILES) $(H_FILES)


all:
	python3 setup.py build

wheel:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	python3 -m build --wheel --no-isolation

format:
	python3 -m black --line-length 100 $(PY_FILES)
	clang-format --style=file -i $(CSRC_FILES)
	python3 -m cpplint --quiet $(CSRC_FILES)

format-check:
	python3 -m black --check --line-length 100 $(PY_FILES)
	clang-format --style=file --dry-run -Werror $(CSRC_FILES)
	python3 -m cpplint $(CSRC_FILES)

test:$(PY_TEST)
	@for test in $^; do \
	  python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

sanitizer:$(PY_TEST)
	@rm -rf /dev/shm/tmp_hpc_*
	@for test in $^; do \
	  PYTORCH_NO_CUDA_MEMORY_CACHING=1 SANITIZER_CHECK=synccheck,memcheck,racecheck NV_SANITIZER_INJECTION_PORT_BASE=1111 python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

clean:
	rm -rf build dist hpc_ops.egg-info hpc.egg-info .pytest_cache tests/__pycache__ site
