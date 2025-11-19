PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") setup.py
PY_TEST=$(shell find tests -name "test_*.py")
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
H_FILES=$(shell find src -name "*.h")
CUH_FILES=$(shell find src -name "*.cuh")

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

test:$(PY_TEST)
	@for test in $^; do \
	  python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

sanitizer:sanitizer-memcheck sanitizer-synccheck sanitizer-racecheck sanitizer-initcheck
	echo "do all sanitizer check"

sanitizer-memcheck:$(PY_TEST)
	@for test in $^; do \
	  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool=memcheck --require-cuda-init=no --kernel-name regex="hpc.+" python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

sanitizer-synccheck:$(PY_TEST)
	@for test in $^; do \
	  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool=synccheck --require-cuda-init=no --kernel-name regex="hpc.+" python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

sanitizer-racecheck:$(PY_TEST)
	@for test in $^; do \
	  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool=racecheck --require-cuda-init=no --kernel-name regex="hpc.+" python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

sanitizer-initcheck:$(PY_TEST)
	@for test in $^; do \
	   true || PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool=initcheck --require-cuda-init=no --check-api-memory-access=no --print-limit=2 --force-blocking-launches python3 -m pytest -v --no-header --disable-warnings $$test || exit 1; \
	done

clean:
	rm -rf build dist hpc_ops.egg-info hpc.egg-info .pytest_cache tests/__pycache__ site
