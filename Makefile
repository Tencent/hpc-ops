PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") $(shell find ./ -maxdepth 1 -name "*.py")
PY_TEST=$(shell find tests -name "test_*.py")
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
H_FILES=$(shell find src -name "*.h")
CUH_FILES=$(shell find src -name "*.cuh")

CSRC_FILES=$(CC_FILES) $(CU_FILES) $(CUH_FILES) $(H_FILES)

# Architectures the build supports. The same list is in CMakeLists.txt and
# setup.py; adding one also needs an HPC_ARCH_EVAL_<arch> block in
# src/utils/utils.h.
KNOWN_ARCHS=90 100 103 120
ALL_ARCHS=$(shell echo $(KNOWN_ARCHS) | tr ' ' ,)

# Target architectures. Each gets its own module; a build for several of them
# produces several .so files in one package, and hpc picks one at import time.
# Unset means the architectures of the local GPUs, detected by setup.py — except
# for `wheel`, which defaults to every architecture in KNOWN_ARCHS, because a
# wheel is a release artifact rather than a local build.
#   make                  build a module for the local architecture (dev)
#   SM_ARCH=90,100 make   build two modules, sm90 and sm100
#   make wheel            wheel covering every architecture in KNOWN_ARCHS
#   make sm90             wheel for sm90 only
#   SM_ARCH=90,100 make wheel   wheel for that subset
export SM_ARCH


all:
	python3 setup.py build

# A release wheel covers everything unless asked for less. The target-specific
# default is overridden by SM_ARCH from the environment or the command line,
# which is how the per architecture shortcuts below narrow it.
wheel: SM_ARCH:=$(if $(strip $(SM_ARCH)),$(SM_ARCH),$(ALL_ARCHS))
wheel:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	python3 -m build --wheel --no-isolation

# Per architecture shortcuts: make sm90 == SM_ARCH=90 make wheel
$(foreach arch,$(KNOWN_ARCHS),sm$(arch)):
	@$(MAKE) --no-print-directory wheel SM_ARCH=$(patsubst sm%,%,$@)

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

.PHONY: all wheel format format-check test sanitizer clean \
        $(foreach arch,$(KNOWN_ARCHS),sm$(arch))
