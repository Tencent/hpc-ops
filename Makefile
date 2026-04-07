PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") $(shell find ./ -maxdepth 1 -name "*.py")
PY_TEST=$(shell find tests -name "test_*.py")
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
H_FILES=$(shell find src -name "*.h")
CUH_FILES=$(shell find src -name "*.cuh")

CSRC_FILES=$(CC_FILES) $(CU_FILES) $(CUH_FILES) $(H_FILES)


# ── Default: build for the GPU installed on this machine ─────────────────────
all:
	python3 setup.py build

# ── Arch-specific wheel targets ───────────────────────────────────────────────
# 'make sm90', 'make sm100', 'make sm89', etc.
# Output wheel goes to dist/
sm%:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	mkdir -p dist
	SM_ARCH=$* python3 -m build --wheel --no-isolation --outdir dist

# ── wheel: build wheel for auto-detected arch ─────────────────────────────────
wheel:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	python3 -m build --wheel --no-isolation

nvshmem:
	cmake -S 3rd/ucl/nvshmem -B 3rd/ucl/nvshmem-build \
		-DMLX5_LIB=/usr/lib64/libmlx5.so \
		-DCMAKE_CUDA_ARCHITECTURES=90a \
		-DNVSHMEM_IBGDA_SUPPORT=ON \
		-DCUDA_HOME=/usr/local/cuda \
		-DCMAKE_INSTALL_PREFIX=./3rd/ucl/nvshmem \
		-DNVSHMEM_SHMEM_SUPPORT=OFF \
		-DNVSHMEM_MPI_SUPPORT=OFF \
		-DNVSHMEM_PMI_SUPPORT=OFF \
		-DNVSHMEM_PMIX_SUPPORT=OFF \
		-DNVSHMEM_PMI2_SUPPORT=OFF \
		-DNVSHMEM_UCX_SUPPORT=OFF \
		-DNVSHMEM_USE_NCCL=OFF \
		-DNVSHMEM_BUILD_HYDRA_LAUNCHER=OFF \
		-DNVSHMEM_BUILD_PYTHON_LIB=OFF \
		-DNVSHMEM_BUILD_TXZ_PACKAGE=OFF \
		-DNVSHMEM_TIMEOUT_DEVICE_POLLING=OFF \
		-DNVSHMEM_USE_GDRCOPY=OFF \
		-DNVSHMEM_BUILD_EXAMPLES=OFF \
		-DNVSHMEM_BUILD_TESTS=OFF \
		-DNVSHMEM_NVTX=OFF
	make -C 3rd/ucl/nvshmem-build -j
	make install -C 3rd/ucl/nvshmem-build
	rm -rf 3rd/ucl/nvshmem-build

ucl: nvshmem
	cmake -S 3rd/ucl -B 3rd/ucl-build \
		-DCMAKE_CUDA_ARCHITECTURES=90a
	make -C 3rd/ucl-build -j
	make install -C 3rd/ucl-build
	rm -rf 3rd/ucl-build

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

sanitizer:$(PY_TEST)
	@rm -rf /dev/shm/tmp_hpc_*
	@for test in $^; do \
	  PYTORCH_NO_CUDA_MEMORY_CACHING=1 SANITIZER_CHECK=synccheck,memcheck,racecheck NV_SANITIZER_INJECTION_PORT_BASE=1111 python3 -m pytest -s -v --no-header --disable-warnings $$test || exit 1; \
	done

clean:
	rm -rf build dist hpc_ops.egg-info hpc.egg-info .pytest_cache tests/__pycache__ site __pycache__
