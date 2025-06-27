PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") setup.py
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")
CUH_FILES=$(shell find src -name "*.cuh")

all:
	python3 setup.py build

wheel:
	python3 setup.py bdist_wheel

doc:
	python3 tools/generate_docs.py
	python3 -m mkdocs build

format:
	python3 -m yapf --style=yapf -i $(PY_FILES)
	clang-format --style=google -i $(CC_FILES) $(CU_FILES) $(CUH_FILES)

test:
	python3 -m pytest tests

clean:
	rm -rf build dist hpc.egg-info
