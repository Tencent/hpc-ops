all:
	python3 setup.py build

PY_FILES=$(shell find hpc -name "*.py") $(shell find tests -name "*.py") setup.py
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")

format:
	python3 -m yapf --style=yapf -i $(PY_FILES)
	clang-format --style=google -i $(CC_FILES) $(CU_FILES)

