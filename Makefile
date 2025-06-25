all:
	python3 setup.py build

PY_FILES=$(shell find hpc -name "*.py")
CC_FILES=$(shell find src -name "*.cc")
CU_FILES=$(shell find src -name "*.cu")

format:$(PY_FILES)
	python3 -m yapf --style=yapf -i $^ 
	clang-format --style=google -i $^

