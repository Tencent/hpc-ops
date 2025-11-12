CU_FILES :=$(shell find ./ -name "*.cu")
EXE_FILES:= $(CU_FILES:.cu=)

all:$(EXE_FILES)

$(EXE_FILES):%:%.cu $(CU_FILES)
	nvcc -o $@ $< -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo -arch=sm_90a -I../../ -I../../3rd/cutlass/include -lcuda
