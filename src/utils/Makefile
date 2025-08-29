all:test

test:test.cu
	nvcc -o $@ $< -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo -arch=sm_90a -I../../ -I../../3rd/cutlass/include -lcuda
