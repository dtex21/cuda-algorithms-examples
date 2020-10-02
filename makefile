all:rk
rk:runge_kutta.cu
	nvcc runge_kutta.cu -o rk -arch=sm_35 -rdc=true -lcudadevrt
