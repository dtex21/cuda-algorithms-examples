all:bfs rk bs ck euler flag
bfs:bfs.cu
	nvcc bfs.cu -o bfs
	
rk:runge_kutta.cu
	nvcc runge_kutta.cu -o rk -arch=sm_35 -rdc=true -lcudadevrt
	
bs:bitonic_sort.cu
	nvcc bitonic_sort.cu -o bs
	
ck:cash_karp.cu
	nvcc cash_karp.cu -o ck -arch=sm_35 -rdc=true -lcudadevrt
	
euler:euler.cu
	nvcc euler.cu -o euler -arch=sm_35 -rdc=true -lcudadevrt
	
flag:flag.cu
	nvcc flag.cu -o flag -I/opt/cuda/samples/common/inc -I/usr/include -lGL -lGLU -lglut -lX11
