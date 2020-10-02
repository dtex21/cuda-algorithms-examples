all:bfs
bfs:bfs.cu
	nvcc bfs.cu -o bfs
