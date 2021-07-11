//For comments go to cash_karp.cu
#include <stdio.h>

using namespace std;

double *x, *y, *d_x, *d_y;
const double step = 0.1;

__device__ double f(double x, double y) {
    return 3 * x * x + y;
}

__global__ void runge_kutta(double *x, double *y) {
    double k1, k2, k3, k4;
    int x_max = 0;
    int idx = threadIdx.x;
    
    while (x_max < 10) {    
        printf("X: %f\tY: %f\n", *x, *y); 
        
        k1 = step * f(x[idx], y[idx]);
        k2 = step * f((x[idx] + step/2), (y[idx] + k1/2));
        k3 = step * f((x[idx] + step/2), (y[idx] + k2/2));
        k4 = step * f((x[idx] + step), (y[idx] + k3));
        
        y[idx] += (k1 + 2*k2 + 2*k3 + k4) / 6;
        x[idx] += step;
        x_max++;
    }
}

__global__ void parent_runge_kutta(double *x, double *y) {
    int i = 0;
    
    printf("!---- Start of Process ----!\n");
        
    while (i < 2) {
        runge_kutta <<< 1, 1 >>>(x, y);
        i++;
        cudaDeviceSynchronize();
    }
    
    printf("!----- End of Process -----!\n");
}

int main() {
    size_t N = sizeof(double);

    cudaMalloc((void **) &d_x, N);
    cudaMalloc((void **) &d_y, N);
    
    x = (double *)malloc(N);
    y = (double *)malloc(N);
    
    cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);
    
    parent_runge_kutta <<< 1, 1 >>>(d_x, d_y);
    
    cudaMemcpy(x, d_x, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N, cudaMemcpyDeviceToHost);
    
    printf("Final Result = X: %g, Y: %.4g\n", *x, *y);
    
    free(x);    free(y);
    cudaFree(d_x);  cudaFree(d_y);
    return 0;
}
