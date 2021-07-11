#include <stdio.h>
#include <math.h>

using namespace std;

double *x, *y, *d_x, *d_y;
const double step = 0.1;

__device__ double f(double x, double y) {                   //Returns the differential equation
    return 3 * x * x + y;
}
  // 
 // Child
// 

__global__ void cash_karp(double *x, double *y) {
    double k1, k2, k3, k4, k5, k6, k5s;
    //double k4s, err;
    int x_max = 0;
    int idx = threadIdx.x;
    
    while (x_max < 10) {                                    //Calculates the first 10 results
        printf("X: %f\tY: %f\n", *x, *y); 
        
        k1 = step * f(x[idx], y[idx]);
        k2 = step * f((x[idx] + step/5), (y[idx] + k1/5));
        k3 = step * f((x[idx] + 3*step/10), (y[idx] + 3*k1/40 + 9*k2/40));
        k4 = step * f((x[idx] + 3*step/5), (y[idx] + 3*k1/10 - 9*k2/10 + 6*k3/5));
        k5 = step * f((x[idx] + step), (y[idx] - 11*k1/54 + 5*k2/2 - 70*k3/27 + 35*k4/27));
        k6 = step * f((x[idx] + 7*step/8), (y[idx] + 1631*k1/55296 + 175*k2/512 + 575*k3/13824 + 44275*k4/110592 + 253*k5/4096));
        
        //The following is needed to calculate the error. I ignore it, so err = 0.
        //k4s = 37*k1/378 + 250*k3/621 + 125*k4/594 + 512*k6/1771;
        k5s = 2825*k1/27648 + 18575*k3/48384 + 13525*k4/55296 + 277*k5/14336 + k6/4;
        //err = max(fabs(k4s-k5s));        
        
        y[idx] += k5s;
        x[idx] += step;
        x_max++;
    }
}
  // 
 // Parent
// 

__global__ void parent_cash_karp(double *x, double *y) {
    int i = 0;
    
    printf("!---- Start of Process ----!\n");
        
    while (i < 2) {                                         //Number of children is 2
        cash_karp <<< 1, 1 >>>(x, y);
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

    parent_cash_karp <<< 1, 1 >>>(d_x, d_y);

    cudaMemcpy(x, d_x, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N, cudaMemcpyDeviceToHost);

    printf("Final Result = X: %g, Y: %.4g\n", *x, *y);

    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
