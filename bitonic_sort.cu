#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>

using namespace std;

#define THREADS 8
#define BLOCKS  8
#define n THREADS * BLOCKS

int *d_arr, *h_arr;

__global__ void bitonicsort(int *d_arr, int j, int k) {
    unsigned int idx, ixj;                                    // Χρησιμοποιούμε τα idx and ixj για την ταξινόμιση
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = idx ^ j;

    // Tα threads με τα μικρότερα idx ταξινομούν το array
    if (ixj > idx) {
        if ((idx&k) == 0) {                                     // Ταξινόμιση κατά αύξουσα σειρά
            if (d_arr[idx] > d_arr[ixj]) {                        // Εναλλαγή του idx με το ixj
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[ixj];
                d_arr[ixj] = temp;
            }
        }
        if ((idx&k) != 0) {                                     // Ταξινόμιση κατά φθίνουσα σειρά
            if (d_arr[idx] < d_arr[ixj]) {                        // Εναλλαγή του idx με το ixj
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[ixj];
                d_arr[ixj] = temp;
            }
        }
    }
}


// Καλεί το kernel, θέτει παραμέτρους, δίνει μνήμη, τυπώνει αποτέλεσμα, ελευθερώνει μνήμη
void runBitonicsort(int *h_arr, size_t N) {
    dim3 blocks(BLOCKS,1);
    dim3 threads(THREADS,1);
    
    cout << "Unsorted Array: ";
    for (int i = 0; i < n; i++)
        cout << h_arr[i] << " ";
    cout <<  endl;
   
    cudaMalloc((void **) &d_arr, N);
   
    cudaMemcpy(d_arr, h_arr, N, cudaMemcpyHostToDevice);
  
    for (int k = 2; k <= n; k <<= 1) {                      // Πάμε κάθε φορά το k προς τα αριστερά κατά 1 bit και το αντικαθιστούμε με το νέο k
        for (int j = k >> 1; j > 0; j--) {                  // Πάμε το k προς τα δεξιά κατά 1 bit κάθε φορά, ενώ μειώνουμε το j
            bitonicsort<<<blocks, threads>>>(d_arr, j, k);
        }
    }
    cudaDeviceSynchronize();
  
    cudaMemcpy(h_arr, d_arr, N, cudaMemcpyDeviceToHost);
  
    cout << "Sorted Array: ";
    for (int i = 0; i < n; i++)
        cout << h_arr[i] << " ";
    cout <<  endl;
    
    cudaFree(d_arr);
    free(h_arr);
}

int main(void) {
    srand(time(0));
    size_t N = n * sizeof(int);
    h_arr = (int*) malloc(N);
    
    cout << "Filling array with random numbers..." << endl;
    for (int i = 0; i < n; i++) 
        h_arr[i] = rand() % 100;

    runBitonicsort(h_arr, N);
        
    return 0;
}
