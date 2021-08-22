#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

//Using a 2D Vector, we store 3-component vectors in each. 3 blocks per row. Or I guess N block being the number of rows I want?
//With two 2D vectors, populate, and iterate, dotting respective rows.

__global__ void dpDevice(int *A, int *B, int *C, int N) {
    __shared__ int cache[256];
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;

    int temp = 0;
    while (idx < N) {
        temp += A[idx] + B[idx];

        idx += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    int i = blockDim.x/2;
    while (i!= 0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x]  +=  cache[threadIdx.x+i];
        }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0) {
        atomicAdd(C,cache[0]);
    }

}

void dpHost(int *A, int *B, int *C, int N) {
    int *temp;
    temp = 0;
    for(int i = 0; i<N;i++) {
        temp += A[i] * B[i];
    }
    C = temp;
}

void random_ints(int *a, int size) {
    for(int i = 0; i<size; i++) {
        a[i] = rand() % 100 + 1;
    }
}

#define checkCudaErrors(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

using namespace std;

int main(void) {

    //Look for CUDA supported device
    int cuda_devices = 0;
    cudaGetDeviceCount(&cuda_devices);

    if (cuda_devices == 0) {
        cout << "No CUDA Devices Found :( Exiting Program... \n";
             
    } else {
        cout << "CUDA Device(s) found: " << cuda_devices << endl;  
    }
    cout << "1" << endl;
    //Blocks and thread set-up
    int N = 1024;
    int block_size = 32;
    dim3 threadsPerBlock(block_size,block_size);
    dim3 blocksPerGrid(N / block_size, N / block_size);

    //initialize host and device variables
    int *a, *b, *c;
    int *d_a,*d_b,*d_c,*hCuda_C;
    int size = sizeof(int) * (N*N);

    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMemset(d_c,0,sizeof(int));
    cout << "1" << endl;
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(sizeof(int));

    for(int i = 0; i<size; i++) {
        a[i] = rand() % 100 + 1;
        b[i] = rand() % 100 + 1;
    }
    unsigned int start_time = clock();
    cout << "1" << endl;
    dpHost(a,b,c,N);

    unsigned int elapsedTime = clock() - start_time;
    float msecPerMatrixMulCpu = elapsedTime;

    cout << "CPU time: " << msecPerMatrixMulCpu << endl;
    cout << "CPU Result: " << *c << endl;
    cout << "1" << endl;
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    dpDevice<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_c,N);
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal;

    cout << "GPU time: " << msecPerMatrixMul << endl;
    checkCudaErrors(cudaMemcpy(hCuda_C, d_c, size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cout << "GPU Result: " << *hCuda_C << endl;
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cout << "1" << endl;
    return 0;

}
