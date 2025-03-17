#include<stdio.h>
#include<assert.h>
#include<time.h>
#include<cuda_runtime.h>
#include<cuda.h>
#define N 15000000
#define FILENAME "array_nums.bin"


__global__ void sumOfElements(double *a, double *res, int n){
    __shared__ double sdata[512];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        sdata[tid] = a[i];
    }
    else{
        sdata[tid] = 0.0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        res[blockIdx.x] = sdata[0];
    } 
} 

void init_array(double *a){
    FILE *fptr = fopen(FILENAME, "rb");

    if(fptr == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    fread(a, sizeof(double), N, fptr);
    fclose(fptr);
}

void serial_sum(double *a, double *res){
    double sum = 0.0;
    double start = clock();
    for(int i = 0; i < N; i++){
        sum += a[i];
    }
    double end = clock();
    *res = sum;


    double time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Serial sum: %f\n", sum);
    printf("Time taken by serial code: %f\n", time);
    return time;
}

int main(void){

    size_t bytes = N * sizeof(double);

    double *a, *res;
    double *dev_a, *dev_res;

    a = (double *)malloc(bytes);
    res = (double *)malloc(sizeof(double));

    init_array(a);

    double serial_time = serial_sum(a, res);

    int blockSize = 512;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaMalloc((void**)&dev_a, N * sizeof(double));
    cudaMalloc((void**)&dev_res, (N / blockSize + 1) * sizeof(double));

    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);

    double start = clock();

    sumOfElements<<<gridSize, blockSize>>>(dev_a, dev_res, N);
    cudaDeviceSynchronize();

    double end = clock();
    cudaMemcpy(res, dev_res, (N / blockSize + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for(int i = 0; i < N / blockSize + 1; i++){
        sum += res[i];
    }

    printf("Parallel sum: %f\n", sum);
    double parallel_time = (double) (end - start) / CLOCKS_PER_SEC;

    printf("Time taken by parallel code: %f\n", parallel_time);

    printf("Speedup: %f\n", serial_time / parallel_time);

    cudaFree(dev_a);
    cudaFree(dev_res);
    free(a);
    free(res);

    return 0;
}