#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 15000000
#define FILENAME "vector_nums.bin"
#define BLOCK_SIZE 512


__global__ void dotProduct(double *a, double *b, double *c, int n) {

    __shared__ double temp[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tempIndex = threadIdx.x;

    double tempSum = 0.0;


    while(tid < n) {
        tempSum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }


    temp[tempIndex] = tempSum;
    __syncthreads();


    int i = blockDim.x / 2;
    while(i != 0) {
        if(tempIndex < i)
            temp[tempIndex] += temp[tempIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(tempIndex == 0)
        c[blockIdx.x] = temp[0];
}


void init_array(double *a, double *b) {

    FILE *fptr = fopen(FILENAME, "rb");
    if(fptr == NULL) {
        printf("Error opening file %s\n", FILENAME);
        exit(1);
    }
    for(size_t i = 0; i < N; i++) {
       fscanf(fptr, "%lf %lf\n", &a[i], &b[i]);
    }
    fclose(fptr);
}


double serial_dot(double *a, double *b) {

    clock_t start = clock();
    double sum = 0.0;

    for(size_t i = 0; i < N; i++){
        sum += a[i] * b[i];
    }
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial dot product: %lf\n", sum);
    //printf("Time taken by serial code: %lf seconds\n", time_taken);
    return time_taken;
}



int main(void) {
    size_t bytes = N * sizeof(double);

    double *a, *b;
    double *dev_a, *dev_b, *dev_partial;
    double parallel_result = 0.0;
    double *partial;

    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    if(a == NULL || b == NULL) {
        printf("Host memory allocation failed\n");
        return 1;
    }


    init_array(a, b);


    double serial_time = serial_dot(a, b);
    printf("Serial dot product execution time: %lf\n", serial_time);

    cudaMalloc((void**)&dev_a, bytes);
    cudaMalloc((void**)&dev_b, bytes);


    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc((void**)&dev_partial, blocks * sizeof(double));
    partial = (double*)malloc(blocks * sizeof(double));

    double parallel_start = clock();
    dotProduct<<<blocks, threads>>>(dev_a, dev_b, dev_partial, N);
    cudaDeviceSynchronize();
    double parallel_end = clock();
    cudaMemcpy(partial, dev_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);


    for (int i = 0; i < blocks; i++) {
        parallel_result += partial[i];
    }


    double parallelTime = (double)(parallel_end - parallel_start) / CLOCKS_PER_SEC;
    printf("Parallel dot product time: %lf seconds\n", parallelTime);
    printf("Parallel dot product result: %lf\n", parallel_result);
    printf("Speedup: %lf\n", serial_time / parallelTime);


    free(a);
    free(b);
    free(partial);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial);
    return 0;
}
