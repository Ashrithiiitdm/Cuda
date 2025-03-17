#include<stdio.h>
#include<assert.h>
#include<time.h>

#define N 15000000
#define FILENAME "vector_nums.bin"

__global__ void mul_vectors(double *a, double *b, double *c, int n){
    //Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Ensure the thread ID is less than the number of elements
    if(tid < n){
        c[tid] = a[tid] * b[tid];
    }
}

void init_array(double *a, double *b, double *c){

    FILE *fptr = fopen(FILENAME, "rb");
    if(fptr == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    for(size_t i = 0; i < N; i++) {
        fscanf(fptr, "%lf %lf\n", &a[i], &b[i]);
    }

    fclose(fptr);

}


double serial_mul(double *a, double *b, double *c){
    
    clock_t start = clock();
    for(size_t i = 0; i < N; i++){
        c[i] = a[i] * b[i];
    }
    clock_t end = clock();

    double time = (double) (end - start) / CLOCKS_PER_SEC;

    printf("Time taken by serial code: %lf\n", time);

    return time;
}


void verify(double *a, double *b, double *c){
    for(size_t i = 0; i < N; i++){
        assert(c[i] == a[i] * b[i]);
    }
    printf("All elements are correct\n");
}


int main(void){

    size_t bytes = N * sizeof(double);
    double *a, *b, *c;
    double *dev_a, *dev_b, *dev_c;

    a = (double *)malloc(bytes);
    b = (double *)malloc(bytes);
    c = (double *)malloc(bytes);


    init_array(a, b, c);

    cudaMalloc((void **) &dev_a, bytes);
    cudaMalloc((void **) &dev_b, bytes);
    cudaMalloc((void **) &dev_c, bytes);

    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

    double serial_time = serial_mul(a, b, c);

    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    clock_t start = clock();

    mul_vectors<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);

    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost);

    clock_t end = clock();

    double parallel_time = (double) (end - start) / CLOCKS_PER_SEC;


    printf("Time taken by parallel code: %lf\n", parallel_time);
    verify(a, b, c);

    printf("Speedup: %lf\n", serial_time / parallel_time);

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
