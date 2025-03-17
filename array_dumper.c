#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 15000000

int main(void){

    FILE *fptr = fopen("vector_nums.bin", "wb");

    if(fptr == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    srand(time(0));
    for(size_t i = 0; i < N; i++){
        double num = ((double) rand() / RAND_MAX) * 1000;
        fprintf(fptr, "%lf\n", num);
    }

    fclose(fptr);

    return 0;
}