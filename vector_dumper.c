#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM 15000000

int main(void){

    FILE *fptr = fopen("vector_nums.bin", "wb");

    if(fptr == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    srand(time(0));
    for(size_t i = 0; i < NUM; i++){
        double num1 = ((double) rand() / RAND_MAX) * 1000;
        double num2 = ((double) rand() / RAND_MAX) * 1000;

        fprintf(fptr, "%lf %lf\n", num1, num2);
    }

    fclose(fptr);

    return 0;
}