/*
 * heapchecker.c
 */
#include <stdio.h>
#include <stdlib.h>

#include "shell_sort.h"

int main(int argc, char * argv[]) {
    unsigned int n = argc - 1;
    unsigned int a[n];
    const char ** strs = (const char **) argv + 1;
    for (unsigned int i = 0; i < n; i++) {
        a[i] = i;
    }

    shell_sort(strs, a, n);

    printf("Result: \n");
    for (unsigned int i=0; i < n; i++)
        printf("%s ", strs[a[i]]);
    printf("\n");

    return EXIT_SUCCESS;
}