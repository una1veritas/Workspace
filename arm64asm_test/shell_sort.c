/*
 * shell_sort.c
 */

 #include <string.h>
 #include "shell_sort.h"

void shell_sort(const char * str[], unsigned int a[], unsigned int n) {
    // Start with a large gap, then reduce it step by step
    for (unsigned int gap = n>>1; gap > 0; gap >>= 1) {
        // Perform a "gapped" swap, from shortest length-1 chain to longer ones
        for (unsigned int i = gap; i < n; ++i) {
            // Current element to be placed correctly
            int tix = a[i];
            // j - gap < j
            // Shift elements that are greater than temp to make space
            unsigned int j;
            for (j = i; j >= gap && (strcmp(str[tix], str[a[j - gap]]) < 0) ; j -= gap ) {
                a[j] = a[j - gap];
            }
            a[j] = tix;
        }
    }
}
