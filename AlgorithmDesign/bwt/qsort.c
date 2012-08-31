#include <stdio.h>
#include <stdlib.h>

int lessthan(const void * i, const void * j) {
  return *((int *)i) - *((int *)j);
}

int main() {
  int a[] = {3, 6, 0, 1, 0, -2, 5, 4};
  int k;

  qsort(a, 7, sizeof(int), &lessthan);
  
  for (k = 0; k < 8; k++)
    printf("%d, ", a[k]);
  printf("\n");

  return 0;
}
