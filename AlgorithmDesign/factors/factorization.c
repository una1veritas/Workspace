#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
  int n = atoi(argv[1]);
  int d;

  printf("Input: %d\n",n);
  if (! (n > 0) )
    return 0;
  printf("%d, ",1);
  for (d = 2; ! (n < d); ) {
    if (n % d == 0) {
      n = n / d;
      printf("%d, ", d);
    } else {
      d++;
    }
  }
  printf("\n");
  return 0;
}

