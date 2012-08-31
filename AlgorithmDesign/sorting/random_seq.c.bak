#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char * argv[]) {
  int n = 7;
  int range = 11;
  int seed;
  int i;

  if (argc > 1)
    n = atoi(argv[1]);
    range = n;
  if (argc > 2) 
    range = atoi(argv[2]);
  if (argc > 3) {
    seed = atoi(argv[3]);
  } else {
    seed = (int) time(NULL);
  }
  srandom(seed);

  for (i = 0; i < n; i++)
    printf("%d\n", (int) random() % range);

  return 0;
}
