#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char * argv[]) {
  int z = 4;
  int c = 1;
  int s = 11;

  if ( argc > 3 ) {
    s = atoi(argv[3]);
  }
  if (argc > 2) {
    c = atoi(argv[2]);
  }
  if (argc > 1) {
    z = atoi(argv[1]);
  }

  srandom(s);
  for ( ; c > 0; c--) {
    printf("%d\t%d\n", 
	   (int) (random() % (int) pow(10,z)), 
	   (int) (random() % (int) pow(10,z)) );
  }
}
