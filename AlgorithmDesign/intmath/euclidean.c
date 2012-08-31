#include <stdio.h>
#include <stdlib.h>

int gcd(int a, int b) {
  int c;
  
  do { /* */ printf("a = %d, b = %d, ", a, b);
    c = a % b; /* */ printf("a mod b = %d;\n", c);
    a = b;
    b = c;
  } while ( c != 0 );
  return a;
}

int main(int argc, char * argv[]) {
  int a, b, divisor;

  if (argc != 3) 
    return 1;
  a = atoi(argv[1]);
  b = atoi(argv[2]);
  divisor = gcd(a, b);

  printf("gcd of %d and %d is: %d.\n", a, b, divisor);

  return 0;
}
