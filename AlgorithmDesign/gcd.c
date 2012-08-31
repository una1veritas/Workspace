#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
  
  printf("The number of args: %d\n", argc);
  printf("%s\n%s\n", argv[0], argv[1]);
  printf("As int: %d\n", atoi(argv[2]));

  return 0;
}
