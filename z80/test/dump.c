#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main(int argc, char * argv[] ) {

  FILE * fp = fopen(argv[1], "rb");
  if ( fp == NULL ) {
    printf("open failed.\n");
    return EXIT_FAILURE;
  }

  int b;
  uint16_t cnt = 0;
  while ( (b = fgetc(fp)) != EOF ) {
    if ( !(cnt & 0x07) ) {
      printf("/* %04x */ ", cnt);
    }
    uint8_t ub = (uint8_t)b;
    printf("0x%02x, ", ub);
    ++cnt;
    if ( !(cnt & 0x07) )
      printf("\n");
  }
  fclose(fp);
  printf("\n");
  return EXIT_SUCCESS;
}
