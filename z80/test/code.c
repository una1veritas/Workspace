#include <stdio.h>

int main() {

  for(int i = 0; i < 256; i++) {
    printf("const char code_%02X[] PROGMEM = \"     \";\n",i);
  }

  printf("const char* const string_table[] PROGMEM = {\n");
  for(int i = 0; i < 256; i += 4) {
    printf("\tcode_%02X, code_%02X, code_%02X, code_%02X, \n",i,i+1,i+2,i+3);
  }
  printf("};\n");
  return 1;
}
