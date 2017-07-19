/*
 * sieve.c
 *
 *  Created on: 2017/07/19
 *      Author: sin
 */

#include <stdio.h>
#include <time.h>

#define Limit 0x6fffffUL
/* 0x5fffff */

int main() {

  unsigned char c[Limit];
  unsigned long i, l;

  printf("unsigned long size: %ld\n",sizeof(unsigned long));

  for (i = 0; i < Limit; i++)
    c[i] = 1;
  c[0] = 0;
  c[1] = 0;
  printf("initialized.\n");

  clock();

  for (i = 2; i < Limit; i++) {
    for (l = 2; i*l < Limit; l++) {
      c[i*l] = 0;
    }
  }
  for (i = Limit - 1; (c[i] == 0) && (i > 0); i--);

  printf("clock -- %f\n",(float) clock()/ (float) CLOCKS_PER_SEC);
  printf("Largest: %ld  Until: %ld\n",i, Limit);
}
