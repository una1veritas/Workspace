#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* global var for passing the args to the comparator and the rotator */
char * TEXT;
int LENGTH;

const char rota(int head, int offset) {
  return TEXT[(head + offset) % LENGTH];
}


/* lexcographic order between '\0'-ending-strings */
int lex_comp(const void * pt1, const void * pt2) {
  int p = * (int *) pt1, q = * (int *) pt2;
  char diff;
  int i;

  for (i = 0; rota(p, i) != '\0' && rota(q, i) != '\0'; i++) {
    diff = rota(p, i) - rota(q, i);
    if ( diff != 0 )
      return diff;
  }
  return 0;
}


int main(int argc, char * argv[]) {
  int *order;
  int i, j, org;
  char * transformed;

  /* preparing an array for the argument's size */
  printf("Input: \"%s\"\n", argv[1]);
  TEXT = argv[1]; /* copying the pointer, not contents of the string */
  LENGTH = strlen(TEXT);
  transformed = (char *) malloc(sizeof(char) * (LENGTH+1));
  
  order = (int *) malloc(sizeof(int) * LENGTH);
  for (i = 0; i < LENGTH; i++)
    order[i] = i;
  
  for (i = 0; i < LENGTH; i++) {
    for (j = 0; j < LENGTH; j++) {
      printf("%c", rota(i,j));
    }
    printf("\n");
  }

  printf("\n --- Result ---\n\n");

  qsort(order, LENGTH, sizeof(int), &lex_comp);

  for (i = 0; i < LENGTH; i++) {
    for (j = 0; j < LENGTH; j++) {
      printf("%c", rota(order[i],j));
    }
    printf("\n");
    if (order[i] == 0)
      org = i;
  }

  for (i = 0; i < LENGTH; i++)
    transformed[i] = rota(order[i],LENGTH-1);
  transformed[LENGTH] = '\0';  /* don't forget placing the terminal */
  printf("\nTransformed into \n\"%s\"\nand the original text is at row %d.\n",transformed, org+1);

  return 0; /* no error */
}
