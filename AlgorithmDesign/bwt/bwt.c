#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* global var for passing the 3rd arg to the comparator */
char ** ARRAY;

/* lexcographic order for '\0'-ending-strings */
int lex_comp(const void * pt1, const void * pt2) {
  char * p = ARRAY[* (int *) pt1], * q = ARRAY[* (int *) pt2];
  char diff;

  for ( ; *p != '\0' && *q != '\0'; p++, q++) {
    diff = *p - *q;
    if ( diff != 0 )
      return diff;
  }
  return 0;
}


int main(int argc, char * argv[]) {
  int n, *order;
  int i, j, org;
  char * transformed;

  /* preparing an array for the argument's size */
  printf("Input: \"%s\"\n", argv[1]);
  n = strlen(argv[1]);
  ARRAY = (char **) malloc(sizeof(char *) * n);
  for (i = 0; i < n; i++) {
    ARRAY[i] = (char *) malloc(sizeof(char) * (n + 1));
  }

  transformed = (char *) malloc(sizeof(char) * (n+1));
  
  order = (int *) malloc(sizeof(int) * n);
  for (i = 0; i < n; i++)
    order[i] = i;
  
  /* copying the content of the string */
  strcpy(ARRAY[0],argv[1]);
  for (i = 1; i < n; i++) {
    for (j = 0; j < n; j++) {
      ARRAY[i][j] = ARRAY[0][(i + j) % n];
    }
    ARRAY[i][n] = '\0';
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%c", ARRAY[i][j]);
    }
    printf("\n");
  }

  printf("\n --- Result ---\n\n");

  qsort(order, n, sizeof(int), &lex_comp);

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%c", ARRAY[order[i]][j]);
    }
    printf("\n");
    if (order[i] == 0)
      org = i;
  }

  for (i = 0; i < n; i++)
    transformed[i] = ARRAY[order[i]][n-1];
  transformed[n] = '\0';
  printf("\nTransformed into \n\"%s\"\nand the original text is at row %d.\n",transformed, org+1);

  return 0; /* no error */
}
