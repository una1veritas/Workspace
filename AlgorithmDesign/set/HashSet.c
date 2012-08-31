#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* datatype 型の定義 */
typedef char * datatype;

int equals(datatype x, datatype y) {
  return strcmp(x,y) == 0;
}

int hash(datatype x) {
  int i, h = 0;
  for (i = 0; i < strlen(x) && i < 4; i++) 
    h = h * 31 + x[i];
  return h;
}

/* HashSet の定義 */
typedef struct HashSet {
  datatype * elements;
  int capacity;
  int tally;
} HashSet;

void initialize_HashSet(HashSet * set, int maxsize) {
  int i;
  set->elements = (datatype *) malloc(sizeof(datatype) * maxsize);
  for (i = 0; i < maxsize; i++) 
    set->elements[i] = NULL;
  set->capacity = maxsize;
  set->tally = 0;
}

void dispose_HashSet(HashSet * set) {
  free(set->elements);
}

int find(HashSet * set, datatype x) {
  int i, p, h = hash(x);
  for (i = 0; i < set->capacity; i++) {
    p = (i + h) % set->capacity;
    if ( set->elements[p] == NULL )
      return p;
    if ( equals(set->elements[p], x) ) 
      return p;
  }
  return -1; /* バグによるエラー */
}

int add(HashSet * set, datatype x) {
  int i, j, 
    p = find(set, x);
  if ( set->elements[p] == NULL ) {
    if (! (set->tally + 1 < set->capacity) )
      return -1; /* capacity が不十分でエラー */
    set->elements[p] = x;
    set->tally++;
  } 
  return p;
}

int delete(HashSet * set, datatype x) {
  int q, 
    p = find(set, x);
  if ( set->elements[p] == NULL ) 
    return -1; /* 削除対象が見つからない */
  set->elements[p] = NULL;
  set->tally--;
  while (1) {
    p = (p + 1) % set->capacity;
    if ( set->elements[p] == NULL )
      break;
    q = find(set, set->elements[p]);
    if ( set->elements[q] == NULL ) {
      set->elements[q] = set->elements[p];
      set->elements[p] = NULL;
    } 
  }
  return p;
}

int size(HashSet * set) {
  return set->tally;
}

datatype element(HashSet * set, int pos) {
  return set->elements[pos];
}

/* テスト用プログラム */
int main(int argc, char * argv[]) {
  HashSet s;
  int i, j;

  initialize_HashSet(&s, 11);
  for (i = 1; i < argc; i++) {
    add( &s, argv[i] );
    printf("%s [%d],\n", argv[i], hash(argv[i]) ); 
  }

  printf("The number of elements = %d.\n", size(&s));

  for ( i = 0; i < s.capacity; i++) {
    printf("%s, ", element(&s, i));
  }
  printf("\n\n");

  delete(&s, "dog.s");
  for ( i = 0; i < s.capacity; i++) {
    printf("%s, ", element(&s, i));
  }
  printf("\n\n");

  delete(&s, "is");
  for ( i = 0; i < s.capacity; i++) {
    printf("%s, ", element(&s, i));
  }
  printf("\n\n");

  delete(&s, "dog.");


  for ( i = 0; i < s.capacity; i++) {
    printf("%s, ", element(&s, i));
  }
  printf("\n\n");

  printf("The number of elements = %d.\n", size(&s));
  for (i = 1; i < argc; i++) {
    j = find(&s, argv[i]);
    if ( element(&s, j) != NULL )
      printf("%s, ", element( &s, find(&s, argv[i]) ));
  }
  printf("\n");

  dispose_HashSet(&s);
}
