#include <stdio.h>
#include <stdlib.h>

/* datatype 型の定義 */
typedef int datatype;

int equals(datatype x, datatype y) {
  return x == y;
}

int lessthan(datatype x, datatype y) {
  return x < y;
}

/* ArraySet の定義 */
typedef struct ArraySet {
  datatype * elements;
  int capacity;
  int tally;
} ArraySet;

void setup_ArraySet(ArraySet * set, int maxsize) {
  set->elements = (datatype *) malloc(sizeof(datatype) * maxsize);
  set->capacity = maxsize;
  set->tally = 0;
}

void dispose_ArraySet(ArraySet * set) {
  free(set->elements);
}

int find(ArraySet * set, datatype x) {
  int i;
  for (i = 0; i < set->tally; i++) {
    if ( equals(set->elements[i], x) ) 
      return i;
  }
  return -1;
}

int add(ArraySet * set, datatype x) {
  int i;
  if ( (i = find(set, x)) != -1)
    return i;
  if (! (set->tally < set->capacity) )
    return -1;
  set->elements[set->tally] = x;
  set->tally++;
  return set->tally - 1;
}

int delete(ArraySet * set, datatype x) {
  int i;
  if ( (i = find(set, x)) == -1)
    return i;
  set->tally--;
  set->elements[i] = set->elements[set->tally];
  return i;
}

int size(ArraySet * set) {
  return set->tally;
}

datatype element(ArraySet * set, int pos) {
  return set->elements[pos];
}

/* テスト用プログラム */
int main(int argc, char * argv[]) {
  ArraySet s;
  int i;
  setup_ArraySet(&s, 10);
  for (i = 1; i < argc; i++) {
    add(&s, atoi(argv[i]));
  }

  delete(&s, 8);
  delete(&s, 22);

  printf("size = %d.\n", size(&s));

  for ( i = 0; i < size(&s); i++) {
    printf("%d, ", element(&s, i));
  }
  printf("\n");

  dispose_ArraySet(&s);
}
