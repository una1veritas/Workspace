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
struct ArraySet {
  datatype * elements;
  int capacity;
  int tally;

  ArraySet(int maxsize) {
    elements = new datatype[maxsize];
    capacity = maxsize;
    tally = 0;
  }

  ~ArraySet() {
    delete elements;
  }

  int find(datatype x) {
    int i;
    for (i = 0; i < tally; i++) 
      if ( equals(elements[i], x) ) 
	return i;
    return -1;
  }
  
  int insert(datatype x) {
    int i;
    if ( (i = find(x)) != -1)
      return i;
    if (! (tally < capacity) )
      return -1;
    elements[tally] = x;
    tally++;
    return tally - 1;
  }

  int remove(datatype x) {
    int i;
    if ( (i = find(x)) == -1)
      return i;
    tally--;
    elements[i] = elements[tally];
    return i;
  }

  int size() {
    return tally;
  }
  
  datatype element(int pos) {
    return elements[pos];
  }
  
};

/* テスト用プログラム */
int main(int argc, char * argv[]) {
  ArraySet s(10);
  int i;

  for (i = 1; i < argc; i++)
    s.insert(atoi(argv[i]));

  for ( i = 0; i < s.size(); i++) 
    printf("%d, ", s.element(i));
  printf("\n");

  s.remove(8);
  s.remove(22);
  
  printf("size = %d.\n", s.size());
  
  for ( i = 0; i < s.size(); i++) 
    printf("%d, ", s.element(i));
  printf("\n");
  
}
