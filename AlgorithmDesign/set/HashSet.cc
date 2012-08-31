#include <stdio.h>
#include <stdlib.h>

#include <string.h>

/* datatype 型の定義 */
typedef char * datatype;

int equals(datatype x, datatype y) {
  return strcmp(x,y) == 0;
}

int hashCode(datatype x) {
  return (int) x[0] + 31 * x[1] + 31* 31 * x[2];
}

/* HashSet の定義 */
struct HashSet {
  datatype * elements;
  int capacity;
  int tally;

  HashSet(int maxsize) {
    elements = new datatype[maxsize];
    capacity = maxsize;
    tally = 0;
  }

  ~HashSet() {
    delete elements;
  }

  int find(datatype x) {
    int i, p;
    for (i = 0, p = hashCode(x); i < tally; i++) {
      if ( equals(elements[(p + i) % capacity], x) ) 
	return i;
      if ( elements[ (p + i) % capacity ] == NULL )
	return i;
    }
    return -1;
  }
  
  int insert(datatype x) {
    int i;
    i = find(x);
    elements[i] = x;
    tally++;
  }

  int remove(datatype x) {

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
  HashSet s(10);
  int i;

  for (i = 1; i < argc; i++) {
    s.insert(argv[i]);
  }

  for ( i = 0; i < s.size(); i++) 
    printf("%d, ", s.element(i));
  printf("\n");

  /*
  s.remove(8);
  s.remove(22);
  */  
  printf("size = %d.\n", s.size());
  
  for ( i = 0; i < s.size(); i++) 
    printf("%d, ", s.element(i));
  printf("\n");
  
}
