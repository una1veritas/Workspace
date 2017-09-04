#include <stdio.h>

class test {
public:
  int * i;
  int * j;
  test() {
    j = new int[10];
  }

  ~test() {
    delete j;
  }
};

void main(int argc, char * argv[]) {
  test * t;
  
  t = new test();
  printf("%d #.\n", t.j[2]);
  delete t;
}
