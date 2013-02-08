#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>

bool is_member(std::vector<int> &v, int m) {
  for(std::vector<int>::iterator i=v.begin(); i!=v.end(); i++) {
    if(*i == m) return true;
  }
  return false;
}

int nrand(int n) {
  int a = 0;
  while(n--) { a += (RAND_MAX / 2) - rand(); }
  return abs(a);
}

int main(int argc, char **argv) {
  if(argc != 5) {
    std::cout << "USEGE : [genset seed/int size/int scale/int output-filename/string]" <<std::endl;
    return 0;
  }
  std::cout << "RND_MIN : " << 0 << " / RND_MAX : " << RAND_MAX;

  int seed = atoi(argv[1]);
  srand(seed);

  int size = atoi(argv[2]);
  int scale = atoi(argv[3]);
  std::vector<int> x(size), y(size);
  for(int i=0; i<size/2; ++i) {
    int a = 0;
    int b = 0;
    do {
      a = nrand(scale);
    } while(is_member(x, a));
    do {
      b = nrand(scale);
    } while(is_member(y, b));
    x[i] = a;
    y[i] = b;
  }
  for(int i=size/2; i<size; ++i) {
    int a = 0;
    int b = 0;
    do {
      a = nrand(scale) / 4 + RAND_MAX/8;
    } while(is_member(x, a));
    do {
      b = nrand(scale) / 4 + RAND_MAX/8;
    } while(is_member(y, b));
    x[i] = a;
    y[i] = b;
  }

  std::ofstream fout(argv[4]);
  if(fout.is_open()) {
    for(int i=0; i<size; ++i) {
      fout << x[i] << " " << y[i] << std::endl;
    }
  }

  return 0;
}
