#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <list>

int list_max(std::list<int> &list) {
  int a = list.front();
  for(std::list<int>::iterator i=list.begin(); i!=list.end(); i++) {
    if(*i > a) a = *i;
  }
  return a;
}

int list_min(std::list<int> &list) {
  int a = list.front();
  for(std::list<int>::iterator i=list.begin(); i!=list.end(); i++) {
    if(*i < a) a = *i;
  }
  return a;
}

double list_average(std::list<int> &list) {
  double a = 0.0;
  for(std::list<int>::iterator i=list.begin(); i!=list.end(); i++) {
    a += (double)*i;
  }
  return a / list.size();
}

int main(int argc, char **argv) {
  if(argc != 2) {
    std::cout << "USAGE : [setinfo.exe input-file/string]" << std::endl;
    return 0;
  }

  std::ifstream fin(argv[1]);
  if(fin.is_open()) {
    // GET
    std::list<int> x;
    std::list<int> y;
    int a, b;
    fin >> a;
    fin >> b;
    while(!fin.eof()) {
      x.push_back(a);
      y.push_back(b);
      fin >> a;
      fin >> b;
    }
    // COUNT
    std::cout << "VERTEX COUNT : " << x.size() << std::endl;
    // MAX
    std::cout << "MAX : " << list_max(x) << " " << list_max(y) << std::endl;
    // MIN
    std::cout << "MIN : " << list_min(x) << " " << list_min(y) << std::endl;
    // AVERAGE
    std::cout << "MAX : " << list_average(x) << " " << list_average(y) << std::endl;
  }

  return 0;
}
