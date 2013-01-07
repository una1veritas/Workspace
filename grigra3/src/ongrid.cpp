#include <fstream>
#include <iostream>
#include <stdlib.h>

int main(int argc, char **argv) {
  if(argc != 3) {
    std::cout << "USAGE : [ongrid.exe grid/int input-file/string]" << std::endl;
    return 0;
  }

  int grid = atoi(argv[1]);
  std::ifstream fin(argv[2]);
  if(fin.is_open()) {
    int a;
    while(!fin.eof()) {
      fin >> a;
      if(a % grid != 0) {
        std::cout << "NOT ON GRID!" << std::endl;
        return 0;
      }
    }
  }

  std::cout << "SUCCESS!" << std::endl;

  return 0;
}
