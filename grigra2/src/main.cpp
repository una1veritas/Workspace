/*
 * main.cpp
 *
 *  Created on: 2013/01/08
 *      Author: sin
 */

//#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
//#include <algorithm>
//#include <functional>
//#include <list>

#include "layout.h"

int32 main(int32 argc, char **argv) {
  if(argc != 3) {
    std::cout << "USAGE [layout.exe grid_size/int input-filename/string]" <<std::endl;
    return 0;
  }

  GridLayout gridlayout(atoi(argv[1]));
  PointSet pointset(argv[2]);

  // Display IN-POINT-SET
  //std::cout << "[IN]" << std::endl;
  //pointset.print(std::cout);
  //std::cout << std::endl;

  // Apply Matching
  if(!pointset.checkIndependent()) {
    std::cout << "POINT-SET IS NOT INDEPENDENT!" << std::endl;
  }
  if(gridlayout.checkMatch(pointset)) {
    std::cout << "POINT-SET IS ALREADY GRID-LAYOUTED!" << std::endl;
  }
  gridlayout.match(pointset);

  // Display OUT-POINT-SET
  //std::cout << "[INDEPENDENT?] " << pointset.checkIndependent() << std::endl;
  //std::cout << "[ONGRID?] " << gridlayout.checkMatch(pointset) << std::endl;
  // if(pointset.checkIndependent() && gridlayout.checkMatch(pointset)) {
  //  std::cout << "[OUT]" << std::endl;
  //  pointset.print(std::cout);
  //  std::cout << std::endl;
  //}

  std::cout << pointset;

  return 0;
}




