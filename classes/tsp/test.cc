#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream.h>

#include "../Point.h"
#include "TravelingSalesman.h"

#define BOX_SIZE 1000000

main(int argc, char * argv[]) {
  Point lowleft (0,0), uppright (BOX_SIZE,BOX_SIZE);
  TravelingSalesman * tsp;
  int times, sizes, i, sz, * tour;
  time_t in_time=0, out_time=0; 
  float ds_sec, ds_sec_sum, factor, ds_length;

  factor = 3.4;
  sz = 100;
  tour = new int[sz];
  tsp =  new TravelingSalesman(sz,lowleft,uppright);
  cerr << ".";
  
  in_time = clock();
  tsp->divide_and_sort(tour,factor);
  out_time = clock();
  ds_sec = ((float)(out_time-in_time)) / CLOCKS_PER_SEC;
  ds_sec_sum += ds_sec;
  ds_length = tsp->tourLength(tour);

  cout << tsp->size() << ", " << ds_sec << ", " << ds_length << "\n";
  cout.flush();
  for(i=0; i < tsp->size(); i++) {
    cout << tsp->city(tour[i]) << "\n";
  }
  delete tour;    

};


