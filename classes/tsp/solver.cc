#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream.h>

#include "../Point.h"
#include "TravelingSalesman.h"

main() {
  TravelingSalesman * tsp;
  int times, sizes, i, sz, * tour;
  time_t in_time=0, out_time=0; 
  float ds_sec, ds_sec_sum, ds_length, nn_length, sp_length;
  char tmp[512];

  for (cin >> tmp; strncmp(tmp,"NAME",4) && (! cin.eof()); 
       cin >> tmp);
  if (tmp[strlen(tmp)-1] != ':')
    cin >> tmp;
  cin >> tmp;
  cout << "Name: " << tmp << "\n";

  tsp =  new TravelingSalesman(cin);
  tour = new int[tsp->size()];

  cerr << "Now computing.. \n";
  in_time = clock();
  tsp->divide_and_sort(tour,3.4);
  out_time = clock();
  ds_sec = out_time - in_time;
  ds_sec = (ds_sec > 0)? ds_sec / CLOCKS_PER_SEC : 0;
  ds_length = tsp->tourLength(tour);
/*
  cerr << *tsp << "\n";
  cerr.flush();
*/
  for (i = 0; i < tsp->size(); i++)
    cout << tsp->city(tour[i]) << "\n"; 
  
  tsp->nearest_neighbor(tour);
  nn_length = tsp->tourLength(tour);
  tsp->doubleSpanningTree(tour);
  sp_length = tsp->tourLength(tour);

  cout << "Size: " << tsp->size() << " Run-time: " << ds_sec 
    << " Tour-length: " << ds_length << " NN-ratio: " << ds_length/ nn_length 
      << " DST-ratio: " << ds_length/sp_length << "\n";

  delete tour;    

};
