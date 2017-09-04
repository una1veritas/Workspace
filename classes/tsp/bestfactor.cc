#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream.h>

#include "../Point.h"
#include "TravelingSalesman.h"

#define BOX_SIZE 20000
#define TRIES /* 1000 */ 200
#define PARAMS 3
#define BASE_FACTOR 3.3

main(int argc, char * argv[]) {
  Point lowleft (0,0), uppright (BOX_SIZE,BOX_SIZE);
  TravelingSalesman * tsp;
  int times, sizes, i, sz, * tour;
  int nocities[] = {100, 250, 500, 750, 1000, 2500, 5000 /* , 7500, 10000, 15000, 20000 */};
  time_t in_time=0, out_time=0; 
  float ds_sec, ds_sec_sum, nn_sec, nn_sec_sum, factor;
  float factors[PARAMS], ratio, rmin[PARAMS], rmax[PARAMS], rsum[PARAMS];
  float ds_length, nn_length, dst_length;

  if (argc > 1) {
    if ( ! strncmp(argv[1], "-f", 2) ) 
      factor = atof((char*)(argv[1]+2));
  } else {
    factor = BASE_FACTOR;
  }

  for (i=0; i< PARAMS; i++) 
    factors[i] = factor+(0.1*i);

  cout << "Cities, Avr. NN run time, Avr. D&S run time";
  for (i=0; i< PARAMS; i++) 
    cout << ", " << factors[i] << ", " << factors[i] << ", "
      << factors[i];
  cout << ", Iteration " << TRIES << ".\n";

  for (sizes = 0; sizes < 8; sizes++) {
    sz = nocities[sizes];
    tour = new int[sz];
    nn_sec_sum = 0;
    ds_sec_sum = 0;
    for (i=0; i< PARAMS; i++) {
      rmax[i] = 0; rmin[i] = 0; rsum[i] = 0;
    }
    for (times = 0; times < TRIES; times++) {
      while ( (clock() - out_time)/CLOCKS_PER_SEC < 0.02) ;
      tsp =  new TravelingSalesman(sz,lowleft,uppright);
      cerr << '.';
      in_time = clock();
      /*
	 tsp->nearest_neighbor(tour);
	 */
      tsp->doubleSpanningTree(tour);
      out_time = clock();
      nn_sec = ((float)(out_time-in_time)) / CLOCKS_PER_SEC;
      nn_sec_sum += nn_sec;
      nn_length = tsp->tourLength(tour);

      for (i=0; i < PARAMS; i++) {
	in_time = clock();
	tsp->divide_and_sort(tour,factors[i]);
	out_time = clock();
	ds_sec = ((float)(out_time-in_time)) / CLOCKS_PER_SEC;
	ds_sec_sum += ds_sec;
	ds_length = tsp->tourLength(tour);
	ratio = ds_length/nn_length;
	rmax[i] = ( ratio > rmax[i] )? ratio : rmax[i];
	rmin[i] = ( ratio < rmin[i] || (rmin[i] == 0) )? ratio : rmin[i];
	rsum[i] += ratio;
      }
      delete tsp;
    }

    cout << sz << ", " << nn_sec_sum/TRIES << ", " << ds_sec_sum/(TRIES*PARAMS) << ", ";
    for (i=0; i< PARAMS; i++) 
      cout << rsum[i]/TRIES << ", " << rmax[i] << ", " << rmin[i] << ", ";
    cout << "\n";
    cout.flush();
    delete tour; 
    cerr << '\n';
  }
};


