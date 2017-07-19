#include <iostream>

// use std::bitset
#include <bitset>

/* NC++ definitions */
#include <random>
std::random_device _randev; 
// may be slow, but no seed is required.
#define ndif if ( _randev() & 1 )
#define ndswitch(x) switch(_randev() % (x))
/* NC++ definitions end */

bool formula(std::bitset<5> x) {
	return (x[0] || !x[1] || x[4])
			&& (!x[1] || x[2] || !x[5])
			&& (!x[1] || !x[3] || x[4])
			&& (!x[2] || !x[4] || x[5]) ;
}

int main(const int argc, const char * argv[]) {
  const int n = 5;
  std::bitset<n> x;
  
  for(int i = 0; i < n; i++) {
    ndif {
      x.set(i);
    } else {
      x.reset(i);
    }
  }
  std::cout << "Boolean assignment: "<< x << std::endl;
  
  if ( formula(x) )
	  std::cout << "yes." << std::endl;
  else
	  std::cout << "halt." << std::endl;
  
  return 0;
}
