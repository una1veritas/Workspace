//============================================================================
// Name        : Knapsack.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

#include <sys/time.h>

typedef bool boolean;
typedef unsigned char byte;

inline long nanos() {
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return long(tv.tv_usec/1000);
}

inline long micros() {
	return nanos()/1000;
}

inline long millis() {
	return micros()/1000;
}

int best(int price[], int items, int budget, boolean buying[]) {
  int itemNo = items - 1;
  if ( items == 0 )
    return 0;
  if ( items == 1 and price[itemNo] <= budget )
    return price[itemNo];

  int buy = price[itemNo]
    + best(price, itemNo, budget - price[itemNo], buying);
  int notbuy = best(price, itemNo, budget, buying);
  if ( buy <= budget and buy >= notbuy ) {
    buying[itemNo] = true;
    return buy;
  } else {
    buying[itemNo] = false;
    return notbuy;
  }
}

int list0[] = {
  108, 78, 78, 58, 58, 68, 0};
int list1[] = {
  108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328,
  228, 238, 198, 298, 64, 178, 0};
int * plist = list1;
const int B = 300;

void setup() {
  cout << "Good day." << endl;

  cout << "Budget: " << B << "Items: ";
  int number;
  for ( number = 0; plist[number] > 0 ; number++) {
    cout << plist[number] << ", ";
  }
  cout << endl;
  cout << "Number of items: " << number << endl;

  boolean buyingGuide[number];

  long swatch_milli = millis();
  long swatch_mu = micros();
  long swatch_nano = nanos();
  int result = best(plist, number, B, buyingGuide);
  swatch_mu = nanos() - swatch_nano;
  swatch_mu = micros() - swatch_mu;
  swatch_milli = millis() - swatch_milli;

  cout << "Recommended total price: " << result << "with items ";
  for ( number = 0; plist[number] > 0 ; number++) {
	if ( buyingGuide[number] ) {
      cout << number << " ";
    }
  }
  cout << "." << endl;

  cout << "Elapsed time: " << swatch_milli;
  if ( swatch_milli == (swatch_mu/1000) ) {
    cout << " milli " << swatch_mu % 1000 << " micro " << swatch_nano << " nano secs. " << endl;
  } else {
    cout << " milli secs. " << endl;
  }
}

void loop() {
}

int main() {
	cout << "Hello World!!!" << endl; // prints Hello World!!!
	setup();
	for(;;)
		loop();
	return 0;
}
