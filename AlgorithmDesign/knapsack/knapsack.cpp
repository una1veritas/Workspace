//============================================================================
// Name        : Knapsack.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>

using namespace std;

#include <sys/time.h>

typedef bool boolean;
typedef unsigned char byte;

int list0[] = {
  108, 78, 78, 58, 58, 68, 0};
int list1[] = {
  38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328,
  228, 238, 198, 298, 64, 178, 228, 138, 298, 100, 0};
int * plist = list1;
const int B = 300;

long micros();
long millis();
long secs();

int best(int price[], int items, int budget, boolean cart[]) {
  if ( items == 0 )
    return 0;
  if ( items == 1 and price[items - 1] <= budget )
    return price[items -1];

  int buy = price[items - 1]
    + best(price, items - 1, budget - price[items - 1], cart);
  int notbuy = best(price, items - 1, budget, cart);
  if ( buy <= budget and buy >= notbuy ) {
    cart[items - 1] = true;
    return buy;
  } else {
    cart[items - 1] = false;
    return notbuy;
  }
}

void setup() {
	int number;

  cout << "Budget: " << B << "," << endl;
  for ( number = 0; plist[number] > 0 ; number++);
  cout << number << " Items: " << endl;
  for ( number = 0; plist[number] > 0 ; number++) {
    cout << plist[number] << ", ";
  }
  cout << endl;

  boolean shoppingCart[number];

  long swatch_milli = millis();
  long swatch_mu = micros();
  long swatch_sec = secs();
  int result = best(plist, number, B, shoppingCart);
  swatch_mu = micros() - swatch_mu;
  swatch_milli = millis() - swatch_milli;
  swatch_sec = secs() - swatch_sec;

  cout << "Recommended total price is " << result << " with these items ";
  for ( number = 0; plist[number] > 0 ; number++) {
	if ( shoppingCart[number] ) {
      cout << number << " ";
    }
  }
  cout << "." << endl;

  cout << "Elapsed " << swatch_sec*1000 + swatch_milli;
  if ( swatch_milli == (swatch_mu/1000) ) {
	swatch_mu %= 1000;
	cout << "." << setw(3) << setfill('0') << swatch_mu;
  }
  cout << " msecs. " << endl;
}

void loop() {
}


int main() {
	cout << "Hello. " << endl; // prints Hello World!!!
	setup();
//	for(;;)
//		loop();
	return 0;
}

long micros() {
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return long(tv.tv_usec);
}

long millis() {
	return micros()/1000;
}

long secs() {
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return long(tv.tv_sec);
}
