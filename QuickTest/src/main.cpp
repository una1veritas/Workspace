/*
 * main.cpp
 *
 *  Created on: 2012/10/20
 *      Author: sin
 */

#include <iostream>
#include <cmath>

typedef uint16_t uint16;
typedef bool boolean;

int main(int argc, char * argv[]) {
	int val;
	float fval;

	if ( argc < 2 )
		return 1; // no arg error

	for (int i = 1; i < argc; i++) {
		std::cout << "From an input " << argv[i];
		fval = atof(argv[i]);
		std::cout << " got fp value " << fval;
		std::cout << ", and it's casted into " << static_cast<int>(fval) << ". " << std::endl;
		std::cout << "On the otherhand, Ceiling = " << ceil(fval) << ", Floor = " << floor(fval) << "." << std::endl;
	}
	std::cout << std::endl;
	return 0;
}
