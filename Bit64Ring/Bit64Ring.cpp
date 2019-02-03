//============================================================================
// Name        : Bit64Ring.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>
#include "Bit64Ring.h"

int main(void) {
	std::cout << "!!!Hello World!!!" << std::endl;

	Bit64Ring bitring;
	bitring.enqueue(1);
	bitring.enqueue(0);
	bitring.enqueue(1);
	bitring.enqueue(1);
	bitring.enqueue(0);
	bitring.enqueue(0);
	bitring.enqueue(1);
	bitring.enqueue(0);
	std::cout << bitring << std::endl;
	return EXIT_SUCCESS;
}
