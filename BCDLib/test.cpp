#include <iostream>

#include "bcdlib.h"

int main(int argc, char **argv) {
	bcdint x = 12;
	bcdint y = 19;
	std::cout << x.sign() << ", " << y.sign() << std::endl;
	std::cout << x << ", " << y << std::endl;
	std::cout << x - y << std::endl;
	std::cout << y - x << std::endl;
	x = -x;
	std::cout << x - y << std::endl;
	std::cout << y - x << std::endl;
	return 0;
}
