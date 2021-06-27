#include <iostream>

#include "bcdlib.h"

int main(int argc, char **argv) {
	bcdint x = 13;
	bcdint y = 409;
	std::cout << x << ", " << y << std::endl;
	std::cout << x - y << std::endl;
	std::cout << y + x << std::endl;
	std::cout << std::dec << (long) (x+y) << std::endl;
	x = -x;
	std::cout << x - y << std::endl;
	std::cout << "mul(x,y) " << bcdint::mul(x, y) << std::endl;
	std::cout << "mul(y,x) " << bcdint::mul(y, x) << std::endl;
	return 0;
}
