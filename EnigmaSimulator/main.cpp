/*
 * main.cpp
 *
 *  Created on: 2014/05/16
 *      Author: sin
 */

#include <iostream>

#include "main.h"

	Rotor RI("EKMFLGDQVZNTOWYHXUSPAIBRCJ", 'Q');
	Rotor RII("AJDKSIRUXBLHWTMCQGZNPYFVOE", 'Q');
	Rotor RIII("BDFHJLCPRTXVZNYEIWGAKMUSQO", 'Q');
	Rotor RV("VZBRGITYUPSDNHLXAWMJQOFECK", 'Q');
	Rotor RfB("YRUHQSLDPXNGOKMIEBFZCWVJAT", '\0');

char code(char c) {

	RIII.advance();
	c = RIII.forward(c);
	c = RII.forward(c);
	c = RI.forward(c);
	c = RfB.forward(c);
	c = RI.reverse(c);
	c = RII.reverse(c);
	c = RIII.reverse(c);
	return c;
}

int main(int argc, char * argv[]) {

	std::cout << "Hi." << std::endl;

	char c = 'A';
	std::cout << "Key " << c << ", Code = " << code(c) << std::endl;
	std::cout << "Key " << c << ", Code = " << code(c) << std::endl;
	std::cout << "Key " << c << ", Code = " << code(c) << std::endl;
	std::cout << "Key " << c << ", Code = " << code(c) << std::endl;
	std::cout << "Key " << c << ", Code = " << code(c) << std::endl;

	return 0;
}
