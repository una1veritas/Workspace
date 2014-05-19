/*
 * main.h
 *
 *  Created on: 2014/05/16
 *      Author: sin
 */

#ifndef MAIN_H_
#define MAIN_H_

extern "C" {
#include <stdlib.h>
}

class Rotor {
	char wiring[26];
	int notch;
	int ringsetting;
	int position;

public:
	Rotor(const char transfer[26], const char carry) {
		memcpy(wiring, transfer, 26);
		notch = carry;
		ringsetting = 0;
		position = 0;
	}

	int advance(const int adv = 1) {
		if ( notch ) {
			position += adv;
		}
		return position;
	}

	char forward(const char c) {
		char t = toupper(c);
		if ( t < 'A' || 'Z' < t )
			return char(0);
		t = wiring[ (t - 'A' + position) % 26];
		return 'A' + ((t - position - 'A' + 26) % 26);
	}

	char reverse(const char c) {
		char t = toupper(c);
		if ( t < 'A' || 'Z' < t )
			return char(0);
		unsigned int p;
		for(p = 0; p < 26; p++) {
			if ( (t - 'A' + position) % 26 == wiring[p] - 'A')
				break;
		}
	//	std::cout << "t = " << t << ", p = " << p << std::endl;
		if ( p == 26 )
			return 0;

		return 'A' + ((p - position + 26 ) % 26);
	}

};


#endif /* MAIN_H_ */
