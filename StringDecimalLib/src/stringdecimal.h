/*
 * stringdecimal.h
 *
 *  Created on: 2021/07/01
 *      Author: sin
 */

#ifndef STRINGDECIMAL_H_
#define STRINGDECIMAL_H_

#include <iostream>

#include <cstdlib>
#include <cstring>
#include <cmath>

struct StringDecimal {
	char * str;

	StringDecimal(const char * s) {
		str = new char[strlen(s)];
		strcpy(str, s);
	}

	StringDecimal(const long d) {
		long d_val;
		int d_sign;
		if (d == 0) {
			d_val = 0;
			d_sign = 0;
		} else if (d > 0) {
			d_val = d;
			d_sign = 1;
		} else {
			d_val = -d;
			d_sign = -1;
		}
		int digits = floor(log10((double)d_val)) + 1;
		str = new char[digits+1];
		strcpy(str,"####");
		char * p = str;
		if (d_sign < 0)
			*p++ = '-';
		std::cout << "digits = " << digits << std::endl;
		for(int i = 0; i < digits; ++i) {
			*(p+digits-1-i) = '0' + (d_val % 10);
			d_val /= 10;
			std::cout << i << std::endl;
		}
		*(p + digits) = (char) 0;
	}

	~StringDecimal() {
		delete [] str;
	}

	friend std::ostream & operator<<(std::ostream & out, const StringDecimal & sd) {
		out << sd.str;
		return out;
	}
};

#endif /* STRINGDECIMAL_H_ */
