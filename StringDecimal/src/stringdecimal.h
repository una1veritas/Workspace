/*
 * stringdecimal.h
 *
 *  Created on: 2021/07/01
 *      Author: sin
 */

#ifndef STRINGDECIMAL_H_
#define STRINGDECIMAL_H_

#include <iostream>

#include <cstdio>
#include <cstring>
#include <cmath>

struct StringDecimal {
	char * str;
	int length;
	int intlen;

	StringDecimal(const char * s) {
		while (isspace(*s)) ++s;
		length = strlen(s);
		str = new char[length];
		strcpy(str, s);
		intlen = intpart_length(str);
	}

	StringDecimal(const long & d) {
		char buf[64];
		length = sprintf(buf, "%ld", d);
		str = new char[length];
		strcpy(str, buf);
		intlen = intpart_length(str);
	}

	StringDecimal(const double fv) {
		char buf[64];
		length = sprintf(buf,"%f",fv);
		str = new char[length];
		strcpy(str, buf);
		intlen = intpart_length(str);
	}

	~StringDecimal() {
		delete [] str;
	}

	static int intpart_length(char * s) {
		int l;
		for(l = 0; s[l] && s[l] != '.'; ++l);
		return l;
	}

	int fraclen(void) const {
		return length - intlen;
	}

	int digit_at(const int & pos) const {
		if (pos < 0) {
			if (intlen - pos < length) {
				return str[intlen - pos];
			} else
				return '0';
		} else {
			if (pos < intlen) {
				return str[intlen - pos - 1];
			} else {
				return '0';
			}
		}
	}

	bool is_negative(void) const {
		return str[0] == '-';
	}

	bool is_positive(void) const {
		return str[0] != '-';
	}

	int compare(const StringDecimal & b) const {
		int pos = intlen > b.intlen ? intlen : b.intlen;
		int fract_len = fraclen() > b.fraclen() ? fraclen() : b.fraclen();
		int both_negative = 1;
		for( ; pos > -fract_len; --pos) {
			if (digit_at(pos) == b.digit_at(pos)) {
				if (digit_at(pos) == '-')
					both_negative = -1;
				continue;
			}
			if (digit_at(pos) == '-')
				return -1;
			if (b.digit_at(pos) == '-')
				return 1;
			if ( digit_at(pos) < b.digit_at(pos) )
				return -1 * both_negative;
			else
				return 1 * both_negative;
		}
		return 0;
	}

	bool operator<(const StringDecimal & b) const {
		return compare(b) < 0;
	}

	bool operator<=(const StringDecimal & b) const {
		return compare(b) <= 0;
	}

	bool operator==(const StringDecimal & b) const {
		return compare(b) == 0;
	}

	StringDecimal & add(const StringDecimal & b) {
		int result_intlen = (intlen > b.intlen) ? intlen : b.intlen;
		int result_fraclen = length - intlen;
		result_fraclen = (result_fraclen > b.length - b.intlen) ? result_fraclen : b.length - b.intlen;
		char result[result_intlen+result_fraclen];
		if (result_fraclen) {
			result[result_intlen] = '.';
		}
		//std::cout << result_fraclen << std::endl;
		char digit, carry = 0;
		for(int pos = -result_fraclen; pos < result_intlen; ++pos) {
			digit = carry;
			carry = 0;
			digit += digit_at(pos) - '0';
			digit += b.digit_at(pos) - '0';
			if (digit > 9) {
				carry = 1;
				digit -= 10;
			}
			if (pos < 0) {
				result[result_intlen - pos] = '0' + digit;
			} else {
				result[result_intlen - pos - 1] = '0' + digit;
			}
		}
		result[result_intlen+result_fraclen] = char(0);
		return *(new StringDecimal(result));
	}

	friend std::ostream & operator<<(std::ostream & out, const StringDecimal & sd) {
		out << sd.str;
		return out;
	}
};

#endif /* STRINGDECIMAL_H_ */
