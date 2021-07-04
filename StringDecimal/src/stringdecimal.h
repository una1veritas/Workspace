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

	StringDecimal(const char * s) {
		int len = strlen(s);
		int offset = 0;
		/*
		switch (s[0]) {
		case ' ':
		case '-':
			break;
		default:
			len += 1;
			offset = 1;
			break;
		}
		*/
		str = new char[len];
		str[0] = ' ';
		strcpy(str+offset, s);
	}

	StringDecimal(const long d) {
		long d_val = (d >= 0) ? d : -d;
		int d_sign = (d == 0) ? 0 : ((d > 0) ? 1 : -1);
		int digits = floor(log10((double)d_val)) + 1;
		str = new char[digits+1];
		char * p = str;
		if (d_sign < 0)
			*p++ = '-';
		//std::cout << "digits = " << digits << std::endl;
		for(int i = 0; i < digits; ++i) {
			*(p+digits-1-i) = '0' + (d_val % 10);
			d_val /= 10;
			//std::cout << i << std::endl;
		}
		*(p + digits) = (char) 0;
	}

	StringDecimal(const double fv) {
		char buf[64];
		sprintf(buf,"%f",fv);
		str = new char[strlen(buf)];
		strcpy(str,buf);
	}

	~StringDecimal() {
		delete [] str;
	}

	int prepoint() const {
		int l;
		for(l = 0; str[l] && str[l] != '.'; ++l);
		return l;
	}

	int postpoint() const {
		return strlen(str) - prepoint();
	}

	StringDecimal & add(const StringDecimal & b) {
		int this_prepoint = prepoint(), this_postpoint = postpoint();
		int b_prepoint = b.prepoint(), b_postpoint = b.postpoint();
		//std::cout << "this pre " << this_prepoint << " b pre " << b_prepoint << std::endl;
		int result_prepoint = 1 + ((this_prepoint > b_prepoint) ? this_prepoint : b_prepoint);
		int result_postpoint = (this_postpoint > b_postpoint) ? this_postpoint : b_postpoint;
		char result[result_prepoint+result_postpoint];
		result[result_prepoint+result_postpoint] = (char) 0;
		memset(result,'0',result_prepoint+result_postpoint);
		if (result_postpoint) {
			result[result_prepoint] = '.';
		}
		//std::cout << result << std::endl;
		char digit, carry = 0;
		for(int pos = result_postpoint; pos > 1; --pos) {
			digit = carry;

			//std::cout << "pos = " << pos << ", "
			//		<< " carry = " << int(carry)
			//		<< ", this = " << char(this->str[this_prepoint+pos-1])
			//		<< ", b = " << char(b.str[b_prepoint+pos-1]) << ", ";
			if (pos <= this_postpoint)
				digit += this->str[this_prepoint+pos-1] - '0';
			//std::cout << pos << " < " << this_postpoint << " digit = " << int(digit) << std::endl;
			if (pos <= b_postpoint)
				digit += b.str[b_prepoint+pos-1] - '0';
			//std::cout << "digit = " << int(digit) << std::endl;
			if (digit > 9) {
				carry = 1;
				digit -= 10;
			} else {
				carry = 0;
			}
			result[result_prepoint+pos-1] = '0' + digit;
		}
		const int this_offset = result_prepoint - this_prepoint;
		const int b_offset = result_prepoint - b_prepoint;
		//std::cout << "carry = " << int(carry) << std::endl;

		for(int pos = result_prepoint; pos > 1 || carry; --pos) {
			digit = carry;
			//std::cout << "pos = " << pos << ", "
			//		<< " carry = " << int(carry)
			//		<< ", this = " << this->str[pos-1-this_offset]
			//		<< ", b = " << b.str[pos-1-b_offset] << std::endl;

			if (pos > this_offset)
				digit += this->str[pos-1-this_offset] - '0';
			if (pos > b_offset)
				digit += b.str[pos-1-b_offset] - '0';
			if (digit > 9) {
				carry = 1;
				digit -= 10;
			} else {
				carry = 0;
			}
			//std::cout << result_prepoint << ", pos: " << pos << std::endl;
			result[pos-1] = '0' + digit;

			//std::cout << "place to " << pos-1 << std::endl;
		}

		return *(new StringDecimal(result));
	}

	friend std::ostream & operator<<(std::ostream & out, const StringDecimal & sd) {
		out << sd.str;
		return out;
	}
};

#endif /* STRINGDECIMAL_H_ */
