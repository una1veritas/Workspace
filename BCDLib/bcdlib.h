/*
 * bcdlib.h
 *
 *  Created on: 2021/06/27
 *      Author: Sin Shimozono
 */

#ifndef BCDLIB_H_
#define BCDLIB_H_

#include <cstdint>
#include <iostream>

#include <cstdio>

struct bcdint {
	uint64_t val;

	bcdint(int64_t v) : val(0) {
		int bits;
		uint64_t negflag = v < 0 ? ((uint64_t)1 << 63) : 0;
		//printf("v = %ld, neg = %ld\n",v,negflag);
		if (negflag) {
			v = -v;
		}
		for( bits = 0; v ; v /= 10, bits += 4 ) {
			val |= (v % 10)<<bits;
		}
		val &= ~((uint64_t)1<<63);
		val |= negflag;
		//printf("%lx\n", val);
	}

	bcdint(const bcdint & v) : val(v.val) {}

	static bool greaterthan(const bcdint & a, const bcdint & b) {
		if (a.sign() >= 0 && b.sign() < 0) {
			return true;
		} else if (a.sign() < 0 && b.sign() >= 0) {
			return true;
		} else if (a.sign() >= 0 && b.sign() >= 0) {
			return a.val > b.val;
		} else {
			return b.val > a.val;
		}
	}

	static bcdint add(const bcdint & a, const bcdint & b) {
		uint64_t a_val = a.val, b_val = b.val;
		bcdint r = 0;
		int carry = 0;
		for(int bits = 0; bits < 64 && (a_val || b_val || carry) ; a_val>>=4, b_val>>=4, bits +=4 ) {
			int d = carry + (a_val & 0x0f) + (b_val & 0x0f);
			carry = 0;
			if (d > 9) {
				carry = 1;
				d -= 10;
			}
			r.val |= d<<bits;
		}
		r.val &=  ~((uint64_t)1<<63);
		return r;
	}

	static bcdint sub(const bcdint & a, const bcdint & b) {
		uint64_t a_val, b_val, sgn;
		bcdint r = 0;
		int borrow = 0;
		if (greaterthan(a,b)) {
			a_val = a.val, b_val = b.val;
			sgn = 0;
		} else {
			printf("negative\n");
			b_val = a.val, a_val = b.val;
			sgn = (uint64_t)1 << 63;
		}
		for(int bits = 0;  bits < 64 && (a_val || b_val || borrow) ; a_val>>=4, b_val>>=4, bits +=4 ) {
			int d = (a_val & 0x0f) - (b_val & 0x0f) - borrow;
			borrow = 0;
			if (d < 0) {
				borrow = 1;
				d += 10;
			}
			r.val |= d<<bits;
		}
		printf("r.val = %ld\n", r.val);
		r.val |= sgn;
		return r;
	}

	static bcdint negate(const bcdint & a) {
		bcdint r = a;
		r.val ^= ((uint64_t)1 << 63);
		return r;
	}
	int sign() const {
		if ( ( val & ~((uint64_t)1 << 63) ) == 0 )
			return 0;
		if ( (val & ((uint64_t)1 << 63)) != 0 )
			return -1;
		else
			return 1;
	}

	uint64_t value() const {
		return ( val & ~((uint64_t)1 << 63) );
	}

	bcdint operator-() {
		return bcdint::negate(*this);
	}

	bcdint operator+(const bcdint & b) {
		return bcdint::add(*this, b);
	}

	bcdint operator-(const bcdint & b) {
		if (this->sign() < 0 && b.sign() < 0) {
			return negate(add(negate(*this), negate(b)));
		} else if (this->sign() < 0 && b.sign() >= 0) {
			return sub(b, negate(*this));
		} else if (this->sign() >= 0 && b.sign() < 0) {
			return add(*this, negate(b));
		}
		return bcdint::sub(*this, b);
	}

	friend std::ostream & operator<<(std::ostream & out, const bcdint & bcd) {
		if ( bcd.sign() < 0 )
			out << "-";
		out << std::hex << bcd.value();
		return out;
	}
};



#endif /* BCDLIB_H_ */
