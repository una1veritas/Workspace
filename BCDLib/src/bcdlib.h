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

	static const uint64_t signflag = ((uint64_t)1 << 63);

	bcdint(int64_t v) : val(0) {
		int bits;
		uint64_t negflag = v < 0 ? signflag : 0;
		if ( negflag )
			v = -v;
		for( bits = 0; v ; v /= 10, bits += 4 ) {
			val |= (v % 10)<<bits;
		}
		val &= ~signflag;
		val |= negflag;
		//printf("%lx\n", val);
	}

	bcdint(const bcdint & v) : val(v.val) {}

	static bool greaterthan(const bcdint & a, const bcdint & b) {
		if (a.nonnegative() && !b.nonnegative()) {
			return true;
		} else if (!a.nonnegative() && b.nonnegative()) {
			return false;
		} else if (a.nonnegative() && b.nonnegative()) {
			return a.val > b.val;
		} else {
			return b.val > a.val;
		}
	}

	// addition of nonnegative values;
	static bcdint add(const bcdint & a, const bcdint & b);

	// subtraction between nonnegative values
	static bcdint sub(const bcdint & a, const bcdint & b);

	static bcdint mul(const bcdint & a, const bcdint & b);

	static bcdint negate(const bcdint & a) {
		bcdint r = a;
		r.val ^= ((uint64_t)1 << 63);
		return r;
	}

	operator long() const {
		long t = 0, d = 1;
		uint64_t v = this->val;
		for(int i = 0; i < sizeof(long)*2; ++i) {
			t += (v & 0x0f) * d;
			v >>= 4;
			d *= 10;
		}
		return t;
	}

	bool nonnegative() const {
		return (val & signflag) == 0;
	}

	int sign() const {
		return (val & ~signflag) ? ((val & signflag) ? -1 : 1) : 0;
	}

	uint64_t value() const {
		return val & ~signflag;
	}

	bcdint operator-() {
		return bcdint::negate(*this);
	}

	bcdint operator+(const bcdint & b) {
		if (this->sign() >= 0) {
			if (b.sign() >= 0) {
				return add(*this, b);
			} else {
				return sub(*this, b);
			}
		}
		if (b.sign() >= 0) {
			return sub(b, *this);
		} else {
			return negate(add(negate(*this), negate(b)));
		}
	}

	bcdint operator-(const bcdint & b) {
		if (this->sign() < 0) {
			if (b.sign() < 0) {
				return negate(add(negate(*this), negate(b)));
			} else {
				return sub(b, negate(*this));
			}
		}
		if (b.sign() < 0) {
			return add(*this, negate(b));
		} else {
			return bcdint::sub(*this, b);
		}
	}

	friend std::ostream & operator<<(std::ostream & out, const bcdint & bcd) {
		if ( bcd.sign() < 0 )
			out << "-";
		out << std::hex << bcd.value();
		return out;
	}
};



#endif /* BCDLIB_H_ */
