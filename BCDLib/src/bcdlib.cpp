/*
 * bcdlib.cpp
 *
 *  Created on: 2021/06/28
 *      Author: Sin Shimozono
 */

#include "bcdlib.h"

// addition of nonnegative values;
bcdint bcdint::add(const bcdint & a, const bcdint & b) {
	uint64_t a_val = a.val & ~signflag, b_val = b.val & ~signflag;
	bcdint r = 0;
	int carry = 0;
	for(int bits = 0; bits < 64 && (a_val || b_val || carry) ; a_val >>= 4, b_val >>= 4, bits +=4 ) {
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

// subtraction between nonnegative values
bcdint bcdint::sub(const bcdint & a, const bcdint & b) {
	uint64_t a_val, b_val, sgn;
	bcdint r = 0;
	int borrow = 0;
	if (greaterthan(a,b)) {
		a_val = a.val & ~signflag, b_val = b.val & ~signflag;
		sgn = 0;
	} else {
		b_val = a.val & ~signflag, a_val = b.val & ~signflag;
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
	r.val |= sgn;
	return r;
}

bcdint bcdint::mul(const bcdint & a, const bcdint & b) {
	uint64_t a_val = a.val & ~signflag;
	uint64_t b_val = b.val & ~signflag;
	bcdint r = 0, s = 0;
	int carry = 0;
	for(int j = 0; b_val; b_val >>= 4, j += 4) {
		int d = b_val & 0x0f;
		s.val = 0;
		for(int bitpos = 0; bitpos < 64; bitpos += 4) {
			uint64_t t = ((a_val>>bitpos) & 0x0f) * d;
			t += carry;
			carry = t / 10;
			t %= 10;
			s.val |= t<<(bitpos+j);
		}
		r = add(r, s);  // replacing bcdint on every occation...
	}
	return r;
}
