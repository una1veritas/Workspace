/*
 * mytypes.h
 *
 *  Created on: 2012/05/05
 *      Author: sin
 */

#ifndef MYTYPES_H_
#define MYTYPES_H_

typedef unsigned int uint;
typedef unsigned long ulong;
typedef bool boolean;
typedef unsigned char byte;
typedef unsigned short int word;

template<class T1, class T2, class T3> struct triple {
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;

	T1 first;
	T2 second;
	T3 third;
	triple() :
			first(T1()), second(T2()), third(T3()) {
	}
	triple(const T1& x, const T2& y, const T3& z) :
			first(x), second(y), third(z) {
	}
	template<class U, class V, class W>
	triple(const triple<U, V, W> &t) :
			first(t.first), second(t.second), third(t.third) {
	}
};

#endif /* MYTYPES_H_ */
