/*
 * pi.cpp
 *
 *  Created on: 2019/02/24
 *      Author: sin
 */

/*
100 REM Pi Calculator
110 p=2
120 s=-1
130 CLS:INPUT "N=";n
140 FOR i=1 TO n
150 q=p
160 p=p+s*2/(2*i+1)
170 IF FRAC(i/200)=0 THEN PRINT i;" PI=";p+q
180 s=-s
190 NEXT i
200 PRINT "I=";i
210 PRINT "PI=";p+q
220 BEEP 0.5
230 END
*/

#include <cmath>

long double Leibniz_pi(const unsigned long & n /*, const long double & error */) {
	long double p = 1.0, q;
	long double sign = -1;
	for ( unsigned long i = 1; i < n; ++i) {
		q = p;
		p += sign / (long double)((i<<1) + 1);
		/*
		if ( abs(p-q) < error )
			break;
		*/
		sign = -sign;
	}
	return (p + q)*2;
}

long double Wallis_pi(const unsigned long & n /*, const long double & error*/) {
	long double p = 1.0, q;
	for ( unsigned long i = 1; i < n; ++i) {
		q = p;
		p *= (long double)(i*2)*(i*2)/(long double)((i*2-1)*(i*2+1));
		/*
		if ( abs(p-q) < error)
			break;
		*/
	}
	return p*2;
}


