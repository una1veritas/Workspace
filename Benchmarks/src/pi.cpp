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

long double Leibniz_pi(const unsigned long & n, const long double & error) {
	long double p = 2.0, q;
	long double sign = -1;
	unsigned long i = 1;
	do {
		q = p;
		p = p + sign * 2 / (2*i + 1);
		if ( abs(p-q) < error)
			break;
		sign = -sign;
		++i;
	} while (i < n);
	return p + q;
}

