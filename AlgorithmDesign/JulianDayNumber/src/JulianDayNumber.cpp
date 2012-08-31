//============================================================================
// Name        : JulianDayNumber.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip.h>

using namespace std;

double JulianDay(int, int, double);
double CalendarDate(double);

inline long integerPart(double df) {
	return (long int) df;
}
inline double fractionalPart(double df) {
	return df - ((long int) df);
}
inline long sign(double d) {
	if ( d < 0 )
		return -1;
	else
		return 1;
}

int main(int argc, char * argv[]) {
	//test
	int year = 2011;
	int month = 10;
	double date = 12;
	double jd = 2436116.31;

	// process input
	if ( argc == 2 ) {
		jd = atof(argv[1]);
	} else if ( argc == 4 ) {
		year = atoi(argv[1]);
		month = atoi(argv[2]);
		date = atof(argv[3]);
	}
	cout << "Input: ";
	cout << "Year " << year << ", month " << month << ", date " << date << endl;
	//
	cout << "INT(-4.98) = " << integerPart(-4.98) << endl;
	//
	jd = JulianDay(year, month, date);
	cout << "Julian Day number = "<< integerPart(jd) << "." << integerPart(jd*10)%10 << integerPart(jd*100)%10 << endl << endl;
	cout << "Day of the week = " << integerPart(integerPart(jd)+1.5) %7 << "." << endl;
	double cal = CalendarDate((double)1507900.13);
	cout << "Calendar date is "<< setprecision(10) << cal << endl;

	return 0;
}

double CalendarDate(double jd) {
	jd += 0.5;
	long z = integerPart(jd);
	long a = z;
	double f = fractionalPart(jd);
	if ( z >= 2299161 ) {
		long alpha = integerPart( (z-1867216.25)/36524.25 );
		a += 1 + alpha - integerPart(alpha/4);
	}
	long b = a + 1524;
	long c = integerPart( (b-122.1)/365.25 );
	long d = integerPart(365.25 * c);
	long e = integerPart( (b-d)/30.6001 );
	double day = b - d - integerPart(30.6001 * e) + f;
	int month;
	if ( e < 13.5 ) {
		month = e - 1;
	} else {
		month = e-13;
	}
	long year;
	if ( month > 2.5) {
		year = c - 4716;
	} else {
		year = c - 4715;
	}
	return sign(year)*(abs(year)*10000 + month*100 + day);
}

double JulianDay(int y, int m, double dd) {
	if ( m <= 2 ) {
		m = m + 12;
		y = y - 1;
	}
	int a = 0, b = 0;
	if ( y*10000+m*100+dd >= 15821015 ) {
		a = integerPart(y/100);
		b = 2-a+integerPart(a/4);
	}
	return integerPart(365.25 * y) + integerPart(30.6001 * (m+1)) + dd + b + 1720994.5;
}
