//============================================================================
// Name        : PrecisionNumber.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : GPL
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdint>
#include <cmath>
#include <algorithm>

using namespace std;

class PrecisionNumber {
private:
	typedef int64_t numint;
	typedef int16_t expint;

private:
	numint numerator;
	numint denominator;
	expint power;

	// utility functions
private:
	static numint gcd(numint a, numint b) {
		while (b != 0) {
			numint t = b;
			b = a % b;
			a = t;
		}
		return a;
	}

	expint find_common_power(const PrecisionNumber& other) const {
		if (this->power >= 0 and other.power >= 0) {
			return std::min(this->power, other.power);
		} else if (this->power < 0 and other.power < 0) {
			return std::max(this->power, other.power);
		}
		return expint(0);
	}

	PrecisionNumber & adjust_exponent(const int32_t & comm_power) {
		if (comm_power > this->power) {
			this->denominator *= pow(10, comm_power - this->power);
		} else if (comm_power < this->power) {
			this->numerator *= pow(10, this->power - comm_power);
		}
		this->power = comm_power;
		return *this;
	}

public:
	static expint significant_number() {
		return expint(19); // 32bit signed integer can hold 9 digits
	}

public:
	PrecisionNumber() : numerator(0), denominator(1), power(0) {}

	// Constructor that takes and normalizes a numerator, denominator, and power
	PrecisionNumber(const uint64_t & n, const uint64_t & d = 1, const expint & p = 0)
	: numerator(n), denominator(d), power(p) {
		numint divisor;
		// Normalize the sign of the fraction
		if (denominator < 0 ){
			numerator = -numerator;
			denominator = -denominator;
		}
		// adjust power to 0 and make pure fraction
		while (power != 0) {
			if ( power > 0 ) {
				numerator *= 10;
				power--;
			} else {
				denominator *= 10;
				power++;
			}
		}
		// Reduce the fraction to its simplest form
		divisor = gcd(std::abs(numerator), std::abs(denominator));
		numerator /= divisor;
		denominator /= divisor;
		// Remove trailing zeros from the numerator and denominator
		while ( (numerator % 10) == 0 ) {
			numerator /= 10;
			power++;
		}
		while ( (denominator % 10) == 0 ) {
			denominator /= 10;
			power--;
		}
	}

	PrecisionNumber(const PrecisionNumber& other) : numerator(other.numerator), denominator(other.denominator), power(other.power) {}

	static PrecisionNumber pi() {
		return PrecisionNumber(3141592653589793238LL, 1, -18);
	}

	// four arithmetic operations, comparison operators, and assignment operator

	PrecisionNumber operator*(const PrecisionNumber& other) const {
		return PrecisionNumber(this->numerator * other.numerator, this->denominator * other.denominator, this->power + other.power);
	}

	PrecisionNumber operator*(const uint64_t & otherint) const {
		return PrecisionNumber(this->numerator * otherint, this->denominator, this->power);
	}

	PrecisionNumber operator/(const PrecisionNumber& other) const {
		return PrecisionNumber(this->numerator * other.denominator, this->denominator * other.numerator, this->power - other.power);
	}

	PrecisionNumber operator/(const uint64_t& otherint) const {
		return PrecisionNumber(this->numerator, this->denominator * otherint, this->power);
	}

	// a ( == *this) + b ( == other) => c (== return value)
	PrecisionNumber operator+(const PrecisionNumber& other) const {
		expint comm_power = find_common_power(other);
		PrecisionNumber a(*this), b(other);
		a.adjust_exponent(comm_power);
		b.adjust_exponent(comm_power);
		//cout << "a = " << a << ", b = " << b << endl;
		numint comm_denom = a.denominator * b.denominator / gcd(a.denominator, b.denominator);
		//cout << "comm_denom = " << comm_denom << endl;

		// Find the least common multiple of the denominators
		a.numerator *= comm_denom / a.denominator;
		b.numerator *= comm_denom / b.denominator;
		a.denominator = comm_denom;
		b.denominator = comm_denom;
		//cout << "a = " << a << ", b = " << b << endl;
		return PrecisionNumber(a.numerator + b.numerator, comm_denom, comm_power);
	}

	PrecisionNumber operator-(const PrecisionNumber& other) const {
		return *this + PrecisionNumber(-other.numerator, other.denominator, other.power);
	}

	PrecisionNumber operator-() const {
		return PrecisionNumber(-numerator, denominator, power);
	}

	PrecisionNumber& operator=(const PrecisionNumber& other) {
		if (this != &other) {
			numerator = other.numerator;
			denominator = other.denominator;
			power = other.power;
		}
		return *this;
	}

	PrecisionNumber operator+=(const PrecisionNumber& other) {
		*this = *this + other;
		return *this;
	}

	bool operator==(const PrecisionNumber& other) const {
		return (this->power == other.power)
				and (this->denominator == other.denominator)
				and (this->numerator == other.numerator);
	}

	bool operator<(const PrecisionNumber& other) const {
		PrecisionNumber r = *this - other;
		return r.numerator < 0;
	}

	// functions

	numint sign() const {
		if (numerator < 0)
			return -1;
		return 1;
	}

	PrecisionNumber abs() const {
		return PrecisionNumber(-numerator, denominator, power);
	}

	// type conversion operators
	operator double() const {
		return static_cast<double>(numerator) / denominator * pow(10, power);
	}

	operator int() const {
		return static_cast<int>(numerator * pow(10, power) / denominator);
	}

	friend std::ostream& operator<<(std::ostream& os, const PrecisionNumber& pn) {
		os << pn.numerator;
		if (pn.denominator != 1) {
			os << "/" << pn.denominator;
		}
		if (pn.power != 0) {
			os << " x 10^" << int(pn.power);
		}
		return os;
	}

	PrecisionNumber sin() const {
		PrecisionNumber sum(*this); 	// 0th term
		PrecisionNumber term(*this); 	// nth term
		PrecisionNumber xsquared = (*this) * (*this); // x^2
		PrecisionNumber denom(1, 1); // 2n * (2n + 1)
		PrecisionNumber error(1, 1, -10); // 10^-17
		cout << "x = " << *this << ", x^2 = " << xsquared << endl;
		return 1;
		for (int n = 1; n < 20; ++n) {
			// 漸化式を用いて次の項を計算する
			// 分母は 2n * (2n + 1)
			denom = (2 * n) * (2 * n + 1);
			term = term * (-xsquared / denom);
			sum += term;
			cout << "n = " << n << ", term = " << term << ", sum = " << sum << endl;
			// もし足す値（term）が極めて小さくなり、sum に影響を与えなくなったら終了
			if (term.abs() < error) {
				break;
			}
		}
		return sum;
	}

	// テイラー展開（7次）による sin(x) の計算
	// ※ x は -pi/4 から pi/4 の範囲を想定
	double sin_taylor(double x) {
		double x2 = x * x;
		// x - x^3/6 + x^5/120 - x^7/5040
		// ホーナー法（Horner's method）で効率的に計算
		return x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))));
	}
	// チェビシェフ多項式をベースに最適化された近似（7次）
	// 区間 [-pi/4, pi/4] で最悪誤差を抑えた係数の例
	double sin_chebyshev(double x) {
		double x2 = x * x;
		// 係数は区間内の最大誤差を最小化するように調整された値
		double c1 =  0.9999999999;
		double c3 = -0.1666666604;
		double c5 =  0.0083331804;
		double c7 = -0.0001980838;
		return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * c7)));
	}

};

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	PrecisionNumber pn1(1, 3), pn2(31415926535,1,-10);
	cout << "pn1: " << pn1 << endl;
	cout << "pn2: " << pn2 << endl;

	cout << "pn1 + pn2: " << (pn1+pn2) << endl;
	cout << "pn1 - pn2: " << (pn1-pn2) << endl;
	cout << "pn1 * pn2: " << (pn1*pn2) << endl;
	cout << "pn1 / pn2: " << (pn1/pn2) << endl;
	cout << "as floating-point (double) / " << double(pn1/pn2) << endl;
	cout << "as int + " << int(pn1+pn2) << endl;
	cout << "pn2^2 " << (pn2*ingarrapn2) << endl;
	PrecisionNumber pi6th = PrecisionNumber::pi() / uint64_t(6);
	cout << "pi = " << PrecisionNumber::pi() << endl;
	cout << "pi/6 = " << pi6th << endl;
	cout << "sin(pi/6) = " << pi6th.sin() << endl;
	cout << "finished " << endl;

	return 0;
}
