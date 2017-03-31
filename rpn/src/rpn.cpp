//============================================================================
// Name        : rpn.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <sstream>
#include <deque>
#include <map>

#include <cmath>
#include <cstdlib>
#include <cerrno>

using namespace std;

class RPN {
	typedef double (*(unary_fp))(const double);
	static map<const string, unary_fp> ufuncdict;
	static map<const string, double> constdict;

	deque<double> stack;

private:
	static double uf_pow2(const double x);

	static double uf_inv(const double x);
	static double uf_int(const double x);
	static double uf_round(const double x);

	static void init_funcdict() {
		ufuncdict.clear();

		ufuncdict["log2"] = &log2;
		ufuncdict["lg"] = &log2;
		ufuncdict["exp"] = &exp;
		ufuncdict["pow2"] = &uf_pow2;

		ufuncdict["sin"] = &sin;
		ufuncdict["cos"] = &cos;
		ufuncdict["tan"] = &tan;
		ufuncdict["asin"] = &asin;
		ufuncdict["acos"] = &acos;
		ufuncdict["atan"] = &atan;

		ufuncdict["inv"] = &uf_inv;
		ufuncdict["//"] = &uf_inv;
		ufuncdict["int"] = &uf_int;
		ufuncdict["trunc"] = &uf_int;
		ufuncdict["round"] = &uf_round;
	}

	static void init_constdict() {
		constdict.clear();

		constdict["pi"] = M_PI;
	}

public:
	static void initialize() { init_constdict(); init_funcdict(); }
	static bool hasUnaryFunction(const string & str) {
		return ufuncdict.find(str) != ufuncdict.end();
	}
	static bool hasConstant(const string & str) {
		return constdict.find(str) != constdict.end();
	}
	static double referConstant(const string & name);


	RPN(void) : stack() { }

	unsigned int size() const { return stack.size(); }
	bool empty() const { return stack.size() == 0; }

	double push(const double & d);
	double pop(void);
	double swap();
	double peek() const { return stack.back(); }
	void clear() { stack.clear(); }

	const double & operator[](unsigned int i) const { return stack[i]; }

	double perform(const string & func);

};

/*
 * class variable
 */
map<const string, RPN::unary_fp> RPN::ufuncdict;
map<const string, double> RPN::constdict;

/*
 * class functions
 * unary function library
 */
double RPN::uf_inv(const double x) {
	return 1.0/x;
}

double RPN::uf_int(const double x) {
	return (double)((int) x);
}

double RPN::uf_round(const double x) {
	return (double)((int) (x + 0.5));
}

double RPN::uf_pow2(const double x) {
	return (double) pow(2,x);
}

/*
 * instance functions
 */
double RPN::push(const double & val) {
	stack.push_back(val);
	return val;
}

double RPN::pop() {
	double val = stack.back();
	stack.pop_back();
	return val;
}

double RPN::swap() {
	double val = pop();
	double x = pop();
	push(val);
	push(x);
	return x;
}

double RPN::perform(const string & func) {
	double val;
	val = pop();
	val = (*ufuncdict[func])(val);
	push(val);
	return val;
}

double RPN::referConstant(const string & name) {
	return constdict[name];
}


int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	string line, word;
	istringstream sstr;

	RPN::initialize();
	RPN rpn;

	double val;

	do {
		if ( rpn.size() != 0 ) {
			cout << " [" << rpn.size() - 1 << "] ";
			cout << rpn.peek();
		} else {
			cout << "[empty]";
		}
		cout << " > ";
		line.clear();
		getline(cin, line);

		sstr.str(line);
		sstr.clear();
		while (!sstr.eof()) {
			sstr >> word;
			char * rptr;
			val = strtod(word.c_str(), &rptr);
			if ( rptr == word.c_str() ) {
				if ( word == "+" || word == "-" || word == "*" || word == "/" ) {
					if ( !(rpn.size() >= 2) ) {
						cout << "Error: requires two arguments." << endl;
					} else {
						val = rpn.pop();
						switch (*word.c_str()) {
						case '+':
							val = rpn.pop() + val;
							break;
						case '-':
							val = rpn.pop() - val;
							break;
						case '*':
							val = rpn.pop() * val;
							break;
						case '/':
							if ( val == 0 ) {
								cout << "Error: divide by zero." << endl;
								rpn.push(val);
							} else {
								val = rpn.pop() / val;
							}
							break;
						}
						rpn.push(val);
						cout << val << endl;
					}
				}
				else if ( rpn.hasConstant(word) ) {
					val = rpn.referConstant(word);
					rpn.push(val);
					cout << val << endl;
				}
				else if ( rpn.hasUnaryFunction(word) ) {
					if ( !(rpn.size() >= 1) ) {
						cout << "Error: no arguments." << endl;
					} else {
						val = rpn.perform(word);
						cout << val << endl;
					}
				}
				else if ( word == "pop" ) {
					if ( rpn.size() == 0 ) {
						cout << "Stack is empty." << endl;
					} else{
						rpn.pop();
					}
				}
				else if ( word == "swap" ) {
					if ( rpn.size() >= 2 )
						rpn.swap();
				}
				else if ( word == "list" ) {
					for(int i = 0; i < rpn.size(); ++i) {
						cout << "[" << i << "] " << rpn[i] << endl;
					}
				}
				else if ( word == "sum" ) {
					val = 0;
					while ( !rpn.empty() ) {
						val += rpn.pop();
					}
					rpn.push(val);
					cout << val << endl;
				}
				else if ( word == "clr" || word == "cls" || word == "clear" ) {
					rpn.clear();
				} else {
					cout << "Error. " << endl;
				}
			} else {
				rpn.push(val);
			}
		}

	} while ( !cin.eof() && line != "quit" );

	return 0;
}
