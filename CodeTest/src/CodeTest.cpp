//============================================================================
// Name        : CodeTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

class Tester {
	static int instance_counter;

	int id;
public:
	Tester(void) : id(0) {
		instance_counter++;
		cout << instance_counter << " void" << endl;
	}
	Tester(int i) : id(i) {
		instance_counter++;
		cout << instance_counter << " int " << i << endl;
	}
	Tester(const Tester & t) :id(t.id) {
		instance_counter++;
		cout << instance_counter << " &" << id << endl;
	}
	Tester(Tester & t) :id(t.id) {
		instance_counter++;
		cout << instance_counter << " call by value" << id << endl;
	}
	Tester(const Tester t) :id(t.id) {
		instance_counter++;
		cout << instance_counter << " call by value as const" << id << endl;
	}
	void set_id(int i) {
		id = i;
	}
};

int Tester::instance_counter = 0;

int main() {
	Tester t, u(3);

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	Tester v(u);

	return 0;
}
