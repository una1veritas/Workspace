//============================================================================
// Name        : OperatorTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstring>

using namespace std;

class TestObject {
private:
	unsigned char age;
	char name[256];

public:
	TestObject(const char * objname) {
		strncpy(name, objname, 255);
		name[255] = (char) 0;
		age = (unsigned char) strlen(name);
	}

	friend ostream & operator<<(ostream & out, const TestObject & obj) {
		out << "(" << obj.name << ": " << (unsigned int) obj.age << ") ";
		return out;
	}

	TestObject & operator=(const TestObject & anotherobj) {
		strcpy(name, anotherobj.name);
		age = anotherobj.age;
		return *this;
	}

	TestObject & operator=(const unsigned char val) {
		age = val;
		return *this;
	}

	operator unsigned char() const {
		return age;
	}
};

int main() {
	cout << "Hello World!!!" << endl; // prints Hello World!!!

	TestObject obj1("Tom"), obj2("Meg");
	cout << obj1 << ", " << obj2 << endl;

	obj1 = obj2;
	cout << obj1 << ", " << obj2 << endl;

	obj1 = 34;
	cout << obj1 << ", " << obj2 << endl;

	unsigned char val = obj2;
	cout << obj2 << ", " << (int) val << endl;

	return 0;
}
