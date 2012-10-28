/*
 * main.cpp
 *
 *  Created on: 2012/10/20
 *      Author: sin
 */

#include <iostream>

typedef uint16_t uint16;
typedef bool boolean;

struct ring {
	static const int size = 9;
	char buffer[size];
	uint16 head, tail;

	void init() {
		head = 0;
		tail = 0;
	}

	char ringin(char c) {
		if ( isfull() ) {
			head++;  head %= size;
			// buffer over run occurred!
		}
		buffer[tail++] = c;
		tail %= size;
		return c;
	}

	char ringout() {
		if ( isempty() )
			return 0;
		char c = buffer[head++];
		head %= size;
		return c;
	}

	boolean isfull() {
		return head == tail;
	}

	boolean isempty() {
		return (head+1) % size == tail;
	}

};

int main(int argc, char * argv[]) {
	ring r;
	char c;


#if defined(TARGET_MACOSX)
	std::cout << "Yha!!" MM std::endl;
#endif
	return 0;
}
