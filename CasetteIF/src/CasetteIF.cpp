//============================================================================
// Name        : CasetteIF.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

struct RingCounter {
  static const unsigned int RING_SIZE = 32;

  char ring[RING_SIZE];
  unsigned long counter[RING_SIZE];
  unsigned int outdex, index;

  RingCounter() {
    clear();
  }

  void clear() {
    outdex = 0;
    index = 0;
    for(int i = 0; i < RING_SIZE; ++i)
      counter[i] = 0;
  }

  bool empty() const {
    return outdex == index; // <==> total == 0
  }

  char peek() const {
    return ring[outdex];
  }

  unsigned long peekCount() const {
    return counter[outdex];
  }

  void enq(char c) {
    unsigned int prev = (index + 0x0f) & 0x0f;
    if ( ring[prev] == c ) {
      counter[prev]++;
      return;
    }
    ring[index] = c;
    counter[index] = 1;
    index = (index + 1) & 0x0f;
  }

  void deq() {
    if ( empty() )
      return;
    if ( counter[outdex] > 1 ) {
      counter[outdex]--;
      return;
    }
    counter[outdex] = 0;
    outdex = (outdex + 1) & 0x0f;
  }

  void clearLast() {
	if ( empty() )
		return;
    counter[outdex] = 0;
    outdex = (outdex + 1) & 0x0f;
  }
};

int main() {
	RingCounter ring;
	char c;

	while ( !std::cin.eof() ) {
		c = (char) std::cin.get();
		ring.enq(c);

		if ( ring.peekCount() > 0 && ring.peek() == '\r' ) {
			std::cout << "IDLE" << std::endl;
			ring.clearLast();
		}
		if ( ring.peekCount() >= 10000 && ring.peek() == '0' ) {
			std::cout << "LEAD0_10000" << std::endl;
			ring.clearLast();
		}
	}
	return 0;
}
