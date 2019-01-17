//============================================================================
// Name        : BitArray.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
//============================================================================

#include <iostream>
#include <vector>

struct BitArray {
	unsigned char * bytearray;
	unsigned long bytecapacity, bitcapacity;

	BitArray(const unsigned long & n) {
		bitcapacity = n;
		bytecapacity = (n>>3) + ((n & 0x07) ? 1 : 0);
		if ( bytecapacity )
			bytearray = new unsigned char[bytecapacity];
	}

	~BitArray() {
		if ( bytecapacity )
			delete [] bytearray;
		bytecapacity = 0;
		bitcapacity = 0;
	}

	void clear_all(const unsigned char & val = 0) {
		unsigned char * ptr = bytearray;
		if ( val ) {
			for ( unsigned long i = 0; i < bytecapacity; ++i )
				*ptr = 0xff;
		} else {
			for ( unsigned long i = 0; i < bytecapacity; ++i )
				*ptr = 0;
		}
	}

	void bitset(const unsigned long & bpos) {
		unsigned long byteindex = (bpos >> 3);
		unsigned char bitmask = 1<<(bpos & 0x07);
		bytearray[byteindex] |= bitmask;
	}

	void bitclear(const unsigned long & bpos) {
		unsigned long byteindex = (bpos >> 3);
		unsigned char bitmask = 1<<(bpos & 0x07);
		bytearray[byteindex] &= ~bitmask;
	}

	unsigned int bitat(const unsigned long & bpos) const {
		unsigned long byteindex = (bpos >> 3);
		unsigned char bitmask = 1<<(bpos & 0x07);
		if ( bytearray[byteindex] & bitmask )
			return 1;
		else
			return 0;
	}

	friend std::ostream & operator<<(std::ostream & stream, const BitArray & barray) {
		for (unsigned long bpos = 0; bpos < barray.bitcapacity; ++bpos) {
			stream << (int) barray.bitat(bpos);
			if ( (bpos & 0x07) == 0x07)
				stream << "  ";
			else
				stream << " ";
		}
		return stream;
	}
};

struct BitStream {
	BitArray bitarray;
	unsigned char * byteptr;
	unsigned int bitptr;
	unsigned int maxpos;

	BitStream(BitArray & barray) : bitarray(barray){
		reset();
	}

	void reset() {
		byteptr = bitarray.bytearray;
		bitptr = 0;
		maxpos = 0;
	}

	void set(const unsigned long & bpos) {
		byteptr = bitarray.bytearray + (bpos >> 3);
		bitptr = bpos & 0x07;
		maxpos = bpos > maxpos ? bpos : maxpos;
	}

	void append(const unsigned int & boolval) {
		if ( boolval ) {
			*byteptr |= (1<<bitptr);
		} else {
			*byteptr &= ~(1<<bitptr);
		}
		forward();
		++maxpos;
	}

	void forward() {
		if ( bitptr < 7 ) {
			++bitptr;
			return;
		} else {
			bitptr = 0;
			++byteptr;
		}
	}

};

int main(void) {

	std::cout << "Hello World!!!" << std::endl;

	BitArray barray(32);
	BitStream bstream(barray);
	unsigned long val = 144065;
	unsigned char * p = (unsigned char *) &val;
	for (int i = 0; i < 8; ++i)
		std::cout << std::hex << static_cast<unsigned int>(*p++) << ' ';
	std::cout << std::endl;

	std::cout << "size of barray = " << sizeof(barray) << std::endl;
	unsigned long bit;
	unsigned int pos;
	for (bit = 1, pos = 0; pos < 31; bit <<= 1, ++pos) {
		bstream.append((bit & val) ? 1 :0) ;
	}
	std::cout << std::endl;
	//char t[16];
	//std::cin.getline(t,15);
	std::cout << barray << std::endl;
	return EXIT_SUCCESS;
}

