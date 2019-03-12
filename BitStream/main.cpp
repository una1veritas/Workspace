#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

class bitstream {

	typedef uint64_t uint64;

	std::vector<uint64> bit64array;
	unsigned int bitcount;
	uint64 cache64;

	inline static uint64 & rotl64(uint64 & x) {
		return x = (x<<1) | (x>>(-1&63));
	}

	inline static uint64 & rotl64(uint64_t & x, const unsigned int & n) {
		return x = (x<<n) | (x>>(-n & 63));
	}

	inline static uint64 & rotr64(uint64 & x) {
		return x = (x>>1) | (x<<(-1&63));
	}

	inline static uint64 & rotr64(uint64_t & x, const unsigned int & n) {
		return x = (x>>n) | (x<<(-n & 63));
	}

public:
	bitstream(void) : bit64array(), bitcount(0), cache64(0) {}
	unsigned int bit_count() const { return bitcount; }

	void append(unsigned char bit) {
		cache64 |= (bit != 0);
		// post process
		rotr64(cache64);
		if ( (bitcount & 63) == 63 ) {
			bit64array.push_back(cache64);
			cache64 = 0;
		}
		++bitcount;
	}

	bool operator[](const unsigned int i) const {
		uint64 t = 1;
		uint64 val;
		if ( i < (bitcount & ~((unsigned int)63)) ) {
			val = bit64array[i>>6];
			return 1 & (val>>(i&63));
		}
		rotr64(t, (bitcount & 63) - i);
		return cache64 & t;
	}

		std::cout << barray.size() << std::endl;
	friend std::ostream & operator<<(std::ostream & stream, const bitstream & bstream) {
		stream << "bit count = " << bstream.bitcount << ", ";
		stream << "array length = " << bstream.bit64array.size() << ", " << std::endl;
		for(unsigned int i = 0; i < bstream.bit_count(); ++i) {
			if ( (i & 0x7) == 0)
				stream << " ";
			stream << std::hex << bstream[i];// << " ";
		}

		stream << ", ";
		return stream;
	}

};

int main(int argc, char **argv) {
	const char message[] = "HELLO, there.\n";
	std::cout << message << std::flush;

	bitstream bstream;
	for(unsigned int ix = 0; ix < 11 && message[ix]; ++ix) {
		for(unsigned int bp = 0; bp < 8; ++bp) {
			bstream.append((message[ix]>>(7-bp)) & 1);
		}
	}
	/*
	bstream.append(1);
	bstream.append(0);
	bstream.append(1);
	bstream.append(1);
	bstream.append(1);
	bstream.append(0);
	bstream.append(1);
	bstream.append(1);
*/
	std::cout << "bit count = " << bstream.bit_count() << std::endl;
	std::cout << "now. " << std::flush;
	std::cout << bstream << std::endl;

	std::cout << "bye." << std::endl;
	return EXIT_SUCCESS;
}
