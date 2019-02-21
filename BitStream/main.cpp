#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

class BitArray {
	typedef uint64_t uint64;
	std::vector<uint64> words;
	unsigned int bitcount;
	uint64 current_word;
	unsigned int current_bit_pos;

	inline static void rotl64(uint64 & x) {
		// assert (n<32);
		x = (x<<1) | (x>>(-1&63));
	}

	inline static void rotl64(uint64_t & x, const unsigned int & n) {
		// assert (n<32);
		x = (x<<n) | (x>>(-n & 63));
	}

public:
	BitArray(const unsigned int & max_bits = 64) :
		words(max_bits>>6), bitcount(max_bits), current_word(0), current_bit_pos(0) { }

	void clearAll(void) {
		for(auto i = words.begin(); i != words.end(); ++i)
			*i = 0;
	}

	bool set(const unsigned int & pos, const unsigned int & bit) {
		if ( bit != 0 ) {
			words[pos>>6] |= ((uint64) 1) << (pos & 0x3f);
			return 1;
		} else {
			words[pos>>6] &= ~(((uint64) 1) << (pos & 0x3f));
			return 0;
		}
	}
	//

	friend std::ostream & operator<<(std::ostream & stream, const BitArray & bstream) {
		uint64 octword;
		for (unsigned int i = 0 ; i < bstream.words.size() ; ++i) {
			stream << std::hex << bstream.words[i];
			stream << " ";
		}
		return stream;
	}

};

int main(int argc, char **argv) {
	std::cout << "Hello, there." << std::endl;

	BitArray bstream(128);
	bstream.set(0,1);
	bstream.set(1,0);
	bstream.set(2,1);
	std::cout << bstream << std::endl;
	return EXIT_SUCCESS;
}
