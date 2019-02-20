#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

typedef uint64_t uint64;
class BitStream {
	std::vector<uint64> words;
	unsigned int bitcount;

	inline static void rotl64(uint64 & x) {
		// assert (n<32);
		x = (x<<1) | (x>>(-1&63));
	}

	inline static void rotl64(uint64_t & x, const unsigned int & n) {
		// assert (n<32);
		x = (x<<n) | (x>>(-n & 63));
	}

public:
	BitStream(void) :
		words(), bitcount(0) {
	}

	void append(const unsigned int & bit) {
		uint64 word = (bit != 0);
		if ( (bitcount & 0x3f ) == 0 ) {
			word <<= 63;
			words.push_back(word);
		} else {
			word <<= (63 - (bitcount & 0x3f));
			words[words.size() - 1] |= word;
		}
		++bitcount;
	}
/*
	unsigned int bitcount(void) const { return ((words.size() - 1)<<5) + bitpos; }
	bool empty() const { return (wordpos == words.begin()) && (bitpos == 0); }
*/
	/*
	bool open() { wordpos = words.begin(); bitpos = 0; return true; }
	void reset() { flush(); open(); }
	void flush() {
	}
	bool close() { flush(); wordpos = words.end(); bitpos = 0; return true; }
*/
	/*
	uint64 read64() {
		cache_word = *wordpos;
		return cache_word;
		++wordpos;
	}
	*/
	//

	friend std::ostream & operator<<(std::ostream & stream, const BitStream & bstream) {
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

	BitStream bstream;
	bstream.append(1);
	bstream.append(0);
	bstream.append(1);
	std::cout << bstream << std::endl;
	return EXIT_SUCCESS;
}
