#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

typedef uint64_t uint64;
class BitStream {
	std::vector<uint64> word;
	uint64 current_word;
	unsigned int bit_pos;

	inline static void rotl64(uint64 & x) {
		// assert (n<32);
		x = (x<<1) | (x>>(-1&63));
	}

	inline static void rotl64(uint64_t & x, const unsigned int & n) {
		// assert (n<32);
		x = (x<<n) | (x>>(-n & 63));
	}

public:
	BitStream(const unsigned int & init_size = 256) :
		word(init_size>>5), current_word(0), bit_pos(0) {
	}

	void append(const unsigned int & bit) {
		current_word |= (uint64) (bit != 0);
		rotl64(current_word);
		++bit_pos;
		if ( bit_pos == 64 ) {
			word.push_back(current_word);
			bit_pos = 0;
			current_word = 0;
		}
	}
/*
	unsigned int bitcount(void) const { return ((words.size() - 1)<<5) + bitpos; }
	bool empty() const { return (wordpos == words.begin()) && (bitpos == 0); }
*/
	/*
	bool open() { wordpos = words.begin(); bitpos = 0; return true; }
	void reset() { flush(); open(); }
	void flush() { rotl64(cache_word, 64 - bitpos); *wordpos = cache_word; }
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
		for (unsigned int i = 0 ; i < bstream.word.size() ; ++i) {
			stream << std::hex << bstream.word[i];
			stream << " ";
		}
		return stream;
	}

};

int main(int argc, char **argv) {
	std::cout << "Hello, there." << std::endl;

	BitStream bstream(500);
	bstream.append(1);
	std::cout << bstream << std::endl;
	return EXIT_SUCCESS;
}
