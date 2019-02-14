#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

typedef uint64_t uint64;
class BitStream {
	std::vector<uint64> words;
	std::vector<uint64>::iterator wordpos;
	unsigned int bitpos;
	uint64 cache_word;

	inline static uint64 rotl64(const uint64_t & x) {
		// assert (n<32);
		return (x<<1) | (x>>(-1&63));
	}

	inline static uint64 rotl64(const uint64_t & x, const unsigned int & n) {
		// assert (n<32);
		return (x<<n) | (x>>(-n & 63));
	}

public:
	BitStream(const unsigned int & init_size = 256) :
		words(init_size>>5), cache_word(0) {
		wordpos = words.end();
		bitpos = 0;
	}

	unsigned int bitcount(void) const { return ((words.size() - 1)<<5) + bitpos; }
	bool empty() const { return (wordpos == words.begin()) && (bitpos == 0); }

	bool open() { wordpos = words.begin(); bitpos = 0; return true; }
	void reset() { flush(); open(); }
	void flush() { rotl64(cache_word, 64 - bitpos); *wordpos = cache_word; }
	bool close() { flush(); wordpos = words.end(); bitpos = 0; return true; }

	uint64 read64() {
		cache_word = *wordpos;
		return cache_word;
		++wordpos;
	}
	//
	friend std::ostream & operator<<(std::ostream & stream, const BitStream & bstream) {
		uint64 octword;
		for (unsigned int i = 0 ; i < bstream.words.size() ; ++i) {
			for(unsigned int j = 0; j < sizeof(uint64)*8; ++j) {
				stream << (unsigned int )(1 & ((unsigned int) rotl64(bstream.words[i],j)));
			}
			stream << " ";
		}
		return stream;
	}

};

int main(int argc, char **argv) {
	std::cout << "Hello, there." << std::endl;

	BitStream bstream(500);
	std::cout << bstream << std::endl;
	return EXIT_SUCCESS;
}
