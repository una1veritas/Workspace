#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

typedef uint64_t uint64;
class BitStream {
	std::vector<uint64> octwords;
	unsigned int bitsize;
	unsigned int head, tail;
	uint64 cache;
	unsigned int cached_oct, cached_bit;

	inline static uint64 rotl64(const uint64_t & x) {
		// assert (n<32);
		return (x<<1) | (x>>(-1&63));
	}

	inline static uint64 rotl64(const uint64_t & x, const unsigned int & n) {
		// assert (n<32);
		return (x<<n) | (x>>(-n & 63));
	}

public:
	BitStream(const unsigned int & maxsize = 256) :
		octwords(maxsize>>5), bitsize(maxsize), head(0), tail(0),
		cache(0), cached_oct(0), cached_bit(0) { }

	unsigned int count() const { return tail - head; }
	bool empty() const { return tail > head; }

	//
	friend std::ostream & operator<<(std::ostream & stream, const BitStream & bstream) {
		uint64 octword;
		for (unsigned int i = 0 ; i < bstream.octwords.size() ; ++i) {
			for(unsigned int j = 0; j < sizeof(uint64)*8; ++j) {
				stream << (unsigned int )(1 & ((unsigned int) rotl64(bstream.octwords[i],j)));
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
