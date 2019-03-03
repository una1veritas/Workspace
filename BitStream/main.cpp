#include <cstdlib>
#include <iostream>

#include <cinttypes>
#include <vector>

class bit64array {
	typedef uint64_t uint64;
	uint64 bitarray;
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
	bit64array(void) : bitarray(0), bitcount(0) { }

	void appendBit(const bool & b) {
		rotl64(bitarray);
		bitarray |= (b? 1 : 0);
		++bitcount;
	}

	void append(const char * str) {
		while (*str) {
			if ( (sizeof(uint64)<<3) - bitcount >= (sizeof(char)<<3) ) {
				rotl64(bitarray,8);
				bitarray |= (uint64)(*str);
				bitcount += 8;
				++str;
				std::cout << "." << std::flush;

			} else {
				for (unsigned int i = (sizeof(char)<<3) - ((sizeof(uint64)<<3) - bitcount);
						i < sizeof(char)<<3; ++i) {
					appendBit(((*str)>>i) & 1);
				}
				std::cout << "!" << std::flush;
				return;
			}
		}
	}

	void append(const uint32_t & val) {
		const unsigned int val_size =  sizeof(uint32_t);
		for (unsigned int i = 0; i < val_size; ++i ) {
			rotl64(bitarray,8);
			bitarray |= (val >> ((val_size- 1 - i)<<3)) & 0xff;
			bitcount += 8;
		}
	}

	bool is_full(void) const {
		return bitcount == 64;
	}

	unsigned int size() const { return (bitcount & 0x3f) + (bitcount & 0x40); }

	bool operator[](const unsigned int i) const {
		return bitarray & (1<<((bitcount-1-i) & 0x3f));
	}

	//

	friend std::ostream & operator<<(std::ostream & stream, const bit64array & barray) {
		std::cout << barray.size() << std::endl;
		for (unsigned int i = 0 ; i < barray.size() ; ++i) {
			stream << std::hex << barray[i];
			stream << " ";
		}
		return stream;
	}

};

int main(int argc, char **argv) {
	std::cout << "Hello, there." << std::endl;

	bit64array barray;
	barray.appendBit(1);
	barray.append("A stack.");
	/*
	barray.append((bool)0);
	barray.append((bool)1);
	barray.append((bool)1);
	barray.append((bool)0);
	*/
	//barray.append((uint32_t)0x2bda0);
	std::cout << "unsigned int size = " << sizeof(unsigned int) << std::endl;
	std::cout << "bit64array (" << barray.size() << ") = " << barray << std::endl;
	return EXIT_SUCCESS;
}
