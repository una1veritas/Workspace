#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>

#include "SMFReader.h"
#include "SMFStream.h"

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

int main(int argc, char **argv) {
	std::fstream infile;

	infile.open(argv[1], (std::ios::in | std::ios::binary) );

	if ( !infile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}

	SMFStream smf(infile);

	// 4d 54 68 64 00 00 00 06 00 00 00 01 00 78
	// 4d 54 72 6b 00 00 1e 71
	// 00 ff 03 0a 50 61 6e 7a 65 72 6c 69 65 64
	// 00 ff 02 24 43 6f 70 79 72 69 67 68 74 20 5f 20 32 30 30 33 20 62 79 20 4a 6f 68 61 6e 6e 20 53 63 68 72 65 69 62 65 72
	// 00 ff 58 04 04 02 18 08
	// 00 ff 59 02 fe 00
	// 00 ff 51 03 08 52 af
	// 00 c0 49
	// 00 b0 07 50
	// 00    0a 19
	// 00 c1 47 00 b1
	/*
	07 5f 00 0a 64 00 c2 3a 00 b2 07 7f 00 0a 6e 00
	c3 39 00 b3 07 64 00 0a 5a 00 c4 38 00 b4 07 5a
	00 0a 1e 00 c5 48 00 b5 07 4b 00 0a 14 00 c6 09
	00 b6 07 41 00 0a 32 00 c7 3c 00 b7 07 5a 00 0a
	28 00 b9 07 41 00 0a 46 82 68 b0 5d 0a 00 5b 0f
	00 90 41 64 00 b1 5d 14 00 5b 0f 00 91 3e 64 00
	b2 5b 1e 00 b3 5b 1e 00 b4 5d 05 00 5b 1e 00 94
	41 64 00 b5 5d 0a 00 5b 14 00 95 4d 64 00 b6 5b
	23 00 96 4d 64 6e 94 41 00 0a 95 4d 00 00 90 41
	*/
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		std::cout << evt << std::endl;
	}
	/*
	uint8 buf[256];
	smf.read_byte(buf,256);
	for(int i = 0; i < 256; ++i) {
		if ( i && (i & 0x0f) == 0 )
			std::cout << std::endl;
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	*/

	std::cout << std::endl;
	return 0;
}
