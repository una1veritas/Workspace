#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>

#include "libsmf/SMFEvent.h"
#include "libsmf/SMFStream.h"

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

int main(int argc, char **argv) {
	std::fstream infile;

	std::cout << "file: " << argv[1] << std::endl;
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

	uint32 delta_total = 0, last_total;
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		if (evt.delta > 0)
			delta_total += evt.delta;
		if ( evt.isMTRK() ) {
			delta_total = 0;
			last_total = 0xffffffff;
		}
		if ( evt.isNoteOn() ) {
			if ( last_total != delta_total ) {
				std::cout << std::endl << std::dec << delta_total << " ";
				last_total = delta_total;
			}
			std::cout << evt << " ";
		}
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
