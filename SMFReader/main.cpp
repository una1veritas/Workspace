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

	uint32 delta_total = 0, last_total;
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		if (evt.delta > 0)
			delta_total += evt.delta;
		if ( evt.isMTRK() ) {
			delta_total = 0;
			last_total = (uint32) -1;
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
	 * binary file dump
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
