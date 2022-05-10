#include <iostream>
//#include <iomanip>
//#include <string>
#include <fstream>
//#include <sstream>
//#include <cctype>
//#include <cstring>

#include <vector>

#include "smf.h"

int main(int argc, char **argv) {
	std::ifstream ifile;

	std::cout << "file: " << argv[1] << std::endl;
	ifile.open(argv[1], (std::ios::in | std::ios::binary) );
	if ( !ifile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}
	std::istreambuf_iterator<char> smfbuf(ifile);

	smf::score midi(smfbuf);
	ifile.close();

	uint64_t globaltime = 0;
	std::vector<smf::event>::const_iterator cursor[midi.numoftracks()], ends[midi.numoftracks()];
	uint32_t remaining[midi.numoftracks()];
	for(int i = 0; i < midi.numoftracks(); ++i) {
		cursor[i] = midi.track(i).events.begin();
		remaining[i] = cursor[i]->delta;
		ends[i] = midi.track(i).events.end();
	}
	uint32_t mindelta;
	bool alleot;
	while (globaltime < 100000) {
		mindelta = 0xffffffff;
		alleot = true;
		for(int i = 0; i < midi.numoftracks(); ++i) {
			if ( cursor[i] != ends[i] ) {
				if ( mindelta > remaining[i] ) {
					mindelta = remaining[i] ;
				}
				alleot = false;
			}
		}
		if ( alleot ) {
			break;
		} else {
			std::cout << "min delta = " << mindelta << std::endl;
		}
		for(int i = 0; i < midi.numoftracks(); ++i) {
			if ( cursor[i] == ends[i] )
				continue;
			remaining[i] -= mindelta;
			while ( remaining[i] == 0 && cursor[i] != ends[i] ) {
				//std::cout << *cursor[i] << ", ";
				++cursor[i];
				remaining[i] += cursor[i]->delta;
			}
			if ( cursor[i] != ends[i] )
				std::cout << *cursor[i] << std::endl;
		}
		globaltime += mindelta;
		for(int i = 0; i < midi.numoftracks(); ++i) {
			std::cout << std::dec << remaining[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "global time = " << globaltime << std::endl << std::endl;
	}

	std::cout << " done. " << std::endl;

	return 0;
}
