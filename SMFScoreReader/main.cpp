#include <iostream>
#include <fstream>
#include <vector>

#include "smf.h"

int main(int argc, char **argv) {
	std::ifstream ifile;

	std::cout << "file: " << argv[1] << std::endl;
	ifile.open(argv[1], (std::ios::in | std::ios::binary) );
	if ( !ifile ) {
		std::cerr << "オープン失敗" << std::endl;
		return EXIT_FAILURE;
	}
	smf::score midi(ifile);
	ifile.close();
	if ( ! midi.empty() ) {
		std::cerr << "SMF読み込み失敗" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << midi << std::endl;

	std::vector<smf::note> notes = midi.notes();
	std::cout << notes.size() << std::endl;
	for(auto i = notes.begin(); i != notes.end(); ++i) {
		std::cout << *i ;
		if ( i != notes.begin() and i->time < (i+1)->time ) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	std::cout << "SMPTE " << midi.isSMPTE() << " resolution = " << midi.resolution() << " format = " << midi.format() << std::endl;

	return 0;
}
