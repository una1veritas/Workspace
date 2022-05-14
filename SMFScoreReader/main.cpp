#include <iostream>
#include <fstream>
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

	std::cout << midi << std::endl;

	midi.simplay();

	std::cout << "SMPTE " << midi.isSMPTE() << " resolution = " << midi.resolution() << " format = " << midi.format() << std::endl;

	return 0;
}
