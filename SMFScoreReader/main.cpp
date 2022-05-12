#include <iostream>
#include <fstream>
#include <vector>

#include "smf.h"

struct smfplayer {
	smf::score & score;
	uint64_t globaltime;
	std::vector<std::vector<smf::event>::const_iterator> cursors;

	smfplayer(smf::score & mid) : score(mid) {
		initialize();
	}

	void initialize() {
		globaltime = 0;
		for(int i = 0; i < score.noftracks() ; ++i) {
			//cursors.push_back(score.track(i).events.begin());
		}
	}
};

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
	std::vector<smf::event>::const_iterator cursor[midi.noftracks()];
	//, ends[midi.noftracks()];
	uint32_t remaining[midi.noftracks()];
	for(int i = 0; i < midi.noftracks(); ++i) {
		cursor[i] = midi.track(i).events.begin();
		while (cursor[i]->delta == 0 and ! cursor[i]->isEoT() )
			++cursor[i];
		remaining[i] = cursor[i]->delta;
		//ends[i] = midi.track(i).events.end();
	}
	uint32_t minremaining;
	while (true) {
		minremaining = 0;
		for(int i = 0; i < midi.noftracks(); ++i) {
			if ( cursor[i]->isEoT() )
				continue;
			if ( minremaining == 0 or minremaining > remaining[i] ) {
				minremaining = remaining[i] ;
			}
		}
		if ( minremaining == 0 )
			break;
		std::cout << "min remaining = " << minremaining << std::endl;
		for(int i = 0; i < midi.noftracks(); ++i) {
			if ( cursor[i]->isEoT() )
				continue;
			remaining[i] -= minremaining;
			if ( remaining[i] == 0 ) {
				// event occurs.
			}
			while ( remaining[i] == 0 && ! cursor[i]->isEoT() ) {
				// event occurring simultaneously.
				++cursor[i];
				remaining[i] += cursor[i]->delta;
			}
		}
		globaltime += minremaining;
		for(int i = 0; i < midi.noftracks(); ++i) {
			std::cout << std::dec << remaining[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "global time = " << globaltime << std::endl << std::endl;
	}

	std::cout << " done. " << std::endl;

	return 0;
}
