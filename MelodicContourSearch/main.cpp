#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>
#include <array>

#include "SMFEvent.h"
#include "SMFStream.h"

#include "kmp.h"
#include "DirectoryLister.h"

std::array<std::vector<int8>, 16> translate(const char * filename) {
	std::fstream infile;
	std::array<std::vector<int8>, 16> melody;
	int last_time[16];

	infile.open(filename, (std::ios::in | std::ios::binary) );
	if ( !infile ) {
		std::cerr << filename << " open failed." << std::endl;
		return melody;
	}

	SMFStream smf(infile);

	for(int i = 0; i < 16; ++i) {
		melody[i].push_back(-1);
		last_time[i] = -1;
	}
	uint32 delta_total = 0;
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		if (evt.delta > 0)
			delta_total += evt.delta; 	// advance the global clock.
		if ( evt.isMTRK() ) {
			delta_total = 0;
			//last_total = (uint32) -1;
		} else {
			if ( last_time[evt.channel()] != delta_total ) {
				// if the global clock is advanced with resp. to channel clock
				// includes delta time == 0 (a note in another channel simultaneously on-ed.)
#ifdef SHOW_EVENTSEQ
				std::cout << std::endl << std::dec << delta_total << " ";
#endif //SHOW_EVENTSEQ
				last_time[evt.channel()] = delta_total;
				if ( melody[evt.channel()].back() != -1 ) {
					// clear the buffer (the last element) of melody queue.
					// push dummy as the last note,
					// assuming a possible note-off (or control) then note-on in the same time
					melody[evt.channel()].push_back(-1);
				}
				if ( evt.isNoteOn() ) {
					melody[evt.channel()].back() = evt.number;
				}
			} else {
				if ( evt.isNoteOn() ) {
					if ( evt.number > melody[evt.channel()].back() ) {
						// force the dummy to be replaced.
						melody[evt.channel()].back() = evt.number;
					}
				}
			}
#ifdef SHOW_EVENTSEQ
			std::cout << evt;
#endif //ifdef SHOW_EVENTSEQ
		}
	}
	for(int i = 0; i < 16; ++i)
		if ( melody[i].back() == -1) melody[i].pop_back();

	return melody;
}

int main(int argc, char **argv) {

	std::cout << "path = " << argv[1] << " melodic contour = " << argv[2] << std::endl;

	const std::regex filepattern(".*\\.mid");
	DirectoryLister dlister(argv[2]);
	kmp mcpat(argv[1]);

	if ( ! dlister() ) {
		std::cerr << "error: opendir returned a NULL pointer for the base path." << std::endl;
		exit(1);
	}

	std::array<std::vector<int8>, 16> melody;
	bool matched;
	std::cout << "search for " << mcpat << std::endl << std::endl;
	int i;
	for(i = 1; dlister.get_next_file(filepattern) != NULL; ++i) {
		melody = translate(dlister.entry_path().c_str());
		matched = false;
		for(int ch = 0; ch < melody.size(); ++ch) {
			if ( melody[ch].size() == 0 )
				continue;
			int res = mcpat.search(melody[ch]);
			if ( res < melody[ch].size() ) {
				matched = true;
				std::cout << ch << " (" << melody[ch].size() << ") @" << res << " ";
			}
		}

		if ( matched ) {
			std::cout << dlister.entry_path().c_str() << std::endl;
		}
	}
	std::cout << i << "songs. " << std::endl;
	return 0;
}
