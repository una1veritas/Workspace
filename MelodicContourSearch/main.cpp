#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <vector>
#include <array>

#include "libsmf/SMFEvent.h"
#include "libsmf/SMFStream.h"

#include "stringmatching.h"

//#define SHOW_EVENTSEQ

typedef unsigned int  uint;
typedef unsigned long ulong;

char contour_symbol(const char before, const char last){
	if ( before == -1 )
		return '.';
	char notediff = last - before;
	if ( notediff < 0 ) {
		if ( notediff < -2 )
			return '-';
		else
			return 'b';
	} else if ( notediff > 0 ) {
		if ( notediff > 2 )
			return '+';
		else
			return '#';
	}
	return '=';
}

std::string & translate(const char * filename, std::string & sequence) {
	std::fstream infile;
	std::array<std::string, 16> melody;
	uint last_time[16];
	struct {
		char last;
		char before; // before last
	} note[16];

	infile.open(filename, (std::ios::in | std::ios::binary) );
	if ( !infile ) {
		std::cerr << filename << " open failed." << std::endl;
		return melody[0];
	}

	SMFStream smf(infile);
	std::cout << "format = " << smf.format() << ", tracks = " << smf.tracks()  << ", resolution = " << smf.resolution() << std::endl;

	for(int i = 0; i < 16; ++i) {
		last_time[i] = -1;
		note[i].before = -1;
		note[i].last = -1;
	}
	uint delta_total = 0;
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		/*
		if ( evt.isMeta(SMFEvent::TIME) ) //|| evt.isMeta(SMFEvent::TEMPO))
			std::cout << evt << std::endl;
			*/
		if (evt.delta > 0) {
			delta_total += evt.delta; 	// advance the global clock.
		}
		if ( evt.isMTRK() ) {
			delta_total = 0;
			//last_total = (uint32) -1;
		} else {
			if ( last_time[evt.channel()] != delta_total ) {
				// if the global clock is advanced with resp. to channel clock
				// includes delta time == 0 (a note in another channel has been simultaneously on-ed.)
#ifdef SHOW_EVENTSEQ
				std::cout << std::endl << std::dec << delta_total << " ";
#endif //SHOW_EVENTSEQ
				last_time[evt.channel()] = delta_total;
				if ( note[evt.channel()].last != -1 ) {
					// clear the buffer (the last element) of melody queue.
					// push dummy as the last note,
					// assuming a possible note-off (or control) then note-on in the same time
					melody[evt.channel()].push_back(contour_symbol(note[evt.channel()].before, note[evt.channel()].last));
					note[evt.channel()].before = note[evt.channel()].last;
					note[evt.channel()].last = -1;
				}
				if ( evt.isNoteOn() ) {
					note[evt.channel()].last = evt.number;
				}
			} else {
				if ( evt.isNoteOn() ) {
					if ( evt.number > note[evt.channel()].last ) {
						// replace the dummy note number.
						note[evt.channel()].last = evt.number;
					}
				}
			}
#ifdef SHOW_EVENTSEQ
			if ( evt.isNoteOn() ) {
				std::cout << evt;
			}
#endif //ifdef SHOW_EVENTSEQ
		}
	}
	for(int i = 0; i < 16; ++i) {
		if ( note[i].last != -1 ) {
			melody[i].push_back(contour_symbol(note[i].before, note[i].last));
		}
	}
	sequence.clear();
	for(int i = 0; i < 16; ++i) {
		if ( melody[i].size() != 0 ) {
			sequence += melody[i];
			sequence.push_back( '\n' );
		}
	}
	return sequence;
}


int main(int argc, char **argv) {

	//const std::regex filepattern(".*\\.mid");
	std::string filename(argv[2]);
	kmp mcpat(argv[1]);

	std::cout << "file: " <<filename << std::endl;
	std::cout << "search for " << mcpat << std::endl << std::endl;

	std::string melody;
	translate(filename.c_str(), melody);
	std::cout << "size = "<< melody.size() << std::endl;

	std::cout << melody << std::endl << "finished." << std::endl;

	int res = mcpat.find(melody);
	if ( res < melody.size() ) {
		std::cout << "match found in " << filename << " " << res << " ` " << melody.size() << std::endl;
	}

	return 0;
}
