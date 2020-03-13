#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <vector>
#include <array>

#include "dirlister.h"
#include "libsmf/SMFEvent.h"
#include "libsmf/SMFStream.h"

#include "stringmatching.h"

//#include "manamatching.h"

//#define SHOW_EVENTSEQ

typedef unsigned int  uint;
typedef unsigned long ulong;

char contour_symbol(const char before, const char last){
	if ( before == -1 and last >= 0 )
		return '.';
	if ( before >= 0 and before == last )
		return '=';
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
	return '?'; // error
}

std::string & translate(const char * filename, std::string & sequence) {
	std::fstream infile;
	std::array<std::stringstream, 16> melody;
	uint last_time[16];
	struct {
		char last;
		char before; // before last
	} note[16];

	infile.open(filename, (std::ios::in | std::ios::binary) );
	if ( !infile ) {
		std::cerr << filename << " open failed." << std::endl;
		return sequence;
	}

	SMFStream smf(infile);
	//std::cout << "format = " << smf.format() << ", tracks = " << smf.tracks()  << ", resolution = " << smf.resolution() << std::endl;

	for(int i = 0; i < 16; ++i) {
		melody[i].str("");
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
					//melody[evt.channel()].push_back(contour_symbol(note[evt.channel()].before, note[evt.channel()].last));
					melody[evt.channel()] << contour_symbol(note[evt.channel()].before, note[evt.channel()].last);
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
			melody[i] << contour_symbol(note[i].before, note[i].last);
		}
	}
	sequence.clear();
	for(int i = 0; i < 16; ++i) {
		sequence += melody[i].str();
	}
	return sequence;
}


int main(int argc, char **argv) {

	std::string path;

	kmp mcpat(argv[1]);

	if (argc >= 3) {
		path = argv[2];
	} else {
		path = "./";
	}

	std::cout << "file path: " << path << std::endl;
	std::cout << "search for     " << mcpat << std::endl;
	std::cout << std::endl;

	//exit(1);

	dirlister dl(path.c_str());

	const std::regex regpat(".*\\.(mid|MID)$");
	int i;
	for(i = 1; dl.get_next_entry(regpat); ++i) {

		std::string melody;
		translate( (dl.fullpath() + "/" + dl.entry_name()).c_str(), melody);
		unsigned int res = mcpat.find(melody);
		if ( res < melody.size() ) {
			std::cout << i << ": " << dl.fullpath() << "/"<< dl.entry_name() << std::endl;
			std::cout << "match found at " << res << " in " << melody.size() << " notes." << std::endl;
			std::cout << std::endl;
		} else {
			//std::cout << "no match." << std::endl;
		}
	}

	std::cout << std::endl << "Finished search among " << i-1 << " files." << std::endl;
	return 0;
}
