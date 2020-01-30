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

typedef unsigned int  uint;
typedef unsigned long ulong;

std::string & translate(const char * filename, std::string & sequence) {
	std::fstream infile;
	std::array<std::string, 16> melody;
//	std::string sequence;
	uint last_time[16];

	infile.open(filename, (std::ios::in | std::ios::binary) );
	if ( !infile ) {
		std::cerr << filename << " open failed." << std::endl;
		return melody[0];
	}

	SMFStream smf(infile);
	std::cout << "format = " << smf.format() << ", tracks = " << smf.tracks()  << ", resolution = " << smf.resolution() << std::endl;

	for(int i = 0; i < 16; ++i) {
		melody[i].push_back(-1);
		last_time[i] = -1;
	}
	uint delta_total = 0;
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		/*
		if ( evt.isMeta(SMFEvent::TIME) ) //|| evt.isMeta(SMFEvent::TEMPO))
			std::cout << evt << std::endl;
			*/
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
	for(int i = 0; i < 16; ++i) {
		if ( melody[i].back() == -1) melody[i].pop_back();
		if ( melody[i].size() != 0 ) {
			for(int j = 0; j < melody[i].size(); ++j) {
				sequence.push_back(melody[i][j]);
			}
			sequence.push_back( -'\n' );
		}
	}
	return sequence;
}

std::stringstream & notename(std::stringstream & ssout, const int & nn) {
	switch(nn%12) {
	case 0:
		ssout << "C";
		break;
	case 1:
		ssout << "C#";
		break;
	case 2:
		ssout << "D";
		break;
	case 3:
		ssout << "D#";
		break;
	case 4:
		ssout << "E";
		break;
	case 5:
		ssout << "F";
		break;
	case 6:
		ssout << "F#";
		break;
	case 7:
		ssout << "G";
		break;
	case 8:
		ssout << "G#";
		break;
	case 9:
		ssout << "A";
		break;
	case 10:
		ssout << "A#";
		break;
	case 11:
		ssout << "B";
		break;
	}
	ssout << (nn/12 - 2);
	return ssout;
}

int main(int argc, char **argv) {

	//const std::regex filepattern(".*\\.mid");
	std::string filename(argv[2]);
	kmp mcpat(argv[1]);

	std::cout << "file: " <<filename << std::endl;
	std::cout << "search for " << mcpat << std::endl << std::endl;

	std::string melody;
	translate(filename.c_str(), melody);

	std::stringstream contour;
	int lastnote;
	int notecounter = 0;
	int track = 0;
	for(auto iter = melody.begin(); iter != melody.end(); ++iter) {
		if ( *iter == -'\n') {
			contour << std::endl;
			++track;
			notecounter = 0;
			continue;
		} else if ( notecounter == 0 ) {
			lastnote = *iter;
			++notecounter;
		} else {
			if ( lastnote == *iter ) {
				contour << '=';
			} else if ( lastnote > *iter ) {
				if ( lastnote > *iter + 2 )
					contour << '-';
				else
					contour << 'b';
			} else if ( lastnote < *iter ) {
				if ( lastnote + 2 < *iter )
					contour << '+';
				else
					contour << '#';
			}
			lastnote = *iter;
			++notecounter;
		}
	}
	contour << std::endl;
	std::cout << contour.str() << std::endl << "finished." << std::endl;
	/*
	int res = mcpat.search(contour);
	if ( res < melody.size() ) {
		std::cout << "match found in " << filename << " " << res << " ` " << melody.size() << std::endl;
	}
	 */
	return 0;
}
