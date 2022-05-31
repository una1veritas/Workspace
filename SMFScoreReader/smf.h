/*
 * smf.h
 *
 *  Created on: 2022/05/10
 *      Author: sin
 */

#ifndef SMF_H_
#define SMF_H_

#include <iostream>
#include <iomanip>
#include <cinttypes>
#include <vector>

namespace smf {

uint32_t get_uint32BE(std::istreambuf_iterator<char> & itr);
uint32_t get_uint16BE(std::istreambuf_iterator<char> & itr);
uint32_t get_uint32VLQ(std::istreambuf_iterator<char> & itr);
bool check_str(const std::string & str, std::istreambuf_iterator<char> & itr);

int octave(uint8_t notenum);
const char * notename(uint8_t notenum);

enum EVENT_TYPE {
	MIDI_NOTEOFF = 0x80,
	MIDI_NOTEON = 0x90,
	MIDI_POLYKEYPRESSURE = 0xa0,
	MIDI_CONTROLCHANGE = 0xb0,
	MIDI_PROGRAMCHANGE = 0xc0,
	MIDI_CHPRESSURE = 0xd0,
	MIDI_PITCHBEND = 0xe0,
	SYSEX = 0xf0, 	// System Exclusive
	ESCSYSEX = 0xf7, 	// Escaped System Exclusive
	META = 0xff, 	// Meta
};

static constexpr char * namesofnote[] = {
		(char *) "C",
		(char *) "C#",
		(char *) "D",
		(char *) "D#",
		(char *) "E",
		(char *) "F",
		(char *) "F#",
		(char *) "G",
		(char *) "G#",
		(char *) "A",
		(char *) "A#",
		(char *) "B",
		(char *) "",
};

struct event {
	uint32_t delta;
	uint8_t  status;
	std::string data;


	static int octave(uint8_t notenum) {
		return (notenum / 12) - 1;
	}

	static const char * notename(uint8_t notenum) {
		return smf::namesofnote[notenum % 12];
	}

	event(void) {
		clear();
	}

	event(std::istreambuf_iterator<char> & itr, uint8_t laststatus);

	void clear() {
		delta = 0;
		status = 0;
		data.clear();
	}

	~event() {
		data.clear();
	}

	bool isMeta(void) const {
		return status == META;
	}

	bool isEoT(void) const {
		return isMeta() && data[0] == 0x2f;
	}

	bool isTempo() const {
		return isMeta() && (data[0] == 0x51);
	}

	uint32_t tempo() const {
		if ( isTempo() )
			return 0;
		uint32_t tmp = 0;
		tmp = uint8_t(data[1]);
		tmp <<= 8;
		tmp |= uint8_t(data[2]);
		tmp <<= 8;
		tmp |= uint8_t(data[3]);
		return tmp;
	}

	bool isNote() const {
		if ( (status & 0xe0) == 0x80 ) {
			return true;
		}
		return false;
	}

	bool isNoteOn() const {
		if ( (status & 0xf0) == smf::MIDI_NOTEON ) {
			return data[1] > 0;
		}
		return false;
	}

	bool isNoteOff() const {
		return (status & 0xf0) == smf::MIDI_NOTEOFF;
	}

	int channel(void) const {
		return 0x0f & status;
	}

	int octave() const {
		if ( isNote() )
			return smf::event::octave(data[0]);
		return -2;
	}

	const uint32_t & deltaTime() const {
		return delta;
	}

	const char * notename() const {
		if ( !isNote() )
			return namesofnote[12];
		return smf::event::notename(data[0]);
	}

	int notenumber() const {
		if ( !isNote() )
			return 128;
		return int(data[0]);
	}

	std::ostream & printOn(std::ostream & out) const;

	std::ostream & printData(std::ostream & out, uint32_t offset = 1) const {
		for(auto i = data.cbegin() + offset; i != data.cend(); ++i) {
			out << *i;
		}
		return out;
	}

	friend std::ostream & operator<<(std::ostream & out, const event & evt) {
		return evt.printOn(out);
	}

};

/*
class header {
	uint16_t length;

public:
	uint16_t format, ntracks, division;

	header(void) : length(0), format(0), ntracks(0), division(0) { }

	header(std::istreambuf_iterator<char> & itr) {
		length = get_uint32BE(itr);
		//std::cout << "header" << std::endl;
		format = get_uint16BE(itr);
		ntracks = get_uint16BE(itr);
		division = get_uint16BE(itr);
	}

	friend std::ostream & operator<<(std::ostream & out, const header & chunk) {
		out << "Header";
		out << "(format = " << chunk.format << ", ntracks = " << chunk.ntracks << ", division = " << chunk.division << ") ";
		return out;
	}
};

class track {
public:
	std::vector<smf::event> events;

	track(void) : length(0) { }

	track(std::istreambuf_iterator<char> & itr) {
		get_uint32BE(itr);
		//std::cout << "track" << std::endl;
		events.clear();
		uint8_t laststatus = 0;
		do {
			event ev(itr, laststatus);
			laststatus = ev.status;
			events.push_back(ev);
		} while ( !events.back().isEoT() );
	}

	~track() {
		events.clear();
	}

	void clear(void) {
		//length = 0;
		events.clear();
	}

	friend std::ostream & operator<<(std::ostream & out, const track & chunk) {
		out << "Track chunk";
		//out << "(length = " << chunk.length << ") ";
		std::cout << std::endl;
		for(auto i = chunk.events.begin(); i != chunk.events.end(); ++i) {
			if ( i->isMeta() || i->isNote() ) {
				if ( i->delta > 0 ) {
					std::cout << std::endl;
				} else {
					std::cout << " ";
				}
				std::cout << *i ;
			}
		}
		return out;
	}
};
*/

struct note {
	uint32_t time;
	uint8_t channel;
	uint8_t number;
	uint32_t duration;

	note(uint32_t t, const smf::event & e, uint32_t d = 0) : time(t), channel(e.channel()), number(e.notenumber()), duration(d) { }

	friend std::ostream & operator<<(std::ostream & out, const note & n) {
		out << "[";
		out << std::dec << n.time << ", " << int(n.channel) << ".";
		out << smf::event::notename(n.number) << smf::event::octave(n.number) << ", " << n.duration;
		out << "]";
		return out;
	}
};

class score {
	uint16_t smfformat, ntracks, division;
	std::vector<std::vector<smf::event>> tracks;

public:
	score(std::istream & smffile) {
		std::istreambuf_iterator<char> itr(smffile);
		std::istreambuf_iterator<char> end_itr;

		if ( check_str("MThd", itr) ) {
			get_uint32BE(itr);
			// The header length is always 6.
			smfformat = get_uint16BE(itr);
			ntracks = get_uint16BE(itr);
			division = get_uint16BE(itr);
		} else {
			smfformat = 0;
			ntracks = 0;
			division = 0;
			tracks.clear();
			return;
		}
		while (itr != end_itr) {
			if ( check_str("MTrk", itr) ) {
				get_uint32BE(itr);
				//std::cout << "track" << std::endl;
				tracks.push_back(std::vector<event>());
				uint8_t laststatus = 0;
				do {
					event ev(itr, laststatus);
					laststatus = ev.status;
					tracks.back().push_back(ev);
				} while ( !tracks.back().back().isEoT() );

			}
		}
	}

	bool empty(void) const {
		if ( ntracks != 0 )
			return true;
		return false;
	}

	uint16_t format() const {
		return smfformat;
	}

	bool isSMPTE() const {
		return (division & (uint16_t(1)<<15)) != 0;
	}

	uint16_t resolution() const {
		if ( ! isSMPTE() ) {
			return division;
		} else {
			uint16_t smpte = -char(division>>8);
			uint16_t tpf = division & 0xff;
			return smpte * tpf;
		}
	}

	uint16_t noftracks() const {
		return ntracks;
	}

	std::vector<smf::event> & track(int i) {
		return tracks[i];
	}

	std::vector<smf::note> notes();

	friend std::ostream & operator<<(std::ostream & out, const score & midi) {
		out << "smf";
		out << "(header: format = " << std::dec << midi.format() << ", ntracks = " << midi.noftracks() << ", resolution = " << midi.resolution() << ") ";
		for(uint16_t i = 0; i < midi.noftracks() ; ++i) {
			out << std::endl << "track " << std::dec << i << ": ";
			for(auto e = midi.tracks[i].cbegin(); e != midi.tracks[i].end() ; ++e) {
				out << *e;
			}

		}
		out << std::endl;
		return out;
	}
};

}

#endif /* SMF_H_ */
