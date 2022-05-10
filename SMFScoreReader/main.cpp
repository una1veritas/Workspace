#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>

uint32_t get_uint32BE(std::istreambuf_iterator<char> & itr) {
	uint32_t res = 0;
	for(uint16_t i = 0; i < 4; ++i) {
		res <<= 8;
		res |= uint8_t(*itr);
		++itr;
	}
	return res;
}

uint32_t get_uint16BE(std::istreambuf_iterator<char> & itr) {
	uint32_t res = *itr;
	++itr;
	res <<= 8;
	res |= *itr;
	++itr;
	return res;
}

uint32_t get_uint32VLQ(std::istreambuf_iterator<char> & itr) {
	uint8_t b;
	uint32_t res = 0;
	for( ; ; ) {
		res <<= 7;
		b = *itr;
		++itr;
		res |= (0x7f & b);
		if ( (b & 0x80) == 0 )
			break;
	}
	return res;
}

struct smfevent {
	uint32_t delta;
	uint8_t status;
	std::string data;

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

	smfevent(void) {
		clear();
	}

	smfevent(std::istreambuf_iterator<char> & itr, uint8_t laststatus) {
		delta = get_uint32VLQ(itr);
		status = laststatus;
		if (((*itr) & 0x80) != 0) {
			status = *itr;
			++itr;
		}
		uint8_t type;
		uint32_t len;
		type = status & 0xf0;
		if ( (MIDI_NOTEOFF <= type && type <= MIDI_CONTROLCHANGE) || (type == MIDI_PITCHBEND) ) {
			data.push_back(*itr);
			++itr;
			data.push_back(*itr);
			++itr;
		} else if ( type == MIDI_PROGRAMCHANGE || type == MIDI_CHPRESSURE ) {
			data.push_back(*itr);
			++itr;
		} else if ( status == SYSEX ) {
			len = get_uint32VLQ(itr);
			for(uint32_t i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else if ( status == ESCSYSEX ) {
			len = get_uint32VLQ(itr);
			for(uint32_t i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else if ( status == META ) {
			data.push_back(*itr); // function
			++itr;
			len = get_uint32VLQ(itr);
			for(uint32_t i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else {
			std::cerr << "error!" << std::dec << delta << std::hex << status << std::endl;
			// error.
		}
	}

	void clear() {
		delta = 0;
		status = 0;
		data.clear();
	}

	~smfevent() {
		data.clear();
	}

	bool isMeta(void) const {
		return status == META;
	}

	bool isEOT(void) const {
		return isMeta() && data[0] == 0x2f;
	}

	bool isNote() const {
		if ( (status & 0xe0) == 0x80 ) {
			return true;
		}
		return false;
	}

	int channel(void) const {
		return 0x0f & status;
	}

	int octave() const {
		if ( isNote() )
			return data[0] / 12;
		return -2;
	}

	int notenumber() const {
		if ( !isNote() )
			return 128;
		return int(data[0]);
	}

	const char * notename() const {
		if ( !isNote() )
			return namesofnote[12];
		return namesofnote[data[0] % 12];
	}

	friend std::ostream & operator<<(std::ostream & out, const smfevent & evt) {
		uint8_t type = evt.status & 0xf0;
		if ( (MIDI_NOTEOFF <= type) && (type <= MIDI_PITCHBEND) ) {
			out << "(";
			if ( evt.delta > 0 )
				out << evt.delta << ", ";
			switch(type) {
			case MIDI_NOTEOFF:
				out << "NOTEOFF:" << evt.channel() << ", "
				<< evt.notename() << evt.octave(); // << ", " << int(evt.data[1]);
				break;
			case MIDI_NOTEON:
				out << "NOTE ON:" << evt.channel() << ", "
				<< evt.notename() << evt.octave() << ", " << int(evt.data[1]);
				break;
			case MIDI_POLYKEYPRESSURE:
				out << "POLYKEY PRESS, " << evt.channel() << ", "
				<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]);
				break;
			case MIDI_CONTROLCHANGE:
				out << "CTL CHANGE, " << evt.channel() << ", "
				<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]);
				break;
			case MIDI_PROGRAMCHANGE:
				out << "PRG CHANGE, " << evt.channel() << ", "
				<< std::dec << int(evt.data[0]);
				break;
			case MIDI_CHPRESSURE:
				out << "CHANNEL PRESS, " << evt.channel() << ", "
				<< std::dec << int(evt.data[0]);
				break;
			case MIDI_PITCHBEND:
				out << "CHANNEL PRESS, " << evt.channel() << ", "
				<< std::dec << (uint16_t(evt.data[1])<<7 | evt.data[0]);
				break;
			}
			out << ")";
		} else if ( evt.status == SYSEX ) {
			out << "(";
			if ( evt.delta != 0 )
				out << evt.delta << ", ";
			out<< "SYSEX " << std::hex << evt.status << ' ';
			for(auto i = evt.data.begin(); i != evt.data.end(); ++i) {
				if ( isprint(*i) && !isspace(*i) ) {
					out << char(*i);
				} else {
					out << std::hex << std::setw(2) << int(*i);
				}
			}
			out << ")";
		} else if ( evt.status == ESCSYSEX ) {
			out << "(";
			if ( evt.delta != 0 )
				out << evt.delta << ", ";
			out<< "ESCSYSEX ";
			for(auto i = evt.data.begin(); i != evt.data.end(); ++i) {
				if ( isprint(*i) && !isspace(*i) ) {
					out << char(*i);
				} else {
					out << std::hex << std::setw(2) << int(*i);
				}
			}
			out << ")";
		} else if ( evt.status == META ) {
			out << "(";
			if ( evt.delta != 0 )
				out << std::dec << evt.delta << ", ";
			out<< "M: ";
			uint32_t tempo;
			switch (evt.data[0]) {
			case 0x01:
				out << "text: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x02:
				out << "copyright: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x03:
				out << "seq. name: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x04:
				out << "instr: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x05:
				out << "lyrics: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x06:
				out << "marker: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x07:
				out << "cue: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x08:
				out << "program: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x09:
				out << "device: ";
				for(auto i = evt.data.begin() + 1; i != evt.data.end(); ++i) {
					out << *i;
				}
				break;
			case 0x21:
				out << "out port " << std::dec << int(evt.data[1]);
				break;
			case 0x2f:
				out << "eot";
				break;
			case 0x51:
				tempo = uint8_t(evt.data[1]);
				tempo <<= 8;
				tempo |= uint8_t(evt.data[2]);
				tempo <<= 8;
				tempo |= uint8_t(evt.data[3]);
				out << "tempo 4th = " << std::dec << (60000000L/tempo);
				break;
			case 0x58:
				out << "time signature " << std::dec << int(evt.data[1]) << "/" << int(1<<evt.data[2]);
				out << ", " << int(evt.data[3]) << " mclk., " << int(evt.data[4]) << " 32nd";
				break;
			default:
				for(auto i = evt.data.begin(); i != evt.data.end(); ++i) {
					if ( isprint(*i) && !isspace(*i) ) {
						out << char(*i);
					} else {
						out << std::hex << std::setw(2) << int(*i) << ' ';
					}
				}
			}
			out << ")";
		} else {
			std::cout << "smfevent::operator<< error!";
			std::cout << std::dec << evt.delta << ", " << std::hex << int(evt.status) << std::endl;
			// error.
		}
		return out;
	}
};


class smfheader {
	uint16_t length;

public:
	uint16_t format, ntracks, division;

	smfheader(void) : length(0), format(0), ntracks(0), division(0) { }

	smfheader(std::istreambuf_iterator<char> & itr) {
		length = get_uint32BE(itr);
		//std::cout << "header" << std::endl;
		format = get_uint16BE(itr);
		ntracks = get_uint16BE(itr);
		division = get_uint16BE(itr);
	}

	void clear(void) {
		length = 0;
		format = 0;
		ntracks = 0;
		division = 0;
	}

	friend std::ostream & operator<<(std::ostream & out, const smfheader & chunk) {
		out << "Header";
		out << "(format = " << chunk.format << ", ntracks = " << chunk.ntracks << ", division = " << chunk.division << ") ";
		return out;
	}
};

class smftrack {
	uint32_t length;

public:
	std::vector<smfevent> events;

	smftrack(void): length(0) { }

	smftrack(std::istreambuf_iterator<char> & itr) {
		length = get_uint32BE(itr);
		//std::cout << "track" << std::endl;
		events.clear();
		uint8_t laststatus = 0;
		do {
			events.push_back(smfevent(itr, laststatus));
			laststatus = events.back().status;
		} while ( !events.back().isEOT() );
	}

	~smftrack() {
		events.clear();
	}

	void clear(void) {
		length = 0;
		events.clear();
	}

	friend std::ostream & operator<<(std::ostream & out, const smftrack & chunk) {
		out << "Track chunk";
		out << "(length = " << chunk.length << ") ";
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

class smf {
	smfheader header;
	std::vector<smftrack> tracks;

	bool verify_signature(std::istreambuf_iterator<char> & itr, const std::string & sig) {
		bool res = true;
		for(auto i = sig.begin(); i != sig.end(); ++i, ++itr) {
			res &= (*i == *itr);
		}
		return res;
	}

public:
	smf(std::istreambuf_iterator<char> & itr) {
		std::istreambuf_iterator<char>  end_itr;

		if ( verify_signature(itr, "MThd") ) {
			header = smfheader(itr);
		}

		while (itr != end_itr) {
			if ( verify_signature(itr, "MTrk") ) {
				tracks.push_back(smftrack(itr));
			}
		}
	}

	uint16_t format() const {
		return header.format;
	}

	int numoftracks() const {
		return tracks.size();
	}

	const smftrack & track(int n) const {
		return tracks[n];
	}

	friend std::ostream & operator<<(std::ostream & out, const smf & midi) {
		out << "smf";
		out << midi.header << std::endl;
		for(auto i = midi.tracks.begin(); i != midi.tracks.end(); ++i) {
			out << *i << std::endl;
		}
		return out;
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

	smf midi(smfbuf);
	ifile.close();

	uint64_t gt = 0;
	std::vector<smfevent>::const_iterator cursor[midi.numoftracks()], ends[midi.numoftracks()];
	uint32_t remaining[midi.numoftracks()];
	for(int i = 0; i < midi.numoftracks(); ++i) {
		cursor[i] = midi.track(i).events.begin();
		remaining[i] = cursor[i]->delta;
		ends[i] = midi.track(i).events.end();
	}
	uint32_t mindelta;
	bool alleot;
	while (gt < 100000) {
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
		gt += mindelta;
		for(int i = 0; i < midi.numoftracks(); ++i) {
			std::cout << std::dec << remaining[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "gt = " << gt << std::endl;
	}

	std::cout << " done. " << std::endl;

	return 0;
}
