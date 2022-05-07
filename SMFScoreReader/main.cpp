#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>


typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

uint32 get_uint32BE(std::vector<char>::iterator & itr) {
	uint32 res = 0;
	for(uint16 i = 0; i < 4; ++i) {
		res <<= 8;
		res |= uint8(*itr);
		++itr;
	}
	return res;
}

uint32 get_uint16BE(std::vector<char>::iterator & itr) {
	uint32 res = *itr;
	++itr;
	res <<= 8;
	res |= *itr;
	++itr;
	return res;
}

uint32 get_uint32VL(std::vector<char>::iterator & itr) {
	uint8 b;
	uint32 res = 0;
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

struct SMFEvent {
	uint32 delta;
	uint8 status;
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

	SMFEvent(void) {
		clear();
	}

	SMFEvent(std::vector<char>::iterator & itr, uint8 laststatus) {
		delta = get_uint32VL(itr);
		status = laststatus;
		if (((*itr) & 0x80) != 0) {
			status = *itr;
			++itr;
		}
		uint8 type;
		uint32 len;
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
			len = get_uint32VL(itr);
			for(uint32 i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else if ( status == ESCSYSEX ) {
			len = get_uint32VL(itr);
			for(uint32 i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else if ( status == META ) {
			data.push_back(*itr); // function
			++itr;
			len = get_uint32VL(itr);
			for(uint32 i = 0; i < len; ++i) {
				data.push_back(*itr);
				++itr;
			}
		} else {
			std::cerr << "error!" << std::endl;
			// error.
		}
	}

	void clear() {
		delta = 0;
		status = 0;
		data.clear();
	}

	~SMFEvent() {
		data.clear();
	}

	int channel(void) const {
		return 0x0f & status;
	}

	friend std::ostream & operator<<(std::ostream & out, const SMFEvent & evt) {
		uint8 type = evt.status & 0xf0;
		if ( MIDI_NOTEOFF == type ) {
			out << "(" << evt.delta << ", NOTE OFF, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]) << ") ";
		} else if ( type == MIDI_NOTEON ) {
			out << "(" << evt.delta << ", NOTE ON, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]) << ") ";
		} else if ( type == MIDI_POLYKEYPRESSURE ) {
			out << "(" << evt.delta << ", POLYKEY PRESS, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]) << ") ";
		} else if ( type == MIDI_CONTROLCHANGE ) {
			out << "(" << evt.delta << ", CONTL CHANGE, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ", " << int(evt.data[1]) << ") ";
		} else if ( type == MIDI_PROGRAMCHANGE ) {
			out << "(" << evt.delta << ", PROG CHANGE, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ") ";
		} else if ( type == MIDI_CHPRESSURE ) {
			out << "(" << evt.delta << ", CHANNEL PRESS, " << evt.channel() << ", "
					<< std::dec << int(evt.data[0]) << ") ";
		} else if ( type == MIDI_PITCHBEND ) {
			out << "(" << evt.delta << ", CHANNEL PRESS, " << evt.channel() << ", "
					<< std::dec << (uint16(evt.data[1])<<7 | evt.data[0]) << ") ";
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
				out << evt.delta << ", ";
			out<< "M: ";
			uint32 tempo;
			switch (evt.data[0]) {
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
			case 0x2f:
				out << "end of track";
				break;
			case 0x51:
				tempo = uint8(evt.data[1]);
				tempo <<= 8;
				tempo |= uint8(evt.data[2]);
				tempo <<= 8;
				tempo |= uint8(evt.data[3]);
				out << "tempo 4th = " << std::dec << (60000000L/tempo);
				break;
			case 0x58:
				out << "time signature " << int(evt.data[1]) << "/" << int(1<<evt.data[2]);
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
			std::cerr << "error!" << std::endl;
			// error.
		}
		return out;
	}
};

struct SMFChunk {
	uint8  type;
	uint32 length;
	uint16 format, ntracks, division;
	std::vector<SMFEvent> events;

	enum CHUNK_TYPE {
		CHUNK_NONE   = 0,
		CHUNK_MThd = 1,
		CHUNK_MTrk  = 2,
	};

	enum EVENT_TYPE {
		SYSEX = 0xf0, 	// System Exclusive
		ESCSYSEX = 0xf7, 	// Escaped System Exclusive
		META = 0xff, 	// Meta
		MIDI = 0x80, 		// Data
	};

	SMFChunk(std::vector<char>::iterator & itr) {
		char t[4];
		for(int i = 0; i < 4; ++i) {
			t[i] = *itr;
			++itr;
		}
		if ( strncmp(t, "MThd", 4) == 0 ) {
			type = CHUNK_MThd;
		} else if ( strncmp(t, "MTrk", 4) == 0 ) {
			type = CHUNK_MTrk;
		} else {
			type = CHUNK_NONE;
		}
		length = get_uint32BE(itr);
		if ( isHeader() ) {
			//std::cout << "header" << std::endl;
			format = get_uint16BE(itr);
			ntracks = get_uint16BE(itr);
			division = get_uint16BE(itr);
		} else if ( isTrack() ) {
			//std::cout << "track" << std::endl;
			events.clear();
			uint8 laststatus = 0;
			auto itr_end = itr + length;
			while ( itr != itr_end ) {
				SMFEvent ev(itr, laststatus);
				laststatus = ev.status;
				events.push_back(ev);
			}
		}
	}

	~SMFChunk() {
		if ( isTrack() ) {
			events.clear();
		}
	}

	void clear(void) {
		format = 0;
		ntracks = 0;
		division = 0;
		events.clear();
	}

	bool isHeader(void) const {
		return type == CHUNK_MThd;
	}

	bool isTrack(void) const {
		return type == CHUNK_MTrk;
	}

	friend std::ostream & operator<<(std::ostream & out, const SMFChunk & chunk) {
		if ( chunk.isHeader() ) {
			out << "Header chunk ";
			out << "(format = " << chunk.format << ", ntracks = " << chunk.ntracks << ", division = " << chunk.division << ") ";
		} else if ( chunk.isTrack() ) {
			out << "Track chunk ";
			out << "(length = " << chunk.length << ") ";
			for(auto i = chunk.events.begin(); i != chunk.events.end(); ++i) {
				std::cout << *i << std::endl;
			}
		} else {
			out << "Unknown chunk ";
		}
		return out;
	}
};

/*
bool get_chunk(std::vector<char>::iterator & itr, SMFChunk & chunk) {
	chunk.clear();
	for(int i = 0; i < 4; ++i)
		chunk.ID[i] = *itr;
	chunk.length = get_uint32BE(itr);
	if ( chunk.isHeader() ) {
		chunk.format = get_uint16BE(itr);
		chunk.ntracks = get_uint16BE(itr);
		chunk.division = get_uint16BE(itr);
		return true;
	} else if ( chunk.isTrack() ) {
		std::vector<char>::iterator itr_end = itr + chunk.length;
		chunk.events.clear();
		// parse events
		uint32 b;
		uint8 status = 0;
		while ( itr != itr_end ) {
			b = get_uint32VL(itr);
			chunk.events.push_back(b);
			if (((*itr) & 0x80) != 0) {
				// not the running state
				status = *itr;
				++itr;
			}
			switch(status & 0xf0) {
			case 0x80:
			case 0x90: // note on
			case 0xa0:
			case 0xb0: // control change
			case 0xe0: // pitch bend
				b = *itr;
				++itr;
				b <<= 8;
				b |= *itr;
				++itr;
				chunk.events.push_back(b);
				break;
			case 0xc0: // prog. change
			case 0xd0: // ch. pressure
				b = *itr;
				++itr;
				chunk.events.push_back(b);
				break;
			case 0xf0: // Sys Ex | Meta
				switch(status) {
				case 0xf0: // sys ex
					b = get_uint32VL(itr);
					chunk.events.push_back(*itr);
					++itr;
					for(uint32 i = 0; i < b; ++i) {
						chunk.events.push_back(*itr);
						++itr;
					}
					break;
				case 0xff: // meta
					b = *itr;
					++itr;
					switch(b) {
					case 0x01:
					case 0x02:
					case 0x03:
					case 0x04:
					case 0x05:
					case 0x06:
					case 0x07:
					case 0x7f:
						chunk.events.push_back(get_uint32VL(itr));
						break;
					case 0x2f:
						// 0
						break;
					case 0x51:
						for(uint32 i = 0; i < 3; ++i) {
							chunk.events.push_back(*itr);
							++itr;
						}
						break;
					case 0x54:
						for(uint32 i = 0; i < 5; ++i) {
							chunk.events.push_back(*itr);
							++itr;
						}
						break;
					case 0x58:
						for(uint32 i = 0; i < 4; ++i) {
							chunk.events.push_back(*itr);
							++itr;
						}
						break;
					case 0x59:
						for(uint32 i = 0; i < 3; ++i) {
							chunk.events.push_back(*itr);
							++itr;
						}
						break;
					default:
						// error
						break;
					}
					break;
				}
				break;
			}
		}
		std::cout << "a track finished." << std::endl;
		return true;
	} else {
		return false;
	}
}
*/

int main(int argc, char **argv) {
	std::ifstream ifile;

	std::cout << "file: " << argv[1] << std::endl;
	ifile.open(argv[1], (std::ios::in | std::ios::binary) );
	if ( !ifile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}

	std::istreambuf_iterator<char> smfbuf(ifile);
	std::istreambuf_iterator<char> end_smfbuf;
	std::vector<char> smf(smfbuf, end_smfbuf);
	ifile.close();

	std::cout << smf.size() << " bytes." << std::endl;
	std::vector<SMFChunk> midi;
	for(auto itr = smf.begin(); itr != smf.end(); ) {
		midi.push_back(SMFChunk(itr));
		std::cout << midi.back() << std::endl;
	}

	std::cout << "done. " << std::endl;
	return 0;
}
