#include <iostream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <cstring>

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

struct SMFEvent {
	uint32 deltatime;
	union {
		uint8 evtype;
		uint8 note;
	};
	union {
		char
		uint16 duration;
	};

	enum EVENT_TYPE {
		STAT_SYSEX = 0xf0, 	// System Exclusive
		STAT_ESCEX = 0xf7, 	// Escaped System Exclusive
		STAT_META = 0xff, 	// Meta
		DATA = 0x0f, 		// Data
	};

	bool isData() const {
		return (evtype & 0x80) == 0;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		if ( isData() ) {
			if (evt.evtype == STAT_SYSEX) {
				ost << "Status (Sys Exclusive) ";
			} else if (evt.evtype == STAT_ESCEX) {
				ost << "Status (Escape Sys Exclusive) ";
			} else if (evt.evtype == STAT_META) {
				if (evt.)
			}
		}
		ost << std::hex << score.status << ", " << score.format << ", " << score.tracks << ", " << score.division;
		return ost;
	}

};

struct SMFStream {
	std::ifstream & smfstream;
	uint16 format, tracks, division;
	uint16 status;

	enum {
		ERROR_HEADER = 1<<0,
		ERROR_MTHD = 1<<1,
		ERROR_MTRK = 1<<2,
	};

	uint8 read_byte() {
		return smfstream.get();
	}

	uint8 * read_byte(uint8 * buf, uint32 num) {
		smfstream.read((char*)buf, num);
		return buf;
	}

	uint32 read_varlenint() {
		char tbyte;
		uint32 val = 0;
		while ( smfstream.get(tbyte) ) {
			val <<= 7;
			val |= 0x07f & tbyte;
			if ( (tbyte & 0x80) == 0 )
				break;
		}
		return val;
	}

	SMFStream(std::ifstream & ifs) : smfstream(ifs),
		format(0), tracks(0), division(0), status(0) {
		unsigned char t[18];
		status = 0;
		if ( !smfstream.read((char*) t, 18) ) {
			status |= ERROR_HEADER;
			return;
		}
		if ( memcmp(t, "MThd", 4) != 0 ) {
			status |= ERROR_MTHD;
			return;
		}
		if ( memcmp(t+14, "MTrk", 4) != 0 ) {
			status |= ERROR_MTRK;
			return;
		}
		format = t[8]<<8 | t[9];
		tracks = t[10]<<8 | t[11];
		division = t[12]<<8 | t[13];
	}

	~SMFStream() {
		std::cerr << "closed." << std::endl;
		smfstream.close();
	}

	SMFEvent getNextEvent() {
		SMFEvent event;
		uint32 deltat;
		uint8 tbyte;
		deltat = read_varlenint();
		std::cout << "delta time = " << deltat << ", ";
		tbyte = read_byte();
		if ( (tbyte & 0x80) != 0 ) {
			// status byte
			std::cout << "status: ";
			switch (tbyte) {
			case SMFEvent::STAT_SYSEX:
			case SMFEvent::STAT_ESCEX:
				std::cout << "escaped/system exclusive event, ";
				break;
			case SMFEvent::STAT_META:
				std::cout << "meta event, ";
				tbyte = read_byte(); // event type
				uint32 len = read_varlenint(); // event size
				std::cout << "len = " << len << ", ";
				if (tbyte == 0x2f) {
					// may be 'end of track'
					// len is always assumed to be zero.
					//score.add(new MusicalNote(-1, -1, -1));
					std::cout << "end of track/new track, ";
				} else {
					do {
						std::cout << std::setw(2) << std::hex << (unsigned int) read_byte()
								<< ", ";
					} while (--len);
					std::cout << std::endl;
				}
				break;
			}
		} else {
			std::cout << "???";
		}
		std::cout << std::endl;
		return event;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFStream & score) {
		ost << std::hex << score.status << ", " << score.format << ", " << score.tracks << ", " << score.division;
		return ost;
	}
};
/*
	private static int parseVarLenInt(InputStream stream) throws IOException {
		int oneByte, value = 0; // for storing unsigned byte
		while ( (oneByte = stream.read()) != -1 ) {
			value = value << 7;
			value += oneByte & 0x7f;
			if ( (oneByte & 0x80) == 0 )
				break;
		}
		if ( oneByte == -1 )
			return -1;
		return value;
	}
*/

int main(int argc, char **argv) {
	std::ifstream infile;

	infile.open(argv[1], (std::ios::in | std::ios::binary) );

	if ( !infile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}

	SMFStream smf(infile);
	std::cout << smf << std::endl;
	uint8 buf[4];
	smf.read_byte(buf, 4);

	SMFEvent e = smf.getNextEvent();
	return 0;
}
