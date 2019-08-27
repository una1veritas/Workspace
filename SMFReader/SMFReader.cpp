#include <iostream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <cstring>

#include "SMFReader.h"

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

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
		event.delta = read_varlenint();
		std::cout << "delta time = " << event.delta << ", ";
		event.type = read_byte();
		if ( (event.type & 0x80) != 0 ) {
			// status byte
			std::cout << "status: ";
			switch (event.type) {
			case SMFEvent::STAT_SYSEX: // f0
				event.sysex.length = read_varlenint();
				event.dataptr = new uint8[event.sysex.length];
				for(int i = 0; i < event.sysex.length; ++i) {
					event.dataptr[i] = read_byte();
				}
				event.dataptr[event.sysex.length-1] = 0;
				break;
			case SMFEvent::STAT_ESCEX:
				std::cout << "escaped/system exclusive event, ";
				event.sysex.length = read_varlenint();
				event.dataptr = new uint8[event.sysex.length];
				for(int i = 0; i < event.sysex.length; ++i) {
					event.dataptr[i] = read_byte();
				}
				break;
			case SMFEvent::STAT_META:
				std::cout << "meta event, ";
				event.meta.type = read_byte(); // event type
				event.meta.length = read_varlenint(); // event size
				std::cout << "len = " << event.meta.length << ", ";
				if (event.meta.type == 0x2f) {
					// may be 'end of track'
					// len is always assumed to be zero.
					//score.add(new MusicalNote(-1, -1, -1));
					std::cout << "end of track/new track, ";
				} else {
					for(int i = 0; i < event.meta.length; ++i) {
						std::cout << std::setw(2) << std::hex << (unsigned int) read_byte()
								<< ", ";
					};
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
	uint8 buf[16];
	smf.read_byte(buf, 4);

	smf.read_byte(buf,16);
	for(int i = 0; i < 16; ++i) {
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	std::cout << std::endl;
	return 0;
}
