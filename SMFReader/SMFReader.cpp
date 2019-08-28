#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>

#include "SMFReader.h"

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

struct SMFStream {
	std::fstream & smfstream;
	uint16 format, tracks, division;
	struct {
		bool omni;
		bool poly;
	} midistatus;
	uint8 status;

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
			//std::cerr << std::hex << (unsigned int) tbyte << ", ";
			val <<= 7;
			val |= (0x07f & tbyte);
			if ( !(tbyte & 0x80) )
				break;
		}
		//std::cerr << std::flush;
		return val;
	}

	SMFStream(std::fstream & fs) : smfstream(fs),
		format(0), tracks(0), division(0), status(0) {
		unsigned char t[18];
		midistatus.omni = true;
		midistatus.poly = false;
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
		smfstream.close();
		std::cerr << "an SMFStream closed." << std::endl;
	}

	SMFEvent getNextEvent() {
		SMFEvent event;
		event.delta = read_varlenint();
		//std::cout << "delta = " << event.delta << ", ";
		event.type = read_byte();
		//std::cout << "type = " << std::hex << (unsigned int) event.type << ", ";
		if ( (event.type & 0xf0) == 0xf0 ) {
			// status byte
			//std::cout << "status: ";
			switch (event.type) {
			case SMFEvent::SYSEX: // status = 0xf0
				event.sysex.length = read_varlenint() - 1;
				event.data = new uint8[event.sysex.length];
				for(int i = 0; i < event.sysex.length; ++i) {
					event.data[i] = read_byte();
				}
				read_byte(); // end-marker x0f7
				break;
			case SMFEvent::ESCSYSEX: // status = 0xf7
				//std::cout << "escaped/system exclusive event, ";
				event.sysex.length = read_varlenint();
				event.data = new uint8[event.sysex.length];
				for(int i = 0; i < event.sysex.length; ++i) {
					event.data[i] = read_byte();
				}
				break;
			case SMFEvent::META: // status = 0xff
				//std::cout << "meta event, ";
				event.meta.type = read_byte(); // event type
				event.meta.length = read_varlenint(); // event size
				//std::cout << "length = " << event.meta.length << " ";
				event.data = new uint8[event.sysex.length];
				for(int i = 0; i < event.meta.length; ++i) {
					event.data[i] = read_byte();
				}
				break;
			}
		} else {
			switch(event.type & 0xf0) {
			case 0x80:
			case 0x90: // note on
				event.midi.channel = event.type & 0x0f;
				event.midi.number = read_byte();
				event.midi.velocity = read_byte();
				break;
			case 0xa0:
				break;
			case 0xb0: // control change
				event.midi.channel = event.type & 0x0f;
				event.midi.number = read_byte();
				event.midi.velocity = read_byte();
				if ( (event.midi.number & 0x78) ) {
					if (event.midi.number == 0x7c)
						midistatus.omni = false;
					else if (event.midi.number == 0x7d)
						midistatus.omni = true;
					else if (event.midi.number == 0x7e) {
						midistatus.poly = false;
						if ( !midistatus.omni )
							read_byte();
					} else if (event.midi.number == 0x7f) {
						midistatus.poly = true;
					}
				}
				break;
			case 0xc0: // prog. change
				event.midi.channel = event.type & 0x0f;
				event.midi.number = read_byte();
				break;
			case 0xd0:
			case 0xe0:
				break;
			}
		}
		//std::cout << std::endl;
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
	std::fstream infile;

	infile.open(argv[1], (std::ios::in | std::ios::binary) );

	if ( !infile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}

	SMFStream smf(infile);
	std::cout << smf << std::endl;
	uint8 buf[16];
	smf.read_byte(buf, 4);
	for(int i = 0; i < 4; ++i) {
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	std::cout << std::endl;

	// 00 f0 05 7e 7f 09 01 f7
	// 00 ff 01 17 72 61 6e 64 6f 6d 5f 73 65 65 64 20 31 33 30 36 38 34 31 32
	for(int i = 0; i < 64; ++i) {
		std::cout << smf.getNextEvent() << std::endl;
		if (!smf.smfstream)
			break;
	}
	/*
	smf.read_byte(buf,32);
	for(int i = 0; i < 32; ++i) {
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	*/

	std::cout << std::endl;
	return 0;
}
