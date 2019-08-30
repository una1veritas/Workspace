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
	struct {
		bool omni;
		bool poly;
	} midistatus;
	uint8 stream_status;
	uint8 current_track;

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

	SMFStream(std::fstream & fs) : smfstream(fs), stream_status(0), current_track(0) {
		unsigned char t[18];
		midistatus.omni = true;
		midistatus.poly = false;
	}

	~SMFStream() {
		smfstream.close();
		std::cerr << "an SMFStream closed." << std::endl;
	}

	SMFEvent getNextEvent() {
		SMFEvent event;
		uint8 tbuf[16];
		event.delta = read_varlenint();
		//std::cout << "delta = " << event.delta << ", ";
		event.type = read_byte();
		//std::cout << "type = " << std::hex << (unsigned int) event.type << ", ";
		if ( event.delta == 'M' && event.type == (uint8)'T' ) {
			read_byte(tbuf,2);
			event.type = tbuf[0];
			read_byte(tbuf, 10);
			event.length = (uint32)tbuf[0]<<24 | (uint32)tbuf[1]<<16 | (uint32)tbuf[2]<<8 | tbuf[3];
			if ( event.mt.mttype == 'h') {
				read_byte(tbuf,event.length);
				event.mt.format = ((uint16)tbuf[0]) << 8 | tbuf[2];
				event.mt.tracks = ((uint16)tbuf[0]) << 8 | tbuf[2];
				event.mt.resolution = ((uint16)tbuf[0]) << 8 | tbuf[2];
				*/
			}
	    } else if ( (event.type & 0xf0) == 0xf0 ) {
			// status byte
			//std::cout << "status: ";
			switch (event.type) {
			case SMFEvent::SYSEX: // status = 0xf0
				event.length = read_varlenint() - 1;
				event.data = new uint8[event.length];
				for(int i = 0; i < event.length; ++i) {
					event.data[i] = read_byte();
				}
				read_byte(); // end-marker x0f7
				break;
			case SMFEvent::ESCSYSEX: // status = 0xf7
				//std::cout << "escaped/system exclusive event, ";
				event.length = read_varlenint();
				event.data = new uint8[event.length];
				for(int i = 0; i < event.length; ++i) {
					event.data[i] = read_byte();
				}
				break;
			case SMFEvent::META: // status = 0xff
				//std::cout << "meta event, ";
				event.metatype = read_byte(); // event type
				event.length = read_varlenint(); // event size
				//std::cout << "length = " << event.meta.length << " ";
				if (event.metatype != 0x2f) {
					event.data = new uint8[event.length];
					for(int i = 0; i < event.length; ++i) {
						event.data[i] = read_byte();
					}
				}
				break;
			}
		} else if ( (event.type & 0xf0) >= 0x80) {
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
		ost << (unsigned int) score.stream_status << " ";
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

	// 00 f0 05 7e 7f 09 01 f7
	// 00 ff 01 17 72 61 6e 64 6f 6d 5f 73 65 65 64 20 31 33 30 36 38 34 31 32
	for(int i = 0; i < 64; ++i) {
		std::cout << smf.getNextEvent() << std::endl;
		if (!smf.smfstream)
			break;
	}
	/*
	uint8 buf[32];
	smf.read_byte(buf,32);
	for(int i = 0; i < 32; ++i) {
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	*/

	std::cout << std::endl;
	return 0;
}
