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
	uint8 last_status;
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

	SMFStream(std::fstream & fs) : smfstream(fs), last_status(0), stream_status(0), current_track(0) {
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
		if ( event.delta == 'M' && event.type == 'T' ) {
			read_byte(tbuf,2);
			event.type = tbuf[0]; // MTRK or MTHD
			if ( event.type == SMFEvent::MTHD) {
				read_byte(tbuf, 10);
				event.length = (uint32)tbuf[0]<<24 | (uint32)tbuf[1]<<16 | (uint32)tbuf[2]<<8 | tbuf[3];
				event.format = (uint16)tbuf[4] << 8 | tbuf[5];
				event.tracks = (uint16)tbuf[6] << 8 | tbuf[7];
				event.resolution = (uint16)tbuf[8] << 8 | tbuf[9];
			} else if ( event.type == SMFEvent::MTRK) {
				read_byte(tbuf, 4);
				event.length = (uint32)tbuf[0]<<24 | (uint32)tbuf[1]<<16 | (uint32)tbuf[2]<<8 | tbuf[3];
			}
			last_status = event.type;
	    } else if ( (event.type & 0xf0) == 0xf0 ) {
			// status byte
			//std::cout << "status: ";
			switch (event.type) {
			case SMFEvent::SYSEX: // status = 0xf0
				event.length = read_varlenint();
				for(int i = 0; i < event.length; ++i) {
					*tbuf = read_byte();
					if ( i < SMFEvent::DATA_MAX_LENGTH && *tbuf != 0xf7)
						event.data[i] = *tbuf;
				}
				break;
			case SMFEvent::ESCSYSEX: // status = 0xf7
				//std::cout << "escaped/system exclusive event, ";
				event.length = read_varlenint();
				for(int i = 0; i < event.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH)
						event.data[i] = read_byte();
				}
				break;
			case SMFEvent::META: // status = 0xff
				//std::cout << "meta event, ";
				event.meta = read_byte(); // meta event type
				event.length = read_varlenint(); // event size
				//std::cout << "length = " << event.meta.length << " ";
				if (event.meta != 0x2f) {
					for(int i = 0; i < event.length; ++i) {
						*tbuf = read_byte();
						if ( i < SMFEvent::DATA_MAX_LENGTH)
							event.data[i] = *tbuf;
					}
				}
				break;
			}
			last_status = event.type;
		} else if ( (event.type & 0xf0) >= 0x80 && (event.type & 0xf0) < 0xf0 ) {
			switch(event.type & 0xf0) {
			case 0x80:
			case 0x90: // note on
				event.number = read_byte();
				event.velocity = read_byte();
				event.duration = 0;
				break;
			case 0xa0:
				break;
			case 0xb0: // control change
				event.number = read_byte();
				event.velocity = read_byte();
				if ( (event.number & 0x78) ) {
					if (event.number == 0x7c)
						midistatus.omni = false;
					else if (event.number == 0x7d)
						midistatus.omni = true;
					else if (event.number == 0x7e) {
						midistatus.poly = false;
						if ( !midistatus.omni )
							read_byte();
					} else if (event.number == 0x7f) {
						midistatus.poly = true;
					}
				}
				break;
			case 0xc0: // prog. change
			case 0xd0: // ch. pressure
				event.number = read_byte();
				break;
			case 0xe0: // pitch bend
				read_byte(tbuf, 2);
				event.pitchbend = (tbuf[0] & 0x7f) | ((uint16)tbuf[1] & 0x7f) << 7;
				break;
			}
			last_status = event.type;
		} else if ( (last_status & 0xf0) >= 0x80 && (last_status & 0xf0) < 0xf0) {
			//std::cout << "running? " << std::hex << (unsigned int) last_status;
			*tbuf = event.type;
			event.type = last_status;
			switch(last_status & 0xf0) {
			case 0x80:
			case 0x90: // note on
				event.number = *tbuf;
				event.velocity = read_byte();
				event.duration = 0;
				break;
			case 0xa0:
				break;
			case 0xb0: // control change
				event.number = *tbuf;
				event.velocity = read_byte();
				if ( (event.number & 0x78) ) {
					if (event.number == 0x7c)
						midistatus.omni = false;
					else if (event.number == 0x7d)
						midistatus.omni = true;
					else if (event.number == 0x7e) {
						midistatus.poly = false;
						if ( !midistatus.omni )
							read_byte();
					} else if (event.number == 0x7f) {
						midistatus.poly = true;
					}
				}
				break;
			case 0xc0: // prog. change
			case 0xd0: // ch. pressure
				event.number = *tbuf;
				break;
			case 0xe0: // pitch bend
				read_byte(tbuf+1, 1);
				event.pitchbend = (tbuf[0] & 0x7f) | ((uint16)tbuf[1] & 0x7f) << 7;
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

	// 4d 54 68 64 00 00 00 06 00 00 00 01 00 78
	// 4d 54 72 6b 00 00 1e 71
	// 00 ff 03 0a 50 61 6e 7a 65 72 6c 69 65 64
	// 00 ff 02 24 43 6f 70 79 72 69 67 68 74 20 5f 20 32 30 30 33 20 62 79 20 4a 6f 68 61 6e 6e 20 53 63 68 72 65 69 62 65 72
	// 00 ff 58 04 04 02 18 08
	// 00 ff 59 02 fe 00
	// 00 ff 51 03 08 52 af
	// 00 c0 49
	// 00 b0 07 50
	// 00    0a 19
	// 00 c1 47 00 b1
	/*
	07 5f 00 0a 64 00 c2 3a 00 b2 07 7f 00 0a 6e 00
	c3 39 00 b3 07 64 00 0a 5a 00 c4 38 00 b4 07 5a
	00 0a 1e 00 c5 48 00 b5 07 4b 00 0a 14 00 c6 09
	00 b6 07 41 00 0a 32 00 c7 3c 00 b7 07 5a 00 0a
	28 00 b9 07 41 00 0a 46 82 68 b0 5d 0a 00 5b 0f
	00 90 41 64 00 b1 5d 14 00 5b 0f 00 91 3e 64 00
	b2 5b 1e 00 b3 5b 1e 00 b4 5d 05 00 5b 1e 00 94
	41 64 00 b5 5d 0a 00 5b 14 00 95 4d 64 00 b6 5b
	23 00 96 4d 64 6e 94 41 00 0a 95 4d 00 00 90 41
	*/
	while ( smf.smfstream ) {
		SMFEvent evt = smf.getNextEvent();
		std::cout << evt << std::endl;
	}
	/*
	uint8 buf[256];
	smf.read_byte(buf,256);
	for(int i = 0; i < 256; ++i) {
		if ( i && (i & 0x0f) == 0 )
			std::cout << std::endl;
		std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) buf[i] << " ";
	}
	*/

	std::cout << std::endl;
	return 0;
}
