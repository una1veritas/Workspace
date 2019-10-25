/*
 * SMFStream.h
 *
 *  Created on: 2019/08/27
 *      Author: sin
 */

#ifndef SMFSTREAM_H_
#define SMFSTREAM_H_

#include <iostream>
#include <iomanip>
#include <fstream>

#include "SMFEvent.h"

/*
typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
*/

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

	uint8 read_byte(void) {
		return smfstream.get();
	}

	uint8 * read_byte(uint8 * buf, const uint32 num) {
		smfstream.read((char*)buf, num);
		return buf;
	}

	uint8 * read_byte(uint8 * buf, const uint32 num, const uint32 limit_num) {
		uint8 t;
		for(uint i = 0; i < num; ++i) {
			t = read_byte();
			if ( i < limit_num)
				buf[i] = t;
		}
		return buf;
	}

	uint32 uint32_from_bytes(const uint8 byte0, const uint8 byte1, const uint8 byte2, const uint8 byte3) {
		return ((uint32) byte0) << 24 | ((uint32) byte1) << 16 | ((uint32) byte2) << 8 | ((uint32) byte3);
	}

	uint16 uint16_from_bytes(const uint8 byte0, const uint8 byte1) {
		return ((uint32) byte0) << 8 | ((uint32) byte1);
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

	SMFEvent & complete_read_MIDIEvent(SMFEvent & event, const uint8 nextByte) {
		switch(event.type & 0xf0) {
		case 0x80:
		case 0x90: // note on
			event.number = nextByte;
			event.velocity = read_byte();
			event.duration = 0;
			break;
		case 0xa0:
			break;
		case 0xb0: // control change
			event.number = nextByte;
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
			event.number = nextByte;
			break;
		case 0xe0: // pitch bend
			read_byte();
			event.pitchbend = ((uint16) read_byte() & 0x7f)<<7;
			event.pitchbend |= nextByte & 0x7f;
			break;
		}
		return event;
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
				event.length = uint32_from_bytes(tbuf[0], tbuf[1], tbuf[2], tbuf[3]);
				event.format = uint16_from_bytes(tbuf[4], tbuf[5]);
				event.tracks = uint16_from_bytes(tbuf[6], tbuf[7]);
				event.resolution = uint16_from_bytes(tbuf[8], tbuf[9]);
			} else if ( event.type == SMFEvent::MTRK) {
				read_byte(tbuf, 4);
				event.length = uint32_from_bytes(tbuf[0], tbuf[1], tbuf[2], tbuf[3]);
			}
			last_status = event.type;
	    } else if ( (event.type & 0xf0) == 0xf0 ) {
			// status byte
			//std::cout << "status: ";
			switch (event.type) {
			case SMFEvent::SYSEX: // status = 0xf0
				event.length = read_varlenint();
				read_byte(event.data, event.length, SMFEvent::DATA_MAX_LENGTH);
				break;
			case SMFEvent::ESCSYSEX: // status = 0xf7
				//std::cout << "escaped/system exclusive event, ";
				event.length = read_varlenint();
				read_byte(event.data, event.length, SMFEvent::DATA_MAX_LENGTH);
				break;
			case SMFEvent::META: // status = 0xff
				//std::cout << "meta event, ";
				event.meta = read_byte(); // meta event type
				event.length = read_varlenint(); // event size
				//std::cout << "length = " << event.meta.length << " ";
				if (event.meta != 0x2f) {
					read_byte(event.data, event.length, SMFEvent::DATA_MAX_LENGTH);
				}
				break;
			}
			last_status = event.type;
		} else if ( (event.type & 0xf0) >= 0x80 && (event.type & 0xf0) < 0xf0 ) {
			*tbuf = read_byte();
			complete_read_MIDIEvent(event, *tbuf);
			last_status = event.type;
		} else if ( (last_status & 0xf0) >= 0x80 && (last_status & 0xf0) < 0xf0) {
			// assumes in the running status mode, since no event type byte matches
			*tbuf = event.type; // extract as the byte next to event type
			event.type = last_status;
			complete_read_MIDIEvent(event, *tbuf);
		}
		//std::cout << std::endl;
		return event;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFStream & score) {
		ost << (unsigned int) score.stream_status << " ";
		return ost;
	}
};

#endif /* SMFREADER_H_ */

