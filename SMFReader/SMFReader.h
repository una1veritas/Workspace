/*
 * SMFReader.h
 *
 *  Created on: 2019/08/27
 *      Author: sin
 */

#ifndef SMFREADER_H_
#define SMFREADER_H_

#include <iostream>
#include <iomanip>
#include <fstream>

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

struct SMFEvent {
	uint32 delta;
	uint8  type;
	union {
		uint8 metatype;
		struct {
			uint8 number, channel, velocity;
		} midi;
		struct {
			uint8 mttype;
			uint16 format, tracks, resolution;
		} mt;
	};
	uint32 length;
	uint8 * data;


	enum EVENT_TYPE {
		SYSEX = 0xf0, 	// System Exclusive
		ESCSYSEX = 0xf7, 	// Escaped System Exclusive
		META = 0xff, 	// Meta
		MIDI = 0x80, 		// Data
		MT = 'T',
	};

	// methods
	SMFEvent() : delta(0), type(0), length(0), data(NULL) {}
	~SMFEvent() {
		if ( !isMIDI() && data != NULL ) {
			delete [] data;
			std::cerr << "allocated data area to an SMFEvent deleted. " << std::endl << std::flush;
		}
	}

	SMFEvent & operator=(const SMFEvent & evt) {
		std::cerr << "substitution operator for SMFEvent has called. " << std::endl << std::flush;
		if ( !isMIDI() && data != NULL )
			delete [] data;
		memcpy((void*)&evt, (void*)this, sizeof(SMFEvent));
		return *this;
	}

	bool isMIDI() const {
		return (type & 0xf0) >= 0x80 && (type & 0xf0) <= 0xe0;
	}

	bool isSys() const {
		return (type == SYSEX) || (type == ESCSYSEX);
	}

	bool isMeta() const {
		return type == META;
	}

	bool isMT() const {
		return type == (uint8)'T';
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		ost << "[" << '+' << std::dec << evt.delta << " ";
		if ( evt.isSys() ) {
			if (evt.type == SYSEX) {
				ost << "(SYSEX) ";
				for(int i = 0; i < evt.length; ++i) {
					ost << std::setw(2) << std::hex << std::setfill('0') << (unsigned int) evt.data[i]<< " ";
				}
			} else if (evt.type == ESCSYSEX) {
				ost << "(ESCSYSEX) ";
			}
		} else if ( evt.isMeta() ) {
			ost << "(META ";
			if ( evt.metatype == 0x01 ) {
				ost << "TEXT" << ") ";
				for(int i = 0; i < evt.length; ++i) {
					ost << (char) evt.data[i];
				}
			} else if ( evt.metatype == 0x59 ) {
				ost << "KEY" << ") ";
				if ( 0 < (char) evt.data[0] ) {
					ost << '#' << (unsigned int) evt.data[0];
				} else if ( evt.data[0] == 0 ) {
					ost << evt.data[1];
				} else {
					ost << 'b' << (unsigned int) evt.data[0];
				}
				ost << " " << (evt.data[1] ? "min" : "maj");
			} else if ( evt.metatype == 0x54 ) {
				ost << "SMTPE OFFSET" << ") ";
				for(int i = 0; i < evt.length; ++i) {
					ost << (unsigned int) evt.data[i]<< " ";
				}
			} else if ( evt.metatype == 0x2f ) {
				ost << "TRACK END" << ") ";
			} else {
				ost << std::hex << (unsigned int) evt.metatype << ") ";
				for(int i = 0; i < evt.length; ++i) {
					ost << (unsigned int) evt.data[i]<< " ";
				}
			}
		} else if ( evt.isMIDI() ) {
			switch ( evt.type & 0xf0 ) {
			case 0xb0:
				ost << "(ctrl change) " << std::hex << (unsigned int) evt.midi.channel
					<< " " << std::hex << (unsigned int) evt.midi.number
					<< " " << std::hex << (unsigned int) evt.midi.velocity;
				break;
			case 0xc0:
				ost << "(prog change) " << std::hex << (unsigned int) evt.midi.channel
					<< " " << std::hex << (unsigned int) evt.midi.number;
				break;
			case 0x80:
			case 0x90:
				if ( (evt.type & 0xf0) == 0x80 || evt.midi.velocity == 0 ) {
					ost << "(noteoff) ";
					ost << (unsigned int) evt.midi.channel
						<< " " << (unsigned int) evt.midi.number;
				} else {
					ost << "(note on) ";
					ost << (unsigned int) evt.midi.channel
						<< " " << (unsigned int) evt.midi.number << ", " << (unsigned int) evt.midi.velocity;
				}
				break;
			}
		} else if ( evt.isMT() ) {
			if (evt.mt.mttype == 'r') {
				ost << "MTrk " << evt.length;
			} else if ( evt.mt.mttype == 'h') {
				ost << "MThd " << evt.length << " format " << evt.mt.format << " tracks " << evt.mt.tracks << " resolution " << evt.mt.resolution;
			}
		} else {
			ost << "UNKNOWN ";
			ost << std::hex << (unsigned int) evt.type;
		}
		ost << "] ";
		return ost;
	}

};


#endif /* SMFREADER_H_ */
