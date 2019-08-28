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
		struct {
			uint32 length;
		} sysex;
		struct {
			uint32 length;
			uint8  type;
		} meta;
		struct {
			uint8 number, channel, velocity;
			uint32 duration;
		} midi;
	};
	uint8 * data;


	enum EVENT_TYPE {
		SYSEX = 0xf0, 	// System Exclusive
		ESCSYSEX = 0xf7, 	// Escaped System Exclusive
		META = 0xff, 	// Meta
		MIDI = 0x0f, 		// Data
	};

	// methods
	SMFEvent() : delta(0), type(0), data(NULL) {}
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
		return (type & 0xf0) != 0xf0;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		ost << "[" << '+' << std::dec << evt.delta << " ";
		if ( ! evt.isMIDI() ) {
			if (evt.type == SYSEX) {
				ost << "(SYSEX) ";
				for(int i = 0; i < evt.meta.length; ++i) {
					ost << std::setw(2) << std::hex << std::setfill('0') << (unsigned int) evt.data[i]<< " ";
				}
			} else if (evt.type == ESCSYSEX) {
				ost << "(ESCSYSEX) ";
			} else if (evt.type == META) {
				ost << "(META ";
				if ( evt.meta.type == 0x01 ) {
					ost << "TEXT" << ") ";
					for(int i = 0; i < evt.meta.length; ++i) {
						ost << (char) evt.data[i];
					}
				} else if ( evt.meta.type == 0x59 ) {
					ost << "KEY" << ") ";
					if ( 0 < (char) evt.data[0] ) {
						ost << '#' << (unsigned int) evt.data[0];
					} else if ( evt.data[0] == 0 ) {
						ost << evt.data[1];
					} else {
						ost << 'b' << (unsigned int) evt.data[0];
					}
					ost << " " << (evt.data[1] ? "min" : "maj");
				} else {
					ost << std::hex << (unsigned int) evt.type << ") ";
					for(int i = 0; i < evt.meta.length; ++i) {
						ost << (unsigned int) evt.data[i]<< " ";
					}
				}
			}
		} else {
			ost << "(MIDI) ";
			switch ( evt.type & 0xf0 ) {
			case 0xb0:
				ost << "ctrl change " << std::hex << (unsigned int) evt.midi.channel
					<< " " << std::hex << (unsigned int) evt.midi.number
					<< " " << std::hex << (unsigned int) evt.midi.velocity;
				break;
			case 0xc0:
				ost << "prog change " << std::hex << (unsigned int) evt.midi.channel
					<< " " << std::hex << (unsigned int) evt.midi.number;
				break;
			case 0x80:
			case 0x90:
				if ( (evt.type & 0xf0) == 0x80 || evt.midi.velocity == 0 ) {
					ost << "note-off ";
					ost << (unsigned int) evt.midi.channel
						<< " " << (unsigned int) evt.midi.number;
				} else {
					ost << "note-on ";
					ost << (unsigned int) evt.midi.channel
						<< " " << (unsigned int) evt.midi.number << " " << (unsigned int) evt.midi.velocity;
				}
				break;
			default:
				ost << std::hex << (unsigned int) evt.type;
				break;
			}
		}
		ost << "] ";
		return ost;
	}

};


#endif /* SMFREADER_H_ */
