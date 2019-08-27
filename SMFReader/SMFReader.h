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
		} note;
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
		if ( data != NULL && isMIDI() ) {
			delete [] data;
			std::cerr << "allocated data area to an SMFEvent deleted. " << std::endl;
		}
	}

	SMFEvent & operator=(const SMFEvent & evt) {
		std::cerr << "substitution operator for SMFEvent has called. " << std::endl;
		if ( data != NULL && isMIDI() )
			delete [] data;
		memcpy((void*)&evt, (void*)this, sizeof(SMFEvent));
		return *this;
	}

	bool isMIDI() const {
		return (type & 0xf0) != 0xf0;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		ost << "[" << '+' << evt.delta << " ";
		if ( ! evt.isMIDI() ) {
			if (evt.type == SYSEX) {
				ost << "(SYSEX) ";
			} else if (evt.type == ESCSYSEX) {
				ost << "(ESCSYSEX) ";
			} else if (evt.type == META) {
				ost << "(META " << std::setw(2) << std::hex << std::setfill('0') << (unsigned int) evt.meta.type << ") ";
				ost << evt.meta.length << " ";
				for(int i = 0; i < evt.meta.length; ++i) {
						ost << (unsigned int) evt.data[i]<< " ";
				}
			}
		} else {
			ost << "MIDI ";
		}
		ost << "] ";
		return ost;
	}

};


#endif /* SMFREADER_H_ */
