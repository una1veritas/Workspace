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

	SMFEvent() : delta(0), type(0), data(NULL) {}
	~SMFEvent() {
		if ( data != NULL && !isNote() ) {
			delete [] data;
			std::cerr << "allocated data area to an SMFEvent deleted. " << std::endl;
		}
	}

	enum EVENT_TYPE {
		STAT_SYSEX = 0xf0, 	// System Exclusive
		STAT_ESCEX = 0xf7, 	// Escaped System Exclusive
		STAT_META = 0xff, 	// Meta
		DATA = 0x0f, 		// Data
	};

	bool isNote() const {
		return (type & 0x80) == 0;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		ost << "[" << '+' << evt.delta << " ";
		if ( not evt.isNote() ) {
			if (evt.type == STAT_SYSEX) {
				ost << "status (sysex) ";
			} else if (evt.type == STAT_ESCEX) {
				ost << "status (esysx) ";
			} else if (evt.type == STAT_META) {
				ost << "status (meta " << std::setw(2) << std::hex << std::setfill('0') << (unsigned int) evt.meta.type << ") ";
				for(int i = 0; i < evt.meta.length; ++i) {
						ost << (unsigned int) evt.data[i]<< " ";
				}
			}
		} else {
			ost << "data ";
		}
		ost << "] ";
		return ost;
	}

};


#endif /* SMFREADER_H_ */
