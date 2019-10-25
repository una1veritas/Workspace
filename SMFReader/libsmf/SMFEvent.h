/*
 * SMFReader.h
 *
 *  Created on: 2019/08/27
 *      Author: sin
 */

#ifndef SMFEVENT_H_
#define SMFEVENT_H_

#include <iostream>
#include <iomanip>
#include <fstream>

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef unsigned int  uint;

struct SMFEvent {
	uint32 delta;
	uint8  type;
	union {
		uint32 length;
		uint32 duration;
	};
	const static uint8 DATA_MAX_LENGTH = 8;
	union {
		struct {
			uint16 format, tracks, resolution;
		};
		struct {
			uint8 number, velocity;
			uint16 pitchbend;
		};
		struct {
			uint8 meta;
			uint8 data[DATA_MAX_LENGTH];
		};
	};

	enum EVENT_TYPE {
		SYSEX = 0xf0, 	// System Exclusive
		ESCSYSEX = 0xf7, 	// Escaped System Exclusive
		META = 0xff, 	// Meta
		MIDI = 0x80, 		// Data
		MTHD = 'h',
		MTRK = 'r',
	};

	// methods
	SMFEvent() : delta(0), type(0), duration(0), number(0), velocity(0) {}
	//~SMFEvent() {}
	/*
	SMFEvent & operator=(const SMFEvent & evt) {
		//std::cerr << "substitution operator for SMFEvent has called. " << std::endl << std::flush;
		if ( !isMIDI() && data != NULL )
			delete [] data;
		memcpy((void*)&evt, (void*)this, sizeof(SMFEvent));
		return *this;
	}
	 */

	bool isMIDI() const {
		return (type & 0xf0) >= 0x80 && (type & 0xf0) <= 0xe0;
	}

	bool isNote() const {
		uint8 t = type & 0xf0;
		return t == 0x80 || t == 0x90;
	}

	bool isNoteOff() const {
		uint8 t = type & 0xf0;
		return ( t == 0x80 || (t == 0x90 && velocity == 0) );
	}

	bool isNoteOn() const {
		return (type & 0xf0) == 0x90 && velocity != 0;
	}

	bool isSys() const {
		return (type == SYSEX) || (type == ESCSYSEX);
	}

	bool isMeta() const {
		return type == META;
	}

	bool isMT() const {
		return type == MTRK || type == MTHD;
	}

	bool isMTRK() const {
		return type == MTRK;
	}

	uint8 channel() const {
		return type & 0x0f;
	}

	friend std::ostream & operator<<(std::ostream & ost, const SMFEvent & evt) {
		ost << "[" << '+' << std::dec << evt.delta << " ";
		if ( evt.isSys() ) {
			if (evt.type == SYSEX) {
				ost << "(SYSEX) "<< evt.length << " ";
 				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) evt.data[i] << " ";
					} else {
						ost << '.';
					}
 				}
			} else if (evt.type == ESCSYSEX) {
				ost << "(ESCSYSEX) ";
			}
		} else if ( evt.isMeta() ) {
			ost << "(M ";
			if ( evt.meta == 0x01 ) {
				ost << "TEXT" << ") ";
				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << (char) evt.data[i];
					} else {
						ost << '.';
					}
				}
			} else if ( evt.meta == 0x02 ) {
				ost << "COPYRIGHT" << ") ";
				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << (char) evt.data[i];
					} else {
						ost << '.';
					}
				}
			} else if ( evt.meta == 0x03 ) {
				ost << "NAME" << ") ";
				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << (char) evt.data[i];
					} else {
						ost << '.';
					}
				}
			} else if ( evt.meta == 0x2f ) {
				ost << "TRACK END" << ") ";
			} else if ( evt.meta == 0x51 ) {
				ost << "TEMPO" << ") " << ((uint32)evt.data[0]<<16 | (uint32)evt.data[1]<<8 | evt.data[2]);
			} else if ( evt.meta == 0x54 ) {
				ost << "SMTPE OFFSET" << ") ";
				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << std::setw(2) << std::hex << (unsigned int) evt.data[i];
					} else {
						ost << '.';
					}
				}
			} else if ( evt.meta == 0x58 ) {
				ost << "TIME" << ") ";
				ost << (unsigned int) evt.data[0] << "/" << ((unsigned int) 1 <<evt.data[1]);
				ost << " clocks " << (unsigned int) evt.data[2] << " 32nds " << (unsigned int) evt.data[3];
			} else if ( evt.meta == 0x59 ) {
				ost << "KEY" << ") ";
				if ( 0 < (char) evt.data[0] ) {
					ost << '#' << (unsigned int) evt.data[0];
				} else if ( evt.data[0] == 0 ) {
					ost << evt.data[1];
				} else {
					ost << 'b' << -((char) evt.data[0]);
				}
				ost << " " << (evt.data[1] ? "min" : "maj");
			} else {
				ost << std::hex << (unsigned int) evt.meta << ") ";
				for(uint i = 0; i < evt.length; ++i) {
					if ( i < SMFEvent::DATA_MAX_LENGTH ) {
						ost << std::setw(2) << std::hex << (uint) evt.data[i];
					} else {
						ost << '.';
					}
				}
			}
		} else if ( evt.isMIDI() ) {
			switch ( evt.type & 0xf0 ) {
			case 0xb0:
				ost << "(ctrl ch) " << std::hex << (uint) evt.channel()
					<< " " << std::hex << (uint) evt.number
					<< " " << std::hex << (uint) evt.velocity;
				break;
			case 0xc0:
				ost << "(prog ch) " << std::hex << (uint) evt.channel()
					<< " " << std::hex << (uint) evt.number;
				break;
			case 0x80:
			case 0x90:
				if ( (evt.type & 0xf0) == 0x80 || evt.velocity == 0 ) {
					ost << "(off) ";
					ost << (unsigned int) evt.channel()
						<< " " << (unsigned int) evt.number;
				} else {
					ost << "(on)  ";
					ost << (unsigned int) evt.channel()
						<< " " << (unsigned int) evt.number << ", " << (unsigned int) evt.velocity;
				}
				break;
			}
		} else if ( evt.isMT() ) {
			if ( evt.type == MTRK ) {
				ost << "MTrk " << evt.length;
			} else if ( evt.type == MTHD ) {
				ost << "MThd " << evt.length << " format " << evt.format << " tracks " << evt.tracks << " resolution " << evt.resolution;
			}
		} else {
			ost << "UNKNOWN ";
			ost << std::hex << (uint) evt.delta << " " << (uint) evt.type << " " << (unsigned int) evt.length;
			for(uint i = 0; i < evt.length; ++i) {
				ost << std::hex << evt.data[i] << " ";
			}
		}
		ost << "] ";
		return ost;
	}

};


#endif /* SMFEVENT_H_ */
