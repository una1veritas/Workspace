/*
 * smf.cpp
 *
 *  Created on: 2022/05/10
 *      Author: sin
 */

#include <iostream>
#include <iomanip>
#include <deque>

#include "smf.h"

uint32_t smf::get_uint32BE(std::istreambuf_iterator<char> & itr) {
	uint32_t res = 0;
	for(uint16_t i = 0; i < 4; ++i) {
		res <<= 8;
		res |= uint8_t(*itr);
		++itr;
	}
	return res;
}

uint32_t smf::get_uint16BE(std::istreambuf_iterator<char> & itr) {
	uint32_t res = *itr;
	++itr;
	res <<= 8;
	res |= *itr;
	++itr;
	return res;
}

uint32_t smf::get_uint32VLQ(std::istreambuf_iterator<char> & itr) {
	uint8_t b;
	uint32_t res = 0;
	for( ; ; ) {
		res <<= 7;
		b = *itr;
		++itr;
		res |= (0x7f & b);
		if ( (b & 0x80) == 0 )
			break;
	}
	return res;
}

smf::event::event(std::istreambuf_iterator<char> & itr, uint8_t laststatus) {
	delta = get_uint32VLQ(itr);
	status = laststatus;
	if (((*itr) & 0x80) != 0) {
		status = *itr;
		++itr;
	}
	uint8_t type;
	uint32_t len;
	type = status & 0xf0;
	if ( (smf::MIDI_NOTEOFF <= type && type <= smf::MIDI_CONTROLCHANGE) || (type == smf::MIDI_PITCHBEND) ) {
		data.push_back(*itr);
		++itr;
		data.push_back(*itr);
		++itr;
	} else if ( type == smf::MIDI_PROGRAMCHANGE || type == smf::MIDI_CHPRESSURE ) {
		data.push_back(*itr);
		++itr;
	} else if ( status == smf::SYSEX ) {
		len = get_uint32VLQ(itr);
		for(uint32_t i = 0; i < len; ++i) {
			data.push_back(*itr);
			++itr;
		}
	} else if ( status == smf::ESCSYSEX ) {
		len = get_uint32VLQ(itr);
		for(uint32_t i = 0; i < len; ++i) {
			data.push_back(*itr);
			++itr;
		}
	} else if ( status == smf::META ) {
		data.push_back(*itr); // function
		++itr;
		len = get_uint32VLQ(itr);
		for(uint32_t i = 0; i < len; ++i) {
			data.push_back(*itr);
			++itr;
		}
	} else {
		std::cerr << "error!" << std::dec << delta << std::hex << status << std::endl;
		// error.
	}
}

std::ostream & smf::event::printOn(std::ostream & out) const {
	uint8_t type = status & 0xf0;
	if ( (smf::MIDI_NOTEOFF <= type) && (type <= smf::MIDI_PITCHBEND) ) {
		out << "(";
		if ( delta > 0 )
			out << delta << ", ";
		switch(type) {
		case smf::MIDI_NOTEOFF:
			out << "NOTEOFF:" << channel() << ", "
			<< notename() << octave(); // << ", " << int(evt.data[1]);
			break;
		case smf::MIDI_NOTEON:
			out << "NOTE ON:" << channel() << ", "
			<< notename() << octave() << ", " << int(data[1]);
			break;
		case smf::MIDI_POLYKEYPRESSURE:
			out << "POLYKEY PRESS, " << channel() << ", "
			<< std::dec << int(data[0]) << ", " << int(data[1]);
			break;
		case smf::MIDI_CONTROLCHANGE:
			out << "CTL CHANGE, " << channel() << ", "
			<< std::dec << int(data[0]) << ", " << int(data[1]);
			break;
		case smf::MIDI_PROGRAMCHANGE:
			out << "PRG CHANGE, " << channel() << ", "
			<< std::dec << int(data[0]);
			break;
		case smf::MIDI_CHPRESSURE:
			out << "CHANNEL PRESS, " << channel() << ", "
			<< std::dec << int(data[0]);
			break;
		case smf::MIDI_PITCHBEND:
			out << "CHANNEL PRESS, " << channel() << ", "
			<< std::dec << (uint16_t(data[1])<<7 | data[0]);
			break;
		}
		out << ")";
	} else if ( status == smf::SYSEX ) {
		out << "(";
		if ( delta != 0 )
			out << delta << ", ";
		out<< "SYSEX " << std::hex << status << ' ';
		for(auto i = data.begin(); i != data.end(); ++i) {
			if ( isprint(*i) && !isspace(*i) ) {
				out << char(*i);
			} else {
				out << std::hex << std::setw(2) << int(*i);
			}
		}
		out << ")";
	} else if ( status == smf::ESCSYSEX ) {
		out << "(";
		if ( delta != 0 )
			out << delta << ", ";
		out<< "ESCSYSEX ";
		for(auto i = data.begin(); i != data.end(); ++i) {
			if ( isprint(*i) && !isspace(*i) ) {
				out << char(*i);
			} else {
				out << std::hex << std::setw(2) << int(*i);
			}
		}
		out << ")";
	} else if ( status == smf::META ) {
		out << "(";
		if ( delta != 0 )
			out << std::dec << delta << ", ";
		out<< "M ";
		uint32_t tempo;
		switch (data[0]) {
		case 0x01:
			out << "text: ";
			printData(out, 1);
			break;
		case 0x02:
			out << "(c): ";
			printData(out, 1);
			break;
		case 0x03:
			out << "seq.name: ";
			printData(out, 1);
			break;
		case 0x04:
			out << "instr: ";
			printData(out, 1);
			break;
		case 0x05:
			out << "lyrics: ";
			printData(out, 1);
			break;
		case 0x06:
			out << "marker: ";
			printData(out, 1);
			break;
		case 0x07:
			out << "cue: ";
			printData(out, 1);
			break;
		case 0x08:
			out << "prog.: ";
			printData(out, 1);
			break;
		case 0x09:
			out << "dev.: ";
			printData(out, 1);
			break;
		case 0x21:
			out << "out port " << std::dec << int(data[1]);
			break;
		case 0x2f:
			out << "EoT";
			break;
		case 0x51:
			tempo = uint8_t(data[1]);
			tempo <<= 8;
			tempo |= uint8_t(data[2]);
			tempo <<= 8;
			tempo |= uint8_t(data[3]);
			out << "tempo 4th = " << std::dec << (60000000L/tempo);
			break;
		case 0x58:
			out << "time sig.: " << std::dec << int(data[1]) << "/" << int(1<<data[2]);
			out << ", " << int(data[3]) << " mclk., " << int(data[4]) << " 32nd";
			break;
		default:
			for(auto i = data.begin(); i != data.end(); ++i) {
				if ( isprint(*i) && !isspace(*i) ) {
					out << char(*i);
				} else {
					out << std::hex << std::setw(2) << int(*i) << ' ';
				}
			}
		}
		out << ")";
	} else {
		std::cout << "smfevent::operator<< error!";
		std::cout << std::dec << delta << ", " << std::hex << int(status) << std::endl;
		// error.
	}
	return out;
}

std::vector<smf::note> & smf::score::playout(std::vector<smf::note> & notes) {
	struct trkinfo {
		std::vector<smf::event>::const_iterator cursor;
		uint32_t to_go;
	} trk[tracks.size()];
	struct soundgenerator {
		struct {
			bool noteon;
			uint64_t time;
			note * ptr;
		} key[128];

		soundgenerator() {
			for(int i = 0; i < 128; ++i) {
				key[i].noteon = false;
				key[i].time = 0;
			}
		}

		void on(uint64_t gt, const smf::event & ev, note * noteitr) {
			key[ev.notenumber()].noteon = true;
			key[ev.notenumber()].time = gt;
			key[ev.notenumber()].ptr = noteitr;
		}

		void off(uint64_t gt, const smf::event & ev) {
			if ( key[ev.notenumber()].noteon) {
				key[ev.notenumber()].noteon = false;
				key[ev.notenumber()].ptr->duration = gt - key[ev.notenumber()].time;
				key[ev.notenumber()].time = 0;
			}
		}
	} sg[16];

	for(int i = 0; i < noftracks(); ++i) {
		trk[i].cursor = tracks[i].cbegin();
	}
	uint64_t globaltime = 0;
	// zero global time events
	for(uint32_t i = 0; i < noftracks(); ++i) {
		trk[i].to_go = 0;
		while ( trk[i].cursor->deltaTime() == 0 && ! trk[i].cursor->isEoT() ) {
			// issue events
			std::cout << i << ": " << *trk[i].cursor << " ";
			if ( trk[i].cursor->isNoteOn() ) {
				notes.push_back(note(globaltime, *trk[i].cursor, 0));
				sg[trk[i].cursor->channel()].on(globaltime, *trk[i].cursor, &notes.back());
			} else if ( trk[i].cursor->isNoteOff() ) {
				sg[trk[i].cursor->channel()].off(globaltime, *trk[i].cursor);
			}
			++trk[i].cursor;
		}
		std::cout << std::endl;
		if ( trk[i].cursor->isEoT() )
			continue;
		trk[i].to_go = trk[i].cursor->deltaTime();

	}
	uint64_t min_to_go;
	while (true) {
		min_to_go = 0;
		for(uint32_t i = 0; i < noftracks(); ++i) {
			if ( trk[i].cursor->isEoT() )
				continue;
			if ( min_to_go == 0 or trk[i].to_go < min_to_go ) {
				min_to_go = trk[i].to_go;
			}
		}
		//std::cout << "min_to_go = " << min_to_go << std::endl;
		globaltime += min_to_go;
		std::cout << "global = " << globaltime << std::endl;
		if (min_to_go == 0)
			break;
		for(uint32_t i = 0; i < noftracks(); ++i) {
			if ( trk[i].cursor->isEoT() )
				continue;
			trk[i].to_go -= min_to_go;
			if ( trk[i].to_go == 0 ) {
				do {
					// events occur
					if ( trk[i].cursor->isNoteOn() ) {
						notes.push_back(note(globaltime, *trk[i].cursor, 0));
					} else if ( trk[i].cursor->isNoteOff() ) {
						for(auto ptr = notes.rbegin(); ptr != notes.rend(); ++ptr) {
							if ( ptr->channel == trk[i].cursor->channel()
									and ptr->number == trk[i].cursor->notenumber() ) {
								ptr->duration = globaltime - ptr->time;
								break;
							}
						}
					}
					++trk[i].cursor;
				} while ( trk[i].cursor->deltaTime() == 0 && ! trk[i].cursor->isEoT() );
				//std::cout << std::endl;
			}
			trk[i].to_go = trk[i].cursor->deltaTime();
		}
	}
	//std::cout << "finished." << std::endl;
	return notes;
}
