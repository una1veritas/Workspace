/*
 * reader.cpp
 *
 *  Created on: 2017/06/09
 *      Author: sin
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <cstdint>

using namespace std;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

struct FileInfo {
	uint8 type;
	uint8 save_name[32];
	uint32 byte_size, load_address, call_address;
	uint16 checksum;

	enum FILEINFOTYPE {
		PC_E500_CSAVE_M = 0x01,
		PC_E500_SAVE_ASC = 0x02,
		PC_E500_CSAVE_BAS = 0x04,

		PC_E500_RAMDISK = 0xff,

		FILEINFOTYPE_UNKNOWN = 0,
	};

	enum SYSTEMTYPE {
		PC_E500_CASSETTE,
		PC_E500_RAMDISK,
	};

	FileInfo(const std::vector<uint8> & buf, const uint8 stype = PC_E500_CASSETTE);

	ostream & printOn(ostream & outs) const;
};

FileInfo::FileInfo(const std::vector<uint8> & buff, const uint8 stype) {
	type = FILEINFOTYPE_UNKNOWN;
	save_name[0] = 0;
	byte_size = buff.size();
	load_address = 0;
	call_address = 0;
	checksum = 0;

	if ( stype == PC_E500_CASSETTE ) {
		type = buff[0];
		switch(type) {
		case PC_E500_CSAVE_M:
		case PC_E500_CSAVE_BAS:
			for(int i = 0; i < 0x11; ++i) {
				save_name[i] = buff[1+i];
			}
			save_name[0x10] = 0;
			byte_size = ((uint32)buff[0x1d])<<16 | ((uint32)buff[0x13])<<8 | buff[0x12];
			load_address = ((uint32)buff[0x1e])<<16 | ((uint32)buff[0x15]) | buff[0x14];
			call_address = ((uint32)buff[0x1f])<<16 | ((uint32)buff[0x17]) | buff[0x16];
			checksum = ((uint16)buff[0x30]<<8) | buff[0x31];
			break;
		case PC_E500_SAVE_ASC:
		default:
			for(int i = 0; i < 0x11; ++i) {
				save_name[i] = buff[1+i];
			}
			save_name[0x10] = 0;
			break;
		}
	}
}

ostream & FileInfo::printOn(ostream & outs) const {
	outs << "type = "<< (int) type << ", ";
	outs << "save name = \""<< save_name << "\", ";
	outs << "size = " << std::dec << byte_size << "(0x" << std::hex << byte_size << "), ";
	outs << "load address = " << "0x" << std::setw(8) << std::setfill('0') << load_address << ", ";
	outs << "call address = " << "0x" << std::setw(8) << std::setfill('0') << call_address << ", ";
	outs << "checksum = " << "0x" << std::setw(4) << std::setfill('0') << checksum << ", ";
	return outs;
}

void dump(std::vector<uint8> & buff);

int main(const int argc, const char * argv[]) {
	cout << "Hello." << endl;

	std::ifstream infile(argv[1], std::ios::binary);

	if ( ! infile )
		return -1;

	std::vector<uint8> buff( (std::istreambuf_iterator<char>(infile)),
			(std::istreambuf_iterator<char>()) );

	infile.close();

	dump(buff);
	FileInfo info(buff);
	info.printOn(std::cout);
	std::cout << std::endl;

	return 0;
}

void dump(std::vector<uint8> & buff) {
	long addr;
	for( addr = 0; addr < buff.size(); addr++) {
		if ( (addr & 0x0f) == 0 ) {
			std::cout << std::endl;
			std::cout << std::setw(4) << std::setfill('0') << std::hex << addr << ": ";
		}
		std::cout << std::setw(2) << std::hex << ((unsigned int) buff[addr] & 0xff) << " ";
	}
	std::cout << std::endl;
}
