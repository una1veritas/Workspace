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

		PC_E500_DISKFILE = 0xff,

		FILEINFOTYPE_UNKNOWN = 0,
	};

	enum SYSTEMTYPE {
		PC_E500_CMT,
		PC_E500_DISK = PC_E500_CMT,
	};

	FileInfo(const std::vector<uint8> & buf, const std::string & name = "", const uint8 stype = PC_E500_CMT);

	ostream & printOn(ostream & outs) const;
};

FileInfo::FileInfo(const std::vector<uint8> & buff, const std::string & name, const uint8 stype) {
	type = FILEINFOTYPE_UNKNOWN;
	save_name[0] = 0;
	byte_size = buff.size();
	load_address = 0;
	call_address = 0;
	checksum = 0;

	if ( stype == PC_E500_CMT ) {
		type = buff[0];
		switch(type) {
		case PC_E500_CSAVE_M:
		case PC_E500_CSAVE_BAS:
			for(int i = 0; i < 0x11; ++i) {
				save_name[i] = buff[1+i];
			}
			save_name[0x10] = 0;
			byte_size = ((uint32)buff[0x1d])<<16 | ((uint32)buff[0x13])<<8 | buff[0x12];
			load_address = ((uint32)buff[0x1e])<<16 | ((uint32)buff[0x15])<<8 | buff[0x14];
			call_address = ((uint32)buff[0x1f])<<16 | ((uint32)buff[0x17])<<8 | buff[0x16];
			checksum = ((uint16)buff[0x30]<<8) | buff[0x31];
			break;
		case PC_E500_DISKFILE:
			if ( name.length() > 0 ) {
				for(int i = 0; i < 16; i++) {
					save_name[i] = toupper(name[i]);
				}
				save_name[16] = 0;
			}
			byte_size = buff[0x05] | ((uint16)buff[0x06])<<8 | ((uint32)buff[0x07])<<16;
			load_address = buff[0x08] | ((uint16)buff[0x09])<<8 | ((uint32)buff[0x0a])<<16;
			// 0x02, 0x03, 0x04 are unknwon bytes.
			//call_address = buff[0x08] | ((uint16)buff[0x09])<<8 | ((uint32)buff[0x0a])<<16;
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
	outs << "size = " << std::dec << byte_size << " (0x" << std::hex << std::uppercase << byte_size << "), ";
	outs << "load address = " << "0x" << std::setw(8) << std::setfill('0') << std::uppercase << load_address << ", ";
	outs << "call address = " << "0x" << std::setw(8) << std::setfill('0') << std::uppercase << call_address << ", ";
	outs << "checksum = " << "0x" << std::setw(4) << std::setfill('0') << std::uppercase << checksum << ", ";
	return outs;
}

void dump(std::vector<uint8> & buff);
void disk2cmt(FileInfo & finfo, std::vector<uint8> & buff);

int main(const int argc, const char * argv[]) {
	cout << "Hello." << endl;

	std::ifstream infile(argv[1], std::ios::binary);
	if ( ! infile )
		return -1;

	std::vector<uint8> buff( (std::istreambuf_iterator<char>(infile)),
			(std::istreambuf_iterator<char>()) );

	infile.close();

	FileInfo info(buff, string(argv[1]));
	info.printOn(std::cout);
	std::cout << std::endl;
	dump(buff);
	std::cout << std::endl;
	disk2cmt(info,buff);
	dump(buff);
	FileInfo info2(buff);
	info2.printOn(std::cout);
	std::cout << std::endl;

	if ( argc >= 3 ) {
		std::ofstream outfile(argv[2], std::ios::binary);
		if ( outfile ) {
			for(std::vector<uint8>::iterator it = buff.begin();
					it != buff.end(); ++it) {
				outfile.put(*it);
			}
			outfile.close();
		} else {
			std::cerr << "Failed to open output file " << argv[2] << std::endl;
		}
	}
	return 0;
}

void disk2cmt(FileInfo & finfo, std::vector<uint8> & buff) {
	std::vector<uint8> newbuff;
	unsigned int ui;

	newbuff.push_back(FileInfo::PC_E500_CSAVE_M);
	for(ui=0; ui< 16 && finfo.save_name[ui] != 0; ++ui) {
		newbuff.push_back(finfo.save_name[ui]);
	}
	for( ; ui< 16; ++ui) {
		newbuff.push_back(0x20);
	}
	newbuff.push_back(0x0D);
	//
	newbuff.push_back(finfo.byte_size & 0xff);
	newbuff.push_back(finfo.byte_size>>8 & 0xff);
	newbuff.push_back(finfo.load_address & 0xff);
	newbuff.push_back(finfo.load_address>>8 & 0xff);
	newbuff.push_back(finfo.call_address & 0xff);
	newbuff.push_back(finfo.call_address>>8 & 0xff);
	for(ui = 0x18; ui < 0x1D; ++ui)
		newbuff.push_back(0x00);
	newbuff.push_back(finfo.byte_size>>16 & 0xff);
	newbuff.push_back(finfo.load_address>>16 & 0xff);
	newbuff.push_back(finfo.call_address>>16 & 0xff);
	for(ui = 0x20; ui < 0x30; ++ui)
		newbuff.push_back(0x00);
	// calculate checksum
	unsigned int xsum = 0;
	for(ui = 0; ui < 0x30; ++ui) {
		for(int bp = 0; bp < 8; bp++)
			xsum += (newbuff[ui]>>bp & 1) == 1;
	}
	newbuff.push_back(xsum>>8 & 0xff);
	newbuff.push_back(xsum & 0xff);
	//
	for(ui = 16; ui < buff.size(); ui++) {
		newbuff.push_back(buff[ui]);
	}
	buff.clear();
	buff.insert(buff.begin(), newbuff.begin(), newbuff.end());
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
