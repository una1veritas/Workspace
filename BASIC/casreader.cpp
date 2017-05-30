/*
 * main.cpp
 *
 *  Created on: 2017/05/22
 *      Author: sin
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <iomanip>

#define MIN(x,y) ((x) <= (y) ? (x) : (y))

int main(const int argc, const char * argv[]) {
	unsigned int data[8*1024];
	unsigned int bytecount;
	unsigned int codelength;
	bool binary = false;
	std::ifstream file;

	if ( argc < 2 || strlen(argv[1]) == 0 ) {
		std::cerr << "Requires file name." << std::endl;
		return -1;
	}

	if (argc == 2 ) {
		file.open(argv[1]);
	} else if (argc == 3 && strcmp(argv[1],"-b") == 0) {
		binary = true;
		file.open(argv[2], std::ios::in | std::ios::binary);
	}

	if ( !file ) {
		std::cerr << "File open failed." << std::endl;
		return -1;
	} else {
		std::cout << "File " << argv[1] << " opened." << std::endl;
	}

	bytecount = 0;
	if ( !binary) {
		std::string str, tmp;
		std::stringstream linebuf;
		while ( !file.eof() ) {
			std::getline(file, str);
			if ( str.length() == 0 )
				break;
			//std::cout << "'" << str << "'" << std::endl;
			linebuf.str(str);
			linebuf.clear();
			while ( true ) {
				tmp.clear();
				linebuf >> tmp;
				if ( linebuf.eof() )
					break;
				if ( tmp.length() == 0 )
					continue;
				unsigned int val = std::stoi(tmp, 0, 16);
				data[bytecount] = val;
				//std::cout << dcount << ": " << val << " " << "'" << tmp << "' ";
				++bytecount;
			}
			//std::cout << std::endl;
		}
	} else {
		while (true) {
			char val;
			file.read(&val,1);
			if ( file.eof() )
				break;
			data[bytecount] = (unsigned char) val;
			//std::cout << dcount << ": " << val << " " << "'" << tmp << "' ";
			++bytecount;
			//std::cout << std::endl;
		}

	}
	file.close();

	std::cout << std::endl <<  "Header " << std::dec << 40 << " bytes (fixed size);" << std::endl;
	unsigned int chksum = 0;
	for(int i = 0; i < MIN(40, bytecount) ; ++i) {
		if ( i == 0 || (i & 0x07) == 0 )
			std::cout << std::endl << std::setfill('0') << std::setw(2) << std::hex << i << ": ";
		chksum += data[i];
		std::cout << std::setfill('0') << std::setw(2) <<std::hex << data[i] << " ";
	}
	std::cout << std::endl;

	std::cout << std::endl << "Program name = '";
	for(int i = 0; i < 16; ++i) {
		if ( data[0x09+i] == 0 )
			break;
		std::cout << (char) data[0x09+i];
	}
	std::cout << "'" << std::endl;

	std::cout << std::endl <<  "checksum: " << std::setfill('0') << std::setw(4) <<std::hex << chksum << std::endl;

	std::cout << std::endl <<  "Check sum for header, 2 bytes (the higher byte first)" << std::endl;

	for(int i = 40; i < MIN(42, bytecount) ; ++i) {
		if ( i == 40 || (i & 0x07) == 0 )
			std::cout << std::endl << std::setfill('0') << std::setw(2) << std::hex << i << ": ";
		std::cout << std::setfill('0') << std::setw(2) <<std::hex << data[i] << " ";
	}
	std::cout << std::endl;

	codelength = (data[0x24]<<8) + data[0x25];
	std::cout << std::endl << "Body " << std::dec << codelength << "(0x" << std::hex << codelength<< ") bytes" << std::endl;
	// line number (2 bytes), length (from the next to the last CR), content, CR (0x0D)
	chksum = 0;
	for(int i = 42; i < 42 + codelength ; ++i) {
		if ( i == 42 || (i & 0x07) == 0 )
			std::cout << std::endl << std::setfill('0') << std::setw(2) << std::hex << i << ": ";
		chksum += data[i];
		std::cout << std::setfill('0') << std::setw(2) <<std::hex << data[i] << " ";
	}
	std::cout << std::endl;

	std::cout << std::endl << "Tail" << std::endl;

	std::cout << std::endl << std::setfill('0') << std::setw(2) << std::hex << (42+codelength) << ": ";
	std::cout << std::setfill('0') << std::setw(2) <<std::hex << data[42+codelength] << " ";

	chksum += data[42+codelength];
	std::cout << std::endl << "checksum: " << std::setfill('0') << std::setw(4) <<std::hex << chksum << std::endl;

	std::cout << std::endl <<  "Check sum for body 2 bytes, and the ending code 0x55 'U'." << std::endl;
	for(int i = 42 + codelength + 1; i < bytecount ; ++i) {
		if ( i == 42 + codelength + 1 || (i & 0x07) == 0 )
			std::cout << std::endl << std::setfill('0') << std::setw(2) << std::hex << i << ": ";
		std::cout << std::setfill('0') << std::setw(2) <<std::hex << data[i] << " ";
	}
	std::cout << std::endl;


	return 0;
}


