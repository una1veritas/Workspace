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

typedef unsigned char byte;

void dumphex(const byte * const data, const unsigned long bytes);
unsigned long read(byte * data, unsigned long bytelimit, std::ifstream & file, const bool bmode = false);

unsigned long PC150xHeader(byte * outdata, const byte * const indata, unsigned long bcount, const char progname[16]);

int main(const int argc, const char * argv[]) {
	const unsigned long limit = 8*1024;
	byte data[limit], outdata[limit];
	unsigned int bytecount;
	unsigned int codelength;
	std::ifstream infile;
	std::ofstream outfile;
	char progname[16];

	bool binary = false;
	std::string infname = "", outfname = "";

	if ( argc < 2 ) {
		std::cerr << "Requires file name." << std::endl;
		return -1;
	} else {
		int argi = 1;
		for( argi = 1; argi < argc; ++argi) {
			std::cout << argv[argi] << std::endl;
			if ( argv[argi][0] == '-' ) {
				if ( strcmp(argv[argi]+1, "b") == 0 ) {
					binary = true;
				}
			} else {
				if ( infname == "" ) {
					infname = argv[argi];
				} else {
					outfname = argv[argi];
				}
			}
		}
	}

	infile.open(infname);
	if ( !infile ) {
		std::cerr << "Input file " << infname << " open failed." << std::endl;
		return -1;
	} else {
		std::cout << "Input file " << infname << " opened." << std::endl;
		bytecount = read(data, limit, infile, binary);
	}

	dumphex(data, bytecount);

	if ( !outfname.empty() ) {
		outfile.open(outfname);
		if ( !outfile ) {
			std::cerr << "Output file " << outfname << " open failed." << std::endl;
			return -1;
		} else {
			std::cout << "Output file " << outfname << " opened." << std::endl;
		}
	}
	outfile.close();

	strncpy(progname, outfname.c_str(), 16);
	codelength = PC150xHeader(outdata, data, bytecount, progname);

	return 0;
}

/* File name and header for PC-1500 */
/*  WriteHeadToB22Wav */
unsigned long PC150xHeader(byte * outdata, const byte * const data, unsigned long bytecount, const char progname[16]) {
	byte  i ;
	unsigned long len;
	unsigned long tmpl ;
	char tmpstr[20] ;
	int  error ;
	unsigned long outcount;
	unsigned int sum;

	/* Search the length */
	for (i = 0; i < 16 && progname[i] != 0; ++i) {
		tmpstr[i] = progname[i];
	}
	for ( ; i < 17; ++i)
		tmpstr[i] = 0;

	std::cout << "program name " << tmpstr << std::endl;

	/* Write the Header */
	outcount = 0;
	sum = 0;

	for ( i = 0x10 ; i < 0x18 ; ++i ) {
		outdata[outcount++] = i;
		sum += i;
	}

	/* Write the Sub-Ident */
//	if (type == TYPE_DAT)
//		tmpL = 0x04 ;
//	else if (type == TYPE_RSV)
//		tmpL = 0x02 ;
//	else if (type == TYPE_BIN)
//		tmpL = 0x00 ;
//	else // TYPE_IMG
	tmpl = 0x01 ;

	outdata[outcount++] = tmpl;
	sum += tmpl;

	/* Write the Name */
	for ( i = 0 ; i < 16 ; ++i ) {
		outdata[outcount++] = tmpstr[i];
		sum += tmpstr[i];
	}

	/* Write 9 null bytes */
	for ( i = 0 ; i < 9 ; ++i ) {
		outdata[outcount++] = 0;
		// sum += 0;
	}

	/* Write the address */
	// if (type==TYPE_IMG && addr==0) addr = 0xC5; /* RSV length before BASIC program */
	tmpl = 0x40C5;

	outdata[outcount++] = tmpl>>8 & 0xff;
	sum += tmpl>>8 & 0xff;
	outdata[outcount++] = tmpl & 0xff;
	sum += tmpl & 0xff;

	/* Write the Buffer Size */
	//if (type == TYPE_DAT)
	//	len = 0 ;
	//else if (type == TYPE_BIN || type == TYPE_RSV)
	//	len = size - 1 ;
	//else
	//	len = size ;
	len = bufferSize; // <==== ????

	outdata[outcount++] = (len >> 8) & 0xFF ;
	sum += (len >> 8) & 0xFF;
	outdata[outcount++] = len & 0xFF ;
	sum += len & 0xff;

	/* Write the entry address */
	/*
	if (type == TYPE_BIN) {
		// if (Acnt<2) eaddr = 0xFFFF ;
		tmpL = (eaddr >> 8) & 0xFF ;
		error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
		if (error != ERR_OK) break ;

		tmpL = eaddr & 0xFF ;
		error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
		if (error != ERR_OK) break ;
	}
	else {
	*/
		tmpl = 0x0000 ;
		outdata[outcount++] = tmpl>>8 & 0xff;
		sum += tmpl>>8 & 0xff;
		outdata[outcount++] = tmpl & 0xff;
		sum += tmpl & 0xff;
/*
	}
*/

	/* Write the checksum */
	outdata[outcount++] = sum>>8 & 0xff;
	outdata[outcount++] = sum & 0xff;

    return outcount;
}

void dumphex(const unsigned char * const data, const unsigned long bytes) {
	for(int i = 0; i < bytes; ++i ) {
		if ( (i & 0x0f) == 0 ) {
			std::cout << std::endl << std::setfill('0') << std::setw(4)
			<< std::hex << i << ": ";
		}
		std::cout << std::setfill('0') << std::setw(2)
		<< std::hex << (int) data[i] << " ";
	}
	std::cout << std::endl;
}

unsigned long read(unsigned char * data, unsigned long bytelimit, std::ifstream & file, const bool bmode) {
	unsigned long bytecount;

	if ( bmode ) {
		std::cout << "binary mode." << std::endl;
	} else {
		std::cout << "ascii hex mode." << std::endl;
	}

	bytecount = 0;
	if ( !bmode ) {
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
				if ( !(bytecount < bytelimit) )
					break;
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
			if ( !(bytecount < bytelimit) )
				break;
		}

	}
	file.close();

	return bytecount;
}

