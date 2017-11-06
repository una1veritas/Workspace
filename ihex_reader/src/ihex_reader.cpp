//============================================================================
// Name        : ihex_reader.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

//#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cctype>

int main(const int argc, const char * argv[]) {

	if ( !(argc >= 2) ) {
		puts("No arguments.\n");
		return EXIT_FAILURE;
	}
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	FILE * fp = fopen(argv[1], "r");
	printf("Open file '%s' ", argv[1]);
	if ( fp == NULL ) {
		printf("failed.\n");
	} else {
		printf("succeeded.\n");
	}

	enum {
		STARTCODE = 0, // awaiting start code ':'
		BYTECOUNT,
		ADDRESS,
		RECORDTYPE,
		DATA,
		CHECKSUM,
		ENDOFFILE,
		CHECKSUMERROR = 0xfd,
		FILEREADERROR = 0xfe,
		SYNTAXERROR = 0xff,
	};
	unsigned char sta = STARTCODE;
	unsigned long baseaddress = 0;
	unsigned int address = 0;
	char buf[256];
	unsigned char bytes[256];
	unsigned int byteindex, bytecount;
	unsigned char xsum, recordtype;

	unsigned int totalbytecount = 0;
	char * ptr;
	do {
		if ( sta == STARTCODE ) {
			if ( fgets(buf, 1+1, fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else if ( buf[0] == ':' ) {
				xsum = 0;
				sta = BYTECOUNT;
				//printf("start code, ");
			} else if ( !iscntrl(buf[0]) ) {
				sta = SYNTAXERROR;
				break;
			}
		} else if ( sta == BYTECOUNT ) {
			if ( fgets(buf,2+1,fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else {
				bytecount = strtol(buf, &ptr, 16);
				if ( *ptr != (char)0 ) {
					sta = SYNTAXERROR;
					break;
				}
				xsum += bytecount;
				sta = ADDRESS;
				//printf("byte count %d, ",bytecount);
			}
		} else if ( sta == ADDRESS ) {
			if ( fgets(buf,4+1,fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else {
				address = strtol(buf, &ptr, 16);
				if ( *ptr != (char)0 ) {
					sta = SYNTAXERROR;
					break;
				}
				//printf("address %04X, ",address);
				xsum += address>>8 & 0xff;
				xsum += address & 0xff;
				sta = RECORDTYPE;
			}
		} else if ( sta == RECORDTYPE ) {
			if ( fgets(buf,2+1,fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else {
				recordtype = strtol(buf, &ptr, 16);
				if ( *ptr != (char)0 ) {
					sta = SYNTAXERROR;
					break;
				}
				xsum += recordtype;
				if ( bytecount ) {
					sta = DATA;
					byteindex = 0;
				} else {
					sta = CHECKSUM;
				}
				//printf("type %02X, ", recordtype);
			}
		} else if ( sta == DATA ) {
			if ( fgets(buf,2+1,fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else {
				bytes[byteindex] = strtol(buf, &ptr, 16);
				if ( *ptr != (char)0 ) {
					sta = SYNTAXERROR;
					break;
				}
				xsum += bytes[byteindex];
				//printf("%02X ", bytes[byteindex]);
				byteindex++;
				if ( byteindex < bytecount ) {
					sta = DATA;
				} else {
					sta = CHECKSUM;
				}
			}
		} else if ( sta == CHECKSUM ) {
			if ( fgets(buf,2+1,fp) == NULL ) {
				sta = FILEREADERROR;
				break;
			} else {
				xsum += strtol(buf, &ptr, 16);
				if ( *ptr != (char)0 ) {
					sta = SYNTAXERROR;
					break;
				}
				//printf(": %02X, ", xsum );
				if ( xsum ) {
					sta = CHECKSUMERROR;
					break;
				}
				if ( recordtype == 1 ) {
					printf("end-of-file.\n");
					sta = ENDOFFILE;
				} else {
					printf("/* %04X */ ", address);
					for(unsigned int i = 0; i < byteindex; i++) {
						printf("0x%02X, ", bytes[i]);
					}
					printf("\n");
					totalbytecount += bytecount;
					sta = STARTCODE;
				}
			}
		}
	} while ( !feof(fp) && (sta != ENDOFFILE) );
	if ( sta != ENDOFFILE ) {
		printf("error %d.\n", sta);
	}
	fclose(fp);
	printf("file closed.\n");

	printf("totally %d bytes.\n", totalbytecount);
	return EXIT_SUCCESS;
}
