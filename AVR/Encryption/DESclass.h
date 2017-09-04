//=============================================================================
// Copyright Atmel Corporation 2003. All Rights Reserved.
//
// File:			DES.h
// Compiler:		Microsoft Visual C++ 6.0
// Output Size:
// Based on work by:OE, VU
// Created:			4-Feb-2003	JP (Atmel Finland)
// Modified:	
//
// Support Mail:	avr@atmel.com
//
// Description:		Please refer to Application Note Documentation for more
//					information.
//
//					For details on DES, please refer to the official FIPS 46-3
//					document:
//
//				http://csrc.nist.gov/publications/fips/fips46-3/fips46-3.pdf
//
//
// Other Info:		Since the numbers in the permutation-tables in the FIPS
//					document start at 1 instead of 0, the numbers in all
//					permutation-tables in this file are 1 less. In other words:
//					The tables in this file are indexed with 0 as the first
//					bit.
//=============================================================================

#ifndef DES_H
#define DES_H

#include "DataBuffer.h"

class Des  
{
public:
	Des(DataBuffer& initialVector);
	virtual ~Des();

	void scheduleKey(DataBuffer key[3]);
	DataBuffer getK(int key, int index);
	void encrypt(unsigned char *buffer, const unsigned char *chainBlock);
	void encryptBuffer(DataBuffer& Buffer);

private:
	unsigned char chainBlockM[8];
	unsigned char kTableM[3][16][6];
	
	unsigned char getBit(const unsigned char *table, unsigned char index);
	void putBit(unsigned char *table, unsigned char index, unsigned char value);
	void permute(unsigned char size, unsigned char *out,
				 const unsigned char *in, const unsigned char *table);
	void xor(unsigned char byte_count, unsigned char *left,
			 const unsigned char *right);
	void s(unsigned char *OutData, const unsigned char *InData);
};

#endif // DES_H
