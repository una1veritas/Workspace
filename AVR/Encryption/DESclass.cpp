//=============================================================================
// Copyright Atmel Corporation 2003. All Rights Reserved.
//
// File:			DES.cpp
// Compiler:		Microsoft Visual C++ 6.0
// Output Size:
// Based on work by:ØE, VU
// Created:			4-Feb-2003	JP (Atmel Finland)
// Modified:	
//
// Support Mail:	avr@atmel.com
//
// Description:		DES encryption algorithm
//
//					Please refer to Application Note Documentation for more
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


#include "DES.h"
#include "DataBuffer.h"
#include "CreateException.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>


//=============================================================================
// S-boxes
//
// The original form of the S-box 1 is as follows.
//
// 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
//  0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
//  4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
// 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13
//
// Since every number only requires 4 bits, two numbers are stored
// as one byte.

const unsigned char sTable[256] =
{
	0xE4, 0xD1, 0x2F, 0xB8, 0x3A, 0x6C, 0x59, 0x07,
	0x0F, 0x74, 0xE2, 0xD1, 0xA6, 0xCB, 0x95, 0x38,
	0x41, 0xE8, 0xD6, 0x2B, 0xFC, 0x97, 0x3A, 0x50,
	0xFC, 0x82, 0x49, 0x17, 0x5B, 0x3E, 0xA0, 0x6D,

	0xF1, 0x8E, 0x6B, 0x34, 0x97, 0x2D, 0xC0, 0x5A,
	0x3D, 0x47, 0xF2, 0x8E, 0xC0, 0x1A, 0x69, 0xB5,
	0x0E, 0x7B, 0xA4, 0xD1, 0x58, 0xC6, 0x93, 0x2F,
	0xD8, 0xA1, 0x3F, 0x42, 0xB6, 0x7C, 0x05, 0xE9,

	0xA0, 0x9E, 0x63, 0xF5, 0x1D, 0xC7, 0xB4, 0x28,
	0xD7, 0x09, 0x34, 0x6A, 0x28, 0x5E, 0xCB, 0xF1,
	0xD6, 0x49, 0x8F, 0x30, 0xB1, 0x2C, 0x5A, 0xE7,
	0x1A, 0xD0, 0x69, 0x87, 0x4F, 0xE3, 0xB5, 0x2C,

	0x7D, 0xE3, 0x06, 0x9A, 0x12, 0x85, 0xBC, 0x4F,
	0xD8, 0xB5, 0x6F, 0x03, 0x47, 0x2C, 0x1A, 0xE9,
	0xA6, 0x90, 0xCB, 0x7D, 0xF1, 0x3E, 0x52, 0x84,
	0x3F, 0x06, 0xA1, 0xD8, 0x94, 0x5B, 0xC7, 0x2E,

	0x2C, 0x41, 0x7A, 0xB6, 0x85, 0x3F, 0xD0, 0xE9,
	0xEB, 0x2C, 0x47, 0xD1, 0x50, 0xFA, 0x39, 0x86,
	0x42, 0x1B, 0xAD, 0x78, 0xF9, 0xC5, 0x63, 0x0E,
	0xB8, 0xC7, 0x1E, 0x2D, 0x6F, 0x09, 0xA4, 0x53,

	0xC1, 0xAF, 0x92, 0x68, 0x0D, 0x34, 0xE7, 0x5B,
	0xAF, 0x42, 0x7C, 0x95, 0x61, 0xDE, 0x0B, 0x38,
	0x9E, 0xF5, 0x28, 0xC3, 0x70, 0x4A, 0x1D, 0xB6,
	0x43, 0x2C, 0x95, 0xFA, 0xBE, 0x17, 0x60, 0x8D,

	0x4B, 0x2E, 0xF0, 0x8D, 0x3C, 0x97, 0x5A, 0x61,
	0xD0, 0xB7, 0x49, 0x1A, 0xE3, 0x5C, 0x2F, 0x86,
	0x14, 0xBD, 0xC3, 0x7E, 0xAF, 0x68, 0x05, 0x92,
	0x6B, 0xD8, 0x14, 0xA7, 0x95, 0x0F, 0xE2, 0x3C,

	0xD2, 0x84, 0x6F, 0xB1, 0xA9, 0x3E, 0x50, 0xC7,
	0x1F, 0xD8, 0xA3, 0x74, 0xC5, 0x6B, 0x0E, 0x92,
	0x7B, 0x41, 0x9C, 0xE2, 0x06, 0xAD, 0xF3, 0x58,
	0x21, 0xE7, 0x4A, 0x8D, 0xFC, 0x90, 0x35, 0x6B
};
  

//=============================================================================
// Selection Order

const unsigned char sOrder[8] = {0, 0, 0, 0, 5, 1, 2, 3};


//=============================================================================
// Initial Permutation bit selection table

const unsigned char ipTable[64] =
{
	57, 49, 41, 33, 25, 17,  9,  1,
	59, 51, 43, 35, 27, 19, 11,  3,
	61, 53, 45, 37, 29, 21, 13,  5,
	63, 55, 47, 39, 31, 23, 15,  7,
	56, 48, 40, 32, 24, 16,  8,  0,
	58, 50, 42, 34, 26, 18, 10,  2,
	60, 52, 44, 36, 28, 20, 12,  4,
	62, 54, 46, 38, 30, 22, 14,  6
};


//=============================================================================
// Inverse Initial Permutation bit selection table

const unsigned char iipTable[64] =
{
	39,  7, 47, 15, 55, 23, 63, 31,
	38,  6, 46, 14, 54, 22, 62, 30,
	37,  5, 45, 13, 53, 21, 61, 29,
	36,  4, 44, 12, 52, 20, 60, 28,
	35,  3, 43, 11, 51, 19, 59, 27,
	34,  2, 42, 10, 50, 18, 58, 26,
	33,  1, 41,  9, 49, 17, 57, 25,
	32,  0, 40,  8, 48, 16, 56, 24
};


//=============================================================================
// Expansion bit selection table

const unsigned char eTable[48] =
{
	31, 0,  1,  2,  3,  4,
	3,  4,  5,  6,  7,  8,
	7,  8,  9,  10, 11, 12,
	11, 12, 13, 14, 15, 16,
	15, 16, 17, 18, 19, 20,
	19, 20, 21, 22, 23, 24,
	23, 24, 25, 26, 27, 28,
	27, 28, 29, 30, 31,  0
};


//=============================================================================
// Permutation bit selection table

const unsigned char pTable[32] =
{
	15, 6,  19, 20,
	28, 11, 27, 16,
	0,  14, 22, 25,
	4,  17, 30, 9,
	1,  7,  23, 13,
	31, 26, 2,  8,
	18, 12, 29, 5,
	21, 10, 3,  24
};


//=============================================================================
// Permuted Choice 1 bit selection table

const unsigned char pc1Table[56] =
{
	56, 48, 40, 32, 24, 16, 8,
	0,  57, 49, 41, 33, 25, 17,
	9,  1,  58, 50, 42, 34, 26,
	18, 10, 2,  59, 51, 43, 35,
	62, 54, 46, 38, 30, 22, 14,
	6,  61, 53, 45, 37, 29, 21,
	13,  5, 60, 52, 44, 36, 28,
	20, 12, 4,  27, 19, 11, 3
};


//=============================================================================
// Permuted Choice 2 bit selection table

const unsigned char pc2Table[48] =
{
	13, 16, 10, 23,  0, 4,
	2,  27, 14, 5,  20, 9,
	22, 18, 11, 3,  25, 7,
	15, 6,  26, 19, 12, 1,
	40, 51, 30, 36, 46, 54,
	29, 39, 50, 44, 32, 47,
	43, 48, 38, 55, 33, 52,
	45, 41, 49, 35, 28, 31
};

	
//=============================================================================
// Key Rotate Left table

const unsigned char rTable[16] =
	{1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};


//=============================================================================
// Constructor

Des::Des(DataBuffer& initialVector)
{
	memcpy(chainBlockM, initialVector.data(), 8);
	srand((unsigned)time(NULL));
}


//=============================================================================
// Destructor

Des::~Des()
{
}


//=============================================================================
// Returns the value of the bit 'index' in the bit table 'table'.

unsigned char Des::getBit(const unsigned char *table, unsigned char index)
{
	return (table[index >> 3] & (0x80 >> (index & 7)));
}


//=============================================================================
// Sets or clears the bit 'index' in the bit table 'table', regarding the value
// of 'value'.

void Des::putBit(unsigned char *table, unsigned char index,
				 unsigned char value)
{
	unsigned char byteIndex;
	unsigned char mask;

	byteIndex = index >> 3;
	mask = 0x80 >> (index & 0x07);

	if (value)
		table[byteIndex] |= mask;
	else
		table[byteIndex] &= ~mask;
}


//=============================================================================
// Permutes 'in' using permutation table 'table' and puts the result to 'out'

void Des::permute(unsigned char size, unsigned char *out,
	const unsigned char *in, const unsigned char *table)
{
	while (size--)
		putBit(out, size, getBit(in, table[size]));
}


//=============================================================================
// Generates the kTable from keys

void Des::scheduleKey(DataBuffer key[3])
{
	unsigned char x, y, z, tmpBit1, tmpBit2;
	unsigned char tempKey[7];
	
	for (int i = 0; i < 3; i++)
	{
		// Check Key Parity
		for (x = 0; x < 8; x++)
		{
			z = key[i][x];
			z ^= z >> 4;
			z ^= z >> 2;
			z ^= z >> 1;

			if ((z & 0x01) == 0)
			{
				char keystr[] = "KEYx";
				keystr[3] = '1' + i;
				throw new CreateException(ERROR_KEY_PARITY, keystr);
			}
		}

		permute(56, tempKey, key[i].data(), pc1Table);
		for (z = 0; z < 16; z++)
		{
			for (y = 0; y < rTable[z]; y++)
			{
				tmpBit1 = getBit(tempKey, 0);
				tmpBit2 = getBit(tempKey, 28);
				for (x = 0; x < 6; x++)
				{
					tempKey[x] <<= 1;
					putBit(&tempKey[x], 7, getBit(&tempKey[x + 1], 0));
				}
				tempKey[6] <<= 1;
				putBit(tempKey, 27, tmpBit1);
				putBit(tempKey, 55, tmpBit2);
			}
			permute(48, kTableM[i][z], tempKey, pc2Table);
		}
	}
}


//=============================================================================
// Return kTable for 'key' at 'index' (six bytes).

DataBuffer Des::getK(int key, int index)
{
	return DataBuffer(kTableM[key][index], 6);
}


//=============================================================================
// Calculates the exclusive or function of 'left' and 'right'. The result
// is written to 'left'. 'byteCount' indicates the size of the two vectors in
// bytes.

void Des::xor(unsigned char byteCount, unsigned char *left,
			  const unsigned char *right)
{
	while (byteCount--)
		*left++ ^= *right++;
}


//=============================================================================
// Generates 32 bits of output with 48 bits input, according to the S-boxes.

void Des::s(unsigned char *out, const unsigned char *in)
{
	unsigned char x, y;
	unsigned char bitCount;
	unsigned char index;
	unsigned char temp; 

	for (x = 0; x < 8; x++)
	{
		if (!(x & 0x01))
			out[x >> 1] = 0;

		bitCount = 6 * x;
		index = ((x << 5) & 0xe0);
    
		for(y = 7; y > 2; y--)
			putBit(&index, y, getBit(in, bitCount + sOrder[y]));
      
		temp = sTable[index];
    
		if (!(getBit(in, bitCount + 4)))
			temp >>= 4;
		temp &= 0x0F;
    
		if (!(x & 0x01))
			temp <<= 4;
		out[x >> 1] |= temp;
	}
}


//=============================================================================
// Encrypts 'buffer' using 3DES encryption algorithm. 'buffer' must be 8 bytes
// long. chainBlock points to the previous cipher block
//
// Before this function is called, the scheduleKey function must be called
// at least once.

void Des::encrypt(unsigned char *buffer, const unsigned char *chainBlock)
{
	unsigned char x, y;
	unsigned char *l, *r, *tmpPointer, tmpBuffer[4];
	unsigned char ipOut[8], iipIn[8];
	unsigned char temp1[6], temp2[4];

	// Cipher-Block-Chaining. Exclusive or with the previous datablock.
	xor(8, buffer, chainBlock);

	l = &ipOut[0];
	r = &ipOut[4];

	// Initial Permutation
	permute(64, ipOut, buffer, ipTable);

	// 3 * 16 iterations (three times DES == triple-DES)
	for (x = 0; x < 48; x++)
	{
		y = x & 0x0F;

		// f(......)
		permute(48, temp1, r, eTable);

		if (x < 16)	
			// f(R, K[Y] -> DES ENCRYPT
			xor(6, temp1, kTableM[0][y]);
		else if (x < 32)
			// f(R, K[15 - Y]) -> DES DECRYPT
			xor(6, temp1, kTableM[1][15 - y]);
		else
			// f(R, K[Y] -> DES ENCRYPT
			xor(6, temp1, kTableM[2][y]);


		s(temp2, temp1);
		permute(32, tmpBuffer, temp2, pTable);

		xor(4, l, tmpBuffer);

		// If not iteration 15., 31., or 47.
		if (y != 0x0F)
		{
			// swap R and L
			tmpPointer = l;
			l = r;
			r = tmpPointer;
		}
	}
  
	// Swap the two buffers L and R (not just the pointers) before Inverse Initial Permutation
	iipIn[0] = l[0];
	iipIn[1] = l[1];
	iipIn[2] = l[2];
	iipIn[3] = l[3];
	iipIn[4] = r[0];
	iipIn[5] = r[1];
	iipIn[6] = r[2];
	iipIn[7] = r[3];
  
	// Inverse Initial Permutation
	permute(64, buffer, iipIn, iipTable);
}


//=============================================================================
// Encrypts a buffer of data. The data length is first aligned to next 8 bytes.

void Des::encryptBuffer(DataBuffer& buffer)
{
	// Check if the buffer needs to be filled.
	if( buffer.size() % 8 ) {
		// Fill the rest (to align to 8 bytes) with random data
		int fillSize = 8 - (buffer.size() % 8);

		while (fillSize--)
			buffer += (unsigned char)rand();
	}
	
	// Encrypt buffer one cipher block at a time
	for (int i = 0; i < buffer.size(); i += 8)
	{
		encrypt(&buffer[i], chainBlockM);
		memcpy(chainBlockM, &buffer[i], 8);
	}
}
