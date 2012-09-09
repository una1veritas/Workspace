/*
 *********************************************************************
 * TITLE  : Common Definition
 *--------------------------------------------------------------------
 * FILE NAME  : cmn_type.h
 * DESCRIPTION  : ALL SFR definitions of TMPA910CRXBG
 *
 * Copyright(C) TOSHIBA CORPORATION 2007 All rights reserved
 *********************************************************************
 */
/*====================================================================
 * File Revision History
 * -------------------------------------------------------------------
 * 07-02-15		(MS6)
 * 				First edition
 *====================================================================
 */
#ifndef	_cmn_type_h_
#define	_cmn_type_h_
/* ************************************************************************ */
/*
 * --------------------------------------------------------------------------
 *   Header Include Area
 * --------------------------------------------------------------------------
 */
#include "cmn_def.h"


/*
 *********************************************************************
 *   TYPE DEFINITIONS
 *********************************************************************
 */

typedef	signed char			CHAR_t;			/*!< signed  8bit 		*/
typedef signed short		SHORT_t;		/*!< signed 16bit		*/
typedef	signed long			LONG_t;			/*!< signed 32bit		*/

typedef	unsigned char		UCHAR_t;		/*!< unsigned  8bit		*/
typedef unsigned short		USHORT_t;		/*!< unsigned 16bit		*/
typedef	unsigned long		ULONG_t;		/*!< unsigned 32bit		*/

typedef	signed char			INT8_t;			/*!< signed  8bit		*/
typedef signed short		INT16_t;		/*!< signed 16bit		*/
typedef	signed long			INT32_t;		/*!< signed 32bit		*/

typedef	unsigned char		UINT8_t;		/*!< unsigned  8bit		*/
typedef unsigned short		UINT16_t;		/*!< unsigned 16bit		*/
typedef	unsigned long		UINT32_t;		/*!< unsigned 32bit		*/

typedef	signed char			BOOL_t;			/*!< signed  8bit		*/

/* **************************************************************** */

typedef struct
{
	UINT8_t		b0 :1;	/*!< byte<0>	*/
	UINT8_t		b1 :1;	/*!< byte<1>	*/
	UINT8_t		b2 :1;	/*!< byte<2>	*/
	UINT8_t		b3 :1;	/*!< byte<3>	*/
	UINT8_t		b4 :1;	/*!< byte<4>	*/
	UINT8_t		b5 :1;	/*!< byte<5>	*/
	UINT8_t		b6 :1;	/*!< byte<6>	*/
	UINT8_t		b7 :1;	/*!< byte<7>	*/
} BIT8_t;									/*!<  8bit structure		*/

typedef struct
{
	UINT16_t	b0 :1;	/*!< word<0>	*/
	UINT16_t	b1 :1;	/*!< word<1>	*/
	UINT16_t	b2 :1;	/*!< word<2>	*/
	UINT16_t	b3 :1;	/*!< word<3>	*/
	UINT16_t	b4 :1;	/*!< word<4>	*/
	UINT16_t	b5 :1;	/*!< word<5>	*/
	UINT16_t	b6 :1;	/*!< word<6>	*/
	UINT16_t	b7 :1;	/*!< word<7>	*/

	UINT16_t	b8 :1;	/*!< word<8>	*/
	UINT16_t	b9 :1;	/*!< word<9>	*/
	UINT16_t	b10:1;	/*!< word<10>	*/
	UINT16_t	b11:1;	/*!< word<11>	*/
	UINT16_t	b12:1;	/*!< word<12>	*/
	UINT16_t	b13:1;	/*!< word<13>	*/
	UINT16_t	b14:1;	/*!< word<14>	*/
	UINT16_t	b15:1;	/*!< word<15>	*/
} BIT16_t;									/*!< 16bit structure		*/

typedef struct
{
	UINT32_t	b0 :1;	/*!< dword<0>	*/
	UINT32_t	b1 :1;	/*!< dword<1>	*/
	UINT32_t	b2 :1;	/*!< dword<2>	*/
	UINT32_t	b3 :1;	/*!< dword<3>	*/
	UINT32_t	b4 :1;	/*!< dword<4>	*/
	UINT32_t	b5 :1;	/*!< dword<5>	*/
	UINT32_t	b6 :1;	/*!< dword<6>	*/
	UINT32_t	b7 :1;	/*!< dword<7>	*/

	UINT32_t	b8 :1;	/*!< dword<8>	*/
	UINT32_t	b9 :1;	/*!< dword<9>	*/
	UINT32_t	b10:1;	/*!< dword<10>	*/
	UINT32_t	b11:1;	/*!< dword<11>	*/
	UINT32_t	b12:1;	/*!< dword<12>	*/
	UINT32_t	b13:1;	/*!< dword<13>	*/
	UINT32_t	b14:1;	/*!< dword<14>	*/
	UINT32_t	b15:1;	/*!< dword<15>	*/

	UINT32_t	b16:1;	/*!< dword<16>	*/
	UINT32_t	b17:1;	/*!< dword<17>	*/
	UINT32_t	b18:1;	/*!< dword<18>	*/
	UINT32_t	b19:1;	/*!< dword<19>	*/
	UINT32_t	b20:1;	/*!< dword<20>	*/
	UINT32_t	b21:1;	/*!< dword<21>	*/
	UINT32_t	b22:1;	/*!< dword<22>	*/
	UINT32_t	b23:1;	/*!< dword<23>	*/

	UINT32_t	b24:1;	/*!< dword<24>	*/
	UINT32_t	b25:1;	/*!< dword<25>	*/
	UINT32_t	b26:1;	/*!< dword<26>	*/
	UINT32_t	b27:1;	/*!< dword<27>	*/
	UINT32_t	b28:1;	/*!< dword<28>	*/
	UINT32_t	b29:1;	/*!< dword<29>	*/
	UINT32_t	b30:1;	/*!< dword<30>	*/
	UINT32_t	b31:1;	/*!< dword<31>	*/
} BIT32_t;									/*!< 32bit structure		*/

/* **************************************************************** */

typedef union
{
	UINT8_t		byte;
	BIT8_t		bit;
} UINT8u_t;									/*!<  8bit union structure	*/

typedef union
{
	UINT16_t	word;
	UINT8_t		byte[CSIZEofUINT16_t/CSIZEofUINT8_t];
	BIT16_t		bit;
} UINT16u_t;								/*!< 16bit union structure	*/

typedef union
{
	UINT32_t	dword;
	UINT16_t	word[CSIZEofUINT32_t/CSIZEofUINT16_t];
	UINT8_t		byte[CSIZEofUINT32_t/CSIZEofUINT8_t];
	BIT32_t		bit;
} UINT32u_t;								/*!< 32bit union structure	*/

/* **************************************************************** */
#endif	/* _cmn_type_h_ */
