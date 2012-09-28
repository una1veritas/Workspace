#ifndef	_cmn_def_h_
#define	_cmn_def_h_
/* ************************************************************************ */
/*
 * ------------------------------------------------------------------------
 *   Application : -
 *   Micon : TMPA910CRAXBG
 *   Copyright(C) TOSHIBA CORPORATION 2008 All rights reserved
 * ------------------------------------------------------------------------
 */

/*! \file cmn_def.h
	\brief Header file of common macro define

	\author TOSHIBA CORPORATION

	\date 2008/08/11 New
	\date 2008/10/24 A910 Sample version 1.0
 */
/* ************************************************************************ */

/* ************************************************************************ */
/*
 * --------------------------------------------------------------------------
 *   Header Include Area
 * --------------------------------------------------------------------------
 */


/*
 * --------------------------------------------------------------------------
 *   Macro Define
 * --------------------------------------------------------------------------
 */
#define CSIZEofUINT8_t		1			/*!< size of UINT8_t		*/
#define CSIZEofUINT16_t		2			/*!< size of UINT16_t		*/
#define CSIZEofUINT32_t		4			/*!< size of UINT32_t		*/

#define CBITofUINT8_t		8			/*!< bit num. of UINT8_t	*/
#define CBITofUINT16_t		16			/*!< bit num. of UINT16_t	*/
#define CBITofUINT32_t		32			/*!< bit num. of UINT32_t	*/

#define CARRAYofUINT16_0	0			/*!< word[0]/s_data[0]		*/
#define CARRAYofUINT16_1	1			/*!< word[1]/s_data[1]		*/

#define CARRAYofUINT8_0		0			/*!< byte[0]/c_data[0]		*/
#define CARRAYofUINT8_1		1			/*!< byte[1]/c_data[1]		*/
#define CARRAYofUINT8_2		2			/*!< byte[2]/c_data[2]		*/
#define CARRAYofUINT8_3		3			/*!< byte[3]/c_data[3]		*/

#define NULL				0			/*!< null pointer			*/
#define FALSE				0			/*!< flase or failed		*/
#define TRUE				1			/*!< true or succeed		*/

#define CBIT_FALSE			FALSE		/*!< bit = 0				*/
#define CBIT_TRUE			TRUE		/*!< bit = 1				*/


/* **************************************************************** */
#endif	/* _cmn_def_h_ */
