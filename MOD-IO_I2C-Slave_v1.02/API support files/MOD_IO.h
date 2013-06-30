/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : MOD_IO.c
 *    Description : API for MOD-IO board by Olimex
 *
 *    History :
 *    1. Date        : 08 Nov 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef _MOD_IO_H
#define _MOD_IO_H

/******************************************************************************
* Description: MODIO_Initialize(..) - initializes pins and registers of the MOD_IO
* Input: 	none
* Output: 	none
* Return:	0 if sucessfully initialized, -1 if error occured 
*******************************************************************************/
char MODIO_Initialize(void);

/******************************************************************************
* Description: MODIO_ReadDINs(..) - reads the digital inputs of the board
* Input: 	none
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_ReadDINs(BYTE *data);

/******************************************************************************
* Description: MODIO_ReadAIN(..) - reads an analog input of the board
* Input: 	channel - analog inpout to scan
* Output: 	data - value of the input level
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_ReadAIN(WORD *data, BYTE channel);

/******************************************************************************
* Description: MODIO_WriteDOUTs(..) - sets stated of digital outputs
* Input: 	bitmap - value to send to the outputs
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_WriteDOUTs(BYTE bitmap);

/******************************************************************************
* Description: MODIO_UpdateSlvAddress(..) - updates the slave address of the board
* Input: 	newaddr - new slave address of the board
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_UpdateSlvAddress(BYTE newaddr);

#endif // _MOD_IO_H



