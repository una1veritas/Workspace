/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : DigitalIns.h
 *    Description : configures and reads the digital inputs of the board
 *
 *    History :
 *    1. Date        : 07 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef DIGITAL_INS_H
#define DIGITAL_INS_H

/******************************************************************************
* Description: DINs_Initialize(..) - initializes pins
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char DINs_Initialize(void);

/******************************************************************************
* Description: DINs_Get(..) - reads inputs
* Input: 	none
* Output: 	none
* Return:	bitmap of inputs, bit0 is DIN0, bit1 is DIN1 and so on
*******************************************************************************/
uint8_t DINs_Get(void);

#endif // DIGITAL_INS_H
