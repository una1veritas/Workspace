/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : DigitalOUTs.h
 *    Description : Simple outputs driver
 *
 *    History :
 *    1. Date        : 03 Nov 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef _DIGITAL_OUTS_H
#define _DIGITAL_OUTS_H

/******************************************************************************
* Description: DOUTs_Initialize(..) - initializes pins and sets to inactive state
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char DOUTs_Initialize(void);

/******************************************************************************
* Description: DOUTs_Set(..) - set outputs
* Input: 	bitmap of outputs, bit0 is OUT0, bit1 is OUT1 and so on
* Output: 	none
* Return:	none
*******************************************************************************/
void DOUTs_Set(uint8_t bitmap);

/******************************************************************************
* Description: DOUTs_Set(..) - get states of outputs
* Input: 	none
* Output: 	none
* Return:	bitmap of outputs, bit0 is OUT0, bit1 is OUT1 and so on
*******************************************************************************/
uint8_t DOUTs_Get(void);

#endif // _DIGITAL_OUTS_H



