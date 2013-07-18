/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : AnalogIns.h
 *    Description : configures and reads the analogue inputs of the board
 *
 *    History :
 *    1. Date        : 07 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef ANALOG_INS_H
#define ANALOG_INS_H

/******************************************************************************
* Description: AINs_Initialize(..) - initializes pins
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char AINs_Initialize(void);

/******************************************************************************
* Description: AINs_Get(..) - sample an analogue channel
* Input: 	channel - the pin to sample
* Output: 	none
* Return:	value in steps of the input
*******************************************************************************/
uint16_t AINs_Get(uint8_t channel);

#endif // ANALOG_INS_H
