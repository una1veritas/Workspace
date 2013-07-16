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
#include <avr/io.h>
#include "DigitalINs.h"

/* DEFINE LOCAL TYPES HERE */

/* DEFINE LOCAL CONSTANTS HERE */
#define DIN3_DDR DDRD
#define DIN3_PORT PIND
#define DIN3_PIN (1 << 3)

#define DIN2_DDR DDRB
#define DIN2_PORT PINB
#define DIN2_PIN (1 << 2)

#define DIN1_DDR DDRB
#define DIN1_PORT PINB
#define DIN1_PIN (1 << 1)

#define DIN0_DDR DDRB
#define DIN0_PORT PINB
#define DIN0_PIN (1 << 0)

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: DINs_Initialize(..) - initializes pins
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char DINs_Initialize(void)
{
	// make pins inputs
	DIN0_DDR &= ~DIN0_PIN;
	DIN1_DDR &= ~DIN1_PIN;
	DIN2_DDR &= ~DIN2_PIN;
	DIN3_DDR &= ~DIN3_PIN;
	
	return 0;
}

/******************************************************************************
* Description: DINs_Get(..) - reads inputs
* Input: 	none
* Output: 	none
* Return:	bitmap of inputs, bit0 is DIN0, bit1 is DIN1 and so on
*******************************************************************************/
uint8_t DINs_Get(void)
{
	uint8_t bitmap = 0;
	
	if(!(DIN0_PORT & DIN0_PIN))
		bitmap |= (1 << 0);
	if(!(DIN1_PORT & DIN1_PIN))
		bitmap |= (1 << 1);
	if(!(DIN2_PORT & DIN2_PIN))
		bitmap |= (1 << 2);
	if(!(DIN3_PORT & DIN3_PIN))
		bitmap |= (1 << 3);
		
	return bitmap;
}


