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
#include <avr/io.h>
#include "DigitalOUTs.h"

/* DEFINE LOCAL TYPES HERE */

/* DEFINE LOCAL CONSTANTS HERE */
#define DOUT0_PORT PORTA
#define DOUT0_DDR DDRA
#define DOUT0_PIN (1 << 3)

#define DOUT1_PORT PORTA
#define DOUT1_DDR DDRA
#define DOUT1_PIN (1 << 2)

#define DOUT2_PORT PORTA
#define DOUT2_DDR DDRA
#define DOUT2_PIN (1 << 1)

#define DOUT3_PORT PORTA
#define DOUT3_DDR DDRA
#define DOUT3_PIN (1 << 0)

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: DOUTs_Initialize(..) - initializes pins and sets to inactive state
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char DOUTs_Initialize(void)
{
	// make pins outputs
	DOUT0_DDR |= DOUT0_PIN;
	DOUT1_DDR |= DOUT1_PIN;
	DOUT2_DDR |= DOUT2_PIN;
	DOUT3_DDR |= DOUT3_PIN;
	
	DOUTs_Set(0x00);
	
	return 0;
}

/******************************************************************************
* Description: DOUTs_Set(..) - set outputs
* Input: 	bitmap of outputs, bit0 is OUT0, bit1 is OUT1 and so on
* Output: 	none
* Return:	none
*******************************************************************************/
void DOUTs_Set(uint8_t bitmap)
{
	if(bitmap & (1 << 0))
		DOUT0_PORT |= DOUT0_PIN;
	else
		DOUT0_PORT &= ~DOUT0_PIN;

	if(bitmap & (1 << 1))
		DOUT1_PORT |= DOUT1_PIN;
	else
		DOUT1_PORT &= ~DOUT1_PIN;

	if(bitmap & (1 << 2))
		DOUT2_PORT |= DOUT2_PIN;
	else
		DOUT2_PORT &= ~DOUT2_PIN;

	if(bitmap & (1 << 3))
		DOUT3_PORT |= DOUT3_PIN;
	else
		DOUT3_PORT &= ~DOUT3_PIN;

}

/******************************************************************************
* Description: DOUTs_Set(..) - get states of outputs
* Input: 	none
* Output: 	none
* Return:	bitmap of outputs, bit0 is OUT0, bit1 is OUT1 and so on
*******************************************************************************/
uint8_t DOUTs_Get(void)
{
	uint8_t bitmap = 0;
	
	if(DOUT0_PORT & DOUT0_PIN)
		bitmap |= (1 << 0);
	if(DOUT1_PORT & DOUT1_PIN)
		bitmap |= (1 << 1);
	if(DOUT2_PORT & DOUT2_PIN)
		bitmap |= (1 << 2);
	if(DOUT3_PORT & DOUT3_PIN)
		bitmap |= (1 << 3);
		
	return bitmap;
}
