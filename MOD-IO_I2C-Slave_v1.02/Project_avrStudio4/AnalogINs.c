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
#include <avr/io.h>
#include "DigitalINs.h"

/* DEFINE LOCAL TYPES HERE */

/* DEFINE LOCAL CONSTANTS HERE */
#define AIN0_DDR DDRA
#define AIN0_PIN (1 << 7)
#define AIN0_MUX 0x07

#define AIN1_DDR DDRA
#define AIN1_PIN (1 << 6)
#define AIN1_MUX 0x06

#define AIN2_DDR DDRA
#define AIN2_PIN (1 << 5)
#define AIN2_MUX 0x05

#define AIN3_DDR DDRA
#define AIN3_PIN (1 << 4)
#define AIN3_MUX 0x04

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: AINs_Initialize(..) - initializes pins
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char AINs_Initialize(void)
{
	// make pins inputs
	AIN0_DDR &= ~AIN0_PIN;
	AIN1_DDR &= ~AIN1_PIN;
	AIN2_DDR &= ~AIN2_PIN;
	AIN3_DDR &= ~AIN3_PIN;
	
	// configure ADC
	ADMUX = AIN0_MUX & 0x1F;
	ADCSRA = (1<<ADEN) | (1<<ADPS2) | (1<<ADPS1);
	
	return 0;
}

/******************************************************************************
* Description: AINs_Get(..) - sample an analogue channel
* Input: 	channel - the pin to sample
* Output: 	none
* Return:	value in steps of the input
*******************************************************************************/
uint16_t AINs_Get(uint8_t channel)
{
	uint16_t adcVal = 0;
	
	switch(channel) {
		case 0:	ADMUX = (ADMUX & 0xE0) | (AIN0_MUX & 0x1F); break;
		case 1:	ADMUX = (ADMUX & 0xE0) | (AIN1_MUX & 0x1F); break;
		case 2:	ADMUX = (ADMUX & 0xE0) | (AIN2_MUX & 0x1F); break;
		case 3:	ADMUX = (ADMUX & 0xE0) | (AIN3_MUX & 0x1F); break;
		default: ADMUX = (ADMUX & 0xE0) | 0x1F; break;		
	}
	
	ADCSRA |= (1<<ADSC);
	
	// wait conversion to finish
	while(ADCSRA & (1<<ADSC))
		;
		
	adcVal = ADCW;
		
	return adcVal;
}


