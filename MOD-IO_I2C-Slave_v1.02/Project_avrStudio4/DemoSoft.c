/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : main.c
 *    Description : main file of the project
 *
 *    History :
 *    1. Date        : 07 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/
#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/fuse.h>
#include "BSP.h"
#include "DigitalINs.h"
#include "AnalogINs.h"
#include "DigitalOUTs.h"
#include "I2CAddress.h"
#include "I2C_FSM.h"

FUSES = 
{
	.low = 0xCF,
	.high = 0x89
};

/* DEFINE LOCAL TYPES HERE */

/* DEFINE LOCAL CONSTANTS HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */
static void InitializeSystem(void);

/* DEFINE FUNCTIONS HERE */

int main(void)
{
	uint32_t delay;
	
	InitializeSystem();

	// check for button pressed at startup to reset slave address to default
	if(BtnPressed()) {
		delay = 30000;
		while(--delay) {
			// button released before timeout occurred
			if(!BtnPressed())
				break;

			// indicate activity
			if(!(delay % 1000))
				ToggleLED();
		}

		// button pressed long enough, reset to default
		if(!delay) {
			I2C_Address_SetDefault();
			I2C_FSM_Initialize(); // load the new address
		}
	}

	delay = 20000;
		
    while(1)
    {
		I2C_FSM_Refresh();

		// toggle LED to indicate activity
		if( (delay--) == 0 ) {
			delay = 200000;
			ToggleLED();
		}			
    }
}

static void InitializeSystem(void)
{
	char result = 0;
	
	// initialize not used pins to inputs
	DDRD &= ~((1<<0) | (1<<1));
	DDRB &= ~((1<<4) | (1<<5) | (1<<6) | (1<<7));
	
	InitLED();
	
	result |= DINs_Initialize();
	result |= AINs_Initialize();
	result |= DOUTs_Initialize();
	result |= I2C_Address_Initialize();
	result |= I2C_FSM_Initialize();
	
	// check for error
	if(result) {
		uint32_t delay;
		
		while(1)
		{
			delay = 10000;
			while(delay--);
		
			ToggleLED();
		}		
	}
	
	sei(); /* enable interrupts */
}

ISR(BADISR_vect)
{
	// user code here
	while(1);
}
