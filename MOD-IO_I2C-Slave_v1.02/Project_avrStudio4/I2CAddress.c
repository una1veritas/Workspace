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
#include <avr/eeprom.h>
#include "I2CAddress.h"

/* DEFINE LOCAL TYPES HERE */

/* DEFINE LOCAL CONSTANTS HERE */
#define I2C_ADDRESS_EEADDRESS 0x00

// this is the value that has to be shifted to the left!
// this means that you can set addresses from 1 - 127
#define I2C_DEFAULT_ADDRESS 0x58

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */
static uint8_t localI2CAddress;

uint8_t ee_I2CAddress __attribute__((section(".eeprom"))) = I2C_DEFAULT_ADDRESS;

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: I2C_Address_Initialize(..) - initializes EEPROM and reads the default address
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char I2C_Address_Initialize(void)
{
	char result = 0;
	
	localI2CAddress = I2C_DEFAULT_ADDRESS;
	
	// read address stored in EEPROM
	eeprom_busy_wait();
	localI2CAddress = eeprom_read_byte(I2C_ADDRESS_EEADDRESS);
	
	return result;
}

/******************************************************************************
* Description: I2C_Address_Get(..) - gets the current I2C address of the devide
* Input: 	none
* Output: 	none
* Return:	the current I2C address 
*******************************************************************************/
uint8_t I2C_Address_Get(void)
{
	return localI2CAddress;
}

/******************************************************************************
* Description: I2C_Address_Set(..) - sets a new address of the I2C of the device
*		Stores it to internal EEPROM
* Input: 	none
* Output: 	none
* Return:	0 if successfully updated, -1 if error occurred 
*******************************************************************************/
char I2C_Address_Set(uint8_t addr)
{
	char result = 0;

	// update address here	
	eeprom_busy_wait();
	cli();
	eeprom_write_byte(I2C_ADDRESS_EEADDRESS, addr);
	sei();

	// verify that data is correct
	eeprom_busy_wait();
	result = (addr == eeprom_read_byte(I2C_ADDRESS_EEADDRESS)) ? 0 : -1;

	if(!result)
		localI2CAddress = addr;
			
	return result;	
}

/******************************************************************************
* Description: I2C_Address_SetDefault(..) - sets a the default I2C address
*		Stores it to internal EEPROM
* Input: 	none
* Output: 	none
* Return:	0 if successfully updated, -1 if error occurred 
*******************************************************************************/
char I2C_Address_SetDefault(void)
{
	return I2C_Address_Set(I2C_DEFAULT_ADDRESS);
}
