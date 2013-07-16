/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : I2CAddress.h
 *    Description : gets/sets the I2C address of the device from/to EEPROM
 *
 *    History :
 *    1. Date        : 07 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef I2C_ADDRESS_H
#define I2C_ADDRESS_H

/******************************************************************************
* Description: I2C_Address_Initialize(..) - initializes EEPROM and reads the default address
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char I2C_Address_Initialize(void);

/******************************************************************************
* Description: I2C_Address_Get(..) - gets the current I2C address of the devide
* Input: 	none
* Output: 	none
* Return:	the current I2C address 
*******************************************************************************/
uint8_t I2C_Address_Get(void);

/******************************************************************************
* Description: I2C_Address_Set(..) - sets a new address of the I2C of the device
*		Stores it to internal EEPROM
* Input: 	none
* Output: 	none
* Return:	0 if successfully updated, -1 if error occurred 
*******************************************************************************/
char I2C_Address_Set(uint8_t addr);

/******************************************************************************
* Description: I2C_Address_SetDefault(..) - sets a the default I2C address
*		Stores it to internal EEPROM
* Input: 	none
* Output: 	none
* Return:	0 if successfully updated, -1 if error occurred 
*******************************************************************************/
char I2C_Address_SetDefault(void);

#endif // I2C_ADDRESS_H
