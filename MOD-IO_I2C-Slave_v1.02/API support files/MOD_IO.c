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
#include "Compiler.h"
#include "GenericTypeDefs.h"
#include "MOD_IO.h"

/* DEFINE LOCAL TYPES HERE */
typedef enum _I2C_COMMAND_CODES {
	I2C_NO_COMMAND = 0x00,
	I2C_SET_OUTPUTS = 0x10,
	I2C_GET_DINPUTS = 0x20,
	I2C_GET_AIN_0 = 0x30,
	I2C_GET_AIN_1,
	I2C_GET_AIN_2,
	I2C_GET_AIN_3,
	I2C_SET_SLAVE_ADDR = 0xF0
} I2C_COMMAND_CODES;

/* DEFINE LOCAL CONSTANTS HERE */
#define MODIO_SDA_TRIS TRISBbits.TRISB5
#define MODIO_SDA_LAT LATBbits.LATB5
#define MODIO_SDA_PORT PORTBbits.RB5

#define MODIO_SCL_TRIS TRISBbits.TRISB4
#define MODIO_SCL_LAT LATBbits.LATB4
#define MODIO_SCL_PORT PORTBbits.RB4

#define MODIO_INT_TRIS TRISCbits.TRISC7
#define MODIO_INT_PORT PORTCbits.RC7

#define MODIO_SLAVE_ADDR (0x01 << 1)

#define ACK 0
#define NACK 1

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */
static void local_I2C_Delay(void);
static void local_I2C_Start(void);
static void local_I2C_Stop(void);
static BYTE local_I2C_ReadByte(char ack);
static char local_I2C_WriteByte(BYTE data); // returns ack state

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: MODIO_Initialize(..) - initializes pins and registers of the MOD_IO
* Input: 	none
* Output: 	none
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
char MODIO_Initialize(void)
{
	char result = 0;
	BYTE id;

	// the idea is to toggle pins as inputs/outputs to emulate I2C open drain mode of operation!
	MODIO_SDA_TRIS = 1;
	MODIO_SCL_TRIS = 1;

	MODIO_SDA_LAT = 0;
	MODIO_SCL_LAT = 0;

	MODIO_INT_TRIS = 1;

	local_I2C_Delay();
	if( !(MODIO_SCL_PORT && MODIO_SDA_PORT) )
		return -1;

	do {
		// check the ID for match
		local_I2C_Start();
		result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x00);
		local_I2C_Stop();

		// is there such a chip on the line at all?
		if(result) break;

	} while(0);
	
	return (result ? -1 : 0);
}

/******************************************************************************
* Description: MODIO_ReadDINs(..) - reads the digital inputs of the board
* Input: 	none
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_ReadDINs(BYTE *data)
{
	char result = 0;
	BYTE bitmap;

	do {
		local_I2C_Start();
		result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x00);
		result |= local_I2C_WriteByte(I2C_GET_DINPUTS);
		local_I2C_Stop();
		if(result) break;

		local_I2C_Start();
		result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x01);
		bitmap = local_I2C_ReadByte(NACK);
		local_I2C_Stop();
		if(result) break;
		
		*data = bitmap;
	} while(0);

	return result;
}

/******************************************************************************
* Description: MODIO_ReadAIN(..) - reads an analog input of the board
* Input: 	channel - analog inpout to scan
* Output: 	data - value of the input level
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_ReadAIN(WORD *data, BYTE channel)
{
	char result = 0;
	WORD_VAL val;

	do {
		local_I2C_Start();
		result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x00);
		result |= local_I2C_WriteByte(I2C_GET_AIN_0 + (channel & 0x03));
		local_I2C_Stop();
		if(result) break;

		local_I2C_Start();
		result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x01);
		val.byte.LB = local_I2C_ReadByte(ACK);
		val.byte.HB = local_I2C_ReadByte(NACK);
		local_I2C_Stop();
		if(result) break;

		*data = val.Val;
	} while(0);

	return result;
}

/******************************************************************************
* Description: MODIO_WriteDOUTs(..) - sets stated of digital outputs
* Input: 	bitmap - value to send to the outputs
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_WriteDOUTs(BYTE bitmap)
{
	char result = 0;

	// configure device here
	local_I2C_Start();
	result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x00);
	result |= local_I2C_WriteByte(I2C_SET_OUTPUTS);
	result |= local_I2C_WriteByte(bitmap); // CTRL_REG1
	local_I2C_Stop();

	return result;
}

/******************************************************************************
* Description: MODIO_UpdateSlvAddress(..) - updates the slave address of the board
* Input: 	newaddr - new slave address of the board
* Output: 	none
* Return:       0 on success, -1 on error
*******************************************************************************/
char MODIO_UpdateSlvAddress(BYTE newaddr)
{
	char result = 0;

	// configure device here
	local_I2C_Start();
	result |= local_I2C_WriteByte(MODIO_SLAVE_ADDR | 0x00);
	result |= local_I2C_WriteByte(I2C_SET_SLAVE_ADDR);
	result |= local_I2C_WriteByte(newaddr); // CTRL_REG1
	local_I2C_Stop();

	return result;
}



/* local functions */

static void local_I2C_Delay(void)
{
	UINT d = 10;
	while(d--) {
		Nop();
	}	
}	

static void local_I2C_Start(void)
{
	MODIO_SDA_TRIS = 0;
	local_I2C_Delay();
	MODIO_SCL_TRIS = 0;
	local_I2C_Delay();
	
}

static void local_I2C_Stop(void)
{
	MODIO_SDA_TRIS = 0;
	local_I2C_Delay();
	MODIO_SCL_TRIS = 1;
	local_I2C_Delay();
	MODIO_SDA_TRIS = 1;
	local_I2C_Delay();
}

static BYTE local_I2C_ReadByte(char ack)
{
	BYTE data = 0;
	char i;
	
	MODIO_SDA_TRIS = 1;
	for(i = 0; i < 8; i++) {
		local_I2C_Delay();
		MODIO_SCL_TRIS = 1;
		local_I2C_Delay();
		data |= MODIO_SDA_PORT & 0x01;
		if(i != 7)
			data <<= 1;
		MODIO_SCL_TRIS = 0;
	}
	
	// read the ack
	local_I2C_Delay();
	MODIO_SDA_TRIS = ack;
	local_I2C_Delay();
	MODIO_SCL_TRIS = 1;
	local_I2C_Delay();
	MODIO_SCL_TRIS = 0;
	local_I2C_Delay();
	
	return data;
}

// returns ack state, 0 means acknowledged
static char local_I2C_WriteByte(BYTE data)
{
	char i;

	// send the 8 bits
	for(i = 0; i < 8; i++) {
		MODIO_SDA_TRIS = (data & 0x80) ? 1 : 0;
		data <<= 1;
		local_I2C_Delay();
		MODIO_SCL_TRIS = 1;
		local_I2C_Delay();
		MODIO_SCL_TRIS = 0;
	}
	
	// read the ack
	MODIO_SDA_TRIS = 1;
	local_I2C_Delay();
	MODIO_SCL_TRIS = 1;
	local_I2C_Delay();
	i = MODIO_SDA_PORT;
	MODIO_SCL_TRIS = 0;
	local_I2C_Delay();
	
	return i;
}


