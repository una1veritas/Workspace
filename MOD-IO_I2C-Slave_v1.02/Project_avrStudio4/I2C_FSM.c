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
#include <string.h>
#include <avr/interrupt.h>
#include "BSP.h"
#include "DigitalINs.h"
#include "AnalogINs.h"
#include "DigitalOUTs.h"
#include "I2CAddress.h"
#include "I2C_FSM.h"

/* DEFINE LOCAL TYPES HERE */
typedef enum _I2C_FSM_STATES {
	I2C_FSM_IDLE = 0,
	I2C_FSM_WAIT_COMMAND,
	I2C_FSM_WAIT_DOUTS,
	I2C_FSM_WAIT_ADDRUPDATE,
	I2C_FSM_WAIT_DUMMY,
	I2C_FSM_ADDRESSED_R,
	I2C_FSM_SEND_DINS,
	I2C_FSM_SEND_AIN0_MSB,
	I2C_FSM_SEND_AIN0_LSB,	
} I2C_FSM_STATES;

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
#define TX_BUF_LENGTH 2

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */
static I2C_COMMAND_CODES i2c_command;
static char readCommandRequested;
static uint32_t addressUpdatedIndTimeout;
static uint8_t txBuf[TX_BUF_LENGTH];
static uint8_t txBufIndex;

static I2C_FSM_STATES i2c_state;

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: I2C_FSM_Initialize(..) - initializes I2C interface and FSM
* Input: 	none
* Output: 	none
* Return:	0 if successfully initialized, -1 if error occurred 
*******************************************************************************/
char I2C_FSM_Initialize(void)
{
	// configure the I2C module
	TWAR = I2C_Address_Get() << 1;
	TWSR = 0x00;
	TWCR = (1<<TWEN) | (1<<TWEA) | (1<<TWIE);
	
	i2c_command = I2C_NO_COMMAND;
	i2c_state = I2C_FSM_IDLE;

	readCommandRequested = 0;
	addressUpdatedIndTimeout = 0;
	
	return 0;
}

/******************************************************************************
* Description: I2C_FSM_Refresh(..) - call periodically to run FSM
* Input: 	none
* Output: 	none
* Return:	none
*******************************************************************************/
void I2C_FSM_Refresh(void)
{
	if(readCommandRequested) {
		switch(i2c_command) {
			case I2C_GET_DINPUTS:
				txBuf[0] = DINs_Get();
				break;
			case I2C_GET_AIN_0:
			case I2C_GET_AIN_1:
			case I2C_GET_AIN_2:
			case I2C_GET_AIN_3:
				*((uint16_t*)txBuf) = AINs_Get(i2c_command - I2C_GET_AIN_0);	
				break;
			default:
				memset(txBuf, 0x00, TX_BUF_LENGTH);
				break;
		}
		
		readCommandRequested = 0;
		txBufIndex = 0;
	}

	// lit LED constatntly for the period of timeout
	if(addressUpdatedIndTimeout) {
		addressUpdatedIndTimeout--;
		SetLED(1);
	}
}


ISR(TWI_vect)
{
	switch(TWSR) {
		case 0xA8: // Own SLA+R has been received; ACK has been returned
		case 0xB8: // Data byte in TWDR has been transmitted; ACK has been received
			if(txBufIndex < TX_BUF_LENGTH)
				TWDR = txBuf[txBufIndex++];
			else
				TWDR = 0x00; // send dummy data
			break;
			
		case 0xC0: // Data byte in TWDR has been transmitted; NOT ACK has been received
			i2c_state = I2C_FSM_IDLE;
			break;
		
		case 0x60: // Own SLA+W has been received; ACK has been returned
			i2c_state = I2C_FSM_WAIT_COMMAND;
			break;
		
		case 0x80: // Previously addressed with own SLA+W; data has been received; ACK has been returned
			switch(i2c_state) {
				case I2C_FSM_WAIT_COMMAND:
					i2c_command = TWDR;
					switch(i2c_command) {
						case I2C_SET_OUTPUTS:
							i2c_state = I2C_FSM_WAIT_DOUTS;
							break;
						case I2C_SET_SLAVE_ADDR:
							i2c_state = I2C_FSM_WAIT_ADDRUPDATE;
							break;
						default:
							i2c_state = I2C_FSM_WAIT_DUMMY;
							break;
					}
					
					readCommandRequested = 1;
					break;
					
				case I2C_FSM_WAIT_DOUTS:
					DOUTs_Set(TWDR);
					i2c_state = I2C_FSM_WAIT_DUMMY;
					break;
					
				case I2C_FSM_WAIT_ADDRUPDATE:
					// only update address if button is pressed
					if( BtnPressed() ) {
						if( I2C_Address_Set(TWDR) == 0) {
							// immediately update own address
							TWAR = I2C_Address_Get() << 1;

							addressUpdatedIndTimeout = 500000;
						}
					}
					
					i2c_state = I2C_FSM_WAIT_DUMMY;
					break;
					
				default:
					break;
			}
			break;
			
		case 0x88: // Previously addressed with own SLA+W; data has been received; NOT ACK has been returned
			i2c_state = I2C_FSM_IDLE;
			break;
		
		default:
			i2c_state = I2C_FSM_IDLE;
			break;
	}

	// clear interrupt flag
	TWCR |= (1<<TWINT);
	
}
