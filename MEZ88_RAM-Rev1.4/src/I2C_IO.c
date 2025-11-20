//**********************************************************
//
// The basic I2C functions were created by @etoolsLab369.
// Please refer to the URL below.
//
// https://qiita.com/etoolsLab369/items/65befd8fe1cccd3afc33
// Redesigned by Akihito Honda(@akih_san)
//  https://twitter.com/akih_san
//  https://github.com/akih-san
//
//  Target: PIC18F57QXX/PIC18F47QXX
//  Date 2025.8.11
//
//**********************************************************

#include "mez88.h"
#include <stdio.h>

#if defined _18F47Q83 || defined _18F47Q84
#define _18F47Q8X
#endif

//*************************************************//
//** Init I2C1
//*************************************************//
void I2C1_Init(void)
{
	//Set Clear Buffer Flag
	I2C1STAT1bits.CLRBF=1;		//I2Cbuffer, RXBF,TXBE celar bit
	I2C1CON0bits.MODE=0b100;	//Host mode, 7bits address

	I2C1CON1bits.ACKCNT=1;		//final ACK value [0:ACK 1:NACK]
	I2C1CON1bits.ACKDT=0;		//ACK value(when ACKCNT !=0) [0:ACK 1:NACK]

	I2C1CON2bits.BFRET=0b00;	//8clocks
	I2C1CON2bits.FME=0;			//MFINTOSC/5 (100kHz)
	I2C1CLK=0b0011;				//MFINTOSC 500kHz
	I2C1PIE=0;					//I2C all interrupts disable
	I2C1PIR=0x00;				//clear I2C1 PIR
	I2C1CON0bits.EN=1;			//enable I2C1

#ifdef _18F47Q8X
	I2C1CNTH = 0;
#endif
}

//*************************************************//
//* I2C1 check ACK/NACK at Acknowledge sequence
//* usees: when host sends client address to device,
//* this function can be checked the device exist.
//* return    0 : device exist
//*        0xFF : no device exist
//*************************************************//
static uint8_t chk_ACK(void) {

	// wait 8th SCL clock (Acknowledge sequence start)
	while(!I2C1CON1bits.ACKT);

	// wait 9th SCL clock (end Acknowledge sequence)
	while(I2C1CON1bits.ACKT);

	// check ACK from client device
	if (I2C1CON1bits.ACKSTAT) {
		I2C1STAT1bits.CLRBF = 1;	// clear I2C buffer,RXBF,TXBE
		return 0xFF;
	}
	return 0;
}

//*************************************************//
// Bus Free check
//   0 Indicates an idle bus
//   1 Bus is not idle
//*************************************************//

uint8_t I2C_Check_BusFree(void) {
	while(!(RC3==1 && RC4==1)) {
		__delay_ms(100);
	}
	if (!(RC3==1 && RC4==1)) return(BUS_NOT_FREE);		//Bus not free 
	return(0);
}

//*************************************************//
//  reset I2C
//   - clear error status
//   - clear Interrupt flag
//   - clear I2C buffer,RXBF,TXBE
//*************************************************//

static void rst_I2C(void) {
	I2C1ERR = 0;				// clear error
	I2C1PIR = 0;				// clear Interrupt Flag Register
	I2C1STAT1bits.CLRBF = 1;	// clear I2C buffer,RXBF,TXBE
}

#if 0
//*************************************************//
//** I2C1: Send 1 byte
//*************************************************//
uint8_t I2C_ByteWrite(uint8_t client_addr, uint8_t data)
{
	uint8_t res;

	rst_I2C();

	I2C1CON2bits.ABD=0;			// address buffer mode
	I2C1CON0bits.RSEN=0;		// disable restart

	I2C1ADB1 = client_addr & 0b11111110;	//set Client address
#ifdef _18F47Q8X
	I2C1CNTL = 1;							//set count 1 (1 byte)
#else
	I2C1CNT = 1;							//set count 1 (1 byte)
#endif
	I2C1TXB = data;							//set tx data
	
	I2C1CON0bits.S=1;			//Start!

	// start sending client address

	if (chk_ACK()) res = NACK_DETECT;
	else res = 0;
	
	while(I2C1STAT0bits.MMA);		//check Host mode active 0:not active 1:active

	return res;
}
#endif
//*************************************************//
//** I2C1 receive 1 byte (no send word address)
//*************************************************//
uint8_t I2C_ByteRead(uint8_t client_addr, uint8_t *buf)
{
	uint8_t res;

	rst_I2C();

	I2C1CON2bits.ABD = 0;			// address buffer mode
	I2C1CON0bits.RSEN=0;			//disable restart
	I2C1ADB1 = client_addr | 1;		//set Client address & R/W bit = 1
#ifdef _18F47Q8X
	I2C1CNTL = 1;							//set count 1 (1 byte)
#else
	I2C1CNT = 1;							//set count 1 (1 byte)
#endif
	
	I2C1CON0bits.S = 1;				//Start!

	// start sending client address

	if (chk_ACK()) res = NACK_DETECT;
	else {
		while(!I2C1STAT1bits.RXBF);		//wait until rx buffer full
		*buf=I2C1RXB;					//get read data
		res = 0;
	}
	while(I2C1STAT0bits.MMA);		//check Host mode active 0:not Active 1:active
	//Set Clear Buffer Flag

	return res;
}
#if 0
//***************************************************//
//** I2C1 receive 1 byte With Specify second Address
//***************************************************//
uint8_t I2C_ByteRead_WSA(uint8_t client_addr,uint8_t address, uint8_t *buf)
{
	uint8_t res;

	res = NACK_DETECT;
	rst_I2C();

	I2C1CON2bits.ABD=0;			// address buffer mode
	I2C1CON0bits.RSEN=1;		//Set restart!

	I2C1ADB1 = client_addr & 0b11111110;	//set Client address (R/W=0)
#ifdef _18F47Q8X
	I2C1CNTL = 1;							//set count 1 (send _address to client)
#else
	I2C1CNT = 1;							//set count 1 (send _address to client)
#endif
	I2C1TXB = address;						//set tx data

	I2C1CON0bits.S=1;				//Start!

	// start sending (client_addr) to device

	if (!chk_ACK()) {
		while(!I2C1STAT1bits.TXBE);		// I2C1TXB data is sent to shift register
										// and count down I2C1CNT to zero

		// sending next data (address) to client device
		if (!chk_ACK()) {
			while(!I2C1CON0bits.MDR);		//wait until MDR=1(address tx complete)

			// next receive process is prepareing
	
			I2C1ADB1 = client_addr|0x01;		//set Client address & R/W bit = 1
#ifdef _18F47Q8X
			I2C1CNTL = 1;						//set read count
#else
			I2C1CNT = 1;						//set read count
#endif

			I2C1CON0bits.RSEN=0;			//reset restart
			I2C1CON0bits.S=1;				//Restart!

			// Rstart! sending (client_addr) to device

			if (!chk_ACK()) {
				while(!I2C1STAT1bits.RXBF);		//wait until read end
				*buf = I2C1RXB;					//get read data
				res = 0;
    		}
    	}
    }
	while(I2C1STAT0bits.MMA);		//check Host mode active 0:not Active 1:active
	return res;
}
#endif
//*************************************************//
//** I2C1 Send n bytes
//*************************************************//
uint8_t write_I2C(uint8_t client_addr, uint8_t wordAdd, uint8_t count, uint8_t *buff) {

	uint8_t cnt;

	cnt = 0xff;
	rst_I2C();

	I2C1CON2bits.ABD=0;						// address buffer mode
	I2C1CON0bits.RSEN=0;					//disable restart
	I2C1ADB1 = client_addr & 0b11111110;	//set Client address
#ifdef _18F47Q8X
	I2C1CNTL = count+1;						//set wordAdd + cont
#else
	I2C1CNT = count+1;						//set wordAdd + cont
#endif
	I2C1TXB = wordAdd;						//set first data : word address(second address)

	I2C1CON0bits.S=1;			//Start!

	// start sending (client_addr) to device

	if (!chk_ACK()) {
		cnt = 0;
		while(!I2C1PIRbits.CNTIF) {
			if (I2C1STAT1bits.TXBE) {		//check tx buffer empty
				I2C1TXB = *buff++;			//set next tx data
				cnt++;
				if (chk_ACK()) {
					cnt = 0xff;		// error
					break;
				}
			}
		}
	}
	while(I2C1STAT0bits.MMA);		//check Host mode active 0:not active 1:active
	return cnt;
}

//*************************************************//
//** I2C1 read n bytes
//*************************************************//
// count = number of read data
uint8_t read_I2C(uint8_t client_addr, uint8_t wordAdd, uint8_t count, uint8_t *buff) {

	uint8_t cnt;
	
	cnt = 0xff;
	rst_I2C();

	I2C1CON2bits.ABD=0;						// address buffer mode
	I2C1CON0bits.RSEN=1;					//Set restart!
	I2C1ADB1 = client_addr & 0b11111110;		//set Client address
#ifdef _18F47Q8X
	I2C1CNTL = 1;							//set count 1 (send _address to client)
#else
	I2C1CNT = 1;							//set count 1 (send _address to client)
#endif
	I2C1TXB = wordAdd;						//set next tx data ( set wordAdd )

	I2C1CON0bits.S=1;				//Start!

	// sending (client_addr) to client device

	if (!chk_ACK()) {
		while(!I2C1STAT1bits.TXBE);		//I2C1TXB data is sent to shift register

		// sending next data (wordAdd) to client device
		if (!chk_ACK()) {
			while(!I2C1CON0bits.MDR);	//wait until MDR=1 (become Resteart condition)

			// next receive process is prepareing
			I2C1ADB1 = client_addr|0x01;	//set Client address & R/W bit = 1
#ifdef _18F47Q8X
			I2C1CNTL = count;				//set read count
#else
			I2C1CNT = count;				//set read count
#endif
			I2C1CON0bits.RSEN=0;			//reset restart
			I2C1PIRbits.CNTIF = 0;			// clear cont flag
			I2C1CON0bits.S=1;

			// Rstart! sending (client_addr) to device
			if (!chk_ACK()) {
				cnt = 0;
				while(!I2C1PIRbits.CNTIF) {
					if (I2C1STAT1bits.RXBF) {	//check rx buffer full
						*buff++ = I2C1RXB;		//get read data
						cnt++;
					}
				}
			}
		}
	}
	while(I2C1STAT0bits.MMA);		//check Host mode active 0:not Active 1:active
	return cnt;
}

//
// check DS1307
//
uint16_t chk_i2cdev(void) {

	uint8_t c;
	uint16_t setup_val;
	setup_val = 0;

	// check I2C bus exist
	if ( I2C_Check_BusFree() ) {
		// I2C bus not free
		printf("I2C bus is not exist.\n\r");
	}
	else {
		printf("DS1307 RTC module ");
		// check RTC module
		if (I2C_ByteRead( DS1307, &c) == NACK_DETECT) {
			// RTC module is not exist
			printf("is not exist.\r\n");
		}
		else {
			// RTC module 
			printf("exists.\r\n");
			if ( !cnv_rtc_tim() ) setup_val = 1;
		}
	}
	return(setup_val);
}
