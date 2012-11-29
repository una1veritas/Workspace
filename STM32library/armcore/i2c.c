/*
 * i2c.c
 *
 *  Created on: 2012/11/03
 *      Author: sin
 */

#include "stm32f4xx_rcc.h"
<<<<<<< HEAD

#include "gpio.h"
#include "delay.h"
#include "i2c.h"

I2CBus Wire1, Wire2, Wire3;
=======

#include "gpio.h"
#include "delay.h"
#include "i2c.h"

typedef enum __I2C_Status {
	NOT_READY = 0xff,
	READY = 0,
	START_ISSUED,
	DST_ADDRESS_SENT,
	SRC_ADDRESS_SENT,
	BYTE_TRANSMITTING,
	BYTE_TRANSMITTED,
	TRANSMISSION_COMPLETED,
	RESTART_ISSUED,
	RECEIVE_BYTE_READY,
	BYTE_RECEIVED,
	BEFORELAST_BYTE_RECEIVED,
	LAST_BYTE_READY,
	RECEIVE_BYTE_COMPLETED,
	RECEIVE_COMPLETED,
//
} I2C_Status;

typedef enum __CommDirection {
	NOT_DEFINED = 0, TRANSMITTER, RECEIVER,
} CommDirection;
>>>>>>> jnosky

//I2C_Status i2c1_status;
//CommDirection i2c1_direction;

<<<<<<< HEAD
boolean i2c_begin(I2CBus * wirex, uint32_t clkspeed) {
//	GPIO_InitTypeDef GPIO_InitStructure;
=======
boolean i2c_begin(uint32_t clkspeed) {
	GPIO_InitTypeDef GPIO_InitStructure;
>>>>>>> jnosky
	I2C_InitTypeDef I2C_InitStructure;

	wirex->I2Cx = I2C1;
	wirex->sda = PB9;
	wirex->scl = PB8;

	/* I2C Periph clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE); //  RCC_APB1PeriphClockCmd(I2C1_RCC, ENABLE);
	/* GPIO Periph clock enable */
	//RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); // PB5 (SMBA), PB6 (SCL), PB9 (SDA)  // RCC_APB2PeriphClockCmd(I2C1_GPIO_RCC, ENABLE);
<<<<<<< HEAD
	GPIOMode(PinPort(wirex->scl), PinBit(wirex->scl), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);
	GPIOMode(PinPort(wirex->sda), PinBit(wirex->sda), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);

	/* Configure I2C pins: SCL and SDA */
	GPIO_PinAFConfig(PinPort(wirex->scl), PinSource(wirex->scl), GPIO_AF_I2C1 );
	GPIO_PinAFConfig(PinPort(wirex->sda), PinSource(wirex->sda), GPIO_AF_I2C1 );
=======
	GPIOMode(PinPort(PB6), PinBit(PB6), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);
	GPIOMode(PinPort(PB9), PinBit(PB9), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);

	/* Configure I2C pins: SCL and SDA */
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_I2C1 );
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource9, GPIO_AF_I2C1 );
>>>>>>> jnosky

	//#if defined (REMAP_I2C1)
//Remap_I2C1_Configuration();
//#endif
	switch (clkspeed) {
	case 100000:
	case 400000:
		break;
	default:
		clkspeed = 100000;
		break;
	}
	/* I2C configuration */
	I2C_InitStructure.I2C_Mode = I2C_Mode_I2C;
	I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2;
	I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;
	I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
	I2C_InitStructure.I2C_ClockSpeed = clkspeed;

	/* Apply I2C configuration after enabling it */
	I2C_Init(wirex->I2Cx, &I2C_InitStructure);
	/* I2C Peripheral Enable */
	I2C_Cmd(wirex->I2Cx, ENABLE);

	wirex->status = NOT_READY;
	wirex->mode = I2C_MODE_NOTDEFINED;

<<<<<<< HEAD
	return true;
}

boolean i2c_transmit(I2CBus * wirex, uint8_t addr, uint8_t * data, uint16_t length) {
	uint16_t i;
	uint16_t wc;
=======
	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;

	return true;
}

boolean i2c_transmit(uint8_t addr, uint8_t * data, uint16_t length) {
	uint16_t i;
	uint32_t resigmillis = millis() + 100;
>>>>>>> jnosky

	wirex->mode = I2C_MODE_MASTERTRANSMITTER;
	//
<<<<<<< HEAD
	if ( !i2c_start(wirex, addr) )
		return false;
=======
	i2c1_status = NOT_READY;
	while (I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = READY;

	/* Send STRAT condition */
	I2C_GenerateSTART(I2C1, ENABLE);
	/* Test on EV5 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = START_ISSUED;

	/* Send address for write */
	I2C_Send7bitAddress(I2C1, addr << 1, I2C_Direction_Transmitter );
	/* Test on EV6 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = DST_ADDRESS_SENT;
>>>>>>> jnosky

	resigmillis = millis() + 100;
	for (i = 0; i < length; i++) {
<<<<<<< HEAD
		I2C_SendData(wirex->I2Cx, data[i]);
		wirex->status = BYTE_TRANSMITTING;
		// Test on EV8 and clear it
		for(wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_BYTE_TRANSMITTED); wc--) {
			if (wc == 0)
				return false;
			delay_us(667);
		}
		wirex->status = BYTE_TRANSMITTED;
=======
		I2C_SendData(I2C1, data[i]);
		i2c1_status = BYTE_TRANSMITTING;
		/* Test on EV8 and clear it */
		while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
			if (millis() > resigmillis)
				return false;
		}
		i2c1_status = BYTE_TRANSMITTED;
>>>>>>> jnosky
	}
	wirex->status = TRANSMISSION_COMPLETED;

<<<<<<< HEAD
	// generate stop condition
	I2C_GenerateSTOP(wirex->I2Cx, ENABLE);
	wirex->status = NOT_READY;
	wirex->mode = I2C_MODE_NOTDEFINED;
=======
	I2C_GenerateSTOP(I2C1, ENABLE);
	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;
>>>>>>> jnosky

	return true;
}

<<<<<<< HEAD
boolean i2c_receive(I2CBus * wirex, uint8_t addr, uint8_t req, uint8_t * recv, uint16_t lim) {
	uint16_t i;
	uint16_t wc;

	wirex->mode = I2C_MODE_MASTERRECEIVER;
	//
	if ( !i2c_start(wirex, addr) )
		return false;
=======
boolean i2c_requestFrom(uint8_t addr, uint8_t req, uint8_t * recv, uint16_t lim) {
	uint16_t i;
	uint32_t resigmillis = millis() + 100;

	i2c1_direction = RECEIVER;
	//
	i2c1_status = NOT_READY;
	/* While the bus is busy */
	while (I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = READY;

	/* Send START condition */
	I2C_GenerateSTART(I2C1, ENABLE);
	/* Test on EV5 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = START_ISSUED;

	/* Send EEPROM address for write */
	I2C_Send7bitAddress(I2C1, addr << 1, I2C_Direction_Transmitter );
	/* Test on EV6 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = DST_ADDRESS_SENT;
>>>>>>> jnosky

	/* Send the EEPROM's internal address to read from: MSB of the address first */
	I2C_SendData(wirex->I2Cx, req);
	wirex->status = BYTE_TRANSMITTING;
	/* Test on EV8 and clear it */
<<<<<<< HEAD
	for (wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_BYTE_TRANSMITTED ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = TRANSMISSION_COMPLETED;
=======
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = TRANSMISSION_COMPLETED;

	//	  I2C_GenerateSTOP(I2C1, ENABLE);
>>>>>>> jnosky

	resigmillis = millis() + 100;
	/* Send STRAT condition a second time */
	I2C_GenerateSTART(wirex->I2Cx, ENABLE);
	/* Test on EV5 and clear it */
<<<<<<< HEAD
	for (wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_MODE_SELECT ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = RESTART_ISSUED;
=======
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = RESTART_ISSUED;
>>>>>>> jnosky

	/* Send EEPROM address for read */
	I2C_Send7bitAddress(wirex->I2Cx, addr << 1, I2C_Direction_Receiver );
	/* Test on EV6 and clear it */
<<<<<<< HEAD
	for (wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = SRC_ADDRESS_SENT;
	for (i = 1; i < lim; i++) {
		wirex->status = RECEIVE_BYTE_READY;
		for(wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_BYTE_RECEIVED ); wc--) {
			if (wc == 0)
				return false;
			delay_us(667);
=======
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = SRC_ADDRESS_SENT;

	for (i = 1; i < lim; i++) {
		i2c1_status = RECEIVE_BYTE_READY;
		while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED )) {
			if (millis() > resigmillis)
				return false;
>>>>>>> jnosky
		}
		/* Read a byte from the EEPROM */
		*recv++ = I2C_ReceiveData(wirex->I2Cx );
		wirex->status = BYTE_RECEIVED;
	}
	wirex->status = BEFORELAST_BYTE_RECEIVED;

	/* Disable Acknowledgement */
	I2C_AcknowledgeConfig(wirex->I2Cx, DISABLE);
	/* Send STOP Condition */
	I2C_GenerateSTOP(wirex->I2Cx, ENABLE);
	wirex->status = LAST_BYTE_READY;

<<<<<<< HEAD
	for(wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_BYTE_RECEIVED ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
=======
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED )) {
		if (millis() > resigmillis)
			return false;
>>>>>>> jnosky
	}
	/* Read a byte from the EEPROM */
	*recv = I2C_ReceiveData(wirex->I2Cx );
	wirex->status = RECEIVE_BYTE_COMPLETED;

	/* Enable Acknowledgement to be ready for another reception */
<<<<<<< HEAD
	I2C_AcknowledgeConfig(wirex->I2Cx, ENABLE);
	wirex->status = NOT_READY;
	wirex->mode = I2C_MODE_NOTDEFINED;

	return true;
}

boolean i2c_start(I2CBus * wirex, uint8_t addr) {
	uint16_t wc;
	//
	wirex->status = NOT_READY;
	for(wc = 5; I2C_GetFlagStatus(wirex->I2Cx, I2C_FLAG_BUSY ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = READY;

	/* Send STRAT condition */
	I2C_GenerateSTART(wirex->I2Cx, ENABLE);
	/* Test on EV5 and clear it */
	for (wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_MODE_SELECT ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = START_ISSUED;

	/* Send address for write */
	I2C_Send7bitAddress(wirex->I2Cx, addr << 1, I2C_Direction_Transmitter);
	/* Test on EV6 and clear it */
	for (wc = 5; !I2C_CheckEvent(wirex->I2Cx, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED ); wc--) {
		if (wc == 0)
			return false;
		delay_us(667);
	}
	wirex->status = DST_ADDRESS_SENT;
	return true;
=======
	I2C_AcknowledgeConfig(I2C1, ENABLE);
	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;

	return true;
}

boolean i2c_send(uint8_t addr, uint8_t * data, uint16_t length) {
return true;
>>>>>>> jnosky
}
