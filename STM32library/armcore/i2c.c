/*
 * i2c.c
 *
 *  Created on: 2012/11/03
 *      Author: sin
 */

#include "stm32f4xx_rcc.h"

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

I2C_Status i2c1_status;
CommDirection i2c1_direction;

boolean i2c_begin(uint32_t clkspeed) {
	GPIO_InitTypeDef GPIO_InitStructure;
	I2C_InitTypeDef I2C_InitStructure;

	/* I2C Periph clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE); //  RCC_APB1PeriphClockCmd(I2C1_RCC, ENABLE);
	/* GPIO Periph clock enable */
	//RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); // PB5 (SMBA), PB6 (SCL), PB9 (SDA)  // RCC_APB2PeriphClockCmd(I2C1_GPIO_RCC, ENABLE);
	GPIOMode(PinPort(PB6), PinBit(PB6), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);
	GPIOMode(PinPort(PB9), PinBit(PB9), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);

	/* Configure I2C pins: SCL and SDA */
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_I2C1 );
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource9, GPIO_AF_I2C1 );

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
	I2C_Init(I2C1, &I2C_InitStructure);
	/* I2C Peripheral Enable */
	I2C_Cmd(I2C1, ENABLE);

	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;

	return true;
}

boolean i2c_transmit(uint8_t addr, uint8_t * data, uint16_t length) {
	uint16_t i;
	uint32_t resigmillis = millis() + 100;

	i2c1_direction = TRANSMITTER;
	//
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

	resigmillis = millis() + 100;
	for (i = 0; i < length; i++) {
		I2C_SendData(I2C1, data[i]);
		i2c1_status = BYTE_TRANSMITTING;
		/* Test on EV8 and clear it */
		while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
			if (millis() > resigmillis)
				return false;
		}
		i2c1_status = BYTE_TRANSMITTED;
	}
	i2c1_status = TRANSMISSION_COMPLETED;

	I2C_GenerateSTOP(I2C1, ENABLE);
	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;

	return true;
}

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

	/* Send the EEPROM's internal address to read from: MSB of the address first */
	I2C_SendData(I2C1, req);
	i2c1_status = BYTE_TRANSMITTING;
	/* Test on EV8 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = TRANSMISSION_COMPLETED;

	//	  I2C_GenerateSTOP(I2C1, ENABLE);

	resigmillis = millis() + 100;
	/* Send STRAT condition a second time */
	I2C_GenerateSTART(I2C1, ENABLE);
	/* Test on EV5 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT )) {
		if (millis() > resigmillis)
			return false;
	}
	i2c1_status = RESTART_ISSUED;

	/* Send EEPROM address for read */
	I2C_Send7bitAddress(I2C1, addr << 1, I2C_Direction_Receiver );
	/* Test on EV6 and clear it */
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
		}
		/* Read a byte from the EEPROM */
		*recv++ = I2C_ReceiveData(I2C1 );
		i2c1_status = BYTE_RECEIVED;
	}
	i2c1_status = BEFORELAST_BYTE_RECEIVED;

	/* Disable Acknowledgement */
	I2C_AcknowledgeConfig(I2C1, DISABLE);
	/* Send STOP Condition */
	I2C_GenerateSTOP(I2C1, ENABLE);
	i2c1_status = LAST_BYTE_READY;

	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED )) {
		if (millis() > resigmillis)
			return false;
	}
	/* Read a byte from the EEPROM */
	*recv = I2C_ReceiveData(I2C1 );
	i2c1_status = RECEIVE_BYTE_COMPLETED;

	/* Enable Acknowledgement to be ready for another reception */
	I2C_AcknowledgeConfig(I2C1, ENABLE);
	i2c1_status = NOT_READY;
	i2c1_direction = NOT_DEFINED;

	return true;
}

boolean i2c_send(uint8_t addr, uint8_t * data, uint16_t length) {
return true;
}
