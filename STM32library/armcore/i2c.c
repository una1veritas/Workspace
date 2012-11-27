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

I2CBus Wire1, Wire2, Wire3;

//I2C_Status i2c1_status;
//CommDirection i2c1_direction;

boolean i2c_begin(uint32_t clkspeed) {
//	GPIO_InitTypeDef GPIO_InitStructure;
	I2C_InitTypeDef I2C_InitStructure;

	Wire1.I2Cx = I2C1;
	Wire1.sda = PB9;
	Wire1.scl = PB8;

	/* I2C Periph clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE); //  RCC_APB1PeriphClockCmd(I2C1_RCC, ENABLE);
	/* GPIO Periph clock enable */
	//RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); // PB5 (SMBA), PB6 (SCL), PB9 (SDA)  // RCC_APB2PeriphClockCmd(I2C1_GPIO_RCC, ENABLE);
	GPIOMode(PinPort(Wire1.scl), PinBit(Wire1.scl), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);
	GPIOMode(PinPort(Wire1.sda), PinBit(Wire1.sda), GPIO_Mode_AF, GPIO_Speed_50MHz,
			GPIO_OType_OD, GPIO_PuPd_UP);

	/* Configure I2C pins: SCL and SDA */
	GPIO_PinAFConfig(PinPort(Wire1.scl), PinSource(Wire1.scl), GPIO_AF_I2C1 );
	GPIO_PinAFConfig(PinPort(Wire1.sda), PinSource(Wire1.sda), GPIO_AF_I2C1 );

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
	I2C_Init(Wire1.I2Cx, &I2C_InitStructure);
	/* I2C Peripheral Enable */
	I2C_Cmd(Wire1.I2Cx, ENABLE);

	Wire1.status = NOT_READY;
	Wire1.mode = I2C_MODE_NOTDEFINED;

	return true;
}

boolean i2c_transmit(uint8_t addr, uint8_t * data, uint16_t length) {
	uint16_t i;
	uint32_t wcount;

	Wire1.mode = I2C_MODE_MASTERTRANSMITTER;
	//
	if ( !i2c_start(addr) )
		return false;

	for (i = 0; i < length; i++) {
		wcount = 5;
		I2C_SendData(Wire1.I2Cx, data[i]);
		Wire1.status = BYTE_TRANSMITTING;
		// Test on EV8 and clear it
		while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
			delay_us(667);
			if (wcount == 0)
				return false;
			wcount--;
		}
		Wire1.status = BYTE_TRANSMITTED;
	}
	Wire1.status = TRANSMISSION_COMPLETED;

	// generate stop condition
	I2C_GenerateSTOP(Wire1.I2Cx, ENABLE);
	Wire1.status = NOT_READY;
	Wire1.mode = I2C_MODE_NOTDEFINED;

	return true;
}

boolean i2c_receive(uint8_t addr, uint8_t req, uint8_t * recv, uint16_t lim) {
	uint16_t i;
	uint32_t wcount;

	Wire1.mode = I2C_MODE_MASTERRECEIVER;
	//
	if ( !i2c_start(addr) )
		return false;

	/* Send the EEPROM's internal address to read from: MSB of the address first */
	I2C_SendData(Wire1.I2Cx, req);
	Wire1.status = BYTE_TRANSMITTING;
	/* Test on EV8 and clear it */
	wcount = 5;
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_BYTE_TRANSMITTED )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = TRANSMISSION_COMPLETED;

	/* Send STRAT condition a second time */
	I2C_GenerateSTART(Wire1.I2Cx, ENABLE);
	/* Test on EV5 and clear it */
	wcount = 5;
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_MODE_SELECT )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = RESTART_ISSUED;

	/* Send EEPROM address for read */
	I2C_Send7bitAddress(Wire1.I2Cx, addr << 1, I2C_Direction_Receiver );
	/* Test on EV6 and clear it */
	wcount = 5;
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = SRC_ADDRESS_SENT;
	wcount = 5;
	for (i = 1; i < lim; i++) {
		Wire1.status = RECEIVE_BYTE_READY;
		while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_BYTE_RECEIVED )) {
			delay_us(667);
			if (wcount == 0)
				return false;
			wcount--;
		}
		/* Read a byte from the EEPROM */
		*recv++ = I2C_ReceiveData(Wire1.I2Cx );
		Wire1.status = BYTE_RECEIVED;
	}
	Wire1.status = BEFORELAST_BYTE_RECEIVED;

	/* Disable Acknowledgement */
	I2C_AcknowledgeConfig(Wire1.I2Cx, DISABLE);
	/* Send STOP Condition */
	I2C_GenerateSTOP(Wire1.I2Cx, ENABLE);
	Wire1.status = LAST_BYTE_READY;

	wcount = 5;
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_BYTE_RECEIVED )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	/* Read a byte from the EEPROM */
	*recv = I2C_ReceiveData(Wire1.I2Cx );
	Wire1.status = RECEIVE_BYTE_COMPLETED;

	/* Enable Acknowledgement to be ready for another reception */
	I2C_AcknowledgeConfig(Wire1.I2Cx, ENABLE);
	Wire1.status = NOT_READY;
	Wire1.mode = I2C_MODE_NOTDEFINED;

	return true;
}

boolean i2c_start(uint8_t addr) {
	uint32_t wcount;
	//
	wcount = 5;
	Wire1.status = NOT_READY;
	while (I2C_GetFlagStatus(Wire1.I2Cx, I2C_FLAG_BUSY )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = READY;

	/* Send STRAT condition */
	wcount = 5;
	I2C_GenerateSTART(Wire1.I2Cx, ENABLE);
	/* Test on EV5 and clear it */
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_MODE_SELECT )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = START_ISSUED;

	/* Send address for write */
	wcount = 5;
	I2C_Send7bitAddress(Wire1.I2Cx, addr << 1, I2C_Direction_Transmitter);
	/* Test on EV6 and clear it */
	while (!I2C_CheckEvent(Wire1.I2Cx, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED )) {
		delay_us(667);
		if (wcount == 0)
			return false;
		wcount--;
	}
	Wire1.status = DST_ADDRESS_SENT;

return true;
}
