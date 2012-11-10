/*
 * i2c.c
 *
 *  Created on: 2012/11/03
 *      Author: sin
 */

#include "stm32f4xx_rcc.h"
#include "i2c.h"

void i2c_begin(uint32_t clkspeed) {
	GPIO_InitTypeDef GPIO_InitStructure;
	I2C_InitTypeDef I2C_InitStructure;

	/* I2C Periph clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE); //  RCC_APB1PeriphClockCmd(I2C1_RCC, ENABLE);
	/* GPIO Periph clock enable */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE); // PB5 (SMBA), PB6 (SCL), PB9 (SDA)  // RCC_APB2PeriphClockCmd(I2C1_GPIO_RCC, ENABLE);

	/* Configure I2C pins: SCL and SDA */
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_I2C1 );
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource9, GPIO_AF_I2C1 );

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

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
}

void i2c_transmit(uint8_t addr, uint8_t * data, uint16_t length) {
	uint16_t i;

	/* Send STRAT condition */
	I2C_GenerateSTART(I2C1, ENABLE);
	/* Test on EV5 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT ))
		;
	/* Send EEPROM address for write */
	I2C_Send7bitAddress(I2C1, addr << 1, I2C_Direction_Transmitter );
	/* Test on EV6 and clear it */
	while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED ))
		;

	for (i = 0; i < length; i++) {
		I2C_SendData(I2C1, data[i]);
		/* Test on EV8 and clear it */
		while (!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED ))
			;
	}
	I2C_GenerateSTOP(I2C1, ENABLE);
}

