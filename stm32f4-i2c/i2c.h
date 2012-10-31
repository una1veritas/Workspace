/*
 * i2c.h
 *
 *  Created on: 2012/10/30
 *      Author: sin
 */

#ifndef I2C_H_
#define I2C_H_

#define I2C_CLOCK 	100000
#define ST7032I_ADDR 	0x3e

/* Private function prototypes -----------------------------------------------*/
void I2C_Configuration(void);
void ST7032i_Command_Write(uint8_t Data);
void ST7032i_Data_Write(uint8_t Data);


#endif /* I2C_H_ */
