/*
 * i2c.h
 *
 *  Created on: 2012/11/03
 *      Author: sin
 */

#ifndef I2C_H_
#define I2C_H_

#ifdef __cplusplus
extern "C" {
#endif

void i2c_begin(uint32_t clkspeed); //I2C_TypeDef * I2Cx, uint32_t clk);
void i2c_transmit(uint8_t addr, uint8_t * data, uint16_t length);
void i2c_send(uint8_t addr, uint8_t * data, uint16_t length);
//void i2c_receive(uint8_t addr, uint8_t * data, uint16_t nlimit);
void i2c_requestFrom(uint8_t addr, uint8_t req, uint8_t * recv, uint16_t lim);

#ifdef __cplusplus
}
#endif

#endif /* I2C_H_ */
