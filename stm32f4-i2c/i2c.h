/*
 * i2c.h
 *
 *  Created on: 2012/10/29
 *      Author: sin
 */

#ifndef I2C_H_
#define I2C_H_

/*
#define  I2C1_PORT                       GPIOB
#define  I2C1_RCC                        RCC_APB1Periph_I2C1
#define  I2C1_GPIO_RCC                   RCC_APB2Periph_GPIOB
//#define  I2C1_SMBAI_PIN                  0
#define  I2C1_SCL_PIN                    GPIO_Pin_6
#define  I2C1_SDA_PIN                    GPIO_Pin_7
//#define REMAP_I2C1
*/

void I2C_Configuration(void);
void ST7032i_Command_Write(uint8_t Data);
void ST7032i_Data_Write(uint8_t Data);

#endif /* I2C_H_ */
