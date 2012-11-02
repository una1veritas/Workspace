/*
 * i2c.h
 *
 *  Created on: 2012/11/01
 *      Author: sin
 */

#ifndef I2C_H_
#define I2C_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_i2c.h>

/**
  * @brief  I2C1 port definitions
  */
//#define IOE_I2C                          I2C1
#define I2C1_CLK                      RCC_APB1Periph_I2C1
#define I2C1_SCL_PIN                  GPIO_Pin_6
#define I2C1_SCL_GPIO_PORT            GPIOB
#define I2C1_SCL_GPIO_CLK             RCC_AHB1Periph_GPIOB
#define I2C1_SCL_SOURCE               GPIO_PinSource6
#define I2C1_SCL_AF                   GPIO_AF_I2C1
#define I2C1_SDA_PIN                  GPIO_Pin_9
#define I2C1_SDA_GPIO_PORT            GPIOB
#define I2C1_SDA_GPIO_CLK             RCC_AHB1Periph_GPIOB
#define I2C1_SDA_SOURCE               GPIO_PinSource9
#define I2C1_SDA_AF                   GPIO_AF_I2C1
#define I2C1_DR                       ((uint32_t)0x40005410)

/* I2C clock speed configuration (in Hz)
  WARNING:
   Make sure that this define is not already declared in other files (ie.
  stm324xg_eval.h file). It can be used in parallel by other modules. */
#ifndef I2C1_SPEED
 #define I2C1_SPEED                        100000
#endif /* I2C1_SPEED */

/**
  * @brief  IO Expander Interrupt line on EXTI
  */
#define IOE_IT_PIN                       GPIO_Pin_2
#define IOE_IT_GPIO_PORT                 GPIOI
#define IOE_IT_GPIO_CLK                  RCC_AHB1Periph_GPIOI
#define IOE_IT_EXTI_PORT_SOURCE          EXTI_PortSourceGPIOI
#define IOE_IT_EXTI_PIN_SOURCE           EXTI_PinSource2
#define IOE_IT_EXTI_LINE                 EXTI_Line2
#define IOE_IT_EXTI_IRQn                 EXTI2_IRQn

/**
  * @brief  IO_Expander Error codes
  */
typedef enum
{
  IOE_OK = 0,
  IOE_FAILURE,
  IOE_TIMEOUT,
  PARAM_ERROR,
  IOE1_NOT_OPERATIONAL,
  IOE2_NOT_OPERATIONAL
} IOE_Status_TypDef;


void I2C1_Config(void);
void I2C1_GPIO_Config(void);

uint8_t IOE_Reset(uint8_t DeviceAddr);

uint8_t I2C_WriteDeviceRegister(uint8_t DeviceAddr, uint8_t RegisterAddr, uint8_t RegisterValue);
#ifdef __cplusplus
}
#endif

#endif /* I2C_H_ */
