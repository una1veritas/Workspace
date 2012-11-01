/*
 * i2c.h
 *
 *  Created on: 2012/11/01
 *      Author: sin
 */

#ifndef I2C_H_
#define I2C_H_

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_i2c.h>

/**
  * @brief  I2C port definitions
  */
#define IOE_I2C                          I2C1
#define IOE_I2C_CLK                      RCC_APB1Periph_I2C1
#define IOE_I2C_SCL_PIN                  GPIO_Pin_6
#define IOE_I2C_SCL_GPIO_PORT            GPIOB
#define IOE_I2C_SCL_GPIO_CLK             RCC_AHB1Periph_GPIOB
#define IOE_I2C_SCL_SOURCE               GPIO_PinSource6
#define IOE_I2C_SCL_AF                   GPIO_AF_I2C1
#define IOE_I2C_SDA_PIN                  GPIO_Pin_9
#define IOE_I2C_SDA_GPIO_PORT            GPIOB
#define IOE_I2C_SDA_GPIO_CLK             RCC_AHB1Periph_GPIOB
#define IOE_I2C_SDA_SOURCE               GPIO_PinSource9
#define IOE_I2C_SDA_AF                   GPIO_AF_I2C1
#define IOE_I2C_DR                       ((uint32_t)0x40005410)

/* I2C clock speed configuration (in Hz)
  WARNING:
   Make sure that this define is not already declared in other files (ie.
  stm324xg_eval.h file). It can be used in parallel by other modules. */
#ifndef I2C_SPEED
 #define I2C_SPEED                        100000
#endif /* I2C_SPEED */

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
}IOE_Status_TypDef;


static void IOE_I2C_Config(void);
static void IOE_GPIO_Config(void);

#endif /* I2C_H_ */
