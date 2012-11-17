

#ifndef HARDWARESPI_H
#define	HARDWARESPI_H

#ifdef	__cplusplus
extern "C" {
#endif

#include "stm32f4xx.h"

/*
 * Using a single SPI port for the output devices
 */
#define SPI SPI2
#define SPI_PORT                  SPI2
#define SPI_PORT_CLOCK            RCC_APB1Periph_SPI2
#define SPI_PORT_CLOCK_INIT       RCC_APB1PeriphClockCmd

#define SPI_SCK_PIN              GPIO_Pin_13
#define SPI_SCK_GPIO_PORT        GPIOB
#define SPI_SCK_GPIO_CLK         RCC_AHB1Periph_GPIOB
#define SPI_SCK_SOURCE           GPIO_PinSource13
#define SPI_SCK_AF               GPIO_AF_SPI2

#define SPI_MOSI_PIN             GPIO_Pin_15
#define SPI_MOSI_GPIO_PORT       GPIOB
#define SPI_MOSI_GPIO_CLK        RCC_AHB1Periph_GPIOB
#define SPI_MOSI_SOURCE          GPIO_PinSource15
#define SPI_MOSI_AF              GPIO_AF_SPI2


//  extern uint16_t spiConfigured;
  void HardwareSPI_init(void);
  void spiPutByte(uint8_t data);
  void spiPutWord(uint16_t data);
  void spiPutBufferPolled(uint8_t * buffer, uint16_t length);

#ifdef	__cplusplus
}
#endif

#endif	/* HARDWARESPI_H */

