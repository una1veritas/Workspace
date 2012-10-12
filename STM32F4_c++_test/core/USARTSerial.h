/*
 * USARTSerial.h
 *
 *  Created on: 2012/10/10
 *      Author: sin
 */

#ifndef USARTSERIAL_H_
#define USARTSERIAL_H_

#include <stm32f4xx.h>
#include <misc.h>			 // I recommend you have a look at these in the ST firmware folder
#include <stm32f4xx_gpio.h>
#include <stm32f4xx_rcc.h>
#include <stm32f4xx_usart.h> // under Libraries/STM32F4xx_StdPeriph_Driver/inc and src

#include "armcore.h"
#include "gpio_digital.h"

class USARTSerial {

//	static const int RXBUFF_LEN = 12; // this is the maximum string length of our string in characters
//	static volatile char rx_buffer[RXBUFF_LEN + 1]; // this will hold the recieved string

private:
	void init(uint32 baudrate);
	void puts(USART_TypeDef* USARTx, volatile char *s);
	void putch(USART_TypeDef* USARTx, int ch);
	int getch(void);

public:
    /* Set up/tear down */
    void begin(uint32 baud);
    void end(void);

    /* I/O */
    uint32 available(void);
    uint8 read(void);
    void flush(void);
    virtual void write(unsigned char);
//    using Print::write;

    /* Pin accessors */
//    int txPin(void) { return this->tx_pin; }
//    int rxPin(void) { return this->rx_pin; }

};

void USART1_IRQHandler(void);

#endif /* USARTSERIAL_H_ */
