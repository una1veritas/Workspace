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
#include "Stream.h"

class USARTSerial : public Stream {

//	static const int RXBUFF_LEN = 12; // this is the maximum string length of our string in characters
//	static volatile char rx_buffer[RXBUFF_LEN + 1]; // this will hold the recieved string

private:
	uint16 channel;
	USART_TypeDef * USARTx;

public:
	USARTSerial(uint16 ch) {
		channel = ch;
	}

	void puts(volatile char *s);
//	void putch(int ch);
	int  getch(void);

	const USART_TypeDef * port() {
		return USARTx;
	}

public:

	virtual int read() {
		return USARTx->DR;
	}
    /* Set up/tear down */
    void begin(uint32 baud);
    void end(void) {}

    /* I/O */
    virtual int available(void) { return 0; }
    virtual int peek() { return 0; }
    virtual void flush(void) { }
    virtual size_t write(uint8 ch);
    using Print::write;
    /* Pin accessors */
//    int txPin(void) { return this->tx_pin; }
//    int rxPin(void) { return this->rx_pin; }

};

//void USART3_IRQHandler(void);

#endif /* USARTSERIAL_H_ */
