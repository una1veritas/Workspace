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

//#include "Print.h"

class USARTSerial {

//	static const int RXBUFF_LEN = 12; // this is the maximum string length of our string in characters
//	static volatile char rx_buffer[RXBUFF_LEN + 1]; // this will hold the recieved string

private:
	uint16 channel;
	USART_TypeDef * USARTx;

public:
	USARTSerial(uint16 ch) {
		channel = ch;
	}

//	void puts(volatile char *s);
//	void putch(int ch);
	int  getch(void);

	const USART_TypeDef * port() {
		return USARTx;
	}

    size_t printNumber(uint32 num, uint8 base);
    size_t printFloat(double fp, uint8 prec);

public:

	virtual int read() {
		return USARTx->DR;
	}
    /* Set up/tear down */
    void begin(uint32 baud);

    /* I/O */
    size_t write(const uint8 ch);
    size_t write(const uint8 * a, const uint16 len);

    size_t print(const char ch) { return write((const char) ch); }
    size_t print(const char * s);
    size_t print(uint32 num, uint8 base = DEC) { return printNumber(num, base); }
    size_t print(int32 num);
    size_t print(double fp) { return printFloat(fp, 2); }
    size_t print(double fp, uint8 prec) { return printFloat(fp, prec); }

    size_t println() { return write('\n'); }
    size_t println(const char * s) { return print(s) + println(); }


    /* Pin accessors */
//    int txPin(void) { return this->tx_pin; }
//    int rxPin(void) { return this->rx_pin; }

};

//void USART3_IRQHandler(void);

#endif /* USARTSERIAL_H_ */
