/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : BSP.h
 *    Description : some helpers
 *
 *    History :
 *    1. Date        : 09 November 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/
#include <avr/io.h>

#ifndef BSP_H
#define BSP_H

/* LED */
#define InitLED() DDRB |= 1 << 3;
#define SetLED(st) \
			if(st) \
				PORTB |= 1 << 3; \
			else \
				PORTB &= ~(1 << 3);
#define ToggleLED() PORTB ^= 1 << 3;

/* BUTTON */
#define InitBtn() DDRD &= ~(1<<2);
#define BtnPressed() (!(PIND & (1<<2)))

#endif // BSP_H

