/*
    ChibiOS/RT - Copyright (C) 2006,2007,2008,2009,2010,
                 2011,2012 Giovanni Di Sirio.

    This file is part of ChibiOS/RT.

    ChibiOS/RT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    ChibiOS/RT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ch.h"
#include "hal.h"

#include "chprintf.h"

#include <util/delay.h>

static WORKING_AREA(waThread1, 128);
static msg_t Thread1(void *arg) {

  while (TRUE) {
    palTogglePad(IOPORT2, 0);
     chThdSleepMilliseconds(200);
  }
  return 0;
}

/*
 * Application entry point.
 */
int main(void) {

  /*
   * System initializations.
   * - HAL initialization, this also initializes the configured device drivers
   *   and performs the board-specific initializations.
   * - Kernel initialization, the main() function becomes a thread and the
   *   RTOS is active.
   */


  halInit();
  chSysInit();


   palSetPadMode(IOPORT2, 0, PAL_MODE_OUTPUT_PUSHPULL);
   PORTB |= 1;
   DDRD |= (1<<DDD7);

  static PWMConfig pwmcfg = {
    10000, /* 10KHz PWM clock frequency. */
    10000, /* PWM period 1S (in ticks). */
	NULL,
	{
	{PWM_OUTPUT_ACTIVE_HIGH, NULL},
	{PWM_OUTPUT_ACTIVE_HIGH, NULL},
	{PWM_OUTPUT_ACTIVE_HIGH, NULL},
	},

    };
      
  
    palSetGroupMode(IOPORT3, 0b00000111, 0, PAL_MODE_OUTPUT_PUSHPULL);

  pwmStart(&PWMD1,&pwmcfg);
  //pwmStart(&PWMD2,&pwmcfg);

    sdStart(&SD1, NULL);

  DDRB|= _BV(DDB7);
  

  chThdCreateStatic(waThread1, sizeof(waThread1), NORMALPRIO, Thread1, NULL);
  uint16_t val = 0;
  while(1)
  {
      //pwmEnableChannel(&PWMD1, 2, val);
      //OCR1C=0x40;
      //pwmEnableChannel(&PWMD2, 0, val); 
      val = (val + 2000) % 10000;
      
      pwmEnableChannel(&PWMD1, 2, val+1);
      
      uint8_t oc = TCCR2A;
      uint8_t oc2 = TCCR2B;
      //chprintf(&SD1,"TCCR1A: %x, TCCR1B: %x, val1: %d, val2 %d\n",TCCR1A,TCCR1B,OCR1CL,OCR1CH);
      chThdSleepMilliseconds(500);

    
  }

  
  
  
  
  
  
}
