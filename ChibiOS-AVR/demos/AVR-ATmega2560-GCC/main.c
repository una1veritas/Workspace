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
#include <util/delay.h>
#include "chprintf.h"
/*
 * Application entry point.
 */


static WORKING_AREA(waThread1, 128);
static msg_t Thread1(void *arg) {
  while(1){
      chprintf(&SD1,"ciaoathread: %d\n",OCR0A);
      
      //PORTB^=_BV(PORTB7);
      _delay_ms(100);
      //chThdSleepMilliseconds(100);
  }
  return 0;
}

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
  
  /*
  TCCR0B &= ~((1 << CS02)  | (1 << CS01)  | (1 << CS00));
  TCCR0B |=(0 << CS02)  | (1 << CS01)  | (1 << CS00);
  OCR0A   = 250;*/
  


  sdStart(&SD1, NULL);
    chThdCreateStatic(waThread1, sizeof(waThread1), NORMALPRIO, Thread1, NULL);
  //cli();
  uint8_t a = 0;
  while(1){
      a++;
      //chprintf(&SD1,"ciaoa: %d\n",OCR0A);
      
      //PORTB^=_BV(PORTB7);
      _delay_ms(100);
      //chThdSleepMilliseconds(100);
  }
    


  
  
  
  
  
  
}
