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

#ifndef _BOARD_H_
#define _BOARD_H_

/*
 * Setup for a generic SPC563M proto board.
 */

/*
 * Board identifiers.
 */
#define BOARD_GENERIC_SPC563M
#define BOARD_NAME "Generic SPC563M"

/*
 * Board frequencies.
 */
#if !defined(EXTCLK)
#define EXTCLK          8000000
#endif

/*
 * I/O definitions.
 */
#define GPIO_SCI_A_TX   89
#define GPIO_SCI_A_RX   90

#define GPIO_BUTTON1    179
#define GPIO_BUTTON2    181
#define GPIO_BUTTON3    183
#define GPIO_BUTTON4    187

#define GPIO_LED1       188
#define GPIO_LED2       189
#define GPIO_LED3       190
#define GPIO_LED4       191

#if !defined(_FROM_ASM_)
#ifdef __cplusplus
extern "C" {
#endif
  void boardInit(void);
#ifdef __cplusplus
}
#endif
#endif /* _FROM_ASM_ */

#endif /* _BOARD_H_ */
