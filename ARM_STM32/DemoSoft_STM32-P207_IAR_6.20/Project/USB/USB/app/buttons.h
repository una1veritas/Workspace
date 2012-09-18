/*************************************************************************
 *
 *    Used with ARM IAR C/C++ Compiler
 *
 *    (c) Copyright IAR Systems 2007
 *
 *    File name      : buttons.h
 *    Description    : Buttons include header
 *
 *    History :
 *    1. Date        : January 11, 2007
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/

#include "includes.h"

#ifndef  __BUTTONS_H
#define  __BUTTONS_H

#define B1_MASK         GPIO_Pin_1
#define B1_PORT         GPIOE

#define TAMPER_MASK     GPIO_Pin_13
#define TAMPER_PORT     GPIOC

#define WAKEUP_MASK     GPIO_Pin_0
#define WAKEUP_PORT     GPIOA

#define JS_RIGHT_MASK   GPIO_Pin_2
#define JS_RIGHT_PORT   GPIOC
#define JS_LEFT_MASK    GPIO_Pin_7
#define JS_LEFT_PORT    GPIOE
#define JS_UP_MASK      GPIO_Pin_3
#define JS_UP_PORT      GPIOC
#define JS_DOWN_MASK    GPIO_Pin_13
#define JS_DOWN_PORT    GPIOE
#define JS_CENTER_MASK  GPIO_Pin_12
#define JS_CENTER_PORT  GPIOE

typedef union _Buttons_t
{
  Int32U Data;
  struct
  {
    Int32U JsUp     : 1;
    Int32U JsDown   : 1;
    Int32U JsRight  : 1;
    Int32U JsLeft   : 1;
    Int32U JsCenter : 1;
    Int32U B1       : 1;
    Int32U Wakeup   : 1;
    Int32U Tamper   : 1;
    Int32U          :24;
  };
} Buttons_t, *pButtons_t;

/*************************************************************************
 * Function Name: ButtonsInit
 * Parameters: none
 * Return: none
 * Description: Init buttons
 *
 *************************************************************************/
void ButtonsInit (void);

/*************************************************************************
 * Function Name: GetButtons
 * Parameters: none
 * Return: Buttons_t
 * Description: Return current buttons states
 *
 *************************************************************************/
Buttons_t GetButtons (void);

#endif  /* __BUTTONS_H */
