/********************************************************
 *
 * This file declares ARM ITM macros used with the C-SPY
 * event logging mechanism.
 *
 * Copyright 2011 IAR Systems. All rights reserved.
 *
 * $Revision: 47760 $
 *
 ********************************************************/

#ifndef __ARM_ITM_INCLUDED
#define __ARM_ITM_INCLUDED

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if ! defined( __ARM6M__ )
  #error "ITM channels are only available in Cortex-M devices."
#endif

#pragma language=extended

#include <intrinsics.h>


/* ITM_EVENTx(channel, value)                                          */
/* Write a 8, 16, or 32-bit value to the specified ITM channel.        */
/* For the event logging mechanism channels 1-4 are available.         */
#define ITM_EVENT8(channel, value)   {while ((*((volatile unsigned long *)(0xE0000000+4*(channel)))) == 0) ; *((volatile unsigned char *)(0xE0000000+4*(channel))) = (value);}
#define ITM_EVENT16(channel, value)  {while ((*((volatile unsigned long *)(0xE0000000+4*(channel)))) == 0) ; *((volatile unsigned short *)(0xE0000000+4*(channel))) = (value);}
#define ITM_EVENT32(channel, value)  {while ((*((volatile unsigned long *)(0xE0000000+4*(channel)))) == 0) ; *((volatile unsigned long *)(0xE0000000+4*(channel))) = (value);}


/* ITM_EVENTx_WITH_PC(channel, value)                                                   */
/* Write a 8, 16, or 32-bit value to the specified ITM channel with a corresponding PC. */
/* For the event logging mechanism channels 1-4 are available.                          */
/* The current PC (program counter) is passed to the debugger using channel 5.          */
#define ITM_EVENT8_WITH_PC(channel, value)    {ITM_EVENT32(5, __get_PC()); ITM_EVENT8((channel), (value))}
#define ITM_EVENT16_WITH_PC(channel, value)   {ITM_EVENT32(5, __get_PC()); ITM_EVENT16((channel), (value))}
#define ITM_EVENT32_WITH_PC(channel, value)   {ITM_EVENT32(5, __get_PC()); ITM_EVENT32((channel), (value))}


#endif /* __ARM_ITM_INCLUDED */




