/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC4320
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 50467 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC4320_H
#define __IOLPC4320_H

#if ( __CORE__ == __ARM6M__ )
#include "NXP/iolpc4320_m0.h"
#endif

#if ( __CORE__ == __ARM7EM__ )
#include "NXP/iolpc4320_m4.h"
#endif

#endif
