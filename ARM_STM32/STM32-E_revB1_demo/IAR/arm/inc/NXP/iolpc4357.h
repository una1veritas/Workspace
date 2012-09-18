/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC4357
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50467 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC4357_H
#define __IOLPC4357_H

#if ( __CORE__ == __ARM6M__ )
#include "NXP/iolpc4357_m0.h"
#endif

#if ( __CORE__ == __ARM7EM__ )
#include "NXP/iolpc4357_m4.h"
#endif

#endif
