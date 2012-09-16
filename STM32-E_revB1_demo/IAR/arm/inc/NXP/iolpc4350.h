/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC4350
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 49116 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC4350_H
#define __IOLPC4350_H

#if ( __CORE__ == __ARM6M__ )
#include "NXP/iolpc4350_m0.h"
#endif

#if ( __CORE__ == __ARM7EM__ )
#include "NXP/iolpc4350_m4.h"
#endif

#endif
