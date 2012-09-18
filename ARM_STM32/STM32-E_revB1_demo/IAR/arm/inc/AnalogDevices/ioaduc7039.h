/***************************************************************************
===============================================															
ADuC7039 HEADER FILE REV 1.4															
===============================================															
***************************************************************************/

#ifndef __IOADI7039_H
#define __IOADI7039_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **                            
 **    ADuC7039 SPECIAL FUNCTION REGISTERS
 **                            
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#endif /* __IAR_SYSTEMS_ICC__ */


/* Common declarations  ****************************************************/

/***************************************************************************
 **
 ** INTERRUPT CONTROLLER
 **
 ***************************************************************************/

__IO_REG32(INTBASE, 0xFFFF0000,__READ_WRITE);
#define IRQSTA INTBASE
__IO_REG32(IRQSIG, 0xFFFF0004,__READ);
__IO_REG32(IRQEN, 0xFFFF0008,__READ_WRITE);
__IO_REG32(IRQCLR, 0xFFFF000C,__WRITE);
__IO_REG32(SWICFG, 0xFFFF0010,__WRITE);
__IO_REG32(FIQSTA, 0xFFFF0100,__READ);
__IO_REG32(FIQSIG, 0xFFFF0104,__READ);
__IO_REG32(FIQEN, 0xFFFF0108,__READ_WRITE);
__IO_REG32(FIQCLR, 0xFFFF010C,__WRITE);

/***************************************************************************
 **
 ** Remap and System Control
 **
 ***************************************************************************/

__IO_REG32(REMAPBASE, 0xFFFF0200,__READ_WRITE);
__IO_REG32(SYSMAP0, 0xFFFF0220,__READ_WRITE);
#define SYSMAP SYSMAP0
__IO_REG32(RSTSTA, 0xFFFF0230,__READ_WRITE);
__IO_REG32(RSTCLR, 0xFFFF0234,__WRITE);
__IO_REG32(SYSCHK, 0xFFFF0244,__READ_WRITE);

/***************************************************************************
 **
 ** 16 bit General Purpose Timer 0
 **
 ***************************************************************************/

__IO_REG32(TIMER0BASE, 0xFFFF0300,__READ_WRITE);
#define T0LD TIMER0BASE
__IO_REG32(T0VAL, 0xFFFF0304,__READ);
__IO_REG32(T0CON, 0xFFFF0308,__READ_WRITE);
__IO_REG32(T0CLRI, 0xFFFF030C,__WRITE);

/***************************************************************************
 **
 ** Wake Up Timer
 **
 ***************************************************************************/

__IO_REG32(TIMER1BASE, 0xFFFF0320,__READ_WRITE);
#define T1LD TIMER1BASE
__IO_REG32(T1VAL, 0xFFFF0324,__READ);
__IO_REG32(T1CON, 0xFFFF0328,__READ_WRITE);
__IO_REG32(T1CLRI, 0xFFFF032C,__WRITE);
__IO_REG32(T1CAP, 0xFFFF0330,__READ);

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/

__IO_REG32(TIMER2BASE, 0xFFFF0340,__READ_WRITE);
#define T2LD TIMER2BASE
__IO_REG32(T2VAL, 0xFFFF0344,__READ);
__IO_REG32(T2CON, 0xFFFF0348,__READ_WRITE);
__IO_REG32(T2CLRI, 0xFFFF034C,__WRITE);

/***************************************************************************
 **
 ** PLL and Oscillator Control
 **
 ***************************************************************************/

__IO_REG32(PLLBASE, 0xFFFF0400,__READ_WRITE);
#define PLLSTA PLLBASE
__IO_REG32(POWKEY0, 0xFFFF0404,__WRITE);
__IO_REG32(POWCON, 0xFFFF0408,__READ_WRITE);
__IO_REG32(POWKEY1, 0xFFFF040C,__WRITE);
__IO_REG32(PLLKEY0, 0xFFFF0410,__WRITE);
__IO_REG32(PLLCON, 0xFFFF0414,__READ_WRITE);
__IO_REG32(PLLKEY1, 0xFFFF0418,__WRITE);
__IO_REG32(OSC0CON, 0xFFFF0440,__READ_WRITE);
#define OSCCON OSC0CON
__IO_REG32(OSC0STA, 0xFFFF0444,__READ);
#define OSCSTA OSC0STA
__IO_REG32(OSC0VAL0, 0xFFFF0448,__READ);
#define OSCVAL0 OSC0VAL0
__IO_REG32(OSC0VAL1, 0xFFFF044C,__READ);
#define OSCVAL1 OSC0VAL1
__IO_REG32(LOCCON, 0xFFFF0480,__READ_WRITE);
__IO_REG32(LOCUSR0, 0xFFFF0484,__READ_WRITE);
__IO_REG32(LOCUSR1, 0xFFFF0488,__READ_WRITE);
__IO_REG32(LOCMAX, 0xFFFF048C,__READ_WRITE);
__IO_REG32(LOCMIN, 0xFFFF0490,__READ_WRITE);
__IO_REG32(LOCSTA, 0xFFFF0494,__READ);
__IO_REG32(LOCVAL0, 0xFFFF0498,__READ);
__IO_REG32(LOCVAL1, 0xFFFF049C,__READ);
__IO_REG32(LOCKEY, 0xFFFF04A0,__WRITE);

/***************************************************************************
 **
 ** ADC interface registers
 **
 ***************************************************************************/

__IO_REG32(ADCBASE, 0xFFFF0500,__READ_WRITE);
#define ADCSTA ADCBASE
__IO_REG32(ADCMSKI, 0xFFFF0504,__READ_WRITE);
__IO_REG32(ADCMDE, 0xFFFF0508,__READ_WRITE);
__IO_REG32(ADC0CON, 0xFFFF050C,__READ_WRITE);
__IO_REG32(ADC1CON, 0xFFFF0510,__READ_WRITE);
__IO_REG32(ADCFLT, 0xFFFF0518,__READ_WRITE);
__IO_REG32(ADCCFG, 0xFFFF051C,__READ_WRITE);
__IO_REG32(ADC0DAT, 0xFFFF0520,__READ);
__IO_REG32(ADC1DAT, 0xFFFF0524,__READ);
__IO_REG32(ADC0OF, 0xFFFF0530,__READ_WRITE);
__IO_REG32(ADC1OF, 0xFFFF0534,__READ_WRITE);
__IO_REG32(ADC2OF, 0xFFFF0538,__READ_WRITE);
__IO_REG32(ADC0GN, 0xFFFF053C,__READ_WRITE);
__IO_REG32(ADC1GN, 0xFFFF0540,__READ_WRITE);
__IO_REG32(ADC2GN, 0xFFFF0544,__READ_WRITE);
__IO_REG32(ADC0RCL, 0xFFFF0548,__READ_WRITE);
__IO_REG32(ADC0RCV, 0xFFFF054C,__READ);
__IO_REG32(ADC0TH, 0xFFFF0550,__READ_WRITE);
__IO_REG32(ADC0ACC, 0xFFFF055C,__READ);

/***************************************************************************
 **
 ** LIN Hardware SYNC Registers.
 **
 ***************************************************************************/

__IO_REG32(LINBASE, 0xFFFF0700,__READ_WRITE);
#define LINCON LINBASE
__IO_REG32(LINCS, 0xFFFF0704,__READ_WRITE);
__IO_REG32(LINBR, 0xFFFF0708,__READ_WRITE);
__IO_REG32(LINBK, 0xFFFF070C,__READ_WRITE);
__IO_REG32(LINSTA, 0xFFFF0710,__READ);
__IO_REG32(LINDAT, 0xFFFF0714,__READ_WRITE);
__IO_REG32(LINLOW, 0xFFFF0718,__READ_WRITE);
__IO_REG32(LINWU, 0xFFFF071C,__READ_WRITE);

/***************************************************************************
 **
 ** High Voltage Interface.
 **
 ***************************************************************************/

__IO_REG32(HVBASE, 0xFFFF0800,__READ_WRITE);
__IO_REG32(HVCON, 0xFFFF0804,__READ_WRITE);
__IO_REG32(HVDAT, 0xFFFF080C,__READ_WRITE);

/***************************************************************************
 **
 ** Serial Port Interface Peripheral
 **
 ***************************************************************************/

__IO_REG32(SPIBASE, 0xFFFF0A00,__READ_WRITE);
#define SPISTA SPIBASE
__IO_REG32(SPIRX, 0xFFFF0A04,__READ);
__IO_REG32(SPITX, 0xFFFF0A08,__WRITE);
__IO_REG32(SPIDIV, 0xFFFF0A0C,__READ_WRITE);
__IO_REG32(SPICON, 0xFFFF0A10,__READ_WRITE);

/***************************************************************************
 **
 ** GPIO + Serial Port Mux (AHB bus)
 **
 ***************************************************************************/

__IO_REG32(GPIOBASE, 0xFFFF0D00,__READ_WRITE);
#define GPCON GPIOBASE
__IO_REG32(GPDAT, 0xFFFF0D10,__READ_WRITE);
__IO_REG32(GPSET, 0xFFFF0D14,__WRITE);
__IO_REG32(GPCLR, 0xFFFF0D18,__WRITE);

/***************************************************************************
 **
 ** Flash Control Interface 64Kbytes (AHB bus)
 **
 ***************************************************************************/

__IO_REG32(FLASHBASE, 0xFFFF0E00,__READ_WRITE);
#define FEESTA FLASHBASE
__IO_REG32(FEEMOD, 0xFFFF0E04,__READ_WRITE);
__IO_REG32(FEECON, 0xFFFF0E08,__READ_WRITE);
__IO_REG32(FEEDAT, 0xFFFF0E0C,__READ_WRITE);
__IO_REG32(FEEADR, 0xFFFF0E10,__READ_WRITE);
__IO_REG32(FEESIG, 0xFFFF0E18,__READ);
__IO_REG32(FEEPRO, 0xFFFF0E1C,__READ_WRITE);
__IO_REG32(FEEHID, 0xFFFF0E20,__READ_WRITE);


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif /* __IOADI7039_H*/
