/*All ADuC706x library code provided by ADI, including this file, is provided
	as is without warranty of any kind, either expressed or implied.
 	You assume any and all risk from the use of this code.  It is the
 	responsibility of the person integrating this code into an application
 	to ensure that the resulting application performs as required and is safe.
*/
/***************************************************************************
===============================================												
ADuC7060 HEADER FILE REV 1.1												
===============================================												
***************************************************************************/

#ifndef __IOADI7060_H
#define __IOADI7060_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    ADuC7060 SPECIAL FUNCTION REGISTERS
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
__IO_REG32(IRQBASE, 0xFFFF0014,__READ_WRITE);
__IO_REG32(IRQVEC, 0xFFFF001C,__READ);
__IO_REG32(IRQP0, 0xFFFF0020,__READ_WRITE);
__IO_REG32(IRQP1, 0xFFFF0024,__READ_WRITE);
__IO_REG32(IRQP2, 0xFFFF0028,__READ_WRITE);
__IO_REG32(IRQP3, 0xFFFF002C,__READ_WRITE);
__IO_REG32(IRQCONN, 0xFFFF0030,__READ_WRITE);
__IO_REG32(IRQCONE, 0xFFFF0034,__READ_WRITE);
__IO_REG32(IRQCLRE, 0xFFFF0038,__READ_WRITE);
__IO_REG32(IRQSTAN, 0xFFFF003C,__READ_WRITE);
__IO_REG32(FIQSTA, 0xFFFF0100,__READ);
__IO_REG32(FIQSIG, 0xFFFF0104,__READ);
__IO_REG32(FIQEN, 0xFFFF0108,__READ_WRITE);
__IO_REG32(FIQCLR, 0xFFFF010C,__WRITE);
__IO_REG32(FIQVEC, 0xFFFF011C,__READ);
__IO_REG32(FIQSTAN, 0xFFFF013C,__READ);

/***************************************************************************
 **
 ** REMAP AND SYSTEM CONTROL
 **
 ***************************************************************************/

__IO_REG32(REMAPBASE, 0xFFFF0200,__READ_WRITE);
__IO_REG32(REMAP, 0xFFFF0220,__READ_WRITE);
__IO_REG32(RSTSTA, 0xFFFF0230,__READ_WRITE);
__IO_REG32(RSTCLR, 0xFFFF0234,__WRITE);

/***************************************************************************
 **
 ** TIMER 0
 **
 ***************************************************************************/

__IO_REG32(T0BASE, 0xFFFF0320,__READ_WRITE);
#define T0LD T0BASE
__IO_REG32(T0VAL, 0xFFFF0324,__READ);
__IO_REG32(T0CON, 0xFFFF0328,__READ_WRITE);
__IO_REG32(T0CLRI, 0xFFFF032C,__WRITE);
__IO_REG32(T0CAP, 0xFFFF0330,__READ);

/***************************************************************************
 **
 ** WAKE UP TIMER
 **
 ***************************************************************************/

__IO_REG32(T1BASE, 0xFFFF0340,__READ_WRITE);
#define T1LD T1BASE
__IO_REG32(T1VAL, 0xFFFF0344,__READ);
__IO_REG32(T1CON, 0xFFFF0348,__READ_WRITE);
__IO_REG32(T1CLRI, 0xFFFF034C,__WRITE);

/***************************************************************************
 **
 ** WATCHDOG TIMER
 **
 ***************************************************************************/

__IO_REG32(T2BASE, 0xFFFF0360,__READ_WRITE);
#define T2LD T2BASE
__IO_REG32(T2VAL, 0xFFFF0364,__READ);
__IO_REG32(T2CON, 0xFFFF0368,__READ_WRITE);
__IO_REG32(T2CLRI, 0xFFFF036C,__WRITE);
__IO_REG32(T2RCFG, 0xFFFF0370,__READ_WRITE);

/***************************************************************************
 **
 ** TIMER 3
 **
 ***************************************************************************/

__IO_REG32(T3BASE, 0xFFFF0380,__READ_WRITE);
#define T3LD T3BASE
__IO_REG32(T3VAL, 0xFFFF0384,__READ);
__IO_REG32(T3CON, 0xFFFF0388,__READ_WRITE);
__IO_REG32(T3CLRI, 0xFFFF038C,__WRITE);
__IO_REG32(T3CAP, 0xFFFF0390,__READ);

/***************************************************************************
 **
 ** PLL AND OSCILLATOR CONTROL
 **
 ***************************************************************************/

__IO_REG32(PLLBASE, 0xFFFF0400,__READ_WRITE);
#define PLLSTA PLLBASE
__IO_REG32(POWKEY1, 0xFFFF0404,__WRITE);
__IO_REG32(POWCON0, 0xFFFF0408,__READ_WRITE);
__IO_REG32(POWKEY2, 0xFFFF040C,__WRITE);
__IO_REG32(PLLKEY1, 0xFFFF0410,__WRITE);
__IO_REG32(PLLCON, 0xFFFF0414,__READ_WRITE);
__IO_REG32(PLLKEY2, 0xFFFF0418,__WRITE);
__IO_REG32(PLLKEY3, 0xFFFF041c,__WRITE);
__IO_REG32(PLLTST, 0xFFFF0420,__READ_WRITE);
__IO_REG32(PLLKEY4, 0xFFFF0424,__WRITE);
__IO_REG32(POWKEY3, 0xFFFF0434,__WRITE);
__IO_REG32(POWCON1, 0xFFFF0438,__READ_WRITE);
__IO_REG32(POWKEY4, 0xFFFF043C,__WRITE);
__IO_REG32(GP0KEY1, 0xFFFF0464,__READ_WRITE);
__IO_REG32(GP0CON1, 0xFFFF0468,__READ_WRITE);
__IO_REG32(GP0KEY2, 0xFFFF046C,__READ_WRITE);

/***************************************************************************
 **
 ** ADC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(ADCBASE, 0xFFFF0500,__READ_WRITE);
#define ADCSTA ADCBASE
__IO_REG32(ADCMSKI, 0xFFFF0504,__READ_WRITE);
__IO_REG32(ADCMDE, 0xFFFF0508,__READ_WRITE);
__IO_REG32(ADC0CON, 0xFFFF050C,__READ_WRITE);
__IO_REG32(ADC1CON, 0xFFFF0510,__READ_WRITE);
__IO_REG32(ADCFLT, 0xFFFF0514,__READ_WRITE);
__IO_REG32(ADCCFG, 0xFFFF0518,__READ_WRITE);
__IO_REG32(ADC0DAT, 0xFFFF051C,__READ_WRITE);
__IO_REG32(ADC1DAT, 0xFFFF0520,__READ_WRITE);
__IO_REG32(ADC0OF, 0xFFFF0524,__READ_WRITE);
__IO_REG32(ADC1OF, 0xFFFF0528,__READ_WRITE);
__IO_REG32(ADC0GN, 0xFFFF052C,__READ_WRITE);
__IO_REG32(ADC1GN, 0xFFFF0530,__READ_WRITE);
__IO_REG32(ADCORCR, 0xFFFF0534,__READ_WRITE);
__IO_REG32(ADCORCV, 0xFFFF0538,__READ);
__IO_REG32(ADCOTH, 0xFFFF053C,__READ_WRITE);
__IO_REG32(ADCOTHC, 0xFFFF0540,__READ_WRITE);
__IO_REG32(ADCOTHV, 0xFFFF0544,__READ_WRITE);
__IO_REG32(ADCOACC, 0xFFFF0548,__READ_WRITE);
__IO_REG32(ADCOATH, 0xFFFF054C,__READ_WRITE);
__IO_REG32(IEXCON, 0xFFFF0570,__READ_WRITE);

/***************************************************************************
 **
 ** DAC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(DACBASE, 0xFFFF0600,__READ_WRITE);
#define DACCON DACBASE
__IO_REG32(DACDAT, 0xFFFF0604,__READ_WRITE);

/***************************************************************************
 **
 ** 450 COMPATIABLE UART CORE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(UARTBASE, 0xFFFF0700,__READ_WRITE);
#define COMTX UARTBASE
#define COMRX UARTBASE
#define COMDIV0 UARTBASE
__IO_REG32(COMIEN0, 0xFFFF0704,__READ_WRITE);
#define COMDIV1 COMIEN0
__IO_REG32(COMIID0, 0xFFFF0708,__READ);
__IO_REG32(COMCON0, 0xFFFF070C,__READ_WRITE);
__IO_REG32(COMCON1, 0xFFFF0710,__READ_WRITE);
__IO_REG32(COMSTA0, 0xFFFF0714,__READ);
__IO_REG32(COMSTA1, 0xFFFF0718,__READ);
__IO_REG32(COMSCR, 0xFFFF071C,__READ_WRITE);
__IO_REG32(COMDIV2, 0xFFFF072C,__READ_WRITE);

/***************************************************************************
 **
 ** I2C BUS PERIPHERAL DEVICE
 **
 ***************************************************************************/

__IO_REG32(I2CBASE, 0xFFFF0900,__READ_WRITE);
#define I2CMCON I2CBASE
__IO_REG32(I2CMSTA, 0xFFFF0904,__READ);
__IO_REG32(I2CMRX, 0xFFFF0908,__READ);
__IO_REG32(I2CMTX, 0xFFFF090C,__WRITE);
__IO_REG32(I2CMCNT0, 0xFFFF0910,__READ_WRITE);
__IO_REG32(I2CMCNT1, 0xFFFF0914,__READ);
__IO_REG32(I2CADR0, 0xFFFF0918,__READ_WRITE);
__IO_REG32(I2CADR1, 0xFFFF091C,__READ_WRITE);
__IO_REG32(I2CREP, 0xFFFF0920,__READ_WRITE);
__IO_REG32(I2CDIV, 0xFFFF0924,__READ_WRITE);
__IO_REG32(I2CSCON, 0xFFFF0928,__READ_WRITE);
__IO_REG32(I2CSSTA, 0xFFFF092C,__READ_WRITE);
__IO_REG32(I2CSRX, 0xFFFF0930,__READ_WRITE);
__IO_REG32(I2CSTX, 0xFFFF0934,__READ_WRITE);
__IO_REG32(I2CALT, 0xFFFF0938,__READ_WRITE);
__IO_REG32(I2CID0, 0xFFFF093C,__READ_WRITE);
__IO_REG32(I2CID1, 0xFFFF0940,__READ_WRITE);
__IO_REG32(I2CID2, 0xFFFF0944,__READ_WRITE);
__IO_REG32(I2CID3, 0xFFFF0948,__READ_WRITE);
__IO_REG32(I2CFSTA, 0xFFFF094C,__READ_WRITE);
__IO_REG32(I2CRCON, 0xFFFF0950,__READ_WRITE);

/***************************************************************************
 **
 ** SERIAL PORT INTERFACE PERIPHERAL
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
 ** GPIO AND SERIAL PORT MUX
 **
 ***************************************************************************/

__IO_REG32(GPIOBASE, 0xFFFF0D00,__READ_WRITE);
#define GP0CON0 GPIOBASE
__IO_REG32(GP1CON, 0xFFFF0D04,__READ_WRITE);
__IO_REG32(GP2CON, 0xFFFF0D08,__READ_WRITE);
__IO_REG32(GP0DAT, 0xFFFF0D20,__READ_WRITE);
__IO_REG32(GP0SET, 0xFFFF0D24,__WRITE);
__IO_REG32(GP0CLR, 0xFFFF0D28,__WRITE);
__IO_REG32(GP0PAR, 0xFFFF0D2C,__READ_WRITE);
__IO_REG32(GP1DAT, 0xFFFF0D30,__READ_WRITE);
__IO_REG32(GP1SET, 0xFFFF0D34,__WRITE);
__IO_REG32(GP1CLR, 0xFFFF0D38,__WRITE);
__IO_REG32(GP1PAR, 0xFFFF0D3C,__READ_WRITE);
__IO_REG32(GP2DAT, 0xFFFF0D40,__READ_WRITE);
__IO_REG32(GP2SET, 0xFFFF0D44,__WRITE);
__IO_REG32(GP2CLR, 0xFFFF0D48,__WRITE);
__IO_REG32(GP2PAR, 0xFFFF0D4C,__READ_WRITE);

/***************************************************************************
 **
 ** FLASH CONTROL INTERFACE
 **
 ***************************************************************************/

__IO_REG32(FLASHBASE, 0xFFFF0E00,__READ_WRITE);
#define FEESTA FLASHBASE
__IO_REG32(FEEMOD, 0xFFFF0E04,__READ_WRITE);
__IO_REG32(FEECON, 0xFFFF0E08,__READ_WRITE);
__IO_REG32(FEEDAT, 0xFFFF0E0C,__READ_WRITE);
__IO_REG32(FEEADR, 0xFFFF0E10,__READ_WRITE);
__IO_REG32(FEESIG, 0xFFFF0E18,__READ);
#define FEESIGN FEESIG
__IO_REG32(FEEPRO, 0xFFFF0E1C,__READ_WRITE);
__IO_REG32(FEEHID, 0xFFFF0E20,__READ_WRITE);
#define FEEHIDE FEEHID

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/

__IO_REG32(PWMBASE, 0xFFFF0F80,__READ_WRITE);
#define PWMCON PWMBASE
__IO_REG32(PWM0COM0, 0xFFFF0F84,__READ_WRITE);
__IO_REG32(PWM0COM1, 0xFFFF0F88,__READ_WRITE);
__IO_REG32(PWM0COM2, 0xFFFF0F8C,__READ_WRITE);
__IO_REG32(PWM0LEN, 0xFFFF0F90,__READ_WRITE);
__IO_REG32(PWM1COM0, 0xFFFF0F94,__READ_WRITE);
__IO_REG32(PWM1COM1, 0xFFFF0F98,__READ_WRITE);
__IO_REG32(PWM1COM2, 0xFFFF0F9C,__READ_WRITE);
__IO_REG32(PWM1LEN, 0xFFFF0FA0,__READ_WRITE);
__IO_REG32(PWM2COM0, 0xFFFF0FA4,__READ_WRITE);
__IO_REG32(PWM2COM1, 0xFFFF0FA8,__READ_WRITE);
__IO_REG32(PWM2COM2, 0xFFFF0FAC,__READ_WRITE);
__IO_REG32(PWM2LEN, 0xFFFF0FB0,__READ_WRITE);
__IO_REG32(PWMCLRI, 0xFFFF0FB8,__READ_WRITE);


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif  /* __IOADI7060_H*/
