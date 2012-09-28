/***************************************************************************
===============================================													
ADuC7023 HEADER FILE REV 1.3													
===============================================													
***************************************************************************/

#ifndef __IOADI7023_H
#define __IOADI7023_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    ADuC7023 SPECIAL FUNCTION REGISTERS
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

__IO_REG32(IRQSTA, 0xFFFF0000,__READ_WRITE);
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

__IO_REG32(T0BASE, 0xFFFF0300,__READ_WRITE);
#define T0LD T0BASE
__IO_REG32(T0VAL, 0xFFFF0304,__READ);
__IO_REG32(T0CON, 0xFFFF0308,__READ_WRITE);
__IO_REG32(T0CLRI, 0xFFFF030C,__WRITE);

/***************************************************************************
 **
 ** GENERAL PURPOSE TIMER
 **
 ***************************************************************************/

__IO_REG32(T1BASE, 0xFFFF0320,__READ_WRITE);
#define T1LD T1BASE
__IO_REG32(T1VAL, 0xFFFF0324,__READ);
__IO_REG32(T1CON, 0xFFFF0328,__READ_WRITE);
__IO_REG32(T1CLRI, 0xFFFF032C,__WRITE);
__IO_REG32(T1CAP, 0xFFFF0330,__READ_WRITE);

/***************************************************************************
 **
 ** WATCHDOG TIMER
 **
 ***************************************************************************/

__IO_REG32(T3BASE, 0xFFFF0360,__READ_WRITE);
#define T3LD T3BASE
__IO_REG32(T3VAL, 0xFFFF0364,__READ);
__IO_REG32(T3CON, 0xFFFF0368,__READ_WRITE);
__IO_REG32(T3CLRI, 0xFFFF036C,__WRITE);

/***************************************************************************
 **
 ** PLL AND OSCILLATOR CONTROL
 **
 ***************************************************************************/

__IO_REG32(PLLBASE, 0xFFFF0400,__READ_WRITE);
#define PLLSTA PLLBASE
__IO_REG32(POWKEY1, 0xFFFF0404,__WRITE);
__IO_REG32(POWCON, 0xFFFF0408,__READ_WRITE);
__IO_REG32(POWKEY2, 0xFFFF040C,__WRITE);
__IO_REG32(PLLKEY1, 0xFFFF0410,__WRITE);
__IO_REG32(PLLCON, 0xFFFF0414,__READ_WRITE);
__IO_REG32(PLLKEY2, 0xFFFF0418,__WRITE);

/***************************************************************************
 **
 ** GLOBAL PERIPHERAL CONTROL
 **
 ***************************************************************************/

__IO_REG32(ALLBASE, 0xFFFF0434,__READ_WRITE);
#define POWKEY3 ALLBASE
__IO_REG32(POWCON1, 0xFFFF0438,__READ_WRITE);
__IO_REG32(POWKEY4, 0xFFFF043C,__READ_WRITE);

/***************************************************************************
 **
 ** POWER SUPPLY MONITOR
 **
 ***************************************************************************/

__IO_REG32(PSMBASE, 0xFFFF0440,__READ_WRITE);
#define PSMCON PSMBASE
__IO_REG32(CMPCON, 0xFFFF0444,__READ_WRITE);

/***************************************************************************
 **
 ** Band Gap Reference
 **
 ***************************************************************************/

__IO_REG32(REFBASE, 0xFFFF0480,__READ_WRITE);
__IO_REG32(REFCON, 0xFFFF048C,__READ_WRITE);

/***************************************************************************
 **
 ** ADC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(ADCBASE, 0xFFFF0500,__READ_WRITE);
#define ADCCON ADCBASE
__IO_REG32(ADCCP, 0xFFFF0504,__READ_WRITE);
__IO_REG32(ADCCN, 0xFFFF0508,__READ_WRITE);
__IO_REG32(ADCSTA, 0xFFFF050C,__READ);
__IO_REG32(ADCDAT, 0xFFFF0510,__READ);
__IO_REG32(ADCRST, 0xFFFF0514,__READ_WRITE);
__IO_REG32(ADCGN, 0xFFFF0530,__READ_WRITE);
__IO_REG32(ADCOF, 0xFFFF0534,__READ_WRITE);

/***************************************************************************
 **
 ** DAC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(DACBASE, 0xFFFF0600,__READ_WRITE);
#define DAC0CON DACBASE
__IO_REG32(DAC0DAT, 0xFFFF0604,__READ_WRITE);
__IO_REG32(DAC1CON, 0xFFFF0608,__READ_WRITE);
__IO_REG32(DAC1DAT, 0xFFFF060C,__READ_WRITE);
__IO_REG32(DAC2CON, 0xFFFF0610,__READ_WRITE);
__IO_REG32(DAC2DAT, 0xFFFF0614,__READ_WRITE);
__IO_REG32(DAC3CON, 0xFFFF0618,__READ_WRITE);
__IO_REG32(DAC3DAT, 0xFFFF061C,__READ_WRITE);

/***************************************************************************
 **
 ** I2C0 BUS PERIPHERAL DEVICE 0
 **
 ***************************************************************************/

__IO_REG32(I2C0BASE, 0xFFFF0800,__READ_WRITE);
#define I2C0MCON I2C0BASE
__IO_REG32(I2C0MSTA, 0xFFFF0804,__READ);
__IO_REG32(I2C0MRX, 0xFFFF0808,__READ);
__IO_REG32(I2C0MTX, 0xFFFF080C,__WRITE);
__IO_REG32(I2C0MCNT0, 0xFFFF0810,__READ_WRITE);
__IO_REG32(I2C0MCNT1, 0xFFFF0814,__READ);
__IO_REG32(I2C0ADR0, 0xFFFF0818,__READ_WRITE);
__IO_REG32(I2C0DIV, 0xFFFF0824,__READ_WRITE);
__IO_REG32(I2C0SCON, 0xFFFF0828,__READ_WRITE);
__IO_REG32(I2C0SSTA, 0xFFFF082C,__READ_WRITE);
__IO_REG32(I2C0SRX, 0xFFFF0830,__READ_WRITE);
__IO_REG32(I2C0STX, 0xFFFF0834,__READ_WRITE);
__IO_REG32(I2C0ALT, 0xFFFF0838,__READ_WRITE);
__IO_REG32(I2C0ID0, 0xFFFF083C,__READ_WRITE);
__IO_REG32(I2C0ID1, 0xFFFF0840,__READ_WRITE);
__IO_REG32(I2C0ID2, 0xFFFF0844,__READ_WRITE);
__IO_REG32(I2C0ID3, 0xFFFF0848,__READ_WRITE);
__IO_REG32(I2C0FSTA, 0xFFFF084C,__READ_WRITE);

/***************************************************************************
 **
 ** I2C1 BUS PERIPHERAL DEVICE 0
 **
 ***************************************************************************/

__IO_REG32(I2C1BASE, 0xFFFF0900,__READ_WRITE);
#define I2C1MCON I2C1BASE
__IO_REG32(I2C1MSTA, 0xFFFF0904,__READ);
__IO_REG32(I2C1MRX, 0xFFFF0908,__READ);
__IO_REG32(I2C1MTX, 0xFFFF090C,__WRITE);
__IO_REG32(I2C1MCNT0, 0xFFFF0910,__READ_WRITE);
__IO_REG32(I2C1MCNT1, 0xFFFF0914,__READ);
__IO_REG32(I2C1ADR0, 0xFFFF0918,__READ_WRITE);
__IO_REG32(I2C1DIV, 0xFFFF0924,__READ_WRITE);
__IO_REG32(I2C1SCON, 0xFFFF0928,__READ_WRITE);
__IO_REG32(I2C1SSTA, 0xFFFF092C,__READ_WRITE);
__IO_REG32(I2C1SRX, 0xFFFF0930,__READ_WRITE);
__IO_REG32(I2C1STX, 0xFFFF0934,__READ_WRITE);
__IO_REG32(I2C1ALT, 0xFFFF0938,__READ_WRITE);
__IO_REG32(I2C1ID0, 0xFFFF093C,__READ_WRITE);
__IO_REG32(I2C1ID1, 0xFFFF0940,__READ_WRITE);
__IO_REG32(I2C1ID2, 0xFFFF0944,__READ_WRITE);
__IO_REG32(I2C1ID3, 0xFFFF0948,__READ_WRITE);
__IO_REG32(I2C1FSTA, 0xFFFF094C,__READ_WRITE);

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
 ** PROGRAMABLE LOGIC ARRAY
 **
 ***************************************************************************/

__IO_REG32(PLABASE, 0xFFFF0B00,__READ_WRITE);
#define PLAELM0 PLABASE
__IO_REG32(PLAELM1, 0xFFFF0B04,__READ_WRITE);
__IO_REG32(PLAELM2, 0xFFFF0B08,__READ_WRITE);
__IO_REG32(PLAELM3, 0xFFFF0B0C,__READ_WRITE);
__IO_REG32(PLAELM4, 0xFFFF0B10,__READ_WRITE);
__IO_REG32(PLAELM5, 0xFFFF0B14,__READ_WRITE);
__IO_REG32(PLAELM6, 0xFFFF0B18,__READ_WRITE);
__IO_REG32(PLAELM7, 0xFFFF0B1C,__READ_WRITE);
__IO_REG32(PLAELM8, 0xFFFF0B20,__READ_WRITE);
__IO_REG32(PLAELM9, 0xFFFF0B24,__READ_WRITE);
__IO_REG32(PLAELM10, 0xFFFF0B28,__READ_WRITE);
__IO_REG32(PLAELM11, 0xFFFF0B2C,__READ_WRITE);
__IO_REG32(PLAELM12, 0xFFFF0B30,__READ_WRITE);
__IO_REG32(PLAELM13, 0xFFFF0B34,__READ_WRITE);
__IO_REG32(PLAELM14, 0xFFFF0B38,__READ_WRITE);
__IO_REG32(PLAELM15, 0xFFFF0B3C,__READ_WRITE);
__IO_REG32(PLACLK, 0xFFFF0B40,__READ_WRITE);
__IO_REG32(PLAIRQ, 0xFFFF0B44,__READ_WRITE);
__IO_REG32(PLAADC, 0xFFFF0B48,__READ_WRITE);
__IO_REG32(PLADIN, 0xFFFF0B4C,__READ_WRITE);
__IO_REG32(PLADOUT, 0xFFFF0B50,__READ);
__IO_REG32(PLALCK, 0xFFFF0B54,__WRITE);
#define PLADLCK PLALCK

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
 **
 ** GPIO AND SERIAL PORT MUX
 **
 ***************************************************************************/

__IO_REG32(GPIOBASE, 0xFFFFF400,__READ_WRITE);
#define GP0CON GPIOBASE
__IO_REG32(GP1CON, 0xFFFFF404,__READ_WRITE);
__IO_REG32(GP2CON, 0xFFFFF408,__READ_WRITE);
__IO_REG32(GP0DAT, 0xFFFFF420,__READ_WRITE);
__IO_REG32(GP0SET, 0xFFFFF424,__WRITE);
__IO_REG32(GP0CLR, 0xFFFFF428,__WRITE);
__IO_REG32(GP0PAR, 0xFFFFF42C,__READ_WRITE);
__IO_REG32(GP1DAT, 0xFFFFF430,__READ_WRITE);
__IO_REG32(GP1SET, 0xFFFFF434,__WRITE);
__IO_REG32(GP1CLR, 0xFFFFF438,__WRITE);
__IO_REG32(GP1PAR, 0xFFFFF43C,__READ_WRITE);
__IO_REG32(GP2DAT, 0xFFFFF440,__READ_WRITE);
__IO_REG32(GP2SET, 0xFFFFF444,__WRITE);
__IO_REG32(GP2CLR, 0xFFFFF448,__WRITE);
__IO_REG32(GP2PAR, 0xFFFFF44C,__READ_WRITE);

/***************************************************************************
 **
 ** FLASH CONTROL INTERFACE
 **
 ***************************************************************************/

__IO_REG32(FLASHBASE, 0xFFFFF800,__READ_WRITE);
#define FEESTA FLASHBASE
__IO_REG32(FEEMOD, 0xFFFFF804,__READ_WRITE);
__IO_REG32(FEECON, 0xFFFFF808,__READ_WRITE);
__IO_REG32(FEEDAT, 0xFFFFF80C,__READ_WRITE);
__IO_REG32(FEEADR, 0xFFFFF810,__READ_WRITE);
__IO_REG32(FEESIGN, 0xFFFFF818,__READ);
__IO_REG32(FEEPRO, 0xFFFFF81C,__READ_WRITE);
__IO_REG32(FEEHIDE, 0xFFFFF820,__READ_WRITE);


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif // __IOADI7023_H
