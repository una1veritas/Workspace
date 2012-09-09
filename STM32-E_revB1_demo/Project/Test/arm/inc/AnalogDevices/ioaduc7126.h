/***************************************************************************
===============================================																																																																																				
ADuC7126 HEADER FILE REV 1.6 09th November 2010																																																																																				
===============================================																																																																																				
***************************************************************************/

#ifndef __IOADI7126_H
#define __IOADI7126_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **                            
 **    ADuC7126 SPECIAL FUNCTION REGISTERS
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

__IO_REG32(IRQSTA, 0xFFFF0000,__READ);
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

__IO_REG32(REMAP, 0xFFFF0220,__READ_WRITE);
__IO_REG32(RSTSTA, 0xFFFF0230,__READ_WRITE);
__IO_REG32(RSTCLR, 0xFFFF0234,__WRITE);
__IO_REG32(RSTKEY0, 0xFFFF0248,__READ_WRITE);
__IO_REG32(RSTCFG, 0xFFFF024C,__READ_WRITE);
__IO_REG32(RSTKEY1, 0xFFFF0250,__READ_WRITE);

/***************************************************************************
 **
 ** TIMER 0
 **
 ***************************************************************************/

__IO_REG32(T0LD, 0xFFFF0300,__READ_WRITE);
__IO_REG32(T0VAL, 0xFFFF0304,__READ);
__IO_REG32(T0CON, 0xFFFF0308,__READ_WRITE);
__IO_REG32(T0CLRI, 0xFFFF030C,__WRITE);

/***************************************************************************
 **
 ** GENERAL PURPOSE TIMER
 **
 ***************************************************************************/

__IO_REG32(T1LD, 0xFFFF0320,__READ_WRITE);
__IO_REG32(T1VAL, 0xFFFF0324,__READ);
__IO_REG32(T1CON, 0xFFFF0328,__READ_WRITE);
__IO_REG32(T1CLRI, 0xFFFF032C,__WRITE);
__IO_REG32(T1CAP, 0xFFFF0330,__READ_WRITE);

/***************************************************************************
 **
 ** GENERAL PURPOSE TIMER
 **
 ***************************************************************************/

__IO_REG32(T2LD, 0xFFFF0340,__READ_WRITE);
__IO_REG32(T2VAL, 0xFFFF0344,__READ);
__IO_REG32(T2CON, 0xFFFF0348,__READ_WRITE);
__IO_REG32(T2CLRI, 0xFFFF034C,__WRITE);

/***************************************************************************
 **
 ** WATCHDOG TIMER
 **
 ***************************************************************************/

__IO_REG32(T3LD, 0xFFFF0360,__READ_WRITE);
__IO_REG32(T3VAL, 0xFFFF0364,__READ);
__IO_REG32(T3CON, 0xFFFF0368,__READ_WRITE);
__IO_REG32(T3CLRI, 0xFFFF036C,__WRITE);

/***************************************************************************
 **
 ** PLL AND OSCILLATOR CONTROL
 **
 ***************************************************************************/

__IO_REG32(POWKEY1, 0xFFFF0404,__WRITE);
__IO_REG32(POWCON0, 0xFFFF0408,__READ_WRITE);
__IO_REG32(POWKEY2, 0xFFFF040C,__WRITE);
__IO_REG32(PLLKEY1, 0xFFFF0410,__WRITE);
__IO_REG32(PLLCON, 0xFFFF0414,__READ_WRITE);
__IO_REG32(PLLKEY2, 0xFFFF0418,__WRITE);

/***************************************************************************
 **
 ** GLOBAL PERIPHERAL CONTROL
 **
 ***************************************************************************/

__IO_REG32(POWKEY3, 0xFFFF0434,__READ_WRITE);
__IO_REG32(POWCON1, 0xFFFF0438,__READ_WRITE);
__IO_REG32(POWKEY4, 0xFFFF043C,__READ_WRITE);

/***************************************************************************
 **
 ** POWER SUPPLY MONITOR
 **
 ***************************************************************************/

__IO_REG32(PSMCON, 0xFFFF0440,__READ_WRITE);
__IO_REG32(CMPCON, 0xFFFF0444,__READ_WRITE);

/***************************************************************************
 **
 ** Band Gap Reference
 **
 ***************************************************************************/

__IO_REG32(REFCON, 0xFFFF048C,__READ_WRITE);

/***************************************************************************
 **
 ** ADC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(ADCCON, 0xFFFF0500,__READ_WRITE);
__IO_REG32(ADCCP, 0xFFFF0504,__READ_WRITE);
__IO_REG32(ADCCN, 0xFFFF0508,__READ_WRITE);
__IO_REG32(ADCSTA, 0xFFFF050C,__READ);
__IO_REG32(ADCDAT, 0xFFFF0510,__READ);
__IO_REG32(ADCRST, 0xFFFF0514,__READ_WRITE);
__IO_REG32(ADCGN, 0xFFFF0530,__READ_WRITE);
__IO_REG32(ADCOF, 0xFFFF0534,__READ_WRITE);
__IO_REG32(TSCON, 0xFFFF0544,__READ_WRITE);
__IO_REG32(TEMPREF, 0xFFFF0548,__READ_WRITE);

/***************************************************************************
 **
 ** DAC INTERFACE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(DAC0CON, 0xFFFF0600,__READ_WRITE);
__IO_REG32(DAC0DAT, 0xFFFF0604,__READ_WRITE);
__IO_REG32(DAC1CON, 0xFFFF0608,__READ_WRITE);
__IO_REG32(DAC1DAT, 0xFFFF060C,__READ_WRITE);
__IO_REG32(DAC2CON, 0xFFFF0610,__READ_WRITE);
__IO_REG32(DAC2DAT, 0xFFFF0614,__READ_WRITE);
__IO_REG32(DAC3CON, 0xFFFF0618,__READ_WRITE);
__IO_REG32(DAC3DAT, 0xFFFF061C,__READ_WRITE);
__IO_REG32(DACBKEY1, 0xFFFF0650,__READ_WRITE);
__IO_REG32(DACBCFG, 0xFFFF0654,__READ_WRITE);
__IO_REG32(DACBKEY2, 0xFFFF0658,__READ_WRITE);

/***************************************************************************
 **
 ** 450 COMPATIABLE UART CORE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(COM0TX, 0xFFFF0700,__WRITE);
#define COM0RX COM0TX
#define COM0DIV0 COM0TX
__IO_REG32(COM0IEN0, 0xFFFF0704,__READ_WRITE);
#define COM0DIV1 COM0IEN0
__IO_REG32(COM0FCR, 0xFFFF0708,__WRITE);
#define COM0IID0 COM0FCR
__IO_REG32(COM0CON0, 0xFFFF070C,__READ_WRITE);
__IO_REG32(COM0CON1, 0xFFFF0710,__READ_WRITE);
__IO_REG32(COM0STA0, 0xFFFF0714,__READ);
__IO_REG32(COM0STA1, 0xFFFF0718,__READ);
__IO_REG32(COM0DIV2, 0xFFFF072C,__READ_WRITE);

/***************************************************************************
 **
 ** 450 COMPATIABLE UART CORE REGISTERS
 **
 ***************************************************************************/

__IO_REG32(COM1TX, 0xFFFF0740,__WRITE);
#define COM1RX COM1TX
#define COM1DIV0 COM1TX
__IO_REG32(COM1IEN0, 0xFFFF0744,__READ_WRITE);
#define COM1DIV1 COM1IEN0
__IO_REG32(COM1FCR, 0xFFFF0748,__WRITE);
#define COM1IID0 COM1FCR
__IO_REG32(COM1CON0, 0xFFFF074C,__READ_WRITE);
__IO_REG32(COM1CON1, 0xFFFF0750,__READ_WRITE);
__IO_REG32(COM1STA0, 0xFFFF0754,__READ);
__IO_REG32(COM1STA1, 0xFFFF0758,__READ);
__IO_REG32(COM1DIV2, 0xFFFF076C,__READ_WRITE);

/***************************************************************************
 **
 ** I2C0 BUS PERIPHERAL DEVICE 0
 **
 ***************************************************************************/

__IO_REG32(I2C0MCON, 0xFFFF0800,__READ_WRITE);
__IO_REG32(I2C0MSTA, 0xFFFF0804,__READ);
__IO_REG32(I2C0MRX, 0xFFFF0808,__READ);
__IO_REG32(I2C0MTX, 0xFFFF080C,__WRITE);
__IO_REG32(I2C0MCNT0, 0xFFFF0810,__READ_WRITE);
__IO_REG32(I2C0MCNT1, 0xFFFF0814,__READ);
__IO_REG32(I2C0ADR0, 0xFFFF0818,__READ_WRITE);
__IO_REG32(I2C0ADR1, 0xFFFF081C,__READ_WRITE);
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

__IO_REG32(I2C1MCON, 0xFFFF0900,__READ_WRITE);
__IO_REG32(I2C1MSTA, 0xFFFF0904,__READ);
__IO_REG32(I2C1MRX, 0xFFFF0908,__READ);
__IO_REG32(I2C1MTX, 0xFFFF090C,__WRITE);
__IO_REG32(I2C1MCNT0, 0xFFFF0910,__READ_WRITE);
__IO_REG32(I2C1MCNT1, 0xFFFF0914,__READ);
__IO_REG32(I2C1ADR0, 0xFFFF0918,__READ_WRITE);
__IO_REG32(I2C1ADR1, 0xFFFF091C,__READ_WRITE);
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

__IO_REG32(SPISTA, 0xFFFF0A00,__READ);
__IO_REG32(SPIRX, 0xFFFF0A04,__READ);
__IO_REG32(SPITX, 0xFFFF0A08,__WRITE);
__IO_REG32(SPIDIV, 0xFFFF0A0C,__READ_WRITE);
__IO_REG32(SPICON, 0xFFFF0A10,__READ_WRITE);

/***************************************************************************
 **
 ** PROGRAMABLE LOGIC ARRAY
 **
 ***************************************************************************/

__IO_REG32(PLAELM0, 0xFFFF0B00,__READ_WRITE);
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

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/

__IO_REG32(PWMCON0, 0xFFFF0F80,__READ_WRITE);
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
__IO_REG32(PWMCON1, 0xFFFF0FB4,__READ_WRITE);
__IO_REG32(PWMCLRI, 0xFFFF0FB8,__READ_WRITE);

/***************************************************************************
 **
 ** EXTERNAL MEMORY REGISTERS
 **
 ***************************************************************************/

__IO_REG32(XMCFG, 0xFFFFF000,__READ_WRITE);
__IO_REG32(XM0CON, 0xFFFFF010,__READ_WRITE);
__IO_REG32(XM1CON, 0xFFFFF014,__READ_WRITE);
__IO_REG32(XM2CON, 0xFFFFF018,__READ_WRITE);
__IO_REG32(XM3CON, 0xFFFFF01C,__READ_WRITE);
__IO_REG32(XM0PAR, 0xFFFFF020,__READ_WRITE);
__IO_REG32(XM1PAR, 0xFFFFF024,__READ_WRITE);
__IO_REG32(XM2PAR, 0xFFFFF028,__READ_WRITE);
__IO_REG32(XM3PAR, 0xFFFFF02C,__READ_WRITE);

/***************************************************************************
 **
 ** GPIO AND SERIAL PORT MUX
 **
 ***************************************************************************/

__IO_REG32(GP0CON, 0xFFFFF400,__READ_WRITE);
__IO_REG32(GP1CON, 0xFFFFF404,__READ_WRITE);
__IO_REG32(GP2CON, 0xFFFFF408,__READ_WRITE);
__IO_REG32(GP3CON, 0xFFFFF40C,__READ_WRITE);
__IO_REG32(GP4CON, 0xFFFFF410,__READ_WRITE);
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
__IO_REG32(GP3DAT, 0xFFFFF450,__READ_WRITE);
__IO_REG32(GP3SET, 0xFFFFF454,__WRITE);
__IO_REG32(GP3CLR, 0xFFFFF458,__WRITE);
__IO_REG32(GP3PAR, 0xFFFFF45C,__READ_WRITE);
__IO_REG32(GP4DAT, 0xFFFFF460,__READ_WRITE);
__IO_REG32(GP4SET, 0xFFFFF464,__WRITE);
__IO_REG32(GP4CLR, 0xFFFFF468,__WRITE);
__IO_REG32(GP4PAR, 0xFFFFF46C,__READ_WRITE);

/***************************************************************************
 **
 ** FLASH CONTROL INTERFACE
 **
 ***************************************************************************/

__IO_REG32(FEE0STA, 0xFFFFF800,__READ);
__IO_REG32(FEE0MOD, 0xFFFFF804,__READ_WRITE);
__IO_REG32(FEE0CON, 0xFFFFF808,__READ_WRITE);
__IO_REG32(FEE0DAT, 0xFFFFF80C,__READ_WRITE);
__IO_REG32(FEE0ADR, 0xFFFFF810,__READ_WRITE);
__IO_REG32(FEE0SGN, 0xFFFFF818,__READ);
__IO_REG32(FEE0PRO, 0xFFFFF81C,__READ_WRITE);
__IO_REG32(FEE0HID, 0xFFFFF820,__READ_WRITE);

/***************************************************************************
 **
 ** FLASH CONTROL INTERFACE
 **
 ***************************************************************************/

__IO_REG32(FEE1STA, 0xFFFFF880,__READ);
__IO_REG32(FEE1MOD, 0xFFFFF884,__READ_WRITE);
__IO_REG32(FEE1CON, 0xFFFFF888,__READ_WRITE);
__IO_REG32(FEE1DAT, 0xFFFFF88C,__READ_WRITE);
__IO_REG32(FEE1ADR, 0xFFFFF890,__READ_WRITE);
__IO_REG32(FEE1SGN, 0xFFFFF898,__READ);
__IO_REG32(FEE1PRO, 0xFFFFF89C,__READ_WRITE);
__IO_REG32(FEE1HID, 0xFFFFF8A0,__READ_WRITE);


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif // __IOADI7126_H
