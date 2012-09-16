/***************************************************************************
===============================================														
ADuC7122 HEADER FILE REV 1.4														
===============================================														
***************************************************************************/

#ifndef __IOADI7122_H
#define __IOADI7122_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **                            
 **    ADuC7122 SPECIAL FUNCTION REGISTERS
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
 ** Interrupt Controller
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
__IO_REG32(IRQCLRE, 0xFFFF0038,__WRITE);
__IO_REG32(IRQSTAN, 0xFFFF003C,__READ_WRITE);
__IO_REG32(FIQSTA, 0xFFFF0100,__READ);
__IO_REG32(FIQSIG, 0xFFFF0104,__READ);
__IO_REG32(FIQEN, 0xFFFF0108,__READ_WRITE);
__IO_REG32(FIQCLR, 0xFFFF010C,__WRITE);
__IO_REG32(FIQVEC, 0xFFFF011C,__READ);
__IO_REG32(FIQSTAN, 0xFFFF013C,__READ_WRITE);

/***************************************************************************
 **
 ** Remap and System Control
 **
 ***************************************************************************/

__IO_REG32(REMAPBASE, 0xFFFF0200,__READ_WRITE);
__IO_REG32(REMAP, 0xFFFF0220,__READ_WRITE);
__IO_REG32(RSTSTA, 0xFFFF0230,__READ_WRITE);
__IO_REG32(RSTCLR, 0xFFFF0234,__WRITE);

/***************************************************************************
 **
 ** 48bit General Purpose Timer 0
 **
 ***************************************************************************/

__IO_REG32(TIMER0BASE, 0xFFFF0300,__READ_WRITE);
#define T0LD TIMER0BASE
__IO_REG32(T0VAL0, 0xFFFF0304,__READ);
__IO_REG32(T0VAL1, 0xFFFF0308,__READ);
__IO_REG32(T0CON, 0xFFFF030C,__READ_WRITE);
__IO_REG32(T0CLRI, 0xFFFF0310,__WRITE);
__IO_REG32(T0CAP, 0xFFFF0314,__READ_WRITE);

/***************************************************************************
 **
 ** General Purpose Timer 1
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
 ** Wake Up Timer
 **
 ***************************************************************************/

__IO_REG32(TIMER2BASE, 0xFFFF0340,__READ_WRITE);
#define T2LD TIMER2BASE
__IO_REG32(T2VAL, 0xFFFF0344,__READ);
__IO_REG32(T2CON, 0xFFFF0348,__READ_WRITE);
__IO_REG32(T2CLRI, 0xFFFF034C,__WRITE);

/***************************************************************************
 **
 ** Watchdog Timer
 **
 ***************************************************************************/

__IO_REG32(TIMER3BASE, 0xFFFF0360,__READ_WRITE);
#define T3LD TIMER3BASE
__IO_REG32(T3VAL, 0xFFFF0364,__READ);
__IO_REG32(T3CON, 0xFFFF0368,__READ_WRITE);
__IO_REG32(T3CLRI, 0xFFFF036C,__WRITE);

/***************************************************************************
 **
 ** General Purpose Timer 4
 **
 ***************************************************************************/

__IO_REG32(TIMER4BASE, 0xFFFF0380,__READ_WRITE);
#define T4LD TIMER4BASE
__IO_REG32(T4VAL, 0xFFFF0384,__READ);
__IO_REG32(T4CON, 0xFFFF0388,__READ_WRITE);
__IO_REG32(T4CLRI, 0xFFFF038C,__WRITE);
__IO_REG32(T4CAP, 0xFFFF0390,__READ);

/***************************************************************************
 **
 ** PLL and Oscillator Control
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
 ** Power Supply Monitor
 **
 ***************************************************************************/

__IO_REG32(PSMBASE, 0xFFFF0440,__READ_WRITE);
#define PSMCON PSMBASE

/***************************************************************************
 **
 ** Band Gap Reference
 **
 ***************************************************************************/

__IO_REG32(BANDGAPBASE, 0xFFFF0480,__READ_WRITE);
#define REFCON BANDGAPBASE

/***************************************************************************
 **
 ** ADC interface registers
 **
 ***************************************************************************/

__IO_REG32(ADCBASE, 0xFFFF0500,__READ_WRITE);
#define ADCCON ADCBASE
__IO_REG32(ADCCP, 0xFFFF0504,__READ_WRITE);
__IO_REG32(ADCCN, 0xFFFF0508,__READ_WRITE);
__IO_REG32(ADCSTA, 0xFFFF050C,__READ_WRITE);
__IO_REG32(ADCDAT, 0xFFFF0510,__READ);
__IO_REG32(ADCRST, 0xFFFF0514,__WRITE);
__IO_REG32(ADCGN, 0xFFFF0518,__READ_WRITE);
__IO_REG32(ADCOF, 0xFFFF051C,__READ_WRITE);
__IO_REG32(PGAGN, 0xFFFF0520,__READ_WRITE);

/***************************************************************************
 **
 ** DAC Interface Peripheral
 **
 ***************************************************************************/

__IO_REG32(DACBASE, 0xFFFF0580,__READ_WRITE);
#define DAC0CON DACBASE
__IO_REG32(DAC0DAT, 0xFFFF0584,__READ_WRITE);
__IO_REG32(DAC1CON, 0xFFFF0588,__READ_WRITE);
__IO_REG32(DAC1DAT, 0xFFFF058C,__READ_WRITE);
__IO_REG32(DAC2CON, 0xFFFF0590,__READ_WRITE);
__IO_REG32(DAC2DAT, 0xFFFF0594,__READ_WRITE);
__IO_REG32(DAC3CON, 0xFFFF0598,__READ_WRITE);
__IO_REG32(DAC3DAT, 0xFFFF059C,__READ_WRITE);
__IO_REG32(DAC4CON, 0xFFFF05A0,__READ_WRITE);
__IO_REG32(DAC4DAT, 0xFFFF05A4,__READ_WRITE);
__IO_REG32(DAC5CON, 0xFFFF05A8,__READ_WRITE);
__IO_REG32(DAC5DAT, 0xFFFF05AC,__READ_WRITE);
__IO_REG32(DAC6CON, 0xFFFF05B0,__READ_WRITE);
__IO_REG32(DAC6DAT, 0xFFFF05B4,__READ_WRITE);
__IO_REG32(DAC7CON, 0xFFFF05B8,__READ_WRITE);
__IO_REG32(DAC7DAT, 0xFFFF05BC,__READ_WRITE);
__IO_REG32(DAC8CON, 0xFFFF05C0,__READ_WRITE);
__IO_REG32(DAC8DAT, 0xFFFF05C4,__READ_WRITE);
__IO_REG32(DAC9CON, 0xFFFF05C8,__READ_WRITE);
__IO_REG32(DAC9DAT, 0xFFFF05CC,__READ_WRITE);
__IO_REG32(DAC10CON, 0xFFFF05D0,__READ_WRITE);
__IO_REG32(DAC10DAT, 0xFFFF05D4,__READ_WRITE);
__IO_REG32(DAC11CON, 0xFFFF05D8,__READ_WRITE);
__IO_REG32(DAC11DAT, 0xFFFF05DC,__READ_WRITE);

/***************************************************************************
 **
 ** 450 Compatible UART core registers
 **
 ***************************************************************************/

__IO_REG32(UARTBASE, 0xFFFF0800,__READ_WRITE);
#define COMTX UARTBASE
#define COMRX UARTBASE
#define COMDIV0 UARTBASE
__IO_REG32(COMIEN0, 0xFFFF0804,__READ_WRITE);
#define COMDIV1 COMIEN0
__IO_REG32(COMIID0, 0xFFFF0808,__READ_WRITE);
__IO_REG32(COMCON0, 0xFFFF080C,__READ_WRITE);
__IO_REG32(COMCON1, 0xFFFF0810,__READ_WRITE);
__IO_REG32(COMSTA0, 0xFFFF0814,__READ_WRITE);
__IO_REG32(COMSTA1, 0xFFFF0818,__READ_WRITE);
__IO_REG32(COMIID1, 0xFFFF0824,__READ_WRITE);
__IO_REG32(COMDIV2, 0xFFFF082C,__READ_WRITE);

/***************************************************************************
 **
 ** I2C0 Port Interface Peripheral
 **
 ***************************************************************************/

__IO_REG32(I2C0BASE, 0xFFFF0880,__READ_WRITE);
#define I2C0MCTL I2C0BASE
__IO_REG32(I2C0MSTA, 0xFFFF0884,__READ);
__IO_REG32(I2C0MRX, 0xFFFF0888,__READ);
__IO_REG32(I2C0MTX, 0xFFFF088C,__READ_WRITE);
__IO_REG32(I2C0MCNT0, 0xFFFF0890,__READ_WRITE);
__IO_REG32(I2C0MCNT1, 0xFFFF0894,__READ);
__IO_REG32(I2C0ADR0, 0xFFFF0898,__READ_WRITE);
__IO_REG32(I2C0ADR1, 0xFFFF089C,__READ_WRITE);
__IO_REG32(I2C0DIV, 0xFFFF08A4,__READ_WRITE);
__IO_REG32(I2C0SCTL, 0xFFFF08A8,__READ_WRITE);
__IO_REG32(I2C0SSTA, 0xFFFF08AC,__READ);
__IO_REG32(I2C0SRX, 0xFFFF08B0,__READ);
__IO_REG32(I2C0STX, 0xFFFF08B4,__READ_WRITE);
__IO_REG32(I2C0ALT, 0xFFFF08B8,__READ_WRITE);
__IO_REG32(I2C0ID0, 0xFFFF08BC,__READ_WRITE);
__IO_REG32(I2C0ID1, 0xFFFF08C0,__READ_WRITE);
__IO_REG32(I2C0ID2, 0xFFFF08C4,__READ_WRITE);
__IO_REG32(I2C0ID3, 0xFFFF08C8,__READ_WRITE);
__IO_REG32(I2C0FSTA, 0xFFFF08CC,__READ_WRITE);

/***************************************************************************
 **
 ** I2C1 Port Interface Peripheral
 **
 ***************************************************************************/

__IO_REG32(I2C1BASE, 0xFFFF0900,__READ_WRITE);
#define I2C1MCTL I2C1BASE
__IO_REG32(I2C1MSTA, 0xFFFF0904,__READ);
__IO_REG32(I2C1MRX, 0xFFFF0908,__READ);
__IO_REG32(I2C1MTX, 0xFFFF090C,__READ_WRITE);
__IO_REG32(I2C1MCNT0, 0xFFFF0910,__READ_WRITE);
__IO_REG32(I2C1MCNT1, 0xFFFF0914,__READ);
__IO_REG32(I2C1ADR0, 0xFFFF0918,__READ_WRITE);
__IO_REG32(I2C1ADR1, 0xFFFF091C,__READ_WRITE);
__IO_REG32(I2C1DIV, 0xFFFF0924,__READ_WRITE);
__IO_REG32(I2C1SCTL, 0xFFFF0928,__READ_WRITE);
__IO_REG32(I2C1SSTA, 0xFFFF092C,__READ);
__IO_REG32(I2C1SRX, 0xFFFF0930,__READ);
__IO_REG32(I2C1STX, 0xFFFF0934,__READ_WRITE);
__IO_REG32(I2C1ALT, 0xFFFF0938,__READ_WRITE);
__IO_REG32(I2C1ID0, 0xFFFF093C,__READ_WRITE);
__IO_REG32(I2C1ID1, 0xFFFF0940,__READ_WRITE);
__IO_REG32(I2C1ID2, 0xFFFF0944,__READ_WRITE);
__IO_REG32(I2C1ID3, 0xFFFF0948,__READ_WRITE);
__IO_REG32(I2C1FSTA, 0xFFFF094C,__READ_WRITE);

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
__IO_REG32(PLAOUT, 0xFFFF0B50,__READ);

/***************************************************************************
 **
 ** GPIO AND SERIAL PORT MUX
 **
 ***************************************************************************/

__IO_REG32(GPIOBASE, 0xFFFF0D00,__READ_WRITE);
#define GP0CON GPIOBASE
__IO_REG32(GP1CON, 0xFFFF0D04,__READ_WRITE);
__IO_REG32(GP2CON, 0xFFFF0D08,__READ_WRITE);
__IO_REG32(GP3CON, 0xFFFF0D0C,__READ_WRITE);
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
__IO_REG32(GP3DAT, 0xFFFF0D50,__READ_WRITE);
__IO_REG32(GP3SET, 0xFFFF0D54,__WRITE);
__IO_REG32(GP3CLR, 0xFFFF0D58,__WRITE);
__IO_REG32(GP3PAR, 0xFFFF0D5C,__READ_WRITE);
__IO_REG32(GP1OCE, 0xFFFF0D70,__READ_WRITE);
__IO_REG32(GP2OCE, 0xFFFF0D74,__READ_WRITE);
__IO_REG32(GP3OCE, 0xFFFF0D78,__READ_WRITE);

/***************************************************************************
 **
 ** Flash Control Interface 0
 **
 ***************************************************************************/

__IO_REG32(FLASH0BASE, 0xFFFF0E00,__READ_WRITE);
#define FEE0STA FLASH0BASE
__IO_REG32(FEE0MOD, 0xFFFF0E04,__READ_WRITE);
__IO_REG32(FEE0CON, 0xFFFF0E08,__READ_WRITE);
__IO_REG32(FEE0DAT, 0xFFFF0E0C,__READ_WRITE);
__IO_REG32(FEE0ADR, 0xFFFF0E10,__READ_WRITE);
__IO_REG32(FEE0SIG, 0xFFFF0E18,__READ);
__IO_REG32(FEE0PRO, 0xFFFF0E1C,__READ_WRITE);
__IO_REG32(FEE0HID, 0xFFFF0E20,__READ_WRITE);

/***************************************************************************
 **
 ** Flash Control Interface 1
 **
 ***************************************************************************/

__IO_REG32(FLASH1BASE, 0xFFFF0E80,__READ_WRITE);
#define FEE1STA FLASH1BASE
__IO_REG32(FEE1MOD, 0xFFFF0E84,__READ_WRITE);
__IO_REG32(FEE1CON, 0xFFFF0E88,__READ_WRITE);
__IO_REG32(FEE1DAT, 0xFFFF0E8C,__READ_WRITE);
__IO_REG32(FEE1ADR, 0xFFFF0E90,__READ_WRITE);
__IO_REG32(FEE1SIG, 0xFFFF0E98,__READ);
__IO_REG32(FEE1PRO, 0xFFFF0E9C,__READ_WRITE);
__IO_REG32(FEE1HID, 0xFFFF0EA0,__READ_WRITE);

/***************************************************************************
 **
 ** Pulse Width Modulator
 **
 ***************************************************************************/

__IO_REG32(PWMBASE, 0xFFFF0F80,__READ_WRITE);
#define PWMCON1 PWMBASE
__IO_REG32(PWM1COM1, 0xFFFF0F84,__READ_WRITE);
__IO_REG32(PWM1COM2, 0xFFFF0F88,__READ_WRITE);
__IO_REG32(PWM1COM3, 0xFFFF0F8C,__READ_WRITE);
__IO_REG32(PWM1LEN, 0xFFFF0F90,__READ_WRITE);
__IO_REG32(PWM2COM1, 0xFFFF0F94,__READ_WRITE);
__IO_REG32(PWM2COM2, 0xFFFF0F98,__READ_WRITE);
__IO_REG32(PWM2COM3, 0xFFFF0F9C,__READ_WRITE);
__IO_REG32(PWM2LEN, 0xFFFF0FA0,__READ_WRITE);
__IO_REG32(PWM3COM1, 0xFFFF0FA4,__READ_WRITE);
__IO_REG32(PWM3COM2, 0xFFFF0FA8,__READ_WRITE);
__IO_REG32(PWM3COM3, 0xFFFF0FAC,__READ_WRITE);
__IO_REG32(PWM3LEN, 0xFFFF0FB0,__READ_WRITE);
__IO_REG32(PWMCON2, 0xFFFF0FB4,__READ_WRITE);
__IO_REG32(PWMCLRI, 0xFFFF0FB8,__WRITE);


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif // __IOADI7122_H
