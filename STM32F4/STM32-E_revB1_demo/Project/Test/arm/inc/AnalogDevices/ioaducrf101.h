#ifndef __IOADIRF101_H
#define __IOADIRF101_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **                            
 **    ADUCRF101 SPECIAL FUNCTION REGISTERS
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
 ** NVIC
 **
 ***************************************************************************/

__IO_REG32(ICTR, 0xE000E004,__READ);
__IO_REG32(STCSR, 0xE000E010,__READ_WRITE);
__IO_REG32(STRVR, 0xE000E014,__READ_WRITE);
__IO_REG32(STCVR, 0xE000E018,__READ_WRITE);
__IO_REG32(STCR, 0xE000E01C,__READ);
__IO_REG32(ISER0, 0xE000E100,__READ_WRITE);
__IO_REG32(ISER1, 0xE000E104,__READ_WRITE);
__IO_REG32(ICER0, 0xE000E180,__READ_WRITE);
__IO_REG32(ICER1, 0xE000E184,__READ_WRITE);
__IO_REG32(ISPR0, 0xE000E200,__READ_WRITE);
__IO_REG32(ISPR1, 0xE000E204,__READ_WRITE);
__IO_REG32(ICPR0, 0xE000E280,__READ_WRITE);
__IO_REG32(ICPR1, 0xE000E284,__READ_WRITE);
__IO_REG32(IABR0, 0xE000E300,__READ_WRITE);
__IO_REG32(IABR1, 0xE000E304,__READ_WRITE);
__IO_REG32(IPR0, 0xE000E400,__READ_WRITE);
__IO_REG32(IPR1, 0xE000E404,__READ_WRITE);
__IO_REG32(IPR2, 0xE000E408,__READ_WRITE);
__IO_REG32(IPR3, 0xE000E40C,__READ_WRITE);
__IO_REG32(IPR4, 0xE000E410,__READ_WRITE);
__IO_REG32(IPR5, 0xE000E414,__READ_WRITE);
__IO_REG32(IPR6, 0xE000E418,__READ_WRITE);
__IO_REG32(IPR7, 0xE000E41C,__READ_WRITE);
__IO_REG32(IPR8, 0xE000E420,__READ_WRITE);
__IO_REG32(IPR9, 0xE000E424,__READ_WRITE);
__IO_REG32(IPR10, 0xE000E428,__READ_WRITE);
__IO_REG32(CPUID, 0xE000ED00,__READ);
__IO_REG32(ICSR, 0xE000ED04,__READ_WRITE);
__IO_REG32(VTOR, 0xE000ED08,__READ_WRITE);
__IO_REG32(AIRCR, 0xE000ED0C,__READ_WRITE);
__IO_REG32(SCR, 0xE000ED10,__READ_WRITE);
__IO_REG32(CCR, 0xE000ED14,__READ_WRITE);
__IO_REG32(SHPR1, 0xE000ED18,__READ_WRITE);
__IO_REG32(SHPR2, 0xE000ED1C,__READ_WRITE);
__IO_REG32(SHPR3, 0xE000ED20,__READ_WRITE);
__IO_REG32(SHCSR, 0xE000ED24,__READ_WRITE);
__IO_REG32(CFSR, 0xE000ED28,__READ_WRITE);
__IO_REG32(HFSR, 0xE000ED2C,__READ_WRITE);
__IO_REG32(MMFAR, 0xE000ED34,__READ_WRITE);
__IO_REG32(BFAR, 0xE000ED38,__READ_WRITE);
__IO_REG32(STIR, 0xE000EF00,__WRITE);

/***************************************************************************
 **
 ** Interrupts
 **
 ***************************************************************************/

__IO_REG16(EI0CFG, 0x40002420,__READ_WRITE);
__IO_REG16(EI1CFG, 0x40002424,__READ_WRITE);
__IO_REG16(EI2CFG, 0x40002428,__READ_WRITE);
__IO_REG16(EICLR, 0x40002430,__READ_WRITE);
__IO_REG8(NMICLR, 0x40002434,__READ_WRITE);

/***************************************************************************
 **
 ** Timer 0
 **
 ***************************************************************************/

__IO_REG16(T0LD, 0x40000000,__READ_WRITE);
__IO_REG16(T0VAL, 0x40000004,__READ);
__IO_REG16(T0CON, 0x40000008,__READ_WRITE);
__IO_REG16(T0CLRI, 0x4000000C,__READ_WRITE);
__IO_REG16(T0CAP, 0x40000010,__READ);
__IO_REG16(T0STA, 0x4000001C,__READ);

/***************************************************************************
 **
 ** Timer 1
 **
 ***************************************************************************/

__IO_REG16(T1LD, 0x40000400,__READ_WRITE);
__IO_REG16(T1VAL, 0x40000404,__READ);
__IO_REG16(T1CON, 0x40000408,__READ_WRITE);
__IO_REG16(T1CLRI, 0x4000040C,__READ_WRITE);
__IO_REG16(T1CAP, 0x40000410,__READ);
__IO_REG16(T1STA, 0x4000041C,__READ);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/

__IO_REG16(PWMCON0, 0x40001000,__READ_WRITE);
__IO_REG8(PWMCON1, 0x40001004,__READ_WRITE);
__IO_REG16(PWMCLRI, 0x40001008,__WRITE);
__IO_REG16(PWM0COM0, 0x40001010,__READ_WRITE);
__IO_REG16(PWM0COM1, 0x40001014,__READ_WRITE);
__IO_REG16(PWM0COM2, 0x40001018,__READ_WRITE);
__IO_REG16(PWM0LEN, 0x4000101C,__READ_WRITE);
__IO_REG16(PWM1COM0, 0x40001020,__READ_WRITE);
__IO_REG16(PWM1COM1, 0x40001024,__READ_WRITE);
__IO_REG16(PWM1COM2, 0x40001028,__READ_WRITE);
__IO_REG16(PWM1LEN, 0x4000102C,__READ_WRITE);
__IO_REG16(PWM2COM0, 0x40001030,__READ_WRITE);
__IO_REG16(PWM2COM1, 0x40001034,__READ_WRITE);
__IO_REG16(PWM2COM2, 0x40001038,__READ_WRITE);
__IO_REG16(PWM2LEN, 0x4000103C,__READ_WRITE);
__IO_REG16(PWM3COM0, 0x40001040,__READ_WRITE);
__IO_REG16(PWM3COM1, 0x40001044,__READ_WRITE);
__IO_REG16(PWM3COM2, 0x40001048,__READ_WRITE);
__IO_REG16(PWM3LEN, 0x4000104C,__READ_WRITE);

/***************************************************************************
 **
 ** PMU (Power Management Unit)
 **
 ***************************************************************************/

__IO_REG16(PWRMOD, 0x40002400,__READ_WRITE);
__IO_REG16(PWRKEY, 0x40002404,__READ_WRITE);
__IO_REG8(PSMCON, 0x40002408,__READ_WRITE);
__IO_REG8(SRAMRET, 0x40002478,__READ_WRITE);
__IO_REG8(SHUTDOWN, 0x4000247C,__READ);

/***************************************************************************
 **
 ** Watchdog Timer
 **
 ***************************************************************************/

__IO_REG16(T3LD, 0x40002580,__READ_WRITE);
__IO_REG16(T3VAL, 0x40002584,__READ);
__IO_REG16(T3CON, 0x40002588,__READ_WRITE);
__IO_REG16(T3CLRI, 0x4000258C,__WRITE);
__IO_REG16(T3STA, 0x40002598,__READ);

/***************************************************************************
 **
 ** WakeUp Timer
 **
 ***************************************************************************/

__IO_REG16(T2VAL0, 0x40002500,__READ);
__IO_REG16(T2VAL1, 0x40002504,__READ);
__IO_REG16(T2CON, 0x40002508,__READ_WRITE);
__IO_REG16(T2INC, 0x4000250C,__READ_WRITE);
__IO_REG16(T2WUFB0, 0x40002510,__READ_WRITE);
__IO_REG16(T2WUFB1, 0x40002514,__READ_WRITE);
__IO_REG16(T2WUFC0, 0x40002518,__READ_WRITE);
__IO_REG16(T2WUFC1, 0x4000251C,__READ_WRITE);
__IO_REG16(T2WUFD0, 0x40002520,__READ_WRITE);
__IO_REG16(T2WUFD1, 0x40002524,__READ_WRITE);
__IO_REG16(T2IEN, 0x40002528,__READ_WRITE);
__IO_REG16(T2STA, 0x4000252C,__READ);
__IO_REG16(T2CLRI, 0x40002530,__WRITE);
__IO_REG16(T2WUFA0, 0x4000253C,__READ_WRITE);
__IO_REG16(T2WUFA1, 0x40002540,__READ_WRITE);

/***************************************************************************
 **
 ** CC (Clock Control)
 **
 ***************************************************************************/

__IO_REG16(CLKCON, 0x40002000,__READ_WRITE);
__IO_REG8(XOSCCON, 0x40002410,__READ_WRITE);
__IO_REG16(CLKACT, 0x40002480,__READ_WRITE);
__IO_REG16(CLKPD, 0x40002484,__READ_WRITE);

/***************************************************************************
 **
 ** Flash Controller
 **
 ***************************************************************************/

__IO_REG16(FEESTA, 0x40002800,__READ);
__IO_REG16(FEECON0, 0x40002804,__READ_WRITE);
__IO_REG16(FEECMD, 0x40002808,__READ_WRITE);
__IO_REG16(FEEADR0L, 0x40002810,__READ_WRITE);
__IO_REG16(FEEADR0H, 0x40002814,__READ_WRITE);
__IO_REG16(FEEADR1L, 0x40002818,__READ_WRITE);
__IO_REG16(FEEADR1H, 0x4000281C,__READ_WRITE);
__IO_REG16(FEEKEY, 0x40002820,__WRITE);
__IO_REG16(FEEPROL, 0x40002828,__READ_WRITE);
__IO_REG16(FEEPROH, 0x4000282C,__READ_WRITE);
__IO_REG16(FEESIGL, 0x40002830,__READ);
__IO_REG16(FEESIGH, 0x40002834,__READ);
__IO_REG16(FEECON1, 0x40002838,__READ_WRITE);
__IO_REG16(FEEADRAL, 0x40002848,__READ);
__IO_REG16(FEEADRAH, 0x4000284C,__READ);
__IO_REG16(FEEAEN0, 0x40002878,__READ_WRITE);
__IO_REG16(FEEAEN1, 0x4000287C,__READ_WRITE);
__IO_REG16(FEEAEN2, 0x40002880,__READ_WRITE);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/

__IO_REG16(I2CMCON, 0x40003000,__READ_WRITE);
__IO_REG16(I2CMSTA, 0x40003004,__READ);
__IO_REG8(I2CMRX, 0x40003008,__READ);
__IO_REG8(I2CMTX, 0x4000300C,__WRITE);
__IO_REG16(I2CMRXCNT, 0x40003010,__READ_WRITE);
__IO_REG16(I2CMCRXCNT, 0x40003014,__READ);
__IO_REG8(I2CADR0, 0x40003018,__READ_WRITE);
__IO_REG8(I2CADR1, 0x4000301C,__READ_WRITE);
__IO_REG16(I2CDIV, 0x40003024,__READ_WRITE);
__IO_REG16(I2CSCON, 0x40003028,__READ_WRITE);
__IO_REG16(I2CSSTA, 0x4000302C,__READ);
__IO_REG16(I2CSRX, 0x40003030,__READ);
__IO_REG16(I2CSTX, 0x40003034,__READ_WRITE);
__IO_REG16(I2CALT, 0x40003038,__READ_WRITE);
__IO_REG16(I2CID0, 0x4000303C,__READ_WRITE);
__IO_REG16(I2CID1, 0x40003040,__READ_WRITE);
__IO_REG16(I2CID2, 0x40003044,__READ_WRITE);
__IO_REG16(I2CID3, 0x40003048,__READ_WRITE);
__IO_REG16(I2CFSTA, 0x4000304C,__READ_WRITE);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/

__IO_REG16(SPI0STA, 0x40004000,__READ);
__IO_REG8(SPI0RX, 0x40004004,__READ);
__IO_REG8(SPI0TX, 0x40004008,__WRITE);
__IO_REG16(SPI0DIV, 0x4000400C,__READ_WRITE);
__IO_REG16(SPI0CON, 0x40004010,__READ_WRITE);
__IO_REG16(SPI0DMA, 0x40004014,__READ_WRITE);
__IO_REG16(SPI0CNT, 0x40004018,__READ_WRITE);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/

__IO_REG16(SPI1STA, 0x40004400,__READ);
__IO_REG8(SPI1RX, 0x40004404,__READ);
__IO_REG8(SPI1TX, 0x40004408,__WRITE);
__IO_REG16(SPI1DIV, 0x4000440C,__READ_WRITE);
__IO_REG16(SPI1CON, 0x40004410,__READ_WRITE);
__IO_REG16(SPI1DMA, 0x40004414,__READ_WRITE);
__IO_REG16(SPI1CNT, 0x40004418,__READ_WRITE);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/

__IO_REG8(COMTX, 0x40005000,__WRITE);
#define COMRX COMTX
__IO_REG8(COMIEN, 0x40005004,__READ_WRITE);
__IO_REG8(COMIIR, 0x40005008,__READ);
__IO_REG8(COMLCR, 0x4000500C,__READ_WRITE);
__IO_REG8(COMMCR, 0x40005010,__READ_WRITE);
__IO_REG8(COMLSR, 0x40005014,__READ);
__IO_REG8(COMMSR, 0x40005018,__READ);
__IO_REG16(COMFBR, 0x40005024,__READ_WRITE);
__IO_REG16(COMDIV, 0x40005028,__READ_WRITE);

/***************************************************************************
 **
 ** GPIO0
 **
 ***************************************************************************/

__IO_REG16(GP0CON, 0x40006000,__READ_WRITE);
__IO_REG8(GP0OEN, 0x40006004,__READ_WRITE);
__IO_REG8(GP0PUL, 0x40006008,__READ_WRITE);
__IO_REG8(GP0OCE, 0x4000600C,__READ_WRITE);
__IO_REG8(GP0IN, 0x40006014,__READ);
__IO_REG8(GP0OUT, 0x40006018,__READ_WRITE);
__IO_REG8(GP0SET, 0x4000601C,__WRITE);
__IO_REG8(GP0CLR, 0x40006020,__WRITE);
__IO_REG8(GP0TGL, 0x40006024,__WRITE);

/***************************************************************************
 **
 ** GPIO1
 **
 ***************************************************************************/

__IO_REG16(GP1CON, 0x40006030,__READ_WRITE);
__IO_REG8(GP1OEN, 0x40006034,__READ_WRITE);
__IO_REG8(GP1PUL, 0x40006038,__READ_WRITE);
__IO_REG8(GP1OCE, 0x4000603C,__READ_WRITE);
__IO_REG8(GP1IN, 0x40006044,__READ);
__IO_REG8(GP1OUT, 0x40006048,__READ_WRITE);
__IO_REG8(GP1SET, 0x4000604C,__WRITE);
__IO_REG8(GP1CLR, 0x40006050,__WRITE);
__IO_REG8(GP1TGL, 0x40006054,__WRITE);

/***************************************************************************
 **
 ** GPIO2
 **
 ***************************************************************************/

__IO_REG16(GP2CON, 0x40006060,__READ_WRITE);
__IO_REG8(GP2OEN, 0x40006064,__READ_WRITE);
__IO_REG8(GP2PUL, 0x40006068,__READ_WRITE);
__IO_REG8(GP2OCE, 0x4000606C,__READ_WRITE);
__IO_REG8(GP2IN, 0x40006074,__READ);
__IO_REG8(GP2OUT, 0x40006078,__READ_WRITE);
__IO_REG8(GP2SET, 0x4000607C,__WRITE);
__IO_REG8(GP2CLR, 0x40006080,__WRITE);
__IO_REG8(GP2TGL, 0x40006084,__WRITE);

/***************************************************************************
 **
 ** GPIO3
 **
 ***************************************************************************/

__IO_REG16(GP3CON, 0x40006090,__READ_WRITE);
__IO_REG8(GP3OEN, 0x40006094,__READ_WRITE);
__IO_REG8(GP3PUL, 0x40006098,__READ_WRITE);
__IO_REG8(GP3OCE, 0x4000609C,__READ_WRITE);
__IO_REG8(GP3IN, 0x400060A4,__READ);
__IO_REG8(GP3OUT, 0x400060A8,__READ_WRITE);
__IO_REG8(GP3SET, 0x400060AC,__WRITE);
__IO_REG8(GP3CLR, 0x400060B0,__WRITE);
__IO_REG8(GP3TGL, 0x400060B4,__WRITE);

/***************************************************************************
 **
 ** GPIO4
 **
 ***************************************************************************/

__IO_REG16(GP4CON, 0x400060C0,__READ_WRITE);
__IO_REG8(GP4OEN, 0x400060C4,__READ_WRITE);
__IO_REG8(GP4PUL, 0x400060C8,__READ_WRITE);
__IO_REG8(GP4OCE, 0x400060CC,__READ_WRITE);
__IO_REG8(GP4IN, 0x400060D4,__READ);
__IO_REG8(GP4OUT, 0x400060D8,__READ_WRITE);
__IO_REG8(GP4SET, 0x400060DC,__WRITE);
__IO_REG8(GP4CLR, 0x400060E0,__WRITE);
__IO_REG8(GP4TGL, 0x400060E4,__WRITE);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/

__IO_REG8(GPDWN, 0x400060F0,__READ_WRITE);

/***************************************************************************
 **
 ** GPIORF
 **
 ***************************************************************************/

__IO_REG16(RFTST, 0x40008824,__READ_WRITE);
__IO_REG8(SWACT, 0x40008830,__READ_WRITE);

/***************************************************************************
 **
 ** DMA
 **
 ***************************************************************************/

__IO_REG32(DMASTA, 0x40010000,__READ);
__IO_REG32(DMACFG, 0x40010004,__READ_WRITE);
__IO_REG32(DMAPDBPTR, 0x40010008,__READ_WRITE);
__IO_REG32(DMAADBPTR, 0x4001000C,__READ);
__IO_REG32(DMASWREQ, 0x40010014,__WRITE);
__IO_REG32(DMARMSKSET, 0x40010020,__READ_WRITE);
__IO_REG32(DMARMSKCLR, 0x40010024,__WRITE);
__IO_REG32(DMAENSET, 0x40010028,__READ_WRITE);
__IO_REG32(DMAENCLR, 0x4001002C,__WRITE);
__IO_REG32(DMAALTSET, 0x40010030,__READ_WRITE);
__IO_REG32(DMAALTCLR, 0x40010034,__WRITE);
__IO_REG32(DMAPRISET, 0x40010038,__READ_WRITE);
__IO_REG32(DMAPRICLR, 0x4001003C,__WRITE);
__IO_REG32(DMAERRCLR, 0x4001004C,__READ_WRITE);
__IO_REG8(DMAPERID4, 0x40010FD0,__READ);
__IO_REG8(DMAPERID0, 0x40010FE0,__READ);
__IO_REG8(DMAPERID1, 0x40010FE4,__READ);
__IO_REG8(DMAPERID2, 0x40010FE8,__READ);
__IO_REG8(DMAPERID3, 0x40010FEC,__READ);
__IO_REG8(DMAPCELLID0, 0x40010FF0,__READ);
__IO_REG8(DMAPCELLID1, 0x40010FF4,__READ);
__IO_REG8(DMAPCELLID2, 0x40010FF8,__READ);
__IO_REG8(DMAPCELLID3, 0x40010FFC,__READ);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/

__IO_REG16(ADCCFG, 0x40050000,__READ_WRITE);
__IO_REG8(ADCCON, 0x40050004,__READ_WRITE);
__IO_REG8(ADCSTA, 0x40050008,__READ);
__IO_REG16(ADCDAT, 0x4005000C,__READ);
__IO_REG16(ADCGN, 0x40050010,__READ_WRITE);
__IO_REG16(ADCOF, 0x40050014,__READ_WRITE);

/***************************************************************************
 **
 ** Reset
 **
 ***************************************************************************/

__IO_REG8(RSTSTA, 0x40002440,__READ);
#define RSTCLR RSTSTA


/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 ***************************************************************************/

#endif // __IOADIRF101_H
