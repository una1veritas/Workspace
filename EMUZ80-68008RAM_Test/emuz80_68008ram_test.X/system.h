/* 
 * File:   system.h
 * Author: sin
 *
 * Created on December 3, 2025, 9:16 PM
 */

#ifndef SYSTEM_H
#define	SYSTEM_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

//#ifndef _XTAL_FREQ
/* cppcheck-suppress misra-c2012-21.1 */
#define _XTAL_FREQ 64000000U
//#endif

#define CLK_68008_FREQ   8000UL

void CLOCK_Initialize(void);

#define MC68K8_RESET_OUT    LATE0
#define MC68K8_BR_OUT       LATE1
#define MC68K8_BG_OUT       LATC2
#define MC68K8_DTACK_OUT    LATC5

#define ALE                 LATE2
#define ALE_OE              LATC3
#define ALE_MODE            TRISE2
#define ALE_OE_MODE         TRISC3

#define ADBUS_OUT    LATB
#define ADBUS_IN     PORTB
#define ADBUS_MODE   TRISB
#define ADBUS_WPU   WPUB
#define ABUS_MID_OUT    LATD
#define ABUS_MID_IN     PORTD
#define ABUS_MID_MODE   TRISD
#define ABUS_HIGH4_OUT  LATA
#define ABUS_HIGH4_IN   PORTA
#define ABUS_HIGH4_MODE TRISA

#define MC68K8_RW_IN        PORTA4
#define MC68K8_RW_MODE      TRISA4

#define _SPI_SS_OUT         LATC7

#define M68K8_AS_IN         PORTC1
#define M68K8_DS_IN         PORTC0
#define M68K8_DS_OUT        LATC0
#define SRAM_CE_OUT         LATC0
#define SRAM_CE_MODE        TRISC0
#define SRAM_WE_OUT         LATC4
#define SRAM_WE_MODE        TRISC4

void PIN_MANAGER_Initialize (void);

//void PIN_MANAGER_IOC(void);

void NCO1_Initialize(void);

#define INTERRUPT_GlobalInterruptEnable() (INTCON0bits.GIE = 1)
#define INTERRUPT_GlobalInterruptDisable() (INTCON0bits.GIE = 0)
#define INTERRUPT_GlobalInterruptStatus() (INTCON0bits.GIE)

void INTERRUPT_Initialize (void);
void SYSTEM_Initialize(void);

#ifdef	__cplusplus
}
#endif

#endif	/* SYSTEM_H */

