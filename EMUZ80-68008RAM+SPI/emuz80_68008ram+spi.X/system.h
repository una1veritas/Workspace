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

#define INPUT   1
#define OUTPUT  0
#define PORT_INPUT  0xff
#define PORT_OUTPUT 0xff

#define HIGH    1
#define LOW     0

#define ANALOG      1
#define DIGITAL     0

#define PULL_UP_ENABLED      1
#define PULL_UP_DISABLED     0


#define MC68K8_RESET_OUT    LATE0
#define MC68K8_BR_OUT       LATE1
#define MC68K8_BG_OUT       LATC2
#define MC68K8_DTACK_OUT    LATC5

#define ALE                 LATE2
#define ALE_OE              LATC3

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

/*
// get/set RA6 aliases
#define IO_RA6_TRIS                 TRISAbits.TRISA6
#define IO_RA6_LAT                  LATAbits.LATA6
#define IO_RA6_PORT                 PORTAbits.RA6
#define IO_RA6_WPU                  WPUAbits.WPUA6
#define IO_RA6_OD                   ODCONAbits.ODCA6
#define IO_RA6_ANS                  ANSELAbits.ANSELA6
#define IO_RA6_SetHigh()            do { LATAbits.LATA6 = 1; } while(0)
#define IO_RA6_SetLow()             do { LATAbits.LATA6 = 0; } while(0)
#define IO_RA6_Toggle()             do { LATAbits.LATA6 = ~LATAbits.LATA6; } while(0)
#define IO_RA6_GetValue()           PORTAbits.RA6
#define IO_RA6_SetDigitalInput()    do { TRISAbits.TRISA6 = 1; } while(0)
#define IO_RA6_SetDigitalOutput()   do { TRISAbits.TRISA6 = 0; } while(0)
#define IO_RA6_SetPullup()          do { WPUAbits.WPUA6 = 1; } while(0)
#define IO_RA6_ResetPullup()        do { WPUAbits.WPUA6 = 0; } while(0)
#define IO_RA6_SetPushPull()        do { ODCONAbits.ODCA6 = 0; } while(0)
#define IO_RA6_SetOpenDrain()       do { ODCONAbits.ODCA6 = 1; } while(0)
#define IO_RA6_SetAnalogMode()      do { ANSELAbits.ANSELA6 = 1; } while(0)
#define IO_RA6_SetDigitalMode()     do { ANSELAbits.ANSELA6 = 0; } while(0)

// get/set RA7 aliases
#define IO_RA7_TRIS                 TRISAbits.TRISA7
#define IO_RA7_LAT                  LATAbits.LATA7
#define IO_RA7_PORT                 PORTAbits.RA7
#define IO_RA7_WPU                  WPUAbits.WPUA7
#define IO_RA7_OD                   ODCONAbits.ODCA7
#define IO_RA7_ANS                  ANSELAbits.ANSELA7
#define IO_RA7_SetHigh()            do { LATAbits.LATA7 = 1; } while(0)
#define IO_RA7_SetLow()             do { LATAbits.LATA7 = 0; } while(0)
#define IO_RA7_Toggle()             do { LATAbits.LATA7 = ~LATAbits.LATA7; } while(0)
#define IO_RA7_GetValue()           PORTAbits.RA7
#define IO_RA7_SetDigitalInput()    do { TRISAbits.TRISA7 = 1; } while(0)
#define IO_RA7_SetDigitalOutput()   do { TRISAbits.TRISA7 = 0; } while(0)
#define IO_RA7_SetPullup()          do { WPUAbits.WPUA7 = 1; } while(0)
#define IO_RA7_ResetPullup()        do { WPUAbits.WPUA7 = 0; } while(0)
#define IO_RA7_SetPushPull()        do { ODCONAbits.ODCA7 = 0; } while(0)
#define IO_RA7_SetOpenDrain()       do { ODCONAbits.ODCA7 = 1; } while(0)
#define IO_RA7_SetAnalogMode()      do { ANSELAbits.ANSELA7 = 1; } while(0)
#define IO_RA7_SetDigitalMode()     do { ANSELAbits.ANSELA7 = 0; } while(0)
*/

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

