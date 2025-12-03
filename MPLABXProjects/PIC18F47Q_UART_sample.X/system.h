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

#ifndef _XTAL_FREQ
/* cppcheck-suppress misra-c2012-21.1 */
#define _XTAL_FREQ 64000000U
#endif

void CLOCK_Initialize(void);

#define INPUT   1
#define OUTPUT  0

#define HIGH    1
#define LOW     0

#define ANALOG      1
#define DIGITAL     0

#define PULL_UP_ENABLED      1
#define PULL_UP_DISABLED     0

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


void PIN_MANAGER_Initialize (void);

void PIN_MANAGER_IOC(void);

#define INTERRUPT_GlobalInterruptEnable() (INTCON0bits.GIE = 1)
#define INTERRUPT_GlobalInterruptDisable() (INTCON0bits.GIE = 0)
#define INTERRUPT_GlobalInterruptStatus() (INTCON0bits.GIE)

void INTERRUPT_Initialize (void);

#define EXT_INT0_InterruptFlagClear()       (PIR1bits.INT0IF = 0)
#define EXT_INT0_InterruptDisable()     (PIE1bits.INT0IE = 0)
#define EXT_INT0_InterruptEnable()       (PIE1bits.INT0IE = 1)
#define EXT_INT0_risingEdgeSet()          (INTCON0bits.INT0EDG = 1)
#define EXT_INT0_fallingEdgeSet()          (INTCON0bits.INT0EDG = 0)
#define EXT_INT1_InterruptFlagClear()       (PIR6bits.INT1IF = 0)
#define EXT_INT1_InterruptDisable()     (PIE6bits.INT1IE = 0)
#define EXT_INT1_InterruptEnable()       (PIE6bits.INT1IE = 1)
#define EXT_INT1_risingEdgeSet()          (INTCON0bits.INT1EDG = 1)
#define EXT_INT1_fallingEdgeSet()          (INTCON0bits.INT1EDG = 0)
#define EXT_INT2_InterruptFlagClear()       (PIR10bits.INT2IF = 0)
#define EXT_INT2_InterruptDisable()     (PIE10bits.INT2IE = 0)
#define EXT_INT2_InterruptEnable()       (PIE10bits.INT2IE = 1)
#define EXT_INT2_risingEdgeSet()          (INTCON0bits.INT2EDG = 1)
#define EXT_INT2_fallingEdgeSet()          (INTCON0bits.INT2EDG = 0)

void INT0_ISR(void);
void INT0_CallBack(void);
void INT0_SetInterruptHandler(void (* InterruptHandler)(void));
extern void (*INT0_InterruptHandler)(void);
void INT0_DefaultInterruptHandler(void);
void INT1_ISR(void);
void INT1_CallBack(void);
void INT1_SetInterruptHandler(void (* InterruptHandler)(void));
extern void (*INT1_InterruptHandler)(void);
void INT1_DefaultInterruptHandler(void);
void INT2_ISR(void);
void INT2_CallBack(void);
void INT2_SetInterruptHandler(void (* InterruptHandler)(void));
extern void (*INT2_InterruptHandler)(void);
void INT2_DefaultInterruptHandler(void);


void SYSTEM_Initialize(void);



#ifdef	__cplusplus
}
#endif

#endif	/* SYSTEM_H */

