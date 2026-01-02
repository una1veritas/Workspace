/**
 * Interrupt Manager Generated Driver API Header File.
 * 
 * @file interrupt.h
 * 
 * @defgroup interrupt INTERRUPT
 * 
 * @brief This file contains API prototypes and the other data types for the Interrupt Manager driver.
 *
 * @version Interrupt Manager Driver Version 2.1.3
*/

/*
� [2026] Microchip Technology Inc. and its subsidiaries.

    Subject to your compliance with these terms, you may use Microchip 
    software and any derivatives exclusively with Microchip products. 
    You are responsible for complying with 3rd party license terms  
    applicable to your use of 3rd party software (including open source  
    software) that may accompany Microchip software. SOFTWARE IS ?AS IS.? 
    NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS 
    SOFTWARE, INCLUDING ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT,  
    MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT 
    WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE, 
    INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY 
    KIND WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF 
    MICROCHIP HAS BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE 
    FORESEEABLE. TO THE FULLEST EXTENT ALLOWED BY LAW, MICROCHIP?S 
    TOTAL LIABILITY ON ALL CLAIMS RELATED TO THE SOFTWARE WILL NOT 
    EXCEED AMOUNT OF FEES, IF ANY, YOU PAID DIRECTLY TO MICROCHIP FOR 
    THIS SOFTWARE.
*/

#ifndef INTERRUPT_H
#define INTERRUPT_H

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptHighEnable()
 * @brief Enables the high priority global interrupts.
 */
#define INTERRUPT_GlobalInterruptHighEnable() (INTCON0bits.GIE = 1)

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptHighDisable()
 * @brief Disables the high priority global interrupts.
 */
#define INTERRUPT_GlobalInterruptHighDisable() (INTCON0bits.GIE = 0)

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptHighStatus()
 * @brief Returns the Global Interrupt Enable bit status.
 * @param None.
 * @retval True - High priority global interrupt is enabled.
 * @retval False - High priority global interrupt is disabled.
 */
#define INTERRUPT_GlobalInterruptHighStatus() (INTCON0bits.GIE)

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptLowEnable()
 * @brief Enables the low priority global interrupts.
 */
#define INTERRUPT_GlobalInterruptLowEnable() (INTCON0bits.GIEL = 1)

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptLowDisable()
 * @brief Disables the low priority global interrupts.
 */
#define INTERRUPT_GlobalInterruptLowDisable() (INTCON0bits.GIEL = 0)

/**
 * @ingroup interrupt
 * @def INTERRUPT_GlobalInterruptLowStatus()
 * @brief Returns the Global Low-Priority Interrupt Enable bit status.
 * @param None.
 * @retval True - Low priority global interrupt is enabled.
 * @retval False - Low priority global interrupt is disabled.
 */
#define INTERRUPT_GlobalInterruptLowStatus() (INTCON0bits.GIEL)

/**
 * @ingroup interrupt
 * @brief Initializes the interrupt controller.
 * @param None.
 * @return None.
 */
void INTERRUPT_Initialize (void);



#endif  // INTERRUPT_H
/**
 End of File
*/
