/**
 * CLC5 Generated Driver API Header File.
 * 
 * @file clc5.h
 * 
 * @defgroup  clc5 CLC5
 * 
 * @brief This file contains the API prototypes for the CLC5 driver.
 *
 * @version CLC5 Driver Version 1.2.0
*/
/*
© [2026] Microchip Technology Inc. and its subsidiaries.

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

#ifndef CLC5_H
 #define CLC5_H

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_Initialize  CLC5_Initialize
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_Enable CLC5_Enable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_Disable CLC5_Disable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_ISR CLC5_ISR
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_OutputStatusGet CLC5_OutputStatusGet
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_RisingEdgeDetectionEnable CLC5_RisingEdgeDetectionEnable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_RisingEdgeDetectionDisable CLC5_RisingEdgeDetectionDisable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_FallingEdgeDetectionEnable CLC5_FallingEdgeDetectionEnable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_FallingEdgeDetectionDisable CLC5_FallingEdgeDetectionDisable
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_CallbackRegister CLC5_CallbackRegister
/* cppcheck-suppress misra-c2012-2.5 */
#define RDY_FF_Tasks CLC5_Tasks


/**
 * @ingroup clc5
 * @brief  Initializes the CLC5 module. This routine configures the CLC5 specific control registers.
 * @param None.
 * @return None.
 */
void CLC5_Initialize(void);

/**
 * @ingroup clc5
 * @brief Enables the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_Enable(void);

/**
 * @ingroup clc5
 * @brief Disables the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_Disable(void);

/**
 * @ingroup clc5
 * @brief Enabes Rising Edge Detection  on CLC5 output for the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_RisingEdgeDetectionEnable(void);

/**
 * @ingroup clc5
 * @brief Disables Rising Edge Detection  on CLC5 output for the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_RisingEdgeDetectionDisable(void);

/**
 * @ingroup clc5
 * @brief Enables Falling Edge Detection  on CLC5 output for the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_FallingEdgeDetectionEnable(void);

/**
 * @ingroup clc5
 * @brief Disables Falling Edge Detection on CLC5 output for the CLC5 module.     
 * @param None.
 * @return None.
 */
void CLC5_FallingEdgeDetectionDisable(void);


/**
 * @ingroup clc5
 * @brief Returns the output pin status of the CLC5 module.
 * @param  None.
 * @retval True - Output is 1
 * @retval False - Output is 0
 */
bool CLC5_OutputStatusGet(void); 

/**
 * @ingroup clc5
 * @brief Setter function for the CLC5 callback.
 * @param CallbackHandler - Pointer to the custom callback
 * @return None.
 */
 void CLC5_CallbackRegister(void (* CallbackHandler)(void));

/**
 * @ingroup clc5
 * @brief Performs tasks to be executed on rising edge or falling edge event in Polling mode.
 * @param None.
 * @return None.
 */
void CLC5_Tasks(void);


#endif  // CLC5_H
/**
 End of File
*/

