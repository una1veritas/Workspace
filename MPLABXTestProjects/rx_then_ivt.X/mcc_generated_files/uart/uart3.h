/**
 * UART3 Generated Driver API Header File
 * 
 * @file uart3.h
 * 
 * @defgroup uart3 UART3
 * 
 * @brief This file contains API prototypes and other data types for the the Universal Asynchronous Receiver and Transmitter (UART) module.
 *
 * @version UART3 Driver Version 3.0.9
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

#ifndef UART3_H
#define UART3_H


#include <stdbool.h>
#include <stdint.h>
/**
    @ingroup uart3 
    @def Standard Input Output functions
    @misradeviation{@required, 21.6} This inclusion is essential for UART module to use Printf function for print the character.
*/
/* cppcheck-suppress misra-c2012-21.6 */
#include <stdio.h>
#include "../system/system.h"

#ifdef __cplusplus  // Provide C++ Compatibility

    extern "C" {

#endif


/**
 @ingroup uart3
 @struct uart3_status_t
 @brief This is a structure defined for errors in reception of data.
 */
 /**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
typedef union {
    struct {
        uint8_t perr : 1;     /**<This is a bit field for Parity Error status*/
        uint8_t ferr : 1;     /**<This is a bit field for Framing Error status*/
        uint8_t oerr : 1;     /**<This is a bit field for Overfrun Error status*/
        uint8_t reserved : 5; /**<Reserved*/
    };
    size_t status;            /**<Group byte for status errors*/
}uart3_status_t;

/**
 * @ingroup uart3
 * @brief Initializes the UART3 module. This routine is called
 *        only once during system initialization, before calling other APIs.
 * @param None.
 * @return None.
 */
void UART3_Initialize(void);

/**
 * @ingroup uart3
 * @brief Deinitializes and disables the UART3 module.

 * @param None.
 * @return None.
 */
void UART3_Deinitialize(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 module.     
 * @param None.
 * @return None.
 */
void UART3_Enable(void);

/**
 * @ingroup uart3
 * @brief Disables the UART3 module.
 * @param None.
 * @return None.
 */
void UART3_Disable(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 transmitter. 
 *        The UART3 must be enabled to send the bytes over to the TX pin.
 * @param None.
 * @return None.
 */
void UART3_TransmitEnable(void);

/**
 * @ingroup uart3
 * @brief Disables the UART3 transmitter.
 * @param None.
 * @return None.
 */
void UART3_TransmitDisable(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 receiver.
 *        The UART3 must be enabled to receive the bytes sent by the RX pin.
 * @param None.
 * @return None.
 */
void UART3_ReceiveEnable(void);

/**
 * @ingroup uart3
 * @brief Disables the UART3 receiver.
 * @param None.
 * @return None.
 */
void UART3_ReceiveDisable(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 to send a break control.
 * @param None.
 * @return None.
 */
void UART3_SendBreakControlEnable(void);

/**
 * @ingroup uart3
 * @brief Disables the UART3 Send Break Control bit.
 * @param None.
 * @return None.
 */
void UART3_SendBreakControlDisable(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 receiver interrupt.
 * @param None.
 * @return None.
 */
void UART3_ReceiveInterruptEnable(void);

/**
 * @ingroup uart3
 * @brief Disables the UART3 receiver interrupt.
 * @param None.
 * @return None.
 */
void UART3_ReceiveInterruptDisable(void);

/**
 * @ingroup uart3
 * @brief Enables the UART3 Auto-Baud Detection (ABR).
 * @param bool enable
 * @return None.
 */
void UART3_AutoBaudSet(bool enable);


/**
 * @ingroup uart3
 * @brief Reads the UART3 Auto-Baud Detection Complete bit.
 * @param None.
 * @return None.
 */
bool UART3_AutoBaudQuery(void);

/**
 * @ingroup uart3
 * @brief Resets the UART3 Auto-Baud Detection Complete bit.
 * @param None.
 * @return None.
 */
void UART3_AutoBaudDetectCompleteReset(void);

/**
 * @ingroup uart3
 * @brief Reads the UART3 Auto-Baud Detection Overflow bit.
 * @param None.
 * @return None.
 */
bool UART3_IsAutoBaudDetectOverflow(void);

/**
 * @ingroup uart3
 * @brief Resets the UART3 Auto-Baud Detection Overflow bit.
 * @param None.
 * @return None.
 */
void UART3_AutoBaudDetectOverflowReset(void);

/**
 * @ingroup uart3
 * @brief Checks if the UART3 receiver has received data and is ready to be read.
 * @param None.
 * @retval True - UART3 receiver FIFO has data
 * @retval False - UART3 receiver FIFO is empty
 */
bool UART3_IsRxReady(void);

/**
 * @ingroup uart3
 * @brief Checks if the UART3 transmitter is ready to accept a data byte.
 * @param None.
 * @retval True -  The UART3 transmitter FIFO has at least a one byte space
 * @retval False - The UART3 transmitter FIFO is full
 */
bool UART3_IsTxReady(void);

/**
 * @ingroup uart3
 * @brief Returns the status of the Transmit Shift Register (TSR).
 * @param None.
 * @retval True - Data completely shifted out from the TSR
 * @retval False - Data is present in Transmit FIFO and/or in TSR
 */
bool UART3_IsTxDone(void);

/**
 * @ingroup uart3
 * @brief Gets the error status of the last read byte. Call 
 *        this function before calling UART3_Read().
 * @pre Call UART3_RxEnable() to enable RX before calling this API.
 * @param None.
 * @return Status of the last read byte. See the uart3_status_t struct for more details.
 */
size_t UART3_ErrorGet(void);

/**
 * @ingroup uart3
 * @brief Reads the eight bits from the Receiver FIFO register.
 * @pre Check the transfer status to see if the receiver is not empty before calling this function. Check 
 *      UART3_IsRxReady() in if () before calling this API.
 * @param None.
 * @return 8-bit data from the RX FIFO register
 */
uint8_t UART3_Read(void);

/**
 * @ingroup uart3
 * @brief Writes a byte of data to the Transmitter FIFO register.
 * @pre Check the transfer status to see if the transmitter is not empty before calling this function. Check
 *      UART3_IsTxReady() in if () before calling this API.
 * @param txData  - Data byte to write to the TX FIFO
 * @return None.
 */
void UART3_Write(uint8_t txData);

/**
 * @ingroup uart3
 * @brief Calls the function upon UART3 framing error.
 * @param callbackHandler - Function pointer called when the framing error condition occurs
 * @return None.
 */
void UART3_FramingErrorCallbackRegister(void (* callbackHandler)(void));

/**
 * @ingroup uart3
 * @brief Calls the function upon UART3 overrun error.
 * @param callbackHandler - Function pointer called when the overrun error condition occurs
 * @return None.
 */
void UART3_OverrunErrorCallbackRegister(void (* callbackHandler)(void));

/**
 * @ingroup uart3
 * @brief Calls the function upon UART3 parity error.
 * @param callbackHandler - Function pointer called when the parity error condition occurs
 * @return None.
 */
void UART3_ParityErrorCallbackRegister(void (* callbackHandler)(void));

/**
 * @ingroup uart3
 * @brief This indicates the function called when the receiver interrupt occurs.
 * @pre Initialize the UART3 module with the receive interrupt enabled.
 * @param None.
 * @return None.
 */
extern void (*UART3_RxInterruptHandler)(void);
/**
 * @ingroup uart3
 * @brief Registers the function to be called when the receiver interrupt occurs
 * @param callbackHandler - Function pointer called when the receiver interrupt condition occurs
 * @return None.
 */
void UART3_RxCompleteCallbackRegister(void (* callbackHandler)(void));

/**
 * @ingroup uart3
 * @brief Implements the ISR for the UART3 receiver interrupt.
 * @param void.
 * @return None.
 */
void UART3_ReceiveISR(void);

/**
 * @ingroup uart3
 * @brief This function used to printf support for reads the 8 bits from the FIFO register receiver.
 * @param None.
 * @return 8-bit data from RX FIFO register.
 */
int getch(void);

/**
 * @ingroup uart3
 * @brief This function used to printf support for writes a byte of data to the transmitter FIFO register.
 * @param txData  - Data byte to write to the TX FIFO.
 * @return None.
 */
void putch(char txData);


#ifdef __cplusplus  // Provide C++ Compatibility


    }

#endif

#endif  // UART3_H
