/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_uart.h
 * - UART hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_UART_H
#define Q32_UART_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Universal Asynchronous Receiver Transmitter support function prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_UART0_Set_Speed(uint32_t baud_rate,
                                uint32_t freq_in_hz);
extern void Sys_UART1_Set_Speed(uint32_t baud_rate,
                                uint32_t freq_in_hz);
extern uint32_t Sys_UART0_Get_Speed(uint32_t freq_in_hz);
extern uint32_t Sys_UART1_Get_Speed(uint32_t freq_in_hz);

/* ----------------------------------------------------------------------------
 * Universal Asynchronous Receiver Transmitter support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART0_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the settings for UART0
 * Inputs        : config - Configuration for UART0; use
 *                 UART_CONTROLLER_CM3/UART_CONTROLLER_DMA,
 *                 UART_ENABLE/UART0_DISABLE and
 *                 UART_PRESCALE_ENABLE/UART_PRESCALE_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART0_Config(uint32_t config)
{
    UART0->CTRL = (config & ((1U << UART_CTRL_CONTROLLER_Pos)
                            | (1U << UART_CTRL_ENABLE_Pos)
                            | (1U << UART_CTRL_PRESCALE_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART1_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the settings for UART1
 * Inputs        : config - Configuration for UART1; use
 *                 UART_CONTROLLER_CM3/UART_CONTROLLER_DMA,
 *                 UART_ENABLE/UART_DISABLE and
 *                 UART_PRESCALE_ENABLE/UART_PRESCALE_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART1_Config(uint32_t config)
{
    UART1->CTRL = (config & ((1U << UART_CTRL_CONTROLLER_Pos)
                            | (1U << UART_CTRL_ENABLE_Pos)
                            | (1U << UART_CTRL_PRESCALE_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART0_Clear_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the UART0 overrun status bit
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART0_Clear_Status()
{
    UART0->STATUS = UART_OVERRUN_CLEAR;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART1_Clear_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the UART1 overrun status bit
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART1_Clear_Status()
{
    UART1->STATUS = UART_OVERRUN_CLEAR;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_UART0_Get_Status()
 * ----------------------------------------------------------------------------
 * Description   : Return the UART0 overrun status
 * Inputs        : None
 * Outputs       : return value - Value of UART0->STATUS; compare with
 *                 UART0_OVERRUN_TRUE/UART0_OVERRUN_FALSE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_UART0_Get_Status()
{
    return UART0->STATUS;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_UART1_Get_Status()
 * ----------------------------------------------------------------------------
 * Description   : Return the UART1 overrun status
 * Inputs        : None
 * Outputs       : return value - Value of UART1->STATUS; compare with
 *                 UART1_OVERRUN_TRUE/UART1_OVERRUN_FALSE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_UART1_Get_Status()
{
    return UART1->STATUS;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART0_Write(uint8_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write data to the UART0 transmit register
 * Inputs        : data - Data to be transferred
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART0_Write(uint8_t data)
{
    UART0->TXDATA = data;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_UART1_Write(uint8_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write data to the UART1 transmit register
 * Inputs        : data - Data to be transferred
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_UART1_Write(uint8_t data)
{
    UART1->TXDATA = data;
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_UART0_Read()
 * ----------------------------------------------------------------------------
 * Description   : Reads data from the UART0 receive register
 * Inputs        : None
 * Outputs       : return value - Data received and stored in UART0->RXDATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_UART0_Read()
{
    return UART0->RXDATA;
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_UART1_Read()
 * ----------------------------------------------------------------------------
 * Description   : Reads data from the UART1 receive register
 * Inputs        : None
 * Outputs       : return value - Data received and stored in UART1->RXDATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_UART1_Read()
{
    return UART1->RXDATA;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_UART_H */
