/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_pcm.h
 * - PCM interface hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_PCM_H
#define Q32_PCM_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * PCM interface support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_PCM_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the PCM scheme and data transfer configuration.
 * Inputs        : config   - Interface operation configuration; use
 *                            PCM_PULLUP_DISABLE/PCM_PULLUP_ENABLE,
 *                            PCM_BIT_ORDER_MSB_FIRST/PCM_BIT_ORDER_LSB_FIRST,
 *                            PCM_TX_ALIGN_MSB/PCM_TX_ALIGN_LSB,
 *                            PCM_WORD_SIZE_*,
 *                            PCM_FRAME_ALIGN_LAST/PCM_FRAME_ALIGN_FIRST,
 *                            PCM_FRAME_WIDTH_SHORT/PCM_FRAME_WIDTH_LONG,
 *                            PCM_FRAME_LENGTH_*,
 *                            PCM_SUBFRAMES_DISABLE/PCM_SUBFRAMES_ENABLE,
 *                            PCM_CONTROLLER_CM3/PCM_CONTROLLER_DMA,
 *                            PCM_DISABLE/PCM_ENABLE,
 *                            and PCM_SELECT_MASTER/PCM_SELECT_SLAVE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_PCM_Config(uint32_t config)
{
    PCM->CTRL = (config & ((1U << PCM_CTRL_PULLUP_ENABLE_Pos)
                          | (1U << PCM_CTRL_BIT_ORDER_Pos)
                          | (1U << PCM_CTRL_TX_ALIGN_Pos)
                          | PCM_CTRL_WORD_SIZE_Mask
                          | (1U << PCM_CTRL_FRAME_ALIGN_Pos)
                          | (1U << PCM_CTRL_FRAME_WIDTH_Pos)
                          | PCM_CTRL_FRAME_LENGTH_Mask
                          | (1U << PCM_CTRL_FRAME_SUBFRAMES_Pos)
                          | (1U << PCM_CTRL_CONTROLLER_Pos)
                          | (1U << PCM_CTRL_ENABLE_Pos)
                          | (1U << PCM_CTRL_SLAVE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_PCM_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable the PCM interface without updating the other PCM
 *                 configuration settings
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_PCM_Enable()
{
    PCM_CTRL->ENABLE_ALIAS = PCM_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_PCM_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable the PCM interface without updating the other PCM
 *                 configuration settings
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_PCM_Disable()
{
    PCM_CTRL->ENABLE_ALIAS = PCM_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_PCM_Read()
 * ----------------------------------------------------------------------------
 * Description   : Read and return the value of the PCM receive data register
 * Inputs        : None
 * Outputs       : data - Value of PCM_RX_DATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_PCM_Read()
{
    return PCM->RX_DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_PCM_Write(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : The supplied data is written to the PCM transmit data
 *                 register.
 * Inputs        : data - Data to be transmitted using the PCM interface
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_PCM_Write(uint32_t data)
{
    PCM->TX_DATA = data;
}


/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_PCM_H */
