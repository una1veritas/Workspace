/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_dma.h
 * - Direct Memory Access (DMA) peripheral support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_DMA_H
#define Q32_DMA_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Direct Memory Access (DMA) peripheral support function prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_DMA_Channel0_Config(uint32_t config,
                                    uint32_t transferLength,
                                    uint32_t counterInt,
                                    uint32_t srcAddr,
                                    uint32_t destAddr);

extern void Sys_DMA_Channel1_Config(uint32_t config,
                                    uint32_t transferLength,
                                    uint32_t counterInt,
                                    uint32_t srcAddr,
                                    uint32_t destAddr);

extern void Sys_DMA_Channel2_Config(uint32_t config,
                                    uint32_t transferLength,
                                    uint32_t counterInt,
                                    uint32_t srcAddr,
                                    uint32_t destAddr);

extern void Sys_DMA_Channel3_Config(uint32_t config,
                                    uint32_t transferLength,
                                    uint32_t counterInt,
                                    uint32_t srcAddr,
                                    uint32_t destAddr);

/* ----------------------------------------------------------------------------
 * Direct Memory Access (DMA) peripheral support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_DMA_Get_Channel0_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current status of DMA channel 0
 * Inputs        : None
 * Outputs       : Return value - The current DMA_STATUS_CH0_BYTE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_DMA_Get_Channel0_Status()
{
    return DMA_STATUS->CH0_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Clear_Channel0_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the current status for DMA channel 0
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Clear_Channel0_Status()
{
    DMA_STATUS->CH0_BYTE =
            ((1U << DMA_STATUS_CH0_BYTE_CH0_ERROR_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH0_BYTE_CH0_COMPLETE_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH0_BYTE_CH0_COUNTER_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH0_BYTE_CH0_START_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH0_BYTE_CH0_DISABLE_INT_STATUS_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel0_SourceAddress(uint32_t srcAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base source address for DMA channel 0
 * Inputs        : srcAddr      - Base source address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel0_SourceAddress(uint32_t srcAddr)
{
    DMA->CH0_SRC_BASE_ADDR = srcAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel0_DestAddress(uint32_t destAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base destination address for DMA channel 0
 * Inputs        : destAddr     - Base destination address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel0_DestAddress(uint32_t destAddr)
{
    DMA->CH0_DEST_BASE_ADDR = destAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel0_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable DMA channel 0
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel0_Enable()
{
    DMA_CH0_CTRL0->ENABLE_ALIAS = DMA_CH0_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel0_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable DMA channel 0
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel0_Disable()
{
    DMA_CH0_CTRL0->ENABLE_ALIAS = DMA_CH0_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_DMA_Get_Channel1_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current status of DMA channel 1
 * Inputs        : None
 * Outputs       : Return value - The current DMA_STATUS_CH1_BYTE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_DMA_Get_Channel1_Status()
{
    return DMA_STATUS->CH1_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Clear_Channel1_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the current status for DMA channel 1
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Clear_Channel1_Status()
{
    DMA_STATUS->CH1_BYTE =
            ((1U << DMA_STATUS_CH1_BYTE_CH1_ERROR_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH1_BYTE_CH1_COMPLETE_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH1_BYTE_CH1_COUNTER_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH1_BYTE_CH1_START_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH1_BYTE_CH1_DISABLE_INT_STATUS_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel1_SourceAddress(uint32_t srcAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base source address for DMA channel 1
 * Inputs        : srcAddr      - Base source address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel1_SourceAddress(uint32_t srcAddr)
{
    DMA->CH1_SRC_BASE_ADDR = srcAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel1_DestAddress(uint32_t destAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base destination address for DMA channel 1
 * Inputs        : destAddr     - Base destination address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel1_DestAddress(uint32_t destAddr)
{
    DMA->CH1_DEST_BASE_ADDR = destAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel1_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable DMA channel 1
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel1_Enable()
{
    DMA_CH1_CTRL0->ENABLE_ALIAS = DMA_CH1_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel1_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable DMA channel 1
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel1_Disable()
{
    DMA_CH1_CTRL0->ENABLE_ALIAS = DMA_CH1_DISABLE_BITBAND;
}


/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_DMA_Get_Channel2_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current status of DMA channel 2
 * Inputs        : None
 * Outputs       : Return value - The current DMA_STATUS_CH2_BYTE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_DMA_Get_Channel2_Status()
{
    return DMA_STATUS->CH2_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Clear_Channel2_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the current status for DMA channel 2
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Clear_Channel2_Status()
{
    DMA_STATUS->CH2_BYTE =
            ((1U << DMA_STATUS_CH2_BYTE_CH2_ERROR_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH2_BYTE_CH2_COMPLETE_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH2_BYTE_CH2_COUNTER_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH2_BYTE_CH2_START_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH2_BYTE_CH2_DISABLE_INT_STATUS_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel2_SourceAddress(uint32_t srcAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base source address for DMA channel 2
 * Inputs        : srcAddr      - Base source address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel2_SourceAddress(uint32_t srcAddr)
{
    DMA->CH2_SRC_BASE_ADDR = srcAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel2_DestAddress(uint32_t destAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base destination address for DMA channel 2
 * Inputs        : destAddr     - Base destination address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel2_DestAddress(uint32_t destAddr)
{
    DMA->CH2_DEST_BASE_ADDR = destAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel2_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable DMA channel 2
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel2_Enable()
{
    DMA_CH2_CTRL0->ENABLE_ALIAS = DMA_CH2_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel2_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable DMA channel 2
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel2_Disable()
{
    DMA_CH2_CTRL0->ENABLE_ALIAS = DMA_CH2_DISABLE_BITBAND;
}


/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_DMA_Get_Channel3_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current status of DMA channel 3
 * Inputs        : None
 * Outputs       : Return value - The current DMA_STATUS_CH3_BYTE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_DMA_Get_Channel3_Status()
{
    return DMA_STATUS->CH3_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Clear_Channel3_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the current status for DMA channel 3
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Clear_Channel3_Status()
{
    DMA_STATUS->CH3_BYTE =
            ((1U << DMA_STATUS_CH3_BYTE_CH3_ERROR_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH3_BYTE_CH3_COMPLETE_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH3_BYTE_CH3_COUNTER_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH3_BYTE_CH3_START_INT_STATUS_Pos)
             | (1U << DMA_STATUS_CH3_BYTE_CH3_DISABLE_INT_STATUS_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel3_SourceAddress(uint32_t srcAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base source address for DMA channel 3
 * Inputs        : srcAddr      - Base source address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel3_SourceAddress(uint32_t srcAddr)
{
    DMA->CH3_SRC_BASE_ADDR = srcAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_Channel3_DestAddress(uint32_t destAddr)
 * ----------------------------------------------------------------------------
 * Description   : Set the base destination address for DMA channel 3
 * Inputs        : destAddr     - Base destination address for the DMA transfer
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_Channel3_DestAddress(uint32_t destAddr)
{
    DMA->CH3_DEST_BASE_ADDR = destAddr;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel3_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable DMA channel 3
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel3_Enable()
{
    DMA_CH3_CTRL0->ENABLE_ALIAS = DMA_CH3_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Channel3_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable DMA channel 3
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Channel3_Disable()
{
    DMA_CH3_CTRL0->ENABLE_ALIAS = DMA_CH3_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_DAC0TimerConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the DMA update timer for DAC0
 * Inputs        : config   - Configuration for the DMA DAC0 update timer; use
 *                            DMA_DAC0_REQUEST_DISABLE/DMA_DAC0_REQUEST_ENABLE
 *                            and a request rate setting for
 *                            DMA_DAC0_REQUEST_RATE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_DAC0TimerConfig(uint32_t config)
{
    DMA->DAC0_REQUEST = (config & ((1U << DMA_DAC0_REQUEST_DMA_ENABLE_Pos)
                                  | DMA_DAC0_REQUEST_RATE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_DMA_Set_DAC1TimerConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the DMA update timer for DAC1
 * Inputs        : config   - Configuration for the DMA DAC1 update timer; use
 *                            DMA_DAC1_REQUEST_DISABLE/DMA_DAC1_REQUEST_ENABLE
 *                            and a request rate setting for
 *                            DMA_DAC1_REQUEST_RATE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_DMA_Set_DAC1TimerConfig(uint32_t config)
{
    DMA->DAC1_REQUEST = (config & ((1U << DMA_DAC1_REQUEST_DMA_ENABLE_Pos)
                                  | DMA_DAC1_REQUEST_RATE_Mask));
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_DMA_H */
