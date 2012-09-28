/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_spi.h
 * - SPI interface hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_SPI_H
#define Q32_SPI_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * SPI interface support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI0_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the SPI0 interface operation and interface transfer
 *                 information. If SQI is enabled, also select the SQI
 *                 functionality for the additional SQI I/O pads multiplexed
 *                 with GPIO IF3/UART1.
 * Inputs        : config   - Interface operation configuration; use
 *                            SPI_SELECT_SPI/SPI_SELECT_SQI,
 *                            SPI_OVERRUN_INT_DISABLE/SPI_OVERRUN_INT_ENABLE,
 *                            SPI_UNDERRUN_INT_DISABLE/SPI_UNDERRUN_INT_ENABLE,
 *                            SPI_CONTROLLER_CM3/SPI_CONTROLLER_DMA,
 *                            SPI_SELECT_MASTER/SPI_SELECT_SLAVE,
 *                            SPI_SERI_PULLUP_DISABLE/SPI_SERI_PULLUP_ENABLE,
 *                            SPI_CLK_POLARITY_NORMAL/SPI_CLK_POLARITY_INVERSE,
 *                            SPI_MODE_SELECT_MANUAL/SPI_MODE_SELECT_AUTO,
 *                            SPI_DISABLE/SPI_ENABLE and SPI_PRESCALE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI0_Config(uint32_t config)
{
    SPI0->CTRL0 = (config & ((1U << SPI_CTRL0_SQI_ENABLE_Pos)
                            | (1U << SPI_CTRL0_OVERRUN_INT_ENABLE_Pos)
                            | (1U << SPI_CTRL0_UNDERRUN_INT_ENABLE_Pos)
                            | (1U << SPI_CTRL0_CONTROLLER_Pos)
                            | (1U << SPI_CTRL0_SLAVE_Pos)
                            | (1U << SPI_CTRL0_SERI_PULLUP_ENABLE_Pos)
                            | (1U << SPI_CTRL0_CLK_POLARITY_Pos)
                            | (1U << SPI_CTRL0_MODE_SELECT_Pos)
                            | (1U << SPI_CTRL0_ENABLE_Pos)
                            | SPI_CTRL0_PRESCALE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI0_TransferConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the SPI0 interface transfer information.
 * Inputs        : config   - Interface transfer configuration; use
 *                            SPI_IDLE/SPI_START,
 *                            SPI_WRITE_DATA/SPI_READ_DATA, SPI_CS_* and
 *                            SPI_WORD_SIZE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI0_TransferConfig(uint32_t config)
{
    SPI0->CTRL1 = (config & ((1U << SPI_CTRL1_START_BUSY_Pos)
                            | (1U << SPI_CTRL1_RW_CMD_Pos)
                            | (1U << SPI_CTRL1_CS_Pos)
                            | SPI_CTRL1_WORD_SIZE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI1_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the SPI1 interface operation and interface transfer
 *                 information.
 * Inputs        : config   - Interface operation configuration; use
 *                            SPI_OVERRUN_INT_DISABLE/SPI_OVERRUN_INT_ENABLE,
 *                            SPI_UNDERRUN_INT_DISABLE/SPI_UNDERRUN_INT_ENABLE,
 *                            SPI_CONTROLLER_CM3/SPI_CONTROLLER_DMA,
 *                            SPI_SELECT_MASTER/SPI_SELECT_SLAVE,
 *                            SPI_SERI_PULLUP_DISABLE/SPI_SERI_PULLUP_ENABLE,
 *                            SPI_CLK_POLARITY_NORMAL/SPI_CLK_POLARITY_INVERSE,
 *                            SPI_MODE_SELECT_MANUAL/SPI_MODE_SELECT_AUTO,
 *                            SPI_DISABLE/SPI_ENABLE and SPI_PRESCALE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI1_Config(uint32_t config)
{
    SPI1->CTRL0 = (config & ((1U << SPI_CTRL0_OVERRUN_INT_ENABLE_Pos)
                            | (1U << SPI_CTRL0_UNDERRUN_INT_ENABLE_Pos)
                            | (1U << SPI_CTRL0_CONTROLLER_Pos)
                            | (1U << SPI_CTRL0_SLAVE_Pos)
                            | (1U << SPI_CTRL0_SERI_PULLUP_ENABLE_Pos)
                            | (1U << SPI_CTRL0_CLK_POLARITY_Pos)
                            | (1U << SPI_CTRL0_MODE_SELECT_Pos)
                            | (1U << SPI_CTRL0_ENABLE_Pos)
                            | SPI_CTRL0_PRESCALE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI1_TransferConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the SPI1 interface transfer information.
 * Inputs        : config   - Interface transfer configuration; use
 *                            SPI_IDLE/SPI_START,
 *                            SPI_WRITE_DATA/SPI_READ_DATA, SPI_CS_* and
 *                            SPI_WORD_SIZE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI1_TransferConfig(uint32_t config)
{
    SPI1->CTRL1 = (config & ((1U << SPI_CTRL1_START_BUSY_Pos)
                            | (1U << SPI_CTRL1_RW_CMD_Pos)
                            | (1U << SPI_CTRL1_CS_Pos)
                            | SPI_CTRL1_WORD_SIZE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI0_MasterInit()
 * ----------------------------------------------------------------------------
 * Description   : Initialize an SPI operation on SPI0 when running SPI0 in
 *                 master mode.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : - The SPI0 interface is currently idle.
 *                 - The SPI0 interface is configured for master mode operation.
 *                 - If writing over SPI0, the data to be written has been
 *                   written to SPI0_DATA or SPI0_DATA_S.
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI0_MasterInit()
{
    SPI0_CTRL1->START_BUSY_ALIAS = SPI_START_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI1_MasterInit()
 * ----------------------------------------------------------------------------
 * Description   : Initialize an SPI operation on SPI1 when running SPI1 in
 *                 master mode.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : - The SPI1 interface is currently idle.
 *                 - The SPI1 interface is configured for master mode operation
 *                 - If writing over SPI1, the data to be written has been.
 *                   written to SPI1_DATA or SPI1_DATA_S.
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI1_MasterInit()
{
    SPI1_CTRL1->START_BUSY_ALIAS = SPI_START_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_SPI0_Read()
 * ----------------------------------------------------------------------------
 * Description   : Read and return the value of the SPI0 data register
 * Inputs        : None
 * Outputs       : data - Value of SPI0_DATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_SPI0_Read()
{
    return SPI0->DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_SPI1_Read()
 * ----------------------------------------------------------------------------
 * Description   : Read and return the value of the SPI1 data register
 * Inputs        : None
 * Outputs       : data - Value of SPI1_DATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_SPI1_Read()
{
    return SPI1->DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI0_Write(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write the supplied data to the SPI0 data register.
 * Inputs        : data - Data to be transmitted
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI0_Write(uint32_t data)
{
    SPI0->DATA = data;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SPI1_Write(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write the supplied data to the SPI1 data register.
 * Inputs        : data - Data to be transmitted
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SPI1_Write(uint32_t data)
{
    SPI1->DATA = data;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_SPI_H */
