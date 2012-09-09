/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_i2c.h
 * - Hardware support functions for the I2C interface
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_I2C_H
#define Q32_I2C_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * I2C interface support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the I2C interface for operation
 * Inputs        : config - I2C interface configuration; use I2C_MASTER_SPEED_*,
 *                          a slave address constant shifted to
 *                          I2C_CTRL0_SLAVE_ADDRESS_POS,
 *                          I2C_CONTROLLER_CM3/I2C_CONTROLLER_DMA,
 *                          I2C_STOP_INT_ENABLE/I2C_STOP_INT_DISABLE,
 *                          I2C_AUTO_ACK_ENABLE/I2C_AUTO_ACK_DISABLE and
 *                          I2C_SLAVE_ENABLE/I2C_SLAVE_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_Config(uint32_t config)
{
    I2C->CTRL0 = (config & (I2C_CTRL0_MASTER_SPEED_PRESCALAR_Mask
                           | I2C_CTRL0_SLAVE_ADDRESS_Mask
                           | (1U << I2C_CTRL0_CONTROLLER_Pos)
                           | (1U << I2C_CTRL0_STOP_INT_ENABLE_Pos)
                           | (1U << I2C_CTRL0_AUTO_ACK_ENABLE_Pos)
                           | (1U << I2C_CTRL0_SLAVE_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_PhysicalConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the I2C interface physical configuration (pull-up
 *                 resistors, signal filtering)
 * Inputs        : config - I2C physical configuration; use I2C_PULLUP_1K/
 *                          I2C_PULLUP_DISABLE and I2C_FILTER_ENABLE/
 *                          I2C_FILTER_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_PhysicalConfig(uint32_t config)
{
    I2C->PHY_CTRL = (config & ((1U << I2C_PHY_CTRL_FILTER_ENABLE_Pos)
                             | (1U << I2C_PHY_CTRL_PULLUP_SELECT_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_I2C_Get_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current I2C interface status
 * Inputs        : None
 * Outputs       : status - Current I2C interface status; compare with
 *                          I2C_DMA_REQUEST/I2C_NO_DMA_REQUEST,
 *                          I2C_STOP_DETECTED/I2C_NO_STOP_DETECTED,
 *                          I2C_DATA_EVENT/I2C_NON_DATA_EVENT,
 *                          I2C_ERROR/I2C_NO_ERROR,
 *                          I2C_BUS_ERROR/I2C_NO_BUS_ERROR,
 *                          I2C_BUFFER_FULL/I2C_BUFFER_EMPTY,
 *                          I2C_CLK_STRETCHED/I2C_CLK_NOT_STRETCHED,
 *                          I2C_BUS_FREE/I2C_BUS_BUSY,
 *                          I2C_DATA_IS_ADDR/I2C_DATA_IS_DATA,
 *                          I2C_IS_READ/I2C_IS_WRITE,
 *                          I2C_ADDR_GEN_CALL/I2C_ADDR_OTHER and
 *                          I2C_HAS_NACK/I2C_HAS_ACK
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_I2C_Get_Status()
{
    return I2C->STATUS;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_StartRead(uint32_t addr)
 * ----------------------------------------------------------------------------
 * Description   : Initialize an I2C master read transfer on the I2C interface
 * Inputs        : addr - I2C slave address to initiate a transfer with
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_StartRead(uint32_t addr)
{
    I2C->ADDR_START = (I2C_READ
                      | ((addr << I2C_ADDR_START_ADDRESS_Pos)
                         & I2C_ADDR_START_ADDRESS_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_StartWrite(uint32_t addr)
 * ----------------------------------------------------------------------------
 * Description   : Initialize an I2C master write transfer on the I2C interface
 * Inputs        : addr - I2C slave address to initiate a transfer with
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_StartWrite(uint32_t addr)
{
    I2C->ADDR_START = (I2C_WRITE
                      | ((addr << I2C_ADDR_START_ADDRESS_Pos)
                         & I2C_ADDR_START_ADDRESS_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_I2C_Read()
 * ----------------------------------------------------------------------------
 * Description   : Read one byte of data from the I2C interface
 * Inputs        : None
 * Outputs       : data - Data read from the I2C interface
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_I2C_Read()
{
    return I2C->DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_Write(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write one byte of data to the I2C interface
 * Inputs        : data - Data to be written over the I2C interface
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_Write(uint32_t data)
{
    I2C->DATA = data;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_ACK()
 * ----------------------------------------------------------------------------
 * Description   : Manually acknowledge the latest transfer
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_ACK()
{
    I2C_CTRL1->ACK_ALIAS = I2C_ACK_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_NACK()
 * ----------------------------------------------------------------------------
 * Description   : Manually not-acknowledge the latest transfer (releases the
 *                 bus to continue with a transfer)
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_NACK()
{
    I2C_CTRL1->NACK_ALIAS = I2C_NACK_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_NACKAndStop()
 * ----------------------------------------------------------------------------
 * Description   : Manually not-acknowledge the latest transfer and send a stop
 *                 condition (Master mode only)
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_NACKAndStop()
{
    I2C_CTRL1->NACK_ALIAS = I2C_NACK_BITBAND;
    I2C_CTRL1->LAST_DATA_ALIAS = I2C_LAST_DATA_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_I2C_Reset()
 * ----------------------------------------------------------------------------
 * Description   : Reset the I2C interface
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_I2C_Reset()
{
    I2C_CTRL1->RESET_ALIAS = I2C_RESET_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_I2C_H */
