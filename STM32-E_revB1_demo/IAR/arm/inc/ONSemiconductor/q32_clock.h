/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_clock.h
 * - System clock divider configuration and hardware support functions
 *
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or
 *            <q32m210.h> instead.
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_CLOCK_H
#define Q32_CLOCK_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif


/* ----------------------------------------------------------------------------
 * System clock divider configuration and hardware support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Root(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select the root-clock source and the RTC clock frequency
 * Inputs        : config   - Root clock and RTC clock configuration; use
 *                            RCLK_SELECT_RC_OSC/RCLK_SELECT_RTC_CLK/
 *                            RCLK_SELECT_RTC_XTAL/RCLK_SELECT_EXT and
 *                            RTC_CLK_SELECT_*HZ
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Root(uint32_t config)
{
    CLK->CTRL0 = (config & (CLK_CTRL0_RTC_CLK_DIV_Mask |
                           CLK_CTRL0_RCLK_SELECT_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Timers(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clocks for the SYSTICK, general-purpose
 *                 and watchdog timers
 * Inputs        : config   - Clock divisor configuration for the timers; use
 *                            SYSTICK_DISABLE/SYSTICK_DIV_*, TIMER02_DIV_*,
 *                            TIMER13_DIV_* and WATCHDOG_CLK_DIV_* or a set of
 *                            shifted constants
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Timers(uint32_t config)
{
    CLK->CTRL1 = (config & (CLK_CTRL1_SYSTICK_DIV_Mask
                           | CLK_CTRL1_TIMER02_DIV_Mask
                           | CLK_CTRL1_TIMER13_DIV_Mask
                           | CLK_CTRL1_WATCHDOG_CLK_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_SYSTICK(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the SYSTICK timer
 * Inputs        : config   - Clock divider configuration for the SYSTICK timer;
 *                            use SYSTICK_DISABLE_BYTE, SYSTICK_DIV_*_BYTE or an
 *                            unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_SYSTICK(uint8_t config)
{
    CLK_CTRL1->SYSTICK_DIV_BYTE = config;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Timer02(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for timers 0 and 2
 * Inputs        : config   - Clock divider configuration for timers 0 and 2;
 *                            use TIMER02_DIV_*_BYTE or an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Timer02(uint8_t config)
{
    CLK_CTRL1->TIMER02_DIV_BYTE = (config &
                                   CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Timer13(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for timers 1 and 3
 * Inputs        : config   - Clock divider configuration for timers 1 and 3;
 *                            use TIMER13_DIV_*_BYTE or an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Timer13(uint8_t config)
{
    CLK_CTRL1->TIMER13_DIV_BYTE = (config &
                                   CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Watchdog(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the watchdog timer
 * Inputs        : config   - Clock divider configuration for the watchdog
 *                            timer; use WATCHDOG_CLK_DIV_*_BYTE or an unshifted
 *                            constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Watchdog(uint8_t config)
{
    CLK_CTRL1->WATCHDOG_CLK_DIV_BYTE = (config &
                          CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_UART0(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the UART0 interface
 * Inputs        : config   - Clock divider configuration for the UART0
 *                            interface; use UART0_CLK_DIV_*_BYTE or an
 *                            unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_UART0(uint8_t config)
{
    CLK_CTRL2->UART0_CLK_DIV_BYTE = (config &
                                CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_UART1(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the UART1 interface
 * Inputs        : config   - Clock divider configuration for the UART1
 *                            interface; use UART1_CLK_DIV_*_BYTE or an
 *                            unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_UART1(uint8_t config)
{
    CLK_CTRL2->UART1_CLK_DIV_BYTE = (config &
                                CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_MCLK(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the MCLK divided clock
 * Inputs        : config   - Clock divider and clock enable configuration for
 *                            MCLK; use MCLK_CLK_ENABLE_BYTE/
 *                            MCLK_CLK_DISABLE_BYTE, and MCLK_CLK_DIV_*_BYTE
 *                            or an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_MCLK(uint8_t config)
{
    CLK_CTRL3->MCLK_DIV_BYTE = (config &
                             ((1U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_ENABLE_Pos)
                             | CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_I2C(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the clock for the I2C interface
 * Inputs        : config   - Clock enable configuration for the I2C
 *                            interface; use I2C_CLK_ENABLE_BYTE/
 *                            I2C_CLK_DISABLE_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_I2C(uint8_t config)
{
    CLK_CTRL3->I2C_CLK_BYTE = (config &
                              (1U << CLK_CTRL_I2C_CLK_BYTE_I2C_CLK_ENABLE_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_PCM(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the PCM interface
 * Inputs        : config   - Clock divider configuration for the PCM
 *                            interface; use PCM_CLK_DIV_*_BYTE or an unshifted
 *                            constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_PCM(uint8_t config)
{
    CLK_CTRL3->PCM_CLK_DIV_BYTE = (config &
                                   CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_GPIO(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for GPIO interface input
 *                 sampling
 * Inputs        : config   - Clock divider configuration for GPIO interface
 *                            input sampling; use GPIO_CLK_DIV_*_BYTE or an
 *                            unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_GPIO(uint8_t config)
{
    CLK_CTRL3->GPIO_CLK_DIV_BYTE = (config &
                                  CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_Core(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clocks for the core system elements
 *                 (ARM Cortex-M3 processor and external clock)
 * Inputs        : config   - Configuration of the divided clocks for the ARM
 *                            Cortex-M3 processor and external clock;
 *                            use CM3_CLK_DIV_*, EXT_CLK_CLK_DIV_*,
 *                            EXT_CLK_ENABLE/EXT_CLK_DISABLE
 *                            and EXT_CLK_DIV2_ENABLE/EXT_CLK_DIV2_DISABLE, or a
 *                            set of shifted constants
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_Core(uint32_t config)
{
    CLK->CTRL4 = (config & (CLK_CTRL4_CM3_CLK_DIV_Mask
                           | CLK_CTRL4_EXT_CLK_DIV_Mask
                           | (1U << CLK_CTRL4_EXT_CLK_DIV2_Pos)
                           | (1U << CLK_CTRL4_EXT_CLK_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_CM3(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the ARM Cortex-M3 processor
 * Inputs        : config   - Clock divider configuration the ARM Cortex-M3
 *                            processor; use CM3_CLK_DIV_*_BYTE or an unshifted
 *                            constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_CM3(uint8_t config)
{
    CLK_CTRL4->CM3_CLK_DIV_BYTE = (config &
                                   CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_EXT_CLK(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock used for the external clock
 * Inputs        : config   - Clock divider and clock enable configuration for
 *                            the external clock; use EXT_CLK_DIV2_ENABLE_BYTE/
 *                            EXT_CLK_DIV2_DISABLE_BYTE, EXT_CLK_ENABLE_BYTE/
 *                            EXT_CLK_DISABLE_BYTE, and EXT_CLK_DIV_*_BYTE or an
 *                            unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_EXT_CLK(uint8_t config)
{
    CLK_CTRL4->EXT_CLK_BYTE= (config &
                             ((1U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV2_Pos)
                             | (1U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_ENABLE_Pos)
                             | CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clocks used for the user clocks
 * Inputs        : config   - Clock divider configuration and enable
 *                            configuration for the three user clocks; use
 *                            USR_CLK*_DIV2_ENABLE/USR_CLK*_DIV2_DISABLE,
 *                            USR_CLK*_ENABLE/USR_CLK*_DISABLE and
 *                            USR_CLK*_DIV_* for each user clock
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK(uint32_t config)
{
    /* To avoid race conditions with a read-modify-write of the entire CLK_CTRL5
     * register, write each of the user clock bytes independently */
    CLK_CTRL5->USR_CLK0_BYTE = (config &
                            ((1U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Mask));

    CLK_CTRL5->USR_CLK1_BYTE = ((config >> CLK_CTRL5_USR_CLK1_DIV_Pos) &
                            ((1U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Mask));

    CLK_CTRL5->USR_CLK2_BYTE = ((config >> CLK_CTRL5_USR_CLK2_DIV_Pos) &
                            ((1U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK0(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock used for user clock 0
 * Inputs        : config   - Clock divider configuration and clock enable
 *                            configuration for user clock 0; use
 *                            USR_CLK0_DIV2_ENABLE_BYTE/
 *                            USR_CLK0_DIV2_DISABLE_BYTE, USR_CLK0_ENABLE_BYTE/
 *                            USR_CLK0_DISABLE_BYTE and USR_CLK0_DIV_*_BYTE or
 *                            an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK0(uint8_t config)
{
    CLK_CTRL5->USR_CLK0_BYTE = (config &
                            ((1U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK1(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock used for user clock 1
 * Inputs        : config   - Clock divider configuration and clock enable
 *                            configuration for user clock 1; use
 *                            USR_CLK1_DIV2_ENABLE_BYTE/
 *                            USR_CLK1_DIV2_DISABLE_BYTE, USR_CLK1_ENABLE_BYTE/
 *                            USR_CLK1_DISABLE_BYTE and USR_CLK1_DIV_*_BYTE or
 *                            an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK1(uint8_t config)
{
    CLK_CTRL5->USR_CLK1_BYTE = (config &
                            ((1U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK2(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock used for user clock 2
 * Inputs        : config   - Clock divider configuration and clock enable
 *                            configuration for user clock 2; use
 *                            USR_CLK2_DIV2_ENABLE_BYTE/
 *                            USR_CLK2_DIV2_DISABLE_BYTE, USR_CLK2_ENABLE_BYTE/
 *                            USR_CLK2_DISABLE_BYTE and USR_CLK2_DIV_*_BYTE or
 *                            an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK2(uint8_t config)
{
    CLK_CTRL5->USR_CLK2_BYTE = (config &
                            ((1U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_Pos)
                            | (1U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_Pos)
                            | CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_LCD(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock used for the LCD driver interface
 * Inputs        : config   - Clock divider and clock enable configuration for
 *                            the LCD driver interface's divided clock; use
 *                            LCD_CLK_ENABLE_BYTE/LCD_CLK_DISABLE_BYTE and
 *                            LCD_CLK_DIV_*_BYTE or an unshifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_LCD(uint8_t config)
{
    CLK_CTRL5->LCD_CLK_BYTE = (config &
                              ((1U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_ENABLE_Pos)
                              | CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_PWM(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clocks for the pulse-width modulators
 * Inputs        : config   - Configuration of the divided clocks for the PWM
 *                            interfaces; use PWM*_CLK_ENABLE/PWM*_CLK_DISABLE
 *                            and PWM*_CLK_DIV_* for each PWM interface
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_PWM(uint32_t config)
{
    CLK->CTRL6 = (config & ((1U << CLK_CTRL6_PWM0_CLK_ENABLE_Pos)
                           | CLK_CTRL6_PWM0_CLK_DIV_Mask
                           | (1U << CLK_CTRL6_PWM1_CLK_ENABLE_Pos)
                           | CLK_CTRL6_PWM1_CLK_DIV_Mask
                           | (1U << CLK_CTRL6_PWM2_CLK_ENABLE_Pos)
                           | CLK_CTRL6_PWM2_CLK_DIV_Mask
                           | (1U << CLK_CTRL6_PWM3_CLK_ENABLE_Pos)
                           | CLK_CTRL6_PWM3_CLK_DIV_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_CP(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the divided clock for the charge pump
 * Inputs        : config   - Configuration of the divided clock for the charge
 *                            pump; use CP_CLK_DIV_* or a shifted constant
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_CP(uint32_t config)
{
    CLK->CTRL7 = (config & CLK_CTRL7_CP_CLK_DIV_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_EXT_CLK_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable the external clock
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_EXT_CLK_Enable()
{
    CLK_CTRL4->EXT_CLK_ENABLE_ALIAS = EXT_CLK_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_EXT_CLK_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable the external clock
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_EXT_CLK_Disable()
{
    CLK_CTRL4->EXT_CLK_ENABLE_ALIAS = EXT_CLK_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK0_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable user clock 0
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK0_Enable()
{
    CLK_CTRL5->USR_CLK0_ENABLE_ALIAS = USR_CLK0_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK0_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable user clock 0
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK0_Disable()
{
    CLK_CTRL5->USR_CLK0_ENABLE_ALIAS = USR_CLK0_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK1_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable user clock 1
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK1_Enable()
{
    CLK_CTRL5->USR_CLK1_ENABLE_ALIAS = USR_CLK1_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK1_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable user clock 1
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK1_Disable()
{
    CLK_CTRL5->USR_CLK1_ENABLE_ALIAS = USR_CLK1_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK2_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable user clock 2
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK2_Enable()
{
    CLK_CTRL5->USR_CLK2_ENABLE_ALIAS = USR_CLK2_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Clk_Config_USR_CLK2_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable user clock 2
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Clk_Config_USR_CLK2_Disable()
{
    CLK_CTRL5->USR_CLK2_ENABLE_ALIAS = USR_CLK2_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_ExternalClockDetect_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the external clock detection circuitry
 * Inputs        : config - Configure external clock detection and clock source
 *                          switching based on this clock detection; use
 *                          CLK_DETECT_ENABLE/CLK_DETECT_DISABLE and
 *                          CLK_DETECT_FORCE_ENABLE/CLK_DETECT_FORCE_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_ExternalClockDetect_Config(uint32_t config)
{
    CLK->DETECT_CTRL = (config &
                       ((1U << CLK_DETECT_CTRL_CLK_DETECT_ENABLE_Pos)
                       | (1U << CLK_DETECT_CTRL_CLK_FORCE_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_ExternalClockDetect_Clear_Status()
 * ----------------------------------------------------------------------------
 * Description   : Clear the external clock detection status bit
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_ExternalClockDetect_Clear_Status()
{
    CLK->DETECT_STATUS = EXT_CLK_DETECT_CLEAR;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_ExternalClockDetect_Get_Status()
 * ----------------------------------------------------------------------------
 * Description   : Get the current external clock detection status bit
 * Inputs        : None
 * Outputs       : status - Current external clock detection status; compare
 *                          against EXT_CLK_NOT_DETECTED/EXT_CLK_DETECTED
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_ExternalClockDetect_Get_Status()
{
    return CLK->DETECT_STATUS;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_CLOCK_H */
