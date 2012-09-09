/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_lcd.h
 * - LCD driver hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_LCD_H
#define Q32_LCD_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * LCD driver support function prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_LCDDriver_SegmentConfig(uint32_t config0,
                                        uint32_t config1,
                                        uint32_t config2,
                                        uint32_t config3);

extern void Sys_LCDDriver_SingleSegmentConfig(uint32_t segmentNumber,
                                              uint32_t config);

/* ----------------------------------------------------------------------------
 * LCD driver support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_LCDDriver_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the LCD driver enable and blanking control
 * Inputs        : config   - Configuration for the LCD driver; use LCD_ENABLE/
 *                            LCD_DISABLE and LCD_BLANK_ENABLE/LCD_BLANK_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_LCDDriver_Config(uint32_t config)
{
    GPIO->IF4_LCD_CTRL = (config & ((1U << GPIO_IF4_LCD_CTRL_LCD_ENABLE_Pos) | 
                                   (1U << GPIO_IF4_LCD_CTRL_BLANK_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_LCDBacklight_Enable(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Enable the LCD backlight driver
 * Inputs        : config   - Backlight configuration; use LCDBACKLIGHT_DISABLE/
 *                            LCDBACKLIGHT_*M**
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_LCDBacklight_Enable(uint32_t config)
{
    /* Use an unaligned half-word write to enable and configure the backlight */
    REG16_POINTER(AFE_PSU_CTRL_BASE
                  + (AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos >> 3))
            = ((config & AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Mask)
                >> AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_LCDBacklight_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable the LCD backlight driver
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_LCDBacklight_Disable()
{
    /* Use an unaligned half-word write to disable the backlight */
    REG16_POINTER(AFE_PSU_CTRL_BASE
                  + (AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos >> 3))
            = (LCDBACKLIGHT_DISABLE >> AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos);
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_LCD_H */
