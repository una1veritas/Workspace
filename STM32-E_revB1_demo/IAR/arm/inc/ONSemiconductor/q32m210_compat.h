/* ----------------------------------------------------------------------------
 * Copyright (c) 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210_compat.h
 * - Peripheral hardware registers and bit-field definitions for backwards
 *   compatibility with earlier EDK and IAR releases
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_COMPAT_H
#define Q32M210_COMPAT_H

/* Include the system library register header */
#include <q32_reg.h>

/* ----------------------------------------------------------------------------
 * Analog Front-End
 * ------------------------------------------------------------------------- */
/* Control and configuration of the analog system components. This block
 * includes the clock, power supply, switch (both SPST switches and multi-
 * switches), input and output control and configuration. */

/* Power Supply Charge Pump and LCD Backlight Control */
/*   Configure the charge pump and LCD backlight */
#define AFE_PSU_CTRL                    REG32_POINTER(AFE_PSU_CTRL_BASE)

/* AFE_PSU_CTRL bit positions (legacy definitions) */
#define AFE_PSU_CTRL_LCDBACKLIGHT_CFG_POS 8
#define AFE_PSU_CTRL_LCDBACKLIGHT_CFG_MASK ((uint32_t)(0x3FFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_POS))
#define AFE_PSU_CTRL_VCP_ENABLE_POS     2
#define AFE_PSU_CTRL_LCDDRIVER_ENABLE_POS 1
#define AFE_PSU_CTRL_VDBL_ENABLE_POS    0

/* AFE_PSU_CTRL bit-band aliases */
#define AFE_PSU_CTRL_VCP_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_PSU_CTRL_BASE, AFE_PSU_CTRL_VCP_ENABLE_POS))
#define AFE_PSU_CTRL_LCDDRIVER_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_PSU_CTRL_BASE, AFE_PSU_CTRL_LCDDRIVER_ENABLE_POS))
#define AFE_PSU_CTRL_VDBL_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_PSU_CTRL_BASE, AFE_PSU_CTRL_VDBL_ENABLE_POS))

/* Operation Control */
/*   Control the operating mode features */
#define AFE_OPMODE_CTRL                 REG32_POINTER(AFE_OPMODE_CTRL_BASE)

/* AFE_OPMODE_CTRL bit positions (legacy definitions) */
#define AFE_OPMODE_CTRL_RC_OSC_ENABLE_POS 3
#define AFE_OPMODE_CTRL_VADC_ENABLE_POS 2
#define AFE_OPMODE_CTRL_STANDBY_MODE_POS 1
#define AFE_OPMODE_CTRL_SLEEP_MODE_POS  0

/* AFE_OPMODE_CTRL bit-band aliases */
#define AFE_OPMODE_CTRL_RC_OSC_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OPMODE_CTRL_BASE, AFE_OPMODE_CTRL_RC_OSC_ENABLE_POS))
#define AFE_OPMODE_CTRL_VADC_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OPMODE_CTRL_BASE, AFE_OPMODE_CTRL_VADC_ENABLE_POS))
#define AFE_OPMODE_CTRL_STANDBY_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OPMODE_CTRL_BASE, AFE_OPMODE_CTRL_STANDBY_MODE_POS))
#define AFE_OPMODE_CTRL_SLEEP_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OPMODE_CTRL_BASE, AFE_OPMODE_CTRL_SLEEP_MODE_POS))

/* 32 kHz Crystal Clock Calibration Control */
/*   Trimming of 32 kHz crystal frequency */
#define AFE_32K_CCR_CTRL                REG32_POINTER(AFE_32K_CCR_CTRL_BASE)

/* AFE_32K_CCR_CTRL bit positions (legacy definitions) */
#define AFE_32K_CCR_CTRL_CCR_CTRL_POS   0
#define AFE_32K_CCR_CTRL_CCR_CTRL_MASK  ((uint32_t)(0xFFU << AFE_32K_CCR_CTRL_CCR_CTRL_POS))

/* 48 MHz Crystal Clock Calibration Control */
/*   Trimming of 48 MHz crystal frequency */
#define AFE_48M_CCR_CTRL                REG32_POINTER(AFE_48M_CCR_CTRL_BASE)

/* AFE_48M_CCR_CTRL bit positions (legacy definitions) */
#define AFE_48M_CCR_CTRL_CCR_CTRL_POS   0
#define AFE_48M_CCR_CTRL_CCR_CTRL_MASK  ((uint32_t)(0xFFU << AFE_48M_CCR_CTRL_CCR_CTRL_POS))

/* RC Clock Calibration Control */
/*   Trimming of RC oscillator frequency */
#define AFE_RC_CCR_CTRL                 REG32_POINTER(AFE_RC_CCR_CTRL_BASE)

/* AFE_RC_CCR_CTRL bit positions (legacy definitions) */
#define AFE_RC_CCR_CTRL_FINE_CTRL_POS   16
#define AFE_RC_CCR_CTRL_FINE_CTRL_MASK  ((uint32_t)(0x3FU << AFE_RC_CCR_CTRL_FINE_CTRL_POS))
#define AFE_RC_CCR_CTRL_RANGE_SEL_POS   8
#define AFE_RC_CCR_CTRL_RANGE_SEL_MASK  ((uint32_t)(0x7U << AFE_RC_CCR_CTRL_RANGE_SEL_POS))
#define AFE_RC_CCR_CTRL_COARSE_CTRL_POS 0
#define AFE_RC_CCR_CTRL_COARSE_CTRL_MASK ((uint32_t)(0x3FU << AFE_RC_CCR_CTRL_COARSE_CTRL_POS))

/* Input Control and Configuration */
/*   Configuration of the inputs to operational amplifier A0 and the ALT
 *   pads */
#define AFE_IN_SW_CTRL                  REG32_POINTER(AFE_IN_SW_CTRL_BASE)

/* AFE_IN_SW_CTRL bit positions (legacy definitions) */
#define AFE_IN_SW_CTRL_ALT1_SW_POS      8
#define AFE_IN_SW_CTRL_ALT1_SW_MASK     ((uint32_t)(0xFU << AFE_IN_SW_CTRL_ALT1_SW_POS))
#define AFE_IN_SW_CTRL_ALT0_SW_POS      4
#define AFE_IN_SW_CTRL_ALT0_SW_MASK     ((uint32_t)(0x7U << AFE_IN_SW_CTRL_ALT0_SW_POS))
#define AFE_IN_SW_CTRL_A0_IN_CFG_POS    0
#define AFE_IN_SW_CTRL_A0_IN_CFG_MASK   ((uint32_t)(0xFU << AFE_IN_SW_CTRL_A0_IN_CFG_POS))

/* Opamp Control */
/*   Enabling of the operational amplifiers A0 to A2 */
#define AFE_AMP_CTRL                    REG32_POINTER(AFE_AMP_CTRL_BASE)

/* AFE_AMP_CTRL bit positions (legacy definitions) */
#define AFE_AMP_CTRL_A2_ENABLE_POS      2
#define AFE_AMP_CTRL_A1_ENABLE_POS      1
#define AFE_AMP_CTRL_A0_ENABLE_POS      0

/* AFE_AMP_CTRL bit-band aliases */
#define AFE_AMP_CTRL_A2_ENABLE_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(AFE_AMP_CTRL_BASE, AFE_AMP_CTRL_A2_ENABLE_POS))
#define AFE_AMP_CTRL_A1_ENABLE_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(AFE_AMP_CTRL_BASE, AFE_AMP_CTRL_A1_ENABLE_POS))
#define AFE_AMP_CTRL_A0_ENABLE_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(AFE_AMP_CTRL_BASE, AFE_AMP_CTRL_A0_ENABLE_POS))

/* Output Control and Configuration */
/*   Output control and configuration for operational amplifiers A0 to A2 */
#define AFE_OUT_SW_CTRL                 REG32_POINTER(AFE_OUT_SW_CTRL_BASE)

/* AFE_OUT_SW_CTRL bit positions (legacy definitions) */
#define AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_POS 5
#define AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_POS 4
#define AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_POS 3
#define AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_POS 2
#define AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_POS 1
#define AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_POS 0

/* AFE_OUT_SW_CTRL bit-band aliases */
#define AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_POS))
#define AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_POS))
#define AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_POS))
#define AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_POS))
#define AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_POS))
#define AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_POS))

/* SPST Switch Control */
/*   Control of SPST switches */
#define AFE_SPST_CTRL                   REG32_POINTER(AFE_SPST_CTRL_BASE)

/* AFE_SPST_CTRL bit positions (legacy definitions) */
#define AFE_SPST_CTRL_SPST3_DISABLE_POS 7
#define AFE_SPST_CTRL_SPST3_SELECT_POS  6
#define AFE_SPST_CTRL_SPST2_DISABLE_POS 5
#define AFE_SPST_CTRL_SPST2_SELECT_POS  4
#define AFE_SPST_CTRL_SPST1_DISABLE_POS 3
#define AFE_SPST_CTRL_SPST1_SELECT_POS  2
#define AFE_SPST_CTRL_SPST0_DISABLE_POS 1
#define AFE_SPST_CTRL_SPST0_SELECT_POS  0

/* AFE_SPST_CTRL bit-band aliases */
#define AFE_SPST_CTRL_SPST3_DISABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST3_DISABLE_POS))
#define AFE_SPST_CTRL_SPST3_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST3_SELECT_POS))
#define AFE_SPST_CTRL_SPST2_DISABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST2_DISABLE_POS))
#define AFE_SPST_CTRL_SPST2_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST2_SELECT_POS))
#define AFE_SPST_CTRL_SPST1_DISABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST1_DISABLE_POS))
#define AFE_SPST_CTRL_SPST1_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST1_SELECT_POS))
#define AFE_SPST_CTRL_SPST0_DISABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST0_DISABLE_POS))
#define AFE_SPST_CTRL_SPST0_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, AFE_SPST_CTRL_SPST0_SELECT_POS))

/* Multi-Switch Control */
/*   Multi-Switch control and configuration */
#define AFE_MSW_CTRL                    REG32_POINTER(AFE_MSW_CTRL_BASE)

/* AFE_MSW_CTRL bit positions (legacy definitions) */
#define AFE_MSW_CTRL_MSW3_SELECT_POS    11
#define AFE_MSW_CTRL_MSW3_SELECT_MASK   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW3_SELECT_POS))
#define AFE_MSW_CTRL_MSW2_SELECT_POS    8
#define AFE_MSW_CTRL_MSW2_SELECT_MASK   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW2_SELECT_POS))
#define AFE_MSW_CTRL_MSW1_SELECT_POS    3
#define AFE_MSW_CTRL_MSW1_SELECT_MASK   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW1_SELECT_POS))
#define AFE_MSW_CTRL_MSW0_SELECT_POS    0
#define AFE_MSW_CTRL_MSW0_SELECT_MASK   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW0_SELECT_POS))

/* Programmable Gain Amplifier 0 Control */
/*   Configure PGA0 and select the inputs for this programmable gain
 *   amplifier */
#define AFE_PGA0_CTRL                   REG32_POINTER(AFE_PGA0_CTRL_BASE)

/* AFE_PGA0_CTRL bit positions (legacy definitions) */
#define AFE_PGA0_CTRL_PGA0_DIF_SELECT_POS 5
#define AFE_PGA0_CTRL_PGA0_DIF_SELECT_MASK ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_POS))
#define AFE_PGA0_CTRL_PGA0_SELECT_POS   1
#define AFE_PGA0_CTRL_PGA0_SELECT_MASK  ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_SELECT_POS))
#define AFE_PGA0_CTRL_PGA0_ENABLE_POS   0

/* AFE_PGA0_CTRL bit-band aliases */
#define AFE_PGA0_CTRL_PGA0_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_PGA0_CTRL_BASE, AFE_PGA0_CTRL_PGA0_ENABLE_POS))

/* Programmable Gain Amplifier 1 Control */
/*   Configure PGA1 and select the inputs for this programmable gain
 *   amplifier */
#define AFE_PGA1_CTRL                   REG32_POINTER(AFE_PGA1_CTRL_BASE)

/* AFE_PGA1_CTRL bit positions (legacy definitions) */
#define AFE_PGA1_CTRL_PGA1_DIF_SELECT_POS 5
#define AFE_PGA1_CTRL_PGA1_DIF_SELECT_MASK ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_POS))
#define AFE_PGA1_CTRL_PGA1_SELECT_POS   1
#define AFE_PGA1_CTRL_PGA1_SELECT_MASK  ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_SELECT_POS))
#define AFE_PGA1_CTRL_PGA1_ENABLE_POS   0

/* AFE_PGA1_CTRL bit-band aliases */
#define AFE_PGA1_CTRL_PGA1_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_PGA1_CTRL_BASE, AFE_PGA1_CTRL_PGA1_ENABLE_POS))

/* Programmable Gain Amplifier Configuration and Control */
/*   Configure the cut-off frequency and gain settings for PGA0, PGA1 */
#define AFE_PGA_GAIN_CTRL               REG32_POINTER(AFE_PGA_GAIN_CTRL_BASE)

/* AFE_PGA_GAIN_CTRL bit positions (legacy definitions) */
#define AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_POS 6
#define AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_MASK ((uint32_t)(0x3U << AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_POS))
#define AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_POS 3
#define AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_MASK ((uint32_t)(0x7U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_POS))
#define AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_POS 0
#define AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_MASK ((uint32_t)(0x7U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_POS))

/* Voltage Threshold Comparator Control */
/*   Configure and control the voltage comparator */
#define AFE_THRESHOLD_COMPARE_CTRL      REG32_POINTER(AFE_THRESHOLD_COMPARE_CTRL_BASE)

/* AFE_THRESHOLD_COMPARE_CTRL bit positions (legacy definitions) */
#define AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_POS 0
#define AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_MASK ((uint32_t)(0x3U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_POS))

/* ADC Configuration and Control */
/*   Configure and control the analog to digital convertors */
#define AFE_ADC_CTRL                    REG32_POINTER(AFE_ADC_CTRL_BASE)

/* AFE_ADC_CTRL bit positions (legacy definitions) */
#define AFE_ADC_CTRL_ADC1_GAIN_ENABLE_POS 7
#define AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_POS 6
#define AFE_ADC_CTRL_ADC0_GAIN_ENABLE_POS 5
#define AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_POS 4
#define AFE_ADC_CTRL_ADC1_FORMAT_CFG_POS 3
#define AFE_ADC_CTRL_ADC0_FORMAT_CFG_POS 2
#define AFE_ADC_CTRL_ADC1_ENABLE_POS    1
#define AFE_ADC_CTRL_ADC0_ENABLE_POS    0

/* AFE_ADC_CTRL bit-band aliases */
#define AFE_ADC_CTRL_ADC1_GAIN_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC1_GAIN_ENABLE_POS))
#define AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_POS))
#define AFE_ADC_CTRL_ADC0_GAIN_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC0_GAIN_ENABLE_POS))
#define AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_POS))
#define AFE_ADC_CTRL_ADC1_FORMAT_CFG_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC1_FORMAT_CFG_POS))
#define AFE_ADC_CTRL_ADC0_FORMAT_CFG_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC0_FORMAT_CFG_POS))
#define AFE_ADC_CTRL_ADC1_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC1_ENABLE_POS))
#define AFE_ADC_CTRL_ADC0_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_ADC_CTRL_BASE, AFE_ADC_CTRL_ADC0_ENABLE_POS))

/* AFE_ADC_CTRL subregister pointers */
#define ADC_CTRL_BYTE                   REG8_POINTER(AFE_ADC_CTRL_BASE + 0)

/* AFE_ADC_CTRL subregister bit positions */
#define ADC_CTRL_BYTE_ADC1_GAIN_ENABLE_POS 7
#define ADC_CTRL_BYTE_ADC1_OFFSET_ENABLE_POS 6
#define ADC_CTRL_BYTE_ADC0_GAIN_ENABLE_POS 5
#define ADC_CTRL_BYTE_ADC0_OFFSET_ENABLE_POS 4
#define ADC_CTRL_BYTE_ADC1_FORMAT_CFG_POS 3
#define ADC_CTRL_BYTE_ADC0_FORMAT_CFG_POS 2
#define ADC_CTRL_BYTE_ADC1_ENABLE_POS   1
#define ADC_CTRL_BYTE_ADC0_ENABLE_POS   0

/* ADC0 Output */
/*   ADC0 converter output */
#define AFE_ADC0_DATA                   READONLY_REG32_POINTER(AFE_ADC0_DATA_BASE)

/* AFE_ADC0_DATA bit positions (legacy definitions) */
#define AFE_ADC0_DATA_ALIGN_POS         16
#define AFE_ADC0_DATA_ALIGN_MASK        ((uint32_t)(0xFFFFU << AFE_ADC0_DATA_ALIGN_POS))

/* ADC1 Output */
/*   ADC1 converter output */
#define AFE_ADC1_DATA                   READONLY_REG32_POINTER(AFE_ADC1_DATA_BASE)

/* AFE_ADC1_DATA bit positions (legacy definitions) */
#define AFE_ADC1_DATA_ALIGN_POS         16
#define AFE_ADC1_DATA_ALIGN_MASK        ((uint32_t)(0xFFFFU << AFE_ADC1_DATA_ALIGN_POS))

/* ADC 0 and 1 Shared Output */
/*   16 MSBs of the ADC0 and ADC1 converter output */
#define AFE_ADC01_DATA                  READONLY_REG32_POINTER(AFE_ADC01_DATA_BASE)

/* AFE_ADC01_DATA bit positions (legacy definitions) */
#define AFE_ADC01_DATA_ADC0_POS         16
#define AFE_ADC01_DATA_ADC0_MASK        ((uint32_t)(0xFFFFU << AFE_ADC01_DATA_ADC0_POS))
#define AFE_ADC01_DATA_ADC1_POS         0
#define AFE_ADC01_DATA_ADC1_MASK        ((uint32_t)(0xFFFFU << AFE_ADC01_DATA_ADC1_POS))

/* AFE_ADC01_DATA subregister pointers */
#define ADC0_DATA_SHORT                 READONLY_REG16_POINTER(AFE_ADC01_DATA_BASE + 2)
#define ADC1_DATA_SHORT                 READONLY_REG16_POINTER(AFE_ADC01_DATA_BASE + 0)

/* ADC0 Offset Calibration */
/*   ADC0 offset calibration */
#define AFE_ADC0_OFFSET                 REG32_POINTER(AFE_ADC0_OFFSET_BASE)

/* AFE_ADC0_OFFSET bit positions (legacy definitions) */
#define AFE_ADC0_OFFSET_CAL_POS         14
#define AFE_ADC0_OFFSET_CAL_MASK        ((uint32_t)(0x3FFFFU << AFE_ADC0_OFFSET_CAL_POS))

/* ADC1  Offset Calibration */
/*   ADC1 offset calibration */
#define AFE_ADC1_OFFSET                 REG32_POINTER(AFE_ADC1_OFFSET_BASE)

/* AFE_ADC1_OFFSET bit positions (legacy definitions) */
#define AFE_ADC1_OFFSET_CAL_POS         14
#define AFE_ADC1_OFFSET_CAL_MASK        ((uint32_t)(0x3FFFFU << AFE_ADC1_OFFSET_CAL_POS))

/* ADC0 Gain Calibration */
/*   ADC0 gain calibration */
#define AFE_ADC0_GAIN                   REG32_POINTER(AFE_ADC0_GAIN_BASE)

/* AFE_ADC0_GAIN bit positions (legacy definitions) */
#define AFE_ADC0_GAIN_CAL_POS           14
#define AFE_ADC0_GAIN_CAL_MASK          ((uint32_t)(0x3FFFFU << AFE_ADC0_GAIN_CAL_POS))

/* ADC1  Gain Calibration */
/*   ADC1 gain calibration */
#define AFE_ADC1_GAIN                   REG32_POINTER(AFE_ADC1_GAIN_BASE)

/* AFE_ADC1_GAIN bit positions (legacy definitions) */
#define AFE_ADC1_GAIN_CAL_POS           14
#define AFE_ADC1_GAIN_CAL_MASK          ((uint32_t)(0x3FFFFU << AFE_ADC1_GAIN_CAL_POS))

/* ADC Data Rate Configuration */
/*   Configure the ADC data rate including the decimation factor */
#define AFE_DATARATE_CFG                REG32_POINTER(AFE_DATARATE_CFG_BASE)

/* AFE_DATARATE_CFG bit positions (legacy definitions) */
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_POS 25
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_MASK ((uint32_t)(0x3FU << AFE_DATARATE_CFG_DECIMATION_FACTOR_POS))
#define AFE_DATARATE_CFG_DECIMATION_ENABLE_POS 24
#define AFE_DATARATE_CFG_ADJUST_POS     0
#define AFE_DATARATE_CFG_ADJUST_MASK    ((uint32_t)(0x3FFU << AFE_DATARATE_CFG_ADJUST_POS))

/* AFE_DATARATE_CFG bit-band aliases */
#define AFE_DATARATE_CFG_DECIMATION_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_DATARATE_CFG_BASE, AFE_DATARATE_CFG_DECIMATION_ENABLE_POS))

/* AFE_DATARATE_CFG subregister pointers */
#define DECIMATION_FACTOR_BYTE          REG8_POINTER(AFE_DATARATE_CFG_BASE + 3)
#define DATARATE_ADJUST_SHORT           REG16_POINTER(AFE_DATARATE_CFG_BASE + 0)

/* AFE_DATARATE_CFG subregister bit positions */
#define DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_POS 1
#define DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_MASK ((uint32_t)(0x3FU << DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_POS))
#define DECIMATION_FACTOR_BYTE_DECIMATION_ENABLE_POS 0
#define DATARATE_ADJUST_SHORT_ADJUST_POS 0
#define DATARATE_ADJUST_SHORT_ADJUST_MASK ((uint32_t)(0x3FFU << DATARATE_ADJUST_SHORT_ADJUST_POS))

/* AFE Data Rate Mode Configuration */
/*   Configure the oversampling mode used by the ADCs */
#define AFE_DATARATE_CFG1               REG32_POINTER(AFE_DATARATE_CFG1_BASE)

/* AFE_DATARATE_CFG1 bit positions (legacy definitions) */
#define AFE_DATARATE_CFG1_FREQ_MODE_POS 0
#define AFE_DATARATE_CFG1_FREQ_MODE_MASK ((uint32_t)(0xFU << AFE_DATARATE_CFG1_FREQ_MODE_POS))

/* PWM0 Period Configuration */
/*   Set the legnth of each PWM0 period */
#define AFE_PWM0_PERIOD                 REG32_POINTER(AFE_PWM0_PERIOD_BASE)

/* AFE_PWM0_PERIOD bit positions (legacy definitions) */
#define AFE_PWM0_PERIOD_PWM0_PERIOD_POS 0
#define AFE_PWM0_PERIOD_PWM0_PERIOD_MASK ((uint32_t)(0xFFU << AFE_PWM0_PERIOD_PWM0_PERIOD_POS))

/* PWM1 Period Configuration */
/*   Set the legnth of each PWM1 period */
#define AFE_PWM1_PERIOD                 REG32_POINTER(AFE_PWM1_PERIOD_BASE)

/* AFE_PWM1_PERIOD bit positions (legacy definitions) */
#define AFE_PWM1_PERIOD_PWM1_PERIOD_POS 0
#define AFE_PWM1_PERIOD_PWM1_PERIOD_MASK ((uint32_t)(0xFFU << AFE_PWM1_PERIOD_PWM1_PERIOD_POS))

/* PWM2  Period Configuration */
/*   Set the legnth of each PWM2 period */
#define AFE_PWM2_PERIOD                 REG32_POINTER(AFE_PWM2_PERIOD_BASE)

/* AFE_PWM2_PERIOD bit positions (legacy definitions) */
#define AFE_PWM2_PERIOD_PWM2_PERIOD_POS 0
#define AFE_PWM2_PERIOD_PWM2_PERIOD_MASK ((uint32_t)(0xFFU << AFE_PWM2_PERIOD_PWM2_PERIOD_POS))

/* PWM3 Period Configuration */
/*   Set the legnth of each PWM3 period */
#define AFE_PWM3_PERIOD                 REG32_POINTER(AFE_PWM3_PERIOD_BASE)

/* AFE_PWM3_PERIOD bit positions (legacy definitions) */
#define AFE_PWM3_PERIOD_PWM3_PERIOD_POS 0
#define AFE_PWM3_PERIOD_PWM3_PERIOD_MASK ((uint32_t)(0xFFU << AFE_PWM3_PERIOD_PWM3_PERIOD_POS))

/* PWM0 Duty Cycle Configuration */
/*   Configure the PWM0 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM0_HI                     REG32_POINTER(AFE_PWM0_HI_BASE)

/* AFE_PWM0_HI bit positions (legacy definitions) */
#define AFE_PWM0_HI_PWM0_HI_POS         0
#define AFE_PWM0_HI_PWM0_HI_MASK        ((uint32_t)(0xFFU << AFE_PWM0_HI_PWM0_HI_POS))

/* PWM1 Duty Cycle Configuration */
/*   Configure the PWM1 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM1_HI                     REG32_POINTER(AFE_PWM1_HI_BASE)

/* AFE_PWM1_HI bit positions (legacy definitions) */
#define AFE_PWM1_HI_PWM1_HI_POS         0
#define AFE_PWM1_HI_PWM1_HI_MASK        ((uint32_t)(0xFFU << AFE_PWM1_HI_PWM1_HI_POS))

/* PWM2  Duty Cycle Configuration */
/*   Configure the PWM2 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM2_HI                     REG32_POINTER(AFE_PWM2_HI_BASE)

/* AFE_PWM2_HI bit positions (legacy definitions) */
#define AFE_PWM2_HI_PWM2_HI_POS         0
#define AFE_PWM2_HI_PWM2_HI_MASK        ((uint32_t)(0xFFU << AFE_PWM2_HI_PWM2_HI_POS))

/* PWM3 Duty Cycle Configuration */
/*   Configure the PWM3 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM3_HI                     REG32_POINTER(AFE_PWM3_HI_BASE)

/* AFE_PWM3_HI bit positions (legacy definitions) */
#define AFE_PWM3_HI_PWM3_HI_POS         0
#define AFE_PWM3_HI_PWM3_HI_MASK        ((uint32_t)(0xFFU << AFE_PWM3_HI_PWM3_HI_POS))

/* Temperature Sensor Control and Configuration */
/*   Temperature sensor control */
#define AFE_TEMP_SENSE_CTRL             REG32_POINTER(AFE_TEMP_SENSE_CTRL_BASE)

/* AFE_TEMP_SENSE_CTRL bit positions (legacy definitions) */
#define AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_POS 0

/* AFE_TEMP_SENSE_CTRL bit-band aliases */
#define AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_TEMP_SENSE_CTRL_BASE, AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_POS))

/* DAC0 Voltage Reference control */
/*   Select voltage references for DAC0 */
#define AFE_DAC0_REF_CTRL               REG32_POINTER(AFE_DAC0_REF_CTRL_BASE)

/* AFE_DAC0_REF_CTRL bit positions (legacy definitions) */
#define AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_POS 0
#define AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_MASK ((uint32_t)(0x3U << AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_POS))

/* DAC Control and Configuration */
/*   DAC Control and Configuration */
#define AFE_DAC_CTRL                    REG32_POINTER(AFE_DAC_CTRL_BASE)

/* AFE_DAC_CTRL bit positions (legacy definitions) */
#define AFE_DAC_CTRL_MODE_SELECT_B_POS  19
#define AFE_DAC_CTRL_MODE_SELECT_B_MASK ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_B_POS))
#define AFE_DAC_CTRL_MODE_SELECT_A_POS  16
#define AFE_DAC_CTRL_MODE_SELECT_A_MASK ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_A_POS))
#define AFE_DAC_CTRL_DAC2_CUR_POS       6
#define AFE_DAC_CTRL_DAC2_CUR_MASK      ((uint32_t)(0x3U << AFE_DAC_CTRL_DAC2_CUR_POS))
#define AFE_DAC_CTRL_DAC2_ENABLE_POS    4
#define AFE_DAC_CTRL_DAC1_ENABLE_POS    2
#define AFE_DAC_CTRL_DAC0_ENABLE_POS    0

/* AFE_DAC_CTRL bit-band aliases */
#define AFE_DAC_CTRL_DAC2_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_DAC_CTRL_BASE, AFE_DAC_CTRL_DAC2_ENABLE_POS))
#define AFE_DAC_CTRL_DAC1_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_DAC_CTRL_BASE, AFE_DAC_CTRL_DAC1_ENABLE_POS))
#define AFE_DAC_CTRL_DAC0_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_DAC_CTRL_BASE, AFE_DAC_CTRL_DAC0_ENABLE_POS))

/* AFE_DAC_CTRL subregister pointers */
#define PGA1_MODE_CTRL_BYTE             REG8_POINTER(AFE_DAC_CTRL_BASE + 2)
#define DAC_CTRL_BYTE                   REG8_POINTER(AFE_DAC_CTRL_BASE + 0)

/* AFE_DAC_CTRL subregister bit positions */
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_POS 3
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_MASK ((uint32_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_POS))
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_POS 0
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_MASK ((uint32_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_POS))
#define DAC_CTRL_BYTE_DAC2_CUR_POS      6
#define DAC_CTRL_BYTE_DAC2_CUR_MASK     ((uint32_t)(0x3U << DAC_CTRL_BYTE_DAC2_CUR_POS))
#define DAC_CTRL_BYTE_DAC2_ENABLE_POS   4
#define DAC_CTRL_BYTE_DAC1_ENABLE_POS   2
#define DAC_CTRL_BYTE_DAC0_ENABLE_POS   0

/* DAC0 Input Data */
/*   DAC0 converter input data */
#define AFE_DAC0_DATA                   REG32_POINTER(AFE_DAC0_DATA_BASE)

/* AFE_DAC0_DATA bit positions (legacy definitions) */
#define AFE_DAC0_DATA_DAC0_DATA_POS     0
#define AFE_DAC0_DATA_DAC0_DATA_MASK    ((uint32_t)(0x3FFU << AFE_DAC0_DATA_DAC0_DATA_POS))

/* DAC1 Input Data */
/*   DAC1 converter input data */
#define AFE_DAC1_DATA                   REG32_POINTER(AFE_DAC1_DATA_BASE)

/* AFE_DAC1_DATA bit positions (legacy definitions) */
#define AFE_DAC1_DATA_DAC1_DATA_POS     0
#define AFE_DAC1_DATA_DAC1_DATA_MASK    ((uint32_t)(0x3FFU << AFE_DAC1_DATA_DAC1_DATA_POS))

/* DAC2 Input Data */
/*   DAC2 converter input data */
#define AFE_DAC2_DATA                   REG32_POINTER(AFE_DAC2_DATA_BASE)

/* AFE_DAC2_DATA bit positions (legacy definitions) */
#define AFE_DAC2_DATA_DAC2_DATA_POS     0
#define AFE_DAC2_DATA_DAC2_DATA_MASK    ((uint32_t)(0x3FFU << AFE_DAC2_DATA_DAC2_DATA_POS))

/* DAC 0 and DAC 1 Input Data */
/*   Input data for the DAC0 and DAC1 converters */
#define AFE_DAC01_DATA                  REG32_POINTER(AFE_DAC01_DATA_BASE)

/* AFE_DAC01_DATA bit positions (legacy definitions) */
#define AFE_DAC01_DATA_DAC0_DATA_POS    16
#define AFE_DAC01_DATA_DAC0_DATA_MASK   ((uint32_t)(0x3FFU << AFE_DAC01_DATA_DAC0_DATA_POS))
#define AFE_DAC01_DATA_DAC1_DATA_POS    0
#define AFE_DAC01_DATA_DAC1_DATA_MASK   ((uint32_t)(0x3FFU << AFE_DAC01_DATA_DAC1_DATA_POS))

/* RTC Clock Control */
/*   Control the real-time clock behavior */
#define AFE_RTC_CTRL                    REG32_POINTER(AFE_RTC_CTRL_BASE)

/* AFE_RTC_CTRL bit positions (legacy definitions) */
#define AFE_RTC_CTRL_RTC_LOAD_POS       3
#define AFE_RTC_CTRL_RTC_BIAS_ENABLE_POS 2
#define AFE_RTC_CTRL_ALARM_ENABLE_POS   1
#define AFE_RTC_CTRL_RTC_MODE_CFG_POS   0

/* AFE_RTC_CTRL bit-band aliases */
#define AFE_RTC_CTRL_RTC_LOAD_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(AFE_RTC_CTRL_BASE, AFE_RTC_CTRL_RTC_LOAD_POS))
#define AFE_RTC_CTRL_RTC_BIAS_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_RTC_CTRL_BASE, AFE_RTC_CTRL_RTC_BIAS_ENABLE_POS))
#define AFE_RTC_CTRL_ALARM_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_RTC_CTRL_BASE, AFE_RTC_CTRL_ALARM_ENABLE_POS))
#define AFE_RTC_CTRL_RTC_MODE_CFG_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_RTC_CTRL_BASE, AFE_RTC_CTRL_RTC_MODE_CFG_POS))

/* RTC Count */
/*   Current count for the real-time clock */
#define AFE_RTC_COUNT                   REG32_POINTER(AFE_RTC_COUNT_BASE)

/* AFE_RTC_COUNT bit positions (legacy definitions) */
#define AFE_RTC_COUNT_RTC_COUNT_POS     0
#define AFE_RTC_COUNT_RTC_COUNT_MASK    ((uint32_t)(0xFFFFFFFFU << AFE_RTC_COUNT_RTC_COUNT_POS))

/* RTC Alarm Count */
/*   Alarm setting for the real-time clock */
#define AFE_RTC_ALARM                   REG32_POINTER(AFE_RTC_ALARM_BASE)

/* AFE_RTC_ALARM bit positions (legacy definitions) */
#define AFE_RTC_ALARM_RTC_ALARM_MODE_POS 31
#define AFE_RTC_ALARM_RTC_ALARM_POS     0
#define AFE_RTC_ALARM_RTC_ALARM_MASK    ((uint32_t)(0x7FFFFFFFU << AFE_RTC_ALARM_RTC_ALARM_POS))

/* AFE_RTC_ALARM bit-band aliases */
#define AFE_RTC_ALARM_RTC_ALARM_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_RTC_ALARM_BASE, AFE_RTC_ALARM_RTC_ALARM_MODE_POS))

/* Analog Interrupt Status */
/*   Status for the interrupts initiating in the analog domain */
#define AFE_INTERRUPT_STATUS            REG32_POINTER(AFE_INTERRUPT_STATUS_BASE)

/* AFE_INTERRUPT_STATUS bit positions (legacy definitions) */
#define AFE_INTERRUPT_STATUS_WAKEUP_INT_POS 7
#define AFE_INTERRUPT_STATUS_IF5_PIN3_POS 6
#define AFE_INTERRUPT_STATUS_IF5_PIN2_POS 5
#define AFE_INTERRUPT_STATUS_IF5_PIN1_POS 4
#define AFE_INTERRUPT_STATUS_RTC_CLOCK_POS 3
#define AFE_INTERRUPT_STATUS_RTC_ALARM_POS 2
#define AFE_INTERRUPT_STATUS_IF5_PIN0_POS 1
#define AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_POS 0

/* AFE_INTERRUPT_STATUS bit-band aliases */
#define AFE_INTERRUPT_STATUS_WAKEUP_INT_BITBAND READONLY_REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_WAKEUP_INT_POS))
#define AFE_INTERRUPT_STATUS_IF5_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_IF5_PIN3_POS))
#define AFE_INTERRUPT_STATUS_IF5_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_IF5_PIN2_POS))
#define AFE_INTERRUPT_STATUS_IF5_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_IF5_PIN1_POS))
#define AFE_INTERRUPT_STATUS_RTC_CLOCK_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_RTC_CLOCK_POS))
#define AFE_INTERRUPT_STATUS_RTC_ALARM_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_RTC_ALARM_POS))
#define AFE_INTERRUPT_STATUS_IF5_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_IF5_PIN0_POS))
#define AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_POS))

/* General-Purpose Data Retention */
/*   General-purpose data retention register, used to maintain one data value
 *   while in sleep mode */
#define AFE_RETENTION_DATA              REG32_POINTER(AFE_RETENTION_DATA_BASE)

/* AFE_RETENTION_DATA bit positions (legacy definitions) */
#define AFE_RETENTION_DATA_RETENTION_DATA_POS 0
#define AFE_RETENTION_DATA_RETENTION_DATA_MASK ((uint32_t)(0xFFU << AFE_RETENTION_DATA_RETENTION_DATA_POS))

/* AFE_RETENTION_DATA subregister pointers */
#define RETENTION_DATA_BYTE             REG8_POINTER(AFE_RETENTION_DATA_BASE + 0)

/* ----------------------------------------------------------------------------
 * DMA Controller Configuration and Control
 * ------------------------------------------------------------------------- */
/* The direct memory access (DMA) controller provides data transfers between
 * peripherals and memory, peripherals and perihperals, and memory-to-memory
 * copy operations.
 *
 * This controller operates in the background without core intervention,
 * allowing the core to be used for other computational needs while allowing
 * high speed sustained transfers to and from the peripherals. */

/* DMA Channel 0 Control and Configuration */
/*   Control and configuration of DMA channel 0, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH0_CTRL0                   REG32_POINTER(DMA_CH0_CTRL0_BASE)

/* DMA_CH0_CTRL0 bit positions (legacy definitions) */
#define DMA_CH0_CTRL0_BYTE_ORDER_POS    25
#define DMA_CH0_CTRL0_DISABLE_INT_ENABLE_POS 24
#define DMA_CH0_CTRL0_ERROR_INT_ENABLE_POS 23
#define DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_POS 22
#define DMA_CH0_CTRL0_COUNTER_INT_ENABLE_POS 21
#define DMA_CH0_CTRL0_START_INT_ENABLE_POS 20
#define DMA_CH0_CTRL0_DEST_WORD_SIZE_POS 18
#define DMA_CH0_CTRL0_DEST_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH0_CTRL0_DEST_WORD_SIZE_POS))
#define DMA_CH0_CTRL0_SRC_WORD_SIZE_POS 16
#define DMA_CH0_CTRL0_SRC_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH0_CTRL0_SRC_WORD_SIZE_POS))
#define DMA_CH0_CTRL0_DEST_SELECT_POS   12
#define DMA_CH0_CTRL0_DEST_SELECT_MASK  ((uint32_t)(0xFU << DMA_CH0_CTRL0_DEST_SELECT_POS))
#define DMA_CH0_CTRL0_SRC_SELECT_POS    8
#define DMA_CH0_CTRL0_SRC_SELECT_MASK   ((uint32_t)(0xFU << DMA_CH0_CTRL0_SRC_SELECT_POS))
#define DMA_CH0_CTRL0_CHANNEL_PRIORITY_POS 6
#define DMA_CH0_CTRL0_CHANNEL_PRIORITY_MASK ((uint32_t)(0x3U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_POS))
#define DMA_CH0_CTRL0_TRANSFER_TYPE_POS 4
#define DMA_CH0_CTRL0_TRANSFER_TYPE_MASK ((uint32_t)(0x3U << DMA_CH0_CTRL0_TRANSFER_TYPE_POS))
#define DMA_CH0_CTRL0_DEST_ADDR_INC_POS 3
#define DMA_CH0_CTRL0_SRC_ADDR_INC_POS  2
#define DMA_CH0_CTRL0_ADDR_MODE_POS     1
#define DMA_CH0_CTRL0_ENABLE_POS        0

/* DMA_CH0_CTRL0 bit-band aliases */
#define DMA_CH0_CTRL0_BYTE_ORDER_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_BYTE_ORDER_POS))
#define DMA_CH0_CTRL0_DISABLE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_DISABLE_INT_ENABLE_POS))
#define DMA_CH0_CTRL0_ERROR_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_ERROR_INT_ENABLE_POS))
#define DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_POS))
#define DMA_CH0_CTRL0_COUNTER_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_COUNTER_INT_ENABLE_POS))
#define DMA_CH0_CTRL0_START_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_START_INT_ENABLE_POS))
#define DMA_CH0_CTRL0_DEST_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_DEST_ADDR_INC_POS))
#define DMA_CH0_CTRL0_SRC_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_SRC_ADDR_INC_POS))
#define DMA_CH0_CTRL0_ADDR_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_ADDR_MODE_POS))
#define DMA_CH0_CTRL0_ENABLE_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, DMA_CH0_CTRL0_ENABLE_POS))

/* DMA Channel 0 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 0 */
#define DMA_CH0_SRC_BASE_ADDR           REG32_POINTER(DMA_CH0_SRC_BASE_ADDR_BASE)

/* DMA Channel 0 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   0 */
#define DMA_CH0_DEST_BASE_ADDR          REG32_POINTER(DMA_CH0_DEST_BASE_ADDR_BASE)

/* DMA Channel 0 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH0_CTRL1                   REG32_POINTER(DMA_CH0_CTRL1_BASE)

/* DMA_CH0_CTRL1 bit positions (legacy definitions) */
#define DMA_CH0_CTRL1_COUNTER_INT_VALUE_POS 16
#define DMA_CH0_CTRL1_COUNTER_INT_VALUE_MASK ((uint32_t)(0xFFFFU << DMA_CH0_CTRL1_COUNTER_INT_VALUE_POS))
#define DMA_CH0_CTRL1_TRANSFER_LENGTH_POS 0
#define DMA_CH0_CTRL1_TRANSFER_LENGTH_MASK ((uint32_t)(0xFFFFU << DMA_CH0_CTRL1_TRANSFER_LENGTH_POS))

/* DMA_CH0_CTRL1 subregister pointers */
#define DMA_CH0_COUNTER_INT_VALUE_SHORT REG16_POINTER(DMA_CH0_CTRL1_BASE + 2)
#define DMA_CH0_TRANSFER_LENGTH_SHORT   REG16_POINTER(DMA_CH0_CTRL1_BASE + 0)

/* DMA Channel 0 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 0 is
 *   loaded from */
#define DMA_CH0_NEXT_SRC_ADDR           READONLY_REG32_POINTER(DMA_CH0_NEXT_SRC_ADDR_BASE)

/* DMA Channel 0 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 0 will be
 *   stored */
#define DMA_CH0_NEXT_DEST_ADDR          READONLY_REG32_POINTER(DMA_CH0_NEXT_DEST_ADDR_BASE)

/* DMA Channel 0 Word Count */
/*   The number of words that have been transferred using DMA channel 0 during
 *   the current transfer */
#define DMA_CH0_WORD_CNT                READONLY_REG32_POINTER(DMA_CH0_WORD_CNT_BASE)

/* DMA channel 1 Control and Configuration */
/*   Control and configuration of DMA channel 1, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH1_CTRL0                   REG32_POINTER(DMA_CH1_CTRL0_BASE)

/* DMA_CH1_CTRL0 bit positions (legacy definitions) */
#define DMA_CH1_CTRL0_BYTE_ORDER_POS    25
#define DMA_CH1_CTRL0_DISABLE_INT_ENABLE_POS 24
#define DMA_CH1_CTRL0_ERROR_INT_ENABLE_POS 23
#define DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_POS 22
#define DMA_CH1_CTRL0_COUNTER_INT_ENABLE_POS 21
#define DMA_CH1_CTRL0_START_INT_ENABLE_POS 20
#define DMA_CH1_CTRL0_DEST_WORD_SIZE_POS 18
#define DMA_CH1_CTRL0_DEST_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH1_CTRL0_DEST_WORD_SIZE_POS))
#define DMA_CH1_CTRL0_SRC_WORD_SIZE_POS 16
#define DMA_CH1_CTRL0_SRC_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH1_CTRL0_SRC_WORD_SIZE_POS))
#define DMA_CH1_CTRL0_DEST_SELECT_POS   12
#define DMA_CH1_CTRL0_DEST_SELECT_MASK  ((uint32_t)(0xFU << DMA_CH1_CTRL0_DEST_SELECT_POS))
#define DMA_CH1_CTRL0_SRC_SELECT_POS    8
#define DMA_CH1_CTRL0_SRC_SELECT_MASK   ((uint32_t)(0xFU << DMA_CH1_CTRL0_SRC_SELECT_POS))
#define DMA_CH1_CTRL0_CHANNEL_PRIORITY_POS 6
#define DMA_CH1_CTRL0_CHANNEL_PRIORITY_MASK ((uint32_t)(0x3U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_POS))
#define DMA_CH1_CTRL0_TRANSFER_TYPE_POS 4
#define DMA_CH1_CTRL0_TRANSFER_TYPE_MASK ((uint32_t)(0x3U << DMA_CH1_CTRL0_TRANSFER_TYPE_POS))
#define DMA_CH1_CTRL0_DEST_ADDR_INC_POS 3
#define DMA_CH1_CTRL0_SRC_ADDR_INC_POS  2
#define DMA_CH1_CTRL0_ADDR_MODE_POS     1
#define DMA_CH1_CTRL0_ENABLE_POS        0

/* DMA_CH1_CTRL0 bit-band aliases */
#define DMA_CH1_CTRL0_BYTE_ORDER_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_BYTE_ORDER_POS))
#define DMA_CH1_CTRL0_DISABLE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_DISABLE_INT_ENABLE_POS))
#define DMA_CH1_CTRL0_ERROR_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_ERROR_INT_ENABLE_POS))
#define DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_POS))
#define DMA_CH1_CTRL0_COUNTER_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_COUNTER_INT_ENABLE_POS))
#define DMA_CH1_CTRL0_START_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_START_INT_ENABLE_POS))
#define DMA_CH1_CTRL0_DEST_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_DEST_ADDR_INC_POS))
#define DMA_CH1_CTRL0_SRC_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_SRC_ADDR_INC_POS))
#define DMA_CH1_CTRL0_ADDR_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_ADDR_MODE_POS))
#define DMA_CH1_CTRL0_ENABLE_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, DMA_CH1_CTRL0_ENABLE_POS))

/* DMA channel 1 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 1 */
#define DMA_CH1_SRC_BASE_ADDR           REG32_POINTER(DMA_CH1_SRC_BASE_ADDR_BASE)

/* DMA channel 1 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   1 */
#define DMA_CH1_DEST_BASE_ADDR          REG32_POINTER(DMA_CH1_DEST_BASE_ADDR_BASE)

/* DMA channel 1 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH1_CTRL1                   REG32_POINTER(DMA_CH1_CTRL1_BASE)

/* DMA_CH1_CTRL1 bit positions (legacy definitions) */
#define DMA_CH1_CTRL1_COUNTER_INT_VALUE_POS 16
#define DMA_CH1_CTRL1_COUNTER_INT_VALUE_MASK ((uint32_t)(0xFFFFU << DMA_CH1_CTRL1_COUNTER_INT_VALUE_POS))
#define DMA_CH1_CTRL1_TRANSFER_LENGTH_POS 0
#define DMA_CH1_CTRL1_TRANSFER_LENGTH_MASK ((uint32_t)(0xFFFFU << DMA_CH1_CTRL1_TRANSFER_LENGTH_POS))

/* DMA_CH1_CTRL1 subregister pointers */
#define DMA_CH1_COUNTER_INT_VALUE_SHORT REG16_POINTER(DMA_CH1_CTRL1_BASE + 2)
#define DMA_CH1_TRANSFER_LENGTH_SHORT   REG16_POINTER(DMA_CH1_CTRL1_BASE + 0)

/* DMA channel 1 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 1 is
 *   loaded from */
#define DMA_CH1_NEXT_SRC_ADDR           READONLY_REG32_POINTER(DMA_CH1_NEXT_SRC_ADDR_BASE)

/* DMA channel 1 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 1 will be
 *   stored */
#define DMA_CH1_NEXT_DEST_ADDR          READONLY_REG32_POINTER(DMA_CH1_NEXT_DEST_ADDR_BASE)

/* DMA channel 1 Word Count */
/*   The number of words that have been transferred using DMA channel 1 during
 *   the current transfer */
#define DMA_CH1_WORD_CNT                READONLY_REG32_POINTER(DMA_CH1_WORD_CNT_BASE)

/* DMA channel 2 Control and Configuration */
/*   Control and configuration of DMA channel 2, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH2_CTRL0                   REG32_POINTER(DMA_CH2_CTRL0_BASE)

/* DMA_CH2_CTRL0 bit positions (legacy definitions) */
#define DMA_CH2_CTRL0_BYTE_ORDER_POS    25
#define DMA_CH2_CTRL0_DISABLE_INT_ENABLE_POS 24
#define DMA_CH2_CTRL0_ERROR_INT_ENABLE_POS 23
#define DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_POS 22
#define DMA_CH2_CTRL0_COUNTER_INT_ENABLE_POS 21
#define DMA_CH2_CTRL0_START_INT_ENABLE_POS 20
#define DMA_CH2_CTRL0_DEST_WORD_SIZE_POS 18
#define DMA_CH2_CTRL0_DEST_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH2_CTRL0_DEST_WORD_SIZE_POS))
#define DMA_CH2_CTRL0_SRC_WORD_SIZE_POS 16
#define DMA_CH2_CTRL0_SRC_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH2_CTRL0_SRC_WORD_SIZE_POS))
#define DMA_CH2_CTRL0_DEST_SELECT_POS   12
#define DMA_CH2_CTRL0_DEST_SELECT_MASK  ((uint32_t)(0xFU << DMA_CH2_CTRL0_DEST_SELECT_POS))
#define DMA_CH2_CTRL0_SRC_SELECT_POS    8
#define DMA_CH2_CTRL0_SRC_SELECT_MASK   ((uint32_t)(0xFU << DMA_CH2_CTRL0_SRC_SELECT_POS))
#define DMA_CH2_CTRL0_CHANNEL_PRIORITY_POS 6
#define DMA_CH2_CTRL0_CHANNEL_PRIORITY_MASK ((uint32_t)(0x3U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_POS))
#define DMA_CH2_CTRL0_TRANSFER_TYPE_POS 4
#define DMA_CH2_CTRL0_TRANSFER_TYPE_MASK ((uint32_t)(0x3U << DMA_CH2_CTRL0_TRANSFER_TYPE_POS))
#define DMA_CH2_CTRL0_DEST_ADDR_INC_POS 3
#define DMA_CH2_CTRL0_SRC_ADDR_INC_POS  2
#define DMA_CH2_CTRL0_ADDR_MODE_POS     1
#define DMA_CH2_CTRL0_ENABLE_POS        0

/* DMA_CH2_CTRL0 bit-band aliases */
#define DMA_CH2_CTRL0_BYTE_ORDER_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_BYTE_ORDER_POS))
#define DMA_CH2_CTRL0_DISABLE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_DISABLE_INT_ENABLE_POS))
#define DMA_CH2_CTRL0_ERROR_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_ERROR_INT_ENABLE_POS))
#define DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_POS))
#define DMA_CH2_CTRL0_COUNTER_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_COUNTER_INT_ENABLE_POS))
#define DMA_CH2_CTRL0_START_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_START_INT_ENABLE_POS))
#define DMA_CH2_CTRL0_DEST_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_DEST_ADDR_INC_POS))
#define DMA_CH2_CTRL0_SRC_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_SRC_ADDR_INC_POS))
#define DMA_CH2_CTRL0_ADDR_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_ADDR_MODE_POS))
#define DMA_CH2_CTRL0_ENABLE_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, DMA_CH2_CTRL0_ENABLE_POS))

/* DMA channel 2 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 2 */
#define DMA_CH2_SRC_BASE_ADDR           REG32_POINTER(DMA_CH2_SRC_BASE_ADDR_BASE)

/* DMA channel 2 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   2 */
#define DMA_CH2_DEST_BASE_ADDR          REG32_POINTER(DMA_CH2_DEST_BASE_ADDR_BASE)

/* DMA channel 2 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH2_CTRL1                   REG32_POINTER(DMA_CH2_CTRL1_BASE)

/* DMA_CH2_CTRL1 bit positions (legacy definitions) */
#define DMA_CH2_CTRL1_COUNTER_INT_VALUE_POS 16
#define DMA_CH2_CTRL1_COUNTER_INT_VALUE_MASK ((uint32_t)(0xFFFFU << DMA_CH2_CTRL1_COUNTER_INT_VALUE_POS))
#define DMA_CH2_CTRL1_TRANSFER_LENGTH_POS 0
#define DMA_CH2_CTRL1_TRANSFER_LENGTH_MASK ((uint32_t)(0xFFFFU << DMA_CH2_CTRL1_TRANSFER_LENGTH_POS))

/* DMA_CH2_CTRL1 subregister pointers */
#define DMA_CH2_COUNTER_INT_VALUE_SHORT REG16_POINTER(DMA_CH2_CTRL1_BASE + 2)
#define DMA_CH2_TRANSFER_LENGTH_SHORT   REG16_POINTER(DMA_CH2_CTRL1_BASE + 0)

/* DMA channel 2 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 2 is
 *   loaded from */
#define DMA_CH2_NEXT_SRC_ADDR           READONLY_REG32_POINTER(DMA_CH2_NEXT_SRC_ADDR_BASE)

/* DMA channel 2 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 2 will be
 *   stored */
#define DMA_CH2_NEXT_DEST_ADDR          READONLY_REG32_POINTER(DMA_CH2_NEXT_DEST_ADDR_BASE)

/* DMA channel 2 Word Count */
/*   The number of words that have been transferred using DMA channel 2 during
 *   the current transfer */
#define DMA_CH2_WORD_CNT                READONLY_REG32_POINTER(DMA_CH2_WORD_CNT_BASE)

/* DMA channel 3 Control and Configuration */
/*   Control and configuration of DMA channel 3, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH3_CTRL0                   REG32_POINTER(DMA_CH3_CTRL0_BASE)

/* DMA_CH3_CTRL0 bit positions (legacy definitions) */
#define DMA_CH3_CTRL0_BYTE_ORDER_POS    25
#define DMA_CH3_CTRL0_DISABLE_INT_ENABLE_POS 24
#define DMA_CH3_CTRL0_ERROR_INT_ENABLE_POS 23
#define DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_POS 22
#define DMA_CH3_CTRL0_COUNTER_INT_ENABLE_POS 21
#define DMA_CH3_CTRL0_START_INT_ENABLE_POS 20
#define DMA_CH3_CTRL0_DEST_WORD_SIZE_POS 18
#define DMA_CH3_CTRL0_DEST_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH3_CTRL0_DEST_WORD_SIZE_POS))
#define DMA_CH3_CTRL0_SRC_WORD_SIZE_POS 16
#define DMA_CH3_CTRL0_SRC_WORD_SIZE_MASK ((uint32_t)(0x3U << DMA_CH3_CTRL0_SRC_WORD_SIZE_POS))
#define DMA_CH3_CTRL0_DEST_SELECT_POS   12
#define DMA_CH3_CTRL0_DEST_SELECT_MASK  ((uint32_t)(0xFU << DMA_CH3_CTRL0_DEST_SELECT_POS))
#define DMA_CH3_CTRL0_SRC_SELECT_POS    8
#define DMA_CH3_CTRL0_SRC_SELECT_MASK   ((uint32_t)(0xFU << DMA_CH3_CTRL0_SRC_SELECT_POS))
#define DMA_CH3_CTRL0_CHANNEL_PRIORITY_POS 6
#define DMA_CH3_CTRL0_CHANNEL_PRIORITY_MASK ((uint32_t)(0x3U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_POS))
#define DMA_CH3_CTRL0_TRANSFER_TYPE_POS 4
#define DMA_CH3_CTRL0_TRANSFER_TYPE_MASK ((uint32_t)(0x3U << DMA_CH3_CTRL0_TRANSFER_TYPE_POS))
#define DMA_CH3_CTRL0_DEST_ADDR_INC_POS 3
#define DMA_CH3_CTRL0_SRC_ADDR_INC_POS  2
#define DMA_CH3_CTRL0_ADDR_MODE_POS     1
#define DMA_CH3_CTRL0_ENABLE_POS        0

/* DMA_CH3_CTRL0 bit-band aliases */
#define DMA_CH3_CTRL0_BYTE_ORDER_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_BYTE_ORDER_POS))
#define DMA_CH3_CTRL0_DISABLE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_DISABLE_INT_ENABLE_POS))
#define DMA_CH3_CTRL0_ERROR_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_ERROR_INT_ENABLE_POS))
#define DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_POS))
#define DMA_CH3_CTRL0_COUNTER_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_COUNTER_INT_ENABLE_POS))
#define DMA_CH3_CTRL0_START_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_START_INT_ENABLE_POS))
#define DMA_CH3_CTRL0_DEST_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_DEST_ADDR_INC_POS))
#define DMA_CH3_CTRL0_SRC_ADDR_INC_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_SRC_ADDR_INC_POS))
#define DMA_CH3_CTRL0_ADDR_MODE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_ADDR_MODE_POS))
#define DMA_CH3_CTRL0_ENABLE_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, DMA_CH3_CTRL0_ENABLE_POS))

/* DMA channel 3 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 3 */
#define DMA_CH3_SRC_BASE_ADDR           REG32_POINTER(DMA_CH3_SRC_BASE_ADDR_BASE)

/* DMA channel 3 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   3 */
#define DMA_CH3_DEST_BASE_ADDR          REG32_POINTER(DMA_CH3_DEST_BASE_ADDR_BASE)

/* DMA channel 3 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH3_CTRL1                   REG32_POINTER(DMA_CH3_CTRL1_BASE)

/* DMA_CH3_CTRL1 bit positions (legacy definitions) */
#define DMA_CH3_CTRL1_COUNTER_INT_VALUE_POS 16
#define DMA_CH3_CTRL1_COUNTER_INT_VALUE_MASK ((uint32_t)(0xFFFFU << DMA_CH3_CTRL1_COUNTER_INT_VALUE_POS))
#define DMA_CH3_CTRL1_TRANSFER_LENGTH_POS 0
#define DMA_CH3_CTRL1_TRANSFER_LENGTH_MASK ((uint32_t)(0xFFFFU << DMA_CH3_CTRL1_TRANSFER_LENGTH_POS))

/* DMA_CH3_CTRL1 subregister pointers */
#define DMA_CH3_COUNTER_INT_VALUE_SHORT REG16_POINTER(DMA_CH3_CTRL1_BASE + 2)
#define DMA_CH3_TRANSFER_LENGTH_SHORT   REG16_POINTER(DMA_CH3_CTRL1_BASE + 0)

/* DMA channel 3 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 3 is
 *   loaded from */
#define DMA_CH3_NEXT_SRC_ADDR           READONLY_REG32_POINTER(DMA_CH3_NEXT_SRC_ADDR_BASE)

/* DMA channel 3 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 3 will be
 *   stored */
#define DMA_CH3_NEXT_DEST_ADDR          READONLY_REG32_POINTER(DMA_CH3_NEXT_DEST_ADDR_BASE)

/* DMA channel 3 Word Count */
/*   The number of words that have been transferred using DMA channel 3 during
 *   the current transfer */
#define DMA_CH3_WORD_CNT                READONLY_REG32_POINTER(DMA_CH3_WORD_CNT_BASE)

/* DMA Request Control for DAC0 */
/*   Control the DAC0 DMA transfers and interrupts */
#define DMA_DAC0_REQUEST                REG32_POINTER(DMA_DAC0_REQUEST_BASE)

/* DMA_DAC0_REQUEST bit positions (legacy definitions) */
#define DMA_DAC0_REQUEST_DMA_ENABLE_POS 16
#define DMA_DAC0_REQUEST_RATE_POS       0
#define DMA_DAC0_REQUEST_RATE_MASK      ((uint32_t)(0x3FFU << DMA_DAC0_REQUEST_RATE_POS))

/* DMA_DAC0_REQUEST bit-band aliases */
#define DMA_DAC0_REQUEST_DMA_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_DAC0_REQUEST_BASE, DMA_DAC0_REQUEST_DMA_ENABLE_POS))

/* DMA Request Control for DAC1 */
/*   Control the DAC1 DMA transfers and interrupts */
#define DMA_DAC1_REQUEST                REG32_POINTER(DMA_DAC1_REQUEST_BASE)

/* DMA_DAC1_REQUEST bit positions (legacy definitions) */
#define DMA_DAC1_REQUEST_DMA_ENABLE_POS 16
#define DMA_DAC1_REQUEST_RATE_POS       0
#define DMA_DAC1_REQUEST_RATE_MASK      ((uint32_t)(0x3FFU << DMA_DAC1_REQUEST_RATE_POS))

/* DMA_DAC1_REQUEST bit-band aliases */
#define DMA_DAC1_REQUEST_DMA_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_DAC1_REQUEST_BASE, DMA_DAC1_REQUEST_DMA_ENABLE_POS))

/* DMA Status */
/*   Status of DMA transfers and interrupts */
#define DMA_STATUS                      REG32_POINTER(DMA_STATUS_BASE)

/* DMA_STATUS bit positions (legacy definitions) */
#define DMA_STATUS_CH3_ERROR_INT_STATUS_POS 31
#define DMA_STATUS_CH3_COMPLETE_INT_STATUS_POS 30
#define DMA_STATUS_CH3_COUNTER_INT_STATUS_POS 29
#define DMA_STATUS_CH3_START_INT_STATUS_POS 28
#define DMA_STATUS_CH3_DISABLE_INT_STATUS_POS 27
#define DMA_STATUS_CH3_STATE_POS        24
#define DMA_STATUS_CH3_STATE_MASK       ((uint32_t)(0x7U << DMA_STATUS_CH3_STATE_POS))
#define DMA_STATUS_CH2_ERROR_INT_STATUS_POS 23
#define DMA_STATUS_CH2_COMPLETE_INT_STATUS_POS 22
#define DMA_STATUS_CH2_COUNTER_INT_STATUS_POS 21
#define DMA_STATUS_CH2_START_INT_STATUS_POS 20
#define DMA_STATUS_CH2_DISABLE_INT_STATUS_POS 19
#define DMA_STATUS_CH2_STATE_POS        16
#define DMA_STATUS_CH2_STATE_MASK       ((uint32_t)(0x7U << DMA_STATUS_CH2_STATE_POS))
#define DMA_STATUS_CH1_ERROR_INT_STATUS_POS 15
#define DMA_STATUS_CH1_COMPLETE_INT_STATUS_POS 14
#define DMA_STATUS_CH1_COUNTER_INT_STATUS_POS 13
#define DMA_STATUS_CH1_START_INT_STATUS_POS 12
#define DMA_STATUS_CH1_DISABLE_INT_STATUS_POS 11
#define DMA_STATUS_CH1_STATE_POS        8
#define DMA_STATUS_CH1_STATE_MASK       ((uint32_t)(0x7U << DMA_STATUS_CH1_STATE_POS))
#define DMA_STATUS_CH0_ERROR_INT_STATUS_POS 7
#define DMA_STATUS_CH0_COMPLETE_INT_STATUS_POS 6
#define DMA_STATUS_CH0_COUNTER_INT_STATUS_POS 5
#define DMA_STATUS_CH0_START_INT_STATUS_POS 4
#define DMA_STATUS_CH0_DISABLE_INT_STATUS_POS 3
#define DMA_STATUS_CH0_STATE_POS        0
#define DMA_STATUS_CH0_STATE_MASK       ((uint32_t)(0x7U << DMA_STATUS_CH0_STATE_POS))

/* DMA_STATUS bit-band aliases */
#define DMA_STATUS_CH3_ERROR_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH3_ERROR_INT_STATUS_POS))
#define DMA_STATUS_CH3_COMPLETE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH3_COMPLETE_INT_STATUS_POS))
#define DMA_STATUS_CH3_COUNTER_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH3_COUNTER_INT_STATUS_POS))
#define DMA_STATUS_CH3_START_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH3_START_INT_STATUS_POS))
#define DMA_STATUS_CH3_DISABLE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH3_DISABLE_INT_STATUS_POS))
#define DMA_STATUS_CH2_ERROR_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH2_ERROR_INT_STATUS_POS))
#define DMA_STATUS_CH2_COMPLETE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH2_COMPLETE_INT_STATUS_POS))
#define DMA_STATUS_CH2_COUNTER_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH2_COUNTER_INT_STATUS_POS))
#define DMA_STATUS_CH2_START_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH2_START_INT_STATUS_POS))
#define DMA_STATUS_CH2_DISABLE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH2_DISABLE_INT_STATUS_POS))
#define DMA_STATUS_CH1_ERROR_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH1_ERROR_INT_STATUS_POS))
#define DMA_STATUS_CH1_COMPLETE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH1_COMPLETE_INT_STATUS_POS))
#define DMA_STATUS_CH1_COUNTER_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH1_COUNTER_INT_STATUS_POS))
#define DMA_STATUS_CH1_START_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH1_START_INT_STATUS_POS))
#define DMA_STATUS_CH1_DISABLE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH1_DISABLE_INT_STATUS_POS))
#define DMA_STATUS_CH0_ERROR_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH0_ERROR_INT_STATUS_POS))
#define DMA_STATUS_CH0_COMPLETE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH0_COMPLETE_INT_STATUS_POS))
#define DMA_STATUS_CH0_COUNTER_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH0_COUNTER_INT_STATUS_POS))
#define DMA_STATUS_CH0_START_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH0_START_INT_STATUS_POS))
#define DMA_STATUS_CH0_DISABLE_INT_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(DMA_STATUS_BASE, DMA_STATUS_CH0_DISABLE_INT_STATUS_POS))

/* DMA_STATUS subregister pointers */
#define DMA_STATUS_CH3_BYTE             REG8_POINTER(DMA_STATUS_BASE + 3)
#define DMA_STATUS_CH2_BYTE             REG8_POINTER(DMA_STATUS_BASE + 2)
#define DMA_STATUS_CH1_BYTE             REG8_POINTER(DMA_STATUS_BASE + 1)
#define DMA_STATUS_CH0_BYTE             REG8_POINTER(DMA_STATUS_BASE + 0)

/* DMA_STATUS subregister bit positions */
#define DMA_STATUS_CH3_BYTE_CH3_ERROR_INT_STATUS_POS 7
#define DMA_STATUS_CH3_BYTE_CH3_COMPLETE_INT_STATUS_POS 6
#define DMA_STATUS_CH3_BYTE_CH3_COUNTER_INT_STATUS_POS 5
#define DMA_STATUS_CH3_BYTE_CH3_START_INT_STATUS_POS 4
#define DMA_STATUS_CH3_BYTE_CH3_DISABLE_INT_STATUS_POS 3
#define DMA_STATUS_CH3_BYTE_CH3_STATE_POS 0
#define DMA_STATUS_CH3_BYTE_CH3_STATE_MASK ((uint32_t)(0x7U << DMA_STATUS_CH3_BYTE_CH3_STATE_POS))
#define DMA_STATUS_CH2_BYTE_CH2_ERROR_INT_STATUS_POS 7
#define DMA_STATUS_CH2_BYTE_CH2_COMPLETE_INT_STATUS_POS 6
#define DMA_STATUS_CH2_BYTE_CH2_COUNTER_INT_STATUS_POS 5
#define DMA_STATUS_CH2_BYTE_CH2_START_INT_STATUS_POS 4
#define DMA_STATUS_CH2_BYTE_CH2_DISABLE_INT_STATUS_POS 3
#define DMA_STATUS_CH2_BYTE_CH2_STATE_POS 0
#define DMA_STATUS_CH2_BYTE_CH2_STATE_MASK ((uint32_t)(0x7U << DMA_STATUS_CH2_BYTE_CH2_STATE_POS))
#define DMA_STATUS_CH1_BYTE_CH1_ERROR_INT_STATUS_POS 7
#define DMA_STATUS_CH1_BYTE_CH1_COMPLETE_INT_STATUS_POS 6
#define DMA_STATUS_CH1_BYTE_CH1_COUNTER_INT_STATUS_POS 5
#define DMA_STATUS_CH1_BYTE_CH1_START_INT_STATUS_POS 4
#define DMA_STATUS_CH1_BYTE_CH1_DISABLE_INT_STATUS_POS 3
#define DMA_STATUS_CH1_BYTE_CH1_STATE_POS 0
#define DMA_STATUS_CH1_BYTE_CH1_STATE_MASK ((uint32_t)(0x7U << DMA_STATUS_CH1_BYTE_CH1_STATE_POS))
#define DMA_STATUS_CH0_BYTE_CH0_ERROR_INT_STATUS_POS 7
#define DMA_STATUS_CH0_BYTE_CH0_COMPLETE_INT_STATUS_POS 6
#define DMA_STATUS_CH0_BYTE_CH0_COUNTER_INT_STATUS_POS 5
#define DMA_STATUS_CH0_BYTE_CH0_START_INT_STATUS_POS 4
#define DMA_STATUS_CH0_BYTE_CH0_DISABLE_INT_STATUS_POS 3
#define DMA_STATUS_CH0_BYTE_CH0_STATE_POS 0
#define DMA_STATUS_CH0_BYTE_CH0_STATE_MASK ((uint32_t)(0x7U << DMA_STATUS_CH0_BYTE_CH0_STATE_POS))

/* ----------------------------------------------------------------------------
 * UART0 Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The universal asynchronous receiver-transmitter (UART) interface provides a
 * general purpose data connection using an RS-232 transmission protocol.
 *
 * UART0 uses a standard data format with one start bit, eight data bits and
 * one stop bit. */

/* UART0 Control and Configuration */
/*   Configuration, control and status of the UART0 interface */
#define UART0_CTRL                      REG32_POINTER(UART0_CTRL_BASE)

/* UART0_CTRL bit positions (legacy definitions) */
#define UART0_CTRL_CONTROLLER_POS       2
#define UART0_CTRL_ENABLE_POS           1
#define UART0_CTRL_PRESCALE_ENABLE_POS  0

/* UART0_CTRL bit-band aliases */
#define UART0_CTRL_CONTROLLER_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(UART0_CTRL_BASE, UART0_CTRL_CONTROLLER_POS))
#define UART0_CTRL_ENABLE_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(UART0_CTRL_BASE, UART0_CTRL_ENABLE_POS))
#define UART0_CTRL_PRESCALE_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(UART0_CTRL_BASE, UART0_CTRL_PRESCALE_ENABLE_POS))

/* UART0_CTRL settings */
#define UART0_CONTROLLER_CM3_BITBAND    0x0
#define UART0_CONTROLLER_DMA_BITBAND    0x1
#define UART0_CONTROLLER_CM3            ((uint32_t)(UART0_CONTROLLER_CM3_BITBAND << UART0_CTRL_CONTROLLER_POS))
#define UART0_CONTROLLER_DMA            ((uint32_t)(UART0_CONTROLLER_DMA_BITBAND << UART0_CTRL_CONTROLLER_POS))

#define UART0_DISABLE_BITBAND           0x0
#define UART0_ENABLE_BITBAND            0x1
#define UART0_DISABLE                   ((uint32_t)(UART0_DISABLE_BITBAND << UART0_CTRL_ENABLE_POS))
#define UART0_ENABLE                    ((uint32_t)(UART0_ENABLE_BITBAND << UART0_CTRL_ENABLE_POS))

#define UART0_PRESCALE_DISABLE_BITBAND  0x0
#define UART0_PRESCALE_ENABLE_BITBAND   0x1
#define UART0_PRESCALE_DISABLE          ((uint32_t)(UART0_PRESCALE_DISABLE_BITBAND << UART0_CTRL_PRESCALE_ENABLE_POS))
#define UART0_PRESCALE_ENABLE           ((uint32_t)(UART0_PRESCALE_ENABLE_BITBAND << UART0_CTRL_PRESCALE_ENABLE_POS))

/* UART0 Baud Rate Configuration */
/*   Configure the UART0 baud rate (divided from UART0_CLK, potentially
 *   including a pre-scaling divisor of 12 based on UART0_CTRL_PRESCALE_ENABLE
 *   in UART0_CTRL) */
#define UART0_SPEED_CTRL                REG32_POINTER(UART0_SPEED_CTRL_BASE)

/* UART0 Status */
/*   Status of the UART0 interface */
#define UART0_STATUS                    REG32_POINTER(UART0_STATUS_BASE)

/* UART0_STATUS bit positions (legacy definitions) */
#define UART0_STATUS_OVERRUN_STATUS_POS 1

/* UART0_STATUS bit-band aliases */
#define UART0_STATUS_OVERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(UART0_STATUS_BASE, UART0_STATUS_OVERRUN_STATUS_POS))

/* UART0_STATUS settings */
#define UART0_OVERRUN_FALSE_BITBAND     0x0
#define UART0_OVERRUN_TRUE_BITBAND      0x1
#define UART0_OVERRUN_CLEAR_BITBAND     0x1
#define UART0_OVERRUN_FALSE             ((uint32_t)(UART0_OVERRUN_FALSE_BITBAND << UART0_STATUS_OVERRUN_STATUS_POS))
#define UART0_OVERRUN_TRUE              ((uint32_t)(UART0_OVERRUN_TRUE_BITBAND << UART0_STATUS_OVERRUN_STATUS_POS))
#define UART0_OVERRUN_CLEAR             ((uint32_t)(UART0_OVERRUN_CLEAR_BITBAND << UART0_STATUS_OVERRUN_STATUS_POS))

/* UART0 Transmit Data */
/*   Byte of data to transmit over the UART interface */
#define UART0_TXDATA                    REG32_POINTER(UART0_TXDATA_BASE)

/* UART0 Receive Data */
/*   Byte of data received from the UART interface */
#define UART0_RXDATA                    READONLY_REG32_POINTER(UART0_RXDATA_BASE)

/* ----------------------------------------------------------------------------
 * UART1 Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The universal asynchronous receiver-transmitter (UART) interface provides a
 * general purpose data connection using an RS-232 transmission protocol.
 *
 * UART1 uses a standard data format with one start bit, eight data bits and
 * one stop bit. */

/* UART1 Control and Configuration */
/*   Configuration and control of the UART1 interface */
#define UART1_CTRL                      REG32_POINTER(UART1_CTRL_BASE)

/* UART1_CTRL bit positions (legacy definitions) */
#define UART1_CTRL_CONTROLLER_POS       2
#define UART1_CTRL_ENABLE_POS           1
#define UART1_CTRL_PRESCALE_ENABLE_POS  0

/* UART1_CTRL bit-band aliases */
#define UART1_CTRL_CONTROLLER_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(UART1_CTRL_BASE, UART1_CTRL_CONTROLLER_POS))
#define UART1_CTRL_ENABLE_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(UART1_CTRL_BASE, UART1_CTRL_ENABLE_POS))
#define UART1_CTRL_PRESCALE_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(UART1_CTRL_BASE, UART1_CTRL_PRESCALE_ENABLE_POS))

/* UART1_CTRL settings */
#define UART1_CONTROLLER_CM3_BITBAND    0x0
#define UART1_CONTROLLER_DMA_BITBAND    0x1
#define UART1_CONTROLLER_CM3            ((uint32_t)(UART1_CONTROLLER_CM3_BITBAND << UART1_CTRL_CONTROLLER_POS))
#define UART1_CONTROLLER_DMA            ((uint32_t)(UART1_CONTROLLER_DMA_BITBAND << UART1_CTRL_CONTROLLER_POS))

#define UART1_DISABLE_BITBAND           0x0
#define UART1_ENABLE_BITBAND            0x1
#define UART1_DISABLE                   ((uint32_t)(UART1_DISABLE_BITBAND << UART1_CTRL_ENABLE_POS))
#define UART1_ENABLE                    ((uint32_t)(UART1_ENABLE_BITBAND << UART1_CTRL_ENABLE_POS))

#define UART1_PRESCALE_DISABLE_BITBAND  0x0
#define UART1_PRESCALE_ENABLE_BITBAND   0x1
#define UART1_PRESCALE_DISABLE          ((uint32_t)(UART1_PRESCALE_DISABLE_BITBAND << UART1_CTRL_PRESCALE_ENABLE_POS))
#define UART1_PRESCALE_ENABLE           ((uint32_t)(UART1_PRESCALE_ENABLE_BITBAND << UART1_CTRL_PRESCALE_ENABLE_POS))

/* UART1 Baud Rate Configuration */
/*   Configure the UART1 baud rate (divided from UART1_CLK, potentially
 *   including a pre-scaling divisor of 12 based on UART1_CTRL_PRESCALE_ENABLE
 *   in UART1_CTRL) */
#define UART1_SPEED_CTRL                REG32_POINTER(UART1_SPEED_CTRL_BASE)

/* UART1 Status */
/*   Status of the UART1 interface */
#define UART1_STATUS                    REG32_POINTER(UART1_STATUS_BASE)

/* UART1_STATUS bit positions (legacy definitions) */
#define UART1_STATUS_OVERRUN_STATUS_POS 1

/* UART1_STATUS bit-band aliases */
#define UART1_STATUS_OVERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(UART1_STATUS_BASE, UART1_STATUS_OVERRUN_STATUS_POS))

/* UART1_STATUS settings */
#define UART1_OVERRUN_FALSE_BITBAND     0x0
#define UART1_OVERRUN_TRUE_BITBAND      0x1
#define UART1_OVERRUN_CLEAR_BITBAND     0x1
#define UART1_OVERRUN_FALSE             ((uint32_t)(UART1_OVERRUN_FALSE_BITBAND << UART1_STATUS_OVERRUN_STATUS_POS))
#define UART1_OVERRUN_TRUE              ((uint32_t)(UART1_OVERRUN_TRUE_BITBAND << UART1_STATUS_OVERRUN_STATUS_POS))
#define UART1_OVERRUN_CLEAR             ((uint32_t)(UART1_OVERRUN_CLEAR_BITBAND << UART1_STATUS_OVERRUN_STATUS_POS))

/* UART1 Transmit Data */
/*   Byte of data to transmit over the UART interface */
#define UART1_TXDATA                    REG32_POINTER(UART1_TXDATA_BASE)

/* UART1 Receive Data */
/*   Byte of data received from the UART interface */
#define UART1_RXDATA                    READONLY_REG32_POINTER(UART1_RXDATA_BASE)

/* ----------------------------------------------------------------------------
 * GPIO and LCD Driver Configuration and Control
 * ------------------------------------------------------------------------- */
/* The general-purpose input/output (GPIO) pins provide user defined signals as
 * inputs or outputs from the system. Many of these general-purpose pins are
 * multiplexed with other interfaces and are only available when those
 * interfaces are not in use. Multiplexed interfaces include the LCD driver,
 * UART0, UART1, SPI0,  SPI1, PCM and User clock interfaces.
 *
 * The LCD driver supports up to 112 segments, using 28 segment drivers and 4
 * COM lines. */

/* GPIO IF0 Function Select */
/*   Select the function for GPIO pins 32 to 35 which are  multiplexed with the
 *   SPI0  interface and user clocks */
#define GPIO_IF0_FUNC_SEL               REG32_POINTER(GPIO_IF0_FUNC_SEL_BASE)

/* GPIO_IF0_FUNC_SEL bit positions (legacy definitions) */
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_POS 15
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_POS 14
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_POS 13
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_POS 12
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS 11
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS 10
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS 9
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS 8
#define GPIO_IF0_FUNC_SEL_INPUT_DATA_POS 4
#define GPIO_IF0_FUNC_SEL_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF0_FUNC_SEL_INPUT_DATA_POS))
#define GPIO_IF0_FUNC_SEL_FUNC_SEL_POS  0
#define GPIO_IF0_FUNC_SEL_FUNC_SEL_MASK ((uint32_t)(0x3U << GPIO_IF0_FUNC_SEL_FUNC_SEL_POS))

/* GPIO_IF0_FUNC_SEL bit-band aliases */
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_POS))
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_POS))
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_POS))
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_POS))
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS))
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS))
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS))
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_FUNC_SEL_BASE, GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS))

/* GPIO_IF0_FUNC_SEL subregister pointers */
#define GPIO_IF0_PIN_CFG_BYTE           REG8_POINTER(GPIO_IF0_FUNC_SEL_BASE + 1)
#define GPIO_IF0_FUNC_SEL_BYTE          REG8_POINTER(GPIO_IF0_FUNC_SEL_BASE + 0)

/* GPIO_IF0_FUNC_SEL subregister bit positions */
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_POS 7
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_POS 6
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_POS 5
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_POS 4
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_POS 3
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_POS 2
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_POS 1
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_POS 0
#define GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_POS 4
#define GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_POS))
#define GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_POS 0
#define GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_MASK ((uint32_t)(0x3U << GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_POS))

/* GPIO IF0 Output */
/*   Output for GPIO pins 32 to 35 */
#define GPIO_IF0_OUT                    REG32_POINTER(GPIO_IF0_OUT_BASE)

/* GPIO_IF0_OUT bit positions (legacy definitions) */
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN3_POS 3
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN2_POS 2
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN1_POS 1
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN0_POS 0

/* GPIO_IF0_OUT bit-band aliases */
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_OUT_BASE, GPIO_IF0_OUT_OUTPUT_DATA_PIN3_POS))
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_OUT_BASE, GPIO_IF0_OUT_OUTPUT_DATA_PIN2_POS))
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_OUT_BASE, GPIO_IF0_OUT_OUTPUT_DATA_PIN1_POS))
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF0_OUT_BASE, GPIO_IF0_OUT_OUTPUT_DATA_PIN0_POS))

/* GPIO IF1 Function Select */
/*   Select the function for GPIO pins 36 to 39 which are  multiplexed with the
 *   SPI1 and PCM interfaces */
#define GPIO_IF1_FUNC_SEL               REG32_POINTER(GPIO_IF1_FUNC_SEL_BASE)

/* GPIO_IF1_FUNC_SEL bit positions (legacy definitions) */
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_POS 15
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_POS 14
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_POS 13
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_POS 12
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS 11
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS 10
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS 9
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS 8
#define GPIO_IF1_FUNC_SEL_INPUT_DATA_POS 4
#define GPIO_IF1_FUNC_SEL_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF1_FUNC_SEL_INPUT_DATA_POS))
#define GPIO_IF1_FUNC_SEL_FUNC_SEL_POS  0
#define GPIO_IF1_FUNC_SEL_FUNC_SEL_MASK ((uint32_t)(0x3U << GPIO_IF1_FUNC_SEL_FUNC_SEL_POS))

/* GPIO_IF1_FUNC_SEL bit-band aliases */
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_POS))
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_POS))
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_POS))
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_POS))
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS))
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS))
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS))
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_FUNC_SEL_BASE, GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS))

/* GPIO_IF1_FUNC_SEL subregister pointers */
#define GPIO_IF1_PIN_CFG_BYTE           REG8_POINTER(GPIO_IF1_FUNC_SEL_BASE + 1)
#define GPIO_IF1_FUNC_SEL_BYTE          REG8_POINTER(GPIO_IF1_FUNC_SEL_BASE + 0)

/* GPIO_IF1_FUNC_SEL subregister bit positions */
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_POS 7
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_POS 6
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_POS 5
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_POS 4
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_POS 3
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_POS 2
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_POS 1
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_POS 0
#define GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_POS 4
#define GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_POS))
#define GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_POS 0
#define GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_MASK ((uint32_t)(0x3U << GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_POS))

/* GPIO IF1 Output */
/*   Output for GPIO pins 36 to 39 */
#define GPIO_IF1_OUT                    REG32_POINTER(GPIO_IF1_OUT_BASE)

/* GPIO_IF1_OUT bit positions (legacy definitions) */
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN3_POS 3
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN2_POS 2
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN1_POS 1
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN0_POS 0

/* GPIO_IF1_OUT bit-band aliases */
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_OUT_BASE, GPIO_IF1_OUT_OUTPUT_DATA_PIN3_POS))
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_OUT_BASE, GPIO_IF1_OUT_OUTPUT_DATA_PIN2_POS))
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_OUT_BASE, GPIO_IF1_OUT_OUTPUT_DATA_PIN1_POS))
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF1_OUT_BASE, GPIO_IF1_OUT_OUTPUT_DATA_PIN0_POS))

/* GPIO IF2/IF3 Function Select */
/*   Select the function for GPIO pins 40 to 41 which are  multiplexed with the
 *   UART0 interface, and  GPIO pins 42 to 43 which are  multiplexed with the
 *   UART1 and SQI interfaces */
#define GPIO_IF23_FUNC_SEL              REG32_POINTER(GPIO_IF23_FUNC_SEL_BASE)

/* GPIO_IF23_FUNC_SEL bit positions (legacy definitions) */
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_POS 15
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_POS 14
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_POS 13
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_POS 12
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_POS 11
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_POS 10
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_POS 9
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_POS 8
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_POS 6
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_POS))
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_POS 4
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_POS))
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_POS 1
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_POS))
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_POS 0

/* GPIO_IF23_FUNC_SEL bit-band aliases */
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_POS))
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_POS))
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_POS))
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_POS))
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_POS))
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_POS))
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_POS))
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_POS))
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_FUNC_SEL_BASE, GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_POS))

/* GPIO_IF23_FUNC_SEL subregister pointers */
#define GPIO_IF23_PIN_CFG_BYTE          REG8_POINTER(GPIO_IF23_FUNC_SEL_BASE + 1)
#define GPIO_IF23_FUNC_SEL_BYTE         REG8_POINTER(GPIO_IF23_FUNC_SEL_BASE + 0)

/* GPIO_IF23_FUNC_SEL subregister bit positions */
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN1_POS 7
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN0_POS 6
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN1_POS 5
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN0_POS 4
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN1_POS 3
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN0_POS 2
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN1_POS 1
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN0_POS 0
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_POS 6
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_POS))
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_POS 4
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_POS))
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_POS 1
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_MASK ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_POS))
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF2_POS 0

/* GPIO IF2/IF3 Output */
/*   Output for GPIO pins 40 to 43 */
#define GPIO_IF23_OUT                   REG32_POINTER(GPIO_IF23_OUT_BASE)

/* GPIO_IF23_OUT bit positions (legacy definitions) */
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN1_POS 3
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN0_POS 2
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN1_POS 1
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN0_POS 0

/* GPIO_IF23_OUT bit-band aliases */
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_OUT_BASE, GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN1_POS))
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_OUT_BASE, GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN0_POS))
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_OUT_BASE, GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN1_POS))
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF23_OUT_BASE, GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN0_POS))

/* GPIO IF0 to IF3 Collected Input and Output */
/*   Collected I/O for GPIO pins 32 to 43 (GPIO interfaces 0, 1, 2 and 3) */
#define GPIO_IF_DATA                    REG32_POINTER(GPIO_IF_DATA_BASE)

/* GPIO_IF_DATA bit positions (legacy definitions) */
#define GPIO_IF_DATA_IF3_DATA_POS       10
#define GPIO_IF_DATA_IF3_DATA_MASK      ((uint32_t)(0x3U << GPIO_IF_DATA_IF3_DATA_POS))
#define GPIO_IF_DATA_IF2_DATA_POS       8
#define GPIO_IF_DATA_IF2_DATA_MASK      ((uint32_t)(0x3U << GPIO_IF_DATA_IF2_DATA_POS))
#define GPIO_IF_DATA_IF1_DATA_POS       4
#define GPIO_IF_DATA_IF1_DATA_MASK      ((uint32_t)(0xFU << GPIO_IF_DATA_IF1_DATA_POS))
#define GPIO_IF_DATA_IF0_DATA_POS       0
#define GPIO_IF_DATA_IF0_DATA_MASK      ((uint32_t)(0xFU << GPIO_IF_DATA_IF0_DATA_POS))

/* GPIO IF4, LCD Driver Function Select */
/*   Select the function for GPIO pins 0 to 31 which are multiplexed with the
 *   segments of the LCD driver */
#define GPIO_IF4_LCD_FUNC_SEL           REG32_POINTER(GPIO_IF4_LCD_FUNC_SEL_BASE)

/* GPIO_IF4_LCD_FUNC_SEL bit positions (legacy definitions) */
#define GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_POS 0
#define GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_MASK ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_POS))

/* GPIO IF4, LCD Driver Select 0 */
/*   Select the  driver waveform for segments 0 to 7 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL0           REG32_POINTER(GPIO_IF4_LCD_DRV_SEL0_BASE)

/* GPIO_IF4_LCD_DRV_SEL0 bit positions (legacy definitions) */
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_POS 28
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_POS 24
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_POS 20
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_POS 16
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_POS 12
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_POS 8
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_POS 4
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_POS))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_POS 0
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_POS))

/* GPIO IF4, LCD Driver Select 1 */
/*   Select the  driver voltage for segments 8 to 15 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL1           REG32_POINTER(GPIO_IF4_LCD_DRV_SEL1_BASE)

/* GPIO_IF4_LCD_DRV_SEL1 bit positions (legacy definitions) */
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_POS 28
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_POS 24
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_POS 20
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_POS 16
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_POS 12
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_POS 8
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_POS 4
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_POS))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_POS 0
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_POS))

/* GPIO IF4, LCD Driver Select 2 */
/*   Select the  driver voltage for segments 16 to 23 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL2           REG32_POINTER(GPIO_IF4_LCD_DRV_SEL2_BASE)

/* GPIO_IF4_LCD_DRV_SEL2 bit positions (legacy definitions) */
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_POS 28
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_POS 24
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_POS 20
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_POS 16
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_POS 12
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_POS 8
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_POS 4
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_POS))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_POS 0
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_POS))

/* GPIO IF4, LCD Driver Select 3 */
/*   Select the  driver voltage for segments 24 to 27 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL3           REG32_POINTER(GPIO_IF4_LCD_DRV_SEL3_BASE)

/* GPIO_IF4_LCD_DRV_SEL3 bit positions (legacy definitions) */
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_POS 12
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_POS))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_POS 8
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_POS))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_POS 4
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_POS))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_POS 0
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_MASK ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_POS))

/* GPIO IF4, LCD Driver Output */
/*   Output for GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_OUT                REG32_POINTER(GPIO_IF4_LCD_OUT_BASE)

/* GPIO_IF4_LCD_OUT bit positions (legacy definitions) */
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN31_POS 31
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN30_POS 30
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN29_POS 29
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN28_POS 28
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN27_POS 27
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN26_POS 26
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN25_POS 25
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN24_POS 24
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN23_POS 23
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN22_POS 22
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN21_POS 21
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN20_POS 20
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN19_POS 19
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN18_POS 18
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN17_POS 17
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN16_POS 16
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN15_POS 15
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN14_POS 14
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN13_POS 13
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN12_POS 12
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN11_POS 11
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN10_POS 10
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN9_POS 9
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN8_POS 8
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN7_POS 7
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN6_POS 6
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN5_POS 5
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN4_POS 4
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN3_POS 3
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN2_POS 2
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN1_POS 1
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN0_POS 0

/* GPIO_IF4_LCD_OUT bit-band aliases */
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN31_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN31_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN30_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN30_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN29_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN29_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN28_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN28_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN27_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN27_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN26_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN26_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN25_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN25_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN24_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN24_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN23_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN23_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN22_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN22_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN21_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN21_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN20_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN20_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN19_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN19_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN18_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN18_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN17_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN17_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN16_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN16_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN15_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN15_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN14_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN14_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN13_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN13_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN12_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN12_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN11_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN11_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN10_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN10_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN9_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN9_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN8_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN8_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN7_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN7_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN6_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN6_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN5_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN5_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN4_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN4_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN3_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN2_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN1_POS))
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN0_POS))

/* GPIO IF4, LCD Driver Output Enable */
/*   Output enable for GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_OE                 REG32_POINTER(GPIO_IF4_LCD_OE_BASE)

/* GPIO_IF4_LCD_OE bit positions (legacy definitions) */
#define GPIO_IF4_LCD_OE_OUTPUT_ENABLE_POS 0
#define GPIO_IF4_LCD_OE_OUTPUT_ENABLE_MASK ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_OE_OUTPUT_ENABLE_POS))

/* GPIO IF4, LCD Driver Input */
/*   Input from GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_IN                 READONLY_REG32_POINTER(GPIO_IF4_LCD_IN_BASE)

/* GPIO_IF4_LCD_IN bit positions (legacy definitions) */
#define GPIO_IF4_LCD_IN_INPUT_DATA_POS  0
#define GPIO_IF4_LCD_IN_INPUT_DATA_MASK ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_IN_INPUT_DATA_POS))

/* GPIO IF4, LCD Driver Pull-down Configuration */
/*   Configure whether GPIO pins 0 to 31 use pull-down resistors */
#define GPIO_IF4_LCD_PULLDOWN           REG32_POINTER(GPIO_IF4_LCD_PULLDOWN_BASE)

/* GPIO_IF4_LCD_PULLDOWN bit positions (legacy definitions) */
#define GPIO_IF4_LCD_PULLDOWN_ENABLE_POS 0
#define GPIO_IF4_LCD_PULLDOWN_ENABLE_MASK ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_PULLDOWN_ENABLE_POS))

/* GPIO LCD Driver Control */
/*   Control the LCD interface */
#define GPIO_IF4_LCD_CTRL               REG32_POINTER(GPIO_IF4_LCD_CTRL_BASE)

/* GPIO_IF4_LCD_CTRL bit positions (legacy definitions) */
#define GPIO_IF4_LCD_CTRL_BLANK_ENABLE_POS 1
#define GPIO_IF4_LCD_CTRL_LCD_ENABLE_POS 0

/* GPIO_IF4_LCD_CTRL bit-band aliases */
#define GPIO_IF4_LCD_CTRL_BLANK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_CTRL_BASE, GPIO_IF4_LCD_CTRL_BLANK_ENABLE_POS))
#define GPIO_IF4_LCD_CTRL_LCD_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_LCD_CTRL_BASE, GPIO_IF4_LCD_CTRL_LCD_ENABLE_POS))

/* GPIO Pulse-Width Modulator Control */
/*   Select PWM functionality for GPIO pins 24 to 27 */
#define GPIO_IF4_PWM_CTRL               REG32_POINTER(GPIO_IF4_PWM_CTRL_BASE)

/* GPIO_IF4_PWM_CTRL bit positions (legacy definitions) */
#define GPIO_IF4_PWM_CTRL_PWM3_ENABLE_POS 3
#define GPIO_IF4_PWM_CTRL_PWM2_ENABLE_POS 2
#define GPIO_IF4_PWM_CTRL_PWM1_ENABLE_POS 1
#define GPIO_IF4_PWM_CTRL_PWM0_ENABLE_POS 0

/* GPIO_IF4_PWM_CTRL bit-band aliases */
#define GPIO_IF4_PWM_CTRL_PWM3_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_PWM_CTRL_BASE, GPIO_IF4_PWM_CTRL_PWM3_ENABLE_POS))
#define GPIO_IF4_PWM_CTRL_PWM2_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_PWM_CTRL_BASE, GPIO_IF4_PWM_CTRL_PWM2_ENABLE_POS))
#define GPIO_IF4_PWM_CTRL_PWM1_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_PWM_CTRL_BASE, GPIO_IF4_PWM_CTRL_PWM1_ENABLE_POS))
#define GPIO_IF4_PWM_CTRL_PWM0_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF4_PWM_CTRL_BASE, GPIO_IF4_PWM_CTRL_PWM0_ENABLE_POS))

/* GPIO IF5, Wakeup Function Select */
/*   Select the function for the dedicated wakeup 3V GPIO pins 0 to 3 */
#define GPIO_IF5_FUNC_SEL               REG32_POINTER(GPIO_IF5_FUNC_SEL_BASE)

/* GPIO_IF5_FUNC_SEL bit positions (legacy definitions) */
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_POS 15
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_POS 14
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_POS 13
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_POS 12
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS 11
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS 10
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS 9
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS 8
#define GPIO_IF5_FUNC_SEL_INPUT_DATA_POS 4
#define GPIO_IF5_FUNC_SEL_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF5_FUNC_SEL_INPUT_DATA_POS))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_POS 3
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_POS 2
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_POS 1
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_POS 0

/* GPIO_IF5_FUNC_SEL bit-band aliases */
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_POS))
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_POS))
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_POS))
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_POS))
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_POS))
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_POS))
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_POS))
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_POS))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_POS))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_POS))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_POS))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_FUNC_SEL_BASE, GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_POS))

/* GPIO_IF5_FUNC_SEL subregister pointers */
#define GPIO_IF5_PIN_CFG_BYTE           REG8_POINTER(GPIO_IF5_FUNC_SEL_BASE + 1)
#define GPIO_IF5_FUNC_SEL_BYTE          REG8_POINTER(GPIO_IF5_FUNC_SEL_BASE + 0)

/* GPIO_IF5_FUNC_SEL subregister bit positions */
#define GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN3_POS 7
#define GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN2_POS 6
#define GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_POS 5
#define GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_POS 4
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_POS 3
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_POS 2
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_POS 1
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_POS 0
#define GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_POS 4
#define GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_MASK ((uint32_t)(0xFU << GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_POS))
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN3_POS 3
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN2_POS 2
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN1_POS 1
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN0_POS 0

/* GPIO IF5, Wakeup Output */
/*   Output for 3V GPIO pins 0 to 3 */
#define GPIO_IF5_OUT                    REG32_POINTER(GPIO_IF5_OUT_BASE)

/* GPIO_IF5_OUT bit positions (legacy definitions) */
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN3_POS 3
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN2_POS 2
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN1_POS 1
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN0_POS 0

/* GPIO_IF5_OUT bit-band aliases */
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN3_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_OUT_BASE, GPIO_IF5_OUT_OUTPUT_DATA_PIN3_POS))
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_OUT_BASE, GPIO_IF5_OUT_OUTPUT_DATA_PIN2_POS))
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN1_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_OUT_BASE, GPIO_IF5_OUT_OUTPUT_DATA_PIN1_POS))
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN0_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF5_OUT_BASE, GPIO_IF5_OUT_OUTPUT_DATA_PIN0_POS))

/* GPIO Interface Input Enable */
/*   Enable or disable GPIO inputs for the GPIO interfaces */
#define GPIO_IF_INPUT_ENABLE            REG32_POINTER(GPIO_IF_INPUT_ENABLE_BASE)

/* GPIO_IF_INPUT_ENABLE bit positions (legacy definitions) */
#define GPIO_IF_INPUT_ENABLE_IF5_INPUT_POS 5
#define GPIO_IF_INPUT_ENABLE_IF4_INPUT_POS 4
#define GPIO_IF_INPUT_ENABLE_IF3_INPUT_POS 3
#define GPIO_IF_INPUT_ENABLE_IF2_INPUT_POS 2
#define GPIO_IF_INPUT_ENABLE_IF1_INPUT_POS 1
#define GPIO_IF_INPUT_ENABLE_IF0_INPUT_POS 0

/* GPIO_IF_INPUT_ENABLE bit-band aliases */
#define GPIO_IF_INPUT_ENABLE_IF5_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF5_INPUT_POS))
#define GPIO_IF_INPUT_ENABLE_IF4_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF4_INPUT_POS))
#define GPIO_IF_INPUT_ENABLE_IF3_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF3_INPUT_POS))
#define GPIO_IF_INPUT_ENABLE_IF2_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF2_INPUT_POS))
#define GPIO_IF_INPUT_ENABLE_IF1_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF1_INPUT_POS))
#define GPIO_IF_INPUT_ENABLE_IF0_INPUT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, GPIO_IF_INPUT_ENABLE_IF0_INPUT_POS))

/* GPIO Interrupt Control 0 */
/*   Configure the GPIO general-purpose interrupts 0 and 1 */
#define GPIO_INT_CTRL0                  REG32_POINTER(GPIO_INT_CTRL0_BASE)

/* GPIO_INT_CTRL0 bit positions (legacy definitions) */
#define GPIO_INT_CTRL0_GP0_INT_TYPE_POS 24
#define GPIO_INT_CTRL0_GP0_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP0_INT_TYPE_POS))
#define GPIO_INT_CTRL0_GP0_INT_SRC_POS  16
#define GPIO_INT_CTRL0_GP0_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL0_GP0_INT_SRC_POS))
#define GPIO_INT_CTRL0_GP1_INT_TYPE_POS 8
#define GPIO_INT_CTRL0_GP1_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP1_INT_TYPE_POS))
#define GPIO_INT_CTRL0_GP1_INT_SRC_POS  0
#define GPIO_INT_CTRL0_GP1_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL0_GP1_INT_SRC_POS))

/* GPIO_INT_CTRL0 subregister pointers */
#define GPIO_INT_CTRL_GP0_SHORT         REG16_POINTER(GPIO_INT_CTRL0_BASE + 2)
#define GPIO_INT_CTRL_GP1_SHORT         REG16_POINTER(GPIO_INT_CTRL0_BASE + 0)

/* GPIO_INT_CTRL0 subregister bit positions */
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_POS))
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_POS))

/* GPIO Interrupt Control 1 */
/*   Configure the GPIO general-purpose interrupts 2 and 3 */
#define GPIO_INT_CTRL1                  REG32_POINTER(GPIO_INT_CTRL1_BASE)

/* GPIO_INT_CTRL1 bit positions (legacy definitions) */
#define GPIO_INT_CTRL1_GP3_INT_TYPE_POS 24
#define GPIO_INT_CTRL1_GP3_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP3_INT_TYPE_POS))
#define GPIO_INT_CTRL1_GP3_INT_SRC_POS  16
#define GPIO_INT_CTRL1_GP3_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL1_GP3_INT_SRC_POS))
#define GPIO_INT_CTRL1_GP2_INT_TYPE_POS 8
#define GPIO_INT_CTRL1_GP2_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP2_INT_TYPE_POS))
#define GPIO_INT_CTRL1_GP2_INT_SRC_POS  0
#define GPIO_INT_CTRL1_GP2_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL1_GP2_INT_SRC_POS))

/* GPIO_INT_CTRL1 subregister pointers */
#define GPIO_INT_CTRL_GP3_SHORT         REG16_POINTER(GPIO_INT_CTRL1_BASE + 2)
#define GPIO_INT_CTRL_GP2_SHORT         REG16_POINTER(GPIO_INT_CTRL1_BASE + 0)

/* GPIO_INT_CTRL1 subregister bit positions */
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_POS))
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_POS))

/* GPIO Interrupt Control 2 */
/*   Configure the GPIO general-purpose interrupts 4 and 5 */
#define GPIO_INT_CTRL2                  REG32_POINTER(GPIO_INT_CTRL2_BASE)

/* GPIO_INT_CTRL2 bit positions (legacy definitions) */
#define GPIO_INT_CTRL2_GP5_INT_TYPE_POS 24
#define GPIO_INT_CTRL2_GP5_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP5_INT_TYPE_POS))
#define GPIO_INT_CTRL2_GP5_INT_SRC_POS  16
#define GPIO_INT_CTRL2_GP5_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL2_GP5_INT_SRC_POS))
#define GPIO_INT_CTRL2_GP4_INT_TYPE_POS 8
#define GPIO_INT_CTRL2_GP4_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP4_INT_TYPE_POS))
#define GPIO_INT_CTRL2_GP4_INT_SRC_POS  0
#define GPIO_INT_CTRL2_GP4_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL2_GP4_INT_SRC_POS))

/* GPIO_INT_CTRL2 subregister pointers */
#define GPIO_INT_CTRL_GP5_SHORT         REG16_POINTER(GPIO_INT_CTRL2_BASE + 2)
#define GPIO_INT_CTRL_GP4_SHORT         REG16_POINTER(GPIO_INT_CTRL2_BASE + 0)

/* GPIO_INT_CTRL2 subregister bit positions */
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_POS))
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_POS))

/* GPIO Interrupt Control 3 */
/*   Configure the GPIO general-purpose interrupts 2 and 3 */
#define GPIO_INT_CTRL3                  REG32_POINTER(GPIO_INT_CTRL3_BASE)

/* GPIO_INT_CTRL3 bit positions (legacy definitions) */
#define GPIO_INT_CTRL3_GP7_INT_TYPE_POS 24
#define GPIO_INT_CTRL3_GP7_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP7_INT_TYPE_POS))
#define GPIO_INT_CTRL3_GP7_INT_SRC_POS  16
#define GPIO_INT_CTRL3_GP7_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL3_GP7_INT_SRC_POS))
#define GPIO_INT_CTRL3_GP6_INT_TYPE_POS 8
#define GPIO_INT_CTRL3_GP6_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP6_INT_TYPE_POS))
#define GPIO_INT_CTRL3_GP6_INT_SRC_POS  0
#define GPIO_INT_CTRL3_GP6_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL3_GP6_INT_SRC_POS))

/* GPIO_INT_CTRL3 subregister pointers */
#define GPIO_INT_CTRL_GP7_SHORT         REG16_POINTER(GPIO_INT_CTRL3_BASE + 2)
#define GPIO_INT_CTRL_GP6_SHORT         REG16_POINTER(GPIO_INT_CTRL3_BASE + 0)

/* GPIO_INT_CTRL3 subregister bit positions */
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_POS))
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_POS 8
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_MASK ((uint32_t)(0x7U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_POS))
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_POS 0
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_MASK ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_POS))

/* ----------------------------------------------------------------------------
 * SPI0 Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The serial-peripheral interface (SPI) module provides a synchronous 4-wire
 * interface including clock, chip select, serial data in, and serial data out.
 *
 * This interface is designed to operate as either a master or slave connecting
 * to external storage devices, advanced displays, wireless transceivers and
 * other data sources or sinks without the need for external components. */

/* SPI0 Control and Configuration Register */
/*   Configure the SPI0 interface */
#define SPI0_CTRL0                      REG32_POINTER(SPI0_CTRL0_BASE)

/* SPI0_CTRL0 bit positions (legacy definitions) */
#define SPI0_CTRL0_SQI_ENABLE_POS       11
#define SPI0_CTRL0_OVERRUN_INT_ENABLE_POS 10
#define SPI0_CTRL0_UNDERRUN_INT_ENABLE_POS 9
#define SPI0_CTRL0_CONTROLLER_POS       8
#define SPI0_CTRL0_SLAVE_POS            7
#define SPI0_CTRL0_SERI_PULLUP_ENABLE_POS 6
#define SPI0_CTRL0_CLK_POLARITY_POS     5
#define SPI0_CTRL0_MODE_SELECT_POS      4
#define SPI0_CTRL0_ENABLE_POS           3
#define SPI0_CTRL0_PRESCALE_POS         0
#define SPI0_CTRL0_PRESCALE_MASK        ((uint32_t)(0x7U << SPI0_CTRL0_PRESCALE_POS))

/* SPI0_CTRL0 bit-band aliases */
#define SPI0_CTRL0_SQI_ENABLE_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_SQI_ENABLE_POS))
#define SPI0_CTRL0_OVERRUN_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_OVERRUN_INT_ENABLE_POS))
#define SPI0_CTRL0_UNDERRUN_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_UNDERRUN_INT_ENABLE_POS))
#define SPI0_CTRL0_CONTROLLER_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_CONTROLLER_POS))
#define SPI0_CTRL0_SLAVE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_SLAVE_POS))
#define SPI0_CTRL0_SERI_PULLUP_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_SERI_PULLUP_ENABLE_POS))
#define SPI0_CTRL0_CLK_POLARITY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_CLK_POLARITY_POS))
#define SPI0_CTRL0_MODE_SELECT_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_MODE_SELECT_POS))
#define SPI0_CTRL0_ENABLE_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL0_BASE, SPI0_CTRL0_ENABLE_POS))

/* SPI0_CTRL0 settings */
#define SPI0_SELECT_SPI_BITBAND         0x0
#define SPI0_SELECT_SQI_BITBAND         0x1
#define SPI0_SELECT_SPI                 ((uint32_t)(SPI0_SELECT_SPI_BITBAND << SPI0_CTRL0_SQI_ENABLE_POS))
#define SPI0_SELECT_SQI                 ((uint32_t)(SPI0_SELECT_SQI_BITBAND << SPI0_CTRL0_SQI_ENABLE_POS))

#define SPI0_OVERRUN_INT_DISABLE_BITBAND 0x0
#define SPI0_OVERRUN_INT_ENABLE_BITBAND 0x1
#define SPI0_OVERRUN_INT_DISABLE        ((uint32_t)(SPI0_OVERRUN_INT_DISABLE_BITBAND << SPI0_CTRL0_OVERRUN_INT_ENABLE_POS))
#define SPI0_OVERRUN_INT_ENABLE         ((uint32_t)(SPI0_OVERRUN_INT_ENABLE_BITBAND << SPI0_CTRL0_OVERRUN_INT_ENABLE_POS))

#define SPI0_UNDERRUN_INT_DISABLE_BITBAND 0x0
#define SPI0_UNDERRUN_INT_ENABLE_BITBAND 0x1
#define SPI0_UNDERRUN_INT_DISABLE       ((uint32_t)(SPI0_UNDERRUN_INT_DISABLE_BITBAND << SPI0_CTRL0_UNDERRUN_INT_ENABLE_POS))
#define SPI0_UNDERRUN_INT_ENABLE        ((uint32_t)(SPI0_UNDERRUN_INT_ENABLE_BITBAND << SPI0_CTRL0_UNDERRUN_INT_ENABLE_POS))

#define SPI0_CONTROLLER_CM3_BITBAND     0x0
#define SPI0_CONTROLLER_DMA_BITBAND     0x1
#define SPI0_CONTROLLER_CM3             ((uint32_t)(SPI0_CONTROLLER_CM3_BITBAND << SPI0_CTRL0_CONTROLLER_POS))
#define SPI0_CONTROLLER_DMA             ((uint32_t)(SPI0_CONTROLLER_DMA_BITBAND << SPI0_CTRL0_CONTROLLER_POS))

#define SPI0_SELECT_MASTER_BITBAND      0x0
#define SPI0_SELECT_SLAVE_BITBAND       0x1
#define SPI0_SELECT_MASTER              ((uint32_t)(SPI0_SELECT_MASTER_BITBAND << SPI0_CTRL0_SLAVE_POS))
#define SPI0_SELECT_SLAVE               ((uint32_t)(SPI0_SELECT_SLAVE_BITBAND << SPI0_CTRL0_SLAVE_POS))

#define SPI0_SERI_PULLUP_DISABLE_BITBAND 0x0
#define SPI0_SERI_PULLUP_ENABLE_BITBAND 0x1
#define SPI0_SERI_PULLUP_DISABLE        ((uint32_t)(SPI0_SERI_PULLUP_DISABLE_BITBAND << SPI0_CTRL0_SERI_PULLUP_ENABLE_POS))
#define SPI0_SERI_PULLUP_ENABLE         ((uint32_t)(SPI0_SERI_PULLUP_ENABLE_BITBAND << SPI0_CTRL0_SERI_PULLUP_ENABLE_POS))

#define SPI0_CLK_POLARITY_NORMAL_BITBAND 0x0
#define SPI0_CLK_POLARITY_INVERSE_BITBAND 0x1
#define SPI0_CLK_POLARITY_NORMAL        ((uint32_t)(SPI0_CLK_POLARITY_NORMAL_BITBAND << SPI0_CTRL0_CLK_POLARITY_POS))
#define SPI0_CLK_POLARITY_INVERSE       ((uint32_t)(SPI0_CLK_POLARITY_INVERSE_BITBAND << SPI0_CTRL0_CLK_POLARITY_POS))

#define SPI0_MODE_SELECT_MANUAL_BITBAND 0x0
#define SPI0_MODE_SELECT_AUTO_BITBAND   0x1
#define SPI0_MODE_SELECT_MANUAL         ((uint32_t)(SPI0_MODE_SELECT_MANUAL_BITBAND << SPI0_CTRL0_MODE_SELECT_POS))
#define SPI0_MODE_SELECT_AUTO           ((uint32_t)(SPI0_MODE_SELECT_AUTO_BITBAND << SPI0_CTRL0_MODE_SELECT_POS))

#define SPI0_DISABLE_BITBAND            0x0
#define SPI0_ENABLE_BITBAND             0x1
#define SPI0_DISABLE                    ((uint32_t)(SPI0_DISABLE_BITBAND << SPI0_CTRL0_ENABLE_POS))
#define SPI0_ENABLE                     ((uint32_t)(SPI0_ENABLE_BITBAND << SPI0_CTRL0_ENABLE_POS))

#define SPI0_PRESCALE_2                 ((uint32_t)(0x0U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_4                 ((uint32_t)(0x1U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_8                 ((uint32_t)(0x2U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_16                ((uint32_t)(0x3U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_32                ((uint32_t)(0x4U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_64                ((uint32_t)(0x5U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_128               ((uint32_t)(0x6U << SPI0_CTRL0_PRESCALE_POS))
#define SPI0_PRESCALE_256               ((uint32_t)(0x7U << SPI0_CTRL0_PRESCALE_POS))

/* SPI0 Transaction Control Register */
/*   Control transactions on the SPI0 interface */
#define SPI0_CTRL1                      REG32_POINTER(SPI0_CTRL1_BASE)

/* SPI0_CTRL1 bit positions (legacy definitions) */
#define SPI0_CTRL1_START_BUSY_POS       7
#define SPI0_CTRL1_RW_CMD_POS           6
#define SPI0_CTRL1_CS_POS               5
#define SPI0_CTRL1_WORD_SIZE_POS        0
#define SPI0_CTRL1_WORD_SIZE_MASK       ((uint32_t)(0x1FU << SPI0_CTRL1_WORD_SIZE_POS))

/* SPI0_CTRL1 bit-band aliases */
#define SPI0_CTRL1_START_BUSY_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL1_BASE, SPI0_CTRL1_START_BUSY_POS))
#define SPI0_CTRL1_RW_CMD_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL1_BASE, SPI0_CTRL1_RW_CMD_POS))
#define SPI0_CTRL1_CS_BITBAND           REG32_POINTER(SYS_CALC_BITBAND(SPI0_CTRL1_BASE, SPI0_CTRL1_CS_POS))

/* SPI0_CTRL1 settings */
#define SPI0_IDLE_BITBAND               0x0
#define SPI0_START_BITBAND              0x1
#define SPI0_BUSY_BITBAND               0x1
#define SPI0_IDLE                       ((uint32_t)(SPI0_IDLE_BITBAND << SPI0_CTRL1_START_BUSY_POS))
#define SPI0_START                      ((uint32_t)(SPI0_START_BITBAND << SPI0_CTRL1_START_BUSY_POS))
#define SPI0_BUSY                       ((uint32_t)(SPI0_BUSY_BITBAND << SPI0_CTRL1_START_BUSY_POS))

#define SPI0_WRITE_DATA_BITBAND         0x0
#define SPI0_READ_DATA_BITBAND          0x1
#define SPI0_WRITE_DATA                 ((uint32_t)(SPI0_WRITE_DATA_BITBAND << SPI0_CTRL1_RW_CMD_POS))
#define SPI0_READ_DATA                  ((uint32_t)(SPI0_READ_DATA_BITBAND << SPI0_CTRL1_RW_CMD_POS))

#define SPI0_CS_0_BITBAND               0x0
#define SPI0_CS_1_BITBAND               0x1
#define SPI0_CS_0                       ((uint32_t)(SPI0_CS_0_BITBAND << SPI0_CTRL1_CS_POS))
#define SPI0_CS_1                       ((uint32_t)(SPI0_CS_1_BITBAND << SPI0_CTRL1_CS_POS))

#define SPI0_WORD_SIZE_1                ((uint32_t)(0x0U << SPI0_CTRL1_WORD_SIZE_POS))
#define SPI0_WORD_SIZE_8                ((uint32_t)(0x7U << SPI0_CTRL1_WORD_SIZE_POS))
#define SPI0_WORD_SIZE_16               ((uint32_t)(0xFU << SPI0_CTRL1_WORD_SIZE_POS))
#define SPI0_WORD_SIZE_24               ((uint32_t)(0x17U << SPI0_CTRL1_WORD_SIZE_POS))
#define SPI0_WORD_SIZE_32               ((uint32_t)(0x1FU << SPI0_CTRL1_WORD_SIZE_POS))

/* SPI0 Data */
/*   Single word buffer for data to be transmitted over the SPI0 interface and
 *   data that has been received over the SPI0 interface */
#define SPI0_DATA                       REG32_POINTER(SPI0_DATA_BASE)

/* Shadow of SPI0 Data */
/*   Shadow register for SPI0_DATA providing no side effect access to the
 *   interface's data */
#define SPI0_DATA_S                     REG32_POINTER(SPI0_DATA_S_BASE)

/* SPI0 Data for Slave Operations */
/*   Single word buffer for data received over the SPI0 interface when
 *   operating in slave mode */
#define SPI0_SLAVE_DATA                 REG32_POINTER(SPI0_SLAVE_DATA_BASE)

/* SPI0_SLAVE_DATA bit positions (legacy definitions) */
#define SPI0_SLAVE_DATA_SPI0_DATA_POS   0
#define SPI0_SLAVE_DATA_SPI0_DATA_MASK  ((uint32_t)(0xFFFFFFFFU << SPI0_SLAVE_DATA_SPI0_DATA_POS))

/* SPI0 Status */
/*   Status of the SPI0 interface */
#define SPI0_STATUS                     REG32_POINTER(SPI0_STATUS_BASE)

/* SPI0_STATUS bit positions (legacy definitions) */
#define SPI0_STATUS_OVERRUN_STATUS_POS  1
#define SPI0_STATUS_UNDERRUN_STATUS_POS 0

/* SPI0_STATUS bit-band aliases */
#define SPI0_STATUS_OVERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_STATUS_BASE, SPI0_STATUS_OVERRUN_STATUS_POS))
#define SPI0_STATUS_UNDERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI0_STATUS_BASE, SPI0_STATUS_UNDERRUN_STATUS_POS))

/* SPI0_STATUS settings */
#define SPI0_OVERRUN_FALSE_BITBAND      0x0
#define SPI0_OVERRUN_TRUE_BITBAND       0x1
#define SPI0_OVERRUN_CLEAR_BITBAND      0x1
#define SPI0_OVERRUN_FALSE              ((uint32_t)(SPI0_OVERRUN_FALSE_BITBAND << SPI0_STATUS_OVERRUN_STATUS_POS))
#define SPI0_OVERRUN_TRUE               ((uint32_t)(SPI0_OVERRUN_TRUE_BITBAND << SPI0_STATUS_OVERRUN_STATUS_POS))
#define SPI0_OVERRUN_CLEAR              ((uint32_t)(SPI0_OVERRUN_CLEAR_BITBAND << SPI0_STATUS_OVERRUN_STATUS_POS))

#define SPI0_UNDERRUN_FALSE_BITBAND     0x0
#define SPI0_UNDERRUN_TRUE_BITBAND      0x1
#define SPI0_UNDERRUN_CLEAR_BITBAND     0x1
#define SPI0_UNDERRUN_FALSE             ((uint32_t)(SPI0_UNDERRUN_FALSE_BITBAND << SPI0_STATUS_UNDERRUN_STATUS_POS))
#define SPI0_UNDERRUN_TRUE              ((uint32_t)(SPI0_UNDERRUN_TRUE_BITBAND << SPI0_STATUS_UNDERRUN_STATUS_POS))
#define SPI0_UNDERRUN_CLEAR             ((uint32_t)(SPI0_UNDERRUN_CLEAR_BITBAND << SPI0_STATUS_UNDERRUN_STATUS_POS))

/* ----------------------------------------------------------------------------
 * SPI1 Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The serial-peripheral interface (SPI) module provides a synchronous 4-wire
 * interface including clock, chip select, serial data in, and serial data out.
 *
 * This interface is designed to operate as either a master or slave connecting
 * to external storage devices, advanced displays, wireless transceivers and
 * other data sources or sinks without the need for external components. */

/* SPI1 Control and Configuration Register */
/*   Configure the SPI1 interface */
#define SPI1_CTRL0                      REG32_POINTER(SPI1_CTRL0_BASE)

/* SPI1_CTRL0 bit positions (legacy definitions) */
#define SPI1_CTRL0_OVERRUN_INT_ENABLE_POS 10
#define SPI1_CTRL0_UNDERRUN_INT_ENABLE_POS 9
#define SPI1_CTRL0_CONTROLLER_POS       8
#define SPI1_CTRL0_SLAVE_POS            7
#define SPI1_CTRL0_SERI_PULLUP_ENABLE_POS 6
#define SPI1_CTRL0_CLK_POLARITY_POS     5
#define SPI1_CTRL0_MODE_SELECT_POS      4
#define SPI1_CTRL0_ENABLE_POS           3
#define SPI1_CTRL0_PRESCALE_POS         0
#define SPI1_CTRL0_PRESCALE_MASK        ((uint32_t)(0x7U << SPI1_CTRL0_PRESCALE_POS))

/* SPI1_CTRL0 bit-band aliases */
#define SPI1_CTRL0_OVERRUN_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_OVERRUN_INT_ENABLE_POS))
#define SPI1_CTRL0_UNDERRUN_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_UNDERRUN_INT_ENABLE_POS))
#define SPI1_CTRL0_CONTROLLER_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_CONTROLLER_POS))
#define SPI1_CTRL0_SLAVE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_SLAVE_POS))
#define SPI1_CTRL0_SERI_PULLUP_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_SERI_PULLUP_ENABLE_POS))
#define SPI1_CTRL0_CLK_POLARITY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_CLK_POLARITY_POS))
#define SPI1_CTRL0_MODE_SELECT_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_MODE_SELECT_POS))
#define SPI1_CTRL0_ENABLE_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL0_BASE, SPI1_CTRL0_ENABLE_POS))

/* SPI1_CTRL0 settings */
#define SPI1_OVERRUN_INT_DISABLE_BITBAND 0x0
#define SPI1_OVERRUN_INT_ENABLE_BITBAND 0x1
#define SPI1_OVERRUN_INT_DISABLE        ((uint32_t)(SPI1_OVERRUN_INT_DISABLE_BITBAND << SPI1_CTRL0_OVERRUN_INT_ENABLE_POS))
#define SPI1_OVERRUN_INT_ENABLE         ((uint32_t)(SPI1_OVERRUN_INT_ENABLE_BITBAND << SPI1_CTRL0_OVERRUN_INT_ENABLE_POS))

#define SPI1_UNDERRUN_INT_DISABLE_BITBAND 0x0
#define SPI1_UNDERRUN_INT_ENABLE_BITBAND 0x1
#define SPI1_UNDERRUN_INT_DISABLE       ((uint32_t)(SPI1_UNDERRUN_INT_DISABLE_BITBAND << SPI1_CTRL0_UNDERRUN_INT_ENABLE_POS))
#define SPI1_UNDERRUN_INT_ENABLE        ((uint32_t)(SPI1_UNDERRUN_INT_ENABLE_BITBAND << SPI1_CTRL0_UNDERRUN_INT_ENABLE_POS))

#define SPI1_CONTROLLER_CM3_BITBAND     0x0
#define SPI1_CONTROLLER_DMA_BITBAND     0x1
#define SPI1_CONTROLLER_CM3             ((uint32_t)(SPI1_CONTROLLER_CM3_BITBAND << SPI1_CTRL0_CONTROLLER_POS))
#define SPI1_CONTROLLER_DMA             ((uint32_t)(SPI1_CONTROLLER_DMA_BITBAND << SPI1_CTRL0_CONTROLLER_POS))

#define SPI1_SELECT_MASTER_BITBAND      0x0
#define SPI1_SELECT_SLAVE_BITBAND       0x1
#define SPI1_SELECT_MASTER              ((uint32_t)(SPI1_SELECT_MASTER_BITBAND << SPI1_CTRL0_SLAVE_POS))
#define SPI1_SELECT_SLAVE               ((uint32_t)(SPI1_SELECT_SLAVE_BITBAND << SPI1_CTRL0_SLAVE_POS))

#define SPI1_SERI_PULLUP_DISABLE_BITBAND 0x0
#define SPI1_SERI_PULLUP_ENABLE_BITBAND 0x1
#define SPI1_SERI_PULLUP_DISABLE        ((uint32_t)(SPI1_SERI_PULLUP_DISABLE_BITBAND << SPI1_CTRL0_SERI_PULLUP_ENABLE_POS))
#define SPI1_SERI_PULLUP_ENABLE         ((uint32_t)(SPI1_SERI_PULLUP_ENABLE_BITBAND << SPI1_CTRL0_SERI_PULLUP_ENABLE_POS))

#define SPI1_CLK_POLARITY_NORMAL_BITBAND 0x0
#define SPI1_CLK_POLARITY_INVERSE_BITBAND 0x1
#define SPI1_CLK_POLARITY_NORMAL        ((uint32_t)(SPI1_CLK_POLARITY_NORMAL_BITBAND << SPI1_CTRL0_CLK_POLARITY_POS))
#define SPI1_CLK_POLARITY_INVERSE       ((uint32_t)(SPI1_CLK_POLARITY_INVERSE_BITBAND << SPI1_CTRL0_CLK_POLARITY_POS))

#define SPI1_MODE_SELECT_MANUAL_BITBAND 0x0
#define SPI1_MODE_SELECT_AUTO_BITBAND   0x1
#define SPI1_MODE_SELECT_MANUAL         ((uint32_t)(SPI1_MODE_SELECT_MANUAL_BITBAND << SPI1_CTRL0_MODE_SELECT_POS))
#define SPI1_MODE_SELECT_AUTO           ((uint32_t)(SPI1_MODE_SELECT_AUTO_BITBAND << SPI1_CTRL0_MODE_SELECT_POS))

#define SPI1_DISABLE_BITBAND            0x0
#define SPI1_ENABLE_BITBAND             0x1
#define SPI1_DISABLE                    ((uint32_t)(SPI1_DISABLE_BITBAND << SPI1_CTRL0_ENABLE_POS))
#define SPI1_ENABLE                     ((uint32_t)(SPI1_ENABLE_BITBAND << SPI1_CTRL0_ENABLE_POS))

#define SPI1_PRESCALE_2                 ((uint32_t)(0x0U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_4                 ((uint32_t)(0x1U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_8                 ((uint32_t)(0x2U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_16                ((uint32_t)(0x3U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_32                ((uint32_t)(0x4U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_64                ((uint32_t)(0x5U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_128               ((uint32_t)(0x6U << SPI1_CTRL0_PRESCALE_POS))
#define SPI1_PRESCALE_256               ((uint32_t)(0x7U << SPI1_CTRL0_PRESCALE_POS))

/* SPI1 Transaction Control Register */
/*   Control transactions on the SPI1 interface */
#define SPI1_CTRL1                      REG32_POINTER(SPI1_CTRL1_BASE)

/* SPI1_CTRL1 bit positions (legacy definitions) */
#define SPI1_CTRL1_START_BUSY_POS       7
#define SPI1_CTRL1_RW_CMD_POS           6
#define SPI1_CTRL1_CS_POS               5
#define SPI1_CTRL1_WORD_SIZE_POS        0
#define SPI1_CTRL1_WORD_SIZE_MASK       ((uint32_t)(0x1FU << SPI1_CTRL1_WORD_SIZE_POS))

/* SPI1_CTRL1 bit-band aliases */
#define SPI1_CTRL1_START_BUSY_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL1_BASE, SPI1_CTRL1_START_BUSY_POS))
#define SPI1_CTRL1_RW_CMD_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL1_BASE, SPI1_CTRL1_RW_CMD_POS))
#define SPI1_CTRL1_CS_BITBAND           REG32_POINTER(SYS_CALC_BITBAND(SPI1_CTRL1_BASE, SPI1_CTRL1_CS_POS))

/* SPI1_CTRL1 settings */
#define SPI1_IDLE_BITBAND               0x0
#define SPI1_START_BITBAND              0x1
#define SPI1_BUSY_BITBAND               0x1
#define SPI1_IDLE                       ((uint32_t)(SPI1_IDLE_BITBAND << SPI1_CTRL1_START_BUSY_POS))
#define SPI1_START                      ((uint32_t)(SPI1_START_BITBAND << SPI1_CTRL1_START_BUSY_POS))
#define SPI1_BUSY                       ((uint32_t)(SPI1_BUSY_BITBAND << SPI1_CTRL1_START_BUSY_POS))

#define SPI1_WRITE_DATA_BITBAND         0x0
#define SPI1_READ_DATA_BITBAND          0x1
#define SPI1_WRITE_DATA                 ((uint32_t)(SPI1_WRITE_DATA_BITBAND << SPI1_CTRL1_RW_CMD_POS))
#define SPI1_READ_DATA                  ((uint32_t)(SPI1_READ_DATA_BITBAND << SPI1_CTRL1_RW_CMD_POS))

#define SPI1_CS_0_BITBAND               0x0
#define SPI1_CS_1_BITBAND               0x1
#define SPI1_CS_0                       ((uint32_t)(SPI1_CS_0_BITBAND << SPI1_CTRL1_CS_POS))
#define SPI1_CS_1                       ((uint32_t)(SPI1_CS_1_BITBAND << SPI1_CTRL1_CS_POS))

#define SPI1_WORD_SIZE_1                ((uint32_t)(0x0U << SPI1_CTRL1_WORD_SIZE_POS))
#define SPI1_WORD_SIZE_8                ((uint32_t)(0x7U << SPI1_CTRL1_WORD_SIZE_POS))
#define SPI1_WORD_SIZE_16               ((uint32_t)(0xFU << SPI1_CTRL1_WORD_SIZE_POS))
#define SPI1_WORD_SIZE_24               ((uint32_t)(0x17U << SPI1_CTRL1_WORD_SIZE_POS))
#define SPI1_WORD_SIZE_32               ((uint32_t)(0x1FU << SPI1_CTRL1_WORD_SIZE_POS))

/* SPI1 Data */
/*   Single word buffer for data to be transmitted over the SPI1 interface and
 *   data that has been received over the SPI1 interface */
#define SPI1_DATA                       REG32_POINTER(SPI1_DATA_BASE)

/* Shadow of SPI1 Data */
/*   Shadow register for SPI1_DATA providing no side effect access to the
 *   interface's data */
#define SPI1_DATA_S                     REG32_POINTER(SPI1_DATA_S_BASE)

/* SPI1 Data for Slave Operations */
/*   Single word buffer for data received over the SPI1 interface when
 *   operating in slave mode */
#define SPI1_SLAVE_DATA                 REG32_POINTER(SPI1_SLAVE_DATA_BASE)

/* SPI1_SLAVE_DATA bit positions (legacy definitions) */
#define SPI1_SLAVE_DATA_SPI1_DATA_POS   0
#define SPI1_SLAVE_DATA_SPI1_DATA_MASK  ((uint32_t)(0xFFFFFFFFU << SPI1_SLAVE_DATA_SPI1_DATA_POS))

/* SPI1 Status */
/*   Status of the SPI1 interface */
#define SPI1_STATUS                     REG32_POINTER(SPI1_STATUS_BASE)

/* SPI1_STATUS bit positions (legacy definitions) */
#define SPI1_STATUS_OVERRUN_STATUS_POS  1
#define SPI1_STATUS_UNDERRUN_STATUS_POS 0

/* SPI1_STATUS bit-band aliases */
#define SPI1_STATUS_OVERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_STATUS_BASE, SPI1_STATUS_OVERRUN_STATUS_POS))
#define SPI1_STATUS_UNDERRUN_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(SPI1_STATUS_BASE, SPI1_STATUS_UNDERRUN_STATUS_POS))

/* SPI1_STATUS settings */
#define SPI1_OVERRUN_FALSE_BITBAND      0x0
#define SPI1_OVERRUN_TRUE_BITBAND       0x1
#define SPI1_OVERRUN_CLEAR_BITBAND      0x1
#define SPI1_OVERRUN_FALSE              ((uint32_t)(SPI1_OVERRUN_FALSE_BITBAND << SPI1_STATUS_OVERRUN_STATUS_POS))
#define SPI1_OVERRUN_TRUE               ((uint32_t)(SPI1_OVERRUN_TRUE_BITBAND << SPI1_STATUS_OVERRUN_STATUS_POS))
#define SPI1_OVERRUN_CLEAR              ((uint32_t)(SPI1_OVERRUN_CLEAR_BITBAND << SPI1_STATUS_OVERRUN_STATUS_POS))

#define SPI1_UNDERRUN_FALSE_BITBAND     0x0
#define SPI1_UNDERRUN_TRUE_BITBAND      0x1
#define SPI1_UNDERRUN_CLEAR_BITBAND     0x1
#define SPI1_UNDERRUN_FALSE             ((uint32_t)(SPI1_UNDERRUN_FALSE_BITBAND << SPI1_STATUS_UNDERRUN_STATUS_POS))
#define SPI1_UNDERRUN_TRUE              ((uint32_t)(SPI1_UNDERRUN_TRUE_BITBAND << SPI1_STATUS_UNDERRUN_STATUS_POS))
#define SPI1_UNDERRUN_CLEAR             ((uint32_t)(SPI1_UNDERRUN_CLEAR_BITBAND << SPI1_STATUS_UNDERRUN_STATUS_POS))

/* ----------------------------------------------------------------------------
 * PCM Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The pulse-code modulation (PCM) interface module provides a data connection
 * to be used with Bluetooth chips, other wireless devices or other data
 * sources without the need for additional external components.
 *
 * The PCM interface shares a set of multiplexed pins with the SPI1
 * interface. */

/* PCM Control */
/*   Configuration of the PCM frame signal, data, and interface behavior. */
#define PCM_CTRL                        REG32_POINTER(PCM_CTRL_BASE)

/* PCM_CTRL bit positions (legacy definitions) */
#define PCM_CTRL_PULLUP_ENABLE_POS      13
#define PCM_CTRL_BIT_ORDER_POS          12
#define PCM_CTRL_TX_ALIGN_POS           11
#define PCM_CTRL_WORD_SIZE_POS          9
#define PCM_CTRL_WORD_SIZE_MASK         ((uint32_t)(0x3U << PCM_CTRL_WORD_SIZE_POS))
#define PCM_CTRL_FRAME_ALIGN_POS        8
#define PCM_CTRL_FRAME_WIDTH_POS        7
#define PCM_CTRL_FRAME_LENGTH_POS       4
#define PCM_CTRL_FRAME_LENGTH_MASK      ((uint32_t)(0x7U << PCM_CTRL_FRAME_LENGTH_POS))
#define PCM_CTRL_FRAME_SUBFRAMES_POS    3
#define PCM_CTRL_CONTROLLER_POS         2
#define PCM_CTRL_ENABLE_POS             1
#define PCM_CTRL_SLAVE_POS              0

/* PCM_CTRL bit-band aliases */
#define PCM_CTRL_PULLUP_ENABLE_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_PULLUP_ENABLE_POS))
#define PCM_CTRL_BIT_ORDER_BITBAND      REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_BIT_ORDER_POS))
#define PCM_CTRL_TX_ALIGN_BITBAND       REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_TX_ALIGN_POS))
#define PCM_CTRL_FRAME_ALIGN_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_FRAME_ALIGN_POS))
#define PCM_CTRL_FRAME_WIDTH_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_FRAME_WIDTH_POS))
#define PCM_CTRL_FRAME_SUBFRAMES_BITBAND REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_FRAME_SUBFRAMES_POS))
#define PCM_CTRL_CONTROLLER_BITBAND     REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_CONTROLLER_POS))
#define PCM_CTRL_ENABLE_BITBAND         REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_ENABLE_POS))
#define PCM_CTRL_SLAVE_BITBAND          REG32_POINTER(SYS_CALC_BITBAND(PCM_CTRL_BASE, PCM_CTRL_SLAVE_POS))

/* PCM Transmit Data */
/*   Single word buffer for the PCM interface containing the next word to
 *   transmit over the PCM interface. */
#define PCM_TX_DATA                     REG32_POINTER(PCM_TX_DATA_BASE)

/* PCM Receive Data */
/*   Single word buffer for the PCM interface containing the most recently
 *   received word (read-only). */
#define PCM_RX_DATA                     READONLY_REG32_POINTER(PCM_RX_DATA_BASE)

/* ----------------------------------------------------------------------------
 * I2C Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The inter-IC (I2C) interface implements an interface which is compatible
 * with the I2C Bus Specification from Philips Semiconductors.
 *
 * This interface is designed to operate in both master (provides limited bus
 * arbitration) and slave modes and provide an interface to external storage
 * devices, control interfaces for external components and other I2C
 * devices. */

/* I2C Interface Configuration and Control */
/*   Configuration of the I2C interface including master clock speed, slave
 *   address and other interface parameters */
#define I2C_CTRL0                       REG32_POINTER(I2C_CTRL0_BASE)

/* I2C_CTRL0 bit positions (legacy definitions) */
#define I2C_CTRL0_MASTER_SPEED_PRESCALAR_POS 16
#define I2C_CTRL0_MASTER_SPEED_PRESCALAR_MASK ((uint32_t)(0xFFU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_POS))
#define I2C_CTRL0_SLAVE_ADDRESS_POS     8
#define I2C_CTRL0_SLAVE_ADDRESS_MASK    ((uint32_t)(0x7FU << I2C_CTRL0_SLAVE_ADDRESS_POS))
#define I2C_CTRL0_CONTROLLER_POS        4
#define I2C_CTRL0_STOP_INT_ENABLE_POS   3
#define I2C_CTRL0_AUTO_ACK_ENABLE_POS   2
#define I2C_CTRL0_SLAVE_ENABLE_POS      0

/* I2C_CTRL0 bit-band aliases */
#define I2C_CTRL0_CONTROLLER_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL0_BASE, I2C_CTRL0_CONTROLLER_POS))
#define I2C_CTRL0_STOP_INT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL0_BASE, I2C_CTRL0_STOP_INT_ENABLE_POS))
#define I2C_CTRL0_AUTO_ACK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL0_BASE, I2C_CTRL0_AUTO_ACK_ENABLE_POS))
#define I2C_CTRL0_SLAVE_ENABLE_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL0_BASE, I2C_CTRL0_SLAVE_ENABLE_POS))

/* I2C_CTRL0 subregister pointers */
#define I2C_MASTER_SPEED_PRESCALAR_BYTE REG8_POINTER(I2C_CTRL0_BASE + 2)
#define I2C_SLAVE_ADDRESS_BYTE          REG8_POINTER(I2C_CTRL0_BASE + 1)
#define I2C_CTRL0_BYTE                  REG8_POINTER(I2C_CTRL0_BASE + 0)

/* I2C_CTRL0 subregister bit positions */
#define I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_POS 0
#define I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_MASK ((uint32_t)(0x7FU << I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_POS))
#define I2C_CTRL0_BYTE_CONTROLLER_POS   4
#define I2C_CTRL0_BYTE_STOP_INT_ENABLE_POS 3
#define I2C_CTRL0_BYTE_AUTO_ACK_ENABLE_POS 2
#define I2C_CTRL0_BYTE_SLAVE_ENABLE_POS 0

/* I2C Interface Status and Control */
/*   Control transfers using the I2C interface */
#define I2C_CTRL1                       REG32_POINTER(I2C_CTRL1_BASE)

/* I2C_CTRL1 bit positions (legacy definitions) */
#define I2C_CTRL1_RESET_POS             5
#define I2C_CTRL1_LAST_DATA_POS         4
#define I2C_CTRL1_STOP_POS              3
#define I2C_CTRL1_NACK_POS              1
#define I2C_CTRL1_ACK_POS               0

/* I2C_CTRL1 bit-band aliases */
#define I2C_CTRL1_RESET_BITBAND         REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL1_BASE, I2C_CTRL1_RESET_POS))
#define I2C_CTRL1_LAST_DATA_BITBAND     REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL1_BASE, I2C_CTRL1_LAST_DATA_POS))
#define I2C_CTRL1_STOP_BITBAND          REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL1_BASE, I2C_CTRL1_STOP_POS))
#define I2C_CTRL1_NACK_BITBAND          REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL1_BASE, I2C_CTRL1_NACK_POS))
#define I2C_CTRL1_ACK_BITBAND           REG32_POINTER(SYS_CALC_BITBAND(I2C_CTRL1_BASE, I2C_CTRL1_ACK_POS))

/* I2C Physical Interface Control */
/*   Configure the physical pads used by the I2C interface */
#define I2C_PHY_CTRL                    REG32_POINTER(I2C_PHY_CTRL_BASE)

/* I2C_PHY_CTRL bit positions (legacy definitions) */
#define I2C_PHY_CTRL_PULLUP_SELECT_POS  1
#define I2C_PHY_CTRL_FILTER_ENABLE_POS  0

/* I2C_PHY_CTRL bit-band aliases */
#define I2C_PHY_CTRL_PULLUP_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(I2C_PHY_CTRL_BASE, I2C_PHY_CTRL_PULLUP_SELECT_POS))
#define I2C_PHY_CTRL_FILTER_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(I2C_PHY_CTRL_BASE, I2C_PHY_CTRL_FILTER_ENABLE_POS))

/* I2C Interface Data */
/*   Single byte buffer for data to be transmitted over the I2C interface and
 *   data that has been received over the I2C interface */
#define I2C_DATA                        REG32_POINTER(I2C_DATA_BASE)

/* I2C Interface Data (Shadow) */
/*   Shadow of the single byte buffer for data to be transmitted over the I2C
 *   interface and data that has been received over the I2C interface */
#define I2C_DATA_S                      REG32_POINTER(I2C_DATA_S_BASE)

/* I2C Master Address and Start */
/*   Start a master I2C transaction with the provided address and read-write
 *   bit */
#define I2C_ADDR_START                  REG32_POINTER(I2C_ADDR_START_BASE)

/* I2C_ADDR_START bit positions (legacy definitions) */
#define I2C_ADDR_START_ADDRESS_POS      1
#define I2C_ADDR_START_ADDRESS_MASK     ((uint32_t)(0x7FU << I2C_ADDR_START_ADDRESS_POS))
#define I2C_ADDR_START_READ_WRITE_POS   0

/* I2C_ADDR_START bit-band aliases */
#define I2C_ADDR_START_READ_WRITE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(I2C_ADDR_START_BASE, I2C_ADDR_START_READ_WRITE_POS))

/* I2C Status */
/*   Indicate the status of the I2C interface */
#define I2C_STATUS                      READONLY_REG32_POINTER(I2C_STATUS_BASE)

/* I2C_STATUS bit positions (legacy definitions) */
#define I2C_STATUS_DMA_REQ_POS          11
#define I2C_STATUS_STOP_DETECT_POS      10
#define I2C_STATUS_DATA_EVENT_POS       9
#define I2C_STATUS_ERROR_POS            8
#define I2C_STATUS_BUS_ERROR_POS        7
#define I2C_STATUS_BUFFER_FULL_POS      6
#define I2C_STATUS_CLK_STRETCH_POS      5
#define I2C_STATUS_BUS_FREE_POS         4
#define I2C_STATUS_ADDR_DATA_POS        3
#define I2C_STATUS_READ_WRITE_POS       2
#define I2C_STATUS_GEN_CALL_POS         1
#define I2C_STATUS_ACK_STATUS_POS       0

/* I2C_STATUS bit-band aliases */
#define I2C_STATUS_DMA_REQ_BITBAND      READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_DMA_REQ_POS))
#define I2C_STATUS_STOP_DETECT_BITBAND  READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_STOP_DETECT_POS))
#define I2C_STATUS_DATA_EVENT_BITBAND   READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_DATA_EVENT_POS))
#define I2C_STATUS_ERROR_BITBAND        READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_ERROR_POS))
#define I2C_STATUS_BUS_ERROR_BITBAND    READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_BUS_ERROR_POS))
#define I2C_STATUS_BUFFER_FULL_BITBAND  READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_BUFFER_FULL_POS))
#define I2C_STATUS_CLK_STRETCH_BITBAND  READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_CLK_STRETCH_POS))
#define I2C_STATUS_BUS_FREE_BITBAND     READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_BUS_FREE_POS))
#define I2C_STATUS_ADDR_DATA_BITBAND    READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_ADDR_DATA_POS))
#define I2C_STATUS_READ_WRITE_BITBAND   READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_READ_WRITE_POS))
#define I2C_STATUS_GEN_CALL_BITBAND     READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_GEN_CALL_POS))
#define I2C_STATUS_ACK_STATUS_BITBAND   READONLY_REG32_POINTER(SYS_CALC_BITBAND(I2C_STATUS_BASE, I2C_STATUS_ACK_STATUS_POS))

/* ----------------------------------------------------------------------------
 * CRC Generator Control
 * ------------------------------------------------------------------------- */
/* The cyclic-redundancy check (CRC) generator provides a hardware
 * implementation of the CRC-CCITT 16-bit CRC calculation.
 *
 * This implementation features:
 *
 * - A CRC polynomial of 0x1021
 *
 * - An initial value of 0xFFFF
 *
 * - No requirement for XOR, bit-reversal or other finalization */

/* CRC Generator Control */
/*   CRC generator data input configuration */
#define CRC_CTRL                        REG32_POINTER(CRC_CTRL_BASE)

/* CRC_CTRL bit positions (legacy definitions) */
#define CRC_CTRL_BYTE_ORDER_POS         0

/* CRC_CTRL bit-band aliases */
#define CRC_CTRL_BYTE_ORDER_BITBAND     REG32_POINTER(SYS_CALC_BITBAND(CRC_CTRL_BASE, CRC_CTRL_BYTE_ORDER_POS))

/* CRC Generator Data */
/*   CRC generator data. Set to 0xFFFF to start the calculation and provides
 *   the current CRC. */
#define CRC_DATA                        REG32_POINTER(CRC_DATA_BASE)

/* CRC_DATA bit positions (legacy definitions) */
#define CRC_DATA_CURRENT_CRC_POS        0
#define CRC_DATA_CURRENT_CRC_MASK       ((uint32_t)(0xFFFFU << CRC_DATA_CURRENT_CRC_POS))

/* CRC Generator - Add 1 Bit */
/*   Add 1 bit to the CRC calculation */
#define CRC_ADD_1                       REG32_POINTER(CRC_ADD_1_BASE)

/* CRC_ADD_1 bit positions (legacy definitions) */
#define CRC_ADD_1_CRC_ADD_1_BIT_POS     0

/* CRC_ADD_1 bit-band aliases */
#define CRC_ADD_1_CRC_ADD_1_BIT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CRC_ADD_1_BASE, CRC_ADD_1_CRC_ADD_1_BIT_POS))

/* CRC Generator - Add 1 Byte */
/*   Add 1 byte (8 bits) to the CRC calculation */
#define CRC_ADD_8                       REG32_POINTER(CRC_ADD_8_BASE)

/* CRC Generator - Add 1 Half-word */
/*   Add 1 half-word (16 bits) to the CRC calculation */
#define CRC_ADD_16                      REG32_POINTER(CRC_ADD_16_BASE)

/* CRC Generator - Add 3 Bytes */
/*   Add 3 bytes (24 bits) to the CRC calculation */
#define CRC_ADD_24                      REG32_POINTER(CRC_ADD_24_BASE)

/* CRC_ADD_24 bit positions (legacy definitions) */
#define CRC_ADD_24_CRC_ADD_24_BITS_POS  0
#define CRC_ADD_24_CRC_ADD_24_BITS_MASK ((uint32_t)(0xFFFFFFU << CRC_ADD_24_CRC_ADD_24_BITS_POS))

/* CRC Generator - Add 1 Word */
/*   Add 1 word (32 bits) to the CRC calculation */
#define CRC_ADD_32                      REG32_POINTER(CRC_ADD_32_BASE)

/* ----------------------------------------------------------------------------
 * USB Interface
 * ------------------------------------------------------------------------- */
/* The universal serial bus (USB) interface implements the USB 1.1 standard in
 * full-speed mode.
 *
 * This register set provides top-level control of the logic behind the USB
 * interface. */

/* USB Control */
/*   Control the USB interface at a high-level. */
#define USB_CTRL                        REG32_POINTER(USB_CTRL_BASE)

/* USB_CTRL bit positions (legacy definitions) */
#define USB_CTRL_CONTROLLER_POS         4
#define USB_CTRL_PHY_STANDBY_POS        3
#define USB_CTRL_REMOTE_WAKEUP_POS      2
#define USB_CTRL_RESET_POS              1
#define USB_CTRL_ENABLE_POS             0

/* USB_CTRL bit-band aliases */
#define USB_CTRL_CONTROLLER_BITBAND     REG32_POINTER(SYS_CALC_BITBAND(USB_CTRL_BASE, USB_CTRL_CONTROLLER_POS))
#define USB_CTRL_PHY_STANDBY_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(USB_CTRL_BASE, USB_CTRL_PHY_STANDBY_POS))
#define USB_CTRL_REMOTE_WAKEUP_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(USB_CTRL_BASE, USB_CTRL_REMOTE_WAKEUP_POS))
#define USB_CTRL_RESET_BITBAND          REG32_POINTER(SYS_CALC_BITBAND(USB_CTRL_BASE, USB_CTRL_RESET_POS))
#define USB_CTRL_ENABLE_BITBAND         REG32_POINTER(SYS_CALC_BITBAND(USB_CTRL_BASE, USB_CTRL_ENABLE_POS))

/* ----------------------------------------------------------------------------
 * Timer 0 to 3 Configuration and Control
 * ------------------------------------------------------------------------- */
/* Four general purpose timers are provided. These timers support generating
 * interrupts at a configurable interval, for both a limited amount of time or
 * continuously */

/* Timer 0 Control and Configuration */
/*   Configure general purpose timer 0 */
#define TIMER0_CTRL                     REG32_POINTER(TIMER0_CTRL_BASE)

/* TIMER0_CTRL bit positions (legacy definitions) */
#define TIMER0_CTRL_MULTI_COUNT_POS     16
#define TIMER0_CTRL_MULTI_COUNT_MASK    ((uint32_t)(0x7U << TIMER0_CTRL_MULTI_COUNT_POS))
#define TIMER0_CTRL_MODE_POS            15
#define TIMER0_CTRL_PRESCALE_POS        12
#define TIMER0_CTRL_PRESCALE_MASK       ((uint32_t)(0x7U << TIMER0_CTRL_PRESCALE_POS))
#define TIMER0_CTRL_TIMEOUT_VALUE_POS   0
#define TIMER0_CTRL_TIMEOUT_VALUE_MASK  ((uint32_t)(0xFFFU << TIMER0_CTRL_TIMEOUT_VALUE_POS))

/* TIMER0_CTRL bit-band aliases */
#define TIMER0_CTRL_MODE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(TIMER0_CTRL_BASE, TIMER0_CTRL_MODE_POS))

/* Timer 1 Control and Configuration */
/*   Configure general purpose timer 1 */
#define TIMER1_CTRL                     REG32_POINTER(TIMER1_CTRL_BASE)

/* TIMER1_CTRL bit positions (legacy definitions) */
#define TIMER1_CTRL_MULTI_COUNT_POS     16
#define TIMER1_CTRL_MULTI_COUNT_MASK    ((uint32_t)(0x7U << TIMER1_CTRL_MULTI_COUNT_POS))
#define TIMER1_CTRL_MODE_POS            15
#define TIMER1_CTRL_PRESCALE_POS        12
#define TIMER1_CTRL_PRESCALE_MASK       ((uint32_t)(0x7U << TIMER1_CTRL_PRESCALE_POS))
#define TIMER1_CTRL_TIMEOUT_VALUE_POS   0
#define TIMER1_CTRL_TIMEOUT_VALUE_MASK  ((uint32_t)(0xFFFU << TIMER1_CTRL_TIMEOUT_VALUE_POS))

/* TIMER1_CTRL bit-band aliases */
#define TIMER1_CTRL_MODE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(TIMER1_CTRL_BASE, TIMER1_CTRL_MODE_POS))

/* Timer 2 Control and Configuration */
/*   Configure general purpose timer 2 */
#define TIMER2_CTRL                     REG32_POINTER(TIMER2_CTRL_BASE)

/* TIMER2_CTRL bit positions (legacy definitions) */
#define TIMER2_CTRL_MULTI_COUNT_POS     16
#define TIMER2_CTRL_MULTI_COUNT_MASK    ((uint32_t)(0x7U << TIMER2_CTRL_MULTI_COUNT_POS))
#define TIMER2_CTRL_MODE_POS            15
#define TIMER2_CTRL_PRESCALE_POS        12
#define TIMER2_CTRL_PRESCALE_MASK       ((uint32_t)(0x7U << TIMER2_CTRL_PRESCALE_POS))
#define TIMER2_CTRL_TIMEOUT_VALUE_POS   0
#define TIMER2_CTRL_TIMEOUT_VALUE_MASK  ((uint32_t)(0xFFFU << TIMER2_CTRL_TIMEOUT_VALUE_POS))

/* TIMER2_CTRL bit-band aliases */
#define TIMER2_CTRL_MODE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(TIMER2_CTRL_BASE, TIMER2_CTRL_MODE_POS))

/* Timer 3 Control and Configuration */
/*   Configure general purpose timer 3 */
#define TIMER3_CTRL                     REG32_POINTER(TIMER3_CTRL_BASE)

/* TIMER3_CTRL bit positions (legacy definitions) */
#define TIMER3_CTRL_MULTI_COUNT_POS     16
#define TIMER3_CTRL_MULTI_COUNT_MASK    ((uint32_t)(0x7U << TIMER3_CTRL_MULTI_COUNT_POS))
#define TIMER3_CTRL_MODE_POS            15
#define TIMER3_CTRL_PRESCALE_POS        12
#define TIMER3_CTRL_PRESCALE_MASK       ((uint32_t)(0x7U << TIMER3_CTRL_PRESCALE_POS))
#define TIMER3_CTRL_TIMEOUT_VALUE_POS   0
#define TIMER3_CTRL_TIMEOUT_VALUE_MASK  ((uint32_t)(0xFFFU << TIMER3_CTRL_TIMEOUT_VALUE_POS))

/* TIMER3_CTRL bit-band aliases */
#define TIMER3_CTRL_MODE_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(TIMER3_CTRL_BASE, TIMER3_CTRL_MODE_POS))

/* Timer Control Status for Timer 0 and Timer 1 */
/*   Control and indicate the status of the general purpose timers */
#define TIMER_CTRL_STATUS               REG32_POINTER(TIMER_CTRL_STATUS_BASE)

/* TIMER_CTRL_STATUS bit positions (legacy definitions) */
#define TIMER_CTRL_STATUS_TIMER3_STATUS_POS 3
#define TIMER_CTRL_STATUS_TIMER2_STATUS_POS 2
#define TIMER_CTRL_STATUS_TIMER1_STATUS_POS 1
#define TIMER_CTRL_STATUS_TIMER0_STATUS_POS 0

/* TIMER_CTRL_STATUS bit-band aliases */
#define TIMER_CTRL_STATUS_TIMER3_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(TIMER_CTRL_STATUS_BASE, TIMER_CTRL_STATUS_TIMER3_STATUS_POS))
#define TIMER_CTRL_STATUS_TIMER2_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(TIMER_CTRL_STATUS_BASE, TIMER_CTRL_STATUS_TIMER2_STATUS_POS))
#define TIMER_CTRL_STATUS_TIMER1_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(TIMER_CTRL_STATUS_BASE, TIMER_CTRL_STATUS_TIMER1_STATUS_POS))
#define TIMER_CTRL_STATUS_TIMER0_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(TIMER_CTRL_STATUS_BASE, TIMER_CTRL_STATUS_TIMER0_STATUS_POS))

/* ----------------------------------------------------------------------------
 * Watchdog Timer Configuration and Control
 * ------------------------------------------------------------------------- */
/* The watchdog timer is an independent timer that monitors whether a device
 * continues to operate.
 *
 * This timer must be refreshed periodically in order to avoid a system
 * reset. */

/* Watchdog Refresh Control */
/*   Used to restart the Watchdog timer to prevent a watchdog reset. */
#define WATCHDOG_REFRESH_CTRL           REG32_POINTER(WATCHDOG_REFRESH_CTRL_BASE)

/* WATCHDOG_REFRESH_CTRL bit positions (legacy definitions) */
#define WATCHDOG_REFRESH_CTRL_REFRESH_POS 0

/* WATCHDOG_REFRESH_CTRL bit-band aliases */
#define WATCHDOG_REFRESH_CTRL_REFRESH_BITBAND REG32_POINTER(SYS_CALC_BITBAND(WATCHDOG_REFRESH_CTRL_BASE, WATCHDOG_REFRESH_CTRL_REFRESH_POS))

/* Watchdog Timer Control */
/*   Configure the watchdog timer timeout period */
#define WATCHDOG_CTRL                   REG32_POINTER(WATCHDOG_CTRL_BASE)

/* WATCHDOG_CTRL bit positions (legacy definitions) */
#define WATCHDOG_CTRL_TIMEOUT_POS       0
#define WATCHDOG_CTRL_TIMEOUT_MASK      ((uint32_t)(0xFU << WATCHDOG_CTRL_TIMEOUT_POS))

/* ----------------------------------------------------------------------------
 * Clock Configuration
 * ------------------------------------------------------------------------- */
/* The clock configuration module configures the clock distribution tree and
 * supports prescaling of the distributed clocks. */

/* Clock control register 0 */
/*   Setup the RTC clock divisor and root clock source. */
#define CLK_CTRL0                       REG32_POINTER(CLK_CTRL0_BASE)

/* CLK_CTRL0 bit positions (legacy definitions) */
#define CLK_CTRL0_RTC_CLK_DIV_POS       2
#define CLK_CTRL0_RTC_CLK_DIV_MASK      ((uint32_t)(0x3U << CLK_CTRL0_RTC_CLK_DIV_POS))
#define CLK_CTRL0_RCLK_SELECT_POS       0
#define CLK_CTRL0_RCLK_SELECT_MASK      ((uint32_t)(0x3U << CLK_CTRL0_RCLK_SELECT_POS))

/* Clock control register 1 */
/*   Configure clock divisors for the SYSTICK timer, general purpose timers and
 *   watchdog timer */
#define CLK_CTRL1                       REG32_POINTER(CLK_CTRL1_BASE)

/* CLK_CTRL1 bit positions (legacy definitions) */
#define CLK_CTRL1_SYSTICK_DIV_POS       24
#define CLK_CTRL1_SYSTICK_DIV_MASK      ((uint32_t)(0xFFU << CLK_CTRL1_SYSTICK_DIV_POS))
#define CLK_CTRL1_TIMER13_DIV_POS       16
#define CLK_CTRL1_TIMER13_DIV_MASK      ((uint32_t)(0x1FU << CLK_CTRL1_TIMER13_DIV_POS))
#define CLK_CTRL1_TIMER02_DIV_POS       8
#define CLK_CTRL1_TIMER02_DIV_MASK      ((uint32_t)(0x1FU << CLK_CTRL1_TIMER02_DIV_POS))
#define CLK_CTRL1_WATCHDOG_CLK_DIV_POS  0
#define CLK_CTRL1_WATCHDOG_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL1_WATCHDOG_CLK_DIV_POS))

/* CLK_CTRL1 subregister pointers */
#define CLK_CTRL_SYSTICK_DIV_BYTE       REG8_POINTER(CLK_CTRL1_BASE + 3)
#define CLK_CTRL_TIMER13_DIV_BYTE       REG8_POINTER(CLK_CTRL1_BASE + 2)
#define CLK_CTRL_TIMER02_DIV_BYTE       REG8_POINTER(CLK_CTRL1_BASE + 1)
#define CLK_CTRL_WATCHDOG_CLK_DIV_BYTE  REG8_POINTER(CLK_CTRL1_BASE + 0)

/* CLK_CTRL1 subregister bit positions */
#define CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_POS 0
#define CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_POS))
#define CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_POS 0
#define CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_POS))
#define CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_POS 0
#define CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_POS))

/* Clock control register 2 */
/*   Configure clock divisors for the UART interfaces */
#define CLK_CTRL2                       REG32_POINTER(CLK_CTRL2_BASE)

/* CLK_CTRL2 bit positions (legacy definitions) */
#define CLK_CTRL2_UART1_CLK_DIV_POS     8
#define CLK_CTRL2_UART1_CLK_DIV_MASK    ((uint32_t)(0x1FU << CLK_CTRL2_UART1_CLK_DIV_POS))
#define CLK_CTRL2_UART0_CLK_DIV_POS     0
#define CLK_CTRL2_UART0_CLK_DIV_MASK    ((uint32_t)(0x1FU << CLK_CTRL2_UART0_CLK_DIV_POS))

/* CLK_CTRL2 subregister pointers */
#define CLK_CTRL_UART1_CLK_DIV_BYTE     REG8_POINTER(CLK_CTRL2_BASE + 1)
#define CLK_CTRL_UART0_CLK_DIV_BYTE     REG8_POINTER(CLK_CTRL2_BASE + 0)

/* CLK_CTRL2 subregister bit positions */
#define CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_POS 0
#define CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_POS))
#define CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_POS 0
#define CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_POS))

/* Clock control register 3 */
/*   Configure clock divisors for the MCLK, I2C, PCM and GPIO interfaces */
#define CLK_CTRL3                       REG32_POINTER(CLK_CTRL3_BASE)

/* CLK_CTRL3 bit positions (legacy definitions) */
#define CLK_CTRL3_MCLK_CLK_ENABLE_POS   29
#define CLK_CTRL3_MCLK_CLK_DIV_POS      24
#define CLK_CTRL3_MCLK_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL3_MCLK_CLK_DIV_POS))
#define CLK_CTRL3_I2C_CLK_ENABLE_POS    21
#define CLK_CTRL3_PCM_CLK_DIV_POS       8
#define CLK_CTRL3_PCM_CLK_DIV_MASK      ((uint32_t)(0x1FU << CLK_CTRL3_PCM_CLK_DIV_POS))
#define CLK_CTRL3_GPIO_CLK_DIV_POS      0
#define CLK_CTRL3_GPIO_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL3_GPIO_CLK_DIV_POS))

/* CLK_CTRL3 bit-band aliases */
#define CLK_CTRL3_MCLK_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL3_BASE, CLK_CTRL3_MCLK_CLK_ENABLE_POS))
#define CLK_CTRL3_I2C_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL3_BASE, CLK_CTRL3_I2C_CLK_ENABLE_POS))

/* CLK_CTRL3 subregister pointers */
#define CLK_CTRL_MCLK_DIV_BYTE          REG8_POINTER(CLK_CTRL3_BASE + 3)
#define CLK_CTRL_I2C_CLK_BYTE           REG8_POINTER(CLK_CTRL3_BASE + 2)
#define CLK_CTRL_PCM_CLK_DIV_BYTE       REG8_POINTER(CLK_CTRL3_BASE + 1)
#define CLK_CTRL_GPIO_CLK_DIV_BYTE      REG8_POINTER(CLK_CTRL3_BASE + 0)

/* CLK_CTRL3 subregister bit positions */
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_ENABLE_POS 5
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_POS 0
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_POS))
#define CLK_CTRL_I2C_CLK_BYTE_I2C_CLK_ENABLE_POS 5
#define CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_POS 0
#define CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_POS))
#define CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_POS 0
#define CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_POS))

/* Clock control register 4 */
/*   Configure clock divisors for the ARM Cortex-M3 core, DMA and external
 *   clock output */
#define CLK_CTRL4                       REG32_POINTER(CLK_CTRL4_BASE)

/* CLK_CTRL4 bit positions (legacy definitions) */
#define CLK_CTRL4_CM3_CLK_DIV_POS       24
#define CLK_CTRL4_CM3_CLK_DIV_MASK      ((uint32_t)(0x1FU << CLK_CTRL4_CM3_CLK_DIV_POS))
#define CLK_CTRL4_EXT_CLK_DIV2_POS      6
#define CLK_CTRL4_EXT_CLK_ENABLE_POS    5
#define CLK_CTRL4_EXT_CLK_DIV_POS       0
#define CLK_CTRL4_EXT_CLK_DIV_MASK      ((uint32_t)(0x1FU << CLK_CTRL4_EXT_CLK_DIV_POS))

/* CLK_CTRL4 bit-band aliases */
#define CLK_CTRL4_EXT_CLK_DIV2_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL4_BASE, CLK_CTRL4_EXT_CLK_DIV2_POS))
#define CLK_CTRL4_EXT_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL4_BASE, CLK_CTRL4_EXT_CLK_ENABLE_POS))

/* CLK_CTRL4 subregister pointers */
#define CLK_CTRL_CM3_CLK_DIV_BYTE       REG8_POINTER(CLK_CTRL4_BASE + 3)
#define CLK_CTRL_EXT_CLK_BYTE           REG8_POINTER(CLK_CTRL4_BASE + 0)

/* CLK_CTRL4 subregister bit positions */
#define CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_POS 0
#define CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_POS))
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV2_POS 6
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_ENABLE_POS 5
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_POS 0
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_POS))

/* Clock control register 5 */
/*   Configure clock divisors and output enables for the user output clocks */
#define CLK_CTRL5                       REG32_POINTER(CLK_CTRL5_BASE)

/* CLK_CTRL5 bit positions (legacy definitions) */
#define CLK_CTRL5_LCD_CLK_ENABLE_POS    31
#define CLK_CTRL5_LCD_CLK_DIV_POS       24
#define CLK_CTRL5_LCD_CLK_DIV_MASK      ((uint32_t)(0x7FU << CLK_CTRL5_LCD_CLK_DIV_POS))
#define CLK_CTRL5_USR_CLK2_DIV2_POS     22
#define CLK_CTRL5_USR_CLK2_ENABLE_POS   21
#define CLK_CTRL5_USR_CLK2_DIV_POS      16
#define CLK_CTRL5_USR_CLK2_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK2_DIV_POS))
#define CLK_CTRL5_USR_CLK1_DIV2_POS     14
#define CLK_CTRL5_USR_CLK1_ENABLE_POS   13
#define CLK_CTRL5_USR_CLK1_DIV_POS      8
#define CLK_CTRL5_USR_CLK1_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK1_DIV_POS))
#define CLK_CTRL5_USR_CLK0_DIV2_POS     6
#define CLK_CTRL5_USR_CLK0_ENABLE_POS   5
#define CLK_CTRL5_USR_CLK0_DIV_POS      0
#define CLK_CTRL5_USR_CLK0_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK0_DIV_POS))

/* CLK_CTRL5 bit-band aliases */
#define CLK_CTRL5_LCD_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_LCD_CLK_ENABLE_POS))
#define CLK_CTRL5_USR_CLK2_DIV2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK2_DIV2_POS))
#define CLK_CTRL5_USR_CLK2_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK2_ENABLE_POS))
#define CLK_CTRL5_USR_CLK1_DIV2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK1_DIV2_POS))
#define CLK_CTRL5_USR_CLK1_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK1_ENABLE_POS))
#define CLK_CTRL5_USR_CLK0_DIV2_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK0_DIV2_POS))
#define CLK_CTRL5_USR_CLK0_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL5_BASE, CLK_CTRL5_USR_CLK0_ENABLE_POS))

/* CLK_CTRL5 subregister pointers */
#define CLK_CTRL_LCD_CLK_BYTE           REG8_POINTER(CLK_CTRL5_BASE + 3)
#define CLK_CTRL_USR_CLK2_BYTE          REG8_POINTER(CLK_CTRL5_BASE + 2)
#define CLK_CTRL_USR_CLK1_BYTE          REG8_POINTER(CLK_CTRL5_BASE + 1)
#define CLK_CTRL_USR_CLK0_BYTE          REG8_POINTER(CLK_CTRL5_BASE + 0)

/* CLK_CTRL5 subregister bit positions */
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_ENABLE_POS 7
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_POS 0
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_MASK ((uint32_t)(0x7FU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_POS))
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_POS 6
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_POS 5
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_POS 0
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_POS))
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_POS 6
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_POS 5
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_POS 0
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_POS))
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_POS 6
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_POS 5
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_POS 0
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_POS))

/* Clock control register 6 */
/*   Configure a clock divisor for the pulse-width modulators */
#define CLK_CTRL6                       REG32_POINTER(CLK_CTRL6_BASE)

/* CLK_CTRL6 bit positions (legacy definitions) */
#define CLK_CTRL6_PWM3_CLK_ENABLE_POS   29
#define CLK_CTRL6_PWM3_CLK_DIV_POS      24
#define CLK_CTRL6_PWM3_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL6_PWM3_CLK_DIV_POS))
#define CLK_CTRL6_PWM2_CLK_ENABLE_POS   21
#define CLK_CTRL6_PWM2_CLK_DIV_POS      16
#define CLK_CTRL6_PWM2_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL6_PWM2_CLK_DIV_POS))
#define CLK_CTRL6_PWM1_CLK_ENABLE_POS   13
#define CLK_CTRL6_PWM1_CLK_DIV_POS      8
#define CLK_CTRL6_PWM1_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL6_PWM1_CLK_DIV_POS))
#define CLK_CTRL6_PWM0_CLK_ENABLE_POS   5
#define CLK_CTRL6_PWM0_CLK_DIV_POS      0
#define CLK_CTRL6_PWM0_CLK_DIV_MASK     ((uint32_t)(0x1FU << CLK_CTRL6_PWM0_CLK_DIV_POS))

/* CLK_CTRL6 bit-band aliases */
#define CLK_CTRL6_PWM3_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL6_BASE, CLK_CTRL6_PWM3_CLK_ENABLE_POS))
#define CLK_CTRL6_PWM2_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL6_BASE, CLK_CTRL6_PWM2_CLK_ENABLE_POS))
#define CLK_CTRL6_PWM1_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL6_BASE, CLK_CTRL6_PWM1_CLK_ENABLE_POS))
#define CLK_CTRL6_PWM0_CLK_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_CTRL6_BASE, CLK_CTRL6_PWM0_CLK_ENABLE_POS))

/* CLK_CTRL6 subregister pointers */
#define CLK_CTRL_PWM3_CLK_BYTE          REG8_POINTER(CLK_CTRL6_BASE + 3)
#define CLK_CTRL_PWM2_CLK_BYTE          REG8_POINTER(CLK_CTRL6_BASE + 2)
#define CLK_CTRL_PWM1_CLK_BYTE          REG8_POINTER(CLK_CTRL6_BASE + 1)
#define CLK_CTRL_PWM0_CLK_BYTE          REG8_POINTER(CLK_CTRL6_BASE + 0)

/* CLK_CTRL6 subregister bit positions */
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_ENABLE_POS 5
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_POS 0
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_POS))
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_ENABLE_POS 5
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_POS 0
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_POS))
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_ENABLE_POS 5
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_POS 0
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_POS))
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_ENABLE_POS 5
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_POS 0
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_POS))

/* Clock control register 7 */
/*   Configure a clock divisor used for the charge pump */
#define CLK_CTRL7                       REG32_POINTER(CLK_CTRL7_BASE)

/* CLK_CTRL7 bit positions (legacy definitions) */
#define CLK_CTRL7_CP_CLK_DIV_POS        0
#define CLK_CTRL7_CP_CLK_DIV_MASK       ((uint32_t)(0x1FU << CLK_CTRL7_CP_CLK_DIV_POS))

/* CLK_CTRL7 subregister pointers */
#define CLK_CTRL_CP_CLK_DIV_BYTE        REG8_POINTER(CLK_CTRL7_BASE + 0)

/* CLK_CTRL7 subregister bit positions */
#define CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_POS 0
#define CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_MASK ((uint32_t)(0x1FU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_POS))

/* External Clock Detection Control and Configuration */
/*   Configure the external clock detection circuitry */
#define CLK_DETECT_CTRL                 REG32_POINTER(CLK_DETECT_CTRL_BASE)

/* CLK_DETECT_CTRL bit positions (legacy definitions) */
#define CLK_DETECT_CTRL_CLK_FORCE_ENABLE_POS 1
#define CLK_DETECT_CTRL_CLK_DETECT_ENABLE_POS 0

/* CLK_DETECT_CTRL bit-band aliases */
#define CLK_DETECT_CTRL_CLK_FORCE_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_DETECT_CTRL_BASE, CLK_DETECT_CTRL_CLK_FORCE_ENABLE_POS))
#define CLK_DETECT_CTRL_CLK_DETECT_ENABLE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_DETECT_CTRL_BASE, CLK_DETECT_CTRL_CLK_DETECT_ENABLE_POS))

/* External Clock Detection Status */
/*   Status from external clock detection circuitry */
#define CLK_DETECT_STATUS               REG32_POINTER(CLK_DETECT_STATUS_BASE)

/* CLK_DETECT_STATUS bit positions (legacy definitions) */
#define CLK_DETECT_STATUS_EXT_CLK_STATUS_POS 0

/* CLK_DETECT_STATUS bit-band aliases */
#define CLK_DETECT_STATUS_EXT_CLK_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(CLK_DETECT_STATUS_BASE, CLK_DETECT_STATUS_EXT_CLK_STATUS_POS))

/* ----------------------------------------------------------------------------
 * Flash Update Control
 * ------------------------------------------------------------------------- */
/* The flash memory can be erased or written using only the flash update logic.
 * Control of this logic is mapped into the peripheral memory space for ease of
 * use. */

/* Flash Error-Correction Coding Status */
/*   Provides the status result from recent flash ECC calculations. */
#define FLASH_ECC_STATUS                REG32_POINTER(FLASH_ECC_STATUS_BASE)

/* FLASH_ECC_STATUS bit positions (legacy definitions) */
#define FLASH_ECC_STATUS_ERROR_DETECT_POS 1
#define FLASH_ECC_STATUS_ERROR_CORRECT_POS 0

/* FLASH_ECC_STATUS bit-band aliases */
#define FLASH_ECC_STATUS_ERROR_DETECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(FLASH_ECC_STATUS_BASE, FLASH_ECC_STATUS_ERROR_DETECT_POS))
#define FLASH_ECC_STATUS_ERROR_CORRECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(FLASH_ECC_STATUS_BASE, FLASH_ECC_STATUS_ERROR_CORRECT_POS))

/* Flash Write Lock */
/*   Lock word allowing a write to the flash memory is allowed */
#define FLASH_WRITE_LOCK                REG32_POINTER(FLASH_WRITE_LOCK_BASE)

/* FLASH_WRITE_LOCK bit positions (legacy definitions) */
#define FLASH_WRITE_LOCK_FLASH_WRITE_KEY_POS 0
#define FLASH_WRITE_LOCK_FLASH_WRITE_KEY_MASK ((uint32_t)(0xFFFFFFFFU << FLASH_WRITE_LOCK_FLASH_WRITE_KEY_POS))

/* ----------------------------------------------------------------------------
 * USB Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The universal serial bus (USB) interface implements the USB 1.1 standard in
 * full-speed mode.
 *
 * This interface provides connectivity between the system and an external PC
 * or other USB device. */

/* USB Bulk 4 and 5 OUT start address register */
#define USB_BULK_OUT45_ADDR             REG32_POINTER(USB_BULK_OUT45_ADDR_BASE)

/* USB_BULK_OUT45_ADDR bit positions (legacy definitions) */
#define USB_BULK_OUT45_ADDR_OUT5_ADDR_POS 8
#define USB_BULK_OUT45_ADDR_OUT5_ADDR_MASK ((uint32_t)(0xFFU << USB_BULK_OUT45_ADDR_OUT5_ADDR_POS))
#define USB_BULK_OUT45_ADDR_OUT4_ADDR_POS 0
#define USB_BULK_OUT45_ADDR_OUT4_ADDR_MASK ((uint32_t)(0xFFU << USB_BULK_OUT45_ADDR_OUT4_ADDR_POS))

/* USB_BULK_OUT45_ADDR subregister pointers */
#define USB_BULK_OUT5_ADDR_BYTE         REG8_POINTER(USB_BULK_OUT45_ADDR_BASE + 1)
#define USB_BULK_OUT4_ADDR_BYTE         REG8_POINTER(USB_BULK_OUT45_ADDR_BASE + 0)

/* USB Bulk 2 and 3 IN start address register */
#define USB_BULK_IN23_ADDR              REG32_POINTER(USB_BULK_IN23_ADDR_BASE)

/* USB_BULK_IN23_ADDR bit positions (legacy definitions) */
#define USB_BULK_IN23_ADDR_IN3_ADDR_POS 24
#define USB_BULK_IN23_ADDR_IN3_ADDR_MASK ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN3_ADDR_POS))
#define USB_BULK_IN23_ADDR_IN2_ADDR_POS 16
#define USB_BULK_IN23_ADDR_IN2_ADDR_MASK ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN2_ADDR_POS))
#define USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_POS 0
#define USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_MASK ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_POS))

/* USB_BULK_IN23_ADDR subregister pointers */
#define USB_BULK_IN3_ADDR_BYTE          REG8_POINTER(USB_BULK_IN23_ADDR_BASE + 3)
#define USB_BULK_IN2_ADDR_BYTE          REG8_POINTER(USB_BULK_IN23_ADDR_BASE + 2)
#define USB_BULK_OFFSET_ADDR_BYTE       REG8_POINTER(USB_BULK_IN23_ADDR_BASE + 0)

/* USB clock gate register */
#define USB_CLOCK_GATE                  REG32_POINTER(USB_CLOCK_GATE_BASE)

/* USB_CLOCK_GATE bit positions (legacy definitions) */
#define USB_CLOCK_GATE_CLOCK_GATE_POS   0
#define USB_CLOCK_GATE_CLOCK_GATE_MASK  ((uint32_t)(0xFFU << USB_CLOCK_GATE_CLOCK_GATE_POS))

/* USB_CLOCK_GATE subregister pointers */
#define USB_CLOCK_GATE_BYTE             REG8_POINTER(USB_CLOCK_GATE_BASE + 0)

/* USB Interrupt Status register */
/*   Interrupt status register. The USB_IVEC bits show the status of the
 *   different interrupt sources and is updated upon interrupt request
 *   generation. The USB_BULK_IN_IRQ bits are set by the CUSB to '1' when it
 *   transmits a bulk IN x data packet and receives an ACK from the host. The
 *   USB_BULK_OUT_IRQ bits are set by the CUSB to '1' when it receives an error
 *   free bulk OUT x data packet. */
#define USB_INT_STATUS                  REG32_POINTER(USB_INT_STATUS_BASE)

/* USB_INT_STATUS bit positions (legacy definitions) */
#define USB_INT_STATUS_RST_IRQ_POS      28
#define USB_INT_STATUS_SUS_IRQ_POS      27
#define USB_INT_STATUS_SETUPTKN_IRQ_POS 26
#define USB_INT_STATUS_SOF_IRQ_POS      25
#define USB_INT_STATUS_DAT_VALID_IRQ_POS 24
#define USB_INT_STATUS_BULK_OUT_5_IRQ_POS 21
#define USB_INT_STATUS_BULK_OUT_4_IRQ_POS 20
#define USB_INT_STATUS_BULK_OUT_0_IRQ_POS 16
#define USB_INT_STATUS_BULK_IN_3_IRQ_POS 11
#define USB_INT_STATUS_BULK_IN_2_IRQ_POS 10
#define USB_INT_STATUS_BULK_IN_0_IRQ_POS 8

/* USB_INT_STATUS bit-band aliases */
#define USB_INT_STATUS_RST_IRQ_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_RST_IRQ_POS))
#define USB_INT_STATUS_SUS_IRQ_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_SUS_IRQ_POS))
#define USB_INT_STATUS_SETUPTKN_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_SETUPTKN_IRQ_POS))
#define USB_INT_STATUS_SOF_IRQ_BITBAND  REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_SOF_IRQ_POS))
#define USB_INT_STATUS_DAT_VALID_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_DAT_VALID_IRQ_POS))
#define USB_INT_STATUS_BULK_OUT_5_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_OUT_5_IRQ_POS))
#define USB_INT_STATUS_BULK_OUT_4_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_OUT_4_IRQ_POS))
#define USB_INT_STATUS_BULK_OUT_0_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_OUT_0_IRQ_POS))
#define USB_INT_STATUS_BULK_IN_3_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_IN_3_IRQ_POS))
#define USB_INT_STATUS_BULK_IN_2_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_IN_2_IRQ_POS))
#define USB_INT_STATUS_BULK_IN_0_IRQ_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_STATUS_BASE, USB_INT_STATUS_BULK_IN_0_IRQ_POS))

/* USB_INT_STATUS subregister pointers */
#define USB_IRQ_BYTE                    REG8_POINTER(USB_INT_STATUS_BASE + 3)
#define USB_BULK_OUT_IRQ_BYTE           REG8_POINTER(USB_INT_STATUS_BASE + 2)
#define USB_BULK_IN_IRQ_BYTE            REG8_POINTER(USB_INT_STATUS_BASE + 1)

/* USB_INT_STATUS subregister bit positions */
#define USB_IRQ_BYTE_RST_IRQ_POS        4
#define USB_IRQ_BYTE_SUS_IRQ_POS        3
#define USB_IRQ_BYTE_SETUPTKN_IRQ_POS   2
#define USB_IRQ_BYTE_SOF_IRQ_POS        1
#define USB_IRQ_BYTE_DAT_VALID_IRQ_POS  0
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_5_IRQ_POS 5
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_4_IRQ_POS 4
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_0_IRQ_POS 0
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_3_IRQ_POS 3
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_2_IRQ_POS 2
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_0_IRQ_POS 0

/* USB Interrupt Control register */
/*   Setting the appropriate bit to '1' enables the appropriate interrupt */
#define USB_INT_CTRL                    REG32_POINTER(USB_INT_CTRL_BASE)

/* USB_INT_CTRL bit positions (legacy definitions) */
#define USB_INT_CTRL_BAV_POS            24
#define USB_INT_CTRL_RST_IEN_POS        20
#define USB_INT_CTRL_SUS_IEN_POS        19
#define USB_INT_CTRL_SETUPTKN_IEN_POS   18
#define USB_INT_CTRL_SOF_IEN_POS        17
#define USB_INT_CTRL_DAT_VALID_IEN_POS  16
#define USB_INT_CTRL_BULK_OUT_5_IEN_POS 13
#define USB_INT_CTRL_BULK_OUT_4_IEN_POS 12
#define USB_INT_CTRL_BULK_OUT_0_IEN_POS 8
#define USB_INT_CTRL_BULK_IN_3_IEN_POS  3
#define USB_INT_CTRL_BULK_IN_2_IEN_POS  2
#define USB_INT_CTRL_BULK_IN_0_IEN_POS  0

/* USB_INT_CTRL bit-band aliases */
#define USB_INT_CTRL_BAV_BITBAND        REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BAV_POS))
#define USB_INT_CTRL_RST_IEN_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_RST_IEN_POS))
#define USB_INT_CTRL_SUS_IEN_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_SUS_IEN_POS))
#define USB_INT_CTRL_SETUPTKN_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_SETUPTKN_IEN_POS))
#define USB_INT_CTRL_SOF_IEN_BITBAND    REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_SOF_IEN_POS))
#define USB_INT_CTRL_DAT_VALID_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_DAT_VALID_IEN_POS))
#define USB_INT_CTRL_BULK_OUT_5_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_OUT_5_IEN_POS))
#define USB_INT_CTRL_BULK_OUT_4_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_OUT_4_IEN_POS))
#define USB_INT_CTRL_BULK_OUT_0_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_OUT_0_IEN_POS))
#define USB_INT_CTRL_BULK_IN_3_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_IN_3_IEN_POS))
#define USB_INT_CTRL_BULK_IN_2_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_IN_2_IEN_POS))
#define USB_INT_CTRL_BULK_IN_0_IEN_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_INT_CTRL_BASE, USB_INT_CTRL_BULK_IN_0_IEN_POS))

/* USB_INT_CTRL subregister pointers */
#define USB_BAV_BYTE                    REG8_POINTER(USB_INT_CTRL_BASE + 3)
#define USB_IEN_BYTE                    REG8_POINTER(USB_INT_CTRL_BASE + 2)
#define USB_BULK_OUT_IEN_BYTE           REG8_POINTER(USB_INT_CTRL_BASE + 1)
#define USB_BULK_IN_IEN_BYTE            REG8_POINTER(USB_INT_CTRL_BASE + 0)

/* USB_INT_CTRL subregister bit positions */
#define USB_BAV_BYTE_BAV_POS            0
#define USB_IEN_BYTE_RST_IEN_POS        4
#define USB_IEN_BYTE_SUS_IEN_POS        3
#define USB_IEN_BYTE_SETUPTKN_IEN_POS   2
#define USB_IEN_BYTE_SOF_IEN_POS        1
#define USB_IEN_BYTE_DAT_VALID_IEN_POS  0
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_5_IEN_POS 5
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_4_IEN_POS 4
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_0_IEN_POS 0
#define USB_BULK_IN_IEN_BYTE_BULK_IN_3_IEN_POS 3
#define USB_BULK_IN_IEN_BYTE_BULK_IN_2_IEN_POS 2
#define USB_BULK_IN_IEN_BYTE_BULK_IN_0_IEN_POS 0

/* USB IN Endpoint 0 Control and Status register */
/*   Various control and status bit fields for the USB IN 0 endpoint */
#define USB_EP0_IN_CTRL                 REG32_POINTER(USB_EP0_IN_CTRL_BASE)

/* USB_EP0_IN_CTRL bit positions (legacy definitions) */
#define USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_POS 8
#define USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_POS))
#define USB_EP0_IN_CTRL_EP0_SETUP_BUF_STATUS_POS 5
#define USB_EP0_IN_CTRL_EP0_DATA_STALL_POS 4
#define USB_EP0_IN_CTRL_EP0_OUT_BUSY_POS 3
#define USB_EP0_IN_CTRL_EP0_IN_BUSY_POS 2
#define USB_EP0_IN_CTRL_EP0_HSNAK_POS   1
#define USB_EP0_IN_CTRL_EP0_CTRL_STALL_POS 0

/* USB_EP0_IN_CTRL bit-band aliases */
#define USB_EP0_IN_CTRL_EP0_SETUP_BUF_STATUS_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_SETUP_BUF_STATUS_POS))
#define USB_EP0_IN_CTRL_EP0_DATA_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_DATA_STALL_POS))
#define USB_EP0_IN_CTRL_EP0_OUT_BUSY_BITBAND READONLY_REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_OUT_BUSY_POS))
#define USB_EP0_IN_CTRL_EP0_IN_BUSY_BITBAND READONLY_REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_IN_BUSY_POS))
#define USB_EP0_IN_CTRL_EP0_HSNAK_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_HSNAK_POS))
#define USB_EP0_IN_CTRL_EP0_CTRL_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP0_IN_CTRL_BASE, USB_EP0_IN_CTRL_EP0_CTRL_STALL_POS))

/* USB_EP0_IN_CTRL subregister pointers */
#define USB_EP0_IN_BYTE_COUNT_BYTE      REG8_POINTER(USB_EP0_IN_CTRL_BASE + 1)
#define USB_EP0_CTRL_BYTE               REG8_POINTER(USB_EP0_IN_CTRL_BASE + 0)

/* USB_EP0_IN_CTRL subregister bit positions */
#define USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_POS 0
#define USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_POS))
#define USB_EP0_CTRL_BYTE_EP0_SETUP_BUF_STATUS_POS 5
#define USB_EP0_CTRL_BYTE_EP0_DATA_STALL_POS 4
#define USB_EP0_CTRL_BYTE_EP0_OUT_BUSY_POS 3
#define USB_EP0_CTRL_BYTE_EP0_IN_BUSY_POS 2
#define USB_EP0_CTRL_BYTE_EP0_HSNAK_POS 1
#define USB_EP0_CTRL_BYTE_EP0_CTRL_STALL_POS 0

/* USB IN Endpoint 2 and 3 Control and Status register */
/*   Various control and status bit fields for the USB IN 2 and 3 endpoints */
#define USB_EP23_IN_CTRL                REG32_POINTER(USB_EP23_IN_CTRL_BASE)

/* USB_EP23_IN_CTRL bit positions (legacy definitions) */
#define USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_POS 24
#define USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_POS))
#define USB_EP23_IN_CTRL_EP3_IN_BUSY_POS 17
#define USB_EP23_IN_CTRL_EP3_IN_STALL_POS 16
#define USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_POS 8
#define USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_POS))
#define USB_EP23_IN_CTRL_EP2_IN_BUSY_POS 1
#define USB_EP23_IN_CTRL_EP2_IN_STALL_POS 0

/* USB_EP23_IN_CTRL bit-band aliases */
#define USB_EP23_IN_CTRL_EP3_IN_BUSY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP23_IN_CTRL_BASE, USB_EP23_IN_CTRL_EP3_IN_BUSY_POS))
#define USB_EP23_IN_CTRL_EP3_IN_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP23_IN_CTRL_BASE, USB_EP23_IN_CTRL_EP3_IN_STALL_POS))
#define USB_EP23_IN_CTRL_EP2_IN_BUSY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP23_IN_CTRL_BASE, USB_EP23_IN_CTRL_EP2_IN_BUSY_POS))
#define USB_EP23_IN_CTRL_EP2_IN_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP23_IN_CTRL_BASE, USB_EP23_IN_CTRL_EP2_IN_STALL_POS))

/* USB_EP23_IN_CTRL subregister pointers */
#define USB_EP3_IN_BYTE_COUNT_BYTE      REG8_POINTER(USB_EP23_IN_CTRL_BASE + 3)
#define USB_EP3_IN_CTRL_BYTE            REG8_POINTER(USB_EP23_IN_CTRL_BASE + 2)
#define USB_EP2_IN_BYTE_COUNT_BYTE      REG8_POINTER(USB_EP23_IN_CTRL_BASE + 1)
#define USB_EP2_IN_CTRL_BYTE            REG8_POINTER(USB_EP23_IN_CTRL_BASE + 0)

/* USB_EP23_IN_CTRL subregister bit positions */
#define USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_POS 0
#define USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_POS))
#define USB_EP3_IN_CTRL_BYTE_EP3_IN_BUSY_POS 1
#define USB_EP3_IN_CTRL_BYTE_EP3_IN_STALL_POS 0
#define USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_POS 0
#define USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_POS))
#define USB_EP2_IN_CTRL_BYTE_EP2_IN_BUSY_POS 1
#define USB_EP2_IN_CTRL_BYTE_EP2_IN_STALL_POS 0

/* USB OUT Endpoint 0 Control and Status register */
/*   Various control and status bit fields for the USB OUT endpoints 0 */
#define USB_EP0_OUT_CTRL                REG32_POINTER(USB_EP0_OUT_CTRL_BASE)

/* USB_EP0_OUT_CTRL bit positions (legacy definitions) */
#define USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_POS 8
#define USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_POS))

/* USB_EP0_OUT_CTRL subregister pointers */
#define USB_EP0_OUT_BYTE_COUNT_BYTE     REG8_POINTER(USB_EP0_OUT_CTRL_BASE + 1)

/* USB_EP0_OUT_CTRL subregister bit positions */
#define USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_POS 0
#define USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_POS))

/* USB OUT Endpoint 4 and 5 Control and Status register */
/*   Various control and status bit fields for the USB OUT endpoints 4 and 5 */
#define USB_EP45_OUT_CTRL               REG32_POINTER(USB_EP45_OUT_CTRL_BASE)

/* USB_EP45_OUT_CTRL bit positions (legacy definitions) */
#define USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_POS 24
#define USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_POS))
#define USB_EP45_OUT_CTRL_EP5_OUT_BUSY_POS 17
#define USB_EP45_OUT_CTRL_EP5_OUT_STALL_POS 16
#define USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_POS 8
#define USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_POS))
#define USB_EP45_OUT_CTRL_EP4_OUT_BUSY_POS 1
#define USB_EP45_OUT_CTRL_EP4_OUT_STALL_POS 0

/* USB_EP45_OUT_CTRL bit-band aliases */
#define USB_EP45_OUT_CTRL_EP5_OUT_BUSY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP45_OUT_CTRL_BASE, USB_EP45_OUT_CTRL_EP5_OUT_BUSY_POS))
#define USB_EP45_OUT_CTRL_EP5_OUT_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP45_OUT_CTRL_BASE, USB_EP45_OUT_CTRL_EP5_OUT_STALL_POS))
#define USB_EP45_OUT_CTRL_EP4_OUT_BUSY_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP45_OUT_CTRL_BASE, USB_EP45_OUT_CTRL_EP4_OUT_BUSY_POS))
#define USB_EP45_OUT_CTRL_EP4_OUT_STALL_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_EP45_OUT_CTRL_BASE, USB_EP45_OUT_CTRL_EP4_OUT_STALL_POS))

/* USB_EP45_OUT_CTRL subregister pointers */
#define USB_EP5_OUT_BYTE_COUNT_BYTE     REG8_POINTER(USB_EP45_OUT_CTRL_BASE + 3)
#define USB_EP5_OUT_CTRL_BYTE           REG8_POINTER(USB_EP45_OUT_CTRL_BASE + 2)
#define USB_EP4_OUT_BYTE_COUNT_BYTE     REG8_POINTER(USB_EP45_OUT_CTRL_BASE + 1)
#define USB_EP4_OUT_CTRL_BYTE           REG8_POINTER(USB_EP45_OUT_CTRL_BASE + 0)

/* USB_EP45_OUT_CTRL subregister bit positions */
#define USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_POS 0
#define USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_POS))
#define USB_EP5_OUT_CTRL_BYTE_EP5_OUT_BUSY_POS 1
#define USB_EP5_OUT_CTRL_BYTE_EP5_OUT_STALL_POS 0
#define USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_POS 0
#define USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_MASK ((uint32_t)(0x7FU << USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_POS))
#define USB_EP4_OUT_CTRL_BYTE_EP4_OUT_BUSY_POS 1
#define USB_EP4_OUT_CTRL_BYTE_EP4_OUT_STALL_POS 0

/* USB Control and Status register 1 */
/*   Various control and status bit fields for USB */
#define USB_SYS_CTRL1                   REG32_POINTER(USB_SYS_CTRL1_BASE)

/* USB_SYS_CTRL1 bit positions (legacy definitions) */
#define USB_SYS_CTRL1_DATA_TOGGLE_STATUS_POS 31
#define USB_SYS_CTRL1_DATA_TOGGLE_DATA1_SET_POS 30
#define USB_SYS_CTRL1_DATA_TOGGLE_RESET_POS 29
#define USB_SYS_CTRL1_DATA_TOGGLE_INOUT_SELECT_POS 28
#define USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_POS 24
#define USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_MASK ((uint32_t)(0x7U << USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_POS))
#define USB_SYS_CTRL1_WAKEUP_SOURCE_POS 23
#define USB_SYS_CTRL1_SOF_GEN_POS       21
#define USB_SYS_CTRL1_DISCONNECT_POS    19
#define USB_SYS_CTRL1_FORCE_J_POS       17
#define USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_POS 16

/* USB_SYS_CTRL1 bit-band aliases */
#define USB_SYS_CTRL1_DATA_TOGGLE_STATUS_BITBAND READONLY_REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_DATA_TOGGLE_STATUS_POS))
#define USB_SYS_CTRL1_DATA_TOGGLE_DATA1_SET_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_DATA_TOGGLE_DATA1_SET_POS))
#define USB_SYS_CTRL1_DATA_TOGGLE_RESET_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_DATA_TOGGLE_RESET_POS))
#define USB_SYS_CTRL1_DATA_TOGGLE_INOUT_SELECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_DATA_TOGGLE_INOUT_SELECT_POS))
#define USB_SYS_CTRL1_WAKEUP_SOURCE_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_WAKEUP_SOURCE_POS))
#define USB_SYS_CTRL1_SOF_GEN_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_SOF_GEN_POS))
#define USB_SYS_CTRL1_DISCONNECT_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_DISCONNECT_POS))
#define USB_SYS_CTRL1_FORCE_J_BITBAND   REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_FORCE_J_POS))
#define USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL1_BASE, USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_POS))

/* USB_SYS_CTRL1 subregister pointers */
#define USB_DATA_TOGGLE_CTRL_BYTE       REG8_POINTER(USB_SYS_CTRL1_BASE + 3)
#define USB_CTRL_STATUS_BYTE            REG8_POINTER(USB_SYS_CTRL1_BASE + 2)

/* USB_SYS_CTRL1 subregister bit positions */
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_STATUS_POS 7
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_DATA1_SET_POS 6
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_RESET_POS 5
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_INOUT_SELECT_POS 4
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_POS 0
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_MASK ((uint32_t)(0x7U << USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_POS))
#define USB_CTRL_STATUS_BYTE_WAKEUP_SOURCE_POS 7
#define USB_CTRL_STATUS_BYTE_SOF_GEN_POS 5
#define USB_CTRL_STATUS_BYTE_DISCONNECT_POS 3
#define USB_CTRL_STATUS_BYTE_FORCE_J_POS 1
#define USB_CTRL_STATUS_BYTE_SIGNAL_REMOTE_RESUME_POS 0

/* USB Control and Status register 2 */
/*   Current USB frame count */
#define USB_SYS_CTRL2                   READONLY_REG32_POINTER(USB_SYS_CTRL2_BASE)

/* USB_SYS_CTRL2 bit positions (legacy definitions) */
#define USB_SYS_CTRL2_FUNCTION_ADDR_POS 24
#define USB_SYS_CTRL2_FUNCTION_ADDR_MASK ((uint32_t)(0x7FU << USB_SYS_CTRL2_FUNCTION_ADDR_POS))
#define USB_SYS_CTRL2_FRAME_COUNT_POS   0
#define USB_SYS_CTRL2_FRAME_COUNT_MASK  ((uint32_t)(0x7FFU << USB_SYS_CTRL2_FRAME_COUNT_POS))

/* USB_SYS_CTRL2 subregister pointers */
#define USB_FUNCTION_ADDR_BYTE          READONLY_REG8_POINTER(USB_SYS_CTRL2_BASE + 3)
#define USB_FRAME_COUNT_SHORT           READONLY_REG16_POINTER(USB_SYS_CTRL2_BASE + 0)

/* USB_SYS_CTRL2 subregister bit positions */
#define USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_POS 0
#define USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_MASK ((uint32_t)(0x7FU << USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_POS))
#define USB_FRAME_COUNT_SHORT_FRAME_COUNT_POS 0
#define USB_FRAME_COUNT_SHORT_FRAME_COUNT_MASK ((uint32_t)(0x7FFU << USB_FRAME_COUNT_SHORT_FRAME_COUNT_POS))

/* USB Control and Status register 3 */
/*   Various control and status bit fields for USB */
#define USB_SYS_CTRL3                   REG32_POINTER(USB_SYS_CTRL3_BASE)

/* USB_SYS_CTRL3 bit positions (legacy definitions) */
#define USB_SYS_CTRL3_OUT5_VALID_POS    29
#define USB_SYS_CTRL3_OUT4_VALID_POS    28
#define USB_SYS_CTRL3_OUT0_VALID_POS    24
#define USB_SYS_CTRL3_IN3_VALID_POS     19
#define USB_SYS_CTRL3_IN2_VALID_POS     18
#define USB_SYS_CTRL3_IN0_VALID_POS     16
#define USB_SYS_CTRL3_USB_PAIR_OUT_EP45_POS 12
#define USB_SYS_CTRL3_USB_PAIR_IN_EP23_POS 8

/* USB_SYS_CTRL3 bit-band aliases */
#define USB_SYS_CTRL3_OUT5_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_OUT5_VALID_POS))
#define USB_SYS_CTRL3_OUT4_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_OUT4_VALID_POS))
#define USB_SYS_CTRL3_OUT0_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_OUT0_VALID_POS))
#define USB_SYS_CTRL3_IN3_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_IN3_VALID_POS))
#define USB_SYS_CTRL3_IN2_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_IN2_VALID_POS))
#define USB_SYS_CTRL3_IN0_VALID_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_IN0_VALID_POS))
#define USB_SYS_CTRL3_USB_PAIR_OUT_EP45_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_USB_PAIR_OUT_EP45_POS))
#define USB_SYS_CTRL3_USB_PAIR_IN_EP23_BITBAND REG32_POINTER(SYS_CALC_BITBAND(USB_SYS_CTRL3_BASE, USB_SYS_CTRL3_USB_PAIR_IN_EP23_POS))

/* USB_SYS_CTRL3 subregister pointers */
#define USB_EP045_OUT_VALID_BYTE        REG8_POINTER(USB_SYS_CTRL3_BASE + 3)
#define USB_EP023_IN_VALID_BYTE         REG8_POINTER(USB_SYS_CTRL3_BASE + 2)
#define USB_EP_PAIRING_BYTE             REG8_POINTER(USB_SYS_CTRL3_BASE + 1)

/* USB_SYS_CTRL3 subregister bit positions */
#define USB_EP045_OUT_VALID_BYTE_OUT5_VALID_POS 5
#define USB_EP045_OUT_VALID_BYTE_OUT4_VALID_POS 4
#define USB_EP045_OUT_VALID_BYTE_OUT0_VALID_POS 0
#define USB_EP023_IN_VALID_BYTE_IN3_VALID_POS 3
#define USB_EP023_IN_VALID_BYTE_IN2_VALID_POS 2
#define USB_EP023_IN_VALID_BYTE_IN0_VALID_POS 0
#define USB_EP_PAIRING_BYTE_USB_PAIR_OUT_EP45_POS 4
#define USB_EP_PAIRING_BYTE_USB_PAIR_IN_EP23_POS 0

/* USB Setup Data Buffer base Lower */
/*   Contains the lower 4 bytes of the SETUP data packet from the latest
 *   CONTROL transfer. */
#define USB_SETUP_DATA_BUF_BASE_0       READONLY_REG32_POINTER(USB_SETUP_DATA_BUF_BASE_0_BASE)

/* USB_SETUP_DATA_BUF_BASE_0 bit positions (legacy definitions) */
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_POS 24
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_POS))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_POS 16
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_POS))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_POS 8
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_POS))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_POS 0
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_POS))

/* USB_SETUP_DATA_BUF_BASE_0 subregister pointers */
#define SETUP_DATA_BUF_BASE1_SHORT      READONLY_REG16_POINTER(USB_SETUP_DATA_BUF_BASE_0_BASE + 2)
#define SETUP_DATA_BUF_BASE0_SHORT      READONLY_REG16_POINTER(USB_SETUP_DATA_BUF_BASE_0_BASE + 0)

/* USB_SETUP_DATA_BUF_BASE_0 subregister bit positions */
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_POS 8
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_POS))
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_POS 0
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_POS))
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_POS 8
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_POS))
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_POS 0
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_POS))

/* USB Setup Data Buffer base Higher */
/*   Contains the upper 4 bytes of the SETUP data packet from the latest
 *   CONTROL transfer. */
#define USB_SETUP_DATA_BUF_BASE_1       READONLY_REG32_POINTER(USB_SETUP_DATA_BUF_BASE_1_BASE)

/* USB_SETUP_DATA_BUF_BASE_1 bit positions (legacy definitions) */
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_POS 24
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_POS))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_POS 16
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_POS))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_POS 8
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_POS))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_POS 0
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_MASK ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_POS))

/* USB_SETUP_DATA_BUF_BASE_1 subregister pointers */
#define SETUP_DATA_BUF_BASE3_SHORT      READONLY_REG16_POINTER(USB_SETUP_DATA_BUF_BASE_1_BASE + 2)
#define SETUP_DATA_BUF_BASE2_SHORT      READONLY_REG16_POINTER(USB_SETUP_DATA_BUF_BASE_1_BASE + 0)

/* USB_SETUP_DATA_BUF_BASE_1 subregister bit positions */
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_POS 8
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_POS))
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_POS 0
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_POS))
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_POS 8
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_POS))
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_POS 0
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_MASK ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_POS))



#endif /* Q32M210_COMPAT_H */
