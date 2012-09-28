/* ----------------------------------------------------------------------------
 * Copyright (c) 2008-2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_analog.h
 * - Analog front-end hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53433 $
 * $Date: 2012-05-20 19:04:19 +0200 (sö, 20 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_ANALOG_H
#define Q32_ANALOG_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Analog Front-End - PRE_REV_E Decimation Filter Defines
 * ------------------------------------------------------------------------- */
/* AFE_DATARATE_CFG bit positions */
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_PRE_REV_E_Pos 24
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_PRE_REV_E_Mask ((uint32_t)(0x7FU << \
        AFE_DATARATE_CFG_DECIMATION_FACTOR_PRE_REV_E_Pos))

/* ----------------------------------------------------------------------------
 * Analog Front-End - Decimation Filter Settings
 *
 * Note: Decimation settings less than 0x100 indicate common settings shared
 *       between all chip revisions.
 *       Decimation settings between 0x100 to 0x200 indicate settings for
 *       pre-revE silicon
 *       Decimation settings between 0x200 to 0x300 indicate settings for
 *       revE or later silicon
 * ------------------------------------------------------------------------- */
#define DATARATE_NO_DECIMATION                  0x0
#define DATARATE_DECIMATE_BY_2                  0x1
#define DATARATE_DECIMATE_BY_4                  0x3
#define DATARATE_DECIMATE_BY_6                  0x204
#define DATARATE_DECIMATE_BY_8                  0x7
#define DATARATE_DECIMATE_BY_10                 0x208
#define DATARATE_DECIMATE_BY_12                 0xB
#define DATARATE_DECIMATE_BY_14                 0x20C
#define DATARATE_DECIMATE_BY_16                 0xF
#define DATARATE_DECIMATE_BY_18                 0x210
#define DATARATE_DECIMATE_BY_20                 0x13
#define DATARATE_DECIMATE_BY_22                 0x214
#define DATARATE_DECIMATE_BY_24                 0x17
#define DATARATE_DECIMATE_BY_26                 0x218
#define DATARATE_DECIMATE_BY_28                 0x1B
#define DATARATE_DECIMATE_BY_30                 0x21C
#define DATARATE_DECIMATE_BY_32                 0x1F
#define DATARATE_DECIMATE_BY_34                 0x220
#define DATARATE_DECIMATE_BY_36                 0x23
#define DATARATE_DECIMATE_BY_38                 0x224
#define DATARATE_DECIMATE_BY_40                 0x27
#define DATARATE_DECIMATE_BY_42                 0x228
#define DATARATE_DECIMATE_BY_44                 0x2B
#define DATARATE_DECIMATE_BY_46                 0x22C
#define DATARATE_DECIMATE_BY_48                 0x2F
#define DATARATE_DECIMATE_BY_50                 0x230
#define DATARATE_DECIMATE_BY_52                 0x130
#define DATARATE_DECIMATE_BY_56                 0x134
#define DATARATE_DECIMATE_BY_60                 0x138
#define DATARATE_DECIMATE_BY_64                 0x13C
#define DATARATE_DECIMATE_BY_68                 0x140
#define DATARATE_DECIMATE_BY_72                 0x144
#define DATARATE_DECIMATE_BY_76                 0x148
#define DATARATE_DECIMATE_BY_80                 0x14C
#define DATARATE_DECIMATE_BY_84                 0x150
#define DATARATE_DECIMATE_BY_88                 0x154
#define DATARATE_DECIMATE_BY_92                 0x158
#define DATARATE_DECIMATE_BY_96                 0x15C
#define DATARATE_DECIMATE_BY_100                0x160

/* Decimation offset defines for firmware usage */
#define DEC_OFFSET_PRE_REV_E                    0x100
#define DEC_OFFSET                              0x200

/* Decimation error defines */
#define DEC_CONFIG_NO_ERR                       0x0
#define DEC_CONFIG_INVALID_SETTING_ERR          0x1
#define DEC_CONFIG_INVALID_DEVICE_ERR           0x2

/* Decimation filter prototypes */
extern uint32_t Sys_Analog_Set_ADCDataRateDecimateConfig(uint32_t config);
extern uint32_t Sys_Analog_Set_ADCDataRateConfigAll(uint32_t config0,
                                                    uint32_t config1);


/* ----------------------------------------------------------------------------
 * Analog Front-End - DataRate Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADCDataRateConfig()
 * ----------------------------------------------------------------------------
 * Description   : Return the current data rate configuration
 * Inputs        : None
 * Outputs       : config - Data rate configuration
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADCDataRateConfig()
{
    return AFE->DATARATE_CFG;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Power Supply Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_PSUControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the power supply unit components
 * Inputs        : config   - Charge-pump and LCD backlight settings; use
 *                            VCP_ENABLE/VCP_DISABLE, LCDDRIVER_ENABLE/
 *                            LCDDRIVER_DISABLE, VDBL_ENABLE/VDBL_DISABLE and
 *                            LCDBACKLIGHT_DISABLE/LCDBACKLIGHT_*M*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_PSUControl(uint32_t config)
{
    AFE->PSU_CTRL = (config & ((1U << AFE_PSU_CTRL_VDBL_ENABLE_Pos)
                              | (1U << AFE_PSU_CTRL_LCDDRIVER_ENABLE_Pos)
                              | (1U << AFE_PSU_CTRL_VCP_ENABLE_Pos)
                              | AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_PSUControl()
 * ----------------------------------------------------------------------------
 * Description   : Read back the current power supply unit component
 *                 configuration settings
 * Inputs        : None
 * Outputs       : config   - Charge-pump and LCD backlight settings; compare
 *                            with CP_LDO_ENABLE/CP_LDO_DISABLE,
 *                            LCDDRIVER_ENABLE/LCDDRIVER_DISBALE,
 *                            CP_ENABLE/CP_DISABLE and LCDBACKLIGHT_DISABLE/
 *                            LCDBACKLIGHT_*_INOM
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_PSUControl()
{
    return AFE->PSU_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_OpModeControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the operation mode control settings
 * Inputs        : config - Operation mode control settings; use
 *                          RC_OSC_ENABLE/RC_OSC_DISABLE,
 *                          VADC_ENABLE/VADC_DISABLE,
 *                          STANDBY_MODE_ENABLE/STANDBY_MODE_DISABLE and
 *                          SLEEP_MODE_ENABLE/SLEEP_MODE_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_OpModeControl(uint32_t config)
{
    AFE->OPMODE_CTRL = (config & ((1U << AFE_OPMODE_CTRL_RC_OSC_ENABLE_Pos)
                                 | (1U << AFE_OPMODE_CTRL_VADC_ENABLE_Pos)
                                 | (1U << AFE_OPMODE_CTRL_STANDBY_MODE_Pos)
                                 | (1U << AFE_OPMODE_CTRL_SLEEP_MODE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_OpModeControl()
 * ----------------------------------------------------------------------------
 * Description   : Get the current operation mode control settings
 * Inputs        : None
 * Outputs       : config - Operation mode control settings; compare with
 *                          RC_OSC_ENABLE/RC_OSC_DISABLE,
 *                          VADC_ENABLE/VADC_DISABLE,
 *                          STANDBY_MODE_ENABLE/STANDBY_MODE_DISABLE and
 *                          SLEEP_MODE_ENABLE/SLEEP_MODE_DISABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_OpModeControl()
{
    return AFE->OPMODE_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Clock Related Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_CCRControl32K(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the CCR trim value for the 32 kHz crystal
 * Inputs        : config - CCR trim value setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_CCRControl32K(uint32_t config)
{
    AFE->OSC_32K_CCR_CTRL = (config & AFE_32K_CCR_CTRL_CCR_CTRL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_CCRControl32K()
 * ----------------------------------------------------------------------------
 * Description   : Read back the current CCR trim value for the 32 kHz crystal
 * Inputs        : None
 * Outputs       : config - CCR trim value setting
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_CCRControl32K()
{
    return AFE->OSC_32K_CCR_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_CCRControl48M(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the CCR trim setting for the 48 MHz crystal
 * Inputs        : config - CCR trim setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_CCRControl48M(uint32_t config)
{
    REG8_POINTER(AFE_48M_CCR_CTRL_BASE) =
                    (uint8_t)(config & AFE_48M_CCR_CTRL_CCR_CTRL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_CCRControl48M()
 * ----------------------------------------------------------------------------
 * Description   : Read back the current CCR trim setting for the 48 MHz crystal
 * Inputs        : None
 * Outputs       : config - CCR trim value setting
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_CCRControl48M()
{
    return (AFE->OSC_48M_CCR_CTRL & AFE_48M_CCR_CTRL_CCR_CTRL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_CCRControlRC(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the CCR trim value for the RC oscillator
 * Inputs        : config - CCR trim value setting; use constants shifted to
 *                          AFE_RC_CCR_CTRL_FINE_CTRL and
 *                          AFE_RC_CCR_CTRL_COARSE_CTRL with RC_OSC_FREQ_RANGE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_CCRControlRC(uint32_t config)
{
    AFE->RC_CCR_CTRL = (config & (AFE_RC_CCR_CTRL_FINE_CTRL_Mask |
                                 AFE_RC_CCR_CTRL_RANGE_SEL_Mask |
                                 AFE_RC_CCR_CTRL_COARSE_CTRL_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_CCRControlRC()
 * ----------------------------------------------------------------------------
 * Description   : Read back the current CCR trim value for the RC oscillator
 * Inputs        : None
 * Outputs       : config - CCR trim value setting; compare with constants
 *                          shifted to AFE_RC_CCR_CTRL_FINE_CTRL and
 *                          AFE_RC_CCR_CTRL_COARSE_CTRL with RC_OSC_FREQ_RANGE_*
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_CCRControlRC()
{
    return AFE->RC_CCR_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Clock Related Function Prototypes
 * ------------------------------------------------------------------------- */

extern FlashStatus Sys_Analog_Set_RCFreq(uint32_t freq);

/* ----------------------------------------------------------------------------
 * Analog Front-End - Operational Amplifier Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_AmplifierInputSelect(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select the configuration of the operational amplifier
 *                 switches by selecting the inputs to A0 and the signal on the
 *                 ALT pad
 * Inputs        : config - Selected switch settings; use
 *                          ALT0_SW_SEL_*, ALT0_SW_NONE or ALT0_SW_DISABLE;
 *                          ALT1_SW_SEL_*, ALT1_SW_NONE or ALT1_SW_DISABLE;
 *                          and A0_CFG_SEL_IN* or A0_CFG_SEL_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_AmplifierInputSelect(uint32_t config)
{
    AFE->IN_SW_CTRL = (config & (AFE_IN_SW_CTRL_ALT1_SW_Mask
                                | AFE_IN_SW_CTRL_ALT0_SW_Mask
                                | AFE_IN_SW_CTRL_A0_IN_CFG_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_AmplifierInputSelect()
 * ----------------------------------------------------------------------------
 * Description   : Return the current configuration of the operational
 *                 amplifier input selection switches
 * Inputs        : None
 * Outputs       : config - Current selected switch settings; compare with
 *                          ALT0_SW_SEL_*, ALT0_SW_NONE or ALT0_SW_DISABLE;
 *                          ALT1_SW_SEL_*, ALT1_SW_NONE or ALT1_SW_DISABLE;
 *                          and A0_CFG_SEL_IN* or A0_CFG_SEL_DISABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_AmplifierInputSelect()
{
    return AFE->IN_SW_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_AmplifierControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select which operational amplifiers are enabled (including
 *                 which are enabled in low-power mode)
 * Inputs        : config - Selected amplifier configuration; use A*_ENABLE/
 *                          A*_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_AmplifierControl(uint32_t config)
{
    AFE->AMP_CTRL = (config & ((1U << AFE_AMP_CTRL_A2_ENABLE_Pos)
                               | (1U << AFE_AMP_CTRL_A1_ENABLE_Pos)
                               | (1U << AFE_AMP_CTRL_A0_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_AmplifierControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current operational amplifier enable settings
 * Inputs        : None
 * Outputs       : config - Current enabled amplifiers; compare with A*_ENABLE/
 *                          A*_DISABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_AmplifierControl()
{
    return AFE->AMP_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void
 *                 Sys_Analog_Set_AmplifierOutputSelect(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select the configuration of the output operational amplifier
 *                 switches
 * Inputs        : config - Selected output switch configuration; use a set of
 *                          A*_OUTA_DISABLE/A*_OUTA_ENABLE and
 *                          A*_OUTB_DISABLE/A*_OUTB_ENABLE settings
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_AmplifierOutputSelect(uint32_t config)
{
    AFE->OUT_SW_CTRL = (config & ((1U << AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_Pos)
                                 | (1U << AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_Pos)
                                 | (1U << AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_Pos)
                                 | (1U << AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_Pos)
                                 | (1U << AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_Pos)
                                 | (1U << AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_AmplifierOutputSelect()
 * ----------------------------------------------------------------------------
 * Description   : Return the current output switch configuration for the
 *                 operational amplifiers
 * Inputs        : None
 * Outputs       : config - Current selected switch settings; compare with a
 *                          set of A*_OUTA_DISABLE/A*_OUTA_ENABLE and
 *                          A*_OUTB_DISABLE/A*_OUTB_ENABLE settings
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_AmplifierOutputSelect()
{
    return AFE->OUT_SW_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Switch Control Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_SPSTControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the single-pole, single-throw switches
 * Inputs        : config - Selected SPST switch configuration; use
 *                          SPST*_SEL_OPEN/SPST*_SEL_CLOSE,
 *                          SPST*_DISABLE/SPST*_ENABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_SPSTControl(uint32_t config)
{
    AFE->SPST_CTRL = (config & ((1U << AFE_SPST_CTRL_SPST3_SELECT_Pos)
                                | (1U << AFE_SPST_CTRL_SPST3_DISABLE_Pos)
                                | (1U << AFE_SPST_CTRL_SPST2_SELECT_Pos)
                                | (1U << AFE_SPST_CTRL_SPST2_DISABLE_Pos)
                                | (1U << AFE_SPST_CTRL_SPST1_SELECT_Pos)
                                | (1U << AFE_SPST_CTRL_SPST1_DISABLE_Pos)
                                | (1U << AFE_SPST_CTRL_SPST0_SELECT_Pos)
                                | (1U << AFE_SPST_CTRL_SPST0_DISABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_SPSTControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current control configuration for the single-pole,
 *                 single-throw switches
 * Inputs        : None
 * Outputs       : config - Current selected switch settings; compare with
 *                          SPST*_SEL_OPEN/SPST*_SEL_CLOSE,
 *                          SPST*_DISABLE/SPST*_ENABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_SPSTControl()
{
    return AFE->SPST_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_MSWControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the multi-switches
 * Inputs        : config - Selected multi-switch configuration; use MSW*_SEL_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_MSWControl(uint32_t config)
{
    AFE->MSW_CTRL = (config & (AFE_MSW_CTRL_MSW3_SELECT_Mask
                              | AFE_MSW_CTRL_MSW2_SELECT_Mask
                              | AFE_MSW_CTRL_MSW1_SELECT_Mask
                              | AFE_MSW_CTRL_MSW0_SELECT_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_MSWControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current control configuration for the
 *                 multi-switches
 * Inputs        : None
 * Outputs       : config - Current selected switch settings; compare with
 *                          MSW*_SEL_*
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_MSWControl()
{
    return AFE->MSW_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Programmable Gain Amplifier Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_PGA0Control(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Enable PGA0, selecting the input and differential input for
 *                 the programmable gain amplifier
 * Inputs        : config - PGA0 control setting; use PGA0_DIF_SEL_*,
 *                          PGA0_SEL_* and PGA0_ENABLE/PGA0_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_PGA0Control(uint32_t config)
{
    AFE->PGA0_CTRL = (config & (AFE_PGA0_CTRL_PGA0_DIF_SELECT_Mask
                              | AFE_PGA0_CTRL_PGA0_SELECT_Mask
                              | (1U << AFE_PGA0_CTRL_PGA0_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_PGA0Control()
 * ----------------------------------------------------------------------------
 * Description   : Return the current enable setting, as well as the input and
 *                 differential input selection for PGA0
 * Inputs        : None
 * Outputs       : config - The current PGA0 control setting; compare with
 *                          PGA0_DIF_SEL_*, PGA0_SEL_* and
 *                          PGA0_ENABLE/PGA0_DISABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_PGA0Control()
{
    return AFE->PGA0_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_PGA1Control(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Enable PGA1, selecting the input and differential input for
 *                 the programmable gain amplifier
 * Inputs        : config - PGA1 control setting; use PGA1_DIF_SEL_*,
 *                          PGA1_SEL_* and PGA1_ENABLE/PGA1_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_PGA1Control(uint32_t config)
{
    AFE->PGA1_CTRL = (config & (AFE_PGA1_CTRL_PGA1_DIF_SELECT_Mask
                              | AFE_PGA1_CTRL_PGA1_SELECT_Mask
                              | (1U << AFE_PGA1_CTRL_PGA1_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_PGA1Control()
 * ----------------------------------------------------------------------------
 * Description   : Return the current enable setting, as well as the input
 *                 and differential input selection for PGA1
 * Inputs        : None
 * Outputs       : config - The current PGA1 control setting; compare with
 *                          PGA1_DIF_SEL_*, PGA1_SEL_* and
 *                          PGA1_ENABLE/PGA1_DISABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_PGA1Control()
{
    return AFE->PGA1_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_PGAGainControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select the cut-off frequency and gain configuration for the
 *                 PGAs
 * Inputs        : config - The programmable gain amplifier gain and cut-off
 *                          frequency configuration; use PGA_CUT_OFF_*,
 *                          PGA0_GAIN_*DB and PGA1_GAIN_*DB
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_PGAGainControl(uint32_t config)
{
    AFE->PGA_GAIN_CTRL = (config & (AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Mask
                                   | AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Mask
                                   | AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_PGAGainControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current cut-off frequency and gain configuration
 *                 for the PGAs
 * Inputs        : None
 * Outputs       : config - The current programmable gain amplifier gain and
 *                          cut-off frequency configuration; compare with
 *                          PGA_CUT_OFF_*, PGA0_GAIN_*DB and PGA1_GAIN_*DB
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_PGAGainControl()
{
    return AFE->PGA_GAIN_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_PGA1Mode(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the input mode for the PGA1 inputs
 * Inputs        : config - PGA1 input modes; use PGA1_A_MODE_*_BYTE and
 *                          PGA1_B_MODE_*_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_PGA1Mode(uint32_t config)
{
    AFE_DAC_CTRL->PGA1_MODE_CTRL_BYTE =
                            (config & (PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Mask
                             | PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_PGA1Mode()
 * ----------------------------------------------------------------------------
 * Description   : Return the current input mode settings for the PGA1 inputs
 * Inputs        : None
 * Outputs       : config - PGA1 input modes; compare with PGA1_A_MODE_*_BYTE
 *                          and PGA1_B_MODE_*_BYTE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_PGA1Mode()
{
    return AFE_DAC_CTRL->PGA1_MODE_CTRL_BYTE;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Threshold Comparator Function Prototypes
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ComparatorThreshold(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the threshold for the threshold comparator
 * Inputs        : config - Desired comparator threshold; use
 *                          THRESHOLD_COMPARE_DISABLE or THRESHOLD_COMPARE_*MV
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ComparatorThreshold(uint32_t config)
{
    AFE->THRESHOLD_COMPARE_CTRL = (config
                                  & AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ComparatorThreshold()
 * ----------------------------------------------------------------------------
 * Description   : Get the current threshold setting for the threshold
 *                 comparator
 * Inputs        : None
 * Outputs       : config - Current selected comparator threshold; compare with
 *                          THRESHOLD_COMPARE_DISABLE or THRESHOLD_COMPARE_*MV
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ComparatorThreshold()
{
    return AFE->THRESHOLD_COMPARE_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - ADC Inline Functions
 * ------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ADCControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the ADCs
 * Inputs        : config - Calibration, data format and enable configuration
 *                          setting for the ADCs; use
 *                          ADC*_GAIN_ENABLE/ADC*_GAIN_DISABLE,
 *                          ADC*_OFFSET_ENABLE/ADC*_OFFSET_DISABLE,
 *                          ADC*_FORMAT_TWOS_COMP/ADC*_FORMAT_UNSIGNED_INT and
 *                          ADC*_DISABLE/ADC*_ENABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADCControl(uint32_t config)
{
    /* Writing to ADC_CTRL_BYTE prevents any accidental update of the reserved
     * byte at (AFE_ADC_CTRL + 1). Note: This function takes advantage of the
     * fact that ADC_CTRL_BYTE is the low byte of AFE_ADC_CTRL and all the
     * relevant bit-fields are located in this byte only. */
    AFE_ADC_CTRL->ADC_CTRL_BYTE = (uint8_t)(config &
                                  ((1U << AFE_ADC_CTRL_ADC1_GAIN_ENABLE_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC0_GAIN_ENABLE_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC1_FORMAT_CFG_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC0_FORMAT_CFG_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC1_ENABLE_Pos)
                                   | (1U << AFE_ADC_CTRL_ADC0_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADCControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current ADC control configuration
 * Inputs        : None
 * Outputs       : config - Calibration, data format and enable configuration
 *                          setting for the ADCs; compare with
 *                          ADC*_GAIN_ENABLE/ADC*_GAIN_DISABLE,
 *                          ADC*_OFFSET_ENABLE/ADC*_OFFSET_DISABLE,
 *                          ADC*_FORMAT_TWOS_COMP/ADC*_FORMAT_UNSIGNED_INT and
 *                          ADC*_DISABLE/ADC*_ENABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADCControl()
{
    return (uint32_t)AFE_ADC_CTRL->ADC_CTRL_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC0Data()
 * ----------------------------------------------------------------------------
 * Description   : Get the most recent sample from ADC0; if ADC0 is configured
 *                 to provide signed data, the return value should be cast as a
 *                 signed int.
 * Inputs        : None
 * Outputs       : data - The most recent sample from ADC0
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC0Data()
{
    return AFE->ADC0_DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC1Data()
 * ----------------------------------------------------------------------------
 * Description   : Get the most recent sample from ADC1; if ADC1 is configured
 *                 to provide signed data, the return value should be cast as a
 *                 signed int.
 * Inputs        : None
 * Outputs       : data - The most recent sample from ADC1
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC1Data()
{
    return AFE->ADC1_DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC01Data()
 * ----------------------------------------------------------------------------
 * Description   : Get the most recent samples from ADC0 and ADC1
 * Inputs        : None
 * Outputs       : data - The most recent samples from ADC0 and ADC1; each
 *                        sample can be extracted by shifting to
 *                        AFE_ADC01_DATA_ADC0_Pos and AFE_ADC01_DATA_ADC1_Pos
 *                        respectively
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC01Data()
{
    return AFE->ADC01_DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : void
 *                 Sys_Analog_Set_ADC0OffsetCalibration(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the offset calibration for ADC0
 * Inputs        : config - Offset calibration value for ADC0
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADC0OffsetCalibration(uint32_t config)
{
    AFE->ADC0_OFFSET = (config & AFE_ADC0_OFFSET_CAL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC0OffsetCalibration()
 * ----------------------------------------------------------------------------
 * Description   : Return the current offset calibration for ADC0
 * Inputs        : None
 * Outputs       : config - Current offset calibration for ADC0
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC0OffsetCalibration()
{
    return AFE->ADC0_OFFSET;
}

/* ----------------------------------------------------------------------------
 * Function      : void
 *                 Sys_Analog_Set_ADC1OffsetCalibration(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the offset calibration for ADC1
 * Inputs        : config - Offset calibration value for ADC1
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADC1OffsetCalibration(uint32_t config)
{
    AFE->ADC1_OFFSET = (config & AFE_ADC1_OFFSET_CAL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC1OffsetCalibration()
 * ----------------------------------------------------------------------------
 * Description   : Return the current offset calibration for ADC1
 * Inputs        : None
 * Outputs       : config - Current offset calibration for ADC1
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC1OffsetCalibration()
{
    return AFE->ADC1_OFFSET;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ADC0GainCalibration(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the gain calibration for ADC0
 * Inputs        : config - Gain calibration value for ADC0
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADC0GainCalibration(uint32_t config)
{
    AFE->ADC0_GAIN = (config & AFE_ADC0_GAIN_CAL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC0GainCalibration()
 * ----------------------------------------------------------------------------
 * Description   : Return the current gain calibration for ADC0
 * Inputs        : None
 * Outputs       : config - Current gain calibration for ADC0
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC0GainCalibration()
{
    return AFE->ADC0_GAIN;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ADC1GainCalibration(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the gain calibration for ADC1
 * Inputs        : config - Gain calibration value for ADC1
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADC1GainCalibration(uint32_t config)
{
    AFE->ADC1_GAIN = (config & AFE_ADC1_GAIN_CAL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_ADC1GainCalibration()
 * ----------------------------------------------------------------------------
 * Description   : Return the current gain calibration for ADC1
 * Inputs        : None
 * Outputs       : config - Current gain calibration for ADC1
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_ADC1GainCalibration()
{
    return AFE->ADC1_GAIN;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ADCDataRateConfig(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the un-decimated ADC data rate. This function
 *                 sets the decimation factor to no decimation (bypass).
 *                 As a result, the data rate can be calculated as:
 *                 - Data Rate = MCLK Frequency /
 *                               (2 * ((2 ^ FREQ_MODE) + FREQ_ADJUST))
 * Inputs        : config - Data rate frequency adjustment factor; use a
 *                          constant shifted to AFE_DATARATE_CFG_ADJUST_Pos
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADCDataRateConfig(uint32_t config)
{
    AFE->DATARATE_CFG = ((config & AFE_DATARATE_CFG_ADJUST_Mask)
                        | DATARATE_NO_DECIMATION);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_ADCFreqMode(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the oversampling mode (FREQ_MODE) used in the
 *                 calculation of the ADC data rate.
 * Inputs        : config - Data rate oversampling mode; use a
 *                          constant between 0x3 and 0xA shifted to
 *                          AFE_DATARATE_CFG1_FREQ_MODE_Pos
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_ADCFreqMode(uint32_t config)
{
    uint32_t *temp = (uint32_t *)0x40000088;

    AFE->DATARATE_CFG1 = (config & AFE_DATARATE_CFG1_FREQ_MODE_Mask);
    
    /* Perform a recalibration of the ADCs to obtain optimal performance */
    switch( config )
    {
    case 0x7:
        *temp = 0x14A;
        break;
    case 0x8:
    default:
        *temp = 0x1E8;
        break;
    case 0x9:        
        *temp = 0x16A;
        break;
    case 0xA:
        *temp = 0x1C9;
        break;
    }
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Pulse-Width Modulator Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM_Enable(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select which pulse-width modulators are enabled and brought
 *                 out to the multiplexed GPIO pads
 * Inputs        : config - The PWM signals to be brought out to the multiplexed
 *                          GPIO pads; use PWM*_ENABLE/PWM*_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM_Enable(uint32_t config)
{
    GPIO->IF4_PWM_CTRL = (config & ((1U << GPIO_IF4_PWM_CTRL_PWM0_ENABLE_Pos)
                             | (1U << GPIO_IF4_PWM_CTRL_PWM1_ENABLE_Pos)
                             | (1U << GPIO_IF4_PWM_CTRL_PWM2_ENABLE_Pos)
                             | (1U << GPIO_IF4_PWM_CTRL_PWM3_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM0_Config(uint32_t period,
 *                                             uint32_t duty)
 * ----------------------------------------------------------------------------
 * Description   : Configure pulse-width modulator 0
 * Inputs        : - period - The period length for PWM0 in cycles
 *                 - duty   - The high part of the period for PWM0 in cycles
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM0_Config(uint32_t period, uint32_t duty)
{
    AFE->PWM0_PERIOD = (period & AFE_PWM0_PERIOD_PWM0_PERIOD_Mask);
    AFE->PWM0_HI = (duty & AFE_PWM0_HI_PWM0_HI_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM1_Config(uint32_t period,
 *                                             uint32_t duty)
 * ----------------------------------------------------------------------------
 * Description   : Configure pulse-width modulator 1
 * Inputs        : - period - The period length for PWM1 in cycles
 *                 - duty   - The high part of the period for PWM1 in cycles
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM1_Config(uint32_t period, uint32_t duty)
{
    AFE->PWM1_PERIOD = (period & AFE_PWM1_PERIOD_PWM1_PERIOD_Mask);
    AFE->PWM1_HI = (duty & AFE_PWM1_HI_PWM1_HI_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM2_Config(uint32_t period,
 *                                             uint32_t duty)
 * ----------------------------------------------------------------------------
 * Description   : Configure pulse-width modulator 2
 * Inputs        : - period - The period length for PWM2 in cycles
 *                 - duty   - The high part of the period for PWM2 in cycles
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM2_Config(uint32_t period, uint32_t duty)
{
    AFE->PWM2_PERIOD = (period & AFE_PWM2_PERIOD_PWM2_PERIOD_Mask);
    AFE->PWM2_HI = (duty & AFE_PWM2_HI_PWM2_HI_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM3_Config(uint32_t period,
 *                                             uint32_t duty)
 * ----------------------------------------------------------------------------
 * Description   : Configure pulse-width modulator 3
 * Inputs        : - period - The period length for PWM3 in cycles
 *                 - duty   - The high part of the period for PWM3 in cycles
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM3_Config(uint32_t period, uint32_t duty)
{
    AFE->PWM3_PERIOD = (period & AFE_PWM3_PERIOD_PWM3_PERIOD_Mask);
    AFE->PWM3_HI = (duty & AFE_PWM3_HI_PWM3_HI_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_PWM_ConfigAll(uint32_t period,
 *                                               uint32_t duty)
 * ----------------------------------------------------------------------------
 * Description   : Configure all four pulse-width modulators with the same
 *                 configuration
 * Inputs        : - period - The period length for the PWMs in cycles
 *                 - duty   - The high part of the period for the PWMs in cycles
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_PWM_ConfigAll(uint32_t period, uint32_t duty)
{
    AFE->PWM0_PERIOD = (period & AFE_PWM0_PERIOD_PWM0_PERIOD_Mask);
    AFE->PWM0_HI = (duty & AFE_PWM0_HI_PWM0_HI_Mask);

    AFE->PWM1_PERIOD = (period & AFE_PWM1_PERIOD_PWM1_PERIOD_Mask);
    AFE->PWM1_HI = (duty & AFE_PWM1_HI_PWM1_HI_Mask);

    AFE->PWM2_PERIOD = (period & AFE_PWM2_PERIOD_PWM2_PERIOD_Mask);
    AFE->PWM2_HI = (duty & AFE_PWM2_HI_PWM2_HI_Mask);

    AFE->PWM3_PERIOD = (period & AFE_PWM3_PERIOD_PWM3_PERIOD_Mask);
    AFE->PWM3_HI = (duty & AFE_PWM3_HI_PWM3_HI_Mask);
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Temperature Sensor Function Prototypes
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_TempSenseControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the temperature sensor
 * Inputs        : config - Temperature sensor configuration; use
 *                          TEMP_SENSE_DISABLE/TEMP_SENSE_ENABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_TempSenseControl(uint32_t config)
{
    AFE->TEMP_SENSE_CTRL = (config
                           & (1U << AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_TempSenseControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current temperature sensor configuration
 * Inputs        : None
 * Outputs       : config - Temperature sensor configuration; compare with
 *                          TEMP_SENSE_DISABLE/TEMP_SENSE_ENABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_TempSenseControl()
{
    return AFE->TEMP_SENSE_CTRL;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - DAC Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DAC0VoltageRef(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the voltage reference range for DAC0
 * Inputs        : config - Voltage reference range configuration for DAC0; use
 *                          DAC0_REF_SELECT_VREF/DAC0_REF_SELECT_2VREF/
 *                          DAC0_REF_SELECT_3VREF
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DAC0VoltageRef(uint32_t config)
{
    AFE->DAC0_REF_CTRL = (config & AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_DAC0VoltageRef()
 * ----------------------------------------------------------------------------
 * Description   : Read back the current voltage reference range configuration
 *                 for DAC0
 * Inputs        : None
 * Outputs       : config - Voltage reference range configuration for DAC0;
 *                          compare with DAC0_REF_SELECT_VREF/
 *                          DAC0_REF_SELECT_2VREF/DAC0_REF_SELECT_3VREF
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_DAC0VoltageRef()
{
    return AFE->DAC0_REF_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DACControl(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select which DACs are enabled and configure the DAC2
 *                 output trim
 * Inputs        : config - Enable configuration and trim configuration; use
 *                          DAC2_CUR_NOMINAL/DAC2_CUR_*_NOMINAL,
 *                          and DAC*_DISABLE/DAC*_ENABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DACControl(uint32_t config)
{
    /* Writing to DAC_CTRL_BYTE prevents any accidental update of the reserved
     * byte at (AFE_DAC_CTRL + 1). Note: This function takes advantage of the
     * fact that DAC_CTRL_BYTE is the low byte of AFE_DAC_CTRL and all the
     * relevant bit-fields are located in this byte only. */
    AFE_DAC_CTRL->DAC_CTRL_BYTE = (uint8_t) (config
                                     & (AFE_DAC_CTRL_DAC2_CUR_Mask
                                       | (1U << AFE_DAC_CTRL_DAC2_ENABLE_Pos)
                                       | (1U << AFE_DAC_CTRL_DAC1_ENABLE_Pos)
                                       | (1U << AFE_DAC_CTRL_DAC0_ENABLE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_DACControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current DAC control configuration
 * Inputs        : None
 * Outputs       : config - Current enable configuration and trim configuration;
 *                          compare with DAC2_CUR_NOMINAL/DAC2_CUR_*_NOMINAL,
 *                          and DAC*_DISABLE/DAC*_ENABLE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_DACControl()
{
    return (uint32_t)AFE_DAC_CTRL->DAC_CTRL_BYTE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DAC0Data(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Set the next output data sample for DAC0
 * Inputs        : data - Data sample to be output on DAC0
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DAC0Data(uint32_t data)
{
    AFE->DAC0_DATA = (data & AFE_DAC0_DATA_DAC0_DATA_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DAC1Data(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Set the next output data sample for DAC1
 * Inputs        : data - Data sample to be output on DAC1
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DAC1Data(uint32_t data)
{
    AFE->DAC1_DATA = (data & AFE_DAC1_DATA_DAC1_DATA_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DAC2Data(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Set the next output data sample for DAC2
 * Inputs        : data - Data sample to be output on DAC2
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DAC2Data(uint32_t data)
{
    AFE->DAC2_DATA = (data & AFE_DAC2_DATA_DAC2_DATA_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_DAC01Data(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Set the next output data samples for DAC0 and for DAC1
 * Inputs        : data - Data samples to be output on DAC0 and DAC1; use 16-bit
 *                        values shifted to the AFE_DAC01_DATA_DAC0_DATA_Pos and
 *                        AFE_DAC01_DATA_DAC1_DATA_Pos
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_DAC01Data(uint32_t data)
{
    AFE->DAC01_DATA = (data & (AFE_DAC01_DATA_DAC0_DATA_Mask
                              | AFE_DAC01_DATA_DAC1_DATA_Mask));
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Real-Time Clock Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_RTCControl()
 * ----------------------------------------------------------------------------
 * Description   : Return the current real-time clock counter and alarm mode
 *                 configuration
 * Inputs        : None
 * Outputs       : config - RTC operation and alarm modes; compare with
 *                          RTC_DISABLE/RTC_ENABLE, ALARM_DISABLE/ALARM_ENABLE
 *                          and RTC_RUN_MODE/RTC_SET_MODE
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_RTCControl()
{
    return AFE->RTC_CTRL;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_RTCCount()
 * ----------------------------------------------------------------------------
 * Description   : Return the current time as indicated by the real-time clock
 *                 counter
 * Inputs        : None
 * Outputs       : data - The current time
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_RTCCount()
{
    return AFE->RTC_COUNT;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_RTCAlarm()
 * ----------------------------------------------------------------------------
 * Description   : Return the current settings for the real-time clock alarm
 * Inputs        : None
 * Outputs       : config - RTC alarm count and mode; compare with
 *                          ALARM_ABSOLUTE_MODE/ALARM_RELATIVE_MODE and a
 *                          a 31-bit RTC alarm count setting
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_RTCAlarm()
{
    return AFE->RTC_ALARM;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - Real-Time Clock Function Prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_Analog_Set_RTCControl(uint32_t freq_in_hz,
                                      uint32_t config);
extern void Sys_Analog_Set_RTCCount(uint32_t freq_in_hz,
                                    uint32_t config);
extern void Sys_Analog_Set_RTCAlarm(uint32_t freq_in_hz,
                                    uint32_t config);

extern void Sys_Analog_RTC_Config(uint32_t freq_in_hz, uint32_t config,
                                  uint32_t count, uint32_t alarm);

/* ----------------------------------------------------------------------------
 * Analog Front-End - AFE Status Related Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Clear_InterruptStatus(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Clear the current AFE interrupt status bits
 * Inputs        : config   - Interrupt status bits to clear; use
 *                            RTC_CLOCK_CLEAR, RTC_ALARM_CLEAR,
 *                            WAKEUP_IF5_PIN0_CLEAR, WAKEUP_IF5_PIN1_CLEAR,
 *                            WAKEUP_IF5_PIN2_CLEAR, WAKEUP_IF5_PIN3_CLEAR,
 *                            and THESHOLD_COMPARE_CLEAR
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Clear_InterruptStatus(uint32_t config)
{
     AFE->INTERRUPT_STATUS = (config &
                             ((1U << AFE_INTERRUPT_STATUS_IF5_PIN3_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_IF5_PIN2_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_IF5_PIN1_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_RTC_CLOCK_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_RTC_ALARM_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_IF5_PIN0_Pos)
                             | (1U << AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Analog_Get_InterruptStatus()
 * ----------------------------------------------------------------------------
 * Description   : Return the current AFE interrupt status
 * Inputs        : None
 * Outputs       : config - Interrupt status; compare with
 *                          WAKEUP_INT_PEND/WAKEUP_INT_CLEARED,
 *                          WAKEUP_IF5_PIN0_SET/WAKEUP_IF5_PIN0_NOT_SET,
 *                          WAKEUP_IF5_PIN1_SET/WAKEUP_IF5_PIN1_NOT_SET,
 *                          WAKEUP_IF5_PIN2_SET/WAKEUP_IF5_PIN2_NOT_SET,
 *                          WAKEUP_IF5_PIN3_SET/WAKEUP_IF5_PIN3_NOT_SET,
 *                          RTC_CLOCK_SET/RTC_CLOCK_NOT_SET,
 *                          RTC_ALARM_SET/RTC_ALARM_NOT_SET,
 *                          and THESHOLD_COMPARE_SET/THESHOLD_COMPARE_NOT_SET
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Analog_Get_InterruptStatus()
{
    return AFE->INTERRUPT_STATUS;
}

/* ----------------------------------------------------------------------------
 * Analog Front-End - General Purpose Retention Register Inline Functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Analog_Set_RetentionReg(uint8_t data)
 * ----------------------------------------------------------------------------
 * Description   : Set the general-purpose data retention register
 * Inputs        : data - Data to be stored in the general-purpose data
 *                        retention register
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Analog_Set_RetentionReg(uint8_t data)
{
    AFE->RETENTION_DATA = data;
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t Sys_Analog_Get_RetentionReg()
 * ----------------------------------------------------------------------------
 * Description   : Return the current value of the general-purpose data
 *                 retention register
 * Inputs        : None
 * Outputs       : data - Data loaded from the general-purpose data retention
 *                        register
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_Analog_Get_RetentionReg()
{
    return AFE->RETENTION_DATA;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_ANALOG_H */
