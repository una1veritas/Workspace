/* ----------------------------------------------------------------------------
 * Copyright (c) 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210_hw.h
 * - Q32M210 hardware registers and bit-field definitions
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_HW_H
#define Q32M210_HW_H

/* Include the system library register header */
#include <q32_reg.h>

/* ----------------------------------------------------------------------------
 * Analog Front-End
 * ------------------------------------------------------------------------- */
/* Control and configuration of the analog system components. This block
 * includes the clock, power supply, switch (both SPST switches and multi-
 * switches), input and output control and configuration. */

typedef struct
{
    __IO uint32_t PSU_CTRL;             /* Configure the charge pump and LCD backlight */
    __IO uint32_t OPMODE_CTRL;          /* Control the operating mode features */
    __IO uint32_t OSC_32K_CCR_CTRL;     /* Trimming of 32 kHz crystal frequency */
         uint32_t RESERVED0;
    __IO uint32_t OSC_48M_CCR_CTRL;     /* Trimming of 48 MHz crystal frequency */
         uint32_t RESERVED1;
    __IO uint32_t RC_CCR_CTRL;          /* Trimming of RC oscillator frequency */
         uint32_t RESERVED2[3];
    __IO uint32_t IN_SW_CTRL;           /* Configuration of the inputs to operational amplifier A0 and the ALT pads */
    __IO uint32_t AMP_CTRL;             /* Enabling of the operational amplifiers A0 to A2 */
    __IO uint32_t OUT_SW_CTRL;          /* Output control and configuration for operational amplifiers A0 to A2 */
         uint32_t RESERVED3;
    __IO uint32_t SPST_CTRL;            /* Control of SPST switches */
    __IO uint32_t MSW_CTRL;             /* Multi-Switch control and configuration */
         uint32_t RESERVED4[2];
    __IO uint32_t PGA0_CTRL;            /* Configure PGA0 and select the inputs for this programmable gain amplifier */
    __IO uint32_t PGA1_CTRL;            /* Configure PGA1 and select the inputs for this programmable gain amplifier */
    __IO uint32_t PGA_GAIN_CTRL;        /* Configure the cut-off frequency and gain settings for PGA0, PGA1 */
    __IO uint32_t THRESHOLD_COMPARE_CTRL;/* Configure and control the voltage comparator */
    __IO uint32_t ADC_CTRL;             /* Configure and control the analog to digital convertors */
    __I  uint32_t ADC0_DATA;            /* ADC0 converter output */
    __I  uint32_t ADC1_DATA;            /* ADC1 converter output */
    __I  uint32_t ADC01_DATA;           /* 16 MSBs of the ADC0 and ADC1 converter output */
    __IO uint32_t ADC0_OFFSET;          /* ADC0 offset calibration */
    __IO uint32_t ADC1_OFFSET;          /* ADC1 offset calibration */
    __IO uint32_t ADC0_GAIN;            /* ADC0 gain calibration */
    __IO uint32_t ADC1_GAIN;            /* ADC1 gain calibration */
    __IO uint32_t DATARATE_CFG;         /* Configure the ADC data rate including the decimation factor */
         uint32_t RESERVED5[2];
    __IO uint32_t DATARATE_CFG1;        /* Configure the oversampling mode used by the ADCs */
    __IO uint32_t PWM0_PERIOD;          /* Set the legnth of each PWM0 period */
    __IO uint32_t PWM1_PERIOD;          /* Set the legnth of each PWM1 period */
    __IO uint32_t PWM2_PERIOD;          /* Set the legnth of each PWM2 period */
    __IO uint32_t PWM3_PERIOD;          /* Set the legnth of each PWM3 period */
    __IO uint32_t PWM0_HI;              /* Configure the PWM0 duty cycle by selecting how many cycles are run in the high state */
    __IO uint32_t PWM1_HI;              /* Configure the PWM1 duty cycle by selecting how many cycles are run in the high state */
    __IO uint32_t PWM2_HI;              /* Configure the PWM2 duty cycle by selecting how many cycles are run in the high state */
    __IO uint32_t PWM3_HI;              /* Configure the PWM3 duty cycle by selecting how many cycles are run in the high state */
    __IO uint32_t TEMP_SENSE_CTRL;      /* Temperature sensor control */
         uint32_t RESERVED6[3];
    __IO uint32_t DAC0_REF_CTRL;        /* Select voltage references for DAC0 */
    __IO uint32_t DAC_CTRL;             /* DAC Control and Configuration */
    __IO uint32_t DAC0_DATA;            /* DAC0 converter input data */
    __IO uint32_t DAC1_DATA;            /* DAC1 converter input data */
    __IO uint32_t DAC2_DATA;            /* DAC2 converter input data */
    __O  uint32_t DAC01_DATA;           /* Input data for the DAC0 and DAC1 converters */
         uint32_t RESERVED7[2];
    __IO uint32_t RTC_CTRL;             /* Control the real-time clock behavior */
    __IO uint32_t RTC_COUNT;            /* Current count for the real-time clock */
    __IO uint32_t RTC_ALARM;            /* Alarm setting for the real-time clock */
         uint32_t RESERVED8[2];
    __IO uint32_t INTERRUPT_STATUS;     /* Status for the interrupts initiating in the analog domain */
    __IO uint32_t RETENTION_DATA;       /* General-purpose data retention register, used to maintain one data value while in sleep mode */
} AFE_Type;

#define AFE_BASE                        0x40000008
#define AFE                             ((AFE_Type *) AFE_BASE)

/* Power Supply Charge Pump and LCD Backlight Control */
/*   Configure the charge pump and LCD backlight */
#define AFE_PSU_CTRL_BASE               0x40000008

/* AFE_PSU_CTRL bit positions */
#define AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos 8
#define AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Mask ((uint32_t)(0x3FFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define AFE_PSU_CTRL_VCP_ENABLE_Pos     2
#define AFE_PSU_CTRL_LCDDRIVER_ENABLE_Pos 1
#define AFE_PSU_CTRL_VDBL_ENABLE_Pos    0

/* AFE_PSU_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t VDBL_ENABLE_ALIAS;    /* Select whether the VDBL is enabled */
    __IO uint32_t LCDDRIVER_ENABLE_ALIAS;/* Select whether the LCD driver is currently enabled */
    __IO uint32_t VCP_ENABLE_ALIAS;     /* Select whether the charge pump LDO is currently enabled */
} AFE_PSU_CTRL_Type;

#define AFE_PSU_CTRL                    ((AFE_PSU_CTRL_Type *) SYS_CALC_BITBAND(AFE_PSU_CTRL_BASE, 0))

/* AFE_PSU_CTRL settings */
#define VDBL_DISABLE_BITBAND            0x0
#define VDBL_ENABLE_BITBAND             0x1
#define VDBL_DISABLE                    ((uint32_t)(VDBL_DISABLE_BITBAND << AFE_PSU_CTRL_VDBL_ENABLE_Pos))
#define VDBL_ENABLE                     ((uint32_t)(VDBL_ENABLE_BITBAND << AFE_PSU_CTRL_VDBL_ENABLE_Pos))

#define LCDDRIVER_DISABLE_BITBAND       0x0
#define LCDDRIVER_ENABLE_BITBAND        0x1
#define LCDDRIVER_DISABLE               ((uint32_t)(LCDDRIVER_DISABLE_BITBAND << AFE_PSU_CTRL_LCDDRIVER_ENABLE_Pos))
#define LCDDRIVER_ENABLE                ((uint32_t)(LCDDRIVER_ENABLE_BITBAND << AFE_PSU_CTRL_LCDDRIVER_ENABLE_Pos))

#define VCP_DISABLE_BITBAND             0x0
#define VCP_ENABLE_BITBAND              0x1
#define VCP_DISABLE                     ((uint32_t)(VCP_DISABLE_BITBAND << AFE_PSU_CTRL_VCP_ENABLE_Pos))
#define VCP_ENABLE                      ((uint32_t)(VCP_ENABLE_BITBAND << AFE_PSU_CTRL_VCP_ENABLE_Pos))

#define LCDBACKLIGHT_DISABLE            ((uint32_t)(0x0U << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_1_INOM             ((uint32_t)(0x1U << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_2_INOM             ((uint32_t)(0x3U << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_3_INOM             ((uint32_t)(0x7U << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_4_INOM             ((uint32_t)(0xFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_5_INOM             ((uint32_t)(0x1FU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_6_INOM             ((uint32_t)(0x3FU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_7_INOM             ((uint32_t)(0x7FU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_8_INOM             ((uint32_t)(0xFFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_9_INOM             ((uint32_t)(0x1FFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))
#define LCDBACKLIGHT_10_INOM            ((uint32_t)(0x3FFU << AFE_PSU_CTRL_LCDBACKLIGHT_CFG_Pos))

/* Operation Control */
/*   Control the operating mode features */
#define AFE_OPMODE_CTRL_BASE            0x4000000C

/* AFE_OPMODE_CTRL bit positions */
#define AFE_OPMODE_CTRL_RC_OSC_ENABLE_Pos 3
#define AFE_OPMODE_CTRL_VADC_ENABLE_Pos 2
#define AFE_OPMODE_CTRL_STANDBY_MODE_Pos 1
#define AFE_OPMODE_CTRL_SLEEP_MODE_Pos  0

/* AFE_OPMODE_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t SLEEP_MODE_ALIAS;     /* Control sleep mode selection */
    __IO uint32_t STANDBY_MODE_ALIAS;   /* Control standby mode selection */
    __IO uint32_t VADC_ENABLE_ALIAS;    /* Select whether VADC is enabled */
    __IO uint32_t RC_OSC_ENABLE_ALIAS;  /* Select whether the RC oscillator is enabled */
} AFE_OPMODE_CTRL_Type;

#define AFE_OPMODE_CTRL                 ((AFE_OPMODE_CTRL_Type *) SYS_CALC_BITBAND(AFE_OPMODE_CTRL_BASE, 0))

/* AFE_OPMODE_CTRL settings */
#define SLEEP_MODE_DISABLE_BITBAND      0x0
#define SLEEP_MODE_ENABLE_BITBAND       0x1
#define SLEEP_MODE_DISABLE              ((uint32_t)(SLEEP_MODE_DISABLE_BITBAND << AFE_OPMODE_CTRL_SLEEP_MODE_Pos))
#define SLEEP_MODE_ENABLE               ((uint32_t)(SLEEP_MODE_ENABLE_BITBAND << AFE_OPMODE_CTRL_SLEEP_MODE_Pos))

#define STANDBY_MODE_DISABLE_BITBAND    0x0
#define STANDBY_MODE_ENABLE_BITBAND     0x1
#define STANDBY_MODE_DISABLE            ((uint32_t)(STANDBY_MODE_DISABLE_BITBAND << AFE_OPMODE_CTRL_STANDBY_MODE_Pos))
#define STANDBY_MODE_ENABLE             ((uint32_t)(STANDBY_MODE_ENABLE_BITBAND << AFE_OPMODE_CTRL_STANDBY_MODE_Pos))

#define VADC_DISABLE_BITBAND            0x0
#define VADC_ENABLE_BITBAND             0x1
#define VADC_DISABLE                    ((uint32_t)(VADC_DISABLE_BITBAND << AFE_OPMODE_CTRL_VADC_ENABLE_Pos))
#define VADC_ENABLE                     ((uint32_t)(VADC_ENABLE_BITBAND << AFE_OPMODE_CTRL_VADC_ENABLE_Pos))

#define RC_OSC_DISABLE_BITBAND          0x0
#define RC_OSC_ENABLE_BITBAND           0x1
#define RC_OSC_DISABLE                  ((uint32_t)(RC_OSC_DISABLE_BITBAND << AFE_OPMODE_CTRL_RC_OSC_ENABLE_Pos))
#define RC_OSC_ENABLE                   ((uint32_t)(RC_OSC_ENABLE_BITBAND << AFE_OPMODE_CTRL_RC_OSC_ENABLE_Pos))

/* 32 kHz Crystal Clock Calibration Control */
/*   Trimming of 32 kHz crystal frequency */
#define AFE_32K_CCR_CTRL_BASE           0x40000010

/* AFE_32K_CCR_CTRL bit positions */
#define AFE_32K_CCR_CTRL_CCR_CTRL_Pos   0
#define AFE_32K_CCR_CTRL_CCR_CTRL_Mask  ((uint32_t)(0xFFU << AFE_32K_CCR_CTRL_CCR_CTRL_Pos))

/* 48 MHz Crystal Clock Calibration Control */
/*   Trimming of 48 MHz crystal frequency */
#define AFE_48M_CCR_CTRL_BASE           0x40000018

/* AFE_48M_CCR_CTRL bit positions */
#define AFE_48M_CCR_CTRL_CCR_CTRL_Pos   0
#define AFE_48M_CCR_CTRL_CCR_CTRL_Mask  ((uint32_t)(0xFFU << AFE_48M_CCR_CTRL_CCR_CTRL_Pos))

/* RC Clock Calibration Control */
/*   Trimming of RC oscillator frequency */
#define AFE_RC_CCR_CTRL_BASE            0x40000020

/* AFE_RC_CCR_CTRL bit positions */
#define AFE_RC_CCR_CTRL_FINE_CTRL_Pos   16
#define AFE_RC_CCR_CTRL_FINE_CTRL_Mask  ((uint32_t)(0x3FU << AFE_RC_CCR_CTRL_FINE_CTRL_Pos))
#define AFE_RC_CCR_CTRL_RANGE_SEL_Pos   8
#define AFE_RC_CCR_CTRL_RANGE_SEL_Mask  ((uint32_t)(0x7U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define AFE_RC_CCR_CTRL_COARSE_CTRL_Pos 0
#define AFE_RC_CCR_CTRL_COARSE_CTRL_Mask ((uint32_t)(0x3FU << AFE_RC_CCR_CTRL_COARSE_CTRL_Pos))

/* AFE_RC_CCR_CTRL settings */
#define RC_OSC_FREQ_RANGE_7             ((uint32_t)(0x0U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_6             ((uint32_t)(0x1U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_5             ((uint32_t)(0x2U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_4             ((uint32_t)(0x3U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_3             ((uint32_t)(0x4U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_2             ((uint32_t)(0x5U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_1             ((uint32_t)(0x6U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))
#define RC_OSC_FREQ_RANGE_0             ((uint32_t)(0x7U << AFE_RC_CCR_CTRL_RANGE_SEL_Pos))

/* Input Control and Configuration */
/*   Configuration of the inputs to operational amplifier A0 and the ALT
 *   pads */
#define AFE_IN_SW_CTRL_BASE             0x40000030

/* AFE_IN_SW_CTRL bit positions */
#define AFE_IN_SW_CTRL_ALT1_SW_Pos      8
#define AFE_IN_SW_CTRL_ALT1_SW_Mask     ((uint32_t)(0xFU << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define AFE_IN_SW_CTRL_ALT0_SW_Pos      4
#define AFE_IN_SW_CTRL_ALT0_SW_Mask     ((uint32_t)(0x7U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define AFE_IN_SW_CTRL_A0_IN_CFG_Pos    0
#define AFE_IN_SW_CTRL_A0_IN_CFG_Mask   ((uint32_t)(0xFU << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))

/* AFE_IN_SW_CTRL settings */
#define A0_CFG_SEL_IN0                  ((uint32_t)(0x0U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN1                  ((uint32_t)(0x1U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN2                  ((uint32_t)(0x2U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN3                  ((uint32_t)(0x3U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN4                  ((uint32_t)(0x4U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN5                  ((uint32_t)(0x5U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN6                  ((uint32_t)(0x6U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_IN7                  ((uint32_t)(0x7U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))
#define A0_CFG_SEL_DISABLE              ((uint32_t)(0x8U << AFE_IN_SW_CTRL_A0_IN_CFG_Pos))

#define ALT0_SW_SEL_NONE                ((uint32_t)(0x0U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_IN0                 ((uint32_t)(0x1U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_IN1                 ((uint32_t)(0x2U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_IN2                 ((uint32_t)(0x3U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_M0_IN0              ((uint32_t)(0x4U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_M0_IN1              ((uint32_t)(0x5U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_SEL_M0_IN2              ((uint32_t)(0x6U << AFE_IN_SW_CTRL_ALT0_SW_Pos))
#define ALT0_SW_DISABLE                 ((uint32_t)(0x7U << AFE_IN_SW_CTRL_ALT0_SW_Pos))

#define ALT1_SW_SEL_NONE                ((uint32_t)(0x0U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_IN3                 ((uint32_t)(0x1U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_IN4                 ((uint32_t)(0x2U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_IN5                 ((uint32_t)(0x3U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_IN6                 ((uint32_t)(0x4U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_IN7                 ((uint32_t)(0x5U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_M0_IN3              ((uint32_t)(0x6U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_M0_IN4              ((uint32_t)(0x7U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_M0_IN5              ((uint32_t)(0x8U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_M0_IN6              ((uint32_t)(0x9U << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_SEL_M0_IN7              ((uint32_t)(0xAU << AFE_IN_SW_CTRL_ALT1_SW_Pos))
#define ALT1_SW_DISABLE                 ((uint32_t)(0xFU << AFE_IN_SW_CTRL_ALT1_SW_Pos))

/* Opamp Control */
/*   Enabling of the operational amplifiers A0 to A2 */
#define AFE_AMP_CTRL_BASE               0x40000034

/* AFE_AMP_CTRL bit positions */
#define AFE_AMP_CTRL_A2_ENABLE_Pos      2
#define AFE_AMP_CTRL_A1_ENABLE_Pos      1
#define AFE_AMP_CTRL_A0_ENABLE_Pos      0

/* AFE_AMP_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t A0_ENABLE_ALIAS;      /* Enable opamp A0 */
    __IO uint32_t A1_ENABLE_ALIAS;      /* Enable opamp A1 */
    __IO uint32_t A2_ENABLE_ALIAS;      /* Enable opamp A2 */
} AFE_AMP_CTRL_Type;

#define AFE_AMP_CTRL                    ((AFE_AMP_CTRL_Type *) SYS_CALC_BITBAND(AFE_AMP_CTRL_BASE, 0))

/* AFE_AMP_CTRL settings */
#define A0_DISABLE_BITBAND              0x0
#define A0_ENABLE_BITBAND               0x1
#define A0_DISABLE                      ((uint32_t)(A0_DISABLE_BITBAND << AFE_AMP_CTRL_A0_ENABLE_Pos))
#define A0_ENABLE                       ((uint32_t)(A0_ENABLE_BITBAND << AFE_AMP_CTRL_A0_ENABLE_Pos))

#define A1_DISABLE_BITBAND              0x0
#define A1_ENABLE_BITBAND               0x1
#define A1_DISABLE                      ((uint32_t)(A1_DISABLE_BITBAND << AFE_AMP_CTRL_A1_ENABLE_Pos))
#define A1_ENABLE                       ((uint32_t)(A1_ENABLE_BITBAND << AFE_AMP_CTRL_A1_ENABLE_Pos))

#define A2_DISABLE_BITBAND              0x0
#define A2_ENABLE_BITBAND               0x1
#define A2_DISABLE                      ((uint32_t)(A2_DISABLE_BITBAND << AFE_AMP_CTRL_A2_ENABLE_Pos))
#define A2_ENABLE                       ((uint32_t)(A2_ENABLE_BITBAND << AFE_AMP_CTRL_A2_ENABLE_Pos))

/* Output Control and Configuration */
/*   Output control and configuration for operational amplifiers A0 to A2 */
#define AFE_OUT_SW_CTRL_BASE            0x40000038

/* AFE_OUT_SW_CTRL bit positions */
#define AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_Pos 5
#define AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_Pos 4
#define AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_Pos 3
#define AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_Pos 2
#define AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_Pos 1
#define AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_Pos 0

/* AFE_OUT_SW_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t A0_OUTA_ENABLE_ALIAS; /* Opamp A0 A output connection */
    __IO uint32_t A0_OUTB_ENABLE_ALIAS; /* Opamp A0 B output connection */
    __IO uint32_t A1_OUTA_ENABLE_ALIAS; /* Opamp A1 A output connection */
    __IO uint32_t A1_OUTB_ENABLE_ALIAS; /* Opamp A1 B output connection */
    __IO uint32_t A2_OUTA_ENABLE_ALIAS; /* Opamp A2 A output connection */
    __IO uint32_t A2_OUTB_ENABLE_ALIAS; /* Opamp A2 B output connection */
} AFE_OUT_SW_CTRL_Type;

#define AFE_OUT_SW_CTRL                 ((AFE_OUT_SW_CTRL_Type *) SYS_CALC_BITBAND(AFE_OUT_SW_CTRL_BASE, 0))

/* AFE_OUT_SW_CTRL settings */
#define A0_OUTA_DISABLE_BITBAND         0x0
#define A0_OUTA_ENABLE_BITBAND          0x1
#define A0_OUTA_DISABLE                 ((uint32_t)(A0_OUTA_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_Pos))
#define A0_OUTA_ENABLE                  ((uint32_t)(A0_OUTA_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A0_OUTA_ENABLE_Pos))

#define A0_OUTB_DISABLE_BITBAND         0x0
#define A0_OUTB_ENABLE_BITBAND          0x1
#define A0_OUTB_DISABLE                 ((uint32_t)(A0_OUTB_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_Pos))
#define A0_OUTB_ENABLE                  ((uint32_t)(A0_OUTB_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A0_OUTB_ENABLE_Pos))

#define A1_OUTA_DISABLE_BITBAND         0x0
#define A1_OUTA_ENABLE_BITBAND          0x1
#define A1_OUTA_DISABLE                 ((uint32_t)(A1_OUTA_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_Pos))
#define A1_OUTA_ENABLE                  ((uint32_t)(A1_OUTA_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A1_OUTA_ENABLE_Pos))

#define A1_OUTB_DISABLE_BITBAND         0x0
#define A1_OUTB_ENABLE_BITBAND          0x1
#define A1_OUTB_DISABLE                 ((uint32_t)(A1_OUTB_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_Pos))
#define A1_OUTB_ENABLE                  ((uint32_t)(A1_OUTB_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A1_OUTB_ENABLE_Pos))

#define A2_OUTA_DISABLE_BITBAND         0x0
#define A2_OUTA_ENABLE_BITBAND          0x1
#define A2_OUTA_DISABLE                 ((uint32_t)(A2_OUTA_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_Pos))
#define A2_OUTA_ENABLE                  ((uint32_t)(A2_OUTA_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A2_OUTA_ENABLE_Pos))

#define A2_OUTB_DISABLE_BITBAND         0x0
#define A2_OUTB_ENABLE_BITBAND          0x1
#define A2_OUTB_DISABLE                 ((uint32_t)(A2_OUTB_DISABLE_BITBAND << AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_Pos))
#define A2_OUTB_ENABLE                  ((uint32_t)(A2_OUTB_ENABLE_BITBAND << AFE_OUT_SW_CTRL_A2_OUTB_ENABLE_Pos))

/* SPST Switch Control */
/*   Control of SPST switches */
#define AFE_SPST_CTRL_BASE              0x40000040

/* AFE_SPST_CTRL bit positions */
#define AFE_SPST_CTRL_SPST3_DISABLE_Pos 7
#define AFE_SPST_CTRL_SPST3_SELECT_Pos  6
#define AFE_SPST_CTRL_SPST2_DISABLE_Pos 5
#define AFE_SPST_CTRL_SPST2_SELECT_Pos  4
#define AFE_SPST_CTRL_SPST1_DISABLE_Pos 3
#define AFE_SPST_CTRL_SPST1_SELECT_Pos  2
#define AFE_SPST_CTRL_SPST0_DISABLE_Pos 1
#define AFE_SPST_CTRL_SPST0_SELECT_Pos  0

/* AFE_SPST_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t SPST0_SELECT_ALIAS;   /* Select whether SPST switch 0 is open or closed */
    __IO uint32_t SPST0_DISABLE_ALIAS;  /* Disable SPST switch 0 */
    __IO uint32_t SPST1_SELECT_ALIAS;   /* Select whether SPST switch 1 is open or closed */
    __IO uint32_t SPST1_DISABLE_ALIAS;  /* Disable SPST switch 1 */
    __IO uint32_t SPST2_SELECT_ALIAS;   /* Select whether SPST switch 2 is open or closed */
    __IO uint32_t SPST2_DISABLE_ALIAS;  /* Disable SPST switch 2 */
    __IO uint32_t SPST3_SELECT_ALIAS;   /* Select whether SPST switch 3 is open or closed */
    __IO uint32_t SPST3_DISABLE_ALIAS;  /* Disable SPST switch 3 */
} AFE_SPST_CTRL_Type;

#define AFE_SPST_CTRL                   ((AFE_SPST_CTRL_Type *) SYS_CALC_BITBAND(AFE_SPST_CTRL_BASE, 0))

/* AFE_SPST_CTRL settings */
#define SPST0_SEL_OPEN_BITBAND          0x0
#define SPST0_SEL_CLOSE_BITBAND         0x1
#define SPST0_SEL_OPEN                  ((uint32_t)(SPST0_SEL_OPEN_BITBAND << AFE_SPST_CTRL_SPST0_SELECT_Pos))
#define SPST0_SEL_CLOSE                 ((uint32_t)(SPST0_SEL_CLOSE_BITBAND << AFE_SPST_CTRL_SPST0_SELECT_Pos))

#define SPST0_ENABLE_BITBAND            0x0
#define SPST0_DISABLE_BITBAND           0x1
#define SPST0_ENABLE                    ((uint32_t)(SPST0_ENABLE_BITBAND << AFE_SPST_CTRL_SPST0_DISABLE_Pos))
#define SPST0_DISABLE                   ((uint32_t)(SPST0_DISABLE_BITBAND << AFE_SPST_CTRL_SPST0_DISABLE_Pos))

#define SPST1_SEL_OPEN_BITBAND          0x0
#define SPST1_SEL_CLOSE_BITBAND         0x1
#define SPST1_SEL_OPEN                  ((uint32_t)(SPST1_SEL_OPEN_BITBAND << AFE_SPST_CTRL_SPST1_SELECT_Pos))
#define SPST1_SEL_CLOSE                 ((uint32_t)(SPST1_SEL_CLOSE_BITBAND << AFE_SPST_CTRL_SPST1_SELECT_Pos))

#define SPST1_ENABLE_BITBAND            0x0
#define SPST1_DISABLE_BITBAND           0x1
#define SPST1_ENABLE                    ((uint32_t)(SPST1_ENABLE_BITBAND << AFE_SPST_CTRL_SPST1_DISABLE_Pos))
#define SPST1_DISABLE                   ((uint32_t)(SPST1_DISABLE_BITBAND << AFE_SPST_CTRL_SPST1_DISABLE_Pos))

#define SPST2_SEL_OPEN_BITBAND          0x0
#define SPST2_SEL_CLOSE_BITBAND         0x1
#define SPST2_SEL_OPEN                  ((uint32_t)(SPST2_SEL_OPEN_BITBAND << AFE_SPST_CTRL_SPST2_SELECT_Pos))
#define SPST2_SEL_CLOSE                 ((uint32_t)(SPST2_SEL_CLOSE_BITBAND << AFE_SPST_CTRL_SPST2_SELECT_Pos))

#define SPST2_ENABLE_BITBAND            0x0
#define SPST2_DISABLE_BITBAND           0x1
#define SPST2_ENABLE                    ((uint32_t)(SPST2_ENABLE_BITBAND << AFE_SPST_CTRL_SPST2_DISABLE_Pos))
#define SPST2_DISABLE                   ((uint32_t)(SPST2_DISABLE_BITBAND << AFE_SPST_CTRL_SPST2_DISABLE_Pos))

#define SPST3_SEL_OPEN_BITBAND          0x0
#define SPST3_SEL_CLOSE_BITBAND         0x1
#define SPST3_SEL_OPEN                  ((uint32_t)(SPST3_SEL_OPEN_BITBAND << AFE_SPST_CTRL_SPST3_SELECT_Pos))
#define SPST3_SEL_CLOSE                 ((uint32_t)(SPST3_SEL_CLOSE_BITBAND << AFE_SPST_CTRL_SPST3_SELECT_Pos))

#define SPST3_ENABLE_BITBAND            0x0
#define SPST3_DISABLE_BITBAND           0x1
#define SPST3_ENABLE                    ((uint32_t)(SPST3_ENABLE_BITBAND << AFE_SPST_CTRL_SPST3_DISABLE_Pos))
#define SPST3_DISABLE                   ((uint32_t)(SPST3_DISABLE_BITBAND << AFE_SPST_CTRL_SPST3_DISABLE_Pos))

/* Multi-Switch Control */
/*   Multi-Switch control and configuration */
#define AFE_MSW_CTRL_BASE               0x40000044

/* AFE_MSW_CTRL bit positions */
#define AFE_MSW_CTRL_MSW3_SELECT_Pos    11
#define AFE_MSW_CTRL_MSW3_SELECT_Mask   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW3_SELECT_Pos))
#define AFE_MSW_CTRL_MSW2_SELECT_Pos    8
#define AFE_MSW_CTRL_MSW2_SELECT_Mask   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW2_SELECT_Pos))
#define AFE_MSW_CTRL_MSW1_SELECT_Pos    3
#define AFE_MSW_CTRL_MSW1_SELECT_Mask   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW1_SELECT_Pos))
#define AFE_MSW_CTRL_MSW0_SELECT_Pos    0
#define AFE_MSW_CTRL_MSW0_SELECT_Mask   ((uint32_t)(0x7U << AFE_MSW_CTRL_MSW0_SELECT_Pos))

/* AFE_MSW_CTRL settings */
#define MSW0_SEL_A                      ((uint32_t)(0x0U << AFE_MSW_CTRL_MSW0_SELECT_Pos))
#define MSW0_SEL_B                      ((uint32_t)(0x1U << AFE_MSW_CTRL_MSW0_SELECT_Pos))
#define MSW0_SEL_A_B                    ((uint32_t)(0x2U << AFE_MSW_CTRL_MSW0_SELECT_Pos))
#define MSW0_SEL_DISABLE                ((uint32_t)(0x3U << AFE_MSW_CTRL_MSW0_SELECT_Pos))
#define MSW0_SEL_PWM0                   ((uint32_t)(0x4U << AFE_MSW_CTRL_MSW0_SELECT_Pos))

#define MSW1_SEL_A                      ((uint32_t)(0x0U << AFE_MSW_CTRL_MSW1_SELECT_Pos))
#define MSW1_SEL_B                      ((uint32_t)(0x1U << AFE_MSW_CTRL_MSW1_SELECT_Pos))
#define MSW1_SEL_A_B                    ((uint32_t)(0x2U << AFE_MSW_CTRL_MSW1_SELECT_Pos))
#define MSW1_SEL_DISABLE                ((uint32_t)(0x3U << AFE_MSW_CTRL_MSW1_SELECT_Pos))
#define MSW1_SEL_PWM1                   ((uint32_t)(0x4U << AFE_MSW_CTRL_MSW1_SELECT_Pos))

#define MSW2_SEL_A                      ((uint32_t)(0x0U << AFE_MSW_CTRL_MSW2_SELECT_Pos))
#define MSW2_SEL_B                      ((uint32_t)(0x1U << AFE_MSW_CTRL_MSW2_SELECT_Pos))
#define MSW2_SEL_A_B                    ((uint32_t)(0x2U << AFE_MSW_CTRL_MSW2_SELECT_Pos))
#define MSW2_SEL_DISABLE                ((uint32_t)(0x3U << AFE_MSW_CTRL_MSW2_SELECT_Pos))
#define MSW2_SEL_PWM2                   ((uint32_t)(0x4U << AFE_MSW_CTRL_MSW2_SELECT_Pos))

#define MSW3_SEL_A                      ((uint32_t)(0x0U << AFE_MSW_CTRL_MSW3_SELECT_Pos))
#define MSW3_SEL_B                      ((uint32_t)(0x1U << AFE_MSW_CTRL_MSW3_SELECT_Pos))
#define MSW3_SEL_A_B                    ((uint32_t)(0x2U << AFE_MSW_CTRL_MSW3_SELECT_Pos))
#define MSW3_SEL_DISABLE                ((uint32_t)(0x3U << AFE_MSW_CTRL_MSW3_SELECT_Pos))
#define MSW3_SEL_PWM3                   ((uint32_t)(0x4U << AFE_MSW_CTRL_MSW3_SELECT_Pos))

/* Programmable Gain Amplifier 0 Control */
/*   Configure PGA0 and select the inputs for this programmable gain
 *   amplifier */
#define AFE_PGA0_CTRL_BASE              0x40000050

/* AFE_PGA0_CTRL bit positions */
#define AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos 5
#define AFE_PGA0_CTRL_PGA0_DIF_SELECT_Mask ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define AFE_PGA0_CTRL_PGA0_SELECT_Pos   1
#define AFE_PGA0_CTRL_PGA0_SELECT_Mask  ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define AFE_PGA0_CTRL_PGA0_ENABLE_Pos   0

/* AFE_PGA0_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t PGA0_ENABLE_ALIAS;    /* Enable PGA0 amplifier */
} AFE_PGA0_CTRL_Type;

#define AFE_PGA0_CTRL                   ((AFE_PGA0_CTRL_Type *) SYS_CALC_BITBAND(AFE_PGA0_CTRL_BASE, 0))

/* AFE_PGA0_CTRL settings */
#define PGA0_DISABLE_BITBAND            0x0
#define PGA0_ENABLE_BITBAND             0x1
#define PGA0_DISABLE                    ((uint32_t)(PGA0_DISABLE_BITBAND << AFE_PGA0_CTRL_PGA0_ENABLE_Pos))
#define PGA0_ENABLE                     ((uint32_t)(PGA0_ENABLE_BITBAND << AFE_PGA0_CTRL_PGA0_ENABLE_Pos))

#define PGA0_SEL_VSS                    ((uint32_t)(0x0U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VBAT_2                 ((uint32_t)(0x1U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VBATA_2                ((uint32_t)(0x2U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VREF                   ((uint32_t)(0x3U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_OUT0                   ((uint32_t)(0x4U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_OUT1                   ((uint32_t)(0x5U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_OUT2                   ((uint32_t)(0x6U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_A0REF                  ((uint32_t)(0x7U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_A1REF                  ((uint32_t)(0x8U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_A2REF                  ((uint32_t)(0x9U << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VWAKEUP                ((uint32_t)(0xAU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VTS1                   ((uint32_t)(0xBU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_AUX_IN2                ((uint32_t)(0xCU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_AUX_IN1                ((uint32_t)(0xDU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_AUX_IN0                ((uint32_t)(0xEU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))
#define PGA0_SEL_VADC_2                 ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_SELECT_Pos))

#define PGA0_DIF_SEL_VSS                ((uint32_t)(0x0U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VBAT_2             ((uint32_t)(0x1U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VBATA_2            ((uint32_t)(0x2U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VREF               ((uint32_t)(0x3U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_OUT0               ((uint32_t)(0x4U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_OUT1               ((uint32_t)(0x5U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_OUT2               ((uint32_t)(0x6U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_A0REF              ((uint32_t)(0x7U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_A1REF              ((uint32_t)(0x8U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_A2REF              ((uint32_t)(0x9U << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VWAKEUP            ((uint32_t)(0xAU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VTS2               ((uint32_t)(0xBU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_AUX_IN1            ((uint32_t)(0xDU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_AUX_IN0            ((uint32_t)(0xEU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))
#define PGA0_DIF_SEL_VADC_2             ((uint32_t)(0xFU << AFE_PGA0_CTRL_PGA0_DIF_SELECT_Pos))

/* Programmable Gain Amplifier 1 Control */
/*   Configure PGA1 and select the inputs for this programmable gain
 *   amplifier */
#define AFE_PGA1_CTRL_BASE              0x40000054

/* AFE_PGA1_CTRL bit positions */
#define AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos 5
#define AFE_PGA1_CTRL_PGA1_DIF_SELECT_Mask ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define AFE_PGA1_CTRL_PGA1_SELECT_Pos   1
#define AFE_PGA1_CTRL_PGA1_SELECT_Mask  ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define AFE_PGA1_CTRL_PGA1_ENABLE_Pos   0

/* AFE_PGA1_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t PGA1_ENABLE_ALIAS;    /* Enable PGA1 amplifier */
} AFE_PGA1_CTRL_Type;

#define AFE_PGA1_CTRL                   ((AFE_PGA1_CTRL_Type *) SYS_CALC_BITBAND(AFE_PGA1_CTRL_BASE, 0))

/* AFE_PGA1_CTRL settings */
#define PGA1_DISABLE_BITBAND            0x0
#define PGA1_ENABLE_BITBAND             0x1
#define PGA1_DISABLE                    ((uint32_t)(PGA1_DISABLE_BITBAND << AFE_PGA1_CTRL_PGA1_ENABLE_Pos))
#define PGA1_ENABLE                     ((uint32_t)(PGA1_ENABLE_BITBAND << AFE_PGA1_CTRL_PGA1_ENABLE_Pos))

#define PGA1_SEL_VSS                    ((uint32_t)(0x0U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VBAT_2                 ((uint32_t)(0x1U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VBATA_2                ((uint32_t)(0x2U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VREF                   ((uint32_t)(0x3U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_OUT0                   ((uint32_t)(0x4U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_OUT1                   ((uint32_t)(0x5U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_OUT2                   ((uint32_t)(0x6U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_A0REF                  ((uint32_t)(0x7U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_A1REF                  ((uint32_t)(0x8U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_A2REF                  ((uint32_t)(0x9U << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VWAKEUP                ((uint32_t)(0xAU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VTS1                   ((uint32_t)(0xBU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_AUX_IN2                ((uint32_t)(0xCU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_AUX_IN1                ((uint32_t)(0xDU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_AUX_IN0                ((uint32_t)(0xEU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))
#define PGA1_SEL_VADC_2                 ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_SELECT_Pos))

#define PGA1_DIF_SEL_VSS                ((uint32_t)(0x0U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VBAT_2             ((uint32_t)(0x1U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VBATA_2            ((uint32_t)(0x2U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VREF               ((uint32_t)(0x3U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_OUT0               ((uint32_t)(0x4U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_OUT1               ((uint32_t)(0x5U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_OUT2               ((uint32_t)(0x6U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_A0REF              ((uint32_t)(0x7U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_A1REF              ((uint32_t)(0x8U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_A2REF              ((uint32_t)(0x9U << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VWAKEUP            ((uint32_t)(0xAU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VTS2               ((uint32_t)(0xBU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_AUX_IN1            ((uint32_t)(0xDU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_AUX_IN0            ((uint32_t)(0xEU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))
#define PGA1_DIF_SEL_VADC_2             ((uint32_t)(0xFU << AFE_PGA1_CTRL_PGA1_DIF_SELECT_Pos))

/* Programmable Gain Amplifier Configuration and Control */
/*   Configure the cut-off frequency and gain settings for PGA0, PGA1 */
#define AFE_PGA_GAIN_CTRL_BASE          0x40000058

/* AFE_PGA_GAIN_CTRL bit positions */
#define AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Pos 6
#define AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Mask ((uint32_t)(0x3U << AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Pos))
#define AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos 3
#define AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Mask ((uint32_t)(0x7U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos 0
#define AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Mask ((uint32_t)(0x7U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))

/* AFE_PGA_GAIN_CTRL settings */
#define PGA0_GAIN_0DB                   ((uint32_t)(0x0U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_6DB                   ((uint32_t)(0x1U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_12DB                  ((uint32_t)(0x2U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_18DB                  ((uint32_t)(0x3U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_24DB                  ((uint32_t)(0x4U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_30DB                  ((uint32_t)(0x5U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_36DB                  ((uint32_t)(0x6U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))
#define PGA0_GAIN_42DB                  ((uint32_t)(0x7U << AFE_PGA_GAIN_CTRL_PGA0_GAIN_CFG_Pos))

#define PGA1_GAIN_0DB                   ((uint32_t)(0x0U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_6DB                   ((uint32_t)(0x1U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_12DB                  ((uint32_t)(0x2U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_18DB                  ((uint32_t)(0x3U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_24DB                  ((uint32_t)(0x4U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_30DB                  ((uint32_t)(0x5U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))
#define PGA1_GAIN_36DB                  ((uint32_t)(0x6U << AFE_PGA_GAIN_CTRL_PGA1_GAIN_CFG_Pos))

#define PGA_CUT_OFF_LOW                 ((uint32_t)(0x1U << AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Pos))
#define PGA_CUT_OFF_HIGH                ((uint32_t)(0x0U << AFE_PGA_GAIN_CTRL_CUT_OFF_CFG_Pos))

/* Voltage Threshold Comparator Control */
/*   Configure and control the voltage comparator */
#define AFE_THRESHOLD_COMPARE_CTRL_BASE 0x4000005C

/* AFE_THRESHOLD_COMPARE_CTRL bit positions */
#define AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos 0
#define AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Mask ((uint32_t)(0x3U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos))

/* AFE_THRESHOLD_COMPARE_CTRL settings */
#define THRESHOLD_COMPARE_DISABLE       ((uint32_t)(0x0U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos))
#define THRESHOLD_COMPARE_40MV          ((uint32_t)(0x1U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos))
#define THRESHOLD_COMPARE_80MV          ((uint32_t)(0x2U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos))
#define THRESHOLD_COMPARE_120MV         ((uint32_t)(0x3U << AFE_THRESHOLD_COMPARE_CTRL_THRESHOLD_Pos))

/* ADC Configuration and Control */
/*   Configure and control the analog to digital convertors */
#define AFE_ADC_CTRL_BASE               0x40000060

/* AFE_ADC_CTRL bit positions */
#define AFE_ADC_CTRL_ADC1_GAIN_ENABLE_Pos 7
#define AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_Pos 6
#define AFE_ADC_CTRL_ADC0_GAIN_ENABLE_Pos 5
#define AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_Pos 4
#define AFE_ADC_CTRL_ADC1_FORMAT_CFG_Pos 3
#define AFE_ADC_CTRL_ADC0_FORMAT_CFG_Pos 2
#define AFE_ADC_CTRL_ADC1_ENABLE_Pos    1
#define AFE_ADC_CTRL_ADC0_ENABLE_Pos    0

/* AFE_ADC_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t ADC_CTRL_BYTE;        
         uint8_t RESERVED0[3];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000060, 0) - (0x40000060 + 4)];
    __IO uint32_t ADC0_ENABLE_ALIAS;    /* Enable ADC0 */
    __IO uint32_t ADC1_ENABLE_ALIAS;    /* Enable ADC1 */
    __IO uint32_t ADC0_FORMAT_CFG_ALIAS;/* ADC output number format */
    __IO uint32_t ADC1_FORMAT_CFG_ALIAS;/* ADC output number format */
    __IO uint32_t ADC0_OFFSET_ENABLE_ALIAS;/* Enable offset correction for ADC0 */
    __IO uint32_t ADC0_GAIN_ENABLE_ALIAS;/* Enable the gain error correction multiplier for ADC0 */
    __IO uint32_t ADC1_OFFSET_ENABLE_ALIAS;/* Enable offset correction for ADC1 */
    __IO uint32_t ADC1_GAIN_ENABLE_ALIAS;/* Enable the gain error correction multiplier for ADC1 */
} AFE_ADC_CTRL_Type;

#define AFE_ADC_CTRL                    ((AFE_ADC_CTRL_Type *) AFE_ADC_CTRL_BASE)

/* AFE_ADC_CTRL settings */
#define ADC0_DISABLE_BITBAND            0x0
#define ADC0_ENABLE_BITBAND             0x1
#define ADC0_DISABLE                    ((uint32_t)(ADC0_DISABLE_BITBAND << AFE_ADC_CTRL_ADC0_ENABLE_Pos))
#define ADC0_ENABLE                     ((uint32_t)(ADC0_ENABLE_BITBAND << AFE_ADC_CTRL_ADC0_ENABLE_Pos))

#define ADC1_DISABLE_BITBAND            0x0
#define ADC1_ENABLE_BITBAND             0x1
#define ADC1_DISABLE                    ((uint32_t)(ADC1_DISABLE_BITBAND << AFE_ADC_CTRL_ADC1_ENABLE_Pos))
#define ADC1_ENABLE                     ((uint32_t)(ADC1_ENABLE_BITBAND << AFE_ADC_CTRL_ADC1_ENABLE_Pos))

#define ADC0_FORMAT_TWOS_COMP_BITBAND   0x0
#define ADC0_FORMAT_UNSIGNED_INT_BITBAND 0x1
#define ADC0_FORMAT_TWOS_COMP           ((uint32_t)(ADC0_FORMAT_TWOS_COMP_BITBAND << AFE_ADC_CTRL_ADC0_FORMAT_CFG_Pos))
#define ADC0_FORMAT_UNSIGNED_INT        ((uint32_t)(ADC0_FORMAT_UNSIGNED_INT_BITBAND << AFE_ADC_CTRL_ADC0_FORMAT_CFG_Pos))

#define ADC1_FORMAT_TWOS_COMP_BITBAND   0x0
#define ADC1_FORMAT_UNSIGNED_INT_BITBAND 0x1
#define ADC1_FORMAT_TWOS_COMP           ((uint32_t)(ADC1_FORMAT_TWOS_COMP_BITBAND << AFE_ADC_CTRL_ADC1_FORMAT_CFG_Pos))
#define ADC1_FORMAT_UNSIGNED_INT        ((uint32_t)(ADC1_FORMAT_UNSIGNED_INT_BITBAND << AFE_ADC_CTRL_ADC1_FORMAT_CFG_Pos))

#define ADC0_OFFSET_DISABLE_BITBAND     0x0
#define ADC0_OFFSET_ENABLE_BITBAND      0x1
#define ADC0_OFFSET_DISABLE             ((uint32_t)(ADC0_OFFSET_DISABLE_BITBAND << AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_Pos))
#define ADC0_OFFSET_ENABLE              ((uint32_t)(ADC0_OFFSET_ENABLE_BITBAND << AFE_ADC_CTRL_ADC0_OFFSET_ENABLE_Pos))

#define ADC0_GAIN_DISABLE_BITBAND       0x0
#define ADC0_GAIN_ENABLE_BITBAND        0x1
#define ADC0_GAIN_DISABLE               ((uint32_t)(ADC0_GAIN_DISABLE_BITBAND << AFE_ADC_CTRL_ADC0_GAIN_ENABLE_Pos))
#define ADC0_GAIN_ENABLE                ((uint32_t)(ADC0_GAIN_ENABLE_BITBAND << AFE_ADC_CTRL_ADC0_GAIN_ENABLE_Pos))

#define ADC1_OFFSET_DISABLE_BITBAND     0x0
#define ADC1_OFFSET_ENABLE_BITBAND      0x1
#define ADC1_OFFSET_DISABLE             ((uint32_t)(ADC1_OFFSET_DISABLE_BITBAND << AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_Pos))
#define ADC1_OFFSET_ENABLE              ((uint32_t)(ADC1_OFFSET_ENABLE_BITBAND << AFE_ADC_CTRL_ADC1_OFFSET_ENABLE_Pos))

#define ADC1_GAIN_DISABLE_BITBAND       0x0
#define ADC1_GAIN_ENABLE_BITBAND        0x1
#define ADC1_GAIN_DISABLE               ((uint32_t)(ADC1_GAIN_DISABLE_BITBAND << AFE_ADC_CTRL_ADC1_GAIN_ENABLE_Pos))
#define ADC1_GAIN_ENABLE                ((uint32_t)(ADC1_GAIN_ENABLE_BITBAND << AFE_ADC_CTRL_ADC1_GAIN_ENABLE_Pos))

/* AFE_ADC_CTRL sub-register bit positions */
#define ADC_CTRL_BYTE_ADC0_ENABLE_Pos   0
#define ADC_CTRL_BYTE_ADC1_ENABLE_Pos   1
#define ADC_CTRL_BYTE_ADC0_FORMAT_CFG_Pos 2
#define ADC_CTRL_BYTE_ADC1_FORMAT_CFG_Pos 3
#define ADC_CTRL_BYTE_ADC0_OFFSET_ENABLE_Pos 4
#define ADC_CTRL_BYTE_ADC0_GAIN_ENABLE_Pos 5
#define ADC_CTRL_BYTE_ADC1_OFFSET_ENABLE_Pos 6
#define ADC_CTRL_BYTE_ADC1_GAIN_ENABLE_Pos 7

/* AFE_ADC_CTRL subregister settings */
#define ADC0_DISABLE_BYTE               ((uint8_t)(ADC0_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC0_ENABLE_Pos))
#define ADC0_ENABLE_BYTE                ((uint8_t)(ADC0_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC0_ENABLE_Pos))

#define ADC1_DISABLE_BYTE               ((uint8_t)(ADC1_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC1_ENABLE_Pos))
#define ADC1_ENABLE_BYTE                ((uint8_t)(ADC1_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC1_ENABLE_Pos))

#define ADC0_FORMAT_TWOS_COMP_BYTE      ((uint8_t)(ADC0_FORMAT_TWOS_COMP_BITBAND << ADC_CTRL_BYTE_ADC0_FORMAT_CFG_Pos))
#define ADC0_FORMAT_UNSIGNED_INT_BYTE   ((uint8_t)(ADC0_FORMAT_UNSIGNED_INT_BITBAND << ADC_CTRL_BYTE_ADC0_FORMAT_CFG_Pos))

#define ADC1_FORMAT_TWOS_COMP_BYTE      ((uint8_t)(ADC1_FORMAT_TWOS_COMP_BITBAND << ADC_CTRL_BYTE_ADC1_FORMAT_CFG_Pos))
#define ADC1_FORMAT_UNSIGNED_INT_BYTE   ((uint8_t)(ADC1_FORMAT_UNSIGNED_INT_BITBAND << ADC_CTRL_BYTE_ADC1_FORMAT_CFG_Pos))

#define ADC0_OFFSET_DISABLE_BYTE        ((uint8_t)(ADC0_OFFSET_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC0_OFFSET_ENABLE_Pos))
#define ADC0_OFFSET_ENABLE_BYTE         ((uint8_t)(ADC0_OFFSET_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC0_OFFSET_ENABLE_Pos))

#define ADC0_GAIN_DISABLE_BYTE          ((uint8_t)(ADC0_GAIN_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC0_GAIN_ENABLE_Pos))
#define ADC0_GAIN_ENABLE_BYTE           ((uint8_t)(ADC0_GAIN_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC0_GAIN_ENABLE_Pos))

#define ADC1_OFFSET_DISABLE_BYTE        ((uint8_t)(ADC1_OFFSET_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC1_OFFSET_ENABLE_Pos))
#define ADC1_OFFSET_ENABLE_BYTE         ((uint8_t)(ADC1_OFFSET_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC1_OFFSET_ENABLE_Pos))

#define ADC1_GAIN_DISABLE_BYTE          ((uint8_t)(ADC1_GAIN_DISABLE_BITBAND << ADC_CTRL_BYTE_ADC1_GAIN_ENABLE_Pos))
#define ADC1_GAIN_ENABLE_BYTE           ((uint8_t)(ADC1_GAIN_ENABLE_BITBAND << ADC_CTRL_BYTE_ADC1_GAIN_ENABLE_Pos))

/* ADC0 Output */
/*   ADC0 converter output */
#define AFE_ADC0_DATA_BASE              0x40000064

/* AFE_ADC0_DATA bit positions */
#define AFE_ADC0_DATA_ALIGN_Pos         16
#define AFE_ADC0_DATA_ALIGN_Mask        ((uint32_t)(0xFFFFU << AFE_ADC0_DATA_ALIGN_Pos))

/* ADC1 Output */
/*   ADC1 converter output */
#define AFE_ADC1_DATA_BASE              0x40000068

/* AFE_ADC1_DATA bit positions */
#define AFE_ADC1_DATA_ALIGN_Pos         16
#define AFE_ADC1_DATA_ALIGN_Mask        ((uint32_t)(0xFFFFU << AFE_ADC1_DATA_ALIGN_Pos))

/* ADC 0 and 1 Shared Output */
/*   16 MSBs of the ADC0 and ADC1 converter output */
#define AFE_ADC01_DATA_BASE             0x4000006C

/* AFE_ADC01_DATA bit positions */
#define AFE_ADC01_DATA_ADC0_Pos         16
#define AFE_ADC01_DATA_ADC0_Mask        ((uint32_t)(0xFFFFU << AFE_ADC01_DATA_ADC0_Pos))
#define AFE_ADC01_DATA_ADC1_Pos         0
#define AFE_ADC01_DATA_ADC1_Mask        ((uint32_t)(0xFFFFU << AFE_ADC01_DATA_ADC1_Pos))

/* AFE_ADC01_DATA sub-register and bit-band aliases */
typedef struct
{
    __I  uint16_t ADC1_DATA_SHORT;      
    __I  uint16_t ADC0_DATA_SHORT;      
} AFE_ADC01_DATA_Type;

#define AFE_ADC01_DATA                  ((AFE_ADC01_DATA_Type *) AFE_ADC01_DATA_BASE)

/* ADC0 Offset Calibration */
/*   ADC0 offset calibration */
#define AFE_ADC0_OFFSET_BASE            0x40000070

/* AFE_ADC0_OFFSET bit positions */
#define AFE_ADC0_OFFSET_CAL_Pos         14
#define AFE_ADC0_OFFSET_CAL_Mask        ((uint32_t)(0x3FFFFU << AFE_ADC0_OFFSET_CAL_Pos))

/* ADC1  Offset Calibration */
/*   ADC1 offset calibration */
#define AFE_ADC1_OFFSET_BASE            0x40000074

/* AFE_ADC1_OFFSET bit positions */
#define AFE_ADC1_OFFSET_CAL_Pos         14
#define AFE_ADC1_OFFSET_CAL_Mask        ((uint32_t)(0x3FFFFU << AFE_ADC1_OFFSET_CAL_Pos))

/* ADC0 Gain Calibration */
/*   ADC0 gain calibration */
#define AFE_ADC0_GAIN_BASE              0x40000078

/* AFE_ADC0_GAIN bit positions */
#define AFE_ADC0_GAIN_CAL_Pos           14
#define AFE_ADC0_GAIN_CAL_Mask          ((uint32_t)(0x3FFFFU << AFE_ADC0_GAIN_CAL_Pos))

/* ADC1  Gain Calibration */
/*   ADC1 gain calibration */
#define AFE_ADC1_GAIN_BASE              0x4000007C

/* AFE_ADC1_GAIN bit positions */
#define AFE_ADC1_GAIN_CAL_Pos           14
#define AFE_ADC1_GAIN_CAL_Mask          ((uint32_t)(0x3FFFFU << AFE_ADC1_GAIN_CAL_Pos))

/* ADC Data Rate Configuration */
/*   Configure the ADC data rate including the decimation factor */
#define AFE_DATARATE_CFG_BASE           0x40000080

/* AFE_DATARATE_CFG bit positions */
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_Pos 25
#define AFE_DATARATE_CFG_DECIMATION_FACTOR_Mask ((uint32_t)(0x3FU << AFE_DATARATE_CFG_DECIMATION_FACTOR_Pos))
#define AFE_DATARATE_CFG_DECIMATION_ENABLE_Pos 24
#define AFE_DATARATE_CFG_ADJUST_Pos     0
#define AFE_DATARATE_CFG_ADJUST_Mask    ((uint32_t)(0x3FFU << AFE_DATARATE_CFG_ADJUST_Pos))

/* AFE_DATARATE_CFG sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t DATARATE_ADJUST_SHORT;
         uint8_t RESERVED0[1];
    __IO uint8_t DECIMATION_FACTOR_BYTE;
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000080, 0) - (0x40000080 + 4)];
         uint32_t RESERVED1[24];
    __IO uint32_t DECIMATION_ENABLE_ALIAS;/* Enable the decimation filter */
} AFE_DATARATE_CFG_Type;

#define AFE_DATARATE_CFG                ((AFE_DATARATE_CFG_Type *) AFE_DATARATE_CFG_BASE)

/* AFE_DATARATE_CFG settings */
#define DATARATE_DECIMATE_ENABLE_BITBAND 0x1
#define DATARATE_DECIMATE_DISABLE_BITBAND 0x0
#define DATARATE_DECIMATE_ENABLE        ((uint32_t)(DATARATE_DECIMATE_ENABLE_BITBAND << AFE_DATARATE_CFG_DECIMATION_ENABLE_Pos))
#define DATARATE_DECIMATE_DISABLE       ((uint32_t)(DATARATE_DECIMATE_DISABLE_BITBAND << AFE_DATARATE_CFG_DECIMATION_ENABLE_Pos))

/* AFE_DATARATE_CFG sub-register bit positions */
#define DATARATE_ADJUST_SHORT_ADJUST_Pos 0
#define DATARATE_ADJUST_SHORT_ADJUST_Mask ((uint32_t)(0x3FFU << DATARATE_ADJUST_SHORT_ADJUST_Pos))
#define DECIMATION_FACTOR_BYTE_DECIMATION_ENABLE_Pos 0
#define DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_Pos 1
#define DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_Mask ((uint32_t)(0x3FU << DECIMATION_FACTOR_BYTE_DECIMATION_FACTOR_Pos))

/* AFE_DATARATE_CFG subregister settings */
#define DATARATE_DECIMATE_ENABLE_BYTE   ((uint8_t)(DATARATE_DECIMATE_ENABLE_BITBAND << DECIMATION_FACTOR_BYTE_DECIMATION_ENABLE_Pos))
#define DATARATE_DECIMATE_DISABLE_BYTE  ((uint8_t)(DATARATE_DECIMATE_DISABLE_BITBAND << DECIMATION_FACTOR_BYTE_DECIMATION_ENABLE_Pos))

/* AFE Data Rate Mode Configuration */
/*   Configure the oversampling mode used by the ADCs */
#define AFE_DATARATE_CFG1_BASE          0x4000008C

/* AFE_DATARATE_CFG1 bit positions */
#define AFE_DATARATE_CFG1_FREQ_MODE_Pos 0
#define AFE_DATARATE_CFG1_FREQ_MODE_Mask ((uint32_t)(0xFU << AFE_DATARATE_CFG1_FREQ_MODE_Pos))

/* PWM0 Period Configuration */
/*   Set the legnth of each PWM0 period */
#define AFE_PWM0_PERIOD_BASE            0x40000090

/* AFE_PWM0_PERIOD bit positions */
#define AFE_PWM0_PERIOD_PWM0_PERIOD_Pos 0
#define AFE_PWM0_PERIOD_PWM0_PERIOD_Mask ((uint32_t)(0xFFU << AFE_PWM0_PERIOD_PWM0_PERIOD_Pos))

/* PWM1 Period Configuration */
/*   Set the legnth of each PWM1 period */
#define AFE_PWM1_PERIOD_BASE            0x40000094

/* AFE_PWM1_PERIOD bit positions */
#define AFE_PWM1_PERIOD_PWM1_PERIOD_Pos 0
#define AFE_PWM1_PERIOD_PWM1_PERIOD_Mask ((uint32_t)(0xFFU << AFE_PWM1_PERIOD_PWM1_PERIOD_Pos))

/* PWM2  Period Configuration */
/*   Set the legnth of each PWM2 period */
#define AFE_PWM2_PERIOD_BASE            0x40000098

/* AFE_PWM2_PERIOD bit positions */
#define AFE_PWM2_PERIOD_PWM2_PERIOD_Pos 0
#define AFE_PWM2_PERIOD_PWM2_PERIOD_Mask ((uint32_t)(0xFFU << AFE_PWM2_PERIOD_PWM2_PERIOD_Pos))

/* PWM3 Period Configuration */
/*   Set the legnth of each PWM3 period */
#define AFE_PWM3_PERIOD_BASE            0x4000009C

/* AFE_PWM3_PERIOD bit positions */
#define AFE_PWM3_PERIOD_PWM3_PERIOD_Pos 0
#define AFE_PWM3_PERIOD_PWM3_PERIOD_Mask ((uint32_t)(0xFFU << AFE_PWM3_PERIOD_PWM3_PERIOD_Pos))

/* PWM0 Duty Cycle Configuration */
/*   Configure the PWM0 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM0_HI_BASE                0x400000A0

/* AFE_PWM0_HI bit positions */
#define AFE_PWM0_HI_PWM0_HI_Pos         0
#define AFE_PWM0_HI_PWM0_HI_Mask        ((uint32_t)(0xFFU << AFE_PWM0_HI_PWM0_HI_Pos))

/* PWM1 Duty Cycle Configuration */
/*   Configure the PWM1 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM1_HI_BASE                0x400000A4

/* AFE_PWM1_HI bit positions */
#define AFE_PWM1_HI_PWM1_HI_Pos         0
#define AFE_PWM1_HI_PWM1_HI_Mask        ((uint32_t)(0xFFU << AFE_PWM1_HI_PWM1_HI_Pos))

/* PWM2  Duty Cycle Configuration */
/*   Configure the PWM2 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM2_HI_BASE                0x400000A8

/* AFE_PWM2_HI bit positions */
#define AFE_PWM2_HI_PWM2_HI_Pos         0
#define AFE_PWM2_HI_PWM2_HI_Mask        ((uint32_t)(0xFFU << AFE_PWM2_HI_PWM2_HI_Pos))

/* PWM3 Duty Cycle Configuration */
/*   Configure the PWM3 duty cycle by selecting how many cycles are run in the
 *   high state */
#define AFE_PWM3_HI_BASE                0x400000AC

/* AFE_PWM3_HI bit positions */
#define AFE_PWM3_HI_PWM3_HI_Pos         0
#define AFE_PWM3_HI_PWM3_HI_Mask        ((uint32_t)(0xFFU << AFE_PWM3_HI_PWM3_HI_Pos))

/* Temperature Sensor Control and Configuration */
/*   Temperature sensor control */
#define AFE_TEMP_SENSE_CTRL_BASE        0x400000B0

/* AFE_TEMP_SENSE_CTRL bit positions */
#define AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_Pos 0

/* AFE_TEMP_SENSE_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t TEMP_SENSE_ENABLE_ALIAS;/* Enable temperature sensor circuitry */
} AFE_TEMP_SENSE_CTRL_Type;

#define AFE_TEMP_SENSE_CTRL             ((AFE_TEMP_SENSE_CTRL_Type *) SYS_CALC_BITBAND(AFE_TEMP_SENSE_CTRL_BASE, 0))

/* AFE_TEMP_SENSE_CTRL settings */
#define TEMP_SENSE_DISABLE_BITBAND      0x0
#define TEMP_SENSE_ENABLE_BITBAND       0x1
#define TEMP_SENSE_DISABLE              ((uint32_t)(TEMP_SENSE_DISABLE_BITBAND << AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_Pos))
#define TEMP_SENSE_ENABLE               ((uint32_t)(TEMP_SENSE_ENABLE_BITBAND << AFE_TEMP_SENSE_CTRL_TEMP_SENSE_ENABLE_Pos))

/* DAC0 Voltage Reference control */
/*   Select voltage references for DAC0 */
#define AFE_DAC0_REF_CTRL_BASE          0x400000C0

/* AFE_DAC0_REF_CTRL bit positions */
#define AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Pos 0
#define AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Mask ((uint32_t)(0x3U << AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Pos))

/* AFE_DAC0_REF_CTRL settings */
#define DAC0_REF_SELECT_VREF            ((uint32_t)(0x0U << AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Pos))
#define DAC0_REF_SELECT_2VREF           ((uint32_t)(0x1U << AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Pos))
#define DAC0_REF_SELECT_3VREF           ((uint32_t)(0x2U << AFE_DAC0_REF_CTRL_DAC0_REF_CTRL_Pos))

/* DAC Control and Configuration */
/*   DAC Control and Configuration */
#define AFE_DAC_CTRL_BASE               0x400000C4

/* AFE_DAC_CTRL bit positions */
#define AFE_DAC_CTRL_MODE_SELECT_B_Pos  19
#define AFE_DAC_CTRL_MODE_SELECT_B_Mask ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_B_Pos))
#define AFE_DAC_CTRL_MODE_SELECT_A_Pos  16
#define AFE_DAC_CTRL_MODE_SELECT_A_Mask ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_A_Pos))
#define AFE_DAC_CTRL_DAC2_CUR_Pos       6
#define AFE_DAC_CTRL_DAC2_CUR_Mask      ((uint32_t)(0x3U << AFE_DAC_CTRL_DAC2_CUR_Pos))
#define AFE_DAC_CTRL_DAC2_ENABLE_Pos    4
#define AFE_DAC_CTRL_DAC1_ENABLE_Pos    2
#define AFE_DAC_CTRL_DAC0_ENABLE_Pos    0

/* AFE_DAC_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t DAC_CTRL_BYTE;        
         uint8_t RESERVED0[1];
    __IO uint8_t PGA1_MODE_CTRL_BYTE;  
         uint8_t RESERVED1[1];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400000C4, 0) - (0x400000C4 + 4)];
    __IO uint32_t DAC0_ENABLE_ALIAS;    /* Enable DAC0 */
         uint32_t RESERVED2;
    __IO uint32_t DAC1_ENABLE_ALIAS;    /* Enable DAC1 */
         uint32_t RESERVED3;
    __IO uint32_t DAC2_ENABLE_ALIAS;    /* Enable DAC2 */
} AFE_DAC_CTRL_Type;

#define AFE_DAC_CTRL                    ((AFE_DAC_CTRL_Type *) AFE_DAC_CTRL_BASE)

/* AFE_DAC_CTRL settings */
#define DAC0_DISABLE_BITBAND            0x0
#define DAC0_ENABLE_BITBAND             0x1
#define DAC0_DISABLE                    ((uint32_t)(DAC0_DISABLE_BITBAND << AFE_DAC_CTRL_DAC0_ENABLE_Pos))
#define DAC0_ENABLE                     ((uint32_t)(DAC0_ENABLE_BITBAND << AFE_DAC_CTRL_DAC0_ENABLE_Pos))

#define DAC1_DISABLE_BITBAND            0x0
#define DAC1_ENABLE_BITBAND             0x1
#define DAC1_DISABLE                    ((uint32_t)(DAC1_DISABLE_BITBAND << AFE_DAC_CTRL_DAC1_ENABLE_Pos))
#define DAC1_ENABLE                     ((uint32_t)(DAC1_ENABLE_BITBAND << AFE_DAC_CTRL_DAC1_ENABLE_Pos))

#define DAC2_DISABLE_BITBAND            0x0
#define DAC2_ENABLE_BITBAND             0x1
#define DAC2_DISABLE                    ((uint32_t)(DAC2_DISABLE_BITBAND << AFE_DAC_CTRL_DAC2_ENABLE_Pos))
#define DAC2_ENABLE                     ((uint32_t)(DAC2_ENABLE_BITBAND << AFE_DAC_CTRL_DAC2_ENABLE_Pos))

#define DAC2_CUR_NOMINAL                ((uint32_t)(0x0U << AFE_DAC_CTRL_DAC2_CUR_Pos))
#define DAC2_CUR_4_NOMINAL              ((uint32_t)(0x1U << AFE_DAC_CTRL_DAC2_CUR_Pos))
#define DAC2_CUR_10_NOMINAL             ((uint32_t)(0x2U << AFE_DAC_CTRL_DAC2_CUR_Pos))
#define DAC2_CUR_20_NOMINAL             ((uint32_t)(0x3U << AFE_DAC_CTRL_DAC2_CUR_Pos))

#define PGA1_A_MODE_DISABLE             ((uint32_t)(0x0U << AFE_DAC_CTRL_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_1                   ((uint32_t)(0x1U << AFE_DAC_CTRL_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_0                   ((uint32_t)(0x2U << AFE_DAC_CTRL_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_2                   ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_A_Pos))

#define PGA1_B_MODE_DISABLE             ((uint32_t)(0x0U << AFE_DAC_CTRL_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_1                   ((uint32_t)(0x1U << AFE_DAC_CTRL_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_0                   ((uint32_t)(0x2U << AFE_DAC_CTRL_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_2                   ((uint32_t)(0x3U << AFE_DAC_CTRL_MODE_SELECT_B_Pos))

/* AFE_DAC_CTRL sub-register bit positions */
#define DAC_CTRL_BYTE_DAC0_ENABLE_Pos   0
#define DAC_CTRL_BYTE_DAC1_ENABLE_Pos   2
#define DAC_CTRL_BYTE_DAC2_ENABLE_Pos   4
#define DAC_CTRL_BYTE_DAC2_CUR_Pos      6
#define DAC_CTRL_BYTE_DAC2_CUR_Mask     ((uint32_t)(0x3U << DAC_CTRL_BYTE_DAC2_CUR_Pos))
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos 0
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Mask ((uint32_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos))
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos 3
#define PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Mask ((uint32_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos))

/* AFE_DAC_CTRL subregister settings */
#define DAC0_DISABLE_BYTE               ((uint8_t)(DAC0_DISABLE_BITBAND << DAC_CTRL_BYTE_DAC0_ENABLE_Pos))
#define DAC0_ENABLE_BYTE                ((uint8_t)(DAC0_ENABLE_BITBAND << DAC_CTRL_BYTE_DAC0_ENABLE_Pos))

#define DAC1_DISABLE_BYTE               ((uint8_t)(DAC1_DISABLE_BITBAND << DAC_CTRL_BYTE_DAC1_ENABLE_Pos))
#define DAC1_ENABLE_BYTE                ((uint8_t)(DAC1_ENABLE_BITBAND << DAC_CTRL_BYTE_DAC1_ENABLE_Pos))

#define DAC2_DISABLE_BYTE               ((uint8_t)(DAC2_DISABLE_BITBAND << DAC_CTRL_BYTE_DAC2_ENABLE_Pos))
#define DAC2_ENABLE_BYTE                ((uint8_t)(DAC2_ENABLE_BITBAND << DAC_CTRL_BYTE_DAC2_ENABLE_Pos))

#define DAC2_CUR_NOMINAL_BYTE           ((uint8_t)(0x0U << DAC_CTRL_BYTE_DAC2_CUR_Pos))
#define DAC2_CUR_4_NOMINAL_BYTE         ((uint8_t)(0x1U << DAC_CTRL_BYTE_DAC2_CUR_Pos))
#define DAC2_CUR_10_NOMINAL_BYTE        ((uint8_t)(0x2U << DAC_CTRL_BYTE_DAC2_CUR_Pos))
#define DAC2_CUR_20_NOMINAL_BYTE        ((uint8_t)(0x3U << DAC_CTRL_BYTE_DAC2_CUR_Pos))

#define PGA1_A_MODE_DISABLE_BYTE        ((uint8_t)(0x0U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_1_BYTE              ((uint8_t)(0x1U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_0_BYTE              ((uint8_t)(0x2U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos))
#define PGA1_A_MODE_2_BYTE              ((uint8_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_A_Pos))

#define PGA1_B_MODE_DISABLE_BYTE        ((uint8_t)(0x0U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_1_BYTE              ((uint8_t)(0x1U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_0_BYTE              ((uint8_t)(0x2U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos))
#define PGA1_B_MODE_2_BYTE              ((uint8_t)(0x3U << PGA1_MODE_CTRL_BYTE_MODE_SELECT_B_Pos))

/* DAC0 Input Data */
/*   DAC0 converter input data */
#define AFE_DAC0_DATA_BASE              0x400000C8

/* AFE_DAC0_DATA bit positions */
#define AFE_DAC0_DATA_DAC0_DATA_Pos     0
#define AFE_DAC0_DATA_DAC0_DATA_Mask    ((uint32_t)(0x3FFU << AFE_DAC0_DATA_DAC0_DATA_Pos))

/* DAC1 Input Data */
/*   DAC1 converter input data */
#define AFE_DAC1_DATA_BASE              0x400000CC

/* AFE_DAC1_DATA bit positions */
#define AFE_DAC1_DATA_DAC1_DATA_Pos     0
#define AFE_DAC1_DATA_DAC1_DATA_Mask    ((uint32_t)(0x3FFU << AFE_DAC1_DATA_DAC1_DATA_Pos))

/* DAC2 Input Data */
/*   DAC2 converter input data */
#define AFE_DAC2_DATA_BASE              0x400000D0

/* AFE_DAC2_DATA bit positions */
#define AFE_DAC2_DATA_DAC2_DATA_Pos     0
#define AFE_DAC2_DATA_DAC2_DATA_Mask    ((uint32_t)(0x3FFU << AFE_DAC2_DATA_DAC2_DATA_Pos))

/* DAC 0 and DAC 1 Input Data */
/*   Input data for the DAC0 and DAC1 converters */
#define AFE_DAC01_DATA_BASE             0x400000D4

/* AFE_DAC01_DATA bit positions */
#define AFE_DAC01_DATA_DAC0_DATA_Pos    16
#define AFE_DAC01_DATA_DAC0_DATA_Mask   ((uint32_t)(0x3FFU << AFE_DAC01_DATA_DAC0_DATA_Pos))
#define AFE_DAC01_DATA_DAC1_DATA_Pos    0
#define AFE_DAC01_DATA_DAC1_DATA_Mask   ((uint32_t)(0x3FFU << AFE_DAC01_DATA_DAC1_DATA_Pos))

/* RTC Clock Control */
/*   Control the real-time clock behavior */
#define AFE_RTC_CTRL_BASE               0x400000E0

/* AFE_RTC_CTRL bit positions */
#define AFE_RTC_CTRL_RTC_LOAD_Pos       3
#define AFE_RTC_CTRL_RTC_BIAS_ENABLE_Pos 2
#define AFE_RTC_CTRL_ALARM_ENABLE_Pos   1
#define AFE_RTC_CTRL_RTC_MODE_CFG_Pos   0

/* AFE_RTC_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t RTC_MODE_CFG_ALIAS;   /* Set RTC operation mode */
    __IO uint32_t ALARM_ENABLE_ALIAS;   /* Alarm enable */
    __O  uint32_t RTC_BIAS_ENABLE_ALIAS;/* Enable the real-time clock */
    __O  uint32_t RTC_LOAD_ALIAS;       /* Load the RTC counter and alarm configuration */
} AFE_RTC_CTRL_Type;

#define AFE_RTC_CTRL                    ((AFE_RTC_CTRL_Type *) SYS_CALC_BITBAND(AFE_RTC_CTRL_BASE, 0))

/* AFE_RTC_CTRL settings */
#define RTC_RUN_MODE_BITBAND            0x0
#define RTC_SET_MODE_BITBAND            0x1
#define RTC_RUN_MODE                    ((uint32_t)(RTC_RUN_MODE_BITBAND << AFE_RTC_CTRL_RTC_MODE_CFG_Pos))
#define RTC_SET_MODE                    ((uint32_t)(RTC_SET_MODE_BITBAND << AFE_RTC_CTRL_RTC_MODE_CFG_Pos))

#define ALARM_DISABLE_BITBAND           0x0
#define ALARM_ENABLE_BITBAND            0x1
#define ALARM_DISABLE                   ((uint32_t)(ALARM_DISABLE_BITBAND << AFE_RTC_CTRL_ALARM_ENABLE_Pos))
#define ALARM_ENABLE                    ((uint32_t)(ALARM_ENABLE_BITBAND << AFE_RTC_CTRL_ALARM_ENABLE_Pos))

#define RTC_DISABLE_BITBAND             0x0
#define RTC_ENABLE_BITBAND              0x1
#define RTC_DISABLE                     ((uint32_t)(RTC_DISABLE_BITBAND << AFE_RTC_CTRL_RTC_BIAS_ENABLE_Pos))
#define RTC_ENABLE                      ((uint32_t)(RTC_ENABLE_BITBAND << AFE_RTC_CTRL_RTC_BIAS_ENABLE_Pos))

#define RTC_LOAD_BITBAND                0x1
#define RTC_LOAD                        ((uint32_t)(RTC_LOAD_BITBAND << AFE_RTC_CTRL_RTC_LOAD_Pos))

/* RTC Count */
/*   Current count for the real-time clock */
#define AFE_RTC_COUNT_BASE              0x400000E4

/* AFE_RTC_COUNT bit positions */
#define AFE_RTC_COUNT_RTC_COUNT_Pos     0
#define AFE_RTC_COUNT_RTC_COUNT_Mask    ((uint32_t)(0xFFFFFFFFU << AFE_RTC_COUNT_RTC_COUNT_Pos))

/* RTC Alarm Count */
/*   Alarm setting for the real-time clock */
#define AFE_RTC_ALARM_BASE              0x400000E8

/* AFE_RTC_ALARM bit positions */
#define AFE_RTC_ALARM_RTC_ALARM_MODE_Pos 31
#define AFE_RTC_ALARM_RTC_ALARM_Pos     0
#define AFE_RTC_ALARM_RTC_ALARM_Mask    ((uint32_t)(0x7FFFFFFFU << AFE_RTC_ALARM_RTC_ALARM_Pos))

/* AFE_RTC_ALARM sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0[31];
    __IO uint32_t RTC_ALARM_MODE_ALIAS; /* Select between relative and absolute alarm mode */
} AFE_RTC_ALARM_Type;

#define AFE_RTC_ALARM                   ((AFE_RTC_ALARM_Type *) SYS_CALC_BITBAND(AFE_RTC_ALARM_BASE, 0))

/* AFE_RTC_ALARM settings */
#define ALARM_ABSOLUTE_MODE_BITBAND     0x0
#define ALARM_RELATIVE_MODE_BITBAND     0x1
#define ALARM_ABSOLUTE_MODE             ((uint32_t)(ALARM_ABSOLUTE_MODE_BITBAND << AFE_RTC_ALARM_RTC_ALARM_MODE_Pos))
#define ALARM_RELATIVE_MODE             ((uint32_t)(ALARM_RELATIVE_MODE_BITBAND << AFE_RTC_ALARM_RTC_ALARM_MODE_Pos))

/* Analog Interrupt Status */
/*   Status for the interrupts initiating in the analog domain */
#define AFE_INTERRUPT_STATUS_BASE       0x400000F4

/* AFE_INTERRUPT_STATUS bit positions */
#define AFE_INTERRUPT_STATUS_WAKEUP_INT_Pos 7
#define AFE_INTERRUPT_STATUS_IF5_PIN3_Pos 6
#define AFE_INTERRUPT_STATUS_IF5_PIN2_Pos 5
#define AFE_INTERRUPT_STATUS_IF5_PIN1_Pos 4
#define AFE_INTERRUPT_STATUS_RTC_CLOCK_Pos 3
#define AFE_INTERRUPT_STATUS_RTC_ALARM_Pos 2
#define AFE_INTERRUPT_STATUS_IF5_PIN0_Pos 1
#define AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_Pos 0

/* AFE_INTERRUPT_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t THRESHOLD_COMPARE_ALIAS;/* Indicate if a threshold comparator event has occurred */
    __IO uint32_t IF5_PIN0_ALIAS;       /* Indicate if an IF5, pin 0 specific wakeup interrupt has occurred */
    __IO uint32_t RTC_ALARM_ALIAS;      /* Indicate if an RTC alarm interrupt has occurred */
    __IO uint32_t RTC_CLOCK_ALIAS;      /* Indicate if an RTC clock interrupt has occurred */
    __IO uint32_t IF5_PIN1_ALIAS;       /* General-purpose wakeup 1 (IF5, pin 1) was the wakeup source */
    __IO uint32_t IF5_PIN2_ALIAS;       /* General-purpose wakeup 2 (IF5, pin 2) was the wakeup source */
    __IO uint32_t IF5_PIN3_ALIAS;       /* General-purpose wakeup 3 (IF5, pin 3) was the wakeup source */
    __I  uint32_t WAKEUP_INT_ALIAS;     /* Analog general-purpose wakeup (IF5, pin 0) was the wakeup source */
} AFE_INTERRUPT_STATUS_Type;

#define AFE_INTERRUPT_STATUS            ((AFE_INTERRUPT_STATUS_Type *) SYS_CALC_BITBAND(AFE_INTERRUPT_STATUS_BASE, 0))

/* AFE_INTERRUPT_STATUS settings */
#define THRESHOLD_COMPARE_NOT_SET_BITBAND 0x0
#define THRESHOLD_COMPARE_SET_BITBAND   0x1
#define THRESHOLD_COMPARE_CLEAR_BITBAND 0x1
#define THRESHOLD_COMPARE_NOT_SET       ((uint32_t)(THRESHOLD_COMPARE_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_Pos))
#define THRESHOLD_COMPARE_SET           ((uint32_t)(THRESHOLD_COMPARE_SET_BITBAND << AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_Pos))
#define THRESHOLD_COMPARE_CLEAR         ((uint32_t)(THRESHOLD_COMPARE_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_THRESHOLD_COMPARE_Pos))

#define WAKEUP_IF5_PIN0_NOT_SET_BITBAND 0x0
#define WAKEUP_IF5_PIN0_SET_BITBAND     0x1
#define WAKEUP_IF5_PIN0_CLEAR_BITBAND   0x1
#define WAKEUP_IF5_PIN0_NOT_SET         ((uint32_t)(WAKEUP_IF5_PIN0_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN0_Pos))
#define WAKEUP_IF5_PIN0_SET             ((uint32_t)(WAKEUP_IF5_PIN0_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN0_Pos))
#define WAKEUP_IF5_PIN0_CLEAR           ((uint32_t)(WAKEUP_IF5_PIN0_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN0_Pos))

#define RTC_ALARM_NOT_SET_BITBAND       0x0
#define RTC_ALARM_SET_BITBAND           0x1
#define RTC_ALARM_CLEAR_BITBAND         0x1
#define RTC_ALARM_NOT_SET               ((uint32_t)(RTC_ALARM_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_RTC_ALARM_Pos))
#define RTC_ALARM_SET                   ((uint32_t)(RTC_ALARM_SET_BITBAND << AFE_INTERRUPT_STATUS_RTC_ALARM_Pos))
#define RTC_ALARM_CLEAR                 ((uint32_t)(RTC_ALARM_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_RTC_ALARM_Pos))

#define RTC_CLOCK_NOT_SET_BITBAND       0x0
#define RTC_CLOCK_SET_BITBAND           0x1
#define RTC_CLOCK_CLEAR_BITBAND         0x1
#define RTC_CLOCK_NOT_SET               ((uint32_t)(RTC_CLOCK_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_RTC_CLOCK_Pos))
#define RTC_CLOCK_SET                   ((uint32_t)(RTC_CLOCK_SET_BITBAND << AFE_INTERRUPT_STATUS_RTC_CLOCK_Pos))
#define RTC_CLOCK_CLEAR                 ((uint32_t)(RTC_CLOCK_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_RTC_CLOCK_Pos))

#define WAKEUP_IF5_PIN1_NOT_SET_BITBAND 0x0
#define WAKEUP_IF5_PIN1_SET_BITBAND     0x1
#define WAKEUP_IF5_PIN1_CLEAR_BITBAND   0x1
#define WAKEUP_IF5_PIN1_NOT_SET         ((uint32_t)(WAKEUP_IF5_PIN1_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN1_Pos))
#define WAKEUP_IF5_PIN1_SET             ((uint32_t)(WAKEUP_IF5_PIN1_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN1_Pos))
#define WAKEUP_IF5_PIN1_CLEAR           ((uint32_t)(WAKEUP_IF5_PIN1_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN1_Pos))

#define WAKEUP_IF5_PIN2_NOT_SET_BITBAND 0x0
#define WAKEUP_IF5_PIN2_SET_BITBAND     0x1
#define WAKEUP_IF5_PIN2_CLEAR_BITBAND   0x1
#define WAKEUP_IF5_PIN2_NOT_SET         ((uint32_t)(WAKEUP_IF5_PIN2_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN2_Pos))
#define WAKEUP_IF5_PIN2_SET             ((uint32_t)(WAKEUP_IF5_PIN2_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN2_Pos))
#define WAKEUP_IF5_PIN2_CLEAR           ((uint32_t)(WAKEUP_IF5_PIN2_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN2_Pos))

#define WAKEUP_IF5_PIN3_NOT_SET_BITBAND 0x0
#define WAKEUP_IF5_PIN3_SET_BITBAND     0x1
#define WAKEUP_IF5_PIN3_CLEAR_BITBAND   0x1
#define WAKEUP_IF5_PIN3_NOT_SET         ((uint32_t)(WAKEUP_IF5_PIN3_NOT_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN3_Pos))
#define WAKEUP_IF5_PIN3_SET             ((uint32_t)(WAKEUP_IF5_PIN3_SET_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN3_Pos))
#define WAKEUP_IF5_PIN3_CLEAR           ((uint32_t)(WAKEUP_IF5_PIN3_CLEAR_BITBAND << AFE_INTERRUPT_STATUS_IF5_PIN3_Pos))

#define WAKEUP_INT_CLEARED_BITBAND      0x0
#define WAKEUP_INT_PEND_BITBAND         0x1
#define WAKEUP_INT_CLEARED              ((uint32_t)(WAKEUP_INT_CLEARED_BITBAND << AFE_INTERRUPT_STATUS_WAKEUP_INT_Pos))
#define WAKEUP_INT_PEND                 ((uint32_t)(WAKEUP_INT_PEND_BITBAND << AFE_INTERRUPT_STATUS_WAKEUP_INT_Pos))

/* General-Purpose Data Retention */
/*   General-purpose data retention register, used to maintain one data value
 *   while in sleep mode */
#define AFE_RETENTION_DATA_BASE         0x400000F8

/* AFE_RETENTION_DATA bit positions */
#define AFE_RETENTION_DATA_RETENTION_DATA_Pos 0
#define AFE_RETENTION_DATA_RETENTION_DATA_Mask ((uint32_t)(0xFFU << AFE_RETENTION_DATA_RETENTION_DATA_Pos))

/* AFE_RETENTION_DATA sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t RETENTION_DATA_BYTE;  
         uint8_t RESERVED0[3];
} AFE_RETENTION_DATA_Type;

#define AFE_RETENTION_DATA              ((AFE_RETENTION_DATA_Type *) AFE_RETENTION_DATA_BASE)

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

typedef struct
{
    __IO uint32_t CH0_CTRL0;            /* Control and configuration of DMA channel 0, specifying the source and destination configurations for this channel */
    __IO uint32_t CH0_SRC_BASE_ADDR;    /* Base address for the source of data transferred using DMA channel 0 */
    __IO uint32_t CH0_DEST_BASE_ADDR;   /* Base address for the destination of data transferred using DMA channel 0 */
    __IO uint32_t CH0_CTRL1;            /* Control the length of and how often interrupts are raised for a DMA transfer */
    __I  uint32_t CH0_NEXT_SRC_ADDR;    /* Address where the next data to be transferred using DMA channel 0 is loaded from */
    __I  uint32_t CH0_NEXT_DEST_ADDR;   /* Address where the next data to be transferred using DMA channel 0 will be stored */
    __I  uint32_t CH0_WORD_CNT;         /* The number of words that have been transferred using DMA channel 0 during the current transfer */
         uint32_t RESERVED0;
    __IO uint32_t CH1_CTRL0;            /* Control and configuration of DMA channel 1, specifying the source and destination configurations for this channel */
    __IO uint32_t CH1_SRC_BASE_ADDR;    /* Base address for the source of data transferred using DMA channel 1 */
    __IO uint32_t CH1_DEST_BASE_ADDR;   /* Base address for the destination of data transferred using DMA channel 1 */
    __IO uint32_t CH1_CTRL1;            /* Control the length of and how often interrupts are raised for a DMA transfer */
    __I  uint32_t CH1_NEXT_SRC_ADDR;    /* Address where the next data to be transferred using DMA channel 1 is loaded from */
    __I  uint32_t CH1_NEXT_DEST_ADDR;   /* Address where the next data to be transferred using DMA channel 1 will be stored */
    __I  uint32_t CH1_WORD_CNT;         /* The number of words that have been transferred using DMA channel 1 during the current transfer */
         uint32_t RESERVED1;
    __IO uint32_t CH2_CTRL0;            /* Control and configuration of DMA channel 2, specifying the source and destination configurations for this channel */
    __IO uint32_t CH2_SRC_BASE_ADDR;    /* Base address for the source of data transferred using DMA channel 2 */
    __IO uint32_t CH2_DEST_BASE_ADDR;   /* Base address for the destination of data transferred using DMA channel 2 */
    __IO uint32_t CH2_CTRL1;            /* Control the length of and how often interrupts are raised for a DMA transfer */
    __I  uint32_t CH2_NEXT_SRC_ADDR;    /* Address where the next data to be transferred using DMA channel 2 is loaded from */
    __I  uint32_t CH2_NEXT_DEST_ADDR;   /* Address where the next data to be transferred using DMA channel 2 will be stored */
    __I  uint32_t CH2_WORD_CNT;         /* The number of words that have been transferred using DMA channel 2 during the current transfer */
         uint32_t RESERVED2;
    __IO uint32_t CH3_CTRL0;            /* Control and configuration of DMA channel 3, specifying the source and destination configurations for this channel */
    __IO uint32_t CH3_SRC_BASE_ADDR;    /* Base address for the source of data transferred using DMA channel 3 */
    __IO uint32_t CH3_DEST_BASE_ADDR;   /* Base address for the destination of data transferred using DMA channel 3 */
    __IO uint32_t CH3_CTRL1;            /* Control the length of and how often interrupts are raised for a DMA transfer */
    __I  uint32_t CH3_NEXT_SRC_ADDR;    /* Address where the next data to be transferred using DMA channel 3 is loaded from */
    __I  uint32_t CH3_NEXT_DEST_ADDR;   /* Address where the next data to be transferred using DMA channel 3 will be stored */
    __I  uint32_t CH3_WORD_CNT;         /* The number of words that have been transferred using DMA channel 3 during the current transfer */
         uint32_t RESERVED3;
    __IO uint32_t DAC0_REQUEST;         /* Control the DAC0 DMA transfers and interrupts */
    __IO uint32_t DAC1_REQUEST;         /* Control the DAC1 DMA transfers and interrupts */
         uint32_t RESERVED4[2];
    __IO uint32_t STATUS;               /* Status of DMA transfers and interrupts */
} DMA_Type;

#define DMA_BASE                        0x40000200
#define DMA                             ((DMA_Type *) DMA_BASE)

/* DMA Channel 0 Control and Configuration */
/*   Control and configuration of DMA channel 0, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH0_CTRL0_BASE              0x40000200

/* DMA_CH0_CTRL0 bit positions */
#define DMA_CH0_CTRL0_BYTE_ORDER_Pos    25
#define DMA_CH0_CTRL0_DISABLE_INT_ENABLE_Pos 24
#define DMA_CH0_CTRL0_ERROR_INT_ENABLE_Pos 23
#define DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_Pos 22
#define DMA_CH0_CTRL0_COUNTER_INT_ENABLE_Pos 21
#define DMA_CH0_CTRL0_START_INT_ENABLE_Pos 20
#define DMA_CH0_CTRL0_DEST_WORD_SIZE_Pos 18
#define DMA_CH0_CTRL0_DEST_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH0_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH0_CTRL0_SRC_WORD_SIZE_Pos 16
#define DMA_CH0_CTRL0_SRC_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH0_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH0_CTRL0_DEST_SELECT_Pos   12
#define DMA_CH0_CTRL0_DEST_SELECT_Mask  ((uint32_t)(0xFU << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_CTRL0_SRC_SELECT_Pos    8
#define DMA_CH0_CTRL0_SRC_SELECT_Mask   ((uint32_t)(0xFU << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos 6
#define DMA_CH0_CTRL0_CHANNEL_PRIORITY_Mask ((uint32_t)(0x3U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH0_CTRL0_TRANSFER_TYPE_Pos 4
#define DMA_CH0_CTRL0_TRANSFER_TYPE_Mask ((uint32_t)(0x3U << DMA_CH0_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH0_CTRL0_DEST_ADDR_INC_Pos 3
#define DMA_CH0_CTRL0_SRC_ADDR_INC_Pos  2
#define DMA_CH0_CTRL0_ADDR_MODE_Pos     1
#define DMA_CH0_CTRL0_ENABLE_Pos        0

/* DMA_CH0_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ENABLE_ALIAS;         /* Enable DMA Channel 0 */
    __IO uint32_t ADDR_MODE_ALIAS;      /* Select the addressing mode for this channel */
    __IO uint32_t SRC_ADDR_INC_ALIAS;   /* Configure whether the source address will be incremented */
    __IO uint32_t DEST_ADDR_INC_ALIAS;  /* Configure whether the destination address will be incremented */
         uint32_t RESERVED0[16];
    __IO uint32_t START_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer starts */
    __IO uint32_t COUNTER_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer reaches the counter value */
    __IO uint32_t COMPLETE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer completes */
    __IO uint32_t ERROR_INT_ENABLE_ALIAS;/* Raise an interrupt when a state machine error occurs during a DMA transfer */
    __IO uint32_t DISABLE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA channel is disabled */
    __IO uint32_t BYTE_ORDER_ALIAS;     /* Select the byte ordering for the DMA channel */
} DMA_CH0_CTRL0_Type;

#define DMA_CH0_CTRL0                   ((DMA_CH0_CTRL0_Type *) SYS_CALC_BITBAND(DMA_CH0_CTRL0_BASE, 0))

/* DMA_CH0_CTRL0 settings */
#define DMA_CH0_DISABLE_BITBAND         0x0
#define DMA_CH0_ENABLE_BITBAND          0x1
#define DMA_CH0_DISABLE                 ((uint32_t)(DMA_CH0_DISABLE_BITBAND << DMA_CH0_CTRL0_ENABLE_Pos))
#define DMA_CH0_ENABLE                  ((uint32_t)(DMA_CH0_ENABLE_BITBAND << DMA_CH0_CTRL0_ENABLE_Pos))

#define DMA_CH0_ADDR_CIRC_BITBAND       0x0
#define DMA_CH0_ADDR_LIN_BITBAND        0x1
#define DMA_CH0_ADDR_CIRC               ((uint32_t)(DMA_CH0_ADDR_CIRC_BITBAND << DMA_CH0_CTRL0_ADDR_MODE_Pos))
#define DMA_CH0_ADDR_LIN                ((uint32_t)(DMA_CH0_ADDR_LIN_BITBAND << DMA_CH0_CTRL0_ADDR_MODE_Pos))

#define DMA_CH0_SRC_ADDR_STATIC_BITBAND 0x0
#define DMA_CH0_SRC_ADDR_INC_BITBAND    0x1
#define DMA_CH0_SRC_ADDR_STATIC         ((uint32_t)(DMA_CH0_SRC_ADDR_STATIC_BITBAND << DMA_CH0_CTRL0_SRC_ADDR_INC_Pos))
#define DMA_CH0_SRC_ADDR_INC            ((uint32_t)(DMA_CH0_SRC_ADDR_INC_BITBAND << DMA_CH0_CTRL0_SRC_ADDR_INC_Pos))

#define DMA_CH0_DEST_ADDR_STATIC_BITBAND 0x0
#define DMA_CH0_DEST_ADDR_INC_BITBAND   0x1
#define DMA_CH0_DEST_ADDR_STATIC        ((uint32_t)(DMA_CH0_DEST_ADDR_STATIC_BITBAND << DMA_CH0_CTRL0_DEST_ADDR_INC_Pos))
#define DMA_CH0_DEST_ADDR_INC           ((uint32_t)(DMA_CH0_DEST_ADDR_INC_BITBAND << DMA_CH0_CTRL0_DEST_ADDR_INC_Pos))

#define DMA_CH0_TRANSFER_M_TO_M         ((uint32_t)(0x0U << DMA_CH0_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH0_TRANSFER_M_TO_P         ((uint32_t)(0x1U << DMA_CH0_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH0_TRANSFER_P_TO_M         ((uint32_t)(0x2U << DMA_CH0_CTRL0_TRANSFER_TYPE_Pos))

#define DMA_CH0_PRIORITY_0              ((uint32_t)(0x0U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH0_PRIORITY_1              ((uint32_t)(0x1U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH0_PRIORITY_2              ((uint32_t)(0x2U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH0_PRIORITY_3              ((uint32_t)(0x3U << DMA_CH0_CTRL0_CHANNEL_PRIORITY_Pos))

#define DMA_CH0_SRC_ADC                 ((uint32_t)(0x0U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_USB                 ((uint32_t)(0x3U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_PCM                 ((uint32_t)(0x6U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_SPI0                ((uint32_t)(0x7U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_SPI1                ((uint32_t)(0x8U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_I2C                 ((uint32_t)(0x9U << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_UART0               ((uint32_t)(0xAU << DMA_CH0_CTRL0_SRC_SELECT_Pos))
#define DMA_CH0_SRC_UART1               ((uint32_t)(0xBU << DMA_CH0_CTRL0_SRC_SELECT_Pos))

#define DMA_CH0_DEST_DAC0               ((uint32_t)(0x1U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_DAC1               ((uint32_t)(0x2U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_USB                ((uint32_t)(0x3U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_PCM                ((uint32_t)(0x6U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_SPI0               ((uint32_t)(0x7U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_SPI1               ((uint32_t)(0x8U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_I2C                ((uint32_t)(0x9U << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_UART0              ((uint32_t)(0xAU << DMA_CH0_CTRL0_DEST_SELECT_Pos))
#define DMA_CH0_DEST_UART1              ((uint32_t)(0xBU << DMA_CH0_CTRL0_DEST_SELECT_Pos))

#define DMA_CH0_SRC_WORD_SIZE_8         ((uint32_t)(0x0U << DMA_CH0_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH0_SRC_WORD_SIZE_16        ((uint32_t)(0x1U << DMA_CH0_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH0_SRC_WORD_SIZE_32        ((uint32_t)(0x2U << DMA_CH0_CTRL0_SRC_WORD_SIZE_Pos))

#define DMA_CH0_DEST_WORD_SIZE_8        ((uint32_t)(0x0U << DMA_CH0_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH0_DEST_WORD_SIZE_16       ((uint32_t)(0x1U << DMA_CH0_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH0_DEST_WORD_SIZE_32       ((uint32_t)(0x2U << DMA_CH0_CTRL0_DEST_WORD_SIZE_Pos))

#define DMA_CH0_START_INT_DISABLE_BITBAND 0x0
#define DMA_CH0_START_INT_ENABLE_BITBAND 0x1
#define DMA_CH0_START_INT_DISABLE       ((uint32_t)(DMA_CH0_START_INT_DISABLE_BITBAND << DMA_CH0_CTRL0_START_INT_ENABLE_Pos))
#define DMA_CH0_START_INT_ENABLE        ((uint32_t)(DMA_CH0_START_INT_ENABLE_BITBAND << DMA_CH0_CTRL0_START_INT_ENABLE_Pos))

#define DMA_CH0_COUNTER_INT_DISABLE_BITBAND 0x0
#define DMA_CH0_COUNTER_INT_ENABLE_BITBAND 0x1
#define DMA_CH0_COUNTER_INT_DISABLE     ((uint32_t)(DMA_CH0_COUNTER_INT_DISABLE_BITBAND << DMA_CH0_CTRL0_COUNTER_INT_ENABLE_Pos))
#define DMA_CH0_COUNTER_INT_ENABLE      ((uint32_t)(DMA_CH0_COUNTER_INT_ENABLE_BITBAND << DMA_CH0_CTRL0_COUNTER_INT_ENABLE_Pos))

#define DMA_CH0_COMPLETE_INT_DISABLE_BITBAND 0x0
#define DMA_CH0_COMPLETE_INT_ENABLE_BITBAND 0x1
#define DMA_CH0_COMPLETE_INT_DISABLE    ((uint32_t)(DMA_CH0_COMPLETE_INT_DISABLE_BITBAND << DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_Pos))
#define DMA_CH0_COMPLETE_INT_ENABLE     ((uint32_t)(DMA_CH0_COMPLETE_INT_ENABLE_BITBAND << DMA_CH0_CTRL0_COMPLETE_INT_ENABLE_Pos))

#define DMA_CH0_ERROR_INT_DISABLE_BITBAND 0x0
#define DMA_CH0_ERROR_INT_ENABLE_BITBAND 0x1
#define DMA_CH0_ERROR_INT_DISABLE       ((uint32_t)(DMA_CH0_ERROR_INT_DISABLE_BITBAND << DMA_CH0_CTRL0_ERROR_INT_ENABLE_Pos))
#define DMA_CH0_ERROR_INT_ENABLE        ((uint32_t)(DMA_CH0_ERROR_INT_ENABLE_BITBAND << DMA_CH0_CTRL0_ERROR_INT_ENABLE_Pos))

#define DMA_CH0_DISABLE_INT_DISABLE_BITBAND 0x0
#define DMA_CH0_DISABLE_INT_ENABLE_BITBAND 0x1
#define DMA_CH0_DISABLE_INT_DISABLE     ((uint32_t)(DMA_CH0_DISABLE_INT_DISABLE_BITBAND << DMA_CH0_CTRL0_DISABLE_INT_ENABLE_Pos))
#define DMA_CH0_DISABLE_INT_ENABLE      ((uint32_t)(DMA_CH0_DISABLE_INT_ENABLE_BITBAND << DMA_CH0_CTRL0_DISABLE_INT_ENABLE_Pos))

#define DMA_CH0_LITTLE_ENDIAN_BITBAND   0x0
#define DMA_CH0_BIG_ENDIAN_BITBAND      0x1
#define DMA_CH0_LITTLE_ENDIAN           ((uint32_t)(DMA_CH0_LITTLE_ENDIAN_BITBAND << DMA_CH0_CTRL0_BYTE_ORDER_Pos))
#define DMA_CH0_BIG_ENDIAN              ((uint32_t)(DMA_CH0_BIG_ENDIAN_BITBAND << DMA_CH0_CTRL0_BYTE_ORDER_Pos))

/* DMA Channel 0 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 0 */
#define DMA_CH0_SRC_BASE_ADDR_BASE      0x40000204

/* DMA Channel 0 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   0 */
#define DMA_CH0_DEST_BASE_ADDR_BASE     0x40000208

/* DMA Channel 0 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH0_CTRL1_BASE              0x4000020C

/* DMA_CH0_CTRL1 bit positions */
#define DMA_CH0_CTRL1_COUNTER_INT_VALUE_Pos 16
#define DMA_CH0_CTRL1_COUNTER_INT_VALUE_Mask ((uint32_t)(0xFFFFU << DMA_CH0_CTRL1_COUNTER_INT_VALUE_Pos))
#define DMA_CH0_CTRL1_TRANSFER_LENGTH_Pos 0
#define DMA_CH0_CTRL1_TRANSFER_LENGTH_Mask ((uint32_t)(0xFFFFU << DMA_CH0_CTRL1_TRANSFER_LENGTH_Pos))

/* DMA_CH0_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t TRANSFER_LENGTH_SHORT;
    __IO uint16_t COUNTER_INT_VALUE_SHORT;
} DMA_CH0_CTRL1_Type;

#define DMA_CH0_CTRL1                   ((DMA_CH0_CTRL1_Type *) DMA_CH0_CTRL1_BASE)

/* DMA Channel 0 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 0 is
 *   loaded from */
#define DMA_CH0_NEXT_SRC_ADDR_BASE      0x40000210

/* DMA Channel 0 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 0 will be
 *   stored */
#define DMA_CH0_NEXT_DEST_ADDR_BASE     0x40000214

/* DMA Channel 0 Word Count */
/*   The number of words that have been transferred using DMA channel 0 during
 *   the current transfer */
#define DMA_CH0_WORD_CNT_BASE           0x40000218

/* DMA channel 1 Control and Configuration */
/*   Control and configuration of DMA channel 1, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH1_CTRL0_BASE              0x40000220

/* DMA_CH1_CTRL0 bit positions */
#define DMA_CH1_CTRL0_BYTE_ORDER_Pos    25
#define DMA_CH1_CTRL0_DISABLE_INT_ENABLE_Pos 24
#define DMA_CH1_CTRL0_ERROR_INT_ENABLE_Pos 23
#define DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_Pos 22
#define DMA_CH1_CTRL0_COUNTER_INT_ENABLE_Pos 21
#define DMA_CH1_CTRL0_START_INT_ENABLE_Pos 20
#define DMA_CH1_CTRL0_DEST_WORD_SIZE_Pos 18
#define DMA_CH1_CTRL0_DEST_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH1_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH1_CTRL0_SRC_WORD_SIZE_Pos 16
#define DMA_CH1_CTRL0_SRC_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH1_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH1_CTRL0_DEST_SELECT_Pos   12
#define DMA_CH1_CTRL0_DEST_SELECT_Mask  ((uint32_t)(0xFU << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_CTRL0_SRC_SELECT_Pos    8
#define DMA_CH1_CTRL0_SRC_SELECT_Mask   ((uint32_t)(0xFU << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos 6
#define DMA_CH1_CTRL0_CHANNEL_PRIORITY_Mask ((uint32_t)(0x3U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH1_CTRL0_TRANSFER_TYPE_Pos 4
#define DMA_CH1_CTRL0_TRANSFER_TYPE_Mask ((uint32_t)(0x3U << DMA_CH1_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH1_CTRL0_DEST_ADDR_INC_Pos 3
#define DMA_CH1_CTRL0_SRC_ADDR_INC_Pos  2
#define DMA_CH1_CTRL0_ADDR_MODE_Pos     1
#define DMA_CH1_CTRL0_ENABLE_Pos        0

/* DMA_CH1_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ENABLE_ALIAS;         /* Enable DMA channel 1 */
    __IO uint32_t ADDR_MODE_ALIAS;      /* Select the addressing mode for this channel */
    __IO uint32_t SRC_ADDR_INC_ALIAS;   /* Configure whether the source address will be incremented */
    __IO uint32_t DEST_ADDR_INC_ALIAS;  /* Configure whether the destination address will be incremented */
         uint32_t RESERVED0[16];
    __IO uint32_t START_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer starts */
    __IO uint32_t COUNTER_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer reaches the counter value */
    __IO uint32_t COMPLETE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer completes */
    __IO uint32_t ERROR_INT_ENABLE_ALIAS;/* Raise an interrupt when a state machine error occurs during a DMA transfer */
    __IO uint32_t DISABLE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA channel is disabled */
    __IO uint32_t BYTE_ORDER_ALIAS;     /* Select the byte ordering for the DMA channel */
} DMA_CH1_CTRL0_Type;

#define DMA_CH1_CTRL0                   ((DMA_CH1_CTRL0_Type *) SYS_CALC_BITBAND(DMA_CH1_CTRL0_BASE, 0))

/* DMA_CH1_CTRL0 settings */
#define DMA_CH1_DISABLE_BITBAND         0x0
#define DMA_CH1_ENABLE_BITBAND          0x1
#define DMA_CH1_DISABLE                 ((uint32_t)(DMA_CH1_DISABLE_BITBAND << DMA_CH1_CTRL0_ENABLE_Pos))
#define DMA_CH1_ENABLE                  ((uint32_t)(DMA_CH1_ENABLE_BITBAND << DMA_CH1_CTRL0_ENABLE_Pos))

#define DMA_CH1_ADDR_CIRC_BITBAND       0x0
#define DMA_CH1_ADDR_LIN_BITBAND        0x1
#define DMA_CH1_ADDR_CIRC               ((uint32_t)(DMA_CH1_ADDR_CIRC_BITBAND << DMA_CH1_CTRL0_ADDR_MODE_Pos))
#define DMA_CH1_ADDR_LIN                ((uint32_t)(DMA_CH1_ADDR_LIN_BITBAND << DMA_CH1_CTRL0_ADDR_MODE_Pos))

#define DMA_CH1_SRC_ADDR_STATIC_BITBAND 0x0
#define DMA_CH1_SRC_ADDR_INC_BITBAND    0x1
#define DMA_CH1_SRC_ADDR_STATIC         ((uint32_t)(DMA_CH1_SRC_ADDR_STATIC_BITBAND << DMA_CH1_CTRL0_SRC_ADDR_INC_Pos))
#define DMA_CH1_SRC_ADDR_INC            ((uint32_t)(DMA_CH1_SRC_ADDR_INC_BITBAND << DMA_CH1_CTRL0_SRC_ADDR_INC_Pos))

#define DMA_CH1_DEST_ADDR_STATIC_BITBAND 0x0
#define DMA_CH1_DEST_ADDR_INC_BITBAND   0x1
#define DMA_CH1_DEST_ADDR_STATIC        ((uint32_t)(DMA_CH1_DEST_ADDR_STATIC_BITBAND << DMA_CH1_CTRL0_DEST_ADDR_INC_Pos))
#define DMA_CH1_DEST_ADDR_INC           ((uint32_t)(DMA_CH1_DEST_ADDR_INC_BITBAND << DMA_CH1_CTRL0_DEST_ADDR_INC_Pos))

#define DMA_CH1_TRANSFER_M_TO_M         ((uint32_t)(0x0U << DMA_CH1_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH1_TRANSFER_M_TO_P         ((uint32_t)(0x1U << DMA_CH1_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH1_TRANSFER_P_TO_M         ((uint32_t)(0x2U << DMA_CH1_CTRL0_TRANSFER_TYPE_Pos))

#define DMA_CH1_PRIORITY_0              ((uint32_t)(0x0U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH1_PRIORITY_1              ((uint32_t)(0x1U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH1_PRIORITY_2              ((uint32_t)(0x2U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH1_PRIORITY_3              ((uint32_t)(0x3U << DMA_CH1_CTRL0_CHANNEL_PRIORITY_Pos))

#define DMA_CH1_SRC_ADC                 ((uint32_t)(0x0U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_USB                 ((uint32_t)(0x3U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_PCM                 ((uint32_t)(0x6U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_SPI0                ((uint32_t)(0x7U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_SPI1                ((uint32_t)(0x8U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_I2C                 ((uint32_t)(0x9U << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_UART0               ((uint32_t)(0xAU << DMA_CH1_CTRL0_SRC_SELECT_Pos))
#define DMA_CH1_SRC_UART1               ((uint32_t)(0xBU << DMA_CH1_CTRL0_SRC_SELECT_Pos))

#define DMA_CH1_DEST_DAC0               ((uint32_t)(0x1U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_DAC1               ((uint32_t)(0x2U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_USB                ((uint32_t)(0x3U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_PCM                ((uint32_t)(0x6U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_SPI0               ((uint32_t)(0x7U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_SPI1               ((uint32_t)(0x8U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_I2C                ((uint32_t)(0x9U << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_UART0              ((uint32_t)(0xAU << DMA_CH1_CTRL0_DEST_SELECT_Pos))
#define DMA_CH1_DEST_UART1              ((uint32_t)(0xBU << DMA_CH1_CTRL0_DEST_SELECT_Pos))

#define DMA_CH1_SRC_WORD_SIZE_8         ((uint32_t)(0x0U << DMA_CH1_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH1_SRC_WORD_SIZE_16        ((uint32_t)(0x1U << DMA_CH1_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH1_SRC_WORD_SIZE_32        ((uint32_t)(0x2U << DMA_CH1_CTRL0_SRC_WORD_SIZE_Pos))

#define DMA_CH1_DEST_WORD_SIZE_8        ((uint32_t)(0x0U << DMA_CH1_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH1_DEST_WORD_SIZE_16       ((uint32_t)(0x1U << DMA_CH1_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH1_DEST_WORD_SIZE_32       ((uint32_t)(0x2U << DMA_CH1_CTRL0_DEST_WORD_SIZE_Pos))

#define DMA_CH1_START_INT_DISABLE_BITBAND 0x0
#define DMA_CH1_START_INT_ENABLE_BITBAND 0x1
#define DMA_CH1_START_INT_DISABLE       ((uint32_t)(DMA_CH1_START_INT_DISABLE_BITBAND << DMA_CH1_CTRL0_START_INT_ENABLE_Pos))
#define DMA_CH1_START_INT_ENABLE        ((uint32_t)(DMA_CH1_START_INT_ENABLE_BITBAND << DMA_CH1_CTRL0_START_INT_ENABLE_Pos))

#define DMA_CH1_COUNTER_INT_DISABLE_BITBAND 0x0
#define DMA_CH1_COUNTER_INT_ENABLE_BITBAND 0x1
#define DMA_CH1_COUNTER_INT_DISABLE     ((uint32_t)(DMA_CH1_COUNTER_INT_DISABLE_BITBAND << DMA_CH1_CTRL0_COUNTER_INT_ENABLE_Pos))
#define DMA_CH1_COUNTER_INT_ENABLE      ((uint32_t)(DMA_CH1_COUNTER_INT_ENABLE_BITBAND << DMA_CH1_CTRL0_COUNTER_INT_ENABLE_Pos))

#define DMA_CH1_COMPLETE_INT_DISABLE_BITBAND 0x0
#define DMA_CH1_COMPLETE_INT_ENABLE_BITBAND 0x1
#define DMA_CH1_COMPLETE_INT_DISABLE    ((uint32_t)(DMA_CH1_COMPLETE_INT_DISABLE_BITBAND << DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_Pos))
#define DMA_CH1_COMPLETE_INT_ENABLE     ((uint32_t)(DMA_CH1_COMPLETE_INT_ENABLE_BITBAND << DMA_CH1_CTRL0_COMPLETE_INT_ENABLE_Pos))

#define DMA_CH1_ERROR_INT_DISABLE_BITBAND 0x0
#define DMA_CH1_ERROR_INT_ENABLE_BITBAND 0x1
#define DMA_CH1_ERROR_INT_DISABLE       ((uint32_t)(DMA_CH1_ERROR_INT_DISABLE_BITBAND << DMA_CH1_CTRL0_ERROR_INT_ENABLE_Pos))
#define DMA_CH1_ERROR_INT_ENABLE        ((uint32_t)(DMA_CH1_ERROR_INT_ENABLE_BITBAND << DMA_CH1_CTRL0_ERROR_INT_ENABLE_Pos))

#define DMA_CH1_DISABLE_INT_DISABLE_BITBAND 0x0
#define DMA_CH1_DISABLE_INT_ENABLE_BITBAND 0x1
#define DMA_CH1_DISABLE_INT_DISABLE     ((uint32_t)(DMA_CH1_DISABLE_INT_DISABLE_BITBAND << DMA_CH1_CTRL0_DISABLE_INT_ENABLE_Pos))
#define DMA_CH1_DISABLE_INT_ENABLE      ((uint32_t)(DMA_CH1_DISABLE_INT_ENABLE_BITBAND << DMA_CH1_CTRL0_DISABLE_INT_ENABLE_Pos))

#define DMA_CH1_LITTLE_ENDIAN_BITBAND   0x0
#define DMA_CH1_BIG_ENDIAN_BITBAND      0x1
#define DMA_CH1_LITTLE_ENDIAN           ((uint32_t)(DMA_CH1_LITTLE_ENDIAN_BITBAND << DMA_CH1_CTRL0_BYTE_ORDER_Pos))
#define DMA_CH1_BIG_ENDIAN              ((uint32_t)(DMA_CH1_BIG_ENDIAN_BITBAND << DMA_CH1_CTRL0_BYTE_ORDER_Pos))

/* DMA channel 1 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 1 */
#define DMA_CH1_SRC_BASE_ADDR_BASE      0x40000224

/* DMA channel 1 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   1 */
#define DMA_CH1_DEST_BASE_ADDR_BASE     0x40000228

/* DMA channel 1 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH1_CTRL1_BASE              0x4000022C

/* DMA_CH1_CTRL1 bit positions */
#define DMA_CH1_CTRL1_COUNTER_INT_VALUE_Pos 16
#define DMA_CH1_CTRL1_COUNTER_INT_VALUE_Mask ((uint32_t)(0xFFFFU << DMA_CH1_CTRL1_COUNTER_INT_VALUE_Pos))
#define DMA_CH1_CTRL1_TRANSFER_LENGTH_Pos 0
#define DMA_CH1_CTRL1_TRANSFER_LENGTH_Mask ((uint32_t)(0xFFFFU << DMA_CH1_CTRL1_TRANSFER_LENGTH_Pos))

/* DMA_CH1_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t TRANSFER_LENGTH_SHORT;
    __IO uint16_t COUNTER_INT_VALUE_SHORT;
} DMA_CH1_CTRL1_Type;

#define DMA_CH1_CTRL1                   ((DMA_CH1_CTRL1_Type *) DMA_CH1_CTRL1_BASE)

/* DMA channel 1 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 1 is
 *   loaded from */
#define DMA_CH1_NEXT_SRC_ADDR_BASE      0x40000230

/* DMA channel 1 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 1 will be
 *   stored */
#define DMA_CH1_NEXT_DEST_ADDR_BASE     0x40000234

/* DMA channel 1 Word Count */
/*   The number of words that have been transferred using DMA channel 1 during
 *   the current transfer */
#define DMA_CH1_WORD_CNT_BASE           0x40000238

/* DMA channel 2 Control and Configuration */
/*   Control and configuration of DMA channel 2, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH2_CTRL0_BASE              0x40000240

/* DMA_CH2_CTRL0 bit positions */
#define DMA_CH2_CTRL0_BYTE_ORDER_Pos    25
#define DMA_CH2_CTRL0_DISABLE_INT_ENABLE_Pos 24
#define DMA_CH2_CTRL0_ERROR_INT_ENABLE_Pos 23
#define DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_Pos 22
#define DMA_CH2_CTRL0_COUNTER_INT_ENABLE_Pos 21
#define DMA_CH2_CTRL0_START_INT_ENABLE_Pos 20
#define DMA_CH2_CTRL0_DEST_WORD_SIZE_Pos 18
#define DMA_CH2_CTRL0_DEST_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH2_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH2_CTRL0_SRC_WORD_SIZE_Pos 16
#define DMA_CH2_CTRL0_SRC_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH2_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH2_CTRL0_DEST_SELECT_Pos   12
#define DMA_CH2_CTRL0_DEST_SELECT_Mask  ((uint32_t)(0xFU << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_CTRL0_SRC_SELECT_Pos    8
#define DMA_CH2_CTRL0_SRC_SELECT_Mask   ((uint32_t)(0xFU << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos 6
#define DMA_CH2_CTRL0_CHANNEL_PRIORITY_Mask ((uint32_t)(0x3U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH2_CTRL0_TRANSFER_TYPE_Pos 4
#define DMA_CH2_CTRL0_TRANSFER_TYPE_Mask ((uint32_t)(0x3U << DMA_CH2_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH2_CTRL0_DEST_ADDR_INC_Pos 3
#define DMA_CH2_CTRL0_SRC_ADDR_INC_Pos  2
#define DMA_CH2_CTRL0_ADDR_MODE_Pos     1
#define DMA_CH2_CTRL0_ENABLE_Pos        0

/* DMA_CH2_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ENABLE_ALIAS;         /* Enable DMA channel 2 */
    __IO uint32_t ADDR_MODE_ALIAS;      /* Select the addressing mode for this channel */
    __IO uint32_t SRC_ADDR_INC_ALIAS;   /* Configure whether the source address will be incremented */
    __IO uint32_t DEST_ADDR_INC_ALIAS;  /* Configure whether the destination address will be incremented */
         uint32_t RESERVED0[16];
    __IO uint32_t START_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer starts */
    __IO uint32_t COUNTER_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer reaches the counter value */
    __IO uint32_t COMPLETE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer completes */
    __IO uint32_t ERROR_INT_ENABLE_ALIAS;/* Raise an interrupt when a state machine error occurs during a DMA transfer */
    __IO uint32_t DISABLE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA channel is disabled */
    __IO uint32_t BYTE_ORDER_ALIAS;     /* Select the byte ordering for the DMA channel */
} DMA_CH2_CTRL0_Type;

#define DMA_CH2_CTRL0                   ((DMA_CH2_CTRL0_Type *) SYS_CALC_BITBAND(DMA_CH2_CTRL0_BASE, 0))

/* DMA_CH2_CTRL0 settings */
#define DMA_CH2_DISABLE_BITBAND         0x0
#define DMA_CH2_ENABLE_BITBAND          0x1
#define DMA_CH2_DISABLE                 ((uint32_t)(DMA_CH2_DISABLE_BITBAND << DMA_CH2_CTRL0_ENABLE_Pos))
#define DMA_CH2_ENABLE                  ((uint32_t)(DMA_CH2_ENABLE_BITBAND << DMA_CH2_CTRL0_ENABLE_Pos))

#define DMA_CH2_ADDR_CIRC_BITBAND       0x0
#define DMA_CH2_ADDR_LIN_BITBAND        0x1
#define DMA_CH2_ADDR_CIRC               ((uint32_t)(DMA_CH2_ADDR_CIRC_BITBAND << DMA_CH2_CTRL0_ADDR_MODE_Pos))
#define DMA_CH2_ADDR_LIN                ((uint32_t)(DMA_CH2_ADDR_LIN_BITBAND << DMA_CH2_CTRL0_ADDR_MODE_Pos))

#define DMA_CH2_SRC_ADDR_STATIC_BITBAND 0x0
#define DMA_CH2_SRC_ADDR_INC_BITBAND    0x1
#define DMA_CH2_SRC_ADDR_STATIC         ((uint32_t)(DMA_CH2_SRC_ADDR_STATIC_BITBAND << DMA_CH2_CTRL0_SRC_ADDR_INC_Pos))
#define DMA_CH2_SRC_ADDR_INC            ((uint32_t)(DMA_CH2_SRC_ADDR_INC_BITBAND << DMA_CH2_CTRL0_SRC_ADDR_INC_Pos))

#define DMA_CH2_DEST_ADDR_STATIC_BITBAND 0x0
#define DMA_CH2_DEST_ADDR_INC_BITBAND   0x1
#define DMA_CH2_DEST_ADDR_STATIC        ((uint32_t)(DMA_CH2_DEST_ADDR_STATIC_BITBAND << DMA_CH2_CTRL0_DEST_ADDR_INC_Pos))
#define DMA_CH2_DEST_ADDR_INC           ((uint32_t)(DMA_CH2_DEST_ADDR_INC_BITBAND << DMA_CH2_CTRL0_DEST_ADDR_INC_Pos))

#define DMA_CH2_TRANSFER_M_TO_M         ((uint32_t)(0x0U << DMA_CH2_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH2_TRANSFER_M_TO_P         ((uint32_t)(0x1U << DMA_CH2_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH2_TRANSFER_P_TO_M         ((uint32_t)(0x2U << DMA_CH2_CTRL0_TRANSFER_TYPE_Pos))

#define DMA_CH2_PRIORITY_0              ((uint32_t)(0x0U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH2_PRIORITY_1              ((uint32_t)(0x1U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH2_PRIORITY_2              ((uint32_t)(0x2U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH2_PRIORITY_3              ((uint32_t)(0x3U << DMA_CH2_CTRL0_CHANNEL_PRIORITY_Pos))

#define DMA_CH2_SRC_ADC                 ((uint32_t)(0x0U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_USB                 ((uint32_t)(0x3U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_PCM                 ((uint32_t)(0x6U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_SPI0                ((uint32_t)(0x7U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_SPI1                ((uint32_t)(0x8U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_I2C                 ((uint32_t)(0x9U << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_UART0               ((uint32_t)(0xAU << DMA_CH2_CTRL0_SRC_SELECT_Pos))
#define DMA_CH2_SRC_UART1               ((uint32_t)(0xBU << DMA_CH2_CTRL0_SRC_SELECT_Pos))

#define DMA_CH2_DEST_DAC0               ((uint32_t)(0x1U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_DAC1               ((uint32_t)(0x2U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_USB                ((uint32_t)(0x3U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_PCM                ((uint32_t)(0x6U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_SPI0               ((uint32_t)(0x7U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_SPI1               ((uint32_t)(0x8U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_I2C                ((uint32_t)(0x9U << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_UART0              ((uint32_t)(0xAU << DMA_CH2_CTRL0_DEST_SELECT_Pos))
#define DMA_CH2_DEST_UART1              ((uint32_t)(0xBU << DMA_CH2_CTRL0_DEST_SELECT_Pos))

#define DMA_CH2_SRC_WORD_SIZE_8         ((uint32_t)(0x0U << DMA_CH2_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH2_SRC_WORD_SIZE_16        ((uint32_t)(0x1U << DMA_CH2_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH2_SRC_WORD_SIZE_32        ((uint32_t)(0x2U << DMA_CH2_CTRL0_SRC_WORD_SIZE_Pos))

#define DMA_CH2_DEST_WORD_SIZE_8        ((uint32_t)(0x0U << DMA_CH2_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH2_DEST_WORD_SIZE_16       ((uint32_t)(0x1U << DMA_CH2_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH2_DEST_WORD_SIZE_32       ((uint32_t)(0x2U << DMA_CH2_CTRL0_DEST_WORD_SIZE_Pos))

#define DMA_CH2_START_INT_DISABLE_BITBAND 0x0
#define DMA_CH2_START_INT_ENABLE_BITBAND 0x1
#define DMA_CH2_START_INT_DISABLE       ((uint32_t)(DMA_CH2_START_INT_DISABLE_BITBAND << DMA_CH2_CTRL0_START_INT_ENABLE_Pos))
#define DMA_CH2_START_INT_ENABLE        ((uint32_t)(DMA_CH2_START_INT_ENABLE_BITBAND << DMA_CH2_CTRL0_START_INT_ENABLE_Pos))

#define DMA_CH2_COUNTER_INT_DISABLE_BITBAND 0x0
#define DMA_CH2_COUNTER_INT_ENABLE_BITBAND 0x1
#define DMA_CH2_COUNTER_INT_DISABLE     ((uint32_t)(DMA_CH2_COUNTER_INT_DISABLE_BITBAND << DMA_CH2_CTRL0_COUNTER_INT_ENABLE_Pos))
#define DMA_CH2_COUNTER_INT_ENABLE      ((uint32_t)(DMA_CH2_COUNTER_INT_ENABLE_BITBAND << DMA_CH2_CTRL0_COUNTER_INT_ENABLE_Pos))

#define DMA_CH2_COMPLETE_INT_DISABLE_BITBAND 0x0
#define DMA_CH2_COMPLETE_INT_ENABLE_BITBAND 0x1
#define DMA_CH2_COMPLETE_INT_DISABLE    ((uint32_t)(DMA_CH2_COMPLETE_INT_DISABLE_BITBAND << DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_Pos))
#define DMA_CH2_COMPLETE_INT_ENABLE     ((uint32_t)(DMA_CH2_COMPLETE_INT_ENABLE_BITBAND << DMA_CH2_CTRL0_COMPLETE_INT_ENABLE_Pos))

#define DMA_CH2_ERROR_INT_DISABLE_BITBAND 0x0
#define DMA_CH2_ERROR_INT_ENABLE_BITBAND 0x1
#define DMA_CH2_ERROR_INT_DISABLE       ((uint32_t)(DMA_CH2_ERROR_INT_DISABLE_BITBAND << DMA_CH2_CTRL0_ERROR_INT_ENABLE_Pos))
#define DMA_CH2_ERROR_INT_ENABLE        ((uint32_t)(DMA_CH2_ERROR_INT_ENABLE_BITBAND << DMA_CH2_CTRL0_ERROR_INT_ENABLE_Pos))

#define DMA_CH2_DISABLE_INT_DISABLE_BITBAND 0x0
#define DMA_CH2_DISABLE_INT_ENABLE_BITBAND 0x1
#define DMA_CH2_DISABLE_INT_DISABLE     ((uint32_t)(DMA_CH2_DISABLE_INT_DISABLE_BITBAND << DMA_CH2_CTRL0_DISABLE_INT_ENABLE_Pos))
#define DMA_CH2_DISABLE_INT_ENABLE      ((uint32_t)(DMA_CH2_DISABLE_INT_ENABLE_BITBAND << DMA_CH2_CTRL0_DISABLE_INT_ENABLE_Pos))

#define DMA_CH2_LITTLE_ENDIAN_BITBAND   0x0
#define DMA_CH2_BIG_ENDIAN_BITBAND      0x1
#define DMA_CH2_LITTLE_ENDIAN           ((uint32_t)(DMA_CH2_LITTLE_ENDIAN_BITBAND << DMA_CH2_CTRL0_BYTE_ORDER_Pos))
#define DMA_CH2_BIG_ENDIAN              ((uint32_t)(DMA_CH2_BIG_ENDIAN_BITBAND << DMA_CH2_CTRL0_BYTE_ORDER_Pos))

/* DMA channel 2 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 2 */
#define DMA_CH2_SRC_BASE_ADDR_BASE      0x40000244

/* DMA channel 2 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   2 */
#define DMA_CH2_DEST_BASE_ADDR_BASE     0x40000248

/* DMA channel 2 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH2_CTRL1_BASE              0x4000024C

/* DMA_CH2_CTRL1 bit positions */
#define DMA_CH2_CTRL1_COUNTER_INT_VALUE_Pos 16
#define DMA_CH2_CTRL1_COUNTER_INT_VALUE_Mask ((uint32_t)(0xFFFFU << DMA_CH2_CTRL1_COUNTER_INT_VALUE_Pos))
#define DMA_CH2_CTRL1_TRANSFER_LENGTH_Pos 0
#define DMA_CH2_CTRL1_TRANSFER_LENGTH_Mask ((uint32_t)(0xFFFFU << DMA_CH2_CTRL1_TRANSFER_LENGTH_Pos))

/* DMA_CH2_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t TRANSFER_LENGTH_SHORT;
    __IO uint16_t COUNTER_INT_VALUE_SHORT;
} DMA_CH2_CTRL1_Type;

#define DMA_CH2_CTRL1                   ((DMA_CH2_CTRL1_Type *) DMA_CH2_CTRL1_BASE)

/* DMA channel 2 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 2 is
 *   loaded from */
#define DMA_CH2_NEXT_SRC_ADDR_BASE      0x40000250

/* DMA channel 2 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 2 will be
 *   stored */
#define DMA_CH2_NEXT_DEST_ADDR_BASE     0x40000254

/* DMA channel 2 Word Count */
/*   The number of words that have been transferred using DMA channel 2 during
 *   the current transfer */
#define DMA_CH2_WORD_CNT_BASE           0x40000258

/* DMA channel 3 Control and Configuration */
/*   Control and configuration of DMA channel 3, specifying the source and
 *   destination configurations for this channel */
#define DMA_CH3_CTRL0_BASE              0x40000260

/* DMA_CH3_CTRL0 bit positions */
#define DMA_CH3_CTRL0_BYTE_ORDER_Pos    25
#define DMA_CH3_CTRL0_DISABLE_INT_ENABLE_Pos 24
#define DMA_CH3_CTRL0_ERROR_INT_ENABLE_Pos 23
#define DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_Pos 22
#define DMA_CH3_CTRL0_COUNTER_INT_ENABLE_Pos 21
#define DMA_CH3_CTRL0_START_INT_ENABLE_Pos 20
#define DMA_CH3_CTRL0_DEST_WORD_SIZE_Pos 18
#define DMA_CH3_CTRL0_DEST_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH3_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH3_CTRL0_SRC_WORD_SIZE_Pos 16
#define DMA_CH3_CTRL0_SRC_WORD_SIZE_Mask ((uint32_t)(0x3U << DMA_CH3_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH3_CTRL0_DEST_SELECT_Pos   12
#define DMA_CH3_CTRL0_DEST_SELECT_Mask  ((uint32_t)(0xFU << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_CTRL0_SRC_SELECT_Pos    8
#define DMA_CH3_CTRL0_SRC_SELECT_Mask   ((uint32_t)(0xFU << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos 6
#define DMA_CH3_CTRL0_CHANNEL_PRIORITY_Mask ((uint32_t)(0x3U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH3_CTRL0_TRANSFER_TYPE_Pos 4
#define DMA_CH3_CTRL0_TRANSFER_TYPE_Mask ((uint32_t)(0x3U << DMA_CH3_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH3_CTRL0_DEST_ADDR_INC_Pos 3
#define DMA_CH3_CTRL0_SRC_ADDR_INC_Pos  2
#define DMA_CH3_CTRL0_ADDR_MODE_Pos     1
#define DMA_CH3_CTRL0_ENABLE_Pos        0

/* DMA_CH3_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ENABLE_ALIAS;         /* Enable DMA channel 3 */
    __IO uint32_t ADDR_MODE_ALIAS;      /* Select the addressing mode for this channel */
    __IO uint32_t SRC_ADDR_INC_ALIAS;   /* Configure whether the source address will be incremented */
    __IO uint32_t DEST_ADDR_INC_ALIAS;  /* Configure whether the destination address will be incremented */
         uint32_t RESERVED0[16];
    __IO uint32_t START_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer starts */
    __IO uint32_t COUNTER_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer reaches the counter value */
    __IO uint32_t COMPLETE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA transfer completes */
    __IO uint32_t ERROR_INT_ENABLE_ALIAS;/* Raise an interrupt when a state machine error occurs during a DMA transfer */
    __IO uint32_t DISABLE_INT_ENABLE_ALIAS;/* Raise an interrupt when the DMA channel is disabled */
    __IO uint32_t BYTE_ORDER_ALIAS;     /* Select the byte ordering for the DMA channel */
} DMA_CH3_CTRL0_Type;

#define DMA_CH3_CTRL0                   ((DMA_CH3_CTRL0_Type *) SYS_CALC_BITBAND(DMA_CH3_CTRL0_BASE, 0))

/* DMA_CH3_CTRL0 settings */
#define DMA_CH3_DISABLE_BITBAND         0x0
#define DMA_CH3_ENABLE_BITBAND          0x1
#define DMA_CH3_DISABLE                 ((uint32_t)(DMA_CH3_DISABLE_BITBAND << DMA_CH3_CTRL0_ENABLE_Pos))
#define DMA_CH3_ENABLE                  ((uint32_t)(DMA_CH3_ENABLE_BITBAND << DMA_CH3_CTRL0_ENABLE_Pos))

#define DMA_CH3_ADDR_CIRC_BITBAND       0x0
#define DMA_CH3_ADDR_LIN_BITBAND        0x1
#define DMA_CH3_ADDR_CIRC               ((uint32_t)(DMA_CH3_ADDR_CIRC_BITBAND << DMA_CH3_CTRL0_ADDR_MODE_Pos))
#define DMA_CH3_ADDR_LIN                ((uint32_t)(DMA_CH3_ADDR_LIN_BITBAND << DMA_CH3_CTRL0_ADDR_MODE_Pos))

#define DMA_CH3_SRC_ADDR_STATIC_BITBAND 0x0
#define DMA_CH3_SRC_ADDR_INC_BITBAND    0x1
#define DMA_CH3_SRC_ADDR_STATIC         ((uint32_t)(DMA_CH3_SRC_ADDR_STATIC_BITBAND << DMA_CH3_CTRL0_SRC_ADDR_INC_Pos))
#define DMA_CH3_SRC_ADDR_INC            ((uint32_t)(DMA_CH3_SRC_ADDR_INC_BITBAND << DMA_CH3_CTRL0_SRC_ADDR_INC_Pos))

#define DMA_CH3_DEST_ADDR_STATIC_BITBAND 0x0
#define DMA_CH3_DEST_ADDR_INC_BITBAND   0x1
#define DMA_CH3_DEST_ADDR_STATIC        ((uint32_t)(DMA_CH3_DEST_ADDR_STATIC_BITBAND << DMA_CH3_CTRL0_DEST_ADDR_INC_Pos))
#define DMA_CH3_DEST_ADDR_INC           ((uint32_t)(DMA_CH3_DEST_ADDR_INC_BITBAND << DMA_CH3_CTRL0_DEST_ADDR_INC_Pos))

#define DMA_CH3_TRANSFER_M_TO_M         ((uint32_t)(0x0U << DMA_CH3_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH3_TRANSFER_M_TO_P         ((uint32_t)(0x1U << DMA_CH3_CTRL0_TRANSFER_TYPE_Pos))
#define DMA_CH3_TRANSFER_P_TO_M         ((uint32_t)(0x2U << DMA_CH3_CTRL0_TRANSFER_TYPE_Pos))

#define DMA_CH3_PRIORITY_0              ((uint32_t)(0x0U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH3_PRIORITY_1              ((uint32_t)(0x1U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH3_PRIORITY_2              ((uint32_t)(0x2U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos))
#define DMA_CH3_PRIORITY_3              ((uint32_t)(0x3U << DMA_CH3_CTRL0_CHANNEL_PRIORITY_Pos))

#define DMA_CH3_SRC_ADC                 ((uint32_t)(0x0U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_USB                 ((uint32_t)(0x3U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_PCM                 ((uint32_t)(0x6U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_SPI0                ((uint32_t)(0x7U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_SPI1                ((uint32_t)(0x8U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_I2C                 ((uint32_t)(0x9U << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_UART0               ((uint32_t)(0xAU << DMA_CH3_CTRL0_SRC_SELECT_Pos))
#define DMA_CH3_SRC_UART1               ((uint32_t)(0xBU << DMA_CH3_CTRL0_SRC_SELECT_Pos))

#define DMA_CH3_DEST_DAC0               ((uint32_t)(0x1U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_DAC1               ((uint32_t)(0x2U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_USB                ((uint32_t)(0x3U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_PCM                ((uint32_t)(0x6U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_SPI0               ((uint32_t)(0x7U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_SPI1               ((uint32_t)(0x8U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_I2C                ((uint32_t)(0x9U << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_UART0              ((uint32_t)(0xAU << DMA_CH3_CTRL0_DEST_SELECT_Pos))
#define DMA_CH3_DEST_UART1              ((uint32_t)(0xBU << DMA_CH3_CTRL0_DEST_SELECT_Pos))

#define DMA_CH3_SRC_WORD_SIZE_8         ((uint32_t)(0x0U << DMA_CH3_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH3_SRC_WORD_SIZE_16        ((uint32_t)(0x1U << DMA_CH3_CTRL0_SRC_WORD_SIZE_Pos))
#define DMA_CH3_SRC_WORD_SIZE_32        ((uint32_t)(0x2U << DMA_CH3_CTRL0_SRC_WORD_SIZE_Pos))

#define DMA_CH3_DEST_WORD_SIZE_8        ((uint32_t)(0x0U << DMA_CH3_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH3_DEST_WORD_SIZE_16       ((uint32_t)(0x1U << DMA_CH3_CTRL0_DEST_WORD_SIZE_Pos))
#define DMA_CH3_DEST_WORD_SIZE_32       ((uint32_t)(0x2U << DMA_CH3_CTRL0_DEST_WORD_SIZE_Pos))

#define DMA_CH3_START_INT_DISABLE_BITBAND 0x0
#define DMA_CH3_START_INT_ENABLE_BITBAND 0x1
#define DMA_CH3_START_INT_DISABLE       ((uint32_t)(DMA_CH3_START_INT_DISABLE_BITBAND << DMA_CH3_CTRL0_START_INT_ENABLE_Pos))
#define DMA_CH3_START_INT_ENABLE        ((uint32_t)(DMA_CH3_START_INT_ENABLE_BITBAND << DMA_CH3_CTRL0_START_INT_ENABLE_Pos))

#define DMA_CH3_COUNTER_INT_DISABLE_BITBAND 0x0
#define DMA_CH3_COUNTER_INT_ENABLE_BITBAND 0x1
#define DMA_CH3_COUNTER_INT_DISABLE     ((uint32_t)(DMA_CH3_COUNTER_INT_DISABLE_BITBAND << DMA_CH3_CTRL0_COUNTER_INT_ENABLE_Pos))
#define DMA_CH3_COUNTER_INT_ENABLE      ((uint32_t)(DMA_CH3_COUNTER_INT_ENABLE_BITBAND << DMA_CH3_CTRL0_COUNTER_INT_ENABLE_Pos))

#define DMA_CH3_COMPLETE_INT_DISABLE_BITBAND 0x0
#define DMA_CH3_COMPLETE_INT_ENABLE_BITBAND 0x1
#define DMA_CH3_COMPLETE_INT_DISABLE    ((uint32_t)(DMA_CH3_COMPLETE_INT_DISABLE_BITBAND << DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_Pos))
#define DMA_CH3_COMPLETE_INT_ENABLE     ((uint32_t)(DMA_CH3_COMPLETE_INT_ENABLE_BITBAND << DMA_CH3_CTRL0_COMPLETE_INT_ENABLE_Pos))

#define DMA_CH3_ERROR_INT_DISABLE_BITBAND 0x0
#define DMA_CH3_ERROR_INT_ENABLE_BITBAND 0x1
#define DMA_CH3_ERROR_INT_DISABLE       ((uint32_t)(DMA_CH3_ERROR_INT_DISABLE_BITBAND << DMA_CH3_CTRL0_ERROR_INT_ENABLE_Pos))
#define DMA_CH3_ERROR_INT_ENABLE        ((uint32_t)(DMA_CH3_ERROR_INT_ENABLE_BITBAND << DMA_CH3_CTRL0_ERROR_INT_ENABLE_Pos))

#define DMA_CH3_DISABLE_INT_DISABLE_BITBAND 0x0
#define DMA_CH3_DISABLE_INT_ENABLE_BITBAND 0x1
#define DMA_CH3_DISABLE_INT_DISABLE     ((uint32_t)(DMA_CH3_DISABLE_INT_DISABLE_BITBAND << DMA_CH3_CTRL0_DISABLE_INT_ENABLE_Pos))
#define DMA_CH3_DISABLE_INT_ENABLE      ((uint32_t)(DMA_CH3_DISABLE_INT_ENABLE_BITBAND << DMA_CH3_CTRL0_DISABLE_INT_ENABLE_Pos))

#define DMA_CH3_LITTLE_ENDIAN_BITBAND   0x0
#define DMA_CH3_BIG_ENDIAN_BITBAND      0x1
#define DMA_CH3_LITTLE_ENDIAN           ((uint32_t)(DMA_CH3_LITTLE_ENDIAN_BITBAND << DMA_CH3_CTRL0_BYTE_ORDER_Pos))
#define DMA_CH3_BIG_ENDIAN              ((uint32_t)(DMA_CH3_BIG_ENDIAN_BITBAND << DMA_CH3_CTRL0_BYTE_ORDER_Pos))

/* DMA channel 3 Source Base Address */
/*   Base address for the source of data transferred using DMA channel 3 */
#define DMA_CH3_SRC_BASE_ADDR_BASE      0x40000264

/* DMA channel 3 Destination Base Address */
/*   Base address for the destination of data transferred using DMA channel
 *   3 */
#define DMA_CH3_DEST_BASE_ADDR_BASE     0x40000268

/* DMA channel 3 Transfer Control */
/*   Control the length of and how often interrupts are raised for a DMA
 *   transfer */
#define DMA_CH3_CTRL1_BASE              0x4000026C

/* DMA_CH3_CTRL1 bit positions */
#define DMA_CH3_CTRL1_COUNTER_INT_VALUE_Pos 16
#define DMA_CH3_CTRL1_COUNTER_INT_VALUE_Mask ((uint32_t)(0xFFFFU << DMA_CH3_CTRL1_COUNTER_INT_VALUE_Pos))
#define DMA_CH3_CTRL1_TRANSFER_LENGTH_Pos 0
#define DMA_CH3_CTRL1_TRANSFER_LENGTH_Mask ((uint32_t)(0xFFFFU << DMA_CH3_CTRL1_TRANSFER_LENGTH_Pos))

/* DMA_CH3_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t TRANSFER_LENGTH_SHORT;
    __IO uint16_t COUNTER_INT_VALUE_SHORT;
} DMA_CH3_CTRL1_Type;

#define DMA_CH3_CTRL1                   ((DMA_CH3_CTRL1_Type *) DMA_CH3_CTRL1_BASE)

/* DMA channel 3 Next Source Address */
/*   Address where the next data to be transferred using DMA channel 3 is
 *   loaded from */
#define DMA_CH3_NEXT_SRC_ADDR_BASE      0x40000270

/* DMA channel 3 Next Destination Address */
/*   Address where the next data to be transferred using DMA channel 3 will be
 *   stored */
#define DMA_CH3_NEXT_DEST_ADDR_BASE     0x40000274

/* DMA channel 3 Word Count */
/*   The number of words that have been transferred using DMA channel 3 during
 *   the current transfer */
#define DMA_CH3_WORD_CNT_BASE           0x40000278

/* DMA Request Control for DAC0 */
/*   Control the DAC0 DMA transfers and interrupts */
#define DMA_DAC0_REQUEST_BASE           0x40000280

/* DMA_DAC0_REQUEST bit positions */
#define DMA_DAC0_REQUEST_DMA_ENABLE_Pos 16
#define DMA_DAC0_REQUEST_RATE_Pos       0
#define DMA_DAC0_REQUEST_RATE_Mask      ((uint32_t)(0x3FFU << DMA_DAC0_REQUEST_RATE_Pos))

/* DMA_DAC0_REQUEST sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0[16];
    __IO uint32_t DMA_ENABLE_ALIAS;     /* Select whether DMA requests are enabled for the DAC0 request line */
} DMA_DAC0_REQUEST_Type;

#define DMA_DAC0_REQUEST                ((DMA_DAC0_REQUEST_Type *) SYS_CALC_BITBAND(DMA_DAC0_REQUEST_BASE, 0))

/* DMA_DAC0_REQUEST settings */
#define DMA_DAC0_REQUEST_DISABLE_BITBAND 0x0
#define DMA_DAC0_REQUEST_ENABLE_BITBAND 0x1
#define DMA_DAC0_REQUEST_DISABLE        ((uint32_t)(DMA_DAC0_REQUEST_DISABLE_BITBAND << DMA_DAC0_REQUEST_DMA_ENABLE_Pos))
#define DMA_DAC0_REQUEST_ENABLE         ((uint32_t)(DMA_DAC0_REQUEST_ENABLE_BITBAND << DMA_DAC0_REQUEST_DMA_ENABLE_Pos))

/* DMA Request Control for DAC1 */
/*   Control the DAC1 DMA transfers and interrupts */
#define DMA_DAC1_REQUEST_BASE           0x40000284

/* DMA_DAC1_REQUEST bit positions */
#define DMA_DAC1_REQUEST_DMA_ENABLE_Pos 16
#define DMA_DAC1_REQUEST_RATE_Pos       0
#define DMA_DAC1_REQUEST_RATE_Mask      ((uint32_t)(0x3FFU << DMA_DAC1_REQUEST_RATE_Pos))

/* DMA_DAC1_REQUEST sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0[16];
    __IO uint32_t DMA_ENABLE_ALIAS;     /* Select whether DMA request are enabled for the DAC1 request line */
} DMA_DAC1_REQUEST_Type;

#define DMA_DAC1_REQUEST                ((DMA_DAC1_REQUEST_Type *) SYS_CALC_BITBAND(DMA_DAC1_REQUEST_BASE, 0))

/* DMA_DAC1_REQUEST settings */
#define DMA_DAC1_REQUEST_DISABLE_BITBAND 0x0
#define DMA_DAC1_REQUEST_ENABLE_BITBAND 0x1
#define DMA_DAC1_REQUEST_DISABLE        ((uint32_t)(DMA_DAC1_REQUEST_DISABLE_BITBAND << DMA_DAC1_REQUEST_DMA_ENABLE_Pos))
#define DMA_DAC1_REQUEST_ENABLE         ((uint32_t)(DMA_DAC1_REQUEST_ENABLE_BITBAND << DMA_DAC1_REQUEST_DMA_ENABLE_Pos))

/* DMA Status */
/*   Status of DMA transfers and interrupts */
#define DMA_STATUS_BASE                 0x40000290

/* DMA_STATUS bit positions */
#define DMA_STATUS_CH3_ERROR_INT_STATUS_Pos 31
#define DMA_STATUS_CH3_COMPLETE_INT_STATUS_Pos 30
#define DMA_STATUS_CH3_COUNTER_INT_STATUS_Pos 29
#define DMA_STATUS_CH3_START_INT_STATUS_Pos 28
#define DMA_STATUS_CH3_DISABLE_INT_STATUS_Pos 27
#define DMA_STATUS_CH3_STATE_Pos        24
#define DMA_STATUS_CH3_STATE_Mask       ((uint32_t)(0x7U << DMA_STATUS_CH3_STATE_Pos))
#define DMA_STATUS_CH2_ERROR_INT_STATUS_Pos 23
#define DMA_STATUS_CH2_COMPLETE_INT_STATUS_Pos 22
#define DMA_STATUS_CH2_COUNTER_INT_STATUS_Pos 21
#define DMA_STATUS_CH2_START_INT_STATUS_Pos 20
#define DMA_STATUS_CH2_DISABLE_INT_STATUS_Pos 19
#define DMA_STATUS_CH2_STATE_Pos        16
#define DMA_STATUS_CH2_STATE_Mask       ((uint32_t)(0x7U << DMA_STATUS_CH2_STATE_Pos))
#define DMA_STATUS_CH1_ERROR_INT_STATUS_Pos 15
#define DMA_STATUS_CH1_COMPLETE_INT_STATUS_Pos 14
#define DMA_STATUS_CH1_COUNTER_INT_STATUS_Pos 13
#define DMA_STATUS_CH1_START_INT_STATUS_Pos 12
#define DMA_STATUS_CH1_DISABLE_INT_STATUS_Pos 11
#define DMA_STATUS_CH1_STATE_Pos        8
#define DMA_STATUS_CH1_STATE_Mask       ((uint32_t)(0x7U << DMA_STATUS_CH1_STATE_Pos))
#define DMA_STATUS_CH0_ERROR_INT_STATUS_Pos 7
#define DMA_STATUS_CH0_COMPLETE_INT_STATUS_Pos 6
#define DMA_STATUS_CH0_COUNTER_INT_STATUS_Pos 5
#define DMA_STATUS_CH0_START_INT_STATUS_Pos 4
#define DMA_STATUS_CH0_DISABLE_INT_STATUS_Pos 3
#define DMA_STATUS_CH0_STATE_Pos        0
#define DMA_STATUS_CH0_STATE_Mask       ((uint32_t)(0x7U << DMA_STATUS_CH0_STATE_Pos))

/* DMA_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t CH0_BYTE;             
    __IO uint8_t CH1_BYTE;             
    __IO uint8_t CH2_BYTE;             
    __IO uint8_t CH3_BYTE;             
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000290, 0) - (0x40000290 + 4)];
         uint32_t RESERVED0[3];
    __IO uint32_t CH0_DISABLE_INT_STATUS_ALIAS;/* Indicate if a channel disable interrupt has occurred on DMA channel 0 */
    __IO uint32_t CH0_START_INT_STATUS_ALIAS;/* Indicate if a start interrupt has occurred on DMA channel 0 */
    __IO uint32_t CH0_COUNTER_INT_STATUS_ALIAS;/* Indicate if a counter interrupt has occurred on DMA channel 0 */
    __IO uint32_t CH0_COMPLETE_INT_STATUS_ALIAS;/* Indicate if a complete interrupt has occurred on DMA channel 0 */
    __IO uint32_t CH0_ERROR_INT_STATUS_ALIAS;/* Indicate if a state machine error interrupt has occurred on DMA channel 0 */
         uint32_t RESERVED1[3];
    __IO uint32_t CH1_DISABLE_INT_STATUS_ALIAS;/* Indicate if a channel disable interrupt has occurred on DMA channel 1 */
    __IO uint32_t CH1_START_INT_STATUS_ALIAS;/* Indicate if a start interrupt has occurred on DMA channel 1 */
    __IO uint32_t CH1_COUNTER_INT_STATUS_ALIAS;/* Indicate if a counter interrupt has occurred on DMA channel 1 */
    __IO uint32_t CH1_COMPLETE_INT_STATUS_ALIAS;/* Indicate if a complete interrupt has occurred on DMA channel 1 */
    __IO uint32_t CH1_ERROR_INT_STATUS_ALIAS;/* Indicate if a state machine error interrupt has occurred on DMA channel 1 */
         uint32_t RESERVED2[3];
    __IO uint32_t CH2_DISABLE_INT_STATUS_ALIAS;/* Indicate if a channel disable interrupt has occurred on DMA channel 2 */
    __IO uint32_t CH2_START_INT_STATUS_ALIAS;/* Indicate if a start interrupt has occurred on DMA channel 2 */
    __IO uint32_t CH2_COUNTER_INT_STATUS_ALIAS;/* Indicate if a counter interrupt has occurred on DMA channel 2 */
    __IO uint32_t CH2_COMPLETE_INT_STATUS_ALIAS;/* Indicate if a complete interrupt has occurred on DMA channel 2 */
    __IO uint32_t CH2_ERROR_INT_STATUS_ALIAS;/* Indicate if a state machine error interrupt has occurred on DMA channel 2 */
         uint32_t RESERVED3[3];
    __IO uint32_t CH3_DISABLE_INT_STATUS_ALIAS;/* Indicate if a channel disable interrupt has occurred on DMA channel 3 */
    __IO uint32_t CH3_START_INT_STATUS_ALIAS;/* Indicate if a start interrupt has occurred on DMA channel 3 */
    __IO uint32_t CH3_COUNTER_INT_STATUS_ALIAS;/* Indicate if a counter interrupt has occurred on DMA channel 3 */
    __IO uint32_t CH3_COMPLETE_INT_STATUS_ALIAS;/* Indicate if a complete interrupt has occurred on DMA channel 3 */
    __IO uint32_t CH3_ERROR_INT_STATUS_ALIAS;/* Indicate if a state machine error interrupt has occurred on DMA channel 3 */
} DMA_STATUS_Type;

#define DMA_STATUS                      ((DMA_STATUS_Type *) DMA_STATUS_BASE)

/* DMA_STATUS settings */
#define DMA_CH0_IDLE                    ((uint32_t)(0x0U << DMA_STATUS_CH0_STATE_Pos))

#define DMA_CH0_DISABLE_INT_BITBAND     0x1
#define DMA_CH0_DISABLE_INT             ((uint32_t)(DMA_CH0_DISABLE_INT_BITBAND << DMA_STATUS_CH0_DISABLE_INT_STATUS_Pos))

#define DMA_CH0_START_INT_BITBAND       0x1
#define DMA_CH0_START_INT               ((uint32_t)(DMA_CH0_START_INT_BITBAND << DMA_STATUS_CH0_START_INT_STATUS_Pos))

#define DMA_CH0_COUNTER_INT_BITBAND     0x1
#define DMA_CH0_COUNTER_INT             ((uint32_t)(DMA_CH0_COUNTER_INT_BITBAND << DMA_STATUS_CH0_COUNTER_INT_STATUS_Pos))

#define DMA_CH0_COMPLETE_INT_BITBAND    0x1
#define DMA_CH0_COMPLETE_INT            ((uint32_t)(DMA_CH0_COMPLETE_INT_BITBAND << DMA_STATUS_CH0_COMPLETE_INT_STATUS_Pos))

#define DMA_CH0_ERROR_INT_BITBAND       0x1
#define DMA_CH0_ERROR_INT               ((uint32_t)(DMA_CH0_ERROR_INT_BITBAND << DMA_STATUS_CH0_ERROR_INT_STATUS_Pos))

#define DMA_CH1_IDLE                    ((uint32_t)(0x0U << DMA_STATUS_CH1_STATE_Pos))

#define DMA_CH1_DISABLE_INT_BITBAND     0x1
#define DMA_CH1_DISABLE_INT             ((uint32_t)(DMA_CH1_DISABLE_INT_BITBAND << DMA_STATUS_CH1_DISABLE_INT_STATUS_Pos))

#define DMA_CH1_START_INT_BITBAND       0x1
#define DMA_CH1_START_INT               ((uint32_t)(DMA_CH1_START_INT_BITBAND << DMA_STATUS_CH1_START_INT_STATUS_Pos))

#define DMA_CH1_COUNTER_INT_BITBAND     0x1
#define DMA_CH1_COUNTER_INT             ((uint32_t)(DMA_CH1_COUNTER_INT_BITBAND << DMA_STATUS_CH1_COUNTER_INT_STATUS_Pos))

#define DMA_CH1_COMPLETE_INT_BITBAND    0x1
#define DMA_CH1_COMPLETE_INT            ((uint32_t)(DMA_CH1_COMPLETE_INT_BITBAND << DMA_STATUS_CH1_COMPLETE_INT_STATUS_Pos))

#define DMA_CH1_ERROR_INT_BITBAND       0x1
#define DMA_CH1_ERROR_INT               ((uint32_t)(DMA_CH1_ERROR_INT_BITBAND << DMA_STATUS_CH1_ERROR_INT_STATUS_Pos))

#define DMA_CH2_IDLE                    ((uint32_t)(0x0U << DMA_STATUS_CH2_STATE_Pos))

#define DMA_CH2_DISABLE_INT_BITBAND     0x1
#define DMA_CH2_DISABLE_INT             ((uint32_t)(DMA_CH2_DISABLE_INT_BITBAND << DMA_STATUS_CH2_DISABLE_INT_STATUS_Pos))

#define DMA_CH2_START_INT_BITBAND       0x1
#define DMA_CH2_START_INT               ((uint32_t)(DMA_CH2_START_INT_BITBAND << DMA_STATUS_CH2_START_INT_STATUS_Pos))

#define DMA_CH2_COUNTER_INT_BITBAND     0x1
#define DMA_CH2_COUNTER_INT             ((uint32_t)(DMA_CH2_COUNTER_INT_BITBAND << DMA_STATUS_CH2_COUNTER_INT_STATUS_Pos))

#define DMA_CH2_COMPLETE_INT_BITBAND    0x1
#define DMA_CH2_COMPLETE_INT            ((uint32_t)(DMA_CH2_COMPLETE_INT_BITBAND << DMA_STATUS_CH2_COMPLETE_INT_STATUS_Pos))

#define DMA_CH2_ERROR_INT_BITBAND       0x1
#define DMA_CH2_ERROR_INT               ((uint32_t)(DMA_CH2_ERROR_INT_BITBAND << DMA_STATUS_CH2_ERROR_INT_STATUS_Pos))

#define DMA_CH3_IDLE                    ((uint32_t)(0x0U << DMA_STATUS_CH3_STATE_Pos))

#define DMA_CH3_DISABLE_INT_BITBAND     0x1
#define DMA_CH3_DISABLE_INT             ((uint32_t)(DMA_CH3_DISABLE_INT_BITBAND << DMA_STATUS_CH3_DISABLE_INT_STATUS_Pos))

#define DMA_CH3_START_INT_BITBAND       0x1
#define DMA_CH3_START_INT               ((uint32_t)(DMA_CH3_START_INT_BITBAND << DMA_STATUS_CH3_START_INT_STATUS_Pos))

#define DMA_CH3_COUNTER_INT_BITBAND     0x1
#define DMA_CH3_COUNTER_INT             ((uint32_t)(DMA_CH3_COUNTER_INT_BITBAND << DMA_STATUS_CH3_COUNTER_INT_STATUS_Pos))

#define DMA_CH3_COMPLETE_INT_BITBAND    0x1
#define DMA_CH3_COMPLETE_INT            ((uint32_t)(DMA_CH3_COMPLETE_INT_BITBAND << DMA_STATUS_CH3_COMPLETE_INT_STATUS_Pos))

#define DMA_CH3_ERROR_INT_BITBAND       0x1
#define DMA_CH3_ERROR_INT               ((uint32_t)(DMA_CH3_ERROR_INT_BITBAND << DMA_STATUS_CH3_ERROR_INT_STATUS_Pos))

/* DMA_STATUS sub-register bit positions */
#define DMA_STATUS_CH0_BYTE_CH0_STATE_Pos 0
#define DMA_STATUS_CH0_BYTE_CH0_STATE_Mask ((uint32_t)(0x7U << DMA_STATUS_CH0_BYTE_CH0_STATE_Pos))
#define DMA_STATUS_CH0_BYTE_CH0_DISABLE_INT_STATUS_Pos 3
#define DMA_STATUS_CH0_BYTE_CH0_START_INT_STATUS_Pos 4
#define DMA_STATUS_CH0_BYTE_CH0_COUNTER_INT_STATUS_Pos 5
#define DMA_STATUS_CH0_BYTE_CH0_COMPLETE_INT_STATUS_Pos 6
#define DMA_STATUS_CH0_BYTE_CH0_ERROR_INT_STATUS_Pos 7
#define DMA_STATUS_CH1_BYTE_CH1_STATE_Pos 0
#define DMA_STATUS_CH1_BYTE_CH1_STATE_Mask ((uint32_t)(0x7U << DMA_STATUS_CH1_BYTE_CH1_STATE_Pos))
#define DMA_STATUS_CH1_BYTE_CH1_DISABLE_INT_STATUS_Pos 3
#define DMA_STATUS_CH1_BYTE_CH1_START_INT_STATUS_Pos 4
#define DMA_STATUS_CH1_BYTE_CH1_COUNTER_INT_STATUS_Pos 5
#define DMA_STATUS_CH1_BYTE_CH1_COMPLETE_INT_STATUS_Pos 6
#define DMA_STATUS_CH1_BYTE_CH1_ERROR_INT_STATUS_Pos 7
#define DMA_STATUS_CH2_BYTE_CH2_STATE_Pos 0
#define DMA_STATUS_CH2_BYTE_CH2_STATE_Mask ((uint32_t)(0x7U << DMA_STATUS_CH2_BYTE_CH2_STATE_Pos))
#define DMA_STATUS_CH2_BYTE_CH2_DISABLE_INT_STATUS_Pos 3
#define DMA_STATUS_CH2_BYTE_CH2_START_INT_STATUS_Pos 4
#define DMA_STATUS_CH2_BYTE_CH2_COUNTER_INT_STATUS_Pos 5
#define DMA_STATUS_CH2_BYTE_CH2_COMPLETE_INT_STATUS_Pos 6
#define DMA_STATUS_CH2_BYTE_CH2_ERROR_INT_STATUS_Pos 7
#define DMA_STATUS_CH3_BYTE_CH3_STATE_Pos 0
#define DMA_STATUS_CH3_BYTE_CH3_STATE_Mask ((uint32_t)(0x7U << DMA_STATUS_CH3_BYTE_CH3_STATE_Pos))
#define DMA_STATUS_CH3_BYTE_CH3_DISABLE_INT_STATUS_Pos 3
#define DMA_STATUS_CH3_BYTE_CH3_START_INT_STATUS_Pos 4
#define DMA_STATUS_CH3_BYTE_CH3_COUNTER_INT_STATUS_Pos 5
#define DMA_STATUS_CH3_BYTE_CH3_COMPLETE_INT_STATUS_Pos 6
#define DMA_STATUS_CH3_BYTE_CH3_ERROR_INT_STATUS_Pos 7

/* DMA_STATUS subregister settings */
#define DMA_CH0_IDLE_BYTE               ((uint8_t)(0x0U << DMA_STATUS_CH0_BYTE_CH0_STATE_Pos))

#define DMA_CH0_DISABLE_INT_BYTE        ((uint8_t)(DMA_CH0_DISABLE_INT_BITBAND << DMA_STATUS_CH0_BYTE_CH0_DISABLE_INT_STATUS_Pos))

#define DMA_CH0_START_INT_BYTE          ((uint8_t)(DMA_CH0_START_INT_BITBAND << DMA_STATUS_CH0_BYTE_CH0_START_INT_STATUS_Pos))

#define DMA_CH0_COUNTER_INT_BYTE        ((uint8_t)(DMA_CH0_COUNTER_INT_BITBAND << DMA_STATUS_CH0_BYTE_CH0_COUNTER_INT_STATUS_Pos))

#define DMA_CH0_COMPLETE_INT_BYTE       ((uint8_t)(DMA_CH0_COMPLETE_INT_BITBAND << DMA_STATUS_CH0_BYTE_CH0_COMPLETE_INT_STATUS_Pos))

#define DMA_CH0_ERROR_INT_BYTE          ((uint8_t)(DMA_CH0_ERROR_INT_BITBAND << DMA_STATUS_CH0_BYTE_CH0_ERROR_INT_STATUS_Pos))

#define DMA_CH1_IDLE_BYTE               ((uint8_t)(0x0U << DMA_STATUS_CH1_BYTE_CH1_STATE_Pos))

#define DMA_CH1_DISABLE_INT_BYTE        ((uint8_t)(DMA_CH1_DISABLE_INT_BITBAND << DMA_STATUS_CH1_BYTE_CH1_DISABLE_INT_STATUS_Pos))

#define DMA_CH1_START_INT_BYTE          ((uint8_t)(DMA_CH1_START_INT_BITBAND << DMA_STATUS_CH1_BYTE_CH1_START_INT_STATUS_Pos))

#define DMA_CH1_COUNTER_INT_BYTE        ((uint8_t)(DMA_CH1_COUNTER_INT_BITBAND << DMA_STATUS_CH1_BYTE_CH1_COUNTER_INT_STATUS_Pos))

#define DMA_CH1_COMPLETE_INT_BYTE       ((uint8_t)(DMA_CH1_COMPLETE_INT_BITBAND << DMA_STATUS_CH1_BYTE_CH1_COMPLETE_INT_STATUS_Pos))

#define DMA_CH1_ERROR_INT_BYTE          ((uint8_t)(DMA_CH1_ERROR_INT_BITBAND << DMA_STATUS_CH1_BYTE_CH1_ERROR_INT_STATUS_Pos))

#define DMA_CH2_IDLE_BYTE               ((uint8_t)(0x0U << DMA_STATUS_CH2_BYTE_CH2_STATE_Pos))

#define DMA_CH2_DISABLE_INT_BYTE        ((uint8_t)(DMA_CH2_DISABLE_INT_BITBAND << DMA_STATUS_CH2_BYTE_CH2_DISABLE_INT_STATUS_Pos))

#define DMA_CH2_START_INT_BYTE          ((uint8_t)(DMA_CH2_START_INT_BITBAND << DMA_STATUS_CH2_BYTE_CH2_START_INT_STATUS_Pos))

#define DMA_CH2_COUNTER_INT_BYTE        ((uint8_t)(DMA_CH2_COUNTER_INT_BITBAND << DMA_STATUS_CH2_BYTE_CH2_COUNTER_INT_STATUS_Pos))

#define DMA_CH2_COMPLETE_INT_BYTE       ((uint8_t)(DMA_CH2_COMPLETE_INT_BITBAND << DMA_STATUS_CH2_BYTE_CH2_COMPLETE_INT_STATUS_Pos))

#define DMA_CH2_ERROR_INT_BYTE          ((uint8_t)(DMA_CH2_ERROR_INT_BITBAND << DMA_STATUS_CH2_BYTE_CH2_ERROR_INT_STATUS_Pos))

#define DMA_CH3_IDLE_BYTE               ((uint8_t)(0x0U << DMA_STATUS_CH3_BYTE_CH3_STATE_Pos))

#define DMA_CH3_DISABLE_INT_BYTE        ((uint8_t)(DMA_CH3_DISABLE_INT_BITBAND << DMA_STATUS_CH3_BYTE_CH3_DISABLE_INT_STATUS_Pos))

#define DMA_CH3_START_INT_BYTE          ((uint8_t)(DMA_CH3_START_INT_BITBAND << DMA_STATUS_CH3_BYTE_CH3_START_INT_STATUS_Pos))

#define DMA_CH3_COUNTER_INT_BYTE        ((uint8_t)(DMA_CH3_COUNTER_INT_BITBAND << DMA_STATUS_CH3_BYTE_CH3_COUNTER_INT_STATUS_Pos))

#define DMA_CH3_COMPLETE_INT_BYTE       ((uint8_t)(DMA_CH3_COMPLETE_INT_BITBAND << DMA_STATUS_CH3_BYTE_CH3_COMPLETE_INT_STATUS_Pos))

#define DMA_CH3_ERROR_INT_BYTE          ((uint8_t)(DMA_CH3_ERROR_INT_BITBAND << DMA_STATUS_CH3_BYTE_CH3_ERROR_INT_STATUS_Pos))

/* ----------------------------------------------------------------------------
 * UART Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The universal asynchronous receiver-transmitter (UART) interface provides a
 * general purpose data connection using an RS-232 transmission protocol.
 *
 * UART uses a standard data format with one start bit, eight data bits and one
 * stop bit. */

typedef struct
{
    __IO uint32_t CTRL;                 /* Configuration, control and status of the UART0 interface */
    __IO uint32_t SPEED_CTRL;           /* Configure the UART0 baud rate (divided from UART0_CLK, potentially including a pre-scaling divisor of 12 based on UART0_CTRL_PRESCALE_ENABLE in UART0_CTRL) */
    __IO uint32_t STATUS;               /* Status of the UART0 interface */
    __IO uint32_t TXDATA;               /* Byte of data to transmit over the UART interface */
    __I  uint32_t RXDATA;               /* Byte of data received from the UART interface */
} UART_Type;

#define UART0_BASE                      0x40000300
#define UART0                           ((UART_Type *) UART0_BASE)

/* UART0 Control and Configuration */
/*   Configuration, control and status of the UART0 interface */
#define UART0_CTRL_BASE                 0x40000300

/* UART_CTRL bit positions */
#define UART_CTRL_CONTROLLER_Pos        2
#define UART_CTRL_ENABLE_Pos            1
#define UART_CTRL_PRESCALE_ENABLE_Pos   0

/* UART_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t PRESCALE_ENABLE_ALIAS;/* Enable or disable a prescale of the supplied UART clock by 12 */
    __IO uint32_t ENABLE_ALIAS;         /* Enable or disable UART0 */
    __IO uint32_t CONTROLLER_ALIAS;     /* Select whether data transfer will be controlled by the ARM Cortex-M3 processor or the DMA for UART0 */
} UART_CTRL_Type;

#define UART0_CTRL                      ((UART_CTRL_Type *) SYS_CALC_BITBAND(UART0_CTRL_BASE, 0))

/* UART_CTRL settings */
#define UART_PRESCALE_DISABLE_BITBAND   0x0
#define UART_PRESCALE_ENABLE_BITBAND    0x1
#define UART_PRESCALE_DISABLE           ((uint32_t)(UART_PRESCALE_DISABLE_BITBAND << UART_CTRL_PRESCALE_ENABLE_Pos))
#define UART_PRESCALE_ENABLE            ((uint32_t)(UART_PRESCALE_ENABLE_BITBAND << UART_CTRL_PRESCALE_ENABLE_Pos))

#define UART_DISABLE_BITBAND            0x0
#define UART_ENABLE_BITBAND             0x1
#define UART_DISABLE                    ((uint32_t)(UART_DISABLE_BITBAND << UART_CTRL_ENABLE_Pos))
#define UART_ENABLE                     ((uint32_t)(UART_ENABLE_BITBAND << UART_CTRL_ENABLE_Pos))

#define UART_CONTROLLER_CM3_BITBAND     0x0
#define UART_CONTROLLER_DMA_BITBAND     0x1
#define UART_CONTROLLER_CM3             ((uint32_t)(UART_CONTROLLER_CM3_BITBAND << UART_CTRL_CONTROLLER_Pos))
#define UART_CONTROLLER_DMA             ((uint32_t)(UART_CONTROLLER_DMA_BITBAND << UART_CTRL_CONTROLLER_Pos))

/* UART0 Baud Rate Configuration */
/*   Configure the UART0 baud rate (divided from UART0_CLK, potentially
 *   including a pre-scaling divisor of 12 based on UART0_CTRL_PRESCALE_ENABLE
 *   in UART0_CTRL) */
#define UART0_SPEED_CTRL_BASE           0x40000304

/* UART0 Status */
/*   Status of the UART0 interface */
#define UART0_STATUS_BASE               0x40000308

/* UART_STATUS bit positions */
#define UART_STATUS_OVERRUN_STATUS_Pos  1

/* UART_STATUS sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0;
    __IO uint32_t OVERRUN_STATUS_ALIAS; /* Indicate that an overrun has occurred when receiving data on the UART0 interface */
} UART_STATUS_Type;

#define UART0_STATUS                    ((UART_STATUS_Type *) SYS_CALC_BITBAND(UART0_STATUS_BASE, 0))

/* UART_STATUS settings */
#define UART_OVERRUN_FALSE_BITBAND      0x0
#define UART_OVERRUN_TRUE_BITBAND       0x1
#define UART_OVERRUN_CLEAR_BITBAND      0x1
#define UART_OVERRUN_FALSE              ((uint32_t)(UART_OVERRUN_FALSE_BITBAND << UART_STATUS_OVERRUN_STATUS_Pos))
#define UART_OVERRUN_TRUE               ((uint32_t)(UART_OVERRUN_TRUE_BITBAND << UART_STATUS_OVERRUN_STATUS_Pos))
#define UART_OVERRUN_CLEAR              ((uint32_t)(UART_OVERRUN_CLEAR_BITBAND << UART_STATUS_OVERRUN_STATUS_Pos))

/* UART0 Transmit Data */
/*   Byte of data to transmit over the UART interface */
#define UART0_TXDATA_BASE               0x4000030C

/* UART0 Receive Data */
/*   Byte of data received from the UART interface */
#define UART0_RXDATA_BASE               0x40000310

#define UART1_BASE                      0x40000400
#define UART1                           ((UART_Type *) UART1_BASE)

/* UART1 Control and Configuration */
/*   Configuration and control of the UART1 interface */
#define UART1_CTRL_BASE                 0x40000400

/* UART_CTRL sub-register and bit-band aliases */
#define UART1_CTRL                      ((UART_CTRL_Type *) SYS_CALC_BITBAND(UART1_CTRL_BASE, 0))

/* UART1 Baud Rate Configuration */
/*   Configure the UART1 baud rate (divided from UART1_CLK, potentially
 *   including a pre-scaling divisor of 12 based on UART1_CTRL_PRESCALE_ENABLE
 *   in UART1_CTRL) */
#define UART1_SPEED_CTRL_BASE           0x40000404

/* UART1 Status */
/*   Status of the UART1 interface */
#define UART1_STATUS_BASE               0x40000408

/* UART_STATUS sub-register and bit-band aliases */
#define UART1_STATUS                    ((UART_STATUS_Type *) SYS_CALC_BITBAND(UART1_STATUS_BASE, 0))

/* UART1 Transmit Data */
/*   Byte of data to transmit over the UART interface */
#define UART1_TXDATA_BASE               0x4000040C

/* UART1 Receive Data */
/*   Byte of data received from the UART interface */
#define UART1_RXDATA_BASE               0x40000410

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

typedef struct
{
    __IO uint32_t IF0_FUNC_SEL;         /* Select the function for GPIO pins 32 to 35 which are  multiplexed with the SPI0  interface and user clocks */
    __IO uint32_t IF0_OUT;              /* Output for GPIO pins 32 to 35 */
    __IO uint32_t IF1_FUNC_SEL;         /* Select the function for GPIO pins 36 to 39 which are  multiplexed with the SPI1 and PCM interfaces */
    __IO uint32_t IF1_OUT;              /* Output for GPIO pins 36 to 39 */
    __IO uint32_t IF23_FUNC_SEL;        /* Select the function for GPIO pins 40 to 41 which are  multiplexed with the UART0 interface, and  GPIO pins 42 to 43 which are  multiplexed with the  UART1 and SQI interfaces */
    __IO uint32_t IF23_OUT;             /* Output for GPIO pins 40 to 43 */
    __IO uint32_t IF_DATA;              /* Collected I/O for GPIO pins 32 to 43 (GPIO interfaces 0, 1, 2 and 3) */
    __IO uint32_t IF4_LCD_FUNC_SEL;     /* Select the function for GPIO pins 0 to 31 which are multiplexed with the segments of the LCD driver */
    __IO uint32_t IF4_LCD_DRV_SEL0;     /* Select the  driver waveform for segments 0 to 7 of the LCD driver */
    __IO uint32_t IF4_LCD_DRV_SEL1;     /* Select the  driver voltage for segments 8 to 15 of the LCD driver */
    __IO uint32_t IF4_LCD_DRV_SEL2;     /* Select the  driver voltage for segments 16 to 23 of the LCD driver */
    __IO uint32_t IF4_LCD_DRV_SEL3;     /* Select the  driver voltage for segments 24 to 27 of the LCD driver */
    __IO uint32_t IF4_LCD_OUT;          /* Output for GPIO pins 0 to 31 */
    __IO uint32_t IF4_LCD_OE;           /* Output enable for GPIO pins 0 to 31 */
    __I  uint32_t IF4_LCD_IN;           /* Input from GPIO pins 0 to 31 */
    __IO uint32_t IF4_LCD_PULLDOWN;     /* Configure whether GPIO pins 0 to 31 use pull-down resistors */
    __IO uint32_t IF4_LCD_CTRL;         /* Control the LCD interface */
    __IO uint32_t IF4_PWM_CTRL;         /* Select PWM functionality for GPIO pins 24 to 27 */
    __IO uint32_t IF5_FUNC_SEL;         /* Select the function for the dedicated wakeup 3V GPIO pins 0 to 3 */
    __IO uint32_t IF5_OUT;              /* Output for 3V GPIO pins 0 to 3 */
    __IO uint32_t IF_INPUT_ENABLE;      /* Enable or disable GPIO inputs for the GPIO interfaces */
         uint32_t RESERVED0[3];
    __IO uint32_t INT_CTRL0;            /* Configure the GPIO general-purpose interrupts 0 and 1 */
    __IO uint32_t INT_CTRL1;            /* Configure the GPIO general-purpose interrupts 2 and 3 */
    __IO uint32_t INT_CTRL2;            /* Configure the GPIO general-purpose interrupts 4 and 5 */
    __IO uint32_t INT_CTRL3;            /* Configure the GPIO general-purpose interrupts 2 and 3 */
} GPIO_Type;

#define GPIO_BASE                       0x40000500
#define GPIO                            ((GPIO_Type *) GPIO_BASE)

/* GPIO IF0 Function Select */
/*   Select the function for GPIO pins 32 to 35 which are  multiplexed with the
 *   SPI0  interface and user clocks */
#define GPIO_IF0_FUNC_SEL_BASE          0x40000500

/* GPIO_IF0_FUNC_SEL bit positions */
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos 15
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos 14
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos 13
#define GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos 12
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos 11
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos 10
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos 9
#define GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos 8
#define GPIO_IF0_FUNC_SEL_INPUT_DATA_Pos 4
#define GPIO_IF0_FUNC_SEL_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF0_FUNC_SEL_INPUT_DATA_Pos))
#define GPIO_IF0_FUNC_SEL_FUNC_SEL_Pos  0
#define GPIO_IF0_FUNC_SEL_FUNC_SEL_Mask ((uint32_t)(0x3U << GPIO_IF0_FUNC_SEL_FUNC_SEL_Pos))

/* GPIO_IF0_FUNC_SEL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t SEL_BYTE;             
    __IO uint8_t PIN_CFG_BYTE;         
         uint8_t RESERVED0[2];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000500, 0) - (0x40000500 + 4)];
         uint32_t RESERVED1[8];
    __IO uint32_t OUTPUT_ENABLE_PIN0_ALIAS;/* Enable output on GPIO IF0, pin 0 */
    __IO uint32_t OUTPUT_ENABLE_PIN1_ALIAS;/* Enable output on GPIO IF0, pin 1 */
    __IO uint32_t OUTPUT_ENABLE_PIN2_ALIAS;/* Enable output on GPIO IF0, pin 2 */
    __IO uint32_t OUTPUT_ENABLE_PIN3_ALIAS;/* Enable output on GPIO IF0, pin 3 */
    __IO uint32_t PULLUP_ENABLE_PIN0_ALIAS;/* Configure the pull-up resistor for GPIO IF0, pin 0 */
    __IO uint32_t PULLUP_ENABLE_PIN1_ALIAS;/* Configure the pull-up resistor for GPIO IF0, pin 1 */
    __IO uint32_t PULLUP_ENABLE_PIN2_ALIAS;/* Configure the pull-up resistor for GPIO IF0, pin 2 */
    __IO uint32_t PULLUP_ENABLE_PIN3_ALIAS;/* Configure the pull-up resistor for GPIO IF0, pin 3 */
} GPIO_IF0_FUNC_SEL_Type;

#define GPIO_IF0_FUNC_SEL               ((GPIO_IF0_FUNC_SEL_Type *) GPIO_IF0_FUNC_SEL_BASE)

/* GPIO_IF0_FUNC_SEL settings */
#define GPIO_IF0_SELECT_SPI0            ((uint32_t)(0x0U << GPIO_IF0_FUNC_SEL_FUNC_SEL_Pos))
#define GPIO_IF0_SELECT_GPIO            ((uint32_t)(0x1U << GPIO_IF0_FUNC_SEL_FUNC_SEL_Pos))
#define GPIO_IF0_SELECT_USR_CLK         ((uint32_t)(0x2U << GPIO_IF0_FUNC_SEL_FUNC_SEL_Pos))

#define GPIO_IF0_PIN0_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN0_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN0_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF0_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF0_PIN0_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF0_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF0_PIN1_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN1_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN1_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF0_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF0_PIN1_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF0_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF0_PIN2_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN2_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN2_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF0_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF0_PIN2_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF0_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF0_PIN3_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN3_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN3_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF0_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF0_PIN3_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF0_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF0_PIN0_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN0_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN0_PULLUP_DISABLE    ((uint32_t)(GPIO_IF0_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF0_PIN0_PULLUP_ENABLE     ((uint32_t)(GPIO_IF0_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF0_PIN1_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN1_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN1_PULLUP_DISABLE    ((uint32_t)(GPIO_IF0_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF0_PIN1_PULLUP_ENABLE     ((uint32_t)(GPIO_IF0_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF0_PIN2_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN2_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN2_PULLUP_DISABLE    ((uint32_t)(GPIO_IF0_PIN2_PULLUP_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos))
#define GPIO_IF0_PIN2_PULLUP_ENABLE     ((uint32_t)(GPIO_IF0_PIN2_PULLUP_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos))

#define GPIO_IF0_PIN3_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF0_PIN3_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF0_PIN3_PULLUP_DISABLE    ((uint32_t)(GPIO_IF0_PIN3_PULLUP_DISABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos))
#define GPIO_IF0_PIN3_PULLUP_ENABLE     ((uint32_t)(GPIO_IF0_PIN3_PULLUP_ENABLE_BITBAND << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos))

/* GPIO_IF0_FUNC_SEL sub-register bit positions */
#define GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Pos 0
#define GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Mask ((uint32_t)(0x3U << GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_Pos 4
#define GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF0_FUNC_SEL_BYTE_INPUT_DATA_Pos))
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos 0
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos 1
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos 2
#define GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos 3
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos 4
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos 5
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos 6
#define GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos 7

/* GPIO_IF0_FUNC_SEL subregister settings */
#define GPIO_IF0_SELECT_SPI0_BYTE       ((uint8_t)(0x0U << GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF0_SELECT_GPIO_BYTE       ((uint8_t)(0x1U << GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF0_SELECT_USR_CLK_BYTE    ((uint8_t)(0x2U << GPIO_IF0_FUNC_SEL_BYTE_FUNC_SEL_Pos))

#define GPIO_IF0_PIN0_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF0_PIN0_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF0_PIN1_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF0_PIN1_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF0_PIN2_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF0_PIN2_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF0_PIN3_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF0_PIN3_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF0_PIN0_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF0_PIN0_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF0_PIN1_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF0_PIN1_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF0_PIN2_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN2_PULLUP_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos))
#define GPIO_IF0_PIN2_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN2_PULLUP_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos))

#define GPIO_IF0_PIN3_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF0_PIN3_PULLUP_DISABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos))
#define GPIO_IF0_PIN3_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF0_PIN3_PULLUP_ENABLE_BITBAND << GPIO_IF0_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos))

/* GPIO IF0 Output */
/*   Output for GPIO pins 32 to 35 */
#define GPIO_IF0_OUT_BASE               0x40000504

/* GPIO_IF0_OUT bit positions */
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN3_Pos 3
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN2_Pos 2
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN1_Pos 1
#define GPIO_IF0_OUT_OUTPUT_DATA_PIN0_Pos 0

/* GPIO_IF0_OUT sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t OUTPUT_DATA_PIN0_ALIAS;/* Output data for GPIO IF0, pin 0 */
    __IO uint32_t OUTPUT_DATA_PIN1_ALIAS;/* Output data for GPIO IF0, pin 1 */
    __IO uint32_t OUTPUT_DATA_PIN2_ALIAS;/* Output data for GPIO IF0, pin 2 */
    __IO uint32_t OUTPUT_DATA_PIN3_ALIAS;/* Output data for GPIO IF0, pin 3 */
} GPIO_IF0_OUT_Type;

#define GPIO_IF0_OUT                    ((GPIO_IF0_OUT_Type *) SYS_CALC_BITBAND(GPIO_IF0_OUT_BASE, 0))

/* GPIO IF1 Function Select */
/*   Select the function for GPIO pins 36 to 39 which are  multiplexed with the
 *   SPI1 and PCM interfaces */
#define GPIO_IF1_FUNC_SEL_BASE          0x40000508

/* GPIO_IF1_FUNC_SEL bit positions */
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos 15
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos 14
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos 13
#define GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos 12
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos 11
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos 10
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos 9
#define GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos 8
#define GPIO_IF1_FUNC_SEL_INPUT_DATA_Pos 4
#define GPIO_IF1_FUNC_SEL_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF1_FUNC_SEL_INPUT_DATA_Pos))
#define GPIO_IF1_FUNC_SEL_FUNC_SEL_Pos  0
#define GPIO_IF1_FUNC_SEL_FUNC_SEL_Mask ((uint32_t)(0x3U << GPIO_IF1_FUNC_SEL_FUNC_SEL_Pos))

/* GPIO_IF1_FUNC_SEL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t SEL_BYTE;             
    __IO uint8_t PIN_CFG_BYTE;         
         uint8_t RESERVED0[2];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000508, 0) - (0x40000508 + 4)];
         uint32_t RESERVED1[8];
    __IO uint32_t OUTPUT_ENABLE_PIN0_ALIAS;/* Enable output on GPIO IF1, pin 0 */
    __IO uint32_t OUTPUT_ENABLE_PIN1_ALIAS;/* Enable output on GPIO IF1, pin 1 */
    __IO uint32_t OUTPUT_ENABLE_PIN2_ALIAS;/* Enable output on GPIO IF1, pin 2 */
    __IO uint32_t OUTPUT_ENABLE_PIN3_ALIAS;/* Enable output on GPIO IF1, pin 3 */
    __IO uint32_t PULLUP_ENABLE_PIN0_ALIAS;/* Configure the pull-up resistor for GPIO IF1, pin 0 */
    __IO uint32_t PULLUP_ENABLE_PIN1_ALIAS;/* Configure the pull-up resistor for GPIO IF1, pin 1 */
    __IO uint32_t PULLUP_ENABLE_PIN2_ALIAS;/* Configure the pull-up resistor for GPIO IF1, pin 2 */
    __IO uint32_t PULLUP_ENABLE_PIN3_ALIAS;/* Configure the pull-up resistor for GPIO IF1, pin 3 */
} GPIO_IF1_FUNC_SEL_Type;

#define GPIO_IF1_FUNC_SEL               ((GPIO_IF1_FUNC_SEL_Type *) GPIO_IF1_FUNC_SEL_BASE)

/* GPIO_IF1_FUNC_SEL settings */
#define GPIO_IF1_SELECT_SPI1            ((uint32_t)(0x0U << GPIO_IF1_FUNC_SEL_FUNC_SEL_Pos))
#define GPIO_IF1_SELECT_GPIO            ((uint32_t)(0x1U << GPIO_IF1_FUNC_SEL_FUNC_SEL_Pos))
#define GPIO_IF1_SELECT_PCM             ((uint32_t)(0x2U << GPIO_IF1_FUNC_SEL_FUNC_SEL_Pos))

#define GPIO_IF1_PIN0_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN0_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN0_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF1_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF1_PIN0_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF1_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF1_PIN1_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN1_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN1_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF1_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF1_PIN1_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF1_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF1_PIN2_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN2_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN2_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF1_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF1_PIN2_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF1_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF1_PIN3_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN3_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN3_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF1_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF1_PIN3_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF1_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF1_PIN0_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN0_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN0_PULLUP_DISABLE    ((uint32_t)(GPIO_IF1_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF1_PIN0_PULLUP_ENABLE     ((uint32_t)(GPIO_IF1_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF1_PIN1_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN1_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN1_PULLUP_DISABLE    ((uint32_t)(GPIO_IF1_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF1_PIN1_PULLUP_ENABLE     ((uint32_t)(GPIO_IF1_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF1_PIN2_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN2_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN2_PULLUP_DISABLE    ((uint32_t)(GPIO_IF1_PIN2_PULLUP_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos))
#define GPIO_IF1_PIN2_PULLUP_ENABLE     ((uint32_t)(GPIO_IF1_PIN2_PULLUP_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos))

#define GPIO_IF1_PIN3_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF1_PIN3_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF1_PIN3_PULLUP_DISABLE    ((uint32_t)(GPIO_IF1_PIN3_PULLUP_DISABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos))
#define GPIO_IF1_PIN3_PULLUP_ENABLE     ((uint32_t)(GPIO_IF1_PIN3_PULLUP_ENABLE_BITBAND << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos))

/* GPIO_IF1_FUNC_SEL sub-register bit positions */
#define GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Pos 0
#define GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Mask ((uint32_t)(0x3U << GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_Pos 4
#define GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF1_FUNC_SEL_BYTE_INPUT_DATA_Pos))
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos 0
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos 1
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos 2
#define GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos 3
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos 4
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos 5
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos 6
#define GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos 7

/* GPIO_IF1_FUNC_SEL subregister settings */
#define GPIO_IF1_SELECT_SPI1_BYTE       ((uint8_t)(0x0U << GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF1_SELECT_GPIO_BYTE       ((uint8_t)(0x1U << GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Pos))
#define GPIO_IF1_SELECT_PCM_BYTE        ((uint8_t)(0x2U << GPIO_IF1_FUNC_SEL_BYTE_FUNC_SEL_Pos))

#define GPIO_IF1_PIN0_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF1_PIN0_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF1_PIN1_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF1_PIN1_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF1_PIN2_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF1_PIN2_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF1_PIN3_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF1_PIN3_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF1_PIN0_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF1_PIN0_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF1_PIN1_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF1_PIN1_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF1_PIN2_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN2_PULLUP_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos))
#define GPIO_IF1_PIN2_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN2_PULLUP_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN2_Pos))

#define GPIO_IF1_PIN3_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF1_PIN3_PULLUP_DISABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos))
#define GPIO_IF1_PIN3_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF1_PIN3_PULLUP_ENABLE_BITBAND << GPIO_IF1_PIN_CFG_BYTE_PULLUP_ENABLE_PIN3_Pos))

/* GPIO IF1 Output */
/*   Output for GPIO pins 36 to 39 */
#define GPIO_IF1_OUT_BASE               0x4000050C

/* GPIO_IF1_OUT bit positions */
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN3_Pos 3
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN2_Pos 2
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN1_Pos 1
#define GPIO_IF1_OUT_OUTPUT_DATA_PIN0_Pos 0

/* GPIO_IF1_OUT sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t OUTPUT_DATA_PIN0_ALIAS;/* Output data for GPIO IF1, pin 0 */
    __IO uint32_t OUTPUT_DATA_PIN1_ALIAS;/* Output data for GPIO IF1, pin 1 */
    __IO uint32_t OUTPUT_DATA_PIN2_ALIAS;/* Output data for GPIO IF1, pin 2 */
    __IO uint32_t OUTPUT_DATA_PIN3_ALIAS;/* Output data for GPIO IF1, pin 3 */
} GPIO_IF1_OUT_Type;

#define GPIO_IF1_OUT                    ((GPIO_IF1_OUT_Type *) SYS_CALC_BITBAND(GPIO_IF1_OUT_BASE, 0))

/* GPIO IF2/IF3 Function Select */
/*   Select the function for GPIO pins 40 to 41 which are  multiplexed with the
 *   UART0 interface, and  GPIO pins 42 to 43 which are  multiplexed with the
 *   UART1 and SQI interfaces */
#define GPIO_IF23_FUNC_SEL_BASE         0x40000510

/* GPIO_IF23_FUNC_SEL bit positions */
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_Pos 15
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_Pos 14
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_Pos 13
#define GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_Pos 12
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_Pos 11
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_Pos 10
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_Pos 9
#define GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_Pos 8
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_Pos 6
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_Pos))
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_Pos 4
#define GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_Pos))
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Pos 1
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Pos))
#define GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_Pos 0

/* GPIO_IF23_FUNC_SEL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t SEL_BYTE;             
    __IO uint8_t PIN_CFG_BYTE;         
         uint8_t RESERVED0[2];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000510, 0) - (0x40000510 + 4)];
    __IO uint32_t FUNC_SEL_IF2_ALIAS;   /* Select the multiplexed function for the GPIO IF2 pins */
         uint32_t RESERVED1[7];
    __IO uint32_t OUTPUT_ENABLE_IF2_PIN0_ALIAS;/* Enable output on GPIO IF2, pin 0 */
    __IO uint32_t OUTPUT_ENABLE_IF2_PIN1_ALIAS;/* Enable output on GPIO IF2, pin 1 */
    __IO uint32_t OUTPUT_ENABLE_IF3_PIN0_ALIAS;/* Enable output on GPIO IF3, pin 0 */
    __IO uint32_t OUTPUT_ENABLE_IF3_PIN1_ALIAS;/* Enable output on GPIO IF3, pin 1 */
    __IO uint32_t PULLUP_ENABLE_IF2_PIN0_ALIAS;/* Configure the pull-up resistor for GPIO IF2, pin 0 */
    __IO uint32_t PULLUP_ENABLE_IF2_PIN1_ALIAS;/* Configure the pull-up resistor for GPIO IF2, pin 1 */
    __IO uint32_t PULLUP_ENABLE_IF3_PIN0_ALIAS;/* Configure the pull-up resistor for GPIO IF3, pin 0 */
    __IO uint32_t PULLUP_ENABLE_IF3_PIN1_ALIAS;/* Configure the pull-up resistor for GPIO IF3, pin 1 */
} GPIO_IF23_FUNC_SEL_Type;

#define GPIO_IF23_FUNC_SEL              ((GPIO_IF23_FUNC_SEL_Type *) GPIO_IF23_FUNC_SEL_BASE)

/* GPIO_IF23_FUNC_SEL settings */
#define GPIO_IF2_SELECT_UART0_BITBAND   0x0
#define GPIO_IF2_SELECT_GPIO_BITBAND    0x1
#define GPIO_IF2_SELECT_UART0           ((uint32_t)(GPIO_IF2_SELECT_UART0_BITBAND << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_Pos))
#define GPIO_IF2_SELECT_GPIO            ((uint32_t)(GPIO_IF2_SELECT_GPIO_BITBAND << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_Pos))

#define GPIO_IF3_SELECT_UART1           ((uint32_t)(0x0U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Pos))
#define GPIO_IF3_SELECT_GPIO            ((uint32_t)(0x1U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Pos))
#define GPIO_IF3_SELECT_SQI             ((uint32_t)(0x2U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Pos))

#define GPIO_IF2_PIN0_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF2_PIN0_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF2_PIN0_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF2_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_Pos))
#define GPIO_IF2_PIN0_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF2_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_Pos))

#define GPIO_IF2_PIN1_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF2_PIN1_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF2_PIN1_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF2_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_Pos))
#define GPIO_IF2_PIN1_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF2_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_Pos))

#define GPIO_IF3_PIN0_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF3_PIN0_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF3_PIN0_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF3_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_Pos))
#define GPIO_IF3_PIN0_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF3_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_Pos))

#define GPIO_IF3_PIN1_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF3_PIN1_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF3_PIN1_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF3_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_Pos))
#define GPIO_IF3_PIN1_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF3_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_Pos))

#define GPIO_IF2_PIN0_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF2_PIN0_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF2_PIN0_PULLUP_DISABLE    ((uint32_t)(GPIO_IF2_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_Pos))
#define GPIO_IF2_PIN0_PULLUP_ENABLE     ((uint32_t)(GPIO_IF2_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_Pos))

#define GPIO_IF2_PIN1_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF2_PIN1_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF2_PIN1_PULLUP_DISABLE    ((uint32_t)(GPIO_IF2_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_Pos))
#define GPIO_IF2_PIN1_PULLUP_ENABLE     ((uint32_t)(GPIO_IF2_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_Pos))

#define GPIO_IF3_PIN0_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF3_PIN0_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF3_PIN0_PULLUP_DISABLE    ((uint32_t)(GPIO_IF3_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_Pos))
#define GPIO_IF3_PIN0_PULLUP_ENABLE     ((uint32_t)(GPIO_IF3_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_Pos))

#define GPIO_IF3_PIN1_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF3_PIN1_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF3_PIN1_PULLUP_DISABLE    ((uint32_t)(GPIO_IF3_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_Pos))
#define GPIO_IF3_PIN1_PULLUP_ENABLE     ((uint32_t)(GPIO_IF3_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_Pos))

/* GPIO_IF23_FUNC_SEL sub-register bit positions */
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF2_Pos 0
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Pos 1
#define GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Pos))
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_Pos 4
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF2_Pos))
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_Pos 6
#define GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_Mask ((uint32_t)(0x3U << GPIO_IF23_FUNC_SEL_BYTE_INPUT_DATA_IF3_Pos))
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN0_Pos 0
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN1_Pos 1
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN0_Pos 2
#define GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN1_Pos 3
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN0_Pos 4
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN1_Pos 5
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN0_Pos 6
#define GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN1_Pos 7

/* GPIO_IF23_FUNC_SEL subregister settings */
#define GPIO_IF2_SELECT_UART0_BYTE      ((uint8_t)(GPIO_IF2_SELECT_UART0_BITBAND << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF2_Pos))
#define GPIO_IF2_SELECT_GPIO_BYTE       ((uint8_t)(GPIO_IF2_SELECT_GPIO_BITBAND << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF2_Pos))

#define GPIO_IF3_SELECT_UART1_BYTE      ((uint8_t)(0x0U << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Pos))
#define GPIO_IF3_SELECT_GPIO_BYTE       ((uint8_t)(0x1U << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Pos))
#define GPIO_IF3_SELECT_SQI_BYTE        ((uint8_t)(0x2U << GPIO_IF23_FUNC_SEL_BYTE_FUNC_SEL_IF3_Pos))

#define GPIO_IF2_PIN0_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF2_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN0_Pos))
#define GPIO_IF2_PIN0_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF2_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN0_Pos))

#define GPIO_IF2_PIN1_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF2_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN1_Pos))
#define GPIO_IF2_PIN1_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF2_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF2_PIN1_Pos))

#define GPIO_IF3_PIN0_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF3_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN0_Pos))
#define GPIO_IF3_PIN0_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF3_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN0_Pos))

#define GPIO_IF3_PIN1_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF3_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN1_Pos))
#define GPIO_IF3_PIN1_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF3_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_OUTPUT_ENABLE_IF3_PIN1_Pos))

#define GPIO_IF2_PIN0_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF2_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN0_Pos))
#define GPIO_IF2_PIN0_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF2_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN0_Pos))

#define GPIO_IF2_PIN1_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF2_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN1_Pos))
#define GPIO_IF2_PIN1_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF2_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF2_PIN1_Pos))

#define GPIO_IF3_PIN0_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF3_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN0_Pos))
#define GPIO_IF3_PIN0_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF3_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN0_Pos))

#define GPIO_IF3_PIN1_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF3_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN1_Pos))
#define GPIO_IF3_PIN1_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF3_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF23_PIN_CFG_BYTE_PULLUP_ENABLE_IF3_PIN1_Pos))

/* GPIO IF2/IF3 Output */
/*   Output for GPIO pins 40 to 43 */
#define GPIO_IF23_OUT_BASE              0x40000514

/* GPIO_IF23_OUT bit positions */
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN1_Pos 3
#define GPIO_IF23_OUT_OUTPUT_DATA_IF3_PIN0_Pos 2
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN1_Pos 1
#define GPIO_IF23_OUT_OUTPUT_DATA_IF2_PIN0_Pos 0

/* GPIO_IF23_OUT sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t OUTPUT_DATA_IF2_PIN0_ALIAS;/* Output data from GPIO IF2, pin 0 */
    __IO uint32_t OUTPUT_DATA_IF2_PIN1_ALIAS;/* Output data from GPIO IF2, pin 1 */
    __IO uint32_t OUTPUT_DATA_IF3_PIN0_ALIAS;/* Output data from GPIO IF3, pin 0 */
    __IO uint32_t OUTPUT_DATA_IF3_PIN1_ALIAS;/* Output data from GPIO IF3, pin 1 */
} GPIO_IF23_OUT_Type;

#define GPIO_IF23_OUT                   ((GPIO_IF23_OUT_Type *) SYS_CALC_BITBAND(GPIO_IF23_OUT_BASE, 0))

/* GPIO IF0 to IF3 Collected Input and Output */
/*   Collected I/O for GPIO pins 32 to 43 (GPIO interfaces 0, 1, 2 and 3) */
#define GPIO_IF_DATA_BASE               0x40000518

/* GPIO_IF_DATA bit positions */
#define GPIO_IF_DATA_IF3_DATA_Pos       10
#define GPIO_IF_DATA_IF3_DATA_Mask      ((uint32_t)(0x3U << GPIO_IF_DATA_IF3_DATA_Pos))
#define GPIO_IF_DATA_IF2_DATA_Pos       8
#define GPIO_IF_DATA_IF2_DATA_Mask      ((uint32_t)(0x3U << GPIO_IF_DATA_IF2_DATA_Pos))
#define GPIO_IF_DATA_IF1_DATA_Pos       4
#define GPIO_IF_DATA_IF1_DATA_Mask      ((uint32_t)(0xFU << GPIO_IF_DATA_IF1_DATA_Pos))
#define GPIO_IF_DATA_IF0_DATA_Pos       0
#define GPIO_IF_DATA_IF0_DATA_Mask      ((uint32_t)(0xFU << GPIO_IF_DATA_IF0_DATA_Pos))

/* GPIO IF4, LCD Driver Function Select */
/*   Select the function for GPIO pins 0 to 31 which are multiplexed with the
 *   segments of the LCD driver */
#define GPIO_IF4_LCD_FUNC_SEL_BASE      0x4000051C

/* GPIO_IF4_LCD_FUNC_SEL bit positions */
#define GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Pos 0
#define GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Mask ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Pos))

/* GPIO_IF4_LCD_FUNC_SEL settings */
#define GPIO_IF4_ALL_GPIO               ((uint32_t)(0x0U << GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Pos))
#define GPIO_IF4_ALL_LCD                ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Pos))

/* GPIO IF4, LCD Driver Select 0 */
/*   Select the  driver waveform for segments 0 to 7 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL0_BASE      0x40000520

/* GPIO_IF4_LCD_DRV_SEL0 bit positions */
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_Pos 28
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN7_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_Pos 24
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN6_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_Pos 20
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN5_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_Pos 16
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN4_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_Pos 12
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN3_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_Pos 8
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN2_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_Pos 4
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN1_Pos))
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_Pos 0
#define GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL0_DRIVER_SELECT_PIN0_Pos))

/* GPIO IF4, LCD Driver Select 1 */
/*   Select the  driver voltage for segments 8 to 15 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL1_BASE      0x40000524

/* GPIO_IF4_LCD_DRV_SEL1 bit positions */
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_Pos 28
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN15_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_Pos 24
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN14_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_Pos 20
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN13_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_Pos 16
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN12_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_Pos 12
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN11_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_Pos 8
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN10_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_Pos 4
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN9_Pos))
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_Pos 0
#define GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL1_DRIVER_SELECT_PIN8_Pos))

/* GPIO IF4, LCD Driver Select 2 */
/*   Select the  driver voltage for segments 16 to 23 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL2_BASE      0x40000528

/* GPIO_IF4_LCD_DRV_SEL2 bit positions */
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_Pos 28
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN23_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_Pos 24
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN22_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_Pos 20
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN21_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_Pos 16
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN20_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_Pos 12
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN19_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_Pos 8
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN18_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_Pos 4
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN17_Pos))
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_Pos 0
#define GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL2_DRIVER_SELECT_PIN16_Pos))

/* GPIO IF4, LCD Driver Select 3 */
/*   Select the  driver voltage for segments 24 to 27 of the LCD driver */
#define GPIO_IF4_LCD_DRV_SEL3_BASE      0x4000052C

/* GPIO_IF4_LCD_DRV_SEL3 bit positions */
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_Pos 12
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN27_Pos))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_Pos 8
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN26_Pos))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_Pos 4
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN25_Pos))
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_Pos 0
#define GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_Mask ((uint32_t)(0xFU << GPIO_IF4_LCD_DRV_SEL3_DRIVER_SELECT_PIN24_Pos))

/* GPIO IF4, LCD Driver Output */
/*   Output for GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_OUT_BASE           0x40000530

/* GPIO_IF4_LCD_OUT bit positions */
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN31_Pos 31
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN30_Pos 30
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN29_Pos 29
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN28_Pos 28
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN27_Pos 27
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN26_Pos 26
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN25_Pos 25
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN24_Pos 24
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN23_Pos 23
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN22_Pos 22
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN21_Pos 21
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN20_Pos 20
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN19_Pos 19
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN18_Pos 18
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN17_Pos 17
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN16_Pos 16
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN15_Pos 15
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN14_Pos 14
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN13_Pos 13
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN12_Pos 12
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN11_Pos 11
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN10_Pos 10
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN9_Pos 9
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN8_Pos 8
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN7_Pos 7
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN6_Pos 6
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN5_Pos 5
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN4_Pos 4
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN3_Pos 3
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN2_Pos 2
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN1_Pos 1
#define GPIO_IF4_LCD_OUT_OUTPUT_DATA_PIN0_Pos 0

/* GPIO_IF4_LCD_OUT sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t OUTPUT_DATA_PIN0_ALIAS;/* Output data for GPIO IF4, pin 0 */
    __IO uint32_t OUTPUT_DATA_PIN1_ALIAS;/* Output data for GPIO IF4, pin 1 */
    __IO uint32_t OUTPUT_DATA_PIN2_ALIAS;/* Output data for GPIO IF4, pin 2 */
    __IO uint32_t OUTPUT_DATA_PIN3_ALIAS;/* Output data for GPIO IF4, pin 3 */
    __IO uint32_t OUTPUT_DATA_PIN4_ALIAS;/* Output data for GPIO IF4, pin 4 */
    __IO uint32_t OUTPUT_DATA_PIN5_ALIAS;/* Output data for GPIO IF4, pin 5 */
    __IO uint32_t OUTPUT_DATA_PIN6_ALIAS;/* Output data for GPIO IF4, pin 6 */
    __IO uint32_t OUTPUT_DATA_PIN7_ALIAS;/* Output data for GPIO IF4, pin 7 */
    __IO uint32_t OUTPUT_DATA_PIN8_ALIAS;/* Output data for GPIO IF4, pin 8 */
    __IO uint32_t OUTPUT_DATA_PIN9_ALIAS;/* Output data for GPIO IF4, pin 9 */
    __IO uint32_t OUTPUT_DATA_PIN10_ALIAS;/* Output data for GPIO IF4, pin 10 */
    __IO uint32_t OUTPUT_DATA_PIN11_ALIAS;/* Output data for GPIO IF4, pin 11 */
    __IO uint32_t OUTPUT_DATA_PIN12_ALIAS;/* Output data for GPIO IF4, pin 12 */
    __IO uint32_t OUTPUT_DATA_PIN13_ALIAS;/* Output data for GPIO IF4, pin 13 */
    __IO uint32_t OUTPUT_DATA_PIN14_ALIAS;/* Output data for GPIO IF4, pin 14 */
    __IO uint32_t OUTPUT_DATA_PIN15_ALIAS;/* Output data for GPIO IF4, pin 15 */
    __IO uint32_t OUTPUT_DATA_PIN16_ALIAS;/* Output data for GPIO IF4, pin 16 */
    __IO uint32_t OUTPUT_DATA_PIN17_ALIAS;/* Output data for GPIO IF4, pin 17 */
    __IO uint32_t OUTPUT_DATA_PIN18_ALIAS;/* Output data for GPIO IF4, pin 18 */
    __IO uint32_t OUTPUT_DATA_PIN19_ALIAS;/* Output data for GPIO IF4, pin 19 */
    __IO uint32_t OUTPUT_DATA_PIN20_ALIAS;/* Output data for GPIO IF4, pin 20 */
    __IO uint32_t OUTPUT_DATA_PIN21_ALIAS;/* Output data for GPIO IF4, pin 21 */
    __IO uint32_t OUTPUT_DATA_PIN22_ALIAS;/* Output data for GPIO IF4, pin 22 */
    __IO uint32_t OUTPUT_DATA_PIN23_ALIAS;/* Output data for GPIO IF4, pin 23 */
    __IO uint32_t OUTPUT_DATA_PIN24_ALIAS;/* Output data for GPIO IF4, pin 24 */
    __IO uint32_t OUTPUT_DATA_PIN25_ALIAS;/* Output data for GPIO IF4, pin 25 */
    __IO uint32_t OUTPUT_DATA_PIN26_ALIAS;/* Output data for GPIO IF4, pin 26 */
    __IO uint32_t OUTPUT_DATA_PIN27_ALIAS;/* Output data for GPIO IF4, pin 27 */
    __IO uint32_t OUTPUT_DATA_PIN28_ALIAS;/* Output data for GPIO IF4, pin 28 */
    __IO uint32_t OUTPUT_DATA_PIN29_ALIAS;/* Output data for GPIO IF4, pin 29 */
    __IO uint32_t OUTPUT_DATA_PIN30_ALIAS;/* Output data for GPIO IF4, pin 30 */
    __IO uint32_t OUTPUT_DATA_PIN31_ALIAS;/* Output data for GPIO IF4, pin 31 */
} GPIO_IF4_LCD_OUT_Type;

#define GPIO_IF4_LCD_OUT                ((GPIO_IF4_LCD_OUT_Type *) SYS_CALC_BITBAND(GPIO_IF4_LCD_OUT_BASE, 0))

/* GPIO IF4, LCD Driver Output Enable */
/*   Output enable for GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_OE_BASE            0x40000534

/* GPIO_IF4_LCD_OE bit positions */
#define GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Pos 0
#define GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Mask ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Pos))

/* GPIO_IF4_LCD_OE settings */
#define GPIO_IF4_OUTPUT_DISABLE_ALL     ((uint32_t)(0x0U << GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Pos))
#define GPIO_IF4_OUTPUT_ENABLE_ALL      ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Pos))

/* GPIO IF4, LCD Driver Input */
/*   Input from GPIO pins 0 to 31 */
#define GPIO_IF4_LCD_IN_BASE            0x40000538

/* GPIO_IF4_LCD_IN bit positions */
#define GPIO_IF4_LCD_IN_INPUT_DATA_Pos  0
#define GPIO_IF4_LCD_IN_INPUT_DATA_Mask ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_IN_INPUT_DATA_Pos))

/* GPIO IF4, LCD Driver Pull-down Configuration */
/*   Configure whether GPIO pins 0 to 31 use pull-down resistors */
#define GPIO_IF4_LCD_PULLDOWN_BASE      0x4000053C

/* GPIO_IF4_LCD_PULLDOWN bit positions */
#define GPIO_IF4_LCD_PULLDOWN_ENABLE_Pos 0
#define GPIO_IF4_LCD_PULLDOWN_ENABLE_Mask ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_PULLDOWN_ENABLE_Pos))

/* GPIO_IF4_LCD_PULLDOWN settings */
#define GPIO_IF4_PULLDOWN_DISABLE_ALL   ((uint32_t)(0x0U << GPIO_IF4_LCD_PULLDOWN_ENABLE_Pos))
#define GPIO_IF4_PULLDOWN_ENABLE_ALL    ((uint32_t)(0xFFFFFFFFU << GPIO_IF4_LCD_PULLDOWN_ENABLE_Pos))

/* GPIO LCD Driver Control */
/*   Control the LCD interface */
#define GPIO_IF4_LCD_CTRL_BASE          0x40000540

/* GPIO_IF4_LCD_CTRL bit positions */
#define GPIO_IF4_LCD_CTRL_BLANK_ENABLE_Pos 1
#define GPIO_IF4_LCD_CTRL_LCD_ENABLE_Pos 0

/* GPIO_IF4_LCD_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t LCD_ENABLE_ALIAS;     /* Enable the LCD driver */
    __IO uint32_t BLANK_ENABLE_ALIAS;   /* Select whether the LCD segment output is blanked */
} GPIO_IF4_LCD_CTRL_Type;

#define GPIO_IF4_LCD_CTRL               ((GPIO_IF4_LCD_CTRL_Type *) SYS_CALC_BITBAND(GPIO_IF4_LCD_CTRL_BASE, 0))

/* GPIO_IF4_LCD_CTRL settings */
#define LCD_DISABLE_BITBAND             0x0
#define LCD_ENABLE_BITBAND              0x1
#define LCD_DISABLE                     ((uint32_t)(LCD_DISABLE_BITBAND << GPIO_IF4_LCD_CTRL_LCD_ENABLE_Pos))
#define LCD_ENABLE                      ((uint32_t)(LCD_ENABLE_BITBAND << GPIO_IF4_LCD_CTRL_LCD_ENABLE_Pos))

#define LCD_BLANK_DISABLE_BITBAND       0x0
#define LCD_BLANK_ENABLE_BITBAND        0x1
#define LCD_BLANK_DISABLE               ((uint32_t)(LCD_BLANK_DISABLE_BITBAND << GPIO_IF4_LCD_CTRL_BLANK_ENABLE_Pos))
#define LCD_BLANK_ENABLE                ((uint32_t)(LCD_BLANK_ENABLE_BITBAND << GPIO_IF4_LCD_CTRL_BLANK_ENABLE_Pos))

/* GPIO Pulse-Width Modulator Control */
/*   Select PWM functionality for GPIO pins 24 to 27 */
#define GPIO_IF4_PWM_CTRL_BASE          0x40000544

/* GPIO_IF4_PWM_CTRL bit positions */
#define GPIO_IF4_PWM_CTRL_PWM3_ENABLE_Pos 3
#define GPIO_IF4_PWM_CTRL_PWM2_ENABLE_Pos 2
#define GPIO_IF4_PWM_CTRL_PWM1_ENABLE_Pos 1
#define GPIO_IF4_PWM_CTRL_PWM0_ENABLE_Pos 0

/* GPIO_IF4_PWM_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t PWM0_ENABLE_ALIAS;    /* Enable PWM0 output. Overrides the GPIO 32 (GPIO IF4 pin 24 and LCD driver segment 24) configuration. */
    __IO uint32_t PWM1_ENABLE_ALIAS;    /* Enable PWM1 output. Overrides the GPIO 33 (GPIO IF4 pin 25 and LCD driver segment 25) configuration. */
    __IO uint32_t PWM2_ENABLE_ALIAS;    /* Enable PWM2 output. Overrides the GPIO 34 (GPIO IF4 pin 26 and LCD driver segment 26) configuration. */
    __IO uint32_t PWM3_ENABLE_ALIAS;    /* Enable PWM3 output. Overrides the GPIO 35 (GPIO IF4 pin 27 and LCD driver segment 27) configuration. */
} GPIO_IF4_PWM_CTRL_Type;

#define GPIO_IF4_PWM_CTRL               ((GPIO_IF4_PWM_CTRL_Type *) SYS_CALC_BITBAND(GPIO_IF4_PWM_CTRL_BASE, 0))

/* GPIO_IF4_PWM_CTRL settings */
#define PWM0_DISABLE_BITBAND            0x0
#define PWM0_ENABLE_BITBAND             0x1
#define PWM0_DISABLE                    ((uint32_t)(PWM0_DISABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM0_ENABLE_Pos))
#define PWM0_ENABLE                     ((uint32_t)(PWM0_ENABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM0_ENABLE_Pos))

#define PWM1_DISABLE_BITBAND            0x0
#define PWM1_ENABLE_BITBAND             0x1
#define PWM1_DISABLE                    ((uint32_t)(PWM1_DISABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM1_ENABLE_Pos))
#define PWM1_ENABLE                     ((uint32_t)(PWM1_ENABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM1_ENABLE_Pos))

#define PWM2_DISABLE_BITBAND            0x0
#define PWM2_ENABLE_BITBAND             0x1
#define PWM2_DISABLE                    ((uint32_t)(PWM2_DISABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM2_ENABLE_Pos))
#define PWM2_ENABLE                     ((uint32_t)(PWM2_ENABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM2_ENABLE_Pos))

#define PWM3_DISABLE_BITBAND            0x0
#define PWM3_ENABLE_BITBAND             0x1
#define PWM3_DISABLE                    ((uint32_t)(PWM3_DISABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM3_ENABLE_Pos))
#define PWM3_ENABLE                     ((uint32_t)(PWM3_ENABLE_BITBAND << GPIO_IF4_PWM_CTRL_PWM3_ENABLE_Pos))

/* GPIO IF5, Wakeup Function Select */
/*   Select the function for the dedicated wakeup 3V GPIO pins 0 to 3 */
#define GPIO_IF5_FUNC_SEL_BASE          0x40000548

/* GPIO_IF5_FUNC_SEL bit positions */
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_Pos 15
#define GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_Pos 14
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos 13
#define GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos 12
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos 11
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos 10
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos 9
#define GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos 8
#define GPIO_IF5_FUNC_SEL_INPUT_DATA_Pos 4
#define GPIO_IF5_FUNC_SEL_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF5_FUNC_SEL_INPUT_DATA_Pos))
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_Pos 3
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_Pos 2
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_Pos 1
#define GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_Pos 0

/* GPIO_IF5_FUNC_SEL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t SEL_BYTE;             
    __IO uint8_t PIN_CFG_BYTE;         
         uint8_t RESERVED0[2];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000548, 0) - (0x40000548 + 4)];
    __IO uint32_t WAKEUP_ENABLE_PIN0_ALIAS;/* Enable wakeup for low-to-high transition GPIO IF5, pin 0 */
    __IO uint32_t WAKEUP_ENABLE_PIN1_ALIAS;/* Enable wakeup for low-to-high transition GPIO IF5, pin 1 */
    __IO uint32_t WAKEUP_ENABLE_PIN2_ALIAS;/* Enable wakeup for high-to-low transition GPIO IF5, pin 2 */
    __IO uint32_t WAKEUP_ENABLE_PIN3_ALIAS;/* Enable wakeup for high-to-low transition GPIO IF5, pin 3 */
         uint32_t RESERVED1[4];
    __IO uint32_t OUTPUT_ENABLE_PIN0_ALIAS;/* Enable output on GPIO IF5, pin 0 */
    __IO uint32_t OUTPUT_ENABLE_PIN1_ALIAS;/* Enable output on GPIO IF5, pin 1 */
    __IO uint32_t OUTPUT_ENABLE_PIN2_ALIAS;/* Enable output on GPIO IF5, pin 2 */
    __IO uint32_t OUTPUT_ENABLE_PIN3_ALIAS;/* Enable output on GPIO IF5, pin 3 */
    __IO uint32_t PULLUP_ENABLE_PIN0_ALIAS;/* Configure the pull-up resistor for GPIO IF5, pin 0 */
    __IO uint32_t PULLUP_ENABLE_PIN1_ALIAS;/* Configure the pull-up resistor for GPIO IF5, pin 1 */
    __IO uint32_t PULLDOWN_ENABLE_PIN2_ALIAS;/* Configure the pull-down resistor for GPIO IF5, pin 2 */
    __IO uint32_t PULLDOWN_ENABLE_PIN3_ALIAS;/* Configure the pull-down resistor for GPIO IF5, pin 3 */
} GPIO_IF5_FUNC_SEL_Type;

#define GPIO_IF5_FUNC_SEL               ((GPIO_IF5_FUNC_SEL_Type *) GPIO_IF5_FUNC_SEL_BASE)

/* GPIO_IF5_FUNC_SEL settings */
#define GPIO_IF5_PIN0_WAKEUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN0_WAKEUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN0_WAKEUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN0_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_WAKEUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN0_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_WAKEUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN1_WAKEUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN1_WAKEUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN1_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_WAKEUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN1_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_WAKEUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN2_WAKEUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN2_WAKEUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN2_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_WAKEUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN2_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_WAKEUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN3_WAKEUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN3_WAKEUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN3_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_WAKEUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN3_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_Pos))

#define GPIO_IF5_PIN0_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN0_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN0_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF5_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF5_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN1_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN1_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF5_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF5_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN2_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN2_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF5_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF5_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_OUTPUT_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN3_OUTPUT_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN3_OUTPUT_DISABLE    ((uint32_t)(GPIO_IF5_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_OUTPUT_ENABLE     ((uint32_t)(GPIO_IF5_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF5_PIN0_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN0_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN0_PULLUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_PULLUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_PULLUP_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN1_PULLUP_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN1_PULLUP_DISABLE    ((uint32_t)(GPIO_IF5_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_PULLUP_ENABLE     ((uint32_t)(GPIO_IF5_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_PULLDOWN_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN2_PULLDOWN_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN2_PULLDOWN_DISABLE  ((uint32_t)(GPIO_IF5_PIN2_PULLDOWN_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_PULLDOWN_ENABLE   ((uint32_t)(GPIO_IF5_PIN2_PULLDOWN_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_PULLDOWN_DISABLE_BITBAND 0x0
#define GPIO_IF5_PIN3_PULLDOWN_ENABLE_BITBAND 0x1
#define GPIO_IF5_PIN3_PULLDOWN_DISABLE  ((uint32_t)(GPIO_IF5_PIN3_PULLDOWN_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_PULLDOWN_ENABLE   ((uint32_t)(GPIO_IF5_PIN3_PULLDOWN_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_Pos))

/* GPIO_IF5_FUNC_SEL sub-register bit positions */
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN0_Pos 0
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN1_Pos 1
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN2_Pos 2
#define GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN3_Pos 3
#define GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_Pos 4
#define GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_Mask ((uint32_t)(0xFU << GPIO_IF5_FUNC_SEL_BYTE_INPUT_DATA_Pos))
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos 0
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos 1
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos 2
#define GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos 3
#define GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos 4
#define GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos 5
#define GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN2_Pos 6
#define GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN3_Pos 7

/* GPIO_IF5_FUNC_SEL subregister settings */
#define GPIO_IF5_PIN0_WAKEUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_WAKEUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_WAKEUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_WAKEUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_WAKEUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_WAKEUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_WAKEUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_WAKEUP_DISABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_WAKEUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_WAKEUP_ENABLE_BITBAND << GPIO_IF5_FUNC_SEL_BYTE_WAKEUP_ENABLE_PIN3_Pos))

#define GPIO_IF5_PIN0_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_OUTPUT_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_OUTPUT_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_OUTPUT_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_OUTPUT_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_OUTPUT_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_OUTPUT_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_OUTPUT_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_OUTPUT_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_OUTPUT_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_OUTPUT_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_OUTPUT_ENABLE_PIN3_Pos))

#define GPIO_IF5_PIN0_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_PULLUP_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))
#define GPIO_IF5_PIN0_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN0_PULLUP_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN0_Pos))

#define GPIO_IF5_PIN1_PULLUP_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_PULLUP_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))
#define GPIO_IF5_PIN1_PULLUP_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN1_PULLUP_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLUP_ENABLE_PIN1_Pos))

#define GPIO_IF5_PIN2_PULLDOWN_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_PULLDOWN_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN2_Pos))
#define GPIO_IF5_PIN2_PULLDOWN_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN2_PULLDOWN_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN2_Pos))

#define GPIO_IF5_PIN3_PULLDOWN_DISABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_PULLDOWN_DISABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN3_Pos))
#define GPIO_IF5_PIN3_PULLDOWN_ENABLE_BYTE ((uint8_t)(GPIO_IF5_PIN3_PULLDOWN_ENABLE_BITBAND << GPIO_IF5_PIN_CFG_BYTE_PULLDOWN_ENABLE_PIN3_Pos))

/* GPIO IF5, Wakeup Output */
/*   Output for 3V GPIO pins 0 to 3 */
#define GPIO_IF5_OUT_BASE               0x4000054C

/* GPIO_IF5_OUT bit positions */
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN3_Pos 3
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN2_Pos 2
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN1_Pos 1
#define GPIO_IF5_OUT_OUTPUT_DATA_PIN0_Pos 0

/* GPIO_IF5_OUT sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t OUTPUT_DATA_PIN0_ALIAS;/* Output data for GPIO IF5, pin 0 */
    __IO uint32_t OUTPUT_DATA_PIN1_ALIAS;/* Output data for GPIO IF5, pin 1 */
    __IO uint32_t OUTPUT_DATA_PIN2_ALIAS;/* Output data for GPIO IF5, pin 2 */
    __IO uint32_t OUTPUT_DATA_PIN3_ALIAS;/* Output data for GPIO IF5, pin 3 */
} GPIO_IF5_OUT_Type;

#define GPIO_IF5_OUT                    ((GPIO_IF5_OUT_Type *) SYS_CALC_BITBAND(GPIO_IF5_OUT_BASE, 0))

/* GPIO Interface Input Enable */
/*   Enable or disable GPIO inputs for the GPIO interfaces */
#define GPIO_IF_INPUT_ENABLE_BASE       0x40000550

/* GPIO_IF_INPUT_ENABLE bit positions */
#define GPIO_IF_INPUT_ENABLE_IF5_INPUT_Pos 5
#define GPIO_IF_INPUT_ENABLE_IF4_INPUT_Pos 4
#define GPIO_IF_INPUT_ENABLE_IF3_INPUT_Pos 3
#define GPIO_IF_INPUT_ENABLE_IF2_INPUT_Pos 2
#define GPIO_IF_INPUT_ENABLE_IF1_INPUT_Pos 1
#define GPIO_IF_INPUT_ENABLE_IF0_INPUT_Pos 0

/* GPIO_IF_INPUT_ENABLE sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t IF0_INPUT_ALIAS;      /* Enable input sampling on GPIO IF0 */
    __IO uint32_t IF1_INPUT_ALIAS;      /* Enable input sampling on GPIO IF1 */
    __IO uint32_t IF2_INPUT_ALIAS;      /* Enable input sampling on GPIO IF2 */
    __IO uint32_t IF3_INPUT_ALIAS;      /* Enable input sampling on GPIO IF3 */
    __IO uint32_t IF4_INPUT_ALIAS;      /* Enable input sampling on GPIO IF4 */
    __IO uint32_t IF5_INPUT_ALIAS;      /* Enable input sampling on GPIO IF5 */
} GPIO_IF_INPUT_ENABLE_Type;

#define GPIO_IF_INPUT_ENABLE            ((GPIO_IF_INPUT_ENABLE_Type *) SYS_CALC_BITBAND(GPIO_IF_INPUT_ENABLE_BASE, 0))

/* GPIO_IF_INPUT_ENABLE settings */
#define GPIO_IF0_IN_DISABLE_BITBAND     0x0
#define GPIO_IF0_IN_ENABLE_BITBAND      0x1
#define GPIO_IF0_IN_DISABLE             ((uint32_t)(GPIO_IF0_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF0_INPUT_Pos))
#define GPIO_IF0_IN_ENABLE              ((uint32_t)(GPIO_IF0_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF0_INPUT_Pos))

#define GPIO_IF1_IN_DISABLE_BITBAND     0x0
#define GPIO_IF1_IN_ENABLE_BITBAND      0x1
#define GPIO_IF1_IN_DISABLE             ((uint32_t)(GPIO_IF1_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF1_INPUT_Pos))
#define GPIO_IF1_IN_ENABLE              ((uint32_t)(GPIO_IF1_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF1_INPUT_Pos))

#define GPIO_IF2_IN_DISABLE_BITBAND     0x0
#define GPIO_IF2_IN_ENABLE_BITBAND      0x1
#define GPIO_IF2_IN_DISABLE             ((uint32_t)(GPIO_IF2_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF2_INPUT_Pos))
#define GPIO_IF2_IN_ENABLE              ((uint32_t)(GPIO_IF2_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF2_INPUT_Pos))

#define GPIO_IF3_IN_DISABLE_BITBAND     0x0
#define GPIO_IF3_IN_ENABLE_BITBAND      0x1
#define GPIO_IF3_IN_DISABLE             ((uint32_t)(GPIO_IF3_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF3_INPUT_Pos))
#define GPIO_IF3_IN_ENABLE              ((uint32_t)(GPIO_IF3_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF3_INPUT_Pos))

#define GPIO_IF4_IN_DISABLE_BITBAND     0x0
#define GPIO_IF4_IN_ENABLE_BITBAND      0x1
#define GPIO_IF4_IN_DISABLE             ((uint32_t)(GPIO_IF4_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF4_INPUT_Pos))
#define GPIO_IF4_IN_ENABLE              ((uint32_t)(GPIO_IF4_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF4_INPUT_Pos))

#define GPIO_IF5_IN_DISABLE_BITBAND     0x0
#define GPIO_IF5_IN_ENABLE_BITBAND      0x1
#define GPIO_IF5_IN_DISABLE             ((uint32_t)(GPIO_IF5_IN_DISABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF5_INPUT_Pos))
#define GPIO_IF5_IN_ENABLE              ((uint32_t)(GPIO_IF5_IN_ENABLE_BITBAND << GPIO_IF_INPUT_ENABLE_IF5_INPUT_Pos))

/* GPIO Interrupt Control 0 */
/*   Configure the GPIO general-purpose interrupts 0 and 1 */
#define GPIO_INT_CTRL0_BASE             0x40000560

/* GPIO_INT_CTRL0 bit positions */
#define GPIO_INT_CTRL0_GP0_INT_TYPE_Pos 24
#define GPIO_INT_CTRL0_GP0_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_INT_CTRL0_GP0_INT_SRC_Pos  16
#define GPIO_INT_CTRL0_GP0_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL0_GP0_INT_SRC_Pos))
#define GPIO_INT_CTRL0_GP1_INT_TYPE_Pos 8
#define GPIO_INT_CTRL0_GP1_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_INT_CTRL0_GP1_INT_SRC_Pos  0
#define GPIO_INT_CTRL0_GP1_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL0_GP1_INT_SRC_Pos))

/* GPIO_INT_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t GP1_SHORT;            
    __IO uint16_t GP0_SHORT;            
} GPIO_INT_CTRL0_Type;

#define GPIO_INT_CTRL0                  ((GPIO_INT_CTRL0_Type *) GPIO_INT_CTRL0_BASE)

/* GPIO_INT_CTRL0 settings */
#define GPIO_GP1_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP1_INT_TYPE_Pos))

#define GPIO_GP0_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL0_GP0_INT_TYPE_Pos))

/* GPIO_INT_CTRL0 sub-register bit positions */
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))

/* GPIO_INT_CTRL0 subregister settings */
#define GPIO_GP1_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))
#define GPIO_GP1_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Pos))

#define GPIO_GP0_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))
#define GPIO_GP0_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Pos))

/* GPIO Interrupt Control 1 */
/*   Configure the GPIO general-purpose interrupts 2 and 3 */
#define GPIO_INT_CTRL1_BASE             0x40000564

/* GPIO_INT_CTRL1 bit positions */
#define GPIO_INT_CTRL1_GP3_INT_TYPE_Pos 24
#define GPIO_INT_CTRL1_GP3_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_INT_CTRL1_GP3_INT_SRC_Pos  16
#define GPIO_INT_CTRL1_GP3_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL1_GP3_INT_SRC_Pos))
#define GPIO_INT_CTRL1_GP2_INT_TYPE_Pos 8
#define GPIO_INT_CTRL1_GP2_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_INT_CTRL1_GP2_INT_SRC_Pos  0
#define GPIO_INT_CTRL1_GP2_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL1_GP2_INT_SRC_Pos))

/* GPIO_INT_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t GP2_SHORT;            
    __IO uint16_t GP3_SHORT;            
} GPIO_INT_CTRL1_Type;

#define GPIO_INT_CTRL1                  ((GPIO_INT_CTRL1_Type *) GPIO_INT_CTRL1_BASE)

/* GPIO_INT_CTRL1 settings */
#define GPIO_GP2_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP2_INT_TYPE_Pos))

#define GPIO_GP3_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL1_GP3_INT_TYPE_Pos))

/* GPIO_INT_CTRL1 sub-register bit positions */
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))

/* GPIO_INT_CTRL1 subregister settings */
#define GPIO_GP2_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))
#define GPIO_GP2_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Pos))

#define GPIO_GP3_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))
#define GPIO_GP3_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Pos))

/* GPIO Interrupt Control 2 */
/*   Configure the GPIO general-purpose interrupts 4 and 5 */
#define GPIO_INT_CTRL2_BASE             0x40000568

/* GPIO_INT_CTRL2 bit positions */
#define GPIO_INT_CTRL2_GP5_INT_TYPE_Pos 24
#define GPIO_INT_CTRL2_GP5_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_INT_CTRL2_GP5_INT_SRC_Pos  16
#define GPIO_INT_CTRL2_GP5_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL2_GP5_INT_SRC_Pos))
#define GPIO_INT_CTRL2_GP4_INT_TYPE_Pos 8
#define GPIO_INT_CTRL2_GP4_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_INT_CTRL2_GP4_INT_SRC_Pos  0
#define GPIO_INT_CTRL2_GP4_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL2_GP4_INT_SRC_Pos))

/* GPIO_INT_CTRL2 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t GP4_SHORT;            
    __IO uint16_t GP5_SHORT;            
} GPIO_INT_CTRL2_Type;

#define GPIO_INT_CTRL2                  ((GPIO_INT_CTRL2_Type *) GPIO_INT_CTRL2_BASE)

/* GPIO_INT_CTRL2 settings */
#define GPIO_GP4_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP4_INT_TYPE_Pos))

#define GPIO_GP5_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL2_GP5_INT_TYPE_Pos))

/* GPIO_INT_CTRL2 sub-register bit positions */
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))

/* GPIO_INT_CTRL2 subregister settings */
#define GPIO_GP4_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))
#define GPIO_GP4_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Pos))

#define GPIO_GP5_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))
#define GPIO_GP5_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Pos))

/* GPIO Interrupt Control 3 */
/*   Configure the GPIO general-purpose interrupts 2 and 3 */
#define GPIO_INT_CTRL3_BASE             0x4000056C

/* GPIO_INT_CTRL3 bit positions */
#define GPIO_INT_CTRL3_GP7_INT_TYPE_Pos 24
#define GPIO_INT_CTRL3_GP7_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_INT_CTRL3_GP7_INT_SRC_Pos  16
#define GPIO_INT_CTRL3_GP7_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL3_GP7_INT_SRC_Pos))
#define GPIO_INT_CTRL3_GP6_INT_TYPE_Pos 8
#define GPIO_INT_CTRL3_GP6_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_INT_CTRL3_GP6_INT_SRC_Pos  0
#define GPIO_INT_CTRL3_GP6_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL3_GP6_INT_SRC_Pos))

/* GPIO_INT_CTRL3 sub-register and bit-band aliases */
typedef struct
{
    __IO uint16_t GP6_SHORT;            
    __IO uint16_t GP7_SHORT;            
} GPIO_INT_CTRL3_Type;

#define GPIO_INT_CTRL3                  ((GPIO_INT_CTRL3_Type *) GPIO_INT_CTRL3_BASE)

/* GPIO_INT_CTRL3 settings */
#define GPIO_GP6_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP6_INT_TYPE_Pos))

#define GPIO_GP7_INT_DISABLE            ((uint32_t)(0x0U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_LOW_TRIGGER        ((uint32_t)(0x2U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_HIGH_TRIGGER       ((uint32_t)(0x3U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_EDGE_TRIGGER       ((uint32_t)(0x4U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_NEG_EDGE_TRIGGER   ((uint32_t)(0x6U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_POS_EDGE_TRIGGER   ((uint32_t)(0x7U << GPIO_INT_CTRL3_GP7_INT_TYPE_Pos))

/* GPIO_INT_CTRL3 sub-register bit positions */
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_Pos 0
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_Mask ((uint32_t)(0x3FU << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_Pos))
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos 8
#define GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Mask ((uint32_t)(0x7U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))

/* GPIO_INT_CTRL3 subregister settings */
#define GPIO_GP6_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))
#define GPIO_GP6_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Pos))

#define GPIO_GP7_INT_DISABLE_SHORT      ((uint16_t)(0x0U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_LOW_TRIGGER_SHORT  ((uint16_t)(0x2U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_HIGH_TRIGGER_SHORT ((uint16_t)(0x3U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_EDGE_TRIGGER_SHORT ((uint16_t)(0x4U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_NEG_EDGE_TRIGGER_SHORT ((uint16_t)(0x6U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))
#define GPIO_GP7_INT_POS_EDGE_TRIGGER_SHORT ((uint16_t)(0x7U << GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Pos))

/* ----------------------------------------------------------------------------
 * SPI Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The serial-peripheral interface (SPI) module provides a synchronous 4-wire
 * interface including clock, chip select, serial data in, and serial data out.
 *
 * This interface is designed to operate as either a master or slave connecting
 * to external storage devices, advanced displays, wireless transceivers and
 * other data sources or sinks without the need for external components. */

typedef struct
{
    __IO uint32_t CTRL0;                /* Configure the SPI0 interface */
    __IO uint32_t CTRL1;                /* Control transactions on the SPI0 interface */
    __IO uint32_t DATA;                 /* Single word buffer for data to be transmitted over the SPI0 interface and data that has been received over the SPI0 interface */
    __IO uint32_t DATA_S;               /* Shadow register for SPI0_DATA providing no side effect access to the interface's data */
    __IO uint32_t SLAVE_DATA;           /* Single word buffer for data received over the SPI0 interface when operating in slave mode */
    __IO uint32_t STATUS;               /* Status of the SPI0 interface */
} SPI_Type;

#define SPI0_BASE                       0x40000600
#define SPI0                            ((SPI_Type *) SPI0_BASE)

/* SPI0 Control and Configuration Register */
/*   Configure the SPI0 interface */
#define SPI0_CTRL0_BASE                 0x40000600

/* SPI_CTRL0 bit positions */
#define SPI_CTRL0_SQI_ENABLE_Pos        11
#define SPI_CTRL0_OVERRUN_INT_ENABLE_Pos 10
#define SPI_CTRL0_UNDERRUN_INT_ENABLE_Pos 9
#define SPI_CTRL0_CONTROLLER_Pos        8
#define SPI_CTRL0_SLAVE_Pos             7
#define SPI_CTRL0_SERI_PULLUP_ENABLE_Pos 6
#define SPI_CTRL0_CLK_POLARITY_Pos      5
#define SPI_CTRL0_MODE_SELECT_Pos       4
#define SPI_CTRL0_ENABLE_Pos            3
#define SPI_CTRL0_PRESCALE_Pos          0
#define SPI_CTRL0_PRESCALE_Mask         ((uint32_t)(0x7U << SPI_CTRL0_PRESCALE_Pos))

/* SPI_CTRL0 sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0[3];
    __IO uint32_t ENABLE_ALIAS;         /* Enable/disable the SPI0 interface */
    __IO uint32_t MODE_SELECT_ALIAS;    /* Select between manual and auto transaction handling modes for SPI0 master transactions */
    __IO uint32_t CLK_POLARITY_ALIAS;   /* Select the polarity of the SPI0 clock */
    __IO uint32_t SERI_PULLUP_ENABLE_ALIAS;/* Configure the optional pull-up resistor for the SERI pad of SPI0 */
    __IO uint32_t SLAVE_ALIAS;          /* Use the SPI0 interface as master or slave */
    __IO uint32_t CONTROLLER_ALIAS;     /* Select whether data transfer will be controlled by the ARM Cortex-M3 processor or the DMA for SPI0 */
    __IO uint32_t UNDERRUN_INT_ENABLE_ALIAS;/* Enable SPI0 underrun interrupts */
    __IO uint32_t OVERRUN_INT_ENABLE_ALIAS;/* Enable SPI0 overrun interrupts */
    __IO uint32_t SQI_ENABLE_ALIAS;     /* Enable SQI operation (nibble width transfers) */
} SPI_CTRL0_Type;

#define SPI0_CTRL0                      ((SPI_CTRL0_Type *) SYS_CALC_BITBAND(SPI0_CTRL0_BASE, 0))

/* SPI_CTRL0 settings */
#define SPI_PRESCALE_2                  ((uint32_t)(0x0U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_4                  ((uint32_t)(0x1U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_8                  ((uint32_t)(0x2U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_16                 ((uint32_t)(0x3U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_32                 ((uint32_t)(0x4U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_64                 ((uint32_t)(0x5U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_128                ((uint32_t)(0x6U << SPI_CTRL0_PRESCALE_Pos))
#define SPI_PRESCALE_256                ((uint32_t)(0x7U << SPI_CTRL0_PRESCALE_Pos))

#define SPI_DISABLE_BITBAND             0x0
#define SPI_ENABLE_BITBAND              0x1
#define SPI_DISABLE                     ((uint32_t)(SPI_DISABLE_BITBAND << SPI_CTRL0_ENABLE_Pos))
#define SPI_ENABLE                      ((uint32_t)(SPI_ENABLE_BITBAND << SPI_CTRL0_ENABLE_Pos))

#define SPI_MODE_SELECT_MANUAL_BITBAND  0x0
#define SPI_MODE_SELECT_AUTO_BITBAND    0x1
#define SPI_MODE_SELECT_MANUAL          ((uint32_t)(SPI_MODE_SELECT_MANUAL_BITBAND << SPI_CTRL0_MODE_SELECT_Pos))
#define SPI_MODE_SELECT_AUTO            ((uint32_t)(SPI_MODE_SELECT_AUTO_BITBAND << SPI_CTRL0_MODE_SELECT_Pos))

#define SPI_CLK_POLARITY_NORMAL_BITBAND 0x0
#define SPI_CLK_POLARITY_INVERSE_BITBAND 0x1
#define SPI_CLK_POLARITY_NORMAL         ((uint32_t)(SPI_CLK_POLARITY_NORMAL_BITBAND << SPI_CTRL0_CLK_POLARITY_Pos))
#define SPI_CLK_POLARITY_INVERSE        ((uint32_t)(SPI_CLK_POLARITY_INVERSE_BITBAND << SPI_CTRL0_CLK_POLARITY_Pos))

#define SPI_SERI_PULLUP_DISABLE_BITBAND 0x0
#define SPI_SERI_PULLUP_ENABLE_BITBAND  0x1
#define SPI_SERI_PULLUP_DISABLE         ((uint32_t)(SPI_SERI_PULLUP_DISABLE_BITBAND << SPI_CTRL0_SERI_PULLUP_ENABLE_Pos))
#define SPI_SERI_PULLUP_ENABLE          ((uint32_t)(SPI_SERI_PULLUP_ENABLE_BITBAND << SPI_CTRL0_SERI_PULLUP_ENABLE_Pos))

#define SPI_SELECT_MASTER_BITBAND       0x0
#define SPI_SELECT_SLAVE_BITBAND        0x1
#define SPI_SELECT_MASTER               ((uint32_t)(SPI_SELECT_MASTER_BITBAND << SPI_CTRL0_SLAVE_Pos))
#define SPI_SELECT_SLAVE                ((uint32_t)(SPI_SELECT_SLAVE_BITBAND << SPI_CTRL0_SLAVE_Pos))

#define SPI_CONTROLLER_CM3_BITBAND      0x0
#define SPI_CONTROLLER_DMA_BITBAND      0x1
#define SPI_CONTROLLER_CM3              ((uint32_t)(SPI_CONTROLLER_CM3_BITBAND << SPI_CTRL0_CONTROLLER_Pos))
#define SPI_CONTROLLER_DMA              ((uint32_t)(SPI_CONTROLLER_DMA_BITBAND << SPI_CTRL0_CONTROLLER_Pos))

#define SPI_UNDERRUN_INT_DISABLE_BITBAND 0x0
#define SPI_UNDERRUN_INT_ENABLE_BITBAND 0x1
#define SPI_UNDERRUN_INT_DISABLE        ((uint32_t)(SPI_UNDERRUN_INT_DISABLE_BITBAND << SPI_CTRL0_UNDERRUN_INT_ENABLE_Pos))
#define SPI_UNDERRUN_INT_ENABLE         ((uint32_t)(SPI_UNDERRUN_INT_ENABLE_BITBAND << SPI_CTRL0_UNDERRUN_INT_ENABLE_Pos))

#define SPI_OVERRUN_INT_DISABLE_BITBAND 0x0
#define SPI_OVERRUN_INT_ENABLE_BITBAND  0x1
#define SPI_OVERRUN_INT_DISABLE         ((uint32_t)(SPI_OVERRUN_INT_DISABLE_BITBAND << SPI_CTRL0_OVERRUN_INT_ENABLE_Pos))
#define SPI_OVERRUN_INT_ENABLE          ((uint32_t)(SPI_OVERRUN_INT_ENABLE_BITBAND << SPI_CTRL0_OVERRUN_INT_ENABLE_Pos))

#define SPI_SELECT_SPI_BITBAND          0x0
#define SPI_SELECT_SQI_BITBAND          0x1
#define SPI_SELECT_SPI                  ((uint32_t)(SPI_SELECT_SPI_BITBAND << SPI_CTRL0_SQI_ENABLE_Pos))
#define SPI_SELECT_SQI                  ((uint32_t)(SPI_SELECT_SQI_BITBAND << SPI_CTRL0_SQI_ENABLE_Pos))

/* SPI0 Transaction Control Register */
/*   Control transactions on the SPI0 interface */
#define SPI0_CTRL1_BASE                 0x40000604

/* SPI_CTRL1 bit positions */
#define SPI_CTRL1_START_BUSY_Pos        7
#define SPI_CTRL1_RW_CMD_Pos            6
#define SPI_CTRL1_CS_Pos                5
#define SPI_CTRL1_WORD_SIZE_Pos         0
#define SPI_CTRL1_WORD_SIZE_Mask        ((uint32_t)(0x1FU << SPI_CTRL1_WORD_SIZE_Pos))

/* SPI_CTRL1 sub-register and bit-band aliases */
typedef struct
{
         uint32_t RESERVED0[5];
    __IO uint32_t CS_ALIAS;             /* Set the chip-select line for SPI0 (master mode); read the chip-select line for SPI0 (slave mode) */
    __IO uint32_t RW_CMD_ALIAS;         /* Issue a read command or write command to the SPI0 interface */
    __IO uint32_t START_BUSY_ALIAS;     /* Start an SPI0 data transfer and indicate if a transfer is in progress */
} SPI_CTRL1_Type;

#define SPI0_CTRL1                      ((SPI_CTRL1_Type *) SYS_CALC_BITBAND(SPI0_CTRL1_BASE, 0))

/* SPI_CTRL1 settings */
#define SPI_WORD_SIZE_1                 ((uint32_t)(0x0U << SPI_CTRL1_WORD_SIZE_Pos))
#define SPI_WORD_SIZE_8                 ((uint32_t)(0x7U << SPI_CTRL1_WORD_SIZE_Pos))
#define SPI_WORD_SIZE_16                ((uint32_t)(0xFU << SPI_CTRL1_WORD_SIZE_Pos))
#define SPI_WORD_SIZE_24                ((uint32_t)(0x17U << SPI_CTRL1_WORD_SIZE_Pos))
#define SPI_WORD_SIZE_32                ((uint32_t)(0x1FU << SPI_CTRL1_WORD_SIZE_Pos))

#define SPI_CS_0_BITBAND                0x0
#define SPI_CS_1_BITBAND                0x1
#define SPI_CS_0                        ((uint32_t)(SPI_CS_0_BITBAND << SPI_CTRL1_CS_Pos))
#define SPI_CS_1                        ((uint32_t)(SPI_CS_1_BITBAND << SPI_CTRL1_CS_Pos))

#define SPI_WRITE_DATA_BITBAND          0x0
#define SPI_READ_DATA_BITBAND           0x1
#define SPI_WRITE_DATA                  ((uint32_t)(SPI_WRITE_DATA_BITBAND << SPI_CTRL1_RW_CMD_Pos))
#define SPI_READ_DATA                   ((uint32_t)(SPI_READ_DATA_BITBAND << SPI_CTRL1_RW_CMD_Pos))

#define SPI_IDLE_BITBAND                0x0
#define SPI_START_BITBAND               0x1
#define SPI_BUSY_BITBAND                0x1
#define SPI_IDLE                        ((uint32_t)(SPI_IDLE_BITBAND << SPI_CTRL1_START_BUSY_Pos))
#define SPI_START                       ((uint32_t)(SPI_START_BITBAND << SPI_CTRL1_START_BUSY_Pos))
#define SPI_BUSY                        ((uint32_t)(SPI_BUSY_BITBAND << SPI_CTRL1_START_BUSY_Pos))

/* SPI0 Data */
/*   Single word buffer for data to be transmitted over the SPI0 interface and
 *   data that has been received over the SPI0 interface */
#define SPI0_DATA_BASE                  0x40000608

/* Shadow of SPI0 Data */
/*   Shadow register for SPI0_DATA providing no side effect access to the
 *   interface's data */
#define SPI0_DATA_S_BASE                0x4000060C

/* SPI0 Data for Slave Operations */
/*   Single word buffer for data received over the SPI0 interface when
 *   operating in slave mode */
#define SPI0_SLAVE_DATA_BASE            0x40000610

/* SPI_SLAVE_DATA bit positions */
#define SPI_SLAVE_DATA_SPI0_DATA_Pos    0
#define SPI_SLAVE_DATA_SPI0_DATA_Mask   ((uint32_t)(0xFFFFFFFFU << SPI_SLAVE_DATA_SPI0_DATA_Pos))

/* SPI0 Status */
/*   Status of the SPI0 interface */
#define SPI0_STATUS_BASE                0x40000614

/* SPI_STATUS bit positions */
#define SPI_STATUS_OVERRUN_STATUS_Pos   1
#define SPI_STATUS_UNDERRUN_STATUS_Pos  0

/* SPI_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t UNDERRUN_STATUS_ALIAS;/* Indicate that an underrun has occurred when transmitting data on the SPI0 interface */
    __IO uint32_t OVERRUN_STATUS_ALIAS; /* Indicate that an overrun has occurred when receiving data on the SPI0 interface */
} SPI_STATUS_Type;

#define SPI0_STATUS                     ((SPI_STATUS_Type *) SYS_CALC_BITBAND(SPI0_STATUS_BASE, 0))

/* SPI_STATUS settings */
#define SPI_UNDERRUN_FALSE_BITBAND      0x0
#define SPI_UNDERRUN_TRUE_BITBAND       0x1
#define SPI_UNDERRUN_CLEAR_BITBAND      0x1
#define SPI_UNDERRUN_FALSE              ((uint32_t)(SPI_UNDERRUN_FALSE_BITBAND << SPI_STATUS_UNDERRUN_STATUS_Pos))
#define SPI_UNDERRUN_TRUE               ((uint32_t)(SPI_UNDERRUN_TRUE_BITBAND << SPI_STATUS_UNDERRUN_STATUS_Pos))
#define SPI_UNDERRUN_CLEAR              ((uint32_t)(SPI_UNDERRUN_CLEAR_BITBAND << SPI_STATUS_UNDERRUN_STATUS_Pos))

#define SPI_OVERRUN_FALSE_BITBAND       0x0
#define SPI_OVERRUN_TRUE_BITBAND        0x1
#define SPI_OVERRUN_CLEAR_BITBAND       0x1
#define SPI_OVERRUN_FALSE               ((uint32_t)(SPI_OVERRUN_FALSE_BITBAND << SPI_STATUS_OVERRUN_STATUS_Pos))
#define SPI_OVERRUN_TRUE                ((uint32_t)(SPI_OVERRUN_TRUE_BITBAND << SPI_STATUS_OVERRUN_STATUS_Pos))
#define SPI_OVERRUN_CLEAR               ((uint32_t)(SPI_OVERRUN_CLEAR_BITBAND << SPI_STATUS_OVERRUN_STATUS_Pos))

#define SPI1_BASE                       0x40000700
#define SPI1                            ((SPI_Type *) SPI1_BASE)

/* SPI1 Control and Configuration Register */
/*   Configure the SPI1 interface */
#define SPI1_CTRL0_BASE                 0x40000700

/* SPI_CTRL0 sub-register and bit-band aliases */
#define SPI1_CTRL0                      ((SPI_CTRL0_Type *) SYS_CALC_BITBAND(SPI1_CTRL0_BASE, 0))

/* SPI1 Transaction Control Register */
/*   Control transactions on the SPI1 interface */
#define SPI1_CTRL1_BASE                 0x40000704

/* SPI_CTRL1 sub-register and bit-band aliases */
#define SPI1_CTRL1                      ((SPI_CTRL1_Type *) SYS_CALC_BITBAND(SPI1_CTRL1_BASE, 0))

/* SPI1 Data */
/*   Single word buffer for data to be transmitted over the SPI1 interface and
 *   data that has been received over the SPI1 interface */
#define SPI1_DATA_BASE                  0x40000708

/* Shadow of SPI1 Data */
/*   Shadow register for SPI1_DATA providing no side effect access to the
 *   interface's data */
#define SPI1_DATA_S_BASE                0x4000070C

/* SPI1 Data for Slave Operations */
/*   Single word buffer for data received over the SPI1 interface when
 *   operating in slave mode */
#define SPI1_SLAVE_DATA_BASE            0x40000710

/* SPI_SLAVE_DATA sub-register and bit-band aliases */
#define SPI1_SLAVE_DATA                 ((SPI_SLAVE_DATA_Type *) SYS_CALC_BITBAND(SPI1_SLAVE_DATA_BASE, 0))

/* SPI1 Status */
/*   Status of the SPI1 interface */
#define SPI1_STATUS_BASE                0x40000714

/* SPI_STATUS sub-register and bit-band aliases */
#define SPI1_STATUS                     ((SPI_STATUS_Type *) SYS_CALC_BITBAND(SPI1_STATUS_BASE, 0))

/* ----------------------------------------------------------------------------
 * PCM Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The pulse-code modulation (PCM) interface module provides a data connection
 * to be used with Bluetooth chips, other wireless devices or other data
 * sources without the need for additional external components.
 *
 * The PCM interface shares a set of multiplexed pins with the SPI1
 * interface. */

typedef struct
{
    __IO uint32_t CTRL;                 /* Configuration of the PCM frame signal, data, and interface behavior. */
    __IO uint32_t TX_DATA;              /* Single word buffer for the PCM interface containing the next word to transmit over the PCM interface. */
    __I  uint32_t RX_DATA;              /* Single word buffer for the PCM interface containing the most recently received word (read-only). */
} PCM_Type;

#define PCM_BASE                        0x40000800
#define PCM                             ((PCM_Type *) PCM_BASE)

/* PCM Control */
/*   Configuration of the PCM frame signal, data, and interface behavior. */
#define PCM_CTRL_BASE                   0x40000800

/* PCM_CTRL bit positions */
#define PCM_CTRL_PULLUP_ENABLE_Pos      13
#define PCM_CTRL_BIT_ORDER_Pos          12
#define PCM_CTRL_TX_ALIGN_Pos           11
#define PCM_CTRL_WORD_SIZE_Pos          9
#define PCM_CTRL_WORD_SIZE_Mask         ((uint32_t)(0x3U << PCM_CTRL_WORD_SIZE_Pos))
#define PCM_CTRL_FRAME_ALIGN_Pos        8
#define PCM_CTRL_FRAME_WIDTH_Pos        7
#define PCM_CTRL_FRAME_LENGTH_Pos       4
#define PCM_CTRL_FRAME_LENGTH_Mask      ((uint32_t)(0x7U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_CTRL_FRAME_SUBFRAMES_Pos    3
#define PCM_CTRL_CONTROLLER_Pos         2
#define PCM_CTRL_ENABLE_Pos             1
#define PCM_CTRL_SLAVE_Pos              0

/* PCM_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t SLAVE_ALIAS;          /* Use the PCM interface as a master/slave */
    __IO uint32_t ENABLE_ALIAS;         /* Enable/disable the PCM interface */
    __IO uint32_t CONTROLLER_ALIAS;     /* Select whether data transfer will be controlled by the ARM Cortex-M3 processor or the DMA for PCM */
    __IO uint32_t FRAME_SUBFRAMES_ALIAS;/* Enable/disable use of PCM subframes */
         uint32_t RESERVED0[3];
    __IO uint32_t FRAME_WIDTH_ALIAS;    /* Use a long/short PCM frame signal */
    __IO uint32_t FRAME_ALIGN_ALIAS;    /* Align the PCM frame signal to the first/last bit */
         uint32_t RESERVED1[2];
    __IO uint32_t TX_ALIGN_ALIAS;       /* Select what bits to use for transmit data */
    __IO uint32_t BIT_ORDER_ALIAS;      /* Select whether the data will be transmitted starting with the MSB or LSB */
    __IO uint32_t PULLUP_ENABLE_ALIAS;  /* Configure the optional pull-up resistors for the PCM interface input pads */
} PCM_CTRL_Type;

#define PCM_CTRL                        ((PCM_CTRL_Type *) SYS_CALC_BITBAND(PCM_CTRL_BASE, 0))

/* PCM_CTRL settings */
#define PCM_SELECT_MASTER_BITBAND       0x0
#define PCM_SELECT_SLAVE_BITBAND        0x1
#define PCM_SELECT_MASTER               ((uint32_t)(PCM_SELECT_MASTER_BITBAND << PCM_CTRL_SLAVE_Pos))
#define PCM_SELECT_SLAVE                ((uint32_t)(PCM_SELECT_SLAVE_BITBAND << PCM_CTRL_SLAVE_Pos))

#define PCM_DISABLE_BITBAND             0x0
#define PCM_ENABLE_BITBAND              0x1
#define PCM_DISABLE                     ((uint32_t)(PCM_DISABLE_BITBAND << PCM_CTRL_ENABLE_Pos))
#define PCM_ENABLE                      ((uint32_t)(PCM_ENABLE_BITBAND << PCM_CTRL_ENABLE_Pos))

#define PCM_CONTROLLER_CM3_BITBAND      0x0
#define PCM_CONTROLLER_DMA_BITBAND      0x1
#define PCM_CONTROLLER_CM3              ((uint32_t)(PCM_CONTROLLER_CM3_BITBAND << PCM_CTRL_CONTROLLER_Pos))
#define PCM_CONTROLLER_DMA              ((uint32_t)(PCM_CONTROLLER_DMA_BITBAND << PCM_CTRL_CONTROLLER_Pos))

#define PCM_SUBFRAMES_DISABLE_BITBAND   0x0
#define PCM_SUBFRAMES_ENABLE_BITBAND    0x1
#define PCM_SUBFRAMES_DISABLE           ((uint32_t)(PCM_SUBFRAMES_DISABLE_BITBAND << PCM_CTRL_FRAME_SUBFRAMES_Pos))
#define PCM_SUBFRAMES_ENABLE            ((uint32_t)(PCM_SUBFRAMES_ENABLE_BITBAND << PCM_CTRL_FRAME_SUBFRAMES_Pos))

#define PCM_FRAME_LENGTH_2              ((uint32_t)(0x0U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_4              ((uint32_t)(0x1U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_6              ((uint32_t)(0x2U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_8              ((uint32_t)(0x3U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_10             ((uint32_t)(0x4U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_12             ((uint32_t)(0x5U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_14             ((uint32_t)(0x6U << PCM_CTRL_FRAME_LENGTH_Pos))
#define PCM_FRAME_LENGTH_16             ((uint32_t)(0x7U << PCM_CTRL_FRAME_LENGTH_Pos))

#define PCM_FRAME_WIDTH_SHORT_BITBAND   0x0
#define PCM_FRAME_WIDTH_LONG_BITBAND    0x1
#define PCM_FRAME_WIDTH_SHORT           ((uint32_t)(PCM_FRAME_WIDTH_SHORT_BITBAND << PCM_CTRL_FRAME_WIDTH_Pos))
#define PCM_FRAME_WIDTH_LONG            ((uint32_t)(PCM_FRAME_WIDTH_LONG_BITBAND << PCM_CTRL_FRAME_WIDTH_Pos))

#define PCM_FRAME_ALIGN_LAST_BITBAND    0x0
#define PCM_FRAME_ALIGN_FIRST_BITBAND   0x1
#define PCM_FRAME_ALIGN_LAST            ((uint32_t)(PCM_FRAME_ALIGN_LAST_BITBAND << PCM_CTRL_FRAME_ALIGN_Pos))
#define PCM_FRAME_ALIGN_FIRST           ((uint32_t)(PCM_FRAME_ALIGN_FIRST_BITBAND << PCM_CTRL_FRAME_ALIGN_Pos))

#define PCM_WORD_SIZE_8                 ((uint32_t)(0x0U << PCM_CTRL_WORD_SIZE_Pos))
#define PCM_WORD_SIZE_16                ((uint32_t)(0x1U << PCM_CTRL_WORD_SIZE_Pos))
#define PCM_WORD_SIZE_24                ((uint32_t)(0x2U << PCM_CTRL_WORD_SIZE_Pos))
#define PCM_WORD_SIZE_32                ((uint32_t)(0x3U << PCM_CTRL_WORD_SIZE_Pos))

#define PCM_TX_ALIGN_MSB_BITBAND        0x0
#define PCM_TX_ALIGN_LSB_BITBAND        0x1
#define PCM_TX_ALIGN_MSB                ((uint32_t)(PCM_TX_ALIGN_MSB_BITBAND << PCM_CTRL_TX_ALIGN_Pos))
#define PCM_TX_ALIGN_LSB                ((uint32_t)(PCM_TX_ALIGN_LSB_BITBAND << PCM_CTRL_TX_ALIGN_Pos))

#define PCM_BIT_ORDER_MSB_FIRST_BITBAND 0x0
#define PCM_BIT_ORDER_LSB_FIRST_BITBAND 0x1
#define PCM_BIT_ORDER_MSB_FIRST         ((uint32_t)(PCM_BIT_ORDER_MSB_FIRST_BITBAND << PCM_CTRL_BIT_ORDER_Pos))
#define PCM_BIT_ORDER_LSB_FIRST         ((uint32_t)(PCM_BIT_ORDER_LSB_FIRST_BITBAND << PCM_CTRL_BIT_ORDER_Pos))

#define PCM_PULLUP_DISABLE_BITBAND      0x0
#define PCM_PULLUP_ENABLE_BITBAND       0x1
#define PCM_PULLUP_DISABLE              ((uint32_t)(PCM_PULLUP_DISABLE_BITBAND << PCM_CTRL_PULLUP_ENABLE_Pos))
#define PCM_PULLUP_ENABLE               ((uint32_t)(PCM_PULLUP_ENABLE_BITBAND << PCM_CTRL_PULLUP_ENABLE_Pos))

/* PCM Transmit Data */
/*   Single word buffer for the PCM interface containing the next word to
 *   transmit over the PCM interface. */
#define PCM_TX_DATA_BASE                0x40000804

/* PCM Receive Data */
/*   Single word buffer for the PCM interface containing the most recently
 *   received word (read-only). */
#define PCM_RX_DATA_BASE                0x40000808

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

typedef struct
{
    __IO uint32_t CTRL0;                /* Configuration of the I2C interface including master clock speed, slave address and other interface parameters */
    __IO uint32_t CTRL1;                /* Control transfers using the I2C interface */
    __IO uint32_t PHY_CTRL;             /* Configure the physical pads used by the I2C interface */
         uint32_t RESERVED0;
    __IO uint32_t DATA;                 /* Single byte buffer for data to be transmitted over the I2C interface and data that has been received over the I2C interface */
    __IO uint32_t DATA_S;               /* Shadow of the single byte buffer for data to be transmitted over the I2C interface and data that has been received over the I2C interface */
    __O  uint32_t ADDR_START;           /* Start a master I2C transaction with the provided address and read-write bit */
    __I  uint32_t STATUS;               /* Indicate the status of the I2C interface */
} I2C_Type;

#define I2C_BASE                        0x40000900
#define I2C                             ((I2C_Type *) I2C_BASE)

/* I2C Interface Configuration and Control */
/*   Configuration of the I2C interface including master clock speed, slave
 *   address and other interface parameters */
#define I2C_CTRL0_BASE                  0x40000900

/* I2C_CTRL0 bit positions */
#define I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos 16
#define I2C_CTRL0_MASTER_SPEED_PRESCALAR_Mask ((uint32_t)(0xFFU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_CTRL0_SLAVE_ADDRESS_Pos     8
#define I2C_CTRL0_SLAVE_ADDRESS_Mask    ((uint32_t)(0x7FU << I2C_CTRL0_SLAVE_ADDRESS_Pos))
#define I2C_CTRL0_CONTROLLER_Pos        4
#define I2C_CTRL0_STOP_INT_ENABLE_Pos   3
#define I2C_CTRL0_AUTO_ACK_ENABLE_Pos   2
#define I2C_CTRL0_SLAVE_ENABLE_Pos      0

/* I2C_CTRL0 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t CTRL0_BYTE;           
    __IO uint8_t SLAVE_ADDRESS_BYTE;   
    __IO uint8_t MASTER_SPEED_PRESCALAR_BYTE;
         uint8_t RESERVED0[1];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000900, 0) - (0x40000900 + 4)];
    __IO uint32_t SLAVE_ENABLE_ALIAS;   /* Select whether the I2C interface will be enabled for slave mode or not */
         uint32_t RESERVED1;
    __IO uint32_t AUTO_ACK_ENABLE_ALIAS;/* Select whether acknowledgement is automatically generated or not */
    __IO uint32_t STOP_INT_ENABLE_ALIAS;/* Configure whether stop interrupts will be generated by the I2C interface */
    __IO uint32_t CONTROLLER_ALIAS;     /* Select whether data transfer will be controlled by the ARM Cortex-M3 processor or the DMA for I2C */
} I2C_CTRL0_Type;

#define I2C_CTRL0                       ((I2C_CTRL0_Type *) I2C_CTRL0_BASE)

/* I2C_CTRL0 settings */
#define I2C_SLAVE_DISABLE_BITBAND       0x0
#define I2C_SLAVE_ENABLE_BITBAND        0x1
#define I2C_SLAVE_DISABLE               ((uint32_t)(I2C_SLAVE_DISABLE_BITBAND << I2C_CTRL0_SLAVE_ENABLE_Pos))
#define I2C_SLAVE_ENABLE                ((uint32_t)(I2C_SLAVE_ENABLE_BITBAND << I2C_CTRL0_SLAVE_ENABLE_Pos))

#define I2C_AUTO_ACK_DISABLE_BITBAND    0x0
#define I2C_AUTO_ACK_ENABLE_BITBAND     0x1
#define I2C_AUTO_ACK_DISABLE            ((uint32_t)(I2C_AUTO_ACK_DISABLE_BITBAND << I2C_CTRL0_AUTO_ACK_ENABLE_Pos))
#define I2C_AUTO_ACK_ENABLE             ((uint32_t)(I2C_AUTO_ACK_ENABLE_BITBAND << I2C_CTRL0_AUTO_ACK_ENABLE_Pos))

#define I2C_STOP_INT_DISABLE_BITBAND    0x0
#define I2C_STOP_INT_ENABLE_BITBAND     0x1
#define I2C_STOP_INT_DISABLE            ((uint32_t)(I2C_STOP_INT_DISABLE_BITBAND << I2C_CTRL0_STOP_INT_ENABLE_Pos))
#define I2C_STOP_INT_ENABLE             ((uint32_t)(I2C_STOP_INT_ENABLE_BITBAND << I2C_CTRL0_STOP_INT_ENABLE_Pos))

#define I2C_CONTROLLER_CM3_BITBAND      0x0
#define I2C_CONTROLLER_DMA_BITBAND      0x1
#define I2C_CONTROLLER_CM3              ((uint32_t)(I2C_CONTROLLER_CM3_BITBAND << I2C_CTRL0_CONTROLLER_Pos))
#define I2C_CONTROLLER_DMA              ((uint32_t)(I2C_CONTROLLER_DMA_BITBAND << I2C_CTRL0_CONTROLLER_Pos))

#define I2C_MASTER_SPEED_3              ((uint32_t)(0x0U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_6              ((uint32_t)(0x1U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_9              ((uint32_t)(0x2U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_12             ((uint32_t)(0x3U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_15             ((uint32_t)(0x4U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_18             ((uint32_t)(0x5U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_21             ((uint32_t)(0x6U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_24             ((uint32_t)(0x7U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_27             ((uint32_t)(0x8U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_30             ((uint32_t)(0x9U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_33             ((uint32_t)(0xAU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_36             ((uint32_t)(0xBU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_39             ((uint32_t)(0xCU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_42             ((uint32_t)(0xDU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_45             ((uint32_t)(0xEU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_48             ((uint32_t)(0xFU << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))
#define I2C_MASTER_SPEED_51             ((uint32_t)(0x10U << I2C_CTRL0_MASTER_SPEED_PRESCALAR_Pos))

/* I2C_CTRL0 sub-register bit positions */
#define I2C_CTRL0_BYTE_SLAVE_ENABLE_Pos 0
#define I2C_CTRL0_BYTE_AUTO_ACK_ENABLE_Pos 2
#define I2C_CTRL0_BYTE_STOP_INT_ENABLE_Pos 3
#define I2C_CTRL0_BYTE_CONTROLLER_Pos   4
#define I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_Pos 0
#define I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_Mask ((uint32_t)(0x7FU << I2C_SLAVE_ADDRESS_BYTE_SLAVE_ADDRESS_Pos))

/* I2C_CTRL0 subregister settings */
#define I2C_SLAVE_DISABLE_BYTE          ((uint8_t)(I2C_SLAVE_DISABLE_BITBAND << I2C_CTRL0_BYTE_SLAVE_ENABLE_Pos))
#define I2C_SLAVE_ENABLE_BYTE           ((uint8_t)(I2C_SLAVE_ENABLE_BITBAND << I2C_CTRL0_BYTE_SLAVE_ENABLE_Pos))

#define I2C_AUTO_ACK_DISABLE_BYTE       ((uint8_t)(I2C_AUTO_ACK_DISABLE_BITBAND << I2C_CTRL0_BYTE_AUTO_ACK_ENABLE_Pos))
#define I2C_AUTO_ACK_ENABLE_BYTE        ((uint8_t)(I2C_AUTO_ACK_ENABLE_BITBAND << I2C_CTRL0_BYTE_AUTO_ACK_ENABLE_Pos))

#define I2C_STOP_INT_DISABLE_BYTE       ((uint8_t)(I2C_STOP_INT_DISABLE_BITBAND << I2C_CTRL0_BYTE_STOP_INT_ENABLE_Pos))
#define I2C_STOP_INT_ENABLE_BYTE        ((uint8_t)(I2C_STOP_INT_ENABLE_BITBAND << I2C_CTRL0_BYTE_STOP_INT_ENABLE_Pos))

#define I2C_CONTROLLER_CM3_BYTE         ((uint8_t)(I2C_CONTROLLER_CM3_BITBAND << I2C_CTRL0_BYTE_CONTROLLER_Pos))
#define I2C_CONTROLLER_DMA_BYTE         ((uint8_t)(I2C_CONTROLLER_DMA_BITBAND << I2C_CTRL0_BYTE_CONTROLLER_Pos))

#define I2C_MASTER_SPEED_3_BYTE         0x00000000
#define I2C_MASTER_SPEED_6_BYTE         0x00000001
#define I2C_MASTER_SPEED_9_BYTE         0x00000002
#define I2C_MASTER_SPEED_12_BYTE        0x00000003
#define I2C_MASTER_SPEED_15_BYTE        0x00000004
#define I2C_MASTER_SPEED_18_BYTE        0x00000005
#define I2C_MASTER_SPEED_21_BYTE        0x00000006
#define I2C_MASTER_SPEED_24_BYTE        0x00000007
#define I2C_MASTER_SPEED_27_BYTE        0x00000008
#define I2C_MASTER_SPEED_30_BYTE        0x00000009
#define I2C_MASTER_SPEED_33_BYTE        0x0000000A
#define I2C_MASTER_SPEED_36_BYTE        0x0000000B
#define I2C_MASTER_SPEED_39_BYTE        0x0000000C
#define I2C_MASTER_SPEED_42_BYTE        0x0000000D
#define I2C_MASTER_SPEED_45_BYTE        0x0000000E
#define I2C_MASTER_SPEED_48_BYTE        0x0000000F
#define I2C_MASTER_SPEED_51_BYTE        0x00000010

/* I2C Interface Status and Control */
/*   Control transfers using the I2C interface */
#define I2C_CTRL1_BASE                  0x40000904

/* I2C_CTRL1 bit positions */
#define I2C_CTRL1_RESET_Pos             5
#define I2C_CTRL1_LAST_DATA_Pos         4
#define I2C_CTRL1_STOP_Pos              3
#define I2C_CTRL1_NACK_Pos              1
#define I2C_CTRL1_ACK_Pos               0

/* I2C_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __O  uint32_t ACK_ALIAS;            /* Issue an acknowledge on the I2C interface bus */
    __O  uint32_t NACK_ALIAS;           /* Issue a not acknowledge on the I2C interface bus */
         uint32_t RESERVED0;
    __O  uint32_t STOP_ALIAS;           /* Issue a stop condition on the I2C interface bus */
    __IO uint32_t LAST_DATA_ALIAS;      /* Indicate that the current data is the last byte of a data transfer */
    __O  uint32_t RESET_ALIAS;          /* Reset the I2C interface */
} I2C_CTRL1_Type;

#define I2C_CTRL1                       ((I2C_CTRL1_Type *) SYS_CALC_BITBAND(I2C_CTRL1_BASE, 0))

/* I2C_CTRL1 settings */
#define I2C_ACK_BITBAND                 0x1
#define I2C_ACK                         ((uint32_t)(I2C_ACK_BITBAND << I2C_CTRL1_ACK_Pos))

#define I2C_NACK_BITBAND                0x1
#define I2C_NACK                        ((uint32_t)(I2C_NACK_BITBAND << I2C_CTRL1_NACK_Pos))

#define I2C_STOP_BITBAND                0x1
#define I2C_STOP                        ((uint32_t)(I2C_STOP_BITBAND << I2C_CTRL1_STOP_Pos))

#define I2C_LAST_DATA_BITBAND           0x1
#define I2C_LAST_DATA                   ((uint32_t)(I2C_LAST_DATA_BITBAND << I2C_CTRL1_LAST_DATA_Pos))

#define I2C_RESET_BITBAND               0x1
#define I2C_RESET                       ((uint32_t)(I2C_RESET_BITBAND << I2C_CTRL1_RESET_Pos))

/* I2C Physical Interface Control */
/*   Configure the physical pads used by the I2C interface */
#define I2C_PHY_CTRL_BASE               0x40000908

/* I2C_PHY_CTRL bit positions */
#define I2C_PHY_CTRL_PULLUP_SELECT_Pos  1
#define I2C_PHY_CTRL_FILTER_ENABLE_Pos  0

/* I2C_PHY_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t FILTER_ENABLE_ALIAS;  /* Configure whether the I2C bus lines will be low-pass filtered */
    __IO uint32_t PULLUP_SELECT_ALIAS;  /* Configure the optional pull-up resistors for the I2C interface lines */
} I2C_PHY_CTRL_Type;

#define I2C_PHY_CTRL                    ((I2C_PHY_CTRL_Type *) SYS_CALC_BITBAND(I2C_PHY_CTRL_BASE, 0))

/* I2C_PHY_CTRL settings */
#define I2C_FILTER_DISABLE_BITBAND      0x0
#define I2C_FILTER_ENABLE_BITBAND       0x1
#define I2C_FILTER_DISABLE              ((uint32_t)(I2C_FILTER_DISABLE_BITBAND << I2C_PHY_CTRL_FILTER_ENABLE_Pos))
#define I2C_FILTER_ENABLE               ((uint32_t)(I2C_FILTER_ENABLE_BITBAND << I2C_PHY_CTRL_FILTER_ENABLE_Pos))

#define I2C_PULLUP_1K_BITBAND           0x0
#define I2C_PULLUP_DISABLE_BITBAND      0x1
#define I2C_PULLUP_1K                   ((uint32_t)(I2C_PULLUP_1K_BITBAND << I2C_PHY_CTRL_PULLUP_SELECT_Pos))
#define I2C_PULLUP_DISABLE              ((uint32_t)(I2C_PULLUP_DISABLE_BITBAND << I2C_PHY_CTRL_PULLUP_SELECT_Pos))

/* I2C Interface Data */
/*   Single byte buffer for data to be transmitted over the I2C interface and
 *   data that has been received over the I2C interface */
#define I2C_DATA_BASE                   0x40000910

/* I2C Interface Data (Shadow) */
/*   Shadow of the single byte buffer for data to be transmitted over the I2C
 *   interface and data that has been received over the I2C interface */
#define I2C_DATA_S_BASE                 0x40000914

/* I2C Master Address and Start */
/*   Start a master I2C transaction with the provided address and read-write
 *   bit */
#define I2C_ADDR_START_BASE             0x40000918

/* I2C_ADDR_START bit positions */
#define I2C_ADDR_START_ADDRESS_Pos      1
#define I2C_ADDR_START_ADDRESS_Mask     ((uint32_t)(0x7FU << I2C_ADDR_START_ADDRESS_Pos))
#define I2C_ADDR_START_READ_WRITE_Pos   0

/* I2C_ADDR_START sub-register and bit-band aliases */
typedef struct
{
    __O  uint32_t READ_WRITE_ALIAS;     /* Select whether a read or a write transaction is started */
} I2C_ADDR_START_Type;

#define I2C_ADDR_START                  ((I2C_ADDR_START_Type *) SYS_CALC_BITBAND(I2C_ADDR_START_BASE, 0))

/* I2C_ADDR_START settings */
#define I2C_WRITE_BITBAND               0x0
#define I2C_READ_BITBAND                0x1
#define I2C_WRITE                       ((uint32_t)(I2C_WRITE_BITBAND << I2C_ADDR_START_READ_WRITE_Pos))
#define I2C_READ                        ((uint32_t)(I2C_READ_BITBAND << I2C_ADDR_START_READ_WRITE_Pos))

/* I2C Status */
/*   Indicate the status of the I2C interface */
#define I2C_STATUS_BASE                 0x4000091C

/* I2C_STATUS bit positions */
#define I2C_STATUS_DMA_REQ_Pos          11
#define I2C_STATUS_STOP_DETECT_Pos      10
#define I2C_STATUS_DATA_EVENT_Pos       9
#define I2C_STATUS_ERROR_Pos            8
#define I2C_STATUS_BUS_ERROR_Pos        7
#define I2C_STATUS_BUFFER_FULL_Pos      6
#define I2C_STATUS_CLK_STRETCH_Pos      5
#define I2C_STATUS_BUS_FREE_Pos         4
#define I2C_STATUS_ADDR_DATA_Pos        3
#define I2C_STATUS_READ_WRITE_Pos       2
#define I2C_STATUS_GEN_CALL_Pos         1
#define I2C_STATUS_ACK_STATUS_Pos       0

/* I2C_STATUS sub-register and bit-band aliases */
typedef struct
{
    __I  uint32_t ACK_STATUS_ALIAS;     /* Indicate whether an acknowledge or a not acknowledge has been received */
    __I  uint32_t GEN_CALL_ALIAS;       /* Indicate whether the I2C bus transfer is using the general call address or another address */
    __I  uint32_t READ_WRITE_ALIAS;     /* Indicate whether the I2C bus transfer is a read or a write */
    __I  uint32_t ADDR_DATA_ALIAS;      /* Indicate if the I2C data register holds an address or data byte */
    __I  uint32_t BUS_FREE_ALIAS;       /* Indicate if the I2C interface bus is free */
    __I  uint32_t CLK_STRETCH_ALIAS;    /* Indicate if the I2C interface is holding the clock signal */
    __I  uint32_t BUFFER_FULL_ALIAS;    /* Indicate if the I2C data buffer is full */
    __I  uint32_t BUS_ERROR_ALIAS;      /* Indicate if the I2C interface state machine indicates a bus error has occurred */
    __I  uint32_t ERROR_ALIAS;          /* Indicate if an error has occurred on the I2C interface since this bit was last cleared */
    __I  uint32_t DATA_EVENT_ALIAS;     /* Indicate that the I2C interface either needs data to transmit or has received data */
    __I  uint32_t STOP_DETECT_ALIAS;    /* Indicate if an I2C stop bit has been detected */
    __I  uint32_t DMA_REQ_ALIAS;        /* Indicate if the I2C interface is currently requesting DMA data */
} I2C_STATUS_Type;

#define I2C_STATUS                      ((I2C_STATUS_Type *) SYS_CALC_BITBAND(I2C_STATUS_BASE, 0))

/* I2C_STATUS settings */
#define I2C_HAS_ACK_BITBAND             0x0
#define I2C_HAS_NACK_BITBAND            0x1
#define I2C_HAS_ACK                     ((uint32_t)(I2C_HAS_ACK_BITBAND << I2C_STATUS_ACK_STATUS_Pos))
#define I2C_HAS_NACK                    ((uint32_t)(I2C_HAS_NACK_BITBAND << I2C_STATUS_ACK_STATUS_Pos))

#define I2C_ADDR_OTHER_BITBAND          0x0
#define I2C_ADDR_GEN_CALL_BITBAND       0x1
#define I2C_ADDR_OTHER                  ((uint32_t)(I2C_ADDR_OTHER_BITBAND << I2C_STATUS_GEN_CALL_Pos))
#define I2C_ADDR_GEN_CALL               ((uint32_t)(I2C_ADDR_GEN_CALL_BITBAND << I2C_STATUS_GEN_CALL_Pos))

#define I2C_IS_WRITE_BITBAND            0x0
#define I2C_IS_READ_BITBAND             0x1
#define I2C_IS_WRITE                    ((uint32_t)(I2C_IS_WRITE_BITBAND << I2C_STATUS_READ_WRITE_Pos))
#define I2C_IS_READ                     ((uint32_t)(I2C_IS_READ_BITBAND << I2C_STATUS_READ_WRITE_Pos))

#define I2C_DATA_IS_DATA_BITBAND        0x0
#define I2C_DATA_IS_ADDR_BITBAND        0x1
#define I2C_DATA_IS_DATA                ((uint32_t)(I2C_DATA_IS_DATA_BITBAND << I2C_STATUS_ADDR_DATA_Pos))
#define I2C_DATA_IS_ADDR                ((uint32_t)(I2C_DATA_IS_ADDR_BITBAND << I2C_STATUS_ADDR_DATA_Pos))

#define I2C_BUS_BUSY_BITBAND            0x0
#define I2C_BUS_FREE_BITBAND            0x1
#define I2C_BUS_BUSY                    ((uint32_t)(I2C_BUS_BUSY_BITBAND << I2C_STATUS_BUS_FREE_Pos))
#define I2C_BUS_FREE                    ((uint32_t)(I2C_BUS_FREE_BITBAND << I2C_STATUS_BUS_FREE_Pos))

#define I2C_CLK_NOT_STRETCHED_BITBAND   0x0
#define I2C_CLK_STRETCHED_BITBAND       0x1
#define I2C_CLK_NOT_STRETCHED           ((uint32_t)(I2C_CLK_NOT_STRETCHED_BITBAND << I2C_STATUS_CLK_STRETCH_Pos))
#define I2C_CLK_STRETCHED               ((uint32_t)(I2C_CLK_STRETCHED_BITBAND << I2C_STATUS_CLK_STRETCH_Pos))

#define I2C_BUFFER_EMPTY_BITBAND        0x0
#define I2C_BUFFER_FULL_BITBAND         0x1
#define I2C_BUFFER_EMPTY                ((uint32_t)(I2C_BUFFER_EMPTY_BITBAND << I2C_STATUS_BUFFER_FULL_Pos))
#define I2C_BUFFER_FULL                 ((uint32_t)(I2C_BUFFER_FULL_BITBAND << I2C_STATUS_BUFFER_FULL_Pos))

#define I2C_NO_BUS_ERROR_BITBAND        0x0
#define I2C_BUS_ERROR_BITBAND           0x1
#define I2C_NO_BUS_ERROR                ((uint32_t)(I2C_NO_BUS_ERROR_BITBAND << I2C_STATUS_BUS_ERROR_Pos))
#define I2C_BUS_ERROR                   ((uint32_t)(I2C_BUS_ERROR_BITBAND << I2C_STATUS_BUS_ERROR_Pos))

#define I2C_NO_ERROR_BITBAND            0x0
#define I2C_ERROR_BITBAND               0x1
#define I2C_NO_ERROR                    ((uint32_t)(I2C_NO_ERROR_BITBAND << I2C_STATUS_ERROR_Pos))
#define I2C_ERROR                       ((uint32_t)(I2C_ERROR_BITBAND << I2C_STATUS_ERROR_Pos))

#define I2C_NON_DATA_EVENT_BITBAND      0x0
#define I2C_DATA_EVENT_BITBAND          0x1
#define I2C_NON_DATA_EVENT              ((uint32_t)(I2C_NON_DATA_EVENT_BITBAND << I2C_STATUS_DATA_EVENT_Pos))
#define I2C_DATA_EVENT                  ((uint32_t)(I2C_DATA_EVENT_BITBAND << I2C_STATUS_DATA_EVENT_Pos))

#define I2C_NO_STOP_DETECTED_BITBAND    0x0
#define I2C_STOP_DETECTED_BITBAND       0x1
#define I2C_NO_STOP_DETECTED            ((uint32_t)(I2C_NO_STOP_DETECTED_BITBAND << I2C_STATUS_STOP_DETECT_Pos))
#define I2C_STOP_DETECTED               ((uint32_t)(I2C_STOP_DETECTED_BITBAND << I2C_STATUS_STOP_DETECT_Pos))

#define I2C_NO_DMA_REQUEST_BITBAND      0x0
#define I2C_DMA_REQUEST_BITBAND         0x1
#define I2C_NO_DMA_REQUEST              ((uint32_t)(I2C_NO_DMA_REQUEST_BITBAND << I2C_STATUS_DMA_REQ_Pos))
#define I2C_DMA_REQUEST                 ((uint32_t)(I2C_DMA_REQUEST_BITBAND << I2C_STATUS_DMA_REQ_Pos))

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

typedef struct
{
    __IO uint32_t CTRL;                 /* CRC generator data input configuration */
    __IO uint32_t DATA;                 /* CRC generator data. Set to 0xFFFF to start the calculation and provides the current CRC. */
    __O  uint32_t ADD_1;                /* Add 1 bit to the CRC calculation */
    __O  uint32_t ADD_8;                /* Add 1 byte (8 bits) to the CRC calculation */
    __O  uint32_t ADD_16;               /* Add 1 half-word (16 bits) to the CRC calculation */
    __O  uint32_t ADD_24;               /* Add 3 bytes (24 bits) to the CRC calculation */
    __O  uint32_t ADD_32;               /* Add 1 word (32 bits) to the CRC calculation */
} CRC_Type;

#define CRC_BASE                        0x40000A00
#define CRC                             ((CRC_Type *) CRC_BASE)

/* CRC Generator Control */
/*   CRC generator data input configuration */
#define CRC_CTRL_BASE                   0x40000A00

/* CRC_CTRL bit positions */
#define CRC_CTRL_BYTE_ORDER_Pos         0

/* CRC_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t BYTE_ORDER_ALIAS;     /* Select the endianess for bytes added to the CRC */
} CRC_CTRL_Type;

#define CRC_CTRL                        ((CRC_CTRL_Type *) SYS_CALC_BITBAND(CRC_CTRL_BASE, 0))

/* CRC_CTRL settings */
#define CRC_BIG_ENDIAN_BITBAND          0x0
#define CRC_LITTLE_ENDIAN_BITBAND       0x1
#define CRC_BIG_ENDIAN                  ((uint32_t)(CRC_BIG_ENDIAN_BITBAND << CRC_CTRL_BYTE_ORDER_Pos))
#define CRC_LITTLE_ENDIAN               ((uint32_t)(CRC_LITTLE_ENDIAN_BITBAND << CRC_CTRL_BYTE_ORDER_Pos))

/* CRC Generator Data */
/*   CRC generator data. Set to 0xFFFF to start the calculation and provides
 *   the current CRC. */
#define CRC_DATA_BASE                   0x40000A04

/* CRC_DATA bit positions */
#define CRC_DATA_CURRENT_CRC_Pos        0
#define CRC_DATA_CURRENT_CRC_Mask       ((uint32_t)(0xFFFFU << CRC_DATA_CURRENT_CRC_Pos))

/* CRC_DATA settings */
#define CRC_INIT_VALUE                  ((uint32_t)(0xFFFFU << CRC_DATA_CURRENT_CRC_Pos))

/* CRC Generator - Add 1 Bit */
/*   Add 1 bit to the CRC calculation */
#define CRC_ADD_1_BASE                  0x40000A08

/* CRC_ADD_1 bit positions */
#define CRC_ADD_1_CRC_ADD_1_BIT_Pos     0

/* CRC_ADD_1 sub-register and bit-band aliases */
typedef struct
{
    __O  uint32_t CRC_ADD_1_BIT_ALIAS;  /* Add one bit to the CRC calculation */
} CRC_ADD_1_Type;

#define CRC_ADD_1                       ((CRC_ADD_1_Type *) SYS_CALC_BITBAND(CRC_ADD_1_BASE, 0))

/* CRC Generator - Add 1 Byte */
/*   Add 1 byte (8 bits) to the CRC calculation */
#define CRC_ADD_8_BASE                  0x40000A0C

/* CRC Generator - Add 1 Half-word */
/*   Add 1 half-word (16 bits) to the CRC calculation */
#define CRC_ADD_16_BASE                 0x40000A10

/* CRC Generator - Add 3 Bytes */
/*   Add 3 bytes (24 bits) to the CRC calculation */
#define CRC_ADD_24_BASE                 0x40000A14

/* CRC_ADD_24 bit positions */
#define CRC_ADD_24_CRC_ADD_24_BITS_Pos  0
#define CRC_ADD_24_CRC_ADD_24_BITS_Mask ((uint32_t)(0xFFFFFFU << CRC_ADD_24_CRC_ADD_24_BITS_Pos))

/* CRC Generator - Add 1 Word */
/*   Add 1 word (32 bits) to the CRC calculation */
#define CRC_ADD_32_BASE                 0x40000A18

/* ----------------------------------------------------------------------------
 * USB Interface
 * ------------------------------------------------------------------------- */
/* The universal serial bus (USB) interface implements the USB 1.1 standard in
 * full-speed mode.
 *
 * This register set provides top-level control of the logic behind the USB
 * interface. */

typedef struct
{
    __IO uint32_t CTRL;                 /* Control the USB interface at a high-level. */
} USB_TOP_Type;

#define USB_TOP_BASE                    0x40000B00
#define USB_TOP                         ((USB_TOP_Type *) USB_TOP_BASE)

/* USB Control */
/*   Control the USB interface at a high-level. */
#define USB_CTRL_BASE                   0x40000B00

/* USB_CTRL bit positions */
#define USB_CTRL_CONTROLLER_Pos         4
#define USB_CTRL_PHY_STANDBY_Pos        3
#define USB_CTRL_REMOTE_WAKEUP_Pos      2
#define USB_CTRL_RESET_Pos              1
#define USB_CTRL_ENABLE_Pos             0

/* USB_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ENABLE_ALIAS;         /* Enable/disable the USB interface controller */
    __O  uint32_t RESET_ALIAS;          /* Reset the USB interface controller */
    __O  uint32_t REMOTE_WAKEUP_ALIAS;  /* Initiate a remote wakeup sequence */
    __IO uint32_t PHY_STANDBY_ALIAS;    /* Put the USB PHY into standby mode */
    __IO uint32_t CONTROLLER_ALIAS;     /* Select whether data transfer will be controlled by the ARM Cortex-M3 processor or the DMA */
} USB_CTRL_Type;

#define USB_CTRL                        ((USB_CTRL_Type *) SYS_CALC_BITBAND(USB_CTRL_BASE, 0))

/* USB_CTRL settings */
#define USB_DISABLE_BITBAND             0x0
#define USB_ENABLE_BITBAND              0x1
#define USB_DISABLE                     ((uint32_t)(USB_DISABLE_BITBAND << USB_CTRL_ENABLE_Pos))
#define USB_ENABLE                      ((uint32_t)(USB_ENABLE_BITBAND << USB_CTRL_ENABLE_Pos))

#define USB_RESET_ENABLE_BITBAND        0x1
#define USB_RESET_DISABLE_BITBAND       0x0
#define USB_RESET_ENABLE                ((uint32_t)(USB_RESET_ENABLE_BITBAND << USB_CTRL_RESET_Pos))
#define USB_RESET_DISABLE               ((uint32_t)(USB_RESET_DISABLE_BITBAND << USB_CTRL_RESET_Pos))

#define USB_REMOTE_WAKEUP_BITBAND       0x1
#define USB_REMOTE_WAKEUP               ((uint32_t)(USB_REMOTE_WAKEUP_BITBAND << USB_CTRL_REMOTE_WAKEUP_Pos))

#define USB_PHY_ENABLED_BITBAND         0x0
#define USB_PHY_STANDBY_BITBAND         0x1
#define USB_PHY_ENABLED                 ((uint32_t)(USB_PHY_ENABLED_BITBAND << USB_CTRL_PHY_STANDBY_Pos))
#define USB_PHY_STANDBY                 ((uint32_t)(USB_PHY_STANDBY_BITBAND << USB_CTRL_PHY_STANDBY_Pos))

#define USB_CONTROLLER_CM3_BITBAND      0x0
#define USB_CONTROLLER_DMA_BITBAND      0x1
#define USB_CONTROLLER_CM3              ((uint32_t)(USB_CONTROLLER_CM3_BITBAND << USB_CTRL_CONTROLLER_Pos))
#define USB_CONTROLLER_DMA              ((uint32_t)(USB_CONTROLLER_DMA_BITBAND << USB_CTRL_CONTROLLER_Pos))

/* ----------------------------------------------------------------------------
 * Timer 0 to 3 Configuration and Control
 * ------------------------------------------------------------------------- */
/* Four general purpose timers are provided. These timers support generating
 * interrupts at a configurable interval, for both a limited amount of time or
 * continuously */

typedef struct
{
    __IO uint32_t TIMER0_CTRL;          /* Configure general purpose timer 0 */
    __IO uint32_t TIMER1_CTRL;          /* Configure general purpose timer 1 */
    __IO uint32_t TIMER2_CTRL;          /* Configure general purpose timer 2 */
    __IO uint32_t TIMER3_CTRL;          /* Configure general purpose timer 3 */
    __IO uint32_t CTRL_STATUS;          /* Control and indicate the status of the general purpose timers */
} TIMER_Type;

#define TIMER_BASE                      0x40000C00
#define TIMER                           ((TIMER_Type *) TIMER_BASE)

/* Timer 0 Control and Configuration */
/*   Configure general purpose timer 0 */
#define TIMER0_CTRL_BASE                0x40000C00

/* TIMER0_CTRL bit positions */
#define TIMER0_CTRL_MULTI_COUNT_Pos     16
#define TIMER0_CTRL_MULTI_COUNT_Mask    ((uint32_t)(0x7U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_CTRL_MODE_Pos            15
#define TIMER0_CTRL_PRESCALE_Pos        12
#define TIMER0_CTRL_PRESCALE_Mask       ((uint32_t)(0x7U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_CTRL_TIMEOUT_VALUE_Pos   0
#define TIMER0_CTRL_TIMEOUT_VALUE_Mask  ((uint32_t)(0xFFFU << TIMER0_CTRL_TIMEOUT_VALUE_Pos))

/* TIMER0_CTRL settings */
#define TIMER0_COUNT_1                  ((uint32_t)(0x0U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_2                  ((uint32_t)(0x1U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_3                  ((uint32_t)(0x2U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_4                  ((uint32_t)(0x3U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_5                  ((uint32_t)(0x4U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_6                  ((uint32_t)(0x5U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_7                  ((uint32_t)(0x6U << TIMER0_CTRL_MULTI_COUNT_Pos))
#define TIMER0_COUNT_8                  ((uint32_t)(0x7U << TIMER0_CTRL_MULTI_COUNT_Pos))

#define TIMER0_FIXED_COUNT_RUN_BITBAND  0x0
#define TIMER0_FREE_RUN_BITBAND         0x1
#define TIMER0_FIXED_COUNT_RUN          ((uint32_t)(TIMER0_FIXED_COUNT_RUN_BITBAND << TIMER0_CTRL_MODE_Pos))
#define TIMER0_FREE_RUN                 ((uint32_t)(TIMER0_FREE_RUN_BITBAND << TIMER0_CTRL_MODE_Pos))

#define TIMER0_PRESCALE_1               ((uint32_t)(0x0U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_2               ((uint32_t)(0x1U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_4               ((uint32_t)(0x2U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_8               ((uint32_t)(0x3U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_16              ((uint32_t)(0x4U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_32              ((uint32_t)(0x5U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_64              ((uint32_t)(0x6U << TIMER0_CTRL_PRESCALE_Pos))
#define TIMER0_PRESCALE_128             ((uint32_t)(0x7U << TIMER0_CTRL_PRESCALE_Pos))

/* Timer 1 Control and Configuration */
/*   Configure general purpose timer 1 */
#define TIMER1_CTRL_BASE                0x40000C04

/* TIMER1_CTRL bit positions */
#define TIMER1_CTRL_MULTI_COUNT_Pos     16
#define TIMER1_CTRL_MULTI_COUNT_Mask    ((uint32_t)(0x7U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_CTRL_MODE_Pos            15
#define TIMER1_CTRL_PRESCALE_Pos        12
#define TIMER1_CTRL_PRESCALE_Mask       ((uint32_t)(0x7U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_CTRL_TIMEOUT_VALUE_Pos   0
#define TIMER1_CTRL_TIMEOUT_VALUE_Mask  ((uint32_t)(0xFFFU << TIMER1_CTRL_TIMEOUT_VALUE_Pos))

/* TIMER1_CTRL settings */
#define TIMER1_COUNT_1                  ((uint32_t)(0x0U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_2                  ((uint32_t)(0x1U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_3                  ((uint32_t)(0x2U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_4                  ((uint32_t)(0x3U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_5                  ((uint32_t)(0x4U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_6                  ((uint32_t)(0x5U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_7                  ((uint32_t)(0x6U << TIMER1_CTRL_MULTI_COUNT_Pos))
#define TIMER1_COUNT_8                  ((uint32_t)(0x7U << TIMER1_CTRL_MULTI_COUNT_Pos))

#define TIMER1_FIXED_COUNT_RUN_BITBAND  0x0
#define TIMER1_FREE_RUN_BITBAND         0x1
#define TIMER1_FIXED_COUNT_RUN          ((uint32_t)(TIMER1_FIXED_COUNT_RUN_BITBAND << TIMER1_CTRL_MODE_Pos))
#define TIMER1_FREE_RUN                 ((uint32_t)(TIMER1_FREE_RUN_BITBAND << TIMER1_CTRL_MODE_Pos))

#define TIMER1_PRESCALE_1               ((uint32_t)(0x0U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_2               ((uint32_t)(0x1U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_4               ((uint32_t)(0x2U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_8               ((uint32_t)(0x3U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_16              ((uint32_t)(0x4U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_32              ((uint32_t)(0x5U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_64              ((uint32_t)(0x6U << TIMER1_CTRL_PRESCALE_Pos))
#define TIMER1_PRESCALE_128             ((uint32_t)(0x7U << TIMER1_CTRL_PRESCALE_Pos))

/* Timer 2 Control and Configuration */
/*   Configure general purpose timer 2 */
#define TIMER2_CTRL_BASE                0x40000C08

/* TIMER2_CTRL bit positions */
#define TIMER2_CTRL_MULTI_COUNT_Pos     16
#define TIMER2_CTRL_MULTI_COUNT_Mask    ((uint32_t)(0x7U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_CTRL_MODE_Pos            15
#define TIMER2_CTRL_PRESCALE_Pos        12
#define TIMER2_CTRL_PRESCALE_Mask       ((uint32_t)(0x7U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_CTRL_TIMEOUT_VALUE_Pos   0
#define TIMER2_CTRL_TIMEOUT_VALUE_Mask  ((uint32_t)(0xFFFU << TIMER2_CTRL_TIMEOUT_VALUE_Pos))

/* TIMER2_CTRL settings */
#define TIMER2_COUNT_1                  ((uint32_t)(0x0U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_2                  ((uint32_t)(0x1U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_3                  ((uint32_t)(0x2U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_4                  ((uint32_t)(0x3U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_5                  ((uint32_t)(0x4U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_6                  ((uint32_t)(0x5U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_7                  ((uint32_t)(0x6U << TIMER2_CTRL_MULTI_COUNT_Pos))
#define TIMER2_COUNT_8                  ((uint32_t)(0x7U << TIMER2_CTRL_MULTI_COUNT_Pos))

#define TIMER2_FIXED_COUNT_RUN_BITBAND  0x0
#define TIMER2_FREE_RUN_BITBAND         0x1
#define TIMER2_FIXED_COUNT_RUN          ((uint32_t)(TIMER2_FIXED_COUNT_RUN_BITBAND << TIMER2_CTRL_MODE_Pos))
#define TIMER2_FREE_RUN                 ((uint32_t)(TIMER2_FREE_RUN_BITBAND << TIMER2_CTRL_MODE_Pos))

#define TIMER2_PRESCALE_1               ((uint32_t)(0x0U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_2               ((uint32_t)(0x1U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_4               ((uint32_t)(0x2U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_8               ((uint32_t)(0x3U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_16              ((uint32_t)(0x4U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_32              ((uint32_t)(0x5U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_64              ((uint32_t)(0x6U << TIMER2_CTRL_PRESCALE_Pos))
#define TIMER2_PRESCALE_128             ((uint32_t)(0x7U << TIMER2_CTRL_PRESCALE_Pos))

/* Timer 3 Control and Configuration */
/*   Configure general purpose timer 3 */
#define TIMER3_CTRL_BASE                0x40000C0C

/* TIMER3_CTRL bit positions */
#define TIMER3_CTRL_MULTI_COUNT_Pos     16
#define TIMER3_CTRL_MULTI_COUNT_Mask    ((uint32_t)(0x7U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_CTRL_MODE_Pos            15
#define TIMER3_CTRL_PRESCALE_Pos        12
#define TIMER3_CTRL_PRESCALE_Mask       ((uint32_t)(0x7U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_CTRL_TIMEOUT_VALUE_Pos   0
#define TIMER3_CTRL_TIMEOUT_VALUE_Mask  ((uint32_t)(0xFFFU << TIMER3_CTRL_TIMEOUT_VALUE_Pos))

/* TIMER3_CTRL settings */
#define TIMER3_COUNT_1                  ((uint32_t)(0x0U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_2                  ((uint32_t)(0x1U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_3                  ((uint32_t)(0x2U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_4                  ((uint32_t)(0x3U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_5                  ((uint32_t)(0x4U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_6                  ((uint32_t)(0x5U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_7                  ((uint32_t)(0x6U << TIMER3_CTRL_MULTI_COUNT_Pos))
#define TIMER3_COUNT_8                  ((uint32_t)(0x7U << TIMER3_CTRL_MULTI_COUNT_Pos))

#define TIMER3_FIXED_COUNT_RUN_BITBAND  0x0
#define TIMER3_FREE_RUN_BITBAND         0x1
#define TIMER3_FIXED_COUNT_RUN          ((uint32_t)(TIMER3_FIXED_COUNT_RUN_BITBAND << TIMER3_CTRL_MODE_Pos))
#define TIMER3_FREE_RUN                 ((uint32_t)(TIMER3_FREE_RUN_BITBAND << TIMER3_CTRL_MODE_Pos))

#define TIMER3_PRESCALE_1               ((uint32_t)(0x0U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_2               ((uint32_t)(0x1U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_4               ((uint32_t)(0x2U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_8               ((uint32_t)(0x3U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_16              ((uint32_t)(0x4U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_32              ((uint32_t)(0x5U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_64              ((uint32_t)(0x6U << TIMER3_CTRL_PRESCALE_Pos))
#define TIMER3_PRESCALE_128             ((uint32_t)(0x7U << TIMER3_CTRL_PRESCALE_Pos))

/* Timer Control Status for Timer 0 and Timer 1 */
/*   Control and indicate the status of the general purpose timers */
#define TIMER_CTRL_STATUS_BASE          0x40000C10

/* TIMER_CTRL_STATUS bit positions */
#define TIMER_CTRL_STATUS_TIMER3_STATUS_Pos 3
#define TIMER_CTRL_STATUS_TIMER2_STATUS_Pos 2
#define TIMER_CTRL_STATUS_TIMER1_STATUS_Pos 1
#define TIMER_CTRL_STATUS_TIMER0_STATUS_Pos 0

/* TIMER_CTRL_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t TIMER0_STATUS_ALIAS;  /* Used to start timer 0 and indicate the timer status */
    __IO uint32_t TIMER1_STATUS_ALIAS;  /* Used to start timer 1 and indicate the timer status */
    __IO uint32_t TIMER2_STATUS_ALIAS;  /* Used to start timer 2 and indicate the timer status */
    __IO uint32_t TIMER3_STATUS_ALIAS;  /* Used to start timer 3 and indicate the timer status */
} TIMER_CTRL_STATUS_Type;

#define TIMER_CTRL_STATUS               ((TIMER_CTRL_STATUS_Type *) SYS_CALC_BITBAND(TIMER_CTRL_STATUS_BASE, 0))

/* TIMER_CTRL_STATUS settings */
#define TIMER0_STOP_BITBAND             0x0
#define TIMER0_START_BITBAND            0x1
#define TIMER0_STOP                     ((uint32_t)(TIMER0_STOP_BITBAND << TIMER_CTRL_STATUS_TIMER0_STATUS_Pos))
#define TIMER0_START                    ((uint32_t)(TIMER0_START_BITBAND << TIMER_CTRL_STATUS_TIMER0_STATUS_Pos))

#define TIMER1_STOP_BITBAND             0x0
#define TIMER1_START_BITBAND            0x1
#define TIMER1_STOP                     ((uint32_t)(TIMER1_STOP_BITBAND << TIMER_CTRL_STATUS_TIMER1_STATUS_Pos))
#define TIMER1_START                    ((uint32_t)(TIMER1_START_BITBAND << TIMER_CTRL_STATUS_TIMER1_STATUS_Pos))

#define TIMER2_STOP_BITBAND             0x0
#define TIMER2_START_BITBAND            0x1
#define TIMER2_STOP                     ((uint32_t)(TIMER2_STOP_BITBAND << TIMER_CTRL_STATUS_TIMER2_STATUS_Pos))
#define TIMER2_START                    ((uint32_t)(TIMER2_START_BITBAND << TIMER_CTRL_STATUS_TIMER2_STATUS_Pos))

#define TIMER3_STOP_BITBAND             0x0
#define TIMER3_START_BITBAND            0x1
#define TIMER3_STOP                     ((uint32_t)(TIMER3_STOP_BITBAND << TIMER_CTRL_STATUS_TIMER3_STATUS_Pos))
#define TIMER3_START                    ((uint32_t)(TIMER3_START_BITBAND << TIMER_CTRL_STATUS_TIMER3_STATUS_Pos))

/* ----------------------------------------------------------------------------
 * Watchdog Timer Configuration and Control
 * ------------------------------------------------------------------------- */
/* The watchdog timer is an independent timer that monitors whether a device
 * continues to operate.
 *
 * This timer must be refreshed periodically in order to avoid a system
 * reset. */

typedef struct
{
    __O  uint32_t REFRESH_CTRL;         /* Used to restart the Watchdog timer to prevent a watchdog reset. */
    __IO uint32_t CTRL;                 /* Configure the watchdog timer timeout period */
} WATCHDOG_Type;

#define WATCHDOG_BASE                   0x40000D00
#define WATCHDOG                        ((WATCHDOG_Type *) WATCHDOG_BASE)

/* Watchdog Refresh Control */
/*   Used to restart the Watchdog timer to prevent a watchdog reset. */
#define WATCHDOG_REFRESH_CTRL_BASE      0x40000D00

/* WATCHDOG_REFRESH_CTRL bit positions */
#define WATCHDOG_REFRESH_CTRL_REFRESH_Pos 0

/* WATCHDOG_REFRESH_CTRL sub-register and bit-band aliases */
typedef struct
{
    __O  uint32_t REFRESH_ALIAS;        /* Used to restart the Watchdog timer to prevent a watchdog reset. An application should periodically write to this bit. */
} WATCHDOG_REFRESH_CTRL_Type;

#define WATCHDOG_REFRESH_CTRL           ((WATCHDOG_REFRESH_CTRL_Type *) SYS_CALC_BITBAND(WATCHDOG_REFRESH_CTRL_BASE, 0))

/* WATCHDOG_REFRESH_CTRL settings */
#define WATCHDOG_REFRESH_BITBAND        0x1
#define WATCHDOG_REFRESH                ((uint32_t)(WATCHDOG_REFRESH_BITBAND << WATCHDOG_REFRESH_CTRL_REFRESH_Pos))

/* Watchdog Timer Control */
/*   Configure the watchdog timer timeout period */
#define WATCHDOG_CTRL_BASE              0x40000D04

/* WATCHDOG_CTRL bit positions */
#define WATCHDOG_CTRL_TIMEOUT_Pos       0
#define WATCHDOG_CTRL_TIMEOUT_Mask      ((uint32_t)(0xFU << WATCHDOG_CTRL_TIMEOUT_Pos))

/* WATCHDOG_CTRL settings */
#define WATCHDOG_TIMEOUT_2E11           ((uint32_t)(0x0U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E12           ((uint32_t)(0x1U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E13           ((uint32_t)(0x2U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E14           ((uint32_t)(0x3U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E15           ((uint32_t)(0x4U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E16           ((uint32_t)(0x5U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E17           ((uint32_t)(0x6U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E18           ((uint32_t)(0x7U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E19           ((uint32_t)(0x8U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E20           ((uint32_t)(0x9U << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E21           ((uint32_t)(0xAU << WATCHDOG_CTRL_TIMEOUT_Pos))
#define WATCHDOG_TIMEOUT_2E22           ((uint32_t)(0xBU << WATCHDOG_CTRL_TIMEOUT_Pos))

/* ----------------------------------------------------------------------------
 * Clock Configuration
 * ------------------------------------------------------------------------- */
/* The clock configuration module configures the clock distribution tree and
 * supports prescaling of the distributed clocks. */

typedef struct
{
    __IO uint32_t CTRL0;                /* Setup the RTC clock divisor and root clock source. */
    __IO uint32_t CTRL1;                /* Configure clock divisors for the SYSTICK timer, general purpose timers and watchdog timer */
    __IO uint32_t CTRL2;                /* Configure clock divisors for the UART interfaces */
    __IO uint32_t CTRL3;                /* Configure clock divisors for the MCLK, I2C, PCM and GPIO interfaces */
    __IO uint32_t CTRL4;                /* Configure clock divisors for the ARM Cortex-M3 core, DMA and external clock output */
    __IO uint32_t CTRL5;                /* Configure clock divisors and output enables for the user output clocks */
    __IO uint32_t CTRL6;                /* Configure a clock divisor for the pulse-width modulators */
    __IO uint32_t CTRL7;                /* Configure a clock divisor used for the charge pump */
    __IO uint32_t DETECT_CTRL;          /* Configure the external clock detection circuitry */
    __IO uint32_t DETECT_STATUS;        /* Status from external clock detection circuitry */
} CLK_Type;

#define CLK_BASE                        0x40000E00
#define CLK                             ((CLK_Type *) CLK_BASE)

/* Clock control register 0 */
/*   Setup the RTC clock divisor and root clock source. */
#define CLK_CTRL0_BASE                  0x40000E00

/* CLK_CTRL0 bit positions */
#define CLK_CTRL0_RTC_CLK_DIV_Pos       2
#define CLK_CTRL0_RTC_CLK_DIV_Mask      ((uint32_t)(0x3U << CLK_CTRL0_RTC_CLK_DIV_Pos))
#define CLK_CTRL0_RCLK_SELECT_Pos       0
#define CLK_CTRL0_RCLK_SELECT_Mask      ((uint32_t)(0x3U << CLK_CTRL0_RCLK_SELECT_Pos))

/* CLK_CTRL0 settings */
#define RCLK_SELECT_RC_OSC              ((uint32_t)(0x0U << CLK_CTRL0_RCLK_SELECT_Pos))
#define RCLK_SELECT_RTC_XTAL            ((uint32_t)(0x2U << CLK_CTRL0_RCLK_SELECT_Pos))
#define RCLK_SELECT_EXT                 ((uint32_t)(0x3U << CLK_CTRL0_RCLK_SELECT_Pos))

#define RTC_CLK_SELECT_2HZ              ((uint32_t)(0x0U << CLK_CTRL0_RTC_CLK_DIV_Pos))
#define RTC_CLK_SELECT_4HZ              ((uint32_t)(0x1U << CLK_CTRL0_RTC_CLK_DIV_Pos))
#define RTC_CLK_SELECT_8HZ              ((uint32_t)(0x2U << CLK_CTRL0_RTC_CLK_DIV_Pos))
#define RTC_CLK_SELECT_16HZ             ((uint32_t)(0x3U << CLK_CTRL0_RTC_CLK_DIV_Pos))

/* Clock control register 1 */
/*   Configure clock divisors for the SYSTICK timer, general purpose timers and
 *   watchdog timer */
#define CLK_CTRL1_BASE                  0x40000E04

/* CLK_CTRL1 bit positions */
#define CLK_CTRL1_SYSTICK_DIV_Pos       24
#define CLK_CTRL1_SYSTICK_DIV_Mask      ((uint32_t)(0xFFU << CLK_CTRL1_SYSTICK_DIV_Pos))
#define CLK_CTRL1_TIMER13_DIV_Pos       16
#define CLK_CTRL1_TIMER13_DIV_Mask      ((uint32_t)(0x1FU << CLK_CTRL1_TIMER13_DIV_Pos))
#define CLK_CTRL1_TIMER02_DIV_Pos       8
#define CLK_CTRL1_TIMER02_DIV_Mask      ((uint32_t)(0x1FU << CLK_CTRL1_TIMER02_DIV_Pos))
#define CLK_CTRL1_WATCHDOG_CLK_DIV_Pos  0
#define CLK_CTRL1_WATCHDOG_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))

/* CLK_CTRL1 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t WATCHDOG_CLK_DIV_BYTE;
    __IO uint8_t TIMER02_DIV_BYTE;     
    __IO uint8_t TIMER13_DIV_BYTE;     
    __IO uint8_t SYSTICK_DIV_BYTE;     
} CLK_CTRL1_Type;

#define CLK_CTRL1                       ((CLK_CTRL1_Type *) CLK_CTRL1_BASE)

/* CLK_CTRL1 settings */
#define WATCHDOG_CLK_DIV_1              ((uint32_t)(0x0U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_2              ((uint32_t)(0x1U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_3              ((uint32_t)(0x2U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_4              ((uint32_t)(0x3U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_5              ((uint32_t)(0x4U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_6              ((uint32_t)(0x5U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_7              ((uint32_t)(0x6U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_8              ((uint32_t)(0x7U << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_16             ((uint32_t)(0xFU << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_32             ((uint32_t)(0x1FU << CLK_CTRL1_WATCHDOG_CLK_DIV_Pos))

#define TIMER02_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL1_TIMER02_DIV_Pos))
#define TIMER02_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL1_TIMER02_DIV_Pos))

#define TIMER13_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL1_TIMER13_DIV_Pos))
#define TIMER13_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL1_TIMER13_DIV_Pos))

#define SYSTICK_DISABLE                 ((uint32_t)(0xFFU << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_2                   ((uint32_t)(0x0U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_4                   ((uint32_t)(0x1U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_6                   ((uint32_t)(0x2U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_8                   ((uint32_t)(0x3U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_10                  ((uint32_t)(0x4U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_12                  ((uint32_t)(0x5U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_14                  ((uint32_t)(0x6U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_16                  ((uint32_t)(0x7U << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_32                  ((uint32_t)(0xFU << CLK_CTRL1_SYSTICK_DIV_Pos))
#define SYSTICK_DIV_64                  ((uint32_t)(0x1FU << CLK_CTRL1_SYSTICK_DIV_Pos))

/* CLK_CTRL1 sub-register bit positions */
#define CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos 0
#define CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos 0
#define CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos 0
#define CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))

/* CLK_CTRL1 subregister settings */
#define WATCHDOG_CLK_DIV_1_BYTE         ((uint8_t)(0x0U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_2_BYTE         ((uint8_t)(0x1U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_3_BYTE         ((uint8_t)(0x2U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_4_BYTE         ((uint8_t)(0x3U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_5_BYTE         ((uint8_t)(0x4U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_6_BYTE         ((uint8_t)(0x5U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_7_BYTE         ((uint8_t)(0x6U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_8_BYTE         ((uint8_t)(0x7U << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_16_BYTE        ((uint8_t)(0xFU << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))
#define WATCHDOG_CLK_DIV_32_BYTE        ((uint8_t)(0x1FU << CLK_CTRL_WATCHDOG_CLK_DIV_BYTE_WATCHDOG_CLK_DIV_Pos))

#define TIMER02_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))
#define TIMER02_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_TIMER02_DIV_BYTE_TIMER02_DIV_Pos))

#define TIMER13_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))
#define TIMER13_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_TIMER13_DIV_BYTE_TIMER13_DIV_Pos))

#define SYSTICK_DISABLE_BYTE            0x000000FF
#define SYSTICK_DIV_2_BYTE              0x00000000
#define SYSTICK_DIV_4_BYTE              0x00000001
#define SYSTICK_DIV_6_BYTE              0x00000002
#define SYSTICK_DIV_8_BYTE              0x00000003
#define SYSTICK_DIV_10_BYTE             0x00000004
#define SYSTICK_DIV_12_BYTE             0x00000005
#define SYSTICK_DIV_14_BYTE             0x00000006
#define SYSTICK_DIV_16_BYTE             0x00000007
#define SYSTICK_DIV_32_BYTE             0x0000000F
#define SYSTICK_DIV_64_BYTE             0x0000001F

/* Clock control register 2 */
/*   Configure clock divisors for the UART interfaces */
#define CLK_CTRL2_BASE                  0x40000E08

/* CLK_CTRL2 bit positions */
#define CLK_CTRL2_UART1_CLK_DIV_Pos     8
#define CLK_CTRL2_UART1_CLK_DIV_Mask    ((uint32_t)(0x1FU << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define CLK_CTRL2_UART0_CLK_DIV_Pos     0
#define CLK_CTRL2_UART0_CLK_DIV_Mask    ((uint32_t)(0x1FU << CLK_CTRL2_UART0_CLK_DIV_Pos))

/* CLK_CTRL2 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t UART0_CLK_DIV_BYTE;   
    __IO uint8_t UART1_CLK_DIV_BYTE;   
         uint8_t RESERVED0[2];
} CLK_CTRL2_Type;

#define CLK_CTRL2                       ((CLK_CTRL2_Type *) CLK_CTRL2_BASE)

/* CLK_CTRL2 settings */
#define UART0_CLK_DIV_1                 ((uint32_t)(0x0U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_2                 ((uint32_t)(0x1U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_3                 ((uint32_t)(0x2U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_4                 ((uint32_t)(0x3U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_5                 ((uint32_t)(0x4U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_6                 ((uint32_t)(0x5U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_7                 ((uint32_t)(0x6U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_8                 ((uint32_t)(0x7U << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_16                ((uint32_t)(0xFU << CLK_CTRL2_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_32                ((uint32_t)(0x1FU << CLK_CTRL2_UART0_CLK_DIV_Pos))

#define UART1_CLK_DIV_1                 ((uint32_t)(0x0U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_2                 ((uint32_t)(0x1U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_3                 ((uint32_t)(0x2U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_4                 ((uint32_t)(0x3U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_5                 ((uint32_t)(0x4U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_6                 ((uint32_t)(0x5U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_7                 ((uint32_t)(0x6U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_8                 ((uint32_t)(0x7U << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_16                ((uint32_t)(0xFU << CLK_CTRL2_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_32                ((uint32_t)(0x1FU << CLK_CTRL2_UART1_CLK_DIV_Pos))

/* CLK_CTRL2 sub-register bit positions */
#define CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos 0
#define CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos 0
#define CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))

/* CLK_CTRL2 subregister settings */
#define UART0_CLK_DIV_1_BYTE            ((uint8_t)(0x0U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_2_BYTE            ((uint8_t)(0x1U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_3_BYTE            ((uint8_t)(0x2U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_4_BYTE            ((uint8_t)(0x3U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_5_BYTE            ((uint8_t)(0x4U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_6_BYTE            ((uint8_t)(0x5U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_7_BYTE            ((uint8_t)(0x6U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_8_BYTE            ((uint8_t)(0x7U << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_16_BYTE           ((uint8_t)(0xFU << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))
#define UART0_CLK_DIV_32_BYTE           ((uint8_t)(0x1FU << CLK_CTRL_UART0_CLK_DIV_BYTE_UART0_CLK_DIV_Pos))

#define UART1_CLK_DIV_1_BYTE            ((uint8_t)(0x0U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_2_BYTE            ((uint8_t)(0x1U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_3_BYTE            ((uint8_t)(0x2U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_4_BYTE            ((uint8_t)(0x3U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_5_BYTE            ((uint8_t)(0x4U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_6_BYTE            ((uint8_t)(0x5U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_7_BYTE            ((uint8_t)(0x6U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_8_BYTE            ((uint8_t)(0x7U << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_16_BYTE           ((uint8_t)(0xFU << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))
#define UART1_CLK_DIV_32_BYTE           ((uint8_t)(0x1FU << CLK_CTRL_UART1_CLK_DIV_BYTE_UART1_CLK_DIV_Pos))

/* Clock control register 3 */
/*   Configure clock divisors for the MCLK, I2C, PCM and GPIO interfaces */
#define CLK_CTRL3_BASE                  0x40000E0C

/* CLK_CTRL3 bit positions */
#define CLK_CTRL3_MCLK_CLK_ENABLE_Pos   29
#define CLK_CTRL3_MCLK_CLK_DIV_Pos      24
#define CLK_CTRL3_MCLK_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define CLK_CTRL3_I2C_CLK_ENABLE_Pos    21
#define CLK_CTRL3_PCM_CLK_DIV_Pos       8
#define CLK_CTRL3_PCM_CLK_DIV_Mask      ((uint32_t)(0x1FU << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define CLK_CTRL3_GPIO_CLK_DIV_Pos      0
#define CLK_CTRL3_GPIO_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL3_GPIO_CLK_DIV_Pos))

/* CLK_CTRL3 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t GPIO_CLK_DIV_BYTE;    
    __IO uint8_t PCM_CLK_DIV_BYTE;     
    __IO uint8_t I2C_CLK_BYTE;         
    __IO uint8_t MCLK_DIV_BYTE;        
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000E0C, 0) - (0x40000E0C + 4)];
         uint32_t RESERVED0[21];
    __IO uint32_t I2C_CLK_ENABLE_ALIAS; /* Enable the I2C clock input to the I2C interface */
         uint32_t RESERVED1[7];
    __IO uint32_t MCLK_CLK_ENABLE_ALIAS;/* Enable the MCLK clock to the analog components */
} CLK_CTRL3_Type;

#define CLK_CTRL3                       ((CLK_CTRL3_Type *) CLK_CTRL3_BASE)

/* CLK_CTRL3 settings */
#define GPIO_CLK_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL3_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL3_GPIO_CLK_DIV_Pos))

#define PCM_CLK_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL3_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL3_PCM_CLK_DIV_Pos))

#define I2C_CLK_DISABLE_BITBAND         0x0
#define I2C_CLK_ENABLE_BITBAND          0x1
#define I2C_CLK_DISABLE                 ((uint32_t)(I2C_CLK_DISABLE_BITBAND << CLK_CTRL3_I2C_CLK_ENABLE_Pos))
#define I2C_CLK_ENABLE                  ((uint32_t)(I2C_CLK_ENABLE_BITBAND << CLK_CTRL3_I2C_CLK_ENABLE_Pos))

#define MCLK_CLK_DIV_2                  ((uint32_t)(0x0U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_4                  ((uint32_t)(0x1U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_6                  ((uint32_t)(0x2U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_8                  ((uint32_t)(0x3U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_10                 ((uint32_t)(0x4U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_12                 ((uint32_t)(0x5U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_14                 ((uint32_t)(0x6U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_16                 ((uint32_t)(0x7U << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_32                 ((uint32_t)(0xFU << CLK_CTRL3_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_64                 ((uint32_t)(0x1FU << CLK_CTRL3_MCLK_CLK_DIV_Pos))

#define MCLK_CLK_DISABLE_BITBAND        0x0
#define MCLK_CLK_ENABLE_BITBAND         0x1
#define MCLK_CLK_DISABLE                ((uint32_t)(MCLK_CLK_DISABLE_BITBAND << CLK_CTRL3_MCLK_CLK_ENABLE_Pos))
#define MCLK_CLK_ENABLE                 ((uint32_t)(MCLK_CLK_ENABLE_BITBAND << CLK_CTRL3_MCLK_CLK_ENABLE_Pos))

/* CLK_CTRL3 sub-register bit positions */
#define CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos 0
#define CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos 0
#define CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define CLK_CTRL_I2C_CLK_BYTE_I2C_CLK_ENABLE_Pos 5
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos 0
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_ENABLE_Pos 5

/* CLK_CTRL3 subregister settings */
#define GPIO_CLK_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))
#define GPIO_CLK_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_GPIO_CLK_DIV_BYTE_GPIO_CLK_DIV_Pos))

#define PCM_CLK_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))
#define PCM_CLK_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_PCM_CLK_DIV_BYTE_PCM_CLK_DIV_Pos))

#define I2C_CLK_DISABLE_BYTE            ((uint8_t)(I2C_CLK_DISABLE_BITBAND << CLK_CTRL_I2C_CLK_BYTE_I2C_CLK_ENABLE_Pos))
#define I2C_CLK_ENABLE_BYTE             ((uint8_t)(I2C_CLK_ENABLE_BITBAND << CLK_CTRL_I2C_CLK_BYTE_I2C_CLK_ENABLE_Pos))

#define MCLK_CLK_DIV_2_BYTE             ((uint8_t)(0x0U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_4_BYTE             ((uint8_t)(0x1U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_6_BYTE             ((uint8_t)(0x2U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_8_BYTE             ((uint8_t)(0x3U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_10_BYTE            ((uint8_t)(0x4U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_12_BYTE            ((uint8_t)(0x5U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_14_BYTE            ((uint8_t)(0x6U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_16_BYTE            ((uint8_t)(0x7U << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_32_BYTE            ((uint8_t)(0xFU << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))
#define MCLK_CLK_DIV_64_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_DIV_Pos))

#define MCLK_CLK_DISABLE_BYTE           ((uint8_t)(MCLK_CLK_DISABLE_BITBAND << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_ENABLE_Pos))
#define MCLK_CLK_ENABLE_BYTE            ((uint8_t)(MCLK_CLK_ENABLE_BITBAND << CLK_CTRL_MCLK_DIV_BYTE_MCLK_CLK_ENABLE_Pos))

/* Clock control register 4 */
/*   Configure clock divisors for the ARM Cortex-M3 core, DMA and external
 *   clock output */
#define CLK_CTRL4_BASE                  0x40000E10

/* CLK_CTRL4 bit positions */
#define CLK_CTRL4_CM3_CLK_DIV_Pos       24
#define CLK_CTRL4_CM3_CLK_DIV_Mask      ((uint32_t)(0x1FU << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CLK_CTRL4_EXT_CLK_DIV2_Pos      6
#define CLK_CTRL4_EXT_CLK_ENABLE_Pos    5
#define CLK_CTRL4_EXT_CLK_DIV_Pos       0
#define CLK_CTRL4_EXT_CLK_DIV_Mask      ((uint32_t)(0x1FU << CLK_CTRL4_EXT_CLK_DIV_Pos))

/* CLK_CTRL4 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t EXT_CLK_BYTE;         
         uint8_t RESERVED0[2];
    __IO uint8_t CM3_CLK_DIV_BYTE;     
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000E10, 0) - (0x40000E10 + 4)];
         uint32_t RESERVED1[5];
    __IO uint32_t EXT_CLK_ENABLE_ALIAS; /* Enable the external clock for output */
    __IO uint32_t EXT_CLK_DIV2_ALIAS;   /* Divide EXT_CLK by 2 (ensures 50% duty cycle) */
} CLK_CTRL4_Type;

#define CLK_CTRL4                       ((CLK_CTRL4_Type *) CLK_CTRL4_BASE)

/* CLK_CTRL4 settings */
#define EXT_CLK_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL4_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL4_EXT_CLK_DIV_Pos))

#define EXT_CLK_DISABLE_BITBAND         0x0
#define EXT_CLK_ENABLE_BITBAND          0x1
#define EXT_CLK_DISABLE                 ((uint32_t)(EXT_CLK_DISABLE_BITBAND << CLK_CTRL4_EXT_CLK_ENABLE_Pos))
#define EXT_CLK_ENABLE                  ((uint32_t)(EXT_CLK_ENABLE_BITBAND << CLK_CTRL4_EXT_CLK_ENABLE_Pos))

#define EXT_CLK_DIV2_DISABLE_BITBAND    0x0
#define EXT_CLK_DIV2_ENABLE_BITBAND     0x1
#define EXT_CLK_DIV2_DISABLE            ((uint32_t)(EXT_CLK_DIV2_DISABLE_BITBAND << CLK_CTRL4_EXT_CLK_DIV2_Pos))
#define EXT_CLK_DIV2_ENABLE             ((uint32_t)(EXT_CLK_DIV2_ENABLE_BITBAND << CLK_CTRL4_EXT_CLK_DIV2_Pos))

#define CM3_CLK_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL4_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL4_CM3_CLK_DIV_Pos))

/* CLK_CTRL4 sub-register bit positions */
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos 0
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_ENABLE_Pos 5
#define CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV2_Pos 6
#define CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos 0
#define CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))

/* CLK_CTRL4 subregister settings */
#define EXT_CLK_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))
#define EXT_CLK_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV_Pos))

#define EXT_CLK_DISABLE_BYTE            ((uint8_t)(EXT_CLK_DISABLE_BITBAND << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_ENABLE_Pos))
#define EXT_CLK_ENABLE_BYTE             ((uint8_t)(EXT_CLK_ENABLE_BITBAND << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_ENABLE_Pos))

#define EXT_CLK_DIV2_DISABLE_BYTE       ((uint8_t)(EXT_CLK_DIV2_DISABLE_BITBAND << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV2_Pos))
#define EXT_CLK_DIV2_ENABLE_BYTE        ((uint8_t)(EXT_CLK_DIV2_ENABLE_BITBAND << CLK_CTRL_EXT_CLK_BYTE_EXT_CLK_DIV2_Pos))

#define CM3_CLK_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))
#define CM3_CLK_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_CM3_CLK_DIV_BYTE_CM3_CLK_DIV_Pos))

/* Clock control register 5 */
/*   Configure clock divisors and output enables for the user output clocks */
#define CLK_CTRL5_BASE                  0x40000E14

/* CLK_CTRL5 bit positions */
#define CLK_CTRL5_LCD_CLK_ENABLE_Pos    31
#define CLK_CTRL5_LCD_CLK_DIV_Pos       24
#define CLK_CTRL5_LCD_CLK_DIV_Mask      ((uint32_t)(0x7FU << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define CLK_CTRL5_USR_CLK2_DIV2_Pos     22
#define CLK_CTRL5_USR_CLK2_ENABLE_Pos   21
#define CLK_CTRL5_USR_CLK2_DIV_Pos      16
#define CLK_CTRL5_USR_CLK2_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define CLK_CTRL5_USR_CLK1_DIV2_Pos     14
#define CLK_CTRL5_USR_CLK1_ENABLE_Pos   13
#define CLK_CTRL5_USR_CLK1_DIV_Pos      8
#define CLK_CTRL5_USR_CLK1_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define CLK_CTRL5_USR_CLK0_DIV2_Pos     6
#define CLK_CTRL5_USR_CLK0_ENABLE_Pos   5
#define CLK_CTRL5_USR_CLK0_DIV_Pos      0
#define CLK_CTRL5_USR_CLK0_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK0_DIV_Pos))

/* CLK_CTRL5 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t USR_CLK0_BYTE;        
    __IO uint8_t USR_CLK1_BYTE;        
    __IO uint8_t USR_CLK2_BYTE;        
    __IO uint8_t LCD_CLK_BYTE;         
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000E14, 0) - (0x40000E14 + 4)];
         uint32_t RESERVED0[5];
    __IO uint32_t USR_CLK0_ENABLE_ALIAS;/* Enable user clock 0 output */
    __IO uint32_t USR_CLK0_DIV2_ALIAS;  /* Divide user clock 0 by 2 (ensures 50% duty cycle) */
         uint32_t RESERVED1[6];
    __IO uint32_t USR_CLK1_ENABLE_ALIAS;/* Enable user clock 1 output */
    __IO uint32_t USR_CLK1_DIV2_ALIAS;  /* Divide user clock 1 by 2 (ensures 50% duty cycle) */
         uint32_t RESERVED2[6];
    __IO uint32_t USR_CLK2_ENABLE_ALIAS;/* Enable user clock 2 for output */
    __IO uint32_t USR_CLK2_DIV2_ALIAS;  /* Divide user clock 2 by 2 (ensures 50% duty cycle) */
         uint32_t RESERVED3[8];
    __IO uint32_t LCD_CLK_ENABLE_ALIAS; /* Enable LCD_CLK */
} CLK_CTRL5_Type;

#define CLK_CTRL5                       ((CLK_CTRL5_Type *) CLK_CTRL5_BASE)

/* CLK_CTRL5 settings */
#define USR_CLK0_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL5_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK0_DIV_Pos))

#define USR_CLK0_DISABLE_BITBAND        0x0
#define USR_CLK0_ENABLE_BITBAND         0x1
#define USR_CLK0_DISABLE                ((uint32_t)(USR_CLK0_DISABLE_BITBAND << CLK_CTRL5_USR_CLK0_ENABLE_Pos))
#define USR_CLK0_ENABLE                 ((uint32_t)(USR_CLK0_ENABLE_BITBAND << CLK_CTRL5_USR_CLK0_ENABLE_Pos))

#define USR_CLK0_DIV2_DISABLE_BITBAND   0x0
#define USR_CLK0_DIV2_ENABLE_BITBAND    0x1
#define USR_CLK0_DIV2_DISABLE           ((uint32_t)(USR_CLK0_DIV2_DISABLE_BITBAND << CLK_CTRL5_USR_CLK0_DIV2_Pos))
#define USR_CLK0_DIV2_ENABLE            ((uint32_t)(USR_CLK0_DIV2_ENABLE_BITBAND << CLK_CTRL5_USR_CLK0_DIV2_Pos))

#define USR_CLK1_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL5_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK1_DIV_Pos))

#define USR_CLK1_DISABLE_BITBAND        0x0
#define USR_CLK1_ENABLE_BITBAND         0x1
#define USR_CLK1_DISABLE                ((uint32_t)(USR_CLK1_DISABLE_BITBAND << CLK_CTRL5_USR_CLK1_ENABLE_Pos))
#define USR_CLK1_ENABLE                 ((uint32_t)(USR_CLK1_ENABLE_BITBAND << CLK_CTRL5_USR_CLK1_ENABLE_Pos))

#define USR_CLK1_DIV2_DISABLE_BITBAND   0x0
#define USR_CLK1_DIV2_ENABLE_BITBAND    0x1
#define USR_CLK1_DIV2_DISABLE           ((uint32_t)(USR_CLK1_DIV2_DISABLE_BITBAND << CLK_CTRL5_USR_CLK1_DIV2_Pos))
#define USR_CLK1_DIV2_ENABLE            ((uint32_t)(USR_CLK1_DIV2_ENABLE_BITBAND << CLK_CTRL5_USR_CLK1_DIV2_Pos))

#define USR_CLK2_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL5_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL5_USR_CLK2_DIV_Pos))

#define USR_CLK2_DISABLE_BITBAND        0x0
#define USR_CLK2_ENABLE_BITBAND         0x1
#define USR_CLK2_DISABLE                ((uint32_t)(USR_CLK2_DISABLE_BITBAND << CLK_CTRL5_USR_CLK2_ENABLE_Pos))
#define USR_CLK2_ENABLE                 ((uint32_t)(USR_CLK2_ENABLE_BITBAND << CLK_CTRL5_USR_CLK2_ENABLE_Pos))

#define USR_CLK2_DIV2_DISABLE_BITBAND   0x0
#define USR_CLK2_DIV2_ENABLE_BITBAND    0x1
#define USR_CLK2_DIV2_DISABLE           ((uint32_t)(USR_CLK2_DIV2_DISABLE_BITBAND << CLK_CTRL5_USR_CLK2_DIV2_Pos))
#define USR_CLK2_DIV2_ENABLE            ((uint32_t)(USR_CLK2_DIV2_ENABLE_BITBAND << CLK_CTRL5_USR_CLK2_DIV2_Pos))

#define LCD_CLK_DIV_1                   ((uint32_t)(0x0U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_2                   ((uint32_t)(0x1U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_3                   ((uint32_t)(0x2U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_4                   ((uint32_t)(0x3U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_5                   ((uint32_t)(0x4U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_6                   ((uint32_t)(0x5U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_7                   ((uint32_t)(0x6U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_8                   ((uint32_t)(0x7U << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_16                  ((uint32_t)(0xFU << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_32                  ((uint32_t)(0x1FU << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_64                  ((uint32_t)(0x3FU << CLK_CTRL5_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_128                 ((uint32_t)(0x7FU << CLK_CTRL5_LCD_CLK_DIV_Pos))

#define LCD_CLK_DISABLE_BITBAND         0x0
#define LCD_CLK_ENABLE_BITBAND          0x1
#define LCD_CLK_DISABLE                 ((uint32_t)(LCD_CLK_DISABLE_BITBAND << CLK_CTRL5_LCD_CLK_ENABLE_Pos))
#define LCD_CLK_ENABLE                  ((uint32_t)(LCD_CLK_ENABLE_BITBAND << CLK_CTRL5_LCD_CLK_ENABLE_Pos))

/* CLK_CTRL5 sub-register bit positions */
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos 0
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_Pos 5
#define CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_Pos 6
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos 0
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_Pos 5
#define CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_Pos 6
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos 0
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_Pos 5
#define CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_Pos 6
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos 0
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Mask ((uint32_t)(0x7FU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_ENABLE_Pos 7

/* CLK_CTRL5 subregister settings */
#define USR_CLK0_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))
#define USR_CLK0_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV_Pos))

#define USR_CLK0_DISABLE_BYTE           ((uint8_t)(USR_CLK0_DISABLE_BITBAND << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_Pos))
#define USR_CLK0_ENABLE_BYTE            ((uint8_t)(USR_CLK0_ENABLE_BITBAND << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_ENABLE_Pos))

#define USR_CLK0_DIV2_DISABLE_BYTE      ((uint8_t)(USR_CLK0_DIV2_DISABLE_BITBAND << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_Pos))
#define USR_CLK0_DIV2_ENABLE_BYTE       ((uint8_t)(USR_CLK0_DIV2_ENABLE_BITBAND << CLK_CTRL_USR_CLK0_BYTE_USR_CLK0_DIV2_Pos))

#define USR_CLK1_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))
#define USR_CLK1_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV_Pos))

#define USR_CLK1_DISABLE_BYTE           ((uint8_t)(USR_CLK1_DISABLE_BITBAND << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_Pos))
#define USR_CLK1_ENABLE_BYTE            ((uint8_t)(USR_CLK1_ENABLE_BITBAND << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_ENABLE_Pos))

#define USR_CLK1_DIV2_DISABLE_BYTE      ((uint8_t)(USR_CLK1_DIV2_DISABLE_BITBAND << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_Pos))
#define USR_CLK1_DIV2_ENABLE_BYTE       ((uint8_t)(USR_CLK1_DIV2_ENABLE_BITBAND << CLK_CTRL_USR_CLK1_BYTE_USR_CLK1_DIV2_Pos))

#define USR_CLK2_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))
#define USR_CLK2_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV_Pos))

#define USR_CLK2_DISABLE_BYTE           ((uint8_t)(USR_CLK2_DISABLE_BITBAND << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_Pos))
#define USR_CLK2_ENABLE_BYTE            ((uint8_t)(USR_CLK2_ENABLE_BITBAND << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_ENABLE_Pos))

#define USR_CLK2_DIV2_DISABLE_BYTE      ((uint8_t)(USR_CLK2_DIV2_DISABLE_BITBAND << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_Pos))
#define USR_CLK2_DIV2_ENABLE_BYTE       ((uint8_t)(USR_CLK2_DIV2_ENABLE_BITBAND << CLK_CTRL_USR_CLK2_BYTE_USR_CLK2_DIV2_Pos))

#define LCD_CLK_DIV_1_BYTE              ((uint8_t)(0x0U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_2_BYTE              ((uint8_t)(0x1U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_3_BYTE              ((uint8_t)(0x2U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_4_BYTE              ((uint8_t)(0x3U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_5_BYTE              ((uint8_t)(0x4U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_6_BYTE              ((uint8_t)(0x5U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_7_BYTE              ((uint8_t)(0x6U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_8_BYTE              ((uint8_t)(0x7U << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_16_BYTE             ((uint8_t)(0xFU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_32_BYTE             ((uint8_t)(0x1FU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_64_BYTE             ((uint8_t)(0x3FU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))
#define LCD_CLK_DIV_128_BYTE            ((uint8_t)(0x7FU << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_DIV_Pos))

#define LCD_CLK_DISABLE_BYTE            ((uint8_t)(LCD_CLK_DISABLE_BITBAND << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_ENABLE_Pos))
#define LCD_CLK_ENABLE_BYTE             ((uint8_t)(LCD_CLK_ENABLE_BITBAND << CLK_CTRL_LCD_CLK_BYTE_LCD_CLK_ENABLE_Pos))

/* Clock control register 6 */
/*   Configure a clock divisor for the pulse-width modulators */
#define CLK_CTRL6_BASE                  0x40000E18

/* CLK_CTRL6 bit positions */
#define CLK_CTRL6_PWM3_CLK_ENABLE_Pos   29
#define CLK_CTRL6_PWM3_CLK_DIV_Pos      24
#define CLK_CTRL6_PWM3_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define CLK_CTRL6_PWM2_CLK_ENABLE_Pos   21
#define CLK_CTRL6_PWM2_CLK_DIV_Pos      16
#define CLK_CTRL6_PWM2_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define CLK_CTRL6_PWM1_CLK_ENABLE_Pos   13
#define CLK_CTRL6_PWM1_CLK_DIV_Pos      8
#define CLK_CTRL6_PWM1_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define CLK_CTRL6_PWM0_CLK_ENABLE_Pos   5
#define CLK_CTRL6_PWM0_CLK_DIV_Pos      0
#define CLK_CTRL6_PWM0_CLK_DIV_Mask     ((uint32_t)(0x1FU << CLK_CTRL6_PWM0_CLK_DIV_Pos))

/* CLK_CTRL6 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t PWM0_CLK_BYTE;        
    __IO uint8_t PWM1_CLK_BYTE;        
    __IO uint8_t PWM2_CLK_BYTE;        
    __IO uint8_t PWM3_CLK_BYTE;        
         uint8_t RESERVED[SYS_CALC_BITBAND(0x40000E18, 0) - (0x40000E18 + 4)];
         uint32_t RESERVED0[5];
    __IO uint32_t PWM0_CLK_ENABLE_ALIAS;/* Enable the PWM0 clock */
         uint32_t RESERVED1[7];
    __IO uint32_t PWM1_CLK_ENABLE_ALIAS;/* Enable the PWM1 clock */
         uint32_t RESERVED2[7];
    __IO uint32_t PWM2_CLK_ENABLE_ALIAS;/* Enable the PWM2 clock */
         uint32_t RESERVED3[7];
    __IO uint32_t PWM3_CLK_ENABLE_ALIAS;/* Enable the PWM3 clock */
} CLK_CTRL6_Type;

#define CLK_CTRL6                       ((CLK_CTRL6_Type *) CLK_CTRL6_BASE)

/* CLK_CTRL6 settings */
#define PWM0_CLK_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL6_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL6_PWM0_CLK_DIV_Pos))

#define PWM0_CLK_DISABLE_BITBAND        0x0
#define PWM0_CLK_ENABLE_BITBAND         0x1
#define PWM0_CLK_DISABLE                ((uint32_t)(PWM0_CLK_DISABLE_BITBAND << CLK_CTRL6_PWM0_CLK_ENABLE_Pos))
#define PWM0_CLK_ENABLE                 ((uint32_t)(PWM0_CLK_ENABLE_BITBAND << CLK_CTRL6_PWM0_CLK_ENABLE_Pos))

#define PWM1_CLK_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL6_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL6_PWM1_CLK_DIV_Pos))

#define PWM1_CLK_DISABLE_BITBAND        0x0
#define PWM1_CLK_ENABLE_BITBAND         0x1
#define PWM1_CLK_DISABLE                ((uint32_t)(PWM1_CLK_DISABLE_BITBAND << CLK_CTRL6_PWM1_CLK_ENABLE_Pos))
#define PWM1_CLK_ENABLE                 ((uint32_t)(PWM1_CLK_ENABLE_BITBAND << CLK_CTRL6_PWM1_CLK_ENABLE_Pos))

#define PWM2_CLK_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL6_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL6_PWM2_CLK_DIV_Pos))

#define PWM2_CLK_DISABLE_BITBAND        0x0
#define PWM2_CLK_ENABLE_BITBAND         0x1
#define PWM2_CLK_DISABLE                ((uint32_t)(PWM2_CLK_DISABLE_BITBAND << CLK_CTRL6_PWM2_CLK_ENABLE_Pos))
#define PWM2_CLK_ENABLE                 ((uint32_t)(PWM2_CLK_ENABLE_BITBAND << CLK_CTRL6_PWM2_CLK_ENABLE_Pos))

#define PWM3_CLK_DIV_1                  ((uint32_t)(0x0U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_2                  ((uint32_t)(0x1U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_3                  ((uint32_t)(0x2U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_4                  ((uint32_t)(0x3U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_5                  ((uint32_t)(0x4U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_6                  ((uint32_t)(0x5U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_7                  ((uint32_t)(0x6U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_8                  ((uint32_t)(0x7U << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_16                 ((uint32_t)(0xFU << CLK_CTRL6_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_32                 ((uint32_t)(0x1FU << CLK_CTRL6_PWM3_CLK_DIV_Pos))

#define PWM3_CLK_DISABLE_BITBAND        0x0
#define PWM3_CLK_ENABLE_BITBAND         0x1
#define PWM3_CLK_DISABLE                ((uint32_t)(PWM3_CLK_DISABLE_BITBAND << CLK_CTRL6_PWM3_CLK_ENABLE_Pos))
#define PWM3_CLK_ENABLE                 ((uint32_t)(PWM3_CLK_ENABLE_BITBAND << CLK_CTRL6_PWM3_CLK_ENABLE_Pos))

/* CLK_CTRL6 sub-register bit positions */
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos 0
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_ENABLE_Pos 5
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos 0
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_ENABLE_Pos 5
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos 0
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_ENABLE_Pos 5
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos 0
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_ENABLE_Pos 5

/* CLK_CTRL6 subregister settings */
#define PWM0_CLK_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))
#define PWM0_CLK_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_DIV_Pos))

#define PWM0_CLK_DISABLE_BYTE           ((uint8_t)(PWM0_CLK_DISABLE_BITBAND << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_ENABLE_Pos))
#define PWM0_CLK_ENABLE_BYTE            ((uint8_t)(PWM0_CLK_ENABLE_BITBAND << CLK_CTRL_PWM0_CLK_BYTE_PWM0_CLK_ENABLE_Pos))

#define PWM1_CLK_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))
#define PWM1_CLK_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_DIV_Pos))

#define PWM1_CLK_DISABLE_BYTE           ((uint8_t)(PWM1_CLK_DISABLE_BITBAND << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_ENABLE_Pos))
#define PWM1_CLK_ENABLE_BYTE            ((uint8_t)(PWM1_CLK_ENABLE_BITBAND << CLK_CTRL_PWM1_CLK_BYTE_PWM1_CLK_ENABLE_Pos))

#define PWM2_CLK_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))
#define PWM2_CLK_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_DIV_Pos))

#define PWM2_CLK_DISABLE_BYTE           ((uint8_t)(PWM2_CLK_DISABLE_BITBAND << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_ENABLE_Pos))
#define PWM2_CLK_ENABLE_BYTE            ((uint8_t)(PWM2_CLK_ENABLE_BITBAND << CLK_CTRL_PWM2_CLK_BYTE_PWM2_CLK_ENABLE_Pos))

#define PWM3_CLK_DIV_1_BYTE             ((uint8_t)(0x0U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_2_BYTE             ((uint8_t)(0x1U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_3_BYTE             ((uint8_t)(0x2U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_4_BYTE             ((uint8_t)(0x3U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_5_BYTE             ((uint8_t)(0x4U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_6_BYTE             ((uint8_t)(0x5U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_7_BYTE             ((uint8_t)(0x6U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_8_BYTE             ((uint8_t)(0x7U << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_16_BYTE            ((uint8_t)(0xFU << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))
#define PWM3_CLK_DIV_32_BYTE            ((uint8_t)(0x1FU << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_DIV_Pos))

#define PWM3_CLK_DISABLE_BYTE           ((uint8_t)(PWM3_CLK_DISABLE_BITBAND << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_ENABLE_Pos))
#define PWM3_CLK_ENABLE_BYTE            ((uint8_t)(PWM3_CLK_ENABLE_BITBAND << CLK_CTRL_PWM3_CLK_BYTE_PWM3_CLK_ENABLE_Pos))

/* Clock control register 7 */
/*   Configure a clock divisor used for the charge pump */
#define CLK_CTRL7_BASE                  0x40000E1C

/* CLK_CTRL7 bit positions */
#define CLK_CTRL7_CP_CLK_DIV_Pos        0
#define CLK_CTRL7_CP_CLK_DIV_Mask       ((uint32_t)(0x1FU << CLK_CTRL7_CP_CLK_DIV_Pos))

/* CLK_CTRL7 sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t CP_CLK_DIV_BYTE;      
         uint8_t RESERVED0[3];
} CLK_CTRL7_Type;

#define CLK_CTRL7                       ((CLK_CTRL7_Type *) CLK_CTRL7_BASE)

/* CLK_CTRL7 settings */
#define CP_CLK_DIV_2                    ((uint32_t)(0x0U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_4                    ((uint32_t)(0x1U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_6                    ((uint32_t)(0x2U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_8                    ((uint32_t)(0x3U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_10                   ((uint32_t)(0x4U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_12                   ((uint32_t)(0x5U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_14                   ((uint32_t)(0x6U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_16                   ((uint32_t)(0x7U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_20                   ((uint32_t)(0x9U << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_24                   ((uint32_t)(0xBU << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_28                   ((uint32_t)(0xDU << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_32                   ((uint32_t)(0xFU << CLK_CTRL7_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_64                   ((uint32_t)(0x1FU << CLK_CTRL7_CP_CLK_DIV_Pos))

/* CLK_CTRL7 sub-register bit positions */
#define CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos 0
#define CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Mask ((uint32_t)(0x1FU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))

/* CLK_CTRL7 subregister settings */
#define CP_CLK_DIV_2_BYTE               ((uint8_t)(0x0U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_4_BYTE               ((uint8_t)(0x1U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_6_BYTE               ((uint8_t)(0x2U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_8_BYTE               ((uint8_t)(0x3U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_10_BYTE              ((uint8_t)(0x4U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_12_BYTE              ((uint8_t)(0x5U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_14_BYTE              ((uint8_t)(0x6U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_16_BYTE              ((uint8_t)(0x7U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_20_BYTE              ((uint8_t)(0x9U << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_24_BYTE              ((uint8_t)(0xBU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_28_BYTE              ((uint8_t)(0xDU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_32_BYTE              ((uint8_t)(0xFU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))
#define CP_CLK_DIV_64_BYTE              ((uint8_t)(0x1FU << CLK_CTRL_CP_CLK_DIV_BYTE_CP_CLK_DIV_Pos))

/* External Clock Detection Control and Configuration */
/*   Configure the external clock detection circuitry */
#define CLK_DETECT_CTRL_BASE            0x40000E20

/* CLK_DETECT_CTRL bit positions */
#define CLK_DETECT_CTRL_CLK_FORCE_ENABLE_Pos 1
#define CLK_DETECT_CTRL_CLK_DETECT_ENABLE_Pos 0

/* CLK_DETECT_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t CLK_DETECT_ENABLE_ALIAS;/* Select whether the system attempts to detect an external clock */
    __IO uint32_t CLK_FORCE_ENABLE_ALIAS;/* Select whether the external clock is forced to RC_CLK when no clock is detected */
} CLK_DETECT_CTRL_Type;

#define CLK_DETECT_CTRL                 ((CLK_DETECT_CTRL_Type *) SYS_CALC_BITBAND(CLK_DETECT_CTRL_BASE, 0))

/* CLK_DETECT_CTRL settings */
#define CLK_DETECT_DISABLE_BITBAND      0x0
#define CLK_DETECT_ENABLE_BITBAND       0x1
#define CLK_DETECT_DISABLE              ((uint32_t)(CLK_DETECT_DISABLE_BITBAND << CLK_DETECT_CTRL_CLK_DETECT_ENABLE_Pos))
#define CLK_DETECT_ENABLE               ((uint32_t)(CLK_DETECT_ENABLE_BITBAND << CLK_DETECT_CTRL_CLK_DETECT_ENABLE_Pos))

#define CLK_DETECT_FORCE_DISABLE_BITBAND 0x0
#define CLK_DETECT_FORCE_ENABLE_BITBAND 0x1
#define CLK_DETECT_FORCE_DISABLE        ((uint32_t)(CLK_DETECT_FORCE_DISABLE_BITBAND << CLK_DETECT_CTRL_CLK_FORCE_ENABLE_Pos))
#define CLK_DETECT_FORCE_ENABLE         ((uint32_t)(CLK_DETECT_FORCE_ENABLE_BITBAND << CLK_DETECT_CTRL_CLK_FORCE_ENABLE_Pos))

/* External Clock Detection Status */
/*   Status from external clock detection circuitry */
#define CLK_DETECT_STATUS_BASE          0x40000E24

/* CLK_DETECT_STATUS bit positions */
#define CLK_DETECT_STATUS_EXT_CLK_STATUS_Pos 0

/* CLK_DETECT_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t EXT_CLK_STATUS_ALIAS; /* Indicates whether an external clock has been present since this bit was last cleared */
} CLK_DETECT_STATUS_Type;

#define CLK_DETECT_STATUS               ((CLK_DETECT_STATUS_Type *) SYS_CALC_BITBAND(CLK_DETECT_STATUS_BASE, 0))

/* CLK_DETECT_STATUS settings */
#define EXT_CLK_DETECTED_BITBAND        0x0
#define EXT_CLK_NOT_DETECTED_BITBAND    0x1
#define EXT_CLK_DETECT_CLEAR_BITBAND    0x1
#define EXT_CLK_DETECTED                ((uint32_t)(EXT_CLK_DETECTED_BITBAND << CLK_DETECT_STATUS_EXT_CLK_STATUS_Pos))
#define EXT_CLK_NOT_DETECTED            ((uint32_t)(EXT_CLK_NOT_DETECTED_BITBAND << CLK_DETECT_STATUS_EXT_CLK_STATUS_Pos))
#define EXT_CLK_DETECT_CLEAR            ((uint32_t)(EXT_CLK_DETECT_CLEAR_BITBAND << CLK_DETECT_STATUS_EXT_CLK_STATUS_Pos))

/* ----------------------------------------------------------------------------
 * Flash Update Control
 * ------------------------------------------------------------------------- */
/* The flash memory can be erased or written using only the flash update logic.
 * Control of this logic is mapped into the peripheral memory space for ease of
 * use. */

typedef struct
{
    __IO uint32_t ECC_STATUS;           /* Provides the status result from recent flash ECC calculations. */
         uint32_t RESERVED0[3];
    __IO uint32_t WRITE_LOCK;           /* Lock word allowing a write to the flash memory is allowed */
} FLASH_Type;

#define FLASH_BASE                      0x40000F04
#define FLASH                           ((FLASH_Type *) FLASH_BASE)

/* Flash Error-Correction Coding Status */
/*   Provides the status result from recent flash ECC calculations. */
#define FLASH_ECC_STATUS_BASE           0x40000F04

/* FLASH_ECC_STATUS bit positions */
#define FLASH_ECC_STATUS_ERROR_DETECT_Pos 1
#define FLASH_ECC_STATUS_ERROR_CORRECT_Pos 0

/* FLASH_ECC_STATUS sub-register and bit-band aliases */
typedef struct
{
    __IO uint32_t ERROR_CORRECT_ALIAS;  /* An ECC error was detected and corrected */
    __IO uint32_t ERROR_DETECT_ALIAS;   /* An ECC error was detected, but not corrected */
} FLASH_ECC_STATUS_Type;

#define FLASH_ECC_STATUS                ((FLASH_ECC_STATUS_Type *) SYS_CALC_BITBAND(FLASH_ECC_STATUS_BASE, 0))

/* FLASH_ECC_STATUS settings */
#define ECC_ERROR_NOT_CORRECTED_BITBAND 0x0
#define ECC_ERROR_CORRECT_CLEAR_BITBAND 0x1
#define ECC_ERROR_CORRECTED_BITBAND     0x1
#define ECC_ERROR_NOT_CORRECTED         ((uint32_t)(ECC_ERROR_NOT_CORRECTED_BITBAND << FLASH_ECC_STATUS_ERROR_CORRECT_Pos))
#define ECC_ERROR_CORRECT_CLEAR         ((uint32_t)(ECC_ERROR_CORRECT_CLEAR_BITBAND << FLASH_ECC_STATUS_ERROR_CORRECT_Pos))
#define ECC_ERROR_CORRECTED             ((uint32_t)(ECC_ERROR_CORRECTED_BITBAND << FLASH_ECC_STATUS_ERROR_CORRECT_Pos))

#define ECC_ERROR_NOT_DETECTED_BITBAND  0x0
#define ECC_ERROR_DETECT_CLEAR_BITBAND  0x1
#define ECC_ERROR_DETECTED_BITBAND      0x1
#define ECC_ERROR_NOT_DETECTED          ((uint32_t)(ECC_ERROR_NOT_DETECTED_BITBAND << FLASH_ECC_STATUS_ERROR_DETECT_Pos))
#define ECC_ERROR_DETECT_CLEAR          ((uint32_t)(ECC_ERROR_DETECT_CLEAR_BITBAND << FLASH_ECC_STATUS_ERROR_DETECT_Pos))
#define ECC_ERROR_DETECTED              ((uint32_t)(ECC_ERROR_DETECTED_BITBAND << FLASH_ECC_STATUS_ERROR_DETECT_Pos))

/* Flash Write Lock */
/*   Lock word allowing a write to the flash memory is allowed */
#define FLASH_WRITE_LOCK_BASE           0x40000F14

/* FLASH_WRITE_LOCK bit positions */
#define FLASH_WRITE_LOCK_FLASH_WRITE_KEY_Pos 0
#define FLASH_WRITE_LOCK_FLASH_WRITE_KEY_Mask ((uint32_t)(0xFFFFFFFFU << FLASH_WRITE_LOCK_FLASH_WRITE_KEY_Pos))

/* FLASH_WRITE_LOCK settings */
#define FLASH_WRITE_KEY                 ((uint32_t)(0xABCD4321U << FLASH_WRITE_LOCK_FLASH_WRITE_KEY_Pos))

/* ----------------------------------------------------------------------------
 * USB Interface Configuration and Control
 * ------------------------------------------------------------------------- */
/* The universal serial bus (USB) interface implements the USB 1.1 standard in
 * full-speed mode.
 *
 * This interface provides connectivity between the system and an external PC
 * or other USB device. */

typedef struct
{
    __O  uint32_t BULK_OUT45_ADDR;          __O  uint32_t BULK_IN23_ADDR;                uint32_t RESERVED0;
    __O  uint32_t CLOCK_GATE;                    uint32_t RESERVED1[5];
    __IO uint32_t INT_STATUS;           /* Interrupt status register. The USB_IVEC bits show the status of the different interrupt sources and is updated upon interrupt request generation. The USB_BULK_IN_IRQ bits are set by the CUSB to '1' when it transmits a bulk IN x data packet and receives an ACK from the host. The USB_BULK_OUT_IRQ bits are set by the CUSB to '1' when it receives an error free bulk OUT x data packet. */
    __IO uint32_t INT_CTRL;             /* Setting the appropriate bit to '1' enables the appropriate interrupt */
         uint32_t RESERVED2;
    __IO uint32_t EP0_IN_CTRL;          /* Various control and status bit fields for the USB IN 0 endpoint */
    __IO uint32_t EP23_IN_CTRL;         /* Various control and status bit fields for the USB IN 2 and 3 endpoints */
         uint32_t RESERVED3[2];
    __IO uint32_t EP0_OUT_CTRL;         /* Various control and status bit fields for the USB OUT endpoints 0 */
         uint32_t RESERVED4;
    __IO uint32_t EP45_OUT_CTRL;        /* Various control and status bit fields for the USB OUT endpoints 4 and 5 */
         uint32_t RESERVED5;
    __IO uint32_t SYS_CTRL1;            /* Various control and status bit fields for USB */
    __I  uint32_t SYS_CTRL2;            /* Current USB frame count */
    __IO uint32_t SYS_CTRL3;            /* Various control and status bit fields for USB */
         uint32_t RESERVED6[2];
    __I  uint32_t SETUP_DATA_BUF_BASE_0;/* Contains the lower 4 bytes of the SETUP data packet from the latest CONTROL transfer. */
    __I  uint32_t SETUP_DATA_BUF_BASE_1;/* Contains the upper 4 bytes of the SETUP data packet from the latest CONTROL transfer. */
} USB_Type;

#define USB_BASE                        0x40001784
#define USB                             ((USB_Type *) USB_BASE)

/* USB Bulk 4 and 5 OUT start address register */
#define USB_BULK_OUT45_ADDR_BASE        0x40001784

/* USB_BULK_OUT45_ADDR bit positions */
#define USB_BULK_OUT45_ADDR_OUT5_ADDR_Pos 8
#define USB_BULK_OUT45_ADDR_OUT5_ADDR_Mask ((uint32_t)(0xFFU << USB_BULK_OUT45_ADDR_OUT5_ADDR_Pos))
#define USB_BULK_OUT45_ADDR_OUT4_ADDR_Pos 0
#define USB_BULK_OUT45_ADDR_OUT4_ADDR_Mask ((uint32_t)(0xFFU << USB_BULK_OUT45_ADDR_OUT4_ADDR_Pos))

/* USB_BULK_OUT45_ADDR sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t OUT4_ADDR_BYTE;       
    __IO uint8_t OUT5_ADDR_BYTE;       
         uint8_t RESERVED0[2];
} USB_BULK_OUT45_ADDR_Type;

#define USB_BULK_OUT45_ADDR             ((USB_BULK_OUT45_ADDR_Type *) USB_BULK_OUT45_ADDR_BASE)

/* USB_BULK_OUT45_ADDR settings */
#define USB_BULK_OUT4_ADDR_VALUE        ((uint32_t)(0x10U << USB_BULK_OUT45_ADDR_OUT4_ADDR_Pos))

#define USB_BULK_OUT5_ADDR_VALUE        ((uint32_t)(0x30U << USB_BULK_OUT45_ADDR_OUT5_ADDR_Pos))

/* USB_BULK_OUT45_ADDR subregister settings */
#define USB_BULK_OUT4_ADDR_VALUE_BYTE   0x00000010

#define USB_BULK_OUT5_ADDR_VALUE_BYTE   0x00000030

/* USB Bulk 2 and 3 IN start address register */
#define USB_BULK_IN23_ADDR_BASE         0x40001788

/* USB_BULK_IN23_ADDR bit positions */
#define USB_BULK_IN23_ADDR_IN3_ADDR_Pos 24
#define USB_BULK_IN23_ADDR_IN3_ADDR_Mask ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN3_ADDR_Pos))
#define USB_BULK_IN23_ADDR_IN2_ADDR_Pos 16
#define USB_BULK_IN23_ADDR_IN2_ADDR_Mask ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN2_ADDR_Pos))
#define USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_Pos 0
#define USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_Mask ((uint32_t)(0xFFU << USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_Pos))

/* USB_BULK_IN23_ADDR sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t OFFSET_ADDR_BYTE;     
         uint8_t RESERVED0[1];
    __IO uint8_t IN2_ADDR_BYTE;        
    __IO uint8_t IN3_ADDR_BYTE;        
} USB_BULK_IN23_ADDR_Type;

#define USB_BULK_IN23_ADDR              ((USB_BULK_IN23_ADDR_Type *) USB_BULK_IN23_ADDR_BASE)

/* USB_BULK_IN23_ADDR settings */
#define USB_BULK_OFFSET_ADDR_VALUE      ((uint32_t)(0x28U << USB_BULK_IN23_ADDR_IN_OFFSET_ADDR_Pos))

#define USB_BULK_IN2_ADDR_VALUE         ((uint32_t)(0x10U << USB_BULK_IN23_ADDR_IN2_ADDR_Pos))

#define USB_BULK_IN3_ADDR_VALUE         ((uint32_t)(0x30U << USB_BULK_IN23_ADDR_IN3_ADDR_Pos))

/* USB_BULK_IN23_ADDR subregister settings */
#define USB_BULK_OFFSET_ADDR_VALUE_BYTE 0x00000028

#define USB_BULK_IN2_ADDR_VALUE_BYTE    0x00000010

#define USB_BULK_IN3_ADDR_VALUE_BYTE    0x00000030

/* USB clock gate register */
#define USB_CLOCK_GATE_BASE             0x40001790

/* USB_CLOCK_GATE bit positions */
#define USB_CLOCK_GATE_CLOCK_GATE_Pos   0
#define USB_CLOCK_GATE_CLOCK_GATE_Mask  ((uint32_t)(0xFFU << USB_CLOCK_GATE_CLOCK_GATE_Pos))

/* USB_CLOCK_GATE sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t GATE_BYTE;            
         uint8_t RESERVED0[3];
} USB_CLOCK_GATE_Type;

#define USB_CLOCK_GATE                  ((USB_CLOCK_GATE_Type *) USB_CLOCK_GATE_BASE)

/* USB Interrupt Status register */
/*   Interrupt status register. The USB_IVEC bits show the status of the
 *   different interrupt sources and is updated upon interrupt request
 *   generation. The USB_BULK_IN_IRQ bits are set by the CUSB to '1' when it
 *   transmits a bulk IN x data packet and receives an ACK from the host. The
 *   USB_BULK_OUT_IRQ bits are set by the CUSB to '1' when it receives an error
 *   free bulk OUT x data packet. */
#define USB_INT_STATUS_BASE             0x400017A8

/* USB_INT_STATUS bit positions */
#define USB_INT_STATUS_RST_IRQ_Pos      28
#define USB_INT_STATUS_SUS_IRQ_Pos      27
#define USB_INT_STATUS_SETUPTKN_IRQ_Pos 26
#define USB_INT_STATUS_SOF_IRQ_Pos      25
#define USB_INT_STATUS_DAT_VALID_IRQ_Pos 24
#define USB_INT_STATUS_BULK_OUT_5_IRQ_Pos 21
#define USB_INT_STATUS_BULK_OUT_4_IRQ_Pos 20
#define USB_INT_STATUS_BULK_OUT_0_IRQ_Pos 16
#define USB_INT_STATUS_BULK_IN_3_IRQ_Pos 11
#define USB_INT_STATUS_BULK_IN_2_IRQ_Pos 10
#define USB_INT_STATUS_BULK_IN_0_IRQ_Pos 8

/* USB_INT_STATUS sub-register and bit-band aliases */
typedef struct
{
         uint8_t RESERVED0[1];
    __IO uint8_t BULK_IN_IRQ_BYTE;     
    __IO uint8_t BULK_OUT_IRQ_BYTE;    
    __IO uint8_t IRQ_BYTE;             
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017A8, 0) - (0x400017A8 + 4)];
         uint32_t RESERVED1[8];
    __IO uint32_t BULK_IN_0_IRQ_ALIAS;  /* Indicate that an EP0 IN interrupt request occurred */
         uint32_t RESERVED2;
    __IO uint32_t BULK_IN_2_IRQ_ALIAS;  /* Indicate that an EP2 IN interrupt request occurred */
    __IO uint32_t BULK_IN_3_IRQ_ALIAS;  /* Indicate that an EP3 IN interrupt request occurred */
         uint32_t RESERVED3[4];
    __IO uint32_t BULK_OUT_0_IRQ_ALIAS; /* Indicate that an EP0 OUT interrupt request occurred */
         uint32_t RESERVED4[3];
    __IO uint32_t BULK_OUT_4_IRQ_ALIAS; /* Indicate that an EP4 OUT interrupt request occurred */
    __IO uint32_t BULK_OUT_5_IRQ_ALIAS; /* Indicate that an EP5 OUT interrupt request occurred */
         uint32_t RESERVED5[2];
    __IO uint32_t DAT_VALID_IRQ_ALIAS;  /* SETUP data valid interrupt request. The USB controller sets this bit to 1 when it receives an error-free SETUP data packet. */
    __IO uint32_t SOF_IRQ_ALIAS;        /* Start-of-Frame interrupt request. The USB controller sets this bit to 1 when it receives a SOF packet. */
    __IO uint32_t SETUPTKN_IRQ_ALIAS;   /* SETUP token interrupt request. The USB controller sets this bit to 1 when it receives a SETUP token. */
    __IO uint32_t SUS_IRQ_ALIAS;        /* USB suspend interrupt request. The USB controller sets this bit to 1 when it detects USB suspend signalling. */
    __IO uint32_t RST_IRQ_ALIAS;        /* USB reset interrupt request. The USB controller sets this bit to 1 when it detects a USB bus reset. */
} USB_INT_STATUS_Type;

#define USB_INT_STATUS                  ((USB_INT_STATUS_Type *) USB_INT_STATUS_BASE)

/* USB_INT_STATUS settings */
#define USB_BULK_IN_0_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_IN_0_IRQ_CLEAR         ((uint32_t)(USB_BULK_IN_0_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_IN_0_IRQ_Pos))

#define USB_BULK_IN_2_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_IN_2_IRQ_CLEAR         ((uint32_t)(USB_BULK_IN_2_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_IN_2_IRQ_Pos))

#define USB_BULK_IN_3_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_IN_3_IRQ_CLEAR         ((uint32_t)(USB_BULK_IN_3_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_IN_3_IRQ_Pos))

#define USB_BULK_OUT_0_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_OUT_0_IRQ_CLEAR        ((uint32_t)(USB_BULK_OUT_0_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_OUT_0_IRQ_Pos))

#define USB_BULK_OUT_4_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_OUT_4_IRQ_CLEAR        ((uint32_t)(USB_BULK_OUT_4_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_OUT_4_IRQ_Pos))

#define USB_BULK_OUT_5_IRQ_CLEAR_BITBAND 0x1
#define USB_BULK_OUT_5_IRQ_CLEAR        ((uint32_t)(USB_BULK_OUT_5_IRQ_CLEAR_BITBAND << USB_INT_STATUS_BULK_OUT_5_IRQ_Pos))

#define USB_DAT_VALID_IRQ_CLEAR_BITBAND 0x1
#define USB_DAT_VALID_IRQ_CLEAR         ((uint32_t)(USB_DAT_VALID_IRQ_CLEAR_BITBAND << USB_INT_STATUS_DAT_VALID_IRQ_Pos))

#define USB_SOF_IRQ_CLEAR_BITBAND       0x1
#define USB_SOF_IRQ_CLEAR               ((uint32_t)(USB_SOF_IRQ_CLEAR_BITBAND << USB_INT_STATUS_SOF_IRQ_Pos))

#define USB_SETUPTKN_IRQ_CLEAR_BITBAND  0x1
#define USB_SETUPTKN_IRQ_CLEAR          ((uint32_t)(USB_SETUPTKN_IRQ_CLEAR_BITBAND << USB_INT_STATUS_SETUPTKN_IRQ_Pos))

#define USB_SUS_IRQ_CLEAR_BITBAND       0x1
#define USB_SUS_IRQ_CLEAR               ((uint32_t)(USB_SUS_IRQ_CLEAR_BITBAND << USB_INT_STATUS_SUS_IRQ_Pos))

#define USB_RST_IRQ_CLEAR_BITBAND       0x1
#define USB_RST_IRQ_CLEAR               ((uint32_t)(USB_RST_IRQ_CLEAR_BITBAND << USB_INT_STATUS_RST_IRQ_Pos))

/* USB_INT_STATUS sub-register bit positions */
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_0_IRQ_Pos 0
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_2_IRQ_Pos 2
#define USB_BULK_IN_IRQ_BYTE_BULK_IN_3_IRQ_Pos 3
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_0_IRQ_Pos 0
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_4_IRQ_Pos 4
#define USB_BULK_OUT_IRQ_BYTE_BULK_OUT_5_IRQ_Pos 5
#define USB_IRQ_BYTE_DAT_VALID_IRQ_Pos  0
#define USB_IRQ_BYTE_SOF_IRQ_Pos        1
#define USB_IRQ_BYTE_SETUPTKN_IRQ_Pos   2
#define USB_IRQ_BYTE_SUS_IRQ_Pos        3
#define USB_IRQ_BYTE_RST_IRQ_Pos        4

/* USB_INT_STATUS subregister settings */
#define USB_BULK_IN_0_IRQ_CLEAR_BYTE    ((uint8_t)(USB_BULK_IN_0_IRQ_CLEAR_BITBAND << USB_BULK_IN_IRQ_BYTE_BULK_IN_0_IRQ_Pos))

#define USB_BULK_IN_2_IRQ_CLEAR_BYTE    ((uint8_t)(USB_BULK_IN_2_IRQ_CLEAR_BITBAND << USB_BULK_IN_IRQ_BYTE_BULK_IN_2_IRQ_Pos))

#define USB_BULK_IN_3_IRQ_CLEAR_BYTE    ((uint8_t)(USB_BULK_IN_3_IRQ_CLEAR_BITBAND << USB_BULK_IN_IRQ_BYTE_BULK_IN_3_IRQ_Pos))

#define USB_BULK_OUT_0_IRQ_CLEAR_BYTE   ((uint8_t)(USB_BULK_OUT_0_IRQ_CLEAR_BITBAND << USB_BULK_OUT_IRQ_BYTE_BULK_OUT_0_IRQ_Pos))

#define USB_BULK_OUT_4_IRQ_CLEAR_BYTE   ((uint8_t)(USB_BULK_OUT_4_IRQ_CLEAR_BITBAND << USB_BULK_OUT_IRQ_BYTE_BULK_OUT_4_IRQ_Pos))

#define USB_BULK_OUT_5_IRQ_CLEAR_BYTE   ((uint8_t)(USB_BULK_OUT_5_IRQ_CLEAR_BITBAND << USB_BULK_OUT_IRQ_BYTE_BULK_OUT_5_IRQ_Pos))

#define USB_DAT_VALID_IRQ_CLEAR_BYTE    ((uint8_t)(USB_DAT_VALID_IRQ_CLEAR_BITBAND << USB_IRQ_BYTE_DAT_VALID_IRQ_Pos))

#define USB_SOF_IRQ_CLEAR_BYTE          ((uint8_t)(USB_SOF_IRQ_CLEAR_BITBAND << USB_IRQ_BYTE_SOF_IRQ_Pos))

#define USB_SETUPTKN_IRQ_CLEAR_BYTE     ((uint8_t)(USB_SETUPTKN_IRQ_CLEAR_BITBAND << USB_IRQ_BYTE_SETUPTKN_IRQ_Pos))

#define USB_SUS_IRQ_CLEAR_BYTE          ((uint8_t)(USB_SUS_IRQ_CLEAR_BITBAND << USB_IRQ_BYTE_SUS_IRQ_Pos))

#define USB_RST_IRQ_CLEAR_BYTE          ((uint8_t)(USB_RST_IRQ_CLEAR_BITBAND << USB_IRQ_BYTE_RST_IRQ_Pos))

/* USB Interrupt Control register */
/*   Setting the appropriate bit to '1' enables the appropriate interrupt */
#define USB_INT_CTRL_BASE               0x400017AC

/* USB_INT_CTRL bit positions */
#define USB_INT_CTRL_BAV_Pos            24
#define USB_INT_CTRL_RST_IEN_Pos        20
#define USB_INT_CTRL_SUS_IEN_Pos        19
#define USB_INT_CTRL_SETUPTKN_IEN_Pos   18
#define USB_INT_CTRL_SOF_IEN_Pos        17
#define USB_INT_CTRL_DAT_VALID_IEN_Pos  16
#define USB_INT_CTRL_BULK_OUT_5_IEN_Pos 13
#define USB_INT_CTRL_BULK_OUT_4_IEN_Pos 12
#define USB_INT_CTRL_BULK_OUT_0_IEN_Pos 8
#define USB_INT_CTRL_BULK_IN_3_IEN_Pos  3
#define USB_INT_CTRL_BULK_IN_2_IEN_Pos  2
#define USB_INT_CTRL_BULK_IN_0_IEN_Pos  0

/* USB_INT_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t BULK_IN_IEN_BYTE;     
    __IO uint8_t BULK_OUT_IEN_BYTE;    
    __IO uint8_t IEN_BYTE;             
    __IO uint8_t BAV_BYTE;             
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017AC, 0) - (0x400017AC + 4)];
    __IO uint32_t BULK_IN_0_IEN_ALIAS;  /* EP0 IN interrupt enable */
         uint32_t RESERVED0;
    __IO uint32_t BULK_IN_2_IEN_ALIAS;  /* EP2 IN interrupt enable */
    __IO uint32_t BULK_IN_3_IEN_ALIAS;  /* EP3 IN interrupt enable */
         uint32_t RESERVED1[4];
    __IO uint32_t BULK_OUT_0_IEN_ALIAS; /* EP0 OUT interrupt enable */
         uint32_t RESERVED2[3];
    __IO uint32_t BULK_OUT_4_IEN_ALIAS; /* EP4 OUT interrupt enable */
    __IO uint32_t BULK_OUT_5_IEN_ALIAS; /* EP5 OUT interrupt enable */
         uint32_t RESERVED3[2];
    __IO uint32_t DAT_VALID_IEN_ALIAS;  /* SETUP data valid interrupt enable */
    __IO uint32_t SOF_IEN_ALIAS;        /* Start-Of-Frame interrupt enable */
    __IO uint32_t SETUPTKN_IEN_ALIAS;   /* SETUP token interrupt enable */
    __IO uint32_t SUS_IEN_ALIAS;        /* USB suspend interrupt enable */
    __IO uint32_t RST_IEN_ALIAS;        /* USB reset interrupt enable */
         uint32_t RESERVED4[3];
    __IO uint32_t BAV_ALIAS;            /* Breakpoint and autovector register. This field must be set to 1. */
} USB_INT_CTRL_Type;

#define USB_INT_CTRL                    ((USB_INT_CTRL_Type *) USB_INT_CTRL_BASE)

/* USB_INT_CTRL settings */
#define USB_BULK_IN_0_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_IN_0_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_IN_0_IEN_ENABLE        ((uint32_t)(USB_BULK_IN_0_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_IN_0_IEN_Pos))
#define USB_BULK_IN_0_IEN_DISABLE       ((uint32_t)(USB_BULK_IN_0_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_IN_0_IEN_Pos))

#define USB_BULK_IN_2_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_IN_2_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_IN_2_IEN_ENABLE        ((uint32_t)(USB_BULK_IN_2_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_IN_2_IEN_Pos))
#define USB_BULK_IN_2_IEN_DISABLE       ((uint32_t)(USB_BULK_IN_2_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_IN_2_IEN_Pos))

#define USB_BULK_IN_3_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_IN_3_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_IN_3_IEN_ENABLE        ((uint32_t)(USB_BULK_IN_3_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_IN_3_IEN_Pos))
#define USB_BULK_IN_3_IEN_DISABLE       ((uint32_t)(USB_BULK_IN_3_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_IN_3_IEN_Pos))

#define USB_BULK_OUT_0_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_OUT_0_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_OUT_0_IEN_ENABLE       ((uint32_t)(USB_BULK_OUT_0_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_OUT_0_IEN_Pos))
#define USB_BULK_OUT_0_IEN_DISABLE      ((uint32_t)(USB_BULK_OUT_0_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_OUT_0_IEN_Pos))

#define USB_BULK_OUT_4_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_OUT_4_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_OUT_4_IEN_ENABLE       ((uint32_t)(USB_BULK_OUT_4_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_OUT_4_IEN_Pos))
#define USB_BULK_OUT_4_IEN_DISABLE      ((uint32_t)(USB_BULK_OUT_4_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_OUT_4_IEN_Pos))

#define USB_BULK_OUT_5_IEN_ENABLE_BITBAND 0x1
#define USB_BULK_OUT_5_IEN_DISABLE_BITBAND 0x0
#define USB_BULK_OUT_5_IEN_ENABLE       ((uint32_t)(USB_BULK_OUT_5_IEN_ENABLE_BITBAND << USB_INT_CTRL_BULK_OUT_5_IEN_Pos))
#define USB_BULK_OUT_5_IEN_DISABLE      ((uint32_t)(USB_BULK_OUT_5_IEN_DISABLE_BITBAND << USB_INT_CTRL_BULK_OUT_5_IEN_Pos))

#define USB_DAT_VALID_IEN_ENABLE_BITBAND 0x1
#define USB_DAT_VALID_IEN_DISABLE_BITBAND 0x0
#define USB_DAT_VALID_IEN_ENABLE        ((uint32_t)(USB_DAT_VALID_IEN_ENABLE_BITBAND << USB_INT_CTRL_DAT_VALID_IEN_Pos))
#define USB_DAT_VALID_IEN_DISABLE       ((uint32_t)(USB_DAT_VALID_IEN_DISABLE_BITBAND << USB_INT_CTRL_DAT_VALID_IEN_Pos))

#define USB_SOF_IEN_ENABLE_BITBAND      0x1
#define USB_SOF_IEN_DISABLE_BITBAND     0x0
#define USB_SOF_IEN_ENABLE              ((uint32_t)(USB_SOF_IEN_ENABLE_BITBAND << USB_INT_CTRL_SOF_IEN_Pos))
#define USB_SOF_IEN_DISABLE             ((uint32_t)(USB_SOF_IEN_DISABLE_BITBAND << USB_INT_CTRL_SOF_IEN_Pos))

#define USB_SETUPTKN_IEN_ENABLE_BITBAND 0x1
#define USB_SETUPTKN_IEN_DISABLE_BITBAND 0x0
#define USB_SETUPTKN_IEN_ENABLE         ((uint32_t)(USB_SETUPTKN_IEN_ENABLE_BITBAND << USB_INT_CTRL_SETUPTKN_IEN_Pos))
#define USB_SETUPTKN_IEN_DISABLE        ((uint32_t)(USB_SETUPTKN_IEN_DISABLE_BITBAND << USB_INT_CTRL_SETUPTKN_IEN_Pos))

#define USB_SUS_IEN_ENABLE_BITBAND      0x1
#define USB_SUS_IEN_DISABLE_BITBAND     0x0
#define USB_SUS_IEN_ENABLE              ((uint32_t)(USB_SUS_IEN_ENABLE_BITBAND << USB_INT_CTRL_SUS_IEN_Pos))
#define USB_SUS_IEN_DISABLE             ((uint32_t)(USB_SUS_IEN_DISABLE_BITBAND << USB_INT_CTRL_SUS_IEN_Pos))

#define USB_RST_IEN_ENABLE_BITBAND      0x1
#define USB_RST_IEN_DISABLE_BITBAND     0x0
#define USB_RST_IEN_ENABLE              ((uint32_t)(USB_RST_IEN_ENABLE_BITBAND << USB_INT_CTRL_RST_IEN_Pos))
#define USB_RST_IEN_DISABLE             ((uint32_t)(USB_RST_IEN_DISABLE_BITBAND << USB_INT_CTRL_RST_IEN_Pos))

#define USB_BAV_ENABLE_BITBAND          0x1
#define USB_BAV_DISABLE_BITBAND         0x0
#define USB_BAV_ENABLE                  ((uint32_t)(USB_BAV_ENABLE_BITBAND << USB_INT_CTRL_BAV_Pos))
#define USB_BAV_DISABLE                 ((uint32_t)(USB_BAV_DISABLE_BITBAND << USB_INT_CTRL_BAV_Pos))

/* USB_INT_CTRL sub-register bit positions */
#define USB_BULK_IN_IEN_BYTE_BULK_IN_0_IEN_Pos 0
#define USB_BULK_IN_IEN_BYTE_BULK_IN_2_IEN_Pos 2
#define USB_BULK_IN_IEN_BYTE_BULK_IN_3_IEN_Pos 3
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_0_IEN_Pos 0
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_4_IEN_Pos 4
#define USB_BULK_OUT_IEN_BYTE_BULK_OUT_5_IEN_Pos 5
#define USB_IEN_BYTE_DAT_VALID_IEN_Pos  0
#define USB_IEN_BYTE_SOF_IEN_Pos        1
#define USB_IEN_BYTE_SETUPTKN_IEN_Pos   2
#define USB_IEN_BYTE_SUS_IEN_Pos        3
#define USB_IEN_BYTE_RST_IEN_Pos        4
#define USB_BAV_BYTE_BAV_Pos            0

/* USB_INT_CTRL subregister settings */
#define USB_BULK_IN_0_IEN_ENABLE_BYTE   ((uint8_t)(USB_BULK_IN_0_IEN_ENABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_0_IEN_Pos))
#define USB_BULK_IN_0_IEN_DISABLE_BYTE  ((uint8_t)(USB_BULK_IN_0_IEN_DISABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_0_IEN_Pos))

#define USB_BULK_IN_2_IEN_ENABLE_BYTE   ((uint8_t)(USB_BULK_IN_2_IEN_ENABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_2_IEN_Pos))
#define USB_BULK_IN_2_IEN_DISABLE_BYTE  ((uint8_t)(USB_BULK_IN_2_IEN_DISABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_2_IEN_Pos))

#define USB_BULK_IN_3_IEN_ENABLE_BYTE   ((uint8_t)(USB_BULK_IN_3_IEN_ENABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_3_IEN_Pos))
#define USB_BULK_IN_3_IEN_DISABLE_BYTE  ((uint8_t)(USB_BULK_IN_3_IEN_DISABLE_BITBAND << USB_BULK_IN_IEN_BYTE_BULK_IN_3_IEN_Pos))

#define USB_BULK_OUT_0_IEN_ENABLE_BYTE  ((uint8_t)(USB_BULK_OUT_0_IEN_ENABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_0_IEN_Pos))
#define USB_BULK_OUT_0_IEN_DISABLE_BYTE ((uint8_t)(USB_BULK_OUT_0_IEN_DISABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_0_IEN_Pos))

#define USB_BULK_OUT_4_IEN_ENABLE_BYTE  ((uint8_t)(USB_BULK_OUT_4_IEN_ENABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_4_IEN_Pos))
#define USB_BULK_OUT_4_IEN_DISABLE_BYTE ((uint8_t)(USB_BULK_OUT_4_IEN_DISABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_4_IEN_Pos))

#define USB_BULK_OUT_5_IEN_ENABLE_BYTE  ((uint8_t)(USB_BULK_OUT_5_IEN_ENABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_5_IEN_Pos))
#define USB_BULK_OUT_5_IEN_DISABLE_BYTE ((uint8_t)(USB_BULK_OUT_5_IEN_DISABLE_BITBAND << USB_BULK_OUT_IEN_BYTE_BULK_OUT_5_IEN_Pos))

#define USB_DAT_VALID_IEN_ENABLE_BYTE   ((uint8_t)(USB_DAT_VALID_IEN_ENABLE_BITBAND << USB_IEN_BYTE_DAT_VALID_IEN_Pos))
#define USB_DAT_VALID_IEN_DISABLE_BYTE  ((uint8_t)(USB_DAT_VALID_IEN_DISABLE_BITBAND << USB_IEN_BYTE_DAT_VALID_IEN_Pos))

#define USB_SOF_IEN_ENABLE_BYTE         ((uint8_t)(USB_SOF_IEN_ENABLE_BITBAND << USB_IEN_BYTE_SOF_IEN_Pos))
#define USB_SOF_IEN_DISABLE_BYTE        ((uint8_t)(USB_SOF_IEN_DISABLE_BITBAND << USB_IEN_BYTE_SOF_IEN_Pos))

#define USB_SETUPTKN_IEN_ENABLE_BYTE    ((uint8_t)(USB_SETUPTKN_IEN_ENABLE_BITBAND << USB_IEN_BYTE_SETUPTKN_IEN_Pos))
#define USB_SETUPTKN_IEN_DISABLE_BYTE   ((uint8_t)(USB_SETUPTKN_IEN_DISABLE_BITBAND << USB_IEN_BYTE_SETUPTKN_IEN_Pos))

#define USB_SUS_IEN_ENABLE_BYTE         ((uint8_t)(USB_SUS_IEN_ENABLE_BITBAND << USB_IEN_BYTE_SUS_IEN_Pos))
#define USB_SUS_IEN_DISABLE_BYTE        ((uint8_t)(USB_SUS_IEN_DISABLE_BITBAND << USB_IEN_BYTE_SUS_IEN_Pos))

#define USB_RST_IEN_ENABLE_BYTE         ((uint8_t)(USB_RST_IEN_ENABLE_BITBAND << USB_IEN_BYTE_RST_IEN_Pos))
#define USB_RST_IEN_DISABLE_BYTE        ((uint8_t)(USB_RST_IEN_DISABLE_BITBAND << USB_IEN_BYTE_RST_IEN_Pos))

#define USB_BAV_ENABLE_BYTE             ((uint8_t)(USB_BAV_ENABLE_BITBAND << USB_BAV_BYTE_BAV_Pos))
#define USB_BAV_DISABLE_BYTE            ((uint8_t)(USB_BAV_DISABLE_BITBAND << USB_BAV_BYTE_BAV_Pos))

/* USB IN Endpoint 0 Control and Status register */
/*   Various control and status bit fields for the USB IN 0 endpoint */
#define USB_EP0_IN_CTRL_BASE            0x400017B4

/* USB_EP0_IN_CTRL bit positions */
#define USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_Pos 8
#define USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP0_IN_CTRL_EP0_IN_BYTE_COUNT_Pos))
#define USB_EP0_IN_CTRL_EP0_SETUP_BUF_STATUS_Pos 5
#define USB_EP0_IN_CTRL_EP0_DATA_STALL_Pos 4
#define USB_EP0_IN_CTRL_EP0_OUT_BUSY_Pos 3
#define USB_EP0_IN_CTRL_EP0_IN_BUSY_Pos 2
#define USB_EP0_IN_CTRL_EP0_HSNAK_Pos   1
#define USB_EP0_IN_CTRL_EP0_CTRL_STALL_Pos 0

/* USB_EP0_IN_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t CTRL_BYTE;            
    __IO uint8_t BYTE_COUNT_BYTE;      
         uint8_t RESERVED0[2];
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017B4, 0) - (0x400017B4 + 4)];
    __IO uint32_t EP0_CTRL_STALL_ALIAS; /* EP0 stall during the data or handshake phases of the CONTROL transfer. If this bit is set to 1, the USB controller sends a STALL handshake for any IN or OUT token. Automatically cleared when a SETUP token arrives. */
    __IO uint32_t EP0_HSNAK_ALIAS;      /* EP0 handshake NAK. If this bit is set to 1, the USB controller responds with a NAK handshake for every packet in the status stage. Automatically set when a SETUP token arrives. */
    __I  uint32_t EP0_IN_BUSY_ALIAS;    /* EP0 IN endpoint busy bit. If this bit is 1, the USB controller takes control of the EP0 IN endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP0 IN endpoint buffer. Automatically cleared when a SETUP token arrives. The ARM Cortex-M3 processor sets this bit by reloading USB_EP0_IN_BYTE_COUNT. */
    __I  uint32_t EP0_OUT_BUSY_ALIAS;   /* EP0 OUT endpoint busy bit. If this bit is 1, the USB controller takes control of the EP0 OUT endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP0 OUT endpoint buffer. Automatically cleared when a SETUP token arrives. The ARM Cortex-M3 processor sets this bit by writing a dummy value to USB_EP0_OUT_BYTE_COUNT. */
    __IO uint32_t EP0_DATA_STALL_ALIAS; /* EP0 stall during the data stage. If this bit is 1, the USB controller sends a STALL handshake for any IN or OUT token. Automatically cleared when a SETUP token arrives. The ARM Cortex-M3 processor should set this bit after the last successful transaction in the data stage. */
    __IO uint32_t EP0_SETUP_BUF_STATUS_ALIAS;/* Indicates that the SETUP buffer contents were changed. If this bit is 1, the SETUP buffer was changed. This bit is automatically cleared when USB controller receives a SETUP data packet. */
} USB_EP0_IN_CTRL_Type;

#define USB_EP0_IN_CTRL                 ((USB_EP0_IN_CTRL_Type *) USB_EP0_IN_CTRL_BASE)

/* USB_EP0_IN_CTRL sub-register bit positions */
#define USB_EP0_CTRL_BYTE_EP0_CTRL_STALL_Pos 0
#define USB_EP0_CTRL_BYTE_EP0_HSNAK_Pos 1
#define USB_EP0_CTRL_BYTE_EP0_IN_BUSY_Pos 2
#define USB_EP0_CTRL_BYTE_EP0_OUT_BUSY_Pos 3
#define USB_EP0_CTRL_BYTE_EP0_DATA_STALL_Pos 4
#define USB_EP0_CTRL_BYTE_EP0_SETUP_BUF_STATUS_Pos 5
#define USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_Pos 0
#define USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP0_IN_BYTE_COUNT_BYTE_EP0_IN_BYTE_COUNT_Pos))

/* USB IN Endpoint 2 and 3 Control and Status register */
/*   Various control and status bit fields for the USB IN 2 and 3 endpoints */
#define USB_EP23_IN_CTRL_BASE           0x400017B8

/* USB_EP23_IN_CTRL bit positions */
#define USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_Pos 24
#define USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP23_IN_CTRL_EP3_IN_BYTE_COUNT_Pos))
#define USB_EP23_IN_CTRL_EP3_IN_BUSY_Pos 17
#define USB_EP23_IN_CTRL_EP3_IN_STALL_Pos 16
#define USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_Pos 8
#define USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP23_IN_CTRL_EP2_IN_BYTE_COUNT_Pos))
#define USB_EP23_IN_CTRL_EP2_IN_BUSY_Pos 1
#define USB_EP23_IN_CTRL_EP2_IN_STALL_Pos 0

/* USB_EP23_IN_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t EP2_IN_CTRL_BYTE;     
    __IO uint8_t EP2_IN_BYTE_COUNT_BYTE;
    __IO uint8_t EP3_IN_CTRL_BYTE;     
    __IO uint8_t EP3_IN_BYTE_COUNT_BYTE;
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017B8, 0) - (0x400017B8 + 4)];
    __IO uint32_t EP2_IN_STALL_ALIAS;   /* EP2 IN endpoint stall bit. If this bit is 1, the USB controller returns a STALL handshake for all requests to EP2. */
    __IO uint32_t EP2_IN_BUSY_ALIAS;    /* EP2 IN endpoint busy bit. If this bit is 1, the USB controller takes control of the EP2 IN endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP2 IN endpoint buffer. Automatically cleared when a SETUP token arrives. */
         uint32_t RESERVED0[14];
    __IO uint32_t EP3_IN_STALL_ALIAS;   /* EP3 IN endpoint stall bit. If this bit is 1, the USB controller returns a STALL handshake for all requests to EP3. */
    __IO uint32_t EP3_IN_BUSY_ALIAS;    /* EP3 IN endpoint busy bit. If this bit is 1, the USB controller takes control of the EP3 IN endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP3 IN endpoint buffer. Automatically cleared when a SETUP token arrives. */
} USB_EP23_IN_CTRL_Type;

#define USB_EP23_IN_CTRL                ((USB_EP23_IN_CTRL_Type *) USB_EP23_IN_CTRL_BASE)

/* USB_EP23_IN_CTRL sub-register bit positions */
#define USB_EP2_IN_CTRL_BYTE_EP2_IN_STALL_Pos 0
#define USB_EP2_IN_CTRL_BYTE_EP2_IN_BUSY_Pos 1
#define USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_Pos 0
#define USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP2_IN_BYTE_COUNT_BYTE_EP2_IN_BYTE_COUNT_Pos))
#define USB_EP3_IN_CTRL_BYTE_EP3_IN_STALL_Pos 0
#define USB_EP3_IN_CTRL_BYTE_EP3_IN_BUSY_Pos 1
#define USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_Pos 0
#define USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP3_IN_BYTE_COUNT_BYTE_EP3_IN_BYTE_COUNT_Pos))

/* USB OUT Endpoint 0 Control and Status register */
/*   Various control and status bit fields for the USB OUT endpoints 0 */
#define USB_EP0_OUT_CTRL_BASE           0x400017C4

/* USB_EP0_OUT_CTRL bit positions */
#define USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_Pos 8
#define USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP0_OUT_CTRL_EP0_OUT_BYTE_COUNT_Pos))

/* USB_EP0_OUT_CTRL sub-register and bit-band aliases */
typedef struct
{
         uint8_t RESERVED0[1];
    __IO uint8_t BYTE_COUNT_BYTE;      
         uint8_t RESERVED1[2];
} USB_EP0_OUT_CTRL_Type;

#define USB_EP0_OUT_CTRL                ((USB_EP0_OUT_CTRL_Type *) USB_EP0_OUT_CTRL_BASE)

/* USB_EP0_OUT_CTRL sub-register bit positions */
#define USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_Pos 0
#define USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP0_OUT_BYTE_COUNT_BYTE_EP0_OUT_BYTE_COUNT_Pos))

/* USB OUT Endpoint 4 and 5 Control and Status register */
/*   Various control and status bit fields for the USB OUT endpoints 4 and 5 */
#define USB_EP45_OUT_CTRL_BASE          0x400017CC

/* USB_EP45_OUT_CTRL bit positions */
#define USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_Pos 24
#define USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP45_OUT_CTRL_EP5_OUT_BYTE_COUNT_Pos))
#define USB_EP45_OUT_CTRL_EP5_OUT_BUSY_Pos 17
#define USB_EP45_OUT_CTRL_EP5_OUT_STALL_Pos 16
#define USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_Pos 8
#define USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP45_OUT_CTRL_EP4_OUT_BYTE_COUNT_Pos))
#define USB_EP45_OUT_CTRL_EP4_OUT_BUSY_Pos 1
#define USB_EP45_OUT_CTRL_EP4_OUT_STALL_Pos 0

/* USB_EP45_OUT_CTRL sub-register and bit-band aliases */
typedef struct
{
    __IO uint8_t EP4_OUT_CTRL_BYTE;    
    __IO uint8_t EP4_OUT_BYTE_COUNT_BYTE;
    __IO uint8_t EP5_OUT_CTRL_BYTE;    
    __IO uint8_t EP5_OUT_BYTE_COUNT_BYTE;
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017CC, 0) - (0x400017CC + 4)];
    __IO uint32_t EP4_OUT_STALL_ALIAS;  /* EP4 OUT endpoint stall bit. If this bit is 1, the USB controller returns a STALL handshake for all requests to EP4. */
    __IO uint32_t EP4_OUT_BUSY_ALIAS;   /* EP4 OUT endpoint busy bit. If this bit is 1, the USB controller takes control of the EP4 OUT endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP4 OUT endpoint buffer. Automatically cleared when a SETUP token arrives. */
         uint32_t RESERVED0[14];
    __IO uint32_t EP5_OUT_STALL_ALIAS;  /* EP5 OUT endpoint stall bit. If this bit is 1, the USB controller returns a STALL handshake for all requests to EP5. */
    __IO uint32_t EP5_OUT_BUSY_ALIAS;   /* EP5 OUT endpoint busy bit. If this bit is 1, the USB controller takes control of the EP5 OUT endpoint buffer. If this bit is 0, the ARM Cortex-M3 processor takes control of the EP5 OUT endpoint buffer. Automatically cleared when a SETUP token arrives. */
} USB_EP45_OUT_CTRL_Type;

#define USB_EP45_OUT_CTRL               ((USB_EP45_OUT_CTRL_Type *) USB_EP45_OUT_CTRL_BASE)

/* USB_EP45_OUT_CTRL sub-register bit positions */
#define USB_EP4_OUT_CTRL_BYTE_EP4_OUT_STALL_Pos 0
#define USB_EP4_OUT_CTRL_BYTE_EP4_OUT_BUSY_Pos 1
#define USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_Pos 0
#define USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP4_OUT_BYTE_COUNT_BYTE_EP4_OUT_BYTE_COUNT_Pos))
#define USB_EP5_OUT_CTRL_BYTE_EP5_OUT_STALL_Pos 0
#define USB_EP5_OUT_CTRL_BYTE_EP5_OUT_BUSY_Pos 1
#define USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_Pos 0
#define USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_Mask ((uint32_t)(0x7FU << USB_EP5_OUT_BYTE_COUNT_BYTE_EP5_OUT_BYTE_COUNT_Pos))

/* USB Control and Status register 1 */
/*   Various control and status bit fields for USB */
#define USB_SYS_CTRL1_BASE              0x400017D4

/* USB_SYS_CTRL1 bit positions */
#define USB_SYS_CTRL1_DATA_TOGGLE_STATUS_Pos 31
#define USB_SYS_CTRL1_DATA_TOGGLE_DATA1_SET_Pos 30
#define USB_SYS_CTRL1_DATA_TOGGLE_RESET_Pos 29
#define USB_SYS_CTRL1_DATA_TOGGLE_INOUT_SELECT_Pos 28
#define USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_Pos 24
#define USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_Mask ((uint32_t)(0x7U << USB_SYS_CTRL1_DATA_TOGGLE_EP_SELECT_Pos))
#define USB_SYS_CTRL1_WAKEUP_SOURCE_Pos 23
#define USB_SYS_CTRL1_SOF_GEN_Pos       21
#define USB_SYS_CTRL1_DISCONNECT_Pos    19
#define USB_SYS_CTRL1_FORCE_J_Pos       17
#define USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_Pos 16

/* USB_SYS_CTRL1 sub-register and bit-band aliases */
typedef struct
{
         uint8_t RESERVED0[2];
    __IO uint8_t CTRL_STATUS_BYTE;     
    __IO uint8_t DATA_TOGGLE_CTRL_BYTE;
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017D4, 0) - (0x400017D4 + 4)];
         uint32_t RESERVED1[16];
    __IO uint32_t SIGNAL_REMOTE_RESUME_ALIAS;/* Signal remote device resume. If the ARM Cortex-M3 processor sets this bit to 1, the USB controller sets the K state on the USB lines. */
    __IO uint32_t FORCE_J_ALIAS;        /* Only use this in the suspend state. The ARM Cortex-M3 processor uses it to drive the J state on the USB lines. */
         uint32_t RESERVED2;
    __IO uint32_t DISCONNECT_ALIAS;     /* Signals a disconnect. If this bit is 1, the external USB transceiver disconnects a microcontroller by disconnecting the D+ or D- pull-up resistor. */
         uint32_t RESERVED3;
    __IO uint32_t SOF_GEN_ALIAS;        /* If this bit is 1, an internal Start-of-Frame timer is used to generate the Start-of-Frame interrupt. */
         uint32_t RESERVED4;
    __IO uint32_t WAKEUP_SOURCE_ALIAS;  /* Wakeup source. If this bit is 1, the wakeup was initiated by the microcontroller by setting the REMOTE_WAKEUP bit in the USB_CTRL register. If this bit is 0, the wakeup was initiated by the USB host. */
         uint32_t RESERVED5[4];
    __IO uint32_t DATA_TOGGLE_INOUT_SELECT_ALIAS;/* Selects IN or OUT endpoint */
    __IO uint32_t DATA_TOGGLE_RESET_ALIAS;/* Resets the data toggle to DATA0. */
    __IO uint32_t DATA_TOGGLE_DATA1_SET_ALIAS;/* Sets the data toggle to DATA1. */
    __I  uint32_t DATA_TOGGLE_STATUS_ALIAS;/* If this bit is 1, it means that the data toggle for selected endpoints is set to DATA1. Otherwise, it means that the data toggle is set to DATA0. */
} USB_SYS_CTRL1_Type;

#define USB_SYS_CTRL1                   ((USB_SYS_CTRL1_Type *) USB_SYS_CTRL1_BASE)

/* USB_SYS_CTRL1 settings */
#define USB_SIGNAL_REMOTE_RESUME_ENABLE_BITBAND 0x1
#define USB_SIGNAL_REMOTE_RESUME_DISABLE_BITBAND 0x0
#define USB_SIGNAL_REMOTE_RESUME_ENABLE ((uint32_t)(USB_SIGNAL_REMOTE_RESUME_ENABLE_BITBAND << USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_Pos))
#define USB_SIGNAL_REMOTE_RESUME_DISABLE ((uint32_t)(USB_SIGNAL_REMOTE_RESUME_DISABLE_BITBAND << USB_SYS_CTRL1_SIGNAL_REMOTE_RESUME_Pos))

#define USB_FORCE_J_ENABLE_BITBAND      0x1
#define USB_FORCE_J_DISABLE_BITBAND     0x0
#define USB_FORCE_J_ENABLE              ((uint32_t)(USB_FORCE_J_ENABLE_BITBAND << USB_SYS_CTRL1_FORCE_J_Pos))
#define USB_FORCE_J_DISABLE             ((uint32_t)(USB_FORCE_J_DISABLE_BITBAND << USB_SYS_CTRL1_FORCE_J_Pos))

#define USB_DISCONNECT_ENABLE_BITBAND   0x1
#define USB_DISCONNECT_DISABLE_BITBAND  0x0
#define USB_DISCONNECT_ENABLE           ((uint32_t)(USB_DISCONNECT_ENABLE_BITBAND << USB_SYS_CTRL1_DISCONNECT_Pos))
#define USB_DISCONNECT_DISABLE          ((uint32_t)(USB_DISCONNECT_DISABLE_BITBAND << USB_SYS_CTRL1_DISCONNECT_Pos))

/* USB_SYS_CTRL1 sub-register bit positions */
#define USB_CTRL_STATUS_BYTE_SIGNAL_REMOTE_RESUME_Pos 0
#define USB_CTRL_STATUS_BYTE_FORCE_J_Pos 1
#define USB_CTRL_STATUS_BYTE_DISCONNECT_Pos 3
#define USB_CTRL_STATUS_BYTE_SOF_GEN_Pos 5
#define USB_CTRL_STATUS_BYTE_WAKEUP_SOURCE_Pos 7
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_Pos 0
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_Mask ((uint32_t)(0x7U << USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_EP_SELECT_Pos))
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_INOUT_SELECT_Pos 4
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_RESET_Pos 5
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_DATA1_SET_Pos 6
#define USB_DATA_TOGGLE_CTRL_BYTE_DATA_TOGGLE_STATUS_Pos 7

/* USB_SYS_CTRL1 subregister settings */
#define USB_SIGNAL_REMOTE_RESUME_ENABLE_BYTE ((uint8_t)(USB_SIGNAL_REMOTE_RESUME_ENABLE_BITBAND << USB_CTRL_STATUS_BYTE_SIGNAL_REMOTE_RESUME_Pos))
#define USB_SIGNAL_REMOTE_RESUME_DISABLE_BYTE ((uint8_t)(USB_SIGNAL_REMOTE_RESUME_DISABLE_BITBAND << USB_CTRL_STATUS_BYTE_SIGNAL_REMOTE_RESUME_Pos))

#define USB_FORCE_J_ENABLE_BYTE         ((uint8_t)(USB_FORCE_J_ENABLE_BITBAND << USB_CTRL_STATUS_BYTE_FORCE_J_Pos))
#define USB_FORCE_J_DISABLE_BYTE        ((uint8_t)(USB_FORCE_J_DISABLE_BITBAND << USB_CTRL_STATUS_BYTE_FORCE_J_Pos))

#define USB_DISCONNECT_ENABLE_BYTE      ((uint8_t)(USB_DISCONNECT_ENABLE_BITBAND << USB_CTRL_STATUS_BYTE_DISCONNECT_Pos))
#define USB_DISCONNECT_DISABLE_BYTE     ((uint8_t)(USB_DISCONNECT_DISABLE_BITBAND << USB_CTRL_STATUS_BYTE_DISCONNECT_Pos))

/* USB Control and Status register 2 */
/*   Current USB frame count */
#define USB_SYS_CTRL2_BASE              0x400017D8

/* USB_SYS_CTRL2 bit positions */
#define USB_SYS_CTRL2_FUNCTION_ADDR_Pos 24
#define USB_SYS_CTRL2_FUNCTION_ADDR_Mask ((uint32_t)(0x7FU << USB_SYS_CTRL2_FUNCTION_ADDR_Pos))
#define USB_SYS_CTRL2_FRAME_COUNT_Pos   0
#define USB_SYS_CTRL2_FRAME_COUNT_Mask  ((uint32_t)(0x7FFU << USB_SYS_CTRL2_FRAME_COUNT_Pos))

/* USB_SYS_CTRL2 sub-register and bit-band aliases */
typedef struct
{
    __I  uint16_t FRAME_COUNT_SHORT;    
         uint8_t RESERVED0[1];
    __I  uint8_t FUNCTION_ADDR_BYTE;   
} USB_SYS_CTRL2_Type;

#define USB_SYS_CTRL2                   ((USB_SYS_CTRL2_Type *) USB_SYS_CTRL2_BASE)

/* USB_SYS_CTRL2 sub-register bit positions */
#define USB_FRAME_COUNT_SHORT_FRAME_COUNT_Pos 0
#define USB_FRAME_COUNT_SHORT_FRAME_COUNT_Mask ((uint32_t)(0x7FFU << USB_FRAME_COUNT_SHORT_FRAME_COUNT_Pos))
#define USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_Pos 0
#define USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_Mask ((uint32_t)(0x7FU << USB_FUNCTION_ADDR_BYTE_FUNCTION_ADDR_Pos))

/* USB Control and Status register 3 */
/*   Various control and status bit fields for USB */
#define USB_SYS_CTRL3_BASE              0x400017DC

/* USB_SYS_CTRL3 bit positions */
#define USB_SYS_CTRL3_OUT5_VALID_Pos    29
#define USB_SYS_CTRL3_OUT4_VALID_Pos    28
#define USB_SYS_CTRL3_OUT0_VALID_Pos    24
#define USB_SYS_CTRL3_IN3_VALID_Pos     19
#define USB_SYS_CTRL3_IN2_VALID_Pos     18
#define USB_SYS_CTRL3_IN0_VALID_Pos     16
#define USB_SYS_CTRL3_USB_PAIR_OUT_EP45_Pos 12
#define USB_SYS_CTRL3_USB_PAIR_IN_EP23_Pos 8

/* USB_SYS_CTRL3 sub-register and bit-band aliases */
typedef struct
{
         uint8_t RESERVED0[1];
    __IO uint8_t EP_PAIRING_BYTE;      
    __IO uint8_t EP023_IN_VALID_BYTE;  
    __IO uint8_t EP045_OUT_VALID_BYTE; 
         uint8_t RESERVED[SYS_CALC_BITBAND(0x400017DC, 0) - (0x400017DC + 4)];
         uint32_t RESERVED1[8];
    __IO uint32_t USB_PAIR_IN_EP23_ALIAS;/* Enable and disable the pairing of EP2 and EP3 IN endpoints */
         uint32_t RESERVED2[3];
    __IO uint32_t USB_PAIR_OUT_EP45_ALIAS;/* Enable and disable the pairing of EP4 and EP5 OUT endpoints */
         uint32_t RESERVED3[3];
    __IO uint32_t IN0_VALID_ALIAS;      /* Indicate if EP0 IN endpoint is valid for USB transfers */
         uint32_t RESERVED4;
    __IO uint32_t IN2_VALID_ALIAS;      /* Indicate if EP2 IN endpoint is valid for USB transfers */
    __IO uint32_t IN3_VALID_ALIAS;      /* Indicate if EP3 IN endpoint is valid for USB transfers */
         uint32_t RESERVED5[4];
    __IO uint32_t OUT0_VALID_ALIAS;     /* Indicate if EP0 OUT endpoint is valid for USB transfers */
         uint32_t RESERVED6[3];
    __IO uint32_t OUT4_VALID_ALIAS;     /* Indicate if EP4 OUT endpoint is valid for USB transfers */
    __IO uint32_t OUT5_VALID_ALIAS;     /* Indicate if EP5 OUT endpoint is valid for USB transfers */
} USB_SYS_CTRL3_Type;

#define USB_SYS_CTRL3                   ((USB_SYS_CTRL3_Type *) USB_SYS_CTRL3_BASE)

/* USB_SYS_CTRL3 settings */
#define USB_PAIR_IN_EP23_ENABLE_BITBAND 0x1
#define USB_PAIR_IN_EP23_DISABLE_BITBAND 0x0
#define USB_PAIR_IN_EP23_ENABLE         ((uint32_t)(USB_PAIR_IN_EP23_ENABLE_BITBAND << USB_SYS_CTRL3_USB_PAIR_IN_EP23_Pos))
#define USB_PAIR_IN_EP23_DISABLE        ((uint32_t)(USB_PAIR_IN_EP23_DISABLE_BITBAND << USB_SYS_CTRL3_USB_PAIR_IN_EP23_Pos))

#define USB_PAIR_OUT_EP45_ENABLE_BITBAND 0x1
#define USB_PAIR_OUT_EP45_DISABLE_BITBAND 0x0
#define USB_PAIR_OUT_EP45_ENABLE        ((uint32_t)(USB_PAIR_OUT_EP45_ENABLE_BITBAND << USB_SYS_CTRL3_USB_PAIR_OUT_EP45_Pos))
#define USB_PAIR_OUT_EP45_DISABLE       ((uint32_t)(USB_PAIR_OUT_EP45_DISABLE_BITBAND << USB_SYS_CTRL3_USB_PAIR_OUT_EP45_Pos))

#define USB_IN0_VALID_ENABLE_BITBAND    0x1
#define USB_IN0_VALID_DISABLE_BITBAND   0x0
#define USB_IN0_VALID_ENABLE            ((uint32_t)(USB_IN0_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_IN0_VALID_Pos))
#define USB_IN0_VALID_DISABLE           ((uint32_t)(USB_IN0_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_IN0_VALID_Pos))

#define USB_IN2_VALID_ENABLE_BITBAND    0x1
#define USB_IN2_VALID_DISABLE_BITBAND   0x0
#define USB_IN2_VALID_ENABLE            ((uint32_t)(USB_IN2_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_IN2_VALID_Pos))
#define USB_IN2_VALID_DISABLE           ((uint32_t)(USB_IN2_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_IN2_VALID_Pos))

#define USB_IN3_VALID_ENABLE_BITBAND    0x1
#define USB_IN3_VALID_DISABLE_BITBAND   0x0
#define USB_IN3_VALID_ENABLE            ((uint32_t)(USB_IN3_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_IN3_VALID_Pos))
#define USB_IN3_VALID_DISABLE           ((uint32_t)(USB_IN3_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_IN3_VALID_Pos))

#define USB_OUT0_VALID_ENABLE_BITBAND   0x1
#define USB_OUT0_VALID_DISABLE_BITBAND  0x0
#define USB_OUT0_VALID_ENABLE           ((uint32_t)(USB_OUT0_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_OUT0_VALID_Pos))
#define USB_OUT0_VALID_DISABLE          ((uint32_t)(USB_OUT0_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_OUT0_VALID_Pos))

#define USB_OUT4_VALID_ENABLE_BITBAND   0x1
#define USB_OUT4_VALID_DISABLE_BITBAND  0x0
#define USB_OUT4_VALID_ENABLE           ((uint32_t)(USB_OUT4_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_OUT4_VALID_Pos))
#define USB_OUT4_VALID_DISABLE          ((uint32_t)(USB_OUT4_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_OUT4_VALID_Pos))

#define USB_OUT5_VALID_ENABLE_BITBAND   0x1
#define USB_OUT5_VALID_DISABLE_BITBAND  0x0
#define USB_OUT5_VALID_ENABLE           ((uint32_t)(USB_OUT5_VALID_ENABLE_BITBAND << USB_SYS_CTRL3_OUT5_VALID_Pos))
#define USB_OUT5_VALID_DISABLE          ((uint32_t)(USB_OUT5_VALID_DISABLE_BITBAND << USB_SYS_CTRL3_OUT5_VALID_Pos))

/* USB_SYS_CTRL3 sub-register bit positions */
#define USB_EP_PAIRING_BYTE_USB_PAIR_IN_EP23_Pos 0
#define USB_EP_PAIRING_BYTE_USB_PAIR_OUT_EP45_Pos 4
#define USB_EP023_IN_VALID_BYTE_IN0_VALID_Pos 0
#define USB_EP023_IN_VALID_BYTE_IN2_VALID_Pos 2
#define USB_EP023_IN_VALID_BYTE_IN3_VALID_Pos 3
#define USB_EP045_OUT_VALID_BYTE_OUT0_VALID_Pos 0
#define USB_EP045_OUT_VALID_BYTE_OUT4_VALID_Pos 4
#define USB_EP045_OUT_VALID_BYTE_OUT5_VALID_Pos 5

/* USB_SYS_CTRL3 subregister settings */
#define USB_PAIR_IN_EP23_ENABLE_BYTE    ((uint8_t)(USB_PAIR_IN_EP23_ENABLE_BITBAND << USB_EP_PAIRING_BYTE_USB_PAIR_IN_EP23_Pos))
#define USB_PAIR_IN_EP23_DISABLE_BYTE   ((uint8_t)(USB_PAIR_IN_EP23_DISABLE_BITBAND << USB_EP_PAIRING_BYTE_USB_PAIR_IN_EP23_Pos))

#define USB_PAIR_OUT_EP45_ENABLE_BYTE   ((uint8_t)(USB_PAIR_OUT_EP45_ENABLE_BITBAND << USB_EP_PAIRING_BYTE_USB_PAIR_OUT_EP45_Pos))
#define USB_PAIR_OUT_EP45_DISABLE_BYTE  ((uint8_t)(USB_PAIR_OUT_EP45_DISABLE_BITBAND << USB_EP_PAIRING_BYTE_USB_PAIR_OUT_EP45_Pos))

#define USB_IN0_VALID_ENABLE_BYTE       ((uint8_t)(USB_IN0_VALID_ENABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN0_VALID_Pos))
#define USB_IN0_VALID_DISABLE_BYTE      ((uint8_t)(USB_IN0_VALID_DISABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN0_VALID_Pos))

#define USB_IN2_VALID_ENABLE_BYTE       ((uint8_t)(USB_IN2_VALID_ENABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN2_VALID_Pos))
#define USB_IN2_VALID_DISABLE_BYTE      ((uint8_t)(USB_IN2_VALID_DISABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN2_VALID_Pos))

#define USB_IN3_VALID_ENABLE_BYTE       ((uint8_t)(USB_IN3_VALID_ENABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN3_VALID_Pos))
#define USB_IN3_VALID_DISABLE_BYTE      ((uint8_t)(USB_IN3_VALID_DISABLE_BITBAND << USB_EP023_IN_VALID_BYTE_IN3_VALID_Pos))

#define USB_OUT0_VALID_ENABLE_BYTE      ((uint8_t)(USB_OUT0_VALID_ENABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT0_VALID_Pos))
#define USB_OUT0_VALID_DISABLE_BYTE     ((uint8_t)(USB_OUT0_VALID_DISABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT0_VALID_Pos))

#define USB_OUT4_VALID_ENABLE_BYTE      ((uint8_t)(USB_OUT4_VALID_ENABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT4_VALID_Pos))
#define USB_OUT4_VALID_DISABLE_BYTE     ((uint8_t)(USB_OUT4_VALID_DISABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT4_VALID_Pos))

#define USB_OUT5_VALID_ENABLE_BYTE      ((uint8_t)(USB_OUT5_VALID_ENABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT5_VALID_Pos))
#define USB_OUT5_VALID_DISABLE_BYTE     ((uint8_t)(USB_OUT5_VALID_DISABLE_BITBAND << USB_EP045_OUT_VALID_BYTE_OUT5_VALID_Pos))

/* USB Setup Data Buffer base Lower */
/*   Contains the lower 4 bytes of the SETUP data packet from the latest
 *   CONTROL transfer. */
#define USB_SETUP_DATA_BUF_BASE_0_BASE  0x400017E8

/* USB_SETUP_DATA_BUF_BASE_0 bit positions */
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_Pos 24
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE3_Pos))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_Pos 16
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE2_Pos))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_Pos 8
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE1_Pos))
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_Pos 0
#define USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_0_SETUP_DATA_BUF_BYTE0_Pos))

/* USB_SETUP_DATA_BUF_BASE_0 sub-register and bit-band aliases */
typedef struct
{
    __I  uint16_t SETUP_DATA_BUF_BASE0_SHORT;
    __I  uint16_t SETUP_DATA_BUF_BASE1_SHORT;
} USB_SETUP_DATA_BUF_BASE_0_Type;

#define USB_SETUP_DATA_BUF_BASE_0       ((USB_SETUP_DATA_BUF_BASE_0_Type *) USB_SETUP_DATA_BUF_BASE_0_BASE)

/* USB_SETUP_DATA_BUF_BASE_0 sub-register bit positions */
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_Pos 0
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE0_Pos))
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_Pos 8
#define SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE0_SHORT_SETUP_DATA_BUF_BYTE1_Pos))
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_Pos 0
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE2_Pos))
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_Pos 8
#define SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE1_SHORT_SETUP_DATA_BUF_BYTE3_Pos))

/* USB Setup Data Buffer base Higher */
/*   Contains the upper 4 bytes of the SETUP data packet from the latest
 *   CONTROL transfer. */
#define USB_SETUP_DATA_BUF_BASE_1_BASE  0x400017EC

/* USB_SETUP_DATA_BUF_BASE_1 bit positions */
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_Pos 24
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE7_Pos))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_Pos 16
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE6_Pos))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_Pos 8
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE5_Pos))
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_Pos 0
#define USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_Mask ((uint32_t)(0xFFU << USB_SETUP_DATA_BUF_BASE_1_SETUP_DATA_BUF_BYTE4_Pos))

/* USB_SETUP_DATA_BUF_BASE_1 sub-register and bit-band aliases */
typedef struct
{
    __I  uint16_t SETUP_DATA_BUF_BASE2_SHORT;
    __I  uint16_t SETUP_DATA_BUF_BASE3_SHORT;
} USB_SETUP_DATA_BUF_BASE_1_Type;

#define USB_SETUP_DATA_BUF_BASE_1       ((USB_SETUP_DATA_BUF_BASE_1_Type *) USB_SETUP_DATA_BUF_BASE_1_BASE)

/* USB_SETUP_DATA_BUF_BASE_1 sub-register bit positions */
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_Pos 0
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE4_Pos))
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_Pos 8
#define SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE2_SHORT_SETUP_DATA_BUF_BYTE5_Pos))
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_Pos 0
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE6_Pos))
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_Pos 8
#define SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_Mask ((uint32_t)(0xFFU << SETUP_DATA_BUF_BASE3_SHORT_SETUP_DATA_BUF_BYTE7_Pos))



#endif  /* Q32M210_HW_H */
