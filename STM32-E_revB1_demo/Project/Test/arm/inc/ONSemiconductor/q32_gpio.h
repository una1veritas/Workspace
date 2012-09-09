/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_gpio.h
 * - Hardware support functions for the GPIO interfaces
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_GPIO_H
#define Q32_GPIO_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * GPIO interface 0 support functions
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_IF0_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure GPIO interface 0; includes pull-up configuration,
 *                 output enable and MUX source selection (SPI0, GPIO or USRCLK)
 * Inputs        : config - GPIO interface 0 configuration; use
 *                          GPIO_IF0_PIN*_PULLUP_*, GPIO_IF0_PIN*_OUTPUT_* and
 *                          GPIO_IF0_SELECT_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF0_Config(uint32_t config)
{
    GPIO->IF0_FUNC_SEL = (config
                         & ((1U << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF0_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos) |
                            GPIO_IF0_FUNC_SEL_FUNC_SEL_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_GPIO_Get_IF0()
 * ----------------------------------------------------------------------------
 * Description   : Read the input data from GPIO interface 0
 * Inputs        : None
 * Outputs       : input -  Data read from the GPIO_IF0_FUNC_SEL_INPUT_DATA
 *                          bit-field
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_GPIO_Get_IF0()
{
    return ((GPIO->IF0_FUNC_SEL & GPIO_IF0_FUNC_SEL_INPUT_DATA_Mask)
            >> GPIO_IF0_FUNC_SEL_INPUT_DATA_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Set_IF0(uint32_t output)
 * ----------------------------------------------------------------------------
 * Description   : Write output data to GPIO interface 0
 * Inputs        : output - Data to write to GPIO interface 0
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Set_IF0(uint32_t output)
{
    GPIO->IF0_OUT = output;
}

/* ----------------------------------------------------------------------------
 * GPIO interface 1 support functions
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_IF1_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure GPIO interface 1; includes pull-up configuration,
 *                 output enable and MUX source selection (SPI1, GPIO or PCM)
 * Inputs        : config - GPIO interface 1 configuration; use
 *                          GPIO_IF1_PIN*_PULLUP_*, GPIO_IF1_PIN*_OUTPUT_* and
 *                          GPIO_IF1_SELECT_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF1_Config(uint32_t config)
{
    GPIO->IF1_FUNC_SEL = (config
                         & ((1U << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF1_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos) |
                            GPIO_IF1_FUNC_SEL_FUNC_SEL_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_GPIO_Get_IF1()
 * ----------------------------------------------------------------------------
 * Description   : Read the input data from GPIO interface 1
 * Inputs        : None
 * Outputs       : input -  Data read from the GPIO_IF1_FUNC_SEL_INPUT_DATA
 *                          bit-field
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_GPIO_Get_IF1()
{
    return ((GPIO->IF1_FUNC_SEL & GPIO_IF1_FUNC_SEL_INPUT_DATA_Mask)
            >> GPIO_IF1_FUNC_SEL_INPUT_DATA_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Set_IF1(uint32_t output)
 * ----------------------------------------------------------------------------
 * Description   : Write output data to GPIO interface 1
 * Inputs        : output - Data to write to GPIO interface 1
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Set_IF1(uint32_t output)
{
    GPIO->IF1_OUT = output;
}

/* ----------------------------------------------------------------------------
 * GPIO interfaces 2 and 3 support functions
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_IF23_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure GPIO interfaces 2 and 3; includes pull-up
 *                 configuration, output enable and MUX source selection (UART0
 *                 or GPIO, and UART1 or GPIO respectively)
 * Inputs        : config - GPIO interfaces 2 and 3 configuration; use
 *                          GPIO_IF2_PIN*_PULLUP_*, GPIO_IF3_PIN*_PULLUP_*,
 *                          GPIO_IF2_PIN*_OUTPUT_*, GPIO_IF3_PIN*_OUTPUT_*,
 *                          GPIO_IF2_SELECT_* and GPIO_IF3_SELECT_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF23_Config(uint32_t config)
{
    GPIO->IF23_FUNC_SEL = (config &
                        ((1U << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN1_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF3_PIN0_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN1_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_PULLUP_ENABLE_IF2_PIN0_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN1_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF3_PIN0_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN1_Pos) |
                         (1U << GPIO_IF23_FUNC_SEL_OUTPUT_ENABLE_IF2_PIN0_Pos) |
                                GPIO_IF23_FUNC_SEL_FUNC_SEL_IF3_Mask |
                         (1U << GPIO_IF23_FUNC_SEL_FUNC_SEL_IF2_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_GPIO_Get_IF23()
 * ----------------------------------------------------------------------------
 * Description   : Read the input data from GPIO interfaces 2 and 3
 * Inputs        : None
 * Outputs       : input -  Data read from the GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2
 *                          and GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3 bit-fields
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_GPIO_Get_IF23()
{
      return ((GPIO->IF23_FUNC_SEL
               & (GPIO_IF23_FUNC_SEL_INPUT_DATA_IF3_Mask
                  | GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_Mask))
              >> GPIO_IF23_FUNC_SEL_INPUT_DATA_IF2_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Set_IF23(uint32_t output)
 * ----------------------------------------------------------------------------
 * Description   : Write output data to GPIO interfaces 2 and 3
 * Inputs        : output - Data to write to GPIO interfaces 2 and 3
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Set_IF23(uint32_t output)
{
    GPIO->IF23_OUT = output;
}

/* ----------------------------------------------------------------------------
 * GPIO interface 4 support functions
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void  Sys_GPIO_IF4_Config(uint32_t functionConfig,
 *                                           uint32_t outputEnableConfig,
 *                                           uint32_t pullDownConfig)
 * ----------------------------------------------------------------------------
 * Description   : Configure GPIO interface 4; includes pull-down
 *                 configuration, output enable and MUX source selection (GPIO
 *                 or LCD)
 * Inputs        : - functionConfig     - GPIO interface 4 function selection;
 *                                        use a function select setting such as
 *                                        GPIO_IF4_ALL_GPIO
 *                 - outputEnableConfig - GPIO interface 4 output enable; use
 *                                        an output enable setting such as
 *                                        GPIO_IF4_OUTPUT_ENABLE_ALL
 *                 - pullDownConfig     - GPIO interface 4 pull-down resistor
 *                                        configuration; use
 *                                        GPIO_IF4_PIN*_PULLDOWN_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF4_Config(uint32_t functionConfig,
                                         uint32_t outputEnableConfig,
                                         uint32_t pullDownConfig)
{
    GPIO->IF4_LCD_FUNC_SEL = (functionConfig
                             & GPIO_IF4_LCD_FUNC_SEL_FUNC_SEL_Mask);
    GPIO->IF4_LCD_OE = (outputEnableConfig & GPIO_IF4_LCD_OE_OUTPUT_ENABLE_Mask);
    GPIO->IF4_LCD_PULLDOWN = (pullDownConfig &
                              GPIO_IF4_LCD_PULLDOWN_ENABLE_Mask);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_GPIO_Get_IF4()
 * ----------------------------------------------------------------------------
 * Description   : Read the input data from GPIO interface 4
 * Inputs        : None
 * Outputs       : input -  Data read from the GPIO_IF4_LCD_IN_INPUT_DATA
 *                          bit-field
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_GPIO_Get_IF4()
{
    return GPIO->IF4_LCD_IN;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Set_IF4(uint32_t output)
 * ----------------------------------------------------------------------------
 * Description   : Write output data to GPIO interface 4
 * Inputs        : output - Data to write to GPIO interface 4
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Set_IF4(uint32_t output)
{
    GPIO->IF4_LCD_OUT = output;
}


/* ----------------------------------------------------------------------------
 * GPIO interface 5 support functions
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_IF5_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure GPIO interface 5; includes pull-up configuration,
 *                 output enable and wakeup trigger enable
 * Inputs        : config - GPIO interface 5 configuration; use
 *                          GPIO_IF5_PIN*_PULLDOWN_*, GPIO_IF5_PIN*_PULLUP_*,
 *                          GPIO_IF5_PIN*_OUTPUT_* and GPIO_IF5_PIN*_WAKEUP_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF5_Config(uint32_t config)
{
    GPIO->IF5_FUNC_SEL = (config
                         & ((1U << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_PULLDOWN_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_PULLUP_ENABLE_PIN0_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_OUTPUT_ENABLE_PIN0_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN3_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN2_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN1_Pos) |
                            (1U << GPIO_IF5_FUNC_SEL_WAKEUP_ENABLE_PIN0_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_GPIO_Get_IF5()
 * ----------------------------------------------------------------------------
 * Description   : Read the input data from GPIO interface 5
 * Inputs        : None
 * Outputs       : input -  Data read from the GPIO_IF5_FUNC_SEL_INPUT_DATA
 *                          bit-field
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_GPIO_Get_IF5()
{
    return ((GPIO->IF5_FUNC_SEL & GPIO_IF5_FUNC_SEL_INPUT_DATA_Mask)
            >> GPIO_IF5_FUNC_SEL_INPUT_DATA_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Set_IF5(uint32_t output)
 * ----------------------------------------------------------------------------
 * Description   : Write output data to GPIO interface 5
 * Inputs        : output - Data to write to GPIO interface 5
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Set_IF5(uint32_t output)
{
    GPIO->IF5_OUT = output;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_IF_Input_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Select which GPIO interfaces are enabled as inputs
 * Inputs        : config - Selection of the GPIO interfaces to allow input on;
 *                          use GPIO_IF*_IN_ENABLE/GPIO_IF*_IN_DISABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_IF_Input_Config(uint32_t config)
{
    GPIO->IF_INPUT_ENABLE = (config
                            & ((1U << GPIO_IF_INPUT_ENABLE_IF0_INPUT_Pos)
                               | (1U << GPIO_IF_INPUT_ENABLE_IF1_INPUT_Pos)
                               | (1U << GPIO_IF_INPUT_ENABLE_IF2_INPUT_Pos)
                               | (1U << GPIO_IF_INPUT_ENABLE_IF3_INPUT_Pos)
                               | (1U << GPIO_IF_INPUT_ENABLE_IF4_INPUT_Pos)
                               | (1U << GPIO_IF_INPUT_ENABLE_IF5_INPUT_Pos)));
}



/* ----------------------------------------------------------------------------
 * GPIO Interrupt configuration function prototypes
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP0Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 0
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            0; use GPIO_GP0_INT_DISABLE_SHORT/
 *                            GPIO_GP0_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP0Config(uint16_t config)
{
    GPIO_INT_CTRL0->GP0_SHORT = (config
                               & (GPIO_INT_CTRL_GP0_SHORT_GP0_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP0_SHORT_GP0_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP1Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 1
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            1; use GPIO_GP1_INT_DISABLE_SHORT/
 *                            GPIO_GP1_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP1Config(uint16_t config)
{
    GPIO_INT_CTRL0->GP1_SHORT = (config
                               & (GPIO_INT_CTRL_GP1_SHORT_GP1_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP1_SHORT_GP1_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP2Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 2
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            2; use GPIO_GP2_INT_DISABLE_SHORT/
 *                            GPIO_GP2_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP2Config(uint16_t config)
{
    GPIO_INT_CTRL1->GP2_SHORT = (config
                               & (GPIO_INT_CTRL_GP2_SHORT_GP2_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP2_SHORT_GP2_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP3Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 3
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            3; use GPIO_GP3_INT_DISABLE_SHORT/
 *                            GPIO_GP3_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP3Config(uint16_t config)
{
    GPIO_INT_CTRL1->GP3_SHORT = (config
                               & (GPIO_INT_CTRL_GP3_SHORT_GP3_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP3_SHORT_GP3_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP4Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 4
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            4; use GPIO_GP4_INT_DISABLE_SHORT/
 *                            GPIO_GP4_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP4Config(uint16_t config)
{
    GPIO_INT_CTRL2->GP4_SHORT = (config
                               & (GPIO_INT_CTRL_GP4_SHORT_GP4_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP4_SHORT_GP4_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP5Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 5
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            5; use GPIO_GP5_INT_DISABLE_SHORT/
 *                            GPIO_GP5_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP5Config(uint16_t config)
{
    GPIO_INT_CTRL2->GP5_SHORT = (config
                               & (GPIO_INT_CTRL_GP5_SHORT_GP5_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP5_SHORT_GP5_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP6Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 6
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            6; use GPIO_GP6_INT_DISABLE_SHORT/
 *                            GPIO_GP6_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP6Config(uint16_t config)
{
    GPIO_INT_CTRL3->GP6_SHORT = (config
                               & (GPIO_INT_CTRL_GP6_SHORT_GP6_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP6_SHORT_GP6_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_GPIO_Int_GP7Config(uint16_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure general-purpose GPIO interrupt 7
 * Inputs        : config   - Configuration for general-purpose GPIO interrupt
 *                            7; use GPIO_GP7_INT_DISABLE_SHORT/
 *                            GPIO_GP7_INT_*_TRIGGER_SHORT and a source index
 *                            GPIO_IF*_PIN*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_GPIO_Int_GP7Config(uint16_t config)
{
    GPIO_INT_CTRL3->GP7_SHORT = (config
                               & (GPIO_INT_CTRL_GP7_SHORT_GP7_INT_TYPE_Mask
                                  | GPIO_INT_CTRL_GP7_SHORT_GP7_INT_SRC_Mask));
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif
#endif /* Q32_GPIO_H */
