/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_usb.h
 * - USB interface controller hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_USB_H
#define Q32_USB_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * USB interface controller type definitions and defines
 * ------------------------------------------------------------------------- */
typedef enum
{
    USB_OUT,
    USB_IN
} Direction;

#define USB_EPSTALL_Mask                0x01
#define USB_HSNAK_Mask                  0x01
#define USB_SETUP_BUFFER_SIZE           0x8

/* ----------------------------------------------------------------------------
 * USB interface controller support function prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_USB_Initialize(void);

extern void Sys_USB_Set_EndpointByteCount(uint32_t EP, Direction direction,
                                          uint8_t count);
extern uint8_t Sys_USB_Get_EndpointByteCount(uint32_t EP);

extern void Sys_USB_SendEndpoint(uint32_t EP, uint32_t size,
                                 uint8_t* data);
extern void Sys_USB_ReceiveEndpoint(uint32_t EP, uint32_t* size,
                                    uint8_t* data);

extern void Sys_USB_Get_SetupBuffer(uint8_t* setupBuffer);
extern void Sys_USB_StallEndpoint(uint32_t EP, Direction direction);

/* ----------------------------------------------------------------------------
 * USB interface controller support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB interface controller
 * Inputs        : config   - Setup for the USB interface controller; use
 *                            USB_CONTROLLER_CM3/USB_CONTROLLER_DMA,
 *                            USB_PHY_ENABLED/USB_PHY_STANDBY,
 *                            USB_REMOTE_WAKEUP_ENABLE,
 *                            USB_RESET_ENABLE/USB_RESET_DISABLE,
 *                            USB_DISABLE/USB_ENABLE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Config(uint32_t config)
{
    USB_TOP->CTRL = (config & ((1U << USB_CTRL_CONTROLLER_Pos) |
                               (1U << USB_CTRL_PHY_STANDBY_Pos) |
                               (1U << USB_CTRL_REMOTE_WAKEUP_Pos) |
                               (1U << USB_CTRL_RESET_Pos) |
                               (1U << USB_CTRL_ENABLE_Pos)));
}


/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Reset()
 * ----------------------------------------------------------------------------
 * Description   : Reset the USB interface
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : USB is enabled.
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Reset()
{
    USB_CTRL->RESET_ALIAS = USB_RESET_ENABLE_BITBAND;
    USB_CTRL->RESET_ALIAS = USB_RESET_DISABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable the USB interface
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Enable()
{
    USB_CTRL->ENABLE_ALIAS = USB_ENABLE_BITBAND;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable the USB interface
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Disable()
{
    USB_CTRL->ENABLE_ALIAS = USB_DISABLE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Set_EndpointIn(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB in endpoints
 * Inputs        : config   - Setup for the USB in endpoints; use
 *                            USB_IN0_VALID_ENABLE_BYTE/
 *                            USB_IN0_VALID_DISABLE_BYTE,
 *                            USB_IN2_VALID_ENABLE_BYTE/
 *                            USB_IN2_VALID_DISABLE_BYTE,
 *                            USB_IN3_VALID_ENABLE_BYTE/
 *                            USB_IN3_VALID_DISABLE_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Set_EndpointIn(uint8_t config)
{
    USB_SYS_CTRL3->EP023_IN_VALID_BYTE = (config & ( USB_IN0_VALID_ENABLE_BYTE |
                                          USB_IN2_VALID_ENABLE_BYTE |
                                          USB_IN3_VALID_ENABLE_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Set_EndpointOut(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB out endpoints
 * Inputs        : config   - Setup for the USB out endpoints; use
 *                            USB_OUT0_VALID_ENABLE_BYTE/
 *                            USB_OUT0_VALID_DISABLE_BYTE
 *                            USB_OUT4_VALID_ENABLE_BYTE/
 *                            USB_OUT4_VALID_DISABLE_BYTE
 *                            USB_OUT5_VALID_ENABLE_BYTE/
 *                            USB_OUT5_VALID_DISABLE_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Set_EndpointOut(uint8_t config)
{
    USB_SYS_CTRL3->EP045_OUT_VALID_BYTE = (config &
                                         ( USB_OUT0_VALID_ENABLE_BYTE |
                                           USB_OUT4_VALID_ENABLE_BYTE |
                                           USB_OUT5_VALID_ENABLE_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Set_EndpointPairing(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB endpoint pairing
 * Inputs        : config   - Setup for the USB endpoint pairing; use
 *                            USB_PAIR_IN_EP23_ENABLE_BYTE/
 *                            USB_PAIR_IN_EP23_DISABLE_BYTE,
 *                            USB_PAIR_OUT_EP45_ENABLE_BYTE/
 *                            USB_PAIR_OUT_EP45_DISABLE_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Set_EndpointPairing(uint8_t config)
{
    USB_SYS_CTRL3->EP_PAIRING_BYTE = 
                                (config & ( USB_PAIR_IN_EP23_ENABLE_BYTE |
                                            USB_PAIR_OUT_EP45_ENABLE_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_InterruptConfig(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB interrupts
 * Inputs        : config   - Setup for USB interface interrupts; use
 *                            USB_DAT_VALID_IEN_ENABLE_BYTE/
 *                            USB_DAT_VALID_IEN_DISABLE_BYTE,
 *                            USB_SOF_IEN_ENABLE_BYTE/USB_SOF_IEN_DISABLE_BYTE,
 *                            USB_SETUPTKN_IEN_ENABLE_BYTE/
 *                            USB_SETUPTKN_IEN_DISABLE_BYTE,
 *                            USB_SUS_IEN_ENABLE_BYTE/USB_SUS_IEN_DISABLE_BYTE,
 *                            USB_RST_IEN_ENABLE_BYTE/USB_RST_IEN_DISABLE_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_InterruptConfig(uint8_t config)
{
    USB_INT_CTRL->IEN_BYTE = (config & ( USB_RST_IEN_ENABLE_BYTE |
                                         USB_SUS_IEN_ENABLE_BYTE |
                                         USB_SETUPTKN_IEN_ENABLE_BYTE |
                                         USB_SOF_IEN_ENABLE_BYTE |
                                         USB_DAT_VALID_IEN_ENABLE_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_BulkInInterruptConfig(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB bulk in endpoint interrupts
 * Inputs        : config   - Setup for the bulk in endpoint interrupts; use
 *                            USB_BULK_IN_0_IEN_ENABLE_BYTE/
 *                            USB_BULK_IN_0_IEN_DISABLE_BYTE,
 *                            USB_BULK_IN_2_IEN_ENABLE_BYTE/
 *                            USB_BULK_IN_2_IEN_DISABLE_BYTE,
 *                            USB_BULK_IN_3_IEN_ENABLE_BYTE/
 *                            USB_BULK_IN_3_IEN_DISABLE_BYTE,
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_BulkInInterruptConfig(uint8_t config)
{
    USB_INT_CTRL->BULK_IN_IEN_BYTE = (config &
                              ((1U << USB_BULK_IN_IEN_BYTE_BULK_IN_0_IEN_Pos) |
                               (1U << USB_BULK_IN_IEN_BYTE_BULK_IN_2_IEN_Pos) |
                               (1U << USB_BULK_IN_IEN_BYTE_BULK_IN_3_IEN_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_BulkOutInterruptConfig(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the USB bulk out endpoint interrupts
 * Inputs        : config   - Setup for the bulk out endpoint interrupts; use
 *                            USB_BULK_OUT_0_IEN_ENABLE_BYTE/
 *                            USB_BULK_OUT_0_IEN_DISABLE_BYTE,
 *                            USB_BULK_OUT_4_IEN_ENABLE_BYTE/
 *                            USB_BULK_OUT_4_IEN_DISABLE_BYTE,
 *                            USB_BULK_OUT_5_IEN_ENABLE_BYTE/
 *                            USB_BULK_OUT_5_IEN_DISABLE_BYTE,
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_BulkOutInterruptConfig(uint8_t config)
{
    USB_INT_CTRL->BULK_OUT_IEN_BYTE = (config &
                            ((1U << USB_BULK_OUT_IEN_BYTE_BULK_OUT_0_IEN_Pos) |
                             (1U << USB_BULK_OUT_IEN_BYTE_BULK_OUT_4_IEN_Pos) |
                             (1U << USB_BULK_OUT_IEN_BYTE_BULK_OUT_5_IEN_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Clear_Interrupt(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Clear the USB interface interrupts
 * Inputs        : config   - Clear the USB interface interrupts; use
 *                            USB_RST_IRQ_CLEAR_BYTE, USB_SUS_IRQ_CLEAR_BYTE,
 *                            USB_SETUPTKN_IRQ_CLEAR_BYTE,
 *                            USB_SOF_IRQ_CLEAR_BYTE,
 *                            USB_DAT_VALID_IRQ_CLEAR_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Clear_Interrupt(uint8_t config)
{
    USB_INT_STATUS->IRQ_BYTE = (config & ( USB_RST_IRQ_CLEAR_BYTE |
                                           USB_SUS_IRQ_CLEAR_BYTE |
                                           USB_SETUPTKN_IRQ_CLEAR_BYTE |
                                           USB_SOF_IRQ_CLEAR_BYTE |
                                           USB_DAT_VALID_IRQ_CLEAR_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Clear_BulkInInterrupt(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Clear the USB bulk in interrupts
 * Inputs        : config   - Clear the USB bulk in interrupts; use
 *                            USB_BULK_IN_0_IRQ_CLEAR_BYTE,
 *                            USB_BULK_IN_2_IRQ_CLEAR_BYTE,
 *                            USB_BULK_IN_3_IRQ_CLEAR_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Clear_BulkInInterrupt(uint8_t config)
{
    USB_INT_STATUS->BULK_IN_IRQ_BYTE = 
                                    (config & ( USB_BULK_IN_0_IRQ_CLEAR_BYTE |
                                                USB_BULK_IN_2_IRQ_CLEAR_BYTE |
                                                USB_BULK_IN_3_IRQ_CLEAR_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_Clear_BulkOutInterrupt(uint8_t config)
 * ----------------------------------------------------------------------------
 * Description   : Clear the USB bulk out interrupts
 * Inputs        : config   - Clear the USB bulk out interrupts; use
 *                            USB_BULK_OUT_0_IRQ_CLEAR_BYTE,
 *                            USB_BULK_OUT_4_IRQ_CLEAR_BYTE,
 *                            USB_BULK_OUT_5_IRQ_CLEAR_BYTE
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_Clear_BulkOutInterrupt(uint8_t config)
{
    USB_INT_STATUS->BULK_OUT_IRQ_BYTE = (config &
                                        ( USB_BULK_OUT_0_IRQ_CLEAR_BYTE |
                                          USB_BULK_OUT_4_IRQ_CLEAR_BYTE |
                                          USB_BULK_OUT_5_IRQ_CLEAR_BYTE ));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_USB_ClearHandshakeNAK()
 * ----------------------------------------------------------------------------
 * Description   : Clear endpoint 0 and send a handshake
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_USB_ClearHandshakeNAK()
{
    USB_EP0_IN_CTRL->EP0_HSNAK_ALIAS = USB_HSNAK_Mask;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_USB_H */
