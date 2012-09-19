/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : usb_trans_layer.h
 *    Description : STM32 USB-OTG-FS to USB Framework translation layer
 *
 *    History :
 *    1. Date        : 3, September 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#include <stdint.h>

#ifndef __USB_TRANS_LAYER_H__
#define __USB_TRANS_LAYER_H__

#define USBF_ERR_NONE 0

extern void SOF_Callback(void);
extern void WKUP_Callback(void);
extern void SUSP_Callback(void);
extern void USBF_EP_TxPktSent(uint32_t EP_num);
extern void USBF_EP_RxPktRdy(uint32_t EP_num);
extern void USBF_EP_DoSetupPkt(void * pBuff);
extern void RESET_Callback(void);

#endif // __USB_TRANS_LAYER_H__
