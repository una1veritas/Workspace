/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : usb_trans_layer.c
 *    Description : STM32 USB-OTG-FS to USB Framework translation layer
 *
 *    History :
 *    1. Date        : 3, September 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/

#include "includes.h"
#include "otgd_fs_cal.h"
#include "otgd_fs_pcd.h"
#include "otgd_fs_dev.h"
#include "usb_trans_layer.h"
#include "usb_hw.h"
#include "usb_hooks.h"


void SOF_Callback(void)
{
  #if USB_SOF_FRAME_NUMB > 0
    USB_FRAME_HOOK(USB_GetFrameNumb());
  #else
    USB_FRAME_HOOK(0);
  #endif
}

void WKUP_Callback(void)
{

}

void SUSP_Callback(void)
{
  UsbDevSuspendCallback(TRUE);
}

void USBF_EP_TxPktSent(uint32_t EP_num)
{
pEpCnfg_t  pEP = &EpCnfg[USB_EpLogToPhysAdd(EP_num)];

USB_OTG_EP *ep =  OTGD_FS_PCD_GetInEP(EP_num & 0x7F);
  if(NULL == ep)
  {
    // invalid EP
    return;
  }

  pEP->Status = COMPLETE;

  // call callback function
  if(pEP->pFn)
  {
    ((void(*)(USB_Endpoint_t))pEP->pFn)(USB_EpLogToPhysAdd(EP_num));
  }
}

void USBF_EP_RxPktRdy(uint32_t EP_num)
{
pEpCnfg_t pEP = &EpCnfg[USB_EpLogToPhysAdd(EP_num)];
USB_OTG_EP *ep =  OTGD_FS_PCD_GetOutEP(EP_num & 0x7F);
  if(NULL == ep)
  {
    // invalid EP
    return;
  }

  if(pEP->Size == ep->xfer_count)
  {
    pEP->Status = COMPLETE;
  }
  else if (pEP->Size > ep->xfer_count)
  {
    pEP->Status = BUFFER_UNDERRUN;
  }
  else
  {
    pEP->Status = BUFFER_OVERRUN;
  }

  pEP->Size = ep->xfer_count;

  // call callback function
  if(pEP->pFn)
  {
    ((void(*)(USB_Endpoint_t))pEP->pFn)(USB_EpLogToPhysAdd(EP_num));
  }
}

void USBF_EP_DoSetupPkt(void * pBuff)
{
  USB_IO_Data(CTRL_ENP_OUT,NULL,-1ul,NULL);
  USB_IO_Data(CTRL_ENP_IN,NULL,-1ul,NULL);
  memcpy(UsbEp0SetupPacket.Data,pBuff,sizeof(UsbEp0SetupPacket));
  USB_SetupHandler();
}

void RESET_Callback(void)
{
  USB_HwReset();
  UsbDevSuspendCallback(FALSE);
  UsbDevResetCallback();
}
