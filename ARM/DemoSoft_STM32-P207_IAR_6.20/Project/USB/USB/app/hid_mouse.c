/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2003
 *
 *    File name      : hid_mouse.c
 *    Description    : Define HID module
 *
 *    History :
 *    1. Date        : December 19, 2005
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#define HID_GLOBAL
#include "hid_mouse.h"

#pragma data_alignment=4
Int8U DataBuffer[sizeof(UsbHidDescriptor_t)];
#pragma data_alignment=4

MouseReport_t MouseReport =
{
#if HID_ID_NUMB > 0
  HID_MOUSE_ID,     // Report ID
#endif // HID_ID_NUMB > 0
  0,                // Buttons
  0,                // X
  0,                // Y
};

#if HID_BOOT_DEVICE > 0
static MouseProtocol_t HidProtocol;
#endif // HID_BOOT_DEVICE > 0

#if HID_IDLE_SUPP > 0
static HidIdleCtrl_t HidIdle[HID_ID_NUMB+1];
#endif // HID_IDLE_SUPP > 0

volatile static Boolean HID_RepotrEpDly;
volatile static Boolean RelativeData0;   // that flag is associated with relative data (X,Y)
volatile static Boolean HID_Cnfg, HID_Decnfg;

/*************************************************************************
 * Function Name: HidInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: Init HID Mouse
 *
 *************************************************************************/
void HidInit (void)
{
  bHidCnfg = HID_Cnfg = FALSE;
  HID_Decnfg = TRUE;
  HidCnfgInit(NULL);
  UsbCoreInit();
}

/*************************************************************************
 * Function Name: HidCnfgInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: HID configure
 *
 *************************************************************************/
void HidCnfgInit (pUsbDevCtrl_t pDev)
{
  if (pDev != NULL && pDev->Configuration)
  {
    HID_Cnfg = TRUE;
    bHidCnfg = TRUE;
    HID_RepotrEpDly = 0;
  #if HID_BOOT_DEVICE > 0
    HidProtocol = ReportProtocol;
  #endif // HID_BOOT_DEVICE > 0
    MouseReport.Buttons = 0;
    MouseReport.X = \
    MouseReport.Y = 0;
    RelativeData0 = FALSE;
  #if HID_IDLE_SUPP > 0
    for(Int32U i = (HID_ID_NUMB?1:0); i < HID_ID_NUMB+1; i++)
    {
      HidIdle[i].Dly  = \
      HidIdle[i].Cnt  = 0;
    }
  #endif // HID_IDLE_SUPP > 0
  }
  else
  {
    HID_Decnfg = TRUE;
    bHidCnfg = FALSE;
  }
}

/*************************************************************************
 * Function Name: UsbClassHid_SOF
 * Parameters: none
 *
 * Return: none
 *
 * Description: Called on every SOF
 *
 *************************************************************************/
void UsbClassHid_SOF (void)
{
  if(bHidCnfg)
  {

    if (HID_RepotrEpDly)
    {
      --HID_RepotrEpDly;
    }
    else if (RelativeData0)
    {
      RelativeData0 = FALSE;
      MouseReport.X = MouseReport.Y = 0;
    }

#if HID_IDLE_SUPP > 0
static Int32U Prescaler = 0;
    if(++Prescaler < 4)
    {
      return;
    }
    Prescaler = 0;

    for(Int32U i = (HID_ID_NUMB?1:0); i < HID_ID_NUMB+1; i++)
    {
      if(HidIdle[i].Dly && (++HidIdle[i].Cnt >= HidIdle[i].Dly))
      {
        // send report
        Boolean Status = HidSendInReport(FALSE,i);
        assert(Status);
        HidIdle[i].Cnt = 0;
      }
    }
#endif // HID_IDLE_SUPP > 0
  }
}

/*************************************************************************
 * Function Name: UsbClassHidDescriptor
 * Parameters:  pUsbSetupPacket_t pSetup
 *
 * Return: UsbCommStatus_t
 *
 * Description: Implement GET DESCRIPTOR
 *
 *************************************************************************/
UsbCommStatus_t UsbClassHidDescriptor (pUsbSetupPacket_t pSetup)
{
  if (pSetup->wIndex.Word == HID_INTERFACE_0)
  {
    switch (pSetup->wValue.Hi)
    {
    case Hid:
      USB_IO_Data(CTRL_ENP_IN,
                 (pInt8U)HidDescriptor,
                  USB_T9_Size(SIZE_OF_HID_DESC,pSetup->wLength.Word),
                 (void *)USB_TransferEndHandler);
      return(UsbPass);
    case HidReport:
      if (HidProtocol == ReportProtocol)
      {
        USB_IO_Data(CTRL_ENP_IN,
                   (pInt8U)HidReportDescriptor,
                    USB_T9_Size(SIZE_OF_HID_MOUSE_DESC,pSetup->wLength.Word),
                   (void *)USB_TransferEndHandler);
        return(UsbPass);
      }
    }
  }
  return(UsbNotSupport);
}

/*************************************************************************
 * Function Name: UsbClassHidRequest
 * Parameters:  pUsbSetupPacket_t pSetup
 *
 * Return: UsbCommStatus_t
 *
 * Description: Implement USB Class Hid requests
 *
 *************************************************************************/
UsbCommStatus_t UsbClassHidRequest (pUsbSetupPacket_t pSetup)
{
  switch (pSetup->bRequest)
  {
  case HID_GET_REPORT:
    if(pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      if (pSetup->wValue.Hi == HID_INPUT)
      {
        if(HidSendInReport(TRUE,pSetup->wValue.Lo))
        {
          return(UsbPass);
        }
      }
      else if (pSetup->wValue.Hi == HID_OUTPUT)
      {
      }
      else if (pSetup->wValue.Hi == HID_FEATURE )
      {
      }
    }
    break;
  case HID_SET_REPORT:
    if (pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      if (pSetup->wValue.Hi == HID_INPUT)
      {
      }
      else if (pSetup->wValue.Hi == HID_OUTPUT)
      {
      }
      else if (pSetup->wValue.Hi == HID_FEATURE)
      {
      }
    }
    break;
#if HID_IDLE_SUPP > 0
  case HID_GET_IDLE:
    if (pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      if(pSetup->wValue.Lo <= HID_ID_NUMB)
      {
        USB_IO_Data(CTRL_ENP_IN,
                   (pInt8U)&HidIdle[pSetup->wValue.Lo].Dly,
                    USB_T9_Size(1,pSetup->wLength.Word),
                   (void *)USB_TransferEndHandler);
        return(UsbPass);
      }
    }
    break;
  case HID_SET_IDLE:
    if (pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      if(pSetup->wValue.Lo <= HID_ID_NUMB)
      {
        if(pSetup->wValue.Lo == 0)
        {
          for(Int32U i = 1; i < HID_ID_NUMB+1; i++)
          {
            HidIdle[i].Dly = pSetup->wValue.Hi;
          }
        }
        HidIdle[pSetup->wValue.Lo].Dly = pSetup->wValue.Hi;
        USB_StatusHandler(CTRL_ENP_OUT);
        return(UsbPass);
      }
    }
    break;
#endif // HID_IDLE_SUPP > 0
#if HID_BOOT_DEVICE > 0
  case HID_GET_PROTOCOL:
    if (pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      USB_IO_Data(CTRL_ENP_IN,
                 (pInt8U)&HidProtocol,
                  USB_T9_Size(1,pSetup->wLength.Word),
                 (void *)USB_TransferEndHandler);
      return(UsbPass);
    }
    break;
  case HID_SET_PROTOCOL:
    if (pSetup->wIndex.Word == HID_INTERFACE_0)
    {
      HidProtocol = (MouseProtocol_t)pSetup->wValue.Word;
      USB_StatusHandler(CTRL_ENP_OUT);
      return(UsbPass);
    }
    break;
#endif // HID_BOOT_DEVICE > 0
  }
  return(UsbNotSupport);
}

/*************************************************************************
 * Function Name: USB_TransferEndHandler
 * Parameters:  USB_Endpoint_t EP
 *
 * Return: none
 *
 * Description: USB HID report end
 *
 *************************************************************************/
void USB_TransferEndHandler (USB_Endpoint_t EP)
{
  if (EP == CTRL_ENP_IN)
  {
    USB_StatusHandler(CTRL_ENP_IN);
  }
  else if (EP == CTRL_ENP_OUT)
  {
    USB_StatusHandler(CTRL_ENP_OUT);
  }
  else if (EP == ReportEpHid)
  {
    HID_RepotrEpDly = 0;
  }
  else
  {
    assert(0);
  }

  if (RelativeData0)
  {
    RelativeData0 = FALSE;
    MouseReport.X = MouseReport.Y = 0;
  }
}

/*************************************************************************
 * Function Name: HidMouseSendReport
 * Parameters:  Int8S X, Int8S Y, Int8U Buttons
 *
 * Return: none
 *
 * Description: USB HID Mouse report send
 *
 *************************************************************************/
void HidMouseSendReport (Int8S X, Int8S Y, Int8U Buttons)
{
  MouseReport.X = X;
  MouseReport.Y = Y;
  if((MouseReport.Buttons != Buttons) ||
     MouseReport.X ||
     MouseReport.Y)
  {
    MouseReport.Buttons = Buttons;
    HidSendInReport(FALSE,HID_MOUSE_ID);
  }
}

/*************************************************************************
 * Function Name: HidSendInReport
 * Parameters:  Boolean Ctrl, Int8U ID
 *
 * Return: none
 *
 * Description: USB HID send different report
 *
 *************************************************************************/
static
Boolean HidSendInReport (Boolean Ctrl, Int8U ID)
{
#if __CORE__ < 7
Int32U Save;
#endif // __CORE__ < 7

#if __CORE__ < 7
  ENTR_CRT_SECTION(Save);
#else
  ENTR_CRT_SECTION();
#endif // __CORE__ < 7

  if(!Ctrl)
  {
    HID_RepotrEpDly = ReportEpPollingPeriod;

#if HID_IDLE_SUPP > 0
    if(HidIdle[ID].Dly)
    {
      if (HidIdle[ID].Cnt < HidIdle[ID].Dly)
      {
      #if __CORE__ < 7
        EXT_CRT_SECTION(Save);
      #else
        EXT_CRT_SECTION();
      #endif // __CORE__ < 7
        return(TRUE);
      }
    }
#endif
  }

  switch(ID)
  {
  case HID_MOUSE_ID:
    RelativeData0 = TRUE;
    USB_IO_Data((Ctrl?CTRL_ENP_IN:ReportEpHid),
              #if HID_ID_NUMB > 0
                (pInt8U)(((pInt8U)&MouseReport) + ((HidProtocol == ReportProtocol)?0:1)),     // first byte is ID
              #else
                (pInt8U)&MouseReport,
              #endif // HID_ID_NUMB > 0
                (HidProtocol == ReportProtocol)?sizeof(MouseReport_t):BootReportMaxSize,
                (void *)USB_TransferEndHandler);
    break;
  default:
    #if __CORE__ < 7
      EXT_CRT_SECTION(Save);
    #else
      EXT_CRT_SECTION();
    #endif // __CORE__ < 7
    return(FALSE);
  }
#if __CORE__ < 7
  EXT_CRT_SECTION(Save);
#else
  EXT_CRT_SECTION();
#endif // __CORE__ < 7
  return(TRUE);
}
