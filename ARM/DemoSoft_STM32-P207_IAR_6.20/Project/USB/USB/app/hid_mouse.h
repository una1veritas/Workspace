/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2005
 *
 *    File name      : hid_mouse.h
 *    Description    : Definition of HID mouse device
 *
 *    History :
 *    1. Date        : December 19, 2005
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#include "includes.h"
#include "hid.h"

#ifndef __HID_MOUSE_H
#define __HID_MOUSE_H

#ifdef HID_GLOBAL
#define HID_EXTERN
#else
#define HID_EXTERN extern
#endif

extern const Int8U HidReportDescriptor[];
extern const Int8U HidDescriptor[];

#define SIZE_OF_HID_MOUSE_DESC  50
#define SIZE_OF_HID_DESC        (sizeof(UsbHidDescriptor_t))

typedef struct _HidIdleCtrl_t
{
  Int8U Dly;
  Int8U Cnt;
} HidIdleCtrl_t, * pHidIdleCtrl_t;

#pragma pack(1)

typedef struct _MouseReport_t
{
#if HID_ID_NUMB > 0
  const Int8U ID;
#endif // HID_ID_NUMB > 0
  Int8U Buttons;
  Int8S X;
  Int8S Y;
} MouseReport_t, *pMouseReport_t;

#pragma pack()

HID_EXTERN Boolean bHidCnfg;

/*************************************************************************
 * Function Name: HidInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: Init HID Mouse
 *
 *************************************************************************/
void HidInit (void);

/*************************************************************************
 * Function Name: HidCnfgInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: HID configure
 *
 *************************************************************************/
void HidCnfgInit (pUsbDevCtrl_t pDev);

/*************************************************************************
 * Function Name: UsbClassHid_SOF
 * Parameters: none
 *
 * Return: none
 *
 * Description: Called on every SOF
 *
 *************************************************************************/
void UsbClassHid_SOF (void);

/*************************************************************************
 * Function Name: UsbClassHidConfigure
 * Parameters:  void * pArg
 *
 * Return: void *
 *
 * Description: USB Class HID configure
 *
 *************************************************************************/
void * UsbClassHidConfigure (void * pArg);

/*************************************************************************
 * Function Name: UsbClassHidDescriptor
 * Parameters:  pUsbSetupPacket_t pSetup
 *
 * Return: UsbCommStatus_t
 *
 * Description: Implement GET DESCRIPTOR
 *
 *************************************************************************/
UsbCommStatus_t UsbClassHidDescriptor (pUsbSetupPacket_t pSetup);

/*************************************************************************
 * Function Name: UsbClassHidRequest
 * Parameters:  pUsbSetupPacket_t pSetup
 *
 * Return: UsbCommStatus_t
 *
 * Description: Implement USB Class Hid requests
 *
 *************************************************************************/
UsbCommStatus_t UsbClassHidRequest (pUsbSetupPacket_t pSetup);

/*************************************************************************
 * Function Name: USB_TransferEndHandler
 * Parameters:  USB_Endpoint_t EP
 *
 * Return: none
 *
 * Description: USB HID report end
 *
 *************************************************************************/
void USB_TransferEndHandler (USB_Endpoint_t EP);

/*************************************************************************
 * Function Name: HidMouseSendReport
 * Parameters:  Int8S X, Int8S Y, Int8U Buttons
 *
 * Return: none
 *
 * Description: USB HID Mouse report send
 *
 *************************************************************************/
void HidMouseSendReport (Int8S X, Int8S Y, Int8U Buttons);

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
Boolean HidSendInReport (Boolean Ctrl, Int8U ID);

#endif //__HID_MOUSE_H
