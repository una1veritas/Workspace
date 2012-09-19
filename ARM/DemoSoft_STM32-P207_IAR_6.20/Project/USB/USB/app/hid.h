/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2005
 *
 *    File name      : hid.h
 *    Description    : Common HID device definitions
 *
 *    History :
 *    1. Date        : December 19, 2005
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#include "arm_comm.h"

#ifndef __HID_H
#define __HID_H

typedef enum _HidRequest_t
{
  HID_GET_REPORT = 1, HID_GET_IDLE, HID_GET_PROTOCOL,
  HID_SET_REPORT = 9, HID_SET_IDLE, HID_SET_PROTOCOL,
} HidRequest_t;

typedef enum _HidReportType_t
{
  HID_INPUT = 1, HID_OUTPUT, HID_FEATURE,
} HidReportType_t;

typedef enum _HidSubclassCodes_t
{
  NoSubclass = 0, BootInterfaceSubclass
} HidSubclassCodes_t;

typedef enum _HidProtocolCodes_t
{
  None = 0, Keyboard, Mouse
} HidProtocolCodes_t;

typedef enum _HidClassDescriptorTypes_t
{
  Hid = 0x21, HidReport, HidPhysical,
} HidClassDescriptorTypes_t;

typedef enum _MouseProtocol_t
{
  BootProtocol = 0, ReportProtocol
} MouseProtocol_t;

#define BootReportMaxSize   3
/*
 Byte 0 - bit 0 - Button 1
          bit 1 - Button 2
          bit 2 - Button 3
          bit 3 - 7 - device specific
 Byte 1 - X
 Byte 2 - Y
 Byte 3 - n - Device specific
*/

#pragma pack(1)

typedef struct _UsbHidDescriptor_t
{
  Int8U       bLength;
  Int8U       bDescriptorType;
  Int8U       bcdHID[2];
  Int8U       bCountryCode;
  Int8U       bNumDescriptors;
  Int8U       bDescriptorHidType;
  Int16U      wDescriptorLength;
} UsbHidDescriptor_t, * pUsbHidDescriptor_t;
#pragma pack()

typedef enum _VendorsPages_t
{
  HID_VENDOR_PAGE_0 = 0, HID_VENDOR_PAGE_1, HID_VENDOR_PAGE_2,
  HID_FLASHLOADER_PAGE = 13,
} VendorsPages_t;


#endif //__HID_H
