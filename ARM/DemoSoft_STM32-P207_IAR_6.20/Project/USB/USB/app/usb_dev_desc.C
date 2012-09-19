/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2007
 *
 *    File name   : usb_desc.c
 *    Description : usb decriptors module
 *
 *    History :
 *    1. Date        : 22, July 2007
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#include "usb_desc.h"

const UsbStandardDeviceDescriptorStr_t UsbStandardDeviceDescriptorStr =
{
  sizeof(UsbStandardDeviceDescriptorStr_t),
  UsbDescriptorDevice,
  0x110,
  0,
  0,
  0,
  Ep0MaxSize,
  0xFFFF,
  0x0035,
  0x0100,
  iManufacturerStr,
  iProductStr,
  iSerialNumberStr,
  1,
};

#pragma data_alignment=4
const Int8U HidReportDescriptor[] =
{
  0x05, 0x01, // Usage Page (Generic Desktop)
  0x09, 0x02, // Usage (Mouse)
  0xA1, 0x01, // Collection (Application)
  0x09, 0x01, //  Usage (Pointer)
  0xA1, 0x00, //  Collection (Physical)
  0x05, 0x09, //    Usage Page (Buttons)
  0x75, 0x01, //    Report Size (1)
  0x95, 0x03, //    Report Count (3)
  0x19, 0x01, //    Usage Minimumù (01)
  0x29, 0x03, //    Usage Maximum (05)
  0x15, 0x00, //    Logical Minimum (0)
  0x25, 0x01, //    Logical Maximum (1)
  0x81, 0x02, //    Input(Data,Var,Abs)
  0x75, 0x05, //    Report Size (5)
  0x95, 0x01, //    Report Count (1)
  0x81, 0x03, //    Input(Const,Var,Abs)
  0x05, 0x01, //    Generic desktop
  0x75, 0x08, //    Report Size (8)
  0x95, 0x02, //    Report Count (2)
  0x09, 0x30, //    Usage(X)
  0x09, 0x31, //    Usage(Y)
  0x15, 0x80, //    Logical Minimum (-128)
  0x25, 0x7F, //    Logical Maximum (127)
  0x81, 0x06, //    Input(Data,Var,Rel)
  0xC0, 0xC0, // End collection x2
};

#pragma data_alignment=4
const Int8U HidDescriptor[] =
{
  sizeof(UsbHidDescriptor_t),
  Hid,
  _2BL(0x0111),
  0,
  1,
  HidReport,
  _2BL(sizeof(HidReportDescriptor)),
};

#pragma data_alignment=4
static const Int8U UsbFS_Cfg0[] =
{
  sizeof(UsbStandardConfigurationDescriptor_t),
  UsbDescriptorConfiguration,
  _2BL(sizeof(UsbStandardConfigurationDescriptor_t)+
   sizeof(UsbStandardInterfaceDescriptor_t)+
   sizeof(UsbHidDescriptor_t)+
   sizeof(UsbStandardEpDescriptor_t)),
  1,
  1,
  0,
  UsbConfigurationCommmonAttr | (USB_REMOTE_WAKEUP?UsbRemoteWakeup:0) | (USB_SELF_POWERED?0:UsbSelfPowered),
  UsbConfigPower_mA(100),

  sizeof(UsbStandardInterfaceDescriptor_t),
  UsbDescriptorInterface,
  HID_INTERFACE_0,
  0,
  1,
  UsbDeviceClassHumanInterface,
  HID_BOOT_DEVICE?BootInterfaceSubclass:NoSubclass,
  HID_BOOT_DEVICE?Mouse:0,
  0,

  sizeof(UsbHidDescriptor_t),
  Hid,
  _2BL(0x0111),
  0,
  1,
  HidReport,
  _2BL(sizeof(HidReportDescriptor)),

  sizeof(UsbStandardEpDescriptor_t),
  UsbDescriptorEp,
  UsbEpIn(ReportEpHid>>1),
  UsbEpTransferInterrupt,
  _2BL(ReportEpMaxSize),
  ReportEpPollingPeriod,
  0,
};

static const Int8U * const UsbFS_CfgArray[] =
{
  UsbFS_Cfg0,
  NULL,
};

static const UsbEP_ExtDesc_t UsbEPExt_Cfg0EP1 =
{
  1,                        // Configuration
  UsbEpIn(ReportEpHid>>1),  // EP address
  {
    0
  }
};

static const UsbEP_ExtDesc_t * const UsbFS_EPExtArray[] =
{
  &UsbEPExt_Cfg0EP1,
  NULL,
};

#pragma data_alignment=4
const Int8U UsbLanguagesStr [] =
{
  // Length of LanguagesStr + 2
  4,
  // Desciptor Type
  UsbDescriptorString,
  // Languages ID
  // Languages1 ID English
  0x09,0x04,
};

#pragma data_alignment=4
const Int8U ManufacturerStrLan1 [] =
{
  24, // length
  UsbDescriptorString,  // Descriptor
  'I',0,'A',0,'R',0,' ',0,'S',0,'y',0,'s',0,'t',0,'e',0,'m',0,'s',0
};

#pragma data_alignment=4
const Int8U ProductStrLan1 [] =
{
  36, //length
  UsbDescriptorString, // Descriptor
  'H',0,'I',0,'D',0,' ',0,'c',0,'l',0,'a',0,'s',0,'s',0,' ',0,'e',0,'x',0,'a',0,'m',0,'p',0,'l',0,'e',0,
};


#pragma data_alignment=4
const Int8U SerialNumberStrLan1 [] =
{
  18, //length
  UsbDescriptorString, // Descriptor
  '0',0,'0',0,'0',0,'0',0,'0',0,'0',0,'0',0,'1',0, // Str
};

const Int8U * const UsbLanguages1Strings[] =
{
  // iManufacturerStr
  ManufacturerStrLan1,
  // iProductStr
  ProductStrLan1,
  // iSerialNumberStr
  SerialNumberStrLan1,
  // Terminator
  NULL
};

static const Int8U * const * const UsbFS_StringArray[] =
{
  UsbLanguages1Strings,
  NULL,
};

const void * const UsbDescArray[] =
{
  UsbFS_CfgArray,
  UsbFS_EPExtArray,
  UsbLanguagesStr,
  UsbFS_StringArray,
  NULL,
};
