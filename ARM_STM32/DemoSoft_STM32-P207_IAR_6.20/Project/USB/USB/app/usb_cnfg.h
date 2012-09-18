/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2007
 *
 *    File name   : usb_cnfg.h
 *    Description : USB config file
 *
 *    History :
 *    1. Date        : June 16, 2007
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/

#include <includes.h>

#ifndef __USB_CNFG_H
#define __USB_CNFG_H

/* The Board */
#define IAR_STM32F107VC_SK

/* USB High Speed support*/
#define USB_HIGH_SPEED                  0

/* USB interrupt priority */
#define USB_INTR_LOW_PRIORITY           2

/* USB Events */
#define USB_SOF_EVENT                   1

/* USB Clock settings */
#define USB_DIVIDER                     RCC_OTGFSCLKSource_PLLVCO_Div3 // when PLL clk 72MHz

/* Device power atrb  */
#define USB_SELF_POWERED                0
#define USB_REMOTE_WAKEUP               0

/* Device Address handling */
#define PRE_DEV_ADDR_SET                1

/* Max Interfaces number*/
#define USB_MAX_INTERFACE               1

/* Endpoint definitions */
// Do not change it
#define Ep0MaxSize                      64

#define ReportEpHid                     ENP1_IN
#define ReportEpMaxSize                 3
#define ReportEpPollingPeriod           2   // resolution 1ms

/* Class defenitions*/
#define HID_INTERFACE_0                 0
#define HID_BOOT_DEVICE                 1
#define HID_IDLE_SUPP                   1
#define HID_ID_NUMB                     0

#define HID_MOUSE_ID                    0
/* Other defenitions */

#endif //__USB_CNFG_H
