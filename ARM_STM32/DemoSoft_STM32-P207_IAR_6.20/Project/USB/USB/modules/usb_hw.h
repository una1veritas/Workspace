/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2009
 *
 *    File name   : STM32F_usb.h
 *    Description : usb module include file
 *
 *    History :
 *    1. Date        : July 28, 2006
 *       Author      : Stanimir Bonev
 *       Description : Create
 *    2. Date        : November 20, 2006
 *       Author      : Stanimir Bonev
 *       Description : Modify
 *      Fix problems with double buffered EPs
 *    3. Date        : April 30, 2007
 *       Author      : Stanimir Bonev
 *       Description : Modify
 *      Adapt for STM32F
 *    4. Date        : July 20, 2007
 *       Author      : Stanimir Bonev
 *       Description : Modify
 *      Adapt for USB framework 2
 *    5. Date        : July 04, 2008
 *       Author      : Stanimir Bonev
 *       Description : Modify
 *      Adapt for STM32F103VE-SK
 *    6. Date        : August 15, 2009
 *       Author      : Stanimir Bonev
 *       Description : Modify
 *      Adapt for STM32F107VC-SK
 *
 *    $Revision: #2 $
 **************************************************************************/

#include "includes.h"

#ifndef __USB_HW_H
#define __USB_HW_H

#ifdef USB_HW_GLOBAL
#define USB_HW_EXTERN
#else
#define USB_HW_EXTERN  extern
#endif

#include "io_macros.h"

typedef enum _USB_PacketType_t
{
	// Packet type parameters
  UsbSetupPacket = 0,UsbDataOutPacket,UsbDataInPacket,UsbDmaPacket,
} USB_PacketType_t;

typedef enum _USB_ErrorCodes_t
{
  USB_OK = 0,USB_PLL_ERROR, USB_INTR_ERROR,
  USB_EP_OCCUPIER, USB_MEMORY_FULL, USB_BUF_OVERFLOW,
  USB_EP_NOT_VALID, UB_EP_SETUP_UNDERRUN, USB_EP_STALLED,
  UB_EP_SETUP_OVERWRITE, USB_EP_FATAL_ERROR,
} USB_ErrorCodes_t;

typedef enum _EpType_t
{
  EP_BULK = 0, EP_CTRL, EP_ISO,
  EP_INTERRUPT,
} EpType_t;

typedef enum _EpSlot_t
{
  EP_SLOT0 = 0, EP_SLOT1, EP_SLOT2, EP_SLOT3, EP_SLOT4,
  EP_SLOT5, EP_SLOT6, EP_SLOT7, EP_MAX_SLOTS
} EpSlot_t;

typedef enum _EpState_t
{
  EP_DISABLED = 0, EP_STALL, EP_NAK, EP_VALID
} EpState_t;

typedef enum _USB_Endpoint_t
{
  CTRL_ENP_OUT=0, CTRL_ENP_IN,
  ENP1_OUT      , ENP1_IN    ,
  ENP2_OUT      , ENP2_IN    ,
  ENP3_OUT      , ENP3_IN    ,
  ENP_MAX_NUMB
} USB_Endpoint_t;

typedef enum _USB_DevStatusReqType_t
{
  USB_DevConnectStatus = 0, USB_SuspendStatus, USB_ResetStatus
} USB_DevStatusReqType_t;

typedef enum _UsbResumeEvent_t
{
  USB_RESUME_SOF_EVENT = 0, USB_RESUME_SOFT_EVENT,
  USB_RESUME_WAKE_UP_EVENT,
} UsbResumeEvent_t, *pUsbResumeEvent_t;

typedef union _RxCount_t
{
  Int16U Count;
  struct
  {
    Int16U CountField     : 10;
    Int16U NubBlockField  :  5;
    Int16U BlSizeField    :  1;
  };
} RxCount_t, *pRxCount_t;

typedef struct _PacketMemUse_t
{
  USB_Endpoint_t RpAddr;
  Int16U         Start;
  Int16U         Size;
  struct _PacketMemUse_t * pNext;
} PacketMemUse_t, *pPacketMemUse_t;

typedef enum _USB_IO_Status_t
{
  NOT_READY = 0, NO_SERVICED, BEGIN_SERVICED, COMPLETE, BUFFER_UNDERRUN, BUFFER_OVERRUN,
  SETUP_OVERWRITE, STALLED, NOT_VALID
} USB_IO_Status_t;

typedef struct _UsbEP_ExtData_t
{
  Int32U Dummy;
} UsbEP_ExtData_t, *pUsbEP_ExtData_t;

typedef struct _EpCnfg_t
{
  Int32U              MaxSize;
  UsbEpTransferType_t EpType;
  void *              pFn;
  Int32U              Offset;
  Int32U              Size;
  USB_IO_Status_t     Status;
  pInt8U              pBuffer;
  union
  {
    Int8U Flags;
    struct
    {
      Int8U bZeroPacket         : 1;
      Int8U bZeroPacketPossible : 1;
    };
  };
} EpCnfg_t, *pEpCnfg_t;

#pragma pack(1)
typedef struct _USB_BuffDeskTbl_t
{
  Int16U    AddrTx;
  Int16U    CountTx;
  Int16U    AddrRx;
  RxCount_t CountRx;
} USB_BuffDeskTbl_t, *pUSB_BuffDeskTbl_t;
#pragma pack()

typedef union _pIntrStatus_t
{
  Int32U Status;
  struct {
    Int32U EP_ID  : 4;
    Int32U DIR    : 1;
    Int32U        : 2;
    Int32U SZDPR  : 1;
    Int32U ESOF   : 1;
    Int32U SOF    : 1;
    Int32U RESET  : 1;
    Int32U SUSP   : 1;
    Int32U WKUP   : 1;
    Int32U ERR    : 1;
    Int32U PMAOVR : 1;
    Int32U CTR    : 1;
    Int32U        :16;
  };
} IntrStatus_t, *pIntrStatus_t;

typedef Int32U UsbDefStatus_t;
typedef void * (* UserFunc_t)(void * Arg);

#define bmCTRM                      0x8000
#define bmPMAOVRM                   0x4000
#define bmERRM                      0x2000
#define bmWKUPM                     0x1000
#define bmSUSPM                     0x0800
#define bmRESETM                    0x0400
#define bmSOFM                      0x0200
#define bmESOFM                     0x0100

#define USB_EP_VALID(pEP)           ((pEP)->MaxSize)

#define OUT_PACKET                  2
#define SETUP_PACKET_COMPLETE       4
#define SETUP_PACKET                6

#define EP2EPI_OUT(EP)              (USB_Endpoint_t)( EP<<1     )
#define EP2EPI_IN(EP)               (USB_Endpoint_t)((EP<<1) | 1)

// Do not change it
#define Ep0MaxSize                      64

USB_HW_EXTERN EpCnfg_t EpCnfg[ENP_MAX_NUMB];

typedef enum
{
  USB_OTG_FS = 0,
  USB_OTG_HS
} USB_select_module_t;

USB_HW_EXTERN USB_select_module_t usb_module_select;

/*************************************************************************
 * Function Name: USB_HwInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: Init USB
 *
 *************************************************************************/
void USB_HwInit(void);

/*************************************************************************
 * Function Name: USB_HwReset
 * Parameters: none
 *
 * Return: none
 *
 * Description: Reset USB engine
 *
 *************************************************************************/
void USB_HwReset (void);

/*************************************************************************
 * Function Name: USB_RealizeEp
 * Parameters: const UsbStandardEpDescriptor_t * pEP_Desc,
 *             const UsbEP_ExtData_t * pUsbEP_ExtData, Boolean Enable
 *
 * Return: USB_ErrorCodes_t
 *
 * Description: Enable or disable an endpoint
 *
 *************************************************************************/
USB_ErrorCodes_t USB_RealizeEp(const UsbStandardEpDescriptor_t * pEP_Desc,
                               const UsbEP_ExtData_t * pUsbEP_ExtData,
                               Boolean Enable);

/*************************************************************************
 * Function Name: USB_SetAdd
 * Parameters: Int32U DevAdd - device address between 0 - 127
 *
 * Return: none
 *
 * Description: Set device address
 *
 *************************************************************************/
void USB_SetAdd(Int32U DevAdd);
#define USB_SetDefAdd() USB_SetAdd(0)

/*************************************************************************
 * Function Name: USB_ConnectRes
 * Parameters: Boolean Conn
 *
 * Return: none
 *
 * Description: Enable Pull-Up resistor
 *
 *************************************************************************/
void USB_ConnectRes (Boolean Conn);

/*************************************************************************
 * Function Name: USB_Configure
 * Parameters: Boolean Configure
 *
 * Return: none
 *
 * Description: Configure device
 *  When Configure != 0 enable all Realize Ep
 *
 *************************************************************************/
inline
void USB_Configure (Boolean Configure)
{}

#if USB_REMOTE_WAKEUP != 0
/*************************************************************************
 * Function Name: USB_WakeUp
 * Parameters: none
 *
 * Return: none
 *
 * Description: Wake up USB
 *
 *************************************************************************/
void USB_WakeUp (void);
#endif // USB_REMOTE_WAKEUP != 0

/*************************************************************************
 * Function Name: USB_GetDevStatus
 * Parameters: USB_DevStatusReqType_t Type
 *
 * Return: Boolean
 *
 * Description: Return USB device status
 *
 *************************************************************************/
Boolean USB_GetDevStatus (USB_DevStatusReqType_t Type);

/*************************************************************************
 * Function Name: USB_EpLogToPhysAdd
 * Parameters: Int8U EpLogAdd
 *
 * Return: USB_Endpoint_t
 *
 * Description: Convert the logical to physical address
 *
 *************************************************************************/
USB_Endpoint_t USB_EpLogToPhysAdd (Int8U EpLogAdd);

/*************************************************************************
 * Function Name: USB_EpPhysToLogAdd
 * Parameters: USB_Endpoint_t EP
 *
 * Return: Int8U
 *
 * Description: Convert physical to logical address
 *
 *************************************************************************/
Int8U USB_EpPhysToLogAdd (USB_Endpoint_t EP);

#if USB_SOF_EVENT > 0
/*************************************************************************
 * Function Name: USB_GetFrameNumb
 * Parameters: none
 *
 * Return: Int32U
 *
 * Description: Return current value of SOF number
 *
 *************************************************************************/
Int32U USB_GetFrameNumb (void);
#endif // USB_SOF_EVENT > 0

/*************************************************************************
 * Function Name: USB_StatusPhase
 * Parameters: Boolean In
 *
 * Return: none
 *
 * Description: Prepare status phase
 *
 *************************************************************************/
void USB_StatusPhase (Boolean In);

/*************************************************************************
 * Function Name: USB_SetStallEP
 * Parameters: USB_Endpoint_t EndPoint, Boolean Stall
 *
 * Return: USB_ErrorCodes_t
 *
 * Description: The endpoint stall/unstall
 *
 *************************************************************************/
USB_ErrorCodes_t USB_SetStallEP (USB_Endpoint_t EndPoint, Boolean Stall);

/*************************************************************************
 * Function Name: USB_StallCtrlEP
 * Parameters: none
 *
 * Return: none
 *
 * Description: Stall both direction of the CTRL EP
 *
 *************************************************************************/
void USB_StallCtrlEP (void);

/*************************************************************************
 * Function Name: USB_GetStallEP
 * Parameters: USB_Endpoint_t EndPoint, pBoolean pStall
 *
 * Return: USB_ErrorCodes_t
 *
 * Description: Get stall state of the endpoint
 *
 *************************************************************************/
USB_ErrorCodes_t USB_GetStallEP (USB_Endpoint_t EndPoint, pBoolean pStall);

/*************************************************************************
 * Function Name: USB_EP_IO
 * Parameters: USB_Endpoint_t EndPoint
 *
 * Return: none
 *
 * Description: Endpoints IO
 *
 *************************************************************************/
void USB_EP_IO(USB_Endpoint_t EP);

#endif //__STM32F_USB_H
