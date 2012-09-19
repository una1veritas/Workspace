/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2009
 *
 *    File name   : usb_hw.c
 *    Description : usb module (HAL)
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

#define USB_HW_GLOBAL
#include "usb_hw.h"
#include "otgd_fs_cal.h"
#include "otgd_fs_pcd.h"
#include "otgd_fs_dev.h"
#include "usb_trans_layer.h"

Int32U DlyCnt;

static const UsbStandardEpDescriptor_t USB_CtrlEpDescr0 =
{
  sizeof(UsbStandardEpDescriptor_t),
  UsbDescriptorEp,
  UsbEpOut(CTRL_ENP_OUT>>1),
  {(Int8U)UsbEpTransferControl | (Int8U)UsbEpSynchNoSynchronization | (Int8U)UsbEpUsageData},
  Ep0MaxSize,
  0
};

static const UsbStandardEpDescriptor_t USB_CtrlEpDescr1 =
{
  sizeof(UsbStandardEpDescriptor_t),
  UsbDescriptorEp,
  UsbEpIn(CTRL_ENP_IN>>1),
  {(Int8U)UsbEpTransferControl | (Int8U)UsbEpSynchNoSynchronization | (Int8U)UsbEpUsageData},
  Ep0MaxSize,
  0
};

/*************************************************************************
 * Function Name: USB_HwInit
 * Parameters: none
 *
 * Return: none
 *
 * Description: Init USB
 *
 *************************************************************************/
void USB_HwInit(void)
{
NVIC_InitTypeDef NVIC_InitStructure;
GPIO_InitTypeDef GPIO_InitStructure;

  if(USB_OTG_FS == usb_module_select)
  {
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  // VBUS, ID
    GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_9 | GPIO_Pin_11 | GPIO_Pin_12; 
    GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
  
    GPIO_PinAFConfig(GPIOA, GPIO_PinSource9,  GPIO_AF_OTG1_FS);
    GPIO_PinAFConfig(GPIOA, GPIO_PinSource11, GPIO_AF_OTG1_FS);
    GPIO_PinAFConfig(GPIOA, GPIO_PinSource12, GPIO_AF_OTG1_FS);
  
    RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_OTG_FS, ENABLE) ;
    /*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_OTG_HS_ULPI, ENABLE) ;*/
    
    // Deinit
    RCC_AHB2PeriphResetCmd(RCC_AHB2Periph_OTG_FS,ENABLE);
    RCC_AHB2PeriphResetCmd(RCC_AHB2Periph_OTG_FS,DISABLE);
     
    OTGD_FS_SetAddress(USB_OTG_FS1_BASE_ADDR);
  }
  else
  {
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
    // VBUS, ID
    GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_12 | GPIO_Pin_14 | GPIO_Pin_15; 
    GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
  
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_DOWN;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
  
  
    GPIO_PinAFConfig(GPIOB, GPIO_PinSource12, GPIO_AF_OTG2_FS);
    GPIO_PinAFConfig(GPIOB, GPIO_PinSource13, GPIO_AF_OTG2_FS);
    GPIO_PinAFConfig(GPIOB, GPIO_PinSource14, GPIO_AF_OTG2_FS);
    GPIO_PinAFConfig(GPIOB, GPIO_PinSource15, GPIO_AF_OTG2_FS);
  
    // Enable USB clock for OTG HS controller
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_OTG_HS, ENABLE) ;
    /*RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_OTG_HS_ULPI, ENABLE) ;*/
    
    // Deinit
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_OTG_HS,ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_OTG_HS,DISABLE);
    /*RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_OTG_HS_ULPI,ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_OTG_HS_ULPI,DISABLE);*/
    OTGD_FS_SetAddress(USB_OTG2_HS_BASE_ADDR);
  }  
  
  OTG_DEV_Init();

  OTGD_FS_EnableDevInt();

  // Disconnect device
  USB_ConnectRes(FALSE);

  // Init controls endpoints
  USB_HwReset();

  if(USB_OTG_FS == usb_module_select)
  {
    // USB interrupt connect to NVIC
    NVIC_InitStructure.NVIC_IRQChannel = OTG_FS_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = USB_INTR_LOW_PRIORITY;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
  }
  else
  {
    // USB interrupt connect to NVIC
    NVIC_InitStructure.NVIC_IRQChannel = OTG_HS_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = USB_INTR_LOW_PRIORITY;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
    NVIC_InitStructure.NVIC_IRQChannel = OTG_HS_EP1_OUT_IRQn;
    NVIC_Init(&NVIC_InitStructure);
    NVIC_InitStructure.NVIC_IRQChannel = OTG_HS_EP1_IN_IRQn;
    NVIC_Init(&NVIC_InitStructure);
  }
  OTGD_FS_EnableGlobalInt();
}

/*************************************************************************
 * Function Name: USB_HwReset
 * Parameters: none
 *
 * Return: none
 *
 * Description: Reset USB engine
 *
 *************************************************************************/
void USB_HwReset (void)
{
Int32U Count;

  // Clear realized EP flag
  for(Count = 0; Count < ENP_MAX_NUMB; Count++)
  {
    EpCnfg[Count].MaxSize = 0;
  }
  // USB_Configure
  USB_Configure(FALSE);
  // Control EP Init
  USB_RealizeEp(&USB_CtrlEpDescr0,NULL,TRUE);
  USB_RealizeEp(&USB_CtrlEpDescr1,NULL,TRUE);
}

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
                               Boolean Enable)
{
USB_Endpoint_t EP;
pEpCnfg_t pEP;

  assert(pEP_Desc);

  EP = (USB_Endpoint_t)USB_EpLogToPhysAdd(pEP_Desc->bEndpointAddress);
  pEP = &EpCnfg[EP];

  if (Enable)
  {
    // Set EP status
    pEP->Status  = NOT_READY;
    // Init EP flags
    pEP->Flags = 0;
    // Set endpoint type
    pEP->EpType = (UsbEpTransferType_t)pEP_Desc->bmAttributes.TransferType;
    // Init EP max packet size
    pEP->MaxSize = pEP_Desc->wMaxPacketSize;
    OTGD_FS_PCD_EP_Open((EP_DESCRIPTOR *)pEP_Desc);
  }
  else
  {
    pEP->MaxSize = 0;
    OTGD_FS_PCD_EP_Close(pEP_Desc->bEndpointAddress);
  }
  return(USB_OK);
}

/*************************************************************************
 * Function Name: USB_SetAdd
 * Parameters: Int32U DevAdd - device address between 0 - 127
 *
 * Return: none
 *
 * Description: Set device address
 *
 *************************************************************************/
void USB_SetAdd(Int32U DevAdd)
{
  OTGD_FS_PCD_EP_SetAddress(DevAdd);
}

/*************************************************************************
 * Function Name: USB_ConnectRes
 * Parameters: Boolean Conn
 *
 * Return: none
 *
 * Description: Enable Pull-Up resistor
 *
 *************************************************************************/
void USB_ConnectRes (Boolean Conn)
{
  if(Conn)
  {
    USB_DevConnect();
  }
  else
  {
    USB_DevDisconnect();
  }
}

#if USB_REMOTE_WAKEUP != 0
/*************************************************************************
 * Function Name: USB_WakeUp
 * Parameters: none
 *
 * Return: none
 *
 * Description: Wake up Usb
 *
 *************************************************************************/
void USB_WakeUp (void)
{
}
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
Boolean USB_GetDevStatus (USB_DevStatusReqType_t Type)
{
  switch (Type)
  {
  case USB_DevConnectStatus:
    return(TRUE);
  }
  return(FALSE);
}

/*************************************************************************
 * Function Name: USB_SetStallEP
 * Parameters: USB_Endpoint_t EP, Boolean Stall
 *
 * Return: USB_ErrorCodes_t
 *
 * Description: The endpoint stall/unstall
 *
 *************************************************************************/
USB_ErrorCodes_t USB_SetStallEP (USB_Endpoint_t EP, Boolean Stall)
{
  if(Stall)
  {
    OTGD_FS_PCD_EP_Stall(USB_EpPhysToLogAdd(EP));
    EpCnfg[EP].Status = STALLED;
  }
  else
  {
    OTGD_FS_PCD_EP_ClrStall(USB_EpPhysToLogAdd(EP));
    EpCnfg[EP].Status = NOT_READY;
  }
  return(USB_OK);
}

/*************************************************************************
 * Function Name: USB_StallCtrlEP
 * Parameters: none
 *
 * Return: none
 *
 * Description: Stall both direction of the CTRL EP
 *
 *************************************************************************/
void USB_StallCtrlEP (void)
{
  EpCnfg[CTRL_ENP_IN].Status  = STALLED;
  EpCnfg[CTRL_ENP_OUT].Status = STALLED;
  USB_SetStallEP(CTRL_ENP_IN,TRUE);
  USB_SetStallEP(CTRL_ENP_OUT,TRUE);
}

/*************************************************************************
 * Function Name: USB_GetStallEP
 * Parameters: USB_Endpoint_t EP, pBoolean pStall
 *
 * Return: USB_ErrorCodes_t
 *
 * Description: Get stall state of the endpoint
 *
 *************************************************************************/
USB_ErrorCodes_t USB_GetStallEP (USB_Endpoint_t EP, pBoolean pStall)
{
  if (!USB_EP_VALID(&EpCnfg[EP]))
  {
    return(USB_EP_NOT_VALID);
  }
  *pStall = STALLED == EpCnfg[EP].Status;
  return (USB_OK);
}

/*************************************************************************
 * Function Name: USB_EP_IO
 * Parameters: USB_Endpoint_t EndPoint
 *
 * Return: none
 *
 * Description: Endpoints IO
 *
 *************************************************************************/
void USB_EP_IO(USB_Endpoint_t EP)
{
pEpCnfg_t pEP = &EpCnfg[EP];
assert(USB_EP_VALID(pEP));

  if(pEP->Status != NO_SERVICED)
  {
    return;
  }

  if(EP & 1)
  {
    // IN
    // Set Status
    pEP->Status = BEGIN_SERVICED;
    OTGD_FS_PCD_EP_Write ( USB_EpPhysToLogAdd(EP),
                           pEP->pBuffer,
                           pEP->Size);
  }
  else
  {
    // OUT
    pEP->Status = BEGIN_SERVICED;
    OTGD_FS_PCD_EP_Read ( USB_EpPhysToLogAdd(EP),
                          pEP->pBuffer,
                          pEP->Size);
  }
}

/*************************************************************************
 * Function Name: USB_EpLogToPhysAdd
 * Parameters: Int8U EpLogAdd
 *
 * Return: USB_Endpoint_t
 *
 * Description: Convert the logical to physical address
 *
 *************************************************************************/
USB_Endpoint_t USB_EpLogToPhysAdd (Int8U EpLogAdd)
{
USB_Endpoint_t Address = (USB_Endpoint_t)((EpLogAdd & 0x0F)<<1);
  if(EpLogAdd & 0x80)
  {
    ++Address;
  }
  return(Address);
}

/*************************************************************************
 * Function Name: USB_EpPhysToLogAdd
 * Parameters: USB_Endpoint_t EP
 *
 * Return: Int8U
 *
 * Description: Convert physical to logical address
 *
 *************************************************************************/
Int8U USB_EpPhysToLogAdd (USB_Endpoint_t EP)
{
Int8U Addr = EP >> 1;
  if(EP & 1)
  {
    Addr |= 0x80;
  }
  return(Addr);
}

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
Int32U USB_GetFrameNumb (void)
{
  return(0);
}
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
void USB_StatusPhase (Boolean In)
{
  if(In)
  {
    if(UsbEp0SetupPacket.bRequest == SET_ADDRESS)
    {
      USB_SetAdd(UsbEp0SetupPacket.wValue.Lo);
    }
    USB_IO_Data(CTRL_ENP_IN,NULL,0,NULL);
  }
  else
  {
    USB_IO_Data(CTRL_ENP_OUT,NULL,0,NULL);
  }
}
