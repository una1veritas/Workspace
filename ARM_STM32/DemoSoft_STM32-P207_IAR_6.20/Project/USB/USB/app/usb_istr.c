/******************** (C) COPYRIGHT 2009 STMicroelectronics ********************
* File Name          : usb_istr.c
* Author             : MCD Application Team
* Version            : V3.1.0
* Date               : 10/30/2009
* Description        : ISTR events interrupt service routines
********************************************************************************
* THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
* WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE TIME.
* AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY DIRECT,
* INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING FROM THE
* CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE CODING
* INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
*******************************************************************************/

/* Includes ------------------------------------------------------------------*/
#include "usb_istr.h"
#include "otgd_fs_regs.h"
#include "otgd_fs_cal.h"
#include "otgd_fs_int.h"


/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
volatile unsigned short wIstr;  /* ISTR register last read value */
volatile unsigned char bIntPackSOF = 0;  /* SOFs received between 2 consecutive packets */

/*******************************************************************************
* Function Name  : STM32_PCD_OTG_ISR_Handler
* Description    : Handles all USB Device Interrupts
* Input          : None
* Output         : None
* Return         : status
*******************************************************************************/
unsigned int  STM32_PCD_OTG_ISR_Handler (void)
{
  USB_OTG_int_sts_data gintr_status;
  u32 retval = 0;

  if (IsDeviceMode()) /* ensure that we are in device mode */
  {
    gintr_status.d32 = OTGD_FS_ReadCoreItr();

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/

    /* If there is no interrupt pending exit the interrupt routine */
    if (!gintr_status.d32)
    {
      return 0;
    }

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Early Suspend interrupt */
#ifdef INTR_ERLYSUSPEND
    if (gintr_status.b.erlysuspend)
    {
      retval |= OTGD_FS_Handle_EarlySuspend_ISR();
    }
#endif /* INTR_ERLYSUSPEND */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* End of Periodic Frame interrupt */
#ifdef INTR_EOPFRAME
    if (gintr_status.b.eopframe)
    {
      retval |= OTGD_FS_Handle_EOPF_ISR();
    }
#endif /* INTR_EOPFRAME */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Non Periodic Tx FIFO Emty interrupt */
#ifdef INTR_NPTXFEMPTY
    if (gintr_status.b.nptxfempty)
    {
      retval |= OTGD_FS_Handle_NPTxFE_ISR();
    }
#endif /* INTR_NPTXFEMPTY */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Wakeup or RemoteWakeup interrupt */
#ifdef INTR_WKUPINTR
    if (gintr_status.b.wkupintr)
    {
      retval |= OTGD_FS_Handle_Wakeup_ISR();
    }
#endif /* INTR_WKUPINTR */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Suspend interrupt */
#ifdef INTR_USBSUSPEND
    if (gintr_status.b.usbsuspend)
    {
      retval |= OTGD_FS_Handle_USBSuspend_ISR();
    }
#endif /* INTR_USBSUSPEND */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Start of Frame interrupt */
#ifdef INTR_SOFINTR
    if (gintr_status.b.sofintr)
    {
      /* Update the frame number variable */
      bIntPackSOF++;

      retval |= OTGD_FS_Handle_Sof_ISR();
    }
#endif /* INTR_SOFINTR */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Receive FIFO Queue Status Level interrupt */
#ifdef INTR_RXSTSQLVL
    if (gintr_status.b.rxstsqlvl)
    {
      retval |= OTGD_FS_Handle_RxStatusQueueLevel_ISR();
    }
#endif /* INTR_RXSTSQLVL */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Enumeration Done interrupt */
#ifdef INTR_ENUMDONE
    if (gintr_status.b.enumdone)
    {
      retval |= OTGD_FS_Handle_EnumDone_ISR();
    }
#endif /* INTR_ENUMDONE */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Reset interrutp */
#ifdef INTR_USBRESET
    if (gintr_status.b.usbreset)
    {
      retval |=OTGD_FS_Handle_UsbReset_ISR();
    }
#endif /* INTR_USBRESET */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* IN Endpoint interrupt */
#ifdef INTR_INEPINTR
    if (gintr_status.b.inepint)
    {
      retval |= OTGD_FS_Handle_InEP_ISR();
    }
#endif /* INTR_INEPINTR */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* OUT Endpoint interrupt */
#ifdef INTR_OUTEPINTR
    if (gintr_status.b.outepintr)
    {
      retval |= OTGD_FS_Handle_OutEP_ISR();
    }
#endif /* INTR_OUTEPINTR */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Mode Mismatch interrupt */
#ifdef INTR_MODEMISMATCH
    if (gintr_status.b.modemismatch)
    {
      retval |= OTGD_FS_Handle_ModeMismatch_ISR();
    }
#endif /* INTR_MODEMISMATCH */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Global IN Endpoints NAK Effective interrupt */
#ifdef INTR_GINNAKEFF
    if (gintr_status.b.ginnakeff)
    {
      retval |= OTGD_FS_Handle_GInNakEff_ISR();
    }
#endif /* INTR_GINNAKEFF */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Global OUT Endpoints NAK effective interrupt */
#ifdef INTR_GOUTNAKEFF
    if (gintr_status.b.goutnakeff)
    {
      retval |= OTGD_FS_Handle_GOutNakEff_ISR();
    }
#endif /* INTR_GOUTNAKEFF */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Isochrounous Out packet Dropped interrupt */
#ifdef INTR_ISOOUTDROP
    if (gintr_status.b.isooutdrop)
    {
      retval |= OTGD_FS_Handle_IsoOutDrop_ISR();
    }
#endif /* INTR_ISOOUTDROP */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Endpoint Mismatch error interrupt */
#ifdef INTR_EPMISMATCH
    if (gintr_status.b.epmismatch)
    {
      retval |= OTGD_FS_Handle_EPMismatch_ISR();
    }
#endif /* INTR_EPMISMATCH */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Incomplete Isochrous IN tranfer error interrupt */
#ifdef INTR_INCOMPLISOIN
    if (gintr_status.b.incomplisoin)
    {
      retval |= OTGD_FS_Handle_IncomplIsoIn_ISR();
    }
#endif /* INTR_INCOMPLISOIN */

   /*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/
    /* Incomplete Isochrous OUT tranfer error interrupt */
#ifdef INTR_INCOMPLISOOUT
    if (gintr_status.b.outepintr)
    {
      retval |= OTGD_FS_Handle_IncomplIsoOut_ISR();
    }
#endif /* INTR_INCOMPLISOOUT */

  }
  return retval;
}

/******************* (C) COPYRIGHT 2009 STMicroelectronics *****END OF FILE****/

