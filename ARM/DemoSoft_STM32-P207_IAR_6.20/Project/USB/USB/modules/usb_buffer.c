/*************************************************************************
 *
 *    Used with ARM IAR C/C++ Compiler.
 *
 *    (c) Copyright IAR Systems 2007
 *
 *    File name      : usb_buffer.c
 *    Description    : USB buffer manager module
 *
 *    History :
 *    1. Date        : June 23, 2007
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/
#define USB_BUFFER_GLOBAL
#include "usb_buffer.h"

/*************************************************************************
 * Function Name: USB_IO_Data
 * Parameters: USB_Endpoint_t EP, pInt8U pBuffer, Int32U Size, void * pFn
 *
 * Return: USB_IO_Status_t
 *
 * Description: Prepare and send
 *
 *************************************************************************/
USB_IO_Status_t USB_IO_Data (USB_Endpoint_t EP, pInt8U pBuffer, Int32U Size, void * pFn)
{
#if __CORE__ < 7
Int32U Save;
#endif // __CORE__ < 7

pEpCnfg_t pEP = &EpCnfg[EP];

  if (Size == (Int32U)-1)
  {
    pEP->Status  = NOT_READY;
    pEP->pFn     = NULL;
  }
  else
  {
  #if __CORE__ < 7
    ENTR_CRT_SECTION(Save);
  #else
    ENTR_CRT_SECTION();
  #endif // __CORE__ < 7
    if (!USB_EP_VALID(pEP))
    {
    #if __CORE__ < 7
      EXT_CRT_SECTION(Save);
    #else
      EXT_CRT_SECTION();
    #endif // __CORE__ < 7
      return(NOT_VALID);
    }
    // lock buffer
    if(pEP->Status == BEGIN_SERVICED)
    {
    #if __CORE__ < 7
      EXT_CRT_SECTION(Save);
    #else
      EXT_CRT_SECTION();
    #endif // __CORE__ < 7
      return(NOT_READY);
    }
    pEP->Offset  = 0;
    pEP->pBuffer = pBuffer;
    pEP->pFn     = pFn;
    if(!(pEP->Size = Size))
    {
      pEP->bZeroPacket = 1;
    }
    pEP->Status  = NO_SERVICED;
    USB_EP_IO(EP);
  #if __CORE__ < 7
    EXT_CRT_SECTION(Save);
  #else
    EXT_CRT_SECTION();
  #endif // __CORE__ < 7
  }
  return(pEP->Status);
}
