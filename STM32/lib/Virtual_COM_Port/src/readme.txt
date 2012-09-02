hw_config.c must be modified for making STBee mini USB port work.
Configuration for reversed polarity required.

void Set_System(void)
{
#ifndef USE_STM3210C_EVAL
  GPIO_InitTypeDef GPIO_InitStructure;
#endif /* USE_STM3210C_EVAL */

  /* SYSCLK, HCLK, PCLK2 and PCLK1 configuration -----------------------------*/
  /* deleted 20100513 yasuokawachi -----------------------------*/


#ifndef USE_STM3210C_EVAL
  /* Enable USB_DISCONNECT GPIO clock */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIO_DISCONNECT, ENABLE);

  /* Configure USB pull-up pin */
  GPIO_InitStructure.GPIO_Pin = USB_DISCONNECT_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
#ifndef USB_DISCONNECT_POLARITY_REVERSE
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
#else
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
#endif
  GPIO_Init(USB_DISCONNECT, &GPIO_InitStructure);
#endif /* USE_STM3210C_EVAL */
}

void USB_Cable_Config (FunctionalState NewState)
{
#ifdef USE_STM3210C_EVAL  
  if (NewState != DISABLE)
  {
    USB_DevConnect();
  }
  else
  {
    USB_DevDisconnect();
  }
#else /* USE_STM3210B_EVAL or USE_STM3210E_EVAL */
  if (NewState != DISABLE)
  {
#ifndef USB_DISCONNECT_POLARITY_REVERSE
    GPIO_ResetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#else
    GPIO_SetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#endif
  }
  else
  {
#ifndef USB_DISCONNECT_POLARITY_REVERSE
    GPIO_SetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#else
    GPIO_ResetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#endif
  }
#endif /* USE_STM3210C_EVAL */
}