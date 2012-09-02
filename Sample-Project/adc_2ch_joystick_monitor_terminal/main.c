/**
  ******************************************************************************
  * @file    adc_2ch_joystick_monitor_terminal/main.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   Main program body
  ******************************************************************************
  * @copy
  *
  * Copyright 2008-2009 Yasuo Kawachi All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *  1. Redistributions of source code must retain the above copyright notice,
  *  this list of conditions and the following disclaimer.
  *  2. Redistributions in binary form must reproduce the above copyright notice,
  *  this list of conditions and the following disclaimer in the documentation
  *  and/or other materials provided with the distribution.
  *
  * THIS SOFTWARE IS PROVIDED BY YASUO KAWACHI "AS IS" AND ANY EXPRESS OR IMPLIE  D
  * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
  * EVENT SHALL YASUO KAWACHI OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  * This software way contain the part of STMicroelectronics firmware.
  * Below notice is applied to the part, but the above BSD license is not.
  *
  * THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
  * WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
  * TIME. AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY
  * DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING
  * FROM THE CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE
  * CODING INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
  *
  * COPYRIGHT 2009 STMicroelectronics
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "platform_config.h"
#include "com_config.h"
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "Tilt joy-sticks. Position will be displayed in realtime.\r\n";
uint8_t RxData;
uint16_t Ex_Param[2], Present_Param[2];
/* Private function prototypes -----------------------------------------------*/
uint16_t ReadADC1(uint8_t channel);
void ADC_Configuration(void);
void Terminal_Back_Space(uint16_t num);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint8_t i;


  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();
  //Configure ADC
  ADC_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  while (1)
    {
      Present_Param[0] = ReadADC1(ADC12_IN8_CH) / 190;
      Present_Param[1] = ReadADC1(ADC12_IN9_CH) / 190;

      if(Present_Param[0] != Ex_Param[0] || Present_Param[1] != Ex_Param[1] )
        {
          Terminal_Back_Space(47);
          RxData = (int8_t)RECEIVE_DATA;
          cprintf("H:");
          for(i=0;i<Present_Param[0];i++)
            {
              cputchar('*');
            }
          for(i=0;i<(21 - Present_Param[0]);i++)
            {
              cputchar(' ');
            }
          cprintf(" V:");
          for(i=0;i<Present_Param[1];i++)
            {
              cputchar('*');
            }
          for(i=0;i<(21 - Present_Param[1]);i++)
            {
              cputchar(' ');
            }
        }

      Ex_Param[0] = Present_Param[0];
      Ex_Param[1] = Present_Param[1];

      delay_ms(10);
    }
}

/**
  * @brief  Send back space to terminal
  * @param  num : number of back space to be send
  * @retval : None
  */
void Terminal_Back_Space(uint16_t num)
{
  uint16_t i;
  for (i=0;i<num;i++)
    {
      cputchar(0x08);
      cputchar(0x20);
      cputchar(0x08);
    }
}

/**
  * @brief  Configure ADC1
  * @param  None
  * @retval : None
  */
void ADC_Configuration(void)
{
  ADC_InitTypeDef  ADC_InitStructure;
  GPIO_InitTypeDef GPIO_InitStructure;

  /* ADCCLK = PCLK2/8 = 72/8 = 9MHz*/
  RCC_ADCCLKConfig(RCC_PCLK2_Div8);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | ADC_IN8_9_GPIO_RCC, ENABLE);

  GPIO_InitStructure.GPIO_Pin = ADC12_IN8_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC12_IN8_PORT, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = ADC12_IN9_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC12_IN9_PORT, &GPIO_InitStructure);

  ADC_DeInit(ADC1);

  /* ADC1 Configuration ------------------------------------------------------*/
  ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
  ADC_InitStructure.ADC_ScanConvMode = DISABLE;
  ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;
  ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
  ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
  ADC_InitStructure.ADC_NbrOfChannel = 1;

  ADC_Init(ADC1, &ADC_InitStructure);
  ADC_Cmd(ADC1, ENABLE);

  /* Enable ADC1 reset calibration register */
  ADC_ResetCalibration(ADC1);
  while(ADC_GetResetCalibrationStatus(ADC1));
  /* Start ADC1 calibration */
  ADC_StartCalibration(ADC1);
  while(ADC_GetCalibrationStatus(ADC1));
}

/**
  * @brief  return ADC value
  * @param  channel : designate ADC channel. Macro Available ex.ADC_Channel_8
  * @retval ADC converted value
  */
uint16_t ReadADC1(uint8_t channel)
{
  ADC_RegularChannelConfig(ADC1, channel, 1, ADC_SampleTime_13Cycles5);
  // Start the conversion
  ADC_SoftwareStartConvCmd(ADC1, ENABLE);
  // Wait until conversion completion
  while(ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC) == RESET);
  // Get the conversion value
  return ADC_GetConversionValue(ADC1);
}

