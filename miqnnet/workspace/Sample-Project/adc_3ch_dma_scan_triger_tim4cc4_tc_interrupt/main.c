/**
  ******************************************************************************
  * @file    adc_3ch_dma_scan_triger_tim4cc4_tc_interrupt/main.c
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
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "ADC Continuous scan mode using DMA transfer.\r\n"
  "Type anything. I report ADC Ch1/8/9 converted value in decimal.\r\n";
__IO int16_t ADCConvertedValue[3], Ex_Value[3];
__IO uint16_t ARR_Val = 20;
/* Private function prototypes -----------------------------------------------*/
void ADC_DMA_Configuration(void);
void NVIC_Configuration(void);
void TIM4_Configuration(void);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{

  // Configure board specific setting
  BoardInit();
  // Setting NVIC
  NVIC_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();
  //Configure TIM4 : Timer must be configured before ADC and DMA configuration.
  TIM4_Configuration();
  //Configure ADC
  ADC_DMA_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  while (1){}
}

/**
  * @brief  Configure ADC1 with DMA1
  * @param  None
  * @retval : None
  */
void ADC_DMA_Configuration(void)
{
  ADC_InitTypeDef  ADC_InitStructure;
  DMA_InitTypeDef DMA_InitStructure;
  GPIO_InitTypeDef GPIO_InitStructure;

  /* ADCCLK = PCLK2/8 = 72/8 = 9MHz*/
  RCC_ADCCLKConfig(RCC_PCLK2_Div8);
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | ADC_IN0_3_GPIO_RCC | ADC_IN8_9_GPIO_RCC, ENABLE);

  GPIO_InitStructure.GPIO_Pin = ADC123_IN1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC123_IN1_PORT, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = ADC12_IN8_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC12_IN8_PORT, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = ADC12_IN9_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC12_IN9_PORT, &GPIO_InitStructure);

  /* DMA1 channel1 configuration ----------------------------------------------*/
  DMA_DeInit(DMA1_Channel1);
  DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&ADC1->DR;
  DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)&ADCConvertedValue;
  DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;
  DMA_InitStructure.DMA_BufferSize = 3;
  DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
  DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
  DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
  DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
  DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
  DMA_InitStructure.DMA_Priority = DMA_Priority_High;
  DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
  DMA_Init(DMA1_Channel1, &DMA_InitStructure);

  /* Enable DMA1 Channel1 Transfer Complete interrupt */
  DMA_ITConfig(DMA1_Channel1, DMA_IT_TC, ENABLE);

  /* Enable DMA1 channel1 */
  DMA_Cmd(DMA1_Channel1, ENABLE);

  ADC_DeInit(ADC1);

  /* ADC1 Configuration ------------------------------------------------------*/
  ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
  ADC_InitStructure.ADC_ScanConvMode = ENABLE;
  ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;
  ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_T4_CC4;
  ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
  ADC_InitStructure.ADC_NbrOfChannel = 3;
  ADC_Init(ADC1, &ADC_InitStructure);

  ADC_RegularChannelConfig(ADC1, ADC123_IN1_CH, 1, ADC_SampleTime_239Cycles5);
  ADC_RegularChannelConfig(ADC1, ADC12_IN8_CH, 2, ADC_SampleTime_239Cycles5);
  ADC_RegularChannelConfig(ADC1, ADC12_IN9_CH, 3, ADC_SampleTime_239Cycles5);

  ADC_DMACmd(ADC1, ENABLE);

  ADC_ExternalTrigConvCmd (ADC1, ENABLE);

  ADC_Cmd(ADC1, ENABLE);

  /* Enable ADC1 reset calibration register */
  ADC_ResetCalibration(ADC1);
  while(ADC_GetResetCalibrationStatus(ADC1));
  /* Start ADC1 calibration */
  ADC_StartCalibration(ADC1);
  while(ADC_GetCalibrationStatus(ADC1));

}

/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable DMA1 channel1 IRQ Channel */
  NVIC_InitStructure.NVIC_IRQChannel = DMA1_Channel1_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  This function handles DMA1 Channel 1 interrupt request.
  * @param  None
  * @retval None
  */
void DMA1_Channel1_IRQHandler(void)
{
  /* Test on DMA1 Channel1 Transfer Complete interrupt */
  if(DMA_GetITStatus(DMA1_IT_TC1))
  {

    if(
        ADCConvertedValue[0] > Ex_Value[0] + 100 || ADCConvertedValue[0] < Ex_Value[0] - 100 ||
        ADCConvertedValue[1] > Ex_Value[1] + 100 || ADCConvertedValue[1] < Ex_Value[1] - 100 ||
        ADCConvertedValue[2] > Ex_Value[2] + 100 || ADCConvertedValue[2] < Ex_Value[2] - 100
        )
      {
        cprintf("Converted result (ch1:8:9) is %d : %d : %d\r\n", ADCConvertedValue[0], ADCConvertedValue[1], ADCConvertedValue[2]);

        Ex_Value[0] = ADCConvertedValue[0];
        Ex_Value[1] = ADCConvertedValue[1];
        Ex_Value[2] = ADCConvertedValue[2];

      }
    /* Clear DMA1 Channel1 Transfer Complete interrupt pending bits */
    DMA_ClearITPendingBit(DMA1_IT_TC1);
  }
}

/**
  * @brief  Configure TIM4
  * @param  None
  * @retval : None
  */
void TIM4_Configuration(void)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM4_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM4 Configuration: Output Compare Toggle Mode:
    TIM4CLK = 72 MHz, Prescaler = 36000, TIM4 counter clock = 2kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = ARR_Val;
  TIM_TimeBaseStructure.TIM_Prescaler = 36000;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel4 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_Pulse = 1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OC4Init(TIM4, &TIM_OCInitStructure);
  TIM_OC4PreloadConfig(TIM4, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM4, ENABLE);

}
