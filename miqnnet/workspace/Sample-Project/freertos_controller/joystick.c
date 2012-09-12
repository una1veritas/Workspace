/**
  ******************************************************************************
  * @file    freertos_controller/joystick.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   psd sensor functions
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
#include "joystick.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TRIM_L_X  (-3)
#define TRIM_L_Y  (+4)
#define TRIM_R_X  (+4)
#define TRIM_R_Y  (+5)
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int16_t Joystick_L_X = 0, Joystick_L_Y = 0, Joystick_R_X = 0, Joystick_R_Y = 0;
__IO uint16_t ADCConvertedValue[2][2];
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for PSD sensor
 * @param  None
 * @retval None
 */
void Joystick_Configuration(void)
{
  // nvic configurationo for dma tc interrupt
  Joystick_NVIC_Configuration();
  // configure timer to trigger adc
  TIM2_Configuration();
  // Configuration ADC for PSD sensor
  Joystick_ADC_Configuration();
}

/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void Joystick_NVIC_Configuration(void)
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
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_Sonic_Process(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  while (1)
    {
    }
}

/**
  * @brief  Configure ADC12 with DMA1
  * @param  None
  * @retval : None
  */
void Joystick_ADC_Configuration(void)
{
  ADC_InitTypeDef  ADC_InitStructure;
  DMA_InitTypeDef DMA_InitStructure;
  GPIO_InitTypeDef GPIO_InitStructure;

  /* ADCCLK = PCLK2/8 = 72/8 = 9MHz*/
  RCC_ADCCLKConfig(RCC_PCLK2_Div8);
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | RCC_APB2Periph_ADC2 | ADC_IN0_3_GPIO_RCC | ADC_IN8_9_GPIO_RCC, ENABLE);

  GPIO_InitStructure.GPIO_Pin = ADC123_IN2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC123_IN2_PORT, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = ADC123_IN3_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(ADC123_IN3_PORT, &GPIO_InitStructure);

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
  DMA_InitStructure.DMA_BufferSize = 2;
  DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
  DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
  DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Word;
  DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Word;
  DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
  DMA_InitStructure.DMA_Priority = DMA_Priority_High;
  DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
  DMA_Init(DMA1_Channel1, &DMA_InitStructure);

  /* Enable DMA1 Channel1 Transfer Complete interrupt */
  DMA_ITConfig(DMA1_Channel1, DMA_IT_TC, ENABLE);

  /* Enable DMA1 channel1 */
  DMA_Cmd(DMA1_Channel1, ENABLE);


  ADC_DeInit(ADC1);
  ADC_DeInit(ADC2);

  /* ADC1 Configuration ------------------------------------------------------*/
  ADC_InitStructure.ADC_Mode = ADC_Mode_RegSimult;
  ADC_InitStructure.ADC_ScanConvMode = ENABLE;
  ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;
  ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_T2_CC2;
  ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
  ADC_InitStructure.ADC_NbrOfChannel = 2;
  ADC_Init(ADC1, &ADC_InitStructure);
  ADC_RegularChannelConfig(ADC1, ADC123_IN2_CH,   1, ADC_SampleTime_239Cycles5);
  ADC_RegularChannelConfig(ADC1, ADC12_IN8_CH,   2, ADC_SampleTime_239Cycles5);
  ADC_DMACmd(ADC1, ENABLE);

  ADC_Init(ADC2, &ADC_InitStructure);
  ADC_RegularChannelConfig(ADC2, ADC123_IN3_CH,   1, ADC_SampleTime_239Cycles5);
  ADC_RegularChannelConfig(ADC2, ADC12_IN9_CH,   2, ADC_SampleTime_239Cycles5);
  ADC_ExternalTrigConvCmd (ADC2, ENABLE);


  ADC_Cmd(ADC1, ENABLE);
  /* Enable ADC1 reset calibration register */
  ADC_ResetCalibration(ADC1);
  while(ADC_GetResetCalibrationStatus(ADC1));
  /* Start ADC1 calibration */
  ADC_StartCalibration(ADC1);
  while(ADC_GetCalibrationStatus(ADC1));

  ADC_Cmd(ADC2, ENABLE);
  /* Enable ADC2 reset calibration register */
  ADC_ResetCalibration(ADC2);
  while(ADC_GetResetCalibrationStatus(ADC2));
  /* Start ADC2 calibration */
  ADC_StartCalibration(ADC2);
  while(ADC_GetCalibrationStatus(ADC2));

  /* Start ADC1 Software Conversion */
  ADC_SoftwareStartConvCmd(ADC1, ENABLE);

}

/**
  * @brief  This function handles DMA1 Channel 1 interrupt request.
  * @param  None
  * @retval None
  */
void DMA1_Channel1_IRQHandler(void)
{
  static uint8_t lr = 0;
  /* Test on DMA1 Channel1 Transfer Complete interrupt */
  if(DMA_GetITStatus(DMA1_IT_TC1))
  {
    if (lr == 0)
      {
        Joystick_L_X = JoystickAdjust(ADCConvertedValue[0][0], 4, TRIM_L_X, 0, 124, 131, 1, 2);
        Joystick_L_Y = JoystickAdjust(ADCConvertedValue[0][1], 4, TRIM_L_Y, 1, 124, 131, 1, 2);
        lr = 1;
      }
    if (lr == 1)
      {
        Joystick_R_X = JoystickAdjust(ADCConvertedValue[1][0], 4, TRIM_R_X, 0, 121, 124, 1, 2);
        Joystick_R_Y = JoystickAdjust(ADCConvertedValue[1][1], 4, TRIM_R_Y, 1, 123, 132, 1, 2);
        lr = 0;
      }

    /* Clear DMA1 Channel1 Transfer Complete interrupt pending bits */
    DMA_ClearITPendingBit(DMA1_IT_TC1);
  }
}

int16_t JoystickAdjust(uint16_t intvalue,uint8_t r_shift, int8_t center_trim, uint8_t invert, int16_t max_plus, int16_t max_minus, int8_t margin_max, int8_t margin_min)
{
  int16_t adjusted;

  adjusted = ((int16_t)(intvalue >> r_shift)) -128 + center_trim;

  if (invert)
    {
      adjusted = -adjusted;
    }

  if (adjusted > 0)
    {
      adjusted = (adjusted * (100 + margin_max + margin_min)) / max_plus;
    }
  else if (adjusted < 0)
    {
      adjusted = (adjusted * (100 + margin_max + margin_min)) / max_minus;
    }
  else
    {
      // do nothing
    }

  if (adjusted > 100 + margin_min)
    {
      adjusted  = 100;
    }
  else if (adjusted > margin_min)
    {
      adjusted  = adjusted - margin_min;
    }
  else if (adjusted < -100 - margin_min)
    {
      adjusted  = -100;
    }
  else if (adjusted < -margin_min)
    {
      adjusted  = adjusted + margin_min;
    }
  else
    {
      adjusted = 0;
    }

  return adjusted;
}

/**
  * @brief  Configure TIM2
  * @param  None
  * @retval : None
  */
void TIM2_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM2_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM2 Configuration: Output Compare PWM Mode:
    TIM4CLK = 72 MHz, Prescaler = 7200, TIM4 counter clock = 10kHz
  ----------------------------------------------------------------------------*/
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 100;
  TIM_TimeBaseStructure.TIM_Prescaler = 7200;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

  TIM_OCInitTypeDef  TIM_OCInitStructure;
  /* Output Compare Toggle Mode configuration: Channel2 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_Pulse = 1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OC2Init(TIM2, &TIM_OCInitStructure);
  TIM_OC2PreloadConfig(TIM2, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM2, ENABLE);

}

