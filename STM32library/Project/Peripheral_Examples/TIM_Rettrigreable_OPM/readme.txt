/**
  @page TIM_Rettrigreable_OPM TIM Rettrigreable One Pulse Mode Example
  
  @verbatim
  ******************** (C) COPYRIGHT 2012 STMicroelectronics *******************
  * @file    TIM_Rettrigreable_OPM/readme.txt 
  * @author  MCD Application Team
  * @version V1.1.0
  * @date    20-September-2012
  * @brief   TIM Rettrigreable One Pulse Mode Example Description.
  ******************************************************************************
  *
  * Licensed under MCD-ST Liberty SW License Agreement V2, (the "License");
  * You may not use this file except in compliance with the License.
  * You may obtain a copy of the License at:
  *
  *        http://www.st.com/software_license_agreement_liberty_v2
  *
  * Unless required by applicable law or agreed to in writing, software 
  * distributed under the License is distributed on an "AS IS" BASIS, 
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  *   
  ******************************************************************************
  @endverbatim

@par Example Description 

This example shows how to use the TIM peripheral to generate a Retrigerrable One pulse Mode 
after a Rising edge of an external signal is received in Timer Input pin.

TIM2CLK = SystemCoreClock, we want to get TIM2 counter clock at 36 MHz:
  - Prescaler = (TIM2CLK / TIM2 counter clock) - 1

SystemCoreClock is set to 72 MHz.

The Autoreload value is 65535 (TIM2->ARR), so the maximum frequency value to 
trigger the TIM2 input is 36000000/65535 = 549.3 Hz.

The TIM2 is configured as follows: 
The Retrigerrable One Pulse mode is used, the external signal is connected to TIM2 CH2 pin (PA.01), 
the rising edge is used as active edge, the One Pulse signal is output on TIM2_CH1 (PA.00).

The TIM_Pulse defines the One Pulse value, the pulse value is fixed to:
Retrigerrable One Pulse value = TIM_Period /TIM2 counter clock 
                              = 65535 / 36000000 = 1.8 ms.
  
@par Directory contents 

  - TIM_Rettrigreable_OPM/stm32f30x_conf.h     Library Configuration file
  - TIM_Rettrigreable_OPM/stm32f30x_it.c       Interrupt handlers
  - TIM_Rettrigreable_OPM/stm32f30x_it.h       Header for stm32f30x_it.c
  - TIM_Rettrigreable_OPM/main.c               Main program
  - TIM_Rettrigreable_OPM/main.h               Header for main.c
  - TIM_Rettrigreable_OPM/system_stm32f30x.c   STM32F30x system source file
           
@note The "system_stm32f30x.c" file contains the system clock configuration for
      STM32F30x devices, and is customized for use with STM32F3-Discovery Kit. 
      The STM32F30x is configured to run at 72 MHz, following the three  
      configuration below:
        + PLL_SOURCE_HSI
           - HSI (~8 MHz) used to clock the PLL, and the PLL is used as system 
             clock source.  
        + PLL_SOURCE_HSE          
           - HSE (8 MHz) used to clock the PLL, and the PLL is used as system
             clock source.
           - The HSE crystal is not provided with the Discovery Kit, some 
             hardware modification are needed in manner to connect this crystal.
             For more details, refer to section "4.10 OSC clock" in "STM32F3 discovery kit User manual (UM1570)"
        + PLL_SOURCE_HSE_BYPASS   
           - HSE bypassed with an external clock (fixed at 8 MHz, coming from 
             ST-Link circuit) used to clock the PLL, and the PLL is used as 
             system clock source.
           - Some  hardware modification are needed in manner to bypass the HSE 
             with clock coming from the ST-Link circuit.
             For more details, refer to section "4.10 OSC clock" in "STM32F3 discovery kit User manual (UM1570)"
      User can select one of the three configuration in system_stm32f30x.c file
      (default configuration is PLL_SOURCE_HSE_BYPASS).           

@par Hardware and Software environment

  - This example runs on STM32F30x Devices.
  
  - This example has been tested with STMicroelectronics STM32F3-Discovery (MB1035) 
    and can be easily tailored to any other supported device and development board.

  - STM32F3-Discovery Set-up
    - Connect the external signal to the TIM2_CH2 pin (PA.01)
    - Connect the TIM2_CH1 (PA.00) pin to an oscilloscope to monitor the waveform.
      
@par How to use it ? 

In order to make the program work, you must do the following :

 + EWARM
    - Open the TIM_Rettrigreable_OPM.eww workspace 
    - Rebuild all files: Project->Rebuild all
    - Load project image: Project->Debug
    - Run program: Debug->Go(F5)

 + MDK-ARM
    - Open the TIM_Rettrigreable_OPM.uvproj project
    - Rebuild all files: Project->Rebuild all target files
    - Load project image: Debug->Start/Stop Debug Session
    - Run program: Debug->Run (F5)    

 + TASKING
    - Open TASKING toolchain.
    - Click on File->Import, select General->'Existing Projects into Workspace' 
      and then click "Next". 
    - Browse to  TASKING workspace directory and select the project "TIM_Rettrigreable_OPM"   
    - Rebuild all project files: Select the project in the "Project explorer" 
      window then click on Project->build project menu.
    - Run program: Select the project in the "Project explorer" window then click 
      Run->Debug (F11)

 + TrueSTUDIO for ARM
    - Open the TrueSTUDIO for ARM toolchain.
    - Click on File->Switch Workspace->Other and browse to TrueSTUDIO workspace 
      directory.
    - Click on File->Import, select General->'Existing Projects into Workspace' 
      and then click "Next". 
    - Browse to the TrueSTUDIO workspace directory and select the project "TIM_Rettrigreable_OPM" 
    - Rebuild all project files: Select the project in the "Project explorer" 
      window then click on Project->build project menu.
    - Run program: Select the project in the "Project explorer" window then click 
      Run->Debug (F11)
  
 * <h3><center>&copy; COPYRIGHT STMicroelectronics</center></h3>
 */
