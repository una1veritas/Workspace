/**
  ******************************************************************************
  * @file    usbd_conf.h
  * @author  MCD Application Team
  * @version V1.1.0
  * @date    19-March-2012
  * @brief   USB device configuration file
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT 2012 STMicroelectronics</center></h2>
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
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USBD_CONF__H__
#define __USBD_CONF__H__

/* Includes ------------------------------------------------------------------*/
#include "usb_conf.h"


/** @defgroup USB_CONF_Exported_Defines
  * @{
  */ 


#define USBD_CFG_MAX_NUM           1
#define USBD_ITF_MAX_NUM           1

#define USB_MAX_STR_DESC_SIZ       64 

#define USBD_SELF_POWERED               

/** @defgroup USB_MSC_Class_Layer_Parameter
  * @{
  */ 
#define MSC_IN_EP                    0x81
#define MSC_OUT_EP                   0x01

#define MSC_MAX_PACKET               512

#define MSC_MEDIA_PACKET             4096
/**
  * @}
  */ 

/** @defgroup USB_HID_Class_Layer_Parameter
  * @{
  */ 
#define HID_IN_EP                    0x81
#define HID_OUT_EP                   0x01

#define HID_IN_PACKET                4
#define HID_OUT_PACKET               4

/**
  * @}
  */ 
/** @defgroup USB_CONF_Exported_Types
  * @{
  */ 
/**
  * @}
  */ 


/** @defgroup USB_CONF_Exported_Macros
  * @{
  */ 
/**
  * @}
  */ 

/** @defgroup USB_CONF_Exported_Variables
  * @{
  */ 
/**
  * @}
  */ 

/** @defgroup USB_CONF_Exported_FunctionsPrototype
  * @{
  */ 
/**
  * @}
  */ 


#endif //__USBD_CONF__H__

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

