/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : sd_ll_spi1.h
 *    Description : SD/MMC low level SPI1 driver include file
 *
 *    History :
 *    1. Date        : Appril 10, 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/

#include "includes.h"

#ifndef __SD_LL_SPI1_H
#define __SD_LL_SPI1_H

#ifdef SD_LL_SPI1_GLOBAL
#define SD_LL_SPI1_EXTERN
#else
#define SD_LL_SPI1_EXTERN  extern
#endif

#define SD_DLY_1MSEC       1000

/*************************************************************************
 * Function Name: SdPowerOn
 * Parameters: none
 * Return: none
 *
 * Description: Set power off state
 *
 *************************************************************************/
void SdPowerOn (void);

/*************************************************************************
 * Function Name: SdPowerOff
 * Parameters: none
 * Return: none
 *
 * Description: Set power off state
 *
 *************************************************************************/
void SdPowerOff (void);

/*************************************************************************
 * Function Name: SdChipSelect
 * Parameters: Boolean Select
 * Return: none
 *
 * Description: SD/MMC Chip select control
 * Select = true  - Chip is enable
 * Select = false - Chip is disable
 *
 *************************************************************************/
void SdChipSelect (Boolean Select);

/*************************************************************************
 * Function Name: SdPresent
 * Parameters: none
 * Return: Boolean - true cart present
 *                 - false cart no present
 *
 * Description: SD/MMC precent check
 *
 *************************************************************************/
Boolean SdPresent(void);

/*************************************************************************
 * Function Name: SdWriteProtect
 * Parameters: none
 * Return: Boolean - true cart is protected
 *                 - false cart no protected
 *
 * Description: SD/MMC Write protect check
 *
 *************************************************************************/
Boolean SdWriteProtect (void);

/*************************************************************************
 * Function Name: SdSetClockFreq
 * Parameters: Int32U Frequency
 * Return: Int32U
 *
 * Description: Set SPI ckl frequency
 *
 *************************************************************************/
Int32U SdSetClockFreq (Int32U Frequency);

/*************************************************************************
 * Function Name: SdInit
 * Parameters: none
 * Return: none
 *
 * Description: Init SPI, Cart Present, Write Protect and Chip select pins
 *
 *************************************************************************/
void SdInit (void);

/*************************************************************************
 * Function Name: SdTranserByte
 * Parameters: Int8U ch
 * Return: Int8U
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
Int8U SdTranserByte (Int8U ch);

/*************************************************************************
 * Function Name: SdSendBlock
 * Parameters: pInt8U pData, Int32U Size
 *
 * Return: void
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
void SdSendBlock (pInt8U pData, Int32U Size);

/*************************************************************************
 * Function Name: SdReceiveBlock
 * Parameters: pInt8U pData, Int32U Size
 *
 * Return: void
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
void SdReceiveBlock (pInt8U pData, Int32U Size);

/*************************************************************************
 * Function Name: SdDly_1ms
 * Parameters: Int32U Delay
 * Return: none
 *
 * Description: Delay [msec]
 *
 *************************************************************************/
void SdDly_1ms (Int32U Delay);

#endif // __SD_LL_SPI1_H
