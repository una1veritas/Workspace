/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2003
 *
 *    File name   : sd_spi_mode.h
 *    Description : define MMC module
 *
 *    History :
 *    1. Date        : July 1, 2005
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #1 $
 **************************************************************************/

#include "includes.h"

#ifndef __SD_SPI_MODE_H
#define __SD_SPI_MODE_H

#ifdef SD_SPI_MODE_GLOBAL
#define SD_SPI_MODE_EXTERN
#else
#define SD_SPI_MODE_EXTERN  extern
#endif

#define SD_DISK_LUN       0
#define SD_DEF_BLK_SIZE   512
#define IdentificationModeClock   (200KHZ)

/*************************************************************************
 * Function Name: SdStatusUpdate
 * Parameters: none
 *
 * Return: none
 *
 * Description: Update status of SD/MMC card
 *
 *************************************************************************/
void SdStatusUpdate (void);

/*************************************************************************
 * Function Name: SdDiskInit
 * Parameters:  none
 *
 * Return: none
 *
 * Description: Init MMC/SD disk
 *
 *************************************************************************/
void SdDiskInit (void);

/*************************************************************************
 * Function Name: SdDiskInfo
 * Parameters:  pInt8U pData, DiskInfoType_t DiskInfoType
 *
 * Return: Int32U
 *
 * Description: Return pointer to Info structure of the disk
 * (Inquiry or Format capacity)
 *
 *************************************************************************/
Int32U SdDiskInfo (pInt8U pData, DiskInfoType_t DiskInfoType);

/*************************************************************************
 * Function Name: SdGetDiskCtrlBkl
 * Parameters:  none
 *
 * Return: pDiskCtrlBlk_t
 *
 * Description: Return pointer to status structure of the disk
 *
 *************************************************************************/
pDiskCtrlBlk_t SdGetDiskCtrlBkl (void);

/*************************************************************************
 * Function Name: SdDiskIO
 * Parameters: pInt8U pData,Int32U BlockStart,
 *             Int32U BlockNum, DiskIoRequest_t IoRequest
 *
 * Return: DiskStatusCode_t
 *
 * Description: MMC/SD disk I/O
 *
 *************************************************************************/
DiskStatusCode_t SdDiskIO (pInt8U pData,Int32U BlockStart,
                              Int32U BlockNum, DiskIoRequest_t IoRequest);

#endif // __MMC_H
