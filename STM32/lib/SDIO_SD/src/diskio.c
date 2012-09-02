// STM32F10 SDIO_DRIVE driver written by J. Shaler

/*-----------------------------------------------------------------------*/
/* Low level disk I/O module skeleton for FatFs     (C)ChaN, 2007        */
/*-----------------------------------------------------------------------*/
/* This is a stub disk I/O module that acts as front end of the existing */
/* disk I/O modules and attach it to FatFs module with common interface. */
/*-----------------------------------------------------------------------*/

#include "diskio.h"

#include "sdcard.h"    // for SDIO peripheral device implementation
//#include "timestamp.h" // for get_fattime implementation


/*-----------------------------------------------------------------------*/
/* Correspondence between physical drive number and physical drive.      */
#define SDIO_DRIVE		0




// Private variables
SD_CardInfo SDCardInfo;



/*-----------------------------------------------------------------------*/
/* Initialize a Drive                                                    */

DSTATUS disk_initialize (
                         BYTE drv				/* Physical drive nmuber (0..) */
                           )
{
  switch (drv)
  {
    case SDIO_DRIVE:
    {
      // Initialize SD Card
      SD_Error status = SD_Init();

      if (status == SD_OK)
      {
        // Read CSD/CID MSD registers
        status = SD_GetCardInfo(&SDCardInfo);
      }

      if (status == SD_OK)
      {
        // Select card
        status = SD_SelectDeselect((uint32_t)(SDCardInfo.RCA << 16));
      }

      if (status == SD_OK)
      {
        status = SD_EnableWideBusOperation(SDIO_BusWide_4b);
      }

      if (status == SD_OK)
      {
        // Set Device Transfer Mode to DMA
        status = SD_SetDeviceMode(SD_DMA_MODE);
      }

      if (status != SD_OK)
        return STA_NOINIT;
      else
        return 0x00;
    }
  }

  return STA_NOINIT;

}



/*-----------------------------------------------------------------------*/
/* Return Disk Status                                                    */

DSTATUS disk_status (
                     BYTE drv		/* Physical drive nmuber (0..) */
                       )
{
  switch (drv)
  {
    case SDIO_DRIVE:
    {
      SD_Error status = SD_GetCardInfo(&SDCardInfo);

      if (status != SD_OK)
        return STA_NOINIT;
      else
        return 0x00;
    }
  }

  return STA_NOINIT;
}



/*-----------------------------------------------------------------------*/
/* Read Sector(s)                                                        */

DRESULT disk_read (
                   BYTE drv,		/* Physical drive nmuber (0..) */
                   BYTE *buff,		/* Data buffer to store read data */
                   DWORD sector,	/* Sector address (LBA) */
                   BYTE count		/* Number of sectors to read (1..255) */
                     )
{
  switch (drv)
  {
    case SDIO_DRIVE:
    {
      SD_Error status = SD_OK;
      for (int secNum = 0; secNum < count && status == SD_OK; secNum++)
      {
        status = SD_ReadBlock((sector+secNum)*512, (uint32_t*)(buff+512*secNum), 512);
      }
      if (status == SD_OK)
        return RES_OK;
      else
        return RES_ERROR;
    }
  }
  return RES_PARERR;
}



/*-----------------------------------------------------------------------*/
/* Write Sector(s)                                                       */

#if _READONLY == 0
DRESULT disk_write (
                    BYTE drv,			/* Physical drive nmuber (0..) */
                    const BYTE *buff,	/* Data to be written */
                    DWORD sector,		/* Sector address (LBA) */
                    BYTE count			/* Number of sectors to write (1..255) */
                      )
{
  switch (drv)
  {
    case SDIO_DRIVE:
    {
      SD_Error status = SD_OK;
      for (int secNum = 0; secNum < count && status == SD_OK; secNum++)
      {
        status = SD_WriteBlock((sector+secNum)*512,
                              (uint32_t*)(buff+512*secNum),
                              512);
      }
      if (status == SD_OK)
        return RES_OK;
      else
        return RES_ERROR;
    }
  }
  return RES_PARERR;
}
#endif /* _READONLY */



/*-----------------------------------------------------------------------*/
/* Miscellaneous Functions                                               */

DRESULT disk_ioctl (
                    BYTE drv,		/* Physical drive nmuber (0..) */
                    BYTE ctrl,		/* Control code */
                    void *buff		/* Buffer to send/receive control data */
                      )
{
  switch (drv)
  {
    case SDIO_DRIVE:
    {
      switch (ctrl)
      {
        case CTRL_SYNC:
          // no synchronization to do since not buffering in this module
          return RES_OK;
        case GET_SECTOR_SIZE:
          *(WORD*)buff = 512;
          return RES_OK;
        case GET_SECTOR_COUNT:
          *(DWORD*)buff = SDCardInfo.CardCapacity / 512;
          return RES_OK;
        case GET_BLOCK_SIZE:
          *(DWORD*)buff = 512;
          return RES_OK;
      }
    }
  }
  return RES_PARERR;
}

