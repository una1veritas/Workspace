/*
             LUFA Library
     Copyright (C) Dean Camera, 2009.
              
  dean [at] fourwalledcubicle [dot] com
      www.fourwalledcubicle.com
*/

/*
  Copyright 2009  Dean Camera (dean [at] fourwalledcubicle [dot] com)

  Permission to use, copy, modify, and distribute this software
  and its documentation for any purpose and without fee is hereby
  granted, provided that the above copyright notice appear in all
  copies and that both that the copyright notice and this
  permission notice and warranty disclaimer appear in supporting
  documentation, and that the name of the author not be used in
  advertising or publicity pertaining to distribution of the
  software without specific, written prior permission.

  The author disclaim all warranties with regard to this
  software, including all implied warranties of merchantability
  and fitness.  In no event shall the author be liable for any
  special, indirect or consequential damages or any damages
  whatsoever resulting from loss of use, data or profits, whether
  in an action of contract, negligence or other tortious action,
  arising out of or in connection with the use or performance of
  this software.
*/

/** \file
 *
 *  Functions to manage the physical dataflash media, including reading and writing of
 *  blocks of data. These functions are called by the SCSI layer when data must be stored
 *  or retrieved to/from the physical storage media. If a different media is used (such
 *  as a SD card or EEPROM), functions similar to these will need to be generated.
 */

#define  INCLUDE_FROM_SDCARDMANAGER_C
#include "SDCardManager.h"
#include "sd_raw.h"

static struct sd_raw_info disk_info;
static uint32_t CachedTotalBlocks = 0;
static uint8_t Buffer[16];
static USB_ClassInfo_MS_Device_t* Current_MS_Device;
void SDCardManager_Init(void)
{
	while(!sd_raw_init());
		//printf_P(PSTR("MMC/SD initialization failed\r\n"));
}

uint32_t SDCardManager_GetNbBlocks(void)
{
	uint32_t TotalBlocks = 0;
	
	if (CachedTotalBlocks != 0)
		return CachedTotalBlocks;
		
	if(!sd_raw_get_info(&disk_info))
	{
		//printf_P(PSTR("Error reading SD card info\r\n"));
		return 0;
	}

	CachedTotalBlocks = disk_info.capacity / 512;
	//printf_P(PSTR("SD blocks: %li\r\n"), TotalBlocks);
	
	return CachedTotalBlocks;
}

/** Writes blocks (OS blocks, not Dataflash pages) to the storage medium, the board dataflash IC(s), from
 *  the pre-selected data OUT endpoint. This routine reads in OS sized blocks from the endpoint and writes
 *  them to the dataflash in Dataflash page sized blocks.
 *
 *  \param[in] BlockAddress  Data block starting address for the write sequence
 *  \param[in] TotalBlocks   Number of blocks of data to write
 */
uintptr_t SDCardManager_WriteBlockHandler(uint8_t* buffer, offset_t offset, void* p)
{
	/* Check if the endpoint is currently empty */
	if (!(Endpoint_IsReadWriteAllowed()))
	{
		/* Clear the current endpoint bank */
		Endpoint_ClearOUT();
		
		/* Wait until the host has sent another packet */
		if (Endpoint_WaitUntilReady())
		  return 0;
	}
	
	/* Write one 16-byte chunk of data to the dataflash */
	buffer[0] = Endpoint_Read_Byte();
	buffer[1] = Endpoint_Read_Byte();
	buffer[2] = Endpoint_Read_Byte();
	buffer[3] = Endpoint_Read_Byte();
	buffer[4] = Endpoint_Read_Byte();
	buffer[5] = Endpoint_Read_Byte();
	buffer[6] = Endpoint_Read_Byte();
	buffer[7] = Endpoint_Read_Byte();
	buffer[8] = Endpoint_Read_Byte();
	buffer[9] = Endpoint_Read_Byte();
	buffer[10] = Endpoint_Read_Byte();
	buffer[11] = Endpoint_Read_Byte();
	buffer[12] = Endpoint_Read_Byte();
	buffer[13] = Endpoint_Read_Byte();
	buffer[14] = Endpoint_Read_Byte();
	buffer[15] = Endpoint_Read_Byte();
	
	return 16;
}

void SDCardManager_WriteBlocks(USB_ClassInfo_MS_Device_t* const MSInterfaceInfo, uint32_t BlockAddress, uint16_t TotalBlocks)
{
	bool     UsingSecondBuffer   = false;
    Current_MS_Device = MSInterfaceInfo;
    
	//printf_P(PSTR("W %li %i\r\n"), BlockAddress, TotalBlocks);

	/* Wait until endpoint is ready before continuing */
	if (Endpoint_WaitUntilReady())
	  return;
	
	while (TotalBlocks)
	{
		sd_raw_write_interval(BlockAddress *  VIRTUAL_MEMORY_BLOCK_SIZE, Buffer, VIRTUAL_MEMORY_BLOCK_SIZE, &SDCardManager_WriteBlockHandler, NULL);
		
		/* Check if the current command is being aborted by the host */
		if (MSInterfaceInfo->State.IsMassStoreReset)
		  return;
			
		/* Decrement the blocks remaining counter and reset the sub block counter */
		BlockAddress++;
		TotalBlocks--;
	}

	/* If the endpoint is empty, clear it ready for the next packet from the host */
	if (!(Endpoint_IsReadWriteAllowed()))
	  Endpoint_ClearOUT();
}

/** Reads blocks (OS blocks, not Dataflash pages) from the storage medium, the board dataflash IC(s), into
 *  the pre-selected data IN endpoint. This routine reads in Dataflash page sized blocks from the Dataflash
 *  and writes them in OS sized blocks to the endpoint.
 *
 *  \param[in] BlockAddress  Data block starting address for the read sequence
 *  \param[in] TotalBlocks   Number of blocks of data to read
 */

uint8_t SDCardManager_ReadBlockHandler(uint8_t* buffer, offset_t offset, void* p)
{
	uint8_t i;

	/* Check if the endpoint is currently full */
	if (!(Endpoint_IsReadWriteAllowed()))
	{
		/* Clear the endpoint bank to send its contents to the host */
		Endpoint_ClearIN();
		
		/* Wait until the endpoint is ready for more data */
		if (Endpoint_WaitUntilReady())
		  return 0;
	}
		
	Endpoint_Write_Byte(buffer[0]);
	Endpoint_Write_Byte(buffer[1]);
	Endpoint_Write_Byte(buffer[2]);
	Endpoint_Write_Byte(buffer[3]);
	Endpoint_Write_Byte(buffer[4]);
	Endpoint_Write_Byte(buffer[5]);
	Endpoint_Write_Byte(buffer[6]);
	Endpoint_Write_Byte(buffer[7]);
	Endpoint_Write_Byte(buffer[8]);
	Endpoint_Write_Byte(buffer[9]);
	Endpoint_Write_Byte(buffer[10]);
	Endpoint_Write_Byte(buffer[11]);
	Endpoint_Write_Byte(buffer[12]);
	Endpoint_Write_Byte(buffer[13]);
	Endpoint_Write_Byte(buffer[14]);
	Endpoint_Write_Byte(buffer[15]);
	
	/* Check if the current command is being aborted by the host */
	if (Current_MS_Device->State.IsMassStoreReset)
	  return 0;
	
	return 1;
}

void SDCardManager_ReadBlocks(USB_ClassInfo_MS_Device_t* const MSInterfaceInfo, uint32_t BlockAddress, uint16_t TotalBlocks)
{
	uint16_t CurrPage          = BlockAddress;
	uint16_t CurrPageByte      = 0;
	
	Current_MS_Device = MSInterfaceInfo;

	//printf_P(PSTR("R %li %i\r\n"), BlockAddress, TotalBlocks);
	
	/* Wait until endpoint is ready before continuing */
	if (Endpoint_WaitUntilReady())
	  return;
	
	while (TotalBlocks)
	{
		/* Read a data block from the SD card */
		sd_raw_read_interval(BlockAddress * VIRTUAL_MEMORY_BLOCK_SIZE, Buffer, 16, 512, &SDCardManager_ReadBlockHandler, NULL);
		
		/* Decrement the blocks remaining counter */
		BlockAddress++;
		TotalBlocks--;
	}
	
	/* If the endpoint is full, send its contents to the host */
	if (!(Endpoint_IsReadWriteAllowed()))
	  Endpoint_ClearIN();
}

/** Performs a simple test on the attached Dataflash IC(s) to ensure that they are working.
 *
 *  \return Boolean true if all media chips are working, false otherwise
 */
bool SDCardManager_CheckDataflashOperation(void)
{	
	return true;
}
