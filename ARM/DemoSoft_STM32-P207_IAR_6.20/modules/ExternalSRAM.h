/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : ExternalSRAM.h
 *    Description : Driver for the external SRAM schip
 *
 *    History :
 *    1. Date        : 27 Oct 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/

#ifndef _EXT_SRAM_H
#define _EXT_SRAM_H

#define EXT_SRAM_BASE_ADDRESS 0x60000000
#define EXT_SRAM_SIZE 0x80000

/******************************************************************************
* Description: ExtSRAM_Initialize(..) - initializes FSMC interface
* Input: 	none
* Output: 	none
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
int ExtSRAM_Initialize(void);

/******************************************************************************
* Description: ExtSRAM_Deinitialize(..) - deinitializes FSMC interface, stopts its clock
* Input: 	none
* Output: 	none
* Return:	none
*******************************************************************************/
void ExtSRAM_Deinitialize(void);

/******************************************************************************
* Description: ExtSRAM_Test(..) - test access to the SRAM
* Input: 	none
* Output: 	none
* Return:	0 if test successful,
*			-1 if error occured on the address bus
*			-2 if error occured on the data bus
*******************************************************************************/
int ExtSRAM_Test(void);


#endif // _EXT_SRAM_H



