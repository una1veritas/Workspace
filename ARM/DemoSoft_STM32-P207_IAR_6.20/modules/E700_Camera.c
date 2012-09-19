/*************************************************************************
 *
 *
 *    (c) Copyright Olimex 2011
 *
 *    File name   : E700_Camera.c
 *    Description : API for the E700 Camera
 *
 *    History :
 *    1. Date        : 18 Oct 2011
 *       Author      : Aleksandar Mitev
 *       Description : Create
 *
 **************************************************************************/
#include <stdio.h>
#include <string.h>
#include <yfuns.h>
#include "includes.h"
#include "ExternalSRAM.h"
#include "E700_Camera.h"

/* DEFINE LOCAL TYPES HERE */
typedef Int8U BYTE;

/* DEFINE LOCAL CONSTANTS HERE */
#define SOFT_I2C_PORT_CLOCK RCC_AHB1Periph_GPIOG
#define SOFT_I2C_PORT GPIOG
#define SOFT_I2C_SCL_PIN GPIO_Pin_10
#define SOFT_I2C_SDA_PIN GPIO_Pin_12

#define ACK 0
#define NACK 1

#define E700_SLAVE_ADDRESS 0x22

#define DCMI_DR_ADDRESS 0x50050028

/* DECLARE EXTERNAL VARIABLES HERE */

/* DEFINE LOCAL MACROS HERE */

/* DEFINE LOCAL VARIABLES HERE */

/* DECLARE EXTERNAL VARIABLES HERE */

/* DECLARE LOCAL FUNCTIONS HERE */
static void local_I2C_Initialize(void);
static void local_I2C_Delay(void);
static void local_I2C_Start(void);
static void local_I2C_Stop(void);
static BYTE local_I2C_ReadByte(char ack);
static char local_I2C_WriteByte(BYTE data); // returns ack state

static void DCMI_Config(void);

/* DEFINE FUNCTIONS HERE */

/******************************************************************************
* Description: E700_Camera_Initialize(..) - initializes DCMI interface, I2C software interface and configures camera
* Input: 	none
* Output: 	none
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
int E700_Camera_Initialize(void)
{
	int result = 0;
	BYTE idVal;
	GPIO_InitTypeDef GPIO_InitStructure;

	local_I2C_Initialize();
	
	// enable MCO1
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	RCC_MCO1Config(RCC_MCO1Source_HSE, RCC_MCO1Div_1);
	
	// enable power
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOF, &GPIO_InitStructure);
	GPIO_WriteBit(GPIOF, GPIO_Pin_9, Bit_SET);
	
	// init DCMI interface and DMI
	DCMI_Config();
	
	// release reset
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOF, &GPIO_InitStructure);
	GPIO_WriteBit(GPIOF, GPIO_Pin_11, Bit_SET);
	
	// enable operation
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	GPIO_WriteBit(GPIOB, GPIO_Pin_2, Bit_SET);
		
	// read the ID of the camera
	do {
		local_I2C_Start();
		result |= local_I2C_WriteByte(E700_SLAVE_ADDRESS | 0x00);
		result |= local_I2C_WriteByte(0x00); // id address
		local_I2C_Stop();
		if(result) break;
		
		local_I2C_Start();
		result |= local_I2C_WriteByte(E700_SLAVE_ADDRESS | 0x01);
		idVal = local_I2C_ReadByte(NACK);
		local_I2C_Stop();			
	} while(0);
	
	if(idVal != 0x40)
		result |= -1;
	                               	
	return (result ? -1 : 0);
}

/******************************************************************************
* Description: E700_Camera_Deinitialize(..) - deinitializes camera stuff
* Input: 	none
* Output: 	none
* Return:	none
*******************************************************************************/
void E700_Camera_Deinitialize(void)
{
   	DMA_DeInit(DMA2_Stream1);
	
	// disable power
	GPIO_WriteBit(GPIOF, GPIO_Pin_9, Bit_RESET);
	
	// hold in reset
	GPIO_WriteBit(GPIOF, GPIO_Pin_11, Bit_RESET);

	// disable interface
	DCMI_DeInit();
	RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_DCMI, DISABLE);
}

/******************************************************************************
* Description: E700_Camera_CaptureImage(..) - enable capture of 1 shot and wait for capturing to complete
* Input: 	none
* Output: 	none
* Return:	pointer to the capture image raw data
*******************************************************************************/
pInt8U E700_Camera_CaptureImage(void)
{
		/* Enable DMA transfer */
		DMA_Cmd(DMA2_Stream1, ENABLE);
		
		/* Start Image capture */
		DCMI_CaptureCmd(ENABLE);
		
		// wait for the transfer to complete
		while( DMA_GetFlagStatus(DMA2_Stream1, DMA_FLAG_TCIF1) != SET )
			;
		DMA_ClearFlag(DMA2_Stream1, DMA_FLAG_TCIF1);
		
		// reload number of transfers for the next capture
		DMA_SetCurrDataCounter(DMA2_Stream1, (CAMERA_HOR_RES * CAMERA_VER_RES * 2) / sizeof(Int32U));
		
		return (pInt8U)EXT_SRAM_BASE_ADDRESS;
}

/******************************************************************************
* Description: E700_Camera_ProcessImage(..) - converts 4:2:2 YCbCr values to 16bit RGB (Little Endian)
* Input: 	pInData - input buffer containing the raw values
*			width - width in px of the image
*           height - height in px of the image
* Output: 	pOutData - output buffer to contain the converted values
* Return:	0 if sucessfully initialized, -1 if error occured
*******************************************************************************/
void E700_Camera_ProcessImage(pInt8U pInData, pInt8U pOutData, Int16U width, Int16U height)
{
	Int32S i = width * height * 2;
	
	while(i >= 0)
	{
		Int8U Y0, Y1, Cb, Cr;
		float calcR, calcG, calcB;
		
		Y0 = *pInData++;
		Cb = *pInData++;
		Y1 = *pInData++;
		Cr = *pInData++;
			
		calcR = Y0 + 1.371f * (Cr - 128);
		calcG = Y0 - 0.698f * (Cr - 128) - 0.336 * (Cb - 128);
		calcB = Y0 + 1.732f * (Cb - 128);
	
		if (calcR < 0) calcR = 0;
		if (calcG < 0) calcG = 0;
		if (calcB < 0) calcB = 0;
	
		*pOutData++ = ((((Int8U)calcG) << 3) & 0xE0) | ((((Int8U)calcB) >> 3) & 0x1F);
		*pOutData++ = (((Int8U)calcR) & 0xF8) | ((((Int8U)calcG) >> 5) & 0x07);

		calcR = Y1 + 1.371f * (Cr - 128);
		calcG = Y1 - 0.698f * (Cr - 128) - 0.336 * (Cb - 128);
		calcB = Y1 + 1.732f * (Cb - 128);
	
		if (calcR < 0) calcR = 0;
		if (calcG < 0) calcG = 0;
		if (calcB < 0) calcB = 0;
		
		*pOutData++ = ((((Int8U)calcG) << 3) & 0xE0) | ((((Int8U)calcB) >> 3) & 0x1F);
		*pOutData++ = (((Int8U)calcR) & 0xF8) | ((((Int8U)calcG) >> 5) & 0x07);
	
		i -= 2;
	}
}

/* local functions */
static void DCMI_Config(void)
{
	DCMI_InitTypeDef DCMI_InitStructure;
	DCMI_CROPInitTypeDef DCMI_CROPStructure;
	GPIO_InitTypeDef GPIO_InitStructure;
	DMA_InitTypeDef  DMA_InitStructure;
	
	/* Enable DCMI GPIOs clocks */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA | RCC_AHB1Periph_GPIOB |
						   RCC_AHB1Periph_GPIOC | RCC_AHB1Periph_GPIOE , ENABLE);
	
	/* Enable DCMI clock */
	RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_DCMI, ENABLE);
	
	/* Connect DCMI pins to AF13 ************************************************/
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource6, GPIO_AF_DCMI); // PCLK
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource4, GPIO_AF_DCMI); // HSYNC
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource7, GPIO_AF_DCMI); // VSYNC
	
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource6, GPIO_AF_DCMI); // Data[0..7]
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource10, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource8, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource9, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOE, GPIO_PinSource4, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOE, GPIO_PinSource5, GPIO_AF_DCMI);
	GPIO_PinAFConfig(GPIOE, GPIO_PinSource6, GPIO_AF_DCMI);
	
	/* DCMI GPIO configuration **************************************************/
	/* HSYNC(PA4) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP ;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	/* VSYNC(PB7) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

	/* PCLK(PA6) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	
	/* D0, D2, D3 (PC6/8/9) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_8 | GPIO_Pin_9;
	GPIO_Init(GPIOC, &GPIO_InitStructure);

	/* D1 (PA10) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	/* D4, D6, D7 (PE4/5/6) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_6;
	GPIO_Init(GPIOE, &GPIO_InitStructure);

	/* D5 (PB6) */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	
	/* DCMI configuration *******************************************************/
	DCMI_DeInit();

	DCMI_InitStructure.DCMI_CaptureMode = DCMI_CaptureMode_SnapShot;
	DCMI_InitStructure.DCMI_SynchroMode = DCMI_SynchroMode_Hardware;
	DCMI_InitStructure.DCMI_PCKPolarity = DCMI_PCKPolarity_Falling;
	DCMI_InitStructure.DCMI_VSPolarity = DCMI_VSPolarity_High;
	DCMI_InitStructure.DCMI_HSPolarity = DCMI_HSPolarity_Low;
	DCMI_InitStructure.DCMI_CaptureRate = DCMI_CaptureRate_All_Frame;
	DCMI_InitStructure.DCMI_ExtendedDataMode = DCMI_ExtendedDataMode_8b;
	
	DCMI_Init(&DCMI_InitStructure);
	
	// resize captured image
	DCMI_CROPStructure.DCMI_VerticalStartLine = 0;
	DCMI_CROPStructure.DCMI_VerticalLineCount = CAMERA_VER_RES - 1;
	DCMI_CROPStructure.DCMI_HorizontalOffsetCount = 0;
	DCMI_CROPStructure.DCMI_CaptureCount = (CAMERA_HOR_RES * 2) - 1;
	DCMI_CROPConfig(&DCMI_CROPStructure);
	DCMI_CROPCmd(ENABLE);
	
	// enable the module
	DCMI_Cmd(ENABLE);
	
	/* Configures the DMA2 to transfer Data from DCMI to the LCD ****************/
	/* Enable DMA2 clock */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_DMA2, ENABLE);
	
	/* DMA2 Stream1 Configuration */
	DMA_DeInit(DMA2_Stream1);
	
	DMA_InitStructure.DMA_Channel = DMA_Channel_1;
	DMA_InitStructure.DMA_PeripheralBaseAddr = DCMI_DR_ADDRESS;	
	DMA_InitStructure.DMA_Memory0BaseAddr = EXT_SRAM_BASE_ADDRESS;
	DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralToMemory;
	DMA_InitStructure.DMA_BufferSize = (CAMERA_HOR_RES * CAMERA_VER_RES * 2) / sizeof(Int32U);
	DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
	DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
	DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Word;
	DMA_InitStructure.DMA_MemoryDataSize = DMA_PeripheralDataSize_Word;
	DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
	DMA_InitStructure.DMA_Priority = DMA_Priority_High;
	DMA_InitStructure.DMA_FIFOMode = DMA_FIFOMode_Enable;
	DMA_InitStructure.DMA_FIFOThreshold = DMA_FIFOThreshold_Full;
	DMA_InitStructure.DMA_MemoryBurst = DMA_MemoryBurst_Single;
	DMA_InitStructure.DMA_PeripheralBurst = DMA_PeripheralBurst_Single;
	
	DMA_Init(DMA2_Stream1, &DMA_InitStructure);

	/* Enable DMA transfer */
	DMA_Cmd(DMA2_Stream1, ENABLE);
}

static void local_I2C_Initialize(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	
	/* Enable the software I2C Clock */
	RCC_AHB1PeriphClockCmd(SOFT_I2C_PORT_CLOCK, ENABLE);
	
	/* Configure the SDA and SCL pins */
	// RG12 - SDA
	// RG10 - SCL
	GPIO_InitStructure.GPIO_Pin = SOFT_I2C_SCL_PIN | SOFT_I2C_SDA_PIN;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_Init(SOFT_I2C_PORT, &GPIO_InitStructure);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);

	local_I2C_Delay();
}

static void local_I2C_Delay(void)
{
	Int32U d = 10000;
	while(d--) {
	}	
}	

static void local_I2C_Start(void)
{
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	
}

static void local_I2C_Stop(void)
{
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
}

static BYTE local_I2C_ReadByte(char ack)
{
	BYTE data = 0;
	char i;
	
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET); // make input
	for(i = 0; i < 8; i++) {
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
		local_I2C_Delay();
		data |= GPIO_ReadInputDataBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN) & 0x01;
		if(i != 7)
			data <<= 1;
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	}
	
	// issue the ack
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, ack ? Bit_SET : Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
	
	return data;
}

// returns ack state, 0 means acknowledged
static char local_I2C_WriteByte(BYTE data)
{
	char i;

	// send the 8 bits
	for(i = 0; i < 8; i++) {
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, (data & 0x80) ? Bit_SET : Bit_RESET);
		data <<= 1;
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
		local_I2C_Delay();
		GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	}
	
	// read the ack
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN, Bit_SET);
	local_I2C_Delay();
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_SET);
	local_I2C_Delay();
	i = GPIO_ReadInputDataBit(SOFT_I2C_PORT, SOFT_I2C_SDA_PIN);
	GPIO_WriteBit(SOFT_I2C_PORT, SOFT_I2C_SCL_PIN, Bit_RESET);
	local_I2C_Delay();
	
	return i;
}



