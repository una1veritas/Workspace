/**
  ******************************************************************************
  * @file    gpio_st7735_sdio_bmp/main.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   main program
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
  *  3. Neither the name of the copyright holders nor the names of contributors
  *  may be used to endorse or promote products derived from this software
  *  without specific prior written permission.
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
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "platform_config.h"
#include "com_config.h"
#include "delay.h"
#include "st7735.h"
#include "sdio_sd_fat.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Some pictures are shown on ST7735 LCD.\r\n";

FRESULT fsresult;
FATFS myfs;
FIL fhandle;
UINT data_read;

const int8_t filename[4][15] =
  {
      "/STM32.bmp",
      "/ilaw.bmp",
      "/flat.bmp",
      "/graphite.bmp"
  };

typedef struct
{
  uint16_t   bfType;
  uint32_t   bfSize;
  uint16_t   bfReserved1;
  uint16_t   bfReserved2;
  uint32_t   bfOffBits;
} __attribute__((packed)) BITMAPFILEHEADER ;
BITMAPFILEHEADER BitmapFileHeader ;
typedef struct
{
  uint32_t  biSize;
  int32_t   biWidth;
  int32_t   biHeight;
  uint16_t  biPlanes;
  uint16_t  biBitCount;
  uint32_t  biCompression;
  uint32_t  biSizeImage;
  int32_t   biXPelsPerMeter;
  int32_t   biYPelsPerMeter;
  uint32_t  biClrUsed;
  uint32_t  biClrImportant;
} __attribute__((packed)) BITMAPINFOHEADER;
BITMAPINFOHEADER BitmapInfoHeader;

uint8_t data_buffer[3];
/* Private function prototypes -----------------------------------------------*/
void show_bmp(const int8_t* filename);
/* Public functions -- -------------------------------------------------------*/
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

  // Setting up COM port for Print function
  COM_Configuration();
  //Print welcome message
  cprintf(Welcome_Message);

  //Initialize ST7735 LCD
  ST7735_Init();

  ST7735_Write_Command(0x2C); //memory write

  //Initialize SDIO
  if(Initialize_SDIO_SD() == SD_FAILED)
    {
      cprintf("SDIO initialization failed.");
      while(1);
    }

  //mount FAT file system
  fsresult = f_mount(0, &myfs);
  if (fsresult != FR_OK)
    {
      cprintf("FAT file system mounting failed.\r\n");
      while(1);
    }

  uint8_t i;
  while(1)
    {
      for (i=0;i<4;i++)
        {
          show_bmp(filename[i]);
          delay_ms(1000);
        }
    }
}

/**
  * @brief  show bitmap file on ST7735 LCD
  * @param  filename : File name of 128 x 128 x 24bit bitmap file
  * @retval : None
  */
void show_bmp(const int8_t* filename)
{
  //Set RAM order inverted
  ST7735_Write_Command(0x36);//MV,MX,MY,RGB
  ST7735_Write_Data(0b01001000);

  //Set RAM address for inverted mode
  ST7735_Write_Command(0x2A);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x02);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x81);

  ST7735_Write_Command(0x2B);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x01);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x80);

  //open a bmp file
  fsresult = f_open(&fhandle, filename, FA_READ);
  if (fsresult != FR_OK)
    {
      cprintf("File open failed. \r\n");
      while(1);
    }
  else
    {
      cprintf("File name opend is :");
      cprintf(filename);
      cprintf("\r\n");
    }

  fsresult = f_read(&fhandle, &BitmapFileHeader, sizeof BitmapFileHeader, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }
  cprintf("File size is : %u bytes\r\n", BitmapFileHeader.bfSize);

  fsresult = f_read(&fhandle, &BitmapInfoHeader, sizeof BitmapInfoHeader, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }
  cprintf("Width is : %u pixel\r\n", BitmapInfoHeader.biWidth);
  cprintf("Height is : %u pixel\r\n", BitmapInfoHeader.biHeight);
  cprintf("Number of bits per pixel is : %u bits\r\n", BitmapInfoHeader.biBitCount);

  //RAMWR (2Ch): Memory Write
  ST7735_Write_Command(0x2C);

  uint16_t i;
  for (i=0;i<128*128;i++)
    {
      fsresult = f_read(&fhandle, &data_buffer, sizeof data_buffer, &data_read);
      if (fsresult != FR_OK)
        {
          cprintf("File read failed.\r\n");
          while(1);
        }
      ST7735_Write_RGB(data_buffer[2]>>3,data_buffer[1]>>2,data_buffer[0]>>3);
    }

  //close a BMP file
  fsresult = f_close(&fhandle);
  if (fsresult != FR_OK)
    {
      cprintf("File close failed.\r\n");
      while(1);
    }
  else
    {
      cprintf("Done!!\r\n\r\n");
    }

}
