//=========================================================
// LPC1114 Project
//=========================================================
// File Name : oled.c
// Function  : OLED Control
//---------------------------------------------------------
// Rev.01 2010.08.13 Munetomo Maruyama
// Rev.02 2011.03.20 Munetomo Maruyama
//        - Add Orientation Mode
//        - Add OLED_GRY (Color Definition)
//        - Correct OLED_printf_Color()
//        - Correct OLED SPI Clock (24MHz-->12MHz)
//---------------------------------------------------------
// Copyright (C) 2010-2011 Munetomo Maruyama
//=========================================================
// ---- License Information -------------------------------
// Anyone can FREELY use this code fully or partially
// under conditions shown below.
// 1. You may use this code only for individual purpose,
//    and educational purpose.
//    Do not use this code for business even if partially.
// 2. You can copy, modify and distribute this code.
// 3. You should remain this header text in your codes
//   including Copyright credit and License Information.
// 4. Your codes should inherit this license information.
//=========================================================
// ---- Patent Notice -------------------------------------
// I have not cared whether this system (hw + sw) causes
// infringement on the patent, copyright, trademark,
// or trade secret rights of others. You have all
// responsibilities for determining if your designs
// and products infringe on the intellectual property
// rights of others, when you use technical information
// included in this system for your business.
//=========================================================
// ---- Disclaimers ---------------------------------------
// The function and reliability of this system are not
// guaranteed. They may cause any damages to loss of
// properties, data, money, profits, life, or business.
// By adopting this system even partially, you assume
// all responsibility for its use.
//=========================================================

#ifdef __USE_CMSIS
#include "LPC11xx.h"
#endif



#include <stdarg.h>
#include <rxduino.h>
#include "font.h"
#include "mary_gpio.h"
#include "oled.h"

#include "iodefine_gcc62n.h"

//#include "systick.h"
#include "utility.h"

extern CSPI SPI;

//=================
// Font Parameters
//=================
#define OLED_FONT_XSIZE 8
#define OLED_FONT_YSIZE 8

//======================
// Define OLED Commands
//======================
#define OLED_COMMAND 0
#define OLED_DATA    1
//
#define C_SET_COLUMN_ADDRESS 0x0215
#define C_SET_ROW_ADDRESS    0x0275
#define D_START_ADDRESS 1
#define D_END_ADDRESS   2
//
#define C_WRITE_RAM_COMMAND 0x005c
#define C_READ_RAM_COMMAND  0x005d
//
#define C_SET_REMAP_COLOR_DEPTH 0x01a0
#define D_SET_REMAP_COLOR_DEPTH 1
//
#define C_SET_DISPLAY_START_LINE 0x01a1
#define D_SET_DISPLAY_START_LINE 1
//
#define C_SET_DISPLAY_OFFSET 0x01a2
#define D_SET_DISPLAY_OFFSET 1
//
#define C_SET_DISPLAY_MODE_ALL_OFF 0x00a4
#define C_SET_DISPLAY_MODE_ALL_ON  0x00a5
#define C_SET_DISPLAY_MODE_RESET   0x00a6
#define C_SET_DISPLAY_MODE_INVERSE 0x00a7
//
#define C_FUNCTION_SELECTION 0x01ab
#define D_FUNCTION_SELECTION 1
//
#define C_SET_SLEEP_MODE_ON  0x00ae
#define C_SET_SLEEP_MODE_OFF 0x00af
//
#define C_SET_RESET_PRECHARGE_PERIOD 0x01b1
#define D_SET_RESET_PRECHARGE_PERIOD 1
//
#define C_ENHANCE_DRIVING_SCHEME_CAPABILITY 0x03b2
#define D_ENHANCE_DRIVING_SCHEME_CAPABILITY_1 1
#define D_ENHANCE_DRIVING_SCHEME_CAPABILITY_2 2
#define D_ENHANCE_DRIVING_SCHEME_CAPABILITY_3 3
//
#define C_FRONT_CLOCK_DRIVER_OSCILLATOR_FREQUENCY 0x01b3
#define D_FRONT_CLOCK_DRIVER_OSCILLATOR_FREQUENCY 1
//
#define C_SET_SEGMENT_LOW_VOLTAGE 0x03b4
#define D_SET_SEGMENT_LOW_VOLTAGE_1 1 // 0xa0 or 0xa2
#define D_SET_SEGMENT_LOW_VOLTAGE_2 2 // 0xb5
#define D_SET_SEGMENT_LOW_VOLTAGE_3 3 // 0x55
//
#define C_SET_GPIO 0x01b5
#define D_SET_GPIO 1
//
#define C_SET_SECOND_PRECHARGE_PERIOD 0x01b6
#define D_SET_SECOND_PRECHARGE_PERIOD 1
//
#define C_LOOKUP_TABLE_FOR_GRAY_SCALE_PULSE_WIDTH 0x3fb8
static const unsigned char GAMMA_TABLE[63] =
{
    0x02, 0x03, 0x04, 0x05,
    0x06, 0x07, 0x08, 0x09,
    0x0a, 0x0b, 0x0c, 0x0d,
    0x0e, 0x0f, 0x10, 0x11,
    //
    0x12, 0x13, 0x15, 0x17,
    0x19, 0x1b, 0x1d, 0x1f,
    0x21, 0x23, 0x25, 0x27,
    0x2a, 0x2d, 0x30, 0x33,
    //
    0x36, 0x39, 0x3c, 0x3f,
    0x42, 0x45, 0x48, 0x4c,
    0x50, 0x54, 0x58, 0x5c,
    0x60, 0x64, 0x68, 0x6c,
    //
    0x70, 0x74, 0x78, 0x7d,
    0x82, 0x87, 0x8c, 0x91,
    0x96, 0x9b, 0xa0, 0xa5,
    0xaa, 0xaf, 0xb4
};
//
#define C_USE_BUILT_IN_LINEAR_LUT 0x00b9
//
#define C_SET_PRECHARGE_VOLTAGE 0x01bb
#define D_SET_PRECHARGE_VOLTAGE 1
//
#define C_SET_VCOMH_VOLTAGE 0x01be
#define D_SET_VCOMH_VOLTAGE 1
//
#define C_SET_CONTRAST_CURRENT_FOR_COLOR_ABC 0x03c1
#define D_SET_CONTRAST_CURRENT_FOR_COLOR_A 1
#define D_SET_CONTRAST_CURRENT_FOR_COLOR_B 2
#define D_SET_CONTRAST_CURRENT_FOR_COLOR_C 3
//
#define C_MASTER_CONTRAST_CURRENT_CONTROL 0x01c7
#define D_MASTER_CONTRAST_CURRENT_CONTROL 1
//
#define C_SET_MUX_RATIO 0x01ca
#define D_SET_MUX_RATIO 1
//
#define C_SET_COMMAND_LOCK 0x01fd
#define D_SET_COMMAND_LOCK 1
//
#define C_HORIZONTAL_SCROLL 0x0596
#define D_HORIZONTAL_SCROLL_A 1
#define D_HORIZONTAL_SCROLL_B 2
#define D_HORIZONTAL_SCROLL_C 3
#define D_HORIZONTAL_SCROLL_D 4
#define D_HORIZONTAL_SCROLL_E 5
//
#define C_STOP_MOVING  0x009e
#define C_START_MOVING 0x009f

//=============
// Globals
//=============
volatile unsigned long gOLED_printf_Font   = OLED_FONT_SMALL;
volatile unsigned long gOLED_printf_ColorF = OLED_WHT;
volatile unsigned long gOLED_printf_ColorB = OLED_BLK;
volatile unsigned long gOLED_printf_PosX = 0;
volatile unsigned long gOLED_printf_PosY = 0;
//
volatile unsigned long gOELD_Orientation_Mode = OLED_TOP_N;

//======================
// Initialize OLED
//======================
int Init_OLED(char portnum)
{
    volatile unsigned long i;
    unsigned long oled[64];

    // OLED Vcc OFF
    // PIO1_4 : OLED Vcc Power
    //LPC_IOCON->PIO1_4 = 0x00000000; // GPIO, disable pu/pd mos
    //GPIOSetValue(1, 4, 0); // low
    //GPIOSetDir(1, 4, 1); // PIO1_4 out
    //GPIOSetValue(1, 4, 0); // low
    switch(portnum){
		case MARY1:
			MARY1_OLED_VCC_ON_H();
			MARY1_OLED_VCC_ON_OUT();
			MARY1_OLED_VCC_ON_L();

			MARY1_CS_H();
			MARY1_CS_OUT();
			MARY1_CS_L();

			MARY1_OLED_RES_H();
			MARY1_OLED_RES_OUT();
			MARY1_OLED_RES_L();
			for(i=100;i!=0;i--);
			MARY1_OLED_RES_H();
			SPI.begin();
			SPI.setBitLength(9);
			SPI.setBitOrder(MSBFIRST);
			SPI.setClockDivider(SPI_CLOCK_DIV4);
			break;

    	case MARY2:
    		MARY2_OLED_VCC_ON_H();		// OLED_VCC_ON(P42) = H
    		MARY2_OLED_VCC_ON_OUT();	// P42 = output
    		MARY2_OLED_VCC_ON_L();		// OLED_VCC_ON = L

    		MARY2_CS_H();				// OLED_CS(PC1) = L
    		MARY2_CS_OUT();				// OLED_CS = OUTPUT
    		MARY2_CS_L();				// OLED_CS = L

    		MARY2_OLED_RES_H();			//	OLED_RES(P13) = H
    		MARY2_OLED_RES_OUT();		//	OLED_RES = OUTPUT
    		MARY2_OLED_RES_L();			//	OLED_RES = L
    		for(i=100;i!=0;i--);
    		MARY2_OLED_RES_H();			//	OLED_RES(P13) = 1
			SPI.begin();
			SPI.setBitLength(9);
			SPI.setBitOrder(MSBFIRST);
			SPI.setClockDivider(SPI_CLOCK_DIV4);
    		break;
    	default:
    		return MARY_FAIL;
    		break;
    }
    // Display OFF

    oled[OLED_COMMAND] = C_SET_DISPLAY_MODE_ALL_OFF;
    OLED_Send_Command(portnum,oled);
    //
    // Initialization Sequence of OLED
    //
    oled[OLED_COMMAND] = C_SET_COMMAND_LOCK;
    oled[D_SET_COMMAND_LOCK] = 0x12; // unlock
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_COMMAND_LOCK;
    oled[D_SET_COMMAND_LOCK] = 0xb1; // unlock
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_SLEEP_MODE_ON;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_FRONT_CLOCK_DRIVER_OSCILLATOR_FREQUENCY;
    oled[D_FRONT_CLOCK_DRIVER_OSCILLATOR_FREQUENCY] = 0xf1;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_MUX_RATIO;
    oled[D_SET_MUX_RATIO] = 0x7f;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_DISPLAY_OFFSET;
    oled[D_SET_DISPLAY_OFFSET] = 0x00;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_DISPLAY_START_LINE;
    oled[D_SET_DISPLAY_START_LINE] = 0x00;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_REMAP_COLOR_DEPTH;
    oled[D_SET_REMAP_COLOR_DEPTH] = 0x74; // 64k colors
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_GPIO;
    oled[D_SET_GPIO] = 0x00;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_FUNCTION_SELECTION;
    oled[D_FUNCTION_SELECTION] = 0x01;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_SEGMENT_LOW_VOLTAGE;
    oled[D_SET_SEGMENT_LOW_VOLTAGE_1] = 0xa0; // use external VSL
    oled[D_SET_SEGMENT_LOW_VOLTAGE_2] = 0xb5;
    oled[D_SET_SEGMENT_LOW_VOLTAGE_3] = 0x55;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_CONTRAST_CURRENT_FOR_COLOR_ABC;
    oled[D_SET_CONTRAST_CURRENT_FOR_COLOR_A] = 0xc8;
    oled[D_SET_CONTRAST_CURRENT_FOR_COLOR_B] = 0x80;
    oled[D_SET_CONTRAST_CURRENT_FOR_COLOR_C] = 0xc8;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_MASTER_CONTRAST_CURRENT_CONTROL;
    oled[D_MASTER_CONTRAST_CURRENT_CONTROL] = 0x0f;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_LOOKUP_TABLE_FOR_GRAY_SCALE_PULSE_WIDTH;
    for (i = 1; i < 64; i++)
    {
        oled[i] = (unsigned long) GAMMA_TABLE[i - 1];
    }
    OLED_Send_Command(portnum,oled);
    //
  //oled[OLED_COMMAND] = C_USE_BUILT_IN_LINEAR_LUT;
  //OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_RESET_PRECHARGE_PERIOD;
    oled[D_SET_RESET_PRECHARGE_PERIOD] = 0x32;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_ENHANCE_DRIVING_SCHEME_CAPABILITY;
    oled[D_ENHANCE_DRIVING_SCHEME_CAPABILITY_1] = 0xa4;
    oled[D_ENHANCE_DRIVING_SCHEME_CAPABILITY_2] = 0x00;
    oled[D_ENHANCE_DRIVING_SCHEME_CAPABILITY_3] = 0x00;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_PRECHARGE_VOLTAGE;
    oled[D_SET_PRECHARGE_VOLTAGE] = 0x17;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_SECOND_PRECHARGE_PERIOD;
    oled[D_SET_SECOND_PRECHARGE_PERIOD] = 0x01;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_VCOMH_VOLTAGE;
    oled[D_SET_VCOMH_VOLTAGE] = 0x05;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_DISPLAY_MODE_RESET;
    OLED_Send_Command(portnum,oled);
    //
    // Clear Screen
    oled[OLED_COMMAND] = C_SET_COLUMN_ADDRESS;
    oled[D_START_ADDRESS] = 0;
    oled[D_END_ADDRESS]   = 127;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_ROW_ADDRESS;
    oled[D_START_ADDRESS] = 0;
    oled[D_END_ADDRESS]   = 127;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_WRITE_RAM_COMMAND;
    OLED_Send_Command(portnum,oled);
    //
    for (i = 0; i < (128 * 128); i++)
    {
        OLED_Send_Pixel(portnum,OLED_BLK);
    }
    // OLED Vcc ON
    switch(portnum){
    	case MARY1:
    		MARY1_OLED_VCC_ON_H();
    		break;
    	case MARY2:
    		MARY2_OLED_VCC_ON_H();
    		break;
    	default:
    		return MARY_FAIL;
    }

	//Wait_N_Ticks(20); // wait for 200ms
	for(i=100;i!=0;i--);
    //
    // Display ON
    oled[OLED_COMMAND] = C_SET_SLEEP_MODE_OFF;
    OLED_Send_Command(portnum,oled);
    //
    // Dummy Print (to make correct link)
    //OLED_printf(" ");
    OLED_printf_Position(0, 0);
    return MARY_SUCCESS;
}

//==================================
// Initialize OLED with Orientation
//==================================
void Init_OLED_with_Orientation(char portnum,unsigned long mode)
{
    unsigned char command;
    unsigned long oled[64];

    //Init_OLED();
    gOELD_Orientation_Mode = mode;
    //
    command = (mode == OLED_TOP_W)? 0x65 :
              (mode == OLED_TOP_S)? 0x66 :
              (mode == OLED_TOP_E)? 0x77 : 0x74;
    oled[OLED_COMMAND] = C_SET_REMAP_COLOR_DEPTH;
    oled[D_SET_REMAP_COLOR_DEPTH] = command;
    OLED_Send_Command(portnum,oled);
}

//=====================
// OLED Send Command
//=====================
void OLED_Send_Command(char portnum, unsigned long *oled)
{
    unsigned long i;
    unsigned long count;
    unsigned long command;
    unsigned long data;

    count = (oled[OLED_COMMAND] >> 8) & 0x0ff;
    command = (oled[OLED_COMMAND] & 0x0ff) | 0x000;
	if(portnum == MARY1) SPI.port = SPI_PORT_CS1_MARY1;
	if(portnum == MARY2) SPI.port = SPI_PORT_CS2_MARY2;
    SPI.transfer(command);
    //
    i = 0;
    while(i < count)
    {
        data = oled[OLED_DATA + i] | 0x100;
        SPI.transfer(data);
        i++;
    }
}

//====================
// OLED Send a Pixel
//====================
void OLED_Send_Pixel(char portnum,unsigned long color)
{
    unsigned long data1, data2;

    data1 = ((color >> 8) & 0xff) | 0x100;
    data2 = (color & 0x0ff) | 0x100;

	if(portnum == MARY1) SPI.port = SPI_PORT_CS1_MARY1;
	if(portnum == MARY2) SPI.port = SPI_PORT_CS2_MARY2;
    SPI.transfer(data1);
    SPI.transfer(data2);
}

void OLED_move_position(char portnum,int x,int y)
{
	unsigned long oled[64];
    oled[OLED_COMMAND] = C_SET_COLUMN_ADDRESS;
    oled[D_START_ADDRESS] = x;
    oled[D_END_ADDRESS]   = 127;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_SET_ROW_ADDRESS;
    oled[D_START_ADDRESS] = y;
    oled[D_END_ADDRESS]   = 127;
    OLED_Send_Command(portnum,oled);
    //
    oled[OLED_COMMAND] = C_WRITE_RAM_COMMAND;
    OLED_Send_Command(portnum,oled);
}

//=================
// OLED_Draw_Text
//=================
unsigned long OLED_Draw_Text_Small(char portnum,char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b)
{
    while(*pStr != '\0')
    {
        OLED_Draw_Char(portnum,*pStr, posx, posy, color_f, color_b, OLED_FONT_SMALL);
        pStr++;
        posx++;
    }
    return posx;
}
unsigned long OLED_Draw_Text_Medium(char portnum,char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b)
{
    while(*pStr != '\0')
    {
        OLED_Draw_Char(portnum,*pStr, posx, posy, color_f, color_b, OLED_FONT_MEDIUM);
        pStr++;
        posx++;
    }
    return posx;
}
unsigned long OLED_Draw_Text_LARGE(char portnum,char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b)
{
    while(*pStr != '\0')
    {
        OLED_Draw_Char(portnum,*pStr, posx, posy, color_f, color_b, OLED_FONT_LARGE);
        pStr++;
        posx++;
    }
    return posx;
}

//=======================
// OLED Draw a Character
//=======================
// scale should be 0, 1 or 2
void OLED_Draw_Char(char portnum,char ch, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b, unsigned long scale)
{
    unsigned long oled[8];
    unsigned long x0, y0;
    unsigned long xsize, ysize;
    unsigned long x, y;
    unsigned long xfont, yfont;
    unsigned long pixel;
    unsigned long color;
    //
    unsigned long col0, col1, row0, row1;

    ch = (ch < 0x20)? 0x20 : (ch > 0x7f)? 0x7f : ch;
    //
    x0 = posx * (OLED_FONT_XSIZE << scale);
    y0 = posy * (OLED_FONT_YSIZE << scale);
    //
    xsize = OLED_FONT_XSIZE * (1 << scale);
    ysize = OLED_FONT_YSIZE * (1 << scale);
    //
    if ((x0 <= (128 - xsize)) && (y0 <= (128 - ysize)))
    {
        col0 = (gOELD_Orientation_Mode == OLED_TOP_W)? y0       :
               (gOELD_Orientation_Mode == OLED_TOP_S)? x0 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? y0 : x0;
        col1 = (gOELD_Orientation_Mode == OLED_TOP_W)? y0 + ysize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? x0 + xsize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? y0 + ysize - 1 : x0 + xsize - 1;
        row0 = (gOELD_Orientation_Mode == OLED_TOP_W)? x0 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? y0 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? x0 : y0;
        row1 = (gOELD_Orientation_Mode == OLED_TOP_W)? x0 + xsize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? y0 + ysize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? x0 + xsize - 1 : y0 + ysize - 1;
        //
        oled[OLED_COMMAND] = C_SET_COLUMN_ADDRESS;
        oled[D_START_ADDRESS] = (col0 > 127)? 127 : col0;
        oled[D_END_ADDRESS]   = (col1 > 127)? 127 : col1;
        OLED_Send_Command(portnum,oled);
        //
        oled[OLED_COMMAND] = C_SET_ROW_ADDRESS;
        oled[D_START_ADDRESS] = (row0 > 127)? 127 : row0;
        oled[D_END_ADDRESS]   = (row1 > 127)? 127 : row1;
        OLED_Send_Command(portnum,oled);
        //
        oled[OLED_COMMAND] = C_WRITE_RAM_COMMAND;
        OLED_Send_Command(portnum,oled);
        //
        for (y = 0; y < ysize; y++)
        {
            for (x = 0; x < xsize; x++)
            {
                xfont = x >> scale;
                yfont = y >> scale;
                pixel = FONT[((unsigned long) ch - 0x20) * 8 + yfont];
                pixel = (pixel >> (OLED_FONT_XSIZE - 1 - xfont)) & 0x01;
                color = (pixel == 1)? color_f : color_b;
                OLED_Send_Pixel(portnum,color);
            }
        }
    }
}

//================
// OLED Fill Rect
//================
void OLED_Fill_Rect(char portnum,signed long x0, signed long y0, signed long xsize, signed long ysize, unsigned long color)
{
    unsigned long i;
    unsigned long oled[8];
    unsigned long pixels;
    signed long x1, y1;
    //
    unsigned long col0, col1, row0, row1;

    x1 = x0 + xsize;
    y1 = y0 + ysize;
    x0 = (x0 <   0)?  0 : x0;
    x0 = (x0 < 128)? x0 : 128;
    y0 = (y0 <   0)?  0 : y0;
    y0 = (y0 < 128)? y0 : 128;
    xsize = x1 - x0;
    ysize = y1 - y0;

    if ((x0 < 128) && (y0 < 128) && (xsize > 0) && (ysize > 0))
    {
        col0 = (gOELD_Orientation_Mode == OLED_TOP_W)? y0       :
               (gOELD_Orientation_Mode == OLED_TOP_S)? x0 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? y0 : x0;
        col1 = (gOELD_Orientation_Mode == OLED_TOP_W)? y0 + ysize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? x0 + xsize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? y0 + ysize - 1 : x0 + xsize - 1;
        row0 = (gOELD_Orientation_Mode == OLED_TOP_W)? x0 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? y0 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? x0 : y0;
        row1 = (gOELD_Orientation_Mode == OLED_TOP_W)? x0 + xsize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_S)? y0 + ysize - 1 :
               (gOELD_Orientation_Mode == OLED_TOP_E)? x0 + xsize - 1 : y0 + ysize - 1;
        //
        oled[OLED_COMMAND] = C_SET_COLUMN_ADDRESS;
        oled[D_START_ADDRESS] = (col0 > 127)? 127 : col0;
        oled[D_END_ADDRESS]   = (col1 > 127)? 127 : col1;
        OLED_Send_Command(portnum,oled);
        //
        oled[OLED_COMMAND] = C_SET_ROW_ADDRESS;
        oled[D_START_ADDRESS] = (row0 > 127)? 127 : row0;
        oled[D_END_ADDRESS]   = (row1 > 127)? 127 : row1;
        OLED_Send_Command(portnum,oled);
        //
        oled[OLED_COMMAND] = C_WRITE_RAM_COMMAND;
        OLED_Send_Command(portnum,oled);
        //
        pixels = xsize * ysize;
        for (i = 0; i < pixels; i++)
        {
            OLED_Send_Pixel(portnum,color);
        }
    }
}

//=================
// OLED Draw Dot
//=================
void OLED_Draw_Dot(char portnum,signed long x, signed long y, signed long size, unsigned long color)
{
    OLED_Fill_Rect(portnum,x, y, size, size, color);
}

//===================
// OLED Clear Screen
//===================
void OLED_Clear_Screen(char portnum,unsigned long color)
{
    OLED_Fill_Rect(portnum, 0, 0, 180, 180, color);
}

//====================
// OLED Num4 to Char
//====================
char OLED_Num4_to_Char(unsigned long num4)
{
    num4 = num4 & 0x0f;
    return (num4 < 0x0a)? (num4 + '0') : (num4 - 0x0a + 'A');
}

//=========================
// OLED Draw Hex (small)
//=========================
unsigned long OLED_Draw_Hex(char portnum,unsigned long bitlen, unsigned long hex, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b)
{
    unsigned long i;
    unsigned long num4;
    char ch;
    char *head = "0x";
    unsigned long len;

    len = bitlen >> 2;
    posx = OLED_Draw_Text_Small(portnum,head, posx, posy, color_f, color_b);
    for (i = 0; i < len; i++)
    {
        num4 = (hex >> (((len - 1) - i) * 4)) & 0x0f;
        ch = OLED_Num4_to_Char(num4);
        OLED_Draw_Char(portnum, ch, posx, posy, color_f, color_b, OLED_FONT_SMALL);
        posx++;
    }
    return posx;
}

//====================
// OLED Make Color
//====================
//      RGB num
// BLK  000 0000-003f (< 64)
// BLU  001 0040-007f (<128)
// CYN  011 0080-00bf (<192)
// GRN  010 00c0-00ff (<256)
// YEL  110 0100-013f (<320)
// WHT  111 0140-017f (<384)
// MAG  101 0180-01bf (<448)
// RED  100 01c0-01ff (<512)
// BLK  000
unsigned long OLED_Make_Color(unsigned long num)
{
    unsigned long grn, red, blu;
    unsigned long zero, full, rise, fall;
    unsigned long color;

    num = num & 0x01ff; // 0-511
    rise = num & 0x3f;
    fall = 0x3f - (num & 0x3f);
    zero = 0x00;
    full = 0x3f;

         if (num <  64) {red = zero; grn = zero; blu = rise;}
    else if (num < 128) {red = zero; grn = rise; blu = full;}
    else if (num < 192) {red = zero; grn = full; blu = fall;}
    else if (num < 256) {red = rise; grn = full; blu = zero;}
    else if (num < 320) {red = full; grn = full; blu = rise;}
    else if (num < 384) {red = full; grn = fall; blu = full;}
    else if (num < 448) {red = full; grn = zero; blu = fall;}
    else                {red = fall; grn = zero; blu = zero;}
    //
    red = red >> 1;
    blu = blu >> 1;
    color = (red << 11) + (grn << 5) + (blu << 0);
    return color;
}

//===========================
// OLED Set printf() Font
//===========================
void OLED_printf_Font(unsigned long font)
{
    gOLED_printf_Font  = font;
}

//===========================
// OLED Set printf() Color
//===========================
void OLED_printf_Color(unsigned long color_f, unsigned long color_b) // corrected 2011.03.20 MM
{
    gOLED_printf_ColorF = color_f;
    gOLED_printf_ColorB = color_b;
}

//===========================
// OLED Set printf() Position
//===========================
void OLED_printf_Position(unsigned long posx, unsigned long posy)
{
    gOLED_printf_PosX = posx;
    gOLED_printf_PosY = posy;
}

//=====================
// OLED printf
//=====================
void OLED_printf(char portnum,const char *format, ...)
{
    va_list ap;
    unsigned char buf[256];
    unsigned char *pStr;

    va_start(ap, format);
    xvsnprintf(buf, 256, format, ap);
    va_end(ap);

    pStr = buf;
    while(*pStr != '\0')
    {
        if (*pStr == '\n')
        {
            gOLED_printf_PosX = 0;
            gOLED_printf_PosY++;
        }
        else
        {
            OLED_Draw_Char(portnum,*pStr, gOLED_printf_PosX, gOLED_printf_PosY,
                    gOLED_printf_ColorF, gOLED_printf_ColorB, gOLED_printf_Font);
            gOLED_printf_PosX++;
        }
        pStr++;
        //
        if (gOLED_printf_PosX >= (128 / (OLED_FONT_XSIZE << gOLED_printf_Font)))
        {
            gOLED_printf_PosX = 0;
            gOLED_printf_PosY++;
        }
        if (gOLED_printf_PosY >= (128 / (OLED_FONT_YSIZE << gOLED_printf_Font)))
        {
            gOLED_printf_PosY = 0;
        }
    }
}


//=========================================================
// End of Program
//=========================================================
