#ifndef BOARDDEF_H
#define BOARDDEF_H

// これはボード固有のカスタマイズを行うためのヘッダファイルです
// トップディレクトリでmake use_***を実行すると自動的に書き換わります
// RXduinoのSerialクラスが使うデフォルトのポート番号

#define  CPU_IS_RX62N
#define  ROMSIZE           (512*1024)
#define  RAMSIZE           (96*1024)
#define  DEFAULT_SERIAL     SCI_USB0
#define  USBVCOM_VID        0x2129
#define  USBVCOM_PID        0x0401

#include "tkdn_hal.h"

// ボードに応じた定数の設定
#ifndef TARGET_BOARD
  #define TARGET_BOARD        BOARD_ULT62N0_SDRAM
#endif
#define CPU_TYPE            RX62N

// ピン番号マクロ
#define PIN_P00      0
#define PIN_P01      1
#define PIN_P02      2
#define PIN_P03      3
#define PIN_P05      4
#define PIN_P07      5
#define PIN_P20      6
#define PIN_P21      7
#define PIN_P32      8
#define PIN_P33      9
#define PIN_P34      10
#define PIN_P35      11
#define PIN_P10      12
#define PIN_P11      13
#define PIN_P12      14
#define PIN_P13      15
#define PIN_P73      16 // CS3
#define PIN_P40      17
#define PIN_P41      18
#define PIN_P42      19
#define PIN_P43      20
#define PIN_P44      21
#define PIN_P45      22
#define PIN_P46      23
#define PIN_P47      24
#define PIN_PC0      25
#define PIN_PC1      26
#define PIN_PC2      27
#define PIN_PC3      28
#define PIN_PC4      29
#define PIN_PC5      30
#define PIN_PC6      31
#define PIN_PC7      32
#define PIN_P60      33 // CS0
#define PIN_P22      34 // EDREQ0
#define PIN_P23      35 // EDACK0
#define PIN_P24      36 // EDREQ1
#define PIN_P25      37 // EDACK1
#define PIN_P52      38
#define PIN_P51      39
#define PIN_P50      40
#define PIN_P53      41 // BCLK
#define PIN_PD0      42 // D0
#define PIN_PD1      43 // D1
#define PIN_PD2      44 // D2
#define PIN_PD3      45 // D3
#define PIN_PD4      46 // D4
#define PIN_PD5      47 // D5
#define PIN_PD6      48 // D6
#define PIN_PD7      49 // D7
#define PIN_PE0      50 // D8
#define PIN_PE1      51 // D9
#define PIN_PE2      52 // D10
#define PIN_PE3      53 // D11
#define PIN_PE4      54 // D12
#define PIN_PE5      55 // D13
#define PIN_PE6      56 // D14
#define PIN_PE7      57 // D15

#define PIN_CS3      16
#define PIN_CS0      33
#define PIN_REQ0     34
#define PIN_ACK0     35
#define PIN_REQ1     36
#define PIN_ACK1     37
#define PIN_BCLK     41
#define PIN_BUSD0    42
#define PIN_BUSD1    43
#define PIN_BUSD2    44
#define PIN_BUSD3    45
#define PIN_BUSD4    46
#define PIN_BUSD5    47
#define PIN_BUSD6    48
#define PIN_BUSD7    49
#define PIN_BUSD8    50
#define PIN_BUSD9    51
#define PIN_BUSD10   52
#define PIN_BUSD11   53
#define PIN_BUSD12   54
#define PIN_BUSD13   55
#define PIN_BUSD14   56
#define PIN_BUSD15   57

#endif /* BOARDDEF_H */

#ifdef COMPILE_TKDNGPIO
const unsigned char GPIO_PINMAP[] =
{
	_PORT0 , _BIT0, // 0
	_PORT0 , _BIT1, // 1
	_PORT0 , _BIT2, // 2
	_PORT0 , _BIT3, // 3
	_PORT0 , _BIT5, // 4
	_PORT0 , _BIT7, // 5
	_PORT2 , _BIT0, // 6
	_PORT2 , _BIT1, // 7
	_PORT3 , _BIT2, // 8
	_PORT3 , _BIT3, // 9
	_PORT3 , _BIT4, // 10
	_PORT3 , _BIT5, // 11
	_PORT1 , _BIT0, // 12
	_PORT1 , _BIT1, // 13
	_PORT1 , _BIT2, // 14
	_PORT1 , _BIT3, // 15
	_PORT7 , _BIT3, // 16
	_PORT4 , _BIT0, // 17
	_PORT4 , _BIT1, // 18
	_PORT4 , _BIT2, // 19
	_PORT4 , _BIT3, // 20
	_PORT4 , _BIT4, // 21
	_PORT4 , _BIT5, // 22
	_PORT4 , _BIT6, // 23
	_PORT4 , _BIT7, // 24
	_PORTC , _BIT0, // 25
	_PORTC , _BIT1, // 26
	_PORTC , _BIT2, // 27
	_PORTC , _BIT3, // 28
	_PORTC , _BIT4, // 29
	_PORTC , _BIT5, // 30
	_PORTC , _BIT6, // 31
	_PORTC , _BIT7, // 32
	_PORT6 , _BIT0, // 33
	_PORT2 , _BIT2, // 34 // EDREQ0
	_PORT2 , _BIT3, // 35 // EDACK0
	_PORT2 , _BIT4, // 36 // EDREQ1
	_PORT2 , _BIT5, // 37 // EDACK1
	_PORT5 , _BIT2, // 38
	_PORT5 , _BIT1, // 39
	_PORT5 , _BIT0, // 40
	_PORT5 , _BIT3, // 41 // BCLK
	_PORTD , _BIT0, // 42
	_PORTD , _BIT1, // 43
	_PORTD , _BIT2, // 44
	_PORTD , _BIT3, // 45
	_PORTD , _BIT4, // 46
	_PORTD , _BIT5, // 47
	_PORTD , _BIT6, // 48
	_PORTD , _BIT7, // 49
	_PORTE , _BIT0, // 50
	_PORTE , _BIT1, // 51
	_PORTE , _BIT2, // 52
	_PORTE , _BIT3, // 53
	_PORTE , _BIT4, // 54
	_PORTE , _BIT5, // 55
	_PORTE , _BIT6, // 56
	_PORTE , _BIT7, // 57
};


const unsigned char SFP_PINMAP[] =
{
	_PORT5 , _BIT5, //          LED0
	_PORT5 , _BIT6, //          LED1
	_PORT5 , _BIT7, //          LED2
	_PORT8 , _BIT4, //          LED3
	_PORT8 , _BIT4, //          BUZZ
	_PORT8 , _BIT5, //          SW
	_PORT3 , _BIT1, //          SPI_CS0
	_PORTC , _BIT0, //          SPI_CS1
	_PORTC , _BIT1, //          SPI_CS2
	_PORTC , _BIT2, //          SPI_CS3
};

#endif
