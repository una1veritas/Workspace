#ifndef BOARDDEF_H
#define BOARDDEF_H

// これはボード固有のカスタマイズを行うためのヘッダファイルです
// トップディレクトリでmake use_***を実行すると自動的に書き換わります
// RXduinoのSerialクラスが使うデフォルトのポート番号

#define  CPU_IS_RX62N
#define  ROMSIZE           (768*1024)
#define  RAMSIZE           (96*1024)
#define  DEFAULT_SERIAL     SCI_USB0
#define  USBVCOM_VID        0x2129
#define  USBVCOM_PID        0x0501

#include "tkdn_hal.h"

// ボードに応じた定数の設定
#ifndef TARGET_BOARD
  #define TARGET_BOARD        BOARD_RAXINO
#endif
#define CPU_TYPE            RX62N

// ピン番号マクロ
#define PIN_P21      0
#define PIN_P20      1
#define PIN_P22      2
#define PIN_P23      3
#define PIN_P24      4
#define PIN_P25      5
#define PIN_P32      6
#define PIN_P33      7
#define PIN_PC2      8
#define PIN_PC3      9
#define PIN_PC4      10
#define PIN_PC6      11
#define PIN_PC7      12
#define PIN_PC5      13
#define PIN_P40      14
#define PIN_P41      15
#define PIN_P42      16
#define PIN_P43      17
#define PIN_P44      18
#define PIN_P45      19
#define PIN_P46      20
#define PIN_P47      21
#define PIN_PC0      22
#define PIN_PC1      23
#define PIN_P50      24
#define PIN_P51      25
#define PIN_P52      26
#define PIN_P53      27
#define PIN_P54      28
#define PIN_P55      29
#define PIN_P12      30
#define PIN_P13      31
#define PIN_P14      32
#define PIN_P15      33
#define PIN_P16      34
#define PIN_P17      35
#define PIN_PD0      36
#define PIN_PD1      37
#define PIN_PD2      38
#define PIN_PD3      39
#define PIN_PD4      40
#define PIN_PD5      41
#define PIN_PD6      42
#define PIN_PD7      43
#define PIN_PE0      44
#define PIN_PE1      45
#define PIN_PE2      46
#define PIN_PE3      47
#define PIN_PE4      48
#define PIN_PE5      49
#define PIN_PE6      50
#define PIN_PE7      51
#define PIN_P07      52
#define PIN_P05      53
#define PIN_P35      54
#define PIN_NMI      54

#endif /* BOARDDEF_H */

// 各種ライブラリで使われる
//#define PIN_SDMMC_CS    PIN_PC0
//#define SDMMC_RSPI_CH   0
//#define PHY_ADDR        0x00
#define ETH_PORT_IS_AB
#define USB_HOST_CH     0
#define USB_FUNC_CH     0

// 以下、ボード上のピン配置を定義する

#ifdef COMPILE_TKDNGPIO
const unsigned char GPIO_PINMAP[] =
{
	_PORT2 , _BIT1, // Arduino-D0  : P21
	_PORT2 , _BIT0, // Arduino-D1  : P20
	_PORT2 , _BIT2, // Arduino-D2  : P22
	_PORT2 , _BIT3, // Arduino-D3  : P23
	_PORT2 , _BIT4, // Arduino-D4  : P24
	_PORT2 , _BIT5, // Arduino-D5  : P25
	_PORT3 , _BIT2, // Arduino-D6  : P32
	_PORT3 , _BIT3, // Arduino-D7  : P33
	_PORTC , _BIT2, // Arduino-D8  : PC2
	_PORTC , _BIT3, // Arduino-D9  : PC3
	_PORTC , _BIT4, // Arduino-D10 : PC4
	_PORTC , _BIT6, // Arduino-D11 : PC6
	_PORTC , _BIT7, // Arduino-D12 : PC7
	_PORTC , _BIT5, // Arduino-D13 : PC5
	_PORT4 , _BIT0, // Arduino-D14 : P40
	_PORT4 , _BIT1, // Arduino-D15 : P41
	_PORT4 , _BIT2, // Arduino-D16 : P42
	_PORT4 , _BIT3, // Arduino-D17 : P43
	_PORT4 , _BIT4, // Arduino-D18 : P44
	_PORT4 , _BIT5, // Arduino-D19 : P45
	_PORT4 , _BIT6, //          20 : P46
	_PORT4 , _BIT7, //          21 : P47
	_PORTC , _BIT0, //          22 : PC0
	_PORTC , _BIT1, //          23 : PC1
	_PORT5 , _BIT0, //          24 : P50
	_PORT5 , _BIT1, //          25 : P51
	_PORT5 , _BIT2, //          26 : P52
	_PORT5 , _BIT3, //          27 : P53
	_PORT5 , _BIT4, //          28 : P54
	_PORT5 , _BIT5, //          29 : P55
	_PORT1 , _BIT2, //          30 : P12
	_PORT1 , _BIT3, //          31 : P13
	_PORT1 , _BIT4, //          32 : P14
	_PORT1 , _BIT5, //          33 : P15
	_PORT1 , _BIT6, //          34 : P16
	_PORT1 , _BIT7, //          35 : P17
	_PORTD , _BIT0, //          36 : PD0
	_PORTD , _BIT1, //          37 : PD1
	_PORTD , _BIT2, //          38 : PD2
	_PORTD , _BIT3, //          39 : PD3
	_PORTD , _BIT4, //          40 : PD4
	_PORTD , _BIT5, //          41 : PD5
	_PORTD , _BIT6, //          42 : PD6
	_PORTD , _BIT7, //          43 : PD7
	_PORTE , _BIT0, //          44 : PE0
	_PORTE , _BIT1, //          45 : PE1
	_PORTE , _BIT2, //          46 : PE2
	_PORTE , _BIT3, //          47 : PE3
	_PORTE , _BIT4, //          48 : PE4
	_PORTE , _BIT5, //          49 : PE5
	_PORTE , _BIT6, //          50 : PE6
	_PORTE , _BIT7, //          51 : PE7
	_PORT0 , _BIT7, //          52 : P07
	_PORT0 , _BIT5, //          53 : P05
	_PORT3 , _BIT5, //          54 : P35/NMI
};

const unsigned char SFP_PINMAP[] =
{
	_PORTA , _BIT0, //          LED0
	_PORTA , _BIT1, //          LED1
	_PORTA , _BIT2, //          LED2
	_PORTA , _BIT6, //          LED3
	_PORTA , _BIT6, //          BUZZ
	_PORTA , _BIT7, //          SW
	_PORTC , _BIT4, //          SPI_CS0
	_PORTC , _BIT0, //          SPI_CS1
	_PORTC , _BIT1, //          SPI_CS2
	_PORTC , _BIT2, //          SPI_CS3
};

#endif
