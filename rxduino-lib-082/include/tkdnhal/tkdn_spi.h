/*******************************************************************************
* RXduinoライブラリ & 特電HAL
* 
* このソフトウェアは特殊電子回路株式会社によって開発されたものであり、当社製品の
* サポートとして提供されます。このライブラリは当社製品および当社がライセンスした
* 製品に対して使用することができます。
* このソフトウェアはあるがままの状態で提供され、内容および動作についての保障はあ
* りません。弊社はファイルの内容および実行結果についていかなる責任も負いません。
* お客様は、お客様の製品開発のために当ソフトウェアのソースコードを自由に参照し、
* 引用していただくことができます。
* このファイルを単体で第三者へ開示・再配布・貸与・譲渡することはできません。
* コンパイル・リンク後のオブジェクトファイル(ELF ファイルまたはMOT,SRECファイル)
* であって、デバッグ情報が削除されている場合は第三者に再配布することができます。
* (C) Copyright 2011-2012 TokushuDenshiKairo Inc. 特殊電子回路株式会社
* http://rx.tokudenkairo.co.jp/
*******************************************************************************/

#ifndef __H_TKDN_SPI
#define __H_TKDN_SPI

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

//各種関数

typedef enum {
	SPI_PORT_NONE         = 255,
	SPI_PORT_CS0_DUINO    = 0, // CS0
	SPI_PORT_CS1_MARY1    = 1, // CS1
	SPI_PORT_CS2_MARY2    = 2, // CS2
	SPI_PORT_CS3_ROM      = 3, // CS3
	SPI_PORT_SDMMC        = 4, // ext
	SPI_SSLB0             = 0, // CS0
	SPI_SSLB1             = 1, // CS1
	SPI_PORT_RAXINO_EXT   = 0, // CS0
	SPI_PORT_RAXINO_SDMMC = 1, // CS1
	SPI_PORT_RAXINO_ROM   = 2, // CS2
	SPI_PORT_RAXINO_ACCEL = 3, // CS3
} SPI_PORT;

typedef enum {SPI_LSBFIRST, SPI_MSBFIRST} SPI_BIT_ORDER;
typedef enum {SPI_CLOCK_DIV2,  SPI_CLOCK_DIV4,  SPI_CLOCK_DIV8, SPI_CLOCK_DIV16,
              SPI_CLOCK_DIV32, SPI_CLOCK_DIV64, SPI_CLOCK_DIV128 } SPI_CLK_DIVIDER;
typedef enum {SPI_MODE0 , SPI_MODE1 , SPI_MODE2 , SPI_MODE3 } SPI_DATA_MODE;

TKDN_HAL
void spi_init(void);

TKDN_HAL
void spi_terminate(void);

TKDN_HAL
void spi_set_port(SPI_PORT port);

TKDN_HAL
void spi_set_bit_length(int bit_length);

TKDN_HAL
void spi_set_bit_order(SPI_BIT_ORDER bit_order) ;

TKDN_HAL
void spi_set_clock_divider(SPI_CLK_DIVIDER divider);

TKDN_HAL
void spi_set_data_mode(SPI_DATA_MODE mode);

TKDN_HAL
unsigned long spi_transfer(unsigned long txbyte);

#ifdef __cplusplus
 }
#endif

#endif // __H_TKDN_SPI
