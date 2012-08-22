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

#ifndef __H_TKDN_HAL
#define __H_TKDN_HAL

// オブジェクトコードを節約したい場合は以下のマクロを有効にしてライブラリを再ビルドすること
// ※ 下記の2つをコメントアウトすると約6kB節約できる
//#define DISABLE_USBVCOM // USB仮想COMポートを使わない場合
//#define DISABLE_ETHER   // イーサネットコントローラを使わない場合


// これ以降、ユーザがカスタマイズする場所はない

#define TKDN_HAL 

#ifdef __GNUC__
  #define __INTTERUPT_FUNC __attribute__ ((interrupt))
#endif
#ifdef __RENESAS
  #define __INTTERUPT_FUNC
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

////////////////////////////////////////////////////////////////
// 以下の番号は特殊電子回路㈱によって割り当てられる
// 勝手に変更してはならない
#define RX62N                0x00000062
#define RX63N                0x00000063
#define BOARD_ULT62N0_SDRAM  0x56280001 // 究極RX62Nボード初期版 SDRAM使用
#define BOARD_ULT62N0_MMC    0x56280002 // 究極RX62Nボード初期版 SDMMCカード使用
#define BOARD_ULT62N         0x56280003 // 究極RX62NボードRevA版
#define BOARD_RXMEGA         0x56280004 // RX-MEGA
#define BOARD_RAXINO         0x56280005 // RAXINO
#define BOARD_FRKRX62N       0x56270006 // FRK-RX62N
////////////////////////////////////////////////////////////////

#define _PINDEF(port,bit) ((port) << 8 | (bit))

#define _PORT0 0
#define _PORT1 1
#define _PORT2 2
#define _PORT3 3
#define _PORT4 4
#define _PORT5 5
#define _PORT6 6
#define _PORT7 7
#define _PORT8 8
#define _PORT9 9
#define _PORTA 10
#define _PORTB 11
#define _PORTC 12
#define _PORTD 13
#define _PORTE 14
#define _PORTF 15
#define _PORTG 16
#define _PORTJ 18

#define _BIT0 (1 << 0)
#define _BIT1 (1 << 1)
#define _BIT2 (1 << 2)
#define _BIT3 (1 << 3)
#define _BIT4 (1 << 4)
#define _BIT5 (1 << 5)
#define _BIT6 (1 << 6)
#define _BIT7 (1 << 7)

#include "boarddef.h" // ボード番号定義を読み込み
#include "tkdn_sci.h"
#include "tkdn_spi.h"
#include "tkdn_i2c.h"
#include "tkdn_dac.h"
#include "tkdn_adc.h"
#include "tkdn_gpio.h"
#include "tkdn_rtc.h"
#include "tkdn_timer.h"
#include "tkdn_ether.h"

#ifdef __cplusplus
	extern "C" {
#endif

#ifdef __cplusplus
	}
#endif

#endif // __H_TKDN_HAL

