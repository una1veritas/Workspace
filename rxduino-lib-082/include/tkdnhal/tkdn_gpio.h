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

#ifndef __H_TKDN_GPIO
#define __H_TKDN_GPIO

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

// Arduino互換ピン
#define PIN_ARD0       0    // Arduino互換コネクタのデジタル0番
#define PIN_ARD1       1    // Arduino互換コネクタのデジタル1番
#define PIN_ARD2       2    // Arduino互換コネクタのデジタル2番
#define PIN_ARD3       3    // Arduino互換コネクタのデジタル3番
#define PIN_ARD4       4    // Arduino互換コネクタのデジタル4番
#define PIN_ARD5       5    // Arduino互換コネクタのデジタル5番
#define PIN_ARD6       6    // Arduino互換コネクタのデジタル6番
#define PIN_ARD7       7    // Arduino互換コネクタのデジタル7番
#define PIN_ARD8       8    // Arduino互換コネクタのデジタル8番
#define PIN_ARD9       9    // Arduino互換コネクタのデジタル9番
#define PIN_ARD10      10   // Arduino互換コネクタのデジタル10番
#define PIN_ARD11      11   // Arduino互換コネクタのデジタル11番
#define PIN_ARD12      12   // Arduino互換コネクタのデジタル12番
#define PIN_ARD13      13   // Arduino互換コネクタのデジタル13番
#define PIN_ARD14      14   // Arduino互換コネクタのデジタル14番/アナログ0番
#define PIN_ARD15      15   // Arduino互換コネクタのデジタル15番/アナログ1番
#define PIN_ARD16      16   // Arduino互換コネクタのデジタル16番/アナログ2番
#define PIN_ARD17      17   // Arduino互換コネクタのデジタル17番/アナログ3番
#define PIN_ARD18      18   // Arduino互換コネクタのデジタル18番/アナログ4番
#define PIN_ARD19      19   // Arduino互換コネクタのデジタル19番/アナログ5番

// ボード上のコンポーネント
#define PIN_FIRST_BOARDPIN 100  // ボード上のコンポーネントの最初のピン番号
#define PIN_LED0           100  // ボード上のLED
#define PIN_LED1           101  // ボード上のLED
#define PIN_LED2           102  // ボード上のLED
#define PIN_LED3           103  // ボード上のLED
#define PIN_BUZZ           104  // ボード上のブザー
#define PIN_SW             105  // RX-MEGAボード上の青色SW
#define PIN_SPI_CS0        106  // ボード上のSPIのCS0番
#define PIN_SPI_CS1        107  // ボード上のSPIのCS1番
#define PIN_SPI_CS2        108  // ボード上のSPIのCS2番
#define PIN_SPI_CS3        109  // ボード上のSPIのCS3番
#define PIN_LAST_BOARDPIN  109  // ボード上のコンポーネントの最後のピン番号

// 引数 pinnum 上記のマクロ定数
//      pinnumに存在しないピン番号を指定しても安全に終了する
//      isoutput 0:入力 1:出力
TKDN_HAL
void gpio_set_pinmode(int pinnum,int isoutput);

// 引数 pinnum 上記のマクロ定数
// 引数 state 0:L 1:H
TKDN_HAL
void gpio_write_port(int pinnum,int state);

// 引数 pinnum 上記のマクロ定数
TKDN_HAL
int  gpio_read_port(int pinnum);

#ifdef __cplusplus
 }
#endif

#endif // __H_TKDN_GPIO
