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

#ifndef __H_TKDN_USB
#define __H_TKDN_USB

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

//==================================================
// USBのベースとなる関数
//==================================================

// USBを初期化する
//  成功すれば0を、そうでなければ負の数を返す
TKDN_HAL
int           TKUSB_Init(void);

// 接続されていれば1を返す。そうでなければ0を返す。
TKDN_HAL
int           TKUSB_IsConnected(void);

// 1バイト送信
// ※実際にはバッファに溜めるだけで、送信はしない
TKDN_HAL
int           TKUSB_SendByte(unsigned char data);

// 送信バッファに溜まっているデータ数を返す
TKDN_HAL
int           TKUSB_SendDataCount(void);

// 1文字受信
// ※この関数は、受信バッファにデータがないときには即座に0をリターンする
//   受信データの有無を調べるために、あらかじめTKUSB_RecvDataCount()を呼ぶこと。
TKDN_HAL
unsigned char TKUSB_RecvByte(void);

// 1文字覗き見る
// ※この関数は、受信バッファにデータがないときには即座に0をリターンする
TKDN_HAL
unsigned char TKUSB_PeekByte(void);

// 受信バッファに溜まっているデータ数を返す
TKDN_HAL
int           TKUSB_RecvDataCount(void);


#ifdef __cplusplus
 }
#endif

#endif /*__H_TKDN_USB*/