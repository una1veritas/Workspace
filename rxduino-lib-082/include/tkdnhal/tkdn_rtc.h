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

#ifndef __H_TKDN_RTC
#define __H_TKDN_RTC

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

typedef struct RX62N_RTC_TIME {
	unsigned short year;
	unsigned char  mon;
	unsigned char  day;
	unsigned char  weekday;
	unsigned char  hour;
	unsigned char  min;
	unsigned char  second;
} RX62N_RTC_TIME;


// RTCに時刻を設定する
// 入力パラメータは上記の構造体
// 成功すると1を返す。失敗すると0を返す。
TKDN_HAL
int rtc_set_time(RX62N_RTC_TIME *time);

// RTCの時刻を取得する
// 成功すると1を返す。失敗すると0を返す。
TKDN_HAL
int rtc_get_time(RX62N_RTC_TIME *time);

#ifdef __cplusplus
 }
#endif

#endif
