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

#ifndef __H_TKDN_TIMER
#define __H_TKDN_TIMER

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

typedef void (* USER_TIMER_FUNC)(void);

TKDN_HAL
void timer_init(void);

TKDN_HAL
void timer_wait_ms(unsigned long ms);

TKDN_HAL
void timer_wait_us(unsigned long us);

TKDN_HAL
unsigned long timer_get_ms(void);

TKDN_HAL
unsigned long timer_get_us(void);

TKDN_HAL
unsigned long timer_regist_userfunc(USER_TIMER_FUNC func);

// 以下の関数はシステムライブラリが使用する。ユーザは実行しないこと
TKDN_HAL
unsigned long timer_regist_disktimer(USER_TIMER_FUNC func);

#ifdef __cplusplus
 }
#endif

#endif // __H_TKDN_TIMER
