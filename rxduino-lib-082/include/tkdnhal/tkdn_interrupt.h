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

#ifndef __H_TKDN_INTERRUPT
#define __H_TKDN_INTERRUPT

#include "tkdn_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*FUNC_INTERRPUT_HANDLER) (void);

extern FUNC_INTERRPUT_HANDLER USBI0_handler;
extern FUNC_INTERRPUT_HANDLER USBI1_handler;

typedef enum 
{
	INTTRIG_LOW      = 0,
	INTTRIG_CHANGE   = 1,
	INTTRIG_RISING   = 2,
	INTTRIG_FALLING  = 3,
} InterruptTrigger;

// PSWを変更してグローバルに割り込みを許可する
void enable_interrupt();

// PSWを変更してグローバルに割り込みを禁止する
void disable_interrupt();

// 割り込みの割り当て
// interrupt 0 ポートP10 (IRQ0) ※TQFP100ピンにはない
//           1 ポートP11 (IRQ1) ※TQFP100ピンにはない
//           2 ポートP12 (IRQ2)
//           3 ポートP13 (IRQ3)
//           4 ポートP14 (IRQ4)
//           5 ポートP15 (IRQ5)
//           6 ポートP16 (IRQ6)
//           7 ポートP17 (IRQ7)
void attach_interrupt(unsigned char interrupt, void (*func)(void), InterruptTrigger mode);

// 割り込みの割り当ての解除
void detach_interrupt(unsigned char interrupt);

#ifdef __cplusplus
}
#endif

#endif
