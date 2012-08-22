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

#include <stdio.h>

//#include "vect.h"
#ifdef __GNUC__
  #ifdef CPU_IS_RX62N
    #include "iodefine_gcc62n.h"
  #endif
  #ifdef CPU_IS_RX63N
    #include "iodefine_gcc63n.h"
  #endif
#endif
#ifdef __RENESAS__
  #include "iodefine.h"
#endif

//const int PCLK       = 48000000;
//const int TIMER_FREQ = 125;
extern int tim_1ms;

#ifdef __cplusplus
extern "C" {
#endif

void tkdn_hwsetup();

#ifdef __cplusplus
}
#endif
