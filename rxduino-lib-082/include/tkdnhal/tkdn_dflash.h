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

#ifndef __H_TKDN_DFLASH
#define __H_TKDN_DFLASH

#include "tkdn_dflash.h"

#ifdef __cplusplus
extern "C" {
#endif

//■■■■■■■■■■■■■■■■■■■■■■■■■
//  ユーザがカスタマイズする場所はありません
//■■■■■■■■■■■■■■■■■■■■■■■■■

TKDN_HAL
int tkdn_dflash_write(unsigned long offset,unsigned char *buf,int len);

TKDN_HAL
int tkdn_dflash_read(unsigned long offset,unsigned char *buf,int len);

TKDN_HAL
int tkdn_dflash_erase(unsigned long offset,int len);

// フラッシュ書き込みモードを終了する
TKDN_HAL
void tkdn_dflash_terminate(void);

// ブランクチェック ブランクなら1 書き込まれていれば0 エラーなら-1
TKDN_HAL
int tkdn_dflash_blank(unsigned long offset,int len);

#ifdef __cplusplus
 }
#endif

#endif // __H_TKDN_TIMER
