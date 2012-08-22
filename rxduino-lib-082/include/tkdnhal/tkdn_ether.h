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

#ifndef	__H_TKDN_ETHER
#define	__H_TKDN_ETHER

#include "tkdn_hal.h"

#ifdef __cplusplus
	extern "C" {
#endif

typedef void (*ETHER_USER_INTERRUPT_FUNCTION)(void);

TKDN_HAL
int  ether_open(unsigned char mac_addr[]);

TKDN_HAL
void ether_close(void);

TKDN_HAL
int  ether_write(unsigned char *buf,int len);

TKDN_HAL
int  ether_read(unsigned char *buf);

TKDN_HAL
int  ether_is_linkup(void);

TKDN_HAL
int  ether_is_100M(void);

TKDN_HAL
int  ether_is_fullduplex(void);

TKDN_HAL
void ether_autonegotiate();

// Linkの状態が変化したときに１回だけTRUEを返す
TKDN_HAL
int  ether_link_changed(void);

// フレームを受信したときに呼び出される関数を登録する
TKDN_HAL
void ether_regist_user_rx_procedure(ETHER_USER_INTERRUPT_FUNCTION func);

#ifdef __cplusplus
	}
#endif

#endif /* R_ETHER_H */

