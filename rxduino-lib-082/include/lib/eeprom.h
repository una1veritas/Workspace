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

#ifndef	__H_RXDUINO_EEPROM
#define	__H_RXDUINO_EEPROM
/**************************************************************************//**
 * @file     eeprom.h
 * @brief    Renesas RX63N core Hardware and Peripheral Access Layer Header for
 *           RXDuino by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	8. May 2012.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright (C) 2011-2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/
#include "rxduino.h"

#ifdef __cplusplus
	extern "C" {
#endif

//------------------------------------------------------------------
// EEPROM (内部のE2データフラッシュでEEPROMをエミュレートする)
//------------------------------------------------------------------
/*! @class EEPROM
	@brief 内蔵E2データフラッシュでEEPROMをエミュレートするクラス。
*/
class EEPROM
{
public:
	// addrの範囲は0～0x7fff
	unsigned char read(unsigned long addr);

	// addrの範囲は0～0x7fff
	// 失敗すると-1を返す。成功すると1を返す
	int write(unsigned long addr,unsigned char data);
};

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_EEPROM

