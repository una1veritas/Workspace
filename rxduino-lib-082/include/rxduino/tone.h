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

#ifndef	__H_RXDUINO_TONE
#define	__H_RXDUINO_TONE
/**************************************************************************//**
 * @file     tone.h
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
// トーン出力
//------------------------------------------------------------------

// 矩形波を生成
// pin はデジタルI/Oのと同じ
// frequencyはHz単位
// duration_msには、ms単位で持続時間を指定する。
// 省略するか、0を指定すると、鳴り続ける
/*********************************************************************//**
 * @brief		矩形波を生成する
 * @param[in]	pin digitalWrite関数で指定するピン番号と同じです
 * @param[in]	frequency 出力したい周波数[Hz]
 * @param[in]	duration_ms 持続時間を指定する[ms]。0を指定すると鳴り続ける
 * @return		なし
 **********************************************************************/
RXDUINO_API
void tone(int pin, int frequency, int duration_ms = 0);

// 矩形波の生成をストップ
/*********************************************************************//**
 * @brief		矩形波の出力を止める
 * @param[in]	なし
 * @return		なし
 **********************************************************************/
RXDUINO_API
void noTone(int pin);

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_TONE
