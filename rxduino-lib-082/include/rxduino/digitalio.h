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

#ifndef	__H_RXDUINO_DIGITALIO
#define	__H_RXDUINO_DIGITALIO
/**************************************************************************//**
 * @file    digitalio.h
 * @brief   Renesas RX62N/63N Arduino Library produced by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	1. August 2011.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright &copy; 2011 - 2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/
#include "rxduino.h"

#ifdef __cplusplus
	extern "C" {
#endif

// digitaiWrite関数で使う定数
/*!
　@def  LOW
　@brief digitaiWrite関数で使う定数(論理ゼロ:LOW)
*/
#define LOW  0

/*!
　@def  HIGH
　@brief digitaiWrite関数で使う定数(論理1:HIGH)
*/
#define HIGH 1

// pinMode関数で使う定数

/*! @enum	pinModeState
	@brief	pinMode関数で使うポートの状態
*/
typedef enum {
	INPUT ,	/*!< 端子は入力モード */
	OUTPUT /*!< 端子は出力モード */
} pinModeState;

//------------------------------------------------------------------
// デジタル入出力
//------------------------------------------------------------------

// ピン番号には、tkdn_gpioで定義したマクロ定数を使う
// 0～13だとArduino互換コネクタから出力
// 100～は、ボード上のコンポーネント
// 200～は、MARY1
// 300～は、MARY2

// ----------------------------------------
// digitalWrite(pin, value)
// HIGHまたはLOWを、指定したピンに出力します。
// [パラメータ]
// pin: ピン番号
// value: 0(LOW)か1(HIGH)
// [戻り値]
// なし
// ----------------------------------------
/*********************************************************************//**
 * @brief		HIGHまたはLOWを、指定したピンに出力
 * @param[in]	duino_portnum ピン番号(tkdn-gpioで宣言したピン番号)
 * @param[in]	value 出力状態の選択
 * 				- 0 : 指定したピンからLOWを出力
				- 1 : 指定したピンからHIGHを出力
 * @return		なし
 **********************************************************************/
RXDUINO_API
void digitalWrite(int duino_portnum,int value);

// ----------------------------------------
// pinMode(pin, mode)
// ピンの動作を入力か出力に設定します。
// [パラメータ]
// pin: 設定したいピンの番号
// mode: INPUTかOUTPUT
// [戻り値]
// なし
// ----------------------------------------
/*********************************************************************//**
 * @brief		指定したピンを入力または出力に設定
 * @param[in]	duino_portnum ピン番号(tkdn-gpioで宣言したピン番号)
 * @param[in]	state 出力状態の選択
 * 				- INPUT : 指定したピンは入力モード
				- OUTPUT : 指定したピンは出力モード
 * @return		なし
 **********************************************************************/
RXDUINO_API
void pinMode(int duino_portnum, pinModeState state);

/*********************************************************************//**
 * @brief		指定したピンの値を読み取り
 * @param[in]	duino_portnum ピン番号(tkdn-gpioで宣言したピン番号)
 *
 * @return		HIGHまたはLOW
 **********************************************************************/
RXDUINO_API
int digitalRead(int duino_portnum);

/*********************************************************************//**
 * @brief		ピンに入力されるパルスを検出する。たとえば、パルスの種類(value)をHIGHに指定した場合、
 * 				pulseIn関数は入力がHIGHに変わると同時に時間の計測を始め、またLOWに戻ったら、
 * 				そこまでの時間(つまりパルスの長さ)をマイクロ秒単位で返す。
 * 				あまりに長いパルスに対してはエラーとなる可能性がある。
 * @param[in]	duino_portnum ピン番号(tkdn-gpioで宣言したピン番号)
 * @param[in]	val 測定するパルスの種類。HIGHまたはLOW
 * @param[in]	timeout タイムアウトまでの時間(単位・マイクロ秒) デフォルトは1秒
 *
 * @return		パルスの長さ(マイクロ秒)。パルスがスタートする前にタイムアウトとなった場合は0 (unsigned long)
 **********************************************************************/
RXDUINO_API
int pulseIn(int duino_portnum, int val, unsigned long timeout = 1000000);


#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_DIGITALIO

