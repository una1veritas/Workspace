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

#ifndef	__H_RXDUINO_SHIFTOUT
#define	__H_RXDUINO_SHIFTOUT
/**************************************************************************//**
 * @file     shiftout.h
 * @brief    Renesas RX62N core Hardware and Peripheral Access Layer Header for
 *           RX62N computer board systems produced by Tokushu Denshi Kairo Inc.
 * @note	Copyright (C) 2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/
#include "rxduino.h"

#ifdef __cplusplus
	extern "C" {
#endif


/*********************************************************************//**
 * @brief		1バイト分のデータを1ビットずつ出力する.
 * 				最上位ビット(MSB)と最下位ビット(LSB)のどちらからもスタートできます。
 * 				各ビットはまずdataPinに出力され、その後clockPinが反転して、そのビットが有効になったことが示されます。
 *
 * @param[in]	dataPin データ出力ピン
 * @param[in]	clockPin クロック入力ピン
 * @param[in]	bitOrder ビットオーダの指定
 * 				- SPI_MSBFIRST : 最上位ビットから送る
 * 				- SPI_LSBFIRST : 最下位ビットから送る
 * @param[in]	value 送信したいデータ(unsigned char)
 *
 * @return		なし
 **********************************************************************/
RXDUINO_API
void shiftOut(int dataPin,int clockPin,SPI_BIT_ORDER bitOrder, unsigned char value);


/*********************************************************************//**
 * @brief		1バイトのデータを1ビットずつ取り込む。
 * 				最上位ビット(MSB)と最下位ビット(LSB)のどちらからもスタートできます。
 * 				各ビットについて次のように動作します。
 * 				まずclockPinがHIGHになり、dataPinから次のビットが読み込まれ、clockPinがLOWに戻ります。
 *
 * @param[in]	dataPin データ出力ピン
 * @param[in]	clockPin クロック入力ピン
 * @param[in]	bitOrder ビットオーダの指定
 * 				- SPI_MSBFIRST : 最上位ビットから送る
 * 				- SPI_LSBFIRST : 最下位ビットから送る
 *
 * @return	value 読み取った値(unsigned char)
 **********************************************************************/
RXDUINO_API
unsigned char shiftIn(unsigned char dataPin, unsigned char clockPin, SPI_BIT_ORDER bitOrder);

/*********************************************************************//**
 * @brief		複数バイト分のデータを1ビットずつ出力する.
 * 				shiftOut関数のRXduinoオリジナル拡張で32bitまで1bit単位で指定できます。
 * 				最上位ビット(MSB)と最下位ビット(LSB)のどちらからもスタートできます。
 * 				各ビットはまずdataPinに出力され、その後clockPinが反転して、そのビットが有効になったことが示されます。
 *
 * @param[in]	dataPin データ出力ピン
 * @param[in]	clockPin クロック入力ピン
 * @param[in]	bitOrder ビットオーダの指定
 * 				- SPI_MSBFIRST : 最上位ビットから送る.32bit未満のデータの場合,valueはMSB詰めで設定してください
 * 				- SPI_LSBFIRST : 最下位ビットから送る.32bit未満のデータの場合,valueはLSB詰めで設定してください
 * @param[in]	len 出力するデータのビット数。
 * @param[in]	value 送信したいデータ(unsigned char)
 *
 * @return		なし
 **********************************************************************/
RXDUINO_API
void shiftOutEx(int dataPin,int clockPin,SPI_BIT_ORDER bitOrder, int len,unsigned long value);

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_SHIFTOUT

