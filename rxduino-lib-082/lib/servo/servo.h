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

#ifndef	__H_RXDUINO_SERVO
#define	__H_RXDUINO_SERVO
/**************************************************************************//**
 * @file     servo.h
 * @brief    Renesas RX63N core Hardware and Peripheral Access Layer Header for
 *           RXDuino by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	8. May 2012.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright (C) 2011-2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/
#include "rxduino.h"
#include "../tkdnhal/tkdn_servo.h"

#ifdef __cplusplus
	extern "C" {
#endif

//------------------------------------------------------------------
// サーボ出力
//------------------------------------------------------------------
/*! @class Servo
	@brief サーボモータを制御するクラス。標準的なサーボでは0~180度の範囲で角度を指定します。
*/
class Servo 
{
private:
	servo_t servo;
	bool _attached;
public:
	//! Servoの初期化
	/*!
	@param なし
	@return なし
	*/
	Servo();

	//! Servoのクローズ
	/*!
	@param なし
	@return なし
	*/
	~Servo();

	//! Servo型変数にピンを割り当てる
	/*!
	@param pin ピン番号
	@return なし
	*/
	void attach(int pin);

	//! Servo型変数にピンを割り当てる
	/*!
	@param pin ピン番号
	@param min サーボの角度が0度のときのパルス幅(マイクロ秒)。デフォルトは544
	@param max サーボの角度が180度のときのパルス幅(マイクロ秒)。デフォルトは2400
	@return なし
	*/
	void attach(int pin,int min,int max);

	//! サーボの角度をセットし、シャフトをその方向に向けます。 連続回転タイプのサーボでは、回転のスピードが設定されます。
	/*!
	@param angle シャフトの角度を指定します
	@return なし
	*/
	void write(int angle);

	//! サーボの角度をマイクロ秒単位で角度を指定します。サーボに与えるパルスで1周期中のHighの時間を直接指定します
	/*!
	@param us マイクロ秒 (int)
	@return なし
	*/
	void writeMicroseconds(int us);

	//! 現在のサーボの角度(最後にwriteで与えた値)を読み取ります
	/*!
	@param なし
	@return 角度(0~180度) (int)
	*/
	int read(void);

	//! ピンにサーボが割り当てられているかを確認
	/*!
	@param なし
	@return (bool)
			- true : 割り当てられている
			- false : 割り当てはない
	*/
	bool attached(void);

	//! サーボの割り当てからピンを開放します
	/*!
	@param なし
	@return なし
	*/
	void detach(void);
};

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_TONE

