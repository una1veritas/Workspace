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

#ifndef	__H_RXDUINO_SERIAL
#define	__H_RXDUINO_SERIAL

/**************************************************************************//**
 * @file    serial.h
 * @brief   Renesas RX62N/63N Arduino Library produced by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	1. August 2011.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright &copy; 2011 - 2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/

#include "rxduino.h"
//#include "Print.h"
#include "../tkdnhal/tkdn_sci.h"

#ifdef __cplusplus
	extern "C" {
#endif

extern SCI_PORT SCI_DEFAULT_PORT;

//------------------------------------------------------------------
// シリアルポート
//------------------------------------------------------------------


/*! @class CSerial
    @brief シリアル通信を制御するクラス
*/
class CSerial {
//class CSerial: public Print {
private:
	SCI_PORT port; /*!< ポート番号。SCI_SCI0P2x SCI_SCI1JTAG SCI_SCI2A SCI_SCI2B SCI_SCI6A SCI_SCI6B SCI_USB0の中から指定 */  
	sci_str sci;

public:
	/*! @enum	PRINT_TYPE
		@brief	println関数などで教示する基数(フォーマット)の指定に使う
		@note	Arduinoとは異なり、Serial.print(a,CSerial::DEC)などのようにデータメンバを指定する必要がある  
	*/

	typedef enum {
		NONE,	/*!< 指定無し */
		BYTE,	/*!< アスキーコードでの表示 */
		BIN,	/*!< 2進数での表示 */
		OCT,	/*!< 8進数での表示 */
		DEC,	/*!< 10進数での表示 */
		HEX		/*!< 16進数での表示 */
	} PRINT_TYPE;

	//! コンストラクタの宣言
	/*!
	   コンストラクタCSerialの宣言
	*/
	CSerial(SCI_PORT port=SCI_NONE); // 

	//! デストラクタの宣言
	/*!
	*/
	~CSerial();

	//! シリアル通信ポートの初期化 デフォルトのポートが使用される
	/*!
		@param bps ボーレート
		@return なし
	*/
	void begin(int bps);

	//! ポートを指定して、シリアル通信ポートの初期化
	/*!
		@param bps ボーレート
		@param port
			- SCI_NONE : SCIポートを使用しない
			- SCI_AUTO : SCIを自動選択
			- SCI_USB0 : USB0 の仮想COMポートを使う
			- SCI_USB1 : USB1 の仮想COMポートを使う （未実装）
			- SCI_SCI0P2x : SCI0 (ポートP20,P21と兼用) を使う
			- SCI_SCI1JTAG : SCI1 (ポートPF0,PF2,JTAGと兼用) を使う
			- SCI_SCI2A : SCI2A (ポートP13,P12)を使う
			- SCI_SCI2B : SCI2B (ポートP50,P52)を使う
			- SCI_SCI6A : SCI6A (ポートP00,P01)を使う
			- SCI_SCI6B : SCI6B (ポートP32,P33)を使う
		@return なし
	*/
	void begin(int bps,SCI_PORT port);

	//! シリアル通信ポートのクローズ
	/*!
		@param なし
		@return なし
	*/
	void end();

	// このシリアルをデフォルトのシリアルに設定し、printf等が使えるようにする
	void setDefault();

	//! シリアルポートから何バイトのデータが読み取れるかを返す
	/*!
	@param なし
	@return シリアルバッファにあるデータのバイト数。0の場合はデータなし
	*/
	int available();

	//! シリアルポートの受信バッファから１バイトのデータを読み出します
	/*!
	@param なし
	@return 先頭のデータ。データなしの場合は-1が返る
	*/
	int read();

	//! シリアルポートの受信バッファにある先頭のデータを読みます。バッファ中の読み込み位置は変更しないので、バッファを覗くだけです。CRLFの変換は行われません。
	/*!
	@param なし
	@return 先頭のデータ。データなしの場合は-1が返る
	*/
	int peek();

	//! シリアルポートの送信バッファが空になるまで待ちます。受信バッファをどうするかは、Arduinoの仕様が変わっているので、検討中です。
	/*!
	@param なし
	@return なし
	*/
	void flush();

	//! この関数は実装していない
//	void serialEvent();

	void write(const char val);
//	using Print::write;

	void write(const char *str);
	void write(const char *buf,int len);

	// ★★Warning★★ 表示タイプは選べません。10進数のみです
	void print(int val, PRINT_TYPE print_type=DEC);
	void print(double val, int fpdigit=2);
	void print(const char *str);
	void println(int val, PRINT_TYPE print_type=DEC);
	void println(double val, int fpdigit=2);
	void println(const char *str);
	
	sci_str *get_handle();
};

extern CSerial Serial;
extern CSerial Serial1;
extern CSerial Serial2;
extern CSerial Serial3;

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_SERIAL

