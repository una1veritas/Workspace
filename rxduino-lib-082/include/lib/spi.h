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

#ifndef	__H_RXDUINO_SPI
#define	__H_RXDUINO_SPI
/**************************************************************************//**
 * @file    spi.h
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

//------------------------------------------------------------------
// SPIポート
//------------------------------------------------------------------
//typedef enum {SPI_PORT_NONE, SPI_PORT_CS0_DUINO, SPI_PORT_CS1_MARY1, SPI_PORT_CS2_MARY2, SPI_PORT_CS3_ROM, SPI_PORT_SDMMC} SPI_PORT;
//typedef enum {LSBFIRST, MSBFIRST} SPI_BIT_ORDER;
//typedef enum {SPI_CLOCK_DIV2,  SPI_CLOCK_DIV4,  SPI_CLOCK_DIV8, SPI_CLOCK_DIV16,
//              SPI_CLOCK_DIV32, SPI_CLOCK_DIV64, SPI_CLOCK_DIV128 } SPI_CLK_DIVIDER;
//typedef enum {SPI_MODE0 , SPI_MODE1 , SPI_MODE2 , SPI_MODE3 } SPI_DATA_MODE;

#define LSBFIRST SPI_LSBFIRST // enum型の定数で実際の値は0
#define MSBFIRST SPI_MSBFIRST // enum型の定数で実際の値は1

/*! @class CSPI
	@brief SPIを制御するクラス
*/
class CSPI {
private:
	/*! @enum	PRINT_TYPE
		@brief	println関数などで教示する基数(フォーマット)の指定に使う
		@note	Arduinoとは異なり、Serial.print(a,CSerial::DEC)などのようにデータメンバを指定する必要がある
	*/
	SPI_BIT_ORDER bitOrder;
	SPI_CLK_DIVIDER divider;
	SPI_DATA_MODE dataMode;

public:
	CSPI(SPI_PORT port);
	~CSPI();
	SPI_PORT port; // ポート番号を変える場合はここを変える

	//! SPIポートの初期化
	/*!
	@param なし
	@return なし
	*/
	void begin() ;

	//! SPIポートのクローズ
	/*!
	@param なし
	@return なし
	*/
	void end() ;

	//! SPIの送信ビット長の指定
	/*!
	@param bitlength 1回に送受信するデータのビット長を指定[bit]．特に指定しない場合、8bitになります
	@return なし
	*/
	void setBitLength(int bitLength);

	//! SPIの送信ビット長の指定
	/*!
	@param bitOrder SPIバスで送受信する際に使用するビットオーダーを設定します。
	- LSBFIRST : (least-significant bit first)
	- MSBFIRST : (most-significant bit first)
	@return なし
	*/
	void setBitOrder(SPI_BIT_ORDER bitOrder) ;

	//! SPIクロック分周比の設定
	/*!
	@param divider SPIクロックをPCLKの分周比で指定します
	- SPI_CLOCK_DIV2
	- SPI_CLOCK_DIV4
	- SPI_CLOCK_DIV8
	- SPI_CLOCK_DIV16
	- SPI_CLOCK_DIV32
	- SPI_CLOCK_DIV64
	- SPI_CLOCK_DIV128
	@return なし
	*/
	void setClockDivider(SPI_CLK_DIVIDER divider);

	//! SPIの転送モードの設定
	/*!
	@param mode モードはクロック極性とクロック位相の組み合わせで決定されます
	- SPI_MODE0
	- SPI_MODE1
	- SPI_MODE2
	- SPI_MODE3
	@return なし
	*/
	void setDataMode(SPI_DATA_MODE mode);

	// note:RXduinoでは最大32bitまでのデータ送信に対応しています
	//! SPIバスで8ビット~32ビットを送受信します．
	/*!
	@param txdata 転送するデータ(long)
	@return 受信データ(long)
	*/
	unsigned long transfer(unsigned long txdata) ;
};

extern CSPI SPI;

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO_SPI

