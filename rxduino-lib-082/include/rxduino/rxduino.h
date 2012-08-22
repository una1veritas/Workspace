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

#ifndef __H_RXDUINO
#define __H_RXDUINO


/**
	@mainpage RXDuino API
	@section abstruct 概要
		RXduino APIはC++で記述されたArduinoライクな環境を提供するためのAPIです
	@section intro 始めに
		RXduino APIはC++で記述されたArduinoライクな環境を提供するためのAPIです
	@section motivation 背景
		RX63N,RX62Nには,GPIO,Ethernet controller,SCI,SPI,I2C,RTC,EXDMAC,USB等,数多くの周辺機能が内蔵されており,これらを秩序正しく制御する必要があります．<br>
		そこで特殊電子回路(株)ではとてもシンプルで可読性の高い高機能な「特電HAL」および｢RXDuino API｣をリリースしました
	@section environmen 環境
		特殊電子回路(株)製rx-elf-gcc-4.6.1以上
		- Windows XP,Windows7環境化ではCygwinが動作する環境
		- Mac OSXではSnow Leopard(10.6.1以上)ではXcode 3.4以上がインストール,Lion(10.7)ではXCode 4.1以上がインストールされており,いずれもMacPortsがインストールされている事が望ましい
	@section history 履歴
	- 2011/08/01    初版作成
 */

/**************************************************************************//**
 * @file    rxduino.h
 * @brief   Renesas RX62N/63N Arduino Library produced by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	1. August 2011.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright &copy; 2011 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/
#define RXDUINO_API

#include "../tkdnhal/tkdn_hal.h"
#include "serial.h"
#include "digitalio.h"
#include "delay.h"
#include "analogio.h"
#include "tone.h"
#include "shiftout.h"
#include "wiring.h"
#include "progmem.h"
#include "interrupt.h"

#ifdef __cplusplus
	extern "C" {
#endif

// 次の2つの関数はユーザが作るもの
extern void setup(void);
extern void loop(void);

#define RXDUINO_VERSION   0x00080000 // version 0.7.1.00

const char RXDUINO_COPYRIGHT[] = "(C)Copyright 2012 Tokushu Denshi Kairo Inc.";

#ifdef __cplusplus
	}
#endif

#endif // __H_RXDUINO

