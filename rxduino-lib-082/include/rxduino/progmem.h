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

#ifndef __H_PROGMEM
#define __H_PROGMEM

/**************************************************************************//**
 * @file    progmem.h
 * @brief   Renesas RX62N/63N Arduino Library produced by Tokushu Denshi Kairo Inc.
 * @version	V0.50
 * @date	1. August 2011.
 * @author	Tokushu Denshi Kairo Inc.
 * @note	Copyright &copy; 2011 - 2012 Tokushu Denshi Kairo Inc. All rights reserved.
 ******************************************************************************/

#ifdef __cplusplus
	extern "C" {
#endif

// PROGMEMは、コード領域にデータを格納するための
/*! @def	PROGMEM
	@brief	コード領域にデータを格納する
*/
#define PROGMEM __attribute__ ((section (".text")))

/*
typedef void          PROGMEM prog_void;
typedef char          PROGMEM prog_char;
typedef unsigned char PROGMEM prog_uchar;
typedef int8_t        PROGMEM prog_int8_t;
typedef uint8_t       PROGMEM prog_uint8_t;
typedef int16_t       PROGMEM prog_int16_t;
typedef uint16_t      PROGMEM prog_uint16_t;
typedef int32_t       PROGMEM prog_int32_t;
typedef uint32_t      PROGMEM prog_uint32_t;
typedef int64_t       PROGMEM prog_int64_t;
typedef uint64_t      PROGMEM prog_uint64_t;
*/

//#ifndef __GNUC__
//#endif

#ifdef __cplusplus
	}
#endif

#endif // __H_PROGMEM

