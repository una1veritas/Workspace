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

#include "../tkdnhal/tkdn_hal.h"
#include "../tkdnhal/tkdn_dflash.h"
#include "eeprom.h"

int EEPROM::write(unsigned long addr,unsigned char data)
{
	int stat;
	int i;
	unsigned char  buf[32];
	unsigned char  blank[16];

	if(tkdn_dflash_blank(addr & 0xfffffffe,2) == 0) // 何か書き込まれている
	{
		for(i=0;i<16;i++)
		{
			if(tkdn_dflash_blank((addr & 0xffffffe0) + i * 2, 2) == 1)
			{
				// そのアドレスはブランク
				if((addr & 0x1e) == i*2) // 目的のアドレスだ
				{
					blank[i] = 0; // 書き込み対象とする
					buf[i*2    ] = 0xff;
					buf[i*2 + 1] = 0xff;
					buf[addr & 0x1f] = data;
				}
				else // 目的のアドレスではない
				{
					blank[i] = 1; // 書き込み対象としない
					buf[i*2    ] = 0xff;
					buf[i*2 + 1] = 0xff;
				}
			}
			else
			{
				// そのアドレスには何か書き込まれている
				blank[i] = 0; // 書き込み対象とする
				tkdn_dflash_read((addr & 0xffffffe0) + i*2,&buf[i*2],2);
				buf[addr & 0x1f] = data;
			}
		}
		addr &= 0xffffffe0;

		stat = tkdn_dflash_erase(addr , 32);    // その領域をすべて消去
		if(stat <= 0) return -1;
		for(i=0;i<16;i++)
		{
			if(blank[i] == 0)
			{
				stat = tkdn_dflash_write(addr+i*2, &buf[i*2], 2);// メモリ書き込み
				if(stat <= 0) return -1;
			}
		}
	}
	else
	{
		buf[(addr & 0x1e)    ] = 0xff;
		buf[(addr & 0x1e) | 1] = 0xff;
		buf[ addr & 0x1f] = data;
		stat = tkdn_dflash_write(addr & 0xfffffffe,&buf[addr & 0x1e],2);// メモリ書き込み
		if(stat <= 0) return -1;
	}
	tkdn_dflash_terminate();
	return 1;
}

unsigned char EEPROM::read(unsigned long addr)
{
	unsigned short val;
	if(tkdn_dflash_blank(addr & 0xfffffffe,2) == 1) // そこはブランク
	{
		return 0xff;
	}
	tkdn_dflash_read(addr & 0xfffffffe,(unsigned char *)&val,2);
	return (addr & 1) ? (val >> 8) : (val & 0xff);
}
