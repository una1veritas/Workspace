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

#include "spi.h"

CSPI SPI(SPI_PORT_NONE);

CSPI::CSPI(SPI_PORT port)
{
	this->port = port;
	bitOrder = LSBFIRST;
	divider = SPI_CLOCK_DIV4;
	dataMode = SPI_MODE0;
	begin();
}

CSPI::~CSPI()
{
	end();
}

void CSPI::setBitLength(int bitlength)
{
	spi_set_bit_length(bitlength);
}

void CSPI::begin()
{
	spi_init();
	spi_set_clock_divider(divider);
	spi_set_bit_length(8);
	spi_set_bit_order(LSBFIRST);
}

void CSPI::end()
{
	spi_terminate();
}

void CSPI::setBitOrder(SPI_BIT_ORDER bitOrder)
{
	this->bitOrder = bitOrder;
	spi_set_bit_order(bitOrder);
}

void CSPI::setClockDivider(SPI_CLK_DIVIDER divider)
{
	this->divider = divider;
	spi_set_clock_divider(divider);
}

void CSPI::setDataMode(SPI_DATA_MODE mode)
{
	this->dataMode = mode;
	spi_set_data_mode(mode);
}

unsigned long CSPI::transfer(unsigned long txdata)
{
	spi_set_port(port);
	return spi_transfer(txdata);
}

