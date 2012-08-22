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

#include "servo.h"

Servo::Servo()
{
	_attached = false;
}

Servo::~Servo()
{
	if(_attached) servo_detach(&servo);
	_attached = false;
}

void Servo::attach(int pin)
{
	attach(pin,0,0);
}

void Servo::attach(int pin,int min,int max)
{
	if(_attached) servo_detach(&servo);
	servo_attach(&servo,pin,min,max);
	_attached = true;
}

void Servo::write(int angle)
{
	if(_attached) servo_write(&servo,angle);
}

void Servo::writeMicroseconds(int us)
{
	if(_attached) servo_write_us(&servo,us);
}

int Servo::read(void)
{
	if(_attached) return servo_read(&servo);
	return 0;
}

bool Servo::attached(void)
{
	return _attached;
}

void Servo::detach(void)
{
	if(_attached) servo_detach(&servo);
	_attached = false;
}
