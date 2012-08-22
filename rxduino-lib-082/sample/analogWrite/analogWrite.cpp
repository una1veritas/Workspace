// RX62NのGCCサンプルプログラム
// PWMによるアナログ出力
// (C)Copyright 2012 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cppに変更して、makeしてください。

// 特電HAL
#include <rxduino.h>

unsigned char count;

void setup()
{
}

void loop()
{
	Serial.print(count);
	Serial.print(" ");
	analogWrite(2,count++);
	analogWrite(4,count++);
	analogWrite(5,count++);
	analogWrite(6,count++);
	analogWrite(7,count++);
	delay(10);
}
