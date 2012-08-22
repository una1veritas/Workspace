// RX62NのGCCサンプルプログラム
// GPIOサンプル
// (C)Copyright 2011 特殊電子回路

// RXDuinoを使う
#include <rxduino.h>

int count = 0;

void setup()
{
	Serial.begin(38400);
	Serial.println("");
	Serial.println("GPIO sample program (RXduino version)");

	pinMode(PIN_LED0 , OUTPUT); // LED0を出力
	pinMode(PIN_LED1 , OUTPUT); // LED1を出力
	pinMode(PIN_LED2 , OUTPUT); // LED2を出力
	pinMode(PIN_LED3 , OUTPUT); // LED3(BUZZ)を出力
	pinMode(PIN_SW   , INPUT);   // SWを入力
}

void loop()
{
	Serial.print(count & 15);
	Serial.print(" ");
	digitalWrite(PIN_LED0 , count & 1);
	digitalWrite(PIN_LED1 , count & 2);
	digitalWrite(PIN_LED2 , count & 4);
	digitalWrite(PIN_LED3 , count & 8);
	if(digitalRead(PIN_SW) == 1) delay(100);
	delay(1);

	count++;
}

