// RX62NのGCCサンプルプログラム
// メイン
// (C)Copyright 2011 特殊電子回路

// ＲＸデュイーノ風ライブラリ
#include <rxduino.h>
#include <liquidcrystal.h>
#include <stdlib.h>

int count = 0;

const unsigned char heart[8] = {0x0a,0x1b,0x1f,0x1f,0x1f,0x0e,0x04,0x00};
/*
   .*.*.
   **.**
   *****
   *****
   *****
   .***.
   ..*..
   .....
*/

const unsigned char progress0[] = {0x0e,0x0e,0x15,0x0e,0x04,0x0a,0x11,0x11};
/*
   .***.
   .***.
   *.*.*
   .***.
   ..*..
   .*.*.
   *...*
   *...*
*/

const unsigned char progress1[] = {0x02,0x1d,0x1d,0x05,0x1e,0x12,0x0d,0x11};
/*
   ...*.
   ***.*
   ***.*
   ..*.*
   ****.
   *..*.
   .**.*
   *...*
*/


const unsigned char progress3[] = {0x08,0x17,0x17,0x14,0x0f,0x09,0x16,0x11};
/*
   .*...
   *.***
   *.***
   *.*..
   .****
   .*..*
   *.**.
   *...*
*/

const unsigned char progress4[] = {0x0e,0x0e,0x05,0x0e,0x14,0x0a,0x09,0x11};
/*
   .***.
   .***.
   ..*.*
   .***.
   *.*..
   .*.*.
   .*..*
   *...*
*/

const unsigned char progress5[] = {0x0e,0x0e,0x06,0x0c,0x07,0x0a,0x19,0x02};
/*
   .***.
   .***.
   ..**.
   .**..
   ..***
   .*.*.
   **..*
   ...*.
*/

const unsigned char progress6[] = {0x0e,0x0e,0x14,0x0e,0x05,0x0a,0x12,0x11};
/*
   .***.
   .***.
   *.*..
   .***.
   ..*.*
   .*.*.
   *..*.
   *...*
*/

const unsigned char progress7[] = {0x0e,0x0e,0x0c,0x06,0x1c,0x0a,0x13,0x08};
/*
   .***.
   .***.
   .**..
   ..**.
   ***..
   .*.*.
   *..**
   .*...
*/

LiquidCrystal *lcd;

void tone(int Hz,int ms)
{
	int start = millis();
	while(1)
	{
		digitalWrite(PIN_BUZZ,HIGH);
		delayMicroseconds(1000000/Hz/2);
		digitalWrite(PIN_BUZZ,LOW);
		delayMicroseconds(1000000/Hz/2);
		if(millis() - start > ms) break;
	}
}

int enableTone = 0;

void local_tone(int Hz,int ms)
{
	int start = millis();
	while(1)
	{
		digitalWrite(PIN_BUZZ,HIGH);
		delayMicroseconds(1000000/Hz/2);
		digitalWrite(PIN_BUZZ,LOW);
		delayMicroseconds(1000000/Hz/2);
		if(millis() - start > ms) break;
	}
}

void setup()
{
	pinMode(PIN_BUZZ,OUTPUT);

	for(int i=0;i<100;i++) { // ピッ
		digitalWrite(PIN_BUZZ,i & 1);
		for(int j=0;j<4000;j++) {}
	}

	Serial.begin(38400);
	Serial.println("--------------------------------------------------------");
	Serial.println("RX62N GCC and RXduino emvironment test");
	Serial.print("Compiled at ");
	Serial.print(__DATE__);
	Serial.print(" ");
	Serial.println(__TIME__);
	Serial.println("--------------------------------------------------------");

	for(int i=0;i<14;i++)
	{
		pinMode(i,OUTPUT);
	}
	for(int i=0;i<6;i++)
	{
		pinMode(14 + i,INPUT);
	}
	for(int i=0;i<5;i++)
	{
		pinMode(PIN_LED0 + i,OUTPUT);
	}
	pinMode(PIN_SW,INPUT);

	lcd = new LiquidCrystal(12,10,5,4,3,2);
//	lcd = new LiquidCrystal(12,11,10,5,4,3,2);

	lcd->begin(20,2);
	lcd->createChar(0x00,heart);
	lcd->createChar(0x08,progress0);
	lcd->createChar(0x10,progress1);
	lcd->createChar(0x18,progress3);
	lcd->createChar(0x20,progress4);
	lcd->createChar(0x28,progress5);
	lcd->createChar(0x30,progress6);
	lcd->createChar(0x38,progress7);

	lcd->writeDelay = 100;
	lcd->print("RXduino/RX-MEGA\n");
	lcd->print(" TOKUDEN Kairo");
	lcd->write(0);
	delay(1500);

	while(1)
	{
		int prev = 0;
		lcd->writeDelay = 0;
		for(int i=0;i<17;i++)
		{
			lcd->setCursor(i,0);
			lcd->write((i & 1) + 4);
			lcd->setCursor(prev,0);
			lcd->write(' ');
			if(enableTone) local_tone((i & 1) ? 500 : 1000 ,10);
			delay(150);
			prev = i;
			if(digitalRead(PIN_SW) == 0)
			{
				enableTone = ~enableTone;
				delay(100);
			}
		}
	
		for(int i=0;i<5;i++)
		{
			lcd->setCursor(rand() % 20,rand() % 2);
			lcd->write('*');
		}

		prev = 16;
		for(int i=16;i>=0;i--)
		{
			lcd->setCursor(i,1);
			lcd->write((i & 1) + 6);
			lcd->setCursor(prev,1);
			lcd->write(' ');
			if(enableTone) local_tone(i & 1 ? 500 : 1000,10);
			delay(150);
			prev = i;
			if(digitalRead(PIN_SW) == 0)
			{
				enableTone = ~enableTone;
				delay(100);
			}
		}

		lcd->setCursor(prev,1);
		lcd->write(' ');

		for(int i=0;i<5;i++)
		{
			lcd->setCursor(rand() % 20,rand() % 2);
			lcd->write(8);
		}
		break;
	}

	lcd->clear();
	lcd->writeDelay = 0;
	lcd->print("println() ﾉ ﾃｽﾄ\n");
	
	lcd->home();
	
	count = 0;
}

void loop()
{
	lcd->setCursor(0,1);
	lcd->print("Count=");
	lcd->print(count);
	count++;
	lcd->print("  ");
	switch((count >> 6) & 0x03)
	{
		case 0:
		case 2:
			lcd->write(1);
			break;
		case 1:
			lcd->write(2);
			break;
		case 3:
			lcd->write(3);
			break;
	}

	if(count > 1000)
	{
		lcd->clear();
		lcd->writeDelay = 100;
		lcd->print("ﾓｳｽｸﾞ .NET MFｶﾞ\n");
		lcd->print("ｷﾄﾞｳｼ");
		lcd->writeDelay = 300;
		lcd->print("ﾅｻｿｳﾃﾞｽ");
		lcd->print("(");
		lcd->scrollDisplayLeft();
		lcd->print(";");
		lcd->scrollDisplayLeft();
		lcd->print("_");
		lcd->scrollDisplayLeft();
		lcd->print(";");
		lcd->scrollDisplayLeft();
		delay(1000);
		lcd->blink();
		delay(2000);
		lcd->noDisplay();
		setup();
	}
}
