// RXduinoのサンプルプログラム
// Arduinoのスケッチ風に簡単にプログラムが作れます
// (C) 2012 Tokushu Denshi Kairo Inc.

#include <rxduino.h> // これをインクルードするだけでOK!!
#include <serial.h>
#include <spi.h>
#include <math.h>
#include <tkdn_tone.h>

volatile int count = 0;
bool accel_sensor_exist = false;

void setup()
{
	Serial.begin(38400);
	Serial.println("Start RXduino !");

	//SPIを加速度センサ用に初期化
	SPI.begin();
	// ★SPI.beginを行うと、ClockDivider、BitLength、BitOrder=LSBFIRSTが初期化されてしまうので、
	// setClockDividerなどはbeginの後に行うこと
	SPI.setClockDivider(SPI_CLOCK_DIV64);
	SPI.setBitLength(8);
	SPI.setBitOrder(MSBFIRST) ;
	SPI.port = SPI_PORT_NONE;

	pinMode(PIN_LED0,OUTPUT);
	pinMode(PIN_LED1,OUTPUT);
	pinMode(PIN_LED2,OUTPUT);
	pinMode(PIN_LED3,OUTPUT);

	//関係ないCSをHレベルにしておくことは必要
	pinMode(PIN_SPI_CS0,OUTPUT);
	digitalWrite(PIN_SPI_CS0,1);
	pinMode(PIN_SPI_CS1,OUTPUT);
	digitalWrite(PIN_SPI_CS1,1);
	pinMode(PIN_SPI_CS2,OUTPUT);
	digitalWrite(PIN_SPI_CS2,1);
	pinMode(PIN_SPI_CS3,OUTPUT);
	digitalWrite(PIN_SPI_CS3,1);
	pinMode(SPI_PORT_RAXINO_SDMMC,OUTPUT);
	digitalWrite(SPI_PORT_RAXINO_SDMMC,1);

	// 加速度センサを初期化
	SPI.port = SPI_PORT_RAXINO_ACCEL;
	SPI.setBitLength(16);
	SPI.transfer(0x2047); //

	unsigned long recv;
	recv = SPI.transfer(0x8f00) & 0xff; //
	Serial.print("ACCEL WHO_AM_I=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(" ");
	if((recv != 0) and (recv != 0xff))
	{
		Serial.print("Accel sensor found.");
		accel_sensor_exist = true;
	}

	recv = SPI.transfer(0xa000) & 0xff; //
	Serial.print("CTRL1_REG1=");
	Serial.print(recv,CSerial::HEX);
	Serial.println(" ");

	// コマンド(8bit)とデータ(24bit)を別々に送受信してみるテスト
	SPI.port = SPI_PORT_NONE; // CSを自分で制御する
	digitalWrite(PIN_SPI_CS2,0); // CS2を下げる
	SPI.setBitLength(8);
	SPI.transfer(0x9f); // コマンド送信
	SPI.setBitLength(24);
	recv = SPI.transfer(0); // データ(24bit)受信
	Serial.print("SPI ROM  JEDEC ID=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(",");
	digitalWrite(PIN_SPI_CS2,1); // CS2を下げる

	digitalWrite(PIN_SPI_CS2,0); // CS2を下げる
	SPI.setBitLength(32);
	SPI.transfer(0x90000000); // コマンド送信 90 00 00 00
	recv = SPI.transfer(0); // データ受信
	Serial.print("ID=");
	Serial.print(recv,CSerial::HEX);
	Serial.print(",");
	digitalWrite(PIN_SPI_CS2,1); // CS2を下げる

	// コマンド(8bit)とデータ(8bit)を一緒に送受信してみる
	// CSを自動で動かしてみるテスト
	SPI.port = SPI_PORT_RAXINO_ROM; // ポート番号を設定してCSを自動的に動かす
	SPI.setBitLength(16);
	recv = SPI.transfer(0x0500); // コマンド送信 05 00
	Serial.print("STATUS_REG=");
	Serial.print(recv & 0xff,CSerial::HEX);
	Serial.println("");

	delay(3000);
}

int anyreceived;

void loop()
{
	digitalWrite(PIN_LED0, count & 1);
	digitalWrite(PIN_LED1, count & 2);
	digitalWrite(PIN_LED2, count & 4);
	digitalWrite(PIN_LED3, count & 8);
	count++;

	unsigned long recv;

	// 加速度センサ
	SPI.port = SPI_PORT_RAXINO_ACCEL;
	SPI.setBitLength(16);

	if(Serial.available())
	{
		anyreceived = 1;
		Serial.read();
	}

	if(accel_sensor_exist) {
		const float fullscale = 2.3; // 2.3g g=重力加速度

		do {
			recv = SPI.transfer(0xa700) & 0xff; //
			if(recv & 0x80) { // オーバーランしている
				SPI.transfer(0xe900); // 読み捨て
				SPI.transfer(0xeb00);
				SPI.transfer(0xed00);
				continue;
			}
		} while ((recv & 0x08) == 0); //  新しいデータがないならループ

		int x,y,z;
		x = (signed char)(SPI.transfer(0xe900) & 0xff);
		y = (signed char)(SPI.transfer(0xeb00) & 0xff);
		z = (signed char)(SPI.transfer(0xed00) & 0xff);

/*
		if(anyreceived)
		{
			Serial.print("X=");
			Serial.print((int)(x/128.*fullscale*1000.),DEC);
			Serial.print("mg ");
			Serial.print("Y=");
			Serial.print((int)(y/128.*fullscale*1000.),DEC);
			Serial.print("mg ");
			Serial.print("Z=");
			Serial.print((int)(z/128.*fullscale*1000.),DEC);
			Serial.println("mg ");
		}
*/
		
		double dx,dy,dz;
		dx = (int)x * (int)x;
		dy = (int)y * (int)y;
		dz = (int)z * (int)z;
		if(digitalRead(PIN_SW) == 1)
		{
			tone_start(PIN_P51,dx+dy+dz + 256,0);
		}
		else
		{
			tone_start(PIN_P51,dx+dy+dz + 256,50);
			delay(50);
		}
		
		if(anyreceived)
		{
			Serial.print("X=");
			Serial.print((int)(dx),CSerial::DEC);
			Serial.print(" Y=");
			Serial.print((int)(dy),CSerial::DEC);
			Serial.print(" Z=");
			Serial.print((int)(dz),CSerial::DEC);
			Serial.print(" Freq=");
			Serial.print((int)sqrt(dx+dy+dz) + 256,CSerial::DEC);
			Serial.println("Hz ");
		}

	}
}
