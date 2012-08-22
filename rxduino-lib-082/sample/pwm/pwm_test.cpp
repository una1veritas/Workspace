// RXduinoのサンプルプログラム

// Arduinoのスケッチ風に簡単にプログラムが作れます

#include <rxduino.h>

void setup()
{
	Serial.begin(38400);
	Serial.println("Hello RXduino!");
	pinMode(PIN_LED0,OUTPUT);
	pinMode(PIN_LED1,OUTPUT);
	pinMode(PIN_LED2,OUTPUT);
	pinMode(PIN_LED3,OUTPUT);
	pinMode(PIN_SW,INPUT);
}

int x = 0;

unsigned char ledbright(unsigned char val)
{
	return (unsigned long)val * (unsigned long)val * (unsigned long)val / 65536;
}

int anyreceived;

void loop()
{
	analogWrite(PIN_LED0,ledbright(x));
	analogWrite(PIN_LED1,ledbright(x + 64));
	analogWrite(PIN_LED2,ledbright(x + 128));
	analogWrite(PIN_LED3,ledbright(x + 192));
	x++;
	x = x & 255;

/*
	int mtu0,mtu1;
	mtu0 = MTU0.TCNT;
	mtu1 = MTU1.TCNT;
	Serial.print("MTU0=");
	Serial.print(mtu0);
	Serial.print(" MTU1=");
	Serial.print(mtu1);
	Serial.print("\n");
*/

	if(Serial.available())
	{
		anyreceived = 1;
		char c = Serial.read();
		analogWrite(PIN_ARD0,x);
		
		digitalWrite(PIN_LED0,c & 1);
		digitalWrite(PIN_LED1,c & 2);
		digitalWrite(PIN_LED2,c & 4);
		digitalWrite(PIN_LED3,c & 8);
		Serial.write(c);
	}
	else
	{
		timer_wait_ms(5);
	}
}
