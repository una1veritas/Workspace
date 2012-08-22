// RX62N��GCC�T���v���v���O����
// GPIO�T���v��
// (C)Copyright 2011 ����d�q��H

// RXDuino���g��
#include <rxduino.h>

int count = 0;

void setup()
{
	Serial.begin(38400);
	Serial.println("");
	Serial.println("GPIO sample program (RXduino version)");

	pinMode(PIN_LED0 , OUTPUT); // LED0���o��
	pinMode(PIN_LED1 , OUTPUT); // LED1���o��
	pinMode(PIN_LED2 , OUTPUT); // LED2���o��
	pinMode(PIN_LED3 , OUTPUT); // LED3(BUZZ)���o��
	pinMode(PIN_SW   , INPUT);   // SW�����
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

