// RX62N��GCC�T���v���v���O����
// PWM�ɂ��A�i���O�o��
// (C)Copyright 2012 ����d�q��H

// �g����
// ���̃v���O�����̃t�@�C������main.cpp�ɕύX���āAmake���Ă��������B

// ���dHAL
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
