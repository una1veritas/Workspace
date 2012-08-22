// RXduino�̃T���v���v���O����

// Arduino�̃X�P�b�`���ɊȒP�Ƀv���O���������܂�

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

	digitalWrite(PIN_LED1, 0);
	digitalWrite(PIN_LED2, 0);
	digitalWrite(PIN_LED3, 0);
}

int x = 0;

void loop()
{
	int tonepin = PIN_P51;
	int wait = 300;

	Serial.print("C");
    tone(tonepin,262,wait) ;  // �h
    delay(wait) ;

	Serial.print("D");
    tone(tonepin,294,wait) ;  // ��
    delay(wait) ;

	Serial.print("E");
    tone(tonepin,330,wait) ;  // �~
    delay(wait) ;

	Serial.print("F");
    tone(tonepin,349,wait) ;  // �t�@
    delay(wait) ;

	Serial.print("G");
    tone(tonepin,392,wait) ;  // �\
    delay(wait) ;

	Serial.print("A");
    tone(tonepin,440,wait) ;  // ��
    delay(wait) ;

	Serial.print("H");
    tone(tonepin,494,wait) ;  // �V
    delay(wait) ;

	Serial.print("C");
    tone(tonepin,523,wait) ;  // �h
    delay(wait) ;
}
