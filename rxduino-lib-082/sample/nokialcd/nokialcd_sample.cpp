#include <rxduino.h>
#include <stdlib.h>
#include "nokia_lcd.h"

void setup()
{
	Serial.begin(38400);

	pinMode(PIN_SW,INPUT);
	pinMode(PIN_LED3,OUTPUT);

	NokiaLCD_reset();
    NokiaLCD_background(0x0000FF);
    NokiaLCD_cls();
}

int count = 0;

void loop()
{
	Serial.print("count=");
	Serial.print(count++);
	Serial.println("");

	int x1,x2,y1,y2,color;
	x1 = rand() & 127;
	x2 = rand() & 127;
	y1 = rand() & 127;
	y2 = rand() & 127;
	if(x1 > x2) {int tmp;tmp = x1;x1 = x2;x2 = tmp;}
	if(y1 > y2) {int tmp;tmp = y1;y1 = y2;y2 = tmp;}

	unsigned char r,g,b;
	r = rand();g = rand();b = rand();
    NokiaLCD_fill(x1,y1,x2-x1,y2-y1,r << 16 | g << 8 | b);

	if(digitalRead(PIN_SW) == 0)
	{
		for(int i=0;i<100;i++)
		{
			digitalWrite(PIN_LED3,LOW);
			delay(1);
			digitalWrite(PIN_LED3,HIGH);
			delay(1);
		}
		delay(100);
	}
}
