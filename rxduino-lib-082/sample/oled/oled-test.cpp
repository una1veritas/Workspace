// RXduinoでOLEDを操作するサンプル
// (C)特殊電子回路

// RXduinoライブラリを使う
#include <rxduino.h>

// OLEDライブラリを使う
#include <oled.h>

#include <stdlib.h>

#define MARY1 1
#define MARY2 2

void setup()
{
	Serial.begin(38400);

	pinMode(PIN_SW,INPUT);
	pinMode(PIN_LED3,OUTPUT);

	Init_OLED(MARY2);
	OLED_printf_Font(OLED_FONT_MEDIUM);
	OLED_printf_Position(0, 0);
	OLED_printf_Color(OLED_WHT, OLED_BLK);

    OLED_printf_Font(OLED_FONT_MEDIUM);
    OLED_printf_Position(0, 1);
    OLED_printf_Color(OLED_WHT, OLED_BLK);
    OLED_printf(MARY2,"TOKUSHU");
    OLED_printf_Position(1, 3);
    OLED_printf_Color(OLED_YEL, OLED_BLK);
	OLED_printf(MARY2,"DENSHI");
    OLED_printf_Position(1, 5);
    OLED_printf_Color(OLED_BLU, OLED_BLK);
	OLED_printf(MARY2,"KAIRO");
    OLED_printf_Position(2, 7);
    OLED_printf_Color(OLED_CYN, OLED_BLK);
	OLED_printf(MARY2,"Inc.");

	delay(1000);
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
	color = (rand() << 1) ^ rand();

	if(digitalRead(PIN_SW))
	{
		OLED_Fill_Rect(MARY2,x1,y1,x2-x1,y2-y1,color);
	}
	else
	{
		OLED_move_position(MARY2,0,0);
		unsigned short *ptr = (unsigned short *)0x00000000;
		ptr += count;
		for(int i=0;i<128*128;i++) OLED_Send_Pixel(MARY2,*ptr++);
	}
}
