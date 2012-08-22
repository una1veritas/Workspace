// キャラクタ液晶のライブラリ
// (C)Copyright 2011 特殊電子回路

#include <rxduino.h>
#include "liquidcrystal.h"

#define D0 0
#define D1 1
#define D2 2
#define D3 3
#define D4 4
#define D5 5
#define D6 6
#define D7 7
#define RS 8
#define RW 9
#define ENABLE 10

void LiquidCrystal::init()
{
	for(int i=0;i<11;i++) pinnum[i] = -1;
	cols = 40;
	rows = 2;
	entry_mode = 0x02;
	display_mode = 0x07;
	cursol_mode = 0x00;
	mode4bit = 0;
	writeDelay = 0;
}

LiquidCrystal::LiquidCrystal(int rs, int enable, int d4, int d5, int d6, int d7)
{
	init();
	pinnum[D4] = d4;
	pinnum[D5] = d5;
	pinnum[D6] = d6;
	pinnum[D7] = d7;
	pinnum[RS] = rs;
	pinnum[ENABLE] = enable;
}

LiquidCrystal::LiquidCrystal(int rs, int rw, int enable, int d4, int d5, int d6, int d7)
{
	init();
	pinnum[D4] = d4;
	pinnum[D5] = d5;
	pinnum[D6] = d6;
	pinnum[D7] = d7;
	pinnum[RS] = rs;
	pinnum[RW] = rw;
	pinnum[ENABLE] = enable;
}

LiquidCrystal::LiquidCrystal(int rs, int enable, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
	init();
	pinnum[D0] = d0;
	pinnum[D1] = d1;
	pinnum[D2] = d2;
	pinnum[D3] = d3;
	pinnum[D4] = d4;
	pinnum[D5] = d5;
	pinnum[D6] = d6;
	pinnum[D7] = d7;
	pinnum[RS] = rs;
	pinnum[ENABLE] = enable;
}

LiquidCrystal::LiquidCrystal(int rs, int rw, int enable, int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
	init();
	pinnum[D0] = d0;
	pinnum[D1] = d1;
	pinnum[D2] = d2;
	pinnum[D3] = d3;
	pinnum[D4] = d4;
	pinnum[D5] = d5;
	pinnum[D6] = d6;
	pinnum[D7] = d7;
	pinnum[RS] = rs;
	pinnum[RW] = rw;
	pinnum[ENABLE] = enable;
}

void LiquidCrystal::begin(int cols,int rows)
{
	this->cols = cols;
	this->rows = rows;
	for(int i=0;i<11;i++)
	{
		pinMode(pinnum[i],OUTPUT);
		digitalWrite(pinnum[i],LOW);
	}

	mode4bit = 0;
	// Function set 8bit
	send_control(0x30);
	delay(5);
	// Function set 8bit
	send_control(0x30);
	// Function set 8bit
	send_control(0x30);
	// Function set 4bit
	send_control(0x20);
	// Function set 4bit
	mode4bit = 1;
	send_control(0x28); // 4bit 2row mode

	noDisplay();
	clear();
	home();
	noCursor();
	noBlink();
	noAutoscroll();
	leftToRight();
	display();
}

void LiquidCrystal::write(unsigned char data)
{
	if(data == '\n')
	{
		setCursor(0,1);
		col = 0;
		row = 1;
		return;
	}
	send_data(data);
	col++;
}

void LiquidCrystal::print(const char *str)
{
	while(*str)
	{
		write(*str++);
		if(writeDelay) delay(writeDelay);
	}
}

void LiquidCrystal::print(int val)
{
	unsigned long tmpval;

	tmpval = val;
	if(val < 0)
	{
		sci_putc('-');
		tmpval = -val;
	}

	char buf[12];
	buf[11] = '\0';

	int i;
	for(i = 10; i > 0 ; i--)
	{
		buf[i] = (tmpval % 10) | 0x30;
		tmpval = tmpval / 10;
		if(tmpval == 0) break;
	}
	print(&buf[i]);
}

void LiquidCrystal::clear()
{
	send_control(0x01);
	delay(10);
	col = 0;
	row = 0;
}

void LiquidCrystal::home()
{
	send_control(0x02);
	col = 0;
	row = 0;
}

void LiquidCrystal::setCursor(unsigned char col,unsigned char row)
{
	send_control(0x80 | (row & 0x01) << 6 | (col & 0x1f));
	col = col;
	row = row;
}

void LiquidCrystal::blink()
{
	display_mode |= 0x01;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::noBlink()
{
	display_mode &= ~0x01;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::display()
{
	display_mode |= 0x04;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::noDisplay()
{
	display_mode &= ~0x04;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::cursor()
{
	display_mode |= 0x02;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::noCursor()
{
	display_mode &= ~0x02;
	send_control(0x08 | display_mode);
}

void LiquidCrystal::leftToRight()
{
	entry_mode |= 0x02;
	send_control(0x04 | entry_mode);
}

void LiquidCrystal::rightToLeft()
{
	entry_mode &= ~0x02;
	send_control(0x04 | entry_mode);
}

void LiquidCrystal::noAutoscroll()
{
	entry_mode &= ~0x01;
	send_control(0x04 | entry_mode);
}

void LiquidCrystal::autoscroll()
{
	entry_mode |= 0x01;
	send_control(0x04 | entry_mode);
}

void LiquidCrystal::scrollDisplayLeft()
{
	send_control(0x18);
}

void LiquidCrystal::scrollDisplayRight()
{
	send_control(0x10);
}

void LiquidCrystal::send_control(unsigned char val)
{
	if(mode4bit)
	{
		digitalWrite(pinnum[D4],(val & 0x10) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(val & 0x20) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(val & 0x40) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(val & 0x80) ? HIGH : LOW);
		digitalWrite(pinnum[RW],LOW);
		digitalWrite(pinnum[RS],LOW);
		digitalWrite(pinnum[ENABLE],HIGH);
		digitalWrite(pinnum[ENABLE],LOW);
		digitalWrite(pinnum[D4],(val & 0x01) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(val & 0x02) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(val & 0x04) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(val & 0x08) ? HIGH : LOW);
	}
	else
	{
		digitalWrite(pinnum[D0],(val & 0x01) ? HIGH : LOW);
		digitalWrite(pinnum[D1],(val & 0x02) ? HIGH : LOW);
		digitalWrite(pinnum[D2],(val & 0x04) ? HIGH : LOW);
		digitalWrite(pinnum[D3],(val & 0x08) ? HIGH : LOW);
		digitalWrite(pinnum[D4],(val & 0x10) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(val & 0x20) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(val & 0x40) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(val & 0x80) ? HIGH : LOW);
	}
	digitalWrite(pinnum[RW],LOW);
	digitalWrite(pinnum[RS],LOW);
	digitalWrite(pinnum[ENABLE],HIGH);
	digitalWrite(pinnum[ENABLE],LOW);
	delay(1);
}

void LiquidCrystal::send_data(unsigned char data)
{
	if(mode4bit)
	{
		digitalWrite(pinnum[D4],(data & 0x10) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(data & 0x20) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(data & 0x40) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(data & 0x80) ? HIGH : LOW);
		digitalWrite(pinnum[RW],LOW);
		digitalWrite(pinnum[RS],HIGH);
		digitalWrite(pinnum[ENABLE],HIGH);
		digitalWrite(pinnum[ENABLE],LOW);
		digitalWrite(pinnum[D4],(data & 0x01) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(data & 0x02) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(data & 0x04) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(data & 0x08) ? HIGH : LOW);
	}
	else
	{
		digitalWrite(pinnum[D0],(data & 0x01) ? HIGH : LOW);
		digitalWrite(pinnum[D1],(data & 0x02) ? HIGH : LOW);
		digitalWrite(pinnum[D2],(data & 0x04) ? HIGH : LOW);
		digitalWrite(pinnum[D3],(data & 0x08) ? HIGH : LOW);
		digitalWrite(pinnum[D4],(data & 0x10) ? HIGH : LOW);
		digitalWrite(pinnum[D5],(data & 0x20) ? HIGH : LOW);
		digitalWrite(pinnum[D6],(data & 0x40) ? HIGH : LOW);
		digitalWrite(pinnum[D7],(data & 0x80) ? HIGH : LOW);
	}
	digitalWrite(pinnum[RW],LOW);
	digitalWrite(pinnum[RS],HIGH);
	digitalWrite(pinnum[ENABLE],HIGH);
	digitalWrite(pinnum[ENABLE],LOW);
	delay(1);
	Serial.write(".");
}

void LiquidCrystal::createChar(unsigned char location,const unsigned char charmap[])
{
	send_control(0x40 | location);
	for(int i=0;i<8;i++) send_data(charmap[i]);
	setCursor(col,row);
}

