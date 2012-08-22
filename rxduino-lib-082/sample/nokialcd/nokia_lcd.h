// (C) Copyright 2011 Tokushu Denshi Kairo Inc.

/*
 * nokia_lcd.h
 *
 *  Created on: 2011/07/12
 *      Author: atsushi
 */

#ifndef NOKIA_LCD_H_
#define NOKIA_LCD_H_


#define LCD_ROWS 16
#define LCD_COLS 16
#define LCD_WIDTH 130
#define LCD_HEIGHT 130
#define LCD_FREQUENCY 5000000

void	NokiaLCD_reset(void);
void	NokiaLCD_locate(int column, int row);
void	NokiaLCD_newline(void);
void	NokiaLCD_putp(int colour);	// Call this by user
void	NokiaLCD_pixel(int x, int y, int colour);
int		NokiaLCD_putc(int value);
void	NokiaLCD_cls(void);
void	NokiaLCD_window(int x, int y, int width, int height);	// Call this by user.
void	NokiaLCD_fill(int x, int y, int width, int height, int colour);
void	NokiaLCD_blit(int x, int y, int width, int height, const int* colour);
void	NokiaLCD_bitblit(int x, int y, int width, int height, const char* bitstream);
void	NokiaLCD_foreground(int c);
void	NokiaLCD_background(int c);
int		NokiaLCD_width(void);
int		NokiaLCD_height(void);
int		NokiaLCD_columns(void);
int		NokiaLCD_rows(void);

#endif /* NOKIA_LCD_H_ */
