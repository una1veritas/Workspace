#include <pololu/orangutan.h>

/*
 * lcd1: for the Orangutan controllers and 3pi robot
 *
 * Uses the OrangutanLCD library to display things on the LCD.
 *
 * http://www.pololu.com/docs/0J20
 * http://www.pololu.com
 * http://forum.pololu.com
 */

int main()
{
  while(1)
  {
    print("Hello");           // display "Hello" at (0, 0), a.k.a. upper-left
    delay_ms(200);
    lcd_scroll(LCD_RIGHT, 3, 200);// shift the display right every 200ms three times
    clear();                  // clear the LCD
    lcd_goto_xy(3, 1);             // go to the fourth character of the second LCD line
    print("Hello");           // display "Hello" at (3, 1), a.k.a. lower-right
    delay_ms(200);
    lcd_scroll(LCD_LEFT, 3, 200); // shift the display left every 200ms three times
    clear();                  // clear the LCD
  }
}

