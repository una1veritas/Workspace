#include "wiring.h"

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "sinwave.h"

#include "serial.h"

// globals for interrupts
volatile byte treduesec = 0;
volatile word sec32th = 0;
volatile int 
	blue = 0, // 0a
	green = 0, // 0b
	red = 0; // 1a

ISR(TIMER0_COMPA_vect) {
	OCR0A = (byte) blue;
}

ISR(TIMER0_COMPB_vect) {
	OCR0B = (byte) green;
}

ISR(TIMER1_COMPA_vect) {
	OCR1AL = (byte) red;
}

ISR(TIMER2_COMPA_vect) {
	treduesec++;
	sec32th++;
	if (sec32th > 60*32 - 1)
		sec32th %= 60*32;
}

/*
#define floor(x) ((long)(x))

long JulianDay(int y, int m, int d, int hour) {
	int a, b;
	if ( m < 3 ) {
		m += 12;
		y -= 1;
	}
	a = y/100; b = 2 - a + (a/4);
	return 365L * y + (long)(30.6001*(m+1)) + d + (long)(1720994.5)+ b;
}
*/

void init() {
	//Timer 0 setup for pwm
	TCCR0A = (0b10 << COM0A0) | (0b10 << COM0B0) | (0b11 << WGM00);
	TCCR0B = 0b010 << CS00;
	OCR0A = 0;
	OCR0B = 0;
	TIMSK0 = (1<<OCIE0A) | (1<<OCIE0B);
	bitSet(DDRD, PD5);
	bitSet(DDRD, PD6);

	// Timer 1 setup for pwm
	TCCR1A = (0b10 << COM1A0) | (0 << COM1B0) | (0b01 << WGM10);
	TCCR1B = (0b01 << WGM12) | (0b010 << CS10);
//	OCR1AH = 0; // will be ignored
	OCR1AL = 0;
	TIMSK1 = (1<<OCIE2A);
	bitSet(DDRB, PB1);
	
	// Timer 2 setup for Async Counter
	TCCR2A = (0b10 << WGM20);
	TCCR2B = (0b111 << CS00);
	OCR2A = 0;
	bitSet(ASSR, AS2);
	bitSet(TIMSK2,OCIE2A);
	
	bitClear(DDRD, PD7); // input
	bitSet(PORTD, PD7);  // weak pull-up
	bitClear(DDRB, PB0); // input
	bitSet(PORTB, PB0);  // weak pull-up
	
	serialBegin(9600);
}

#define LCD_CLEARDISPLAY 0x01
#define LCD_RETURNHOME 0x02
void LCD_command(byte cmd);

int main(void) {
	byte sec, minu, hour, day;
	int	blueyellow = 0, redgreen = 0;
	int i;
	byte updated = !0;
	char mes[20];
	
	init();
	
	LCD_command(LCD_CLEARDISPLAY);
	LCD_command(LCD_RETURNHOME);
	
	sei();
    for(;;){
		blueyellow = sinwave8(sec32th/8);
		redgreen = sinwave8(sec32th/8+periodwave8()/4);
		blue = min(max(blueyellow*1+redgreen*0, 0), 255);
		red = min(max(blueyellow*(-0.866) + redgreen*1, 0), 255);
		green = min(max(blueyellow*(-0.866) + redgreen*(-1), 0)*0.8, 255);
		
		if (bitRead(PIND, PD7) == 0 ) {
			treduesec += 4;
		} else if ( bitRead(PINB, PB0) == 0 ) {
			sec++;
		}
		
		if ( treduesec >= 32 ) {			
			treduesec %= 32;
			sec++;
			updated = !0;
		}
		if (sec >= 60) {
			sec -= 60;
			minu++;
		}
		if ( minu >= 60 ) {
			minu -= 60;
			hour++;
		}
		if ( hour >= 24 ) {
			hour -= 24;
			day++;
		}
		
		if (updated) {
			updated = 0;
			sprintf(mes, "%d%d:%d%d:%d%d", hour/10, hour%10, minu/10, minu%10, sec/10, sec%10);
			serialPrint(mes);
			LCD_command(LCD_RETURNHOME);
		}
    }
    return 0;               /* never reached */
}

void LCD_command(byte cmd) {
	serialWrite(0xfe);
	_delay_ms(5);
	serialWrite(cmd);
	_delay_ms(5);
}

