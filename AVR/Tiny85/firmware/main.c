#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <avr/sleep.h>
#include <avr/pgmspace.h>

#include "wiring.h"
#include "melodies.h"

#define maskByteWrite(port, mask, vals)  ((port) = ((port)&(~(mask))) | (vals & mask))

volatile word t0compa_cnt;
//volatile word t1ovf_cnt;
volatile byte tcnt1h;

ISR(TIM0_COMPA_vect) {
   t0compa_cnt++;
   // automatically adc starts w/ compa interrupt
}

ISR(TIM1_OVF_vect) {
   tcnt1h++;
   //	t1ovf_cnt++;
}


ISR(ADC_vect) {
   // only to wake up, do nothing specially
}

void init() {
   
   maskByteWrite(ADCSRA, 0b0111<<ADPS0, 6<<ADPS0); // sampling prescaler
   maskByteWrite(ADMUX, 0b11<<REFS0, 0<<REFS0); // ref V. select
   bitWrite(ADMUX, ADLAR, 0); // result align left
   maskByteWrite(ADMUX, 0b1111, 0b0001); // select adc channel
   bitSet(ADCSRA, ADEN); // enable adc
   bitSet(ADCSRA, ADATE); // auto trigger enable
   bitSet(ADCSRA, ADIE); // adc complete interrupt enable
   maskByteWrite(ADCSRB,0b111<<ADTS0, 0b011<<ADTS0); // Timer/Counter0 Compare Match A
   
   
   maskByteWrite(TCCR0A, 0b1111<<COM0B0, 0<<COM0B0); 
   maskByteWrite(TCCR0A, 0b11<<WGM00, 0b10<<WGM00); //CTC
   maskByteWrite(TCCR0B, 0b1<<WGM02, 0<<WGM02);
   maskByteWrite(TCCR0B, 0b111<<CS00, 0b101<<CS00); // prescaler 0b100 = 1/256, 0b101 = 1/1024
   OCR0A = 32;
   bitSet(TIMSK, OCIE0A);
   bitSet(ACSR, ACD);
   
   TCCR1 |= 0<<CTC1 | 0b0101<<CS10;  // 0101 -- 1/16
   GTCCR |= 0<<PWM1B | 0<<COM1B1 | 0<<COM1B0;
   //OCR1C = 99;
   //	bitSet(TIMSK, TOIE1);
   
   // set portb pb3 (pin 2) as output, low
   bitSet(DDRB, PB3);
   asm("nop");
   bitClear(PORTB, PB3);
   
   // set portb pb4 (pin 3) as closed (input w/ high-impedance)
   //	bitSet(DDRB, PB4);
   bitClear(DDRB, PB4);
   asm("nop");
   bitClear(PORTB, PB4);
   
   return;
}

word tone[] PROGMEM = {
   0,
   //	1920, // C1
   //	1800, 
   //	1724, // D1
   //	1600,
   //	1526, // E1
   //	1450, // F1
   //	1364,
   1298, // G1 1
   1200,
   1152, // A1 440Hz 3
   1030,
   1030, // B1  5
   970,  // C2 6
   970,		
   862,  // D2	8
   863,
   763, // E2	10
   725, // F	11
   672, // F#
   646,  // G, 13
   647,
   575,  // A  15
   548,
   504,  // B	17
   480,  // C3, 18
   454,
   428,  //D3  20
   408,
   379,	//E3 22
   362,	//F3 23
   342,
   320,	//G  25
   307,
   286,	//A  27
   276,
   249,	//B  29
   238,	//C4 30
   228,	//
   214,	//D  32
   202,
   190,	//E  34
   182,	//F  35
   176,	
   160,	//G		37
   152,	
   141		//A	 39
};

void music(byte s) {
   int osc0tc;
   boolean osc0on;
   byte * score = melody(s);
   byte notenum;
   word cycle, duration;
   byte sTCNT1;
   byte diff;
   
   // set portb pb4 (pin 3) as output
   bitSet(DDRB, PB4);
   
   //	bitClear(TIMSK, OCIE0A);
   bitSet(TIMSK, TOIE1);
   
   osc0tc = 0;
   osc0on = false;
   t0compa_cnt = 0;
   sTCNT1 = TCNT1;
   while (true) {
	if ( t0compa_cnt ) {
	   if ( duration > 0 ) {
		duration--;
		if ( duration < 3 ) 
		   osc0on = false;
	   }
	   t0compa_cnt--;
	}
	if ( duration == 0 ) {
	   notenum = pgm_read_byte(score++);
	   cycle = pgm_read_word(tone+ notenum);
	   duration = pgm_read_byte(score++) * 2;
	   if (cycle)
		osc0on = true;
	   else
		osc0on = false;
	   if ( duration == 0 && cycle == 0) 
		break;
	}
	
	//		wave = 0;
	diff = TCNT1 - ((signed char) sTCNT1);
	sTCNT1 += diff;
	if ( osc0on ) {
	   osc0tc -= diff;
	   if ( osc0tc < 0 ) {
		osc0tc += /* osc0 */ cycle;
		//				wave++;
		bitSet(PINB, PB4);
	   }
	}
   }
   
   bitClear(TIMSK, TOIE1);
   //	bitSet(TIMSK, OCIE0A);
   
   // set portb pb4 (pin 3) as closed (input w/ high-impedance)
   bitClear(DDRB, PB4);
   asm("nop");
   bitClear(PORTB, PB4);	
}


#define SEQMAX 16

void analyzeSeq(int seq[]);

int main(void) {
   int seq[SEQMAX];
   word adc1val, sinceTriggered;
   int dumax = 0;
   byte q = 0;
   boolean listening = false;
   
   
   init();
   
   sei();
   music(0);
   
   for (;;){
	
	adc1val = 0;
	//
	set_sleep_mode(0);
	sleep_mode();
	
	if ( ! bitRead(ADCSRA, ADSC) ) // adc completed.
	{
	   adc1val = ADCL + (ADCH<<8);
	   if ( adc1val > 10 ) {
		sinceTriggered = t0compa_cnt;
	   }
	}
	
	if ( adc1val > 10 && (sinceTriggered > 48)) {
	   if ( listening ) {
		seq[q++] = t0compa_cnt;
		seq[q] = 0;
		dumax = max(dumax, t0compa_cnt);
		t0compa_cnt = 0;
	   } else {
		listening = true;
		q = 0;
		seq[q] = 0;
		dumax = 0;
		t0compa_cnt = 0;
	   }
	}
	
	if (listening ) {
	   if (q >= SEQMAX) {
		listening = false;
	   } else 
		if ( (dumax != 0) && (t0compa_cnt > dumax*4) ) {
		   listening = false;
		} else 
		   if ( t0compa_cnt > 400 ) {
			listening = false;
		   }
	}
	
	if ( (!listening) && q > 0 ) {
	   analyzeSeq(seq);
	   q = 0;
	} else {
	   bitClear(PORTB, PB3);
	}
   }
   return 0;               /* never reached */
}

void	analyzeSeq(int seq[]) {
   int dmin, dmax;
   int base;
   byte pattern[3][4] = {
	{ 1, 1, 2, 0 },
	{ 2, 1, 1, 0 },
	{ 1, 2, 1, 0 }
   };
   
   
   bitSet(PORTB, PB3);
   _delay_ms(12);
   bitClear(PORTB, PB3);
   
   dmax = seq[0];
   dmin = dmax;
   int i;
   for (i = 0; seq[i] > 0 && i < SEQMAX; i++) {
	dmax = max(dmax, seq[i]);
	dmin = min(dmin, seq[i]);
	_delay_ms(4*(seq[i]-3));
	bitSet(PORTB, PB3);
	_delay_ms(12);
	bitClear(PORTB, PB3);
   }
   base = (dmax + (dmin/2)) / dmin;
   base = dmax/8 / base;
   for (i = 0; seq[i] > 0 && i < SEQMAX; i++) {
	seq[i] = (seq[i] + (base/2) ) / base;
   }
   int n;
   int error;
   for ( n = 0; n < 3; n++) {
	error = 0;
	for (i = 0; seq[i] != 0 && pattern[n][i] != 0; i++) {
	   error += abs(seq[i] - pattern[n][i]*8);
	}
	if ( error > 5 ) 
	   continue;
	if ( i > 0 && (seq[i] == 0 && pattern[n][i] == 0) ) {
	   bitSet(PORTB, PB3);
	   switch(n) {
		case 0:
		   music(1);
		   break;
		case 1:
		   music(2);
		   break;
		case 2:
		   music(3);
		   break;
	   }
	   bitClear(PORTB, PB3);
	}
   }
}
