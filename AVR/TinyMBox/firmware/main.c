#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <avr/sleep.h>
#include <avr/pgmspace.h>

#include "wiring.h"
#include "melodies.h"

#define bitsClear(port, mask)  ((port) &= (~(mask)) )
#define bitsSet(port, mask)  ((port) |= (mask)) )

inline void speaker_on() {
  bitSet(DDRB, PB4);
  asm("nop");
  bitClear(PORTB, PB4);
}

inline void speaker_off() {
  bitClear(DDRB, PB4);
  asm("nop");
  bitClear(PORTB, PB4);
}

inline void led_on() {
  bitSet(PORTB, PB3);
}

inline void led_off() {
  bitClear(PORTB, PB3);
}

volatile word t0compa_cnt;
//volatile word t1ovf_cnt;
volatile byte tcnt1h;

ISR(TIM0_COMPA_vect) {
  t0compa_cnt++;
  // automatically adc starts w/ compa interrupt
}

ISR(TIM1_OVF_vect) {
  tcnt1h++;
  // t1ovf_cnt++;
}


ISR(ADC_vect) {
  // only to wake up, do nothing specially
}

void init() {
  byte t;
  
  // Timer/Counter 0
  bitsClear(TCCR0A, 0b11<<COM0A0 | 0b11 << COM0B0); // disconnect port from OC0A/OC0B
  
  // Waveform generation mode = CTC
  t = 2;
  bitWrite(TCCR0A, WGM00, t & (1<<0));
  bitWrite(TCCR0A, WGM01, t & (1<<1));
  bitWrite(TCCR0B, WGM02, t & (1<<2));
  
  // select Prescaler
  // prescaler 0b100 = 1/256, 0b101 = 1/1024
  t = 0b101;
  bitWrite(TCCR0B, CS02, t & (1<<2));
  bitWrite(TCCR0B, CS01, t & (1<<1));
  bitWrite(TCCR0B, CS00, t & (1<<0));
  
  OCR0A = 32;
  bitSet(TIMSK, OCIE0A);
  
  //TimerCounter1
  TCCR1 |= 0<<CTC1 | 0b0101<<CS10;  // 0101 -- 1/16
  GTCCR |= 0<<PWM1B | 0<<COM1B1 | 0<<COM1B0;
  //OCR1C = 99;
  bitSet(TIMSK, TOIE1);
  
  // PB3 -- led/motor
  // set portb pb3 (pin 2) as output, low
  bitSet(DDRB, PB3);
  asm("nop");
  bitClear(PORTB, PB3);
  
  // PB4 -- speaker
  // set portb pb4 (pin 3) as closed (input w/ high-impedance)
  //	bitSet(DDRB, PB4);
  speaker_on();
  
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
  682, // F#
  646,  // G, 13
  647,
  575,  // A  15
  575,
  504,  // B	17
  480,  // C3, 18
  480,
  431,  //D3  20
  408,
  380,	//E3 22
  362,	//F3 23
  342,
  323,	//G  25
  307,
  287,	//A  27
  276,
  252,	//B  29
  240,	//C4 30
  228,	//
  214,	//D  32
  202,
  190,	//E  34
  182,	//F  35
  176,	
  162,	//G		37
  152,	
  144		//A	 39
};

void music(byte s) {
  int osc0tc; //, osc1tc;
//  word osc0cycle, osc1cycle;
  boolean osc0on; //, osc1on;
//  int osc0duration, osc1duration;
  byte * score = melody(s);
  byte notenum;
  word cycle, duration;
//  byte wave;
  byte sTCNT1;
  byte diff;
  
  speaker_on();
  
  //	bitClear(TIMSK, OCIE0A);
  bitSet(TIMSK, TOIE1);
  
  osc0tc = 0;
/*  osc1tc = 0;
  osc0duration = 0;
  osc1duration = 0;
  osc0on = false;
  osc1on = false;
   */
  t0compa_cnt = 0;
  sTCNT1 = TCNT1;
  while (true) {
	if ( t0compa_cnt ) {
	  if ( /* osc0 */ duration > 0 ) {
		/* osc0 */ duration--;
		if ( /* osc0 */ duration < 2 ) 
		  osc0on = false;
	  }
	  t0compa_cnt--;
	}
	if ( /* osc0 */ duration <= 0 ) {
	  notenum = pgm_read_byte(score++);
	  cycle = pgm_read_word(tone+ (notenum & 0x7f));
	  duration = pgm_read_byte(score++)*4;
	  //			if ( notenum < 127 ) {
	  if (cycle)
		osc0on = true;
	  else
		osc0on = false;
//	  osc0cycle = cycle;
//	  osc0duration = duration;

	  if ( !duration && !cycle ) 
		break;
	}
	
	// wave = 0;
	diff = TCNT1 - ((signed char) sTCNT1);
	sTCNT1 += diff;
	if ( osc0on ) {
	  osc0tc -= diff;
	  if ( osc0tc < 0 ) {
		osc0tc += /* osc0 */ cycle;
		// wave++;
		bitSet(PINB, PB4);
	  }
	}
  }
  
  bitClear(TIMSK, TOIE1);
  //	bitSet(TIMSK, OCIE0A);
  // set portb pb4 (pin 3) as closed (input w/ high-impedance)
  speaker_off();
}


#define SEQMAX 16

void analyzeSeq(int seq[]);

int main(void) {
  int seq[SEQMAX];
  word adc1val, sinceTriggered;
  int dumax = 0;
  byte q = 0;
  boolean listening = false;
  
  long i = 0;
  
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
		led_on();
		_delay_ms(20);
		led_off();
		_delay_ms(20);

		sinceTriggered = t0compa_cnt;
	  } else {
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
	  } else if ( (dumax != 0) && (t0compa_cnt > dumax*4) ) {
		  listening = false;
	  } else if ( t0compa_cnt > 400 ) {
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
  byte pattern[2][4] = {
	{ 1, 1, 2, 0 },
	{ 2, 1, 1, 0 }
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
  for ( n = 0; n < 2; n++) {
	error = 0;
	for (i = 0; seq[i] != 0 && pattern[n][i] != 0; i++) {
	  error += abs(seq[i] - pattern[n][i]*8);
	}
	if ( error > 5 ) 
	  continue;
	if ( i > 0 && (seq[i] == 0 && pattern[n][i] == 0) ) {
	  if ( n == 0 ) {
		bitSet(PORTB, PB3);
		//	_delay_ms(2000);
		music(1);
		bitClear(PORTB, PB3);
	  } else {
		bitSet(PORTB, PB3);
		//	_delay_ms(2000);
		music(2);
		bitClear(PORTB, PB3);
	  }
	  break;
	}
  }
}
