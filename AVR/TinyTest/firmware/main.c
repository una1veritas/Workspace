#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "wiring.h"
#include "fixedmath.h"

#define bset(p, bp)				(p |= (1<<bp))
#define bclear(p, bp)			(p &= ~(1<<bp))
#define bflip(p, bp)			(p ^= (1<<bp))

word tone[] = {
	185, // C (60)
	176, // C#
	166, // D
	157, // D#
	148, // E
	140, // F
	133, // F#
	125, // G
	118, // G#
	110, // A
	106, // A#
	99, // B
	93, // C
	91,
	93
};

word note[] = {
	0, 2, 4, 5, 7, 9, 11, 12
};


inline void TC1_prescaler(byte cs) {
	TCCR1 &= ~(0b1111)<<CS10;
	TCCR1 |= (cs & 0b1111)<<CS10;
}

inline void TC1_CTC(byte f) {
	TCCR1 &= ~(1<<CTC1);
	TCCR1 |= (f & 1)<<CTC1;
}

inline void TC1B_outputmode(byte m) {
	GTCCR = (GTCCR & 0b11001111) | ((m & 0b11)<<COM1B0);
}

inline void PWM1B_enable(byte b) {
	GTCCR |= 1<<PWM1B;
}

inline void PWM1B_out(byte v) {
	OCR1B = v;
}

inline void TC1_period(byte p) {
	OCR1C = p;
}

volatile word t1ovf_cnt = 0;
volatile word wave_period;
volatile byte pwm_cycle = 100;
volatile byte amplitude = 0;

ISR(TIM1_OVF_vect) {
	t1ovf_cnt--;
	if ( t1ovf_cnt < 0 ) {
		t1ovf_cnt = wave_period;
		amplitude = (pwm_cycle) - amplitude;
	}
	PWM1B_out(amplitude);
}

void init() {
	/*
	PLLCSR |= bit(LSM) | bit(PLLE);
	_delay_us(100);
	while ( !bitRead(PLLCSR, PLOCK) );
	PLLCSR |= bit(PCKE);
	*/
	TC1B_outputmode(1);
	PWM1B_enable(1);
//	TC1_CTC(1);
	TC1_period(pwm_cycle);
	TC1_prescaler(1);
	PWM1B_out(0);
	
	bset(TIMSK, TOIE1);
	
	bset(DDRB, PB4);
}

int main(void) {
	int i = 0, j;
	init();
	
	amplitude = 30;
	sei();
    for(;;){
		wave_period = 200;//*tone[note[i]];
		_delay_ms(2000);
		i++;
		i %= 4;
	}
    return 0;               /* never reached */
}
