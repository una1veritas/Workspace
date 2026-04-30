/*
	153.6kHz baud rate generator
	Device: PIC12F1822
	Compiler: XC8
*/

#include <xc.h>

#pragma config FOSC = INTOSC
#pragma config WDTE = OFF
#pragma config MCLRE = ON
#pragma config CLKOUTEN = OFF
#pragma config PLLEN = ON

void main(){
	OSCCON = 0b01110000; //32MHz
	OSCTUNE = 63; //tune
	ANSELA = 0; //digital
	nWPUEN = 0;
	TRISA  = 0b11111011;

	CCP1CON = 0b00001100; //PWM mode
	PR2 = 51; //PWM cycle
	CCPR1L = PR2 / 2; //duty cycle
	T2CON = 0; //not use prescale
	TMR2ON = 1; //TMR2 start
	while(1);
}