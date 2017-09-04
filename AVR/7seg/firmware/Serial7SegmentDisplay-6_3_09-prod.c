/*
	Serial 7-Segment Display v1.0
    6-3-09
    Spark Fun Electronics
    Jim Lindblom
*/

#include <avr/io.h>
#include <avr/interrupt.h>

#define F_CPU 8000000
#define FOSC 8000000
#define BAUD 9600
#define MYUBRR FOSC/16/BAUD-1
#define BRIGHT_ADDRESS 0
#define UART_ADDRESS 2
#define DDR_SPI    PORTB
#define DD_MISO    PINB4

// SBI and CBI to set bits
#define sbi(port_name, pin_number)   (port_name |= 1<<pin_number)
#define cbi(port_name, pin_number)   ((port_name) &= (uint8_t)~(1 << pin_number))

//Declare functions
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void ioinit();
void display(char number, int digit);
void check_Special(void);
void delay_ms(uint16_t x);
void delay_us(uint16_t x);
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

//Declare global variables
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
char data0=0;	// Digit 0 data
char data1=0;	// Digit 1 data
char data2=0;	// Digit 2 data
char data3=0;	// Digit 3 data
int receiveCount = 0;	// Will count between 0 and 3
int uartMode = 0;
int spiMode = 1;
uint16_t bright_level;
char DPStatus = 0;	// Decimal point status, each bit represents one DP
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

// SPI interrupt, enter when SPIF is set
SIGNAL(SIG_SPI)
{
	// Set mode to SPI
	// if previously in UART mode, reset receiveCount
	spiMode = 1;
	
	if (uartMode)
	{
		uartMode = 0;
		receiveCount = 0;
	}

	switch(receiveCount)
	{
		case 0:
			data0 = SPDR;
			receiveCount++;
			break;
		case 1:
			data1 = SPDR;
			receiveCount++;
			break;
		case 2:
			data2 = SPDR;
			receiveCount++;
			break;
		case 3:
			data3 = SPDR;
			receiveCount = 0;
			break;
	}
	check_Special();
}

// UART interrupt, enter when receive a character over RX
SIGNAL(SIG_USART_RECV)
{
	uartMode = 1;
	if (spiMode)
	{
		spiMode = 0;
		receiveCount = 0;
	}

	switch(receiveCount)
	{
		case 0:
			data0 = UDR0;
			receiveCount++;
			break;
		case 1:
			data1 = UDR0;
			receiveCount++;
			break;
		case 2:
			data2 = UDR0;
			receiveCount++;
			break;
		case 3:
			data3 = UDR0;
			receiveCount = 0;
			break;
	}
	check_Special();
}

int main()
{
	ioinit();

	// Read brightness from EEPROM
	while(EECR & (1<<EEPE))
		;						// Wait for completion of previous write
	EEAR = BRIGHT_ADDRESS;		// Set up address register
	EECR |= (1<<EERE);			// Start eeprom read by writing EERE
	bright_level = 118*EEDR;	// Return data from Data Register */

	// Read UART value from EEPROM
	while(EECR & (1<<EEPE))
		;						// Wait for completion of previous write
	EEAR = UART_ADDRESS;		// Set up address register
	EECR |= (1<<EERE);			// Start eeprom read by writing EERE

	switch(EEDR)
	{
		case 0:	// 2400
			UBRR0H = 207 >> 8;
    		UBRR0L = 207;
			break;
		case 1:	// 4800
			UBRR0H = 103 >> 8;
    		UBRR0L = 103;
			break;
		case 2:	// 9600
			UBRR0H = 51 >> 8;
    		UBRR0L = 51;
			break;
		case 3:	// 14400
			UBRR0H = 34 >> 8;
    		UBRR0L = 34;
			break;
		case 4:	// 19200
			UBRR0H = 25 >> 8;
    		UBRR0L = 25;
			break;
		case 5:	// 38400
			UBRR0H = 12 >> 8;
    		UBRR0L = 12;
			break;
		case 6:	// 57600
			UBRR0H = 8 >> 8;
    		UBRR0L = 8;
			break;
	}

	// Main loop: update display
	while(1)
	{
		//if((data0!='z')&&(data1!='z')&&(receiveCount==0))
		if((data0!='z')&&(data0!='w')&&(data0!='y'))
		{
			// Display numbers
			display(data0, 0); 
			delay_us(5);
			display(data1, 1); 
			delay_us(5);
			display(data2, 2); 
			delay_us(5);
			display(data3, 3); 
			delay_us(5);
			display(0, 4);
			delay_us(5);
		}

		// clear display
		PORTC = PORTC | 0b0011111;;
		PORTD = 0;
		delay_us(bright_level);
	}
}

void ioinit()
{
	char junk;

	sei();	// Enable interrupts

	DDRC = DDRC | 0b0111111;	// Set data direction for port C (DIG1,...DP)
	PORTC = PORTC | 0b0011111;	// Initialize all digits off
	DDRD = DDRD | 0b11111110;	// Set data direction for port D (A,B,...,G)
	PORTD = PORTD | 0b00000000;	// Initialize all digits off

	//Init Timer0 for delay_us
	//Set Prescaler to clk/8 : 1click = 1us. CS01=1 
	TCCR0B = (1<<CS01); 

	// intialize USART Baud rate: 9600
	// enable rx and rx interrupt
    UBRR0H = MYUBRR >> 8;
    UBRR0L = MYUBRR;
    UCSR0B = (1<<RXCIE0)|(1<<RXEN0);

	/* Set MISO output, all others input */
	DDR_SPI = (1<<DD_MISO);
	/* Enable SPI */
	SPCR = (1<<SPIE) | (1<<SPE);

	junk = SPDR;
}

// Output number to digit 0,1,2, or 3, 4 to display dots
void display(char number, int digit)
{
	// Clear display initially
	PORTC = 0b0011111;;
	PORTD = 0;
	if (number != 'x')
		cbi(PORTC, digit);	// Turn on corresponding digit

	if (digit == 4)
	{	
		// Digit 4 is COL dots
		if (DPStatus & (1<<4))
			PORTD = PORTD | 0b00000010;
		if (DPStatus & (1<<5))
			PORTD = PORTD | 0b00000100;
		if (DPStatus & (1<<6))
			PORTD = PORTD | 0b00001000;
	}
	else
	{
		switch(number)	// Set PORTD, display pins, to correct output
		{
			case 0:
			case '0':
				PORTD = 0b01111110;
				break;
			case 1:
			case '1':
				PORTD = 0b00001100;
				break;
			case 2:
			case '2':
				PORTD = 0b10110110;
				break;
			case 3:
			case '3':
				PORTD = 0b10011110;
				break;
			case 4:
			case '4':
				PORTD = 0b11001100;
				break;
			case 5:
			case '5':
				PORTD = 0b11011010;
				break;
			case 6:
			case '6':
				PORTD = 0b11111010;
				break;
			case 7:
			case '7':
				PORTD = 0b00001110;
				break;
			case 8:
			case '8':
				PORTD = 0b11111110;
				break;
			case 9:
			case '9':
				PORTD = 0b11011110;
				break;
			case 10:
			case 'a':
			case 'A':
				PORTD = 0b11101110;
				break;
			case 11:
			case 'b':
			case 'B':
				PORTD = 0b11111000;
				break;
			case 12:
			case 'c':
			case 'C':
				PORTD = 0b01110010;
				break;
			case 13:
			case 'd':
			case 'D':
				PORTD = 0b10111100;
				break;
			case 14:
			case 'e':
			case 'E':
				PORTD = 0b11110010;
				break;
			case 15:
			case 'f':
			case 'F':
				PORTD = 0b11100010;
				break;
		}

		// Turn on decimal points depending on DPStatus
		if ((DPStatus & (1<<0))&&(digit==0))
		{
			cbi(PORTC, digit);
			PORTC = PORTC | 0b0100000;
		}
		if ((DPStatus & (1<<1))&&(digit==1))
		{
			cbi(PORTC, digit);
			PORTC = PORTC | 0b0100000;
		}
		if ((DPStatus & (1<<2))&&(digit==2))
		{
			cbi(PORTC, digit);
			PORTC = PORTC | 0b0100000;
		}
		if ((DPStatus & (1<<3))&&(digit==3))
		{
			cbi(PORTC, digit);
			PORTC = PORTC | 0b0100000;
		}
	}
}

void check_Special(void)
{
	// If sent special character z
	// Update brightness
	if ((data0=='z')&&(receiveCount==2))
	{
		// Write bright_level into EEPROM
		/* Wait for completion of previous write */
		while(EECR & (1<<EEPE))
			;
		/* Set up address and Data Registers */
		EEAR = BRIGHT_ADDRESS;
		EEDR = data1;	// Write data1 into EEPROM
		/* Write logical one to EEMPE */
		EECR |= (1<<EEMPE);
		/* Start eeprom write by setting EEPE */
		EECR |= (1<<EEPE);

		bright_level = 118*data1;
		
		// Clear non-displayable data, reset receiveCount
		receiveCount = 0;
		data0 = 'x';
		data1 = 'x';
		data2 = 'x';
		data3 = 'x';
	}
	// If sent special character w
	// Update DPStatus
	if ((data0=='w')&&(receiveCount==2))
	{
		DPStatus = data1;

		// Clear non-displayable data, reset receiveCount
		receiveCount = 0;
		data0 = 'x';
		data1 = 'x';
		data2 = 'x';
		data3 = 'x';
	}
	// If sent special character y
	// Update baud rate
	// Warning: numbers are static and depend on 8MHz clock
	if ((data0=='y')&&(receiveCount==2))
	{
		switch(data1)
		{
			case 0:	// 2400
				UBRR0H = 207 >> 8;
    			UBRR0L = 207;
				break;
			case 1:	// 4800
				UBRR0H = 103 >> 8;
    			UBRR0L = 103;
				break;
			case 2:	// 9600
				UBRR0H = 51 >> 8;
    			UBRR0L = 51;
				break;
			case 3:	// 14400
				UBRR0H = 34 >> 8;
    			UBRR0L = 34;
				break;
			case 4:	// 19200
				UBRR0H = 25 >> 8;
    			UBRR0L = 25;
				break;
			case 5:	// 38400
				UBRR0H = 12 >> 8;
    			UBRR0L = 12;
				break;
			case 6:	// 57600
				UBRR0H = 8 >> 8;
    			UBRR0L = 8;
				break;
		}

		// Write updated UART value to EEPROM
		/* Wait for completion of previous write */
		while(EECR & (1<<EEPE))
			;
		/* Set up address and Data Registers */
		EEAR = UART_ADDRESS;
		EEDR = data1;	// Write data1 into EEPROM
		/* Write logical one to EEMPE */
		EECR |= (1<<EEMPE);
		/* Start eeprom write by setting EEPE */
		EECR |= (1<<EEPE);

		// Clear non-displayable data, reset receiveCount
		receiveCount = 0;
		data0 = 'x';
		data1 = 'x';
		data2 = 'x';
		data3 = 'x';
	}
}

// Long delays
void delay_ms(uint16_t x)
{
	for (; x > 0 ; x--)
	{
		delay_us(250);
		delay_us(250);
		delay_us(250);
		delay_us(250);
	}
}

// For short delays
void delay_us(uint16_t x)
{
	if (x != 0)
	{
		while(x > 256)
		{
			TIFR0 = (1<<TOV0); //Clear any interrupt flags on Timer2
			TCNT0 = 0; //256 - 125 = 131 : Preload timer 2 for x clicks. Should be 1us per click
			while( (TIFR0 & (1<<TOV0)) == 0);
			
			x -= 256;
		}
	
		TIFR0 = (1<<TOV0); //Clear any interrupt flags on Timer2
		TCNT0 = 256 - x; //256 - 125 = 131 : Preload timer 2 for x clicks. Should be 1us per click
		while( (TIFR0 & (1<<TOV0)) == 0);
	}
}
