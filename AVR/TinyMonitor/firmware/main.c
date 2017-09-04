#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>


#define PORTBMode(bits)	(DDRB = ((unsigned char) (bits)))
#define PORTDMode(bits)	(DDRD = ((unsigned char) (bits)))

#define SetBit(port, bit)		((port) |= 0x01 << (bit))
#define SetByte(port, onebyte)		((port) |= (onebyte))
#define ClearBit(port, bit)		((port) &= ~(0x01<< (bit)))
#define ClearByte(port, onebyte)	((port) &= ~(onebyte))
#define TestBit(port, bit)		((port) & (0x01 << (bit)))


#define USART_BAUDRATE 9600 
#define BAUD_PRESCALE (((F_CPU / (USART_BAUDRATE * 16UL))) - 1)

/*
 .def	temp = r16	;general purpose variable
 .def	inchar = r17     ;char destined to go out the uart
 .def	outchar = r18     ;char coming in from the uart
 .def	inbytel = r19     ;lower byte for asci-hex conversion
 .def	inbyteh = r20     ;higher byte for asci-hex conversion
 .def	currentadd = r21  ;pointer to write address for first page of memory	
 */

void USART_Init( unsigned int baud );
void USART_Transmit( unsigned char data );
unsigned char USART_Receive( void );

volatile void send_char(unsigned char c) {
	USART_Transmit(c);
}

volatile void send_ln() {
	USART_Transmit('\n');
	USART_Transmit('\r');
}

void send_str(unsigned char * msg) {
	while (*msg) {
		USART_Transmit(*msg);
		msg++;
	}
}

void send_hex(unsigned int val, unsigned int digits) {
	char i;
	unsigned char tmp;
	
	for ( ; digits > 0; digits--) {
		tmp = val;
		for (i = 1; i < digits; i++) {
			tmp = tmp>>4;
		}
		tmp &= 0x0f;
		if (tmp > 9) {
			tmp += ('A'-10);
		} else {
			tmp += '0';
		}
		send_char(tmp);
	}
}

unsigned int hexToInt(unsigned char * buf, unsigned int digits) {
	unsigned int tmp = 0;
	
	for ( ; digits > 0 && (*buf != 0); ) {
		if ( '0' <= *buf && *buf <= '9') {
			tmp = tmp<<4;
			tmp += (*buf - '0') & 0x0f;
			digits--;
		} else if ( ('a' <= *buf && *buf <= 'f') 
				   || ('A' <= *buf && *buf <= 'F') ) {
			tmp = tmp<<4;
			tmp += ((*buf & 0x1f) + 9) & 0x0f; // A or a
			digits--;
		}
		buf++;
	}
	return tmp;
}


unsigned char buf[16];
unsigned char bix = 0;
volatile unsigned int address;

int main(void) {
	int sreg = 0;
	int spl = 0;
	PORTDMode(0b11111110);
	
	USART_Init(BAUD_PRESCALE);
	SetBit(UCSRB, RXCIE);
	sei();
	
	for (;;) {
		asm("lds r2,0x5f;");
		asm("lds r3,0x5d;");
	}
    return 0;               /* never reached */
}


ISR(USART_RX_vect) { 
	asm("lds r4,0x5f;");
	asm("lds r5,0x5d;");

	unsigned char receivedByte;
	receivedByte = UDR; // Fetch the recieved byte value into the variable 
	buf[bix] = receivedByte;
	if (receivedByte == '.') {
		buf[bix] = 0;
		bix = 0;
	} else {
		bix++;
		bix %= sizeof(buf);
		return;
	}
	//send_str(buf);
	
	address = hexToInt(buf,2);
	send_ln();
	send_str("RAM ");
	send_str("$");
	send_hex(address, 2);
	send_str(": ");
	send_hex(*(unsigned char *) address,2);
	
	send_ln();
	send_str("SREG : ");
	address = 0x5f;
	send_hex(*(unsigned char *) address,2);
	send_ln();
	send_str("SPL : ");
	address = 0x5d;
	send_hex(*(unsigned char *) address,2);
	send_ln();
	for (address = 0x00; address < 0xe0; address++) {
		if ( (address & 0x0f) == 0) {
			send_hex(address,2);
			send_str(": ");
		}
		send_hex(*(unsigned char*)address,2);
		switch (address & 0x0f) {
			case 0x0f:
				send_ln();
				break;
			case 0x07:
				send_str(" ");
			default:
				send_str(" ");
		}
	}
	send_ln();
}

void USART_Init( unsigned int speed )
{
	/* Set baud rate */
	UBRRH = (unsigned char)(speed>>8);
	UBRRL = (unsigned char)speed;
	/* Enable receiver and transmitter */
	UCSRB = (1<<RXEN)|(1<<TXEN)|(0<<UCSZ2);
	/* Set frame format: 8data, 1stop bit */
	UCSRC = (0<<USBS)|(0b11<<UCSZ0);
}

void USART_Transmit( unsigned char data )
{
	/* Wait for empty transmit buffer */
	while ( !( UCSRA & (1<<UDRE)) )
		;
	/* Put data into buffer, sends the data */
	UDR = data;
}

/* 9 bit
void USART_Transmit( unsigned int data )
{
	// Wait for empty transmit buffer 
	while ( !( UCSRA & (1<<UDRE))) )
		;
	// Copy 9th bit to TXB8 
	UCSRB &= ~(1<<TXB8);
	if ( data & 0x0100 )
		UCSRB |= (1<<TXB8);
	// Put data into buffer, sends the data 
	UDR = data;
}
*/

unsigned char USART_Receive( void )
{
	/* Wait for data to be received */
	while ( !(UCSRA & (1<<RXC)) )
		;
	/* Get and return received data from buffer */
	return UDR;
}

/*
 unsigned int USART_Receive( void )
 {
 unsigned char status, resh, resl;
 // Wait for data to be received 
while ( !(UCSRA & (1<<RXC)) )
;
// Get status and 9th bit, then data 
// from buffer 
status = UCSRA;
resh = UCSRB;
resl = UDR;
// If error, return -1 
if ( status & (1<<FE)|(1<<DOR)|(1<<UPE) )
return -1;
// Filter the 9th bit, then return 
resh = (resh >> 1) & 0x01;
return ((resh << 8) | resl);
}
*/

void USART_Flush( void )
{
	unsigned char dummy;
	while ( UCSRA & (1<<RXC) ) dummy = UDR;
}
