/*
 *  This source is for PIC18F47Q43 UART, I2C, SPI and TIMER0
 *
 * Base source code is maked by @hanyazou
 *  https://twitter.com/hanyazou
 *
 * Redesigned by Akihito Honda(Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 *  Target: MEZ88_RAM
 *  Date. 2025.8.10
*/

#define BOARD_DEPENDENT_SOURCE

#include <stdio.h>

#include "../mez88.h"
#include "../../drivers/SDCard.h"
#include "../../drivers/picregister.h"

#define SPI_PREFIX      SPI_SD
#define SPI_HW_INST     SPI1

#include "../../drivers/SPI.h"

#define I88_ADBUS		B
#define I88_ADR_H		D

#define I88_A16		A0
#define I88_A17		A1
#define I88_A18		A2

#define I88_IOM		C0
#define I88_ALE		C5

#define I88_RESET	E0
#define I88_HOLD	E1		// BUS request
#define I88_HOLDA	A5

// /WR
#define I88_WR		A3
// /RD
#define I88_RD		A4

// CLK
#define I88_CLK		C7
#define I88_NMI		C6

#define SPI_SS		E2
#define SPI_SD_POCI	B2

#define SPI_SD_PICO     B0
#define SPI_SD_CLK      B1
#define SPI_SD_SS       SPI_SS

#define URT5_RXD		C2
#define URT5_TXD		C1

#define CMD_REQ CLC3OUT

//#define INCLUDE_MEZ88_COM_C
#ifdef INCLUDE_MEZ88_COM_C
#include "mez88_com.c"
#else

#define BOARD_DEPENDENT_SOURCE

#define clk5m
//#define clk8m
//#define clk9_10m

// console input buffers
#define U3B_SIZE 128

unsigned char rx_buf[U3B_SIZE];	//UART Rx ring buffer
unsigned int rx_wp, rx_rp, rx_cnt;

// AUX: input buffers
#define U5B_SIZE 128
unsigned char ax_buf[U5B_SIZE];	//UART Rx ring buffer
unsigned int ax_wp, ax_rp, ax_cnt;

TPB tim_pb;			// TIME device parameter block

// RTC DS1307 format
// rtc[0] : seconds (BCD) 00-59
// rtc[1] : minuts  (BCD) 00-59
// rtc[2] : hours   (BCD) 00-23 (or 1-12 when bit6=1. bit5: AM(0)/PM(1) )
// rtc[3] : day     (BCD) week day 01-07
// rtc[4] : date    (BCD) 01-31
// rtc[5] : month   (BCD) 01-12
// rtc[6] : year    (BCD) 00-99 : range : (19)80-(19)99, (20)00-(20)79
uint8_t rtc[7];

static uint8_t cnt_sec;		// sec timer (1000ms = 10ms * 100)

void enable_int(void) {
	GIE = 1;             // Global interrupt enable
	TMR0IF = 0;			// Clear timer0 interrupt flag
	TMR0IE = 1;			// Enable timer0 interrupt
}

void uart5_init(void) {
	ax_wp = 0;
	ax_rp = 0;
	ax_cnt = 0;

	U5RXIE = 1;			// Receiver interrupt enable

}
//initialize TIMER0 & TIM device parameter block
void timer0_init(void) {
	
	uint16_t year, month, date;

	cnt_sec = 0;	// set initial adjust timer counter
	tim_pb.TIM_DAYS = TIM20250601;		//set 2025.06.01
	tim_pb.TIM_MINS = 0;
	tim_pb.TIM_HRS = 0;
	tim_pb.TIM_SECS = 0;
	tim_pb.TIM_HSEC = 0;

	// convert number of days to year, month and date
	cnv_ymd(tim_pb.TIM_DAYS, &year, &month, &date );
	// convert bin to BCD
	rtc[0] = cnv_bcd(tim_pb.TIM_SECS);
	rtc[1] = cnv_bcd(tim_pb.TIM_MINS);
	rtc[2] = cnv_bcd(tim_pb.TIM_HRS);
	rtc[4] = cnv_bcd((uint8_t)date);
	rtc[5] = cnv_bcd((uint8_t)month);
	rtc[6] = cnv_bcd((uint8_t)year);
}

void cvt_bcd_bin(void) {
	uint16_t year, month, date;
	// convert BCD to bin
	rtc[0] &=0x7f;		// mask bit 7(CH: clock disable bit)

	TMR0IE = 0;			// disable timer0 interrupt
	tim_pb.TIM_SECS = cnv_byte(rtc[0]);
	tim_pb.TIM_MINS = cnv_byte(rtc[1]);
	tim_pb.TIM_HRS  = cnv_byte(rtc[2]);
	date  = (uint16_t)cnv_byte(rtc[4]);
	month = (uint16_t)cnv_byte(rtc[5]);
	year  = (uint16_t)cnv_byte(rtc[6]);
	if (year >= 80) year += 1900;
	else year += 2000;

	// convert year, month and date to number of days from 1980
	tim_pb.TIM_DAYS = days_from_1980(year, month, date);
	TMR0IE = 1;			// Enable timer0 interrupt
}

int cnv_rtc_tim(void) {
	if ( read_I2C(DS1307, 0, 7, &rtc[0]) == 0xFF) return 1;
	cvt_bcd_bin();
	return 0;
}

void datcnv_tim_rtc(void) {
	uint16_t year, month, date;
	
	cnv_ymd(tim_pb.TIM_DAYS, &year, &month, &date );
	// convert bin to BCD
	rtc[0] = cnv_bcd(tim_pb.TIM_SECS);
	rtc[1] = cnv_bcd(tim_pb.TIM_MINS);
	rtc[2] = cnv_bcd(tim_pb.TIM_HRS);
	rtc[4] = cnv_bcd((uint8_t)date);
	rtc[5] = cnv_bcd((uint8_t)month);
	rtc[6] = cnv_bcd((uint8_t)year);
}

static void base_pin_definition()
{
    // System initialize
    OSCFRQ = 0x08;      // 64MHz internal OSC

	// Disable analog function
    ANSELA = 0x00;
    ANSELB = 0x00;
    ANSELC = 0x00;
    ANSELD = 0x00;
    ANSELE0 = 0;
    ANSELE1 = 0;
    ANSELE2 = 0;

    // RESET output pin
	LAT(I88_RESET) = 1;        // Reset
    TRIS(I88_RESET) = 0;       // Set as output

	// HOLDA
	WPU(I88_HOLDA) = 0;     // disable week pull up
	LAT(I88_HOLDA) = 1;     // set HOLDA=0 for PIC controls ALE(RC5) dualing RESET period
	TRIS(I88_HOLDA) = 0;    // Set as output during RESET period

	// ALE
	WPU(I88_ALE) = 0;     // disable week pull up
	LAT(I88_ALE) = 0;
    TRIS(I88_ALE) = 0;    // Set as output for PIC R/W RAM

	// Init address LATCH to 0
	// Address bus A7-A0 pin
    WPU(I88_ADBUS) = 0xff;       // Week pull up
    LAT(I88_ADBUS) = 0x00;
    TRIS(I88_ADBUS) = 0x00;      // Set as output

	// Address bus A15-A8 pin
    WPU(I88_ADR_H) = 0xff;       // Week pull up
    LAT(I88_ADR_H) = 0x00;
    TRIS(I88_ADR_H) = 0x00;      // Set as output

	WPU(I88_A16) = 1;     // A16 Week pull up
	LAT(I88_A16) = 1;     // init A16=0
    TRIS(I88_A16) = 0;    // Set as output

	WPU(I88_A17) = 1;     // A17 Week pull up
	LAT(I88_A17) = 1;     // init A17=0
    TRIS(I88_A17) = 0;    // Set as output

	WPU(I88_A18) = 1;     // A17 Week pull up
	LAT(I88_A18) = 1;     // init A17=0
    TRIS(I88_A18) = 0;    // Set as output

	// NMI definition
	WPU(I88_NMI) = 0;     // disable week pull up
	PPS(I88_NMI) = 0;     // set as latch port
	LAT(I88_NMI) = 0;     // NMI=0
	TRIS(I88_NMI) = 0;    // Set as output

	// HOLD output pin
	WPU(I88_HOLD) = 0;	 // disable week pull up
    LAT(I88_HOLD) = 1;
    TRIS(I88_HOLD) = 0;  // Set as output
	
	// IO/#M
	WPU(I88_IOM) = 1;     // I88_IOM Week pull up
	LAT(I88_IOM) = 0;     // memory /CE active
	TRIS(I88_IOM) = 0;    // Set as onput

	// #WR output pin
	WPU(I88_WR) = 1;		// /WR Week pull up
    LAT(I88_WR) = 1;		// disactive
    TRIS(I88_WR) = 0;		// Set as output

	// #RD output pin
	WPU(I88_RD) = 1;		// /WR Week pull up
    LAT(I88_RD) = 1;		// disactive
    TRIS(I88_RD) = 0;		// Set as output

	// #SPI_SS
	WPU(SPI_SS) = 1;		// SPI_SS Week pull up
	LAT(SPI_SS) = 1;		// set SPI disable
	TRIS(SPI_SS) = 0;		// Set as onput

	WPU(I88_CLK) = 0;		// disable week pull up
	LAT(I88_CLK) = 1;		// 8088_CLK = 1
    TRIS(I88_CLK) = 0;		// set as output pin
}
	
// UART3 Transmit
// if TXIF is busy, return status BUSY(not ZERO)
//
uint8_t out_chr(char c) {
	if (!U3TXIF) return 1;		// retrun BUSY
    U3TXB = c;                  // Write data
	return 0;
}

// UART3 Transmit
void putch(char c) {
    while(!U3TXIF);             // Wait or Tx interrupt flag set
    U3TXB = c;                  // Write data
}

// UART3 Recive
int getch(void) {
	char c;

	while(!rx_cnt);             // Wait for Rx interrupt flag set
	U3RXIE = 0;					// disable Rx interruot
	c = rx_buf[rx_rp];
	rx_rp = (rx_rp + 1) & ( U3B_SIZE - 1);
	rx_cnt--;
	U3RXIE = 1;					// enable Rx interruot
    return c;               // Read data
}

// clear rx buffer and enable rx interrupt
void clr_uart_rx_buf(void) {
	rx_wp = 0;
	rx_rp = 0;
	rx_cnt = 0;
    U3RXIE = 1;          // Receiver interrupt enable
}

unsigned int get_str(char *buf, uint8_t cnt) {
	unsigned int c, i;
	
	U3RXIE = 0;					// disable Rx interruot
	i = ( (unsigned int)cnt > rx_cnt ) ? rx_cnt : (unsigned int)cnt;
	c = i;
	while(i--) {
		*buf++ = rx_buf[rx_rp];
		rx_rp = (rx_rp + 1) & ( U3B_SIZE - 1);
		rx_cnt--;
	}
	U3RXIE = 1;					// enable Rx interruot
	return c;
}

// UART5 Transmit
void putax(char c) {
    while(!U5TXIF);             // Wait or Tx interrupt flag set
    U5TXB = c;                  // Write data
}

// UART5 Recive
int getax(void) {
	char c;

	while(!ax_cnt);			// Wait for Rx interrupt flag set
	U5RXIE = 0;					// disable Rx interruot

	c = ax_buf[ax_rp];
	ax_rp = (ax_rp + 1) & ( U5B_SIZE - 1);
	ax_cnt--;
    U5RXIE = 1;          // Receiver interrupt enable

	return c;               // Read data
}

//
// define interrupt
//
// Never called, logically
void __interrupt(irq(default),base(8)) Default_ISR(){}

////////////// TIMER0 vector interrupt ////////////////////////////
//TIMER0 interrupt
/////////////////////////////////////////////////////////////////
void __interrupt(irq(TMR0),base(8)) TIMER0_ISR(){

	TMR0IF =0; // Clear timer0 interrupt flag

	if (++cnt_sec == 100) {
		cnt_sec = 0;
		
		if( ++tim_pb.TIM_SECS == 60 ) {
			tim_pb.TIM_SECS = 0;
			if ( ++tim_pb.TIM_MINS == 60 ) {
				tim_pb.TIM_MINS = 0;
				if ( ++tim_pb.TIM_HRS == 24 ) {
					tim_pb.TIM_HRS = 0;
					tim_pb.TIM_DAYS++;
				}
			}
		}
		tim_pb.TIM_HSEC = 0;
	}
}

////////////// UART3 Receive interrupt ////////////////////////////
// UART3 Rx interrupt
// PIR9 (bit0:U3RXIF bit1:U3TXIF)
/////////////////////////////////////////////////////////////////
void __interrupt(irq(U3RX),base(8)) URT3Rx_ISR(){

	unsigned char rx_data;

	rx_data = U3RXB;			// get rx data

	if ( ctlq_ev == CTL_Q && rx_data == CTL_Q ) nmi_sig = 1;
	if (rx_cnt < U3B_SIZE) {
		ctlq_ev = (uint8_t)rx_data;
		rx_buf[rx_wp] = rx_data;
		rx_wp = (rx_wp + 1) & (U3B_SIZE - 1);
		rx_cnt++;
	}
}

////////////// UART5 Receive interrupt ////////////////////////////
// UART5 Rx interrupt
// PIR13 (bit0:U5RXIF)
/////////////////////////////////////////////////////////////////
void __interrupt(irq(U5RX),base(8)) URT5Rx_ISR(){

	unsigned char ax_data;

	ax_data = U5RXB;			// get rx data

	if (ax_cnt < U5B_SIZE) {
		ax_buf[ax_wp] = ax_data;
		ax_wp = (ax_wp + 1) & (U5B_SIZE - 1);
		ax_cnt++;
	}
}

uint32_t get_physical_addr(uint16_t ah, uint16_t al)
{
// real 32 bit address
//	return (uint32_t)ah*0x1000 + (uint32_t)al;

// 8088 : segment:offset
	return (uint32_t)ah*0x10 + (uint32_t)al;
}

// Address Bus
//union address_bus_u {
//    uint32_t w;             // 32 bits Address
//    struct {
//        uint8_t ll;        // Address L low
//        uint8_t lh;        // Address L high
//        uint8_t hl;        // Address H low
//        uint8_t hh;        // Address H high
//    };
//};
void write_sram(uint32_t addr, uint8_t *buf, unsigned int len)
{
    union address_bus_u ab;
    unsigned int i;

	ab.w = addr;
	TRIS(I88_ADBUS) = 0x00;					// Set as output
	i = 0;

	while( i < len ) {

	    LAT(I88_ADBUS) = ab.ll;
		LAT(I88_ADR_H) = ab.lh;
		LAT(I88_A16) = ab.hl & 0x01;
		LAT(I88_A17) = (ab.hl & 0x02) >> 1;
		LAT(I88_A18) = (ab.hl & 0x04) >> 2;

		// Latch address A0 - A7 & A16-A13 to 74LS373
		LAT(I88_ALE) = 1;
		LAT(I88_ALE) = 0;

        LAT(I88_WR) = 0;					// activate /WE
        LAT(I88_ADBUS) = ((uint8_t*)buf)[i];
        LAT(I88_WR) = 1;					// deactivate /WE

		i++;
		ab.w++;
    }
	TRIS(I88_ADBUS) = 0xff;					// Set as input
}

void read_sram(uint32_t addr, uint8_t *buf, unsigned int len)
{
    union address_bus_u ab;
    unsigned int i;

	ab.w = addr;

	LAT(I88_RD) = 0;						// activate /OE
	i = 0;
	while( i < len ) {
	
		TRIS(I88_ADBUS) = 0x00;					// Set as output

		LAT(I88_ADBUS) = ab.ll;
		LAT(I88_ADR_H) = ab.lh;
		LAT(I88_A16) = ab.hl & 0x01;
		LAT(I88_A17) = (ab.hl & 0x02) >> 1;
		LAT(I88_A18) = (ab.hl & 0x04) >> 2;

		// Latch address A0 - A7 & A16-A13 to 74LS373
		LAT(I88_ALE) = 1;
		LAT(I88_ALE) = 0;

		TRIS(I88_ADBUS) = 0xff;					// Set as input
		ab.w++;									// Ensure bus data setup time from HiZ to valid data
		((uint8_t*)buf)[i] = PORT(I88_ADBUS);	// read data

		i++;
    }

	LAT(I88_RD) = 1;						// deactivate /OE
}

static void wait_for_programmer()
{
    //
    // Give a chance to use PRC (RB6) and PRD (RB7) to PIC programer.
    //
    printf("\r\n");
    printf("wait for programmer ...\r");
    __delay_ms(200);
    printf("                       \r");

    printf("\r\n");
}

#endif

static void reset_ioreq(void);
void (*bus_master_operation)(void);

/*********** CLOCK TIMING ************************
 I88_CLK TIMING REQUIREMENTS(MUST)
 CLK Low Time  : minimum 118ns
 CLK High Time : minimum 69ns
*************************************************/

/**************** PWM1 ********************
// 8088 TIMING REQUIREMENTS(MUST)
// CLK Low Time  : minimum 118ns
// CLK High Time : minimum 69ns
// CLK duty 33%

// P64 = 1/64MHz = 15.625ns
// P5  = 1/5MHz  = 200ns = P64 * 12.8
// P8  = 1/8MHz  = 125ns = P64 * 8
// P10 = 1/10MHz = 100ns = P64 * 6.4
// P16 = 1/16MHz = 62.5ns= P64 * 4
//
// --- 4.92MHz ---
// Set PWM Left Aligned mode
// PR = 12
// P1 = 5 : P64*5 = 78.125ns
// P2 = 8 : P64*8 = 125ns
// MODE = 0
//     high period time: 78.125ns
//     low period time: 125ns
//     78.125 + 125ns = 203.125ns f = 4.92MHz
//     duty = 38.4%
// --- 8MHz ---
// Set PWM Left Aligned mode
// PR = 7
// P1 = 3 : P64*3 = 46.875ns
// P2 = 5 : P64*5 = 78.125ns
// MODE = 0
//     high period time: 46.875ns
//     low period time: 78.125ns
//     46.875ns + 78.125 = 125ns f = 8MHz
//     duty = 37.5%
// --- 10MHz(9.14MHz) ---	H min 44 L min 53
// Set PWM Left Aligned mode
// PR = 6
// P1 = 3 : P64*3 = 46.875ns
// P2 = 4 : P64*4 = 62.5ns
// MODE = 0
//     high period time: 46.875ns
//     low period time: 62.5ns
//     46.875ns + 62.5 = 109.375 f = 9.14MHz
//     duty = 42.9%
// --- 10MHz(10.67MHz) ---	H min 44 L min 46
// Set PWM Left Aligned mode
// PR = 5
// P1 = 3 : P64*3 = 46.875ns
// P2 = 3 : P64*3 = 46.875ns
// MODE = 0
//     high period time: 46.875ns
//     low period time: 46.875ns
//     46.875 + 46.875 = 93.75 f = 10.67MHz
//     duty = 50%
// --- 12.8MHz ---
// Set PWM Left Aligned mode
// PR = 4
// P1 = 2 : P64*2 = 31.25ns
// P2 = 3 : P64*2 = 46.875ns
// MODE = 0
//     high period time: 31.255ns
//     low period time: 46.875ns
//     31.25 + 46.875 = 78.125 f = 12.8MHz
//     duty = 60%
// --- 16MHz ---	H min 25 L min 25
// Set PWM Left Aligned mode
// PR = 3
// P1 = 2 : P64*2 = 31.25ns
// P2 = 2 : P64*2 = 31.255ns
// MODE = 0
//     high period time: 31.255ns
//     low period time: 31.25ns
//     31.25 + 31.25 = 62.5 f = 16MHz
//     duty = 50%
******************************************/

void reset_clk(uint16_t clk_fs)
{
	PWM1CON = 0x00;		// reset PWM
	PWM1CLK = 0x02;		// Fsoc
	PWM1GIE = 0x00;		// interrupt disable

	switch (clk_fs) {
		case 8:		// 8MHz duty 37.5%
			PWM1PR = 0x0007;	// 8 periods ( 0 - 7 )
			PWM1S1P1 = 0x0003;	// P1 = 3
			PWM1S1P2 = 0x0005;	// P2 = 5
			break;
		case 9:		// 9.14MHz duty 42.9%
			PWM1PR = 0x0006;	// 7 periods ( 0 - 6 )
			PWM1S1P1 = 0x0003;	// P1 = 3
			PWM1S1P2 = 0x0004;	// P2 = 4
			break;
		case 10:	// 10.67MHz duty 50%
			PWM1PR = 0x0005;	// 6 periods ( 0 - 5 )
			PWM1S1P1 = 0x0003;	// P1 = 3
			PWM1S1P2 = 0x0003;	// P2 = 3
			break;
		case 12:	// 16MHz duty 50%
			PWM1PR = 0x0004;	// 5 periods ( 0 - 4 )
			PWM1S1P1 = 0x0003;	// P1 = 3
			PWM1S1P2 = 0x0002;	// P2 = 2
			break;
		case 16:	// 16MHz duty 50%
			PWM1PR = 0x0003;	// 4 periods ( 0 - 3 )
			PWM1S1P1 = 0x0002;	// P1 = 2
			PWM1S1P2 = 0x0002;	// P2 = 2
			break;
		default:	// 4.92MHz  duty 38.4%
			PWM1PR = 0x000C;	// 13 periods ( 0 - 12 )
			PWM1S1P1 = 0x0005;	// P1 = 5
			PWM1S1P2 = 0x0008;	// P2 = 8
	}

	PWM1S1CFG = 0x00;	// (POL1, POL2)= 0, PPEN = 0 MODE = 0 (Left Aligned mode)
	PWM1CON = 0x84;		// EN=1, LD=1
	RC7PPS = 0x18;		// PWM1S1P1_OUT
}
	
void sys_init()
{
    base_pin_definition();

	// SPI data and clock pins slew at maximum rate

	SLRCON(SPI_SD_PICO) = 0;
	SLRCON(SPI_SD_CLK) = 0;
	SLRCON(SPI_SD_POCI) = 0;

	//
	// UART3 initialize
	//
	U3BRG = 416;        // 9600bps @ 64MHz
    U3RXEN = 1;         // Receiver enable
    U3TXEN = 1;         // Transmitter enable

    // UART3 Receiver
    TRISA7 = 1;         // RX set as input
    U3RXPPS = 0x07;     // RA7->UART3:RXD;

    // UART3 Transmitter
    LATA6 = 1;          // Default level
    TRISA6 = 0;         // TX set as output
    RA6PPS = 0x26;      // UART3:TXD -> RA6;

    U3ON = 1;           // Serial port enable

	//
	// UART5 initialize
	//
	U5BRG = 416;		// 9600bps @ 64MHz
	U5RXEN = 1;			// Receiver enable
    U5TXEN = 1;			// Transmitter enable
	// UART5 Receiver
	TRIS(URT5_RXD) = 1;	// RXD set as input
	U5RXPPS = 0x12;	 	// RC2->UART5:RXD;
    // UART5 Transmitter
    LAT(URT5_TXD) = 1;		// Default level
    TRIS(URT5_TXD) = 0;		// TXD set as output
    RC1PPS = 0x2C;			// UART5:TXD -> RC1;

    U5ON = 1;			// Serial port enable

	// ************ timer0 setup ******************
	T0CON0 = 0x89;	// timer enable, 8bit mode , 1:10 Postscaler 10ms
//	T0CON0 = 0x80;	// timer enable, 8bit mode , 1:1 Postscaler  1ms
//	T0CON0 = 0x84;	// timer enable, 8bit mode , 1:5 Postscaler  5ms
//	T0CON0 = 0x81;	// timer enable, 8bit mode , 1:2 Postscaler  2ms
//	T0CON0 = 0x82;	// timer enable, 8bit mode , 1:3 Postscaler  3ms
//	T0CON0 = 0x83;	// timer enable, 8bit mode , 1:4 Postscaler  4ms
	T0CON1 = 0xa1;	// sorce clk:MFINTOSC (500 kHz), 1:2 Prescaler
	MFOEN = 1;
	TMR0H = 0xff;
	TMR0L = 0x00;

	// Setup CLC
//
	//========== CLC pin assign ===========
    CLCIN4PPS = 0x15;			// assign RC5(ALE)
    CLCIN5PPS = 0x10;			// assign RC0(IO/M#)

	CLCIN6PPS = 0x1f;			// assign RD7(A15)
    CLCIN7PPS = 0x1e;			// assign RD6(A14)
    CLCIN2PPS = 0x1d;			// assign RD5(A13)
    CLCIN3PPS = 0x1c;			// assign RD4(A12)

	//========== CLC1 : address decoder ==========
	// 0xf000

	CLCSELECT = 0;		// CLC1 select

	CLCnSEL0 = 6;		// CLCIN6PPS : AD15
    CLCnSEL1 = 7;		// CLCIN7PPS : AD14
	CLCnSEL2 = 2;		// CLCIN2PPS : AD13
	CLCnSEL3 = 3;		// CLCIN3PPS : AD12

    CLCnGLS0 = 0x02;	// log1 
	CLCnGLS1 = 0x08;	// log2
    CLCnGLS2 = 0x20;	// log3
    CLCnGLS3 = 0x80;	// log4

    CLCnPOL = 0x00;		// POL=0
    CLCnCON = 0x82;		// 4-Input AND , no interrupt occurs

	//========== CLC3 : IOREQ ==========
	// reset DFF with software(reset_ioreq();)
	// Check #IO request & IO address 0xf000
	
	CLCSELECT = 2;		// CLC3 select

	CLCnSEL0 = 4;		// d1s : CLCIN4PPS [RC0(ALE)]
    CLCnSEL1 = 5;		// d2s : CLCIN5PPS [RA2(#IO/M)]
	CLCnSEL2 = 53;		// d3s : CLC3OUT
	CLCnSEL3 = 51;		// d4s : CLC1OUT

    CLCnGLS0 = 0x01;	// Gate1 : not(d1s) -> log1[DFF(CK)]
	CLCnGLS1 = 0x44;	// Gate2 : not(d2s) | not(d4s), POL2=1 - > log2[OR A input]
    CLCnGLS2 = 0x00;	// Gate3 : not gated (log3 = 0)
    CLCnGLS3 = 0x20;	// Gate4 : s3s -> log4[OR B input]

    CLCnPOL = 0x02;		// POL2 = 1
    CLCnCON = 0x85;		// 2-Input DFF with R , no interrupt occurs

	// reset CLC3
	reset_ioreq();

    wait_for_programmer();
}

void setup_sd(void) {
    //
    // Initialize SD Card
    //
    static int retry;
    for (retry = 0; 1; retry++) {
        if (20 <= retry) {
            printf("No SD Card?\r\n");
            while(1);
        }
//        if (SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_2MHZ, /* timeout */ 100) == SDCARD_SUCCESS)
        if (SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_4MHZ, /* timeout */ 100) == SDCARD_SUCCESS)
//        if (SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_8MHZ, /* timeout */ 100) == SDCARD_SUCCESS)
            break;
        __delay_ms(200);
    }
}

void setup_I2C(void) {
	//Source clock enable
	MFOEN = 1;		// MFINTOSC is explicitly enabled
	
	// I2C PIN definition
	LATC4 = 0;			// Set as output
	LATC3 = 0;			// Set as output
	TRISC4 = 0;			// Set as output
	TRISC3 = 0;			// Set as output
	WPUC4 = 1;			// week pull up
	WPUC3 = 1;			// week pull up
	
	RC4PPS = 0x38;			// set RC4PPS for I2C1 SDA
	I2C1SDAPPS = 0x14;		// set RC4 for I2C SDA port

	RC3PPS = 0x37;			// set RC3PPS for I2C SCL
	I2C1SCLPPS = 0x13;		// set RC3 for I2C_SCL port

	//Open-Drain Control
	ODCONC = 0x18;		// RC4 and RC3 are Open-Drain output

	//set I2C Pad Control Register TH=01 (I2C-specific input thresholds)
	RC4I2C = 0x01;		// Std GPIO Slew Rate, Std GPIO weak pull-up, I2C-specific input thresholds
	RC3I2C = 0x01;		// Std GPIO Slew Rate, Std GPIO weak pull-up, I2C-specific input thresholds
//	RC4I2C = 0x41;		// Fast mode (400 kHz), Std GPIO weak pull-up, I2C-specific input thresholds
//	RC3I2C = 0x41;		// Fast mode (400 kHz), Std GPIO weak pull-up, I2C-specific input thresholds
//	RC4I2C = 0xc1;		// Fast mode Plus (1 MHz), Std GPIO weak pull-up, I2C-specific input thresholds
//	RC3I2C = 0xc1;		// Fast mode Plus (1 MHz), Std GPIO weak pull-up, I2C-specific input thresholds

	I2C1_Init();

}

void start_i88(void)
{

    // AD bus A7-A0 pin
    TRIS(I88_ADBUS) = 0xff;	// Set as input
    TRIS(I88_ADR_H) = 0xff;	// Set as input
    TRIS(I88_A16) = 1;			// Set as input
    TRIS(I88_A17) = 1;			// Set as input
    TRIS(I88_A18) = 1;			// Set as input

    TRIS(I88_RD) = 1;           // Set as input
    TRIS(I88_WR) = 1;           // Set as input
	TRIS(I88_IOM) = 1;			// Set as input

	// Unlock IVT
    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x00;

    // Default IVT base address
    IVTBASE = 0x000008;

    // Lock IVT
    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x01;

	reset_ioreq();			// reset CLC3 (CMD_REQ : CLC3OUT = 0)
	LAT(I88_ALE) = 0;    	// set 0 before Hi-z
	TRIS(I88_ALE) = 1;		// ALE is set as input
	LAT(I88_HOLD) = 0;		// Release HOLD
	// I88 start
    LAT(I88_RESET) = 0;		// Release reset

}

static void reset_ioreq(void)
{
	// Release wait (D-FF reset)
	G3POL = 1;
	G3POL = 0;
}

static void set_hold_pin(void)
{
	LAT(I88_HOLD) = 1;
	while( !R(I88_HOLDA) ) {}		// wait until bus release
}

static void reset_hold_pin(void)
{
	LAT(I88_HOLD) = 0;
	while( R(I88_HOLDA) ) {}		// wait until bus release
}

static void nmi_sig_on(void)
{
	LAT(I88_NMI) = 1;			// NMI interrupt occurs
}

static void nmi_sig_off(void)
{
	LAT(I88_NMI) = 0;
}

static void bus_hold_req(void) {
	// Set address bus as output
	TRIS(I88_ADBUS) = 0x00;	// A7-A0
	TRIS(I88_ADR_H) = 0x00;	// A8-A15
	TRIS(I88_A16) = 0;			// Set as output
	TRIS(I88_A17) = 0;			// Set as output
	TRIS(I88_A18) = 0;			// Set as output

	TRIS(I88_RD) = 0;           // output
	TRIS(I88_WR) = 0;           // output
	// SRAM HiZ

	LAT(I88_IOM) = 0;     // memory CE active
	TRIS(I88_IOM) = 0;    // Set as output
	LAT(I88_ALE) = 0;
	TRIS(I88_ALE) = 0;    // Set as output
}

static void bus_release_req(void) {
	// Set address bus as input
	LAT(I88_IOM) = 0;    // memory CE active
	LAT(I88_ALE) = 0;
	TRIS(I88_ADBUS) = 0xff;    // A7-A0
	TRIS(I88_ADR_H) = 0xff;    // A8-A15
	TRIS(I88_A16) = 1;    // Set as input
	TRIS(I88_A17) = 1;    // Set as input
	TRIS(I88_A18) = 1;    // Set as input

	// Set /RD and /WR as input
	TRIS(I88_ALE) = 1;           // input
	TRIS(I88_RD) = 1;           // input
	TRIS(I88_WR) = 1;           // input
	TRIS(I88_IOM) = 1;          // input
}

//--------------------------------
// event loop ( PIC MAIN LOOP )
//--------------------------------
void board_event_loop(void) {

	while(1) {
		if (CMD_REQ) {					// CLC3OUT =1
			set_hold_pin();				// HOLD = 1, wait until HOLDA = 1
		    bus_hold_req();				// PIC becomes a busmaster
			(*bus_master_operation)();
			if ( terminate ) {
				nmi_sig_off();
				LAT(I88_RESET) = 1;        // Reset CPU
				break;
			}
			bus_release_req();
			reset_ioreq();				// reset CLC3 (CMD_REQ : CLC3OUT = 0)
			reset_hold_pin();			// HOLD = 0, wait until HOLDA = 0
		}
		if (nmi_sig) {
			nmi_sig_on();
			nmi_sig = 0;
		}
		nmi_sig_off();
	}
}

#include "../../drivers/pic18f57q43_spi.c"
#include "../../drivers/SDCard.c"

