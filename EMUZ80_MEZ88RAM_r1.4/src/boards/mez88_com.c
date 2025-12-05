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
 *  Date. 2026.8.10
*/

#define BOARD_DEPENDENT_SOURCE

#include "../mez88.h"

//#define clk5m
//#define clk8m
#define clk9_10m

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
    printf("\n\r");
    printf("wait for programmer ...\r");
    __delay_ms(200);
    printf("                       \r");

    printf("\n\r");
}
