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
 *  Target: MEZ68K8_RAM
 *  Date. 2024.5.14
*/

#define BOARD_DEPENDENT_SOURCE

#include "../mez68k8.h"

// console input buffers
#define U3B_SIZE 128
unsigned char rx_buf[U3B_SIZE];	//UART Rx ring buffer
unsigned int rx_wp, rx_rp, rx_cnt;

//TIMER0 seconds counter
static union {
    unsigned int w; //16 bits Address
    struct {
        unsigned char l; //Address low
        unsigned char h; //Address high
    };
} adjCnt;

TPB tim_pb;			// TIME device parameter block

//initialize TIMER0 & TIM device parameter block
void timer0_init(void) {
	adjCnt.w = TIMER0_INITC;	// set initial adjust timer counter
	tim_pb.TIM_DAYS = TIM20240101;
	tim_pb.TIM_MINS = 0;
	tim_pb.TIM_HRS = 0;
	tim_pb.TIM_SECS = 0;
	tim_pb.TIM_HSEC = 0;
}

//
// define interrupt
//
// Never called, logically
void __interrupt(irq(default),base(8)) Default_ISR(){}

////////////// UART3 Receive interrupt ////////////////////////////
// UART3 Rx interrupt
// PIR9 (bit0:U3RXIF bit1:U3TXIF)
/////////////////////////////////////////////////////////////////
void __interrupt(irq(U3RX),base(8)) URT3Rx_ISR(){

	unsigned char rx_data;

	rx_data = U3RXB;			// get rx data

	if (rx_cnt < U3B_SIZE) {
		rx_buf[rx_wp] = rx_data;
		rx_wp = (rx_wp + 1) & (U3B_SIZE - 1);
		rx_cnt++;
	}
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
	GIE = 0;                // Disable interrupt
	c = rx_buf[rx_rp];
	rx_rp = (rx_rp + 1) & ( U3B_SIZE - 1);
	rx_cnt--;
	GIE = 1;                // enable interrupt
    return c;               // Read data
}

void devio_init(void) {
	rx_wp = 0;
	rx_rp = 0;
	rx_cnt = 0;
    U3RXIE = 1;          // Receiver interrupt enable
}

static void m68k8_common_sys_init()
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
	LAT(M68K_RESET) = 0;        // Reset
    TRIS(M68K_RESET) = 0;       // Set as output

	// R/W
	WPU(M68K_RW) = 1;		// week pull up
	LAT(M68K_RW) = 1;		// Active state READ-High
	TRIS(M68K_RW) = 0;		// Set as output

	// LE
	WPU(M68K_LE) = 0;		// disable week pull up
	LAT(M68K_LE) = 0;		// Disactive state Low
    TRIS(M68K_LE) = 0;		// Set as output

    // UART3 initialize
    U3BRG = 416;			// 9600bps @ 64MHz
    U3RXEN = 1;				// Receiver enable
    U3TXEN = 1;				// Transmitter enable

    // UART3 Receiver
    TRISA7 = 1;				// RX set as input
    U3RXPPS = 0x07;			// RA7->UART3:RXD;

    // UART3 Transmitter
    LATA6 = 1;				// Default level
    TRISA6 = 0;				// TX set as output
    RA6PPS = 0x26;			// UART3:TXD -> RA6;

    U3ON = 1;				// Serial port enable

	// Init address LATCH to 0
	// Address bus A7-A0 pin
    WPU(M68K_ADBUS) = 0xff;       // Week pull up
    LAT(M68K_ADBUS) = 0x00;
    TRIS(M68K_ADBUS) = 0x00;      // Set as output

	// Address bus A15-A8 pin
    WPU(M68K_ADR_H) = 0xff;       // Week pull up
    LAT(M68K_ADR_H) = 0x00;
    TRIS(M68K_ADR_H) = 0x00;      // Set as output

	WPU(M68K_A16) = 1;     // A16 Week pull up
	LAT(M68K_A16) = 0;     // init A16=0
    TRIS(M68K_A16) = 0;    // Set as output

	WPU(M68K_A17) = 1;     // A17 Week pull up
	LAT(M68K_A17) = 0;     // init A17=0
    TRIS(M68K_A17) = 0;    // Set as output

	WPU(M68K_A18) = 1;     // A18 Week pull up
	LAT(M68K_A18) = 0;     // init A18=0
    TRIS(M68K_A18) = 0;    // Set as output

	// A19 : PIC BR trigger
	WPU(M68K_A19) = 0;     // Disable pull up
	LAT(M68K_A19) = 0;     // init A19=0
    TRIS(M68K_A19) = 0;    // Set as output(SRAM #CE enable)
}

static void mez88_common_start_M68K(void)
{
    // AD bus A7-A0 pin
    TRIS(M68K_ADBUS) = 0xff;	// Set as input
    TRIS(M68K_ADR_H) = 0xff;	// Set as input
    TRIS(M68K_A16) = 1;			// Set as input
    TRIS(M68K_A17) = 1;			// Set as input
    TRIS(M68K_A18) = 1;			// Set as input
    TRIS(M68K_A19) = 1;			// Set as input

    TRIS(M68K_RW) = 1;           // Set as input
    TRIS(M68K_AS) = 1;           // Set as input
	TRIS(M68K_DS) = 1;			// Set as input
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

	LAT(M68K_LTOE) = 0;			// Set 74LS373 as active
	LAT(M68K_DS) = 0;			// activate SRAM #CE
	ab.w = addr;
	i = 0;

	while( i < len ) {

	    LAT(M68K_ADBUS) = ab.ll;
		LAT(M68K_ADR_H) = ab.lh;
		LAT(M68K_A16) = ab.hl & 0x01;
		LAT(M68K_A17) = (ab.hl & 0x02) >> 1;
		LAT(M68K_A18) = (ab.hl & 0x04) >> 2;

		// Latch address A0 - A7 & A16-A13 to 74LS373
		LAT(M68K_LE) = 1;
		LAT(M68K_LE) = 0;

        LAT(M68K_RW) = 0;					// activate /WE
        LAT(M68K_ADBUS) = ((uint8_t*)buf)[i];
        LAT(M68K_RW) = 1;					// deactivate /WE

		i++;
		ab.w++;
    }
	LAT(M68K_DS) = 1;			// disactivate SRAM #CE
	LAT(M68K_LTOE) = 1;			// Set 74LS373 as Hiz
}

void read_sram(uint32_t addr, uint8_t *buf, unsigned int len)
{
    union address_bus_u ab;
    unsigned int i;

	ab.w = addr;

	LAT(M68K_LTOE) = 0;			// Set 74LS373 as active
	LAT(M68K_DS) = 0;			// activate SRAM #CE
	i = 0;
	while( i < len ) {
	
		TRIS(M68K_ADBUS) = 0x00;					// Set as output

		LAT(M68K_ADBUS) = ab.ll;
		LAT(M68K_ADR_H) = ab.lh;
		LAT(M68K_A16) = ab.hl & 0x01;
		LAT(M68K_A17) = (ab.hl & 0x02) >> 1;
		LAT(M68K_A18) = (ab.hl & 0x04) >> 2;

		// Latch address A0 - A7 & A16-A13 to 74LS373
		LAT(M68K_LE) = 1;
		LAT(M68K_LE) = 0;

		TRIS(M68K_ADBUS) = 0xff;				// Set as input
		ab.w++;									// Ensure bus data setup time from HiZ to valid data
		((uint8_t*)buf)[i] = PORT(M68K_ADBUS);	// read data
		i++;
    }
	LAT(M68K_DS) = 1;							// disactivate SRAM #CE
	TRIS(M68K_ADBUS) = 0x00;					// Set as output
	LAT(M68K_LTOE) = 1;							// Set 74LS373 as Hiz
}

static void emu88_common_wait_for_programmer()
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
