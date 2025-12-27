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

#include "../picconfig.h"
#include <stdio.h>

#include "../mez68k8.h"
#include "../drivers/SDCard.h"
#include "../drivers/picregister.h"

#include "../drivers/SPI.h"
#define SPI_PREFIX      SPI_SD
#define SPI_HW_INST     SPI1

#define M68K_ADBUS		B
#define M68K_ADR_H		D

#define M68K_A16		A0
#define M68K_A17		A1
#define M68K_A18		A2
#define M68K_A19		A3

#define M68K_DS			C0
#define M68K_AS			C1
#define M68K_LE			E2

#define M68K_LTOE		C3
#define M68K_SRAM_WE	C4

#define M68K_DTACK		C5

#define M68K_RESET		E0
#define M68K_BR			E1		// BUS request
#define M68K_BG			C2		// BUS GRANT

// RD/#WR
#define M68K_RW			A4

// CLK
#define M68K_CLK		A5

// SPI
#define MISO			C6
#define MOSI			B0
#define SPI_CK			B1
#define SPI_SS			C7

//SD IO
#define SPI_SD_POCI		MISO
#define SPI_SD_PICO		MOSI
#define SPI_SD_CLK		SPI_CK
#define SPI_SD_SS       SPI_SS

#define CMD_REQ CLC3OUT

#define _XTAL_FREQ 64000000UL
#define UART3_BAUD 57600UL

#ifdef C_SCR_INCLUDE
#include "m68k8_cmn.c"
#else
//#define BOARD_DEPENDENT_SOURCE

//#include "../mez68k8.h"

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
    U3BRG = (_XTAL_FREQ/(16UL * UART3_BAUD)) - 1; //416;			// 9600bps @ 64MHz
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

#endif

static void reset_ioreq(void);
void bus_master_operation(void);
void sys_init()
{
    m68k8_common_sys_init();

	// #LAT_CE
	WPU(M68K_LTOE) = 0;		// disable pull up
	LAT(M68K_LTOE) = 1;		// 74LS373 as Hiz
	TRIS(M68K_LTOE) = 0;	// Set as onput

	// #SRAM_WE
	WPU(M68K_SRAM_WE) = 0;	// disable pull up
	LAT(M68K_SRAM_WE) = 1;	// SRAM WE disactive
	TRIS(M68K_SRAM_WE) = 0;	// Set as onput

	// #AS
	WPU(M68K_AS) = 1;		// Week pull up
	LAT(M68K_AS) = 1;
	TRIS(M68K_AS) = 0;		// Set as onput

	// #DS
	WPU(M68K_DS) = 1;		// week pull up
	LAT(M68K_DS) = 1;
	TRIS(M68K_DS) = 0;    // Set as output

	// #DTACK output pin
	WPU(M68K_DTACK) = 0;		// disable week pull up
    LAT(M68K_DTACK) = 1;
    TRIS(M68K_DTACK) = 0;		// Set as output
	
	// #BR output pin
	WPU(M68K_BR) = 0;		// disable week pull up
    LAT(M68K_BR) = 1;
    TRIS(M68K_BR) = 0;		// Set as output

	// #BG input pin
	WPU(M68K_BG) = 1;			// week pull up
    LAT(M68K_BG) = 1;
    TRIS(M68K_BG) = 1;			// Set as input

	// SPI_SS
	WPU(SPI_SS) = 1;		// SPI_SS Week pull up
	LAT(SPI_SS) = 1;		// set SPI disable
	TRIS(SPI_SS) = 0;		// Set as onput

	WPU(M68K_CLK) = 0;		// disable week pull up
	LAT(M68K_CLK) = 1;		// init CLK = 1
    TRIS(M68K_CLK) = 0;		// set as output pin
	
// Setup CLC
//
	//========== CLC pin assign ===========
    CLCIN0PPS = 0x03;			// assign RA3(A19)
    CLCIN1PPS = 0x04;			// assign RA4(R/#W)
    CLCIN4PPS = 0x12;			// assign RC2(#BG)
    CLCIN5PPS = 0x11;			// assign RC1(#AS)

	//========== CLC1 : Make #SRAM_WE ==========

	CLCSELECT = 0;		// CLC1 select

	CLCnSEL0 = 0;		// CLCIN0PPS : RA3(A19)
	CLCnSEL1 = 1;		// CLCIN1PPS : RA4(R/#W)
	CLCnSEL2 = 127;		// NC
	CLCnSEL3 = 127;		// NC
	
    CLCnGLS0 = 0x01;	// invert RA3(A19) -> lcxg1
	CLCnGLS1 = 0x04;	// invert RA4(R/#W) -> lcxg2
    CLCnGLS2 = 0x00;	// 0 -> lcxg3
    CLCnGLS3 = 0x00;	// 0 -> lcxg4
	
    CLCnPOL = 0x8C;		// POL=1(out : invert X), G4POL=1, G3POL=1
    CLCnCON = 0x82;		// 4-input AND gate
	
	RC4PPS = 0x01;		// CLC1OUT(#SRAM_WE) -> RC4(output)

	//========== CLC2 : Make #DTACK ==========
	//input	#AS
	//		NCO1

	CLCSELECT = 1;		// CLC2 select

//*
	CLCnSEL0 = 0x2a;	// NCO1
    CLCnSEL1 = 5;		// CLCIN5PPS : RC1(#AS)
	CLCnSEL2 = 127;		// NC
	CLCnSEL3 = 127;		// NC

    CLCnGLS0 = 0x02;	// NCO1 -> lcxg1(DFF CK)
	CLCnGLS1 = 0x04;	// invert RC1(#AS) -> lcxg2(DFF D)
    CLCnGLS2 = 0x00;	// 0 -> lcxg3(DFF R)
    CLCnGLS3 = 0x00;	// 0 -> lcxg4(DFF S)
	
    CLCnPOL = 0x80;		// POL=1 invert(DFF Q)
    CLCnCON = 0x84;		// 1-input DFF with Set and Rese, no interrupt occurs
	
	RC5PPS = 0x02;		// CLC2OUT(#DTACK) -> RC5(output)

	reset_ioreq();		// reset DFF

//*/
/*
	CLCnSEL0 = 5;		// CLCIN5PPS : RC1(#AS)
    CLCnSEL1 = 127;		// NC
	CLCnSEL2 = 127;		// NC
	CLCnSEL3 = 127;		// NC

    CLCnGLS0 = 0x02;	// RC1(#AS)  -> lcxg1
	CLCnGLS1 = 0x00;	// 0 -> lcxg2
    CLCnGLS2 = 0x00;	// 0 -> lcxg3
    CLCnGLS3 = 0x00;	// 0 -> lcxg4
	
    CLCnPOL = 0x0e;		// POL=0, G4POL=G3POL=G2POL=1, G1POL=0
    CLCnCON = 0x82;		// 4-input AND

	RC5PPS = 0x02;		// CLC2OUT(#DTACK) -> RC5(output)
*/
	//========== CLC3 : Make #BR trigger(CLC3OUT) ==========

	CLCSELECT = 2;		// CLC3 select

    CLCnSEL0 = 5;		// CLCIN5PPS : RC1(#AS)
	CLCnSEL1 = 0;		// CLCIN0PPS : RA3(A19)
	CLCnSEL2 = 0x35;	// CLC3OUT
	CLCnSEL3 = 127;		// NC

    CLCnGLS0 = 0x01;	// invert RC1(#AS) -> lcxg1(DFF clock)
	CLCnGLS1 = 0x08;	// RA3(A19) -> lcxg2(DFF OR input 2)
    CLCnGLS2 = 0x00;	// 0 -> lcxg3(DFF R)
    CLCnGLS3 = 0x20;	// CLC3OUT -> lcxg4(DFF OR input 1)

    CLCnPOL = 0x00;		// POL=0
    CLCnCON = 0x85;		// 2-Input DFF with R , no interrupt occurs

	reset_ioreq();		// reset DFF

	// SPI data and clock pins slew at maximum rate

	SLRCON(SPI_SD_PICO) = 0;
	SLRCON(SPI_SD_CLK) = 0;
	SLRCON(SPI_SD_POCI) = 0;

#define CLK_68k8_6M 196608  // 6MHz
#define CLK_68k8_10M 327680 // 10MHz

	// 68008 clock(RA5) by NCO FDC mode

//	NCO1INC = CLK_68k8_6M;
	NCO1INC = CLK_68k8_10M;
	NCO1CLK = 0x00;		// Clock source Fosc
	NCO1PFM = 0;		// FDC mode
	NCO1OUT = 1;		// NCO output enable
	NCO1EN = 1;			// NCO enable

	RA5PPS = 0x3f;		// RA5 assign NCO1

    emu88_common_wait_for_programmer();
}

void setup_sd(void) {
    //
    // Initialize SD Card
    //
    int result;
    static int retry;
    for (retry = 0; 1; retry++) {
        if (20 <= retry) {
            printf("No SD Card? Halt.\r\n");
            while(1);
        }
//        if (SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_2MHZ, /* timeout */ 100) == SDCARD_SUCCESS)
        if ((result = SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_4MHZ, /* timeout */ 100)) == SDCARD_SUCCESS) {
//        if (SDCard_init(SPI_CLOCK_100KHZ, SPI_CLOCK_8MHZ, /* timeout */ 100) == SDCARD_SUCCESS)
            break;
        } else {
            printf("SDCrd_init error code: %d\r\n", result);
        }
        __delay_ms(200);
    }
}

void start_M68K(void)
{

    mez88_common_start_M68K();

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

	// reset CLC2
	CLCSELECT = 1;		// CLC2 select
	reset_ioreq();

	// reset CLC3
	CLCSELECT = 2;		// CLC3 select
	reset_ioreq();			// reset CLC3 (CMD_REQ : CLC3OUT = 0)

	// M68K start
    LAT(M68K_RESET) = 1;		// Release reset
	TRIS(M68K_RESET) = 1;		// Set as input
}

static void reset_ioreq(void)
{
	// Release wait (D-FF reset)
	G3POL = 1;
	G3POL = 0;
}

static void set_BR_pin(void)
{
	LAT(M68K_BR) = 0;			// request BR
	while( R(M68K_BG) ) {}		// wait until bus release
}

static void reset_BR_pin(void)
{
	LAT(M68K_BR) = 1;
	while( !R(M68K_BG) ) {}		// wait until bus release
}

static void bus_hold_req(void) {
	// Set address bus as output
	TRIS(M68K_ADBUS) = 0x00;	// A7-A0
	TRIS(M68K_ADR_H) = 0x00;	// A8-A15
	TRIS(M68K_A16) = 0;			// Set as output
	TRIS(M68K_A17) = 0;			// Set as output
	TRIS(M68K_A18) = 0;			// Set as output
	TRIS(M68K_A19) = 0;			// Set as output

	TRIS(M68K_RW) = 0;			// output
	TRIS(M68K_AS) = 0;			// output
	TRIS(M68K_DS) = 0;			// output

	LAT(M68K_RW) = 1;			// SRAM READ mode
	LAT(M68K_DS) = 1;			// memory #OE disactive
	LAT(M68K_AS) = 1;			// AS disactive
	LAT(M68K_LE) = 0;			// Latch LE disactive
	LAT(M68K_A19) = 0;
}

static void bus_release_req(void) {
	// Set address bus as input
	TRIS(M68K_ADBUS) = 0xff;	// A7-A0
	TRIS(M68K_ADR_H) = 0xff;	// A8-A15
	TRIS(M68K_A16) = 1;			// Set as input
	TRIS(M68K_A17) = 1;			// Set as input
	TRIS(M68K_A18) = 1;			// Set as input
	TRIS(M68K_A19) = 1;			// Set as input

	TRIS(M68K_AS) = 1;			// input
	TRIS(M68K_RW) = 1;			// input
	TRIS(M68K_DS) = 1;			// input
}

//--------------------------------
// event loop ( PIC MAIN LOOP )
//--------------------------------
void board_event_loop(void) {

	while(1) {
		if (CMD_REQ) {					// CLC3OUT =1
			set_BR_pin();				// HOLD = 1, wait until HOLDA = 1
		    bus_hold_req();				// PIC becomes a busmaster
			bus_master_operation();
			bus_release_req();
			reset_ioreq();				// reset CLC3 (CMD_REQ : CLC3OUT = 0)
			reset_BR_pin();				// HOLD = 0, wait until HOLDA = 0
		}
	}
}

#ifdef C_SRC_INCLUDE
#include "../drivers/pic18f57q43_spi.c"
#else

//#include "../drivers/SPI.h"
//#include "../picconfig.h"

struct SPI_HW {
    struct SPI spi;
    uint8_t bus_acquired;
    uint8_t tris;
};
static struct SPI_HW pic18f47q43_spi_ctx = { 0 };
struct SPI *SPI(ctx) = (struct SPI *)&pic18f47q43_spi_ctx;

void SPI(select)(struct SPI *ctx_, int select);

static void acquire_bus(struct SPI *ctx_)
{
    struct SPI_HW *ctx = (struct SPI_HW *)ctx_;
    if (ctx->bus_acquired == 0) {
        PPS(SPI(CLK)) = PPS_OUT(SPIx(SCK));
                                    // Set as CLK output
        PPS(SPI(PICO)) = PPS_OUT(SPIx(SDO));
                                    // Set as data output
        ctx->tris = TRIS(M68K_ADBUS); // save direction settings
        TRIS(SPI(PICO)) = 0;        // set PICO as output
        TRIS(SPI(CLK)) = 0;         // set clock as output
        TRIS(SPI(POCI)) = 1;        // set POCI as input
    }
    ctx->bus_acquired++;
}

static void release_bus(struct SPI *ctx_)
{
    struct SPI_HW *ctx = (struct SPI_HW *)ctx_;
    if (--ctx->bus_acquired <= 0) {
        PPS(SPI(CLK)) = 0x00;       // Release CLK output
        PPS(SPI(PICO)) = 0x00;      // Release data output
        TRIS(M68K_ADBUS) = ctx->tris; // restore direction settings
    }
}

void SPI(begin)(struct SPI *ctx_)
{
    struct SPI_HW *ctx = (struct SPI_HW *)ctx_;
    ctx->bus_acquired = 0;

    SPIx(CON0) = 0;
    SPIx(CON1) = 0;
    SPIx(SCKPPS) = PPS_IN(SPI(CLK));    // Assign CLK input pin (?)
    SPIx(SDIPPS) = PPS_IN(SPI(POCI));   // Assign data input pin
    TRIS(SPI(SS)) = 0;                  // Set as output
    SPIx(CON0bits).EN = 1;              // Enable SPI
}

void SPI(configure)(struct SPI *ctx_, int clock_speed, uint8_t bit_order, uint8_t data_mode)
{
    struct SPI_HW *ctx = (struct SPI_HW *)ctx_;

    SPIx(CON0bits).MST = 1;     // Host mode
    SPIx(CON0bits).BMODE = 1;   // Byte transfer mode
    SPIx(TWIDTH) = 0;           // 8 bit
    SPIx(INTE) = 0;             // Interrupts are not used
    SPIx(CON1bits).FST = 0;     // Delay to first SCK will be at least 1‚Å?2 baud period
    SPIx(CON2bits).TXR = 1;     // Full duplex mode (TXR and RXR are both enabled)
    SPIx(CON2bits).RXR = 1;

    if (bit_order == SPI_MSBFIRST)
        SPIx(CON0bits).LSBF = 0;
    else
        SPIx(CON0bits).LSBF = 1;

    if (data_mode == SPI_MODE0) {
        SPIx(CON1bits).SMP = 0; // SDI input is sampled in the middle of data output time
        SPIx(CON1bits).CKE = 1; // Output data changes on transition from Active to Idle clock state
        SPIx(CON1bits).CKP = 0; // Idle state for SCK is low level
    } else {
        printf("%s: ERROR: mode %d is not supported\n\r", __func__, data_mode);
        while (1);
    }

    SPIx(CLK) = 0;      // FOSC (System Clock)
    switch (clock_speed) {
    case SPI_CLOCK_100KHZ:
        SPIx(CLK) = 2;      // MFINTOSC (500 kHz)
        SPIx(BAUD) = 2;     // 500 kHz / (2 * ( 2 + 1)) = 83 kHz
        break;
    case SPI_CLOCK_2MHZ:
        SPIx(BAUD) = 15;    // 64 MHz / (2 * (15 + 1)) = 2.0 MHz
        break;
    case SPI_CLOCK_4MHZ:
        SPIx(BAUD) = 7;     // 64 MHz / (2 * ( 7 + 1)) = 4.0 MHz
        break;
    case SPI_CLOCK_5MHZ:
        SPIx(BAUD) = 5;     // 64 MHz / (2 * ( 5 + 1)) = 5.3 MHz
        break;
    case SPI_CLOCK_6MHZ:
        SPIx(BAUD) = 4;     // 64 MHz / (2 * ( 4 + 1)) = 6.4 MHz
        break;
    case SPI_CLOCK_8MHZ:
        SPIx(BAUD) = 3;     // 64 MHz / (2 * ( 3 + 1)) = 8.0 MHz
        break;
    case SPI_CLOCK_10MHZ:
        SPIx(BAUD) = 2;     // 64 MHz / (2 * ( 2 + 1)) = 10.7 MHz
        break;
    default:
        printf("%s: ERROR: clock speed %d is not supported\n\r", __func__, clock_speed);
        break;
    }
}

void SPI(begin_transaction)(struct SPI *ctx_)
{
    acquire_bus(ctx_);
    SPI(select)(ctx_, 1);  // select the chip and start transaction
}

void SPI(end_transaction)(struct SPI *ctx_)
{
    SPI(select)(ctx_, 0);  // de-select the chip and end transaction
    release_bus(ctx_);
}

uint8_t SPI(transfer_byte)(struct SPI *ctx_, uint8_t output)
{
    SPIx(TCNTH) = 0;
    SPIx(TCNTL) = 1;
    SPIx(TXB) = output;
    while(!SPIx(RXIF));
    return SPIx(RXB);
}

void SPI(transfer)(struct SPI *ctx_, void *buf, unsigned int count)
{
    uint8_t *p = (uint8_t*)buf;
    for (int i = 0; i < count; i++) {
        *p = SPI(transfer_byte)(ctx_, *p);
        p++;
    }
}

void SPI(send)(struct SPI *ctx_, const void *buf, unsigned int count)
{
    uint8_t *p = (uint8_t*)buf;
    uint8_t dummy;

    if (count == 0)
        return;

    SPIx(TCNTH) = (count >> 8);
    SPIx(TCNTL) = (count & 0xff);

    SPIx(TXB) = *p++;
    for (int i = 1; i < count; i++) {
        SPIx(TXB) = *p++;
        while(!SPIx(RXIF));
        dummy = SPIx(RXB);
    }
    while(!SPIx(RXIF));
    dummy = SPIx(RXB);
}

void SPI(receive)(struct SPI *ctx_, void *buf, unsigned int count)
{
    uint8_t *p = (uint8_t*)buf;

    if (count == 0)
        return;

    SPIx(TCNTH) = (count >> 8);
    SPIx(TCNTL) = (count & 0xff);

    if ((count & 0x07) || 255 < count / 8) {
        SPIx(TXB) = 0xff;
        for (int i = 1; i < count; i++) {
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
        }
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
    } else {
        SPIx(TXB) = 0xff;
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        SPIx(TXB) = 0xff;
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
        uint8_t repeat = (uint8_t)(count / 8);
        for (uint8_t i = 1; i < repeat; i++) {
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
            SPIx(TXB) = 0xff;
            while(!SPIx(RXIF));
            *p++ = SPIx(RXB);
        }
        while(!SPIx(RXIF));
        *p++ = SPIx(RXB);
    }
}

void SPI(dummy_clocks)(struct SPI *ctx_, unsigned int clocks)
{
    uint8_t dummy = 0xff;
    acquire_bus(ctx_);
    for (int i = 0; i < clocks; i++) {
        SPI(send)(ctx_, &dummy, 1);
    }
    release_bus(ctx_);
}

uint8_t SPI(receive_byte)(struct SPI *ctx_)
{
    uint8_t dummy = 0xff;
    SPI(receive)(ctx_, &dummy, 1);
    return dummy;
}

void SPI(select)(struct SPI *ctx_, int select)
{
    LAT(SPI(SS)) = select ? 0 : 1;
}

#endif

#ifdef C_SRC_INCLUDE
#include "../drivers/SDCard.c"
#else

#include "../drivers/utils.h"

#define DEBUG
#if defined(DEBUG)
static int debug_flags = 0;
#else
static const int debug_flags = 0;
#endif

#define dprintf(args) do { if (debug_flags) printf args; } while(0)
#define drprintf(args) do { if (debug_flags & SDCARD_DEBUG_READ) printf args; } while(0)
#define dwprintf(args) do { if (debug_flags & SDCARD_DEBUG_WRITE) printf args; } while(0)

static struct SDCard {
    struct SPI *spi;
    uint16_t timeout;
    unsigned int calc_read_crc :1;
} ctx_ = { 0 };

void SDCard_end_transaction()
{
    struct SPI *spi = ctx_.spi;
    SPI(end_transaction)(spi);
    SPI(dummy_clocks)(spi, 1);
}

int SDCard_init(int initial_clock_speed, int clock_speed, uint16_t timeout)
{
    ctx_.spi = SPI(ctx);
    ctx_.timeout = timeout;
    ctx_.calc_read_crc = 0;
    struct SPI *spi = ctx_.spi;
    SPI(begin)(spi);

    uint8_t buf[5];
    dprintf(("\r\nSD Card: initialize ...\r\n"));

    SPI(configure)(spi, initial_clock_speed, SPI_MSBFIRST, SPI_MODE0);
    SPI(begin_transaction)(spi);
    SPI(dummy_clocks)(spi, 10);
    SDCard_end_transaction();

    // CMD0 go idle state
    SDCard_command(0, 0, buf, 1);
    dprintf(("SD Card: CMD0, R1=%02x\r\n", buf[0]));
    if (buf[0] != SDCARD_R1_IDLE_STATE) {
        dprintf(("SD Card: timeout\r\n"));
        return SDCARD_TIMEOUT;
    }

    // CMD8 send interface condition
    SDCard_command(8, 0x000001aa, buf, 5);
    dprintf(("SD Card: CMD8, R7=%02x %02x %02x %02x %02x\r\n",
             buf[0], buf[1], buf[2], buf[3], buf[4]));
    if (buf[0] != SDCARD_R1_IDLE_STATE || (buf[3] & 0x01) != 0x01 || buf[4] != 0xaa) {
        dprintf(("SD Card: not supoprted\r\n"));
        return SDCARD_NOTSUPPORTED;
    }

    // ACMD41 send operating condition
    for (int i = 0; i < 3000; i++) {
        SDCard_command(55, 0, buf, 1);
        SDCard_command(41, 1UL << 30 /* HCS bit (Host Capacity Support) is 1 */, buf, 5);
        if (buf[0] == 0x00)
            break;
    }
    dprintf(("SD Card: ACMD41, R1=%02x\n\r", buf[0]));
    if (buf[0] != 0x00) {
        dprintf(("SD Card: ACMD41 response is %02x\n\r", buf[0]));
        return SDCARD_TIMEOUT;
    }

    // CMD58 read OCR register
    SDCard_command(58, 0, buf, 5);
    dprintf(("SD Card: CMD58, R3=%02x %02x %02x %02x %02x\n\r",
             buf[0], buf[1], buf[2], buf[3], buf[4]));
    if (buf[0] & 0xfe) {
        dprintf(("SD Card: unexpected response %02x\n\r", buf[0]));
        return SDCARD_BADRESPONSE;
    }
    if (!(buf[1] & 0x40)) {
        dprintf(("SD Card: CCS (Card Capacity Status) is 0\n\r"));
        return SDCARD_NOTSUPPORTED;
    }
    dprintf(("SD Card: SDHC or SDXC card detected\n\r"));

    if (!(buf[1] & 0x80)) {
        dprintf(("SD Card: Card power up status bis is 0\n\r"));
        return SDCARD_BADRESPONSE;
    }
    dprintf(("SD Card: ready.\n\r"));

    // CMD59 turn on CRC
    SDCard_command(59, 1, buf, 1);
    if (buf[0] != 0x00) {
        dprintf(("SD Card: CMD59 response is %02x\n\r", buf[0]));
        return SDCARD_BADRESPONSE;
    }

    SPI(configure)(spi, clock_speed, SPI_MSBFIRST, SPI_MODE0);

    dprintf(("SD Card: initialize ... succeeded\n\r"));

    return SDCARD_SUCCESS;
}

static uint8_t __SDCard_wait_response(uint8_t no_response, unsigned int attempts)
{
    struct SPI *spi = ctx_.spi;
    uint8_t response;
    do {
        response = SPI(receive_byte)(spi);
    } while ((response == no_response) && 0 < --attempts);
    return response;
}

static int __SDCard_command_r1(uint8_t command, uint32_t argument, uint8_t *r1)
{
    struct SPI *spi = ctx_.spi;
    uint8_t buf[6];
    uint8_t response;

    buf[0] = command | 0x40;
    buf[1] = (argument >> 24) & 0xff;
    buf[2] = (argument >> 16) & 0xff;
    buf[3] = (argument >>  8) & 0xff;
    buf[4] = (argument >>  0) & 0xff;
    buf[5] = SDCard_crc(buf, 5) | 0x01;

    SPI(begin_transaction)(spi);
    SPI(dummy_clocks)(spi, 1);
    SPI(send)(spi, buf, 6);

    response = __SDCard_wait_response(0xff, ctx_.timeout);
    *r1 = response;
    if (response == 0xff) {
        return SDCARD_TIMEOUT;
    }

    return SDCARD_SUCCESS;
}

int SDCard_read512(uint32_t addr, unsigned int offs, void *buf, unsigned int count)
{
    struct SPI *spi = ctx_.spi;
    int result;
    uint8_t response;
    uint16_t crc, resp_crc;
    int retry = 5;

    drprintf(("SD Card:  read512: addr=%8ld, offs=%d, count=%d\n\r", addr, offs, count));

 retry:
    result = __SDCard_command_r1(17, addr, &response);
    if (result != SDCARD_SUCCESS) {
        goto done;
    }
    if (response != 0) {
        result = SDCARD_BADRESPONSE;
        goto done;
    }

    response = __SDCard_wait_response(0xff, 3000);
    if (response == 0xff) {
        result = SDCARD_TIMEOUT;
        goto done;
    }
    if (response != 0xfe) {
        result = SDCARD_BADRESPONSE;
        goto done;
    }

    crc = 0;
    for (int i = 0; i < offs; i++) {
        response = SPI(receive_byte)(spi);
        if (ctx_.calc_read_crc)
            crc = __SDCard_crc16(crc, &response, 1);
    }
    SPI(receive)(spi, buf, count);
    if (ctx_.calc_read_crc)
        crc = __SDCard_crc16(crc, buf, count);
    for (unsigned int i = 0; i < 512 - offs - count; i++) {
        response = SPI(receive_byte)(spi);
        if (ctx_.calc_read_crc)
            crc = __SDCard_crc16(crc, &response, 1);
    }
    if ((debug_flags & SDCARD_DEBUG_READ) && (debug_flags & SDCARD_DEBUG_VERBOSE)) {
        util_addrdump("SD: ", (addr * 512) + offs, buf, count);
    }

    resp_crc = (uint16_t)SPI(receive_byte)(spi) << 8;
    resp_crc |= SPI(receive_byte)(spi);
    if (ctx_.calc_read_crc && resp_crc != crc) {
        dprintf(("SD Card: read512: CRC error (%04x != %04x, retry=%d)\n\r",
                 crc, resp_crc, retry));
        if (--retry < 1) {
            result = SDCARD_CRC_ERROR;
            goto done;
        }
        SDCard_end_transaction();
        goto retry;
    }

    result = SDCARD_SUCCESS;

 done:
    SDCard_end_transaction();
    return result;
}

int SDCard_write512(uint32_t addr, unsigned int offs, const void *buf, unsigned int count)
{
    struct SPI *spi = ctx_.spi;
    int result;
    uint8_t response;
    uint16_t crc;
    int retry = 5;

    dwprintf(("SD Card: write512: addr=%8ld, offs=%d, count=%d\n\r", addr, offs, count));
    if ((debug_flags & SDCARD_DEBUG_WRITE) && (debug_flags & SDCARD_DEBUG_VERBOSE)) {
        util_addrdump("SD: ", (addr * 512) + offs, buf, count);
    }

    crc = 0;
    response = 0xff;
    for (int i = 0; i < offs; i++) {
        crc = __SDCard_crc16(crc, &response, 1);
    }
    crc = __SDCard_crc16(crc, buf, count);
    for (unsigned int i = 0; i < 512 - offs - count; i++) {
        crc = __SDCard_crc16(crc, &response, 1);
    }

 retry:
    result = __SDCard_command_r1(24, addr, &response);
    if (result != SDCARD_SUCCESS) {
        goto done;
    }
    if (response != 0) {
        result = SDCARD_BADRESPONSE;
        goto done;
    }

    response = 0xfe;
    SPI(send)(spi, &response, 1);
    SPI(dummy_clocks)(spi, offs);
    SPI(send)(spi, buf, count);
    SPI(dummy_clocks)(spi, 512 - offs - count);
    response = (crc >> 8) & 0xff;
    SPI(send)(spi, &response, 1);
    response = crc & 0xff;
    SPI(send)(spi, &response, 1);

    response = __SDCard_wait_response(0xff, 3000);
    if (response == 0xff) {
        dprintf(("SD Card: write512: failed to get token, timeout\n\r"));
        result = SDCARD_TIMEOUT;
        goto done;
    }
    if ((response & 0x1f) != 0x05) {
        dprintf(("SD Card: write512: token is %02x\n\r", response));
        if ((response & 0x1f) == 0x0b) {
            dprintf(("SD Card: write512: CRC error (retry=%d)\n\r", retry));
            if (--retry < 1) {
                result = SDCARD_CRC_ERROR;
                goto done;
            }
            __SDCard_wait_response(0xff, 30000);
            SDCard_end_transaction();
            goto retry;
        }
        result = SDCARD_BADRESPONSE;
        goto done;
    }

    response = __SDCard_wait_response(0x00, 30000);
    if (response == 0x00) {
        dprintf(("SD Card: write512: timeout, response is %02x\n\r", response));
        result = SDCARD_TIMEOUT;
        goto done;
    }

    result = SDCARD_SUCCESS;

 done:
    SDCard_end_transaction();
    return result;
}

int SDCard_command(uint8_t command, uint32_t argument, void *response_buffer, unsigned int length)
{
    struct SPI *spi = ctx_.spi;
    int result;
    uint8_t *responsep = (uint8_t*)response_buffer;

    result = __SDCard_command_r1(command, argument, responsep);
    if (result != SDCARD_SUCCESS) {
        SDCard_end_transaction();
        return result;
    }

    SPI(receive)(spi, &responsep[1], length - 1);
    SDCard_end_transaction();

    return SDCARD_SUCCESS;
}

uint8_t SDCard_crc(const void *buf, unsigned int count)
{
    uint8_t crc = 0;
    uint8_t *p = (uint8_t*)buf;
    uint8_t *endp = p + count;

    while (p < endp) {
        crc ^= *p++;
        for (int i = 0; i < 8; i++) {
            if (crc & 0x80)
                crc ^= 0x89;
            crc <<= 1;
        }
    }

    return crc;
}

uint16_t __SDCard_crc16(uint16_t crc, const void *buf, unsigned int count)
{
    uint8_t *p = (uint8_t*)buf;
    uint8_t *endp = p + count;

    while (p < endp) {
        crc = (crc >> 8)|(crc << 8);
        crc ^= *p++;
        crc ^= ((crc & 0xff) >> 4);
        crc ^= (crc << 12);
        crc ^= ((crc & 0xff) << 5);
    }

    return crc;
}

uint16_t SDCard_crc16(const void *buf, unsigned int count)
{
    return __SDCard_crc16(0, buf, count);
}

int SDCard_debug(int newval)
{
    int res = debug_flags;
#if defined(DEBUG)
    debug_flags = newval;
#endif
    return res;
}

#endif

