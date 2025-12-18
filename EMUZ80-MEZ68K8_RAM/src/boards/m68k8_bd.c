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

#include "../../src/mez68k8.h"
#include <stdio.h>
#include "../../drivers/SDCard.h"
#include "../../drivers/picregister.h"

#include "../../drivers/SPI.h"

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

//#include "m68k8_cmn.c"
#include "m68k8_cmn.h"

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

#include "../../drivers/pic18f57q43_spi.c"
#include "../../drivers/SDCard.c"

