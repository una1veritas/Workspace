#include "system.h"

void CLC_init(void) {
   	//========== CLC input pin assign ===========
	// 0,1,4,5 = Port A, C
	// 2,3,6,7 = Port B, D
	CLCIN2PPS = 0x1f;	// RD7 = A15
	CLCIN3PPS = 0x1e;	// RD6 = A14
	CLCIN4PPS = 0x04;	// RA4 = R/W
	CLCIN6PPS = 0x1d;	// RD5 = A13 (CLCx Input 7))
	CLCIN7PPS = 0x1c;	// RD4 = A12

	//========== CLC1 /OE ==========
	CLCSELECT = 0;		// CLC1 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IORQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x08;	// R/W noninverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC1 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC2 /WE ==========
	CLCSELECT = 1;		// CLC2 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IOREQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x04;	// R/W inverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC2 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC3 /IORQ :IO area 0xB000 - 0xBFFF ==========
	CLCSELECT = 2;		// CLC3 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 2;		// CLCIN2PPS <- A15
	CLCnSEL1 = 3;		// CLCIN3PPS <- A14
	CLCnSEL2 = 6;		// CLCIN6PPS <- A13
	CLCnSEL3 = 7;		// CLCIN7PPS <- A12
 
	CLCnGLS0 = 0x02;	// A15 noninverted
	CLCnGLS1 = 0x04;	// A14 inverted
	CLCnGLS2 = 0x20;	// A13 noninverted
	CLCnGLS3 = 0x80;	// A12 noninverted

	CLCnPOL = 0x80;		// inverted the CLC3 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC5 RDY ==========
	CLCSELECT = 4;		// CLC5 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// D-FF CLK <- CLC3 (/IORQ)
	CLCnSEL1 = 127;		// D-FF D NC
	CLCnSEL2 = 127;		// D-FF S NC
	CLCnSEL3 = 127;		// D-FF R NC

	CLCnGLS0 = 0x1;		// LCG1D1N
	CLCnGLS1 = 0x0;		// Connect none
	CLCnGLS2 = 0x0;		// Connect none
	CLCnGLS3 = 0x0;		// Connect none

	CLCnPOL = 0x82;		// inverted the CLC5 output, G2 inverted
	CLCnCON = 0x84;		// Select D-FF

	//========== CLC output pin assign ===========
	// 1,2,5,6 = Port A, C
	// 3,4,7,8 = Port B, D
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY

    printf("CLC setup finished.\r\n");
}

