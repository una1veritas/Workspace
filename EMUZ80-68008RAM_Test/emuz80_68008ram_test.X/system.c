/**
 * System Driver Source File
 * 
 * @file system.c
 * 
 * @ingroup systemdriver
 * 
 * @brief This file contains the API implementation for the System driver.
 *
 * @version Driver Version 2.0.3
 *
 * @version Package Version 5.3.5
*/

#include <xc.h>

#include "system.h"
#include "uart3.h"

#include "pic18common.h"

void SYSTEM_Initialize(void)
{
    CLOCK_Initialize();
    PIN_MANAGER_Initialize();
    NCO1_Initialize();
    UART3_Initialize();
    INTERRUPT_Initialize();
}

void CLOCK_Initialize(void)
{
    // Set the CLOCK CONTROL module to the options selected in the user interface.
    OSCCON1 = (0 << _OSCCON1_NDIV_POSN)   // NDIV 1
        | (6 << _OSCCON1_NOSC_POSN);  // NOSC HFINTOSC
    OSCCON3 = (0 << _OSCCON3_SOSCPWR_POSN)   // SOSCPWR Low power
        | (0 << _OSCCON3_CSWHOLD_POSN);  // CSWHOLD may proceed
    OSCEN = (0 << _OSCEN_EXTOEN_POSN)   // EXTOEN disabled
        | (0 << _OSCEN_HFOEN_POSN)   // HFOEN disabled
        | (0 << _OSCEN_MFOEN_POSN)   // MFOEN disabled
        | (0 << _OSCEN_LFOEN_POSN)   // LFOEN disabled
        | (0 << _OSCEN_SOSCEN_POSN)   // SOSCEN disabled
        | (0 << _OSCEN_ADOEN_POSN)   // ADOEN disabled
        | (0 << _OSCEN_PLLEN_POSN);  // PLLEN disabled
    OSCFRQ = (8 << _OSCFRQ_HFFRQ_POSN);  // HFFRQ 64_MHz
    OSCTUNE = (0 << _OSCTUNE_TUN_POSN);  // TUN 0x0
    ACTCON = (0 << _ACTCON_ACTEN_POSN)   // ACTEN disabled
        | (0 << _ACTCON_ACTUD_POSN);  // ACTUD enabled

}

void NCO1_Initialize(void) {
    // NCO1 setup as clock source to PH2IN on RA3 
    // CPU clock (RA5) by NCO FDC mode

	RA5PPS = 0x3f;	// RA15 --> NCO1
    ANSELAbits.ANSELA5 = 0; // Disable analog function
	//TRISA5 = 0;		// NCO output pin
    TRISAbits.TRISA5 = 0;
	NCO1INC = (unsigned int)(CLK_68008_FREQ / 30.5175781);
	NCO1CLK = 0x00; // (0<<5 | 0x1 ) NCO output is active for 1 input clock periods, Clock source HFINTOSC
	NCO1PFM = 0;	// FDC mode (fixed to 50% duty)
	NCO1OUT = 1;	// NCO output enable
	NCO1EN = 1;		// NCO enable
    // NCO1 setup finished

}

void PIN_MANAGER_Initialize(void)
{
   /**
    Clear LATx registers
    */
    LATA = 0x0;
    LATB = 0x0;
    LATC = 0x0;
    LATD = 0x0;
    LATE = 0x0;
    /**
    ODx registers open drain off
    */
    ODCONA = 0x0;
    ODCONB = 0x0;
    ODCONC = 0x0;
    ODCOND = 0x0;
    ODCONE = 0x0;

    /**
    TRISx registers set all input except RA6 (TX3)
    */
    TRISA = 0xBF;
    TRISB = 0xFF;
    TRISC = 0xFF;
    TRISD = 0xFF;
    TRISE = 0xF;

    /**
    ANSELx registers off (digital only)
    */
    ANSELA = 0x00;
    ANSELB = 0x00;
    ANSELC = 0x00;
    ANSELD = 0x00;
    ANSELE = 0x00;

    /**
    WPUx registers all off
    */
    WPUA = 0x0;
    WPUB = 0x0;
    WPUC = 0x0;
    WPUD = 0x0;
    WPUE = 0x0;


    /**
    SLRCONx registers maximum slew rate
    */
    SLRCONA = 0xFF;
    SLRCONB = 0xFF;
    SLRCONC = 0xFF;
    SLRCOND = 0xFF;
    SLRCONE = 0x7;

    /**
    INLVLx registers ST mode (INLVL=1, default) mode use rising threshold around 1.3V
    */
    INLVLA = 0;//0xFF;
    INLVLB = 0;//0xFF;
    INLVLC = 0;//0xFF;
    INLVLD = 0;//0xFF;
    INLVLE = 0;//0xF;

   /**
    RxyI2C | RxyFEAT registers   
    */
    RB1I2C = 0x0;
    RB2I2C = 0x0;
    RC3I2C = 0x0;
    RC4I2C = 0x0;
    
    /**
    PPS registers
    */
    // UART3
    ANSELA7 = 0;
    U3RXPPS = 0x7; //RA7->UART3:RX3;
    RA6PPS = 0x26;  //RA6->UART3:TX3;

   /**
    IOCx registers interrupt on pin change 
    */
    IOCAP = 0x0;
    IOCAN = 0x0;
    IOCAF = 0x0;
    IOCBP = 0x0;
    IOCBN = 0x0;
    IOCBF = 0x0;
    IOCCP = 0x0;
    IOCCN = 0x0;
    IOCCF = 0x0;
    IOCEP = 0x0;
    IOCEN = 0x0;
    IOCEF = 0x0;

}

/*
void PIN_MANAGER_IOC(void)
{
}
 * */