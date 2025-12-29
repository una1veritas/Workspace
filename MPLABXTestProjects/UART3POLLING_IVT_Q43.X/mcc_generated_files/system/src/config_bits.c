/**
 * CONFIGURATION BITS Generated Driver Source File
 * 
 * @file config_bits.c
 * 
 * @ingroup config_bitsdriver
 * 
 * @brief This file contains the API Implementation for the Device Configuration Bits driver.
 *
 * @version Driver Version 1.0.1
 *
 * @version Package Version 1.0.4
*/

/*
� [2025] Microchip Technology Inc. and its subsidiaries.

    Subject to your compliance with these terms, you may use Microchip 
    software and any derivatives exclusively with Microchip products. 
    You are responsible for complying with 3rd party license terms  
    applicable to your use of 3rd party software (including open source  
    software) that may accompany Microchip software. SOFTWARE IS ?AS IS.? 
    NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS 
    SOFTWARE, INCLUDING ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT,  
    MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT 
    WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE, 
    INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY 
    KIND WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF 
    MICROCHIP HAS BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE 
    FORESEEABLE. TO THE FULLEST EXTENT ALLOWED BY LAW, MICROCHIP?S 
    TOTAL LIABILITY ON ALL CLAIMS RELATED TO THE SOFTWARE WILL NOT 
    EXCEED AMOUNT OF FEES, IF ANY, YOU PAID DIRECTLY TO MICROCHIP FOR 
    THIS SOFTWARE.
*/

// Configuration bits: selected in the GUI

//CONFIG1
#pragma config FEXTOSC = OFF     // External Oscillator Selection->Oscillator not enabled
#pragma config RSTOSC = HFINTOSC_64MHZ     // Reset Oscillator Selection->HFINTOSC with HFFRQ = 64 MHz and CDIV = 1:1

//CONFIG2
#pragma config CLKOUTEN = OFF     // Clock out Enable bit->CLKOUT function is disabled
#pragma config PR1WAY = ON     // PRLOCKED One-Way Set Enable bit->PRLOCKED bit can be cleared and set only once
#pragma config CSWEN = ON     // Clock Switch Enable bit->Writing to NOSC and NDIV is allowed
#ifdef _18F47Q83
#pragma config JTAGEN = OFF     // JTAG Enable bit->Disable JTAG Boundary Scan mode, JTAG pins revert to user functions
#endif
#pragma config FCMEN = ON     // Fail-Safe Clock Monitor Enable bit->Fail-Safe Clock Monitor enabled
#ifdef _18F47Q83
#pragma config FCMENP = ON     // Fail-Safe Clock Monitor -Primary XTAL Enable bit->FSCM timer will set FSCMP bit and OSFIF interrupt on Primary XTAL failure
#pragma config FCMENS = ON     // Fail-Safe Clock Monitor -Secondary XTAL Enable bit->FSCM timer will set FSCMS bit and OSFIF interrupt on Secondary XTAL failure
#endif

//CONFIG3
#pragma config MCLRE = EXTMCLR     // MCLR Enable bit->If LVP = 0, MCLR pin is MCLR; If LVP = 1, RE3 pin function is MCLR 
#pragma config PWRTS = PWRT_OFF     // Power-up timer selection bits->PWRT is disabled
#pragma config MVECEN = ON     // Multi-vector enable bit->Multi-vector enabled, Vector table used for interrupts
#pragma config IVT1WAY = ON     // IVTLOCK bit One-way set enable bit->IVTLOCKED bit can be cleared and set only once
#pragma config LPBOREN = OFF     // Low Power BOR Enable bit->Low-Power BOR disabled
#pragma config BOREN = SBORDIS     // Brown-out Reset Enable bits->Brown-out Reset enabled , SBOREN bit is ignored

//CONFIG4
#pragma config BORV = VBOR_1P9     // Brown-out Reset Voltage Selection bits->Brown-out Reset Voltage (VBOR) set to 1.9V
#pragma config ZCD = OFF     // ZCD Disable bit->ZCD module is disabled. ZCD can be enabled by setting the ZCDSEN bit of ZCDCON
#pragma config PPS1WAY = ON     // PPSLOCK bit One-Way Set Enable bit->PPSLOCKED bit can be cleared and set only once; PPS registers remain locked after one clear/set cycle
#pragma config STVREN = ON     // Stack Full/Underflow Reset Enable bit->Stack full/underflow will cause Reset
#pragma config LVP = ON     // Low Voltage Programming Enable bit->Low voltage programming enabled. MCLR/VPP pin function is MCLR. MCLRE configuration bit is ignored
#pragma config XINST = OFF     // Extended Instruction Set Enable bit->Extended Instruction Set and Indexed Addressing Mode disabled

//CONFIG5
#pragma config WDTCPS = WDTCPS_31     // WDT Period selection bits->Divider ratio 1:65536; software control of WDTPS
#pragma config WDTE = OFF     // WDT operating mode->WDT Disabled; SWDTEN is ignored

//CONFIG6
#pragma config WDTCWS = WDTCWS_7     // WDT Window Select bits->window always open (100%); software control; keyed access not required
#pragma config WDTCCS = SC     // WDT input clock selector->Software Control

//CONFIG7
#pragma config BBSIZE = BBSIZE_512     // Boot Block Size selection bits->Boot Block size is 512 words
#pragma config BBEN = OFF     // Boot Block enable bit->Boot block disabled
#pragma config SAFEN = OFF     // Storage Area Flash enable bit->SAF disabled

//CONFIG8
#pragma config WRTB = OFF     // Boot Block Write Protection bit->Boot Block not Write protected
#pragma config WRTC = OFF     // Configuration Register Write Protection bit->Configuration registers not Write protected
#pragma config WRTD = OFF     // Data EEPROM Write Protection bit->Data EEPROM not Write protected
#pragma config WRTSAF = OFF     // SAF Write protection bit->SAF not Write Protected
#pragma config WRTAPP = OFF     // Application Block write protection bit->Application Block not write protected

//CONFIG9
#ifdef _18F47Q83
#pragma config BOOTPINSEL = RC5     // CRC on boot output pin selection->CRC on boot output pin is RC5
#pragma config BPEN = OFF     // CRC on boot output pin enable bit->CRC on boot output pin disabled
#pragma config ODCON = OFF     // CRC on boot output pin open drain bit->Pin drives both high-going and low-going signals
#endif

//CONFIG10
#pragma config CP = OFF     // PFM and Data EEPROM Code Protection bit->PFM and Data EEPROM code protection disabled

//CONFIG11
#ifdef _18F47Q83
#pragma config BOOTSCEN = OFF     // CRC on boot scan enable for boot area->CRC on boot will not include the boot area of program memory in its calculation
#pragma config BOOTCOE = HALT     // CRC on boot Continue on Error for boot areas bit->CRC on boot will stop device if error is detected in boot areas
#pragma config APPSCEN = OFF     // CRC on boot application code scan enable->CRC on boot will not include the application area of program memory in its calculation
#pragma config SAFSCEN = OFF     // CRC on boot SAF area scan enable->CRC on boot will not include the SAF area of program memory in its calculation
#pragma config DATASCEN = OFF     // CRC on boot Data EEPROM scan enable->CRC on boot will not include data EEPROM in its calculation
#pragma config CFGSCEN = OFF     // CRC on boot Config fuses scan enable->CRC on boot will not include the configuration fuses in its calculation
#pragma config COE = HALT     // CRC on boot Continue on Error for non-boot areas bit->CRC on boot will stop device if error is detected in non-boot areas
#pragma config BOOTPOR = OFF     // Boot on CRC Enable bit->CRC on boot will not run
#endif

//CONFIG12
#ifdef _18F47Q83
#pragma config BCRCPOLT = hFF     // Boot Sector Polynomial for CRC on boot bits 31-24->Bits 31:24 of BCRCPOL are 0xFF
#endif

//CONFIG13
#ifdef _18F47Q83
#pragma config BCRCPOLU = hFF     // Boot Sector Polynomial for CRC on boot bits 23-16->Bits 23:16 of BCRCPOL are 0xFF
#endif

//CONFIG14
#ifdef _18F47Q83
#pragma config BCRCPOLH = hFF     // Boot Sector Polynomial for CRC on boot bits 15-8->Bits 15:8 of BCRCPOL are 0xFF
#endif

//CONFIG15
#ifdef _18F47Q83
#pragma config BCRCPOLL = hFF     // Boot Sector Polynomial for CRC on boot bits 7-0->Bits 7:0 of BCRCPOL are 0xFF
#endif

//CONFIG16
#ifdef _18F47Q83
#pragma config BCRCSEEDT = hFF     // Boot Sector Seed for CRC on boot bits 31-24->Bits 31:24 of BCRCSEED are 0xFF
#endif

//CONFIG17
#ifdef _18F47Q83
#pragma config BCRCSEEDU = hFF     // Boot Sector Seed for CRC on boot bits 23-16->Bits 23:16 of BCRCSEED are 0xFF
#endif

//CONFIG18
#ifdef _18F47Q83
#pragma config BCRCSEEDH = hFF     // Boot Sector Seed for CRC on boot bits 15-8->Bits 15:8 of BCRCSEED are 0xFF
#endif

//CONFIG19
#ifdef _18F47Q83
#pragma config BCRCSEEDL = hFF     // Boot Sector Seed for CRC on boot bits 7-0->Bits 7:0 of BCRCSEED are 0xFF
#endif

//CONFIG20
#ifdef _18F47Q83
#pragma config BCRCEREST = hFF     // Boot Sector Expected Result for CRC on boot bits 31-24->Bits 31:24 of BCRCERES are 0xFF
#endif

//CONFIG21
#ifdef _18F47Q83
#pragma config BCRCERESU = hFF     // Boot Sector Expected Result for CRC on boot bits 23-16->Bits 23:16 of BCRCERES are 0xFF
#endif

//CONFIG22
#ifdef _18F47Q83
#pragma config BCRCERESH = hFF     // Boot Sector Expected Result for CRC on boot bits 15-8->Bits 15:8 of BCRCERES are 0xFF
#endif

//CONFIG23
#ifdef _18F47Q83
#pragma config BCRCERESL = hFF     // Boot Sector Expected Result for CRC on boot bits 7-0->Bits 7:0 of BCRCERES are 0xFF
#endif

//CONFIG24
#ifdef _18F47Q83
#pragma config CRCPOLT = hFF     // Non-Boot Sector Polynomial for CRC on boot bits 31-24->Bits 31:24 of CRCPOL are 0xFF
#endif

//CONFIG25
#ifdef _18F47Q83
#pragma config CRCPOLU = hFF     // Non-Boot Sector Polynomial for CRC on boot bits 23-16->Bits 23:16 of CRCPOL are 0xFF
#endif

//CONFIG26
#ifdef _18F47Q83
#pragma config CRCPOLH = hFF     // Non-Boot Sector Polynomial for CRC on boot bits 15-8->Bits 15:8 of CRCPOL are 0xFF
#endif

//CONFIG27
#ifdef _18F47Q83
#pragma config CRCPOLL = hFF     // Non-Boot Sector Polynomial for CRC on boot bits 7-0->Bits 7:0 of CRCPOL are 0xFF
#endif

//CONFIG28
#ifdef _18F47Q83
#pragma config CRCSEEDT = hFF     // Non-Boot Sector Seed for CRC on boot bits 31-24->Bits 31:24 of CRCSEED are 0xFF
#endif

//CONFIG29
#ifdef _18F47Q83
#pragma config CRCSEEDU = hFF     // Non-Boot Sector Seed for CRC on boot bits 23-16->Bits 23:16 of CRCSEED are 0xFF
#endif

//CONFIG30
#ifdef _18F47Q83
#pragma config CRCSEEDH = hFF     // Non-Boot Sector Seed for CRC on boot bits 15-8->Bits 15:8 of CRCSEED are 0xFF
#endif

//CONFIG31
#ifdef _18F47Q83
#pragma config CRCSEEDL = hFF     // Non-Boot Sector Seed for CRC on boot bits 7-0->Bits 7:0 of CRCSEED are 0xFF
#endif

//CONFIG32
#ifdef _18F47Q83
#pragma config CRCEREST = hFF     // Non-Boot Sector Expected Result for CRC on boot bits 31-24->Bits 31:24 of CRCERES are 0xFF
#endif

//CONFIG33
#ifdef _18F47Q83
#pragma config CRCERESU = hFF     // Non-Boot Sector Expected Result for CRC on boot bits 23-16->Bits 23:16 of CRCERES are 0xFF
#endif

//CONFIG34
#ifdef _18F47Q83
#pragma config CRCERESH = hFF     // Non-Boot Sector Expected Result for CRC on boot bits 15-8->Bits 15:8 of CRCERES are 0xFF
#endif

//CONFIG35
#ifdef _18F47Q83
#pragma config CRCERESL = hFF     // Non-Boot Sector Expected Result for CRC on boot bits 7-0->Bits 7:0 of CRCERES are 0xFF
#endif
/**
 End of File
*/