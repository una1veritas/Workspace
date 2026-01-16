
#ifndef CONFIG_BITS_H
#define	CONFIG_BITS_H

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

// Configuration bits: selected in the GUI

//CONFIG1
#pragma config FEXTOSC = OFF     // External Oscillator Selection->Oscillator not enabled
#pragma config RSTOSC = HFINTOSC_64MHZ     // Reset Oscillator Selection->HFINTOSC with HFFRQ = 64 MHz and CDIV = 1:1

//CONFIG2
#pragma config CLKOUTEN = OFF     // Clock out Enable bit->CLKOUT function is disabled
#pragma config FCMEN = ON     // Fail-Safe Clock Monitor Enable bit->Fail-Safe Clock Monitor enabled
#pragma config CSWEN = ON     // Clock Switch Enable bit->Writing to NOSC and NDIV is allowed
#pragma config PR1WAY = ON     // PRLOCKED One-Way Set Enable bit->PRLOCKED bit can be cleared and set only once

//CONFIG3
#pragma config MVECEN = ON     // Multi-vector enable bit->Multi-vector enabled, Vector table used for interrupts
#pragma config MCLRE = EXTMCLR     // MCLR Enable bit->If LVP = 0, MCLR pin is MCLR; If LVP = 1, RE3 pin function is MCLR 
#pragma config BOREN = SBORDIS     // Brown-out Reset Enable bits->Brown-out Reset enabled , SBOREN bit is ignored
#pragma config PWRTS = PWRT_OFF     // Power-up timer selection bits->PWRT is disabled
#pragma config IVT1WAY = ON     // IVTLOCK bit One-way set enable bit->IVTLOCKED bit can be cleared and set only once
#pragma config LPBOREN = OFF     // Low Power BOR Enable bit->Low-Power BOR disabled

//CONFIG4
#pragma config XINST = OFF     // Extended Instruction Set Enable bit->Extended Instruction Set and Indexed Addressing Mode disabled
#pragma config LVP = ON     // Low Voltage Programming Enable bit->Low voltage programming enabled. MCLR/VPP pin function is MCLR. MCLRE configuration bit is ignored
#pragma config ZCD = OFF     // ZCD Disable bit->ZCD module is disabled. ZCD can be enabled by setting the ZCDSEN bit of ZCDCON
#pragma config STVREN = ON     // Stack Full/Underflow Reset Enable bit->Stack full/underflow will cause Reset
#pragma config BORV = VBOR_1P9     // Brown-out Reset Voltage Selection bits->Brown-out Reset Voltage (VBOR) set to 1.9V
#pragma config PPS1WAY = ON     // PPSLOCK bit One-Way Set Enable bit->PPSLOCKED bit can be cleared and set only once; PPS registers remain locked after one clear/set cycle

//CONFIG5
#pragma config WDTCPS = WDTCPS_31     // WDT Period selection bits->Divider ratio 1:65536; software control of WDTPS
#pragma config WDTE = OFF     // WDT operating mode->WDT Disabled; SWDTEN is ignored

//CONFIG6
#pragma config WDTCWS = WDTCWS_7     // WDT Window Select bits->window always open (100%); software control; keyed access not required
#pragma config WDTCCS = SC     // WDT input clock selector->Software Control

//CONFIG7
#pragma config SAFEN = OFF     // Storage Area Flash enable bit->SAF disabled
#pragma config BBEN = OFF     // Boot Block enable bit->Boot block disabled
#pragma config BBSIZE = BBSIZE_512     // Boot Block Size selection bits->Boot Block size is 512 words

//CONFIG8
#pragma config WRTB = OFF     // Boot Block Write Protection bit->Boot Block not Write protected
#pragma config WRTC = OFF     // Configuration Register Write Protection bit->Configuration registers not Write protected
#pragma config WRTD = OFF     // Data EEPROM Write Protection bit->Data EEPROM not Write protected
#pragma config WRTAPP = OFF     // Application Block write protection bit->Application Block not write protected
#pragma config WRTSAF = OFF     // SAF Write protection bit->SAF not Write Protected

//CONFIG10
#pragma config CP = OFF     // PFM and Data EEPROM Code Protection bit->PFM and Data EEPROM code protection disabled

#endif	/* CONFIG_BITS_H */
/**
 End of File
*/
