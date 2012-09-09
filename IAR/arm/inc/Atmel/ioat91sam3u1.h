// - ----------------------------------------------------------------------------
// -          ATMEL Microcontroller Software Support  -  ROUSSET  -
// - ----------------------------------------------------------------------------
// -  DISCLAIMER:  THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR
// -  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// -  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE
// -  DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR ANY DIRECT, INDIRECT,
// -  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// -  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// -  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// -  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// -  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// - ----------------------------------------------------------------------------
// - File Name           : AT91SAM3U1.h
// - Object              : AT91SAM3U1 definitions
// - Generated           : AT91 SW Application Group  11/17/2009 (13:04:57)
// - 
// - CVS Reference       : /AT91SAM3U1.pl/1.39/Wed Jul 29 14:37:37 2009//
// - CVS Reference       : /ADC_SAM3UE.pl/1.4/Fri Feb 20 12:19:18 2009//
// - CVS Reference       : /CORTEX_M3_MPU.pl/1.3/Fri Oct 17 13:27:48 2008//
// - CVS Reference       : /CORTEX_M3.pl/1.1/Mon Sep 15 15:22:06 2008//
// - CVS Reference       : /CORTEX_M3_NVIC.pl/1.8/Fri Jun 19 12:00:55 2009//
// - CVS Reference       : /DBGU_SAM3U4.pl/1.3/Tue May  5 11:28:09 2009//
// - CVS Reference       : /EBI_SAM9260.pl/1.1/Fri Sep 30 12:12:14 2005//
// - CVS Reference       : /EFC2_SAM3U4.pl/1.3/Mon Mar  2 10:12:06 2009//
// - CVS Reference       : /HDMA_SAM3U4.pl/1.4/Thu Jun  4 09:24:04 2009//
// - CVS Reference       : /HECC_6143A.pl/1.1/Wed Feb  9 17:16:57 2005//
// - CVS Reference       : /HMATRIX2_SAM3U4.pl/1.5/Fri Jun 26 07:25:14 2009//
// - CVS Reference       : /MCI_6101F.pl/1.3/Fri Jan 23 09:15:32 2009//
// - CVS Reference       : /PDC_6074C.pl/1.2/Thu Feb  3 09:02:11 2005//
// - CVS Reference       : /PIO3_xxxx.pl/1.6/Mon Mar  9 10:43:37 2009//
// - CVS Reference       : /PITC_6079A.pl/1.2/Thu Nov  4 13:56:22 2004//
// - CVS Reference       : /PMC_SAM3U4.pl/1.8/Tue Nov 17 10:57:14 2009//
// - CVS Reference       : /PWM_6343B_V400.pl/1.3/Fri Oct 17 13:27:54 2008//
// - CVS Reference       : /RSTC_6098A.pl/1.4/Fri Oct 17 13:27:55 2008//
// - CVS Reference       : /RTC_1245D.pl/1.3/Fri Sep 17 14:01:31 2004//
// - CVS Reference       : /RTTC_6081A.pl/1.2/Thu Nov  4 13:57:22 2004//
// - CVS Reference       : /HSDRAMC1_6100A.pl/1.2/Mon Aug  9 10:52:25 2004//
// - CVS Reference       : /SHDWC_6122A.pl/1.3/Wed Oct  6 14:16:58 2004//
// - CVS Reference       : /HSMC4_xxxx.pl/1.9/Fri Oct 17 13:27:56 2008//
// - CVS Reference       : /SPI2.pl/1.6/Mon Jul  6 07:40:56 2009//
// - CVS Reference       : /SSC_SAM3U4.pl/1.1/Thu Jun  4 09:02:35 2009//
// - CVS Reference       : /SUPC_SAM3U4.pl/1.2/Tue May  5 11:29:05 2009//
// - CVS Reference       : /SYS_SAM3U4.pl/1.4/Fri Oct 17 13:27:57 2008//
// - CVS Reference       : /TC_6082A.pl/1.8/Fri Oct 17 13:27:58 2008//
// - CVS Reference       : /TWI_6061B.pl/1.3/Fri Oct 17 13:27:59 2008//
// - CVS Reference       : /UDPHS_SAM9_7ept6dma4iso.pl/1.4/Tue Jun 24 13:05:14 2008//
// - CVS Reference       : /US_6089J.pl/1.4/Tue Jul  7 12:01:26 2009//
// - CVS Reference       : /WDTC_6080A.pl/1.3/Thu Nov  4 13:58:52 2004//
// - ----------------------------------------------------------------------------

#ifndef AT91SAM3U1_H
#define AT91SAM3U1_H

#ifdef __IAR_SYSTEMS_ICC__

typedef volatile unsigned int AT91_REG;// Hardware register definition

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR System Peripherals
// *****************************************************************************
typedef struct _AT91S_SYS {
	AT91_REG	 HSMC4_CFG; 	// Configuration Register
	AT91_REG	 HSMC4_CTRL; 	// Control Register
	AT91_REG	 HSMC4_SR; 	// Status Register
	AT91_REG	 HSMC4_IER; 	// Interrupt Enable Register
	AT91_REG	 HSMC4_IDR; 	// Interrupt Disable Register
	AT91_REG	 HSMC4_IMR; 	// Interrupt Mask Register
	AT91_REG	 HSMC4_ADDR; 	// Address Cycle Zero Register
	AT91_REG	 HSMC4_BANK; 	// Bank Register
	AT91_REG	 HSMC4_ECCCR; 	// ECC reset register
	AT91_REG	 HSMC4_ECCCMD; 	// ECC Page size register
	AT91_REG	 HSMC4_ECCSR1; 	// ECC Status register 1
	AT91_REG	 HSMC4_ECCPR0; 	// ECC Parity register 0
	AT91_REG	 HSMC4_ECCPR1; 	// ECC Parity register 1
	AT91_REG	 HSMC4_ECCSR2; 	// ECC Status register 2
	AT91_REG	 HSMC4_ECCPR2; 	// ECC Parity register 2
	AT91_REG	 HSMC4_ECCPR3; 	// ECC Parity register 3
	AT91_REG	 HSMC4_ECCPR4; 	// ECC Parity register 4
	AT91_REG	 HSMC4_ECCPR5; 	// ECC Parity register 5
	AT91_REG	 HSMC4_ECCPR6; 	// ECC Parity register 6
	AT91_REG	 HSMC4_ECCPR7; 	// ECC Parity register 7
	AT91_REG	 HSMC4_ECCPR8; 	// ECC Parity register 8
	AT91_REG	 HSMC4_ECCPR9; 	// ECC Parity register 9
	AT91_REG	 HSMC4_ECCPR10; 	// ECC Parity register 10
	AT91_REG	 HSMC4_ECCPR11; 	// ECC Parity register 11
	AT91_REG	 HSMC4_ECCPR12; 	// ECC Parity register 12
	AT91_REG	 HSMC4_ECCPR13; 	// ECC Parity register 13
	AT91_REG	 HSMC4_ECCPR14; 	// ECC Parity register 14
	AT91_REG	 HSMC4_Eccpr15; 	// ECC Parity register 15
	AT91_REG	 Reserved0[40]; 	// 
	AT91_REG	 HSMC4_OCMS; 	// OCMS MODE register
	AT91_REG	 HSMC4_KEY1; 	// KEY1 Register
	AT91_REG	 HSMC4_KEY2; 	// KEY2 Register
	AT91_REG	 Reserved1[50]; 	// 
	AT91_REG	 HSMC4_WPCR; 	// Write Protection Control register
	AT91_REG	 HSMC4_WPSR; 	// Write Protection Status Register
	AT91_REG	 HSMC4_ADDRSIZE; 	// Write Protection Status Register
	AT91_REG	 HSMC4_IPNAME1; 	// Write Protection Status Register
	AT91_REG	 HSMC4_IPNAME2; 	// Write Protection Status Register
	AT91_REG	 HSMC4_FEATURES; 	// Write Protection Status Register
	AT91_REG	 HSMC4_VER; 	// HSMC4 Version Register
	AT91_REG	 HMATRIX_MCFG0; 	//  Master Configuration Register 0 : ARM I and D
	AT91_REG	 HMATRIX_MCFG1; 	//  Master Configuration Register 1 : ARM S
	AT91_REG	 HMATRIX_MCFG2; 	//  Master Configuration Register 2
	AT91_REG	 HMATRIX_MCFG3; 	//  Master Configuration Register 3
	AT91_REG	 HMATRIX_MCFG4; 	//  Master Configuration Register 4
	AT91_REG	 HMATRIX_MCFG5; 	//  Master Configuration Register 5
	AT91_REG	 HMATRIX_MCFG6; 	//  Master Configuration Register 6
	AT91_REG	 HMATRIX_MCFG7; 	//  Master Configuration Register 7
	AT91_REG	 Reserved2[8]; 	// 
	AT91_REG	 HMATRIX_SCFG0; 	//  Slave Configuration Register 0
	AT91_REG	 HMATRIX_SCFG1; 	//  Slave Configuration Register 1
	AT91_REG	 HMATRIX_SCFG2; 	//  Slave Configuration Register 2
	AT91_REG	 HMATRIX_SCFG3; 	//  Slave Configuration Register 3
	AT91_REG	 HMATRIX_SCFG4; 	//  Slave Configuration Register 4
	AT91_REG	 HMATRIX_SCFG5; 	//  Slave Configuration Register 5
	AT91_REG	 HMATRIX_SCFG6; 	//  Slave Configuration Register 6
	AT91_REG	 HMATRIX_SCFG7; 	//  Slave Configuration Register 5
	AT91_REG	 HMATRIX_SCFG8; 	//  Slave Configuration Register 8
	AT91_REG	 HMATRIX_SCFG9; 	//  Slave Configuration Register 9
	AT91_REG	 Reserved3[42]; 	// 
	AT91_REG	 HMATRIX_SFR0 ; 	//  Special Function Register 0
	AT91_REG	 HMATRIX_SFR1 ; 	//  Special Function Register 1
	AT91_REG	 HMATRIX_SFR2 ; 	//  Special Function Register 2
	AT91_REG	 HMATRIX_SFR3 ; 	//  Special Function Register 3
	AT91_REG	 HMATRIX_SFR4 ; 	//  Special Function Register 4
	AT91_REG	 HMATRIX_SFR5 ; 	//  Special Function Register 5
	AT91_REG	 HMATRIX_SFR6 ; 	//  Special Function Register 6
	AT91_REG	 HMATRIX_SFR7 ; 	//  Special Function Register 7
	AT91_REG	 HMATRIX_SFR8 ; 	//  Special Function Register 8
	AT91_REG	 HMATRIX_SFR9 ; 	//  Special Function Register 9
	AT91_REG	 HMATRIX_SFR10; 	//  Special Function Register 10
	AT91_REG	 HMATRIX_SFR11; 	//  Special Function Register 11
	AT91_REG	 HMATRIX_SFR12; 	//  Special Function Register 12
	AT91_REG	 HMATRIX_SFR13; 	//  Special Function Register 13
	AT91_REG	 HMATRIX_SFR14; 	//  Special Function Register 14
	AT91_REG	 HMATRIX_SFR15; 	//  Special Function Register 15
	AT91_REG	 Reserved4[39]; 	// 
	AT91_REG	 HMATRIX_ADDRSIZE; 	// HMATRIX2 ADDRSIZE REGISTER 
	AT91_REG	 HMATRIX_IPNAME1; 	// HMATRIX2 IPNAME1 REGISTER 
	AT91_REG	 HMATRIX_IPNAME2; 	// HMATRIX2 IPNAME2 REGISTER 
	AT91_REG	 HMATRIX_FEATURES; 	// HMATRIX2 FEATURES REGISTER 
	AT91_REG	 HMATRIX_VER; 	// HMATRIX2 VERSION REGISTER 
	AT91_REG	 PMC_SCER; 	// System Clock Enable Register
	AT91_REG	 PMC_SCDR; 	// System Clock Disable Register
	AT91_REG	 PMC_SCSR; 	// System Clock Status Register
	AT91_REG	 Reserved5[1]; 	// 
	AT91_REG	 PMC_PCER; 	// Peripheral Clock Enable Register
	AT91_REG	 PMC_PCDR; 	// Peripheral Clock Disable Register
	AT91_REG	 PMC_PCSR; 	// Peripheral Clock Status Register
	AT91_REG	 PMC_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 PMC_MOR; 	// Main Oscillator Register
	AT91_REG	 PMC_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 PMC_PLLAR; 	// PLL Register
	AT91_REG	 Reserved6[1]; 	// 
	AT91_REG	 PMC_MCKR; 	// Master Clock Register
	AT91_REG	 Reserved7[3]; 	// 
	AT91_REG	 PMC_PCKR[8]; 	// Programmable Clock Register
	AT91_REG	 PMC_IER; 	// Interrupt Enable Register
	AT91_REG	 PMC_IDR; 	// Interrupt Disable Register
	AT91_REG	 PMC_SR; 	// Status Register
	AT91_REG	 PMC_IMR; 	// Interrupt Mask Register
	AT91_REG	 PMC_FSMR; 	// Fast Startup Mode Register
	AT91_REG	 PMC_FSPR; 	// Fast Startup Polarity Register
	AT91_REG	 PMC_FOCR; 	// Fault Output Clear Register
	AT91_REG	 Reserved8[28]; 	// 
	AT91_REG	 PMC_ADDRSIZE; 	// PMC ADDRSIZE REGISTER 
	AT91_REG	 PMC_IPNAME1; 	// PMC IPNAME1 REGISTER 
	AT91_REG	 PMC_IPNAME2; 	// PMC IPNAME2 REGISTER 
	AT91_REG	 PMC_FEATURES; 	// PMC FEATURES REGISTER 
	AT91_REG	 PMC_VER; 	// APMC VERSION REGISTER
	AT91_REG	 Reserved9[64]; 	// 
	AT91_REG	 DBGU_CR; 	// Control Register
	AT91_REG	 DBGU_MR; 	// Mode Register
	AT91_REG	 DBGU_IER; 	// Interrupt Enable Register
	AT91_REG	 DBGU_IDR; 	// Interrupt Disable Register
	AT91_REG	 DBGU_IMR; 	// Interrupt Mask Register
	AT91_REG	 DBGU_CSR; 	// Channel Status Register
	AT91_REG	 DBGU_RHR; 	// Receiver Holding Register
	AT91_REG	 DBGU_THR; 	// Transmitter Holding Register
	AT91_REG	 DBGU_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 Reserved10[9]; 	// 
	AT91_REG	 DBGU_FNTR; 	// Force NTRST Register
	AT91_REG	 Reserved11[40]; 	// 
	AT91_REG	 DBGU_ADDRSIZE; 	// DBGU ADDRSIZE REGISTER 
	AT91_REG	 DBGU_IPNAME1; 	// DBGU IPNAME1 REGISTER 
	AT91_REG	 DBGU_IPNAME2; 	// DBGU IPNAME2 REGISTER 
	AT91_REG	 DBGU_FEATURES; 	// DBGU FEATURES REGISTER 
	AT91_REG	 DBGU_VER; 	// DBGU VERSION REGISTER 
	AT91_REG	 DBGU_RPR; 	// Receive Pointer Register
	AT91_REG	 DBGU_RCR; 	// Receive Counter Register
	AT91_REG	 DBGU_TPR; 	// Transmit Pointer Register
	AT91_REG	 DBGU_TCR; 	// Transmit Counter Register
	AT91_REG	 DBGU_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DBGU_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DBGU_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DBGU_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DBGU_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DBGU_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved12[6]; 	// 
	AT91_REG	 DBGU_CIDR; 	// Chip ID Register
	AT91_REG	 DBGU_EXID; 	// Chip ID Extension Register
	AT91_REG	 Reserved13[46]; 	// 
	AT91_REG	 EFC0_FMR; 	// EFC Flash Mode Register
	AT91_REG	 EFC0_FCR; 	// EFC Flash Command Register
	AT91_REG	 EFC0_FSR; 	// EFC Flash Status Register
	AT91_REG	 EFC0_FRR; 	// EFC Flash Result Register
	AT91_REG	 Reserved14[1]; 	// 
	AT91_REG	 EFC0_FVR; 	// EFC Flash Version Register
	AT91_REG	 Reserved15[122]; 	// 
	AT91_REG	 EFC1_FMR; 	// EFC Flash Mode Register
	AT91_REG	 EFC1_FCR; 	// EFC Flash Command Register
	AT91_REG	 EFC1_FSR; 	// EFC Flash Status Register
	AT91_REG	 EFC1_FRR; 	// EFC Flash Result Register
	AT91_REG	 Reserved16[1]; 	// 
	AT91_REG	 EFC1_FVR; 	// EFC Flash Version Register
	AT91_REG	 Reserved17[122]; 	// 
	AT91_REG	 PIOA_PER; 	// PIO Enable Register
	AT91_REG	 PIOA_PDR; 	// PIO Disable Register
	AT91_REG	 PIOA_PSR; 	// PIO Status Register
	AT91_REG	 Reserved18[1]; 	// 
	AT91_REG	 PIOA_OER; 	// Output Enable Register
	AT91_REG	 PIOA_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOA_OSR; 	// Output Status Register
	AT91_REG	 Reserved19[1]; 	// 
	AT91_REG	 PIOA_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOA_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOA_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved20[1]; 	// 
	AT91_REG	 PIOA_SODR; 	// Set Output Data Register
	AT91_REG	 PIOA_CODR; 	// Clear Output Data Register
	AT91_REG	 PIOA_ODSR; 	// Output Data Status Register
	AT91_REG	 PIOA_PDSR; 	// Pin Data Status Register
	AT91_REG	 PIOA_IER; 	// Interrupt Enable Register
	AT91_REG	 PIOA_IDR; 	// Interrupt Disable Register
	AT91_REG	 PIOA_IMR; 	// Interrupt Mask Register
	AT91_REG	 PIOA_ISR; 	// Interrupt Status Register
	AT91_REG	 PIOA_MDER; 	// Multi-driver Enable Register
	AT91_REG	 PIOA_MDDR; 	// Multi-driver Disable Register
	AT91_REG	 PIOA_MDSR; 	// Multi-driver Status Register
	AT91_REG	 Reserved21[1]; 	// 
	AT91_REG	 PIOA_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOA_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOA_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved22[1]; 	// 
	AT91_REG	 PIOA_ABSR; 	// Peripheral AB Select Register
	AT91_REG	 Reserved23[3]; 	// 
	AT91_REG	 PIOA_SCIFSR; 	// System Clock Glitch Input Filter Select Register
	AT91_REG	 PIOA_DIFSR; 	// Debouncing Input Filter Select Register
	AT91_REG	 PIOA_IFDGSR; 	// Glitch or Debouncing Input Filter Clock Selection Status Register
	AT91_REG	 PIOA_SCDR; 	// Slow Clock Divider Debouncing Register
	AT91_REG	 Reserved24[4]; 	// 
	AT91_REG	 PIOA_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOA_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOA_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved25[1]; 	// 
	AT91_REG	 PIOA_AIMER; 	// Additional Interrupt Modes Enable Register
	AT91_REG	 PIOA_AIMDR; 	// Additional Interrupt Modes Disables Register
	AT91_REG	 PIOA_AIMMR; 	// Additional Interrupt Modes Mask Register
	AT91_REG	 Reserved26[1]; 	// 
	AT91_REG	 PIOA_ESR; 	// Edge Select Register
	AT91_REG	 PIOA_LSR; 	// Level Select Register
	AT91_REG	 PIOA_ELSR; 	// Edge/Level Status Register
	AT91_REG	 Reserved27[1]; 	// 
	AT91_REG	 PIOA_FELLSR; 	// Falling Edge/Low Level Select Register
	AT91_REG	 PIOA_REHLSR; 	// Rising Edge/ High Level Select Register
	AT91_REG	 PIOA_FRLHSR; 	// Fall/Rise - Low/High Status Register
	AT91_REG	 Reserved28[1]; 	// 
	AT91_REG	 PIOA_LOCKSR; 	// Lock Status Register
	AT91_REG	 Reserved29[6]; 	// 
	AT91_REG	 PIOA_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 Reserved30[8]; 	// 
	AT91_REG	 PIOA_KER; 	// Keypad Controller Enable Register
	AT91_REG	 PIOA_KRCR; 	// Keypad Controller Row Column Register
	AT91_REG	 PIOA_KDR; 	// Keypad Controller Debouncing Register
	AT91_REG	 Reserved31[1]; 	// 
	AT91_REG	 PIOA_KIER; 	// Keypad Controller Interrupt Enable Register
	AT91_REG	 PIOA_KIDR; 	// Keypad Controller Interrupt Disable Register
	AT91_REG	 PIOA_KIMR; 	// Keypad Controller Interrupt Mask Register
	AT91_REG	 PIOA_KSR; 	// Keypad Controller Status Register
	AT91_REG	 PIOA_KKPR; 	// Keypad Controller Key Press Register
	AT91_REG	 PIOA_KKRR; 	// Keypad Controller Key Release Register
	AT91_REG	 Reserved32[46]; 	// 
	AT91_REG	 PIOB_PER; 	// PIO Enable Register
	AT91_REG	 PIOB_PDR; 	// PIO Disable Register
	AT91_REG	 PIOB_PSR; 	// PIO Status Register
	AT91_REG	 Reserved33[1]; 	// 
	AT91_REG	 PIOB_OER; 	// Output Enable Register
	AT91_REG	 PIOB_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOB_OSR; 	// Output Status Register
	AT91_REG	 Reserved34[1]; 	// 
	AT91_REG	 PIOB_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOB_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOB_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved35[1]; 	// 
	AT91_REG	 PIOB_SODR; 	// Set Output Data Register
	AT91_REG	 PIOB_CODR; 	// Clear Output Data Register
	AT91_REG	 PIOB_ODSR; 	// Output Data Status Register
	AT91_REG	 PIOB_PDSR; 	// Pin Data Status Register
	AT91_REG	 PIOB_IER; 	// Interrupt Enable Register
	AT91_REG	 PIOB_IDR; 	// Interrupt Disable Register
	AT91_REG	 PIOB_IMR; 	// Interrupt Mask Register
	AT91_REG	 PIOB_ISR; 	// Interrupt Status Register
	AT91_REG	 PIOB_MDER; 	// Multi-driver Enable Register
	AT91_REG	 PIOB_MDDR; 	// Multi-driver Disable Register
	AT91_REG	 PIOB_MDSR; 	// Multi-driver Status Register
	AT91_REG	 Reserved36[1]; 	// 
	AT91_REG	 PIOB_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOB_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOB_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved37[1]; 	// 
	AT91_REG	 PIOB_ABSR; 	// Peripheral AB Select Register
	AT91_REG	 Reserved38[3]; 	// 
	AT91_REG	 PIOB_SCIFSR; 	// System Clock Glitch Input Filter Select Register
	AT91_REG	 PIOB_DIFSR; 	// Debouncing Input Filter Select Register
	AT91_REG	 PIOB_IFDGSR; 	// Glitch or Debouncing Input Filter Clock Selection Status Register
	AT91_REG	 PIOB_SCDR; 	// Slow Clock Divider Debouncing Register
	AT91_REG	 Reserved39[4]; 	// 
	AT91_REG	 PIOB_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOB_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOB_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved40[1]; 	// 
	AT91_REG	 PIOB_AIMER; 	// Additional Interrupt Modes Enable Register
	AT91_REG	 PIOB_AIMDR; 	// Additional Interrupt Modes Disables Register
	AT91_REG	 PIOB_AIMMR; 	// Additional Interrupt Modes Mask Register
	AT91_REG	 Reserved41[1]; 	// 
	AT91_REG	 PIOB_ESR; 	// Edge Select Register
	AT91_REG	 PIOB_LSR; 	// Level Select Register
	AT91_REG	 PIOB_ELSR; 	// Edge/Level Status Register
	AT91_REG	 Reserved42[1]; 	// 
	AT91_REG	 PIOB_FELLSR; 	// Falling Edge/Low Level Select Register
	AT91_REG	 PIOB_REHLSR; 	// Rising Edge/ High Level Select Register
	AT91_REG	 PIOB_FRLHSR; 	// Fall/Rise - Low/High Status Register
	AT91_REG	 Reserved43[1]; 	// 
	AT91_REG	 PIOB_LOCKSR; 	// Lock Status Register
	AT91_REG	 Reserved44[6]; 	// 
	AT91_REG	 PIOB_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 Reserved45[8]; 	// 
	AT91_REG	 PIOB_KER; 	// Keypad Controller Enable Register
	AT91_REG	 PIOB_KRCR; 	// Keypad Controller Row Column Register
	AT91_REG	 PIOB_KDR; 	// Keypad Controller Debouncing Register
	AT91_REG	 Reserved46[1]; 	// 
	AT91_REG	 PIOB_KIER; 	// Keypad Controller Interrupt Enable Register
	AT91_REG	 PIOB_KIDR; 	// Keypad Controller Interrupt Disable Register
	AT91_REG	 PIOB_KIMR; 	// Keypad Controller Interrupt Mask Register
	AT91_REG	 PIOB_KSR; 	// Keypad Controller Status Register
	AT91_REG	 PIOB_KKPR; 	// Keypad Controller Key Press Register
	AT91_REG	 PIOB_KKRR; 	// Keypad Controller Key Release Register
	AT91_REG	 Reserved47[46]; 	// 
	AT91_REG	 PIOC_PER; 	// PIO Enable Register
	AT91_REG	 PIOC_PDR; 	// PIO Disable Register
	AT91_REG	 PIOC_PSR; 	// PIO Status Register
	AT91_REG	 Reserved48[1]; 	// 
	AT91_REG	 PIOC_OER; 	// Output Enable Register
	AT91_REG	 PIOC_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOC_OSR; 	// Output Status Register
	AT91_REG	 Reserved49[1]; 	// 
	AT91_REG	 PIOC_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOC_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOC_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved50[1]; 	// 
	AT91_REG	 PIOC_SODR; 	// Set Output Data Register
	AT91_REG	 PIOC_CODR; 	// Clear Output Data Register
	AT91_REG	 PIOC_ODSR; 	// Output Data Status Register
	AT91_REG	 PIOC_PDSR; 	// Pin Data Status Register
	AT91_REG	 PIOC_IER; 	// Interrupt Enable Register
	AT91_REG	 PIOC_IDR; 	// Interrupt Disable Register
	AT91_REG	 PIOC_IMR; 	// Interrupt Mask Register
	AT91_REG	 PIOC_ISR; 	// Interrupt Status Register
	AT91_REG	 PIOC_MDER; 	// Multi-driver Enable Register
	AT91_REG	 PIOC_MDDR; 	// Multi-driver Disable Register
	AT91_REG	 PIOC_MDSR; 	// Multi-driver Status Register
	AT91_REG	 Reserved51[1]; 	// 
	AT91_REG	 PIOC_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOC_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOC_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved52[1]; 	// 
	AT91_REG	 PIOC_ABSR; 	// Peripheral AB Select Register
	AT91_REG	 Reserved53[3]; 	// 
	AT91_REG	 PIOC_SCIFSR; 	// System Clock Glitch Input Filter Select Register
	AT91_REG	 PIOC_DIFSR; 	// Debouncing Input Filter Select Register
	AT91_REG	 PIOC_IFDGSR; 	// Glitch or Debouncing Input Filter Clock Selection Status Register
	AT91_REG	 PIOC_SCDR; 	// Slow Clock Divider Debouncing Register
	AT91_REG	 Reserved54[4]; 	// 
	AT91_REG	 PIOC_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOC_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOC_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved55[1]; 	// 
	AT91_REG	 PIOC_AIMER; 	// Additional Interrupt Modes Enable Register
	AT91_REG	 PIOC_AIMDR; 	// Additional Interrupt Modes Disables Register
	AT91_REG	 PIOC_AIMMR; 	// Additional Interrupt Modes Mask Register
	AT91_REG	 Reserved56[1]; 	// 
	AT91_REG	 PIOC_ESR; 	// Edge Select Register
	AT91_REG	 PIOC_LSR; 	// Level Select Register
	AT91_REG	 PIOC_ELSR; 	// Edge/Level Status Register
	AT91_REG	 Reserved57[1]; 	// 
	AT91_REG	 PIOC_FELLSR; 	// Falling Edge/Low Level Select Register
	AT91_REG	 PIOC_REHLSR; 	// Rising Edge/ High Level Select Register
	AT91_REG	 PIOC_FRLHSR; 	// Fall/Rise - Low/High Status Register
	AT91_REG	 Reserved58[1]; 	// 
	AT91_REG	 PIOC_LOCKSR; 	// Lock Status Register
	AT91_REG	 Reserved59[6]; 	// 
	AT91_REG	 PIOC_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 Reserved60[8]; 	// 
	AT91_REG	 PIOC_KER; 	// Keypad Controller Enable Register
	AT91_REG	 PIOC_KRCR; 	// Keypad Controller Row Column Register
	AT91_REG	 PIOC_KDR; 	// Keypad Controller Debouncing Register
	AT91_REG	 Reserved61[1]; 	// 
	AT91_REG	 PIOC_KIER; 	// Keypad Controller Interrupt Enable Register
	AT91_REG	 PIOC_KIDR; 	// Keypad Controller Interrupt Disable Register
	AT91_REG	 PIOC_KIMR; 	// Keypad Controller Interrupt Mask Register
	AT91_REG	 PIOC_KSR; 	// Keypad Controller Status Register
	AT91_REG	 PIOC_KKPR; 	// Keypad Controller Key Press Register
	AT91_REG	 PIOC_KKRR; 	// Keypad Controller Key Release Register
	AT91_REG	 Reserved62[46]; 	// 
	AT91_REG	 RSTC_RCR; 	// Reset Control Register
	AT91_REG	 RSTC_RSR; 	// Reset Status Register
	AT91_REG	 RSTC_RMR; 	// Reset Mode Register
	AT91_REG	 Reserved63[1]; 	// 
	AT91_REG	 SUPC_CR; 	// Control Register
	AT91_REG	 SUPC_BOMR; 	// Brown Out Mode Register
	AT91_REG	 SUPC_MR; 	// Mode Register
	AT91_REG	 SUPC_WUMR; 	// Wake Up Mode Register
	AT91_REG	 SUPC_WUIR; 	// Wake Up Inputs Register
	AT91_REG	 SUPC_SR; 	// Status Register
	AT91_REG	 SUPC_FWUTR; 	// Flash Wake-up Timer Register
	AT91_REG	 Reserved64[1]; 	// 
	AT91_REG	 RTTC_RTMR; 	// Real-time Mode Register
	AT91_REG	 RTTC_RTAR; 	// Real-time Alarm Register
	AT91_REG	 RTTC_RTVR; 	// Real-time Value Register
	AT91_REG	 RTTC_RTSR; 	// Real-time Status Register
	AT91_REG	 Reserved65[4]; 	// 
	AT91_REG	 WDTC_WDCR; 	// Watchdog Control Register
	AT91_REG	 WDTC_WDMR; 	// Watchdog Mode Register
	AT91_REG	 WDTC_WDSR; 	// Watchdog Status Register
	AT91_REG	 Reserved66[1]; 	// 
	AT91_REG	 RTC_CR; 	// Control Register
	AT91_REG	 RTC_MR; 	// Mode Register
	AT91_REG	 RTC_TIMR; 	// Time Register
	AT91_REG	 RTC_CALR; 	// Calendar Register
	AT91_REG	 RTC_TIMALR; 	// Time Alarm Register
	AT91_REG	 RTC_CALALR; 	// Calendar Alarm Register
	AT91_REG	 RTC_SR; 	// Status Register
	AT91_REG	 RTC_SCCR; 	// Status Clear Command Register
	AT91_REG	 RTC_IER; 	// Interrupt Enable Register
	AT91_REG	 RTC_IDR; 	// Interrupt Disable Register
	AT91_REG	 RTC_IMR; 	// Interrupt Mask Register
	AT91_REG	 RTC_VER; 	// Valid Entry Register
	AT91_REG	 SYS_GPBR[8]; 	// General Purpose Register
	AT91_REG	 Reserved67[19]; 	// 
	AT91_REG	 RSTC_VER; 	// Version Register
} AT91S_SYS, *AT91PS_SYS;

// -------- GPBR : (SYS Offset: 0x1290) GPBR General Purpose Register -------- 
#define AT91C_GPBR_GPRV       ((unsigned int) 0x0 <<  0) // (SYS) General Purpose Register Value

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR HSMC4 Chip Select interface
// *****************************************************************************
typedef struct _AT91S_HSMC4_CS {
	AT91_REG	 HSMC4_SETUP; 	// Setup Register
	AT91_REG	 HSMC4_PULSE; 	// Pulse Register
	AT91_REG	 HSMC4_CYCLE; 	// Cycle Register
	AT91_REG	 HSMC4_TIMINGS; 	// Timmings Register
	AT91_REG	 HSMC4_MODE; 	// Mode Register
} AT91S_HSMC4_CS, *AT91PS_HSMC4_CS;

// -------- HSMC4_SETUP : (HSMC4_CS Offset: 0x0) HSMC4 SETUP -------- 
#define AT91C_HSMC4_NWE_SETUP ((unsigned int) 0x3F <<  0) // (HSMC4_CS) NWE Setup length
#define AT91C_HSMC4_NCS_WR_SETUP ((unsigned int) 0x3F <<  8) // (HSMC4_CS) NCS Setup length in Write access
#define AT91C_HSMC4_NRD_SETUP ((unsigned int) 0x3F << 16) // (HSMC4_CS) NRD Setup length
#define AT91C_HSMC4_NCS_RD_SETUP ((unsigned int) 0x3F << 24) // (HSMC4_CS) NCS Setup legnth in Read access
// -------- HSMC4_PULSE : (HSMC4_CS Offset: 0x4) HSMC4 PULSE -------- 
#define AT91C_HSMC4_NWE_PULSE ((unsigned int) 0x3F <<  0) // (HSMC4_CS) NWE Pulse Length
#define AT91C_HSMC4_NCS_WR_PULSE ((unsigned int) 0x3F <<  8) // (HSMC4_CS) NCS Pulse length in WRITE access
#define AT91C_HSMC4_NRD_PULSE ((unsigned int) 0x3F << 16) // (HSMC4_CS) NRD Pulse length
#define AT91C_HSMC4_NCS_RD_PULSE ((unsigned int) 0x3F << 24) // (HSMC4_CS) NCS Pulse length in READ access
// -------- HSMC4_CYCLE : (HSMC4_CS Offset: 0x8) HSMC4 CYCLE -------- 
#define AT91C_HSMC4_NWE_CYCLE ((unsigned int) 0x1FF <<  0) // (HSMC4_CS) Total Write Cycle Length
#define AT91C_HSMC4_NRD_CYCLE ((unsigned int) 0x1FF << 16) // (HSMC4_CS) Total Read Cycle Length
// -------- HSMC4_TIMINGS : (HSMC4_CS Offset: 0xc) HSMC4 TIMINGS -------- 
#define AT91C_HSMC4_TCLR      ((unsigned int) 0xF <<  0) // (HSMC4_CS) CLE to REN low delay
#define AT91C_HSMC4_TADL      ((unsigned int) 0xF <<  4) // (HSMC4_CS) ALE to data start
#define AT91C_HSMC4_TAR       ((unsigned int) 0xF <<  8) // (HSMC4_CS) ALE to REN low delay
#define AT91C_HSMC4_OCMSEN    ((unsigned int) 0x1 << 12) // (HSMC4_CS) Off Chip Memory Scrambling Enable
#define AT91C_HSMC4_TRR       ((unsigned int) 0xF << 16) // (HSMC4_CS) Ready to REN low delay
#define AT91C_HSMC4_TWB       ((unsigned int) 0xF << 24) // (HSMC4_CS) WEN high to REN to busy
#define AT91C_HSMC4_RBNSEL    ((unsigned int) 0x7 << 28) // (HSMC4_CS) Ready/Busy Line Selection
#define AT91C_HSMC4_NFSEL     ((unsigned int) 0x1 << 31) // (HSMC4_CS) Nand Flash Selection
// -------- HSMC4_MODE : (HSMC4_CS Offset: 0x10) HSMC4 MODE -------- 
#define AT91C_HSMC4_READ_MODE ((unsigned int) 0x1 <<  0) // (HSMC4_CS) Read Mode
#define AT91C_HSMC4_WRITE_MODE ((unsigned int) 0x1 <<  1) // (HSMC4_CS) Write Mode
#define AT91C_HSMC4_EXNW_MODE ((unsigned int) 0x3 <<  4) // (HSMC4_CS) NWAIT Mode
#define 	AT91C_HSMC4_EXNW_MODE_NWAIT_DISABLE        ((unsigned int) 0x0 <<  4) // (HSMC4_CS) External NWAIT disabled.
#define 	AT91C_HSMC4_EXNW_MODE_NWAIT_ENABLE_FROZEN  ((unsigned int) 0x2 <<  4) // (HSMC4_CS) External NWAIT enabled in frozen mode.
#define 	AT91C_HSMC4_EXNW_MODE_NWAIT_ENABLE_READY   ((unsigned int) 0x3 <<  4) // (HSMC4_CS) External NWAIT enabled in ready mode.
#define AT91C_HSMC4_BAT       ((unsigned int) 0x1 <<  8) // (HSMC4_CS) Byte Access Type
#define 	AT91C_HSMC4_BAT_BYTE_SELECT          ((unsigned int) 0x0 <<  8) // (HSMC4_CS) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
#define 	AT91C_HSMC4_BAT_BYTE_WRITE           ((unsigned int) 0x1 <<  8) // (HSMC4_CS) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
#define AT91C_HSMC4_DBW       ((unsigned int) 0x3 << 12) // (HSMC4_CS) Data Bus Width
#define 	AT91C_HSMC4_DBW_WIDTH_EIGTH_BITS     ((unsigned int) 0x0 << 12) // (HSMC4_CS) 8 bits.
#define 	AT91C_HSMC4_DBW_WIDTH_SIXTEEN_BITS   ((unsigned int) 0x1 << 12) // (HSMC4_CS) 16 bits.
#define 	AT91C_HSMC4_DBW_WIDTH_THIRTY_TWO_BITS ((unsigned int) 0x2 << 12) // (HSMC4_CS) 32 bits.
#define AT91C_HSMC4_TDF_CYCLES ((unsigned int) 0xF << 16) // (HSMC4_CS) Data Float Time.
#define AT91C_HSMC4_TDF_MODE  ((unsigned int) 0x1 << 20) // (HSMC4_CS) TDF Enabled.
#define AT91C_HSMC4_PMEN      ((unsigned int) 0x1 << 24) // (HSMC4_CS) Page Mode Enabled.
#define AT91C_HSMC4_PS        ((unsigned int) 0x3 << 28) // (HSMC4_CS) Page Size
#define 	AT91C_HSMC4_PS_SIZE_FOUR_BYTES      ((unsigned int) 0x0 << 28) // (HSMC4_CS) 4 bytes.
#define 	AT91C_HSMC4_PS_SIZE_EIGHT_BYTES     ((unsigned int) 0x1 << 28) // (HSMC4_CS) 8 bytes.
#define 	AT91C_HSMC4_PS_SIZE_SIXTEEN_BYTES   ((unsigned int) 0x2 << 28) // (HSMC4_CS) 16 bytes.
#define 	AT91C_HSMC4_PS_SIZE_THIRTY_TWO_BYTES ((unsigned int) 0x3 << 28) // (HSMC4_CS) 32 bytes.

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR AHB Static Memory Controller 4 Interface
// *****************************************************************************
typedef struct _AT91S_HSMC4 {
	AT91_REG	 HSMC4_CFG; 	// Configuration Register
	AT91_REG	 HSMC4_CTRL; 	// Control Register
	AT91_REG	 HSMC4_SR; 	// Status Register
	AT91_REG	 HSMC4_IER; 	// Interrupt Enable Register
	AT91_REG	 HSMC4_IDR; 	// Interrupt Disable Register
	AT91_REG	 HSMC4_IMR; 	// Interrupt Mask Register
	AT91_REG	 HSMC4_ADDR; 	// Address Cycle Zero Register
	AT91_REG	 HSMC4_BANK; 	// Bank Register
	AT91_REG	 HSMC4_ECCCR; 	// ECC reset register
	AT91_REG	 HSMC4_ECCCMD; 	// ECC Page size register
	AT91_REG	 HSMC4_ECCSR1; 	// ECC Status register 1
	AT91_REG	 HSMC4_ECCPR0; 	// ECC Parity register 0
	AT91_REG	 HSMC4_ECCPR1; 	// ECC Parity register 1
	AT91_REG	 HSMC4_ECCSR2; 	// ECC Status register 2
	AT91_REG	 HSMC4_ECCPR2; 	// ECC Parity register 2
	AT91_REG	 HSMC4_ECCPR3; 	// ECC Parity register 3
	AT91_REG	 HSMC4_ECCPR4; 	// ECC Parity register 4
	AT91_REG	 HSMC4_ECCPR5; 	// ECC Parity register 5
	AT91_REG	 HSMC4_ECCPR6; 	// ECC Parity register 6
	AT91_REG	 HSMC4_ECCPR7; 	// ECC Parity register 7
	AT91_REG	 HSMC4_ECCPR8; 	// ECC Parity register 8
	AT91_REG	 HSMC4_ECCPR9; 	// ECC Parity register 9
	AT91_REG	 HSMC4_ECCPR10; 	// ECC Parity register 10
	AT91_REG	 HSMC4_ECCPR11; 	// ECC Parity register 11
	AT91_REG	 HSMC4_ECCPR12; 	// ECC Parity register 12
	AT91_REG	 HSMC4_ECCPR13; 	// ECC Parity register 13
	AT91_REG	 HSMC4_ECCPR14; 	// ECC Parity register 14
	AT91_REG	 HSMC4_Eccpr15; 	// ECC Parity register 15
	AT91_REG	 Reserved0[40]; 	// 
	AT91_REG	 HSMC4_OCMS; 	// OCMS MODE register
	AT91_REG	 HSMC4_KEY1; 	// KEY1 Register
	AT91_REG	 HSMC4_KEY2; 	// KEY2 Register
	AT91_REG	 Reserved1[50]; 	// 
	AT91_REG	 HSMC4_WPCR; 	// Write Protection Control register
	AT91_REG	 HSMC4_WPSR; 	// Write Protection Status Register
	AT91_REG	 HSMC4_ADDRSIZE; 	// Write Protection Status Register
	AT91_REG	 HSMC4_IPNAME1; 	// Write Protection Status Register
	AT91_REG	 HSMC4_IPNAME2; 	// Write Protection Status Register
	AT91_REG	 HSMC4_FEATURES; 	// Write Protection Status Register
	AT91_REG	 HSMC4_VER; 	// HSMC4 Version Register
	AT91_REG	 HSMC4_DUMMY; 	// This rtegister was created only ti have AHB constants
} AT91S_HSMC4, *AT91PS_HSMC4;

// -------- HSMC4_CFG : (HSMC4 Offset: 0x0) Configuration Register -------- 
#define AT91C_HSMC4_PAGESIZE  ((unsigned int) 0x3 <<  0) // (HSMC4) PAGESIZE field description
#define 	AT91C_HSMC4_PAGESIZE_528_Bytes            ((unsigned int) 0x0) // (HSMC4) 512 bytes plus 16 bytes page size
#define 	AT91C_HSMC4_PAGESIZE_1056_Bytes           ((unsigned int) 0x1) // (HSMC4) 1024 bytes plus 32 bytes page size
#define 	AT91C_HSMC4_PAGESIZE_2112_Bytes           ((unsigned int) 0x2) // (HSMC4) 2048 bytes plus 64 bytes page size
#define 	AT91C_HSMC4_PAGESIZE_4224_Bytes           ((unsigned int) 0x3) // (HSMC4) 4096 bytes plus 128 bytes page size
#define AT91C_HSMC4_WSPARE    ((unsigned int) 0x1 <<  8) // (HSMC4) Spare area access in Write Mode
#define AT91C_HSMC4_RSPARE    ((unsigned int) 0x1 <<  9) // (HSMC4) Spare area access in Read Mode
#define AT91C_HSMC4_EDGECTRL  ((unsigned int) 0x1 << 12) // (HSMC4) Rising/Falling Edge Detection Control
#define AT91C_HSMC4_RBEDGE    ((unsigned int) 0x1 << 13) // (HSMC4) Ready/Busy Signal edge Detection
#define AT91C_HSMC4_DTOCYC    ((unsigned int) 0xF << 16) // (HSMC4) Data Timeout Cycle Number
#define AT91C_HSMC4_DTOMUL    ((unsigned int) 0x7 << 20) // (HSMC4) Data Timeout Multiplier
#define 	AT91C_HSMC4_DTOMUL_1                    ((unsigned int) 0x0 << 20) // (HSMC4) DTOCYC x 1
#define 	AT91C_HSMC4_DTOMUL_16                   ((unsigned int) 0x1 << 20) // (HSMC4) DTOCYC x 16
#define 	AT91C_HSMC4_DTOMUL_128                  ((unsigned int) 0x2 << 20) // (HSMC4) DTOCYC x 128
#define 	AT91C_HSMC4_DTOMUL_256                  ((unsigned int) 0x3 << 20) // (HSMC4) DTOCYC x 256
#define 	AT91C_HSMC4_DTOMUL_1024                 ((unsigned int) 0x4 << 20) // (HSMC4) DTOCYC x 1024
#define 	AT91C_HSMC4_DTOMUL_4096                 ((unsigned int) 0x5 << 20) // (HSMC4) DTOCYC x 4096
#define 	AT91C_HSMC4_DTOMUL_65536                ((unsigned int) 0x6 << 20) // (HSMC4) DTOCYC x 65536
#define 	AT91C_HSMC4_DTOMUL_1048576              ((unsigned int) 0x7 << 20) // (HSMC4) DTOCYC x 1048576
// -------- HSMC4_CTRL : (HSMC4 Offset: 0x4) Control Register -------- 
#define AT91C_HSMC4_NFCEN     ((unsigned int) 0x1 <<  0) // (HSMC4) Nand Flash Controller Host Enable
#define AT91C_HSMC4_NFCDIS    ((unsigned int) 0x1 <<  1) // (HSMC4) Nand Flash Controller Host Disable
#define AT91C_HSMC4_HOSTEN    ((unsigned int) 0x1 <<  8) // (HSMC4) If set to one, the Host controller is activated and perform a data transfer phase.
#define AT91C_HSMC4_HOSTWR    ((unsigned int) 0x1 << 11) // (HSMC4) If this field is set to one, the host transfers data from the internal SRAM to the Memory Device.
#define AT91C_HSMC4_HOSTCSID  ((unsigned int) 0x7 << 12) // (HSMC4) Host Controller Chip select Id
#define 	AT91C_HSMC4_HOSTCSID_0                    ((unsigned int) 0x0 << 12) // (HSMC4) CS0
#define 	AT91C_HSMC4_HOSTCSID_1                    ((unsigned int) 0x1 << 12) // (HSMC4) CS1
#define 	AT91C_HSMC4_HOSTCSID_2                    ((unsigned int) 0x2 << 12) // (HSMC4) CS2
#define 	AT91C_HSMC4_HOSTCSID_3                    ((unsigned int) 0x3 << 12) // (HSMC4) CS3
#define 	AT91C_HSMC4_HOSTCSID_4                    ((unsigned int) 0x4 << 12) // (HSMC4) CS4
#define 	AT91C_HSMC4_HOSTCSID_5                    ((unsigned int) 0x5 << 12) // (HSMC4) CS5
#define 	AT91C_HSMC4_HOSTCSID_6                    ((unsigned int) 0x6 << 12) // (HSMC4) CS6
#define 	AT91C_HSMC4_HOSTCSID_7                    ((unsigned int) 0x7 << 12) // (HSMC4) CS7
#define AT91C_HSMC4_VALID     ((unsigned int) 0x1 << 15) // (HSMC4) When set to 1, a write operation modifies both HOSTCSID and HOSTWR fields.
// -------- HSMC4_SR : (HSMC4 Offset: 0x8) HSMC4 Status Register -------- 
#define AT91C_HSMC4_NFCSTS    ((unsigned int) 0x1 <<  0) // (HSMC4) Nand Flash Controller status
#define AT91C_HSMC4_RBRISE    ((unsigned int) 0x1 <<  4) // (HSMC4) Selected Ready Busy Rising Edge Detected flag
#define AT91C_HSMC4_RBFALL    ((unsigned int) 0x1 <<  5) // (HSMC4) Selected Ready Busy Falling Edge Detected flag
#define AT91C_HSMC4_HOSTBUSY  ((unsigned int) 0x1 <<  8) // (HSMC4) Host Busy
#define AT91C_HSMC4_HOSTW     ((unsigned int) 0x1 << 11) // (HSMC4) Host Write/Read Operation
#define AT91C_HSMC4_HOSTCS    ((unsigned int) 0x7 << 12) // (HSMC4) Host Controller Chip select Id
#define 	AT91C_HSMC4_HOSTCS_0                    ((unsigned int) 0x0 << 12) // (HSMC4) CS0
#define 	AT91C_HSMC4_HOSTCS_1                    ((unsigned int) 0x1 << 12) // (HSMC4) CS1
#define 	AT91C_HSMC4_HOSTCS_2                    ((unsigned int) 0x2 << 12) // (HSMC4) CS2
#define 	AT91C_HSMC4_HOSTCS_3                    ((unsigned int) 0x3 << 12) // (HSMC4) CS3
#define 	AT91C_HSMC4_HOSTCS_4                    ((unsigned int) 0x4 << 12) // (HSMC4) CS4
#define 	AT91C_HSMC4_HOSTCS_5                    ((unsigned int) 0x5 << 12) // (HSMC4) CS5
#define 	AT91C_HSMC4_HOSTCS_6                    ((unsigned int) 0x6 << 12) // (HSMC4) CS6
#define 	AT91C_HSMC4_HOSTCS_7                    ((unsigned int) 0x7 << 12) // (HSMC4) CS7
#define AT91C_HSMC4_XFRDONE   ((unsigned int) 0x1 << 16) // (HSMC4) Host Data Transfer Terminated
#define AT91C_HSMC4_CMDDONE   ((unsigned int) 0x1 << 17) // (HSMC4) Command Done
#define AT91C_HSMC4_ECCRDY    ((unsigned int) 0x1 << 18) // (HSMC4) ECC ready
#define AT91C_HSMC4_DTOE      ((unsigned int) 0x1 << 20) // (HSMC4) Data timeout Error
#define AT91C_HSMC4_UNDEF     ((unsigned int) 0x1 << 21) // (HSMC4) Undefined Area Error
#define AT91C_HSMC4_AWB       ((unsigned int) 0x1 << 22) // (HSMC4) Accessing While Busy Error
#define AT91C_HSMC4_HASE      ((unsigned int) 0x1 << 23) // (HSMC4) Host Controller Access Size Error
#define AT91C_HSMC4_RBEDGE0   ((unsigned int) 0x1 << 24) // (HSMC4) Ready Busy line 0 Edge detected
#define AT91C_HSMC4_RBEDGE1   ((unsigned int) 0x1 << 25) // (HSMC4) Ready Busy line 1 Edge detected
#define AT91C_HSMC4_RBEDGE2   ((unsigned int) 0x1 << 26) // (HSMC4) Ready Busy line 2 Edge detected
#define AT91C_HSMC4_RBEDGE3   ((unsigned int) 0x1 << 27) // (HSMC4) Ready Busy line 3 Edge detected
#define AT91C_HSMC4_RBEDGE4   ((unsigned int) 0x1 << 28) // (HSMC4) Ready Busy line 4 Edge detected
#define AT91C_HSMC4_RBEDGE5   ((unsigned int) 0x1 << 29) // (HSMC4) Ready Busy line 5 Edge detected
#define AT91C_HSMC4_RBEDGE6   ((unsigned int) 0x1 << 30) // (HSMC4) Ready Busy line 6 Edge detected
#define AT91C_HSMC4_RBEDGE7   ((unsigned int) 0x1 << 31) // (HSMC4) Ready Busy line 7 Edge detected
// -------- HSMC4_IER : (HSMC4 Offset: 0xc) HSMC4 Interrupt Enable Register -------- 
// -------- HSMC4_IDR : (HSMC4 Offset: 0x10) HSMC4 Interrupt Disable Register -------- 
// -------- HSMC4_IMR : (HSMC4 Offset: 0x14) HSMC4 Interrupt Mask Register -------- 
// -------- HSMC4_ADDR : (HSMC4 Offset: 0x18) Address Cycle Zero Register -------- 
#define AT91C_HSMC4_ADDRCYCLE0 ((unsigned int) 0xFF <<  0) // (HSMC4) Nand Flash Array Address cycle 0
// -------- HSMC4_BANK : (HSMC4 Offset: 0x1c) Bank Register -------- 
#define AT91C_BANK            ((unsigned int) 0x7 <<  0) // (HSMC4) Bank identifier
#define 	AT91C_BANK_0                    ((unsigned int) 0x0) // (HSMC4) BANK0
#define 	AT91C_BANK_1                    ((unsigned int) 0x1) // (HSMC4) BANK1
#define 	AT91C_BANK_2                    ((unsigned int) 0x2) // (HSMC4) BANK2
#define 	AT91C_BANK_3                    ((unsigned int) 0x3) // (HSMC4) BANK3
#define 	AT91C_BANK_4                    ((unsigned int) 0x4) // (HSMC4) BANK4
#define 	AT91C_BANK_5                    ((unsigned int) 0x5) // (HSMC4) BANK5
#define 	AT91C_BANK_6                    ((unsigned int) 0x6) // (HSMC4) BANK6
#define 	AT91C_BANK_7                    ((unsigned int) 0x7) // (HSMC4) BANK7
// -------- HSMC4_ECCCR : (HSMC4 Offset: 0x20) ECC Control Register -------- 
#define AT91C_HSMC4_ECCRESET  ((unsigned int) 0x1 <<  0) // (HSMC4) Reset ECC
// -------- HSMC4_ECCCMD : (HSMC4 Offset: 0x24) ECC mode register -------- 
#define AT91C_ECC_PAGE_SIZE   ((unsigned int) 0x3 <<  0) // (HSMC4) Nand Flash page size
#define 	AT91C_ECC_PAGE_SIZE_528_Bytes            ((unsigned int) 0x0) // (HSMC4) 512 bytes plus 16 bytes page size
#define 	AT91C_ECC_PAGE_SIZE_1056_Bytes           ((unsigned int) 0x1) // (HSMC4) 1024 bytes plus 32 bytes page size
#define 	AT91C_ECC_PAGE_SIZE_2112_Bytes           ((unsigned int) 0x2) // (HSMC4) 2048 bytes plus 64 bytes page size
#define 	AT91C_ECC_PAGE_SIZE_4224_Bytes           ((unsigned int) 0x3) // (HSMC4) 4096 bytes plus 128 bytes page size
#define AT91C_ECC_TYPCORRECT  ((unsigned int) 0x3 <<  4) // (HSMC4) Nand Flash page size
#define 	AT91C_ECC_TYPCORRECT_ONE_PER_PAGE         ((unsigned int) 0x0 <<  4) // (HSMC4) 
#define 	AT91C_ECC_TYPCORRECT_ONE_EVERY_256_BYTES  ((unsigned int) 0x1 <<  4) // (HSMC4) 
#define 	AT91C_ECC_TYPCORRECT_ONE_EVERY_512_BYTES  ((unsigned int) 0x2 <<  4) // (HSMC4) 
// -------- HSMC4_ECCSR1 : (HSMC4 Offset: 0x28) ECC Status Register 1 -------- 
#define AT91C_HSMC4_ECC_RECERR0 ((unsigned int) 0x1 <<  0) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR0 ((unsigned int) 0x1 <<  1) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR0 ((unsigned int) 0x1 <<  2) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR1 ((unsigned int) 0x1 <<  4) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR1 ((unsigned int) 0x1 <<  5) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR1 ((unsigned int) 0x1 <<  6) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR2 ((unsigned int) 0x1 <<  8) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR2 ((unsigned int) 0x1 <<  9) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR2 ((unsigned int) 0x1 << 10) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR3 ((unsigned int) 0x1 << 12) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR3 ((unsigned int) 0x1 << 13) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR3 ((unsigned int) 0x1 << 14) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR4 ((unsigned int) 0x1 << 16) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR4 ((unsigned int) 0x1 << 17) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR4 ((unsigned int) 0x1 << 18) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR5 ((unsigned int) 0x1 << 20) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR5 ((unsigned int) 0x1 << 21) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR5 ((unsigned int) 0x1 << 22) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR6 ((unsigned int) 0x1 << 24) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR6 ((unsigned int) 0x1 << 25) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR6 ((unsigned int) 0x1 << 26) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR7 ((unsigned int) 0x1 << 28) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR7 ((unsigned int) 0x1 << 29) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR7 ((unsigned int) 0x1 << 30) // (HSMC4) Multiple Error
// -------- HSMC4_ECCPR0 : (HSMC4 Offset: 0x2c) HSMC4 ECC parity Register 0 -------- 
#define AT91C_HSMC4_ECC_BITADDR ((unsigned int) 0x7 <<  0) // (HSMC4) Corrupted Bit Address in the page
#define AT91C_HSMC4_ECC_WORDADDR ((unsigned int) 0xFF <<  3) // (HSMC4) Corrupted Word Address in the page
#define AT91C_HSMC4_ECC_NPARITY ((unsigned int) 0x7FF << 12) // (HSMC4) Parity N
// -------- HSMC4_ECCPR1 : (HSMC4 Offset: 0x30) HSMC4 ECC parity Register 1 -------- 
// -------- HSMC4_ECCSR2 : (HSMC4 Offset: 0x34) ECC Status Register 2 -------- 
#define AT91C_HSMC4_ECC_RECERR8 ((unsigned int) 0x1 <<  0) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR8 ((unsigned int) 0x1 <<  1) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR8 ((unsigned int) 0x1 <<  2) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR9 ((unsigned int) 0x1 <<  4) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR9 ((unsigned int) 0x1 <<  5) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR9 ((unsigned int) 0x1 <<  6) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR10 ((unsigned int) 0x1 <<  8) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR10 ((unsigned int) 0x1 <<  9) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR10 ((unsigned int) 0x1 << 10) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR11 ((unsigned int) 0x1 << 12) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR11 ((unsigned int) 0x1 << 13) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR11 ((unsigned int) 0x1 << 14) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR12 ((unsigned int) 0x1 << 16) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR12 ((unsigned int) 0x1 << 17) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR12 ((unsigned int) 0x1 << 18) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR13 ((unsigned int) 0x1 << 20) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR13 ((unsigned int) 0x1 << 21) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR13 ((unsigned int) 0x1 << 22) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR14 ((unsigned int) 0x1 << 24) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR14 ((unsigned int) 0x1 << 25) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR14 ((unsigned int) 0x1 << 26) // (HSMC4) Multiple Error
#define AT91C_HSMC4_ECC_RECERR15 ((unsigned int) 0x1 << 28) // (HSMC4) Recoverable Error
#define AT91C_HSMC4_ECC_ECCERR15 ((unsigned int) 0x1 << 29) // (HSMC4) ECC Error
#define AT91C_HSMC4_ECC_MULERR15 ((unsigned int) 0x1 << 30) // (HSMC4) Multiple Error
// -------- HSMC4_ECCPR2 : (HSMC4 Offset: 0x38) HSMC4 ECC parity Register 2 -------- 
// -------- HSMC4_ECCPR3 : (HSMC4 Offset: 0x3c) HSMC4 ECC parity Register 3 -------- 
// -------- HSMC4_ECCPR4 : (HSMC4 Offset: 0x40) HSMC4 ECC parity Register 4 -------- 
// -------- HSMC4_ECCPR5 : (HSMC4 Offset: 0x44) HSMC4 ECC parity Register 5 -------- 
// -------- HSMC4_ECCPR6 : (HSMC4 Offset: 0x48) HSMC4 ECC parity Register 6 -------- 
// -------- HSMC4_ECCPR7 : (HSMC4 Offset: 0x4c) HSMC4 ECC parity Register 7 -------- 
// -------- HSMC4_ECCPR8 : (HSMC4 Offset: 0x50) HSMC4 ECC parity Register 8 -------- 
// -------- HSMC4_ECCPR9 : (HSMC4 Offset: 0x54) HSMC4 ECC parity Register 9 -------- 
// -------- HSMC4_ECCPR10 : (HSMC4 Offset: 0x58) HSMC4 ECC parity Register 10 -------- 
// -------- HSMC4_ECCPR11 : (HSMC4 Offset: 0x5c) HSMC4 ECC parity Register 11 -------- 
// -------- HSMC4_ECCPR12 : (HSMC4 Offset: 0x60) HSMC4 ECC parity Register 12 -------- 
// -------- HSMC4_ECCPR13 : (HSMC4 Offset: 0x64) HSMC4 ECC parity Register 13 -------- 
// -------- HSMC4_ECCPR14 : (HSMC4 Offset: 0x68) HSMC4 ECC parity Register 14 -------- 
// -------- HSMC4_ECCPR15 : (HSMC4 Offset: 0x6c) HSMC4 ECC parity Register 15 -------- 
// -------- HSMC4_OCMS : (HSMC4 Offset: 0x110) HSMC4 OCMS Register -------- 
#define AT91C_HSMC4_OCMS_SRSE ((unsigned int) 0x1 <<  0) // (HSMC4) Static Memory Controller Scrambling Enable
#define AT91C_HSMC4_OCMS_SMSE ((unsigned int) 0x1 <<  1) // (HSMC4) SRAM Scramling Enable
// -------- HSMC4_KEY1 : (HSMC4 Offset: 0x114) HSMC4 OCMS KEY1 Register -------- 
#define AT91C_HSMC4_OCMS_KEY1 ((unsigned int) 0x0 <<  0) // (HSMC4) OCMS Key 2
// -------- HSMC4_OCMS_KEY2 : (HSMC4 Offset: 0x118) HSMC4 OCMS KEY2 Register -------- 
#define AT91C_HSMC4_OCMS_KEY2 ((unsigned int) 0x0 <<  0) // (HSMC4) OCMS Key 2
// -------- HSMC4_WPCR : (HSMC4 Offset: 0x1e4) HSMC4 Witre Protection Control Register -------- 
#define AT91C_HSMC4_WP_EN     ((unsigned int) 0x1 <<  0) // (HSMC4) Write Protection Enable
#define AT91C_HSMC4_WP_KEY    ((unsigned int) 0xFFFFFF <<  8) // (HSMC4) Protection Password
// -------- HSMC4_WPSR : (HSMC4 Offset: 0x1e8) HSMC4 WPSR Register -------- 
#define AT91C_HSMC4_WP_VS     ((unsigned int) 0xF <<  0) // (HSMC4) Write Protection Violation Status
#define 	AT91C_HSMC4_WP_VS_WP_VS0               ((unsigned int) 0x0) // (HSMC4) No write protection violation since the last read of this register
#define 	AT91C_HSMC4_WP_VS_WP_VS1               ((unsigned int) 0x1) // (HSMC4) write protection detected unauthorized attempt to write a control register had occured (since the last read)
#define 	AT91C_HSMC4_WP_VS_WP_VS2               ((unsigned int) 0x2) // (HSMC4) Software reset had been performed while write protection was enabled (since the last read)
#define 	AT91C_HSMC4_WP_VS_WP_VS3               ((unsigned int) 0x3) // (HSMC4) Both write protection violation and software reset with write protection enabled had occured since the last read
#define AT91C_                ((unsigned int) 0x0 <<  8) // (HSMC4) 
// -------- HSMC4_VER : (HSMC4 Offset: 0x1fc) HSMC4 VERSION Register -------- 
// -------- HSMC4_DUMMY : (HSMC4 Offset: 0x200) HSMC4 DUMMY REGISTER -------- 
#define AT91C_HSMC4_CMD1      ((unsigned int) 0xFF <<  2) // (HSMC4) Command Register Value for Cycle 1
#define AT91C_HSMC4_CMD2      ((unsigned int) 0xFF << 10) // (HSMC4) Command Register Value for Cycle 2
#define AT91C_HSMC4_VCMD2     ((unsigned int) 0x1 << 18) // (HSMC4) Valid Cycle 2 Command
#define AT91C_HSMC4_ACYCLE    ((unsigned int) 0x7 << 19) // (HSMC4) Number of Address required for the current command
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_NONE    ((unsigned int) 0x0 << 19) // (HSMC4) No address cycle
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_ONE     ((unsigned int) 0x1 << 19) // (HSMC4) One address cycle
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_TWO     ((unsigned int) 0x2 << 19) // (HSMC4) Two address cycles
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_THREE   ((unsigned int) 0x3 << 19) // (HSMC4) Three address cycles
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_FOUR    ((unsigned int) 0x4 << 19) // (HSMC4) Four address cycles
#define 	AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_FIVE    ((unsigned int) 0x5 << 19) // (HSMC4) Five address cycles
#define AT91C_HSMC4_CSID      ((unsigned int) 0x7 << 22) // (HSMC4) Chip Select Identifier
#define 	AT91C_HSMC4_CSID_0                    ((unsigned int) 0x0 << 22) // (HSMC4) CS0
#define 	AT91C_HSMC4_CSID_1                    ((unsigned int) 0x1 << 22) // (HSMC4) CS1
#define 	AT91C_HSMC4_CSID_2                    ((unsigned int) 0x2 << 22) // (HSMC4) CS2
#define 	AT91C_HSMC4_CSID_3                    ((unsigned int) 0x3 << 22) // (HSMC4) CS3
#define 	AT91C_HSMC4_CSID_4                    ((unsigned int) 0x4 << 22) // (HSMC4) CS4
#define 	AT91C_HSMC4_CSID_5                    ((unsigned int) 0x5 << 22) // (HSMC4) CS5
#define 	AT91C_HSMC4_CSID_6                    ((unsigned int) 0x6 << 22) // (HSMC4) CS6
#define 	AT91C_HSMC4_CSID_7                    ((unsigned int) 0x7 << 22) // (HSMC4) CS7
#define AT91C_HSMC4_HOST_EN   ((unsigned int) 0x1 << 25) // (HSMC4) Host Main Controller Enable
#define AT91C_HSMC4_HOST_WR   ((unsigned int) 0x1 << 26) // (HSMC4) HOSTWR : Host Main Controller Write Enable
#define AT91C_HSMC4_HOSTCMD   ((unsigned int) 0x1 << 27) // (HSMC4) Host Command Enable

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR AHB Matrix2 Interface
// *****************************************************************************
typedef struct _AT91S_HMATRIX2 {
	AT91_REG	 HMATRIX2_MCFG0; 	//  Master Configuration Register 0 : ARM I and D
	AT91_REG	 HMATRIX2_MCFG1; 	//  Master Configuration Register 1 : ARM S
	AT91_REG	 HMATRIX2_MCFG2; 	//  Master Configuration Register 2
	AT91_REG	 HMATRIX2_MCFG3; 	//  Master Configuration Register 3
	AT91_REG	 HMATRIX2_MCFG4; 	//  Master Configuration Register 4
	AT91_REG	 HMATRIX2_MCFG5; 	//  Master Configuration Register 5
	AT91_REG	 HMATRIX2_MCFG6; 	//  Master Configuration Register 6
	AT91_REG	 HMATRIX2_MCFG7; 	//  Master Configuration Register 7
	AT91_REG	 Reserved0[8]; 	// 
	AT91_REG	 HMATRIX2_SCFG0; 	//  Slave Configuration Register 0
	AT91_REG	 HMATRIX2_SCFG1; 	//  Slave Configuration Register 1
	AT91_REG	 HMATRIX2_SCFG2; 	//  Slave Configuration Register 2
	AT91_REG	 HMATRIX2_SCFG3; 	//  Slave Configuration Register 3
	AT91_REG	 HMATRIX2_SCFG4; 	//  Slave Configuration Register 4
	AT91_REG	 HMATRIX2_SCFG5; 	//  Slave Configuration Register 5
	AT91_REG	 HMATRIX2_SCFG6; 	//  Slave Configuration Register 6
	AT91_REG	 HMATRIX2_SCFG7; 	//  Slave Configuration Register 5
	AT91_REG	 HMATRIX2_SCFG8; 	//  Slave Configuration Register 8
	AT91_REG	 HMATRIX2_SCFG9; 	//  Slave Configuration Register 9
	AT91_REG	 Reserved1[42]; 	// 
	AT91_REG	 HMATRIX2_SFR0 ; 	//  Special Function Register 0
	AT91_REG	 HMATRIX2_SFR1 ; 	//  Special Function Register 1
	AT91_REG	 HMATRIX2_SFR2 ; 	//  Special Function Register 2
	AT91_REG	 HMATRIX2_SFR3 ; 	//  Special Function Register 3
	AT91_REG	 HMATRIX2_SFR4 ; 	//  Special Function Register 4
	AT91_REG	 HMATRIX2_SFR5 ; 	//  Special Function Register 5
	AT91_REG	 HMATRIX2_SFR6 ; 	//  Special Function Register 6
	AT91_REG	 HMATRIX2_SFR7 ; 	//  Special Function Register 7
	AT91_REG	 HMATRIX2_SFR8 ; 	//  Special Function Register 8
	AT91_REG	 HMATRIX2_SFR9 ; 	//  Special Function Register 9
	AT91_REG	 HMATRIX2_SFR10; 	//  Special Function Register 10
	AT91_REG	 HMATRIX2_SFR11; 	//  Special Function Register 11
	AT91_REG	 HMATRIX2_SFR12; 	//  Special Function Register 12
	AT91_REG	 HMATRIX2_SFR13; 	//  Special Function Register 13
	AT91_REG	 HMATRIX2_SFR14; 	//  Special Function Register 14
	AT91_REG	 HMATRIX2_SFR15; 	//  Special Function Register 15
	AT91_REG	 Reserved2[39]; 	// 
	AT91_REG	 HMATRIX2_ADDRSIZE; 	// HMATRIX2 ADDRSIZE REGISTER 
	AT91_REG	 HMATRIX2_IPNAME1; 	// HMATRIX2 IPNAME1 REGISTER 
	AT91_REG	 HMATRIX2_IPNAME2; 	// HMATRIX2 IPNAME2 REGISTER 
	AT91_REG	 HMATRIX2_FEATURES; 	// HMATRIX2 FEATURES REGISTER 
	AT91_REG	 HMATRIX2_VER; 	// HMATRIX2 VERSION REGISTER 
} AT91S_HMATRIX2, *AT91PS_HMATRIX2;

// -------- MATRIX_MCFG0 : (HMATRIX2 Offset: 0x0) Master Configuration Register ARM bus I and D -------- 
#define AT91C_MATRIX_ULBT     ((unsigned int) 0x7 <<  0) // (HMATRIX2) Undefined Length Burst Type
#define 	AT91C_MATRIX_ULBT_INFINIT_LENGTH       ((unsigned int) 0x0) // (HMATRIX2) infinite length burst
#define 	AT91C_MATRIX_ULBT_SINGLE_ACCESS        ((unsigned int) 0x1) // (HMATRIX2) Single Access
#define 	AT91C_MATRIX_ULBT_4_BEAT               ((unsigned int) 0x2) // (HMATRIX2) 4 Beat Burst
#define 	AT91C_MATRIX_ULBT_8_BEAT               ((unsigned int) 0x3) // (HMATRIX2) 8 Beat Burst
#define 	AT91C_MATRIX_ULBT_16_BEAT              ((unsigned int) 0x4) // (HMATRIX2) 16 Beat Burst
#define 	AT91C_MATRIX_ULBT_32_BEAT              ((unsigned int) 0x5) // (HMATRIX2) 32 Beat Burst
#define 	AT91C_MATRIX_ULBT_64_BEAT              ((unsigned int) 0x6) // (HMATRIX2) 64 Beat Burst
#define 	AT91C_MATRIX_ULBT_128_BEAT             ((unsigned int) 0x7) // (HMATRIX2) 128 Beat Burst
// -------- MATRIX_MCFG1 : (HMATRIX2 Offset: 0x4) Master Configuration Register ARM bus S -------- 
// -------- MATRIX_MCFG2 : (HMATRIX2 Offset: 0x8) Master Configuration Register -------- 
// -------- MATRIX_MCFG3 : (HMATRIX2 Offset: 0xc) Master Configuration Register -------- 
// -------- MATRIX_MCFG4 : (HMATRIX2 Offset: 0x10) Master Configuration Register -------- 
// -------- MATRIX_MCFG5 : (HMATRIX2 Offset: 0x14) Master Configuration Register -------- 
// -------- MATRIX_MCFG6 : (HMATRIX2 Offset: 0x18) Master Configuration Register -------- 
// -------- MATRIX_MCFG7 : (HMATRIX2 Offset: 0x1c) Master Configuration Register -------- 
// -------- MATRIX_SCFG0 : (HMATRIX2 Offset: 0x40) Slave Configuration Register 0 -------- 
#define AT91C_MATRIX_SLOT_CYCLE ((unsigned int) 0xFF <<  0) // (HMATRIX2) Maximum Number of Allowed Cycles for a Burst
#define AT91C_MATRIX_DEFMSTR_TYPE ((unsigned int) 0x3 << 16) // (HMATRIX2) Default Master Type
#define 	AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR           ((unsigned int) 0x0 << 16) // (HMATRIX2) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.
#define 	AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR         ((unsigned int) 0x1 << 16) // (HMATRIX2) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.
#define 	AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR        ((unsigned int) 0x2 << 16) // (HMATRIX2) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG0 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG0_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG1 : (HMATRIX2 Offset: 0x44) Slave Configuration Register 1 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG1 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG1_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG2 : (HMATRIX2 Offset: 0x48) Slave Configuration Register 2 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG2 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG2_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG3 : (HMATRIX2 Offset: 0x4c) Slave Configuration Register 3 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG3 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG3_ARMC                 ((unsigned int) 0x0 << 18) // (HMATRIX2) ARMC is Default Master
// -------- MATRIX_SCFG4 : (HMATRIX2 Offset: 0x50) Slave Configuration Register 4 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG4 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG4_ARMC                 ((unsigned int) 0x0 << 18) // (HMATRIX2) ARMC is Default Master
// -------- MATRIX_SCFG5 : (HMATRIX2 Offset: 0x54) Slave Configuration Register 5 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG5 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG5_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG6 : (HMATRIX2 Offset: 0x58) Slave Configuration Register 6 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG6 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG6_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG7 : (HMATRIX2 Offset: 0x5c) Slave Configuration Register 7 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG7 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG7_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
// -------- MATRIX_SCFG8 : (HMATRIX2 Offset: 0x60) Slave Configuration Register 8 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG8 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG8_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG8_HDMA                 ((unsigned int) 0x4 << 18) // (HMATRIX2) HDMA is Default Master
// -------- MATRIX_SCFG9 : (HMATRIX2 Offset: 0x64) Slave Configuration Register 9 -------- 
#define AT91C_MATRIX_FIXED_DEFMSTR_SCFG9 ((unsigned int) 0x7 << 18) // (HMATRIX2) Fixed Index of Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG9_ARMS                 ((unsigned int) 0x1 << 18) // (HMATRIX2) ARMS is Default Master
#define 	AT91C_MATRIX_FIXED_DEFMSTR_SCFG9_HDMA                 ((unsigned int) 0x4 << 18) // (HMATRIX2) HDMA is Default Master
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x110) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x114) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x118) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x11c) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x120) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x124) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x128) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x12c) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x130) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x134) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x138) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x13c) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x140) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x144) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x148) Special Function Register 0 -------- 
// -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x14c) Special Function Register 0 -------- 
// -------- HMATRIX2_VER : (HMATRIX2 Offset: 0x1fc)  VERSION  Register -------- 
#define AT91C_HMATRIX2_VER    ((unsigned int) 0xF <<  0) // (HMATRIX2)  VERSION  Register

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR NESTED vector Interrupt Controller
// *****************************************************************************
typedef struct _AT91S_NVIC {
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 NVIC_ICTR; 	// Interrupt Control Type Register
	AT91_REG	 Reserved1[2]; 	// 
	AT91_REG	 NVIC_STICKCSR; 	// SysTick Control and Status Register
	AT91_REG	 NVIC_STICKRVR; 	// SysTick Reload Value Register
	AT91_REG	 NVIC_STICKCVR; 	// SysTick Current Value Register
	AT91_REG	 NVIC_STICKCALVR; 	// SysTick Calibration Value Register
	AT91_REG	 Reserved2[56]; 	// 
	AT91_REG	 NVIC_ISER[8]; 	// Set Enable Register
	AT91_REG	 Reserved3[24]; 	// 
	AT91_REG	 NVIC_ICER[8]; 	// Clear enable Register
	AT91_REG	 Reserved4[24]; 	// 
	AT91_REG	 NVIC_ISPR[8]; 	// Set Pending Register
	AT91_REG	 Reserved5[24]; 	// 
	AT91_REG	 NVIC_ICPR[8]; 	// Clear Pending Register
	AT91_REG	 Reserved6[24]; 	// 
	AT91_REG	 NVIC_ABR[8]; 	// Active Bit Register
	AT91_REG	 Reserved7[56]; 	// 
	AT91_REG	 NVIC_IPR[60]; 	// Interrupt Mask Register
	AT91_REG	 Reserved8[516]; 	// 
	AT91_REG	 NVIC_CPUID; 	// CPUID Base Register
	AT91_REG	 NVIC_ICSR; 	// Interrupt Control State Register
	AT91_REG	 NVIC_VTOFFR; 	// Vector Table Offset Register
	AT91_REG	 NVIC_AIRCR; 	// Application Interrupt/Reset Control Reg
	AT91_REG	 NVIC_SCR; 	// System Control Register
	AT91_REG	 NVIC_CCR; 	// Configuration Control Register
	AT91_REG	 NVIC_HAND4PR; 	// System Handlers 4-7 Priority Register
	AT91_REG	 NVIC_HAND8PR; 	// System Handlers 8-11 Priority Register
	AT91_REG	 NVIC_HAND12PR; 	// System Handlers 12-15 Priority Register
	AT91_REG	 NVIC_HANDCSR; 	// System Handler Control and State Register
	AT91_REG	 NVIC_CFSR; 	// Configurable Fault Status Register
	AT91_REG	 NVIC_HFSR; 	// Hard Fault Status Register
	AT91_REG	 NVIC_DFSR; 	// Debug Fault Status Register
	AT91_REG	 NVIC_MMAR; 	// Mem Manage Address Register
	AT91_REG	 NVIC_BFAR; 	// Bus Fault Address Register
	AT91_REG	 NVIC_AFSR; 	// Auxiliary Fault Status Register
	AT91_REG	 NVIC_PFR0; 	// Processor Feature register0
	AT91_REG	 NVIC_PFR1; 	// Processor Feature register1
	AT91_REG	 NVIC_DFR0; 	// Debug Feature register0
	AT91_REG	 NVIC_AFR0; 	// Auxiliary Feature register0
	AT91_REG	 NVIC_MMFR0; 	// Memory Model Feature register0
	AT91_REG	 NVIC_MMFR1; 	// Memory Model Feature register1
	AT91_REG	 NVIC_MMFR2; 	// Memory Model Feature register2
	AT91_REG	 NVIC_MMFR3; 	// Memory Model Feature register3
	AT91_REG	 NVIC_ISAR0; 	// ISA Feature register0
	AT91_REG	 NVIC_ISAR1; 	// ISA Feature register1
	AT91_REG	 NVIC_ISAR2; 	// ISA Feature register2
	AT91_REG	 NVIC_ISAR3; 	// ISA Feature register3
	AT91_REG	 NVIC_ISAR4; 	// ISA Feature register4
	AT91_REG	 Reserved9[99]; 	// 
	AT91_REG	 NVIC_STIR; 	// Software Trigger Interrupt Register
	AT91_REG	 Reserved10[51]; 	// 
	AT91_REG	 NVIC_PID4; 	// Peripheral identification register
	AT91_REG	 NVIC_PID5; 	// Peripheral identification register
	AT91_REG	 NVIC_PID6; 	// Peripheral identification register
	AT91_REG	 NVIC_PID7; 	// Peripheral identification register
	AT91_REG	 NVIC_PID0; 	// Peripheral identification register b7:0
	AT91_REG	 NVIC_PID1; 	// Peripheral identification register b15:8
	AT91_REG	 NVIC_PID2; 	// Peripheral identification register b23:16
	AT91_REG	 NVIC_PID3; 	// Peripheral identification register b31:24
	AT91_REG	 NVIC_CID0; 	// Component identification register b7:0
	AT91_REG	 NVIC_CID1; 	// Component identification register b15:8
	AT91_REG	 NVIC_CID2; 	// Component identification register b23:16
	AT91_REG	 NVIC_CID3; 	// Component identification register b31:24
} AT91S_NVIC, *AT91PS_NVIC;

// -------- NVIC_ICTR : (NVIC Offset: 0x4) Interrupt Controller Type Register -------- 
#define AT91C_NVIC_INTLINESNUM ((unsigned int) 0xF <<  0) // (NVIC) Total number of interrupt lines
#define 	AT91C_NVIC_INTLINESNUM_32                   ((unsigned int) 0x0) // (NVIC) up to 32 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_64                   ((unsigned int) 0x1) // (NVIC) up to 64 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_96                   ((unsigned int) 0x2) // (NVIC) up to 96 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_128                  ((unsigned int) 0x3) // (NVIC) up to 128 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_160                  ((unsigned int) 0x4) // (NVIC) up to 160 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_192                  ((unsigned int) 0x5) // (NVIC) up to 192 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_224                  ((unsigned int) 0x6) // (NVIC) up to 224 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_256                  ((unsigned int) 0x7) // (NVIC) up to 256 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_288                  ((unsigned int) 0x8) // (NVIC) up to 288 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_320                  ((unsigned int) 0x9) // (NVIC) up to 320 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_352                  ((unsigned int) 0xA) // (NVIC) up to 352 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_384                  ((unsigned int) 0xB) // (NVIC) up to 384 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_416                  ((unsigned int) 0xC) // (NVIC) up to 416 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_448                  ((unsigned int) 0xD) // (NVIC) up to 448 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_480                  ((unsigned int) 0xE) // (NVIC) up to 480 interrupt lines supported
#define 	AT91C_NVIC_INTLINESNUM_496                  ((unsigned int) 0xF) // (NVIC) up to 496 interrupt lines supported)
// -------- NVIC_STICKCSR : (NVIC Offset: 0x10) SysTick Control and Status Register -------- 
#define AT91C_NVIC_STICKENABLE ((unsigned int) 0x1 <<  0) // (NVIC) SysTick counter enable.
#define AT91C_NVIC_STICKINT   ((unsigned int) 0x1 <<  1) // (NVIC) SysTick interrupt enable.
#define AT91C_NVIC_STICKCLKSOURCE ((unsigned int) 0x1 <<  2) // (NVIC) Reference clock selection.
#define AT91C_NVIC_STICKCOUNTFLAG ((unsigned int) 0x1 << 16) // (NVIC) Return 1 if timer counted to 0 since last read.
// -------- NVIC_STICKRVR : (NVIC Offset: 0x14) SysTick Reload Value Register -------- 
#define AT91C_NVIC_STICKRELOAD ((unsigned int) 0xFFFFFF <<  0) // (NVIC) SysTick reload value.
// -------- NVIC_STICKCVR : (NVIC Offset: 0x18) SysTick Current Value Register -------- 
#define AT91C_NVIC_STICKCURRENT ((unsigned int) 0x7FFFFFFF <<  0) // (NVIC) SysTick current value.
// -------- NVIC_STICKCALVR : (NVIC Offset: 0x1c) SysTick Calibration Value Register -------- 
#define AT91C_NVIC_STICKTENMS ((unsigned int) 0xFFFFFF <<  0) // (NVIC) Reload value to use for 10ms timing.
#define AT91C_NVIC_STICKSKEW  ((unsigned int) 0x1 << 30) // (NVIC) Read as 1 if the calibration value is not exactly 10ms because of clock frequency.
#define AT91C_NVIC_STICKNOREF ((unsigned int) 0x1 << 31) // (NVIC) Read as 1 if the reference clock is not provided.
// -------- NVIC_IPR : (NVIC Offset: 0x400) Interrupt Priority Registers -------- 
#define AT91C_NVIC_PRI_N      ((unsigned int) 0xFF <<  0) // (NVIC) Priority of interrupt N (0, 4, 8, etc)
#define AT91C_NVIC_PRI_N1     ((unsigned int) 0xFF <<  8) // (NVIC) Priority of interrupt N+1 (1, 5, 9, etc)
#define AT91C_NVIC_PRI_N2     ((unsigned int) 0xFF << 16) // (NVIC) Priority of interrupt N+2 (2, 6, 10, etc)
#define AT91C_NVIC_PRI_N3     ((unsigned int) 0xFF << 24) // (NVIC) Priority of interrupt N+3 (3, 7, 11, etc)
// -------- NVIC_CPUID : (NVIC Offset: 0xd00) CPU ID Base Register -------- 
#define AT91C_NVIC_REVISION   ((unsigned int) 0xF <<  0) // (NVIC) Implementation defined revision number.
#define AT91C_NVIC_PARTNO     ((unsigned int) 0xFFF <<  4) // (NVIC) Number of processor within family
#define AT91C_NVIC_CONSTANT   ((unsigned int) 0xF << 16) // (NVIC) Reads as 0xF
#define AT91C_NVIC_VARIANT    ((unsigned int) 0xF << 20) // (NVIC) Implementation defined variant number.
#define AT91C_NVIC_IMPLEMENTER ((unsigned int) 0xFF << 24) // (NVIC) Implementer code. ARM is 0x41
// -------- NVIC_ICSR : (NVIC Offset: 0xd04) Interrupt Control State Register -------- 
#define AT91C_NVIC_VECTACTIVE ((unsigned int) 0x1FF <<  0) // (NVIC) Read-only Active ISR number field
#define AT91C_NVIC_RETTOBASE  ((unsigned int) 0x1 << 11) // (NVIC) Read-only
#define AT91C_NVIC_VECTPENDING ((unsigned int) 0x1FF << 12) // (NVIC) Read-only Pending ISR number field
#define AT91C_NVIC_ISRPENDING ((unsigned int) 0x1 << 22) // (NVIC) Read-only Interrupt pending flag.
#define AT91C_NVIC_ISRPREEMPT ((unsigned int) 0x1 << 23) // (NVIC) Read-only You must only use this at debug time
#define AT91C_NVIC_PENDSTCLR  ((unsigned int) 0x1 << 25) // (NVIC) Write-only Clear pending SysTick bit
#define AT91C_NVIC_PENDSTSET  ((unsigned int) 0x1 << 26) // (NVIC) Read/write Set a pending SysTick bit
#define AT91C_NVIC_PENDSVCLR  ((unsigned int) 0x1 << 27) // (NVIC) Write-only Clear pending pendSV bit
#define AT91C_NVIC_PENDSVSET  ((unsigned int) 0x1 << 28) // (NVIC) Read/write Set pending pendSV bit
#define AT91C_NVIC_NMIPENDSET ((unsigned int) 0x1 << 31) // (NVIC) Read/write Set pending NMI
// -------- NVIC_VTOFFR : (NVIC Offset: 0xd08) Vector Table Offset Register -------- 
#define AT91C_NVIC_TBLOFF     ((unsigned int) 0x3FFFFF <<  7) // (NVIC) Vector table base offset field
#define AT91C_NVIC_TBLBASE    ((unsigned int) 0x1 << 29) // (NVIC) Table base is in Code (0) or RAM (1)
#define 	AT91C_NVIC_TBLBASE_CODE                 ((unsigned int) 0x0 << 29) // (NVIC) Table base is in CODE
#define 	AT91C_NVIC_TBLBASE_RAM                  ((unsigned int) 0x1 << 29) // (NVIC) Table base is in RAM
// -------- NVIC_AIRCR : (NVIC Offset: 0xd0c) Application Interrupt and Reset Control Register -------- 
#define AT91C_NVIC_VECTRESET  ((unsigned int) 0x1 <<  0) // (NVIC) System Reset bit
#define AT91C_NVIC_VECTCLRACTIVE ((unsigned int) 0x1 <<  1) // (NVIC) Clear active vector bit
#define AT91C_NVIC_SYSRESETREQ ((unsigned int) 0x1 <<  2) // (NVIC) Causes a signal to be asserted to the outer system that indicates a reset is requested
#define AT91C_NVIC_PRIGROUP   ((unsigned int) 0x7 <<  8) // (NVIC) Interrupt priority grouping field
#define 	AT91C_NVIC_PRIGROUP_3                    ((unsigned int) 0x3 <<  8) // (NVIC) indicates four bits of pre-emption priority, none bit of subpriority
#define 	AT91C_NVIC_PRIGROUP_4                    ((unsigned int) 0x4 <<  8) // (NVIC) indicates three bits of pre-emption priority, one bit of subpriority
#define 	AT91C_NVIC_PRIGROUP_5                    ((unsigned int) 0x5 <<  8) // (NVIC) indicates two bits of pre-emption priority, two bits of subpriority
#define 	AT91C_NVIC_PRIGROUP_6                    ((unsigned int) 0x6 <<  8) // (NVIC) indicates one bit of pre-emption priority, three bits of subpriority
#define 	AT91C_NVIC_PRIGROUP_7                    ((unsigned int) 0x7 <<  8) // (NVIC) indicates no pre-emption priority, four bits of subpriority
#define AT91C_NVIC_ENDIANESS  ((unsigned int) 0x1 << 15) // (NVIC) Data endianness bit
#define AT91C_NVIC_VECTKEY    ((unsigned int) 0xFFFF << 16) // (NVIC) Register key
// -------- NVIC_SCR : (NVIC Offset: 0xd10) System Control Register -------- 
#define AT91C_NVIC_SLEEPONEXIT ((unsigned int) 0x1 <<  1) // (NVIC) Sleep on exit when returning from Handler mode to Thread mode
#define AT91C_NVIC_SLEEPDEEP  ((unsigned int) 0x1 <<  2) // (NVIC) Sleep deep bit
#define AT91C_NVIC_SEVONPEND  ((unsigned int) 0x1 <<  4) // (NVIC) When enabled, this causes WFE to wake up when an interrupt moves from inactive to pended
// -------- NVIC_CCR : (NVIC Offset: 0xd14) Configuration Control Register -------- 
#define AT91C_NVIC_NONEBASETHRDENA ((unsigned int) 0x1 <<  0) // (NVIC) When 0, default, It is only possible to enter Thread mode when returning from the last exception
#define AT91C_NVIC_USERSETMPEND ((unsigned int) 0x1 <<  1) // (NVIC) 
#define AT91C_NVIC_UNALIGN_TRP ((unsigned int) 0x1 <<  3) // (NVIC) Trap for unaligned access
#define AT91C_NVIC_DIV_0_TRP  ((unsigned int) 0x1 <<  4) // (NVIC) Trap on Divide by 0
#define AT91C_NVIC_BFHFNMIGN  ((unsigned int) 0x1 <<  8) // (NVIC) 
#define AT91C_NVIC_STKALIGN   ((unsigned int) 0x1 <<  9) // (NVIC) 
// -------- NVIC_HAND4PR : (NVIC Offset: 0xd18) System Handlers 4-7 Priority Register -------- 
#define AT91C_NVIC_PRI_4      ((unsigned int) 0xFF <<  0) // (NVIC) 
#define AT91C_NVIC_PRI_5      ((unsigned int) 0xFF <<  8) // (NVIC) 
#define AT91C_NVIC_PRI_6      ((unsigned int) 0xFF << 16) // (NVIC) 
#define AT91C_NVIC_PRI_7      ((unsigned int) 0xFF << 24) // (NVIC) 
// -------- NVIC_HAND8PR : (NVIC Offset: 0xd1c) System Handlers 8-11 Priority Register -------- 
#define AT91C_NVIC_PRI_8      ((unsigned int) 0xFF <<  0) // (NVIC) 
#define AT91C_NVIC_PRI_9      ((unsigned int) 0xFF <<  8) // (NVIC) 
#define AT91C_NVIC_PRI_10     ((unsigned int) 0xFF << 16) // (NVIC) 
#define AT91C_NVIC_PRI_11     ((unsigned int) 0xFF << 24) // (NVIC) 
// -------- NVIC_HAND12PR : (NVIC Offset: 0xd20) System Handlers 12-15 Priority Register -------- 
#define AT91C_NVIC_PRI_12     ((unsigned int) 0xFF <<  0) // (NVIC) 
#define AT91C_NVIC_PRI_13     ((unsigned int) 0xFF <<  8) // (NVIC) 
#define AT91C_NVIC_PRI_14     ((unsigned int) 0xFF << 16) // (NVIC) 
#define AT91C_NVIC_PRI_15     ((unsigned int) 0xFF << 24) // (NVIC) 
// -------- NVIC_HANDCSR : (NVIC Offset: 0xd24) System Handler Control and State Register -------- 
#define AT91C_NVIC_MEMFAULTACT ((unsigned int) 0x1 <<  0) // (NVIC) 
#define AT91C_NVIC_BUSFAULTACT ((unsigned int) 0x1 <<  1) // (NVIC) 
#define AT91C_NVIC_USGFAULTACT ((unsigned int) 0x1 <<  3) // (NVIC) 
#define AT91C_NVIC_SVCALLACT  ((unsigned int) 0x1 <<  7) // (NVIC) 
#define AT91C_NVIC_MONITORACT ((unsigned int) 0x1 <<  8) // (NVIC) 
#define AT91C_NVIC_PENDSVACT  ((unsigned int) 0x1 << 10) // (NVIC) 
#define AT91C_NVIC_SYSTICKACT ((unsigned int) 0x1 << 11) // (NVIC) 
#define AT91C_NVIC_USGFAULTPENDED ((unsigned int) 0x1 << 12) // (NVIC) 
#define AT91C_NVIC_MEMFAULTPENDED ((unsigned int) 0x1 << 13) // (NVIC) 
#define AT91C_NVIC_BUSFAULTPENDED ((unsigned int) 0x1 << 14) // (NVIC) 
#define AT91C_NVIC_SVCALLPENDED ((unsigned int) 0x1 << 15) // (NVIC) 
#define AT91C_NVIC_MEMFAULTENA ((unsigned int) 0x1 << 16) // (NVIC) 
#define AT91C_NVIC_BUSFAULTENA ((unsigned int) 0x1 << 17) // (NVIC) 
#define AT91C_NVIC_USGFAULTENA ((unsigned int) 0x1 << 18) // (NVIC) 
// -------- NVIC_CFSR : (NVIC Offset: 0xd28) Configurable Fault Status Registers -------- 
#define AT91C_NVIC_MEMMANAGE  ((unsigned int) 0xFF <<  0) // (NVIC) 
#define AT91C_NVIC_BUSFAULT   ((unsigned int) 0xFF <<  8) // (NVIC) 
#define AT91C_NVIC_USAGEFAULT ((unsigned int) 0xFF << 16) // (NVIC) 
// -------- NVIC_BFAR : (NVIC Offset: 0xd38) Bus Fault Address Register -------- 
#define AT91C_NVIC_IBUSERR    ((unsigned int) 0x1 <<  0) // (NVIC) This bit indicates a bus fault on an instruction prefetch
#define AT91C_NVIC_PRECISERR  ((unsigned int) 0x1 <<  1) // (NVIC) Precise data access error. The BFAR is written with the faulting address
#define AT91C_NVIC_IMPRECISERR ((unsigned int) 0x1 <<  2) // (NVIC) Imprecise data access error
#define AT91C_NVIC_UNSTKERR   ((unsigned int) 0x1 <<  3) // (NVIC) This bit indicates a derived bus fault has occurred on exception return
#define AT91C_NVIC_STKERR     ((unsigned int) 0x1 <<  4) // (NVIC) This bit indicates a derived bus fault has occurred on exception entry
#define AT91C_NVIC_BFARVALID  ((unsigned int) 0x1 <<  7) // (NVIC) This bit is set if the BFAR register has valid contents
// -------- NVIC_PFR0 : (NVIC Offset: 0xd40) Processor Feature register0 (ID_PFR0) -------- 
#define AT91C_NVIC_ID_PFR0_0  ((unsigned int) 0xF <<  0) // (NVIC) State0 (T-bit == 0)
#define AT91C_NVIC_ID_PRF0_1  ((unsigned int) 0xF <<  4) // (NVIC) State1 (T-bit == 1)
// -------- NVIC_PFR1 : (NVIC Offset: 0xd44) Processor Feature register1 (ID_PFR1) -------- 
#define AT91C_NVIC_ID_PRF1_MODEL ((unsigned int) 0xF <<  8) // (NVIC) Microcontroller programmer’s model
// -------- NVIC_DFR0 : (NVIC Offset: 0xd48) Debug Feature register0 (ID_DFR0) -------- 
#define AT91C_NVIC_ID_DFR0_MODEL ((unsigned int) 0xF << 20) // (NVIC) Microcontroller Debug Model – memory mapped
// -------- NVIC_MMFR0 : (NVIC Offset: 0xd50) Memory Model Feature register0 (ID_MMFR0) -------- 
#define AT91C_NVIC_ID_MMFR0_PMSA ((unsigned int) 0xF <<  4) // (NVIC) Microcontroller Debug Model – memory mapped
#define AT91C_NVIC_ID_MMFR0_CACHE ((unsigned int) 0xF <<  8) // (NVIC) Microcontroller Debug Model – memory mapped

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR NESTED vector Interrupt Controller
// *****************************************************************************
typedef struct _AT91S_MPU {
	AT91_REG	 MPU_TYPE; 	// MPU Type Register
	AT91_REG	 MPU_CTRL; 	// MPU Control Register
	AT91_REG	 MPU_REG_NB; 	// MPU Region Number Register
	AT91_REG	 MPU_REG_BASE_ADDR; 	// MPU Region Base Address Register
	AT91_REG	 MPU_ATTR_SIZE; 	// MPU  Attribute and Size Register
	AT91_REG	 MPU_REG_BASE_ADDR1; 	// MPU Region Base Address Register alias 1
	AT91_REG	 MPU_ATTR_SIZE1; 	// MPU  Attribute and Size Register alias 1
	AT91_REG	 MPU_REG_BASE_ADDR2; 	// MPU Region Base Address Register alias 2
	AT91_REG	 MPU_ATTR_SIZE2; 	// MPU  Attribute and Size Register alias 2
	AT91_REG	 MPU_REG_BASE_ADDR3; 	// MPU Region Base Address Register alias 3
	AT91_REG	 MPU_ATTR_SIZE3; 	// MPU  Attribute and Size Register alias 3
} AT91S_MPU, *AT91PS_MPU;

// -------- MPU_TYPE : (MPU Offset: 0x0)  -------- 
#define AT91C_MPU_SEPARATE    ((unsigned int) 0x1 <<  0) // (MPU) 
#define AT91C_MPU_DREGION     ((unsigned int) 0xFF <<  8) // (MPU) 
#define AT91C_MPU_IREGION     ((unsigned int) 0xFF << 16) // (MPU) 
// -------- MPU_CTRL : (MPU Offset: 0x4)  -------- 
#define AT91C_MPU_ENABLE      ((unsigned int) 0x1 <<  0) // (MPU) 
#define AT91C_MPU_HFNMIENA    ((unsigned int) 0x1 <<  1) // (MPU) 
#define AT91C_MPU_PRIVDEFENA  ((unsigned int) 0x1 <<  2) // (MPU) 
// -------- MPU_REG_NB : (MPU Offset: 0x8)  -------- 
#define AT91C_MPU_REGION      ((unsigned int) 0xFF <<  0) // (MPU) 
// -------- MPU_REG_BASE_ADDR : (MPU Offset: 0xc)  -------- 
#define AT91C_MPU_REG         ((unsigned int) 0xF <<  0) // (MPU) 
#define AT91C_MPU_VALID       ((unsigned int) 0x1 <<  4) // (MPU) 
#define AT91C_MPU_ADDR        ((unsigned int) 0x3FFFFFF <<  5) // (MPU) 
// -------- MPU_ATTR_SIZE : (MPU Offset: 0x10)  -------- 
#define AT91C_MPU_ENA         ((unsigned int) 0x1 <<  0) // (MPU) 
#define AT91C_MPU_SIZE        ((unsigned int) 0xF <<  1) // (MPU) 
#define AT91C_MPU_SRD         ((unsigned int) 0xFF <<  8) // (MPU) 
#define AT91C_MPU_B           ((unsigned int) 0x1 << 16) // (MPU) 
#define AT91C_MPU_C           ((unsigned int) 0x1 << 17) // (MPU) 
#define AT91C_MPU_S           ((unsigned int) 0x1 << 18) // (MPU) 
#define AT91C_MPU_TEX         ((unsigned int) 0x7 << 19) // (MPU) 
#define AT91C_MPU_AP          ((unsigned int) 0x7 << 24) // (MPU) 
#define AT91C_MPU_XN          ((unsigned int) 0x7 << 28) // (MPU) 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR CORTEX_M3 Registers
// *****************************************************************************
typedef struct _AT91S_CM3 {
	AT91_REG	 CM3_CPUID; 	// CPU ID Base Register
	AT91_REG	 CM3_ICSR; 	// Interrupt Control State Register
	AT91_REG	 CM3_VTOR; 	// Vector Table Offset Register
	AT91_REG	 CM3_AIRCR; 	// Application Interrupt and Reset Control Register
	AT91_REG	 CM3_SCR; 	// System Controller Register
	AT91_REG	 CM3_CCR; 	// Configuration Control Register
	AT91_REG	 CM3_SHPR[3]; 	// System Handler Priority Register
	AT91_REG	 CM3_SHCSR; 	// System Handler Control and State Register
} AT91S_CM3, *AT91PS_CM3;

// -------- CM3_CPUID : (CM3 Offset: 0x0)  -------- 
// -------- CM3_AIRCR : (CM3 Offset: 0xc)  -------- 
#define AT91C_CM3_SYSRESETREQ ((unsigned int) 0x1 <<  2) // (CM3) A reset is requested by the processor.
// -------- CM3_SCR : (CM3 Offset: 0x10)  -------- 
#define AT91C_CM3_SLEEPONEXIT ((unsigned int) 0x1 <<  1) // (CM3) Sleep on exit when returning from Handler mode to Thread mode. Enables interrupt driven applications to avoid returning to empty main application.
#define AT91C_CM3_SLEEPDEEP   ((unsigned int) 0x1 <<  2) // (CM3) Sleep deep bit.
#define AT91C_CM3_SEVONPEND   ((unsigned int) 0x1 <<  4) // (CM3) When enabled, this causes WFE to wake up when an interrupt moves from inactive to pended.
// -------- CM3_SHCSR : (CM3 Offset: 0x24)  -------- 
#define AT91C_CM3_SYSTICKACT  ((unsigned int) 0x1 << 11) // (CM3) Reads as 1 if SysTick is active.

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller
// *****************************************************************************
typedef struct _AT91S_PDC {
	AT91_REG	 PDC_RPR; 	// Receive Pointer Register
	AT91_REG	 PDC_RCR; 	// Receive Counter Register
	AT91_REG	 PDC_TPR; 	// Transmit Pointer Register
	AT91_REG	 PDC_TCR; 	// Transmit Counter Register
	AT91_REG	 PDC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PDC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PDC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PDC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PDC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PDC_PTSR; 	// PDC Transfer Status Register
} AT91S_PDC, *AT91PS_PDC;

// -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- 
#define AT91C_PDC_RXTEN       ((unsigned int) 0x1 <<  0) // (PDC) Receiver Transfer Enable
#define AT91C_PDC_RXTDIS      ((unsigned int) 0x1 <<  1) // (PDC) Receiver Transfer Disable
#define AT91C_PDC_TXTEN       ((unsigned int) 0x1 <<  8) // (PDC) Transmitter Transfer Enable
#define AT91C_PDC_TXTDIS      ((unsigned int) 0x1 <<  9) // (PDC) Transmitter Transfer Disable
// -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Debug Unit
// *****************************************************************************
typedef struct _AT91S_DBGU {
	AT91_REG	 DBGU_CR; 	// Control Register
	AT91_REG	 DBGU_MR; 	// Mode Register
	AT91_REG	 DBGU_IER; 	// Interrupt Enable Register
	AT91_REG	 DBGU_IDR; 	// Interrupt Disable Register
	AT91_REG	 DBGU_IMR; 	// Interrupt Mask Register
	AT91_REG	 DBGU_CSR; 	// Channel Status Register
	AT91_REG	 DBGU_RHR; 	// Receiver Holding Register
	AT91_REG	 DBGU_THR; 	// Transmitter Holding Register
	AT91_REG	 DBGU_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 Reserved0[9]; 	// 
	AT91_REG	 DBGU_FNTR; 	// Force NTRST Register
	AT91_REG	 Reserved1[40]; 	// 
	AT91_REG	 DBGU_ADDRSIZE; 	// DBGU ADDRSIZE REGISTER 
	AT91_REG	 DBGU_IPNAME1; 	// DBGU IPNAME1 REGISTER 
	AT91_REG	 DBGU_IPNAME2; 	// DBGU IPNAME2 REGISTER 
	AT91_REG	 DBGU_FEATURES; 	// DBGU FEATURES REGISTER 
	AT91_REG	 DBGU_VER; 	// DBGU VERSION REGISTER 
	AT91_REG	 DBGU_RPR; 	// Receive Pointer Register
	AT91_REG	 DBGU_RCR; 	// Receive Counter Register
	AT91_REG	 DBGU_TPR; 	// Transmit Pointer Register
	AT91_REG	 DBGU_TCR; 	// Transmit Counter Register
	AT91_REG	 DBGU_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DBGU_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DBGU_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DBGU_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DBGU_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DBGU_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved2[6]; 	// 
	AT91_REG	 DBGU_CIDR; 	// Chip ID Register
	AT91_REG	 DBGU_EXID; 	// Chip ID Extension Register
} AT91S_DBGU, *AT91PS_DBGU;

// -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- 
#define AT91C_DBGU_RSTRX      ((unsigned int) 0x1 <<  2) // (DBGU) Reset Receiver
#define AT91C_DBGU_RSTTX      ((unsigned int) 0x1 <<  3) // (DBGU) Reset Transmitter
#define AT91C_DBGU_RXEN       ((unsigned int) 0x1 <<  4) // (DBGU) Receiver Enable
#define AT91C_DBGU_RXDIS      ((unsigned int) 0x1 <<  5) // (DBGU) Receiver Disable
#define AT91C_DBGU_TXEN       ((unsigned int) 0x1 <<  6) // (DBGU) Transmitter Enable
#define AT91C_DBGU_TXDIS      ((unsigned int) 0x1 <<  7) // (DBGU) Transmitter Disable
#define AT91C_DBGU_RSTSTA     ((unsigned int) 0x1 <<  8) // (DBGU) Reset Status Bits
// -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- 
#define AT91C_DBGU_PAR        ((unsigned int) 0x7 <<  9) // (DBGU) Parity type
#define 	AT91C_DBGU_PAR_EVEN                 ((unsigned int) 0x0 <<  9) // (DBGU) Even Parity
#define 	AT91C_DBGU_PAR_ODD                  ((unsigned int) 0x1 <<  9) // (DBGU) Odd Parity
#define 	AT91C_DBGU_PAR_SPACE                ((unsigned int) 0x2 <<  9) // (DBGU) Parity forced to 0 (Space)
#define 	AT91C_DBGU_PAR_MARK                 ((unsigned int) 0x3 <<  9) // (DBGU) Parity forced to 1 (Mark)
#define 	AT91C_DBGU_PAR_NONE                 ((unsigned int) 0x4 <<  9) // (DBGU) No Parity
#define AT91C_DBGU_CHMODE     ((unsigned int) 0x3 << 14) // (DBGU) Channel Mode
#define 	AT91C_DBGU_CHMODE_NORMAL               ((unsigned int) 0x0 << 14) // (DBGU) Normal Mode: The debug unit channel operates as an RX/TX debug unit.
#define 	AT91C_DBGU_CHMODE_AUTO                 ((unsigned int) 0x1 << 14) // (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.
#define 	AT91C_DBGU_CHMODE_LOCAL                ((unsigned int) 0x2 << 14) // (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
#define 	AT91C_DBGU_CHMODE_REMOTE               ((unsigned int) 0x3 << 14) // (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.
// -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- 
#define AT91C_DBGU_RXRDY      ((unsigned int) 0x1 <<  0) // (DBGU) RXRDY Interrupt
#define AT91C_DBGU_TXRDY      ((unsigned int) 0x1 <<  1) // (DBGU) TXRDY Interrupt
#define AT91C_DBGU_ENDRX      ((unsigned int) 0x1 <<  3) // (DBGU) End of Receive Transfer Interrupt
#define AT91C_DBGU_ENDTX      ((unsigned int) 0x1 <<  4) // (DBGU) End of Transmit Interrupt
#define AT91C_DBGU_OVRE       ((unsigned int) 0x1 <<  5) // (DBGU) Overrun Interrupt
#define AT91C_DBGU_FRAME      ((unsigned int) 0x1 <<  6) // (DBGU) Framing Error Interrupt
#define AT91C_DBGU_PARE       ((unsigned int) 0x1 <<  7) // (DBGU) Parity Error Interrupt
#define AT91C_DBGU_TXEMPTY    ((unsigned int) 0x1 <<  9) // (DBGU) TXEMPTY Interrupt
#define AT91C_DBGU_TXBUFE     ((unsigned int) 0x1 << 11) // (DBGU) TXBUFE Interrupt
#define AT91C_DBGU_RXBUFF     ((unsigned int) 0x1 << 12) // (DBGU) RXBUFF Interrupt
#define AT91C_DBGU_COMM_TX    ((unsigned int) 0x1 << 30) // (DBGU) COMM_TX Interrupt
#define AT91C_DBGU_COMM_RX    ((unsigned int) 0x1 << 31) // (DBGU) COMM_RX Interrupt
// -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- 
// -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- 
// -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- 
// -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- 
#define AT91C_DBGU_FORCE_NTRST ((unsigned int) 0x1 <<  0) // (DBGU) Force NTRST in JTAG

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler
// *****************************************************************************
typedef struct _AT91S_PIO {
	AT91_REG	 PIO_PER; 	// PIO Enable Register
	AT91_REG	 PIO_PDR; 	// PIO Disable Register
	AT91_REG	 PIO_PSR; 	// PIO Status Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PIO_OER; 	// Output Enable Register
	AT91_REG	 PIO_ODR; 	// Output Disable Registerr
	AT91_REG	 PIO_OSR; 	// Output Status Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 PIO_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIO_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIO_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved2[1]; 	// 
	AT91_REG	 PIO_SODR; 	// Set Output Data Register
	AT91_REG	 PIO_CODR; 	// Clear Output Data Register
	AT91_REG	 PIO_ODSR; 	// Output Data Status Register
	AT91_REG	 PIO_PDSR; 	// Pin Data Status Register
	AT91_REG	 PIO_IER; 	// Interrupt Enable Register
	AT91_REG	 PIO_IDR; 	// Interrupt Disable Register
	AT91_REG	 PIO_IMR; 	// Interrupt Mask Register
	AT91_REG	 PIO_ISR; 	// Interrupt Status Register
	AT91_REG	 PIO_MDER; 	// Multi-driver Enable Register
	AT91_REG	 PIO_MDDR; 	// Multi-driver Disable Register
	AT91_REG	 PIO_MDSR; 	// Multi-driver Status Register
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 PIO_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIO_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIO_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved4[1]; 	// 
	AT91_REG	 PIO_ABSR; 	// Peripheral AB Select Register
	AT91_REG	 Reserved5[3]; 	// 
	AT91_REG	 PIO_SCIFSR; 	// System Clock Glitch Input Filter Select Register
	AT91_REG	 PIO_DIFSR; 	// Debouncing Input Filter Select Register
	AT91_REG	 PIO_IFDGSR; 	// Glitch or Debouncing Input Filter Clock Selection Status Register
	AT91_REG	 PIO_SCDR; 	// Slow Clock Divider Debouncing Register
	AT91_REG	 Reserved6[4]; 	// 
	AT91_REG	 PIO_OWER; 	// Output Write Enable Register
	AT91_REG	 PIO_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIO_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved7[1]; 	// 
	AT91_REG	 PIO_AIMER; 	// Additional Interrupt Modes Enable Register
	AT91_REG	 PIO_AIMDR; 	// Additional Interrupt Modes Disables Register
	AT91_REG	 PIO_AIMMR; 	// Additional Interrupt Modes Mask Register
	AT91_REG	 Reserved8[1]; 	// 
	AT91_REG	 PIO_ESR; 	// Edge Select Register
	AT91_REG	 PIO_LSR; 	// Level Select Register
	AT91_REG	 PIO_ELSR; 	// Edge/Level Status Register
	AT91_REG	 Reserved9[1]; 	// 
	AT91_REG	 PIO_FELLSR; 	// Falling Edge/Low Level Select Register
	AT91_REG	 PIO_REHLSR; 	// Rising Edge/ High Level Select Register
	AT91_REG	 PIO_FRLHSR; 	// Fall/Rise - Low/High Status Register
	AT91_REG	 Reserved10[1]; 	// 
	AT91_REG	 PIO_LOCKSR; 	// Lock Status Register
	AT91_REG	 Reserved11[6]; 	// 
	AT91_REG	 PIO_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 Reserved12[8]; 	// 
	AT91_REG	 PIO_KER; 	// Keypad Controller Enable Register
	AT91_REG	 PIO_KRCR; 	// Keypad Controller Row Column Register
	AT91_REG	 PIO_KDR; 	// Keypad Controller Debouncing Register
	AT91_REG	 Reserved13[1]; 	// 
	AT91_REG	 PIO_KIER; 	// Keypad Controller Interrupt Enable Register
	AT91_REG	 PIO_KIDR; 	// Keypad Controller Interrupt Disable Register
	AT91_REG	 PIO_KIMR; 	// Keypad Controller Interrupt Mask Register
	AT91_REG	 PIO_KSR; 	// Keypad Controller Status Register
	AT91_REG	 PIO_KKPR; 	// Keypad Controller Key Press Register
	AT91_REG	 PIO_KKRR; 	// Keypad Controller Key Release Register
} AT91S_PIO, *AT91PS_PIO;

// -------- PIO_KER : (PIO Offset: 0x120) Keypad Controller Enable Register -------- 
#define AT91C_PIO_KCE         ((unsigned int) 0x1 <<  0) // (PIO) Keypad Controller Enable
// -------- PIO_KRCR : (PIO Offset: 0x124) Keypad Controller Row Column Register -------- 
#define AT91C_PIO_NBR         ((unsigned int) 0x7 <<  0) // (PIO) Number of Columns of the Keypad Matrix
#define AT91C_PIO_NBC         ((unsigned int) 0x7 <<  8) // (PIO) Number of Rows of the Keypad Matrix
// -------- PIO_KDR : (PIO Offset: 0x128) Keypad Controller Debouncing Register -------- 
#define AT91C_PIO_DBC         ((unsigned int) 0x3FF <<  0) // (PIO) Debouncing Value
// -------- PIO_KIER : (PIO Offset: 0x130) Keypad Controller Interrupt Enable Register -------- 
#define AT91C_PIO_KPR         ((unsigned int) 0x1 <<  0) // (PIO) Key Press Interrupt Enable
#define AT91C_PIO_KRL         ((unsigned int) 0x1 <<  1) // (PIO) Key Release Interrupt Enable
// -------- PIO_KIDR : (PIO Offset: 0x134) Keypad Controller Interrupt Disable Register -------- 
// -------- PIO_KIMR : (PIO Offset: 0x138) Keypad Controller Interrupt Mask Register -------- 
// -------- PIO_KSR : (PIO Offset: 0x13c) Keypad Controller Status Register -------- 
#define AT91C_PIO_NBKPR       ((unsigned int) 0x3 <<  8) // (PIO) Number of Simultaneous Key Presses
#define AT91C_PIO_NBKRL       ((unsigned int) 0x3 << 16) // (PIO) Number of Simultaneous Key Releases
// -------- PIO_KKPR : (PIO Offset: 0x140) Keypad Controller Key Press Register -------- 
#define AT91C_KEY0ROW         ((unsigned int) 0x7 <<  0) // (PIO) Row index of the first detected Key Press
#define AT91C_KEY0COL         ((unsigned int) 0x7 <<  4) // (PIO) Column index of the first detected Key Press
#define AT91C_KEY1ROW         ((unsigned int) 0x7 <<  8) // (PIO) Row index of the second detected Key Press
#define AT91C_KEY1COL         ((unsigned int) 0x7 << 12) // (PIO) Column index of the second detected Key Press
#define AT91C_KEY2ROW         ((unsigned int) 0x7 << 16) // (PIO) Row index of the third detected Key Press
#define AT91C_KEY2COL         ((unsigned int) 0x7 << 20) // (PIO) Column index of the third detected Key Press
#define AT91C_KEY3ROW         ((unsigned int) 0x7 << 24) // (PIO) Row index of the fourth detected Key Press
#define AT91C_KEY3COL         ((unsigned int) 0x7 << 28) // (PIO) Column index of the fourth detected Key Press
// -------- PIO_KKRR : (PIO Offset: 0x144) Keypad Controller Key Release Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Power Management Controler
// *****************************************************************************
typedef struct _AT91S_PMC {
	AT91_REG	 PMC_SCER; 	// System Clock Enable Register
	AT91_REG	 PMC_SCDR; 	// System Clock Disable Register
	AT91_REG	 PMC_SCSR; 	// System Clock Status Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PMC_PCER; 	// Peripheral Clock Enable Register
	AT91_REG	 PMC_PCDR; 	// Peripheral Clock Disable Register
	AT91_REG	 PMC_PCSR; 	// Peripheral Clock Status Register
	AT91_REG	 PMC_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 PMC_MOR; 	// Main Oscillator Register
	AT91_REG	 PMC_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 PMC_PLLAR; 	// PLL Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 PMC_MCKR; 	// Master Clock Register
	AT91_REG	 Reserved2[3]; 	// 
	AT91_REG	 PMC_PCKR[8]; 	// Programmable Clock Register
	AT91_REG	 PMC_IER; 	// Interrupt Enable Register
	AT91_REG	 PMC_IDR; 	// Interrupt Disable Register
	AT91_REG	 PMC_SR; 	// Status Register
	AT91_REG	 PMC_IMR; 	// Interrupt Mask Register
	AT91_REG	 PMC_FSMR; 	// Fast Startup Mode Register
	AT91_REG	 PMC_FSPR; 	// Fast Startup Polarity Register
	AT91_REG	 PMC_FOCR; 	// Fault Output Clear Register
	AT91_REG	 Reserved3[28]; 	// 
	AT91_REG	 PMC_ADDRSIZE; 	// PMC ADDRSIZE REGISTER 
	AT91_REG	 PMC_IPNAME1; 	// PMC IPNAME1 REGISTER 
	AT91_REG	 PMC_IPNAME2; 	// PMC IPNAME2 REGISTER 
	AT91_REG	 PMC_FEATURES; 	// PMC FEATURES REGISTER 
	AT91_REG	 PMC_VER; 	// APMC VERSION REGISTER
} AT91S_PMC, *AT91PS_PMC;

// -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) // (PMC) Processor Clock
#define AT91C_PMC_PCK0        ((unsigned int) 0x1 <<  8) // (PMC) Programmable Clock Output
#define AT91C_PMC_PCK1        ((unsigned int) 0x1 <<  9) // (PMC) Programmable Clock Output
#define AT91C_PMC_PCK2        ((unsigned int) 0x1 << 10) // (PMC) Programmable Clock Output
// -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- 
// -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- 
// -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- 
#define AT91C_CKGR_UPLLEN     ((unsigned int) 0x1 << 16) // (PMC) UTMI PLL Enable
#define 	AT91C_CKGR_UPLLEN_DISABLED             ((unsigned int) 0x0 << 16) // (PMC) The UTMI PLL is disabled
#define 	AT91C_CKGR_UPLLEN_ENABLED              ((unsigned int) 0x1 << 16) // (PMC) The UTMI PLL is enabled
#define AT91C_CKGR_UPLLCOUNT  ((unsigned int) 0xF << 20) // (PMC) UTMI Oscillator Start-up Time
#define AT91C_CKGR_BIASEN     ((unsigned int) 0x1 << 24) // (PMC) UTMI BIAS Enable
#define 	AT91C_CKGR_BIASEN_DISABLED             ((unsigned int) 0x0 << 24) // (PMC) The UTMI BIAS is disabled
#define 	AT91C_CKGR_BIASEN_ENABLED              ((unsigned int) 0x1 << 24) // (PMC) The UTMI BIAS is enabled
#define AT91C_CKGR_BIASCOUNT  ((unsigned int) 0xF << 28) // (PMC) UTMI BIAS Start-up Time
// -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- 
#define AT91C_CKGR_MOSCXTEN   ((unsigned int) 0x1 <<  0) // (PMC) Main Crystal Oscillator Enable
#define AT91C_CKGR_MOSCXTBY   ((unsigned int) 0x1 <<  1) // (PMC) Main Crystal Oscillator Bypass
#define AT91C_CKGR_WAITMODE   ((unsigned int) 0x1 <<  2) // (PMC) Main Crystal Oscillator Bypass
#define AT91C_CKGR_MOSCRCEN   ((unsigned int) 0x1 <<  3) // (PMC) Main On-Chip RC Oscillator Enable
#define AT91C_CKGR_MOSCRCF    ((unsigned int) 0x7 <<  4) // (PMC) Main On-Chip RC Oscillator Frequency Selection
#define AT91C_CKGR_MOSCXTST   ((unsigned int) 0xFF <<  8) // (PMC) Main Crystal Oscillator Start-up Time
#define AT91C_CKGR_KEY        ((unsigned int) 0xFF << 16) // (PMC) Clock Generator Controller Writing Protection Key
#define AT91C_CKGR_MOSCSEL    ((unsigned int) 0x1 << 24) // (PMC) Main Oscillator Selection
#define AT91C_CKGR_CFDEN      ((unsigned int) 0x1 << 25) // (PMC) Clock Failure Detector Enable
// -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- 
#define AT91C_CKGR_MAINF      ((unsigned int) 0xFFFF <<  0) // (PMC) Main Clock Frequency
#define AT91C_CKGR_MAINRDY    ((unsigned int) 0x1 << 16) // (PMC) Main Clock Ready
// -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- 
#define AT91C_CKGR_DIVA       ((unsigned int) 0xFF <<  0) // (PMC) Divider Selected
#define 	AT91C_CKGR_DIVA_0                    ((unsigned int) 0x0) // (PMC) Divider output is 0
#define 	AT91C_CKGR_DIVA_BYPASS               ((unsigned int) 0x1) // (PMC) Divider is bypassed
#define AT91C_CKGR_PLLACOUNT  ((unsigned int) 0x3F <<  8) // (PMC) PLLA Counter
#define AT91C_CKGR_STMODE     ((unsigned int) 0x3 << 14) // (PMC) Start Mode
#define 	AT91C_CKGR_STMODE_0                    ((unsigned int) 0x0 << 14) // (PMC) Fast startup
#define 	AT91C_CKGR_STMODE_1                    ((unsigned int) 0x1 << 14) // (PMC) Reserved
#define 	AT91C_CKGR_STMODE_2                    ((unsigned int) 0x2 << 14) // (PMC) Normal startup
#define 	AT91C_CKGR_STMODE_3                    ((unsigned int) 0x3 << 14) // (PMC) Reserved
#define AT91C_CKGR_MULA       ((unsigned int) 0x7FF << 16) // (PMC) PLL Multiplier
#define AT91C_CKGR_SRC        ((unsigned int) 0x1 << 29) // (PMC) 
// -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
#define AT91C_PMC_CSS         ((unsigned int) 0x7 <<  0) // (PMC) Programmable Clock Selection
#define 	AT91C_PMC_CSS_SLOW_CLK             ((unsigned int) 0x0) // (PMC) Slow Clock is selected
#define 	AT91C_PMC_CSS_MAIN_CLK             ((unsigned int) 0x1) // (PMC) Main Clock is selected
#define 	AT91C_PMC_CSS_PLLA_CLK             ((unsigned int) 0x2) // (PMC) Clock from PLL A is selected
#define 	AT91C_PMC_CSS_UPLL_CLK             ((unsigned int) 0x3) // (PMC) Clock from UPLL is selected
#define 	AT91C_PMC_CSS_SYS_CLK              ((unsigned int) 0x4) // (PMC) System clock is selected
#define AT91C_PMC_PRES        ((unsigned int) 0x7 <<  4) // (PMC) Programmable Clock Prescaler
#define 	AT91C_PMC_PRES_CLK                  ((unsigned int) 0x0 <<  4) // (PMC) Selected clock
#define 	AT91C_PMC_PRES_CLK_2                ((unsigned int) 0x1 <<  4) // (PMC) Selected clock divided by 2
#define 	AT91C_PMC_PRES_CLK_4                ((unsigned int) 0x2 <<  4) // (PMC) Selected clock divided by 4
#define 	AT91C_PMC_PRES_CLK_8                ((unsigned int) 0x3 <<  4) // (PMC) Selected clock divided by 8
#define 	AT91C_PMC_PRES_CLK_16               ((unsigned int) 0x4 <<  4) // (PMC) Selected clock divided by 16
#define 	AT91C_PMC_PRES_CLK_32               ((unsigned int) 0x5 <<  4) // (PMC) Selected clock divided by 32
#define 	AT91C_PMC_PRES_CLK_64               ((unsigned int) 0x6 <<  4) // (PMC) Selected clock divided by 64
#define 	AT91C_PMC_PRES_CLK_3                ((unsigned int) 0x7 <<  4) // (PMC) Selected clock divided by 3
// -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- 
// -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
#define AT91C_PMC_MOSCXTS     ((unsigned int) 0x1 <<  0) // (PMC) Main Crystal Oscillator Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKA       ((unsigned int) 0x1 <<  1) // (PMC) PLL A Status/Enable/Disable/Mask
#define AT91C_PMC_MCKRDY      ((unsigned int) 0x1 <<  3) // (PMC) Master Clock Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKU       ((unsigned int) 0x1 <<  6) // (PMC) PLL UTMI Status/Enable/Disable/Mask
#define AT91C_PMC_PCKRDY0     ((unsigned int) 0x1 <<  8) // (PMC) PCK0_RDY Status/Enable/Disable/Mask
#define AT91C_PMC_PCKRDY1     ((unsigned int) 0x1 <<  9) // (PMC) PCK1_RDY Status/Enable/Disable/Mask
#define AT91C_PMC_PCKRDY2     ((unsigned int) 0x1 << 10) // (PMC) PCK2_RDY Status/Enable/Disable/Mask
#define AT91C_PMC_MOSCSELS    ((unsigned int) 0x1 << 16) // (PMC) Main Oscillator Selection Status
#define AT91C_PMC_MOSCRCS     ((unsigned int) 0x1 << 17) // (PMC) Main On-Chip RC Oscillator Status
#define AT91C_PMC_CFDEV       ((unsigned int) 0x1 << 18) // (PMC) Clock Failure Detector Event
// -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- 
// -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- 
#define AT91C_PMC_OSCSELS     ((unsigned int) 0x1 <<  7) // (PMC) Slow Clock Oscillator Selection
#define AT91C_PMC_CFDS        ((unsigned int) 0x1 << 19) // (PMC) Clock Failure Detector Status
#define AT91C_PMC_FOS         ((unsigned int) 0x1 << 20) // (PMC) Clock Failure Detector Fault Output Status
// -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- 
// -------- PMC_FSMR : (PMC Offset: 0x70) Fast Startup Mode Register -------- 
#define AT91C_PMC_FSTT        ((unsigned int) 0xFFFF <<  0) // (PMC) Fast Start-up Input Enable 0 to 15
#define AT91C_PMC_RTTAL       ((unsigned int) 0x1 << 16) // (PMC) RTT Alarm Enable
#define AT91C_PMC_RTCAL       ((unsigned int) 0x1 << 17) // (PMC) RTC Alarm Enable
#define AT91C_PMC_USBAL       ((unsigned int) 0x1 << 18) // (PMC) USB Alarm Enable
#define AT91C_PMC_LPM         ((unsigned int) 0x1 << 20) // (PMC) Low Power Mode
// -------- PMC_FSPR : (PMC Offset: 0x74) Fast Startup Polarity Register -------- 
#define AT91C_PMC_FSTP        ((unsigned int) 0xFFFF <<  0) // (PMC) Fast Start-up Input Polarity 0 to 15
// -------- PMC_FOCR : (PMC Offset: 0x78) Fault Output Clear Register -------- 
#define AT91C_PMC_FOCLR       ((unsigned int) 0x1 <<  0) // (PMC) Fault Output Clear

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// *****************************************************************************
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 CKGR_MOR; 	// Main Oscillator Register
	AT91_REG	 CKGR_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 CKGR_PLLAR; 	// PLL Register
} AT91S_CKGR, *AT91PS_CKGR;

// -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Reset Controller Interface
// *****************************************************************************
typedef struct _AT91S_RSTC {
	AT91_REG	 RSTC_RCR; 	// Reset Control Register
	AT91_REG	 RSTC_RSR; 	// Reset Status Register
	AT91_REG	 RSTC_RMR; 	// Reset Mode Register
	AT91_REG	 Reserved0[60]; 	// 
	AT91_REG	 RSTC_VER; 	// Version Register
} AT91S_RSTC, *AT91PS_RSTC;

// -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- 
#define AT91C_RSTC_PROCRST    ((unsigned int) 0x1 <<  0) // (RSTC) Processor Reset
#define AT91C_RSTC_ICERST     ((unsigned int) 0x1 <<  1) // (RSTC) ICE Interface Reset
#define AT91C_RSTC_PERRST     ((unsigned int) 0x1 <<  2) // (RSTC) Peripheral Reset
#define AT91C_RSTC_EXTRST     ((unsigned int) 0x1 <<  3) // (RSTC) External Reset
#define AT91C_RSTC_KEY        ((unsigned int) 0xFF << 24) // (RSTC) Password
// -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- 
#define AT91C_RSTC_URSTS      ((unsigned int) 0x1 <<  0) // (RSTC) User Reset Status
#define AT91C_RSTC_RSTTYP     ((unsigned int) 0x7 <<  8) // (RSTC) Reset Type
#define 	AT91C_RSTC_RSTTYP_GENERAL              ((unsigned int) 0x0 <<  8) // (RSTC) General reset. Both VDDCORE and VDDBU rising.
#define 	AT91C_RSTC_RSTTYP_WAKEUP               ((unsigned int) 0x1 <<  8) // (RSTC) WakeUp Reset. VDDCORE rising.
#define 	AT91C_RSTC_RSTTYP_WATCHDOG             ((unsigned int) 0x2 <<  8) // (RSTC) Watchdog Reset. Watchdog overflow occured.
#define 	AT91C_RSTC_RSTTYP_SOFTWARE             ((unsigned int) 0x3 <<  8) // (RSTC) Software Reset. Processor reset required by the software.
#define 	AT91C_RSTC_RSTTYP_USER                 ((unsigned int) 0x4 <<  8) // (RSTC) User Reset. NRST pin detected low.
#define AT91C_RSTC_NRSTL      ((unsigned int) 0x1 << 16) // (RSTC) NRST pin level
#define AT91C_RSTC_SRCMP      ((unsigned int) 0x1 << 17) // (RSTC) Software Reset Command in Progress.
// -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- 
#define AT91C_RSTC_URSTEN     ((unsigned int) 0x1 <<  0) // (RSTC) User Reset Enable
#define AT91C_RSTC_URSTIEN    ((unsigned int) 0x1 <<  4) // (RSTC) User Reset Interrupt Enable
#define AT91C_RSTC_ERSTL      ((unsigned int) 0xF <<  8) // (RSTC) User Reset Enable

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Supply Controller Interface
// *****************************************************************************
typedef struct _AT91S_SUPC {
	AT91_REG	 SUPC_CR; 	// Control Register
	AT91_REG	 SUPC_BOMR; 	// Brown Out Mode Register
	AT91_REG	 SUPC_MR; 	// Mode Register
	AT91_REG	 SUPC_WUMR; 	// Wake Up Mode Register
	AT91_REG	 SUPC_WUIR; 	// Wake Up Inputs Register
	AT91_REG	 SUPC_SR; 	// Status Register
	AT91_REG	 SUPC_FWUTR; 	// Flash Wake-up Timer Register
} AT91S_SUPC, *AT91PS_SUPC;

// -------- SUPC_CR : (SUPC Offset: 0x0) Control Register -------- 
#define AT91C_SUPC_SHDW       ((unsigned int) 0x1 <<  0) // (SUPC) Shut Down Command
#define AT91C_SUPC_SHDWEOF    ((unsigned int) 0x1 <<  1) // (SUPC) Shut Down after End Of Frame
#define AT91C_SUPC_VROFF      ((unsigned int) 0x1 <<  2) // (SUPC) Voltage Regulator Off
#define AT91C_SUPC_XTALSEL    ((unsigned int) 0x1 <<  3) // (SUPC) Crystal Oscillator Select
#define AT91C_SUPC_KEY        ((unsigned int) 0xFF << 24) // (SUPC) Supply Controller Writing Protection Key
// -------- SUPC_BOMR : (SUPC Offset: 0x4) Brown Out Mode Register -------- 
#define AT91C_SUPC_BODTH      ((unsigned int) 0xF <<  0) // (SUPC) Brown Out Threshold
#define AT91C_SUPC_BODSMPL    ((unsigned int) 0x7 <<  8) // (SUPC) Brown Out Sampling Period
#define 	AT91C_SUPC_BODSMPL_DISABLED             ((unsigned int) 0x0 <<  8) // (SUPC) Brown Out Detector disabled
#define 	AT91C_SUPC_BODSMPL_CONTINUOUS           ((unsigned int) 0x1 <<  8) // (SUPC) Continuous Brown Out Detector
#define 	AT91C_SUPC_BODSMPL_32_SLCK              ((unsigned int) 0x2 <<  8) // (SUPC) Brown Out Detector enabled one SLCK period every 32 SLCK periods
#define 	AT91C_SUPC_BODSMPL_256_SLCK             ((unsigned int) 0x3 <<  8) // (SUPC) Brown Out Detector enabled one SLCK period every 256 SLCK periods
#define 	AT91C_SUPC_BODSMPL_2048_SLCK            ((unsigned int) 0x4 <<  8) // (SUPC) Brown Out Detector enabled one SLCK period every 2048 SLCK periods
#define AT91C_SUPC_BODRSTEN   ((unsigned int) 0x1 << 12) // (SUPC) Brownout Reset Enable
// -------- SUPC_MR : (SUPC Offset: 0x8) Supply Controller Mode Register -------- 
#define AT91C_SUPC_LCDOUT     ((unsigned int) 0xF <<  0) // (SUPC) LCD Charge Pump Output Voltage Selection
#define AT91C_SUPC_LCDMODE    ((unsigned int) 0x3 <<  4) // (SUPC) Segment LCD Supply Mode
#define 	AT91C_SUPC_LCDMODE_OFF                  ((unsigned int) 0x0 <<  4) // (SUPC) The internal and external supply sources are both deselected and the on-chip charge pump is turned off
#define 	AT91C_SUPC_LCDMODE_OFF_AFTER_EOF        ((unsigned int) 0x1 <<  4) // (SUPC) At the End Of Frame from LCD controller, the internal and external supply sources are both deselected and the on-chip charge pump is turned off
#define 	AT91C_SUPC_LCDMODE_EXTERNAL             ((unsigned int) 0x2 <<  4) // (SUPC) The external supply source is selected
#define 	AT91C_SUPC_LCDMODE_INTERNAL             ((unsigned int) 0x3 <<  4) // (SUPC) The internal supply source is selected and the on-chip charge pump is turned on
#define AT91C_SUPC_VRDEEP     ((unsigned int) 0x1 <<  8) // (SUPC) Voltage Regulator Deep Mode
#define AT91C_SUPC_VRVDD      ((unsigned int) 0x7 <<  9) // (SUPC) Voltage Regulator Output Voltage Selection
#define AT91C_SUPC_VRRSTEN    ((unsigned int) 0x1 << 12) // (SUPC) Voltage Regulation Loss Reset Enable
#define AT91C_SUPC_GPBRON     ((unsigned int) 0x1 << 16) // (SUPC) GPBR ON
#define AT91C_SUPC_SRAMON     ((unsigned int) 0x1 << 17) // (SUPC) SRAM ON
#define AT91C_SUPC_RTCON      ((unsigned int) 0x1 << 18) // (SUPC) Real Time Clock Power switch ON
#define AT91C_SUPC_FLASHON    ((unsigned int) 0x1 << 19) // (SUPC) Flash Power switch On
#define AT91C_SUPC_BYPASS     ((unsigned int) 0x1 << 20) // (SUPC) 32kHz oscillator bypass
#define AT91C_SUPC_MKEY       ((unsigned int) 0xFF << 24) // (SUPC) Supply Controller Writing Protection Key
// -------- SUPC_WUMR : (SUPC Offset: 0xc) Wake Up Mode Register -------- 
#define AT91C_SUPC_FWUPEN     ((unsigned int) 0x1 <<  0) // (SUPC) Force Wake Up Enable
#define AT91C_SUPC_BODEN      ((unsigned int) 0x1 <<  1) // (SUPC) Brown Out Wake Up Enable
#define AT91C_SUPC_RTTEN      ((unsigned int) 0x1 <<  2) // (SUPC) Real Time Timer Wake Up Enable
#define AT91C_SUPC_RTCEN      ((unsigned int) 0x1 <<  3) // (SUPC) Real Time Clock Wake Up Enable
#define AT91C_SUPC_FWUPDBC    ((unsigned int) 0x7 <<  8) // (SUPC) Force Wake Up debouncer
#define 	AT91C_SUPC_FWUPDBC_IMMEDIATE            ((unsigned int) 0x0 <<  8) // (SUPC) Immediate, No debouncing, detected active at least one Slow clock edge
#define 	AT91C_SUPC_FWUPDBC_3_SLCK               ((unsigned int) 0x1 <<  8) // (SUPC) An enabled Wake Up input shall be low for at least 3 SLCK periods
#define 	AT91C_SUPC_FWUPDBC_32_SLCK              ((unsigned int) 0x2 <<  8) // (SUPC) An enabled Wake Up input  shall be low for at least 32 SLCK periods
#define 	AT91C_SUPC_FWUPDBC_512_SLCK             ((unsigned int) 0x3 <<  8) // (SUPC) An enabled Wake Up input  shall be low for at least 512 SLCK periods
#define 	AT91C_SUPC_FWUPDBC_4096_SLCK            ((unsigned int) 0x4 <<  8) // (SUPC) An enabled Wake Up input  shall be low for at least 4096 SLCK periods
#define 	AT91C_SUPC_FWUPDBC_32768_SLCK           ((unsigned int) 0x5 <<  8) // (SUPC) An enabled Wake Up input  shall be low for at least 32768 SLCK periods
#define AT91C_SUPC_WKUPDBC    ((unsigned int) 0x7 << 12) // (SUPC) Force Wake Up debouncer
#define 	AT91C_SUPC_WKUPDBC_IMMEDIATE            ((unsigned int) 0x0 << 12) // (SUPC) Immediate, No debouncing, detected active at least one Slow clock edge
#define 	AT91C_SUPC_WKUPDBC_3_SLCK               ((unsigned int) 0x1 << 12) // (SUPC) FWUP shall be low for at least 3 SLCK periods
#define 	AT91C_SUPC_WKUPDBC_32_SLCK              ((unsigned int) 0x2 << 12) // (SUPC) FWUP shall be low for at least 32 SLCK periods
#define 	AT91C_SUPC_WKUPDBC_512_SLCK             ((unsigned int) 0x3 << 12) // (SUPC) FWUP shall be low for at least 512 SLCK periods
#define 	AT91C_SUPC_WKUPDBC_4096_SLCK            ((unsigned int) 0x4 << 12) // (SUPC) FWUP shall be low for at least 4096 SLCK periods
#define 	AT91C_SUPC_WKUPDBC_32768_SLCK           ((unsigned int) 0x5 << 12) // (SUPC) FWUP shall be low for at least 32768 SLCK periods
// -------- SUPC_WUIR : (SUPC Offset: 0x10) Wake Up Inputs Register -------- 
#define AT91C_SUPC_WKUPEN0    ((unsigned int) 0x1 <<  0) // (SUPC) Wake Up Input Enable 0
#define AT91C_SUPC_WKUPEN1    ((unsigned int) 0x1 <<  1) // (SUPC) Wake Up Input Enable 1
#define AT91C_SUPC_WKUPEN2    ((unsigned int) 0x1 <<  2) // (SUPC) Wake Up Input Enable 2
#define AT91C_SUPC_WKUPEN3    ((unsigned int) 0x1 <<  3) // (SUPC) Wake Up Input Enable 3
#define AT91C_SUPC_WKUPEN4    ((unsigned int) 0x1 <<  4) // (SUPC) Wake Up Input Enable 4
#define AT91C_SUPC_WKUPEN5    ((unsigned int) 0x1 <<  5) // (SUPC) Wake Up Input Enable 5
#define AT91C_SUPC_WKUPEN6    ((unsigned int) 0x1 <<  6) // (SUPC) Wake Up Input Enable 6
#define AT91C_SUPC_WKUPEN7    ((unsigned int) 0x1 <<  7) // (SUPC) Wake Up Input Enable 7
#define AT91C_SUPC_WKUPEN8    ((unsigned int) 0x1 <<  8) // (SUPC) Wake Up Input Enable 8
#define AT91C_SUPC_WKUPEN9    ((unsigned int) 0x1 <<  9) // (SUPC) Wake Up Input Enable 9
#define AT91C_SUPC_WKUPEN10   ((unsigned int) 0x1 << 10) // (SUPC) Wake Up Input Enable 10
#define AT91C_SUPC_WKUPEN11   ((unsigned int) 0x1 << 11) // (SUPC) Wake Up Input Enable 11
#define AT91C_SUPC_WKUPEN12   ((unsigned int) 0x1 << 12) // (SUPC) Wake Up Input Enable 12
#define AT91C_SUPC_WKUPEN13   ((unsigned int) 0x1 << 13) // (SUPC) Wake Up Input Enable 13
#define AT91C_SUPC_WKUPEN14   ((unsigned int) 0x1 << 14) // (SUPC) Wake Up Input Enable 14
#define AT91C_SUPC_WKUPEN15   ((unsigned int) 0x1 << 15) // (SUPC) Wake Up Input Enable 15
#define AT91C_SUPC_WKUPT0     ((unsigned int) 0x1 << 16) // (SUPC) Wake Up Input Transition 0
#define AT91C_SUPC_WKUPT1     ((unsigned int) 0x1 << 17) // (SUPC) Wake Up Input Transition 1
#define AT91C_SUPC_WKUPT2     ((unsigned int) 0x1 << 18) // (SUPC) Wake Up Input Transition 2
#define AT91C_SUPC_WKUPT3     ((unsigned int) 0x1 << 19) // (SUPC) Wake Up Input Transition 3
#define AT91C_SUPC_WKUPT4     ((unsigned int) 0x1 << 20) // (SUPC) Wake Up Input Transition 4
#define AT91C_SUPC_WKUPT5     ((unsigned int) 0x1 << 21) // (SUPC) Wake Up Input Transition 5
#define AT91C_SUPC_WKUPT6     ((unsigned int) 0x1 << 22) // (SUPC) Wake Up Input Transition 6
#define AT91C_SUPC_WKUPT7     ((unsigned int) 0x1 << 23) // (SUPC) Wake Up Input Transition 7
#define AT91C_SUPC_WKUPT8     ((unsigned int) 0x1 << 24) // (SUPC) Wake Up Input Transition 8
#define AT91C_SUPC_WKUPT9     ((unsigned int) 0x1 << 25) // (SUPC) Wake Up Input Transition 9
#define AT91C_SUPC_WKUPT10    ((unsigned int) 0x1 << 26) // (SUPC) Wake Up Input Transition 10
#define AT91C_SUPC_WKUPT11    ((unsigned int) 0x1 << 27) // (SUPC) Wake Up Input Transition 11
#define AT91C_SUPC_WKUPT12    ((unsigned int) 0x1 << 28) // (SUPC) Wake Up Input Transition 12
#define AT91C_SUPC_WKUPT13    ((unsigned int) 0x1 << 29) // (SUPC) Wake Up Input Transition 13
#define AT91C_SUPC_WKUPT14    ((unsigned int) 0x1 << 30) // (SUPC) Wake Up Input Transition 14
#define AT91C_SUPC_WKUPT15    ((unsigned int) 0x1 << 31) // (SUPC) Wake Up Input Transition 15
// -------- SUPC_SR : (SUPC Offset: 0x14) Status Register -------- 
#define AT91C_SUPC_FWUPS      ((unsigned int) 0x1 <<  0) // (SUPC) Force Wake Up Status
#define AT91C_SUPC_WKUPS      ((unsigned int) 0x1 <<  1) // (SUPC) Wake Up Status
#define AT91C_SUPC_BODWS      ((unsigned int) 0x1 <<  2) // (SUPC) BOD Detection Wake Up Status
#define AT91C_SUPC_VRRSTS     ((unsigned int) 0x1 <<  3) // (SUPC) Voltage regulation Loss Reset Status
#define AT91C_SUPC_BODRSTS    ((unsigned int) 0x1 <<  4) // (SUPC) BOD detection Reset Status
#define AT91C_SUPC_BODS       ((unsigned int) 0x1 <<  5) // (SUPC) BOD Status
#define AT91C_SUPC_BROWNOUT   ((unsigned int) 0x1 <<  6) // (SUPC) BOD Output Status
#define AT91C_SUPC_OSCSEL     ((unsigned int) 0x1 <<  7) // (SUPC) 32kHz Oscillator Selection Status
#define AT91C_SUPC_LCDS       ((unsigned int) 0x1 <<  8) // (SUPC) LCD Status
#define AT91C_SUPC_GPBRS      ((unsigned int) 0x1 <<  9) // (SUPC) General Purpose Back-up registers Status
#define AT91C_SUPC_RTS        ((unsigned int) 0x1 << 10) // (SUPC) Clock Status
#define AT91C_SUPC_FLASHS     ((unsigned int) 0x1 << 11) // (SUPC) FLASH Memory Status
#define AT91C_SUPC_FWUPIS     ((unsigned int) 0x1 << 12) // (SUPC) WKUP Input Status
#define AT91C_SUPC_WKUPIS0    ((unsigned int) 0x1 << 16) // (SUPC) WKUP Input 0 Status
#define AT91C_SUPC_WKUPIS1    ((unsigned int) 0x1 << 17) // (SUPC) WKUP Input 1 Status
#define AT91C_SUPC_WKUPIS2    ((unsigned int) 0x1 << 18) // (SUPC) WKUP Input 2 Status
#define AT91C_SUPC_WKUPIS3    ((unsigned int) 0x1 << 19) // (SUPC) WKUP Input 3 Status
#define AT91C_SUPC_WKUPIS4    ((unsigned int) 0x1 << 20) // (SUPC) WKUP Input 4 Status
#define AT91C_SUPC_WKUPIS5    ((unsigned int) 0x1 << 21) // (SUPC) WKUP Input 5 Status
#define AT91C_SUPC_WKUPIS6    ((unsigned int) 0x1 << 22) // (SUPC) WKUP Input 6 Status
#define AT91C_SUPC_WKUPIS7    ((unsigned int) 0x1 << 23) // (SUPC) WKUP Input 7 Status
#define AT91C_SUPC_WKUPIS8    ((unsigned int) 0x1 << 24) // (SUPC) WKUP Input 8 Status
#define AT91C_SUPC_WKUPIS9    ((unsigned int) 0x1 << 25) // (SUPC) WKUP Input 9 Status
#define AT91C_SUPC_WKUPIS10   ((unsigned int) 0x1 << 26) // (SUPC) WKUP Input 10 Status
#define AT91C_SUPC_WKUPIS11   ((unsigned int) 0x1 << 27) // (SUPC) WKUP Input 11 Status
#define AT91C_SUPC_WKUPIS12   ((unsigned int) 0x1 << 28) // (SUPC) WKUP Input 12 Status
#define AT91C_SUPC_WKUPIS13   ((unsigned int) 0x1 << 29) // (SUPC) WKUP Input 13 Status
#define AT91C_SUPC_WKUPIS14   ((unsigned int) 0x1 << 30) // (SUPC) WKUP Input 14 Status
#define AT91C_SUPC_WKUPIS15   ((unsigned int) 0x1 << 31) // (SUPC) WKUP Input 15 Status
// -------- SUPC_FWUTR : (SUPC Offset: 0x18) Flash Wake Up Timer Register -------- 
#define AT91C_SUPC_FWUT       ((unsigned int) 0x3FF <<  0) // (SUPC) Flash Wake Up Timer

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface
// *****************************************************************************
typedef struct _AT91S_RTTC {
	AT91_REG	 RTTC_RTMR; 	// Real-time Mode Register
	AT91_REG	 RTTC_RTAR; 	// Real-time Alarm Register
	AT91_REG	 RTTC_RTVR; 	// Real-time Value Register
	AT91_REG	 RTTC_RTSR; 	// Real-time Status Register
} AT91S_RTTC, *AT91PS_RTTC;

// -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- 
#define AT91C_RTTC_RTPRES     ((unsigned int) 0xFFFF <<  0) // (RTTC) Real-time Timer Prescaler Value
#define AT91C_RTTC_ALMIEN     ((unsigned int) 0x1 << 16) // (RTTC) Alarm Interrupt Enable
#define AT91C_RTTC_RTTINCIEN  ((unsigned int) 0x1 << 17) // (RTTC) Real Time Timer Increment Interrupt Enable
#define AT91C_RTTC_RTTRST     ((unsigned int) 0x1 << 18) // (RTTC) Real Time Timer Restart
// -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- 
#define AT91C_RTTC_ALMV       ((unsigned int) 0x0 <<  0) // (RTTC) Alarm Value
// -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- 
#define AT91C_RTTC_CRTV       ((unsigned int) 0x0 <<  0) // (RTTC) Current Real-time Value
// -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- 
#define AT91C_RTTC_ALMS       ((unsigned int) 0x1 <<  0) // (RTTC) Real-time Alarm Status
#define AT91C_RTTC_RTTINC     ((unsigned int) 0x1 <<  1) // (RTTC) Real-time Timer Increment

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface
// *****************************************************************************
typedef struct _AT91S_WDTC {
	AT91_REG	 WDTC_WDCR; 	// Watchdog Control Register
	AT91_REG	 WDTC_WDMR; 	// Watchdog Mode Register
	AT91_REG	 WDTC_WDSR; 	// Watchdog Status Register
} AT91S_WDTC, *AT91PS_WDTC;

// -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- 
#define AT91C_WDTC_WDRSTT     ((unsigned int) 0x1 <<  0) // (WDTC) Watchdog Restart
#define AT91C_WDTC_KEY        ((unsigned int) 0xFF << 24) // (WDTC) Watchdog KEY Password
// -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- 
#define AT91C_WDTC_WDV        ((unsigned int) 0xFFF <<  0) // (WDTC) Watchdog Timer Restart
#define AT91C_WDTC_WDFIEN     ((unsigned int) 0x1 << 12) // (WDTC) Watchdog Fault Interrupt Enable
#define AT91C_WDTC_WDRSTEN    ((unsigned int) 0x1 << 13) // (WDTC) Watchdog Reset Enable
#define AT91C_WDTC_WDRPROC    ((unsigned int) 0x1 << 14) // (WDTC) Watchdog Timer Restart
#define AT91C_WDTC_WDDIS      ((unsigned int) 0x1 << 15) // (WDTC) Watchdog Disable
#define AT91C_WDTC_WDD        ((unsigned int) 0xFFF << 16) // (WDTC) Watchdog Delta Value
#define AT91C_WDTC_WDDBGHLT   ((unsigned int) 0x1 << 28) // (WDTC) Watchdog Debug Halt
#define AT91C_WDTC_WDIDLEHLT  ((unsigned int) 0x1 << 29) // (WDTC) Watchdog Idle Halt
// -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- 
#define AT91C_WDTC_WDUNF      ((unsigned int) 0x1 <<  0) // (WDTC) Watchdog Underflow
#define AT91C_WDTC_WDERR      ((unsigned int) 0x1 <<  1) // (WDTC) Watchdog Error

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface
// *****************************************************************************
typedef struct _AT91S_RTC {
	AT91_REG	 RTC_CR; 	// Control Register
	AT91_REG	 RTC_MR; 	// Mode Register
	AT91_REG	 RTC_TIMR; 	// Time Register
	AT91_REG	 RTC_CALR; 	// Calendar Register
	AT91_REG	 RTC_TIMALR; 	// Time Alarm Register
	AT91_REG	 RTC_CALALR; 	// Calendar Alarm Register
	AT91_REG	 RTC_SR; 	// Status Register
	AT91_REG	 RTC_SCCR; 	// Status Clear Command Register
	AT91_REG	 RTC_IER; 	// Interrupt Enable Register
	AT91_REG	 RTC_IDR; 	// Interrupt Disable Register
	AT91_REG	 RTC_IMR; 	// Interrupt Mask Register
	AT91_REG	 RTC_VER; 	// Valid Entry Register
} AT91S_RTC, *AT91PS_RTC;

// -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- 
#define AT91C_RTC_UPDTIM      ((unsigned int) 0x1 <<  0) // (RTC) Update Request Time Register
#define AT91C_RTC_UPDCAL      ((unsigned int) 0x1 <<  1) // (RTC) Update Request Calendar Register
#define AT91C_RTC_TIMEVSEL    ((unsigned int) 0x3 <<  8) // (RTC) Time Event Selection
#define 	AT91C_RTC_TIMEVSEL_MINUTE               ((unsigned int) 0x0 <<  8) // (RTC) Minute change.
#define 	AT91C_RTC_TIMEVSEL_HOUR                 ((unsigned int) 0x1 <<  8) // (RTC) Hour change.
#define 	AT91C_RTC_TIMEVSEL_DAY24                ((unsigned int) 0x2 <<  8) // (RTC) Every day at midnight.
#define 	AT91C_RTC_TIMEVSEL_DAY12                ((unsigned int) 0x3 <<  8) // (RTC) Every day at noon.
#define AT91C_RTC_CALEVSEL    ((unsigned int) 0x3 << 16) // (RTC) Calendar Event Selection
#define 	AT91C_RTC_CALEVSEL_WEEK                 ((unsigned int) 0x0 << 16) // (RTC) Week change (every Monday at time 00:00:00).
#define 	AT91C_RTC_CALEVSEL_MONTH                ((unsigned int) 0x1 << 16) // (RTC) Month change (every 01 of each month at time 00:00:00).
#define 	AT91C_RTC_CALEVSEL_YEAR                 ((unsigned int) 0x2 << 16) // (RTC) Year change (every January 1 at time 00:00:00).
// -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- 
#define AT91C_RTC_HRMOD       ((unsigned int) 0x1 <<  0) // (RTC) 12-24 hour Mode
// -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- 
#define AT91C_RTC_SEC         ((unsigned int) 0x7F <<  0) // (RTC) Current Second
#define AT91C_RTC_MIN         ((unsigned int) 0x7F <<  8) // (RTC) Current Minute
#define AT91C_RTC_HOUR        ((unsigned int) 0x3F << 16) // (RTC) Current Hour
#define AT91C_RTC_AMPM        ((unsigned int) 0x1 << 22) // (RTC) Ante Meridiem, Post Meridiem Indicator
// -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- 
#define AT91C_RTC_CENT        ((unsigned int) 0x3F <<  0) // (RTC) Current Century
#define AT91C_RTC_YEAR        ((unsigned int) 0xFF <<  8) // (RTC) Current Year
#define AT91C_RTC_MONTH       ((unsigned int) 0x1F << 16) // (RTC) Current Month
#define AT91C_RTC_DAY         ((unsigned int) 0x7 << 21) // (RTC) Current Day
#define AT91C_RTC_DATE        ((unsigned int) 0x3F << 24) // (RTC) Current Date
// -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- 
#define AT91C_RTC_SECEN       ((unsigned int) 0x1 <<  7) // (RTC) Second Alarm Enable
#define AT91C_RTC_MINEN       ((unsigned int) 0x1 << 15) // (RTC) Minute Alarm
#define AT91C_RTC_HOUREN      ((unsigned int) 0x1 << 23) // (RTC) Current Hour
// -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- 
#define AT91C_RTC_MONTHEN     ((unsigned int) 0x1 << 23) // (RTC) Month Alarm Enable
#define AT91C_RTC_DATEEN      ((unsigned int) 0x1 << 31) // (RTC) Date Alarm Enable
// -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- 
#define AT91C_RTC_ACKUPD      ((unsigned int) 0x1 <<  0) // (RTC) Acknowledge for Update
#define AT91C_RTC_ALARM       ((unsigned int) 0x1 <<  1) // (RTC) Alarm Flag
#define AT91C_RTC_SECEV       ((unsigned int) 0x1 <<  2) // (RTC) Second Event
#define AT91C_RTC_TIMEV       ((unsigned int) 0x1 <<  3) // (RTC) Time Event
#define AT91C_RTC_CALEV       ((unsigned int) 0x1 <<  4) // (RTC) Calendar event
// -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- 
// -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- 
// -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- 
// -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- 
// -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- 
#define AT91C_RTC_NVTIM       ((unsigned int) 0x1 <<  0) // (RTC) Non valid Time
#define AT91C_RTC_NVCAL       ((unsigned int) 0x1 <<  1) // (RTC) Non valid Calendar
#define AT91C_RTC_NVTIMALR    ((unsigned int) 0x1 <<  2) // (RTC) Non valid time Alarm
#define AT91C_RTC_NVCALALR    ((unsigned int) 0x1 <<  3) // (RTC) Nonvalid Calendar Alarm

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Analog to Digital Convertor
// *****************************************************************************
typedef struct _AT91S_ADC {
	AT91_REG	 ADC_CR; 	// ADC Control Register
	AT91_REG	 ADC_MR; 	// ADC Mode Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 ADC_CHER; 	// ADC Channel Enable Register
	AT91_REG	 ADC_CHDR; 	// ADC Channel Disable Register
	AT91_REG	 ADC_CHSR; 	// ADC Channel Status Register
	AT91_REG	 ADC_SR; 	// ADC Status Register
	AT91_REG	 ADC_LCDR; 	// ADC Last Converted Data Register
	AT91_REG	 ADC_IER; 	// ADC Interrupt Enable Register
	AT91_REG	 ADC_IDR; 	// ADC Interrupt Disable Register
	AT91_REG	 ADC_IMR; 	// ADC Interrupt Mask Register
	AT91_REG	 ADC_CDR0; 	// ADC Channel Data Register 0
	AT91_REG	 ADC_CDR1; 	// ADC Channel Data Register 1
	AT91_REG	 ADC_CDR2; 	// ADC Channel Data Register 2
	AT91_REG	 ADC_CDR3; 	// ADC Channel Data Register 3
	AT91_REG	 ADC_CDR4; 	// ADC Channel Data Register 4
	AT91_REG	 ADC_CDR5; 	// ADC Channel Data Register 5
	AT91_REG	 ADC_CDR6; 	// ADC Channel Data Register 6
	AT91_REG	 ADC_CDR7; 	// ADC Channel Data Register 7
	AT91_REG	 Reserved1[5]; 	// 
	AT91_REG	 ADC_ACR; 	// Analog Control Register
	AT91_REG	 ADC_EMR; 	// Extended Mode Register
	AT91_REG	 Reserved2[32]; 	// 
	AT91_REG	 ADC_ADDRSIZE; 	// ADC ADDRSIZE REGISTER 
	AT91_REG	 ADC_IPNAME1; 	// ADC IPNAME1 REGISTER 
	AT91_REG	 ADC_IPNAME2; 	// ADC IPNAME2 REGISTER 
	AT91_REG	 ADC_FEATURES; 	// ADC FEATURES REGISTER 
	AT91_REG	 ADC_VER; 	// ADC VERSION REGISTER
	AT91_REG	 ADC_RPR; 	// Receive Pointer Register
	AT91_REG	 ADC_RCR; 	// Receive Counter Register
	AT91_REG	 ADC_TPR; 	// Transmit Pointer Register
	AT91_REG	 ADC_TCR; 	// Transmit Counter Register
	AT91_REG	 ADC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 ADC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 ADC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 ADC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 ADC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 ADC_PTSR; 	// PDC Transfer Status Register
} AT91S_ADC, *AT91PS_ADC;

// -------- ADC_CR : (ADC Offset: 0x0) ADC Control Register -------- 
#define AT91C_ADC_SWRST       ((unsigned int) 0x1 <<  0) // (ADC) Software Reset
#define AT91C_ADC_START       ((unsigned int) 0x1 <<  1) // (ADC) Start Conversion
// -------- ADC_MR : (ADC Offset: 0x4) ADC Mode Register -------- 
#define AT91C_ADC_TRGEN       ((unsigned int) 0x1 <<  0) // (ADC) Trigger Enable
#define 	AT91C_ADC_TRGEN_DIS                  ((unsigned int) 0x0) // (ADC) Hradware triggers are disabled. Starting a conversion is only possible by software
#define 	AT91C_ADC_TRGEN_EN                   ((unsigned int) 0x1) // (ADC) Hardware trigger selected by TRGSEL field is enabled.
#define AT91C_ADC_TRGSEL      ((unsigned int) 0x7 <<  1) // (ADC) Trigger Selection
#define 	AT91C_ADC_TRGSEL_EXT                  ((unsigned int) 0x0 <<  1) // (ADC) Selected TRGSEL = External Trigger
#define 	AT91C_ADC_TRGSEL_TIOA0                ((unsigned int) 0x1 <<  1) // (ADC) Selected TRGSEL = TIAO0
#define 	AT91C_ADC_TRGSEL_TIOA1                ((unsigned int) 0x2 <<  1) // (ADC) Selected TRGSEL = TIAO1
#define 	AT91C_ADC_TRGSEL_TIOA2                ((unsigned int) 0x3 <<  1) // (ADC) Selected TRGSEL = TIAO2
#define 	AT91C_ADC_TRGSEL_PWM0_TRIG            ((unsigned int) 0x4 <<  1) // (ADC) Selected TRGSEL = PWM trigger
#define 	AT91C_ADC_TRGSEL_PWM1_TRIG            ((unsigned int) 0x5 <<  1) // (ADC) Selected TRGSEL = PWM Trigger
#define 	AT91C_ADC_TRGSEL_RESERVED             ((unsigned int) 0x6 <<  1) // (ADC) Selected TRGSEL = Reserved
#define AT91C_ADC_LOWRES      ((unsigned int) 0x1 <<  4) // (ADC) Resolution.
#define 	AT91C_ADC_LOWRES_10_BIT               ((unsigned int) 0x0 <<  4) // (ADC) 10-bit resolution
#define 	AT91C_ADC_LOWRES_8_BIT                ((unsigned int) 0x1 <<  4) // (ADC) 8-bit resolution
#define AT91C_ADC_SLEEP       ((unsigned int) 0x1 <<  5) // (ADC) Sleep Mode
#define 	AT91C_ADC_SLEEP_NORMAL_MODE          ((unsigned int) 0x0 <<  5) // (ADC) Normal Mode
#define 	AT91C_ADC_SLEEP_MODE                 ((unsigned int) 0x1 <<  5) // (ADC) Sleep Mode
#define AT91C_ADC_PRESCAL     ((unsigned int) 0x3F <<  8) // (ADC) Prescaler rate selection
#define AT91C_ADC_STARTUP     ((unsigned int) 0x1F << 16) // (ADC) Startup Time
#define AT91C_ADC_SHTIM       ((unsigned int) 0xF << 24) // (ADC) Sample & Hold Time
// -------- 	ADC_CHER : (ADC Offset: 0x10) ADC Channel Enable Register -------- 
#define AT91C_ADC_CH0         ((unsigned int) 0x1 <<  0) // (ADC) Channel 0
#define AT91C_ADC_CH1         ((unsigned int) 0x1 <<  1) // (ADC) Channel 1
#define AT91C_ADC_CH2         ((unsigned int) 0x1 <<  2) // (ADC) Channel 2
#define AT91C_ADC_CH3         ((unsigned int) 0x1 <<  3) // (ADC) Channel 3
#define AT91C_ADC_CH4         ((unsigned int) 0x1 <<  4) // (ADC) Channel 4
#define AT91C_ADC_CH5         ((unsigned int) 0x1 <<  5) // (ADC) Channel 5
#define AT91C_ADC_CH6         ((unsigned int) 0x1 <<  6) // (ADC) Channel 6
#define AT91C_ADC_CH7         ((unsigned int) 0x1 <<  7) // (ADC) Channel 7
// -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- 
// -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- 
// -------- ADC_SR : (ADC Offset: 0x1c) ADC Status Register -------- 
#define AT91C_ADC_EOC0        ((unsigned int) 0x1 <<  0) // (ADC) End of Conversion
#define AT91C_ADC_EOC1        ((unsigned int) 0x1 <<  1) // (ADC) End of Conversion
#define AT91C_ADC_EOC2        ((unsigned int) 0x1 <<  2) // (ADC) End of Conversion
#define AT91C_ADC_EOC3        ((unsigned int) 0x1 <<  3) // (ADC) End of Conversion
#define AT91C_ADC_EOC4        ((unsigned int) 0x1 <<  4) // (ADC) End of Conversion
#define AT91C_ADC_EOC5        ((unsigned int) 0x1 <<  5) // (ADC) End of Conversion
#define AT91C_ADC_EOC6        ((unsigned int) 0x1 <<  6) // (ADC) End of Conversion
#define AT91C_ADC_EOC7        ((unsigned int) 0x1 <<  7) // (ADC) End of Conversion
#define AT91C_ADC_OVRE0       ((unsigned int) 0x1 <<  8) // (ADC) Overrun Error
#define AT91C_ADC_OVRE1       ((unsigned int) 0x1 <<  9) // (ADC) Overrun Error
#define AT91C_ADC_OVRE2       ((unsigned int) 0x1 << 10) // (ADC) Overrun Error
#define AT91C_ADC_OVRE3       ((unsigned int) 0x1 << 11) // (ADC) Overrun Error
#define AT91C_ADC_OVRE4       ((unsigned int) 0x1 << 12) // (ADC) Overrun Error
#define AT91C_ADC_OVRE5       ((unsigned int) 0x1 << 13) // (ADC) Overrun Error
#define AT91C_ADC_OVRE6       ((unsigned int) 0x1 << 14) // (ADC) Overrun Error
#define AT91C_ADC_OVRE7       ((unsigned int) 0x1 << 15) // (ADC) Overrun Error
#define AT91C_ADC_DRDY        ((unsigned int) 0x1 << 16) // (ADC) Data Ready
#define AT91C_ADC_GOVRE       ((unsigned int) 0x1 << 17) // (ADC) General Overrun
#define AT91C_ADC_ENDRX       ((unsigned int) 0x1 << 18) // (ADC) End of Receiver Transfer
#define AT91C_ADC_RXBUFF      ((unsigned int) 0x1 << 19) // (ADC) RXBUFF Interrupt
// -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- 
#define AT91C_ADC_LDATA       ((unsigned int) 0x3FF <<  0) // (ADC) Last Data Converted
// -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- 
// -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- 
// -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- 
// -------- ADC_CDR0 : (ADC Offset: 0x30) ADC Channel Data Register 0 -------- 
#define AT91C_ADC_DATA        ((unsigned int) 0x3FF <<  0) // (ADC) Converted Data
// -------- ADC_CDR1 : (ADC Offset: 0x34) ADC Channel Data Register 1 -------- 
// -------- ADC_CDR2 : (ADC Offset: 0x38) ADC Channel Data Register 2 -------- 
// -------- ADC_CDR3 : (ADC Offset: 0x3c) ADC Channel Data Register 3 -------- 
// -------- ADC_CDR4 : (ADC Offset: 0x40) ADC Channel Data Register 4 -------- 
// -------- ADC_CDR5 : (ADC Offset: 0x44) ADC Channel Data Register 5 -------- 
// -------- ADC_CDR6 : (ADC Offset: 0x48) ADC Channel Data Register 6 -------- 
// -------- ADC_CDR7 : (ADC Offset: 0x4c) ADC Channel Data Register 7 -------- 
// -------- ADC_ACR : (ADC Offset: 0x64) ADC Analog Controler Register -------- 
#define AT91C_ADC_GAIN        ((unsigned int) 0x3 <<  0) // (ADC) Input Gain
#define AT91C_ADC_IBCTL       ((unsigned int) 0x3 <<  6) // (ADC) Bias Current Control
#define 	AT91C_ADC_IBCTL_00                   ((unsigned int) 0x0 <<  6) // (ADC) typ - 20%
#define 	AT91C_ADC_IBCTL_01                   ((unsigned int) 0x1 <<  6) // (ADC) typ
#define 	AT91C_ADC_IBCTL_10                   ((unsigned int) 0x2 <<  6) // (ADC) typ + 20%
#define 	AT91C_ADC_IBCTL_11                   ((unsigned int) 0x3 <<  6) // (ADC) typ + 40%
#define AT91C_ADC_DIFF        ((unsigned int) 0x1 << 16) // (ADC) Differential Mode
#define AT91C_ADC_OFFSET      ((unsigned int) 0x1 << 17) // (ADC) Input OFFSET
// -------- ADC_EMR : (ADC Offset: 0x68) ADC Extended Mode Register -------- 
#define AT91C_OFFMODES        ((unsigned int) 0x1 <<  0) // (ADC) Off Mode if
#define AT91C_OFF_MODE_STARTUP_TIME ((unsigned int) 0x1 << 16) // (ADC) Startup Time
// -------- ADC_VER : (ADC Offset: 0xfc) ADC VER -------- 
#define AT91C_ADC_VER         ((unsigned int) 0xF <<  0) // (ADC) ADC VER

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface
// *****************************************************************************
typedef struct _AT91S_TC {
	AT91_REG	 TC_CCR; 	// Channel Control Register
	AT91_REG	 TC_CMR; 	// Channel Mode Register (Capture Mode / Waveform Mode)
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 TC_CV; 	// Counter Value
	AT91_REG	 TC_RA; 	// Register A
	AT91_REG	 TC_RB; 	// Register B
	AT91_REG	 TC_RC; 	// Register C
	AT91_REG	 TC_SR; 	// Status Register
	AT91_REG	 TC_IER; 	// Interrupt Enable Register
	AT91_REG	 TC_IDR; 	// Interrupt Disable Register
	AT91_REG	 TC_IMR; 	// Interrupt Mask Register
} AT91S_TC, *AT91PS_TC;

// -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- 
#define AT91C_TC_CLKEN        ((unsigned int) 0x1 <<  0) // (TC) Counter Clock Enable Command
#define AT91C_TC_CLKDIS       ((unsigned int) 0x1 <<  1) // (TC) Counter Clock Disable Command
#define AT91C_TC_SWTRG        ((unsigned int) 0x1 <<  2) // (TC) Software Trigger Command
// -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- 
#define AT91C_TC_CLKS         ((unsigned int) 0x7 <<  0) // (TC) Clock Selection
#define 	AT91C_TC_CLKS_TIMER_DIV1_CLOCK     ((unsigned int) 0x0) // (TC) Clock selected: TIMER_DIV1_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV2_CLOCK     ((unsigned int) 0x1) // (TC) Clock selected: TIMER_DIV2_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV3_CLOCK     ((unsigned int) 0x2) // (TC) Clock selected: TIMER_DIV3_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV4_CLOCK     ((unsigned int) 0x3) // (TC) Clock selected: TIMER_DIV4_CLOCK
#define 	AT91C_TC_CLKS_TIMER_DIV5_CLOCK     ((unsigned int) 0x4) // (TC) Clock selected: TIMER_DIV5_CLOCK
#define 	AT91C_TC_CLKS_XC0                  ((unsigned int) 0x5) // (TC) Clock selected: XC0
#define 	AT91C_TC_CLKS_XC1                  ((unsigned int) 0x6) // (TC) Clock selected: XC1
#define 	AT91C_TC_CLKS_XC2                  ((unsigned int) 0x7) // (TC) Clock selected: XC2
#define AT91C_TC_CLKI         ((unsigned int) 0x1 <<  3) // (TC) Clock Invert
#define AT91C_TC_BURST        ((unsigned int) 0x3 <<  4) // (TC) Burst Signal Selection
#define 	AT91C_TC_BURST_NONE                 ((unsigned int) 0x0 <<  4) // (TC) The clock is not gated by an external signal
#define 	AT91C_TC_BURST_XC0                  ((unsigned int) 0x1 <<  4) // (TC) XC0 is ANDed with the selected clock
#define 	AT91C_TC_BURST_XC1                  ((unsigned int) 0x2 <<  4) // (TC) XC1 is ANDed with the selected clock
#define 	AT91C_TC_BURST_XC2                  ((unsigned int) 0x3 <<  4) // (TC) XC2 is ANDed with the selected clock
#define AT91C_TC_CPCSTOP      ((unsigned int) 0x1 <<  6) // (TC) Counter Clock Stopped with RC Compare
#define AT91C_TC_LDBSTOP      ((unsigned int) 0x1 <<  6) // (TC) Counter Clock Stopped with RB Loading
#define AT91C_TC_CPCDIS       ((unsigned int) 0x1 <<  7) // (TC) Counter Clock Disable with RC Compare
#define AT91C_TC_LDBDIS       ((unsigned int) 0x1 <<  7) // (TC) Counter Clock Disabled with RB Loading
#define AT91C_TC_ETRGEDG      ((unsigned int) 0x3 <<  8) // (TC) External Trigger Edge Selection
#define 	AT91C_TC_ETRGEDG_NONE                 ((unsigned int) 0x0 <<  8) // (TC) Edge: None
#define 	AT91C_TC_ETRGEDG_RISING               ((unsigned int) 0x1 <<  8) // (TC) Edge: rising edge
#define 	AT91C_TC_ETRGEDG_FALLING              ((unsigned int) 0x2 <<  8) // (TC) Edge: falling edge
#define 	AT91C_TC_ETRGEDG_BOTH                 ((unsigned int) 0x3 <<  8) // (TC) Edge: each edge
#define AT91C_TC_EEVTEDG      ((unsigned int) 0x3 <<  8) // (TC) External Event Edge Selection
#define 	AT91C_TC_EEVTEDG_NONE                 ((unsigned int) 0x0 <<  8) // (TC) Edge: None
#define 	AT91C_TC_EEVTEDG_RISING               ((unsigned int) 0x1 <<  8) // (TC) Edge: rising edge
#define 	AT91C_TC_EEVTEDG_FALLING              ((unsigned int) 0x2 <<  8) // (TC) Edge: falling edge
#define 	AT91C_TC_EEVTEDG_BOTH                 ((unsigned int) 0x3 <<  8) // (TC) Edge: each edge
#define AT91C_TC_EEVT         ((unsigned int) 0x3 << 10) // (TC) External Event  Selection
#define 	AT91C_TC_EEVT_TIOB                 ((unsigned int) 0x0 << 10) // (TC) Signal selected as external event: TIOB TIOB direction: input
#define 	AT91C_TC_EEVT_XC0                  ((unsigned int) 0x1 << 10) // (TC) Signal selected as external event: XC0 TIOB direction: output
#define 	AT91C_TC_EEVT_XC1                  ((unsigned int) 0x2 << 10) // (TC) Signal selected as external event: XC1 TIOB direction: output
#define 	AT91C_TC_EEVT_XC2                  ((unsigned int) 0x3 << 10) // (TC) Signal selected as external event: XC2 TIOB direction: output
#define AT91C_TC_ABETRG       ((unsigned int) 0x1 << 10) // (TC) TIOA or TIOB External Trigger Selection
#define AT91C_TC_ENETRG       ((unsigned int) 0x1 << 12) // (TC) External Event Trigger enable
#define AT91C_TC_WAVESEL      ((unsigned int) 0x3 << 13) // (TC) Waveform  Selection
#define 	AT91C_TC_WAVESEL_UP                   ((unsigned int) 0x0 << 13) // (TC) UP mode without atomatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UPDOWN               ((unsigned int) 0x1 << 13) // (TC) UPDOWN mode without automatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UP_AUTO              ((unsigned int) 0x2 << 13) // (TC) UP mode with automatic trigger on RC Compare
#define 	AT91C_TC_WAVESEL_UPDOWN_AUTO          ((unsigned int) 0x3 << 13) // (TC) UPDOWN mode with automatic trigger on RC Compare
#define AT91C_TC_CPCTRG       ((unsigned int) 0x1 << 14) // (TC) RC Compare Trigger Enable
#define AT91C_TC_WAVE         ((unsigned int) 0x1 << 15) // (TC) 
#define AT91C_TC_ACPA         ((unsigned int) 0x3 << 16) // (TC) RA Compare Effect on TIOA
#define 	AT91C_TC_ACPA_NONE                 ((unsigned int) 0x0 << 16) // (TC) Effect: none
#define 	AT91C_TC_ACPA_SET                  ((unsigned int) 0x1 << 16) // (TC) Effect: set
#define 	AT91C_TC_ACPA_CLEAR                ((unsigned int) 0x2 << 16) // (TC) Effect: clear
#define 	AT91C_TC_ACPA_TOGGLE               ((unsigned int) 0x3 << 16) // (TC) Effect: toggle
#define AT91C_TC_LDRA         ((unsigned int) 0x3 << 16) // (TC) RA Loading Selection
#define 	AT91C_TC_LDRA_NONE                 ((unsigned int) 0x0 << 16) // (TC) Edge: None
#define 	AT91C_TC_LDRA_RISING               ((unsigned int) 0x1 << 16) // (TC) Edge: rising edge of TIOA
#define 	AT91C_TC_LDRA_FALLING              ((unsigned int) 0x2 << 16) // (TC) Edge: falling edge of TIOA
#define 	AT91C_TC_LDRA_BOTH                 ((unsigned int) 0x3 << 16) // (TC) Edge: each edge of TIOA
#define AT91C_TC_ACPC         ((unsigned int) 0x3 << 18) // (TC) RC Compare Effect on TIOA
#define 	AT91C_TC_ACPC_NONE                 ((unsigned int) 0x0 << 18) // (TC) Effect: none
#define 	AT91C_TC_ACPC_SET                  ((unsigned int) 0x1 << 18) // (TC) Effect: set
#define 	AT91C_TC_ACPC_CLEAR                ((unsigned int) 0x2 << 18) // (TC) Effect: clear
#define 	AT91C_TC_ACPC_TOGGLE               ((unsigned int) 0x3 << 18) // (TC) Effect: toggle
#define AT91C_TC_LDRB         ((unsigned int) 0x3 << 18) // (TC) RB Loading Selection
#define 	AT91C_TC_LDRB_NONE                 ((unsigned int) 0x0 << 18) // (TC) Edge: None
#define 	AT91C_TC_LDRB_RISING               ((unsigned int) 0x1 << 18) // (TC) Edge: rising edge of TIOA
#define 	AT91C_TC_LDRB_FALLING              ((unsigned int) 0x2 << 18) // (TC) Edge: falling edge of TIOA
#define 	AT91C_TC_LDRB_BOTH                 ((unsigned int) 0x3 << 18) // (TC) Edge: each edge of TIOA
#define AT91C_TC_AEEVT        ((unsigned int) 0x3 << 20) // (TC) External Event Effect on TIOA
#define 	AT91C_TC_AEEVT_NONE                 ((unsigned int) 0x0 << 20) // (TC) Effect: none
#define 	AT91C_TC_AEEVT_SET                  ((unsigned int) 0x1 << 20) // (TC) Effect: set
#define 	AT91C_TC_AEEVT_CLEAR                ((unsigned int) 0x2 << 20) // (TC) Effect: clear
#define 	AT91C_TC_AEEVT_TOGGLE               ((unsigned int) 0x3 << 20) // (TC) Effect: toggle
#define AT91C_TC_ASWTRG       ((unsigned int) 0x3 << 22) // (TC) Software Trigger Effect on TIOA
#define 	AT91C_TC_ASWTRG_NONE                 ((unsigned int) 0x0 << 22) // (TC) Effect: none
#define 	AT91C_TC_ASWTRG_SET                  ((unsigned int) 0x1 << 22) // (TC) Effect: set
#define 	AT91C_TC_ASWTRG_CLEAR                ((unsigned int) 0x2 << 22) // (TC) Effect: clear
#define 	AT91C_TC_ASWTRG_TOGGLE               ((unsigned int) 0x3 << 22) // (TC) Effect: toggle
#define AT91C_TC_BCPB         ((unsigned int) 0x3 << 24) // (TC) RB Compare Effect on TIOB
#define 	AT91C_TC_BCPB_NONE                 ((unsigned int) 0x0 << 24) // (TC) Effect: none
#define 	AT91C_TC_BCPB_SET                  ((unsigned int) 0x1 << 24) // (TC) Effect: set
#define 	AT91C_TC_BCPB_CLEAR                ((unsigned int) 0x2 << 24) // (TC) Effect: clear
#define 	AT91C_TC_BCPB_TOGGLE               ((unsigned int) 0x3 << 24) // (TC) Effect: toggle
#define AT91C_TC_BCPC         ((unsigned int) 0x3 << 26) // (TC) RC Compare Effect on TIOB
#define 	AT91C_TC_BCPC_NONE                 ((unsigned int) 0x0 << 26) // (TC) Effect: none
#define 	AT91C_TC_BCPC_SET                  ((unsigned int) 0x1 << 26) // (TC) Effect: set
#define 	AT91C_TC_BCPC_CLEAR                ((unsigned int) 0x2 << 26) // (TC) Effect: clear
#define 	AT91C_TC_BCPC_TOGGLE               ((unsigned int) 0x3 << 26) // (TC) Effect: toggle
#define AT91C_TC_BEEVT        ((unsigned int) 0x3 << 28) // (TC) External Event Effect on TIOB
#define 	AT91C_TC_BEEVT_NONE                 ((unsigned int) 0x0 << 28) // (TC) Effect: none
#define 	AT91C_TC_BEEVT_SET                  ((unsigned int) 0x1 << 28) // (TC) Effect: set
#define 	AT91C_TC_BEEVT_CLEAR                ((unsigned int) 0x2 << 28) // (TC) Effect: clear
#define 	AT91C_TC_BEEVT_TOGGLE               ((unsigned int) 0x3 << 28) // (TC) Effect: toggle
#define AT91C_TC_BSWTRG       ((unsigned int) 0x3 << 30) // (TC) Software Trigger Effect on TIOB
#define 	AT91C_TC_BSWTRG_NONE                 ((unsigned int) 0x0 << 30) // (TC) Effect: none
#define 	AT91C_TC_BSWTRG_SET                  ((unsigned int) 0x1 << 30) // (TC) Effect: set
#define 	AT91C_TC_BSWTRG_CLEAR                ((unsigned int) 0x2 << 30) // (TC) Effect: clear
#define 	AT91C_TC_BSWTRG_TOGGLE               ((unsigned int) 0x3 << 30) // (TC) Effect: toggle
// -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- 
#define AT91C_TC_COVFS        ((unsigned int) 0x1 <<  0) // (TC) Counter Overflow
#define AT91C_TC_LOVRS        ((unsigned int) 0x1 <<  1) // (TC) Load Overrun
#define AT91C_TC_CPAS         ((unsigned int) 0x1 <<  2) // (TC) RA Compare
#define AT91C_TC_CPBS         ((unsigned int) 0x1 <<  3) // (TC) RB Compare
#define AT91C_TC_CPCS         ((unsigned int) 0x1 <<  4) // (TC) RC Compare
#define AT91C_TC_LDRAS        ((unsigned int) 0x1 <<  5) // (TC) RA Loading
#define AT91C_TC_LDRBS        ((unsigned int) 0x1 <<  6) // (TC) RB Loading
#define AT91C_TC_ETRGS        ((unsigned int) 0x1 <<  7) // (TC) External Trigger
#define AT91C_TC_CLKSTA       ((unsigned int) 0x1 << 16) // (TC) Clock Enabling
#define AT91C_TC_MTIOA        ((unsigned int) 0x1 << 17) // (TC) TIOA Mirror
#define AT91C_TC_MTIOB        ((unsigned int) 0x1 << 18) // (TC) TIOA Mirror
// -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- 
// -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- 
// -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Timer Counter Interface
// *****************************************************************************
typedef struct _AT91S_TCB {
	AT91S_TC	 TCB_TC0; 	// TC Channel 0
	AT91_REG	 Reserved0[4]; 	// 
	AT91S_TC	 TCB_TC1; 	// TC Channel 1
	AT91_REG	 Reserved1[4]; 	// 
	AT91S_TC	 TCB_TC2; 	// TC Channel 2
	AT91_REG	 Reserved2[4]; 	// 
	AT91_REG	 TCB_BCR; 	// TC Block Control Register
	AT91_REG	 TCB_BMR; 	// TC Block Mode Register
	AT91_REG	 Reserved3[9]; 	// 
	AT91_REG	 TCB_ADDRSIZE; 	// TC ADDRSIZE REGISTER 
	AT91_REG	 TCB_IPNAME1; 	// TC IPNAME1 REGISTER 
	AT91_REG	 TCB_IPNAME2; 	// TC IPNAME2 REGISTER 
	AT91_REG	 TCB_FEATURES; 	// TC FEATURES REGISTER 
	AT91_REG	 TCB_VER; 	//  Version Register
} AT91S_TCB, *AT91PS_TCB;

// -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- 
#define AT91C_TCB_SYNC        ((unsigned int) 0x1 <<  0) // (TCB) Synchro Command
// -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- 
#define AT91C_TCB_TC0XC0S     ((unsigned int) 0x3 <<  0) // (TCB) External Clock Signal 0 Selection
#define 	AT91C_TCB_TC0XC0S_TCLK0                ((unsigned int) 0x0) // (TCB) TCLK0 connected to XC0
#define 	AT91C_TCB_TC0XC0S_NONE                 ((unsigned int) 0x1) // (TCB) None signal connected to XC0
#define 	AT91C_TCB_TC0XC0S_TIOA1                ((unsigned int) 0x2) // (TCB) TIOA1 connected to XC0
#define 	AT91C_TCB_TC0XC0S_TIOA2                ((unsigned int) 0x3) // (TCB) TIOA2 connected to XC0
#define AT91C_TCB_TC1XC1S     ((unsigned int) 0x3 <<  2) // (TCB) External Clock Signal 1 Selection
#define 	AT91C_TCB_TC1XC1S_TCLK1                ((unsigned int) 0x0 <<  2) // (TCB) TCLK1 connected to XC1
#define 	AT91C_TCB_TC1XC1S_NONE                 ((unsigned int) 0x1 <<  2) // (TCB) None signal connected to XC1
#define 	AT91C_TCB_TC1XC1S_TIOA0                ((unsigned int) 0x2 <<  2) // (TCB) TIOA0 connected to XC1
#define 	AT91C_TCB_TC1XC1S_TIOA2                ((unsigned int) 0x3 <<  2) // (TCB) TIOA2 connected to XC1
#define AT91C_TCB_TC2XC2S     ((unsigned int) 0x3 <<  4) // (TCB) External Clock Signal 2 Selection
#define 	AT91C_TCB_TC2XC2S_TCLK2                ((unsigned int) 0x0 <<  4) // (TCB) TCLK2 connected to XC2
#define 	AT91C_TCB_TC2XC2S_NONE                 ((unsigned int) 0x1 <<  4) // (TCB) None signal connected to XC2
#define 	AT91C_TCB_TC2XC2S_TIOA0                ((unsigned int) 0x2 <<  4) // (TCB) TIOA0 connected to XC2
#define 	AT91C_TCB_TC2XC2S_TIOA1                ((unsigned int) 0x3 <<  4) // (TCB) TIOA2 connected to XC2

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Embedded Flash Controller 2.0
// *****************************************************************************
typedef struct _AT91S_EFC {
	AT91_REG	 EFC_FMR; 	// EFC Flash Mode Register
	AT91_REG	 EFC_FCR; 	// EFC Flash Command Register
	AT91_REG	 EFC_FSR; 	// EFC Flash Status Register
	AT91_REG	 EFC_FRR; 	// EFC Flash Result Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 EFC_FVR; 	// EFC Flash Version Register
} AT91S_EFC, *AT91PS_EFC;

// -------- EFC_FMR : (EFC Offset: 0x0) EFC Flash Mode Register -------- 
#define AT91C_EFC_FRDY        ((unsigned int) 0x1 <<  0) // (EFC) Ready Interrupt Enable
#define AT91C_EFC_FWS         ((unsigned int) 0xF <<  8) // (EFC) Flash Wait State.
#define 	AT91C_EFC_FWS_0WS                  ((unsigned int) 0x0 <<  8) // (EFC) 0 Wait State
#define 	AT91C_EFC_FWS_1WS                  ((unsigned int) 0x1 <<  8) // (EFC) 1 Wait State
#define 	AT91C_EFC_FWS_2WS                  ((unsigned int) 0x2 <<  8) // (EFC) 2 Wait States
#define 	AT91C_EFC_FWS_3WS                  ((unsigned int) 0x3 <<  8) // (EFC) 3 Wait States
// -------- EFC_FCR : (EFC Offset: 0x4) EFC Flash Command Register -------- 
#define AT91C_EFC_FCMD        ((unsigned int) 0xFF <<  0) // (EFC) Flash Command
#define 	AT91C_EFC_FCMD_GETD                 ((unsigned int) 0x0) // (EFC) Get Flash Descriptor
#define 	AT91C_EFC_FCMD_WP                   ((unsigned int) 0x1) // (EFC) Write Page
#define 	AT91C_EFC_FCMD_WPL                  ((unsigned int) 0x2) // (EFC) Write Page and Lock
#define 	AT91C_EFC_FCMD_EWP                  ((unsigned int) 0x3) // (EFC) Erase Page and Write Page
#define 	AT91C_EFC_FCMD_EWPL                 ((unsigned int) 0x4) // (EFC) Erase Page and Write Page then Lock
#define 	AT91C_EFC_FCMD_EA                   ((unsigned int) 0x5) // (EFC) Erase All
#define 	AT91C_EFC_FCMD_EPL                  ((unsigned int) 0x6) // (EFC) Erase Plane
#define 	AT91C_EFC_FCMD_EPA                  ((unsigned int) 0x7) // (EFC) Erase Pages
#define 	AT91C_EFC_FCMD_SLB                  ((unsigned int) 0x8) // (EFC) Set Lock Bit
#define 	AT91C_EFC_FCMD_CLB                  ((unsigned int) 0x9) // (EFC) Clear Lock Bit
#define 	AT91C_EFC_FCMD_GLB                  ((unsigned int) 0xA) // (EFC) Get Lock Bit
#define 	AT91C_EFC_FCMD_SFB                  ((unsigned int) 0xB) // (EFC) Set Fuse Bit
#define 	AT91C_EFC_FCMD_CFB                  ((unsigned int) 0xC) // (EFC) Clear Fuse Bit
#define 	AT91C_EFC_FCMD_GFB                  ((unsigned int) 0xD) // (EFC) Get Fuse Bit
#define 	AT91C_EFC_FCMD_STUI                 ((unsigned int) 0xE) // (EFC) Start Read Unique ID
#define 	AT91C_EFC_FCMD_SPUI                 ((unsigned int) 0xF) // (EFC) Stop Read Unique ID
#define AT91C_EFC_FARG        ((unsigned int) 0xFFFF <<  8) // (EFC) Flash Command Argument
#define AT91C_EFC_FKEY        ((unsigned int) 0xFF << 24) // (EFC) Flash Writing Protection Key
// -------- EFC_FSR : (EFC Offset: 0x8) EFC Flash Status Register -------- 
#define AT91C_EFC_FRDY_S      ((unsigned int) 0x1 <<  0) // (EFC) Flash Ready Status
#define AT91C_EFC_FCMDE       ((unsigned int) 0x1 <<  1) // (EFC) Flash Command Error Status
#define AT91C_EFC_LOCKE       ((unsigned int) 0x1 <<  2) // (EFC) Flash Lock Error Status
// -------- EFC_FRR : (EFC Offset: 0xc) EFC Flash Result Register -------- 
#define AT91C_EFC_FVALUE      ((unsigned int) 0x0 <<  0) // (EFC) Flash Result Value

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Multimedia Card Interface
// *****************************************************************************
typedef struct _AT91S_MCI {
	AT91_REG	 MCI_CR; 	// MCI Control Register
	AT91_REG	 MCI_MR; 	// MCI Mode Register
	AT91_REG	 MCI_DTOR; 	// MCI Data Timeout Register
	AT91_REG	 MCI_SDCR; 	// MCI SD/SDIO Card Register
	AT91_REG	 MCI_ARGR; 	// MCI Argument Register
	AT91_REG	 MCI_CMDR; 	// MCI Command Register
	AT91_REG	 MCI_BLKR; 	// MCI Block Register
	AT91_REG	 MCI_CSTOR; 	// MCI Completion Signal Timeout Register
	AT91_REG	 MCI_RSPR[4]; 	// MCI Response Register
	AT91_REG	 MCI_RDR; 	// MCI Receive Data Register
	AT91_REG	 MCI_TDR; 	// MCI Transmit Data Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 MCI_SR; 	// MCI Status Register
	AT91_REG	 MCI_IER; 	// MCI Interrupt Enable Register
	AT91_REG	 MCI_IDR; 	// MCI Interrupt Disable Register
	AT91_REG	 MCI_IMR; 	// MCI Interrupt Mask Register
	AT91_REG	 MCI_DMA; 	// MCI DMA Configuration Register
	AT91_REG	 MCI_CFG; 	// MCI Configuration Register
	AT91_REG	 Reserved1[35]; 	// 
	AT91_REG	 MCI_WPCR; 	// MCI Write Protection Control Register
	AT91_REG	 MCI_WPSR; 	// MCI Write Protection Status Register
	AT91_REG	 MCI_ADDRSIZE; 	// MCI ADDRSIZE REGISTER 
	AT91_REG	 MCI_IPNAME1; 	// MCI IPNAME1 REGISTER 
	AT91_REG	 MCI_IPNAME2; 	// MCI IPNAME2 REGISTER 
	AT91_REG	 MCI_FEATURES; 	// MCI FEATURES REGISTER 
	AT91_REG	 MCI_VER; 	// MCI VERSION REGISTER 
	AT91_REG	 MCI_RPR; 	// Receive Pointer Register
	AT91_REG	 MCI_RCR; 	// Receive Counter Register
	AT91_REG	 MCI_TPR; 	// Transmit Pointer Register
	AT91_REG	 MCI_TCR; 	// Transmit Counter Register
	AT91_REG	 MCI_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 MCI_RNCR; 	// Receive Next Counter Register
	AT91_REG	 MCI_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 MCI_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 MCI_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 MCI_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved2[54]; 	// 
	AT91_REG	 MCI_FIFO; 	// MCI FIFO Aperture Register
} AT91S_MCI, *AT91PS_MCI;

// -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- 
#define AT91C_MCI_MCIEN       ((unsigned int) 0x1 <<  0) // (MCI) Multimedia Interface Enable
#define 	AT91C_MCI_MCIEN_0                    ((unsigned int) 0x0) // (MCI) No effect
#define 	AT91C_MCI_MCIEN_1                    ((unsigned int) 0x1) // (MCI) Enable the MultiMedia Interface if MCIDIS is 0
#define AT91C_MCI_MCIDIS      ((unsigned int) 0x1 <<  1) // (MCI) Multimedia Interface Disable
#define 	AT91C_MCI_MCIDIS_0                    ((unsigned int) 0x0 <<  1) // (MCI) No effect
#define 	AT91C_MCI_MCIDIS_1                    ((unsigned int) 0x1 <<  1) // (MCI) Disable the MultiMedia Interface
#define AT91C_MCI_PWSEN       ((unsigned int) 0x1 <<  2) // (MCI) Power Save Mode Enable
#define 	AT91C_MCI_PWSEN_0                    ((unsigned int) 0x0 <<  2) // (MCI) No effect
#define 	AT91C_MCI_PWSEN_1                    ((unsigned int) 0x1 <<  2) // (MCI) Enable the Power-saving mode if PWSDIS is 0.
#define AT91C_MCI_PWSDIS      ((unsigned int) 0x1 <<  3) // (MCI) Power Save Mode Disable
#define 	AT91C_MCI_PWSDIS_0                    ((unsigned int) 0x0 <<  3) // (MCI) No effect
#define 	AT91C_MCI_PWSDIS_1                    ((unsigned int) 0x1 <<  3) // (MCI) Disable the Power-saving mode.
#define AT91C_MCI_IOWAITEN    ((unsigned int) 0x1 <<  4) // (MCI) SDIO Read Wait Enable
#define 	AT91C_MCI_IOWAITEN_0                    ((unsigned int) 0x0 <<  4) // (MCI) No effect
#define 	AT91C_MCI_IOWAITEN_1                    ((unsigned int) 0x1 <<  4) // (MCI) Enables the SDIO Read Wait Operation.
#define AT91C_MCI_IOWAITDIS   ((unsigned int) 0x1 <<  5) // (MCI) SDIO Read Wait Disable
#define 	AT91C_MCI_IOWAITDIS_0                    ((unsigned int) 0x0 <<  5) // (MCI) No effect
#define 	AT91C_MCI_IOWAITDIS_1                    ((unsigned int) 0x1 <<  5) // (MCI) Disables the SDIO Read Wait Operation.
#define AT91C_MCI_SWRST       ((unsigned int) 0x1 <<  7) // (MCI) MCI Software reset
#define 	AT91C_MCI_SWRST_0                    ((unsigned int) 0x0 <<  7) // (MCI) No effect
#define 	AT91C_MCI_SWRST_1                    ((unsigned int) 0x1 <<  7) // (MCI) Resets the MCI
// -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- 
#define AT91C_MCI_CLKDIV      ((unsigned int) 0xFF <<  0) // (MCI) Clock Divider
#define AT91C_MCI_PWSDIV      ((unsigned int) 0x7 <<  8) // (MCI) Power Saving Divider
#define AT91C_MCI_RDPROOF     ((unsigned int) 0x1 << 11) // (MCI) Read Proof Enable
#define 	AT91C_MCI_RDPROOF_DISABLE              ((unsigned int) 0x0 << 11) // (MCI) Disables Read Proof
#define 	AT91C_MCI_RDPROOF_ENABLE               ((unsigned int) 0x1 << 11) // (MCI) Enables Read Proof
#define AT91C_MCI_WRPROOF     ((unsigned int) 0x1 << 12) // (MCI) Write Proof Enable
#define 	AT91C_MCI_WRPROOF_DISABLE              ((unsigned int) 0x0 << 12) // (MCI) Disables Write Proof
#define 	AT91C_MCI_WRPROOF_ENABLE               ((unsigned int) 0x1 << 12) // (MCI) Enables Write Proof
#define AT91C_MCI_PDCFBYTE    ((unsigned int) 0x1 << 13) // (MCI) PDC Force Byte Transfer
#define 	AT91C_MCI_PDCFBYTE_DISABLE              ((unsigned int) 0x0 << 13) // (MCI) Disables PDC Force Byte Transfer
#define 	AT91C_MCI_PDCFBYTE_ENABLE               ((unsigned int) 0x1 << 13) // (MCI) Enables PDC Force Byte Transfer
#define AT91C_MCI_PDCPADV     ((unsigned int) 0x1 << 14) // (MCI) PDC Padding Value
#define AT91C_MCI_PDCMODE     ((unsigned int) 0x1 << 15) // (MCI) PDC Oriented Mode
#define 	AT91C_MCI_PDCMODE_DISABLE              ((unsigned int) 0x0 << 15) // (MCI) Disables PDC Transfer
#define 	AT91C_MCI_PDCMODE_ENABLE               ((unsigned int) 0x1 << 15) // (MCI) Enables PDC Transfer
#define AT91C_MCI_BLKLEN      ((unsigned int) 0xFFFF << 16) // (MCI) Data Block Length
// -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- 
#define AT91C_MCI_DTOCYC      ((unsigned int) 0xF <<  0) // (MCI) Data Timeout Cycle Number
#define AT91C_MCI_DTOMUL      ((unsigned int) 0x7 <<  4) // (MCI) Data Timeout Multiplier
#define 	AT91C_MCI_DTOMUL_1                    ((unsigned int) 0x0 <<  4) // (MCI) DTOCYC x 1
#define 	AT91C_MCI_DTOMUL_16                   ((unsigned int) 0x1 <<  4) // (MCI) DTOCYC x 16
#define 	AT91C_MCI_DTOMUL_128                  ((unsigned int) 0x2 <<  4) // (MCI) DTOCYC x 128
#define 	AT91C_MCI_DTOMUL_256                  ((unsigned int) 0x3 <<  4) // (MCI) DTOCYC x 256
#define 	AT91C_MCI_DTOMUL_1024                 ((unsigned int) 0x4 <<  4) // (MCI) DTOCYC x 1024
#define 	AT91C_MCI_DTOMUL_4096                 ((unsigned int) 0x5 <<  4) // (MCI) DTOCYC x 4096
#define 	AT91C_MCI_DTOMUL_65536                ((unsigned int) 0x6 <<  4) // (MCI) DTOCYC x 65536
#define 	AT91C_MCI_DTOMUL_1048576              ((unsigned int) 0x7 <<  4) // (MCI) DTOCYC x 1048576
// -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- 
#define AT91C_MCI_SCDSEL      ((unsigned int) 0x3 <<  0) // (MCI) SD Card/SDIO Selector
#define 	AT91C_MCI_SCDSEL_SLOTA                ((unsigned int) 0x0) // (MCI) Slot A selected
#define 	AT91C_MCI_SCDSEL_SLOTB                ((unsigned int) 0x1) // (MCI) Slot B selected
#define 	AT91C_MCI_SCDSEL_SLOTC                ((unsigned int) 0x2) // (MCI) Slot C selected
#define 	AT91C_MCI_SCDSEL_SLOTD                ((unsigned int) 0x3) // (MCI) Slot D selected
#define AT91C_MCI_SCDBUS      ((unsigned int) 0x3 <<  6) // (MCI) SDCard/SDIO Bus Width
#define 	AT91C_MCI_SCDBUS_1BIT                 ((unsigned int) 0x0 <<  6) // (MCI) 1-bit data bus
#define 	AT91C_MCI_SCDBUS_4BITS                ((unsigned int) 0x2 <<  6) // (MCI) 4-bits data bus
#define 	AT91C_MCI_SCDBUS_8BITS                ((unsigned int) 0x3 <<  6) // (MCI) 8-bits data bus
// -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- 
#define AT91C_MCI_CMDNB       ((unsigned int) 0x3F <<  0) // (MCI) Command Number
#define AT91C_MCI_RSPTYP      ((unsigned int) 0x3 <<  6) // (MCI) Response Type
#define 	AT91C_MCI_RSPTYP_NO                   ((unsigned int) 0x0 <<  6) // (MCI) No response
#define 	AT91C_MCI_RSPTYP_48                   ((unsigned int) 0x1 <<  6) // (MCI) 48-bit response
#define 	AT91C_MCI_RSPTYP_136                  ((unsigned int) 0x2 <<  6) // (MCI) 136-bit response
#define 	AT91C_MCI_RSPTYP_R1B                  ((unsigned int) 0x3 <<  6) // (MCI) R1b response
#define AT91C_MCI_SPCMD       ((unsigned int) 0x7 <<  8) // (MCI) Special CMD
#define 	AT91C_MCI_SPCMD_NONE                 ((unsigned int) 0x0 <<  8) // (MCI) Not a special CMD
#define 	AT91C_MCI_SPCMD_INIT                 ((unsigned int) 0x1 <<  8) // (MCI) Initialization CMD
#define 	AT91C_MCI_SPCMD_SYNC                 ((unsigned int) 0x2 <<  8) // (MCI) Synchronized CMD
#define 	AT91C_MCI_SPCMD_CE_ATA               ((unsigned int) 0x3 <<  8) // (MCI) CE-ATA Completion Signal disable CMD
#define 	AT91C_MCI_SPCMD_IT_CMD               ((unsigned int) 0x4 <<  8) // (MCI) Interrupt command
#define 	AT91C_MCI_SPCMD_IT_REP               ((unsigned int) 0x5 <<  8) // (MCI) Interrupt response
#define AT91C_MCI_OPDCMD      ((unsigned int) 0x1 << 11) // (MCI) Open Drain Command
#define 	AT91C_MCI_OPDCMD_PUSHPULL             ((unsigned int) 0x0 << 11) // (MCI) Push/pull command
#define 	AT91C_MCI_OPDCMD_OPENDRAIN            ((unsigned int) 0x1 << 11) // (MCI) Open drain command
#define AT91C_MCI_MAXLAT      ((unsigned int) 0x1 << 12) // (MCI) Maximum Latency for Command to respond
#define 	AT91C_MCI_MAXLAT_5                    ((unsigned int) 0x0 << 12) // (MCI) 5 cycles maximum latency
#define 	AT91C_MCI_MAXLAT_64                   ((unsigned int) 0x1 << 12) // (MCI) 64 cycles maximum latency
#define AT91C_MCI_TRCMD       ((unsigned int) 0x3 << 16) // (MCI) Transfer CMD
#define 	AT91C_MCI_TRCMD_NO                   ((unsigned int) 0x0 << 16) // (MCI) No transfer
#define 	AT91C_MCI_TRCMD_START                ((unsigned int) 0x1 << 16) // (MCI) Start transfer
#define 	AT91C_MCI_TRCMD_STOP                 ((unsigned int) 0x2 << 16) // (MCI) Stop transfer
#define AT91C_MCI_TRDIR       ((unsigned int) 0x1 << 18) // (MCI) Transfer Direction
#define 	AT91C_MCI_TRDIR_WRITE                ((unsigned int) 0x0 << 18) // (MCI) Write
#define 	AT91C_MCI_TRDIR_READ                 ((unsigned int) 0x1 << 18) // (MCI) Read
#define AT91C_MCI_TRTYP       ((unsigned int) 0x7 << 19) // (MCI) Transfer Type
#define 	AT91C_MCI_TRTYP_BLOCK                ((unsigned int) 0x0 << 19) // (MCI) MMC/SDCard Single Block Transfer type
#define 	AT91C_MCI_TRTYP_MULTIPLE             ((unsigned int) 0x1 << 19) // (MCI) MMC/SDCard Multiple Block transfer type
#define 	AT91C_MCI_TRTYP_STREAM               ((unsigned int) 0x2 << 19) // (MCI) MMC Stream transfer type
#define 	AT91C_MCI_TRTYP_SDIO_BYTE            ((unsigned int) 0x4 << 19) // (MCI) SDIO Byte transfer type
#define 	AT91C_MCI_TRTYP_SDIO_BLOCK           ((unsigned int) 0x5 << 19) // (MCI) SDIO Block transfer type
#define AT91C_MCI_IOSPCMD     ((unsigned int) 0x3 << 24) // (MCI) SDIO Special Command
#define 	AT91C_MCI_IOSPCMD_NONE                 ((unsigned int) 0x0 << 24) // (MCI) NOT a special command
#define 	AT91C_MCI_IOSPCMD_SUSPEND              ((unsigned int) 0x1 << 24) // (MCI) SDIO Suspend Command
#define 	AT91C_MCI_IOSPCMD_RESUME               ((unsigned int) 0x2 << 24) // (MCI) SDIO Resume Command
#define AT91C_MCI_ATACS       ((unsigned int) 0x1 << 26) // (MCI) ATA with command completion signal
#define 	AT91C_MCI_ATACS_NORMAL               ((unsigned int) 0x0 << 26) // (MCI) normal operation mode
#define 	AT91C_MCI_ATACS_COMPLETION           ((unsigned int) 0x1 << 26) // (MCI) completion signal is expected within MCI_CSTOR
// -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- 
#define AT91C_MCI_BCNT        ((unsigned int) 0xFFFF <<  0) // (MCI) MMC/SDIO Block Count / SDIO Byte Count
// -------- MCI_CSTOR : (MCI Offset: 0x1c) MCI Completion Signal Timeout Register -------- 
#define AT91C_MCI_CSTOCYC     ((unsigned int) 0xF <<  0) // (MCI) Completion Signal Timeout Cycle Number
#define AT91C_MCI_CSTOMUL     ((unsigned int) 0x7 <<  4) // (MCI) Completion Signal Timeout Multiplier
#define 	AT91C_MCI_CSTOMUL_1                    ((unsigned int) 0x0 <<  4) // (MCI) CSTOCYC x 1
#define 	AT91C_MCI_CSTOMUL_16                   ((unsigned int) 0x1 <<  4) // (MCI) CSTOCYC x  16
#define 	AT91C_MCI_CSTOMUL_128                  ((unsigned int) 0x2 <<  4) // (MCI) CSTOCYC x  128
#define 	AT91C_MCI_CSTOMUL_256                  ((unsigned int) 0x3 <<  4) // (MCI) CSTOCYC x  256
#define 	AT91C_MCI_CSTOMUL_1024                 ((unsigned int) 0x4 <<  4) // (MCI) CSTOCYC x  1024
#define 	AT91C_MCI_CSTOMUL_4096                 ((unsigned int) 0x5 <<  4) // (MCI) CSTOCYC x  4096
#define 	AT91C_MCI_CSTOMUL_65536                ((unsigned int) 0x6 <<  4) // (MCI) CSTOCYC x  65536
#define 	AT91C_MCI_CSTOMUL_1048576              ((unsigned int) 0x7 <<  4) // (MCI) CSTOCYC x  1048576
// -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- 
#define AT91C_MCI_CMDRDY      ((unsigned int) 0x1 <<  0) // (MCI) Command Ready flag
#define AT91C_MCI_RXRDY       ((unsigned int) 0x1 <<  1) // (MCI) RX Ready flag
#define AT91C_MCI_TXRDY       ((unsigned int) 0x1 <<  2) // (MCI) TX Ready flag
#define AT91C_MCI_BLKE        ((unsigned int) 0x1 <<  3) // (MCI) Data Block Transfer Ended flag
#define AT91C_MCI_DTIP        ((unsigned int) 0x1 <<  4) // (MCI) Data Transfer in Progress flag
#define AT91C_MCI_NOTBUSY     ((unsigned int) 0x1 <<  5) // (MCI) Data Line Not Busy flag
#define AT91C_MCI_ENDRX       ((unsigned int) 0x1 <<  6) // (MCI) End of RX Buffer flag
#define AT91C_MCI_ENDTX       ((unsigned int) 0x1 <<  7) // (MCI) End of TX Buffer flag
#define AT91C_MCI_SDIOIRQA    ((unsigned int) 0x1 <<  8) // (MCI) SDIO Interrupt for Slot A
#define AT91C_MCI_SDIOIRQB    ((unsigned int) 0x1 <<  9) // (MCI) SDIO Interrupt for Slot B
#define AT91C_MCI_SDIOIRQC    ((unsigned int) 0x1 << 10) // (MCI) SDIO Interrupt for Slot C
#define AT91C_MCI_SDIOIRQD    ((unsigned int) 0x1 << 11) // (MCI) SDIO Interrupt for Slot D
#define AT91C_MCI_SDIOWAIT    ((unsigned int) 0x1 << 12) // (MCI) SDIO Read Wait operation flag
#define AT91C_MCI_CSRCV       ((unsigned int) 0x1 << 13) // (MCI) CE-ATA Completion Signal flag
#define AT91C_MCI_RXBUFF      ((unsigned int) 0x1 << 14) // (MCI) RX Buffer Full flag
#define AT91C_MCI_TXBUFE      ((unsigned int) 0x1 << 15) // (MCI) TX Buffer Empty flag
#define AT91C_MCI_RINDE       ((unsigned int) 0x1 << 16) // (MCI) Response Index Error flag
#define AT91C_MCI_RDIRE       ((unsigned int) 0x1 << 17) // (MCI) Response Direction Error flag
#define AT91C_MCI_RCRCE       ((unsigned int) 0x1 << 18) // (MCI) Response CRC Error flag
#define AT91C_MCI_RENDE       ((unsigned int) 0x1 << 19) // (MCI) Response End Bit Error flag
#define AT91C_MCI_RTOE        ((unsigned int) 0x1 << 20) // (MCI) Response Time-out Error flag
#define AT91C_MCI_DCRCE       ((unsigned int) 0x1 << 21) // (MCI) data CRC Error flag
#define AT91C_MCI_DTOE        ((unsigned int) 0x1 << 22) // (MCI) Data timeout Error flag
#define AT91C_MCI_CSTOE       ((unsigned int) 0x1 << 23) // (MCI) Completion Signal timeout Error flag
#define AT91C_MCI_BLKOVRE     ((unsigned int) 0x1 << 24) // (MCI) DMA Block Overrun Error flag
#define AT91C_MCI_DMADONE     ((unsigned int) 0x1 << 25) // (MCI) DMA Transfer Done flag
#define AT91C_MCI_FIFOEMPTY   ((unsigned int) 0x1 << 26) // (MCI) FIFO Empty flag
#define AT91C_MCI_XFRDONE     ((unsigned int) 0x1 << 27) // (MCI) Transfer Done flag
#define AT91C_MCI_OVRE        ((unsigned int) 0x1 << 30) // (MCI) Overrun flag
#define AT91C_MCI_UNRE        ((unsigned int) 0x1 << 31) // (MCI) Underrun flag
// -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- 
// -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- 
// -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- 
// -------- MCI_DMA : (MCI Offset: 0x50) MCI DMA Configuration Register -------- 
#define AT91C_MCI_OFFSET      ((unsigned int) 0x3 <<  0) // (MCI) DMA Write Buffer Offset
#define AT91C_MCI_CHKSIZE     ((unsigned int) 0x7 <<  4) // (MCI) DMA Channel Read/Write Chunk Size
#define 	AT91C_MCI_CHKSIZE_1                    ((unsigned int) 0x0 <<  4) // (MCI) Number of data transferred is 1
#define 	AT91C_MCI_CHKSIZE_4                    ((unsigned int) 0x1 <<  4) // (MCI) Number of data transferred is 4
#define 	AT91C_MCI_CHKSIZE_8                    ((unsigned int) 0x2 <<  4) // (MCI) Number of data transferred is 8
#define 	AT91C_MCI_CHKSIZE_16                   ((unsigned int) 0x3 <<  4) // (MCI) Number of data transferred is 16
#define 	AT91C_MCI_CHKSIZE_32                   ((unsigned int) 0x4 <<  4) // (MCI) Number of data transferred is 32
#define AT91C_MCI_DMAEN       ((unsigned int) 0x1 <<  8) // (MCI) DMA Hardware Handshaking Enable
#define 	AT91C_MCI_DMAEN_DISABLE              ((unsigned int) 0x0 <<  8) // (MCI) DMA interface is disabled
#define 	AT91C_MCI_DMAEN_ENABLE               ((unsigned int) 0x1 <<  8) // (MCI) DMA interface is enabled
// -------- MCI_CFG : (MCI Offset: 0x54) MCI Configuration Register -------- 
#define AT91C_MCI_FIFOMODE    ((unsigned int) 0x1 <<  0) // (MCI) MCI Internal FIFO Control Mode
#define 	AT91C_MCI_FIFOMODE_AMOUNTDATA           ((unsigned int) 0x0) // (MCI) A write transfer starts when a sufficient amount of datas is written into the FIFO
#define 	AT91C_MCI_FIFOMODE_ONEDATA              ((unsigned int) 0x1) // (MCI) A write transfer starts as soon as one data is written into the FIFO
#define AT91C_MCI_FERRCTRL    ((unsigned int) 0x1 <<  4) // (MCI) Flow Error Flag Reset Control Mode
#define 	AT91C_MCI_FERRCTRL_RWCMD                ((unsigned int) 0x0 <<  4) // (MCI) When an underflow/overflow condition flag is set, a new Write/Read command is needed to reset the flag
#define 	AT91C_MCI_FERRCTRL_READSR               ((unsigned int) 0x1 <<  4) // (MCI) When an underflow/overflow condition flag is set, a read status resets the flag
#define AT91C_MCI_HSMODE      ((unsigned int) 0x1 <<  8) // (MCI) High Speed Mode
#define 	AT91C_MCI_HSMODE_DISABLE              ((unsigned int) 0x0 <<  8) // (MCI) Default Bus Timing Mode
#define 	AT91C_MCI_HSMODE_ENABLE               ((unsigned int) 0x1 <<  8) // (MCI) High Speed Mode
#define AT91C_MCI_LSYNC       ((unsigned int) 0x1 << 12) // (MCI) Synchronize on last block
#define 	AT91C_MCI_LSYNC_CURRENT              ((unsigned int) 0x0 << 12) // (MCI) Pending command sent at end of current data block
#define 	AT91C_MCI_LSYNC_INFINITE             ((unsigned int) 0x1 << 12) // (MCI) Pending command sent at end of block transfer when transfer length is not infinite
// -------- MCI_WPCR : (MCI Offset: 0xe4) Write Protection Control Register -------- 
#define AT91C_MCI_WP_EN       ((unsigned int) 0x1 <<  0) // (MCI) Write Protection Enable
#define 	AT91C_MCI_WP_EN_DISABLE              ((unsigned int) 0x0) // (MCI) Write Operation is disabled (if WP_KEY corresponds)
#define 	AT91C_MCI_WP_EN_ENABLE               ((unsigned int) 0x1) // (MCI) Write Operation is enabled (if WP_KEY corresponds)
#define AT91C_MCI_WP_KEY      ((unsigned int) 0xFFFFFF <<  8) // (MCI) Write Protection Key
// -------- MCI_WPSR : (MCI Offset: 0xe8) Write Protection Status Register -------- 
#define AT91C_MCI_WP_VS       ((unsigned int) 0xF <<  0) // (MCI) Write Protection Violation Status
#define 	AT91C_MCI_WP_VS_NO_VIOLATION         ((unsigned int) 0x0) // (MCI) No Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_WRITE             ((unsigned int) 0x1) // (MCI) Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_RESET             ((unsigned int) 0x2) // (MCI) Software Reset Violation detected since last read
#define 	AT91C_MCI_WP_VS_ON_BOTH              ((unsigned int) 0x3) // (MCI) Write Protection and Software Reset Violation detected since last read
#define AT91C_MCI_WP_VSRC     ((unsigned int) 0xF <<  8) // (MCI) Write Protection Violation Source
#define 	AT91C_MCI_WP_VSRC_NO_VIOLATION         ((unsigned int) 0x0 <<  8) // (MCI) No Write Protection Violation detected since last read
#define 	AT91C_MCI_WP_VSRC_MCI_MR               ((unsigned int) 0x1 <<  8) // (MCI) Write Protection Violation detected on MCI_MR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DTOR             ((unsigned int) 0x2 <<  8) // (MCI) Write Protection Violation detected on MCI_DTOR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_SDCR             ((unsigned int) 0x3 <<  8) // (MCI) Write Protection Violation detected on MCI_SDCR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_CSTOR            ((unsigned int) 0x4 <<  8) // (MCI) Write Protection Violation detected on MCI_CSTOR since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DMA              ((unsigned int) 0x5 <<  8) // (MCI) Write Protection Violation detected on MCI_DMA since last read
#define 	AT91C_MCI_WP_VSRC_MCI_CFG              ((unsigned int) 0x6 <<  8) // (MCI) Write Protection Violation detected on MCI_CFG since last read
#define 	AT91C_MCI_WP_VSRC_MCI_DEL              ((unsigned int) 0x7 <<  8) // (MCI) Write Protection Violation detected on MCI_DEL since last read
// -------- MCI_VER : (MCI Offset: 0xfc)  VERSION  Register -------- 
#define AT91C_MCI_VER         ((unsigned int) 0xF <<  0) // (MCI)  VERSION  Register

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Two-wire Interface
// *****************************************************************************
typedef struct _AT91S_TWI {
	AT91_REG	 TWI_CR; 	// Control Register
	AT91_REG	 TWI_MMR; 	// Master Mode Register
	AT91_REG	 TWI_SMR; 	// Slave Mode Register
	AT91_REG	 TWI_IADR; 	// Internal Address Register
	AT91_REG	 TWI_CWGR; 	// Clock Waveform Generator Register
	AT91_REG	 Reserved0[3]; 	// 
	AT91_REG	 TWI_SR; 	// Status Register
	AT91_REG	 TWI_IER; 	// Interrupt Enable Register
	AT91_REG	 TWI_IDR; 	// Interrupt Disable Register
	AT91_REG	 TWI_IMR; 	// Interrupt Mask Register
	AT91_REG	 TWI_RHR; 	// Receive Holding Register
	AT91_REG	 TWI_THR; 	// Transmit Holding Register
	AT91_REG	 Reserved1[45]; 	// 
	AT91_REG	 TWI_ADDRSIZE; 	// TWI ADDRSIZE REGISTER 
	AT91_REG	 TWI_IPNAME1; 	// TWI IPNAME1 REGISTER 
	AT91_REG	 TWI_IPNAME2; 	// TWI IPNAME2 REGISTER 
	AT91_REG	 TWI_FEATURES; 	// TWI FEATURES REGISTER 
	AT91_REG	 TWI_VER; 	// Version Register
	AT91_REG	 TWI_RPR; 	// Receive Pointer Register
	AT91_REG	 TWI_RCR; 	// Receive Counter Register
	AT91_REG	 TWI_TPR; 	// Transmit Pointer Register
	AT91_REG	 TWI_TCR; 	// Transmit Counter Register
	AT91_REG	 TWI_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 TWI_RNCR; 	// Receive Next Counter Register
	AT91_REG	 TWI_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 TWI_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 TWI_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 TWI_PTSR; 	// PDC Transfer Status Register
} AT91S_TWI, *AT91PS_TWI;

// -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- 
#define AT91C_TWI_START       ((unsigned int) 0x1 <<  0) // (TWI) Send a START Condition
#define AT91C_TWI_STOP        ((unsigned int) 0x1 <<  1) // (TWI) Send a STOP Condition
#define AT91C_TWI_MSEN        ((unsigned int) 0x1 <<  2) // (TWI) TWI Master Transfer Enabled
#define AT91C_TWI_MSDIS       ((unsigned int) 0x1 <<  3) // (TWI) TWI Master Transfer Disabled
#define AT91C_TWI_SVEN        ((unsigned int) 0x1 <<  4) // (TWI) TWI Slave mode Enabled
#define AT91C_TWI_SVDIS       ((unsigned int) 0x1 <<  5) // (TWI) TWI Slave mode Disabled
#define AT91C_TWI_SWRST       ((unsigned int) 0x1 <<  7) // (TWI) Software Reset
// -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- 
#define AT91C_TWI_IADRSZ      ((unsigned int) 0x3 <<  8) // (TWI) Internal Device Address Size
#define 	AT91C_TWI_IADRSZ_NO                   ((unsigned int) 0x0 <<  8) // (TWI) No internal device address
#define 	AT91C_TWI_IADRSZ_1_BYTE               ((unsigned int) 0x1 <<  8) // (TWI) One-byte internal device address
#define 	AT91C_TWI_IADRSZ_2_BYTE               ((unsigned int) 0x2 <<  8) // (TWI) Two-byte internal device address
#define 	AT91C_TWI_IADRSZ_3_BYTE               ((unsigned int) 0x3 <<  8) // (TWI) Three-byte internal device address
#define AT91C_TWI_MREAD       ((unsigned int) 0x1 << 12) // (TWI) Master Read Direction
#define AT91C_TWI_DADR        ((unsigned int) 0x7F << 16) // (TWI) Device Address
// -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- 
#define AT91C_TWI_SADR        ((unsigned int) 0x7F << 16) // (TWI) Slave Address
// -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- 
#define AT91C_TWI_CLDIV       ((unsigned int) 0xFF <<  0) // (TWI) Clock Low Divider
#define AT91C_TWI_CHDIV       ((unsigned int) 0xFF <<  8) // (TWI) Clock High Divider
#define AT91C_TWI_CKDIV       ((unsigned int) 0x7 << 16) // (TWI) Clock Divider
// -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- 
#define AT91C_TWI_TXCOMP_SLAVE ((unsigned int) 0x1 <<  0) // (TWI) Transmission Completed
#define AT91C_TWI_TXCOMP_MASTER ((unsigned int) 0x1 <<  0) // (TWI) Transmission Completed
#define AT91C_TWI_RXRDY       ((unsigned int) 0x1 <<  1) // (TWI) Receive holding register ReaDY
#define AT91C_TWI_TXRDY_MASTER ((unsigned int) 0x1 <<  2) // (TWI) Transmit holding register ReaDY
#define AT91C_TWI_TXRDY_SLAVE ((unsigned int) 0x1 <<  2) // (TWI) Transmit holding register ReaDY
#define AT91C_TWI_SVREAD      ((unsigned int) 0x1 <<  3) // (TWI) Slave READ (used only in Slave mode)
#define AT91C_TWI_SVACC       ((unsigned int) 0x1 <<  4) // (TWI) Slave ACCess (used only in Slave mode)
#define AT91C_TWI_GACC        ((unsigned int) 0x1 <<  5) // (TWI) General Call ACcess (used only in Slave mode)
#define AT91C_TWI_OVRE        ((unsigned int) 0x1 <<  6) // (TWI) Overrun Error (used only in Master and Multi-master mode)
#define AT91C_TWI_NACK_SLAVE  ((unsigned int) 0x1 <<  8) // (TWI) Not Acknowledged
#define AT91C_TWI_NACK_MASTER ((unsigned int) 0x1 <<  8) // (TWI) Not Acknowledged
#define AT91C_TWI_ARBLST_MULTI_MASTER ((unsigned int) 0x1 <<  9) // (TWI) Arbitration Lost (used only in Multimaster mode)
#define AT91C_TWI_SCLWS       ((unsigned int) 0x1 << 10) // (TWI) Clock Wait State (used only in Slave mode)
#define AT91C_TWI_EOSACC      ((unsigned int) 0x1 << 11) // (TWI) End Of Slave ACCess (used only in Slave mode)
#define AT91C_TWI_ENDRX       ((unsigned int) 0x1 << 12) // (TWI) End of Receiver Transfer
#define AT91C_TWI_ENDTX       ((unsigned int) 0x1 << 13) // (TWI) End of Receiver Transfer
#define AT91C_TWI_RXBUFF      ((unsigned int) 0x1 << 14) // (TWI) RXBUFF Interrupt
#define AT91C_TWI_TXBUFE      ((unsigned int) 0x1 << 15) // (TWI) TXBUFE Interrupt
// -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- 
// -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- 
// -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Usart
// *****************************************************************************
typedef struct _AT91S_USART {
	AT91_REG	 US_CR; 	// Control Register
	AT91_REG	 US_MR; 	// Mode Register
	AT91_REG	 US_IER; 	// Interrupt Enable Register
	AT91_REG	 US_IDR; 	// Interrupt Disable Register
	AT91_REG	 US_IMR; 	// Interrupt Mask Register
	AT91_REG	 US_CSR; 	// Channel Status Register
	AT91_REG	 US_RHR; 	// Receiver Holding Register
	AT91_REG	 US_THR; 	// Transmitter Holding Register
	AT91_REG	 US_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 US_RTOR; 	// Receiver Time-out Register
	AT91_REG	 US_TTGR; 	// Transmitter Time-guard Register
	AT91_REG	 Reserved0[5]; 	// 
	AT91_REG	 US_FIDI; 	// FI_DI_Ratio Register
	AT91_REG	 US_NER; 	// Nb Errors Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 US_IF; 	// IRDA_FILTER Register
	AT91_REG	 US_MAN; 	// Manchester Encoder Decoder Register
	AT91_REG	 Reserved2[38]; 	// 
	AT91_REG	 US_ADDRSIZE; 	// US ADDRSIZE REGISTER 
	AT91_REG	 US_IPNAME1; 	// US IPNAME1 REGISTER 
	AT91_REG	 US_IPNAME2; 	// US IPNAME2 REGISTER 
	AT91_REG	 US_FEATURES; 	// US FEATURES REGISTER 
	AT91_REG	 US_VER; 	// VERSION Register
	AT91_REG	 US_RPR; 	// Receive Pointer Register
	AT91_REG	 US_RCR; 	// Receive Counter Register
	AT91_REG	 US_TPR; 	// Transmit Pointer Register
	AT91_REG	 US_TCR; 	// Transmit Counter Register
	AT91_REG	 US_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 US_RNCR; 	// Receive Next Counter Register
	AT91_REG	 US_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 US_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 US_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 US_PTSR; 	// PDC Transfer Status Register
} AT91S_USART, *AT91PS_USART;

// -------- US_CR : (USART Offset: 0x0)  Control Register -------- 
#define AT91C_US_RSTRX        ((unsigned int) 0x1 <<  2) // (USART) Reset Receiver
#define AT91C_US_RSTTX        ((unsigned int) 0x1 <<  3) // (USART) Reset Transmitter
#define AT91C_US_RXEN         ((unsigned int) 0x1 <<  4) // (USART) Receiver Enable
#define AT91C_US_RXDIS        ((unsigned int) 0x1 <<  5) // (USART) Receiver Disable
#define AT91C_US_TXEN         ((unsigned int) 0x1 <<  6) // (USART) Transmitter Enable
#define AT91C_US_TXDIS        ((unsigned int) 0x1 <<  7) // (USART) Transmitter Disable
#define AT91C_US_RSTSTA       ((unsigned int) 0x1 <<  8) // (USART) Reset Status Bits
#define AT91C_US_STTBRK       ((unsigned int) 0x1 <<  9) // (USART) Start Break
#define AT91C_US_STPBRK       ((unsigned int) 0x1 << 10) // (USART) Stop Break
#define AT91C_US_STTTO        ((unsigned int) 0x1 << 11) // (USART) Start Time-out
#define AT91C_US_SENDA        ((unsigned int) 0x1 << 12) // (USART) Send Address
#define AT91C_US_RSTIT        ((unsigned int) 0x1 << 13) // (USART) Reset Iterations
#define AT91C_US_RSTNACK      ((unsigned int) 0x1 << 14) // (USART) Reset Non Acknowledge
#define AT91C_US_RETTO        ((unsigned int) 0x1 << 15) // (USART) Rearm Time-out
#define AT91C_US_DTREN        ((unsigned int) 0x1 << 16) // (USART) Data Terminal ready Enable
#define AT91C_US_DTRDIS       ((unsigned int) 0x1 << 17) // (USART) Data Terminal ready Disable
#define AT91C_US_RTSEN        ((unsigned int) 0x1 << 18) // (USART) Request to Send enable
#define AT91C_US_RTSDIS       ((unsigned int) 0x1 << 19) // (USART) Request to Send Disable
// -------- US_MR : (USART Offset: 0x4)  Mode Register -------- 
#define AT91C_US_USMODE       ((unsigned int) 0xF <<  0) // (USART) Usart mode
#define 	AT91C_US_USMODE_NORMAL               ((unsigned int) 0x0) // (USART) Normal
#define 	AT91C_US_USMODE_RS485                ((unsigned int) 0x1) // (USART) RS485
#define 	AT91C_US_USMODE_HWHSH                ((unsigned int) 0x2) // (USART) Hardware Handshaking
#define 	AT91C_US_USMODE_MODEM                ((unsigned int) 0x3) // (USART) Modem
#define 	AT91C_US_USMODE_ISO7816_0            ((unsigned int) 0x4) // (USART) ISO7816 protocol: T = 0
#define 	AT91C_US_USMODE_ISO7816_1            ((unsigned int) 0x6) // (USART) ISO7816 protocol: T = 1
#define 	AT91C_US_USMODE_IRDA                 ((unsigned int) 0x8) // (USART) IrDA
#define AT91C_US_CLKS         ((unsigned int) 0x3 <<  4) // (USART) Clock Selection (Baud Rate generator Input Clock
#define 	AT91C_US_CLKS_CLOCK                ((unsigned int) 0x0 <<  4) // (USART) Clock
#define 	AT91C_US_CLKS_FDIV1                ((unsigned int) 0x1 <<  4) // (USART) fdiv1
#define 	AT91C_US_CLKS_SLOW                 ((unsigned int) 0x2 <<  4) // (USART) slow_clock (ARM)
#define 	AT91C_US_CLKS_EXT                  ((unsigned int) 0x3 <<  4) // (USART) External (SCK)
#define AT91C_US_CHRL         ((unsigned int) 0x3 <<  6) // (USART) Clock Selection (Baud Rate generator Input Clock
#define 	AT91C_US_CHRL_5_BITS               ((unsigned int) 0x0 <<  6) // (USART) Character Length: 5 bits
#define 	AT91C_US_CHRL_6_BITS               ((unsigned int) 0x1 <<  6) // (USART) Character Length: 6 bits
#define 	AT91C_US_CHRL_7_BITS               ((unsigned int) 0x2 <<  6) // (USART) Character Length: 7 bits
#define 	AT91C_US_CHRL_8_BITS               ((unsigned int) 0x3 <<  6) // (USART) Character Length: 8 bits
#define AT91C_US_SYNC         ((unsigned int) 0x1 <<  8) // (USART) Synchronous Mode Select
#define AT91C_US_PAR          ((unsigned int) 0x7 <<  9) // (USART) Parity type
#define 	AT91C_US_PAR_EVEN                 ((unsigned int) 0x0 <<  9) // (USART) Even Parity
#define 	AT91C_US_PAR_ODD                  ((unsigned int) 0x1 <<  9) // (USART) Odd Parity
#define 	AT91C_US_PAR_SPACE                ((unsigned int) 0x2 <<  9) // (USART) Parity forced to 0 (Space)
#define 	AT91C_US_PAR_MARK                 ((unsigned int) 0x3 <<  9) // (USART) Parity forced to 1 (Mark)
#define 	AT91C_US_PAR_NONE                 ((unsigned int) 0x4 <<  9) // (USART) No Parity
#define 	AT91C_US_PAR_MULTI_DROP           ((unsigned int) 0x6 <<  9) // (USART) Multi-drop mode
#define AT91C_US_NBSTOP       ((unsigned int) 0x3 << 12) // (USART) Number of Stop bits
#define 	AT91C_US_NBSTOP_1_BIT                ((unsigned int) 0x0 << 12) // (USART) 1 stop bit
#define 	AT91C_US_NBSTOP_15_BIT               ((unsigned int) 0x1 << 12) // (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits
#define 	AT91C_US_NBSTOP_2_BIT                ((unsigned int) 0x2 << 12) // (USART) 2 stop bits
#define AT91C_US_CHMODE       ((unsigned int) 0x3 << 14) // (USART) Channel Mode
#define 	AT91C_US_CHMODE_NORMAL               ((unsigned int) 0x0 << 14) // (USART) Normal Mode: The USART channel operates as an RX/TX USART.
#define 	AT91C_US_CHMODE_AUTO                 ((unsigned int) 0x1 << 14) // (USART) Automatic Echo: Receiver Data Input is connected to the TXD pin.
#define 	AT91C_US_CHMODE_LOCAL                ((unsigned int) 0x2 << 14) // (USART) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
#define 	AT91C_US_CHMODE_REMOTE               ((unsigned int) 0x3 << 14) // (USART) Remote Loopback: RXD pin is internally connected to TXD pin.
#define AT91C_US_MSBF         ((unsigned int) 0x1 << 16) // (USART) Bit Order
#define AT91C_US_MODE9        ((unsigned int) 0x1 << 17) // (USART) 9-bit Character length
#define AT91C_US_CKLO         ((unsigned int) 0x1 << 18) // (USART) Clock Output Select
#define AT91C_US_OVER         ((unsigned int) 0x1 << 19) // (USART) Over Sampling Mode
#define AT91C_US_INACK        ((unsigned int) 0x1 << 20) // (USART) Inhibit Non Acknowledge
#define AT91C_US_DSNACK       ((unsigned int) 0x1 << 21) // (USART) Disable Successive NACK
#define AT91C_US_VAR_SYNC     ((unsigned int) 0x1 << 22) // (USART) Variable synchronization of command/data sync Start Frame Delimiter
#define AT91C_US_MAX_ITER     ((unsigned int) 0x1 << 24) // (USART) Number of Repetitions
#define AT91C_US_FILTER       ((unsigned int) 0x1 << 28) // (USART) Receive Line Filter
#define AT91C_US_MANMODE      ((unsigned int) 0x1 << 29) // (USART) Manchester Encoder/Decoder Enable
#define AT91C_US_MODSYNC      ((unsigned int) 0x1 << 30) // (USART) Manchester Synchronization mode
#define AT91C_US_ONEBIT       ((unsigned int) 0x1 << 31) // (USART) Start Frame Delimiter selector
// -------- US_IER : (USART Offset: 0x8)  Interrupt Enable Register -------- 
#define AT91C_US_RXRDY        ((unsigned int) 0x1 <<  0) // (USART) RXRDY Interrupt
#define AT91C_US_TXRDY        ((unsigned int) 0x1 <<  1) // (USART) TXRDY Interrupt
#define AT91C_US_RXBRK        ((unsigned int) 0x1 <<  2) // (USART) Break Received/End of Break
#define AT91C_US_ENDRX        ((unsigned int) 0x1 <<  3) // (USART) End of Receive Transfer Interrupt
#define AT91C_US_ENDTX        ((unsigned int) 0x1 <<  4) // (USART) End of Transmit Interrupt
#define AT91C_US_OVRE         ((unsigned int) 0x1 <<  5) // (USART) Overrun Interrupt
#define AT91C_US_FRAME        ((unsigned int) 0x1 <<  6) // (USART) Framing Error Interrupt
#define AT91C_US_PARE         ((unsigned int) 0x1 <<  7) // (USART) Parity Error Interrupt
#define AT91C_US_TIMEOUT      ((unsigned int) 0x1 <<  8) // (USART) Receiver Time-out
#define AT91C_US_TXEMPTY      ((unsigned int) 0x1 <<  9) // (USART) TXEMPTY Interrupt
#define AT91C_US_ITERATION    ((unsigned int) 0x1 << 10) // (USART) Max number of Repetitions Reached
#define AT91C_US_TXBUFE       ((unsigned int) 0x1 << 11) // (USART) TXBUFE Interrupt
#define AT91C_US_RXBUFF       ((unsigned int) 0x1 << 12) // (USART) RXBUFF Interrupt
#define AT91C_US_NACK         ((unsigned int) 0x1 << 13) // (USART) Non Acknowledge
#define AT91C_US_RIIC         ((unsigned int) 0x1 << 16) // (USART) Ring INdicator Input Change Flag
#define AT91C_US_DSRIC        ((unsigned int) 0x1 << 17) // (USART) Data Set Ready Input Change Flag
#define AT91C_US_DCDIC        ((unsigned int) 0x1 << 18) // (USART) Data Carrier Flag
#define AT91C_US_CTSIC        ((unsigned int) 0x1 << 19) // (USART) Clear To Send Input Change Flag
#define AT91C_US_MANE         ((unsigned int) 0x1 << 20) // (USART) Manchester Error Interrupt
// -------- US_IDR : (USART Offset: 0xc)  Interrupt Disable Register -------- 
// -------- US_IMR : (USART Offset: 0x10)  Interrupt Mask Register -------- 
// -------- US_CSR : (USART Offset: 0x14)  Channel Status Register -------- 
#define AT91C_US_RI           ((unsigned int) 0x1 << 20) // (USART) Image of RI Input
#define AT91C_US_DSR          ((unsigned int) 0x1 << 21) // (USART) Image of DSR Input
#define AT91C_US_DCD          ((unsigned int) 0x1 << 22) // (USART) Image of DCD Input
#define AT91C_US_CTS          ((unsigned int) 0x1 << 23) // (USART) Image of CTS Input
#define AT91C_US_MANERR       ((unsigned int) 0x1 << 24) // (USART) Manchester Error
// -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- 
#define AT91C_US_TX_PL        ((unsigned int) 0xF <<  0) // (USART) Transmitter Preamble Length
#define AT91C_US_TX_PP        ((unsigned int) 0x3 <<  8) // (USART) Transmitter Preamble Pattern
#define 	AT91C_US_TX_PP_ALL_ONE              ((unsigned int) 0x0 <<  8) // (USART) ALL_ONE
#define 	AT91C_US_TX_PP_ALL_ZERO             ((unsigned int) 0x1 <<  8) // (USART) ALL_ZERO
#define 	AT91C_US_TX_PP_ZERO_ONE             ((unsigned int) 0x2 <<  8) // (USART) ZERO_ONE
#define 	AT91C_US_TX_PP_ONE_ZERO             ((unsigned int) 0x3 <<  8) // (USART) ONE_ZERO
#define AT91C_US_TX_MPOL      ((unsigned int) 0x1 << 12) // (USART) Transmitter Manchester Polarity
#define AT91C_US_RX_PL        ((unsigned int) 0xF << 16) // (USART) Receiver Preamble Length
#define AT91C_US_RX_PP        ((unsigned int) 0x3 << 24) // (USART) Receiver Preamble Pattern detected
#define 	AT91C_US_RX_PP_ALL_ONE              ((unsigned int) 0x0 << 24) // (USART) ALL_ONE
#define 	AT91C_US_RX_PP_ALL_ZERO             ((unsigned int) 0x1 << 24) // (USART) ALL_ZERO
#define 	AT91C_US_RX_PP_ZERO_ONE             ((unsigned int) 0x2 << 24) // (USART) ZERO_ONE
#define 	AT91C_US_RX_PP_ONE_ZERO             ((unsigned int) 0x3 << 24) // (USART) ONE_ZERO
#define AT91C_US_RX_MPOL      ((unsigned int) 0x1 << 28) // (USART) Receiver Manchester Polarity
#define AT91C_US_DRIFT        ((unsigned int) 0x1 << 30) // (USART) Drift compensation

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface
// *****************************************************************************
typedef struct _AT91S_SSC {
	AT91_REG	 SSC_CR; 	// Control Register
	AT91_REG	 SSC_CMR; 	// Clock Mode Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 SSC_RCMR; 	// Receive Clock ModeRegister
	AT91_REG	 SSC_RFMR; 	// Receive Frame Mode Register
	AT91_REG	 SSC_TCMR; 	// Transmit Clock Mode Register
	AT91_REG	 SSC_TFMR; 	// Transmit Frame Mode Register
	AT91_REG	 SSC_RHR; 	// Receive Holding Register
	AT91_REG	 SSC_THR; 	// Transmit Holding Register
	AT91_REG	 Reserved1[2]; 	// 
	AT91_REG	 SSC_RSHR; 	// Receive Sync Holding Register
	AT91_REG	 SSC_TSHR; 	// Transmit Sync Holding Register
	AT91_REG	 SSC_RC0R; 	// Receive Compare 0 Register
	AT91_REG	 SSC_RC1R; 	// Receive Compare 1 Register
	AT91_REG	 SSC_SR; 	// Status Register
	AT91_REG	 SSC_IER; 	// Interrupt Enable Register
	AT91_REG	 SSC_IDR; 	// Interrupt Disable Register
	AT91_REG	 SSC_IMR; 	// Interrupt Mask Register
} AT91S_SSC, *AT91PS_SSC;

// -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- 
#define AT91C_SSC_RXEN        ((unsigned int) 0x1 <<  0) // (SSC) Receive Enable
#define AT91C_SSC_RXDIS       ((unsigned int) 0x1 <<  1) // (SSC) Receive Disable
#define AT91C_SSC_TXEN        ((unsigned int) 0x1 <<  8) // (SSC) Transmit Enable
#define AT91C_SSC_TXDIS       ((unsigned int) 0x1 <<  9) // (SSC) Transmit Disable
#define AT91C_SSC_SWRST       ((unsigned int) 0x1 << 15) // (SSC) Software Reset
// -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- 
#define AT91C_SSC_CKS         ((unsigned int) 0x3 <<  0) // (SSC) Receive/Transmit Clock Selection
#define 	AT91C_SSC_CKS_DIV                  ((unsigned int) 0x0) // (SSC) Divided Clock
#define 	AT91C_SSC_CKS_TK                   ((unsigned int) 0x1) // (SSC) TK Clock signal
#define 	AT91C_SSC_CKS_RK                   ((unsigned int) 0x2) // (SSC) RK pin
#define AT91C_SSC_CKO         ((unsigned int) 0x7 <<  2) // (SSC) Receive/Transmit Clock Output Mode Selection
#define 	AT91C_SSC_CKO_NONE                 ((unsigned int) 0x0 <<  2) // (SSC) Receive/Transmit Clock Output Mode: None RK pin: Input-only
#define 	AT91C_SSC_CKO_CONTINOUS            ((unsigned int) 0x1 <<  2) // (SSC) Continuous Receive/Transmit Clock RK pin: Output
#define 	AT91C_SSC_CKO_DATA_TX              ((unsigned int) 0x2 <<  2) // (SSC) Receive/Transmit Clock only during data transfers RK pin: Output
#define AT91C_SSC_CKI         ((unsigned int) 0x1 <<  5) // (SSC) Receive/Transmit Clock Inversion
#define AT91C_SSC_CKG         ((unsigned int) 0x3 <<  6) // (SSC) Receive/Transmit Clock Gating Selection
#define 	AT91C_SSC_CKG_NONE                 ((unsigned int) 0x0 <<  6) // (SSC) Receive/Transmit Clock Gating: None, continuous clock
#define 	AT91C_SSC_CKG_LOW                  ((unsigned int) 0x1 <<  6) // (SSC) Receive/Transmit Clock enabled only if RF Low
#define 	AT91C_SSC_CKG_HIGH                 ((unsigned int) 0x2 <<  6) // (SSC) Receive/Transmit Clock enabled only if RF High
#define AT91C_SSC_START       ((unsigned int) 0xF <<  8) // (SSC) Receive/Transmit Start Selection
#define 	AT91C_SSC_START_CONTINOUS            ((unsigned int) 0x0 <<  8) // (SSC) Continuous, as soon as the receiver is enabled, and immediately after the end of transfer of the previous data.
#define 	AT91C_SSC_START_TX                   ((unsigned int) 0x1 <<  8) // (SSC) Transmit/Receive start
#define 	AT91C_SSC_START_LOW_RF               ((unsigned int) 0x2 <<  8) // (SSC) Detection of a low level on RF input
#define 	AT91C_SSC_START_HIGH_RF              ((unsigned int) 0x3 <<  8) // (SSC) Detection of a high level on RF input
#define 	AT91C_SSC_START_FALL_RF              ((unsigned int) 0x4 <<  8) // (SSC) Detection of a falling edge on RF input
#define 	AT91C_SSC_START_RISE_RF              ((unsigned int) 0x5 <<  8) // (SSC) Detection of a rising edge on RF input
#define 	AT91C_SSC_START_LEVEL_RF             ((unsigned int) 0x6 <<  8) // (SSC) Detection of any level change on RF input
#define 	AT91C_SSC_START_EDGE_RF              ((unsigned int) 0x7 <<  8) // (SSC) Detection of any edge on RF input
#define 	AT91C_SSC_START_0                    ((unsigned int) 0x8 <<  8) // (SSC) Compare 0
#define AT91C_SSC_STOP        ((unsigned int) 0x1 << 12) // (SSC) Receive Stop Selection
#define AT91C_SSC_STTOUT      ((unsigned int) 0x1 << 15) // (SSC) Receive/Transmit Start Output Selection
#define AT91C_SSC_STTDLY      ((unsigned int) 0xFF << 16) // (SSC) Receive/Transmit Start Delay
#define AT91C_SSC_PERIOD      ((unsigned int) 0xFF << 24) // (SSC) Receive/Transmit Period Divider Selection
// -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- 
#define AT91C_SSC_DATLEN      ((unsigned int) 0x1F <<  0) // (SSC) Data Length
#define AT91C_SSC_LOOP        ((unsigned int) 0x1 <<  5) // (SSC) Loop Mode
#define AT91C_SSC_MSBF        ((unsigned int) 0x1 <<  7) // (SSC) Most Significant Bit First
#define AT91C_SSC_DATNB       ((unsigned int) 0xF <<  8) // (SSC) Data Number per Frame
#define AT91C_SSC_FSLEN       ((unsigned int) 0xF << 16) // (SSC) Receive/Transmit Frame Sync length
#define AT91C_SSC_FSOS        ((unsigned int) 0x7 << 20) // (SSC) Receive/Transmit Frame Sync Output Selection
#define 	AT91C_SSC_FSOS_NONE                 ((unsigned int) 0x0 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: None RK pin Input-only
#define 	AT91C_SSC_FSOS_NEGATIVE             ((unsigned int) 0x1 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Negative Pulse
#define 	AT91C_SSC_FSOS_POSITIVE             ((unsigned int) 0x2 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Positive Pulse
#define 	AT91C_SSC_FSOS_LOW                  ((unsigned int) 0x3 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Driver Low during data transfer
#define 	AT91C_SSC_FSOS_HIGH                 ((unsigned int) 0x4 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Driver High during data transfer
#define 	AT91C_SSC_FSOS_TOGGLE               ((unsigned int) 0x5 << 20) // (SSC) Selected Receive/Transmit Frame Sync Signal: Toggling at each start of data transfer
#define AT91C_SSC_FSEDGE      ((unsigned int) 0x1 << 24) // (SSC) Frame Sync Edge Detection
// -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- 
// -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- 
#define AT91C_SSC_DATDEF      ((unsigned int) 0x1 <<  5) // (SSC) Data Default Value
#define AT91C_SSC_FSDEN       ((unsigned int) 0x1 << 23) // (SSC) Frame Sync Data Enable
// -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- 
#define AT91C_SSC_TXRDY       ((unsigned int) 0x1 <<  0) // (SSC) Transmit Ready
#define AT91C_SSC_TXEMPTY     ((unsigned int) 0x1 <<  1) // (SSC) Transmit Empty
#define AT91C_SSC_ENDTX       ((unsigned int) 0x1 <<  2) // (SSC) End Of Transmission
#define AT91C_SSC_TXBUFE      ((unsigned int) 0x1 <<  3) // (SSC) Transmit Buffer Empty
#define AT91C_SSC_RXRDY       ((unsigned int) 0x1 <<  4) // (SSC) Receive Ready
#define AT91C_SSC_OVRUN       ((unsigned int) 0x1 <<  5) // (SSC) Receive Overrun
#define AT91C_SSC_ENDRX       ((unsigned int) 0x1 <<  6) // (SSC) End of Reception
#define AT91C_SSC_RXBUFF      ((unsigned int) 0x1 <<  7) // (SSC) Receive Buffer Full
#define AT91C_SSC_CP0         ((unsigned int) 0x1 <<  8) // (SSC) Compare 0
#define AT91C_SSC_CP1         ((unsigned int) 0x1 <<  9) // (SSC) Compare 1
#define AT91C_SSC_TXSYN       ((unsigned int) 0x1 << 10) // (SSC) Transmit Sync
#define AT91C_SSC_RXSYN       ((unsigned int) 0x1 << 11) // (SSC) Receive Sync
#define AT91C_SSC_TXENA       ((unsigned int) 0x1 << 16) // (SSC) Transmit Enable
#define AT91C_SSC_RXENA       ((unsigned int) 0x1 << 17) // (SSC) Receive Enable
// -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- 
// -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- 
// -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR PWMC Channel Interface
// *****************************************************************************
typedef struct _AT91S_PWMC_CH {
	AT91_REG	 PWMC_CMR; 	// Channel Mode Register
	AT91_REG	 PWMC_CDTYR; 	// Channel Duty Cycle Register
	AT91_REG	 PWMC_CDTYUPDR; 	// Channel Duty Cycle Update Register
	AT91_REG	 PWMC_CPRDR; 	// Channel Period Register
	AT91_REG	 PWMC_CPRDUPDR; 	// Channel Period Update Register
	AT91_REG	 PWMC_CCNTR; 	// Channel Counter Register
	AT91_REG	 PWMC_DTR; 	// Channel Dead Time Value Register
	AT91_REG	 PWMC_DTUPDR; 	// Channel Dead Time Update Value Register
} AT91S_PWMC_CH, *AT91PS_PWMC_CH;

// -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- 
#define AT91C_PWMC_CPRE       ((unsigned int) 0xF <<  0) // (PWMC_CH) Channel Pre-scaler : PWMC_CLKx
#define 	AT91C_PWMC_CPRE_MCK                  ((unsigned int) 0x0) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_2            ((unsigned int) 0x1) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_4            ((unsigned int) 0x2) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_8            ((unsigned int) 0x3) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_16           ((unsigned int) 0x4) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_32           ((unsigned int) 0x5) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_64           ((unsigned int) 0x6) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_128          ((unsigned int) 0x7) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_256          ((unsigned int) 0x8) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_512          ((unsigned int) 0x9) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCK_DIV_1024         ((unsigned int) 0xA) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCKA                 ((unsigned int) 0xB) // (PWMC_CH) 
#define 	AT91C_PWMC_CPRE_MCKB                 ((unsigned int) 0xC) // (PWMC_CH) 
#define AT91C_PWMC_CALG       ((unsigned int) 0x1 <<  8) // (PWMC_CH) Channel Alignment
#define AT91C_PWMC_CPOL       ((unsigned int) 0x1 <<  9) // (PWMC_CH) Channel Polarity
#define AT91C_PWMC_CES        ((unsigned int) 0x1 << 10) // (PWMC_CH) Counter Event Selection
#define AT91C_PWMC_DTE        ((unsigned int) 0x1 << 16) // (PWMC_CH) Dead Time Genrator Enable
#define AT91C_PWMC_DTHI       ((unsigned int) 0x1 << 17) // (PWMC_CH) Dead Time PWMHx Output Inverted
#define AT91C_PWMC_DTLI       ((unsigned int) 0x1 << 18) // (PWMC_CH) Dead Time PWMLx Output Inverted
// -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- 
#define AT91C_PWMC_CDTY       ((unsigned int) 0xFFFFFF <<  0) // (PWMC_CH) Channel Duty Cycle
// -------- PWMC_CDTYUPDR : (PWMC_CH Offset: 0x8) PWMC Channel Duty Cycle Update Register -------- 
#define AT91C_PWMC_CDTYUPD    ((unsigned int) 0xFFFFFF <<  0) // (PWMC_CH) Channel Duty Cycle Update
// -------- PWMC_CPRDR : (PWMC_CH Offset: 0xc) PWMC Channel Period Register -------- 
#define AT91C_PWMC_CPRD       ((unsigned int) 0xFFFFFF <<  0) // (PWMC_CH) Channel Period
// -------- PWMC_CPRDUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Period Update Register -------- 
#define AT91C_PWMC_CPRDUPD    ((unsigned int) 0xFFFFFF <<  0) // (PWMC_CH) Channel Period Update
// -------- PWMC_CCNTR : (PWMC_CH Offset: 0x14) PWMC Channel Counter Register -------- 
#define AT91C_PWMC_CCNT       ((unsigned int) 0xFFFFFF <<  0) // (PWMC_CH) Channel Counter
// -------- PWMC_DTR : (PWMC_CH Offset: 0x18) Channel Dead Time Value Register -------- 
#define AT91C_PWMC_DTL        ((unsigned int) 0xFFFF <<  0) // (PWMC_CH) Channel Dead Time for PWML
#define AT91C_PWMC_DTH        ((unsigned int) 0xFFFF << 16) // (PWMC_CH) Channel Dead Time for PWMH
// -------- PWMC_DTUPDR : (PWMC_CH Offset: 0x1c) Channel Dead Time Value Register -------- 
#define AT91C_PWMC_DTLUPD     ((unsigned int) 0xFFFF <<  0) // (PWMC_CH) Channel Dead Time Update for PWML.
#define AT91C_PWMC_DTHUPD     ((unsigned int) 0xFFFF << 16) // (PWMC_CH) Channel Dead Time Update for PWMH.

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface
// *****************************************************************************
typedef struct _AT91S_PWMC {
	AT91_REG	 PWMC_MR; 	// PWMC Mode Register
	AT91_REG	 PWMC_ENA; 	// PWMC Enable Register
	AT91_REG	 PWMC_DIS; 	// PWMC Disable Register
	AT91_REG	 PWMC_SR; 	// PWMC Status Register
	AT91_REG	 PWMC_IER1; 	// PWMC Interrupt Enable Register 1
	AT91_REG	 PWMC_IDR1; 	// PWMC Interrupt Disable Register 1
	AT91_REG	 PWMC_IMR1; 	// PWMC Interrupt Mask Register 1
	AT91_REG	 PWMC_ISR1; 	// PWMC Interrupt Status Register 1
	AT91_REG	 PWMC_SYNC; 	// PWM Synchronized Channels Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PWMC_UPCR; 	// PWM Update Control Register
	AT91_REG	 PWMC_SCUP; 	// PWM Update Period Register
	AT91_REG	 PWMC_SCUPUPD; 	// PWM Update Period Update Register
	AT91_REG	 PWMC_IER2; 	// PWMC Interrupt Enable Register 2
	AT91_REG	 PWMC_IDR2; 	// PWMC Interrupt Disable Register 2
	AT91_REG	 PWMC_IMR2; 	// PWMC Interrupt Mask Register 2
	AT91_REG	 PWMC_ISR2; 	// PWMC Interrupt Status Register 2
	AT91_REG	 PWMC_OOV; 	// PWM Output Override Value Register
	AT91_REG	 PWMC_OS; 	// PWM Output Selection Register
	AT91_REG	 PWMC_OSS; 	// PWM Output Selection Set Register
	AT91_REG	 PWMC_OSC; 	// PWM Output Selection Clear Register
	AT91_REG	 PWMC_OSSUPD; 	// PWM Output Selection Set Update Register
	AT91_REG	 PWMC_OSCUPD; 	// PWM Output Selection Clear Update Register
	AT91_REG	 PWMC_FMR; 	// PWM Fault Mode Register
	AT91_REG	 PWMC_FSR; 	// PWM Fault Mode Status Register
	AT91_REG	 PWMC_FCR; 	// PWM Fault Mode Clear Register
	AT91_REG	 PWMC_FPV; 	// PWM Fault Protection Value Register
	AT91_REG	 PWMC_FPER1; 	// PWM Fault Protection Enable Register 1
	AT91_REG	 PWMC_FPER2; 	// PWM Fault Protection Enable Register 2
	AT91_REG	 PWMC_FPER3; 	// PWM Fault Protection Enable Register 3
	AT91_REG	 PWMC_FPER4; 	// PWM Fault Protection Enable Register 4
	AT91_REG	 PWMC_EL0MR; 	// PWM Event Line 0 Mode Register
	AT91_REG	 PWMC_EL1MR; 	// PWM Event Line 1 Mode Register
	AT91_REG	 PWMC_EL2MR; 	// PWM Event Line 2 Mode Register
	AT91_REG	 PWMC_EL3MR; 	// PWM Event Line 3 Mode Register
	AT91_REG	 PWMC_EL4MR; 	// PWM Event Line 4 Mode Register
	AT91_REG	 PWMC_EL5MR; 	// PWM Event Line 5 Mode Register
	AT91_REG	 PWMC_EL6MR; 	// PWM Event Line 6 Mode Register
	AT91_REG	 PWMC_EL7MR; 	// PWM Event Line 7 Mode Register
	AT91_REG	 Reserved1[18]; 	// 
	AT91_REG	 PWMC_WPCR; 	// PWM Write Protection Enable Register
	AT91_REG	 PWMC_WPSR; 	// PWM Write Protection Status Register
	AT91_REG	 PWMC_ADDRSIZE; 	// PWMC ADDRSIZE REGISTER 
	AT91_REG	 PWMC_IPNAME1; 	// PWMC IPNAME1 REGISTER 
	AT91_REG	 PWMC_IPNAME2; 	// PWMC IPNAME2 REGISTER 
	AT91_REG	 PWMC_FEATURES; 	// PWMC FEATURES REGISTER 
	AT91_REG	 PWMC_VER; 	// PWMC Version Register
	AT91_REG	 PWMC_RPR; 	// Receive Pointer Register
	AT91_REG	 PWMC_RCR; 	// Receive Counter Register
	AT91_REG	 PWMC_TPR; 	// Transmit Pointer Register
	AT91_REG	 PWMC_TCR; 	// Transmit Counter Register
	AT91_REG	 PWMC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PWMC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PWMC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PWMC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PWMC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PWMC_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 PWMC_CMP0V; 	// PWM Comparison Value 0 Register
	AT91_REG	 PWMC_CMP0VUPD; 	// PWM Comparison Value 0 Update Register
	AT91_REG	 PWMC_CMP0M; 	// PWM Comparison Mode 0 Register
	AT91_REG	 PWMC_CMP0MUPD; 	// PWM Comparison Mode 0 Update Register
	AT91_REG	 PWMC_CMP1V; 	// PWM Comparison Value 1 Register
	AT91_REG	 PWMC_CMP1VUPD; 	// PWM Comparison Value 1 Update Register
	AT91_REG	 PWMC_CMP1M; 	// PWM Comparison Mode 1 Register
	AT91_REG	 PWMC_CMP1MUPD; 	// PWM Comparison Mode 1 Update Register
	AT91_REG	 PWMC_CMP2V; 	// PWM Comparison Value 2 Register
	AT91_REG	 PWMC_CMP2VUPD; 	// PWM Comparison Value 2 Update Register
	AT91_REG	 PWMC_CMP2M; 	// PWM Comparison Mode 2 Register
	AT91_REG	 PWMC_CMP2MUPD; 	// PWM Comparison Mode 2 Update Register
	AT91_REG	 PWMC_CMP3V; 	// PWM Comparison Value 3 Register
	AT91_REG	 PWMC_CMP3VUPD; 	// PWM Comparison Value 3 Update Register
	AT91_REG	 PWMC_CMP3M; 	// PWM Comparison Mode 3 Register
	AT91_REG	 PWMC_CMP3MUPD; 	// PWM Comparison Mode 3 Update Register
	AT91_REG	 PWMC_CMP4V; 	// PWM Comparison Value 4 Register
	AT91_REG	 PWMC_CMP4VUPD; 	// PWM Comparison Value 4 Update Register
	AT91_REG	 PWMC_CMP4M; 	// PWM Comparison Mode 4 Register
	AT91_REG	 PWMC_CMP4MUPD; 	// PWM Comparison Mode 4 Update Register
	AT91_REG	 PWMC_CMP5V; 	// PWM Comparison Value 5 Register
	AT91_REG	 PWMC_CMP5VUPD; 	// PWM Comparison Value 5 Update Register
	AT91_REG	 PWMC_CMP5M; 	// PWM Comparison Mode 5 Register
	AT91_REG	 PWMC_CMP5MUPD; 	// PWM Comparison Mode 5 Update Register
	AT91_REG	 PWMC_CMP6V; 	// PWM Comparison Value 6 Register
	AT91_REG	 PWMC_CMP6VUPD; 	// PWM Comparison Value 6 Update Register
	AT91_REG	 PWMC_CMP6M; 	// PWM Comparison Mode 6 Register
	AT91_REG	 PWMC_CMP6MUPD; 	// PWM Comparison Mode 6 Update Register
	AT91_REG	 PWMC_CMP7V; 	// PWM Comparison Value 7 Register
	AT91_REG	 PWMC_CMP7VUPD; 	// PWM Comparison Value 7 Update Register
	AT91_REG	 PWMC_CMP7M; 	// PWM Comparison Mode 7 Register
	AT91_REG	 PWMC_CMP7MUPD; 	// PWM Comparison Mode 7 Update Register
	AT91_REG	 Reserved3[20]; 	// 
	AT91S_PWMC_CH	 PWMC_CH[8]; 	// PWMC Channel 0
} AT91S_PWMC, *AT91PS_PWMC;

// -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- 
#define AT91C_PWMC_DIVA       ((unsigned int) 0xFF <<  0) // (PWMC) CLKA divide factor.
#define AT91C_PWMC_PREA       ((unsigned int) 0xF <<  8) // (PWMC) Divider Input Clock Prescaler A
#define 	AT91C_PWMC_PREA_MCK                  ((unsigned int) 0x0 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_2            ((unsigned int) 0x1 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_4            ((unsigned int) 0x2 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_8            ((unsigned int) 0x3 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_16           ((unsigned int) 0x4 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_32           ((unsigned int) 0x5 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_64           ((unsigned int) 0x6 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_128          ((unsigned int) 0x7 <<  8) // (PWMC) 
#define 	AT91C_PWMC_PREA_MCK_DIV_256          ((unsigned int) 0x8 <<  8) // (PWMC) 
#define AT91C_PWMC_DIVB       ((unsigned int) 0xFF << 16) // (PWMC) CLKB divide factor.
#define AT91C_PWMC_PREB       ((unsigned int) 0xF << 24) // (PWMC) Divider Input Clock Prescaler B
#define 	AT91C_PWMC_PREB_MCK                  ((unsigned int) 0x0 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_2            ((unsigned int) 0x1 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_4            ((unsigned int) 0x2 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_8            ((unsigned int) 0x3 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_16           ((unsigned int) 0x4 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_32           ((unsigned int) 0x5 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_64           ((unsigned int) 0x6 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_128          ((unsigned int) 0x7 << 24) // (PWMC) 
#define 	AT91C_PWMC_PREB_MCK_DIV_256          ((unsigned int) 0x8 << 24) // (PWMC) 
#define AT91C_PWMC_CLKSEL     ((unsigned int) 0x1 << 31) // (PWMC) CCK Source Clock Selection
// -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- 
#define AT91C_PWMC_CHID0      ((unsigned int) 0x1 <<  0) // (PWMC) Channel ID 0
#define AT91C_PWMC_CHID1      ((unsigned int) 0x1 <<  1) // (PWMC) Channel ID 1
#define AT91C_PWMC_CHID2      ((unsigned int) 0x1 <<  2) // (PWMC) Channel ID 2
#define AT91C_PWMC_CHID3      ((unsigned int) 0x1 <<  3) // (PWMC) Channel ID 3
#define AT91C_PWMC_CHID4      ((unsigned int) 0x1 <<  4) // (PWMC) Channel ID 4
#define AT91C_PWMC_CHID5      ((unsigned int) 0x1 <<  5) // (PWMC) Channel ID 5
#define AT91C_PWMC_CHID6      ((unsigned int) 0x1 <<  6) // (PWMC) Channel ID 6
#define AT91C_PWMC_CHID7      ((unsigned int) 0x1 <<  7) // (PWMC) Channel ID 7
#define AT91C_PWMC_CHID8      ((unsigned int) 0x1 <<  8) // (PWMC) Channel ID 8
#define AT91C_PWMC_CHID9      ((unsigned int) 0x1 <<  9) // (PWMC) Channel ID 9
#define AT91C_PWMC_CHID10     ((unsigned int) 0x1 << 10) // (PWMC) Channel ID 10
#define AT91C_PWMC_CHID11     ((unsigned int) 0x1 << 11) // (PWMC) Channel ID 11
#define AT91C_PWMC_CHID12     ((unsigned int) 0x1 << 12) // (PWMC) Channel ID 12
#define AT91C_PWMC_CHID13     ((unsigned int) 0x1 << 13) // (PWMC) Channel ID 13
#define AT91C_PWMC_CHID14     ((unsigned int) 0x1 << 14) // (PWMC) Channel ID 14
#define AT91C_PWMC_CHID15     ((unsigned int) 0x1 << 15) // (PWMC) Channel ID 15
// -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- 
// -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- 
// -------- PWMC_IER1 : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- 
#define AT91C_PWMC_FCHID0     ((unsigned int) 0x1 << 16) // (PWMC) Fault Event Channel ID 0
#define AT91C_PWMC_FCHID1     ((unsigned int) 0x1 << 17) // (PWMC) Fault Event Channel ID 1
#define AT91C_PWMC_FCHID2     ((unsigned int) 0x1 << 18) // (PWMC) Fault Event Channel ID 2
#define AT91C_PWMC_FCHID3     ((unsigned int) 0x1 << 19) // (PWMC) Fault Event Channel ID 3
#define AT91C_PWMC_FCHID4     ((unsigned int) 0x1 << 20) // (PWMC) Fault Event Channel ID 4
#define AT91C_PWMC_FCHID5     ((unsigned int) 0x1 << 21) // (PWMC) Fault Event Channel ID 5
#define AT91C_PWMC_FCHID6     ((unsigned int) 0x1 << 22) // (PWMC) Fault Event Channel ID 6
#define AT91C_PWMC_FCHID7     ((unsigned int) 0x1 << 23) // (PWMC) Fault Event Channel ID 7
#define AT91C_PWMC_FCHID8     ((unsigned int) 0x1 << 24) // (PWMC) Fault Event Channel ID 8
#define AT91C_PWMC_FCHID9     ((unsigned int) 0x1 << 25) // (PWMC) Fault Event Channel ID 9
#define AT91C_PWMC_FCHID10    ((unsigned int) 0x1 << 26) // (PWMC) Fault Event Channel ID 10
#define AT91C_PWMC_FCHID11    ((unsigned int) 0x1 << 27) // (PWMC) Fault Event Channel ID 11
#define AT91C_PWMC_FCHID12    ((unsigned int) 0x1 << 28) // (PWMC) Fault Event Channel ID 12
#define AT91C_PWMC_FCHID13    ((unsigned int) 0x1 << 29) // (PWMC) Fault Event Channel ID 13
#define AT91C_PWMC_FCHID14    ((unsigned int) 0x1 << 30) // (PWMC) Fault Event Channel ID 14
#define AT91C_PWMC_FCHID15    ((unsigned int) 0x1 << 31) // (PWMC) Fault Event Channel ID 15
// -------- PWMC_IDR1 : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- 
// -------- PWMC_IMR1 : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- 
// -------- PWMC_ISR1 : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- 
// -------- PWMC_SYNC : (PWMC Offset: 0x20) PWMC Synchronous Channels Register -------- 
#define AT91C_PWMC_SYNC0      ((unsigned int) 0x1 <<  0) // (PWMC) Synchronous Channel ID 0
#define AT91C_PWMC_SYNC1      ((unsigned int) 0x1 <<  1) // (PWMC) Synchronous Channel ID 1
#define AT91C_PWMC_SYNC2      ((unsigned int) 0x1 <<  2) // (PWMC) Synchronous Channel ID 2
#define AT91C_PWMC_SYNC3      ((unsigned int) 0x1 <<  3) // (PWMC) Synchronous Channel ID 3
#define AT91C_PWMC_SYNC4      ((unsigned int) 0x1 <<  4) // (PWMC) Synchronous Channel ID 4
#define AT91C_PWMC_SYNC5      ((unsigned int) 0x1 <<  5) // (PWMC) Synchronous Channel ID 5
#define AT91C_PWMC_SYNC6      ((unsigned int) 0x1 <<  6) // (PWMC) Synchronous Channel ID 6
#define AT91C_PWMC_SYNC7      ((unsigned int) 0x1 <<  7) // (PWMC) Synchronous Channel ID 7
#define AT91C_PWMC_SYNC8      ((unsigned int) 0x1 <<  8) // (PWMC) Synchronous Channel ID 8
#define AT91C_PWMC_SYNC9      ((unsigned int) 0x1 <<  9) // (PWMC) Synchronous Channel ID 9
#define AT91C_PWMC_SYNC10     ((unsigned int) 0x1 << 10) // (PWMC) Synchronous Channel ID 10
#define AT91C_PWMC_SYNC11     ((unsigned int) 0x1 << 11) // (PWMC) Synchronous Channel ID 11
#define AT91C_PWMC_SYNC12     ((unsigned int) 0x1 << 12) // (PWMC) Synchronous Channel ID 12
#define AT91C_PWMC_SYNC13     ((unsigned int) 0x1 << 13) // (PWMC) Synchronous Channel ID 13
#define AT91C_PWMC_SYNC14     ((unsigned int) 0x1 << 14) // (PWMC) Synchronous Channel ID 14
#define AT91C_PWMC_SYNC15     ((unsigned int) 0x1 << 15) // (PWMC) Synchronous Channel ID 15
#define AT91C_PWMC_UPDM       ((unsigned int) 0x3 << 16) // (PWMC) Synchronous Channels Update mode
#define 	AT91C_PWMC_UPDM_MODE0                ((unsigned int) 0x0 << 16) // (PWMC) Manual write of data and manual trigger of the update
#define 	AT91C_PWMC_UPDM_MODE1                ((unsigned int) 0x1 << 16) // (PWMC) Manual write of data and automatic trigger of the update
#define 	AT91C_PWMC_UPDM_MODE2                ((unsigned int) 0x2 << 16) // (PWMC) Automatic write of data and automatic trigger of the update
// -------- PWMC_UPCR : (PWMC Offset: 0x28) PWMC Update Control Register -------- 
#define AT91C_PWMC_UPDULOCK   ((unsigned int) 0x1 <<  0) // (PWMC) Synchronized Channels Duty Cycle Update Unlock
// -------- PWMC_SCUP : (PWMC Offset: 0x2c) PWM Update Period Register -------- 
#define AT91C_PWMC_UPR        ((unsigned int) 0xF <<  0) // (PWMC) PWM Update Period.
#define AT91C_PWMC_UPRCNT     ((unsigned int) 0xF <<  4) // (PWMC) PWM Update Period Counter.
// -------- PWMC_SCUPUPD : (PWMC Offset: 0x30) PWM Update Period Update Register -------- 
#define AT91C_PWMC_UPVUPDAL   ((unsigned int) 0xF <<  0) // (PWMC) PWM Update Period Update.
// -------- PWMC_IER2 : (PWMC Offset: 0x34) PWMC Interrupt Enable Register -------- 
#define AT91C_PWMC_WRDY       ((unsigned int) 0x1 <<  0) // (PWMC) PDC Write Ready
#define AT91C_PWMC_ENDTX      ((unsigned int) 0x1 <<  1) // (PWMC) PDC End of TX Buffer
#define AT91C_PWMC_TXBUFE     ((unsigned int) 0x1 <<  2) // (PWMC) PDC End of TX Buffer
#define AT91C_PWMC_UNRE       ((unsigned int) 0x1 <<  3) // (PWMC) PDC End of TX Buffer
// -------- PWMC_IDR2 : (PWMC Offset: 0x38) PWMC Interrupt Disable Register -------- 
// -------- PWMC_IMR2 : (PWMC Offset: 0x3c) PWMC Interrupt Mask Register -------- 
// -------- PWMC_ISR2 : (PWMC Offset: 0x40) PWMC Interrupt Status Register -------- 
#define AT91C_PWMC_CMPM0      ((unsigned int) 0x1 <<  8) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM1      ((unsigned int) 0x1 <<  9) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM2      ((unsigned int) 0x1 << 10) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM3      ((unsigned int) 0x1 << 11) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM4      ((unsigned int) 0x1 << 12) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM5      ((unsigned int) 0x1 << 13) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM6      ((unsigned int) 0x1 << 14) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPM7      ((unsigned int) 0x1 << 15) // (PWMC) Comparison x Match
#define AT91C_PWMC_CMPU0      ((unsigned int) 0x1 << 16) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU1      ((unsigned int) 0x1 << 17) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU2      ((unsigned int) 0x1 << 18) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU3      ((unsigned int) 0x1 << 19) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU4      ((unsigned int) 0x1 << 20) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU5      ((unsigned int) 0x1 << 21) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU6      ((unsigned int) 0x1 << 22) // (PWMC) Comparison x Update
#define AT91C_PWMC_CMPU7      ((unsigned int) 0x1 << 23) // (PWMC) Comparison x Update
// -------- PWMC_OOV : (PWMC Offset: 0x44) PWM Output Override Value Register -------- 
#define AT91C_PWMC_OOVH0      ((unsigned int) 0x1 <<  0) // (PWMC) Output Override Value for PWMH output of the channel 0
#define AT91C_PWMC_OOVH1      ((unsigned int) 0x1 <<  1) // (PWMC) Output Override Value for PWMH output of the channel 1
#define AT91C_PWMC_OOVH2      ((unsigned int) 0x1 <<  2) // (PWMC) Output Override Value for PWMH output of the channel 2
#define AT91C_PWMC_OOVH3      ((unsigned int) 0x1 <<  3) // (PWMC) Output Override Value for PWMH output of the channel 3
#define AT91C_PWMC_OOVH4      ((unsigned int) 0x1 <<  4) // (PWMC) Output Override Value for PWMH output of the channel 4
#define AT91C_PWMC_OOVH5      ((unsigned int) 0x1 <<  5) // (PWMC) Output Override Value for PWMH output of the channel 5
#define AT91C_PWMC_OOVH6      ((unsigned int) 0x1 <<  6) // (PWMC) Output Override Value for PWMH output of the channel 6
#define AT91C_PWMC_OOVH7      ((unsigned int) 0x1 <<  7) // (PWMC) Output Override Value for PWMH output of the channel 7
#define AT91C_PWMC_OOVH8      ((unsigned int) 0x1 <<  8) // (PWMC) Output Override Value for PWMH output of the channel 8
#define AT91C_PWMC_OOVH9      ((unsigned int) 0x1 <<  9) // (PWMC) Output Override Value for PWMH output of the channel 9
#define AT91C_PWMC_OOVH10     ((unsigned int) 0x1 << 10) // (PWMC) Output Override Value for PWMH output of the channel 10
#define AT91C_PWMC_OOVH11     ((unsigned int) 0x1 << 11) // (PWMC) Output Override Value for PWMH output of the channel 11
#define AT91C_PWMC_OOVH12     ((unsigned int) 0x1 << 12) // (PWMC) Output Override Value for PWMH output of the channel 12
#define AT91C_PWMC_OOVH13     ((unsigned int) 0x1 << 13) // (PWMC) Output Override Value for PWMH output of the channel 13
#define AT91C_PWMC_OOVH14     ((unsigned int) 0x1 << 14) // (PWMC) Output Override Value for PWMH output of the channel 14
#define AT91C_PWMC_OOVH15     ((unsigned int) 0x1 << 15) // (PWMC) Output Override Value for PWMH output of the channel 15
#define AT91C_PWMC_OOVL0      ((unsigned int) 0x1 << 16) // (PWMC) Output Override Value for PWML output of the channel 0
#define AT91C_PWMC_OOVL1      ((unsigned int) 0x1 << 17) // (PWMC) Output Override Value for PWML output of the channel 1
#define AT91C_PWMC_OOVL2      ((unsigned int) 0x1 << 18) // (PWMC) Output Override Value for PWML output of the channel 2
#define AT91C_PWMC_OOVL3      ((unsigned int) 0x1 << 19) // (PWMC) Output Override Value for PWML output of the channel 3
#define AT91C_PWMC_OOVL4      ((unsigned int) 0x1 << 20) // (PWMC) Output Override Value for PWML output of the channel 4
#define AT91C_PWMC_OOVL5      ((unsigned int) 0x1 << 21) // (PWMC) Output Override Value for PWML output of the channel 5
#define AT91C_PWMC_OOVL6      ((unsigned int) 0x1 << 22) // (PWMC) Output Override Value for PWML output of the channel 6
#define AT91C_PWMC_OOVL7      ((unsigned int) 0x1 << 23) // (PWMC) Output Override Value for PWML output of the channel 7
#define AT91C_PWMC_OOVL8      ((unsigned int) 0x1 << 24) // (PWMC) Output Override Value for PWML output of the channel 8
#define AT91C_PWMC_OOVL9      ((unsigned int) 0x1 << 25) // (PWMC) Output Override Value for PWML output of the channel 9
#define AT91C_PWMC_OOVL10     ((unsigned int) 0x1 << 26) // (PWMC) Output Override Value for PWML output of the channel 10
#define AT91C_PWMC_OOVL11     ((unsigned int) 0x1 << 27) // (PWMC) Output Override Value for PWML output of the channel 11
#define AT91C_PWMC_OOVL12     ((unsigned int) 0x1 << 28) // (PWMC) Output Override Value for PWML output of the channel 12
#define AT91C_PWMC_OOVL13     ((unsigned int) 0x1 << 29) // (PWMC) Output Override Value for PWML output of the channel 13
#define AT91C_PWMC_OOVL14     ((unsigned int) 0x1 << 30) // (PWMC) Output Override Value for PWML output of the channel 14
#define AT91C_PWMC_OOVL15     ((unsigned int) 0x1 << 31) // (PWMC) Output Override Value for PWML output of the channel 15
// -------- PWMC_OS : (PWMC Offset: 0x48) PWM Output Selection Register -------- 
#define AT91C_PWMC_OSH0       ((unsigned int) 0x1 <<  0) // (PWMC) Output Selection for PWMH output of the channel 0
#define AT91C_PWMC_OSH1       ((unsigned int) 0x1 <<  1) // (PWMC) Output Selection for PWMH output of the channel 1
#define AT91C_PWMC_OSH2       ((unsigned int) 0x1 <<  2) // (PWMC) Output Selection for PWMH output of the channel 2
#define AT91C_PWMC_OSH3       ((unsigned int) 0x1 <<  3) // (PWMC) Output Selection for PWMH output of the channel 3
#define AT91C_PWMC_OSH4       ((unsigned int) 0x1 <<  4) // (PWMC) Output Selection for PWMH output of the channel 4
#define AT91C_PWMC_OSH5       ((unsigned int) 0x1 <<  5) // (PWMC) Output Selection for PWMH output of the channel 5
#define AT91C_PWMC_OSH6       ((unsigned int) 0x1 <<  6) // (PWMC) Output Selection for PWMH output of the channel 6
#define AT91C_PWMC_OSH7       ((unsigned int) 0x1 <<  7) // (PWMC) Output Selection for PWMH output of the channel 7
#define AT91C_PWMC_OSH8       ((unsigned int) 0x1 <<  8) // (PWMC) Output Selection for PWMH output of the channel 8
#define AT91C_PWMC_OSH9       ((unsigned int) 0x1 <<  9) // (PWMC) Output Selection for PWMH output of the channel 9
#define AT91C_PWMC_OSH10      ((unsigned int) 0x1 << 10) // (PWMC) Output Selection for PWMH output of the channel 10
#define AT91C_PWMC_OSH11      ((unsigned int) 0x1 << 11) // (PWMC) Output Selection for PWMH output of the channel 11
#define AT91C_PWMC_OSH12      ((unsigned int) 0x1 << 12) // (PWMC) Output Selection for PWMH output of the channel 12
#define AT91C_PWMC_OSH13      ((unsigned int) 0x1 << 13) // (PWMC) Output Selection for PWMH output of the channel 13
#define AT91C_PWMC_OSH14      ((unsigned int) 0x1 << 14) // (PWMC) Output Selection for PWMH output of the channel 14
#define AT91C_PWMC_OSH15      ((unsigned int) 0x1 << 15) // (PWMC) Output Selection for PWMH output of the channel 15
#define AT91C_PWMC_OSL0       ((unsigned int) 0x1 << 16) // (PWMC) Output Selection for PWML output of the channel 0
#define AT91C_PWMC_OSL1       ((unsigned int) 0x1 << 17) // (PWMC) Output Selection for PWML output of the channel 1
#define AT91C_PWMC_OSL2       ((unsigned int) 0x1 << 18) // (PWMC) Output Selection for PWML output of the channel 2
#define AT91C_PWMC_OSL3       ((unsigned int) 0x1 << 19) // (PWMC) Output Selection for PWML output of the channel 3
#define AT91C_PWMC_OSL4       ((unsigned int) 0x1 << 20) // (PWMC) Output Selection for PWML output of the channel 4
#define AT91C_PWMC_OSL5       ((unsigned int) 0x1 << 21) // (PWMC) Output Selection for PWML output of the channel 5
#define AT91C_PWMC_OSL6       ((unsigned int) 0x1 << 22) // (PWMC) Output Selection for PWML output of the channel 6
#define AT91C_PWMC_OSL7       ((unsigned int) 0x1 << 23) // (PWMC) Output Selection for PWML output of the channel 7
#define AT91C_PWMC_OSL8       ((unsigned int) 0x1 << 24) // (PWMC) Output Selection for PWML output of the channel 8
#define AT91C_PWMC_OSL9       ((unsigned int) 0x1 << 25) // (PWMC) Output Selection for PWML output of the channel 9
#define AT91C_PWMC_OSL10      ((unsigned int) 0x1 << 26) // (PWMC) Output Selection for PWML output of the channel 10
#define AT91C_PWMC_OSL11      ((unsigned int) 0x1 << 27) // (PWMC) Output Selection for PWML output of the channel 11
#define AT91C_PWMC_OSL12      ((unsigned int) 0x1 << 28) // (PWMC) Output Selection for PWML output of the channel 12
#define AT91C_PWMC_OSL13      ((unsigned int) 0x1 << 29) // (PWMC) Output Selection for PWML output of the channel 13
#define AT91C_PWMC_OSL14      ((unsigned int) 0x1 << 30) // (PWMC) Output Selection for PWML output of the channel 14
#define AT91C_PWMC_OSL15      ((unsigned int) 0x1 << 31) // (PWMC) Output Selection for PWML output of the channel 15
// -------- PWMC_OSS : (PWMC Offset: 0x4c) PWM Output Selection Set Register -------- 
#define AT91C_PWMC_OSSH0      ((unsigned int) 0x1 <<  0) // (PWMC) Output Selection Set for PWMH output of the channel 0
#define AT91C_PWMC_OSSH1      ((unsigned int) 0x1 <<  1) // (PWMC) Output Selection Set for PWMH output of the channel 1
#define AT91C_PWMC_OSSH2      ((unsigned int) 0x1 <<  2) // (PWMC) Output Selection Set for PWMH output of the channel 2
#define AT91C_PWMC_OSSH3      ((unsigned int) 0x1 <<  3) // (PWMC) Output Selection Set for PWMH output of the channel 3
#define AT91C_PWMC_OSSH4      ((unsigned int) 0x1 <<  4) // (PWMC) Output Selection Set for PWMH output of the channel 4
#define AT91C_PWMC_OSSH5      ((unsigned int) 0x1 <<  5) // (PWMC) Output Selection Set for PWMH output of the channel 5
#define AT91C_PWMC_OSSH6      ((unsigned int) 0x1 <<  6) // (PWMC) Output Selection Set for PWMH output of the channel 6
#define AT91C_PWMC_OSSH7      ((unsigned int) 0x1 <<  7) // (PWMC) Output Selection Set for PWMH output of the channel 7
#define AT91C_PWMC_OSSH8      ((unsigned int) 0x1 <<  8) // (PWMC) Output Selection Set for PWMH output of the channel 8
#define AT91C_PWMC_OSSH9      ((unsigned int) 0x1 <<  9) // (PWMC) Output Selection Set for PWMH output of the channel 9
#define AT91C_PWMC_OSSH10     ((unsigned int) 0x1 << 10) // (PWMC) Output Selection Set for PWMH output of the channel 10
#define AT91C_PWMC_OSSH11     ((unsigned int) 0x1 << 11) // (PWMC) Output Selection Set for PWMH output of the channel 11
#define AT91C_PWMC_OSSH12     ((unsigned int) 0x1 << 12) // (PWMC) Output Selection Set for PWMH output of the channel 12
#define AT91C_PWMC_OSSH13     ((unsigned int) 0x1 << 13) // (PWMC) Output Selection Set for PWMH output of the channel 13
#define AT91C_PWMC_OSSH14     ((unsigned int) 0x1 << 14) // (PWMC) Output Selection Set for PWMH output of the channel 14
#define AT91C_PWMC_OSSH15     ((unsigned int) 0x1 << 15) // (PWMC) Output Selection Set for PWMH output of the channel 15
#define AT91C_PWMC_OSSL0      ((unsigned int) 0x1 << 16) // (PWMC) Output Selection Set for PWML output of the channel 0
#define AT91C_PWMC_OSSL1      ((unsigned int) 0x1 << 17) // (PWMC) Output Selection Set for PWML output of the channel 1
#define AT91C_PWMC_OSSL2      ((unsigned int) 0x1 << 18) // (PWMC) Output Selection Set for PWML output of the channel 2
#define AT91C_PWMC_OSSL3      ((unsigned int) 0x1 << 19) // (PWMC) Output Selection Set for PWML output of the channel 3
#define AT91C_PWMC_OSSL4      ((unsigned int) 0x1 << 20) // (PWMC) Output Selection Set for PWML output of the channel 4
#define AT91C_PWMC_OSSL5      ((unsigned int) 0x1 << 21) // (PWMC) Output Selection Set for PWML output of the channel 5
#define AT91C_PWMC_OSSL6      ((unsigned int) 0x1 << 22) // (PWMC) Output Selection Set for PWML output of the channel 6
#define AT91C_PWMC_OSSL7      ((unsigned int) 0x1 << 23) // (PWMC) Output Selection Set for PWML output of the channel 7
#define AT91C_PWMC_OSSL8      ((unsigned int) 0x1 << 24) // (PWMC) Output Selection Set for PWML output of the channel 8
#define AT91C_PWMC_OSSL9      ((unsigned int) 0x1 << 25) // (PWMC) Output Selection Set for PWML output of the channel 9
#define AT91C_PWMC_OSSL10     ((unsigned int) 0x1 << 26) // (PWMC) Output Selection Set for PWML output of the channel 10
#define AT91C_PWMC_OSSL11     ((unsigned int) 0x1 << 27) // (PWMC) Output Selection Set for PWML output of the channel 11
#define AT91C_PWMC_OSSL12     ((unsigned int) 0x1 << 28) // (PWMC) Output Selection Set for PWML output of the channel 12
#define AT91C_PWMC_OSSL13     ((unsigned int) 0x1 << 29) // (PWMC) Output Selection Set for PWML output of the channel 13
#define AT91C_PWMC_OSSL14     ((unsigned int) 0x1 << 30) // (PWMC) Output Selection Set for PWML output of the channel 14
#define AT91C_PWMC_OSSL15     ((unsigned int) 0x1 << 31) // (PWMC) Output Selection Set for PWML output of the channel 15
// -------- PWMC_OSC : (PWMC Offset: 0x50) PWM Output Selection Clear Register -------- 
#define AT91C_PWMC_OSCH0      ((unsigned int) 0x1 <<  0) // (PWMC) Output Selection Clear for PWMH output of the channel 0
#define AT91C_PWMC_OSCH1      ((unsigned int) 0x1 <<  1) // (PWMC) Output Selection Clear for PWMH output of the channel 1
#define AT91C_PWMC_OSCH2      ((unsigned int) 0x1 <<  2) // (PWMC) Output Selection Clear for PWMH output of the channel 2
#define AT91C_PWMC_OSCH3      ((unsigned int) 0x1 <<  3) // (PWMC) Output Selection Clear for PWMH output of the channel 3
#define AT91C_PWMC_OSCH4      ((unsigned int) 0x1 <<  4) // (PWMC) Output Selection Clear for PWMH output of the channel 4
#define AT91C_PWMC_OSCH5      ((unsigned int) 0x1 <<  5) // (PWMC) Output Selection Clear for PWMH output of the channel 5
#define AT91C_PWMC_OSCH6      ((unsigned int) 0x1 <<  6) // (PWMC) Output Selection Clear for PWMH output of the channel 6
#define AT91C_PWMC_OSCH7      ((unsigned int) 0x1 <<  7) // (PWMC) Output Selection Clear for PWMH output of the channel 7
#define AT91C_PWMC_OSCH8      ((unsigned int) 0x1 <<  8) // (PWMC) Output Selection Clear for PWMH output of the channel 8
#define AT91C_PWMC_OSCH9      ((unsigned int) 0x1 <<  9) // (PWMC) Output Selection Clear for PWMH output of the channel 9
#define AT91C_PWMC_OSCH10     ((unsigned int) 0x1 << 10) // (PWMC) Output Selection Clear for PWMH output of the channel 10
#define AT91C_PWMC_OSCH11     ((unsigned int) 0x1 << 11) // (PWMC) Output Selection Clear for PWMH output of the channel 11
#define AT91C_PWMC_OSCH12     ((unsigned int) 0x1 << 12) // (PWMC) Output Selection Clear for PWMH output of the channel 12
#define AT91C_PWMC_OSCH13     ((unsigned int) 0x1 << 13) // (PWMC) Output Selection Clear for PWMH output of the channel 13
#define AT91C_PWMC_OSCH14     ((unsigned int) 0x1 << 14) // (PWMC) Output Selection Clear for PWMH output of the channel 14
#define AT91C_PWMC_OSCH15     ((unsigned int) 0x1 << 15) // (PWMC) Output Selection Clear for PWMH output of the channel 15
#define AT91C_PWMC_OSCL0      ((unsigned int) 0x1 << 16) // (PWMC) Output Selection Clear for PWML output of the channel 0
#define AT91C_PWMC_OSCL1      ((unsigned int) 0x1 << 17) // (PWMC) Output Selection Clear for PWML output of the channel 1
#define AT91C_PWMC_OSCL2      ((unsigned int) 0x1 << 18) // (PWMC) Output Selection Clear for PWML output of the channel 2
#define AT91C_PWMC_OSCL3      ((unsigned int) 0x1 << 19) // (PWMC) Output Selection Clear for PWML output of the channel 3
#define AT91C_PWMC_OSCL4      ((unsigned int) 0x1 << 20) // (PWMC) Output Selection Clear for PWML output of the channel 4
#define AT91C_PWMC_OSCL5      ((unsigned int) 0x1 << 21) // (PWMC) Output Selection Clear for PWML output of the channel 5
#define AT91C_PWMC_OSCL6      ((unsigned int) 0x1 << 22) // (PWMC) Output Selection Clear for PWML output of the channel 6
#define AT91C_PWMC_OSCL7      ((unsigned int) 0x1 << 23) // (PWMC) Output Selection Clear for PWML output of the channel 7
#define AT91C_PWMC_OSCL8      ((unsigned int) 0x1 << 24) // (PWMC) Output Selection Clear for PWML output of the channel 8
#define AT91C_PWMC_OSCL9      ((unsigned int) 0x1 << 25) // (PWMC) Output Selection Clear for PWML output of the channel 9
#define AT91C_PWMC_OSCL10     ((unsigned int) 0x1 << 26) // (PWMC) Output Selection Clear for PWML output of the channel 10
#define AT91C_PWMC_OSCL11     ((unsigned int) 0x1 << 27) // (PWMC) Output Selection Clear for PWML output of the channel 11
#define AT91C_PWMC_OSCL12     ((unsigned int) 0x1 << 28) // (PWMC) Output Selection Clear for PWML output of the channel 12
#define AT91C_PWMC_OSCL13     ((unsigned int) 0x1 << 29) // (PWMC) Output Selection Clear for PWML output of the channel 13
#define AT91C_PWMC_OSCL14     ((unsigned int) 0x1 << 30) // (PWMC) Output Selection Clear for PWML output of the channel 14
#define AT91C_PWMC_OSCL15     ((unsigned int) 0x1 << 31) // (PWMC) Output Selection Clear for PWML output of the channel 15
// -------- PWMC_OSSUPD : (PWMC Offset: 0x54) Output Selection Set for PWMH / PWML output of the channel x -------- 
#define AT91C_PWMC_OSSUPDH0   ((unsigned int) 0x1 <<  0) // (PWMC) Output Selection Set for PWMH output of the channel 0
#define AT91C_PWMC_OSSUPDH1   ((unsigned int) 0x1 <<  1) // (PWMC) Output Selection Set for PWMH output of the channel 1
#define AT91C_PWMC_OSSUPDH2   ((unsigned int) 0x1 <<  2) // (PWMC) Output Selection Set for PWMH output of the channel 2
#define AT91C_PWMC_OSSUPDH3   ((unsigned int) 0x1 <<  3) // (PWMC) Output Selection Set for PWMH output of the channel 3
#define AT91C_PWMC_OSSUPDH4   ((unsigned int) 0x1 <<  4) // (PWMC) Output Selection Set for PWMH output of the channel 4
#define AT91C_PWMC_OSSUPDH5   ((unsigned int) 0x1 <<  5) // (PWMC) Output Selection Set for PWMH output of the channel 5
#define AT91C_PWMC_OSSUPDH6   ((unsigned int) 0x1 <<  6) // (PWMC) Output Selection Set for PWMH output of the channel 6
#define AT91C_PWMC_OSSUPDH7   ((unsigned int) 0x1 <<  7) // (PWMC) Output Selection Set for PWMH output of the channel 7
#define AT91C_PWMC_OSSUPDH8   ((unsigned int) 0x1 <<  8) // (PWMC) Output Selection Set for PWMH output of the channel 8
#define AT91C_PWMC_OSSUPDH9   ((unsigned int) 0x1 <<  9) // (PWMC) Output Selection Set for PWMH output of the channel 9
#define AT91C_PWMC_OSSUPDH10  ((unsigned int) 0x1 << 10) // (PWMC) Output Selection Set for PWMH output of the channel 10
#define AT91C_PWMC_OSSUPDH11  ((unsigned int) 0x1 << 11) // (PWMC) Output Selection Set for PWMH output of the channel 11
#define AT91C_PWMC_OSSUPDH12  ((unsigned int) 0x1 << 12) // (PWMC) Output Selection Set for PWMH output of the channel 12
#define AT91C_PWMC_OSSUPDH13  ((unsigned int) 0x1 << 13) // (PWMC) Output Selection Set for PWMH output of the channel 13
#define AT91C_PWMC_OSSUPDH14  ((unsigned int) 0x1 << 14) // (PWMC) Output Selection Set for PWMH output of the channel 14
#define AT91C_PWMC_OSSUPDH15  ((unsigned int) 0x1 << 15) // (PWMC) Output Selection Set for PWMH output of the channel 15
#define AT91C_PWMC_OSSUPDL0   ((unsigned int) 0x1 << 16) // (PWMC) Output Selection Set for PWML output of the channel 0
#define AT91C_PWMC_OSSUPDL1   ((unsigned int) 0x1 << 17) // (PWMC) Output Selection Set for PWML output of the channel 1
#define AT91C_PWMC_OSSUPDL2   ((unsigned int) 0x1 << 18) // (PWMC) Output Selection Set for PWML output of the channel 2
#define AT91C_PWMC_OSSUPDL3   ((unsigned int) 0x1 << 19) // (PWMC) Output Selection Set for PWML output of the channel 3
#define AT91C_PWMC_OSSUPDL4   ((unsigned int) 0x1 << 20) // (PWMC) Output Selection Set for PWML output of the channel 4
#define AT91C_PWMC_OSSUPDL5   ((unsigned int) 0x1 << 21) // (PWMC) Output Selection Set for PWML output of the channel 5
#define AT91C_PWMC_OSSUPDL6   ((unsigned int) 0x1 << 22) // (PWMC) Output Selection Set for PWML output of the channel 6
#define AT91C_PWMC_OSSUPDL7   ((unsigned int) 0x1 << 23) // (PWMC) Output Selection Set for PWML output of the channel 7
#define AT91C_PWMC_OSSUPDL8   ((unsigned int) 0x1 << 24) // (PWMC) Output Selection Set for PWML output of the channel 8
#define AT91C_PWMC_OSSUPDL9   ((unsigned int) 0x1 << 25) // (PWMC) Output Selection Set for PWML output of the channel 9
#define AT91C_PWMC_OSSUPDL10  ((unsigned int) 0x1 << 26) // (PWMC) Output Selection Set for PWML output of the channel 10
#define AT91C_PWMC_OSSUPDL11  ((unsigned int) 0x1 << 27) // (PWMC) Output Selection Set for PWML output of the channel 11
#define AT91C_PWMC_OSSUPDL12  ((unsigned int) 0x1 << 28) // (PWMC) Output Selection Set for PWML output of the channel 12
#define AT91C_PWMC_OSSUPDL13  ((unsigned int) 0x1 << 29) // (PWMC) Output Selection Set for PWML output of the channel 13
#define AT91C_PWMC_OSSUPDL14  ((unsigned int) 0x1 << 30) // (PWMC) Output Selection Set for PWML output of the channel 14
#define AT91C_PWMC_OSSUPDL15  ((unsigned int) 0x1 << 31) // (PWMC) Output Selection Set for PWML output of the channel 15
// -------- PWMC_OSCUPD : (PWMC Offset: 0x58) Output Selection Clear for PWMH / PWML output of the channel x -------- 
#define AT91C_PWMC_OSCUPDH0   ((unsigned int) 0x1 <<  0) // (PWMC) Output Selection Clear for PWMH output of the channel 0
#define AT91C_PWMC_OSCUPDH1   ((unsigned int) 0x1 <<  1) // (PWMC) Output Selection Clear for PWMH output of the channel 1
#define AT91C_PWMC_OSCUPDH2   ((unsigned int) 0x1 <<  2) // (PWMC) Output Selection Clear for PWMH output of the channel 2
#define AT91C_PWMC_OSCUPDH3   ((unsigned int) 0x1 <<  3) // (PWMC) Output Selection Clear for PWMH output of the channel 3
#define AT91C_PWMC_OSCUPDH4   ((unsigned int) 0x1 <<  4) // (PWMC) Output Selection Clear for PWMH output of the channel 4
#define AT91C_PWMC_OSCUPDH5   ((unsigned int) 0x1 <<  5) // (PWMC) Output Selection Clear for PWMH output of the channel 5
#define AT91C_PWMC_OSCUPDH6   ((unsigned int) 0x1 <<  6) // (PWMC) Output Selection Clear for PWMH output of the channel 6
#define AT91C_PWMC_OSCUPDH7   ((unsigned int) 0x1 <<  7) // (PWMC) Output Selection Clear for PWMH output of the channel 7
#define AT91C_PWMC_OSCUPDH8   ((unsigned int) 0x1 <<  8) // (PWMC) Output Selection Clear for PWMH output of the channel 8
#define AT91C_PWMC_OSCUPDH9   ((unsigned int) 0x1 <<  9) // (PWMC) Output Selection Clear for PWMH output of the channel 9
#define AT91C_PWMC_OSCUPDH10  ((unsigned int) 0x1 << 10) // (PWMC) Output Selection Clear for PWMH output of the channel 10
#define AT91C_PWMC_OSCUPDH11  ((unsigned int) 0x1 << 11) // (PWMC) Output Selection Clear for PWMH output of the channel 11
#define AT91C_PWMC_OSCUPDH12  ((unsigned int) 0x1 << 12) // (PWMC) Output Selection Clear for PWMH output of the channel 12
#define AT91C_PWMC_OSCUPDH13  ((unsigned int) 0x1 << 13) // (PWMC) Output Selection Clear for PWMH output of the channel 13
#define AT91C_PWMC_OSCUPDH14  ((unsigned int) 0x1 << 14) // (PWMC) Output Selection Clear for PWMH output of the channel 14
#define AT91C_PWMC_OSCUPDH15  ((unsigned int) 0x1 << 15) // (PWMC) Output Selection Clear for PWMH output of the channel 15
#define AT91C_PWMC_OSCUPDL0   ((unsigned int) 0x1 << 16) // (PWMC) Output Selection Clear for PWML output of the channel 0
#define AT91C_PWMC_OSCUPDL1   ((unsigned int) 0x1 << 17) // (PWMC) Output Selection Clear for PWML output of the channel 1
#define AT91C_PWMC_OSCUPDL2   ((unsigned int) 0x1 << 18) // (PWMC) Output Selection Clear for PWML output of the channel 2
#define AT91C_PWMC_OSCUPDL3   ((unsigned int) 0x1 << 19) // (PWMC) Output Selection Clear for PWML output of the channel 3
#define AT91C_PWMC_OSCUPDL4   ((unsigned int) 0x1 << 20) // (PWMC) Output Selection Clear for PWML output of the channel 4
#define AT91C_PWMC_OSCUPDL5   ((unsigned int) 0x1 << 21) // (PWMC) Output Selection Clear for PWML output of the channel 5
#define AT91C_PWMC_OSCUPDL6   ((unsigned int) 0x1 << 22) // (PWMC) Output Selection Clear for PWML output of the channel 6
#define AT91C_PWMC_OSCUPDL7   ((unsigned int) 0x1 << 23) // (PWMC) Output Selection Clear for PWML output of the channel 7
#define AT91C_PWMC_OSCUPDL8   ((unsigned int) 0x1 << 24) // (PWMC) Output Selection Clear for PWML output of the channel 8
#define AT91C_PWMC_OSCUPDL9   ((unsigned int) 0x1 << 25) // (PWMC) Output Selection Clear for PWML output of the channel 9
#define AT91C_PWMC_OSCUPDL10  ((unsigned int) 0x1 << 26) // (PWMC) Output Selection Clear for PWML output of the channel 10
#define AT91C_PWMC_OSCUPDL11  ((unsigned int) 0x1 << 27) // (PWMC) Output Selection Clear for PWML output of the channel 11
#define AT91C_PWMC_OSCUPDL12  ((unsigned int) 0x1 << 28) // (PWMC) Output Selection Clear for PWML output of the channel 12
#define AT91C_PWMC_OSCUPDL13  ((unsigned int) 0x1 << 29) // (PWMC) Output Selection Clear for PWML output of the channel 13
#define AT91C_PWMC_OSCUPDL14  ((unsigned int) 0x1 << 30) // (PWMC) Output Selection Clear for PWML output of the channel 14
#define AT91C_PWMC_OSCUPDL15  ((unsigned int) 0x1 << 31) // (PWMC) Output Selection Clear for PWML output of the channel 15
// -------- PWMC_FMR : (PWMC Offset: 0x5c) PWM Fault Mode Register -------- 
#define AT91C_PWMC_FPOL0      ((unsigned int) 0x1 <<  0) // (PWMC) Fault Polarity on fault input 0
#define AT91C_PWMC_FPOL1      ((unsigned int) 0x1 <<  1) // (PWMC) Fault Polarity on fault input 1
#define AT91C_PWMC_FPOL2      ((unsigned int) 0x1 <<  2) // (PWMC) Fault Polarity on fault input 2
#define AT91C_PWMC_FPOL3      ((unsigned int) 0x1 <<  3) // (PWMC) Fault Polarity on fault input 3
#define AT91C_PWMC_FPOL4      ((unsigned int) 0x1 <<  4) // (PWMC) Fault Polarity on fault input 4
#define AT91C_PWMC_FPOL5      ((unsigned int) 0x1 <<  5) // (PWMC) Fault Polarity on fault input 5
#define AT91C_PWMC_FPOL6      ((unsigned int) 0x1 <<  6) // (PWMC) Fault Polarity on fault input 6
#define AT91C_PWMC_FPOL7      ((unsigned int) 0x1 <<  7) // (PWMC) Fault Polarity on fault input 7
#define AT91C_PWMC_FMOD0      ((unsigned int) 0x1 <<  8) // (PWMC) Fault Activation Mode on fault input 0
#define AT91C_PWMC_FMOD1      ((unsigned int) 0x1 <<  9) // (PWMC) Fault Activation Mode on fault input 1
#define AT91C_PWMC_FMOD2      ((unsigned int) 0x1 << 10) // (PWMC) Fault Activation Mode on fault input 2
#define AT91C_PWMC_FMOD3      ((unsigned int) 0x1 << 11) // (PWMC) Fault Activation Mode on fault input 3
#define AT91C_PWMC_FMOD4      ((unsigned int) 0x1 << 12) // (PWMC) Fault Activation Mode on fault input 4
#define AT91C_PWMC_FMOD5      ((unsigned int) 0x1 << 13) // (PWMC) Fault Activation Mode on fault input 5
#define AT91C_PWMC_FMOD6      ((unsigned int) 0x1 << 14) // (PWMC) Fault Activation Mode on fault input 6
#define AT91C_PWMC_FMOD7      ((unsigned int) 0x1 << 15) // (PWMC) Fault Activation Mode on fault input 7
#define AT91C_PWMC_FFIL00     ((unsigned int) 0x1 << 16) // (PWMC) Fault Filtering on fault input 0
#define AT91C_PWMC_FFIL01     ((unsigned int) 0x1 << 17) // (PWMC) Fault Filtering on fault input 1
#define AT91C_PWMC_FFIL02     ((unsigned int) 0x1 << 18) // (PWMC) Fault Filtering on fault input 2
#define AT91C_PWMC_FFIL03     ((unsigned int) 0x1 << 19) // (PWMC) Fault Filtering on fault input 3
#define AT91C_PWMC_FFIL04     ((unsigned int) 0x1 << 20) // (PWMC) Fault Filtering on fault input 4
#define AT91C_PWMC_FFIL05     ((unsigned int) 0x1 << 21) // (PWMC) Fault Filtering on fault input 5
#define AT91C_PWMC_FFIL06     ((unsigned int) 0x1 << 22) // (PWMC) Fault Filtering on fault input 6
#define AT91C_PWMC_FFIL07     ((unsigned int) 0x1 << 23) // (PWMC) Fault Filtering on fault input 7
// -------- PWMC_FSR : (PWMC Offset: 0x60) Fault Input x Value -------- 
#define AT91C_PWMC_FIV0       ((unsigned int) 0x1 <<  0) // (PWMC) Fault Input 0 Value
#define AT91C_PWMC_FIV1       ((unsigned int) 0x1 <<  1) // (PWMC) Fault Input 1 Value
#define AT91C_PWMC_FIV2       ((unsigned int) 0x1 <<  2) // (PWMC) Fault Input 2 Value
#define AT91C_PWMC_FIV3       ((unsigned int) 0x1 <<  3) // (PWMC) Fault Input 3 Value
#define AT91C_PWMC_FIV4       ((unsigned int) 0x1 <<  4) // (PWMC) Fault Input 4 Value
#define AT91C_PWMC_FIV5       ((unsigned int) 0x1 <<  5) // (PWMC) Fault Input 5 Value
#define AT91C_PWMC_FIV6       ((unsigned int) 0x1 <<  6) // (PWMC) Fault Input 6 Value
#define AT91C_PWMC_FIV7       ((unsigned int) 0x1 <<  7) // (PWMC) Fault Input 7 Value
#define AT91C_PWMC_FS0        ((unsigned int) 0x1 <<  8) // (PWMC) Fault 0 Status
#define AT91C_PWMC_FS1        ((unsigned int) 0x1 <<  9) // (PWMC) Fault 1 Status
#define AT91C_PWMC_FS2        ((unsigned int) 0x1 << 10) // (PWMC) Fault 2 Status
#define AT91C_PWMC_FS3        ((unsigned int) 0x1 << 11) // (PWMC) Fault 3 Status
#define AT91C_PWMC_FS4        ((unsigned int) 0x1 << 12) // (PWMC) Fault 4 Status
#define AT91C_PWMC_FS5        ((unsigned int) 0x1 << 13) // (PWMC) Fault 5 Status
#define AT91C_PWMC_FS6        ((unsigned int) 0x1 << 14) // (PWMC) Fault 6 Status
#define AT91C_PWMC_FS7        ((unsigned int) 0x1 << 15) // (PWMC) Fault 7 Status
// -------- PWMC_FCR : (PWMC Offset: 0x64) Fault y Clear -------- 
#define AT91C_PWMC_FCLR0      ((unsigned int) 0x1 <<  0) // (PWMC) Fault 0 Clear
#define AT91C_PWMC_FCLR1      ((unsigned int) 0x1 <<  1) // (PWMC) Fault 1 Clear
#define AT91C_PWMC_FCLR2      ((unsigned int) 0x1 <<  2) // (PWMC) Fault 2 Clear
#define AT91C_PWMC_FCLR3      ((unsigned int) 0x1 <<  3) // (PWMC) Fault 3 Clear
#define AT91C_PWMC_FCLR4      ((unsigned int) 0x1 <<  4) // (PWMC) Fault 4 Clear
#define AT91C_PWMC_FCLR5      ((unsigned int) 0x1 <<  5) // (PWMC) Fault 5 Clear
#define AT91C_PWMC_FCLR6      ((unsigned int) 0x1 <<  6) // (PWMC) Fault 6 Clear
#define AT91C_PWMC_FCLR7      ((unsigned int) 0x1 <<  7) // (PWMC) Fault 7 Clear
// -------- PWMC_FPV : (PWMC Offset: 0x68) PWM Fault Protection Value -------- 
#define AT91C_PWMC_FPVH0      ((unsigned int) 0x1 <<  0) // (PWMC) Fault Protection Value for PWMH output on channel 0
#define AT91C_PWMC_FPVH1      ((unsigned int) 0x1 <<  1) // (PWMC) Fault Protection Value for PWMH output on channel 1
#define AT91C_PWMC_FPVH2      ((unsigned int) 0x1 <<  2) // (PWMC) Fault Protection Value for PWMH output on channel 2
#define AT91C_PWMC_FPVH3      ((unsigned int) 0x1 <<  3) // (PWMC) Fault Protection Value for PWMH output on channel 3
#define AT91C_PWMC_FPVH4      ((unsigned int) 0x1 <<  4) // (PWMC) Fault Protection Value for PWMH output on channel 4
#define AT91C_PWMC_FPVH5      ((unsigned int) 0x1 <<  5) // (PWMC) Fault Protection Value for PWMH output on channel 5
#define AT91C_PWMC_FPVH6      ((unsigned int) 0x1 <<  6) // (PWMC) Fault Protection Value for PWMH output on channel 6
#define AT91C_PWMC_FPVH7      ((unsigned int) 0x1 <<  7) // (PWMC) Fault Protection Value for PWMH output on channel 7
#define AT91C_PWMC_FPVL0      ((unsigned int) 0x1 << 16) // (PWMC) Fault Protection Value for PWML output on channel 0
#define AT91C_PWMC_FPVL1      ((unsigned int) 0x1 << 17) // (PWMC) Fault Protection Value for PWML output on channel 1
#define AT91C_PWMC_FPVL2      ((unsigned int) 0x1 << 18) // (PWMC) Fault Protection Value for PWML output on channel 2
#define AT91C_PWMC_FPVL3      ((unsigned int) 0x1 << 19) // (PWMC) Fault Protection Value for PWML output on channel 3
#define AT91C_PWMC_FPVL4      ((unsigned int) 0x1 << 20) // (PWMC) Fault Protection Value for PWML output on channel 4
#define AT91C_PWMC_FPVL5      ((unsigned int) 0x1 << 21) // (PWMC) Fault Protection Value for PWML output on channel 5
#define AT91C_PWMC_FPVL6      ((unsigned int) 0x1 << 22) // (PWMC) Fault Protection Value for PWML output on channel 6
#define AT91C_PWMC_FPVL7      ((unsigned int) 0x1 << 23) // (PWMC) Fault Protection Value for PWML output on channel 7
// -------- PWMC_FPER1 : (PWMC Offset: 0x6c) PWM Fault Protection Enable Register 1 -------- 
#define AT91C_PWMC_FPE0       ((unsigned int) 0xFF <<  0) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 0
#define AT91C_PWMC_FPE1       ((unsigned int) 0xFF <<  8) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 1
#define AT91C_PWMC_FPE2       ((unsigned int) 0xFF << 16) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 2
#define AT91C_PWMC_FPE3       ((unsigned int) 0xFF << 24) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 3
// -------- PWMC_FPER2 : (PWMC Offset: 0x70) PWM Fault Protection Enable Register 2 -------- 
#define AT91C_PWMC_FPE4       ((unsigned int) 0xFF <<  0) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 4
#define AT91C_PWMC_FPE5       ((unsigned int) 0xFF <<  8) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 5
#define AT91C_PWMC_FPE6       ((unsigned int) 0xFF << 16) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 6
#define AT91C_PWMC_FPE7       ((unsigned int) 0xFF << 24) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 7
// -------- PWMC_FPER3 : (PWMC Offset: 0x74) PWM Fault Protection Enable Register 3 -------- 
#define AT91C_PWMC_FPE8       ((unsigned int) 0xFF <<  0) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 8
#define AT91C_PWMC_FPE9       ((unsigned int) 0xFF <<  8) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 9
#define AT91C_PWMC_FPE10      ((unsigned int) 0xFF << 16) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 10
#define AT91C_PWMC_FPE11      ((unsigned int) 0xFF << 24) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 11
// -------- PWMC_FPER4 : (PWMC Offset: 0x78) PWM Fault Protection Enable Register 4 -------- 
#define AT91C_PWMC_FPE12      ((unsigned int) 0xFF <<  0) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 12
#define AT91C_PWMC_FPE13      ((unsigned int) 0xFF <<  8) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 13
#define AT91C_PWMC_FPE14      ((unsigned int) 0xFF << 16) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 14
#define AT91C_PWMC_FPE15      ((unsigned int) 0xFF << 24) // (PWMC) Fault Protection Enable with Fault Input y for PWM channel 15
// -------- PWMC_EL0MR : (PWMC Offset: 0x7c) PWM Event Line 0 Mode Register -------- 
#define AT91C_PWMC_L0CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L0CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L0CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L0CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L0CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L0CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L0CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L0CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL1MR : (PWMC Offset: 0x80) PWM Event Line 1 Mode Register -------- 
#define AT91C_PWMC_L1CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L1CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L1CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L1CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L1CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L1CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L1CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L1CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL2MR : (PWMC Offset: 0x84) PWM Event line 2 Mode Register -------- 
#define AT91C_PWMC_L2CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L2CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L2CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L2CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L2CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L2CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L2CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L2CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL3MR : (PWMC Offset: 0x88) PWM Event line 3 Mode Register -------- 
#define AT91C_PWMC_L3CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L3CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L3CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L3CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L3CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L3CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L3CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L3CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL4MR : (PWMC Offset: 0x8c) PWM Event line 4 Mode Register -------- 
#define AT91C_PWMC_L4CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L4CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L4CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L4CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L4CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L4CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L4CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L4CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL5MR : (PWMC Offset: 0x90) PWM Event line 5 Mode Register -------- 
#define AT91C_PWMC_L5CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L5CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L5CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L5CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L5CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L5CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L5CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L5CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL6MR : (PWMC Offset: 0x94) PWM Event line 6 Mode Register -------- 
#define AT91C_PWMC_L6CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L6CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L6CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L6CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L6CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L6CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L6CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L6CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_EL7MR : (PWMC Offset: 0x98) PWM Event line 7 Mode Register -------- 
#define AT91C_PWMC_L7CSEL0    ((unsigned int) 0x1 <<  0) // (PWMC) Comparison 0 Selection
#define AT91C_PWMC_L7CSEL1    ((unsigned int) 0x1 <<  1) // (PWMC) Comparison 1 Selection
#define AT91C_PWMC_L7CSEL2    ((unsigned int) 0x1 <<  2) // (PWMC) Comparison 2 Selection
#define AT91C_PWMC_L7CSEL3    ((unsigned int) 0x1 <<  3) // (PWMC) Comparison 3 Selection
#define AT91C_PWMC_L7CSEL4    ((unsigned int) 0x1 <<  4) // (PWMC) Comparison 4 Selection
#define AT91C_PWMC_L7CSEL5    ((unsigned int) 0x1 <<  5) // (PWMC) Comparison 5 Selection
#define AT91C_PWMC_L7CSEL6    ((unsigned int) 0x1 <<  6) // (PWMC) Comparison 6 Selection
#define AT91C_PWMC_L7CSEL7    ((unsigned int) 0x1 <<  7) // (PWMC) Comparison 7 Selection
// -------- PWMC_WPCR : (PWMC Offset: 0xe4) PWM Write Protection Control Register -------- 
#define AT91C_PWMC_WPCMD      ((unsigned int) 0x3 <<  0) // (PWMC) Write Protection Command
#define AT91C_PWMC_WPRG0      ((unsigned int) 0x1 <<  2) // (PWMC) Write Protect Register Group 0
#define AT91C_PWMC_WPRG1      ((unsigned int) 0x1 <<  3) // (PWMC) Write Protect Register Group 1
#define AT91C_PWMC_WPRG2      ((unsigned int) 0x1 <<  4) // (PWMC) Write Protect Register Group 2
#define AT91C_PWMC_WPRG3      ((unsigned int) 0x1 <<  5) // (PWMC) Write Protect Register Group 3
#define AT91C_PWMC_WPRG4      ((unsigned int) 0x1 <<  6) // (PWMC) Write Protect Register Group 4
#define AT91C_PWMC_WPRG5      ((unsigned int) 0x1 <<  7) // (PWMC) Write Protect Register Group 5
#define AT91C_PWMC_WPKEY      ((unsigned int) 0xFFFFFF <<  8) // (PWMC) Protection Password
// -------- PWMC_WPVS : (PWMC Offset: 0xe8) Write Protection Status Register -------- 
#define AT91C_PWMC_WPSWS0     ((unsigned int) 0x1 <<  0) // (PWMC) Write Protect SW Group 0 Status 
#define AT91C_PWMC_WPSWS1     ((unsigned int) 0x1 <<  1) // (PWMC) Write Protect SW Group 1 Status 
#define AT91C_PWMC_WPSWS2     ((unsigned int) 0x1 <<  2) // (PWMC) Write Protect SW Group 2 Status 
#define AT91C_PWMC_WPSWS3     ((unsigned int) 0x1 <<  3) // (PWMC) Write Protect SW Group 3 Status 
#define AT91C_PWMC_WPSWS4     ((unsigned int) 0x1 <<  4) // (PWMC) Write Protect SW Group 4 Status 
#define AT91C_PWMC_WPSWS5     ((unsigned int) 0x1 <<  5) // (PWMC) Write Protect SW Group 5 Status 
#define AT91C_PWMC_WPVS       ((unsigned int) 0x1 <<  7) // (PWMC) Write Protection Enable
#define AT91C_PWMC_WPHWS0     ((unsigned int) 0x1 <<  8) // (PWMC) Write Protect HW Group 0 Status 
#define AT91C_PWMC_WPHWS1     ((unsigned int) 0x1 <<  9) // (PWMC) Write Protect HW Group 1 Status 
#define AT91C_PWMC_WPHWS2     ((unsigned int) 0x1 << 10) // (PWMC) Write Protect HW Group 2 Status 
#define AT91C_PWMC_WPHWS3     ((unsigned int) 0x1 << 11) // (PWMC) Write Protect HW Group 3 Status 
#define AT91C_PWMC_WPHWS4     ((unsigned int) 0x1 << 12) // (PWMC) Write Protect HW Group 4 Status 
#define AT91C_PWMC_WPHWS5     ((unsigned int) 0x1 << 13) // (PWMC) Write Protect HW Group 5 Status 
#define AT91C_PWMC_WPVSRC     ((unsigned int) 0xFFFF << 16) // (PWMC) Write Protection Violation Source
// -------- PWMC_CMP0V : (PWMC Offset: 0x130) PWM Comparison Value 0 Register -------- 
#define AT91C_PWMC_CV         ((unsigned int) 0xFFFFFF <<  0) // (PWMC) PWM Comparison Value 0.
#define AT91C_PWMC_CVM        ((unsigned int) 0x1 << 24) // (PWMC) Comparison Value 0 Mode.
// -------- PWMC_CMP0VUPD : (PWMC Offset: 0x134) PWM Comparison Value 0 Update Register -------- 
#define AT91C_PWMC_CVUPD      ((unsigned int) 0xFFFFFF <<  0) // (PWMC) PWM Comparison Value Update.
#define AT91C_PWMC_CVMUPD     ((unsigned int) 0x1 << 24) // (PWMC) Comparison Value Update Mode.
// -------- PWMC_CMP0M : (PWMC Offset: 0x138) PWM Comparison 0 Mode Register -------- 
#define AT91C_PWMC_CEN        ((unsigned int) 0x1 <<  0) // (PWMC) Comparison Enable.
#define AT91C_PWMC_CTR        ((unsigned int) 0xF <<  4) // (PWMC) PWM Comparison Trigger.
#define AT91C_PWMC_CPR        ((unsigned int) 0xF <<  8) // (PWMC) PWM Comparison Period.
#define AT91C_PWMC_CPRCNT     ((unsigned int) 0xF << 12) // (PWMC) PWM Comparison Period Counter.
#define AT91C_PWMC_CUPR       ((unsigned int) 0xF << 16) // (PWMC) PWM Comparison Update Period.
#define AT91C_PWMC_CUPRCNT    ((unsigned int) 0xF << 20) // (PWMC) PWM Comparison Update Period Counter.
// -------- PWMC_CMP0MUPD : (PWMC Offset: 0x13c) PWM Comparison 0 Mode Update Register -------- 
#define AT91C_PWMC_CENUPD     ((unsigned int) 0x1 <<  0) // (PWMC) Comparison Enable Update.
#define AT91C_PWMC_CTRUPD     ((unsigned int) 0xF <<  4) // (PWMC) PWM Comparison Trigger Update.
#define AT91C_PWMC_CPRUPD     ((unsigned int) 0xF <<  8) // (PWMC) PWM Comparison Period Update.
#define AT91C_PWMC_CUPRUPD    ((unsigned int) 0xF << 16) // (PWMC) PWM Comparison Update Period Update.
// -------- PWMC_CMP1V : (PWMC Offset: 0x140) PWM Comparison Value 1 Register -------- 
// -------- PWMC_CMP1VUPD : (PWMC Offset: 0x144) PWM Comparison Value 1 Update Register -------- 
// -------- PWMC_CMP1M : (PWMC Offset: 0x148) PWM Comparison 1 Mode Register -------- 
// -------- PWMC_CMP1MUPD : (PWMC Offset: 0x14c) PWM Comparison 1 Mode Update Register -------- 
// -------- PWMC_CMP2V : (PWMC Offset: 0x150) PWM Comparison Value 2 Register -------- 
// -------- PWMC_CMP2VUPD : (PWMC Offset: 0x154) PWM Comparison Value 2 Update Register -------- 
// -------- PWMC_CMP2M : (PWMC Offset: 0x158) PWM Comparison 2 Mode Register -------- 
// -------- PWMC_CMP2MUPD : (PWMC Offset: 0x15c) PWM Comparison 2 Mode Update Register -------- 
// -------- PWMC_CMP3V : (PWMC Offset: 0x160) PWM Comparison Value 3 Register -------- 
// -------- PWMC_CMP3VUPD : (PWMC Offset: 0x164) PWM Comparison Value 3 Update Register -------- 
// -------- PWMC_CMP3M : (PWMC Offset: 0x168) PWM Comparison 3 Mode Register -------- 
// -------- PWMC_CMP3MUPD : (PWMC Offset: 0x16c) PWM Comparison 3 Mode Update Register -------- 
// -------- PWMC_CMP4V : (PWMC Offset: 0x170) PWM Comparison Value 4 Register -------- 
// -------- PWMC_CMP4VUPD : (PWMC Offset: 0x174) PWM Comparison Value 4 Update Register -------- 
// -------- PWMC_CMP4M : (PWMC Offset: 0x178) PWM Comparison 4 Mode Register -------- 
// -------- PWMC_CMP4MUPD : (PWMC Offset: 0x17c) PWM Comparison 4 Mode Update Register -------- 
// -------- PWMC_CMP5V : (PWMC Offset: 0x180) PWM Comparison Value 5 Register -------- 
// -------- PWMC_CMP5VUPD : (PWMC Offset: 0x184) PWM Comparison Value 5 Update Register -------- 
// -------- PWMC_CMP5M : (PWMC Offset: 0x188) PWM Comparison 5 Mode Register -------- 
// -------- PWMC_CMP5MUPD : (PWMC Offset: 0x18c) PWM Comparison 5 Mode Update Register -------- 
// -------- PWMC_CMP6V : (PWMC Offset: 0x190) PWM Comparison Value 6 Register -------- 
// -------- PWMC_CMP6VUPD : (PWMC Offset: 0x194) PWM Comparison Value 6 Update Register -------- 
// -------- PWMC_CMP6M : (PWMC Offset: 0x198) PWM Comparison 6 Mode Register -------- 
// -------- PWMC_CMP6MUPD : (PWMC Offset: 0x19c) PWM Comparison 6 Mode Update Register -------- 
// -------- PWMC_CMP7V : (PWMC Offset: 0x1a0) PWM Comparison Value 7 Register -------- 
// -------- PWMC_CMP7VUPD : (PWMC Offset: 0x1a4) PWM Comparison Value 7 Update Register -------- 
// -------- PWMC_CMP7M : (PWMC Offset: 0x1a8) PWM Comparison 7 Mode Register -------- 
// -------- PWMC_CMP7MUPD : (PWMC Offset: 0x1ac) PWM Comparison 7 Mode Update Register -------- 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Serial Parallel Interface
// *****************************************************************************
typedef struct _AT91S_SPI {
	AT91_REG	 SPI_CR; 	// Control Register
	AT91_REG	 SPI_MR; 	// Mode Register
	AT91_REG	 SPI_RDR; 	// Receive Data Register
	AT91_REG	 SPI_TDR; 	// Transmit Data Register
	AT91_REG	 SPI_SR; 	// Status Register
	AT91_REG	 SPI_IER; 	// Interrupt Enable Register
	AT91_REG	 SPI_IDR; 	// Interrupt Disable Register
	AT91_REG	 SPI_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved0[4]; 	// 
	AT91_REG	 SPI_CSR[4]; 	// Chip Select Register
	AT91_REG	 Reserved1[43]; 	// 
	AT91_REG	 SPI_ADDRSIZE; 	// SPI ADDRSIZE REGISTER 
	AT91_REG	 SPI_IPNAME1; 	// SPI IPNAME1 REGISTER 
	AT91_REG	 SPI_IPNAME2; 	// SPI IPNAME2 REGISTER 
	AT91_REG	 SPI_FEATURES; 	// SPI FEATURES REGISTER 
	AT91_REG	 SPI_VER; 	// Version Register
} AT91S_SPI, *AT91PS_SPI;

// -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- 
#define AT91C_SPI_SPIEN       ((unsigned int) 0x1 <<  0) // (SPI) SPI Enable
#define AT91C_SPI_SPIDIS      ((unsigned int) 0x1 <<  1) // (SPI) SPI Disable
#define AT91C_SPI_SWRST       ((unsigned int) 0x1 <<  7) // (SPI) SPI Software reset
#define AT91C_SPI_LASTXFER    ((unsigned int) 0x1 << 24) // (SPI) SPI Last Transfer
// -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- 
#define AT91C_SPI_MSTR        ((unsigned int) 0x1 <<  0) // (SPI) Master/Slave Mode
#define AT91C_SPI_PS          ((unsigned int) 0x1 <<  1) // (SPI) Peripheral Select
#define 	AT91C_SPI_PS_FIXED                ((unsigned int) 0x0 <<  1) // (SPI) Fixed Peripheral Select
#define 	AT91C_SPI_PS_VARIABLE             ((unsigned int) 0x1 <<  1) // (SPI) Variable Peripheral Select
#define AT91C_SPI_PCSDEC      ((unsigned int) 0x1 <<  2) // (SPI) Chip Select Decode
#define AT91C_SPI_FDIV        ((unsigned int) 0x1 <<  3) // (SPI) Clock Selection
#define AT91C_SPI_MODFDIS     ((unsigned int) 0x1 <<  4) // (SPI) Mode Fault Detection
#define AT91C_SPI_LLB         ((unsigned int) 0x1 <<  7) // (SPI) Clock Selection
#define AT91C_SPI_PCS         ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select
#define AT91C_SPI_DLYBCS      ((unsigned int) 0xFF << 24) // (SPI) Delay Between Chip Selects
// -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- 
#define AT91C_SPI_RD          ((unsigned int) 0xFFFF <<  0) // (SPI) Receive Data
#define AT91C_SPI_RPCS        ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select Status
// -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- 
#define AT91C_SPI_TD          ((unsigned int) 0xFFFF <<  0) // (SPI) Transmit Data
#define AT91C_SPI_TPCS        ((unsigned int) 0xF << 16) // (SPI) Peripheral Chip Select Status
// -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- 
#define AT91C_SPI_RDRF        ((unsigned int) 0x1 <<  0) // (SPI) Receive Data Register Full
#define AT91C_SPI_TDRE        ((unsigned int) 0x1 <<  1) // (SPI) Transmit Data Register Empty
#define AT91C_SPI_MODF        ((unsigned int) 0x1 <<  2) // (SPI) Mode Fault Error
#define AT91C_SPI_OVRES       ((unsigned int) 0x1 <<  3) // (SPI) Overrun Error Status
#define AT91C_SPI_ENDRX       ((unsigned int) 0x1 <<  4) // (SPI) End of Receiver Transfer
#define AT91C_SPI_ENDTX       ((unsigned int) 0x1 <<  5) // (SPI) End of Receiver Transfer
#define AT91C_SPI_RXBUFF      ((unsigned int) 0x1 <<  6) // (SPI) RXBUFF Interrupt
#define AT91C_SPI_TXBUFE      ((unsigned int) 0x1 <<  7) // (SPI) TXBUFE Interrupt
#define AT91C_SPI_NSSR        ((unsigned int) 0x1 <<  8) // (SPI) NSSR Interrupt
#define AT91C_SPI_TXEMPTY     ((unsigned int) 0x1 <<  9) // (SPI) TXEMPTY Interrupt
#define AT91C_SPI_SPIENS      ((unsigned int) 0x1 << 16) // (SPI) Enable Status
// -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- 
// -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- 
// -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- 
// -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- 
#define AT91C_SPI_CPOL        ((unsigned int) 0x1 <<  0) // (SPI) Clock Polarity
#define AT91C_SPI_NCPHA       ((unsigned int) 0x1 <<  1) // (SPI) Clock Phase
#define AT91C_SPI_CSNAAT      ((unsigned int) 0x1 <<  2) // (SPI) Chip Select Not Active After Transfer (Ignored if CSAAT = 1)
#define AT91C_SPI_CSAAT       ((unsigned int) 0x1 <<  3) // (SPI) Chip Select Active After Transfer
#define AT91C_SPI_BITS        ((unsigned int) 0xF <<  4) // (SPI) Bits Per Transfer
#define 	AT91C_SPI_BITS_8                    ((unsigned int) 0x0 <<  4) // (SPI) 8 Bits Per transfer
#define 	AT91C_SPI_BITS_9                    ((unsigned int) 0x1 <<  4) // (SPI) 9 Bits Per transfer
#define 	AT91C_SPI_BITS_10                   ((unsigned int) 0x2 <<  4) // (SPI) 10 Bits Per transfer
#define 	AT91C_SPI_BITS_11                   ((unsigned int) 0x3 <<  4) // (SPI) 11 Bits Per transfer
#define 	AT91C_SPI_BITS_12                   ((unsigned int) 0x4 <<  4) // (SPI) 12 Bits Per transfer
#define 	AT91C_SPI_BITS_13                   ((unsigned int) 0x5 <<  4) // (SPI) 13 Bits Per transfer
#define 	AT91C_SPI_BITS_14                   ((unsigned int) 0x6 <<  4) // (SPI) 14 Bits Per transfer
#define 	AT91C_SPI_BITS_15                   ((unsigned int) 0x7 <<  4) // (SPI) 15 Bits Per transfer
#define 	AT91C_SPI_BITS_16                   ((unsigned int) 0x8 <<  4) // (SPI) 16 Bits Per transfer
#define AT91C_SPI_SCBR        ((unsigned int) 0xFF <<  8) // (SPI) Serial Clock Baud Rate
#define AT91C_SPI_DLYBS       ((unsigned int) 0xFF << 16) // (SPI) Delay Before SPCK
#define AT91C_SPI_DLYBCT      ((unsigned int) 0xFF << 24) // (SPI) Delay Between Consecutive Transfers

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register
// *****************************************************************************
typedef struct _AT91S_UDPHS_EPTFIFO {
	AT91_REG	 UDPHS_READEPT0[16384]; 	// FIFO Endpoint Data Register 0
	AT91_REG	 UDPHS_READEPT1[16384]; 	// FIFO Endpoint Data Register 1
	AT91_REG	 UDPHS_READEPT2[16384]; 	// FIFO Endpoint Data Register 2
	AT91_REG	 UDPHS_READEPT3[16384]; 	// FIFO Endpoint Data Register 3
	AT91_REG	 UDPHS_READEPT4[16384]; 	// FIFO Endpoint Data Register 4
	AT91_REG	 UDPHS_READEPT5[16384]; 	// FIFO Endpoint Data Register 5
	AT91_REG	 UDPHS_READEPT6[16384]; 	// FIFO Endpoint Data Register 6
} AT91S_UDPHS_EPTFIFO, *AT91PS_UDPHS_EPTFIFO;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct
// *****************************************************************************
typedef struct _AT91S_UDPHS_EPT {
	AT91_REG	 UDPHS_EPTCFG; 	// UDPHS Endpoint Config Register
	AT91_REG	 UDPHS_EPTCTLENB; 	// UDPHS Endpoint Control Enable Register
	AT91_REG	 UDPHS_EPTCTLDIS; 	// UDPHS Endpoint Control Disable Register
	AT91_REG	 UDPHS_EPTCTL; 	// UDPHS Endpoint Control Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 UDPHS_EPTSETSTA; 	// UDPHS Endpoint Set Status Register
	AT91_REG	 UDPHS_EPTCLRSTA; 	// UDPHS Endpoint Clear Status Register
	AT91_REG	 UDPHS_EPTSTA; 	// UDPHS Endpoint Status Register
} AT91S_UDPHS_EPT, *AT91PS_UDPHS_EPT;

// -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- 
#define AT91C_UDPHS_EPT_SIZE  ((unsigned int) 0x7 <<  0) // (UDPHS_EPT) Endpoint Size
#define 	AT91C_UDPHS_EPT_SIZE_8                    ((unsigned int) 0x0) // (UDPHS_EPT)    8 bytes
#define 	AT91C_UDPHS_EPT_SIZE_16                   ((unsigned int) 0x1) // (UDPHS_EPT)   16 bytes
#define 	AT91C_UDPHS_EPT_SIZE_32                   ((unsigned int) 0x2) // (UDPHS_EPT)   32 bytes
#define 	AT91C_UDPHS_EPT_SIZE_64                   ((unsigned int) 0x3) // (UDPHS_EPT)   64 bytes
#define 	AT91C_UDPHS_EPT_SIZE_128                  ((unsigned int) 0x4) // (UDPHS_EPT)  128 bytes
#define 	AT91C_UDPHS_EPT_SIZE_256                  ((unsigned int) 0x5) // (UDPHS_EPT)  256 bytes (if possible)
#define 	AT91C_UDPHS_EPT_SIZE_512                  ((unsigned int) 0x6) // (UDPHS_EPT)  512 bytes (if possible)
#define 	AT91C_UDPHS_EPT_SIZE_1024                 ((unsigned int) 0x7) // (UDPHS_EPT) 1024 bytes (if possible)
#define AT91C_UDPHS_EPT_DIR   ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Endpoint Direction 0:OUT, 1:IN
#define 	AT91C_UDPHS_EPT_DIR_OUT                  ((unsigned int) 0x0 <<  3) // (UDPHS_EPT) Direction OUT
#define 	AT91C_UDPHS_EPT_DIR_IN                   ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Direction IN
#define AT91C_UDPHS_EPT_TYPE  ((unsigned int) 0x3 <<  4) // (UDPHS_EPT) Endpoint Type
#define 	AT91C_UDPHS_EPT_TYPE_CTL_EPT              ((unsigned int) 0x0 <<  4) // (UDPHS_EPT) Control endpoint
#define 	AT91C_UDPHS_EPT_TYPE_ISO_EPT              ((unsigned int) 0x1 <<  4) // (UDPHS_EPT) Isochronous endpoint
#define 	AT91C_UDPHS_EPT_TYPE_BUL_EPT              ((unsigned int) 0x2 <<  4) // (UDPHS_EPT) Bulk endpoint
#define 	AT91C_UDPHS_EPT_TYPE_INT_EPT              ((unsigned int) 0x3 <<  4) // (UDPHS_EPT) Interrupt endpoint
#define AT91C_UDPHS_BK_NUMBER ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Number of Banks
#define 	AT91C_UDPHS_BK_NUMBER_0                    ((unsigned int) 0x0 <<  6) // (UDPHS_EPT) Zero Bank, the EndPoint is not mapped in memory
#define 	AT91C_UDPHS_BK_NUMBER_1                    ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) One Bank (Bank0)
#define 	AT91C_UDPHS_BK_NUMBER_2                    ((unsigned int) 0x2 <<  6) // (UDPHS_EPT) Double bank (Ping-Pong : Bank0 / Bank1)
#define 	AT91C_UDPHS_BK_NUMBER_3                    ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2) (if possible)
#define AT91C_UDPHS_NB_TRANS  ((unsigned int) 0x3 <<  8) // (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)
#define AT91C_UDPHS_EPT_MAPD  ((unsigned int) 0x1 << 31) // (UDPHS_EPT) Endpoint Mapped (read only
// -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- 
#define AT91C_UDPHS_EPT_ENABL ((unsigned int) 0x1 <<  0) // (UDPHS_EPT) Endpoint Enable
#define AT91C_UDPHS_AUTO_VALID ((unsigned int) 0x1 <<  1) // (UDPHS_EPT) Packet Auto-Valid Enable/Disable
#define AT91C_UDPHS_INTDIS_DMA ((unsigned int) 0x1 <<  3) // (UDPHS_EPT) Endpoint Interrupts DMA Request Enable/Disable
#define AT91C_UDPHS_NYET_DIS  ((unsigned int) 0x1 <<  4) // (UDPHS_EPT) NYET Enable/Disable
#define AT91C_UDPHS_DATAX_RX  ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) DATAx Interrupt Enable/Disable
#define AT91C_UDPHS_MDATA_RX  ((unsigned int) 0x1 <<  7) // (UDPHS_EPT) MDATA Interrupt Enabled/Disable
#define AT91C_UDPHS_ERR_OVFLW ((unsigned int) 0x1 <<  8) // (UDPHS_EPT) OverFlow Error Interrupt Enable/Disable/Status
#define AT91C_UDPHS_RX_BK_RDY ((unsigned int) 0x1 <<  9) // (UDPHS_EPT) Received OUT Data
#define AT91C_UDPHS_TX_COMPLT ((unsigned int) 0x1 << 10) // (UDPHS_EPT) Transmitted IN Data Complete Interrupt Enable/Disable or Transmitted IN Data Complete (clear)
#define AT91C_UDPHS_ERR_TRANS ((unsigned int) 0x1 << 11) // (UDPHS_EPT) Transaction Error Interrupt Enable/Disable
#define AT91C_UDPHS_TX_PK_RDY ((unsigned int) 0x1 << 11) // (UDPHS_EPT) TX Packet Ready Interrupt Enable/Disable
#define AT91C_UDPHS_RX_SETUP  ((unsigned int) 0x1 << 12) // (UDPHS_EPT) Received SETUP Interrupt Enable/Disable
#define AT91C_UDPHS_ERR_FL_ISO ((unsigned int) 0x1 << 12) // (UDPHS_EPT) Error Flow Clear/Interrupt Enable/Disable
#define AT91C_UDPHS_STALL_SNT ((unsigned int) 0x1 << 13) // (UDPHS_EPT) Stall Sent Clear
#define AT91C_UDPHS_ERR_CRISO ((unsigned int) 0x1 << 13) // (UDPHS_EPT) CRC error / Error NB Trans / Interrupt Enable/Disable
#define AT91C_UDPHS_NAK_IN    ((unsigned int) 0x1 << 14) // (UDPHS_EPT) NAKIN ERROR FLUSH / Clear / Interrupt Enable/Disable
#define AT91C_UDPHS_NAK_OUT   ((unsigned int) 0x1 << 15) // (UDPHS_EPT) NAKOUT / Clear / Interrupt Enable/Disable
#define AT91C_UDPHS_BUSY_BANK ((unsigned int) 0x1 << 18) // (UDPHS_EPT) Busy Bank Interrupt Enable/Disable
#define AT91C_UDPHS_SHRT_PCKT ((unsigned int) 0x1 << 31) // (UDPHS_EPT) Short Packet / Interrupt Enable/Disable
// -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- 
#define AT91C_UDPHS_EPT_DISABL ((unsigned int) 0x1 <<  0) // (UDPHS_EPT) Endpoint Disable
// -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- 
// -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- 
#define AT91C_UDPHS_FRCESTALL ((unsigned int) 0x1 <<  5) // (UDPHS_EPT) Stall Handshake Request Set/Clear/Status
#define AT91C_UDPHS_KILL_BANK ((unsigned int) 0x1 <<  9) // (UDPHS_EPT) KILL Bank
// -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- 
#define AT91C_UDPHS_TOGGLESQ  ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) Data Toggle Clear
// -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- 
#define AT91C_UDPHS_TOGGLESQ_STA ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) Toggle Sequencing
#define 	AT91C_UDPHS_TOGGLESQ_STA_00                   ((unsigned int) 0x0 <<  6) // (UDPHS_EPT) Data0
#define 	AT91C_UDPHS_TOGGLESQ_STA_01                   ((unsigned int) 0x1 <<  6) // (UDPHS_EPT) Data1
#define 	AT91C_UDPHS_TOGGLESQ_STA_10                   ((unsigned int) 0x2 <<  6) // (UDPHS_EPT) Data2 (only for High-Bandwidth Isochronous EndPoint)
#define 	AT91C_UDPHS_TOGGLESQ_STA_11                   ((unsigned int) 0x3 <<  6) // (UDPHS_EPT) MData (only for High-Bandwidth Isochronous EndPoint)
#define AT91C_UDPHS_CONTROL_DIR ((unsigned int) 0x3 << 16) // (UDPHS_EPT) 
#define 	AT91C_UDPHS_CONTROL_DIR_00                   ((unsigned int) 0x0 << 16) // (UDPHS_EPT) Bank 0
#define 	AT91C_UDPHS_CONTROL_DIR_01                   ((unsigned int) 0x1 << 16) // (UDPHS_EPT) Bank 1
#define 	AT91C_UDPHS_CONTROL_DIR_10                   ((unsigned int) 0x2 << 16) // (UDPHS_EPT) Bank 2
#define 	AT91C_UDPHS_CONTROL_DIR_11                   ((unsigned int) 0x3 << 16) // (UDPHS_EPT) Invalid
#define AT91C_UDPHS_CURRENT_BANK ((unsigned int) 0x3 << 16) // (UDPHS_EPT) 
#define 	AT91C_UDPHS_CURRENT_BANK_00                   ((unsigned int) 0x0 << 16) // (UDPHS_EPT) Bank 0
#define 	AT91C_UDPHS_CURRENT_BANK_01                   ((unsigned int) 0x1 << 16) // (UDPHS_EPT) Bank 1
#define 	AT91C_UDPHS_CURRENT_BANK_10                   ((unsigned int) 0x2 << 16) // (UDPHS_EPT) Bank 2
#define 	AT91C_UDPHS_CURRENT_BANK_11                   ((unsigned int) 0x3 << 16) // (UDPHS_EPT) Invalid
#define AT91C_UDPHS_BUSY_BANK_STA ((unsigned int) 0x3 << 18) // (UDPHS_EPT) Busy Bank Number
#define 	AT91C_UDPHS_BUSY_BANK_STA_00                   ((unsigned int) 0x0 << 18) // (UDPHS_EPT) All banks are free
#define 	AT91C_UDPHS_BUSY_BANK_STA_01                   ((unsigned int) 0x1 << 18) // (UDPHS_EPT) 1 busy bank
#define 	AT91C_UDPHS_BUSY_BANK_STA_10                   ((unsigned int) 0x2 << 18) // (UDPHS_EPT) 2 busy banks
#define 	AT91C_UDPHS_BUSY_BANK_STA_11                   ((unsigned int) 0x3 << 18) // (UDPHS_EPT) 3 busy banks (if possible)
#define AT91C_UDPHS_BYTE_COUNT ((unsigned int) 0x7FF << 20) // (UDPHS_EPT) UDPHS Byte Count

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS DMA struct
// *****************************************************************************
typedef struct _AT91S_UDPHS_DMA {
	AT91_REG	 UDPHS_DMANXTDSC; 	// UDPHS DMA Channel Next Descriptor Address
	AT91_REG	 UDPHS_DMAADDRESS; 	// UDPHS DMA Channel Address Register
	AT91_REG	 UDPHS_DMACONTROL; 	// UDPHS DMA Channel Control Register
	AT91_REG	 UDPHS_DMASTATUS; 	// UDPHS DMA Channel Status Register
} AT91S_UDPHS_DMA, *AT91PS_UDPHS_DMA;

// -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- 
#define AT91C_UDPHS_NXT_DSC_ADD ((unsigned int) 0xFFFFFFF <<  4) // (UDPHS_DMA) next Channel Descriptor
// -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- 
#define AT91C_UDPHS_BUFF_ADD  ((unsigned int) 0x0 <<  0) // (UDPHS_DMA) starting address of a DMA Channel transfer
// -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- 
#define AT91C_UDPHS_CHANN_ENB ((unsigned int) 0x1 <<  0) // (UDPHS_DMA) Channel Enabled
#define AT91C_UDPHS_LDNXT_DSC ((unsigned int) 0x1 <<  1) // (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable
#define AT91C_UDPHS_END_TR_EN ((unsigned int) 0x1 <<  2) // (UDPHS_DMA) Buffer Close Input Enable
#define AT91C_UDPHS_END_B_EN  ((unsigned int) 0x1 <<  3) // (UDPHS_DMA) End of DMA Buffer Packet Validation
#define AT91C_UDPHS_END_TR_IT ((unsigned int) 0x1 <<  4) // (UDPHS_DMA) End Of Transfer Interrupt Enable
#define AT91C_UDPHS_END_BUFFIT ((unsigned int) 0x1 <<  5) // (UDPHS_DMA) End Of Channel Buffer Interrupt Enable
#define AT91C_UDPHS_DESC_LD_IT ((unsigned int) 0x1 <<  6) // (UDPHS_DMA) Descriptor Loaded Interrupt Enable
#define AT91C_UDPHS_BURST_LCK ((unsigned int) 0x1 <<  7) // (UDPHS_DMA) Burst Lock Enable
#define AT91C_UDPHS_BUFF_LENGTH ((unsigned int) 0xFFFF << 16) // (UDPHS_DMA) Buffer Byte Length (write only)
// -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- 
#define AT91C_UDPHS_CHANN_ACT ((unsigned int) 0x1 <<  1) // (UDPHS_DMA) 
#define AT91C_UDPHS_END_TR_ST ((unsigned int) 0x1 <<  4) // (UDPHS_DMA) 
#define AT91C_UDPHS_END_BF_ST ((unsigned int) 0x1 <<  5) // (UDPHS_DMA) 
#define AT91C_UDPHS_DESC_LDST ((unsigned int) 0x1 <<  6) // (UDPHS_DMA) 
#define AT91C_UDPHS_BUFF_COUNT ((unsigned int) 0xFFFF << 16) // (UDPHS_DMA) 

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface
// *****************************************************************************
typedef struct _AT91S_UDPHS {
	AT91_REG	 UDPHS_CTRL; 	// UDPHS Control Register
	AT91_REG	 UDPHS_FNUM; 	// UDPHS Frame Number Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 UDPHS_IEN; 	// UDPHS Interrupt Enable Register
	AT91_REG	 UDPHS_INTSTA; 	// UDPHS Interrupt Status Register
	AT91_REG	 UDPHS_CLRINT; 	// UDPHS Clear Interrupt Register
	AT91_REG	 UDPHS_EPTRST; 	// UDPHS Endpoints Reset Register
	AT91_REG	 Reserved1[44]; 	// 
	AT91_REG	 UDPHS_TSTSOFCNT; 	// UDPHS Test SOF Counter Register
	AT91_REG	 UDPHS_TSTCNTA; 	// UDPHS Test A Counter Register
	AT91_REG	 UDPHS_TSTCNTB; 	// UDPHS Test B Counter Register
	AT91_REG	 UDPHS_TSTMODREG; 	// UDPHS Test Mode Register
	AT91_REG	 UDPHS_TST; 	// UDPHS Test Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 UDPHS_RIPPADDRSIZE; 	// UDPHS PADDRSIZE Register
	AT91_REG	 UDPHS_RIPNAME1; 	// UDPHS Name1 Register
	AT91_REG	 UDPHS_RIPNAME2; 	// UDPHS Name2 Register
	AT91_REG	 UDPHS_IPFEATURES; 	// UDPHS Features Register
	AT91_REG	 UDPHS_IPVERSION; 	// UDPHS Version Register
	AT91S_UDPHS_EPT	 UDPHS_EPT[7]; 	// UDPHS Endpoint struct
	AT91_REG	 Reserved3[72]; 	// 
	AT91S_UDPHS_DMA	 UDPHS_DMA[6]; 	// UDPHS DMA channel struct (not use [0])
} AT91S_UDPHS, *AT91PS_UDPHS;

// -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- 
#define AT91C_UDPHS_DEV_ADDR  ((unsigned int) 0x7F <<  0) // (UDPHS) UDPHS Address
#define AT91C_UDPHS_FADDR_EN  ((unsigned int) 0x1 <<  7) // (UDPHS) Function Address Enable
#define AT91C_UDPHS_EN_UDPHS  ((unsigned int) 0x1 <<  8) // (UDPHS) UDPHS Enable
#define AT91C_UDPHS_DETACH    ((unsigned int) 0x1 <<  9) // (UDPHS) Detach Command
#define AT91C_UDPHS_REWAKEUP  ((unsigned int) 0x1 << 10) // (UDPHS) Send Remote Wake Up
#define AT91C_UDPHS_PULLD_DIS ((unsigned int) 0x1 << 11) // (UDPHS) PullDown Disable
// -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- 
#define AT91C_UDPHS_MICRO_FRAME_NUM ((unsigned int) 0x7 <<  0) // (UDPHS) Micro Frame Number
#define AT91C_UDPHS_FRAME_NUMBER ((unsigned int) 0x7FF <<  3) // (UDPHS) Frame Number as defined in the Packet Field Formats
#define AT91C_UDPHS_FNUM_ERR  ((unsigned int) 0x1 << 31) // (UDPHS) Frame Number CRC Error
// -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- 
#define AT91C_UDPHS_DET_SUSPD ((unsigned int) 0x1 <<  1) // (UDPHS) Suspend Interrupt Enable/Clear/Status
#define AT91C_UDPHS_MICRO_SOF ((unsigned int) 0x1 <<  2) // (UDPHS) Micro-SOF Interrupt Enable/Clear/Status
#define AT91C_UDPHS_IEN_SOF   ((unsigned int) 0x1 <<  3) // (UDPHS) SOF Interrupt Enable/Clear/Status
#define AT91C_UDPHS_ENDRESET  ((unsigned int) 0x1 <<  4) // (UDPHS) End Of Reset Interrupt Enable/Clear/Status
#define AT91C_UDPHS_WAKE_UP   ((unsigned int) 0x1 <<  5) // (UDPHS) Wake Up CPU Interrupt Enable/Clear/Status
#define AT91C_UDPHS_ENDOFRSM  ((unsigned int) 0x1 <<  6) // (UDPHS) End Of Resume Interrupt Enable/Clear/Status
#define AT91C_UDPHS_UPSTR_RES ((unsigned int) 0x1 <<  7) // (UDPHS) Upstream Resume Interrupt Enable/Clear/Status
#define AT91C_UDPHS_EPT_INT_0 ((unsigned int) 0x1 <<  8) // (UDPHS) Endpoint 0 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_1 ((unsigned int) 0x1 <<  9) // (UDPHS) Endpoint 1 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_2 ((unsigned int) 0x1 << 10) // (UDPHS) Endpoint 2 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_3 ((unsigned int) 0x1 << 11) // (UDPHS) Endpoint 3 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_4 ((unsigned int) 0x1 << 12) // (UDPHS) Endpoint 4 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_5 ((unsigned int) 0x1 << 13) // (UDPHS) Endpoint 5 Interrupt Enable/Status
#define AT91C_UDPHS_EPT_INT_6 ((unsigned int) 0x1 << 14) // (UDPHS) Endpoint 6 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_1 ((unsigned int) 0x1 << 25) // (UDPHS) DMA Channel 1 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_2 ((unsigned int) 0x1 << 26) // (UDPHS) DMA Channel 2 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_3 ((unsigned int) 0x1 << 27) // (UDPHS) DMA Channel 3 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_4 ((unsigned int) 0x1 << 28) // (UDPHS) DMA Channel 4 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_5 ((unsigned int) 0x1 << 29) // (UDPHS) DMA Channel 5 Interrupt Enable/Status
#define AT91C_UDPHS_DMA_INT_6 ((unsigned int) 0x1 << 30) // (UDPHS) DMA Channel 6 Interrupt Enable/Status
// -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- 
#define AT91C_UDPHS_SPEED     ((unsigned int) 0x1 <<  0) // (UDPHS) Speed Status
// -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- 
// -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- 
#define AT91C_UDPHS_RST_EPT_0 ((unsigned int) 0x1 <<  0) // (UDPHS) Endpoint Reset 0
#define AT91C_UDPHS_RST_EPT_1 ((unsigned int) 0x1 <<  1) // (UDPHS) Endpoint Reset 1
#define AT91C_UDPHS_RST_EPT_2 ((unsigned int) 0x1 <<  2) // (UDPHS) Endpoint Reset 2
#define AT91C_UDPHS_RST_EPT_3 ((unsigned int) 0x1 <<  3) // (UDPHS) Endpoint Reset 3
#define AT91C_UDPHS_RST_EPT_4 ((unsigned int) 0x1 <<  4) // (UDPHS) Endpoint Reset 4
#define AT91C_UDPHS_RST_EPT_5 ((unsigned int) 0x1 <<  5) // (UDPHS) Endpoint Reset 5
#define AT91C_UDPHS_RST_EPT_6 ((unsigned int) 0x1 <<  6) // (UDPHS) Endpoint Reset 6
// -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- 
#define AT91C_UDPHS_SOFCNTMAX ((unsigned int) 0x3 <<  0) // (UDPHS) SOF Counter Max Value
#define AT91C_UDPHS_SOFCTLOAD ((unsigned int) 0x1 <<  7) // (UDPHS) SOF Counter Load
// -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- 
#define AT91C_UDPHS_CNTAMAX   ((unsigned int) 0x7FFF <<  0) // (UDPHS) A Counter Max Value
#define AT91C_UDPHS_CNTALOAD  ((unsigned int) 0x1 << 15) // (UDPHS) A Counter Load
// -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- 
#define AT91C_UDPHS_CNTBMAX   ((unsigned int) 0x7FFF <<  0) // (UDPHS) B Counter Max Value
#define AT91C_UDPHS_CNTBLOAD  ((unsigned int) 0x1 << 15) // (UDPHS) B Counter Load
// -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- 
#define AT91C_UDPHS_TSTMODE   ((unsigned int) 0x1F <<  1) // (UDPHS) UDPHS Core TestModeReg
// -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- 
#define AT91C_UDPHS_SPEED_CFG ((unsigned int) 0x3 <<  0) // (UDPHS) Speed Configuration
#define 	AT91C_UDPHS_SPEED_CFG_NM                   ((unsigned int) 0x0) // (UDPHS) Normal Mode
#define 	AT91C_UDPHS_SPEED_CFG_RS                   ((unsigned int) 0x1) // (UDPHS) Reserved
#define 	AT91C_UDPHS_SPEED_CFG_HS                   ((unsigned int) 0x2) // (UDPHS) Force High Speed
#define 	AT91C_UDPHS_SPEED_CFG_FS                   ((unsigned int) 0x3) // (UDPHS) Force Full-Speed
#define AT91C_UDPHS_TST_J     ((unsigned int) 0x1 <<  2) // (UDPHS) TestJMode
#define AT91C_UDPHS_TST_K     ((unsigned int) 0x1 <<  3) // (UDPHS) TestKMode
#define AT91C_UDPHS_TST_PKT   ((unsigned int) 0x1 <<  4) // (UDPHS) TestPacketMode
#define AT91C_UDPHS_OPMODE2   ((unsigned int) 0x1 <<  5) // (UDPHS) OpMode2
// -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- 
#define AT91C_UDPHS_IPPADDRSIZE ((unsigned int) 0x0 <<  0) // (UDPHS) 2^UDPHSDEV_PADDR_SIZE
// -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- 
#define AT91C_UDPHS_IPNAME1   ((unsigned int) 0x0 <<  0) // (UDPHS) ASCII string HUSB
// -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- 
#define AT91C_UDPHS_IPNAME2   ((unsigned int) 0x0 <<  0) // (UDPHS) ASCII string 2DEV
// -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- 
#define AT91C_UDPHS_EPT_NBR_MAX ((unsigned int) 0xF <<  0) // (UDPHS) Max Number of Endpoints
#define AT91C_UDPHS_DMA_CHANNEL_NBR ((unsigned int) 0x7 <<  4) // (UDPHS) Number of DMA Channels
#define AT91C_UDPHS_DMA_B_SIZ ((unsigned int) 0x1 <<  7) // (UDPHS) DMA Buffer Size
#define AT91C_UDPHS_DMA_FIFO_WORD_DEPTH ((unsigned int) 0xF <<  8) // (UDPHS) DMA FIFO Depth in words
#define AT91C_UDPHS_FIFO_MAX_SIZE ((unsigned int) 0x7 << 12) // (UDPHS) DPRAM size
#define AT91C_UDPHS_BW_DPRAM  ((unsigned int) 0x1 << 15) // (UDPHS) DPRAM byte write capability
#define AT91C_UDPHS_DATAB16_8 ((unsigned int) 0x1 << 16) // (UDPHS) UTMI DataBus16_8
#define AT91C_UDPHS_ISO_EPT_1 ((unsigned int) 0x1 << 17) // (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_2 ((unsigned int) 0x1 << 18) // (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_5 ((unsigned int) 0x1 << 21) // (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability
#define AT91C_UDPHS_ISO_EPT_6 ((unsigned int) 0x1 << 22) // (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability
// -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- 
#define AT91C_UDPHS_VERSION_NUM ((unsigned int) 0xFFFF <<  0) // (UDPHS) Give the IP version
#define AT91C_UDPHS_METAL_FIX_NUM ((unsigned int) 0x7 << 16) // (UDPHS) Give the number of metal fixes

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR HDMA Channel structure
// *****************************************************************************
typedef struct _AT91S_HDMA_CH {
	AT91_REG	 HDMA_SADDR; 	// HDMA Channel Source Address Register
	AT91_REG	 HDMA_DADDR; 	// HDMA Channel Destination Address Register
	AT91_REG	 HDMA_DSCR; 	// HDMA Channel Descriptor Address Register
	AT91_REG	 HDMA_CTRLA; 	// HDMA Channel Control A Register
	AT91_REG	 HDMA_CTRLB; 	// HDMA Channel Control B Register
	AT91_REG	 HDMA_CFG; 	// HDMA Channel Configuration Register
} AT91S_HDMA_CH, *AT91PS_HDMA_CH;

// -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- 
#define AT91C_SADDR           ((unsigned int) 0x0 <<  0) // (HDMA_CH) 
// -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- 
#define AT91C_DADDR           ((unsigned int) 0x0 <<  0) // (HDMA_CH) 
// -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- 
#define AT91C_HDMA_DSCR       ((unsigned int) 0x3FFFFFFF <<  2) // (HDMA_CH) Buffer Transfer descriptor address. This address is word aligned.
// -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- 
#define AT91C_HDMA_BTSIZE     ((unsigned int) 0xFFFF <<  0) // (HDMA_CH) Buffer Transfer Size.
#define AT91C_HDMA_SCSIZE     ((unsigned int) 0x1 << 16) // (HDMA_CH) Source Chunk Transfer Size.
#define 	AT91C_HDMA_SCSIZE_1                    ((unsigned int) 0x0 << 16) // (HDMA_CH) 1.
#define 	AT91C_HDMA_SCSIZE_4                    ((unsigned int) 0x1 << 16) // (HDMA_CH) 4.
#define AT91C_HDMA_DCSIZE     ((unsigned int) 0x1 << 20) // (HDMA_CH) Destination Chunk Transfer Size
#define 	AT91C_HDMA_DCSIZE_1                    ((unsigned int) 0x0 << 20) // (HDMA_CH) 1.
#define 	AT91C_HDMA_DCSIZE_4                    ((unsigned int) 0x1 << 20) // (HDMA_CH) 4.
#define AT91C_HDMA_SRC_WIDTH  ((unsigned int) 0x3 << 24) // (HDMA_CH) Source Single Transfer Size
#define 	AT91C_HDMA_SRC_WIDTH_BYTE                 ((unsigned int) 0x0 << 24) // (HDMA_CH) BYTE.
#define 	AT91C_HDMA_SRC_WIDTH_HALFWORD             ((unsigned int) 0x1 << 24) // (HDMA_CH) HALF-WORD.
#define 	AT91C_HDMA_SRC_WIDTH_WORD                 ((unsigned int) 0x2 << 24) // (HDMA_CH) WORD.
#define AT91C_HDMA_DST_WIDTH  ((unsigned int) 0x3 << 28) // (HDMA_CH) Destination Single Transfer Size
#define 	AT91C_HDMA_DST_WIDTH_BYTE                 ((unsigned int) 0x0 << 28) // (HDMA_CH) BYTE.
#define 	AT91C_HDMA_DST_WIDTH_HALFWORD             ((unsigned int) 0x1 << 28) // (HDMA_CH) HALF-WORD.
#define 	AT91C_HDMA_DST_WIDTH_WORD                 ((unsigned int) 0x2 << 28) // (HDMA_CH) WORD.
#define AT91C_HDMA_DONE       ((unsigned int) 0x1 << 31) // (HDMA_CH) 
// -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- 
#define AT91C_HDMA_SRC_DSCR   ((unsigned int) 0x1 << 16) // (HDMA_CH) Source Buffer Descriptor Fetch operation
#define 	AT91C_HDMA_SRC_DSCR_FETCH_FROM_MEM       ((unsigned int) 0x0 << 16) // (HDMA_CH) Source address is updated when the descriptor is fetched from the memory.
#define 	AT91C_HDMA_SRC_DSCR_FETCH_DISABLE        ((unsigned int) 0x1 << 16) // (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the Source.
#define AT91C_HDMA_DST_DSCR   ((unsigned int) 0x1 << 20) // (HDMA_CH) Destination Buffer Descriptor operation
#define 	AT91C_HDMA_DST_DSCR_FETCH_FROM_MEM       ((unsigned int) 0x0 << 20) // (HDMA_CH) Destination address is updated when the descriptor is fetched from the memory.
#define 	AT91C_HDMA_DST_DSCR_FETCH_DISABLE        ((unsigned int) 0x1 << 20) // (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the destination.
#define AT91C_HDMA_FC         ((unsigned int) 0x7 << 21) // (HDMA_CH) This field defines which devices controls the size of the buffer transfer, also referred as to the Flow Controller.
#define 	AT91C_HDMA_FC_MEM2MEM              ((unsigned int) 0x0 << 21) // (HDMA_CH) Memory-to-Memory (DMA Controller).
#define 	AT91C_HDMA_FC_MEM2PER              ((unsigned int) 0x1 << 21) // (HDMA_CH) Memory-to-Peripheral (DMA Controller).
#define 	AT91C_HDMA_FC_PER2MEM              ((unsigned int) 0x2 << 21) // (HDMA_CH) Peripheral-to-Memory (DMA Controller).
#define 	AT91C_HDMA_FC_PER2PER              ((unsigned int) 0x3 << 21) // (HDMA_CH) Peripheral-to-Peripheral (DMA Controller).
#define AT91C_HDMA_SRC_ADDRESS_MODE ((unsigned int) 0x3 << 24) // (HDMA_CH) Type of addressing mode
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_INCR                 ((unsigned int) 0x0 << 24) // (HDMA_CH) Incrementing Mode.
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_DECR                 ((unsigned int) 0x1 << 24) // (HDMA_CH) Decrementing Mode.
#define 	AT91C_HDMA_SRC_ADDRESS_MODE_FIXED                ((unsigned int) 0x2 << 24) // (HDMA_CH) Fixed Mode.
#define AT91C_HDMA_DST_ADDRESS_MODE ((unsigned int) 0x3 << 28) // (HDMA_CH) Type of addressing mode
#define 	AT91C_HDMA_DST_ADDRESS_MODE_INCR                 ((unsigned int) 0x0 << 28) // (HDMA_CH) Incrementing Mode.
#define 	AT91C_HDMA_DST_ADDRESS_MODE_DECR                 ((unsigned int) 0x1 << 28) // (HDMA_CH) Decrementing Mode.
#define 	AT91C_HDMA_DST_ADDRESS_MODE_FIXED                ((unsigned int) 0x2 << 28) // (HDMA_CH) Fixed Mode.
#define AT91C_HDMA_IEN        ((unsigned int) 0x1 << 30) // (HDMA_CH) buffer transfer completed
// -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- 
#define AT91C_HDMA_SRC_PER    ((unsigned int) 0xF <<  0) // (HDMA_CH) Channel Source Request is associated with peripheral identifier coded SRC_PER handshaking interface.
#define 	AT91C_HDMA_SRC_PER_0                    ((unsigned int) 0x0) // (HDMA_CH) HW Handshaking Interface number 0.
#define 	AT91C_HDMA_SRC_PER_1                    ((unsigned int) 0x1) // (HDMA_CH) HW Handshaking Interface number 1.
#define 	AT91C_HDMA_SRC_PER_2                    ((unsigned int) 0x2) // (HDMA_CH) HW Handshaking Interface number 2.
#define 	AT91C_HDMA_SRC_PER_3                    ((unsigned int) 0x3) // (HDMA_CH) HW Handshaking Interface number 3.
#define 	AT91C_HDMA_SRC_PER_4                    ((unsigned int) 0x4) // (HDMA_CH) HW Handshaking Interface number 4.
#define 	AT91C_HDMA_SRC_PER_5                    ((unsigned int) 0x5) // (HDMA_CH) HW Handshaking Interface number 5.
#define 	AT91C_HDMA_SRC_PER_6                    ((unsigned int) 0x6) // (HDMA_CH) HW Handshaking Interface number 6.
#define 	AT91C_HDMA_SRC_PER_7                    ((unsigned int) 0x7) // (HDMA_CH) HW Handshaking Interface number 7.
#define AT91C_HDMA_DST_PER    ((unsigned int) 0xF <<  4) // (HDMA_CH) Channel Destination Request is associated with peripheral identifier coded DST_PER handshaking interface.
#define 	AT91C_HDMA_DST_PER_0                    ((unsigned int) 0x0 <<  4) // (HDMA_CH) HW Handshaking Interface number 0.
#define 	AT91C_HDMA_DST_PER_1                    ((unsigned int) 0x1 <<  4) // (HDMA_CH) HW Handshaking Interface number 1.
#define 	AT91C_HDMA_DST_PER_2                    ((unsigned int) 0x2 <<  4) // (HDMA_CH) HW Handshaking Interface number 2.
#define 	AT91C_HDMA_DST_PER_3                    ((unsigned int) 0x3 <<  4) // (HDMA_CH) HW Handshaking Interface number 3.
#define 	AT91C_HDMA_DST_PER_4                    ((unsigned int) 0x4 <<  4) // (HDMA_CH) HW Handshaking Interface number 4.
#define 	AT91C_HDMA_DST_PER_5                    ((unsigned int) 0x5 <<  4) // (HDMA_CH) HW Handshaking Interface number 5.
#define 	AT91C_HDMA_DST_PER_6                    ((unsigned int) 0x6 <<  4) // (HDMA_CH) HW Handshaking Interface number 6.
#define 	AT91C_HDMA_DST_PER_7                    ((unsigned int) 0x7 <<  4) // (HDMA_CH) HW Handshaking Interface number 7.
#define AT91C_HDMA_SRC_H2SEL  ((unsigned int) 0x1 <<  9) // (HDMA_CH) Source Handshaking Mode
#define 	AT91C_HDMA_SRC_H2SEL_SW                   ((unsigned int) 0x0 <<  9) // (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
#define 	AT91C_HDMA_SRC_H2SEL_HW                   ((unsigned int) 0x1 <<  9) // (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
#define AT91C_HDMA_DST_H2SEL  ((unsigned int) 0x1 << 13) // (HDMA_CH) Destination Handshaking Mode
#define 	AT91C_HDMA_DST_H2SEL_SW                   ((unsigned int) 0x0 << 13) // (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
#define 	AT91C_HDMA_DST_H2SEL_HW                   ((unsigned int) 0x1 << 13) // (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
#define AT91C_HDMA_SOD        ((unsigned int) 0x1 << 16) // (HDMA_CH) STOP ON DONE
#define 	AT91C_HDMA_SOD_DISABLE              ((unsigned int) 0x0 << 16) // (HDMA_CH) STOP ON DONE disabled, the descriptor fetch operation ignores DONE Field of CTRLA register.
#define 	AT91C_HDMA_SOD_ENABLE               ((unsigned int) 0x1 << 16) // (HDMA_CH) STOP ON DONE activated, the DMAC module is automatically disabled if DONE FIELD is set to 1.
#define AT91C_HDMA_LOCK_IF    ((unsigned int) 0x1 << 20) // (HDMA_CH) Interface Lock
#define 	AT91C_HDMA_LOCK_IF_DISABLE              ((unsigned int) 0x0 << 20) // (HDMA_CH) Interface Lock capability is disabled.
#define 	AT91C_HDMA_LOCK_IF_ENABLE               ((unsigned int) 0x1 << 20) // (HDMA_CH) Interface Lock capability is enabled.
#define AT91C_HDMA_LOCK_B     ((unsigned int) 0x1 << 21) // (HDMA_CH) AHB Bus Lock
#define 	AT91C_HDMA_LOCK_B_DISABLE              ((unsigned int) 0x0 << 21) // (HDMA_CH) AHB Bus Locking capability is disabled.
#define 	AT91C_HDMA_LOCK_B_ENABLE               ((unsigned int) 0x1 << 21) // (HDMA_CH) AHB Bus Locking capability is enabled.
#define AT91C_HDMA_LOCK_IF_L  ((unsigned int) 0x1 << 22) // (HDMA_CH) Master Interface Arbiter Lock
#define 	AT91C_HDMA_LOCK_IF_L_CHUNK                ((unsigned int) 0x0 << 22) // (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a chunk transfer.
#define 	AT91C_HDMA_LOCK_IF_L_BUFFER               ((unsigned int) 0x1 << 22) // (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a buffer transfer.
#define AT91C_HDMA_AHB_PROT   ((unsigned int) 0x7 << 24) // (HDMA_CH) AHB Prot
#define AT91C_HDMA_FIFOCFG    ((unsigned int) 0x3 << 28) // (HDMA_CH) FIFO Request Configuration
#define 	AT91C_HDMA_FIFOCFG_LARGESTBURST         ((unsigned int) 0x0 << 28) // (HDMA_CH) The largest defined length AHB burst is performed on the destination AHB interface.
#define 	AT91C_HDMA_FIFOCFG_HALFFIFO             ((unsigned int) 0x1 << 28) // (HDMA_CH) When half fifo size is available/filled a source/destination request is serviced.
#define 	AT91C_HDMA_FIFOCFG_ENOUGHSPACE          ((unsigned int) 0x2 << 28) // (HDMA_CH) When there is enough space/data available to perfom a single AHB access then the request is serviced.

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR HDMA controller
// *****************************************************************************
typedef struct _AT91S_HDMA {
	AT91_REG	 HDMA_GCFG; 	// HDMA Global Configuration Register
	AT91_REG	 HDMA_EN; 	// HDMA Controller Enable Register
	AT91_REG	 HDMA_SREQ; 	// HDMA Software Single Request Register
	AT91_REG	 HDMA_CREQ; 	// HDMA Software Chunk Transfer Request Register
	AT91_REG	 HDMA_LAST; 	// HDMA Software Last Transfer Flag Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 HDMA_EBCIER; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
	AT91_REG	 HDMA_EBCIDR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register
	AT91_REG	 HDMA_EBCIMR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
	AT91_REG	 HDMA_EBCISR; 	// HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
	AT91_REG	 HDMA_CHER; 	// HDMA Channel Handler Enable Register
	AT91_REG	 HDMA_CHDR; 	// HDMA Channel Handler Disable Register
	AT91_REG	 HDMA_CHSR; 	// HDMA Channel Handler Status Register
	AT91_REG	 Reserved1[2]; 	// 
	AT91S_HDMA_CH	 HDMA_CH[4]; 	// HDMA Channel structure
	AT91_REG	 Reserved2[84]; 	// 
	AT91_REG	 HDMA_ADDRSIZE; 	// HDMA ADDRSIZE REGISTER 
	AT91_REG	 HDMA_IPNAME1; 	// HDMA IPNAME1 REGISTER 
	AT91_REG	 HDMA_IPNAME2; 	// HDMA IPNAME2 REGISTER 
	AT91_REG	 HDMA_FEATURES; 	// HDMA FEATURES REGISTER 
	AT91_REG	 HDMA_VER; 	// HDMA VERSION REGISTER 
} AT91S_HDMA, *AT91PS_HDMA;

// -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- 
#define AT91C_HDMA_ARB_CFG    ((unsigned int) 0x1 <<  4) // (HDMA) Arbiter mode.
#define 	AT91C_HDMA_ARB_CFG_FIXED                ((unsigned int) 0x0 <<  4) // (HDMA) Fixed priority arbiter.
#define 	AT91C_HDMA_ARB_CFG_ROUND_ROBIN          ((unsigned int) 0x1 <<  4) // (HDMA) Modified round robin arbiter.
// -------- HDMA_EN : (HDMA Offset: 0x4)  -------- 
#define AT91C_HDMA_ENABLE     ((unsigned int) 0x1 <<  0) // (HDMA) 
#define 	AT91C_HDMA_ENABLE_DISABLE              ((unsigned int) 0x0) // (HDMA) Disables HDMA.
#define 	AT91C_HDMA_ENABLE_ENABLE               ((unsigned int) 0x1) // (HDMA) Enables HDMA.
// -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- 
#define AT91C_HDMA_SSREQ0     ((unsigned int) 0x1 <<  0) // (HDMA) Request a source single transfer on channel 0
#define 	AT91C_HDMA_SSREQ0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ0_1                    ((unsigned int) 0x1) // (HDMA) Request a source single transfer on channel 0.
#define AT91C_HDMA_DSREQ0     ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination single transfer on channel 0
#define 	AT91C_HDMA_DSREQ0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination single transfer on channel 0.
#define AT91C_HDMA_SSREQ1     ((unsigned int) 0x1 <<  2) // (HDMA) Request a source single transfer on channel 1
#define 	AT91C_HDMA_SSREQ1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Request a source single transfer on channel 1.
#define AT91C_HDMA_DSREQ1     ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination single transfer on channel 1
#define 	AT91C_HDMA_DSREQ1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination single transfer on channel 1.
#define AT91C_HDMA_SSREQ2     ((unsigned int) 0x1 <<  4) // (HDMA) Request a source single transfer on channel 2
#define 	AT91C_HDMA_SSREQ2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Request a source single transfer on channel 2.
#define AT91C_HDMA_DSREQ2     ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination single transfer on channel 2
#define 	AT91C_HDMA_DSREQ2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination single transfer on channel 2.
#define AT91C_HDMA_SSREQ3     ((unsigned int) 0x1 <<  6) // (HDMA) Request a source single transfer on channel 3
#define 	AT91C_HDMA_SSREQ3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SSREQ3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Request a source single transfer on channel 3.
#define AT91C_HDMA_DSREQ3     ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination single transfer on channel 3
#define 	AT91C_HDMA_DSREQ3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DSREQ3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination single transfer on channel 3.
// -------- HDMA_CREQ : (HDMA Offset: 0xc)  -------- 
#define AT91C_HDMA_SCREQ0     ((unsigned int) 0x1 <<  0) // (HDMA) Request a source chunk transfer on channel 0
#define 	AT91C_HDMA_SCREQ0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ0_1                    ((unsigned int) 0x1) // (HDMA) Request a source chunk transfer on channel 0.
#define AT91C_HDMA_DCREQ0     ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination chunk transfer on channel 0
#define 	AT91C_HDMA_DCREQ0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Request a destination chunk transfer on channel 0.
#define AT91C_HDMA_SCREQ1     ((unsigned int) 0x1 <<  2) // (HDMA) Request a source chunk transfer on channel 1
#define 	AT91C_HDMA_SCREQ1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Request a source chunk transfer on channel 1.
#define AT91C_HDMA_DCREQ1     ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination chunk transfer on channel 1
#define 	AT91C_HDMA_DCREQ1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Request a destination chunk transfer on channel 1.
#define AT91C_HDMA_SCREQ2     ((unsigned int) 0x1 <<  4) // (HDMA) Request a source chunk transfer on channel 2
#define 	AT91C_HDMA_SCREQ2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Request a source chunk transfer on channel 2.
#define AT91C_HDMA_DCREQ2     ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination chunk transfer on channel 2
#define 	AT91C_HDMA_DCREQ2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Request a destination chunk transfer on channel 2.
#define AT91C_HDMA_SCREQ3     ((unsigned int) 0x1 <<  6) // (HDMA) Request a source chunk transfer on channel 3
#define 	AT91C_HDMA_SCREQ3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SCREQ3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Request a source chunk transfer on channel 3.
#define AT91C_HDMA_DCREQ3     ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination chunk transfer on channel 3
#define 	AT91C_HDMA_DCREQ3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DCREQ3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Request a destination chunk transfer on channel 3.
// -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- 
#define AT91C_HDMA_SLAST0     ((unsigned int) 0x1 <<  0) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 0
#define 	AT91C_HDMA_SLAST0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST0_1                    ((unsigned int) 0x1) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 0.
#define AT91C_HDMA_DLAST0     ((unsigned int) 0x1 <<  1) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 0
#define 	AT91C_HDMA_DLAST0_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST0_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 0.
#define AT91C_HDMA_SLAST1     ((unsigned int) 0x1 <<  2) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 1
#define 	AT91C_HDMA_SLAST1_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST1_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 1.
#define AT91C_HDMA_DLAST1     ((unsigned int) 0x1 <<  3) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 1
#define 	AT91C_HDMA_DLAST1_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST1_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 1.
#define AT91C_HDMA_SLAST2     ((unsigned int) 0x1 <<  4) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 2
#define 	AT91C_HDMA_SLAST2_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST2_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 2.
#define AT91C_HDMA_DLAST2     ((unsigned int) 0x1 <<  5) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 2
#define 	AT91C_HDMA_DLAST2_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST2_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 2.
#define AT91C_HDMA_SLAST3     ((unsigned int) 0x1 <<  6) // (HDMA) Indicates that this source request is the last transfer of the buffer on channel 3
#define 	AT91C_HDMA_SLAST3_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_SLAST3_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 3.
#define AT91C_HDMA_DLAST3     ((unsigned int) 0x1 <<  7) // (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 3
#define 	AT91C_HDMA_DLAST3_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DLAST3_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 3.
// -------- HDMA_EBCIER : (HDMA Offset: 0x18) Buffer Transfer Completed/Chained Buffer Transfer Completed/Access Error Interrupt Enable Register -------- 
#define AT91C_HDMA_BTC0       ((unsigned int) 0x1 <<  0) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC1       ((unsigned int) 0x1 <<  1) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC2       ((unsigned int) 0x1 <<  2) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC3       ((unsigned int) 0x1 <<  3) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC4       ((unsigned int) 0x1 <<  4) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC5       ((unsigned int) 0x1 <<  5) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC6       ((unsigned int) 0x1 <<  6) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_BTC7       ((unsigned int) 0x1 <<  7) // (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC0      ((unsigned int) 0x1 <<  8) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC1      ((unsigned int) 0x1 <<  9) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC2      ((unsigned int) 0x1 << 10) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC3      ((unsigned int) 0x1 << 11) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC4      ((unsigned int) 0x1 << 12) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC5      ((unsigned int) 0x1 << 13) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC6      ((unsigned int) 0x1 << 14) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_CBTC7      ((unsigned int) 0x1 << 15) // (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR0       ((unsigned int) 0x1 << 16) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR1       ((unsigned int) 0x1 << 17) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR2       ((unsigned int) 0x1 << 18) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR3       ((unsigned int) 0x1 << 19) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR4       ((unsigned int) 0x1 << 20) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR5       ((unsigned int) 0x1 << 21) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR6       ((unsigned int) 0x1 << 22) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
#define AT91C_HDMA_ERR7       ((unsigned int) 0x1 << 23) // (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
// -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- 
// -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- 
// -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- 
// -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- 
#define AT91C_HDMA_ENA0       ((unsigned int) 0x1 <<  0) // (HDMA) When set, channel 0 enabled.
#define 	AT91C_HDMA_ENA0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA0_1                    ((unsigned int) 0x1) // (HDMA) Channel 0 enabled.
#define AT91C_HDMA_ENA1       ((unsigned int) 0x1 <<  1) // (HDMA) When set, channel 1 enabled.
#define 	AT91C_HDMA_ENA1_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA1_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Channel 1 enabled.
#define AT91C_HDMA_ENA2       ((unsigned int) 0x1 <<  2) // (HDMA) When set, channel 2 enabled.
#define 	AT91C_HDMA_ENA2_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA2_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Channel 2 enabled.
#define AT91C_HDMA_ENA3       ((unsigned int) 0x1 <<  3) // (HDMA) When set, channel 3 enabled.
#define 	AT91C_HDMA_ENA3_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA3_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Channel 3 enabled.
#define AT91C_HDMA_ENA4       ((unsigned int) 0x1 <<  4) // (HDMA) When set, channel 4 enabled.
#define 	AT91C_HDMA_ENA4_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA4_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Channel 4 enabled.
#define AT91C_HDMA_ENA5       ((unsigned int) 0x1 <<  5) // (HDMA) When set, channel 5 enabled.
#define 	AT91C_HDMA_ENA5_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA5_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Channel 5 enabled.
#define AT91C_HDMA_ENA6       ((unsigned int) 0x1 <<  6) // (HDMA) When set, channel 6 enabled.
#define 	AT91C_HDMA_ENA6_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA6_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Channel 6 enabled.
#define AT91C_HDMA_ENA7       ((unsigned int) 0x1 <<  7) // (HDMA) When set, channel 7 enabled.
#define 	AT91C_HDMA_ENA7_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_ENA7_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Channel 7 enabled.
#define AT91C_HDMA_SUSP0      ((unsigned int) 0x1 <<  8) // (HDMA) When set, channel 0 freezed and its current context.
#define 	AT91C_HDMA_SUSP0_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP0_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Channel 0 freezed.
#define AT91C_HDMA_SUSP1      ((unsigned int) 0x1 <<  9) // (HDMA) When set, channel 1 freezed and its current context.
#define 	AT91C_HDMA_SUSP1_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP1_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Channel 1 freezed.
#define AT91C_HDMA_SUSP2      ((unsigned int) 0x1 << 10) // (HDMA) When set, channel 2 freezed and its current context.
#define 	AT91C_HDMA_SUSP2_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP2_1                    ((unsigned int) 0x1 << 10) // (HDMA) Channel 2 freezed.
#define AT91C_HDMA_SUSP3      ((unsigned int) 0x1 << 11) // (HDMA) When set, channel 3 freezed and its current context.
#define 	AT91C_HDMA_SUSP3_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP3_1                    ((unsigned int) 0x1 << 11) // (HDMA) Channel 3 freezed.
#define AT91C_HDMA_SUSP4      ((unsigned int) 0x1 << 12) // (HDMA) When set, channel 4 freezed and its current context.
#define 	AT91C_HDMA_SUSP4_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP4_1                    ((unsigned int) 0x1 << 12) // (HDMA) Channel 4 freezed.
#define AT91C_HDMA_SUSP5      ((unsigned int) 0x1 << 13) // (HDMA) When set, channel 5 freezed and its current context.
#define 	AT91C_HDMA_SUSP5_0                    ((unsigned int) 0x0 << 13) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP5_1                    ((unsigned int) 0x1 << 13) // (HDMA) Channel 5 freezed.
#define AT91C_HDMA_SUSP6      ((unsigned int) 0x1 << 14) // (HDMA) When set, channel 6 freezed and its current context.
#define 	AT91C_HDMA_SUSP6_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP6_1                    ((unsigned int) 0x1 << 14) // (HDMA) Channel 6 freezed.
#define AT91C_HDMA_SUSP7      ((unsigned int) 0x1 << 15) // (HDMA) When set, channel 7 freezed and its current context.
#define 	AT91C_HDMA_SUSP7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_SUSP7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Channel 7 freezed.
#define AT91C_HDMA_KEEP0      ((unsigned int) 0x1 << 24) // (HDMA) When set, it resumes the channel 0 from an automatic stall state.
#define 	AT91C_HDMA_KEEP0_0                    ((unsigned int) 0x0 << 24) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP0_1                    ((unsigned int) 0x1 << 24) // (HDMA) Resumes the channel 0.
#define AT91C_HDMA_KEEP1      ((unsigned int) 0x1 << 25) // (HDMA) When set, it resumes the channel 1 from an automatic stall state.
#define 	AT91C_HDMA_KEEP1_0                    ((unsigned int) 0x0 << 25) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP1_1                    ((unsigned int) 0x1 << 25) // (HDMA) Resumes the channel 1.
#define AT91C_HDMA_KEEP2      ((unsigned int) 0x1 << 26) // (HDMA) When set, it resumes the channel 2 from an automatic stall state.
#define 	AT91C_HDMA_KEEP2_0                    ((unsigned int) 0x0 << 26) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP2_1                    ((unsigned int) 0x1 << 26) // (HDMA) Resumes the channel 2.
#define AT91C_HDMA_KEEP3      ((unsigned int) 0x1 << 27) // (HDMA) When set, it resumes the channel 3 from an automatic stall state.
#define 	AT91C_HDMA_KEEP3_0                    ((unsigned int) 0x0 << 27) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP3_1                    ((unsigned int) 0x1 << 27) // (HDMA) Resumes the channel 3.
#define AT91C_HDMA_KEEP4      ((unsigned int) 0x1 << 28) // (HDMA) When set, it resumes the channel 4 from an automatic stall state.
#define 	AT91C_HDMA_KEEP4_0                    ((unsigned int) 0x0 << 28) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP4_1                    ((unsigned int) 0x1 << 28) // (HDMA) Resumes the channel 4.
#define AT91C_HDMA_KEEP5      ((unsigned int) 0x1 << 29) // (HDMA) When set, it resumes the channel 5 from an automatic stall state.
#define 	AT91C_HDMA_KEEP5_0                    ((unsigned int) 0x0 << 29) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP5_1                    ((unsigned int) 0x1 << 29) // (HDMA) Resumes the channel 5.
#define AT91C_HDMA_KEEP6      ((unsigned int) 0x1 << 30) // (HDMA) When set, it resumes the channel 6 from an automatic stall state.
#define 	AT91C_HDMA_KEEP6_0                    ((unsigned int) 0x0 << 30) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP6_1                    ((unsigned int) 0x1 << 30) // (HDMA) Resumes the channel 6.
#define AT91C_HDMA_KEEP7      ((unsigned int) 0x1 << 31) // (HDMA) When set, it resumes the channel 7 from an automatic stall state.
#define 	AT91C_HDMA_KEEP7_0                    ((unsigned int) 0x0 << 31) // (HDMA) No effect.
#define 	AT91C_HDMA_KEEP7_1                    ((unsigned int) 0x1 << 31) // (HDMA) Resumes the channel 7.
// -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- 
#define AT91C_HDMA_DIS0       ((unsigned int) 0x1 <<  0) // (HDMA) Write one to this field to disable the channel 0.
#define 	AT91C_HDMA_DIS0_0                    ((unsigned int) 0x0) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS0_1                    ((unsigned int) 0x1) // (HDMA) Disables the channel 0.
#define AT91C_HDMA_DIS1       ((unsigned int) 0x1 <<  1) // (HDMA) Write one to this field to disable the channel 1.
#define 	AT91C_HDMA_DIS1_0                    ((unsigned int) 0x0 <<  1) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS1_1                    ((unsigned int) 0x1 <<  1) // (HDMA) Disables the channel 1.
#define AT91C_HDMA_DIS2       ((unsigned int) 0x1 <<  2) // (HDMA) Write one to this field to disable the channel 2.
#define 	AT91C_HDMA_DIS2_0                    ((unsigned int) 0x0 <<  2) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS2_1                    ((unsigned int) 0x1 <<  2) // (HDMA) Disables the channel 2.
#define AT91C_HDMA_DIS3       ((unsigned int) 0x1 <<  3) // (HDMA) Write one to this field to disable the channel 3.
#define 	AT91C_HDMA_DIS3_0                    ((unsigned int) 0x0 <<  3) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS3_1                    ((unsigned int) 0x1 <<  3) // (HDMA) Disables the channel 3.
#define AT91C_HDMA_DIS4       ((unsigned int) 0x1 <<  4) // (HDMA) Write one to this field to disable the channel 4.
#define 	AT91C_HDMA_DIS4_0                    ((unsigned int) 0x0 <<  4) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS4_1                    ((unsigned int) 0x1 <<  4) // (HDMA) Disables the channel 4.
#define AT91C_HDMA_DIS5       ((unsigned int) 0x1 <<  5) // (HDMA) Write one to this field to disable the channel 5.
#define 	AT91C_HDMA_DIS5_0                    ((unsigned int) 0x0 <<  5) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS5_1                    ((unsigned int) 0x1 <<  5) // (HDMA) Disables the channel 5.
#define AT91C_HDMA_DIS6       ((unsigned int) 0x1 <<  6) // (HDMA) Write one to this field to disable the channel 6.
#define 	AT91C_HDMA_DIS6_0                    ((unsigned int) 0x0 <<  6) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS6_1                    ((unsigned int) 0x1 <<  6) // (HDMA) Disables the channel 6.
#define AT91C_HDMA_DIS7       ((unsigned int) 0x1 <<  7) // (HDMA) Write one to this field to disable the channel 7.
#define 	AT91C_HDMA_DIS7_0                    ((unsigned int) 0x0 <<  7) // (HDMA) No effect.
#define 	AT91C_HDMA_DIS7_1                    ((unsigned int) 0x1 <<  7) // (HDMA) Disables the channel 7.
#define AT91C_HDMA_RES0       ((unsigned int) 0x1 <<  8) // (HDMA) Write one to this field to resume the channel 0 transfer restoring its context.
#define 	AT91C_HDMA_RES0_0                    ((unsigned int) 0x0 <<  8) // (HDMA) No effect.
#define 	AT91C_HDMA_RES0_1                    ((unsigned int) 0x1 <<  8) // (HDMA) Resumes the channel 0.
#define AT91C_HDMA_RES1       ((unsigned int) 0x1 <<  9) // (HDMA) Write one to this field to resume the channel 1 transfer restoring its context.
#define 	AT91C_HDMA_RES1_0                    ((unsigned int) 0x0 <<  9) // (HDMA) No effect.
#define 	AT91C_HDMA_RES1_1                    ((unsigned int) 0x1 <<  9) // (HDMA) Resumes the channel 1.
#define AT91C_HDMA_RES2       ((unsigned int) 0x1 << 10) // (HDMA) Write one to this field to resume the channel 2 transfer restoring its context.
#define 	AT91C_HDMA_RES2_0                    ((unsigned int) 0x0 << 10) // (HDMA) No effect.
#define 	AT91C_HDMA_RES2_1                    ((unsigned int) 0x1 << 10) // (HDMA) Resumes the channel 2.
#define AT91C_HDMA_RES3       ((unsigned int) 0x1 << 11) // (HDMA) Write one to this field to resume the channel 3 transfer restoring its context.
#define 	AT91C_HDMA_RES3_0                    ((unsigned int) 0x0 << 11) // (HDMA) No effect.
#define 	AT91C_HDMA_RES3_1                    ((unsigned int) 0x1 << 11) // (HDMA) Resumes the channel 3.
#define AT91C_HDMA_RES4       ((unsigned int) 0x1 << 12) // (HDMA) Write one to this field to resume the channel 4 transfer restoring its context.
#define 	AT91C_HDMA_RES4_0                    ((unsigned int) 0x0 << 12) // (HDMA) No effect.
#define 	AT91C_HDMA_RES4_1                    ((unsigned int) 0x1 << 12) // (HDMA) Resumes the channel 4.
#define AT91C_HDMA_RES5       ((unsigned int) 0x1 << 13) // (HDMA) Write one to this field to resume the channel 5 transfer restoring its context.
#define 	AT91C_HDMA_RES5_0                    ((unsigned int) 0x0 << 13) // (HDMA) No effect.
#define 	AT91C_HDMA_RES5_1                    ((unsigned int) 0x1 << 13) // (HDMA) Resumes the channel 5.
#define AT91C_HDMA_RES6       ((unsigned int) 0x1 << 14) // (HDMA) Write one to this field to resume the channel 6 transfer restoring its context.
#define 	AT91C_HDMA_RES6_0                    ((unsigned int) 0x0 << 14) // (HDMA) No effect.
#define 	AT91C_HDMA_RES6_1                    ((unsigned int) 0x1 << 14) // (HDMA) Resumes the channel 6.
#define AT91C_HDMA_RES7       ((unsigned int) 0x1 << 15) // (HDMA) Write one to this field to resume the channel 7 transfer restoring its context.
#define 	AT91C_HDMA_RES7_0                    ((unsigned int) 0x0 << 15) // (HDMA) No effect.
#define 	AT91C_HDMA_RES7_1                    ((unsigned int) 0x1 << 15) // (HDMA) Resumes the channel 7.
// -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- 
#define AT91C_HDMA_EMPT0      ((unsigned int) 0x1 << 16) // (HDMA) When set, channel 0 is empty.
#define 	AT91C_HDMA_EMPT0_0                    ((unsigned int) 0x0 << 16) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT0_1                    ((unsigned int) 0x1 << 16) // (HDMA) Channel 0 empty.
#define AT91C_HDMA_EMPT1      ((unsigned int) 0x1 << 17) // (HDMA) When set, channel 1 is empty.
#define 	AT91C_HDMA_EMPT1_0                    ((unsigned int) 0x0 << 17) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT1_1                    ((unsigned int) 0x1 << 17) // (HDMA) Channel 1 empty.
#define AT91C_HDMA_EMPT2      ((unsigned int) 0x1 << 18) // (HDMA) When set, channel 2 is empty.
#define 	AT91C_HDMA_EMPT2_0                    ((unsigned int) 0x0 << 18) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT2_1                    ((unsigned int) 0x1 << 18) // (HDMA) Channel 2 empty.
#define AT91C_HDMA_EMPT3      ((unsigned int) 0x1 << 19) // (HDMA) When set, channel 3 is empty.
#define 	AT91C_HDMA_EMPT3_0                    ((unsigned int) 0x0 << 19) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT3_1                    ((unsigned int) 0x1 << 19) // (HDMA) Channel 3 empty.
#define AT91C_HDMA_EMPT4      ((unsigned int) 0x1 << 20) // (HDMA) When set, channel 4 is empty.
#define 	AT91C_HDMA_EMPT4_0                    ((unsigned int) 0x0 << 20) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT4_1                    ((unsigned int) 0x1 << 20) // (HDMA) Channel 4 empty.
#define AT91C_HDMA_EMPT5      ((unsigned int) 0x1 << 21) // (HDMA) When set, channel 5 is empty.
#define 	AT91C_HDMA_EMPT5_0                    ((unsigned int) 0x0 << 21) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT5_1                    ((unsigned int) 0x1 << 21) // (HDMA) Channel 5 empty.
#define AT91C_HDMA_EMPT6      ((unsigned int) 0x1 << 22) // (HDMA) When set, channel 6 is empty.
#define 	AT91C_HDMA_EMPT6_0                    ((unsigned int) 0x0 << 22) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT6_1                    ((unsigned int) 0x1 << 22) // (HDMA) Channel 6 empty.
#define AT91C_HDMA_EMPT7      ((unsigned int) 0x1 << 23) // (HDMA) When set, channel 7 is empty.
#define 	AT91C_HDMA_EMPT7_0                    ((unsigned int) 0x0 << 23) // (HDMA) No effect.
#define 	AT91C_HDMA_EMPT7_1                    ((unsigned int) 0x1 << 23) // (HDMA) Channel 7 empty.
#define AT91C_HDMA_STAL0      ((unsigned int) 0x1 << 24) // (HDMA) When set, channel 0 is stalled.
#define 	AT91C_HDMA_STAL0_0                    ((unsigned int) 0x0 << 24) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL0_1                    ((unsigned int) 0x1 << 24) // (HDMA) Channel 0 stalled.
#define AT91C_HDMA_STAL1      ((unsigned int) 0x1 << 25) // (HDMA) When set, channel 1 is stalled.
#define 	AT91C_HDMA_STAL1_0                    ((unsigned int) 0x0 << 25) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL1_1                    ((unsigned int) 0x1 << 25) // (HDMA) Channel 1 stalled.
#define AT91C_HDMA_STAL2      ((unsigned int) 0x1 << 26) // (HDMA) When set, channel 2 is stalled.
#define 	AT91C_HDMA_STAL2_0                    ((unsigned int) 0x0 << 26) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL2_1                    ((unsigned int) 0x1 << 26) // (HDMA) Channel 2 stalled.
#define AT91C_HDMA_STAL3      ((unsigned int) 0x1 << 27) // (HDMA) When set, channel 3 is stalled.
#define 	AT91C_HDMA_STAL3_0                    ((unsigned int) 0x0 << 27) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL3_1                    ((unsigned int) 0x1 << 27) // (HDMA) Channel 3 stalled.
#define AT91C_HDMA_STAL4      ((unsigned int) 0x1 << 28) // (HDMA) When set, channel 4 is stalled.
#define 	AT91C_HDMA_STAL4_0                    ((unsigned int) 0x0 << 28) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL4_1                    ((unsigned int) 0x1 << 28) // (HDMA) Channel 4 stalled.
#define AT91C_HDMA_STAL5      ((unsigned int) 0x1 << 29) // (HDMA) When set, channel 5 is stalled.
#define 	AT91C_HDMA_STAL5_0                    ((unsigned int) 0x0 << 29) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL5_1                    ((unsigned int) 0x1 << 29) // (HDMA) Channel 5 stalled.
#define AT91C_HDMA_STAL6      ((unsigned int) 0x1 << 30) // (HDMA) When set, channel 6 is stalled.
#define 	AT91C_HDMA_STAL6_0                    ((unsigned int) 0x0 << 30) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL6_1                    ((unsigned int) 0x1 << 30) // (HDMA) Channel 6 stalled.
#define AT91C_HDMA_STAL7      ((unsigned int) 0x1 << 31) // (HDMA) When set, channel 7 is stalled.
#define 	AT91C_HDMA_STAL7_0                    ((unsigned int) 0x0 << 31) // (HDMA) No effect.
#define 	AT91C_HDMA_STAL7_1                    ((unsigned int) 0x1 << 31) // (HDMA) Channel 7 stalled.
// -------- HDMA_VER : (HDMA Offset: 0x1fc)  -------- 

// *****************************************************************************
//               REGISTER ADDRESS DEFINITION FOR AT91SAM3U1
// *****************************************************************************
// ========== Register definition for SYS peripheral ========== 
#define AT91C_SYS_GPBR  ((AT91_REG *) 	0x400E1290) // (SYS) General Purpose Register
// ========== Register definition for HSMC4_CS0 peripheral ========== 
#define AT91C_CS0_MODE  ((AT91_REG *) 	0x400E0080) // (HSMC4_CS0) Mode Register
#define AT91C_CS0_PULSE ((AT91_REG *) 	0x400E0074) // (HSMC4_CS0) Pulse Register
#define AT91C_CS0_CYCLE ((AT91_REG *) 	0x400E0078) // (HSMC4_CS0) Cycle Register
#define AT91C_CS0_TIMINGS ((AT91_REG *) 	0x400E007C) // (HSMC4_CS0) Timmings Register
#define AT91C_CS0_SETUP ((AT91_REG *) 	0x400E0070) // (HSMC4_CS0) Setup Register
// ========== Register definition for HSMC4_CS1 peripheral ========== 
#define AT91C_CS1_CYCLE ((AT91_REG *) 	0x400E008C) // (HSMC4_CS1) Cycle Register
#define AT91C_CS1_PULSE ((AT91_REG *) 	0x400E0088) // (HSMC4_CS1) Pulse Register
#define AT91C_CS1_MODE  ((AT91_REG *) 	0x400E0094) // (HSMC4_CS1) Mode Register
#define AT91C_CS1_SETUP ((AT91_REG *) 	0x400E0084) // (HSMC4_CS1) Setup Register
#define AT91C_CS1_TIMINGS ((AT91_REG *) 	0x400E0090) // (HSMC4_CS1) Timmings Register
// ========== Register definition for HSMC4_CS2 peripheral ========== 
#define AT91C_CS2_PULSE ((AT91_REG *) 	0x400E009C) // (HSMC4_CS2) Pulse Register
#define AT91C_CS2_TIMINGS ((AT91_REG *) 	0x400E00A4) // (HSMC4_CS2) Timmings Register
#define AT91C_CS2_CYCLE ((AT91_REG *) 	0x400E00A0) // (HSMC4_CS2) Cycle Register
#define AT91C_CS2_MODE  ((AT91_REG *) 	0x400E00A8) // (HSMC4_CS2) Mode Register
#define AT91C_CS2_SETUP ((AT91_REG *) 	0x400E0098) // (HSMC4_CS2) Setup Register
// ========== Register definition for HSMC4_CS3 peripheral ========== 
#define AT91C_CS3_MODE  ((AT91_REG *) 	0x400E00BC) // (HSMC4_CS3) Mode Register
#define AT91C_CS3_TIMINGS ((AT91_REG *) 	0x400E00B8) // (HSMC4_CS3) Timmings Register
#define AT91C_CS3_SETUP ((AT91_REG *) 	0x400E00AC) // (HSMC4_CS3) Setup Register
#define AT91C_CS3_CYCLE ((AT91_REG *) 	0x400E00B4) // (HSMC4_CS3) Cycle Register
#define AT91C_CS3_PULSE ((AT91_REG *) 	0x400E00B0) // (HSMC4_CS3) Pulse Register
// ========== Register definition for HSMC4_NFC peripheral ========== 
#define AT91C_NFC_MODE  ((AT91_REG *) 	0x400E010C) // (HSMC4_NFC) Mode Register
#define AT91C_NFC_CYCLE ((AT91_REG *) 	0x400E0104) // (HSMC4_NFC) Cycle Register
#define AT91C_NFC_PULSE ((AT91_REG *) 	0x400E0100) // (HSMC4_NFC) Pulse Register
#define AT91C_NFC_SETUP ((AT91_REG *) 	0x400E00FC) // (HSMC4_NFC) Setup Register
#define AT91C_NFC_TIMINGS ((AT91_REG *) 	0x400E0108) // (HSMC4_NFC) Timmings Register
// ========== Register definition for HSMC4 peripheral ========== 
#define AT91C_HSMC4_IPNAME1 ((AT91_REG *) 	0x400E01F0) // (HSMC4) Write Protection Status Register
#define AT91C_HSMC4_ECCPR6 ((AT91_REG *) 	0x400E0048) // (HSMC4) ECC Parity register 6
#define AT91C_HSMC4_ADDRSIZE ((AT91_REG *) 	0x400E01EC) // (HSMC4) Write Protection Status Register
#define AT91C_HSMC4_ECCPR11 ((AT91_REG *) 	0x400E005C) // (HSMC4) ECC Parity register 11
#define AT91C_HSMC4_SR  ((AT91_REG *) 	0x400E0008) // (HSMC4) Status Register
#define AT91C_HSMC4_IMR ((AT91_REG *) 	0x400E0014) // (HSMC4) Interrupt Mask Register
#define AT91C_HSMC4_WPSR ((AT91_REG *) 	0x400E01E8) // (HSMC4) Write Protection Status Register
#define AT91C_HSMC4_BANK ((AT91_REG *) 	0x400E001C) // (HSMC4) Bank Register
#define AT91C_HSMC4_ECCPR8 ((AT91_REG *) 	0x400E0050) // (HSMC4) ECC Parity register 8
#define AT91C_HSMC4_WPCR ((AT91_REG *) 	0x400E01E4) // (HSMC4) Write Protection Control register
#define AT91C_HSMC4_ECCPR2 ((AT91_REG *) 	0x400E0038) // (HSMC4) ECC Parity register 2
#define AT91C_HSMC4_ECCPR1 ((AT91_REG *) 	0x400E0030) // (HSMC4) ECC Parity register 1
#define AT91C_HSMC4_ECCSR2 ((AT91_REG *) 	0x400E0034) // (HSMC4) ECC Status register 2
#define AT91C_HSMC4_OCMS ((AT91_REG *) 	0x400E0110) // (HSMC4) OCMS MODE register
#define AT91C_HSMC4_ECCPR9 ((AT91_REG *) 	0x400E0054) // (HSMC4) ECC Parity register 9
#define AT91C_HSMC4_DUMMY ((AT91_REG *) 	0x400E0200) // (HSMC4) This rtegister was created only ti have AHB constants
#define AT91C_HSMC4_ECCPR5 ((AT91_REG *) 	0x400E0044) // (HSMC4) ECC Parity register 5
#define AT91C_HSMC4_ECCCR ((AT91_REG *) 	0x400E0020) // (HSMC4) ECC reset register
#define AT91C_HSMC4_KEY2 ((AT91_REG *) 	0x400E0118) // (HSMC4) KEY2 Register
#define AT91C_HSMC4_IER ((AT91_REG *) 	0x400E000C) // (HSMC4) Interrupt Enable Register
#define AT91C_HSMC4_ECCSR1 ((AT91_REG *) 	0x400E0028) // (HSMC4) ECC Status register 1
#define AT91C_HSMC4_IDR ((AT91_REG *) 	0x400E0010) // (HSMC4) Interrupt Disable Register
#define AT91C_HSMC4_ECCPR0 ((AT91_REG *) 	0x400E002C) // (HSMC4) ECC Parity register 0
#define AT91C_HSMC4_FEATURES ((AT91_REG *) 	0x400E01F8) // (HSMC4) Write Protection Status Register
#define AT91C_HSMC4_ECCPR7 ((AT91_REG *) 	0x400E004C) // (HSMC4) ECC Parity register 7
#define AT91C_HSMC4_ECCPR12 ((AT91_REG *) 	0x400E0060) // (HSMC4) ECC Parity register 12
#define AT91C_HSMC4_ECCPR10 ((AT91_REG *) 	0x400E0058) // (HSMC4) ECC Parity register 10
#define AT91C_HSMC4_KEY1 ((AT91_REG *) 	0x400E0114) // (HSMC4) KEY1 Register
#define AT91C_HSMC4_VER ((AT91_REG *) 	0x400E01FC) // (HSMC4) HSMC4 Version Register
#define AT91C_HSMC4_Eccpr15 ((AT91_REG *) 	0x400E006C) // (HSMC4) ECC Parity register 15
#define AT91C_HSMC4_ECCPR4 ((AT91_REG *) 	0x400E0040) // (HSMC4) ECC Parity register 4
#define AT91C_HSMC4_IPNAME2 ((AT91_REG *) 	0x400E01F4) // (HSMC4) Write Protection Status Register
#define AT91C_HSMC4_ECCCMD ((AT91_REG *) 	0x400E0024) // (HSMC4) ECC Page size register
#define AT91C_HSMC4_ADDR ((AT91_REG *) 	0x400E0018) // (HSMC4) Address Cycle Zero Register
#define AT91C_HSMC4_ECCPR3 ((AT91_REG *) 	0x400E003C) // (HSMC4) ECC Parity register 3
#define AT91C_HSMC4_CFG ((AT91_REG *) 	0x400E0000) // (HSMC4) Configuration Register
#define AT91C_HSMC4_CTRL ((AT91_REG *) 	0x400E0004) // (HSMC4) Control Register
#define AT91C_HSMC4_ECCPR13 ((AT91_REG *) 	0x400E0064) // (HSMC4) ECC Parity register 13
#define AT91C_HSMC4_ECCPR14 ((AT91_REG *) 	0x400E0068) // (HSMC4) ECC Parity register 14
// ========== Register definition for MATRIX peripheral ========== 
#define AT91C_MATRIX_SFR2  ((AT91_REG *) 	0x400E0318) // (MATRIX)  Special Function Register 2
#define AT91C_MATRIX_SFR3  ((AT91_REG *) 	0x400E031C) // (MATRIX)  Special Function Register 3
#define AT91C_MATRIX_SCFG8 ((AT91_REG *) 	0x400E0260) // (MATRIX)  Slave Configuration Register 8
#define AT91C_MATRIX_MCFG2 ((AT91_REG *) 	0x400E0208) // (MATRIX)  Master Configuration Register 2
#define AT91C_MATRIX_MCFG7 ((AT91_REG *) 	0x400E021C) // (MATRIX)  Master Configuration Register 7
#define AT91C_MATRIX_SCFG3 ((AT91_REG *) 	0x400E024C) // (MATRIX)  Slave Configuration Register 3
#define AT91C_MATRIX_SCFG0 ((AT91_REG *) 	0x400E0240) // (MATRIX)  Slave Configuration Register 0
#define AT91C_MATRIX_SFR12 ((AT91_REG *) 	0x400E0340) // (MATRIX)  Special Function Register 12
#define AT91C_MATRIX_SCFG1 ((AT91_REG *) 	0x400E0244) // (MATRIX)  Slave Configuration Register 1
#define AT91C_MATRIX_SFR8  ((AT91_REG *) 	0x400E0330) // (MATRIX)  Special Function Register 8
#define AT91C_MATRIX_VER ((AT91_REG *) 	0x400E03FC) // (MATRIX) HMATRIX2 VERSION REGISTER 
#define AT91C_MATRIX_SFR13 ((AT91_REG *) 	0x400E0344) // (MATRIX)  Special Function Register 13
#define AT91C_MATRIX_SFR5  ((AT91_REG *) 	0x400E0324) // (MATRIX)  Special Function Register 5
#define AT91C_MATRIX_MCFG0 ((AT91_REG *) 	0x400E0200) // (MATRIX)  Master Configuration Register 0 : ARM I and D
#define AT91C_MATRIX_SCFG6 ((AT91_REG *) 	0x400E0258) // (MATRIX)  Slave Configuration Register 6
#define AT91C_MATRIX_SFR14 ((AT91_REG *) 	0x400E0348) // (MATRIX)  Special Function Register 14
#define AT91C_MATRIX_SFR1  ((AT91_REG *) 	0x400E0314) // (MATRIX)  Special Function Register 1
#define AT91C_MATRIX_SFR15 ((AT91_REG *) 	0x400E034C) // (MATRIX)  Special Function Register 15
#define AT91C_MATRIX_SFR6  ((AT91_REG *) 	0x400E0328) // (MATRIX)  Special Function Register 6
#define AT91C_MATRIX_SFR11 ((AT91_REG *) 	0x400E033C) // (MATRIX)  Special Function Register 11
#define AT91C_MATRIX_IPNAME2 ((AT91_REG *) 	0x400E03F4) // (MATRIX) HMATRIX2 IPNAME2 REGISTER 
#define AT91C_MATRIX_ADDRSIZE ((AT91_REG *) 	0x400E03EC) // (MATRIX) HMATRIX2 ADDRSIZE REGISTER 
#define AT91C_MATRIX_MCFG5 ((AT91_REG *) 	0x400E0214) // (MATRIX)  Master Configuration Register 5
#define AT91C_MATRIX_SFR9  ((AT91_REG *) 	0x400E0334) // (MATRIX)  Special Function Register 9
#define AT91C_MATRIX_MCFG3 ((AT91_REG *) 	0x400E020C) // (MATRIX)  Master Configuration Register 3
#define AT91C_MATRIX_SCFG4 ((AT91_REG *) 	0x400E0250) // (MATRIX)  Slave Configuration Register 4
#define AT91C_MATRIX_MCFG1 ((AT91_REG *) 	0x400E0204) // (MATRIX)  Master Configuration Register 1 : ARM S
#define AT91C_MATRIX_SCFG7 ((AT91_REG *) 	0x400E025C) // (MATRIX)  Slave Configuration Register 5
#define AT91C_MATRIX_SFR10 ((AT91_REG *) 	0x400E0338) // (MATRIX)  Special Function Register 10
#define AT91C_MATRIX_SCFG2 ((AT91_REG *) 	0x400E0248) // (MATRIX)  Slave Configuration Register 2
#define AT91C_MATRIX_SFR7  ((AT91_REG *) 	0x400E032C) // (MATRIX)  Special Function Register 7
#define AT91C_MATRIX_IPNAME1 ((AT91_REG *) 	0x400E03F0) // (MATRIX) HMATRIX2 IPNAME1 REGISTER 
#define AT91C_MATRIX_MCFG4 ((AT91_REG *) 	0x400E0210) // (MATRIX)  Master Configuration Register 4
#define AT91C_MATRIX_SFR0  ((AT91_REG *) 	0x400E0310) // (MATRIX)  Special Function Register 0
#define AT91C_MATRIX_FEATURES ((AT91_REG *) 	0x400E03F8) // (MATRIX) HMATRIX2 FEATURES REGISTER 
#define AT91C_MATRIX_SCFG5 ((AT91_REG *) 	0x400E0254) // (MATRIX)  Slave Configuration Register 5
#define AT91C_MATRIX_MCFG6 ((AT91_REG *) 	0x400E0218) // (MATRIX)  Master Configuration Register 6
#define AT91C_MATRIX_SCFG9 ((AT91_REG *) 	0x400E0264) // (MATRIX)  Slave Configuration Register 9
#define AT91C_MATRIX_SFR4  ((AT91_REG *) 	0x400E0320) // (MATRIX)  Special Function Register 4
// ========== Register definition for NVIC peripheral ========== 
#define AT91C_NVIC_MMAR ((AT91_REG *) 	0xE000ED34) // (NVIC) Mem Manage Address Register
#define AT91C_NVIC_STIR ((AT91_REG *) 	0xE000EF00) // (NVIC) Software Trigger Interrupt Register
#define AT91C_NVIC_MMFR2 ((AT91_REG *) 	0xE000ED58) // (NVIC) Memory Model Feature register2
#define AT91C_NVIC_CPUID ((AT91_REG *) 	0xE000ED00) // (NVIC) CPUID Base Register
#define AT91C_NVIC_DFSR ((AT91_REG *) 	0xE000ED30) // (NVIC) Debug Fault Status Register
#define AT91C_NVIC_HAND4PR ((AT91_REG *) 	0xE000ED18) // (NVIC) System Handlers 4-7 Priority Register
#define AT91C_NVIC_HFSR ((AT91_REG *) 	0xE000ED2C) // (NVIC) Hard Fault Status Register
#define AT91C_NVIC_PID6 ((AT91_REG *) 	0xE000EFD8) // (NVIC) Peripheral identification register
#define AT91C_NVIC_PFR0 ((AT91_REG *) 	0xE000ED40) // (NVIC) Processor Feature register0
#define AT91C_NVIC_VTOFFR ((AT91_REG *) 	0xE000ED08) // (NVIC) Vector Table Offset Register
#define AT91C_NVIC_ISPR ((AT91_REG *) 	0xE000E200) // (NVIC) Set Pending Register
#define AT91C_NVIC_PID0 ((AT91_REG *) 	0xE000EFE0) // (NVIC) Peripheral identification register b7:0
#define AT91C_NVIC_PID7 ((AT91_REG *) 	0xE000EFDC) // (NVIC) Peripheral identification register
#define AT91C_NVIC_STICKRVR ((AT91_REG *) 	0xE000E014) // (NVIC) SysTick Reload Value Register
#define AT91C_NVIC_PID2 ((AT91_REG *) 	0xE000EFE8) // (NVIC) Peripheral identification register b23:16
#define AT91C_NVIC_ISAR0 ((AT91_REG *) 	0xE000ED60) // (NVIC) ISA Feature register0
#define AT91C_NVIC_SCR  ((AT91_REG *) 	0xE000ED10) // (NVIC) System Control Register
#define AT91C_NVIC_PID4 ((AT91_REG *) 	0xE000EFD0) // (NVIC) Peripheral identification register
#define AT91C_NVIC_ISAR2 ((AT91_REG *) 	0xE000ED68) // (NVIC) ISA Feature register2
#define AT91C_NVIC_ISER ((AT91_REG *) 	0xE000E100) // (NVIC) Set Enable Register
#define AT91C_NVIC_IPR  ((AT91_REG *) 	0xE000E400) // (NVIC) Interrupt Mask Register
#define AT91C_NVIC_AIRCR ((AT91_REG *) 	0xE000ED0C) // (NVIC) Application Interrupt/Reset Control Reg
#define AT91C_NVIC_CID2 ((AT91_REG *) 	0xE000EFF8) // (NVIC) Component identification register b23:16
#define AT91C_NVIC_ICPR ((AT91_REG *) 	0xE000E280) // (NVIC) Clear Pending Register
#define AT91C_NVIC_CID3 ((AT91_REG *) 	0xE000EFFC) // (NVIC) Component identification register b31:24
#define AT91C_NVIC_CFSR ((AT91_REG *) 	0xE000ED28) // (NVIC) Configurable Fault Status Register
#define AT91C_NVIC_AFR0 ((AT91_REG *) 	0xE000ED4C) // (NVIC) Auxiliary Feature register0
#define AT91C_NVIC_ICSR ((AT91_REG *) 	0xE000ED04) // (NVIC) Interrupt Control State Register
#define AT91C_NVIC_CCR  ((AT91_REG *) 	0xE000ED14) // (NVIC) Configuration Control Register
#define AT91C_NVIC_CID0 ((AT91_REG *) 	0xE000EFF0) // (NVIC) Component identification register b7:0
#define AT91C_NVIC_ISAR1 ((AT91_REG *) 	0xE000ED64) // (NVIC) ISA Feature register1
#define AT91C_NVIC_STICKCVR ((AT91_REG *) 	0xE000E018) // (NVIC) SysTick Current Value Register
#define AT91C_NVIC_STICKCSR ((AT91_REG *) 	0xE000E010) // (NVIC) SysTick Control and Status Register
#define AT91C_NVIC_CID1 ((AT91_REG *) 	0xE000EFF4) // (NVIC) Component identification register b15:8
#define AT91C_NVIC_DFR0 ((AT91_REG *) 	0xE000ED48) // (NVIC) Debug Feature register0
#define AT91C_NVIC_MMFR3 ((AT91_REG *) 	0xE000ED5C) // (NVIC) Memory Model Feature register3
#define AT91C_NVIC_MMFR0 ((AT91_REG *) 	0xE000ED50) // (NVIC) Memory Model Feature register0
#define AT91C_NVIC_STICKCALVR ((AT91_REG *) 	0xE000E01C) // (NVIC) SysTick Calibration Value Register
#define AT91C_NVIC_PID1 ((AT91_REG *) 	0xE000EFE4) // (NVIC) Peripheral identification register b15:8
#define AT91C_NVIC_HAND12PR ((AT91_REG *) 	0xE000ED20) // (NVIC) System Handlers 12-15 Priority Register
#define AT91C_NVIC_MMFR1 ((AT91_REG *) 	0xE000ED54) // (NVIC) Memory Model Feature register1
#define AT91C_NVIC_AFSR ((AT91_REG *) 	0xE000ED3C) // (NVIC) Auxiliary Fault Status Register
#define AT91C_NVIC_HANDCSR ((AT91_REG *) 	0xE000ED24) // (NVIC) System Handler Control and State Register
#define AT91C_NVIC_ISAR4 ((AT91_REG *) 	0xE000ED70) // (NVIC) ISA Feature register4
#define AT91C_NVIC_ABR  ((AT91_REG *) 	0xE000E300) // (NVIC) Active Bit Register
#define AT91C_NVIC_PFR1 ((AT91_REG *) 	0xE000ED44) // (NVIC) Processor Feature register1
#define AT91C_NVIC_PID5 ((AT91_REG *) 	0xE000EFD4) // (NVIC) Peripheral identification register
#define AT91C_NVIC_ICTR ((AT91_REG *) 	0xE000E004) // (NVIC) Interrupt Control Type Register
#define AT91C_NVIC_ICER ((AT91_REG *) 	0xE000E180) // (NVIC) Clear enable Register
#define AT91C_NVIC_PID3 ((AT91_REG *) 	0xE000EFEC) // (NVIC) Peripheral identification register b31:24
#define AT91C_NVIC_ISAR3 ((AT91_REG *) 	0xE000ED6C) // (NVIC) ISA Feature register3
#define AT91C_NVIC_HAND8PR ((AT91_REG *) 	0xE000ED1C) // (NVIC) System Handlers 8-11 Priority Register
#define AT91C_NVIC_BFAR ((AT91_REG *) 	0xE000ED38) // (NVIC) Bus Fault Address Register
// ========== Register definition for MPU peripheral ========== 
#define AT91C_MPU_REG_BASE_ADDR3 ((AT91_REG *) 	0xE000EDB4) // (MPU) MPU Region Base Address Register alias 3
#define AT91C_MPU_REG_NB ((AT91_REG *) 	0xE000ED98) // (MPU) MPU Region Number Register
#define AT91C_MPU_ATTR_SIZE1 ((AT91_REG *) 	0xE000EDA8) // (MPU) MPU  Attribute and Size Register alias 1
#define AT91C_MPU_REG_BASE_ADDR1 ((AT91_REG *) 	0xE000EDA4) // (MPU) MPU Region Base Address Register alias 1
#define AT91C_MPU_ATTR_SIZE3 ((AT91_REG *) 	0xE000EDB8) // (MPU) MPU  Attribute and Size Register alias 3
#define AT91C_MPU_CTRL  ((AT91_REG *) 	0xE000ED94) // (MPU) MPU Control Register
#define AT91C_MPU_ATTR_SIZE2 ((AT91_REG *) 	0xE000EDB0) // (MPU) MPU  Attribute and Size Register alias 2
#define AT91C_MPU_REG_BASE_ADDR ((AT91_REG *) 	0xE000ED9C) // (MPU) MPU Region Base Address Register
#define AT91C_MPU_REG_BASE_ADDR2 ((AT91_REG *) 	0xE000EDAC) // (MPU) MPU Region Base Address Register alias 2
#define AT91C_MPU_ATTR_SIZE ((AT91_REG *) 	0xE000EDA0) // (MPU) MPU  Attribute and Size Register
#define AT91C_MPU_TYPE  ((AT91_REG *) 	0xE000ED90) // (MPU) MPU Type Register
// ========== Register definition for CM3 peripheral ========== 
#define AT91C_CM3_SHCSR ((AT91_REG *) 	0xE000ED24) // (CM3) System Handler Control and State Register
#define AT91C_CM3_CCR   ((AT91_REG *) 	0xE000ED14) // (CM3) Configuration Control Register
#define AT91C_CM3_ICSR  ((AT91_REG *) 	0xE000ED04) // (CM3) Interrupt Control State Register
#define AT91C_CM3_CPUID ((AT91_REG *) 	0xE000ED00) // (CM3) CPU ID Base Register
#define AT91C_CM3_SCR   ((AT91_REG *) 	0xE000ED10) // (CM3) System Controller Register
#define AT91C_CM3_AIRCR ((AT91_REG *) 	0xE000ED0C) // (CM3) Application Interrupt and Reset Control Register
#define AT91C_CM3_SHPR  ((AT91_REG *) 	0xE000ED18) // (CM3) System Handler Priority Register
#define AT91C_CM3_VTOR  ((AT91_REG *) 	0xE000ED08) // (CM3) Vector Table Offset Register
// ========== Register definition for PDC_DBGU peripheral ========== 
#define AT91C_DBGU_TPR  ((AT91_REG *) 	0x400E0708) // (PDC_DBGU) Transmit Pointer Register
#define AT91C_DBGU_PTCR ((AT91_REG *) 	0x400E0720) // (PDC_DBGU) PDC Transfer Control Register
#define AT91C_DBGU_TNCR ((AT91_REG *) 	0x400E071C) // (PDC_DBGU) Transmit Next Counter Register
#define AT91C_DBGU_PTSR ((AT91_REG *) 	0x400E0724) // (PDC_DBGU) PDC Transfer Status Register
#define AT91C_DBGU_RNCR ((AT91_REG *) 	0x400E0714) // (PDC_DBGU) Receive Next Counter Register
#define AT91C_DBGU_RPR  ((AT91_REG *) 	0x400E0700) // (PDC_DBGU) Receive Pointer Register
#define AT91C_DBGU_TCR  ((AT91_REG *) 	0x400E070C) // (PDC_DBGU) Transmit Counter Register
#define AT91C_DBGU_RNPR ((AT91_REG *) 	0x400E0710) // (PDC_DBGU) Receive Next Pointer Register
#define AT91C_DBGU_TNPR ((AT91_REG *) 	0x400E0718) // (PDC_DBGU) Transmit Next Pointer Register
#define AT91C_DBGU_RCR  ((AT91_REG *) 	0x400E0704) // (PDC_DBGU) Receive Counter Register
// ========== Register definition for DBGU peripheral ========== 
#define AT91C_DBGU_CR   ((AT91_REG *) 	0x400E0600) // (DBGU) Control Register
#define AT91C_DBGU_IDR  ((AT91_REG *) 	0x400E060C) // (DBGU) Interrupt Disable Register
#define AT91C_DBGU_CIDR ((AT91_REG *) 	0x400E0740) // (DBGU) Chip ID Register
#define AT91C_DBGU_IPNAME2 ((AT91_REG *) 	0x400E06F4) // (DBGU) DBGU IPNAME2 REGISTER 
#define AT91C_DBGU_FEATURES ((AT91_REG *) 	0x400E06F8) // (DBGU) DBGU FEATURES REGISTER 
#define AT91C_DBGU_FNTR ((AT91_REG *) 	0x400E0648) // (DBGU) Force NTRST Register
#define AT91C_DBGU_RHR  ((AT91_REG *) 	0x400E0618) // (DBGU) Receiver Holding Register
#define AT91C_DBGU_THR  ((AT91_REG *) 	0x400E061C) // (DBGU) Transmitter Holding Register
#define AT91C_DBGU_ADDRSIZE ((AT91_REG *) 	0x400E06EC) // (DBGU) DBGU ADDRSIZE REGISTER 
#define AT91C_DBGU_MR   ((AT91_REG *) 	0x400E0604) // (DBGU) Mode Register
#define AT91C_DBGU_IER  ((AT91_REG *) 	0x400E0608) // (DBGU) Interrupt Enable Register
#define AT91C_DBGU_BRGR ((AT91_REG *) 	0x400E0620) // (DBGU) Baud Rate Generator Register
#define AT91C_DBGU_CSR  ((AT91_REG *) 	0x400E0614) // (DBGU) Channel Status Register
#define AT91C_DBGU_VER  ((AT91_REG *) 	0x400E06FC) // (DBGU) DBGU VERSION REGISTER 
#define AT91C_DBGU_IMR  ((AT91_REG *) 	0x400E0610) // (DBGU) Interrupt Mask Register
#define AT91C_DBGU_IPNAME1 ((AT91_REG *) 	0x400E06F0) // (DBGU) DBGU IPNAME1 REGISTER 
#define AT91C_DBGU_EXID ((AT91_REG *) 	0x400E0744) // (DBGU) Chip ID Extension Register
// ========== Register definition for PIOA peripheral ========== 
#define AT91C_PIOA_PDR  ((AT91_REG *) 	0x400E0C04) // (PIOA) PIO Disable Register
#define AT91C_PIOA_FRLHSR ((AT91_REG *) 	0x400E0CD8) // (PIOA) Fall/Rise - Low/High Status Register
#define AT91C_PIOA_KIMR ((AT91_REG *) 	0x400E0D38) // (PIOA) Keypad Controller Interrupt Mask Register
#define AT91C_PIOA_LSR  ((AT91_REG *) 	0x400E0CC4) // (PIOA) Level Select Register
#define AT91C_PIOA_IFSR ((AT91_REG *) 	0x400E0C28) // (PIOA) Input Filter Status Register
#define AT91C_PIOA_KKRR ((AT91_REG *) 	0x400E0D44) // (PIOA) Keypad Controller Key Release Register
#define AT91C_PIOA_ODR  ((AT91_REG *) 	0x400E0C14) // (PIOA) Output Disable Registerr
#define AT91C_PIOA_SCIFSR ((AT91_REG *) 	0x400E0C80) // (PIOA) System Clock Glitch Input Filter Select Register
#define AT91C_PIOA_PER  ((AT91_REG *) 	0x400E0C00) // (PIOA) PIO Enable Register
#define AT91C_PIOA_VER  ((AT91_REG *) 	0x400E0CFC) // (PIOA) PIO VERSION REGISTER 
#define AT91C_PIOA_OWSR ((AT91_REG *) 	0x400E0CA8) // (PIOA) Output Write Status Register
#define AT91C_PIOA_KSR  ((AT91_REG *) 	0x400E0D3C) // (PIOA) Keypad Controller Status Register
#define AT91C_PIOA_IMR  ((AT91_REG *) 	0x400E0C48) // (PIOA) Interrupt Mask Register
#define AT91C_PIOA_OWDR ((AT91_REG *) 	0x400E0CA4) // (PIOA) Output Write Disable Register
#define AT91C_PIOA_MDSR ((AT91_REG *) 	0x400E0C58) // (PIOA) Multi-driver Status Register
#define AT91C_PIOA_IFDR ((AT91_REG *) 	0x400E0C24) // (PIOA) Input Filter Disable Register
#define AT91C_PIOA_AIMDR ((AT91_REG *) 	0x400E0CB4) // (PIOA) Additional Interrupt Modes Disables Register
#define AT91C_PIOA_CODR ((AT91_REG *) 	0x400E0C34) // (PIOA) Clear Output Data Register
#define AT91C_PIOA_SCDR ((AT91_REG *) 	0x400E0C8C) // (PIOA) Slow Clock Divider Debouncing Register
#define AT91C_PIOA_KIER ((AT91_REG *) 	0x400E0D30) // (PIOA) Keypad Controller Interrupt Enable Register
#define AT91C_PIOA_REHLSR ((AT91_REG *) 	0x400E0CD4) // (PIOA) Rising Edge/ High Level Select Register
#define AT91C_PIOA_ISR  ((AT91_REG *) 	0x400E0C4C) // (PIOA) Interrupt Status Register
#define AT91C_PIOA_ESR  ((AT91_REG *) 	0x400E0CC0) // (PIOA) Edge Select Register
#define AT91C_PIOA_PPUDR ((AT91_REG *) 	0x400E0C60) // (PIOA) Pull-up Disable Register
#define AT91C_PIOA_MDDR ((AT91_REG *) 	0x400E0C54) // (PIOA) Multi-driver Disable Register
#define AT91C_PIOA_PSR  ((AT91_REG *) 	0x400E0C08) // (PIOA) PIO Status Register
#define AT91C_PIOA_PDSR ((AT91_REG *) 	0x400E0C3C) // (PIOA) Pin Data Status Register
#define AT91C_PIOA_IFDGSR ((AT91_REG *) 	0x400E0C88) // (PIOA) Glitch or Debouncing Input Filter Clock Selection Status Register
#define AT91C_PIOA_FELLSR ((AT91_REG *) 	0x400E0CD0) // (PIOA) Falling Edge/Low Level Select Register
#define AT91C_PIOA_PPUSR ((AT91_REG *) 	0x400E0C68) // (PIOA) Pull-up Status Register
#define AT91C_PIOA_OER  ((AT91_REG *) 	0x400E0C10) // (PIOA) Output Enable Register
#define AT91C_PIOA_OSR  ((AT91_REG *) 	0x400E0C18) // (PIOA) Output Status Register
#define AT91C_PIOA_KKPR ((AT91_REG *) 	0x400E0D40) // (PIOA) Keypad Controller Key Press Register
#define AT91C_PIOA_AIMMR ((AT91_REG *) 	0x400E0CB8) // (PIOA) Additional Interrupt Modes Mask Register
#define AT91C_PIOA_KRCR ((AT91_REG *) 	0x400E0D24) // (PIOA) Keypad Controller Row Column Register
#define AT91C_PIOA_IER  ((AT91_REG *) 	0x400E0C40) // (PIOA) Interrupt Enable Register
#define AT91C_PIOA_KER  ((AT91_REG *) 	0x400E0D20) // (PIOA) Keypad Controller Enable Register
#define AT91C_PIOA_PPUER ((AT91_REG *) 	0x400E0C64) // (PIOA) Pull-up Enable Register
#define AT91C_PIOA_KIDR ((AT91_REG *) 	0x400E0D34) // (PIOA) Keypad Controller Interrupt Disable Register
#define AT91C_PIOA_ABSR ((AT91_REG *) 	0x400E0C70) // (PIOA) Peripheral AB Select Register
#define AT91C_PIOA_LOCKSR ((AT91_REG *) 	0x400E0CE0) // (PIOA) Lock Status Register
#define AT91C_PIOA_DIFSR ((AT91_REG *) 	0x400E0C84) // (PIOA) Debouncing Input Filter Select Register
#define AT91C_PIOA_MDER ((AT91_REG *) 	0x400E0C50) // (PIOA) Multi-driver Enable Register
#define AT91C_PIOA_AIMER ((AT91_REG *) 	0x400E0CB0) // (PIOA) Additional Interrupt Modes Enable Register
#define AT91C_PIOA_ELSR ((AT91_REG *) 	0x400E0CC8) // (PIOA) Edge/Level Status Register
#define AT91C_PIOA_IFER ((AT91_REG *) 	0x400E0C20) // (PIOA) Input Filter Enable Register
#define AT91C_PIOA_KDR  ((AT91_REG *) 	0x400E0D28) // (PIOA) Keypad Controller Debouncing Register
#define AT91C_PIOA_IDR  ((AT91_REG *) 	0x400E0C44) // (PIOA) Interrupt Disable Register
#define AT91C_PIOA_OWER ((AT91_REG *) 	0x400E0CA0) // (PIOA) Output Write Enable Register
#define AT91C_PIOA_ODSR ((AT91_REG *) 	0x400E0C38) // (PIOA) Output Data Status Register
#define AT91C_PIOA_SODR ((AT91_REG *) 	0x400E0C30) // (PIOA) Set Output Data Register
// ========== Register definition for PIOB peripheral ========== 
#define AT91C_PIOB_KIDR ((AT91_REG *) 	0x400E0F34) // (PIOB) Keypad Controller Interrupt Disable Register
#define AT91C_PIOB_OWSR ((AT91_REG *) 	0x400E0EA8) // (PIOB) Output Write Status Register
#define AT91C_PIOB_PSR  ((AT91_REG *) 	0x400E0E08) // (PIOB) PIO Status Register
#define AT91C_PIOB_MDER ((AT91_REG *) 	0x400E0E50) // (PIOB) Multi-driver Enable Register
#define AT91C_PIOB_ODR  ((AT91_REG *) 	0x400E0E14) // (PIOB) Output Disable Registerr
#define AT91C_PIOB_IDR  ((AT91_REG *) 	0x400E0E44) // (PIOB) Interrupt Disable Register
#define AT91C_PIOB_AIMER ((AT91_REG *) 	0x400E0EB0) // (PIOB) Additional Interrupt Modes Enable Register
#define AT91C_PIOB_DIFSR ((AT91_REG *) 	0x400E0E84) // (PIOB) Debouncing Input Filter Select Register
#define AT91C_PIOB_PDR  ((AT91_REG *) 	0x400E0E04) // (PIOB) PIO Disable Register
#define AT91C_PIOB_REHLSR ((AT91_REG *) 	0x400E0ED4) // (PIOB) Rising Edge/ High Level Select Register
#define AT91C_PIOB_PDSR ((AT91_REG *) 	0x400E0E3C) // (PIOB) Pin Data Status Register
#define AT91C_PIOB_PPUDR ((AT91_REG *) 	0x400E0E60) // (PIOB) Pull-up Disable Register
#define AT91C_PIOB_LSR  ((AT91_REG *) 	0x400E0EC4) // (PIOB) Level Select Register
#define AT91C_PIOB_OWDR ((AT91_REG *) 	0x400E0EA4) // (PIOB) Output Write Disable Register
#define AT91C_PIOB_FELLSR ((AT91_REG *) 	0x400E0ED0) // (PIOB) Falling Edge/Low Level Select Register
#define AT91C_PIOB_IFER ((AT91_REG *) 	0x400E0E20) // (PIOB) Input Filter Enable Register
#define AT91C_PIOB_ABSR ((AT91_REG *) 	0x400E0E70) // (PIOB) Peripheral AB Select Register
#define AT91C_PIOB_KIMR ((AT91_REG *) 	0x400E0F38) // (PIOB) Keypad Controller Interrupt Mask Register
#define AT91C_PIOB_KKPR ((AT91_REG *) 	0x400E0F40) // (PIOB) Keypad Controller Key Press Register
#define AT91C_PIOB_FRLHSR ((AT91_REG *) 	0x400E0ED8) // (PIOB) Fall/Rise - Low/High Status Register
#define AT91C_PIOB_AIMDR ((AT91_REG *) 	0x400E0EB4) // (PIOB) Additional Interrupt Modes Disables Register
#define AT91C_PIOB_SCIFSR ((AT91_REG *) 	0x400E0E80) // (PIOB) System Clock Glitch Input Filter Select Register
#define AT91C_PIOB_VER  ((AT91_REG *) 	0x400E0EFC) // (PIOB) PIO VERSION REGISTER 
#define AT91C_PIOB_PER  ((AT91_REG *) 	0x400E0E00) // (PIOB) PIO Enable Register
#define AT91C_PIOB_ELSR ((AT91_REG *) 	0x400E0EC8) // (PIOB) Edge/Level Status Register
#define AT91C_PIOB_IMR  ((AT91_REG *) 	0x400E0E48) // (PIOB) Interrupt Mask Register
#define AT91C_PIOB_PPUSR ((AT91_REG *) 	0x400E0E68) // (PIOB) Pull-up Status Register
#define AT91C_PIOB_SCDR ((AT91_REG *) 	0x400E0E8C) // (PIOB) Slow Clock Divider Debouncing Register
#define AT91C_PIOB_KSR  ((AT91_REG *) 	0x400E0F3C) // (PIOB) Keypad Controller Status Register
#define AT91C_PIOB_IFDGSR ((AT91_REG *) 	0x400E0E88) // (PIOB) Glitch or Debouncing Input Filter Clock Selection Status Register
#define AT91C_PIOB_ESR  ((AT91_REG *) 	0x400E0EC0) // (PIOB) Edge Select Register
#define AT91C_PIOB_ODSR ((AT91_REG *) 	0x400E0E38) // (PIOB) Output Data Status Register
#define AT91C_PIOB_IFDR ((AT91_REG *) 	0x400E0E24) // (PIOB) Input Filter Disable Register
#define AT91C_PIOB_SODR ((AT91_REG *) 	0x400E0E30) // (PIOB) Set Output Data Register
#define AT91C_PIOB_IER  ((AT91_REG *) 	0x400E0E40) // (PIOB) Interrupt Enable Register
#define AT91C_PIOB_MDSR ((AT91_REG *) 	0x400E0E58) // (PIOB) Multi-driver Status Register
#define AT91C_PIOB_ISR  ((AT91_REG *) 	0x400E0E4C) // (PIOB) Interrupt Status Register
#define AT91C_PIOB_IFSR ((AT91_REG *) 	0x400E0E28) // (PIOB) Input Filter Status Register
#define AT91C_PIOB_KER  ((AT91_REG *) 	0x400E0F20) // (PIOB) Keypad Controller Enable Register
#define AT91C_PIOB_KKRR ((AT91_REG *) 	0x400E0F44) // (PIOB) Keypad Controller Key Release Register
#define AT91C_PIOB_PPUER ((AT91_REG *) 	0x400E0E64) // (PIOB) Pull-up Enable Register
#define AT91C_PIOB_LOCKSR ((AT91_REG *) 	0x400E0EE0) // (PIOB) Lock Status Register
#define AT91C_PIOB_OWER ((AT91_REG *) 	0x400E0EA0) // (PIOB) Output Write Enable Register
#define AT91C_PIOB_KIER ((AT91_REG *) 	0x400E0F30) // (PIOB) Keypad Controller Interrupt Enable Register
#define AT91C_PIOB_MDDR ((AT91_REG *) 	0x400E0E54) // (PIOB) Multi-driver Disable Register
#define AT91C_PIOB_KRCR ((AT91_REG *) 	0x400E0F24) // (PIOB) Keypad Controller Row Column Register
#define AT91C_PIOB_CODR ((AT91_REG *) 	0x400E0E34) // (PIOB) Clear Output Data Register
#define AT91C_PIOB_KDR  ((AT91_REG *) 	0x400E0F28) // (PIOB) Keypad Controller Debouncing Register
#define AT91C_PIOB_AIMMR ((AT91_REG *) 	0x400E0EB8) // (PIOB) Additional Interrupt Modes Mask Register
#define AT91C_PIOB_OER  ((AT91_REG *) 	0x400E0E10) // (PIOB) Output Enable Register
#define AT91C_PIOB_OSR  ((AT91_REG *) 	0x400E0E18) // (PIOB) Output Status Register
// ========== Register definition for PIOC peripheral ========== 
#define AT91C_PIOC_FELLSR ((AT91_REG *) 	0x400E10D0) // (PIOC) Falling Edge/Low Level Select Register
#define AT91C_PIOC_FRLHSR ((AT91_REG *) 	0x400E10D8) // (PIOC) Fall/Rise - Low/High Status Register
#define AT91C_PIOC_MDDR ((AT91_REG *) 	0x400E1054) // (PIOC) Multi-driver Disable Register
#define AT91C_PIOC_IFDGSR ((AT91_REG *) 	0x400E1088) // (PIOC) Glitch or Debouncing Input Filter Clock Selection Status Register
#define AT91C_PIOC_ABSR ((AT91_REG *) 	0x400E1070) // (PIOC) Peripheral AB Select Register
#define AT91C_PIOC_KIMR ((AT91_REG *) 	0x400E1138) // (PIOC) Keypad Controller Interrupt Mask Register
#define AT91C_PIOC_KRCR ((AT91_REG *) 	0x400E1124) // (PIOC) Keypad Controller Row Column Register
#define AT91C_PIOC_ODSR ((AT91_REG *) 	0x400E1038) // (PIOC) Output Data Status Register
#define AT91C_PIOC_OSR  ((AT91_REG *) 	0x400E1018) // (PIOC) Output Status Register
#define AT91C_PIOC_IFER ((AT91_REG *) 	0x400E1020) // (PIOC) Input Filter Enable Register
#define AT91C_PIOC_KKPR ((AT91_REG *) 	0x400E1140) // (PIOC) Keypad Controller Key Press Register
#define AT91C_PIOC_MDSR ((AT91_REG *) 	0x400E1058) // (PIOC) Multi-driver Status Register
#define AT91C_PIOC_IFDR ((AT91_REG *) 	0x400E1024) // (PIOC) Input Filter Disable Register
#define AT91C_PIOC_MDER ((AT91_REG *) 	0x400E1050) // (PIOC) Multi-driver Enable Register
#define AT91C_PIOC_SCDR ((AT91_REG *) 	0x400E108C) // (PIOC) Slow Clock Divider Debouncing Register
#define AT91C_PIOC_SCIFSR ((AT91_REG *) 	0x400E1080) // (PIOC) System Clock Glitch Input Filter Select Register
#define AT91C_PIOC_IER  ((AT91_REG *) 	0x400E1040) // (PIOC) Interrupt Enable Register
#define AT91C_PIOC_KDR  ((AT91_REG *) 	0x400E1128) // (PIOC) Keypad Controller Debouncing Register
#define AT91C_PIOC_OWDR ((AT91_REG *) 	0x400E10A4) // (PIOC) Output Write Disable Register
#define AT91C_PIOC_IFSR ((AT91_REG *) 	0x400E1028) // (PIOC) Input Filter Status Register
#define AT91C_PIOC_ISR  ((AT91_REG *) 	0x400E104C) // (PIOC) Interrupt Status Register
#define AT91C_PIOC_PPUDR ((AT91_REG *) 	0x400E1060) // (PIOC) Pull-up Disable Register
#define AT91C_PIOC_PDSR ((AT91_REG *) 	0x400E103C) // (PIOC) Pin Data Status Register
#define AT91C_PIOC_KKRR ((AT91_REG *) 	0x400E1144) // (PIOC) Keypad Controller Key Release Register
#define AT91C_PIOC_AIMDR ((AT91_REG *) 	0x400E10B4) // (PIOC) Additional Interrupt Modes Disables Register
#define AT91C_PIOC_LSR  ((AT91_REG *) 	0x400E10C4) // (PIOC) Level Select Register
#define AT91C_PIOC_PPUER ((AT91_REG *) 	0x400E1064) // (PIOC) Pull-up Enable Register
#define AT91C_PIOC_AIMER ((AT91_REG *) 	0x400E10B0) // (PIOC) Additional Interrupt Modes Enable Register
#define AT91C_PIOC_OER  ((AT91_REG *) 	0x400E1010) // (PIOC) Output Enable Register
#define AT91C_PIOC_CODR ((AT91_REG *) 	0x400E1034) // (PIOC) Clear Output Data Register
#define AT91C_PIOC_AIMMR ((AT91_REG *) 	0x400E10B8) // (PIOC) Additional Interrupt Modes Mask Register
#define AT91C_PIOC_OWER ((AT91_REG *) 	0x400E10A0) // (PIOC) Output Write Enable Register
#define AT91C_PIOC_VER  ((AT91_REG *) 	0x400E10FC) // (PIOC) PIO VERSION REGISTER 
#define AT91C_PIOC_IMR  ((AT91_REG *) 	0x400E1048) // (PIOC) Interrupt Mask Register
#define AT91C_PIOC_PPUSR ((AT91_REG *) 	0x400E1068) // (PIOC) Pull-up Status Register
#define AT91C_PIOC_IDR  ((AT91_REG *) 	0x400E1044) // (PIOC) Interrupt Disable Register
#define AT91C_PIOC_DIFSR ((AT91_REG *) 	0x400E1084) // (PIOC) Debouncing Input Filter Select Register
#define AT91C_PIOC_KIDR ((AT91_REG *) 	0x400E1134) // (PIOC) Keypad Controller Interrupt Disable Register
#define AT91C_PIOC_KSR  ((AT91_REG *) 	0x400E113C) // (PIOC) Keypad Controller Status Register
#define AT91C_PIOC_REHLSR ((AT91_REG *) 	0x400E10D4) // (PIOC) Rising Edge/ High Level Select Register
#define AT91C_PIOC_ESR  ((AT91_REG *) 	0x400E10C0) // (PIOC) Edge Select Register
#define AT91C_PIOC_KIER ((AT91_REG *) 	0x400E1130) // (PIOC) Keypad Controller Interrupt Enable Register
#define AT91C_PIOC_ELSR ((AT91_REG *) 	0x400E10C8) // (PIOC) Edge/Level Status Register
#define AT91C_PIOC_SODR ((AT91_REG *) 	0x400E1030) // (PIOC) Set Output Data Register
#define AT91C_PIOC_PSR  ((AT91_REG *) 	0x400E1008) // (PIOC) PIO Status Register
#define AT91C_PIOC_KER  ((AT91_REG *) 	0x400E1120) // (PIOC) Keypad Controller Enable Register
#define AT91C_PIOC_ODR  ((AT91_REG *) 	0x400E1014) // (PIOC) Output Disable Registerr
#define AT91C_PIOC_OWSR ((AT91_REG *) 	0x400E10A8) // (PIOC) Output Write Status Register
#define AT91C_PIOC_PDR  ((AT91_REG *) 	0x400E1004) // (PIOC) PIO Disable Register
#define AT91C_PIOC_LOCKSR ((AT91_REG *) 	0x400E10E0) // (PIOC) Lock Status Register
#define AT91C_PIOC_PER  ((AT91_REG *) 	0x400E1000) // (PIOC) PIO Enable Register
// ========== Register definition for PMC peripheral ========== 
#define AT91C_PMC_PLLAR ((AT91_REG *) 	0x400E0428) // (PMC) PLL Register
#define AT91C_PMC_UCKR  ((AT91_REG *) 	0x400E041C) // (PMC) UTMI Clock Configuration Register
#define AT91C_PMC_FSMR  ((AT91_REG *) 	0x400E0470) // (PMC) Fast Startup Mode Register
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0x400E0430) // (PMC) Master Clock Register
#define AT91C_PMC_SCER  ((AT91_REG *) 	0x400E0400) // (PMC) System Clock Enable Register
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0x400E0418) // (PMC) Peripheral Clock Status Register
#define AT91C_PMC_MCFR  ((AT91_REG *) 	0x400E0424) // (PMC) Main Clock  Frequency Register
#define AT91C_PMC_FOCR  ((AT91_REG *) 	0x400E0478) // (PMC) Fault Output Clear Register
#define AT91C_PMC_FSPR  ((AT91_REG *) 	0x400E0474) // (PMC) Fast Startup Polarity Register
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0x400E0408) // (PMC) System Clock Status Register
#define AT91C_PMC_IDR   ((AT91_REG *) 	0x400E0464) // (PMC) Interrupt Disable Register
#define AT91C_PMC_VER   ((AT91_REG *) 	0x400E04FC) // (PMC) APMC VERSION REGISTER
#define AT91C_PMC_IMR   ((AT91_REG *) 	0x400E046C) // (PMC) Interrupt Mask Register
#define AT91C_PMC_IPNAME2 ((AT91_REG *) 	0x400E04F4) // (PMC) PMC IPNAME2 REGISTER 
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0x400E0404) // (PMC) System Clock Disable Register
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0x400E0440) // (PMC) Programmable Clock Register
#define AT91C_PMC_ADDRSIZE ((AT91_REG *) 	0x400E04EC) // (PMC) PMC ADDRSIZE REGISTER 
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0x400E0414) // (PMC) Peripheral Clock Disable Register
#define AT91C_PMC_MOR   ((AT91_REG *) 	0x400E0420) // (PMC) Main Oscillator Register
#define AT91C_PMC_SR    ((AT91_REG *) 	0x400E0468) // (PMC) Status Register
#define AT91C_PMC_IER   ((AT91_REG *) 	0x400E0460) // (PMC) Interrupt Enable Register
#define AT91C_PMC_IPNAME1 ((AT91_REG *) 	0x400E04F0) // (PMC) PMC IPNAME1 REGISTER 
#define AT91C_PMC_PCER  ((AT91_REG *) 	0x400E0410) // (PMC) Peripheral Clock Enable Register
#define AT91C_PMC_FEATURES ((AT91_REG *) 	0x400E04F8) // (PMC) PMC FEATURES REGISTER 
// ========== Register definition for CKGR peripheral ========== 
#define AT91C_CKGR_PLLAR ((AT91_REG *) 	0x400E0428) // (CKGR) PLL Register
#define AT91C_CKGR_UCKR ((AT91_REG *) 	0x400E041C) // (CKGR) UTMI Clock Configuration Register
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0x400E0420) // (CKGR) Main Oscillator Register
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0x400E0424) // (CKGR) Main Clock  Frequency Register
// ========== Register definition for RSTC peripheral ========== 
#define AT91C_RSTC_VER  ((AT91_REG *) 	0x400E12FC) // (RSTC) Version Register
#define AT91C_RSTC_RCR  ((AT91_REG *) 	0x400E1200) // (RSTC) Reset Control Register
#define AT91C_RSTC_RMR  ((AT91_REG *) 	0x400E1208) // (RSTC) Reset Mode Register
#define AT91C_RSTC_RSR  ((AT91_REG *) 	0x400E1204) // (RSTC) Reset Status Register
// ========== Register definition for SUPC peripheral ========== 
#define AT91C_SUPC_WUIR ((AT91_REG *) 	0x400E1220) // (SUPC) Wake Up Inputs Register
#define AT91C_SUPC_CR   ((AT91_REG *) 	0x400E1210) // (SUPC) Control Register
#define AT91C_SUPC_MR   ((AT91_REG *) 	0x400E1218) // (SUPC) Mode Register
#define AT91C_SUPC_FWUTR ((AT91_REG *) 	0x400E1228) // (SUPC) Flash Wake-up Timer Register
#define AT91C_SUPC_SR   ((AT91_REG *) 	0x400E1224) // (SUPC) Status Register
#define AT91C_SUPC_WUMR ((AT91_REG *) 	0x400E121C) // (SUPC) Wake Up Mode Register
#define AT91C_SUPC_BOMR ((AT91_REG *) 	0x400E1214) // (SUPC) Brown Out Mode Register
// ========== Register definition for RTTC peripheral ========== 
#define AT91C_RTTC_RTVR ((AT91_REG *) 	0x400E1238) // (RTTC) Real-time Value Register
#define AT91C_RTTC_RTAR ((AT91_REG *) 	0x400E1234) // (RTTC) Real-time Alarm Register
#define AT91C_RTTC_RTMR ((AT91_REG *) 	0x400E1230) // (RTTC) Real-time Mode Register
#define AT91C_RTTC_RTSR ((AT91_REG *) 	0x400E123C) // (RTTC) Real-time Status Register
// ========== Register definition for WDTC peripheral ========== 
#define AT91C_WDTC_WDSR ((AT91_REG *) 	0x400E1258) // (WDTC) Watchdog Status Register
#define AT91C_WDTC_WDMR ((AT91_REG *) 	0x400E1254) // (WDTC) Watchdog Mode Register
#define AT91C_WDTC_WDCR ((AT91_REG *) 	0x400E1250) // (WDTC) Watchdog Control Register
// ========== Register definition for RTC peripheral ========== 
#define AT91C_RTC_IMR   ((AT91_REG *) 	0x400E1288) // (RTC) Interrupt Mask Register
#define AT91C_RTC_SCCR  ((AT91_REG *) 	0x400E127C) // (RTC) Status Clear Command Register
#define AT91C_RTC_CALR  ((AT91_REG *) 	0x400E126C) // (RTC) Calendar Register
#define AT91C_RTC_MR    ((AT91_REG *) 	0x400E1264) // (RTC) Mode Register
#define AT91C_RTC_TIMR  ((AT91_REG *) 	0x400E1268) // (RTC) Time Register
#define AT91C_RTC_CALALR ((AT91_REG *) 	0x400E1274) // (RTC) Calendar Alarm Register
#define AT91C_RTC_VER   ((AT91_REG *) 	0x400E128C) // (RTC) Valid Entry Register
#define AT91C_RTC_CR    ((AT91_REG *) 	0x400E1260) // (RTC) Control Register
#define AT91C_RTC_IDR   ((AT91_REG *) 	0x400E1284) // (RTC) Interrupt Disable Register
#define AT91C_RTC_TIMALR ((AT91_REG *) 	0x400E1270) // (RTC) Time Alarm Register
#define AT91C_RTC_IER   ((AT91_REG *) 	0x400E1280) // (RTC) Interrupt Enable Register
#define AT91C_RTC_SR    ((AT91_REG *) 	0x400E1278) // (RTC) Status Register
// ========== Register definition for ADC0 peripheral ========== 
#define AT91C_ADC0_IPNAME2 ((AT91_REG *) 	0x400AC0F4) // (ADC0) ADC IPNAME2 REGISTER 
#define AT91C_ADC0_ADDRSIZE ((AT91_REG *) 	0x400AC0EC) // (ADC0) ADC ADDRSIZE REGISTER 
#define AT91C_ADC0_IDR  ((AT91_REG *) 	0x400AC028) // (ADC0) ADC Interrupt Disable Register
#define AT91C_ADC0_CHSR ((AT91_REG *) 	0x400AC018) // (ADC0) ADC Channel Status Register
#define AT91C_ADC0_FEATURES ((AT91_REG *) 	0x400AC0F8) // (ADC0) ADC FEATURES REGISTER 
#define AT91C_ADC0_CDR0 ((AT91_REG *) 	0x400AC030) // (ADC0) ADC Channel Data Register 0
#define AT91C_ADC0_LCDR ((AT91_REG *) 	0x400AC020) // (ADC0) ADC Last Converted Data Register
#define AT91C_ADC0_EMR  ((AT91_REG *) 	0x400AC068) // (ADC0) Extended Mode Register
#define AT91C_ADC0_CDR3 ((AT91_REG *) 	0x400AC03C) // (ADC0) ADC Channel Data Register 3
#define AT91C_ADC0_CDR7 ((AT91_REG *) 	0x400AC04C) // (ADC0) ADC Channel Data Register 7
#define AT91C_ADC0_SR   ((AT91_REG *) 	0x400AC01C) // (ADC0) ADC Status Register
#define AT91C_ADC0_ACR  ((AT91_REG *) 	0x400AC064) // (ADC0) Analog Control Register
#define AT91C_ADC0_CDR5 ((AT91_REG *) 	0x400AC044) // (ADC0) ADC Channel Data Register 5
#define AT91C_ADC0_IPNAME1 ((AT91_REG *) 	0x400AC0F0) // (ADC0) ADC IPNAME1 REGISTER 
#define AT91C_ADC0_CDR6 ((AT91_REG *) 	0x400AC048) // (ADC0) ADC Channel Data Register 6
#define AT91C_ADC0_MR   ((AT91_REG *) 	0x400AC004) // (ADC0) ADC Mode Register
#define AT91C_ADC0_CDR1 ((AT91_REG *) 	0x400AC034) // (ADC0) ADC Channel Data Register 1
#define AT91C_ADC0_CDR2 ((AT91_REG *) 	0x400AC038) // (ADC0) ADC Channel Data Register 2
#define AT91C_ADC0_CDR4 ((AT91_REG *) 	0x400AC040) // (ADC0) ADC Channel Data Register 4
#define AT91C_ADC0_CHER ((AT91_REG *) 	0x400AC010) // (ADC0) ADC Channel Enable Register
#define AT91C_ADC0_VER  ((AT91_REG *) 	0x400AC0FC) // (ADC0) ADC VERSION REGISTER
#define AT91C_ADC0_CHDR ((AT91_REG *) 	0x400AC014) // (ADC0) ADC Channel Disable Register
#define AT91C_ADC0_CR   ((AT91_REG *) 	0x400AC000) // (ADC0) ADC Control Register
#define AT91C_ADC0_IMR  ((AT91_REG *) 	0x400AC02C) // (ADC0) ADC Interrupt Mask Register
#define AT91C_ADC0_IER  ((AT91_REG *) 	0x400AC024) // (ADC0) ADC Interrupt Enable Register
// ========== Register definition for TC0 peripheral ========== 
#define AT91C_TC0_IER   ((AT91_REG *) 	0x40080024) // (TC0) Interrupt Enable Register
#define AT91C_TC0_CV    ((AT91_REG *) 	0x40080010) // (TC0) Counter Value
#define AT91C_TC0_RA    ((AT91_REG *) 	0x40080014) // (TC0) Register A
#define AT91C_TC0_RB    ((AT91_REG *) 	0x40080018) // (TC0) Register B
#define AT91C_TC0_IDR   ((AT91_REG *) 	0x40080028) // (TC0) Interrupt Disable Register
#define AT91C_TC0_SR    ((AT91_REG *) 	0x40080020) // (TC0) Status Register
#define AT91C_TC0_IMR   ((AT91_REG *) 	0x4008002C) // (TC0) Interrupt Mask Register
#define AT91C_TC0_CMR   ((AT91_REG *) 	0x40080004) // (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC0_RC    ((AT91_REG *) 	0x4008001C) // (TC0) Register C
#define AT91C_TC0_CCR   ((AT91_REG *) 	0x40080000) // (TC0) Channel Control Register
// ========== Register definition for TC1 peripheral ========== 
#define AT91C_TC1_SR    ((AT91_REG *) 	0x40080060) // (TC1) Status Register
#define AT91C_TC1_RA    ((AT91_REG *) 	0x40080054) // (TC1) Register A
#define AT91C_TC1_IER   ((AT91_REG *) 	0x40080064) // (TC1) Interrupt Enable Register
#define AT91C_TC1_RB    ((AT91_REG *) 	0x40080058) // (TC1) Register B
#define AT91C_TC1_IDR   ((AT91_REG *) 	0x40080068) // (TC1) Interrupt Disable Register
#define AT91C_TC1_CCR   ((AT91_REG *) 	0x40080040) // (TC1) Channel Control Register
#define AT91C_TC1_IMR   ((AT91_REG *) 	0x4008006C) // (TC1) Interrupt Mask Register
#define AT91C_TC1_RC    ((AT91_REG *) 	0x4008005C) // (TC1) Register C
#define AT91C_TC1_CMR   ((AT91_REG *) 	0x40080044) // (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC1_CV    ((AT91_REG *) 	0x40080050) // (TC1) Counter Value
// ========== Register definition for TC2 peripheral ========== 
#define AT91C_TC2_RA    ((AT91_REG *) 	0x40080094) // (TC2) Register A
#define AT91C_TC2_RB    ((AT91_REG *) 	0x40080098) // (TC2) Register B
#define AT91C_TC2_CMR   ((AT91_REG *) 	0x40080084) // (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC2_SR    ((AT91_REG *) 	0x400800A0) // (TC2) Status Register
#define AT91C_TC2_CCR   ((AT91_REG *) 	0x40080080) // (TC2) Channel Control Register
#define AT91C_TC2_IMR   ((AT91_REG *) 	0x400800AC) // (TC2) Interrupt Mask Register
#define AT91C_TC2_CV    ((AT91_REG *) 	0x40080090) // (TC2) Counter Value
#define AT91C_TC2_RC    ((AT91_REG *) 	0x4008009C) // (TC2) Register C
#define AT91C_TC2_IER   ((AT91_REG *) 	0x400800A4) // (TC2) Interrupt Enable Register
#define AT91C_TC2_IDR   ((AT91_REG *) 	0x400800A8) // (TC2) Interrupt Disable Register
// ========== Register definition for TCB0 peripheral ========== 
#define AT91C_TCB0_BCR  ((AT91_REG *) 	0x400800C0) // (TCB0) TC Block Control Register
#define AT91C_TCB0_IPNAME2 ((AT91_REG *) 	0x400800F4) // (TCB0) TC IPNAME2 REGISTER 
#define AT91C_TCB0_IPNAME1 ((AT91_REG *) 	0x400800F0) // (TCB0) TC IPNAME1 REGISTER 
#define AT91C_TCB0_ADDRSIZE ((AT91_REG *) 	0x400800EC) // (TCB0) TC ADDRSIZE REGISTER 
#define AT91C_TCB0_FEATURES ((AT91_REG *) 	0x400800F8) // (TCB0) TC FEATURES REGISTER 
#define AT91C_TCB0_BMR  ((AT91_REG *) 	0x400800C4) // (TCB0) TC Block Mode Register
#define AT91C_TCB0_VER  ((AT91_REG *) 	0x400800FC) // (TCB0)  Version Register
// ========== Register definition for TCB1 peripheral ========== 
#define AT91C_TCB1_BCR  ((AT91_REG *) 	0x40080100) // (TCB1) TC Block Control Register
#define AT91C_TCB1_VER  ((AT91_REG *) 	0x4008013C) // (TCB1)  Version Register
#define AT91C_TCB1_FEATURES ((AT91_REG *) 	0x40080138) // (TCB1) TC FEATURES REGISTER 
#define AT91C_TCB1_IPNAME2 ((AT91_REG *) 	0x40080134) // (TCB1) TC IPNAME2 REGISTER 
#define AT91C_TCB1_BMR  ((AT91_REG *) 	0x40080104) // (TCB1) TC Block Mode Register
#define AT91C_TCB1_ADDRSIZE ((AT91_REG *) 	0x4008012C) // (TCB1) TC ADDRSIZE REGISTER 
#define AT91C_TCB1_IPNAME1 ((AT91_REG *) 	0x40080130) // (TCB1) TC IPNAME1 REGISTER 
// ========== Register definition for TCB2 peripheral ========== 
#define AT91C_TCB2_FEATURES ((AT91_REG *) 	0x40080178) // (TCB2) TC FEATURES REGISTER 
#define AT91C_TCB2_VER  ((AT91_REG *) 	0x4008017C) // (TCB2)  Version Register
#define AT91C_TCB2_ADDRSIZE ((AT91_REG *) 	0x4008016C) // (TCB2) TC ADDRSIZE REGISTER 
#define AT91C_TCB2_IPNAME1 ((AT91_REG *) 	0x40080170) // (TCB2) TC IPNAME1 REGISTER 
#define AT91C_TCB2_IPNAME2 ((AT91_REG *) 	0x40080174) // (TCB2) TC IPNAME2 REGISTER 
#define AT91C_TCB2_BMR  ((AT91_REG *) 	0x40080144) // (TCB2) TC Block Mode Register
#define AT91C_TCB2_BCR  ((AT91_REG *) 	0x40080140) // (TCB2) TC Block Control Register
// ========== Register definition for EFC0 peripheral ========== 
#define AT91C_EFC0_FCR  ((AT91_REG *) 	0x400E0804) // (EFC0) EFC Flash Command Register
#define AT91C_EFC0_FRR  ((AT91_REG *) 	0x400E080C) // (EFC0) EFC Flash Result Register
#define AT91C_EFC0_FMR  ((AT91_REG *) 	0x400E0800) // (EFC0) EFC Flash Mode Register
#define AT91C_EFC0_FSR  ((AT91_REG *) 	0x400E0808) // (EFC0) EFC Flash Status Register
#define AT91C_EFC0_FVR  ((AT91_REG *) 	0x400E0814) // (EFC0) EFC Flash Version Register
// ========== Register definition for EFC1 peripheral ========== 
#define AT91C_EFC1_FMR  ((AT91_REG *) 	0x400E0A00) // (EFC1) EFC Flash Mode Register
#define AT91C_EFC1_FVR  ((AT91_REG *) 	0x400E0A14) // (EFC1) EFC Flash Version Register
#define AT91C_EFC1_FSR  ((AT91_REG *) 	0x400E0A08) // (EFC1) EFC Flash Status Register
#define AT91C_EFC1_FCR  ((AT91_REG *) 	0x400E0A04) // (EFC1) EFC Flash Command Register
#define AT91C_EFC1_FRR  ((AT91_REG *) 	0x400E0A0C) // (EFC1) EFC Flash Result Register
// ========== Register definition for MCI0 peripheral ========== 
#define AT91C_MCI0_DMA  ((AT91_REG *) 	0x40000050) // (MCI0) MCI DMA Configuration Register
#define AT91C_MCI0_SDCR ((AT91_REG *) 	0x4000000C) // (MCI0) MCI SD/SDIO Card Register
#define AT91C_MCI0_IPNAME1 ((AT91_REG *) 	0x400000F0) // (MCI0) MCI IPNAME1 REGISTER 
#define AT91C_MCI0_CSTOR ((AT91_REG *) 	0x4000001C) // (MCI0) MCI Completion Signal Timeout Register
#define AT91C_MCI0_RDR  ((AT91_REG *) 	0x40000030) // (MCI0) MCI Receive Data Register
#define AT91C_MCI0_CMDR ((AT91_REG *) 	0x40000014) // (MCI0) MCI Command Register
#define AT91C_MCI0_IDR  ((AT91_REG *) 	0x40000048) // (MCI0) MCI Interrupt Disable Register
#define AT91C_MCI0_ADDRSIZE ((AT91_REG *) 	0x400000EC) // (MCI0) MCI ADDRSIZE REGISTER 
#define AT91C_MCI0_WPCR ((AT91_REG *) 	0x400000E4) // (MCI0) MCI Write Protection Control Register
#define AT91C_MCI0_RSPR ((AT91_REG *) 	0x40000020) // (MCI0) MCI Response Register
#define AT91C_MCI0_IPNAME2 ((AT91_REG *) 	0x400000F4) // (MCI0) MCI IPNAME2 REGISTER 
#define AT91C_MCI0_CR   ((AT91_REG *) 	0x40000000) // (MCI0) MCI Control Register
#define AT91C_MCI0_IMR  ((AT91_REG *) 	0x4000004C) // (MCI0) MCI Interrupt Mask Register
#define AT91C_MCI0_WPSR ((AT91_REG *) 	0x400000E8) // (MCI0) MCI Write Protection Status Register
#define AT91C_MCI0_DTOR ((AT91_REG *) 	0x40000008) // (MCI0) MCI Data Timeout Register
#define AT91C_MCI0_MR   ((AT91_REG *) 	0x40000004) // (MCI0) MCI Mode Register
#define AT91C_MCI0_SR   ((AT91_REG *) 	0x40000040) // (MCI0) MCI Status Register
#define AT91C_MCI0_IER  ((AT91_REG *) 	0x40000044) // (MCI0) MCI Interrupt Enable Register
#define AT91C_MCI0_VER  ((AT91_REG *) 	0x400000FC) // (MCI0) MCI VERSION REGISTER 
#define AT91C_MCI0_FEATURES ((AT91_REG *) 	0x400000F8) // (MCI0) MCI FEATURES REGISTER 
#define AT91C_MCI0_BLKR ((AT91_REG *) 	0x40000018) // (MCI0) MCI Block Register
#define AT91C_MCI0_ARGR ((AT91_REG *) 	0x40000010) // (MCI0) MCI Argument Register
#define AT91C_MCI0_FIFO ((AT91_REG *) 	0x40000200) // (MCI0) MCI FIFO Aperture Register
#define AT91C_MCI0_TDR  ((AT91_REG *) 	0x40000034) // (MCI0) MCI Transmit Data Register
#define AT91C_MCI0_CFG  ((AT91_REG *) 	0x40000054) // (MCI0) MCI Configuration Register
// ========== Register definition for PDC_TWI0 peripheral ========== 
#define AT91C_TWI0_TNCR ((AT91_REG *) 	0x4008411C) // (PDC_TWI0) Transmit Next Counter Register
#define AT91C_TWI0_PTCR ((AT91_REG *) 	0x40084120) // (PDC_TWI0) PDC Transfer Control Register
#define AT91C_TWI0_PTSR ((AT91_REG *) 	0x40084124) // (PDC_TWI0) PDC Transfer Status Register
#define AT91C_TWI0_RCR  ((AT91_REG *) 	0x40084104) // (PDC_TWI0) Receive Counter Register
#define AT91C_TWI0_TNPR ((AT91_REG *) 	0x40084118) // (PDC_TWI0) Transmit Next Pointer Register
#define AT91C_TWI0_RNPR ((AT91_REG *) 	0x40084110) // (PDC_TWI0) Receive Next Pointer Register
#define AT91C_TWI0_RPR  ((AT91_REG *) 	0x40084100) // (PDC_TWI0) Receive Pointer Register
#define AT91C_TWI0_RNCR ((AT91_REG *) 	0x40084114) // (PDC_TWI0) Receive Next Counter Register
#define AT91C_TWI0_TPR  ((AT91_REG *) 	0x40084108) // (PDC_TWI0) Transmit Pointer Register
#define AT91C_TWI0_TCR  ((AT91_REG *) 	0x4008410C) // (PDC_TWI0) Transmit Counter Register
// ========== Register definition for PDC_TWI1 peripheral ========== 
#define AT91C_TWI1_TNCR ((AT91_REG *) 	0x4008811C) // (PDC_TWI1) Transmit Next Counter Register
#define AT91C_TWI1_PTCR ((AT91_REG *) 	0x40088120) // (PDC_TWI1) PDC Transfer Control Register
#define AT91C_TWI1_RNCR ((AT91_REG *) 	0x40088114) // (PDC_TWI1) Receive Next Counter Register
#define AT91C_TWI1_RCR  ((AT91_REG *) 	0x40088104) // (PDC_TWI1) Receive Counter Register
#define AT91C_TWI1_RPR  ((AT91_REG *) 	0x40088100) // (PDC_TWI1) Receive Pointer Register
#define AT91C_TWI1_TNPR ((AT91_REG *) 	0x40088118) // (PDC_TWI1) Transmit Next Pointer Register
#define AT91C_TWI1_RNPR ((AT91_REG *) 	0x40088110) // (PDC_TWI1) Receive Next Pointer Register
#define AT91C_TWI1_TCR  ((AT91_REG *) 	0x4008810C) // (PDC_TWI1) Transmit Counter Register
#define AT91C_TWI1_TPR  ((AT91_REG *) 	0x40088108) // (PDC_TWI1) Transmit Pointer Register
#define AT91C_TWI1_PTSR ((AT91_REG *) 	0x40088124) // (PDC_TWI1) PDC Transfer Status Register
// ========== Register definition for TWI0 peripheral ========== 
#define AT91C_TWI0_FEATURES ((AT91_REG *) 	0x400840F8) // (TWI0) TWI FEATURES REGISTER 
#define AT91C_TWI0_IPNAME1 ((AT91_REG *) 	0x400840F0) // (TWI0) TWI IPNAME1 REGISTER 
#define AT91C_TWI0_SMR  ((AT91_REG *) 	0x40084008) // (TWI0) Slave Mode Register
#define AT91C_TWI0_MMR  ((AT91_REG *) 	0x40084004) // (TWI0) Master Mode Register
#define AT91C_TWI0_SR   ((AT91_REG *) 	0x40084020) // (TWI0) Status Register
#define AT91C_TWI0_IPNAME2 ((AT91_REG *) 	0x400840F4) // (TWI0) TWI IPNAME2 REGISTER 
#define AT91C_TWI0_CR   ((AT91_REG *) 	0x40084000) // (TWI0) Control Register
#define AT91C_TWI0_IER  ((AT91_REG *) 	0x40084024) // (TWI0) Interrupt Enable Register
#define AT91C_TWI0_RHR  ((AT91_REG *) 	0x40084030) // (TWI0) Receive Holding Register
#define AT91C_TWI0_ADDRSIZE ((AT91_REG *) 	0x400840EC) // (TWI0) TWI ADDRSIZE REGISTER 
#define AT91C_TWI0_THR  ((AT91_REG *) 	0x40084034) // (TWI0) Transmit Holding Register
#define AT91C_TWI0_VER  ((AT91_REG *) 	0x400840FC) // (TWI0) Version Register
#define AT91C_TWI0_IADR ((AT91_REG *) 	0x4008400C) // (TWI0) Internal Address Register
#define AT91C_TWI0_IMR  ((AT91_REG *) 	0x4008402C) // (TWI0) Interrupt Mask Register
#define AT91C_TWI0_CWGR ((AT91_REG *) 	0x40084010) // (TWI0) Clock Waveform Generator Register
#define AT91C_TWI0_IDR  ((AT91_REG *) 	0x40084028) // (TWI0) Interrupt Disable Register
// ========== Register definition for TWI1 peripheral ========== 
#define AT91C_TWI1_VER  ((AT91_REG *) 	0x400880FC) // (TWI1) Version Register
#define AT91C_TWI1_IDR  ((AT91_REG *) 	0x40088028) // (TWI1) Interrupt Disable Register
#define AT91C_TWI1_IPNAME2 ((AT91_REG *) 	0x400880F4) // (TWI1) TWI IPNAME2 REGISTER 
#define AT91C_TWI1_CWGR ((AT91_REG *) 	0x40088010) // (TWI1) Clock Waveform Generator Register
#define AT91C_TWI1_CR   ((AT91_REG *) 	0x40088000) // (TWI1) Control Register
#define AT91C_TWI1_ADDRSIZE ((AT91_REG *) 	0x400880EC) // (TWI1) TWI ADDRSIZE REGISTER 
#define AT91C_TWI1_IADR ((AT91_REG *) 	0x4008800C) // (TWI1) Internal Address Register
#define AT91C_TWI1_IER  ((AT91_REG *) 	0x40088024) // (TWI1) Interrupt Enable Register
#define AT91C_TWI1_SMR  ((AT91_REG *) 	0x40088008) // (TWI1) Slave Mode Register
#define AT91C_TWI1_RHR  ((AT91_REG *) 	0x40088030) // (TWI1) Receive Holding Register
#define AT91C_TWI1_FEATURES ((AT91_REG *) 	0x400880F8) // (TWI1) TWI FEATURES REGISTER 
#define AT91C_TWI1_IMR  ((AT91_REG *) 	0x4008802C) // (TWI1) Interrupt Mask Register
#define AT91C_TWI1_SR   ((AT91_REG *) 	0x40088020) // (TWI1) Status Register
#define AT91C_TWI1_THR  ((AT91_REG *) 	0x40088034) // (TWI1) Transmit Holding Register
#define AT91C_TWI1_MMR  ((AT91_REG *) 	0x40088004) // (TWI1) Master Mode Register
#define AT91C_TWI1_IPNAME1 ((AT91_REG *) 	0x400880F0) // (TWI1) TWI IPNAME1 REGISTER 
// ========== Register definition for PDC_US0 peripheral ========== 
#define AT91C_US0_RNCR  ((AT91_REG *) 	0x40090114) // (PDC_US0) Receive Next Counter Register
#define AT91C_US0_TNPR  ((AT91_REG *) 	0x40090118) // (PDC_US0) Transmit Next Pointer Register
#define AT91C_US0_TPR   ((AT91_REG *) 	0x40090108) // (PDC_US0) Transmit Pointer Register
#define AT91C_US0_RCR   ((AT91_REG *) 	0x40090104) // (PDC_US0) Receive Counter Register
#define AT91C_US0_RNPR  ((AT91_REG *) 	0x40090110) // (PDC_US0) Receive Next Pointer Register
#define AT91C_US0_TNCR  ((AT91_REG *) 	0x4009011C) // (PDC_US0) Transmit Next Counter Register
#define AT91C_US0_PTSR  ((AT91_REG *) 	0x40090124) // (PDC_US0) PDC Transfer Status Register
#define AT91C_US0_RPR   ((AT91_REG *) 	0x40090100) // (PDC_US0) Receive Pointer Register
#define AT91C_US0_PTCR  ((AT91_REG *) 	0x40090120) // (PDC_US0) PDC Transfer Control Register
#define AT91C_US0_TCR   ((AT91_REG *) 	0x4009010C) // (PDC_US0) Transmit Counter Register
// ========== Register definition for US0 peripheral ========== 
#define AT91C_US0_NER   ((AT91_REG *) 	0x40090044) // (US0) Nb Errors Register
#define AT91C_US0_RHR   ((AT91_REG *) 	0x40090018) // (US0) Receiver Holding Register
#define AT91C_US0_IPNAME1 ((AT91_REG *) 	0x400900F0) // (US0) US IPNAME1 REGISTER 
#define AT91C_US0_MR    ((AT91_REG *) 	0x40090004) // (US0) Mode Register
#define AT91C_US0_RTOR  ((AT91_REG *) 	0x40090024) // (US0) Receiver Time-out Register
#define AT91C_US0_IF    ((AT91_REG *) 	0x4009004C) // (US0) IRDA_FILTER Register
#define AT91C_US0_ADDRSIZE ((AT91_REG *) 	0x400900EC) // (US0) US ADDRSIZE REGISTER 
#define AT91C_US0_IDR   ((AT91_REG *) 	0x4009000C) // (US0) Interrupt Disable Register
#define AT91C_US0_IMR   ((AT91_REG *) 	0x40090010) // (US0) Interrupt Mask Register
#define AT91C_US0_IER   ((AT91_REG *) 	0x40090008) // (US0) Interrupt Enable Register
#define AT91C_US0_TTGR  ((AT91_REG *) 	0x40090028) // (US0) Transmitter Time-guard Register
#define AT91C_US0_IPNAME2 ((AT91_REG *) 	0x400900F4) // (US0) US IPNAME2 REGISTER 
#define AT91C_US0_FIDI  ((AT91_REG *) 	0x40090040) // (US0) FI_DI_Ratio Register
#define AT91C_US0_CR    ((AT91_REG *) 	0x40090000) // (US0) Control Register
#define AT91C_US0_BRGR  ((AT91_REG *) 	0x40090020) // (US0) Baud Rate Generator Register
#define AT91C_US0_MAN   ((AT91_REG *) 	0x40090050) // (US0) Manchester Encoder Decoder Register
#define AT91C_US0_VER   ((AT91_REG *) 	0x400900FC) // (US0) VERSION Register
#define AT91C_US0_FEATURES ((AT91_REG *) 	0x400900F8) // (US0) US FEATURES REGISTER 
#define AT91C_US0_CSR   ((AT91_REG *) 	0x40090014) // (US0) Channel Status Register
#define AT91C_US0_THR   ((AT91_REG *) 	0x4009001C) // (US0) Transmitter Holding Register
// ========== Register definition for PDC_US1 peripheral ========== 
#define AT91C_US1_TNPR  ((AT91_REG *) 	0x40094118) // (PDC_US1) Transmit Next Pointer Register
#define AT91C_US1_TPR   ((AT91_REG *) 	0x40094108) // (PDC_US1) Transmit Pointer Register
#define AT91C_US1_RNCR  ((AT91_REG *) 	0x40094114) // (PDC_US1) Receive Next Counter Register
#define AT91C_US1_TNCR  ((AT91_REG *) 	0x4009411C) // (PDC_US1) Transmit Next Counter Register
#define AT91C_US1_RNPR  ((AT91_REG *) 	0x40094110) // (PDC_US1) Receive Next Pointer Register
#define AT91C_US1_TCR   ((AT91_REG *) 	0x4009410C) // (PDC_US1) Transmit Counter Register
#define AT91C_US1_PTSR  ((AT91_REG *) 	0x40094124) // (PDC_US1) PDC Transfer Status Register
#define AT91C_US1_RCR   ((AT91_REG *) 	0x40094104) // (PDC_US1) Receive Counter Register
#define AT91C_US1_RPR   ((AT91_REG *) 	0x40094100) // (PDC_US1) Receive Pointer Register
#define AT91C_US1_PTCR  ((AT91_REG *) 	0x40094120) // (PDC_US1) PDC Transfer Control Register
// ========== Register definition for US1 peripheral ========== 
#define AT91C_US1_IMR   ((AT91_REG *) 	0x40094010) // (US1) Interrupt Mask Register
#define AT91C_US1_RTOR  ((AT91_REG *) 	0x40094024) // (US1) Receiver Time-out Register
#define AT91C_US1_RHR   ((AT91_REG *) 	0x40094018) // (US1) Receiver Holding Register
#define AT91C_US1_IPNAME1 ((AT91_REG *) 	0x400940F0) // (US1) US IPNAME1 REGISTER 
#define AT91C_US1_VER   ((AT91_REG *) 	0x400940FC) // (US1) VERSION Register
#define AT91C_US1_MR    ((AT91_REG *) 	0x40094004) // (US1) Mode Register
#define AT91C_US1_FEATURES ((AT91_REG *) 	0x400940F8) // (US1) US FEATURES REGISTER 
#define AT91C_US1_NER   ((AT91_REG *) 	0x40094044) // (US1) Nb Errors Register
#define AT91C_US1_IPNAME2 ((AT91_REG *) 	0x400940F4) // (US1) US IPNAME2 REGISTER 
#define AT91C_US1_CR    ((AT91_REG *) 	0x40094000) // (US1) Control Register
#define AT91C_US1_BRGR  ((AT91_REG *) 	0x40094020) // (US1) Baud Rate Generator Register
#define AT91C_US1_IF    ((AT91_REG *) 	0x4009404C) // (US1) IRDA_FILTER Register
#define AT91C_US1_IER   ((AT91_REG *) 	0x40094008) // (US1) Interrupt Enable Register
#define AT91C_US1_TTGR  ((AT91_REG *) 	0x40094028) // (US1) Transmitter Time-guard Register
#define AT91C_US1_FIDI  ((AT91_REG *) 	0x40094040) // (US1) FI_DI_Ratio Register
#define AT91C_US1_MAN   ((AT91_REG *) 	0x40094050) // (US1) Manchester Encoder Decoder Register
#define AT91C_US1_ADDRSIZE ((AT91_REG *) 	0x400940EC) // (US1) US ADDRSIZE REGISTER 
#define AT91C_US1_CSR   ((AT91_REG *) 	0x40094014) // (US1) Channel Status Register
#define AT91C_US1_THR   ((AT91_REG *) 	0x4009401C) // (US1) Transmitter Holding Register
#define AT91C_US1_IDR   ((AT91_REG *) 	0x4009400C) // (US1) Interrupt Disable Register
// ========== Register definition for PDC_US2 peripheral ========== 
#define AT91C_US2_RPR   ((AT91_REG *) 	0x40098100) // (PDC_US2) Receive Pointer Register
#define AT91C_US2_TPR   ((AT91_REG *) 	0x40098108) // (PDC_US2) Transmit Pointer Register
#define AT91C_US2_TCR   ((AT91_REG *) 	0x4009810C) // (PDC_US2) Transmit Counter Register
#define AT91C_US2_PTSR  ((AT91_REG *) 	0x40098124) // (PDC_US2) PDC Transfer Status Register
#define AT91C_US2_PTCR  ((AT91_REG *) 	0x40098120) // (PDC_US2) PDC Transfer Control Register
#define AT91C_US2_RNPR  ((AT91_REG *) 	0x40098110) // (PDC_US2) Receive Next Pointer Register
#define AT91C_US2_TNCR  ((AT91_REG *) 	0x4009811C) // (PDC_US2) Transmit Next Counter Register
#define AT91C_US2_RNCR  ((AT91_REG *) 	0x40098114) // (PDC_US2) Receive Next Counter Register
#define AT91C_US2_TNPR  ((AT91_REG *) 	0x40098118) // (PDC_US2) Transmit Next Pointer Register
#define AT91C_US2_RCR   ((AT91_REG *) 	0x40098104) // (PDC_US2) Receive Counter Register
// ========== Register definition for US2 peripheral ========== 
#define AT91C_US2_MAN   ((AT91_REG *) 	0x40098050) // (US2) Manchester Encoder Decoder Register
#define AT91C_US2_ADDRSIZE ((AT91_REG *) 	0x400980EC) // (US2) US ADDRSIZE REGISTER 
#define AT91C_US2_MR    ((AT91_REG *) 	0x40098004) // (US2) Mode Register
#define AT91C_US2_IPNAME1 ((AT91_REG *) 	0x400980F0) // (US2) US IPNAME1 REGISTER 
#define AT91C_US2_IF    ((AT91_REG *) 	0x4009804C) // (US2) IRDA_FILTER Register
#define AT91C_US2_BRGR  ((AT91_REG *) 	0x40098020) // (US2) Baud Rate Generator Register
#define AT91C_US2_FIDI  ((AT91_REG *) 	0x40098040) // (US2) FI_DI_Ratio Register
#define AT91C_US2_IER   ((AT91_REG *) 	0x40098008) // (US2) Interrupt Enable Register
#define AT91C_US2_RTOR  ((AT91_REG *) 	0x40098024) // (US2) Receiver Time-out Register
#define AT91C_US2_CR    ((AT91_REG *) 	0x40098000) // (US2) Control Register
#define AT91C_US2_THR   ((AT91_REG *) 	0x4009801C) // (US2) Transmitter Holding Register
#define AT91C_US2_CSR   ((AT91_REG *) 	0x40098014) // (US2) Channel Status Register
#define AT91C_US2_VER   ((AT91_REG *) 	0x400980FC) // (US2) VERSION Register
#define AT91C_US2_FEATURES ((AT91_REG *) 	0x400980F8) // (US2) US FEATURES REGISTER 
#define AT91C_US2_IDR   ((AT91_REG *) 	0x4009800C) // (US2) Interrupt Disable Register
#define AT91C_US2_TTGR  ((AT91_REG *) 	0x40098028) // (US2) Transmitter Time-guard Register
#define AT91C_US2_IPNAME2 ((AT91_REG *) 	0x400980F4) // (US2) US IPNAME2 REGISTER 
#define AT91C_US2_RHR   ((AT91_REG *) 	0x40098018) // (US2) Receiver Holding Register
#define AT91C_US2_NER   ((AT91_REG *) 	0x40098044) // (US2) Nb Errors Register
#define AT91C_US2_IMR   ((AT91_REG *) 	0x40098010) // (US2) Interrupt Mask Register
// ========== Register definition for PDC_US3 peripheral ========== 
#define AT91C_US3_TPR   ((AT91_REG *) 	0x4009C108) // (PDC_US3) Transmit Pointer Register
#define AT91C_US3_PTCR  ((AT91_REG *) 	0x4009C120) // (PDC_US3) PDC Transfer Control Register
#define AT91C_US3_TCR   ((AT91_REG *) 	0x4009C10C) // (PDC_US3) Transmit Counter Register
#define AT91C_US3_RCR   ((AT91_REG *) 	0x4009C104) // (PDC_US3) Receive Counter Register
#define AT91C_US3_RNCR  ((AT91_REG *) 	0x4009C114) // (PDC_US3) Receive Next Counter Register
#define AT91C_US3_RNPR  ((AT91_REG *) 	0x4009C110) // (PDC_US3) Receive Next Pointer Register
#define AT91C_US3_RPR   ((AT91_REG *) 	0x4009C100) // (PDC_US3) Receive Pointer Register
#define AT91C_US3_PTSR  ((AT91_REG *) 	0x4009C124) // (PDC_US3) PDC Transfer Status Register
#define AT91C_US3_TNCR  ((AT91_REG *) 	0x4009C11C) // (PDC_US3) Transmit Next Counter Register
#define AT91C_US3_TNPR  ((AT91_REG *) 	0x4009C118) // (PDC_US3) Transmit Next Pointer Register
// ========== Register definition for US3 peripheral ========== 
#define AT91C_US3_MAN   ((AT91_REG *) 	0x4009C050) // (US3) Manchester Encoder Decoder Register
#define AT91C_US3_CSR   ((AT91_REG *) 	0x4009C014) // (US3) Channel Status Register
#define AT91C_US3_BRGR  ((AT91_REG *) 	0x4009C020) // (US3) Baud Rate Generator Register
#define AT91C_US3_IPNAME2 ((AT91_REG *) 	0x4009C0F4) // (US3) US IPNAME2 REGISTER 
#define AT91C_US3_RTOR  ((AT91_REG *) 	0x4009C024) // (US3) Receiver Time-out Register
#define AT91C_US3_ADDRSIZE ((AT91_REG *) 	0x4009C0EC) // (US3) US ADDRSIZE REGISTER 
#define AT91C_US3_CR    ((AT91_REG *) 	0x4009C000) // (US3) Control Register
#define AT91C_US3_IF    ((AT91_REG *) 	0x4009C04C) // (US3) IRDA_FILTER Register
#define AT91C_US3_FEATURES ((AT91_REG *) 	0x4009C0F8) // (US3) US FEATURES REGISTER 
#define AT91C_US3_VER   ((AT91_REG *) 	0x4009C0FC) // (US3) VERSION Register
#define AT91C_US3_RHR   ((AT91_REG *) 	0x4009C018) // (US3) Receiver Holding Register
#define AT91C_US3_TTGR  ((AT91_REG *) 	0x4009C028) // (US3) Transmitter Time-guard Register
#define AT91C_US3_NER   ((AT91_REG *) 	0x4009C044) // (US3) Nb Errors Register
#define AT91C_US3_IMR   ((AT91_REG *) 	0x4009C010) // (US3) Interrupt Mask Register
#define AT91C_US3_THR   ((AT91_REG *) 	0x4009C01C) // (US3) Transmitter Holding Register
#define AT91C_US3_IDR   ((AT91_REG *) 	0x4009C00C) // (US3) Interrupt Disable Register
#define AT91C_US3_MR    ((AT91_REG *) 	0x4009C004) // (US3) Mode Register
#define AT91C_US3_IER   ((AT91_REG *) 	0x4009C008) // (US3) Interrupt Enable Register
#define AT91C_US3_FIDI  ((AT91_REG *) 	0x4009C040) // (US3) FI_DI_Ratio Register
#define AT91C_US3_IPNAME1 ((AT91_REG *) 	0x4009C0F0) // (US3) US IPNAME1 REGISTER 
// ========== Register definition for PDC_SSC0 peripheral ========== 
#define AT91C_SSC0_RNCR ((AT91_REG *) 	0x40004114) // (PDC_SSC0) Receive Next Counter Register
#define AT91C_SSC0_TPR  ((AT91_REG *) 	0x40004108) // (PDC_SSC0) Transmit Pointer Register
#define AT91C_SSC0_TCR  ((AT91_REG *) 	0x4000410C) // (PDC_SSC0) Transmit Counter Register
#define AT91C_SSC0_PTCR ((AT91_REG *) 	0x40004120) // (PDC_SSC0) PDC Transfer Control Register
#define AT91C_SSC0_TNPR ((AT91_REG *) 	0x40004118) // (PDC_SSC0) Transmit Next Pointer Register
#define AT91C_SSC0_RPR  ((AT91_REG *) 	0x40004100) // (PDC_SSC0) Receive Pointer Register
#define AT91C_SSC0_TNCR ((AT91_REG *) 	0x4000411C) // (PDC_SSC0) Transmit Next Counter Register
#define AT91C_SSC0_RNPR ((AT91_REG *) 	0x40004110) // (PDC_SSC0) Receive Next Pointer Register
#define AT91C_SSC0_RCR  ((AT91_REG *) 	0x40004104) // (PDC_SSC0) Receive Counter Register
#define AT91C_SSC0_PTSR ((AT91_REG *) 	0x40004124) // (PDC_SSC0) PDC Transfer Status Register
// ========== Register definition for SSC0 peripheral ========== 
#define AT91C_SSC0_CR   ((AT91_REG *) 	0x40004000) // (SSC0) Control Register
#define AT91C_SSC0_RHR  ((AT91_REG *) 	0x40004020) // (SSC0) Receive Holding Register
#define AT91C_SSC0_TSHR ((AT91_REG *) 	0x40004034) // (SSC0) Transmit Sync Holding Register
#define AT91C_SSC0_RFMR ((AT91_REG *) 	0x40004014) // (SSC0) Receive Frame Mode Register
#define AT91C_SSC0_IDR  ((AT91_REG *) 	0x40004048) // (SSC0) Interrupt Disable Register
#define AT91C_SSC0_TFMR ((AT91_REG *) 	0x4000401C) // (SSC0) Transmit Frame Mode Register
#define AT91C_SSC0_RSHR ((AT91_REG *) 	0x40004030) // (SSC0) Receive Sync Holding Register
#define AT91C_SSC0_RC1R ((AT91_REG *) 	0x4000403C) // (SSC0) Receive Compare 1 Register
#define AT91C_SSC0_TCMR ((AT91_REG *) 	0x40004018) // (SSC0) Transmit Clock Mode Register
#define AT91C_SSC0_RCMR ((AT91_REG *) 	0x40004010) // (SSC0) Receive Clock ModeRegister
#define AT91C_SSC0_SR   ((AT91_REG *) 	0x40004040) // (SSC0) Status Register
#define AT91C_SSC0_RC0R ((AT91_REG *) 	0x40004038) // (SSC0) Receive Compare 0 Register
#define AT91C_SSC0_THR  ((AT91_REG *) 	0x40004024) // (SSC0) Transmit Holding Register
#define AT91C_SSC0_CMR  ((AT91_REG *) 	0x40004004) // (SSC0) Clock Mode Register
#define AT91C_SSC0_IER  ((AT91_REG *) 	0x40004044) // (SSC0) Interrupt Enable Register
#define AT91C_SSC0_IMR  ((AT91_REG *) 	0x4000404C) // (SSC0) Interrupt Mask Register
// ========== Register definition for PDC_PWMC peripheral ========== 
#define AT91C_PWMC_TNCR ((AT91_REG *) 	0x4008C11C) // (PDC_PWMC) Transmit Next Counter Register
#define AT91C_PWMC_TPR  ((AT91_REG *) 	0x4008C108) // (PDC_PWMC) Transmit Pointer Register
#define AT91C_PWMC_RPR  ((AT91_REG *) 	0x4008C100) // (PDC_PWMC) Receive Pointer Register
#define AT91C_PWMC_TCR  ((AT91_REG *) 	0x4008C10C) // (PDC_PWMC) Transmit Counter Register
#define AT91C_PWMC_PTSR ((AT91_REG *) 	0x4008C124) // (PDC_PWMC) PDC Transfer Status Register
#define AT91C_PWMC_RNPR ((AT91_REG *) 	0x4008C110) // (PDC_PWMC) Receive Next Pointer Register
#define AT91C_PWMC_RCR  ((AT91_REG *) 	0x4008C104) // (PDC_PWMC) Receive Counter Register
#define AT91C_PWMC_RNCR ((AT91_REG *) 	0x4008C114) // (PDC_PWMC) Receive Next Counter Register
#define AT91C_PWMC_PTCR ((AT91_REG *) 	0x4008C120) // (PDC_PWMC) PDC Transfer Control Register
#define AT91C_PWMC_TNPR ((AT91_REG *) 	0x4008C118) // (PDC_PWMC) Transmit Next Pointer Register
// ========== Register definition for PWMC_CH0 peripheral ========== 
#define AT91C_PWMC_CH0_DTR ((AT91_REG *) 	0x4008C218) // (PWMC_CH0) Channel Dead Time Value Register
#define AT91C_PWMC_CH0_CMR ((AT91_REG *) 	0x4008C200) // (PWMC_CH0) Channel Mode Register
#define AT91C_PWMC_CH0_CCNTR ((AT91_REG *) 	0x4008C214) // (PWMC_CH0) Channel Counter Register
#define AT91C_PWMC_CH0_CPRDR ((AT91_REG *) 	0x4008C20C) // (PWMC_CH0) Channel Period Register
#define AT91C_PWMC_CH0_DTUPDR ((AT91_REG *) 	0x4008C21C) // (PWMC_CH0) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH0_CPRDUPDR ((AT91_REG *) 	0x4008C210) // (PWMC_CH0) Channel Period Update Register
#define AT91C_PWMC_CH0_CDTYUPDR ((AT91_REG *) 	0x4008C208) // (PWMC_CH0) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH0_CDTYR ((AT91_REG *) 	0x4008C204) // (PWMC_CH0) Channel Duty Cycle Register
// ========== Register definition for PWMC_CH1 peripheral ========== 
#define AT91C_PWMC_CH1_CCNTR ((AT91_REG *) 	0x4008C234) // (PWMC_CH1) Channel Counter Register
#define AT91C_PWMC_CH1_DTR ((AT91_REG *) 	0x4008C238) // (PWMC_CH1) Channel Dead Time Value Register
#define AT91C_PWMC_CH1_CDTYUPDR ((AT91_REG *) 	0x4008C228) // (PWMC_CH1) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH1_DTUPDR ((AT91_REG *) 	0x4008C23C) // (PWMC_CH1) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH1_CDTYR ((AT91_REG *) 	0x4008C224) // (PWMC_CH1) Channel Duty Cycle Register
#define AT91C_PWMC_CH1_CPRDR ((AT91_REG *) 	0x4008C22C) // (PWMC_CH1) Channel Period Register
#define AT91C_PWMC_CH1_CPRDUPDR ((AT91_REG *) 	0x4008C230) // (PWMC_CH1) Channel Period Update Register
#define AT91C_PWMC_CH1_CMR ((AT91_REG *) 	0x4008C220) // (PWMC_CH1) Channel Mode Register
// ========== Register definition for PWMC_CH2 peripheral ========== 
#define AT91C_PWMC_CH2_CDTYR ((AT91_REG *) 	0x4008C244) // (PWMC_CH2) Channel Duty Cycle Register
#define AT91C_PWMC_CH2_DTUPDR ((AT91_REG *) 	0x4008C25C) // (PWMC_CH2) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH2_CCNTR ((AT91_REG *) 	0x4008C254) // (PWMC_CH2) Channel Counter Register
#define AT91C_PWMC_CH2_CMR ((AT91_REG *) 	0x4008C240) // (PWMC_CH2) Channel Mode Register
#define AT91C_PWMC_CH2_CPRDR ((AT91_REG *) 	0x4008C24C) // (PWMC_CH2) Channel Period Register
#define AT91C_PWMC_CH2_CPRDUPDR ((AT91_REG *) 	0x4008C250) // (PWMC_CH2) Channel Period Update Register
#define AT91C_PWMC_CH2_CDTYUPDR ((AT91_REG *) 	0x4008C248) // (PWMC_CH2) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH2_DTR ((AT91_REG *) 	0x4008C258) // (PWMC_CH2) Channel Dead Time Value Register
// ========== Register definition for PWMC_CH3 peripheral ========== 
#define AT91C_PWMC_CH3_CPRDUPDR ((AT91_REG *) 	0x4008C270) // (PWMC_CH3) Channel Period Update Register
#define AT91C_PWMC_CH3_DTR ((AT91_REG *) 	0x4008C278) // (PWMC_CH3) Channel Dead Time Value Register
#define AT91C_PWMC_CH3_CDTYR ((AT91_REG *) 	0x4008C264) // (PWMC_CH3) Channel Duty Cycle Register
#define AT91C_PWMC_CH3_DTUPDR ((AT91_REG *) 	0x4008C27C) // (PWMC_CH3) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH3_CDTYUPDR ((AT91_REG *) 	0x4008C268) // (PWMC_CH3) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH3_CCNTR ((AT91_REG *) 	0x4008C274) // (PWMC_CH3) Channel Counter Register
#define AT91C_PWMC_CH3_CMR ((AT91_REG *) 	0x4008C260) // (PWMC_CH3) Channel Mode Register
#define AT91C_PWMC_CH3_CPRDR ((AT91_REG *) 	0x4008C26C) // (PWMC_CH3) Channel Period Register
// ========== Register definition for PWMC peripheral ========== 
#define AT91C_PWMC_CMP6MUPD ((AT91_REG *) 	0x4008C19C) // (PWMC) PWM Comparison Mode 6 Update Register
#define AT91C_PWMC_ISR1 ((AT91_REG *) 	0x4008C01C) // (PWMC) PWMC Interrupt Status Register 1
#define AT91C_PWMC_CMP5V ((AT91_REG *) 	0x4008C180) // (PWMC) PWM Comparison Value 5 Register
#define AT91C_PWMC_CMP4MUPD ((AT91_REG *) 	0x4008C17C) // (PWMC) PWM Comparison Mode 4 Update Register
#define AT91C_PWMC_FMR  ((AT91_REG *) 	0x4008C05C) // (PWMC) PWM Fault Mode Register
#define AT91C_PWMC_CMP6V ((AT91_REG *) 	0x4008C190) // (PWMC) PWM Comparison Value 6 Register
#define AT91C_PWMC_EL4MR ((AT91_REG *) 	0x4008C08C) // (PWMC) PWM Event Line 4 Mode Register
#define AT91C_PWMC_UPCR ((AT91_REG *) 	0x4008C028) // (PWMC) PWM Update Control Register
#define AT91C_PWMC_CMP1VUPD ((AT91_REG *) 	0x4008C144) // (PWMC) PWM Comparison Value 1 Update Register
#define AT91C_PWMC_CMP0M ((AT91_REG *) 	0x4008C138) // (PWMC) PWM Comparison Mode 0 Register
#define AT91C_PWMC_CMP5VUPD ((AT91_REG *) 	0x4008C184) // (PWMC) PWM Comparison Value 5 Update Register
#define AT91C_PWMC_FPER3 ((AT91_REG *) 	0x4008C074) // (PWMC) PWM Fault Protection Enable Register 3
#define AT91C_PWMC_OSCUPD ((AT91_REG *) 	0x4008C058) // (PWMC) PWM Output Selection Clear Update Register
#define AT91C_PWMC_FPER1 ((AT91_REG *) 	0x4008C06C) // (PWMC) PWM Fault Protection Enable Register 1
#define AT91C_PWMC_SCUPUPD ((AT91_REG *) 	0x4008C030) // (PWMC) PWM Update Period Update Register
#define AT91C_PWMC_DIS  ((AT91_REG *) 	0x4008C008) // (PWMC) PWMC Disable Register
#define AT91C_PWMC_IER1 ((AT91_REG *) 	0x4008C010) // (PWMC) PWMC Interrupt Enable Register 1
#define AT91C_PWMC_IMR2 ((AT91_REG *) 	0x4008C03C) // (PWMC) PWMC Interrupt Mask Register 2
#define AT91C_PWMC_CMP0V ((AT91_REG *) 	0x4008C130) // (PWMC) PWM Comparison Value 0 Register
#define AT91C_PWMC_SR   ((AT91_REG *) 	0x4008C00C) // (PWMC) PWMC Status Register
#define AT91C_PWMC_CMP4M ((AT91_REG *) 	0x4008C178) // (PWMC) PWM Comparison Mode 4 Register
#define AT91C_PWMC_CMP3M ((AT91_REG *) 	0x4008C168) // (PWMC) PWM Comparison Mode 3 Register
#define AT91C_PWMC_IER2 ((AT91_REG *) 	0x4008C034) // (PWMC) PWMC Interrupt Enable Register 2
#define AT91C_PWMC_CMP3VUPD ((AT91_REG *) 	0x4008C164) // (PWMC) PWM Comparison Value 3 Update Register
#define AT91C_PWMC_CMP2M ((AT91_REG *) 	0x4008C158) // (PWMC) PWM Comparison Mode 2 Register
#define AT91C_PWMC_IDR2 ((AT91_REG *) 	0x4008C038) // (PWMC) PWMC Interrupt Disable Register 2
#define AT91C_PWMC_EL2MR ((AT91_REG *) 	0x4008C084) // (PWMC) PWM Event Line 2 Mode Register
#define AT91C_PWMC_CMP7V ((AT91_REG *) 	0x4008C1A0) // (PWMC) PWM Comparison Value 7 Register
#define AT91C_PWMC_CMP1M ((AT91_REG *) 	0x4008C148) // (PWMC) PWM Comparison Mode 1 Register
#define AT91C_PWMC_CMP0VUPD ((AT91_REG *) 	0x4008C134) // (PWMC) PWM Comparison Value 0 Update Register
#define AT91C_PWMC_WPSR ((AT91_REG *) 	0x4008C0E8) // (PWMC) PWM Write Protection Status Register
#define AT91C_PWMC_CMP6VUPD ((AT91_REG *) 	0x4008C194) // (PWMC) PWM Comparison Value 6 Update Register
#define AT91C_PWMC_CMP1MUPD ((AT91_REG *) 	0x4008C14C) // (PWMC) PWM Comparison Mode 1 Update Register
#define AT91C_PWMC_CMP1V ((AT91_REG *) 	0x4008C140) // (PWMC) PWM Comparison Value 1 Register
#define AT91C_PWMC_FCR  ((AT91_REG *) 	0x4008C064) // (PWMC) PWM Fault Mode Clear Register
#define AT91C_PWMC_VER  ((AT91_REG *) 	0x4008C0FC) // (PWMC) PWMC Version Register
#define AT91C_PWMC_EL1MR ((AT91_REG *) 	0x4008C080) // (PWMC) PWM Event Line 1 Mode Register
#define AT91C_PWMC_EL6MR ((AT91_REG *) 	0x4008C094) // (PWMC) PWM Event Line 6 Mode Register
#define AT91C_PWMC_ISR2 ((AT91_REG *) 	0x4008C040) // (PWMC) PWMC Interrupt Status Register 2
#define AT91C_PWMC_CMP4VUPD ((AT91_REG *) 	0x4008C174) // (PWMC) PWM Comparison Value 4 Update Register
#define AT91C_PWMC_CMP5MUPD ((AT91_REG *) 	0x4008C18C) // (PWMC) PWM Comparison Mode 5 Update Register
#define AT91C_PWMC_OS   ((AT91_REG *) 	0x4008C048) // (PWMC) PWM Output Selection Register
#define AT91C_PWMC_FPV  ((AT91_REG *) 	0x4008C068) // (PWMC) PWM Fault Protection Value Register
#define AT91C_PWMC_FPER2 ((AT91_REG *) 	0x4008C070) // (PWMC) PWM Fault Protection Enable Register 2
#define AT91C_PWMC_EL7MR ((AT91_REG *) 	0x4008C098) // (PWMC) PWM Event Line 7 Mode Register
#define AT91C_PWMC_OSSUPD ((AT91_REG *) 	0x4008C054) // (PWMC) PWM Output Selection Set Update Register
#define AT91C_PWMC_FEATURES ((AT91_REG *) 	0x4008C0F8) // (PWMC) PWMC FEATURES REGISTER 
#define AT91C_PWMC_CMP2V ((AT91_REG *) 	0x4008C150) // (PWMC) PWM Comparison Value 2 Register
#define AT91C_PWMC_FSR  ((AT91_REG *) 	0x4008C060) // (PWMC) PWM Fault Mode Status Register
#define AT91C_PWMC_ADDRSIZE ((AT91_REG *) 	0x4008C0EC) // (PWMC) PWMC ADDRSIZE REGISTER 
#define AT91C_PWMC_OSC  ((AT91_REG *) 	0x4008C050) // (PWMC) PWM Output Selection Clear Register
#define AT91C_PWMC_SCUP ((AT91_REG *) 	0x4008C02C) // (PWMC) PWM Update Period Register
#define AT91C_PWMC_CMP7MUPD ((AT91_REG *) 	0x4008C1AC) // (PWMC) PWM Comparison Mode 7 Update Register
#define AT91C_PWMC_CMP2VUPD ((AT91_REG *) 	0x4008C154) // (PWMC) PWM Comparison Value 2 Update Register
#define AT91C_PWMC_FPER4 ((AT91_REG *) 	0x4008C078) // (PWMC) PWM Fault Protection Enable Register 4
#define AT91C_PWMC_IMR1 ((AT91_REG *) 	0x4008C018) // (PWMC) PWMC Interrupt Mask Register 1
#define AT91C_PWMC_EL3MR ((AT91_REG *) 	0x4008C088) // (PWMC) PWM Event Line 3 Mode Register
#define AT91C_PWMC_CMP3V ((AT91_REG *) 	0x4008C160) // (PWMC) PWM Comparison Value 3 Register
#define AT91C_PWMC_IPNAME1 ((AT91_REG *) 	0x4008C0F0) // (PWMC) PWMC IPNAME1 REGISTER 
#define AT91C_PWMC_OSS  ((AT91_REG *) 	0x4008C04C) // (PWMC) PWM Output Selection Set Register
#define AT91C_PWMC_CMP0MUPD ((AT91_REG *) 	0x4008C13C) // (PWMC) PWM Comparison Mode 0 Update Register
#define AT91C_PWMC_CMP2MUPD ((AT91_REG *) 	0x4008C15C) // (PWMC) PWM Comparison Mode 2 Update Register
#define AT91C_PWMC_CMP4V ((AT91_REG *) 	0x4008C170) // (PWMC) PWM Comparison Value 4 Register
#define AT91C_PWMC_ENA  ((AT91_REG *) 	0x4008C004) // (PWMC) PWMC Enable Register
#define AT91C_PWMC_CMP3MUPD ((AT91_REG *) 	0x4008C16C) // (PWMC) PWM Comparison Mode 3 Update Register
#define AT91C_PWMC_EL0MR ((AT91_REG *) 	0x4008C07C) // (PWMC) PWM Event Line 0 Mode Register
#define AT91C_PWMC_OOV  ((AT91_REG *) 	0x4008C044) // (PWMC) PWM Output Override Value Register
#define AT91C_PWMC_WPCR ((AT91_REG *) 	0x4008C0E4) // (PWMC) PWM Write Protection Enable Register
#define AT91C_PWMC_CMP7M ((AT91_REG *) 	0x4008C1A8) // (PWMC) PWM Comparison Mode 7 Register
#define AT91C_PWMC_CMP6M ((AT91_REG *) 	0x4008C198) // (PWMC) PWM Comparison Mode 6 Register
#define AT91C_PWMC_CMP5M ((AT91_REG *) 	0x4008C188) // (PWMC) PWM Comparison Mode 5 Register
#define AT91C_PWMC_IPNAME2 ((AT91_REG *) 	0x4008C0F4) // (PWMC) PWMC IPNAME2 REGISTER 
#define AT91C_PWMC_CMP7VUPD ((AT91_REG *) 	0x4008C1A4) // (PWMC) PWM Comparison Value 7 Update Register
#define AT91C_PWMC_SYNC ((AT91_REG *) 	0x4008C020) // (PWMC) PWM Synchronized Channels Register
#define AT91C_PWMC_MR   ((AT91_REG *) 	0x4008C000) // (PWMC) PWMC Mode Register
#define AT91C_PWMC_IDR1 ((AT91_REG *) 	0x4008C014) // (PWMC) PWMC Interrupt Disable Register 1
#define AT91C_PWMC_EL5MR ((AT91_REG *) 	0x4008C090) // (PWMC) PWM Event Line 5 Mode Register
// ========== Register definition for SPI0 peripheral ========== 
#define AT91C_SPI0_ADDRSIZE ((AT91_REG *) 	0x400080EC) // (SPI0) SPI ADDRSIZE REGISTER 
#define AT91C_SPI0_RDR  ((AT91_REG *) 	0x40008008) // (SPI0) Receive Data Register
#define AT91C_SPI0_FEATURES ((AT91_REG *) 	0x400080F8) // (SPI0) SPI FEATURES REGISTER 
#define AT91C_SPI0_CR   ((AT91_REG *) 	0x40008000) // (SPI0) Control Register
#define AT91C_SPI0_IPNAME1 ((AT91_REG *) 	0x400080F0) // (SPI0) SPI IPNAME1 REGISTER 
#define AT91C_SPI0_VER  ((AT91_REG *) 	0x400080FC) // (SPI0) Version Register
#define AT91C_SPI0_IDR  ((AT91_REG *) 	0x40008018) // (SPI0) Interrupt Disable Register
#define AT91C_SPI0_TDR  ((AT91_REG *) 	0x4000800C) // (SPI0) Transmit Data Register
#define AT91C_SPI0_MR   ((AT91_REG *) 	0x40008004) // (SPI0) Mode Register
#define AT91C_SPI0_IER  ((AT91_REG *) 	0x40008014) // (SPI0) Interrupt Enable Register
#define AT91C_SPI0_IMR  ((AT91_REG *) 	0x4000801C) // (SPI0) Interrupt Mask Register
#define AT91C_SPI0_IPNAME2 ((AT91_REG *) 	0x400080F4) // (SPI0) SPI IPNAME2 REGISTER 
#define AT91C_SPI0_CSR  ((AT91_REG *) 	0x40008030) // (SPI0) Chip Select Register
#define AT91C_SPI0_SR   ((AT91_REG *) 	0x40008010) // (SPI0) Status Register
// ========== Register definition for UDPHS_EPTFIFO peripheral ========== 
#define AT91C_UDPHS_EPTFIFO_READEPT6 ((AT91_REG *) 	0x201E0000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6
#define AT91C_UDPHS_EPTFIFO_READEPT2 ((AT91_REG *) 	0x201A0000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2
#define AT91C_UDPHS_EPTFIFO_READEPT1 ((AT91_REG *) 	0x20190000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1
#define AT91C_UDPHS_EPTFIFO_READEPT0 ((AT91_REG *) 	0x20180000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0
#define AT91C_UDPHS_EPTFIFO_READEPT5 ((AT91_REG *) 	0x201D0000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5
#define AT91C_UDPHS_EPTFIFO_READEPT4 ((AT91_REG *) 	0x201C0000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4
#define AT91C_UDPHS_EPTFIFO_READEPT3 ((AT91_REG *) 	0x201B0000) // (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3
// ========== Register definition for UDPHS_EPT_0 peripheral ========== 
#define AT91C_UDPHS_EPT_0_EPTCTL ((AT91_REG *) 	0x400A410C) // (UDPHS_EPT_0) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_0_EPTSTA ((AT91_REG *) 	0x400A411C) // (UDPHS_EPT_0) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_0_EPTCLRSTA ((AT91_REG *) 	0x400A4118) // (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_0_EPTCTLDIS ((AT91_REG *) 	0x400A4108) // (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_0_EPTCFG ((AT91_REG *) 	0x400A4100) // (UDPHS_EPT_0) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_0_EPTSETSTA ((AT91_REG *) 	0x400A4114) // (UDPHS_EPT_0) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_0_EPTCTLENB ((AT91_REG *) 	0x400A4104) // (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register
// ========== Register definition for UDPHS_EPT_1 peripheral ========== 
#define AT91C_UDPHS_EPT_1_EPTSTA ((AT91_REG *) 	0x400A413C) // (UDPHS_EPT_1) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_1_EPTSETSTA ((AT91_REG *) 	0x400A4134) // (UDPHS_EPT_1) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_1_EPTCTL ((AT91_REG *) 	0x400A412C) // (UDPHS_EPT_1) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_1_EPTCFG ((AT91_REG *) 	0x400A4120) // (UDPHS_EPT_1) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_1_EPTCTLDIS ((AT91_REG *) 	0x400A4128) // (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_1_EPTCLRSTA ((AT91_REG *) 	0x400A4138) // (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_1_EPTCTLENB ((AT91_REG *) 	0x400A4124) // (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register
// ========== Register definition for UDPHS_EPT_2 peripheral ========== 
#define AT91C_UDPHS_EPT_2_EPTCTLENB ((AT91_REG *) 	0x400A4144) // (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_2_EPTCLRSTA ((AT91_REG *) 	0x400A4158) // (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_2_EPTCFG ((AT91_REG *) 	0x400A4140) // (UDPHS_EPT_2) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_2_EPTCTL ((AT91_REG *) 	0x400A414C) // (UDPHS_EPT_2) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_2_EPTSETSTA ((AT91_REG *) 	0x400A4154) // (UDPHS_EPT_2) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_2_EPTSTA ((AT91_REG *) 	0x400A415C) // (UDPHS_EPT_2) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_2_EPTCTLDIS ((AT91_REG *) 	0x400A4148) // (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register
// ========== Register definition for UDPHS_EPT_3 peripheral ========== 
#define AT91C_UDPHS_EPT_3_EPTCTLDIS ((AT91_REG *) 	0x400A4168) // (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_3_EPTCTLENB ((AT91_REG *) 	0x400A4164) // (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_3_EPTSETSTA ((AT91_REG *) 	0x400A4174) // (UDPHS_EPT_3) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_3_EPTCLRSTA ((AT91_REG *) 	0x400A4178) // (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_3_EPTCFG ((AT91_REG *) 	0x400A4160) // (UDPHS_EPT_3) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_3_EPTSTA ((AT91_REG *) 	0x400A417C) // (UDPHS_EPT_3) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_3_EPTCTL ((AT91_REG *) 	0x400A416C) // (UDPHS_EPT_3) UDPHS Endpoint Control Register
// ========== Register definition for UDPHS_EPT_4 peripheral ========== 
#define AT91C_UDPHS_EPT_4_EPTSETSTA ((AT91_REG *) 	0x400A4194) // (UDPHS_EPT_4) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_4_EPTCTLDIS ((AT91_REG *) 	0x400A4188) // (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_4_EPTCTL ((AT91_REG *) 	0x400A418C) // (UDPHS_EPT_4) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_4_EPTCFG ((AT91_REG *) 	0x400A4180) // (UDPHS_EPT_4) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_4_EPTCTLENB ((AT91_REG *) 	0x400A4184) // (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_4_EPTSTA ((AT91_REG *) 	0x400A419C) // (UDPHS_EPT_4) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_4_EPTCLRSTA ((AT91_REG *) 	0x400A4198) // (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register
// ========== Register definition for UDPHS_EPT_5 peripheral ========== 
#define AT91C_UDPHS_EPT_5_EPTCFG ((AT91_REG *) 	0x400A41A0) // (UDPHS_EPT_5) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_5_EPTCTL ((AT91_REG *) 	0x400A41AC) // (UDPHS_EPT_5) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_5_EPTCTLENB ((AT91_REG *) 	0x400A41A4) // (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_5_EPTSTA ((AT91_REG *) 	0x400A41BC) // (UDPHS_EPT_5) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_5_EPTSETSTA ((AT91_REG *) 	0x400A41B4) // (UDPHS_EPT_5) UDPHS Endpoint Set Status Register
#define AT91C_UDPHS_EPT_5_EPTCTLDIS ((AT91_REG *) 	0x400A41A8) // (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_5_EPTCLRSTA ((AT91_REG *) 	0x400A41B8) // (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register
// ========== Register definition for UDPHS_EPT_6 peripheral ========== 
#define AT91C_UDPHS_EPT_6_EPTCLRSTA ((AT91_REG *) 	0x400A41D8) // (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register
#define AT91C_UDPHS_EPT_6_EPTCTL ((AT91_REG *) 	0x400A41CC) // (UDPHS_EPT_6) UDPHS Endpoint Control Register
#define AT91C_UDPHS_EPT_6_EPTCFG ((AT91_REG *) 	0x400A41C0) // (UDPHS_EPT_6) UDPHS Endpoint Config Register
#define AT91C_UDPHS_EPT_6_EPTCTLDIS ((AT91_REG *) 	0x400A41C8) // (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register
#define AT91C_UDPHS_EPT_6_EPTSTA ((AT91_REG *) 	0x400A41DC) // (UDPHS_EPT_6) UDPHS Endpoint Status Register
#define AT91C_UDPHS_EPT_6_EPTCTLENB ((AT91_REG *) 	0x400A41C4) // (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register
#define AT91C_UDPHS_EPT_6_EPTSETSTA ((AT91_REG *) 	0x400A41D4) // (UDPHS_EPT_6) UDPHS Endpoint Set Status Register
// ========== Register definition for UDPHS_DMA_1 peripheral ========== 
#define AT91C_UDPHS_DMA_1_DMASTATUS ((AT91_REG *) 	0x400A431C) // (UDPHS_DMA_1) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_1_DMACONTROL ((AT91_REG *) 	0x400A4318) // (UDPHS_DMA_1) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_1_DMANXTDSC ((AT91_REG *) 	0x400A4310) // (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_1_DMAADDRESS ((AT91_REG *) 	0x400A4314) // (UDPHS_DMA_1) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS_DMA_2 peripheral ========== 
#define AT91C_UDPHS_DMA_2_DMASTATUS ((AT91_REG *) 	0x400A432C) // (UDPHS_DMA_2) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_2_DMANXTDSC ((AT91_REG *) 	0x400A4320) // (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_2_DMACONTROL ((AT91_REG *) 	0x400A4328) // (UDPHS_DMA_2) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_2_DMAADDRESS ((AT91_REG *) 	0x400A4324) // (UDPHS_DMA_2) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS_DMA_3 peripheral ========== 
#define AT91C_UDPHS_DMA_3_DMACONTROL ((AT91_REG *) 	0x400A4338) // (UDPHS_DMA_3) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_3_DMANXTDSC ((AT91_REG *) 	0x400A4330) // (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_3_DMASTATUS ((AT91_REG *) 	0x400A433C) // (UDPHS_DMA_3) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_3_DMAADDRESS ((AT91_REG *) 	0x400A4334) // (UDPHS_DMA_3) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS_DMA_4 peripheral ========== 
#define AT91C_UDPHS_DMA_4_DMAADDRESS ((AT91_REG *) 	0x400A4344) // (UDPHS_DMA_4) UDPHS DMA Channel Address Register
#define AT91C_UDPHS_DMA_4_DMANXTDSC ((AT91_REG *) 	0x400A4340) // (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_4_DMASTATUS ((AT91_REG *) 	0x400A434C) // (UDPHS_DMA_4) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_4_DMACONTROL ((AT91_REG *) 	0x400A4348) // (UDPHS_DMA_4) UDPHS DMA Channel Control Register
// ========== Register definition for UDPHS_DMA_5 peripheral ========== 
#define AT91C_UDPHS_DMA_5_DMACONTROL ((AT91_REG *) 	0x400A4358) // (UDPHS_DMA_5) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_5_DMAADDRESS ((AT91_REG *) 	0x400A4354) // (UDPHS_DMA_5) UDPHS DMA Channel Address Register
#define AT91C_UDPHS_DMA_5_DMANXTDSC ((AT91_REG *) 	0x400A4350) // (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_5_DMASTATUS ((AT91_REG *) 	0x400A435C) // (UDPHS_DMA_5) UDPHS DMA Channel Status Register
// ========== Register definition for UDPHS_DMA_6 peripheral ========== 
#define AT91C_UDPHS_DMA_6_DMASTATUS ((AT91_REG *) 	0x400A436C) // (UDPHS_DMA_6) UDPHS DMA Channel Status Register
#define AT91C_UDPHS_DMA_6_DMACONTROL ((AT91_REG *) 	0x400A4368) // (UDPHS_DMA_6) UDPHS DMA Channel Control Register
#define AT91C_UDPHS_DMA_6_DMANXTDSC ((AT91_REG *) 	0x400A4360) // (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address
#define AT91C_UDPHS_DMA_6_DMAADDRESS ((AT91_REG *) 	0x400A4364) // (UDPHS_DMA_6) UDPHS DMA Channel Address Register
// ========== Register definition for UDPHS peripheral ========== 
#define AT91C_UDPHS_EPTRST ((AT91_REG *) 	0x400A401C) // (UDPHS) UDPHS Endpoints Reset Register
#define AT91C_UDPHS_IEN ((AT91_REG *) 	0x400A4010) // (UDPHS) UDPHS Interrupt Enable Register
#define AT91C_UDPHS_TSTCNTB ((AT91_REG *) 	0x400A40D8) // (UDPHS) UDPHS Test B Counter Register
#define AT91C_UDPHS_RIPNAME2 ((AT91_REG *) 	0x400A40F4) // (UDPHS) UDPHS Name2 Register
#define AT91C_UDPHS_RIPPADDRSIZE ((AT91_REG *) 	0x400A40EC) // (UDPHS) UDPHS PADDRSIZE Register
#define AT91C_UDPHS_TSTMODREG ((AT91_REG *) 	0x400A40DC) // (UDPHS) UDPHS Test Mode Register
#define AT91C_UDPHS_TST ((AT91_REG *) 	0x400A40E0) // (UDPHS) UDPHS Test Register
#define AT91C_UDPHS_TSTSOFCNT ((AT91_REG *) 	0x400A40D0) // (UDPHS) UDPHS Test SOF Counter Register
#define AT91C_UDPHS_FNUM ((AT91_REG *) 	0x400A4004) // (UDPHS) UDPHS Frame Number Register
#define AT91C_UDPHS_TSTCNTA ((AT91_REG *) 	0x400A40D4) // (UDPHS) UDPHS Test A Counter Register
#define AT91C_UDPHS_INTSTA ((AT91_REG *) 	0x400A4014) // (UDPHS) UDPHS Interrupt Status Register
#define AT91C_UDPHS_IPFEATURES ((AT91_REG *) 	0x400A40F8) // (UDPHS) UDPHS Features Register
#define AT91C_UDPHS_CLRINT ((AT91_REG *) 	0x400A4018) // (UDPHS) UDPHS Clear Interrupt Register
#define AT91C_UDPHS_RIPNAME1 ((AT91_REG *) 	0x400A40F0) // (UDPHS) UDPHS Name1 Register
#define AT91C_UDPHS_CTRL ((AT91_REG *) 	0x400A4000) // (UDPHS) UDPHS Control Register
#define AT91C_UDPHS_IPVERSION ((AT91_REG *) 	0x400A40FC) // (UDPHS) UDPHS Version Register
// ========== Register definition for HDMA_CH_0 peripheral ========== 
#define AT91C_HDMA_CH_0_SADDR ((AT91_REG *) 	0x400B003C) // (HDMA_CH_0) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_0_DADDR ((AT91_REG *) 	0x400B0040) // (HDMA_CH_0) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_0_CFG ((AT91_REG *) 	0x400B0050) // (HDMA_CH_0) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_0_CTRLB ((AT91_REG *) 	0x400B004C) // (HDMA_CH_0) HDMA Channel Control B Register
#define AT91C_HDMA_CH_0_CTRLA ((AT91_REG *) 	0x400B0048) // (HDMA_CH_0) HDMA Channel Control A Register
#define AT91C_HDMA_CH_0_DSCR ((AT91_REG *) 	0x400B0044) // (HDMA_CH_0) HDMA Channel Descriptor Address Register
// ========== Register definition for HDMA_CH_1 peripheral ========== 
#define AT91C_HDMA_CH_1_DSCR ((AT91_REG *) 	0x400B006C) // (HDMA_CH_1) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_1_CTRLB ((AT91_REG *) 	0x400B0074) // (HDMA_CH_1) HDMA Channel Control B Register
#define AT91C_HDMA_CH_1_SADDR ((AT91_REG *) 	0x400B0064) // (HDMA_CH_1) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_1_CFG ((AT91_REG *) 	0x400B0078) // (HDMA_CH_1) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_1_DADDR ((AT91_REG *) 	0x400B0068) // (HDMA_CH_1) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_1_CTRLA ((AT91_REG *) 	0x400B0070) // (HDMA_CH_1) HDMA Channel Control A Register
// ========== Register definition for HDMA_CH_2 peripheral ========== 
#define AT91C_HDMA_CH_2_CTRLA ((AT91_REG *) 	0x400B0098) // (HDMA_CH_2) HDMA Channel Control A Register
#define AT91C_HDMA_CH_2_SADDR ((AT91_REG *) 	0x400B008C) // (HDMA_CH_2) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_2_CTRLB ((AT91_REG *) 	0x400B009C) // (HDMA_CH_2) HDMA Channel Control B Register
#define AT91C_HDMA_CH_2_DADDR ((AT91_REG *) 	0x400B0090) // (HDMA_CH_2) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_2_CFG ((AT91_REG *) 	0x400B00A0) // (HDMA_CH_2) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_2_DSCR ((AT91_REG *) 	0x400B0094) // (HDMA_CH_2) HDMA Channel Descriptor Address Register
// ========== Register definition for HDMA_CH_3 peripheral ========== 
#define AT91C_HDMA_CH_3_DSCR ((AT91_REG *) 	0x400B00BC) // (HDMA_CH_3) HDMA Channel Descriptor Address Register
#define AT91C_HDMA_CH_3_SADDR ((AT91_REG *) 	0x400B00B4) // (HDMA_CH_3) HDMA Channel Source Address Register
#define AT91C_HDMA_CH_3_CTRLB ((AT91_REG *) 	0x400B00C4) // (HDMA_CH_3) HDMA Channel Control B Register
#define AT91C_HDMA_CH_3_CFG ((AT91_REG *) 	0x400B00C8) // (HDMA_CH_3) HDMA Channel Configuration Register
#define AT91C_HDMA_CH_3_DADDR ((AT91_REG *) 	0x400B00B8) // (HDMA_CH_3) HDMA Channel Destination Address Register
#define AT91C_HDMA_CH_3_CTRLA ((AT91_REG *) 	0x400B00C0) // (HDMA_CH_3) HDMA Channel Control A Register
// ========== Register definition for HDMA peripheral ========== 
#define AT91C_HDMA_VER  ((AT91_REG *) 	0x400B01FC) // (HDMA) HDMA VERSION REGISTER 
#define AT91C_HDMA_CHSR ((AT91_REG *) 	0x400B0030) // (HDMA) HDMA Channel Handler Status Register
#define AT91C_HDMA_IPNAME2 ((AT91_REG *) 	0x400B01F4) // (HDMA) HDMA IPNAME2 REGISTER 
#define AT91C_HDMA_EBCIMR ((AT91_REG *) 	0x400B0020) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
#define AT91C_HDMA_CHDR ((AT91_REG *) 	0x400B002C) // (HDMA) HDMA Channel Handler Disable Register
#define AT91C_HDMA_EN   ((AT91_REG *) 	0x400B0004) // (HDMA) HDMA Controller Enable Register
#define AT91C_HDMA_GCFG ((AT91_REG *) 	0x400B0000) // (HDMA) HDMA Global Configuration Register
#define AT91C_HDMA_IPNAME1 ((AT91_REG *) 	0x400B01F0) // (HDMA) HDMA IPNAME1 REGISTER 
#define AT91C_HDMA_LAST ((AT91_REG *) 	0x400B0010) // (HDMA) HDMA Software Last Transfer Flag Register
#define AT91C_HDMA_FEATURES ((AT91_REG *) 	0x400B01F8) // (HDMA) HDMA FEATURES REGISTER 
#define AT91C_HDMA_CREQ ((AT91_REG *) 	0x400B000C) // (HDMA) HDMA Software Chunk Transfer Request Register
#define AT91C_HDMA_EBCIER ((AT91_REG *) 	0x400B0018) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
#define AT91C_HDMA_CHER ((AT91_REG *) 	0x400B0028) // (HDMA) HDMA Channel Handler Enable Register
#define AT91C_HDMA_ADDRSIZE ((AT91_REG *) 	0x400B01EC) // (HDMA) HDMA ADDRSIZE REGISTER 
#define AT91C_HDMA_EBCISR ((AT91_REG *) 	0x400B0024) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
#define AT91C_HDMA_SREQ ((AT91_REG *) 	0x400B0008) // (HDMA) HDMA Software Single Request Register
#define AT91C_HDMA_EBCIDR ((AT91_REG *) 	0x400B001C) // (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register

// *****************************************************************************
//               PIO DEFINITIONS FOR AT91SAM3U1
// *****************************************************************************
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) // Pin Controlled by PA0
#define AT91C_PA0_TIOB0    ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PA0_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) // Pin Controlled by PA1
#define AT91C_PA1_TIOA0    ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PA1_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) // Pin Controlled by PA10
#define AT91C_PA10_TWCK0    ((unsigned int) AT91C_PIO_PA10) //  
#define AT91C_PA10_PWML3    ((unsigned int) AT91C_PIO_PA10) //  
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) // Pin Controlled by PA11
#define AT91C_PA11_DRXD     ((unsigned int) AT91C_PIO_PA11) //  
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) // Pin Controlled by PA12
#define AT91C_PA12_DTXD     ((unsigned int) AT91C_PIO_PA12) //  
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) // Pin Controlled by PA13
#define AT91C_PA13_SPI0_MISO ((unsigned int) AT91C_PIO_PA13) //  
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) // Pin Controlled by PA14
#define AT91C_PA14_SPI0_MOSI ((unsigned int) AT91C_PIO_PA14) //  
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) // Pin Controlled by PA15
#define AT91C_PA15_SPI0_SPCK ((unsigned int) AT91C_PIO_PA15) //  
#define AT91C_PA15_PWMH2    ((unsigned int) AT91C_PIO_PA15) //  
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) // Pin Controlled by PA16
#define AT91C_PA16_SPI0_NPCS0 ((unsigned int) AT91C_PIO_PA16) //  
#define AT91C_PA16_NCS1     ((unsigned int) AT91C_PIO_PA16) //  
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) // Pin Controlled by PA17
#define AT91C_PA17_SCK0     ((unsigned int) AT91C_PIO_PA17) //  
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) // Pin Controlled by PA18
#define AT91C_PA18_TXD0     ((unsigned int) AT91C_PIO_PA18) //  
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) // Pin Controlled by PA19
#define AT91C_PA19_RXD0     ((unsigned int) AT91C_PIO_PA19) //  
#define AT91C_PA19_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PA19) //  
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) // Pin Controlled by PA2
#define AT91C_PA2_TCLK0    ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PA2_ADTRG1   ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) // Pin Controlled by PA20
#define AT91C_PA20_TXD1     ((unsigned int) AT91C_PIO_PA20) //  
#define AT91C_PA20_PWMH3    ((unsigned int) AT91C_PIO_PA20) //  
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) // Pin Controlled by PA21
#define AT91C_PA21_RXD1     ((unsigned int) AT91C_PIO_PA21) //  
#define AT91C_PA21_PCK0     ((unsigned int) AT91C_PIO_PA21) //  
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) // Pin Controlled by PA22
#define AT91C_PA22_TXD2     ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PA22_RTS1     ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) // Pin Controlled by PA23
#define AT91C_PA23_RXD2     ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PA23_CTS1     ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) // Pin Controlled by PA24
#define AT91C_PA24_TWD1     ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PA24_SCK1     ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) // Pin Controlled by PA25
#define AT91C_PA25_TWCK1    ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PA25_SCK2     ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PIO_PA26       ((unsigned int) 1 << 26) // Pin Controlled by PA26
#define AT91C_PA26_TD0      ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PA26_TCLK2    ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PIO_PA27       ((unsigned int) 1 << 27) // Pin Controlled by PA27
#define AT91C_PA27_RD0      ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PA27_PCK0     ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PIO_PA28       ((unsigned int) 1 << 28) // Pin Controlled by PA28
#define AT91C_PA28_TK0      ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PA28_PWMH0    ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PIO_PA29       ((unsigned int) 1 << 29) // Pin Controlled by PA29
#define AT91C_PA29_RK0      ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PA29_PWMH1    ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) // Pin Controlled by PA3
#define AT91C_PA3_MCI0_CK  ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PA3_PCK1     ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PIO_PA30       ((unsigned int) 1 << 30) // Pin Controlled by PA30
#define AT91C_PA30_TF0      ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PA30_TIOA2    ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PIO_PA31       ((unsigned int) 1 << 31) // Pin Controlled by PA31
#define AT91C_PA31_RF0      ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PA31_TIOB2    ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) // Pin Controlled by PA4
#define AT91C_PA4_MCI0_CDA ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PA4_PWMH0    ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) // Pin Controlled by PA5
#define AT91C_PA5_MCI0_DA0 ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PA5_PWMH1    ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) // Pin Controlled by PA6
#define AT91C_PA6_MCI0_DA1 ((unsigned int) AT91C_PIO_PA6) //  
#define AT91C_PA6_PWMH2    ((unsigned int) AT91C_PIO_PA6) //  
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) // Pin Controlled by PA7
#define AT91C_PA7_MCI0_DA2 ((unsigned int) AT91C_PIO_PA7) //  
#define AT91C_PA7_PWML0    ((unsigned int) AT91C_PIO_PA7) //  
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) // Pin Controlled by PA8
#define AT91C_PA8_MCI0_DA3 ((unsigned int) AT91C_PIO_PA8) //  
#define AT91C_PA8_PWML1    ((unsigned int) AT91C_PIO_PA8) //  
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) // Pin Controlled by PA9
#define AT91C_PA9_TWD0     ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PA9_PWML2    ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) // Pin Controlled by PB0
#define AT91C_PB0_PWMH0    ((unsigned int) AT91C_PIO_PB0) //  
#define AT91C_PB0_A2       ((unsigned int) AT91C_PIO_PB0) //  
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) // Pin Controlled by PB1
#define AT91C_PB1_PWMH1    ((unsigned int) AT91C_PIO_PB1) //  
#define AT91C_PB1_A3       ((unsigned int) AT91C_PIO_PB1) //  
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) // Pin Controlled by PB10
#define AT91C_PB10_D1       ((unsigned int) AT91C_PIO_PB10) //  
#define AT91C_PB10_DSR0     ((unsigned int) AT91C_PIO_PB10) //  
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) // Pin Controlled by PB11
#define AT91C_PB11_D2       ((unsigned int) AT91C_PIO_PB11) //  
#define AT91C_PB11_DCD0     ((unsigned int) AT91C_PIO_PB11) //  
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) // Pin Controlled by PB12
#define AT91C_PB12_D3       ((unsigned int) AT91C_PIO_PB12) //  
#define AT91C_PB12_RI0      ((unsigned int) AT91C_PIO_PB12) //  
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) // Pin Controlled by PB13
#define AT91C_PB13_D4       ((unsigned int) AT91C_PIO_PB13) //  
#define AT91C_PB13_PWMH0    ((unsigned int) AT91C_PIO_PB13) //  
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) // Pin Controlled by PB14
#define AT91C_PB14_D5       ((unsigned int) AT91C_PIO_PB14) //  
#define AT91C_PB14_PWMH1    ((unsigned int) AT91C_PIO_PB14) //  
#define AT91C_PIO_PB15       ((unsigned int) 1 << 15) // Pin Controlled by PB15
#define AT91C_PB15_D6       ((unsigned int) AT91C_PIO_PB15) //  
#define AT91C_PB15_PWMH2    ((unsigned int) AT91C_PIO_PB15) //  
#define AT91C_PIO_PB16       ((unsigned int) 1 << 16) // Pin Controlled by PB16
#define AT91C_PB16_D7       ((unsigned int) AT91C_PIO_PB16) //  
#define AT91C_PB16_PWMH3    ((unsigned int) AT91C_PIO_PB16) //  
#define AT91C_PIO_PB17       ((unsigned int) 1 << 17) // Pin Controlled by PB17
#define AT91C_PB17_NANDOE   ((unsigned int) AT91C_PIO_PB17) //  
#define AT91C_PB17_PWML0    ((unsigned int) AT91C_PIO_PB17) //  
#define AT91C_PIO_PB18       ((unsigned int) 1 << 18) // Pin Controlled by PB18
#define AT91C_PB18_NANDWE   ((unsigned int) AT91C_PIO_PB18) //  
#define AT91C_PB18_PWML1    ((unsigned int) AT91C_PIO_PB18) //  
#define AT91C_PIO_PB19       ((unsigned int) 1 << 19) // Pin Controlled by PB19
#define AT91C_PB19_NRD      ((unsigned int) AT91C_PIO_PB19) //  
#define AT91C_PB19_PWML2    ((unsigned int) AT91C_PIO_PB19) //  
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) // Pin Controlled by PB2
#define AT91C_PB2_PWMH2    ((unsigned int) AT91C_PIO_PB2) //  
#define AT91C_PB2_A4       ((unsigned int) AT91C_PIO_PB2) //  
#define AT91C_PIO_PB20       ((unsigned int) 1 << 20) // Pin Controlled by PB20
#define AT91C_PB20_NCS0     ((unsigned int) AT91C_PIO_PB20) //  
#define AT91C_PB20_PWML3    ((unsigned int) AT91C_PIO_PB20) //  
#define AT91C_PIO_PB21       ((unsigned int) 1 << 21) // Pin Controlled by PB21
#define AT91C_PB21_A21_NANDALE ((unsigned int) AT91C_PIO_PB21) //  
#define AT91C_PB21_RTS2     ((unsigned int) AT91C_PIO_PB21) //  
#define AT91C_PIO_PB22       ((unsigned int) 1 << 22) // Pin Controlled by PB22
#define AT91C_PB22_A22_NANDCLE ((unsigned int) AT91C_PIO_PB22) //  
#define AT91C_PB22_CTS2     ((unsigned int) AT91C_PIO_PB22) //  
#define AT91C_PIO_PB23       ((unsigned int) 1 << 23) // Pin Controlled by PB23
#define AT91C_PB23_NWR0_NWE ((unsigned int) AT91C_PIO_PB23) //  
#define AT91C_PB23_PCK2     ((unsigned int) AT91C_PIO_PB23) //  
#define AT91C_PIO_PB24       ((unsigned int) 1 << 24) // Pin Controlled by PB24
#define AT91C_PB24_NANDRDY  ((unsigned int) AT91C_PIO_PB24) //  
#define AT91C_PB24_PCK1     ((unsigned int) AT91C_PIO_PB24) //  
#define AT91C_PIO_PB25       ((unsigned int) 1 << 25) // Pin Controlled by PB25
#define AT91C_PB25_D8       ((unsigned int) AT91C_PIO_PB25) //  
#define AT91C_PB25_PWML0    ((unsigned int) AT91C_PIO_PB25) //  
#define AT91C_PIO_PB26       ((unsigned int) 1 << 26) // Pin Controlled by PB26
#define AT91C_PB26_D9       ((unsigned int) AT91C_PIO_PB26) //  
#define AT91C_PB26_PWML1    ((unsigned int) AT91C_PIO_PB26) //  
#define AT91C_PIO_PB27       ((unsigned int) 1 << 27) // Pin Controlled by PB27
#define AT91C_PB27_D10      ((unsigned int) AT91C_PIO_PB27) //  
#define AT91C_PB27_PWML2    ((unsigned int) AT91C_PIO_PB27) //  
#define AT91C_PIO_PB28       ((unsigned int) 1 << 28) // Pin Controlled by PB28
#define AT91C_PB28_D11      ((unsigned int) AT91C_PIO_PB28) //  
#define AT91C_PB28_PWML3    ((unsigned int) AT91C_PIO_PB28) //  
#define AT91C_PIO_PB29       ((unsigned int) 1 << 29) // Pin Controlled by PB29
#define AT91C_PB29_D12      ((unsigned int) AT91C_PIO_PB29) //  
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) // Pin Controlled by PB3
#define AT91C_PB3_PWMH3    ((unsigned int) AT91C_PIO_PB3) //  
#define AT91C_PB3_A5       ((unsigned int) AT91C_PIO_PB3) //  
#define AT91C_PIO_PB30       ((unsigned int) 1 << 30) // Pin Controlled by PB30
#define AT91C_PB30_D13      ((unsigned int) AT91C_PIO_PB30) //  
#define AT91C_PIO_PB31       ((unsigned int) 1 << 31) // Pin Controlled by PB31
#define AT91C_PB31_D14      ((unsigned int) AT91C_PIO_PB31) //  
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) // Pin Controlled by PB4
#define AT91C_PB4_TCLK1    ((unsigned int) AT91C_PIO_PB4) //  
#define AT91C_PB4_A6       ((unsigned int) AT91C_PIO_PB4) //  
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) // Pin Controlled by PB5
#define AT91C_PB5_TIOA1    ((unsigned int) AT91C_PIO_PB5) //  
#define AT91C_PB5_A7       ((unsigned int) AT91C_PIO_PB5) //  
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) // Pin Controlled by PB6
#define AT91C_PB6_TIOB1    ((unsigned int) AT91C_PIO_PB6) //  
#define AT91C_PB6_D15      ((unsigned int) AT91C_PIO_PB6) //  
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) // Pin Controlled by PB7
#define AT91C_PB7_RTS0     ((unsigned int) AT91C_PIO_PB7) //  
#define AT91C_PB7_A0_NBS0  ((unsigned int) AT91C_PIO_PB7) //  
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) // Pin Controlled by PB8
#define AT91C_PB8_CTS0     ((unsigned int) AT91C_PIO_PB8) //  
#define AT91C_PB8_A1       ((unsigned int) AT91C_PIO_PB8) //  
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) // Pin Controlled by PB9
#define AT91C_PB9_D0       ((unsigned int) AT91C_PIO_PB9) //  
#define AT91C_PB9_DTR0     ((unsigned int) AT91C_PIO_PB9) //  
#define AT91C_PIO_PC0        ((unsigned int) 1 <<  0) // Pin Controlled by PC0
#define AT91C_PC0_A2       ((unsigned int) AT91C_PIO_PC0) //  
#define AT91C_PIO_PC1        ((unsigned int) 1 <<  1) // Pin Controlled by PC1
#define AT91C_PC1_A3       ((unsigned int) AT91C_PIO_PC1) //  
#define AT91C_PIO_PC10       ((unsigned int) 1 << 10) // Pin Controlled by PC10
#define AT91C_PC10_A12      ((unsigned int) AT91C_PIO_PC10) //  
#define AT91C_PC10_CTS3     ((unsigned int) AT91C_PIO_PC10) //  
#define AT91C_PIO_PC11       ((unsigned int) 1 << 11) // Pin Controlled by PC11
#define AT91C_PC11_A13      ((unsigned int) AT91C_PIO_PC11) //  
#define AT91C_PC11_RTS3     ((unsigned int) AT91C_PIO_PC11) //  
#define AT91C_PIO_PC12       ((unsigned int) 1 << 12) // Pin Controlled by PC12
#define AT91C_PC12_NCS1     ((unsigned int) AT91C_PIO_PC12) //  
#define AT91C_PC12_TXD3     ((unsigned int) AT91C_PIO_PC12) //  
#define AT91C_PIO_PC13       ((unsigned int) 1 << 13) // Pin Controlled by PC13
#define AT91C_PC13_A2       ((unsigned int) AT91C_PIO_PC13) //  
#define AT91C_PC13_RXD3     ((unsigned int) AT91C_PIO_PC13) //  
#define AT91C_PIO_PC14       ((unsigned int) 1 << 14) // Pin Controlled by PC14
#define AT91C_PC14_A3       ((unsigned int) AT91C_PIO_PC14) //  
#define AT91C_PC14_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PC14) //  
#define AT91C_PIO_PC15       ((unsigned int) 1 << 15) // Pin Controlled by PC15
#define AT91C_PC15_NWR1_NBS1 ((unsigned int) AT91C_PIO_PC15) //  
#define AT91C_PIO_PC16       ((unsigned int) 1 << 16) // Pin Controlled by PC16
#define AT91C_PC16_NCS2     ((unsigned int) AT91C_PIO_PC16) //  
#define AT91C_PC16_PWML3    ((unsigned int) AT91C_PIO_PC16) //  
#define AT91C_PIO_PC17       ((unsigned int) 1 << 17) // Pin Controlled by PC17
#define AT91C_PC17_NCS3     ((unsigned int) AT91C_PIO_PC17) //  
#define AT91C_PC17_A24      ((unsigned int) AT91C_PIO_PC17) //  
#define AT91C_PIO_PC18       ((unsigned int) 1 << 18) // Pin Controlled by PC18
#define AT91C_PC18_NWAIT    ((unsigned int) AT91C_PIO_PC18) //  
#define AT91C_PIO_PC19       ((unsigned int) 1 << 19) // Pin Controlled by PC19
#define AT91C_PC19_SCK3     ((unsigned int) AT91C_PIO_PC19) //  
#define AT91C_PC19_NPCS1    ((unsigned int) AT91C_PIO_PC19) //  
#define AT91C_PIO_PC2        ((unsigned int) 1 <<  2) // Pin Controlled by PC2
#define AT91C_PC2_A4       ((unsigned int) AT91C_PIO_PC2) //  
#define AT91C_PIO_PC20       ((unsigned int) 1 << 20) // Pin Controlled by PC20
#define AT91C_PC20_A14      ((unsigned int) AT91C_PIO_PC20) //  
#define AT91C_PIO_PC21       ((unsigned int) 1 << 21) // Pin Controlled by PC21
#define AT91C_PC21_A15      ((unsigned int) AT91C_PIO_PC21) //  
#define AT91C_PIO_PC22       ((unsigned int) 1 << 22) // Pin Controlled by PC22
#define AT91C_PC22_A16      ((unsigned int) AT91C_PIO_PC22) //  
#define AT91C_PIO_PC23       ((unsigned int) 1 << 23) // Pin Controlled by PC23
#define AT91C_PC23_A17      ((unsigned int) AT91C_PIO_PC23) //  
#define AT91C_PIO_PC24       ((unsigned int) 1 << 24) // Pin Controlled by PC24
#define AT91C_PC24_A18      ((unsigned int) AT91C_PIO_PC24) //  
#define AT91C_PC24_PWMH0    ((unsigned int) AT91C_PIO_PC24) //  
#define AT91C_PIO_PC25       ((unsigned int) 1 << 25) // Pin Controlled by PC25
#define AT91C_PC25_A19      ((unsigned int) AT91C_PIO_PC25) //  
#define AT91C_PC25_PWMH1    ((unsigned int) AT91C_PIO_PC25) //  
#define AT91C_PIO_PC26       ((unsigned int) 1 << 26) // Pin Controlled by PC26
#define AT91C_PC26_A20      ((unsigned int) AT91C_PIO_PC26) //  
#define AT91C_PC26_PWMH2    ((unsigned int) AT91C_PIO_PC26) //  
#define AT91C_PIO_PC27       ((unsigned int) 1 << 27) // Pin Controlled by PC27
#define AT91C_PC27_A23      ((unsigned int) AT91C_PIO_PC27) //  
#define AT91C_PC27_PWMH3    ((unsigned int) AT91C_PIO_PC27) //  
#define AT91C_PIO_PC28       ((unsigned int) 1 << 28) // Pin Controlled by PC28
#define AT91C_PC28_A24      ((unsigned int) AT91C_PIO_PC28) //  
#define AT91C_PC28_MCI0_DA4 ((unsigned int) AT91C_PIO_PC28) //  
#define AT91C_PIO_PC29       ((unsigned int) 1 << 29) // Pin Controlled by PC29
#define AT91C_PC29_PWML0    ((unsigned int) AT91C_PIO_PC29) //  
#define AT91C_PC29_MCI0_DA5 ((unsigned int) AT91C_PIO_PC29) //  
#define AT91C_PIO_PC3        ((unsigned int) 1 <<  3) // Pin Controlled by PC3
#define AT91C_PC3_A5       ((unsigned int) AT91C_PIO_PC3) //  
#define AT91C_PC3_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PC3) //  
#define AT91C_PIO_PC30       ((unsigned int) 1 << 30) // Pin Controlled by PC30
#define AT91C_PC30_PWML1    ((unsigned int) AT91C_PIO_PC30) //  
#define AT91C_PC30_MCI0_DA6 ((unsigned int) AT91C_PIO_PC30) //  
#define AT91C_PIO_PC31       ((unsigned int) 1 << 31) // Pin Controlled by PC31
#define AT91C_PC31_PWML2    ((unsigned int) AT91C_PIO_PC31) //  
#define AT91C_PC31_MCI0_DA7 ((unsigned int) AT91C_PIO_PC31) //  
#define AT91C_PIO_PC4        ((unsigned int) 1 <<  4) // Pin Controlled by PC4
#define AT91C_PC4_A6       ((unsigned int) AT91C_PIO_PC4) //  
#define AT91C_PC4_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PC4) //  
#define AT91C_PIO_PC5        ((unsigned int) 1 <<  5) // Pin Controlled by PC5
#define AT91C_PC5_A7       ((unsigned int) AT91C_PIO_PC5) //  
#define AT91C_PC5_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PC5) //  
#define AT91C_PIO_PC6        ((unsigned int) 1 <<  6) // Pin Controlled by PC6
#define AT91C_PC6_A8       ((unsigned int) AT91C_PIO_PC6) //  
#define AT91C_PC6_PWML0    ((unsigned int) AT91C_PIO_PC6) //  
#define AT91C_PIO_PC7        ((unsigned int) 1 <<  7) // Pin Controlled by PC7
#define AT91C_PC7_A9       ((unsigned int) AT91C_PIO_PC7) //  
#define AT91C_PC7_PWML1    ((unsigned int) AT91C_PIO_PC7) //  
#define AT91C_PIO_PC8        ((unsigned int) 1 <<  8) // Pin Controlled by PC8
#define AT91C_PC8_A10      ((unsigned int) AT91C_PIO_PC8) //  
#define AT91C_PC8_PWML2    ((unsigned int) AT91C_PIO_PC8) //  
#define AT91C_PIO_PC9        ((unsigned int) 1 <<  9) // Pin Controlled by PC9
#define AT91C_PC9_A11      ((unsigned int) AT91C_PIO_PC9) //  
#define AT91C_PC9_PWML3    ((unsigned int) AT91C_PIO_PC9) //  

// *****************************************************************************
//               PERIPHERAL ID DEFINITIONS FOR AT91SAM3U1
// *****************************************************************************
#define AT91C_ID_SUPC   ((unsigned int)  0) // SUPPLY CONTROLLER
#define AT91C_ID_RSTC   ((unsigned int)  1) // RESET CONTROLLER
#define AT91C_ID_RTC    ((unsigned int)  2) // REAL TIME CLOCK
#define AT91C_ID_RTT    ((unsigned int)  3) // REAL TIME TIMER
#define AT91C_ID_WDG    ((unsigned int)  4) // WATCHDOG TIMER
#define AT91C_ID_PMC    ((unsigned int)  5) // PMC
#define AT91C_ID_EFC0   ((unsigned int)  6) // EFC0
#define AT91C_ID_EFC1   ((unsigned int)  7) // EFC1
#define AT91C_ID_DBGU   ((unsigned int)  8) // DBGU
#define AT91C_ID_HSMC4  ((unsigned int)  9) // HSMC4
#define AT91C_ID_PIOA   ((unsigned int) 10) // Parallel IO Controller A
#define AT91C_ID_PIOB   ((unsigned int) 11) // Parallel IO Controller B
#define AT91C_ID_PIOC   ((unsigned int) 12) // Parallel IO Controller C
#define AT91C_ID_US0    ((unsigned int) 13) // USART 0
#define AT91C_ID_US1    ((unsigned int) 14) // USART 1
#define AT91C_ID_US2    ((unsigned int) 15) // USART 2
#define AT91C_ID_US3    ((unsigned int) 16) // USART 3
#define AT91C_ID_MCI0   ((unsigned int) 17) // Multimedia Card Interface
#define AT91C_ID_TWI0   ((unsigned int) 18) // TWI 0
#define AT91C_ID_TWI1   ((unsigned int) 19) // TWI 1
#define AT91C_ID_SPI0   ((unsigned int) 20) // Serial Peripheral Interface
#define AT91C_ID_SSC0   ((unsigned int) 21) // Serial Synchronous Controller 0
#define AT91C_ID_TC0    ((unsigned int) 22) // Timer Counter 0
#define AT91C_ID_TC1    ((unsigned int) 23) // Timer Counter 1
#define AT91C_ID_TC2    ((unsigned int) 24) // Timer Counter 2
#define AT91C_ID_PWMC   ((unsigned int) 25) // Pulse Width Modulation Controller
#define AT91C_ID_ADC    ((unsigned int) 27) // ADC controller
#define AT91C_ID_HDMA   ((unsigned int) 28) // HDMA
#define AT91C_ID_UDPHS  ((unsigned int) 29) // USB Device High Speed
#define AT91C_ALL_INT   ((unsigned int) 0x3BFFFFFF) // ALL VALID INTERRUPTS

// *****************************************************************************
//               BASE ADDRESS DEFINITIONS FOR AT91SAM3U1
// *****************************************************************************
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0x400E0000) // (SYS) Base Address
#define AT91C_BASE_HSMC4_CS0 ((AT91PS_HSMC4_CS) 	0x400E0070) // (HSMC4_CS0) Base Address
#define AT91C_BASE_HSMC4_CS1 ((AT91PS_HSMC4_CS) 	0x400E0084) // (HSMC4_CS1) Base Address
#define AT91C_BASE_HSMC4_CS2 ((AT91PS_HSMC4_CS) 	0x400E0098) // (HSMC4_CS2) Base Address
#define AT91C_BASE_HSMC4_CS3 ((AT91PS_HSMC4_CS) 	0x400E00AC) // (HSMC4_CS3) Base Address
#define AT91C_BASE_HSMC4_NFC ((AT91PS_HSMC4_CS) 	0x400E00FC) // (HSMC4_NFC) Base Address
#define AT91C_BASE_HSMC4     ((AT91PS_HSMC4) 	0x400E0000) // (HSMC4) Base Address
#define AT91C_BASE_MATRIX    ((AT91PS_HMATRIX2) 	0x400E0200) // (MATRIX) Base Address
#define AT91C_BASE_NVIC      ((AT91PS_NVIC) 	0xE000E000) // (NVIC) Base Address
#define AT91C_BASE_MPU       ((AT91PS_MPU) 	0xE000ED90) // (MPU) Base Address
#define AT91C_BASE_CM3       ((AT91PS_CM3) 	0xE000ED00) // (CM3) Base Address
#define AT91C_BASE_PDC_DBGU  ((AT91PS_PDC) 	0x400E0700) // (PDC_DBGU) Base Address
#define AT91C_BASE_DBGU      ((AT91PS_DBGU) 	0x400E0600) // (DBGU) Base Address
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0x400E0C00) // (PIOA) Base Address
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0x400E0E00) // (PIOB) Base Address
#define AT91C_BASE_PIOC      ((AT91PS_PIO) 	0x400E1000) // (PIOC) Base Address
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0x400E0400) // (PMC) Base Address
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0x400E041C) // (CKGR) Base Address
#define AT91C_BASE_RSTC      ((AT91PS_RSTC) 	0x400E1200) // (RSTC) Base Address
#define AT91C_BASE_SUPC      ((AT91PS_SUPC) 	0x400E1210) // (SUPC) Base Address
#define AT91C_BASE_RTTC      ((AT91PS_RTTC) 	0x400E1230) // (RTTC) Base Address
#define AT91C_BASE_WDTC      ((AT91PS_WDTC) 	0x400E1250) // (WDTC) Base Address
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0x400E1260) // (RTC) Base Address
#define AT91C_BASE_ADC0      ((AT91PS_ADC) 	0x400AC000) // (ADC0) Base Address
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0x40080000) // (TC0) Base Address
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0x40080040) // (TC1) Base Address
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0x40080080) // (TC2) Base Address
#define AT91C_BASE_TCB0      ((AT91PS_TCB) 	0x40080000) // (TCB0) Base Address
#define AT91C_BASE_TCB1      ((AT91PS_TCB) 	0x40080040) // (TCB1) Base Address
#define AT91C_BASE_TCB2      ((AT91PS_TCB) 	0x40080080) // (TCB2) Base Address
#define AT91C_BASE_EFC0      ((AT91PS_EFC) 	0x400E0800) // (EFC0) Base Address
#define AT91C_BASE_EFC1      ((AT91PS_EFC) 	0x400E0A00) // (EFC1) Base Address
#define AT91C_BASE_MCI0      ((AT91PS_MCI) 	0x40000000) // (MCI0) Base Address
#define AT91C_BASE_PDC_TWI0  ((AT91PS_PDC) 	0x40084100) // (PDC_TWI0) Base Address
#define AT91C_BASE_PDC_TWI1  ((AT91PS_PDC) 	0x40088100) // (PDC_TWI1) Base Address
#define AT91C_BASE_TWI0      ((AT91PS_TWI) 	0x40084000) // (TWI0) Base Address
#define AT91C_BASE_TWI1      ((AT91PS_TWI) 	0x40088000) // (TWI1) Base Address
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0x40090100) // (PDC_US0) Base Address
#define AT91C_BASE_US0       ((AT91PS_USART) 	0x40090000) // (US0) Base Address
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0x40094100) // (PDC_US1) Base Address
#define AT91C_BASE_US1       ((AT91PS_USART) 	0x40094000) // (US1) Base Address
#define AT91C_BASE_PDC_US2   ((AT91PS_PDC) 	0x40098100) // (PDC_US2) Base Address
#define AT91C_BASE_US2       ((AT91PS_USART) 	0x40098000) // (US2) Base Address
#define AT91C_BASE_PDC_US3   ((AT91PS_PDC) 	0x4009C100) // (PDC_US3) Base Address
#define AT91C_BASE_US3       ((AT91PS_USART) 	0x4009C000) // (US3) Base Address
#define AT91C_BASE_PDC_SSC0  ((AT91PS_PDC) 	0x40004100) // (PDC_SSC0) Base Address
#define AT91C_BASE_SSC0      ((AT91PS_SSC) 	0x40004000) // (SSC0) Base Address
#define AT91C_BASE_PDC_PWMC  ((AT91PS_PDC) 	0x4008C100) // (PDC_PWMC) Base Address
#define AT91C_BASE_PWMC_CH0  ((AT91PS_PWMC_CH) 	0x4008C200) // (PWMC_CH0) Base Address
#define AT91C_BASE_PWMC_CH1  ((AT91PS_PWMC_CH) 	0x4008C220) // (PWMC_CH1) Base Address
#define AT91C_BASE_PWMC_CH2  ((AT91PS_PWMC_CH) 	0x4008C240) // (PWMC_CH2) Base Address
#define AT91C_BASE_PWMC_CH3  ((AT91PS_PWMC_CH) 	0x4008C260) // (PWMC_CH3) Base Address
#define AT91C_BASE_PWMC      ((AT91PS_PWMC) 	0x4008C000) // (PWMC) Base Address
#define AT91C_BASE_SPI0      ((AT91PS_SPI) 	0x40008000) // (SPI0) Base Address
#define AT91C_BASE_UDPHS_EPTFIFO ((AT91PS_UDPHS_EPTFIFO) 	0x20180000) // (UDPHS_EPTFIFO) Base Address
#define AT91C_BASE_UDPHS_EPT_0 ((AT91PS_UDPHS_EPT) 	0x400A4100) // (UDPHS_EPT_0) Base Address
#define AT91C_BASE_UDPHS_EPT_1 ((AT91PS_UDPHS_EPT) 	0x400A4120) // (UDPHS_EPT_1) Base Address
#define AT91C_BASE_UDPHS_EPT_2 ((AT91PS_UDPHS_EPT) 	0x400A4140) // (UDPHS_EPT_2) Base Address
#define AT91C_BASE_UDPHS_EPT_3 ((AT91PS_UDPHS_EPT) 	0x400A4160) // (UDPHS_EPT_3) Base Address
#define AT91C_BASE_UDPHS_EPT_4 ((AT91PS_UDPHS_EPT) 	0x400A4180) // (UDPHS_EPT_4) Base Address
#define AT91C_BASE_UDPHS_EPT_5 ((AT91PS_UDPHS_EPT) 	0x400A41A0) // (UDPHS_EPT_5) Base Address
#define AT91C_BASE_UDPHS_EPT_6 ((AT91PS_UDPHS_EPT) 	0x400A41C0) // (UDPHS_EPT_6) Base Address
#define AT91C_BASE_UDPHS_DMA_1 ((AT91PS_UDPHS_DMA) 	0x400A4310) // (UDPHS_DMA_1) Base Address
#define AT91C_BASE_UDPHS_DMA_2 ((AT91PS_UDPHS_DMA) 	0x400A4320) // (UDPHS_DMA_2) Base Address
#define AT91C_BASE_UDPHS_DMA_3 ((AT91PS_UDPHS_DMA) 	0x400A4330) // (UDPHS_DMA_3) Base Address
#define AT91C_BASE_UDPHS_DMA_4 ((AT91PS_UDPHS_DMA) 	0x400A4340) // (UDPHS_DMA_4) Base Address
#define AT91C_BASE_UDPHS_DMA_5 ((AT91PS_UDPHS_DMA) 	0x400A4350) // (UDPHS_DMA_5) Base Address
#define AT91C_BASE_UDPHS_DMA_6 ((AT91PS_UDPHS_DMA) 	0x400A4360) // (UDPHS_DMA_6) Base Address
#define AT91C_BASE_UDPHS     ((AT91PS_UDPHS) 	0x400A4000) // (UDPHS) Base Address
#define AT91C_BASE_HDMA_CH_0 ((AT91PS_HDMA_CH) 	0x400B003C) // (HDMA_CH_0) Base Address
#define AT91C_BASE_HDMA_CH_1 ((AT91PS_HDMA_CH) 	0x400B0064) // (HDMA_CH_1) Base Address
#define AT91C_BASE_HDMA_CH_2 ((AT91PS_HDMA_CH) 	0x400B008C) // (HDMA_CH_2) Base Address
#define AT91C_BASE_HDMA_CH_3 ((AT91PS_HDMA_CH) 	0x400B00B4) // (HDMA_CH_3) Base Address
#define AT91C_BASE_HDMA      ((AT91PS_HDMA) 	0x400B0000) // (HDMA) Base Address

// *****************************************************************************
//               MEMORY MAPPING DEFINITIONS FOR AT91SAM3U1
// *****************************************************************************
// ITCM
#define AT91C_ITCM 	 ((char *) 	0x00100000) // Maximum ITCM Area base address
#define AT91C_ITCM_SIZE	 ((unsigned int) 0x00010000) // Maximum ITCM Area size in byte (64 Kbytes)
// DTCM
#define AT91C_DTCM 	 ((char *) 	0x00200000) // Maximum DTCM Area base address
#define AT91C_DTCM_SIZE	 ((unsigned int) 0x00010000) // Maximum DTCM Area size in byte (64 Kbytes)
// IRAM
#define AT91C_IRAM 	 ((char *) 	0x20000000) // Maximum Internal SRAM base address
#define AT91C_IRAM_SIZE	 ((unsigned int) 0x00002000) // Maximum Internal SRAM size in byte (8 Kbytes)
// IRAM_MIN
#define AT91C_IRAM_MIN	 ((char *) 	0x00300000) // Minimum Internal RAM base address
#define AT91C_IRAM_MIN_SIZE	 ((unsigned int) 0x00004000) // Minimum Internal RAM size in byte (16 Kbytes)
// IROM
#define AT91C_IROM 	 ((char *) 	0x00180000) // Internal ROM base address
#define AT91C_IROM_SIZE	 ((unsigned int) 0x00008000) // Internal ROM size in byte (32 Kbytes)
// IFLASH0
#define AT91C_IFLASH0	 ((char *) 	0x00080000) // Maximum IFLASH Area : 64Kbyte base address
#define AT91C_IFLASH0_SIZE	 ((unsigned int) 0x00010000) // Maximum IFLASH Area : 64Kbyte size in byte (64 Kbytes)
#define AT91C_IFLASH0_PAGE_SIZE	 ((unsigned int) 256) // Maximum IFLASH Area : 64Kbyte Page Size: 256 bytes
#define AT91C_IFLASH0_LOCK_REGION_SIZE	 ((unsigned int) 8192) // Maximum IFLASH Area : 64Kbyte Lock Region Size: 8 Kbytes
#define AT91C_IFLASH0_NB_OF_PAGES	 ((unsigned int) 512) // Maximum IFLASH Area : 64Kbyte Number of Pages: 512 bytes
#define AT91C_IFLASH0_NB_OF_LOCK_BITS	 ((unsigned int) 16) // Maximum IFLASH Area : 64Kbyte Number of Lock Bits: 16 bytes

// EBI_CS0
#define AT91C_EBI_CS0	 ((char *) 	0x10000000) // EBI Chip Select 0 base address
#define AT91C_EBI_CS0_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 0 size in byte (262144 Kbytes)
// EBI_CS1
#define AT91C_EBI_CS1	 ((char *) 	0x20000000) // EBI Chip Select 1 base address
#define AT91C_EBI_CS1_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 1 size in byte (262144 Kbytes)
// EBI_SDRAM
#define AT91C_EBI_SDRAM	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_SIZE	 ((unsigned int) 0x10000000) // SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// EBI_SDRAM_16BIT
#define AT91C_EBI_SDRAM_16BIT	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_16BIT_SIZE	 ((unsigned int) 0x02000000) // SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// EBI_SDRAM_32BIT
#define AT91C_EBI_SDRAM_32BIT	 ((char *) 	0x20000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_32BIT_SIZE	 ((unsigned int) 0x04000000) // SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
// EBI_CS2
#define AT91C_EBI_CS2	 ((char *) 	0x30000000) // EBI Chip Select 2 base address
#define AT91C_EBI_CS2_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 2 size in byte (262144 Kbytes)
// EBI_CS3
#define AT91C_EBI_CS3	 ((char *) 	0x40000000) // EBI Chip Select 3 base address
#define AT91C_EBI_CS3_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 3 size in byte (262144 Kbytes)
// EBI_SM
#define AT91C_EBI_SM	 ((char *) 	0x40000000) // NANDFLASH on EBI Chip Select 3 base address
#define AT91C_EBI_SM_SIZE	 ((unsigned int) 0x10000000) // NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)
// EBI_CS4
#define AT91C_EBI_CS4	 ((char *) 	0x50000000) // EBI Chip Select 4 base address
#define AT91C_EBI_CS4_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CF0
#define AT91C_EBI_CF0	 ((char *) 	0x50000000) // CompactFlash 0 on EBI Chip Select 4 base address
#define AT91C_EBI_CF0_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CS5
#define AT91C_EBI_CS5	 ((char *) 	0x60000000) // EBI Chip Select 5 base address
#define AT91C_EBI_CS5_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 5 size in byte (262144 Kbytes)
// EBI_CF1
#define AT91C_EBI_CF1	 ((char *) 	0x60000000) // CompactFlash 1 on EBIChip Select 5 base address
#define AT91C_EBI_CF1_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

// - Hardware register definition

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR System Peripherals
// - *****************************************************************************
// - -------- GPBR : (SYS Offset: 0x1290) GPBR General Purpose Register -------- 
AT91C_GPBR_GPRV           EQU (0x0 <<  0) ;- (SYS) General Purpose Register Value

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR HSMC4 Chip Select interface
// - *****************************************************************************
// - -------- HSMC4_SETUP : (HSMC4_CS Offset: 0x0) HSMC4 SETUP -------- 
AT91C_HSMC4_NWE_SETUP     EQU (0x3F <<  0) ;- (HSMC4_CS) NWE Setup length
AT91C_HSMC4_NCS_WR_SETUP  EQU (0x3F <<  8) ;- (HSMC4_CS) NCS Setup length in Write access
AT91C_HSMC4_NRD_SETUP     EQU (0x3F << 16) ;- (HSMC4_CS) NRD Setup length
AT91C_HSMC4_NCS_RD_SETUP  EQU (0x3F << 24) ;- (HSMC4_CS) NCS Setup legnth in Read access
// - -------- HSMC4_PULSE : (HSMC4_CS Offset: 0x4) HSMC4 PULSE -------- 
AT91C_HSMC4_NWE_PULSE     EQU (0x3F <<  0) ;- (HSMC4_CS) NWE Pulse Length
AT91C_HSMC4_NCS_WR_PULSE  EQU (0x3F <<  8) ;- (HSMC4_CS) NCS Pulse length in WRITE access
AT91C_HSMC4_NRD_PULSE     EQU (0x3F << 16) ;- (HSMC4_CS) NRD Pulse length
AT91C_HSMC4_NCS_RD_PULSE  EQU (0x3F << 24) ;- (HSMC4_CS) NCS Pulse length in READ access
// - -------- HSMC4_CYCLE : (HSMC4_CS Offset: 0x8) HSMC4 CYCLE -------- 
AT91C_HSMC4_NWE_CYCLE     EQU (0x1FF <<  0) ;- (HSMC4_CS) Total Write Cycle Length
AT91C_HSMC4_NRD_CYCLE     EQU (0x1FF << 16) ;- (HSMC4_CS) Total Read Cycle Length
// - -------- HSMC4_TIMINGS : (HSMC4_CS Offset: 0xc) HSMC4 TIMINGS -------- 
AT91C_HSMC4_TCLR          EQU (0xF <<  0) ;- (HSMC4_CS) CLE to REN low delay
AT91C_HSMC4_TADL          EQU (0xF <<  4) ;- (HSMC4_CS) ALE to data start
AT91C_HSMC4_TAR           EQU (0xF <<  8) ;- (HSMC4_CS) ALE to REN low delay
AT91C_HSMC4_OCMSEN        EQU (0x1 << 12) ;- (HSMC4_CS) Off Chip Memory Scrambling Enable
AT91C_HSMC4_TRR           EQU (0xF << 16) ;- (HSMC4_CS) Ready to REN low delay
AT91C_HSMC4_TWB           EQU (0xF << 24) ;- (HSMC4_CS) WEN high to REN to busy
AT91C_HSMC4_RBNSEL        EQU (0x7 << 28) ;- (HSMC4_CS) Ready/Busy Line Selection
AT91C_HSMC4_NFSEL         EQU (0x1 << 31) ;- (HSMC4_CS) Nand Flash Selection
// - -------- HSMC4_MODE : (HSMC4_CS Offset: 0x10) HSMC4 MODE -------- 
AT91C_HSMC4_READ_MODE     EQU (0x1 <<  0) ;- (HSMC4_CS) Read Mode
AT91C_HSMC4_WRITE_MODE    EQU (0x1 <<  1) ;- (HSMC4_CS) Write Mode
AT91C_HSMC4_EXNW_MODE     EQU (0x3 <<  4) ;- (HSMC4_CS) NWAIT Mode
AT91C_HSMC4_EXNW_MODE_NWAIT_DISABLE EQU (0x0 <<  4) ;- (HSMC4_CS) External NWAIT disabled.
AT91C_HSMC4_EXNW_MODE_NWAIT_ENABLE_FROZEN EQU (0x2 <<  4) ;- (HSMC4_CS) External NWAIT enabled in frozen mode.
AT91C_HSMC4_EXNW_MODE_NWAIT_ENABLE_READY EQU (0x3 <<  4) ;- (HSMC4_CS) External NWAIT enabled in ready mode.
AT91C_HSMC4_BAT           EQU (0x1 <<  8) ;- (HSMC4_CS) Byte Access Type
AT91C_HSMC4_BAT_BYTE_SELECT EQU (0x0 <<  8) ;- (HSMC4_CS) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
AT91C_HSMC4_BAT_BYTE_WRITE EQU (0x1 <<  8) ;- (HSMC4_CS) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
AT91C_HSMC4_DBW           EQU (0x3 << 12) ;- (HSMC4_CS) Data Bus Width
AT91C_HSMC4_DBW_WIDTH_EIGTH_BITS EQU (0x0 << 12) ;- (HSMC4_CS) 8 bits.
AT91C_HSMC4_DBW_WIDTH_SIXTEEN_BITS EQU (0x1 << 12) ;- (HSMC4_CS) 16 bits.
AT91C_HSMC4_DBW_WIDTH_THIRTY_TWO_BITS EQU (0x2 << 12) ;- (HSMC4_CS) 32 bits.
AT91C_HSMC4_TDF_CYCLES    EQU (0xF << 16) ;- (HSMC4_CS) Data Float Time.
AT91C_HSMC4_TDF_MODE      EQU (0x1 << 20) ;- (HSMC4_CS) TDF Enabled.
AT91C_HSMC4_PMEN          EQU (0x1 << 24) ;- (HSMC4_CS) Page Mode Enabled.
AT91C_HSMC4_PS            EQU (0x3 << 28) ;- (HSMC4_CS) Page Size
AT91C_HSMC4_PS_SIZE_FOUR_BYTES EQU (0x0 << 28) ;- (HSMC4_CS) 4 bytes.
AT91C_HSMC4_PS_SIZE_EIGHT_BYTES EQU (0x1 << 28) ;- (HSMC4_CS) 8 bytes.
AT91C_HSMC4_PS_SIZE_SIXTEEN_BYTES EQU (0x2 << 28) ;- (HSMC4_CS) 16 bytes.
AT91C_HSMC4_PS_SIZE_THIRTY_TWO_BYTES EQU (0x3 << 28) ;- (HSMC4_CS) 32 bytes.

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR AHB Static Memory Controller 4 Interface
// - *****************************************************************************
// - -------- HSMC4_CFG : (HSMC4 Offset: 0x0) Configuration Register -------- 
AT91C_HSMC4_PAGESIZE      EQU (0x3 <<  0) ;- (HSMC4) PAGESIZE field description
AT91C_HSMC4_PAGESIZE_528_Bytes EQU (0x0) ;- (HSMC4) 512 bytes plus 16 bytes page size
AT91C_HSMC4_PAGESIZE_1056_Bytes EQU (0x1) ;- (HSMC4) 1024 bytes plus 32 bytes page size
AT91C_HSMC4_PAGESIZE_2112_Bytes EQU (0x2) ;- (HSMC4) 2048 bytes plus 64 bytes page size
AT91C_HSMC4_PAGESIZE_4224_Bytes EQU (0x3) ;- (HSMC4) 4096 bytes plus 128 bytes page size
AT91C_HSMC4_WSPARE        EQU (0x1 <<  8) ;- (HSMC4) Spare area access in Write Mode
AT91C_HSMC4_RSPARE        EQU (0x1 <<  9) ;- (HSMC4) Spare area access in Read Mode
AT91C_HSMC4_EDGECTRL      EQU (0x1 << 12) ;- (HSMC4) Rising/Falling Edge Detection Control
AT91C_HSMC4_RBEDGE        EQU (0x1 << 13) ;- (HSMC4) Ready/Busy Signal edge Detection
AT91C_HSMC4_DTOCYC        EQU (0xF << 16) ;- (HSMC4) Data Timeout Cycle Number
AT91C_HSMC4_DTOMUL        EQU (0x7 << 20) ;- (HSMC4) Data Timeout Multiplier
AT91C_HSMC4_DTOMUL_1      EQU (0x0 << 20) ;- (HSMC4) DTOCYC x 1
AT91C_HSMC4_DTOMUL_16     EQU (0x1 << 20) ;- (HSMC4) DTOCYC x 16
AT91C_HSMC4_DTOMUL_128    EQU (0x2 << 20) ;- (HSMC4) DTOCYC x 128
AT91C_HSMC4_DTOMUL_256    EQU (0x3 << 20) ;- (HSMC4) DTOCYC x 256
AT91C_HSMC4_DTOMUL_1024   EQU (0x4 << 20) ;- (HSMC4) DTOCYC x 1024
AT91C_HSMC4_DTOMUL_4096   EQU (0x5 << 20) ;- (HSMC4) DTOCYC x 4096
AT91C_HSMC4_DTOMUL_65536  EQU (0x6 << 20) ;- (HSMC4) DTOCYC x 65536
AT91C_HSMC4_DTOMUL_1048576 EQU (0x7 << 20) ;- (HSMC4) DTOCYC x 1048576
// - -------- HSMC4_CTRL : (HSMC4 Offset: 0x4) Control Register -------- 
AT91C_HSMC4_NFCEN         EQU (0x1 <<  0) ;- (HSMC4) Nand Flash Controller Host Enable
AT91C_HSMC4_NFCDIS        EQU (0x1 <<  1) ;- (HSMC4) Nand Flash Controller Host Disable
AT91C_HSMC4_HOSTEN        EQU (0x1 <<  8) ;- (HSMC4) If set to one, the Host controller is activated and perform a data transfer phase.
AT91C_HSMC4_HOSTWR        EQU (0x1 << 11) ;- (HSMC4) If this field is set to one, the host transfers data from the internal SRAM to the Memory Device.
AT91C_HSMC4_HOSTCSID      EQU (0x7 << 12) ;- (HSMC4) Host Controller Chip select Id
AT91C_HSMC4_HOSTCSID_0    EQU (0x0 << 12) ;- (HSMC4) CS0
AT91C_HSMC4_HOSTCSID_1    EQU (0x1 << 12) ;- (HSMC4) CS1
AT91C_HSMC4_HOSTCSID_2    EQU (0x2 << 12) ;- (HSMC4) CS2
AT91C_HSMC4_HOSTCSID_3    EQU (0x3 << 12) ;- (HSMC4) CS3
AT91C_HSMC4_HOSTCSID_4    EQU (0x4 << 12) ;- (HSMC4) CS4
AT91C_HSMC4_HOSTCSID_5    EQU (0x5 << 12) ;- (HSMC4) CS5
AT91C_HSMC4_HOSTCSID_6    EQU (0x6 << 12) ;- (HSMC4) CS6
AT91C_HSMC4_HOSTCSID_7    EQU (0x7 << 12) ;- (HSMC4) CS7
AT91C_HSMC4_VALID         EQU (0x1 << 15) ;- (HSMC4) When set to 1, a write operation modifies both HOSTCSID and HOSTWR fields.
// - -------- HSMC4_SR : (HSMC4 Offset: 0x8) HSMC4 Status Register -------- 
AT91C_HSMC4_NFCSTS        EQU (0x1 <<  0) ;- (HSMC4) Nand Flash Controller status
AT91C_HSMC4_RBRISE        EQU (0x1 <<  4) ;- (HSMC4) Selected Ready Busy Rising Edge Detected flag
AT91C_HSMC4_RBFALL        EQU (0x1 <<  5) ;- (HSMC4) Selected Ready Busy Falling Edge Detected flag
AT91C_HSMC4_HOSTBUSY      EQU (0x1 <<  8) ;- (HSMC4) Host Busy
AT91C_HSMC4_HOSTW         EQU (0x1 << 11) ;- (HSMC4) Host Write/Read Operation
AT91C_HSMC4_HOSTCS        EQU (0x7 << 12) ;- (HSMC4) Host Controller Chip select Id
AT91C_HSMC4_HOSTCS_0      EQU (0x0 << 12) ;- (HSMC4) CS0
AT91C_HSMC4_HOSTCS_1      EQU (0x1 << 12) ;- (HSMC4) CS1
AT91C_HSMC4_HOSTCS_2      EQU (0x2 << 12) ;- (HSMC4) CS2
AT91C_HSMC4_HOSTCS_3      EQU (0x3 << 12) ;- (HSMC4) CS3
AT91C_HSMC4_HOSTCS_4      EQU (0x4 << 12) ;- (HSMC4) CS4
AT91C_HSMC4_HOSTCS_5      EQU (0x5 << 12) ;- (HSMC4) CS5
AT91C_HSMC4_HOSTCS_6      EQU (0x6 << 12) ;- (HSMC4) CS6
AT91C_HSMC4_HOSTCS_7      EQU (0x7 << 12) ;- (HSMC4) CS7
AT91C_HSMC4_XFRDONE       EQU (0x1 << 16) ;- (HSMC4) Host Data Transfer Terminated
AT91C_HSMC4_CMDDONE       EQU (0x1 << 17) ;- (HSMC4) Command Done
AT91C_HSMC4_ECCRDY        EQU (0x1 << 18) ;- (HSMC4) ECC ready
AT91C_HSMC4_DTOE          EQU (0x1 << 20) ;- (HSMC4) Data timeout Error
AT91C_HSMC4_UNDEF         EQU (0x1 << 21) ;- (HSMC4) Undefined Area Error
AT91C_HSMC4_AWB           EQU (0x1 << 22) ;- (HSMC4) Accessing While Busy Error
AT91C_HSMC4_HASE          EQU (0x1 << 23) ;- (HSMC4) Host Controller Access Size Error
AT91C_HSMC4_RBEDGE0       EQU (0x1 << 24) ;- (HSMC4) Ready Busy line 0 Edge detected
AT91C_HSMC4_RBEDGE1       EQU (0x1 << 25) ;- (HSMC4) Ready Busy line 1 Edge detected
AT91C_HSMC4_RBEDGE2       EQU (0x1 << 26) ;- (HSMC4) Ready Busy line 2 Edge detected
AT91C_HSMC4_RBEDGE3       EQU (0x1 << 27) ;- (HSMC4) Ready Busy line 3 Edge detected
AT91C_HSMC4_RBEDGE4       EQU (0x1 << 28) ;- (HSMC4) Ready Busy line 4 Edge detected
AT91C_HSMC4_RBEDGE5       EQU (0x1 << 29) ;- (HSMC4) Ready Busy line 5 Edge detected
AT91C_HSMC4_RBEDGE6       EQU (0x1 << 30) ;- (HSMC4) Ready Busy line 6 Edge detected
AT91C_HSMC4_RBEDGE7       EQU (0x1 << 31) ;- (HSMC4) Ready Busy line 7 Edge detected
// - -------- HSMC4_IER : (HSMC4 Offset: 0xc) HSMC4 Interrupt Enable Register -------- 
// - -------- HSMC4_IDR : (HSMC4 Offset: 0x10) HSMC4 Interrupt Disable Register -------- 
// - -------- HSMC4_IMR : (HSMC4 Offset: 0x14) HSMC4 Interrupt Mask Register -------- 
// - -------- HSMC4_ADDR : (HSMC4 Offset: 0x18) Address Cycle Zero Register -------- 
AT91C_HSMC4_ADDRCYCLE0    EQU (0xFF <<  0) ;- (HSMC4) Nand Flash Array Address cycle 0
// - -------- HSMC4_BANK : (HSMC4 Offset: 0x1c) Bank Register -------- 
AT91C_BANK                EQU (0x7 <<  0) ;- (HSMC4) Bank identifier
AT91C_BANK_0              EQU (0x0) ;- (HSMC4) BANK0
AT91C_BANK_1              EQU (0x1) ;- (HSMC4) BANK1
AT91C_BANK_2              EQU (0x2) ;- (HSMC4) BANK2
AT91C_BANK_3              EQU (0x3) ;- (HSMC4) BANK3
AT91C_BANK_4              EQU (0x4) ;- (HSMC4) BANK4
AT91C_BANK_5              EQU (0x5) ;- (HSMC4) BANK5
AT91C_BANK_6              EQU (0x6) ;- (HSMC4) BANK6
AT91C_BANK_7              EQU (0x7) ;- (HSMC4) BANK7
// - -------- HSMC4_ECCCR : (HSMC4 Offset: 0x20) ECC Control Register -------- 
AT91C_HSMC4_ECCRESET      EQU (0x1 <<  0) ;- (HSMC4) Reset ECC
// - -------- HSMC4_ECCCMD : (HSMC4 Offset: 0x24) ECC mode register -------- 
AT91C_ECC_PAGE_SIZE       EQU (0x3 <<  0) ;- (HSMC4) Nand Flash page size
AT91C_ECC_PAGE_SIZE_528_Bytes EQU (0x0) ;- (HSMC4) 512 bytes plus 16 bytes page size
AT91C_ECC_PAGE_SIZE_1056_Bytes EQU (0x1) ;- (HSMC4) 1024 bytes plus 32 bytes page size
AT91C_ECC_PAGE_SIZE_2112_Bytes EQU (0x2) ;- (HSMC4) 2048 bytes plus 64 bytes page size
AT91C_ECC_PAGE_SIZE_4224_Bytes EQU (0x3) ;- (HSMC4) 4096 bytes plus 128 bytes page size
AT91C_ECC_TYPCORRECT      EQU (0x3 <<  4) ;- (HSMC4) Nand Flash page size
AT91C_ECC_TYPCORRECT_ONE_PER_PAGE EQU (0x0 <<  4) ;- (HSMC4) 
AT91C_ECC_TYPCORRECT_ONE_EVERY_256_BYTES EQU (0x1 <<  4) ;- (HSMC4) 
AT91C_ECC_TYPCORRECT_ONE_EVERY_512_BYTES EQU (0x2 <<  4) ;- (HSMC4) 
// - -------- HSMC4_ECCSR1 : (HSMC4 Offset: 0x28) ECC Status Register 1 -------- 
AT91C_HSMC4_ECC_RECERR0   EQU (0x1 <<  0) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR0   EQU (0x1 <<  1) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR0   EQU (0x1 <<  2) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR1   EQU (0x1 <<  4) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR1   EQU (0x1 <<  5) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR1   EQU (0x1 <<  6) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR2   EQU (0x1 <<  8) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR2   EQU (0x1 <<  9) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR2   EQU (0x1 << 10) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR3   EQU (0x1 << 12) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR3   EQU (0x1 << 13) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR3   EQU (0x1 << 14) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR4   EQU (0x1 << 16) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR4   EQU (0x1 << 17) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR4   EQU (0x1 << 18) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR5   EQU (0x1 << 20) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR5   EQU (0x1 << 21) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR5   EQU (0x1 << 22) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR6   EQU (0x1 << 24) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR6   EQU (0x1 << 25) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR6   EQU (0x1 << 26) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR7   EQU (0x1 << 28) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR7   EQU (0x1 << 29) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR7   EQU (0x1 << 30) ;- (HSMC4) Multiple Error
// - -------- HSMC4_ECCPR0 : (HSMC4 Offset: 0x2c) HSMC4 ECC parity Register 0 -------- 
AT91C_HSMC4_ECC_BITADDR   EQU (0x7 <<  0) ;- (HSMC4) Corrupted Bit Address in the page
AT91C_HSMC4_ECC_WORDADDR  EQU (0xFF <<  3) ;- (HSMC4) Corrupted Word Address in the page
AT91C_HSMC4_ECC_NPARITY   EQU (0x7FF << 12) ;- (HSMC4) Parity N
// - -------- HSMC4_ECCPR1 : (HSMC4 Offset: 0x30) HSMC4 ECC parity Register 1 -------- 
// - -------- HSMC4_ECCSR2 : (HSMC4 Offset: 0x34) ECC Status Register 2 -------- 
AT91C_HSMC4_ECC_RECERR8   EQU (0x1 <<  0) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR8   EQU (0x1 <<  1) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR8   EQU (0x1 <<  2) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR9   EQU (0x1 <<  4) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR9   EQU (0x1 <<  5) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR9   EQU (0x1 <<  6) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR10  EQU (0x1 <<  8) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR10  EQU (0x1 <<  9) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR10  EQU (0x1 << 10) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR11  EQU (0x1 << 12) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR11  EQU (0x1 << 13) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR11  EQU (0x1 << 14) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR12  EQU (0x1 << 16) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR12  EQU (0x1 << 17) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR12  EQU (0x1 << 18) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR13  EQU (0x1 << 20) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR13  EQU (0x1 << 21) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR13  EQU (0x1 << 22) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR14  EQU (0x1 << 24) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR14  EQU (0x1 << 25) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR14  EQU (0x1 << 26) ;- (HSMC4) Multiple Error
AT91C_HSMC4_ECC_RECERR15  EQU (0x1 << 28) ;- (HSMC4) Recoverable Error
AT91C_HSMC4_ECC_ECCERR15  EQU (0x1 << 29) ;- (HSMC4) ECC Error
AT91C_HSMC4_ECC_MULERR15  EQU (0x1 << 30) ;- (HSMC4) Multiple Error
// - -------- HSMC4_ECCPR2 : (HSMC4 Offset: 0x38) HSMC4 ECC parity Register 2 -------- 
// - -------- HSMC4_ECCPR3 : (HSMC4 Offset: 0x3c) HSMC4 ECC parity Register 3 -------- 
// - -------- HSMC4_ECCPR4 : (HSMC4 Offset: 0x40) HSMC4 ECC parity Register 4 -------- 
// - -------- HSMC4_ECCPR5 : (HSMC4 Offset: 0x44) HSMC4 ECC parity Register 5 -------- 
// - -------- HSMC4_ECCPR6 : (HSMC4 Offset: 0x48) HSMC4 ECC parity Register 6 -------- 
// - -------- HSMC4_ECCPR7 : (HSMC4 Offset: 0x4c) HSMC4 ECC parity Register 7 -------- 
// - -------- HSMC4_ECCPR8 : (HSMC4 Offset: 0x50) HSMC4 ECC parity Register 8 -------- 
// - -------- HSMC4_ECCPR9 : (HSMC4 Offset: 0x54) HSMC4 ECC parity Register 9 -------- 
// - -------- HSMC4_ECCPR10 : (HSMC4 Offset: 0x58) HSMC4 ECC parity Register 10 -------- 
// - -------- HSMC4_ECCPR11 : (HSMC4 Offset: 0x5c) HSMC4 ECC parity Register 11 -------- 
// - -------- HSMC4_ECCPR12 : (HSMC4 Offset: 0x60) HSMC4 ECC parity Register 12 -------- 
// - -------- HSMC4_ECCPR13 : (HSMC4 Offset: 0x64) HSMC4 ECC parity Register 13 -------- 
// - -------- HSMC4_ECCPR14 : (HSMC4 Offset: 0x68) HSMC4 ECC parity Register 14 -------- 
// - -------- HSMC4_ECCPR15 : (HSMC4 Offset: 0x6c) HSMC4 ECC parity Register 15 -------- 
// - -------- HSMC4_OCMS : (HSMC4 Offset: 0x110) HSMC4 OCMS Register -------- 
AT91C_HSMC4_OCMS_SRSE     EQU (0x1 <<  0) ;- (HSMC4) Static Memory Controller Scrambling Enable
AT91C_HSMC4_OCMS_SMSE     EQU (0x1 <<  1) ;- (HSMC4) SRAM Scramling Enable
// - -------- HSMC4_KEY1 : (HSMC4 Offset: 0x114) HSMC4 OCMS KEY1 Register -------- 
AT91C_HSMC4_OCMS_KEY1     EQU (0x0 <<  0) ;- (HSMC4) OCMS Key 2
// - -------- HSMC4_OCMS_KEY2 : (HSMC4 Offset: 0x118) HSMC4 OCMS KEY2 Register -------- 
AT91C_HSMC4_OCMS_KEY2     EQU (0x0 <<  0) ;- (HSMC4) OCMS Key 2
// - -------- HSMC4_WPCR : (HSMC4 Offset: 0x1e4) HSMC4 Witre Protection Control Register -------- 
AT91C_HSMC4_WP_EN         EQU (0x1 <<  0) ;- (HSMC4) Write Protection Enable
AT91C_HSMC4_WP_KEY        EQU (0xFFFFFF <<  8) ;- (HSMC4) Protection Password
// - -------- HSMC4_WPSR : (HSMC4 Offset: 0x1e8) HSMC4 WPSR Register -------- 
AT91C_HSMC4_WP_VS         EQU (0xF <<  0) ;- (HSMC4) Write Protection Violation Status
AT91C_HSMC4_WP_VS_WP_VS0  EQU (0x0) ;- (HSMC4) No write protection violation since the last read of this register
AT91C_HSMC4_WP_VS_WP_VS1  EQU (0x1) ;- (HSMC4) write protection detected unauthorized attempt to write a control register had occured (since the last read)
AT91C_HSMC4_WP_VS_WP_VS2  EQU (0x2) ;- (HSMC4) Software reset had been performed while write protection was enabled (since the last read)
AT91C_HSMC4_WP_VS_WP_VS3  EQU (0x3) ;- (HSMC4) Both write protection violation and software reset with write protection enabled had occured since the last read
AT91C_                    EQU (0x0 <<  8) ;- (HSMC4) 
// - -------- HSMC4_VER : (HSMC4 Offset: 0x1fc) HSMC4 VERSION Register -------- 
// - -------- HSMC4_DUMMY : (HSMC4 Offset: 0x200) HSMC4 DUMMY REGISTER -------- 
AT91C_HSMC4_CMD1          EQU (0xFF <<  2) ;- (HSMC4) Command Register Value for Cycle 1
AT91C_HSMC4_CMD2          EQU (0xFF << 10) ;- (HSMC4) Command Register Value for Cycle 2
AT91C_HSMC4_VCMD2         EQU (0x1 << 18) ;- (HSMC4) Valid Cycle 2 Command
AT91C_HSMC4_ACYCLE        EQU (0x7 << 19) ;- (HSMC4) Number of Address required for the current command
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_NONE EQU (0x0 << 19) ;- (HSMC4) No address cycle
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_ONE EQU (0x1 << 19) ;- (HSMC4) One address cycle
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_TWO EQU (0x2 << 19) ;- (HSMC4) Two address cycles
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_THREE EQU (0x3 << 19) ;- (HSMC4) Three address cycles
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_FOUR EQU (0x4 << 19) ;- (HSMC4) Four address cycles
AT91C_HSMC4_ACYCLE_HSMC4_ACYCLE_FIVE EQU (0x5 << 19) ;- (HSMC4) Five address cycles
AT91C_HSMC4_CSID          EQU (0x7 << 22) ;- (HSMC4) Chip Select Identifier
AT91C_HSMC4_CSID_0        EQU (0x0 << 22) ;- (HSMC4) CS0
AT91C_HSMC4_CSID_1        EQU (0x1 << 22) ;- (HSMC4) CS1
AT91C_HSMC4_CSID_2        EQU (0x2 << 22) ;- (HSMC4) CS2
AT91C_HSMC4_CSID_3        EQU (0x3 << 22) ;- (HSMC4) CS3
AT91C_HSMC4_CSID_4        EQU (0x4 << 22) ;- (HSMC4) CS4
AT91C_HSMC4_CSID_5        EQU (0x5 << 22) ;- (HSMC4) CS5
AT91C_HSMC4_CSID_6        EQU (0x6 << 22) ;- (HSMC4) CS6
AT91C_HSMC4_CSID_7        EQU (0x7 << 22) ;- (HSMC4) CS7
AT91C_HSMC4_HOST_EN       EQU (0x1 << 25) ;- (HSMC4) Host Main Controller Enable
AT91C_HSMC4_HOST_WR       EQU (0x1 << 26) ;- (HSMC4) HOSTWR : Host Main Controller Write Enable
AT91C_HSMC4_HOSTCMD       EQU (0x1 << 27) ;- (HSMC4) Host Command Enable

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR AHB Matrix2 Interface
// - *****************************************************************************
// - -------- MATRIX_MCFG0 : (HMATRIX2 Offset: 0x0) Master Configuration Register ARM bus I and D -------- 
AT91C_MATRIX_ULBT         EQU (0x7 <<  0) ;- (HMATRIX2) Undefined Length Burst Type
AT91C_MATRIX_ULBT_INFINIT_LENGTH EQU (0x0) ;- (HMATRIX2) infinite length burst
AT91C_MATRIX_ULBT_SINGLE_ACCESS EQU (0x1) ;- (HMATRIX2) Single Access
AT91C_MATRIX_ULBT_4_BEAT  EQU (0x2) ;- (HMATRIX2) 4 Beat Burst
AT91C_MATRIX_ULBT_8_BEAT  EQU (0x3) ;- (HMATRIX2) 8 Beat Burst
AT91C_MATRIX_ULBT_16_BEAT EQU (0x4) ;- (HMATRIX2) 16 Beat Burst
AT91C_MATRIX_ULBT_32_BEAT EQU (0x5) ;- (HMATRIX2) 32 Beat Burst
AT91C_MATRIX_ULBT_64_BEAT EQU (0x6) ;- (HMATRIX2) 64 Beat Burst
AT91C_MATRIX_ULBT_128_BEAT EQU (0x7) ;- (HMATRIX2) 128 Beat Burst
// - -------- MATRIX_MCFG1 : (HMATRIX2 Offset: 0x4) Master Configuration Register ARM bus S -------- 
// - -------- MATRIX_MCFG2 : (HMATRIX2 Offset: 0x8) Master Configuration Register -------- 
// - -------- MATRIX_MCFG3 : (HMATRIX2 Offset: 0xc) Master Configuration Register -------- 
// - -------- MATRIX_MCFG4 : (HMATRIX2 Offset: 0x10) Master Configuration Register -------- 
// - -------- MATRIX_MCFG5 : (HMATRIX2 Offset: 0x14) Master Configuration Register -------- 
// - -------- MATRIX_MCFG6 : (HMATRIX2 Offset: 0x18) Master Configuration Register -------- 
// - -------- MATRIX_MCFG7 : (HMATRIX2 Offset: 0x1c) Master Configuration Register -------- 
// - -------- MATRIX_SCFG0 : (HMATRIX2 Offset: 0x40) Slave Configuration Register 0 -------- 
AT91C_MATRIX_SLOT_CYCLE   EQU (0xFF <<  0) ;- (HMATRIX2) Maximum Number of Allowed Cycles for a Burst
AT91C_MATRIX_DEFMSTR_TYPE EQU (0x3 << 16) ;- (HMATRIX2) Default Master Type
AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR EQU (0x0 << 16) ;- (HMATRIX2) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.
AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR EQU (0x1 << 16) ;- (HMATRIX2) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.
AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR EQU (0x2 << 16) ;- (HMATRIX2) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.
AT91C_MATRIX_FIXED_DEFMSTR_SCFG0 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG0_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG1 : (HMATRIX2 Offset: 0x44) Slave Configuration Register 1 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG1 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG1_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG2 : (HMATRIX2 Offset: 0x48) Slave Configuration Register 2 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG2 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG2_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG3 : (HMATRIX2 Offset: 0x4c) Slave Configuration Register 3 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG3 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG3_ARMC EQU (0x0 << 18) ;- (HMATRIX2) ARMC is Default Master
// - -------- MATRIX_SCFG4 : (HMATRIX2 Offset: 0x50) Slave Configuration Register 4 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG4 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG4_ARMC EQU (0x0 << 18) ;- (HMATRIX2) ARMC is Default Master
// - -------- MATRIX_SCFG5 : (HMATRIX2 Offset: 0x54) Slave Configuration Register 5 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG5 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG5_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG6 : (HMATRIX2 Offset: 0x58) Slave Configuration Register 6 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG6 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG6_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG7 : (HMATRIX2 Offset: 0x5c) Slave Configuration Register 7 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG7 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG7_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
// - -------- MATRIX_SCFG8 : (HMATRIX2 Offset: 0x60) Slave Configuration Register 8 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG8 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG8_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG8_HDMA EQU (0x4 << 18) ;- (HMATRIX2) HDMA is Default Master
// - -------- MATRIX_SCFG9 : (HMATRIX2 Offset: 0x64) Slave Configuration Register 9 -------- 
AT91C_MATRIX_FIXED_DEFMSTR_SCFG9 EQU (0x7 << 18) ;- (HMATRIX2) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG9_ARMS EQU (0x1 << 18) ;- (HMATRIX2) ARMS is Default Master
AT91C_MATRIX_FIXED_DEFMSTR_SCFG9_HDMA EQU (0x4 << 18) ;- (HMATRIX2) HDMA is Default Master
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x110) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x114) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x118) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x11c) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x120) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x124) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x128) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x12c) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x130) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x134) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x138) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x13c) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x140) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x144) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x148) Special Function Register 0 -------- 
// - -------- MATRIX_SFR0 : (HMATRIX2 Offset: 0x14c) Special Function Register 0 -------- 
// - -------- HMATRIX2_VER : (HMATRIX2 Offset: 0x1fc)  VERSION  Register -------- 
AT91C_HMATRIX2_VER        EQU (0xF <<  0) ;- (HMATRIX2)  VERSION  Register

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR NESTED vector Interrupt Controller
// - *****************************************************************************
// - -------- NVIC_ICTR : (NVIC Offset: 0x4) Interrupt Controller Type Register -------- 
AT91C_NVIC_INTLINESNUM    EQU (0xF <<  0) ;- (NVIC) Total number of interrupt lines
AT91C_NVIC_INTLINESNUM_32 EQU (0x0) ;- (NVIC) up to 32 interrupt lines supported
AT91C_NVIC_INTLINESNUM_64 EQU (0x1) ;- (NVIC) up to 64 interrupt lines supported
AT91C_NVIC_INTLINESNUM_96 EQU (0x2) ;- (NVIC) up to 96 interrupt lines supported
AT91C_NVIC_INTLINESNUM_128 EQU (0x3) ;- (NVIC) up to 128 interrupt lines supported
AT91C_NVIC_INTLINESNUM_160 EQU (0x4) ;- (NVIC) up to 160 interrupt lines supported
AT91C_NVIC_INTLINESNUM_192 EQU (0x5) ;- (NVIC) up to 192 interrupt lines supported
AT91C_NVIC_INTLINESNUM_224 EQU (0x6) ;- (NVIC) up to 224 interrupt lines supported
AT91C_NVIC_INTLINESNUM_256 EQU (0x7) ;- (NVIC) up to 256 interrupt lines supported
AT91C_NVIC_INTLINESNUM_288 EQU (0x8) ;- (NVIC) up to 288 interrupt lines supported
AT91C_NVIC_INTLINESNUM_320 EQU (0x9) ;- (NVIC) up to 320 interrupt lines supported
AT91C_NVIC_INTLINESNUM_352 EQU (0xA) ;- (NVIC) up to 352 interrupt lines supported
AT91C_NVIC_INTLINESNUM_384 EQU (0xB) ;- (NVIC) up to 384 interrupt lines supported
AT91C_NVIC_INTLINESNUM_416 EQU (0xC) ;- (NVIC) up to 416 interrupt lines supported
AT91C_NVIC_INTLINESNUM_448 EQU (0xD) ;- (NVIC) up to 448 interrupt lines supported
AT91C_NVIC_INTLINESNUM_480 EQU (0xE) ;- (NVIC) up to 480 interrupt lines supported
AT91C_NVIC_INTLINESNUM_496 EQU (0xF) ;- (NVIC) up to 496 interrupt lines supported)
// - -------- NVIC_STICKCSR : (NVIC Offset: 0x10) SysTick Control and Status Register -------- 
AT91C_NVIC_STICKENABLE    EQU (0x1 <<  0) ;- (NVIC) SysTick counter enable.
AT91C_NVIC_STICKINT       EQU (0x1 <<  1) ;- (NVIC) SysTick interrupt enable.
AT91C_NVIC_STICKCLKSOURCE EQU (0x1 <<  2) ;- (NVIC) Reference clock selection.
AT91C_NVIC_STICKCOUNTFLAG EQU (0x1 << 16) ;- (NVIC) Return 1 if timer counted to 0 since last read.
// - -------- NVIC_STICKRVR : (NVIC Offset: 0x14) SysTick Reload Value Register -------- 
AT91C_NVIC_STICKRELOAD    EQU (0xFFFFFF <<  0) ;- (NVIC) SysTick reload value.
// - -------- NVIC_STICKCVR : (NVIC Offset: 0x18) SysTick Current Value Register -------- 
AT91C_NVIC_STICKCURRENT   EQU (0x7FFFFFFF <<  0) ;- (NVIC) SysTick current value.
// - -------- NVIC_STICKCALVR : (NVIC Offset: 0x1c) SysTick Calibration Value Register -------- 
AT91C_NVIC_STICKTENMS     EQU (0xFFFFFF <<  0) ;- (NVIC) Reload value to use for 10ms timing.
AT91C_NVIC_STICKSKEW      EQU (0x1 << 30) ;- (NVIC) Read as 1 if the calibration value is not exactly 10ms because of clock frequency.
AT91C_NVIC_STICKNOREF     EQU (0x1 << 31) ;- (NVIC) Read as 1 if the reference clock is not provided.
// - -------- NVIC_IPR : (NVIC Offset: 0x400) Interrupt Priority Registers -------- 
AT91C_NVIC_PRI_N          EQU (0xFF <<  0) ;- (NVIC) Priority of interrupt N (0, 4, 8, etc)
AT91C_NVIC_PRI_N1         EQU (0xFF <<  8) ;- (NVIC) Priority of interrupt N+1 (1, 5, 9, etc)
AT91C_NVIC_PRI_N2         EQU (0xFF << 16) ;- (NVIC) Priority of interrupt N+2 (2, 6, 10, etc)
AT91C_NVIC_PRI_N3         EQU (0xFF << 24) ;- (NVIC) Priority of interrupt N+3 (3, 7, 11, etc)
// - -------- NVIC_CPUID : (NVIC Offset: 0xd00) CPU ID Base Register -------- 
AT91C_NVIC_REVISION       EQU (0xF <<  0) ;- (NVIC) Implementation defined revision number.
AT91C_NVIC_PARTNO         EQU (0xFFF <<  4) ;- (NVIC) Number of processor within family
AT91C_NVIC_CONSTANT       EQU (0xF << 16) ;- (NVIC) Reads as 0xF
AT91C_NVIC_VARIANT        EQU (0xF << 20) ;- (NVIC) Implementation defined variant number.
AT91C_NVIC_IMPLEMENTER    EQU (0xFF << 24) ;- (NVIC) Implementer code. ARM is 0x41
// - -------- NVIC_ICSR : (NVIC Offset: 0xd04) Interrupt Control State Register -------- 
AT91C_NVIC_VECTACTIVE     EQU (0x1FF <<  0) ;- (NVIC) Read-only Active ISR number field
AT91C_NVIC_RETTOBASE      EQU (0x1 << 11) ;- (NVIC) Read-only
AT91C_NVIC_VECTPENDING    EQU (0x1FF << 12) ;- (NVIC) Read-only Pending ISR number field
AT91C_NVIC_ISRPENDING     EQU (0x1 << 22) ;- (NVIC) Read-only Interrupt pending flag.
AT91C_NVIC_ISRPREEMPT     EQU (0x1 << 23) ;- (NVIC) Read-only You must only use this at debug time
AT91C_NVIC_PENDSTCLR      EQU (0x1 << 25) ;- (NVIC) Write-only Clear pending SysTick bit
AT91C_NVIC_PENDSTSET      EQU (0x1 << 26) ;- (NVIC) Read/write Set a pending SysTick bit
AT91C_NVIC_PENDSVCLR      EQU (0x1 << 27) ;- (NVIC) Write-only Clear pending pendSV bit
AT91C_NVIC_PENDSVSET      EQU (0x1 << 28) ;- (NVIC) Read/write Set pending pendSV bit
AT91C_NVIC_NMIPENDSET     EQU (0x1 << 31) ;- (NVIC) Read/write Set pending NMI
// - -------- NVIC_VTOFFR : (NVIC Offset: 0xd08) Vector Table Offset Register -------- 
AT91C_NVIC_TBLOFF         EQU (0x3FFFFF <<  7) ;- (NVIC) Vector table base offset field
AT91C_NVIC_TBLBASE        EQU (0x1 << 29) ;- (NVIC) Table base is in Code (0) or RAM (1)
AT91C_NVIC_TBLBASE_CODE   EQU (0x0 << 29) ;- (NVIC) Table base is in CODE
AT91C_NVIC_TBLBASE_RAM    EQU (0x1 << 29) ;- (NVIC) Table base is in RAM
// - -------- NVIC_AIRCR : (NVIC Offset: 0xd0c) Application Interrupt and Reset Control Register -------- 
AT91C_NVIC_VECTRESET      EQU (0x1 <<  0) ;- (NVIC) System Reset bit
AT91C_NVIC_VECTCLRACTIVE  EQU (0x1 <<  1) ;- (NVIC) Clear active vector bit
AT91C_NVIC_SYSRESETREQ    EQU (0x1 <<  2) ;- (NVIC) Causes a signal to be asserted to the outer system that indicates a reset is requested
AT91C_NVIC_PRIGROUP       EQU (0x7 <<  8) ;- (NVIC) Interrupt priority grouping field
AT91C_NVIC_PRIGROUP_3     EQU (0x3 <<  8) ;- (NVIC) indicates four bits of pre-emption priority, none bit of subpriority
AT91C_NVIC_PRIGROUP_4     EQU (0x4 <<  8) ;- (NVIC) indicates three bits of pre-emption priority, one bit of subpriority
AT91C_NVIC_PRIGROUP_5     EQU (0x5 <<  8) ;- (NVIC) indicates two bits of pre-emption priority, two bits of subpriority
AT91C_NVIC_PRIGROUP_6     EQU (0x6 <<  8) ;- (NVIC) indicates one bit of pre-emption priority, three bits of subpriority
AT91C_NVIC_PRIGROUP_7     EQU (0x7 <<  8) ;- (NVIC) indicates no pre-emption priority, four bits of subpriority
AT91C_NVIC_ENDIANESS      EQU (0x1 << 15) ;- (NVIC) Data endianness bit
AT91C_NVIC_VECTKEY        EQU (0xFFFF << 16) ;- (NVIC) Register key
// - -------- NVIC_SCR : (NVIC Offset: 0xd10) System Control Register -------- 
AT91C_NVIC_SLEEPONEXIT    EQU (0x1 <<  1) ;- (NVIC) Sleep on exit when returning from Handler mode to Thread mode
AT91C_NVIC_SLEEPDEEP      EQU (0x1 <<  2) ;- (NVIC) Sleep deep bit
AT91C_NVIC_SEVONPEND      EQU (0x1 <<  4) ;- (NVIC) When enabled, this causes WFE to wake up when an interrupt moves from inactive to pended
// - -------- NVIC_CCR : (NVIC Offset: 0xd14) Configuration Control Register -------- 
AT91C_NVIC_NONEBASETHRDENA EQU (0x1 <<  0) ;- (NVIC) When 0, default, It is only possible to enter Thread mode when returning from the last exception
AT91C_NVIC_USERSETMPEND   EQU (0x1 <<  1) ;- (NVIC) 
AT91C_NVIC_UNALIGN_TRP    EQU (0x1 <<  3) ;- (NVIC) Trap for unaligned access
AT91C_NVIC_DIV_0_TRP      EQU (0x1 <<  4) ;- (NVIC) Trap on Divide by 0
AT91C_NVIC_BFHFNMIGN      EQU (0x1 <<  8) ;- (NVIC) 
AT91C_NVIC_STKALIGN       EQU (0x1 <<  9) ;- (NVIC) 
// - -------- NVIC_HAND4PR : (NVIC Offset: 0xd18) System Handlers 4-7 Priority Register -------- 
AT91C_NVIC_PRI_4          EQU (0xFF <<  0) ;- (NVIC) 
AT91C_NVIC_PRI_5          EQU (0xFF <<  8) ;- (NVIC) 
AT91C_NVIC_PRI_6          EQU (0xFF << 16) ;- (NVIC) 
AT91C_NVIC_PRI_7          EQU (0xFF << 24) ;- (NVIC) 
// - -------- NVIC_HAND8PR : (NVIC Offset: 0xd1c) System Handlers 8-11 Priority Register -------- 
AT91C_NVIC_PRI_8          EQU (0xFF <<  0) ;- (NVIC) 
AT91C_NVIC_PRI_9          EQU (0xFF <<  8) ;- (NVIC) 
AT91C_NVIC_PRI_10         EQU (0xFF << 16) ;- (NVIC) 
AT91C_NVIC_PRI_11         EQU (0xFF << 24) ;- (NVIC) 
// - -------- NVIC_HAND12PR : (NVIC Offset: 0xd20) System Handlers 12-15 Priority Register -------- 
AT91C_NVIC_PRI_12         EQU (0xFF <<  0) ;- (NVIC) 
AT91C_NVIC_PRI_13         EQU (0xFF <<  8) ;- (NVIC) 
AT91C_NVIC_PRI_14         EQU (0xFF << 16) ;- (NVIC) 
AT91C_NVIC_PRI_15         EQU (0xFF << 24) ;- (NVIC) 
// - -------- NVIC_HANDCSR : (NVIC Offset: 0xd24) System Handler Control and State Register -------- 
AT91C_NVIC_MEMFAULTACT    EQU (0x1 <<  0) ;- (NVIC) 
AT91C_NVIC_BUSFAULTACT    EQU (0x1 <<  1) ;- (NVIC) 
AT91C_NVIC_USGFAULTACT    EQU (0x1 <<  3) ;- (NVIC) 
AT91C_NVIC_SVCALLACT      EQU (0x1 <<  7) ;- (NVIC) 
AT91C_NVIC_MONITORACT     EQU (0x1 <<  8) ;- (NVIC) 
AT91C_NVIC_PENDSVACT      EQU (0x1 << 10) ;- (NVIC) 
AT91C_NVIC_SYSTICKACT     EQU (0x1 << 11) ;- (NVIC) 
AT91C_NVIC_USGFAULTPENDED EQU (0x1 << 12) ;- (NVIC) 
AT91C_NVIC_MEMFAULTPENDED EQU (0x1 << 13) ;- (NVIC) 
AT91C_NVIC_BUSFAULTPENDED EQU (0x1 << 14) ;- (NVIC) 
AT91C_NVIC_SVCALLPENDED   EQU (0x1 << 15) ;- (NVIC) 
AT91C_NVIC_MEMFAULTENA    EQU (0x1 << 16) ;- (NVIC) 
AT91C_NVIC_BUSFAULTENA    EQU (0x1 << 17) ;- (NVIC) 
AT91C_NVIC_USGFAULTENA    EQU (0x1 << 18) ;- (NVIC) 
// - -------- NVIC_CFSR : (NVIC Offset: 0xd28) Configurable Fault Status Registers -------- 
AT91C_NVIC_MEMMANAGE      EQU (0xFF <<  0) ;- (NVIC) 
AT91C_NVIC_BUSFAULT       EQU (0xFF <<  8) ;- (NVIC) 
AT91C_NVIC_USAGEFAULT     EQU (0xFF << 16) ;- (NVIC) 
// - -------- NVIC_BFAR : (NVIC Offset: 0xd38) Bus Fault Address Register -------- 
AT91C_NVIC_IBUSERR        EQU (0x1 <<  0) ;- (NVIC) This bit indicates a bus fault on an instruction prefetch
AT91C_NVIC_PRECISERR      EQU (0x1 <<  1) ;- (NVIC) Precise data access error. The BFAR is written with the faulting address
AT91C_NVIC_IMPRECISERR    EQU (0x1 <<  2) ;- (NVIC) Imprecise data access error
AT91C_NVIC_UNSTKERR       EQU (0x1 <<  3) ;- (NVIC) This bit indicates a derived bus fault has occurred on exception return
AT91C_NVIC_STKERR         EQU (0x1 <<  4) ;- (NVIC) This bit indicates a derived bus fault has occurred on exception entry
AT91C_NVIC_BFARVALID      EQU (0x1 <<  7) ;- (NVIC) This bit is set if the BFAR register has valid contents
// - -------- NVIC_PFR0 : (NVIC Offset: 0xd40) Processor Feature register0 (ID_PFR0) -------- 
AT91C_NVIC_ID_PFR0_0      EQU (0xF <<  0) ;- (NVIC) State0 (T-bit == 0)
AT91C_NVIC_ID_PRF0_1      EQU (0xF <<  4) ;- (NVIC) State1 (T-bit == 1)
// - -------- NVIC_PFR1 : (NVIC Offset: 0xd44) Processor Feature register1 (ID_PFR1) -------- 
AT91C_NVIC_ID_PRF1_MODEL  EQU (0xF <<  8) ;- (NVIC) Microcontroller programmer’s model
// - -------- NVIC_DFR0 : (NVIC Offset: 0xd48) Debug Feature register0 (ID_DFR0) -------- 
AT91C_NVIC_ID_DFR0_MODEL  EQU (0xF << 20) ;- (NVIC) Microcontroller Debug Model – memory mapped
// - -------- NVIC_MMFR0 : (NVIC Offset: 0xd50) Memory Model Feature register0 (ID_MMFR0) -------- 
AT91C_NVIC_ID_MMFR0_PMSA  EQU (0xF <<  4) ;- (NVIC) Microcontroller Debug Model – memory mapped
AT91C_NVIC_ID_MMFR0_CACHE EQU (0xF <<  8) ;- (NVIC) Microcontroller Debug Model – memory mapped

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR NESTED vector Interrupt Controller
// - *****************************************************************************
// - -------- MPU_TYPE : (MPU Offset: 0x0)  -------- 
AT91C_MPU_SEPARATE        EQU (0x1 <<  0) ;- (MPU) 
AT91C_MPU_DREGION         EQU (0xFF <<  8) ;- (MPU) 
AT91C_MPU_IREGION         EQU (0xFF << 16) ;- (MPU) 
// - -------- MPU_CTRL : (MPU Offset: 0x4)  -------- 
AT91C_MPU_ENABLE          EQU (0x1 <<  0) ;- (MPU) 
AT91C_MPU_HFNMIENA        EQU (0x1 <<  1) ;- (MPU) 
AT91C_MPU_PRIVDEFENA      EQU (0x1 <<  2) ;- (MPU) 
// - -------- MPU_REG_NB : (MPU Offset: 0x8)  -------- 
AT91C_MPU_REGION          EQU (0xFF <<  0) ;- (MPU) 
// - -------- MPU_REG_BASE_ADDR : (MPU Offset: 0xc)  -------- 
AT91C_MPU_REG             EQU (0xF <<  0) ;- (MPU) 
AT91C_MPU_VALID           EQU (0x1 <<  4) ;- (MPU) 
AT91C_MPU_ADDR            EQU (0x3FFFFFF <<  5) ;- (MPU) 
// - -------- MPU_ATTR_SIZE : (MPU Offset: 0x10)  -------- 
AT91C_MPU_ENA             EQU (0x1 <<  0) ;- (MPU) 
AT91C_MPU_SIZE            EQU (0xF <<  1) ;- (MPU) 
AT91C_MPU_SRD             EQU (0xFF <<  8) ;- (MPU) 
AT91C_MPU_B               EQU (0x1 << 16) ;- (MPU) 
AT91C_MPU_C               EQU (0x1 << 17) ;- (MPU) 
AT91C_MPU_S               EQU (0x1 << 18) ;- (MPU) 
AT91C_MPU_TEX             EQU (0x7 << 19) ;- (MPU) 
AT91C_MPU_AP              EQU (0x7 << 24) ;- (MPU) 
AT91C_MPU_XN              EQU (0x7 << 28) ;- (MPU) 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR CORTEX_M3 Registers
// - *****************************************************************************
// - -------- CM3_CPUID : (CM3 Offset: 0x0)  -------- 
// - -------- CM3_AIRCR : (CM3 Offset: 0xc)  -------- 
AT91C_CM3_SYSRESETREQ     EQU (0x1 <<  2) ;- (CM3) A reset is requested by the processor.
// - -------- CM3_SCR : (CM3 Offset: 0x10)  -------- 
AT91C_CM3_SLEEPONEXIT     EQU (0x1 <<  1) ;- (CM3) Sleep on exit when returning from Handler mode to Thread mode. Enables interrupt driven applications to avoid returning to empty main application.
AT91C_CM3_SLEEPDEEP       EQU (0x1 <<  2) ;- (CM3) Sleep deep bit.
AT91C_CM3_SEVONPEND       EQU (0x1 <<  4) ;- (CM3) When enabled, this causes WFE to wake up when an interrupt moves from inactive to pended.
// - -------- CM3_SHCSR : (CM3 Offset: 0x24)  -------- 
AT91C_CM3_SYSTICKACT      EQU (0x1 << 11) ;- (CM3) Reads as 1 if SysTick is active.

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller
// - *****************************************************************************
// - -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- 
AT91C_PDC_RXTEN           EQU (0x1 <<  0) ;- (PDC) Receiver Transfer Enable
AT91C_PDC_RXTDIS          EQU (0x1 <<  1) ;- (PDC) Receiver Transfer Disable
AT91C_PDC_TXTEN           EQU (0x1 <<  8) ;- (PDC) Transmitter Transfer Enable
AT91C_PDC_TXTDIS          EQU (0x1 <<  9) ;- (PDC) Transmitter Transfer Disable
// - -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Debug Unit
// - *****************************************************************************
// - -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- 
AT91C_DBGU_RSTRX          EQU (0x1 <<  2) ;- (DBGU) Reset Receiver
AT91C_DBGU_RSTTX          EQU (0x1 <<  3) ;- (DBGU) Reset Transmitter
AT91C_DBGU_RXEN           EQU (0x1 <<  4) ;- (DBGU) Receiver Enable
AT91C_DBGU_RXDIS          EQU (0x1 <<  5) ;- (DBGU) Receiver Disable
AT91C_DBGU_TXEN           EQU (0x1 <<  6) ;- (DBGU) Transmitter Enable
AT91C_DBGU_TXDIS          EQU (0x1 <<  7) ;- (DBGU) Transmitter Disable
AT91C_DBGU_RSTSTA         EQU (0x1 <<  8) ;- (DBGU) Reset Status Bits
// - -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- 
AT91C_DBGU_PAR            EQU (0x7 <<  9) ;- (DBGU) Parity type
AT91C_DBGU_PAR_EVEN       EQU (0x0 <<  9) ;- (DBGU) Even Parity
AT91C_DBGU_PAR_ODD        EQU (0x1 <<  9) ;- (DBGU) Odd Parity
AT91C_DBGU_PAR_SPACE      EQU (0x2 <<  9) ;- (DBGU) Parity forced to 0 (Space)
AT91C_DBGU_PAR_MARK       EQU (0x3 <<  9) ;- (DBGU) Parity forced to 1 (Mark)
AT91C_DBGU_PAR_NONE       EQU (0x4 <<  9) ;- (DBGU) No Parity
AT91C_DBGU_CHMODE         EQU (0x3 << 14) ;- (DBGU) Channel Mode
AT91C_DBGU_CHMODE_NORMAL  EQU (0x0 << 14) ;- (DBGU) Normal Mode: The debug unit channel operates as an RX/TX debug unit.
AT91C_DBGU_CHMODE_AUTO    EQU (0x1 << 14) ;- (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.
AT91C_DBGU_CHMODE_LOCAL   EQU (0x2 << 14) ;- (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
AT91C_DBGU_CHMODE_REMOTE  EQU (0x3 << 14) ;- (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.
// - -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- 
AT91C_DBGU_RXRDY          EQU (0x1 <<  0) ;- (DBGU) RXRDY Interrupt
AT91C_DBGU_TXRDY          EQU (0x1 <<  1) ;- (DBGU) TXRDY Interrupt
AT91C_DBGU_ENDRX          EQU (0x1 <<  3) ;- (DBGU) End of Receive Transfer Interrupt
AT91C_DBGU_ENDTX          EQU (0x1 <<  4) ;- (DBGU) End of Transmit Interrupt
AT91C_DBGU_OVRE           EQU (0x1 <<  5) ;- (DBGU) Overrun Interrupt
AT91C_DBGU_FRAME          EQU (0x1 <<  6) ;- (DBGU) Framing Error Interrupt
AT91C_DBGU_PARE           EQU (0x1 <<  7) ;- (DBGU) Parity Error Interrupt
AT91C_DBGU_TXEMPTY        EQU (0x1 <<  9) ;- (DBGU) TXEMPTY Interrupt
AT91C_DBGU_TXBUFE         EQU (0x1 << 11) ;- (DBGU) TXBUFE Interrupt
AT91C_DBGU_RXBUFF         EQU (0x1 << 12) ;- (DBGU) RXBUFF Interrupt
AT91C_DBGU_COMM_TX        EQU (0x1 << 30) ;- (DBGU) COMM_TX Interrupt
AT91C_DBGU_COMM_RX        EQU (0x1 << 31) ;- (DBGU) COMM_RX Interrupt
// - -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- 
// - -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- 
// - -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- 
// - -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- 
AT91C_DBGU_FORCE_NTRST    EQU (0x1 <<  0) ;- (DBGU) Force NTRST in JTAG

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler
// - *****************************************************************************
// - -------- PIO_KER : (PIO Offset: 0x120) Keypad Controller Enable Register -------- 
AT91C_PIO_KCE             EQU (0x1 <<  0) ;- (PIO) Keypad Controller Enable
// - -------- PIO_KRCR : (PIO Offset: 0x124) Keypad Controller Row Column Register -------- 
AT91C_PIO_NBR             EQU (0x7 <<  0) ;- (PIO) Number of Columns of the Keypad Matrix
AT91C_PIO_NBC             EQU (0x7 <<  8) ;- (PIO) Number of Rows of the Keypad Matrix
// - -------- PIO_KDR : (PIO Offset: 0x128) Keypad Controller Debouncing Register -------- 
AT91C_PIO_DBC             EQU (0x3FF <<  0) ;- (PIO) Debouncing Value
// - -------- PIO_KIER : (PIO Offset: 0x130) Keypad Controller Interrupt Enable Register -------- 
AT91C_PIO_KPR             EQU (0x1 <<  0) ;- (PIO) Key Press Interrupt Enable
AT91C_PIO_KRL             EQU (0x1 <<  1) ;- (PIO) Key Release Interrupt Enable
// - -------- PIO_KIDR : (PIO Offset: 0x134) Keypad Controller Interrupt Disable Register -------- 
// - -------- PIO_KIMR : (PIO Offset: 0x138) Keypad Controller Interrupt Mask Register -------- 
// - -------- PIO_KSR : (PIO Offset: 0x13c) Keypad Controller Status Register -------- 
AT91C_PIO_NBKPR           EQU (0x3 <<  8) ;- (PIO) Number of Simultaneous Key Presses
AT91C_PIO_NBKRL           EQU (0x3 << 16) ;- (PIO) Number of Simultaneous Key Releases
// - -------- PIO_KKPR : (PIO Offset: 0x140) Keypad Controller Key Press Register -------- 
AT91C_KEY0ROW             EQU (0x7 <<  0) ;- (PIO) Row index of the first detected Key Press
AT91C_KEY0COL             EQU (0x7 <<  4) ;- (PIO) Column index of the first detected Key Press
AT91C_KEY1ROW             EQU (0x7 <<  8) ;- (PIO) Row index of the second detected Key Press
AT91C_KEY1COL             EQU (0x7 << 12) ;- (PIO) Column index of the second detected Key Press
AT91C_KEY2ROW             EQU (0x7 << 16) ;- (PIO) Row index of the third detected Key Press
AT91C_KEY2COL             EQU (0x7 << 20) ;- (PIO) Column index of the third detected Key Press
AT91C_KEY3ROW             EQU (0x7 << 24) ;- (PIO) Row index of the fourth detected Key Press
AT91C_KEY3COL             EQU (0x7 << 28) ;- (PIO) Column index of the fourth detected Key Press
// - -------- PIO_KKRR : (PIO Offset: 0x144) Keypad Controller Key Release Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Power Management Controler
// - *****************************************************************************
// - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
AT91C_PMC_PCK0            EQU (0x1 <<  8) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK1            EQU (0x1 <<  9) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK2            EQU (0x1 << 10) ;- (PMC) Programmable Clock Output
// - -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- 
// - -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- 
// - -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- 
AT91C_CKGR_UPLLEN         EQU (0x1 << 16) ;- (PMC) UTMI PLL Enable
AT91C_CKGR_UPLLEN_DISABLED EQU (0x0 << 16) ;- (PMC) The UTMI PLL is disabled
AT91C_CKGR_UPLLEN_ENABLED EQU (0x1 << 16) ;- (PMC) The UTMI PLL is enabled
AT91C_CKGR_UPLLCOUNT      EQU (0xF << 20) ;- (PMC) UTMI Oscillator Start-up Time
AT91C_CKGR_BIASEN         EQU (0x1 << 24) ;- (PMC) UTMI BIAS Enable
AT91C_CKGR_BIASEN_DISABLED EQU (0x0 << 24) ;- (PMC) The UTMI BIAS is disabled
AT91C_CKGR_BIASEN_ENABLED EQU (0x1 << 24) ;- (PMC) The UTMI BIAS is enabled
AT91C_CKGR_BIASCOUNT      EQU (0xF << 28) ;- (PMC) UTMI BIAS Start-up Time
// - -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- 
AT91C_CKGR_MOSCXTEN       EQU (0x1 <<  0) ;- (PMC) Main Crystal Oscillator Enable
AT91C_CKGR_MOSCXTBY       EQU (0x1 <<  1) ;- (PMC) Main Crystal Oscillator Bypass
AT91C_CKGR_WAITMODE       EQU (0x1 <<  2) ;- (PMC) Main Crystal Oscillator Bypass
AT91C_CKGR_MOSCRCEN       EQU (0x1 <<  3) ;- (PMC) Main On-Chip RC Oscillator Enable
AT91C_CKGR_MOSCRCF        EQU (0x7 <<  4) ;- (PMC) Main On-Chip RC Oscillator Frequency Selection
AT91C_CKGR_MOSCXTST       EQU (0xFF <<  8) ;- (PMC) Main Crystal Oscillator Start-up Time
AT91C_CKGR_KEY            EQU (0xFF << 16) ;- (PMC) Clock Generator Controller Writing Protection Key
AT91C_CKGR_MOSCSEL        EQU (0x1 << 24) ;- (PMC) Main Oscillator Selection
AT91C_CKGR_CFDEN          EQU (0x1 << 25) ;- (PMC) Clock Failure Detector Enable
// - -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- 
AT91C_CKGR_MAINF          EQU (0xFFFF <<  0) ;- (PMC) Main Clock Frequency
AT91C_CKGR_MAINRDY        EQU (0x1 << 16) ;- (PMC) Main Clock Ready
// - -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- 
AT91C_CKGR_DIVA           EQU (0xFF <<  0) ;- (PMC) Divider Selected
AT91C_CKGR_DIVA_0         EQU (0x0) ;- (PMC) Divider output is 0
AT91C_CKGR_DIVA_BYPASS    EQU (0x1) ;- (PMC) Divider is bypassed
AT91C_CKGR_PLLACOUNT      EQU (0x3F <<  8) ;- (PMC) PLLA Counter
AT91C_CKGR_STMODE         EQU (0x3 << 14) ;- (PMC) Start Mode
AT91C_CKGR_STMODE_0       EQU (0x0 << 14) ;- (PMC) Fast startup
AT91C_CKGR_STMODE_1       EQU (0x1 << 14) ;- (PMC) Reserved
AT91C_CKGR_STMODE_2       EQU (0x2 << 14) ;- (PMC) Normal startup
AT91C_CKGR_STMODE_3       EQU (0x3 << 14) ;- (PMC) Reserved
AT91C_CKGR_MULA           EQU (0x7FF << 16) ;- (PMC) PLL Multiplier
AT91C_CKGR_SRC            EQU (0x1 << 29) ;- (PMC) 
// - -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
AT91C_PMC_CSS             EQU (0x7 <<  0) ;- (PMC) Programmable Clock Selection
AT91C_PMC_CSS_SLOW_CLK    EQU (0x0) ;- (PMC) Slow Clock is selected
AT91C_PMC_CSS_MAIN_CLK    EQU (0x1) ;- (PMC) Main Clock is selected
AT91C_PMC_CSS_PLLA_CLK    EQU (0x2) ;- (PMC) Clock from PLL A is selected
AT91C_PMC_CSS_UPLL_CLK    EQU (0x3) ;- (PMC) Clock from UPLL is selected
AT91C_PMC_CSS_SYS_CLK     EQU (0x4) ;- (PMC) System clock is selected
AT91C_PMC_PRES            EQU (0x7 <<  4) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_PRES_CLK        EQU (0x0 <<  4) ;- (PMC) Selected clock
AT91C_PMC_PRES_CLK_2      EQU (0x1 <<  4) ;- (PMC) Selected clock divided by 2
AT91C_PMC_PRES_CLK_4      EQU (0x2 <<  4) ;- (PMC) Selected clock divided by 4
AT91C_PMC_PRES_CLK_8      EQU (0x3 <<  4) ;- (PMC) Selected clock divided by 8
AT91C_PMC_PRES_CLK_16     EQU (0x4 <<  4) ;- (PMC) Selected clock divided by 16
AT91C_PMC_PRES_CLK_32     EQU (0x5 <<  4) ;- (PMC) Selected clock divided by 32
AT91C_PMC_PRES_CLK_64     EQU (0x6 <<  4) ;- (PMC) Selected clock divided by 64
AT91C_PMC_PRES_CLK_3      EQU (0x7 <<  4) ;- (PMC) Selected clock divided by 3
// - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- 
// - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
AT91C_PMC_MOSCXTS         EQU (0x1 <<  0) ;- (PMC) Main Crystal Oscillator Status/Enable/Disable/Mask
AT91C_PMC_LOCKA           EQU (0x1 <<  1) ;- (PMC) PLL A Status/Enable/Disable/Mask
AT91C_PMC_MCKRDY          EQU (0x1 <<  3) ;- (PMC) Master Clock Status/Enable/Disable/Mask
AT91C_PMC_LOCKU           EQU (0x1 <<  6) ;- (PMC) PLL UTMI Status/Enable/Disable/Mask
AT91C_PMC_PCKRDY0         EQU (0x1 <<  8) ;- (PMC) PCK0_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCKRDY1         EQU (0x1 <<  9) ;- (PMC) PCK1_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCKRDY2         EQU (0x1 << 10) ;- (PMC) PCK2_RDY Status/Enable/Disable/Mask
AT91C_PMC_MOSCSELS        EQU (0x1 << 16) ;- (PMC) Main Oscillator Selection Status
AT91C_PMC_MOSCRCS         EQU (0x1 << 17) ;- (PMC) Main On-Chip RC Oscillator Status
AT91C_PMC_CFDEV           EQU (0x1 << 18) ;- (PMC) Clock Failure Detector Event
// - -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- 
// - -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- 
AT91C_PMC_OSCSELS         EQU (0x1 <<  7) ;- (PMC) Slow Clock Oscillator Selection
AT91C_PMC_CFDS            EQU (0x1 << 19) ;- (PMC) Clock Failure Detector Status
AT91C_PMC_FOS             EQU (0x1 << 20) ;- (PMC) Clock Failure Detector Fault Output Status
// - -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- 
// - -------- PMC_FSMR : (PMC Offset: 0x70) Fast Startup Mode Register -------- 
AT91C_PMC_FSTT            EQU (0xFFFF <<  0) ;- (PMC) Fast Start-up Input Enable 0 to 15
AT91C_PMC_RTTAL           EQU (0x1 << 16) ;- (PMC) RTT Alarm Enable
AT91C_PMC_RTCAL           EQU (0x1 << 17) ;- (PMC) RTC Alarm Enable
AT91C_PMC_USBAL           EQU (0x1 << 18) ;- (PMC) USB Alarm Enable
AT91C_PMC_LPM             EQU (0x1 << 20) ;- (PMC) Low Power Mode
// - -------- PMC_FSPR : (PMC Offset: 0x74) Fast Startup Polarity Register -------- 
AT91C_PMC_FSTP            EQU (0xFFFF <<  0) ;- (PMC) Fast Start-up Input Polarity 0 to 15
// - -------- PMC_FOCR : (PMC Offset: 0x78) Fault Output Clear Register -------- 
AT91C_PMC_FOCLR           EQU (0x1 <<  0) ;- (PMC) Fault Output Clear

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// - *****************************************************************************
// - -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// - -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// - -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// - -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Reset Controller Interface
// - *****************************************************************************
// - -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- 
AT91C_RSTC_PROCRST        EQU (0x1 <<  0) ;- (RSTC) Processor Reset
AT91C_RSTC_ICERST         EQU (0x1 <<  1) ;- (RSTC) ICE Interface Reset
AT91C_RSTC_PERRST         EQU (0x1 <<  2) ;- (RSTC) Peripheral Reset
AT91C_RSTC_EXTRST         EQU (0x1 <<  3) ;- (RSTC) External Reset
AT91C_RSTC_KEY            EQU (0xFF << 24) ;- (RSTC) Password
// - -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- 
AT91C_RSTC_URSTS          EQU (0x1 <<  0) ;- (RSTC) User Reset Status
AT91C_RSTC_RSTTYP         EQU (0x7 <<  8) ;- (RSTC) Reset Type
AT91C_RSTC_RSTTYP_GENERAL EQU (0x0 <<  8) ;- (RSTC) General reset. Both VDDCORE and VDDBU rising.
AT91C_RSTC_RSTTYP_WAKEUP  EQU (0x1 <<  8) ;- (RSTC) WakeUp Reset. VDDCORE rising.
AT91C_RSTC_RSTTYP_WATCHDOG EQU (0x2 <<  8) ;- (RSTC) Watchdog Reset. Watchdog overflow occured.
AT91C_RSTC_RSTTYP_SOFTWARE EQU (0x3 <<  8) ;- (RSTC) Software Reset. Processor reset required by the software.
AT91C_RSTC_RSTTYP_USER    EQU (0x4 <<  8) ;- (RSTC) User Reset. NRST pin detected low.
AT91C_RSTC_NRSTL          EQU (0x1 << 16) ;- (RSTC) NRST pin level
AT91C_RSTC_SRCMP          EQU (0x1 << 17) ;- (RSTC) Software Reset Command in Progress.
// - -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- 
AT91C_RSTC_URSTEN         EQU (0x1 <<  0) ;- (RSTC) User Reset Enable
AT91C_RSTC_URSTIEN        EQU (0x1 <<  4) ;- (RSTC) User Reset Interrupt Enable
AT91C_RSTC_ERSTL          EQU (0xF <<  8) ;- (RSTC) User Reset Enable

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Supply Controller Interface
// - *****************************************************************************
// - -------- SUPC_CR : (SUPC Offset: 0x0) Control Register -------- 
AT91C_SUPC_SHDW           EQU (0x1 <<  0) ;- (SUPC) Shut Down Command
AT91C_SUPC_SHDWEOF        EQU (0x1 <<  1) ;- (SUPC) Shut Down after End Of Frame
AT91C_SUPC_VROFF          EQU (0x1 <<  2) ;- (SUPC) Voltage Regulator Off
AT91C_SUPC_XTALSEL        EQU (0x1 <<  3) ;- (SUPC) Crystal Oscillator Select
AT91C_SUPC_KEY            EQU (0xFF << 24) ;- (SUPC) Supply Controller Writing Protection Key
// - -------- SUPC_BOMR : (SUPC Offset: 0x4) Brown Out Mode Register -------- 
AT91C_SUPC_BODTH          EQU (0xF <<  0) ;- (SUPC) Brown Out Threshold
AT91C_SUPC_BODSMPL        EQU (0x7 <<  8) ;- (SUPC) Brown Out Sampling Period
AT91C_SUPC_BODSMPL_DISABLED EQU (0x0 <<  8) ;- (SUPC) Brown Out Detector disabled
AT91C_SUPC_BODSMPL_CONTINUOUS EQU (0x1 <<  8) ;- (SUPC) Continuous Brown Out Detector
AT91C_SUPC_BODSMPL_32_SLCK EQU (0x2 <<  8) ;- (SUPC) Brown Out Detector enabled one SLCK period every 32 SLCK periods
AT91C_SUPC_BODSMPL_256_SLCK EQU (0x3 <<  8) ;- (SUPC) Brown Out Detector enabled one SLCK period every 256 SLCK periods
AT91C_SUPC_BODSMPL_2048_SLCK EQU (0x4 <<  8) ;- (SUPC) Brown Out Detector enabled one SLCK period every 2048 SLCK periods
AT91C_SUPC_BODRSTEN       EQU (0x1 << 12) ;- (SUPC) Brownout Reset Enable
// - -------- SUPC_MR : (SUPC Offset: 0x8) Supply Controller Mode Register -------- 
AT91C_SUPC_LCDOUT         EQU (0xF <<  0) ;- (SUPC) LCD Charge Pump Output Voltage Selection
AT91C_SUPC_LCDMODE        EQU (0x3 <<  4) ;- (SUPC) Segment LCD Supply Mode
AT91C_SUPC_LCDMODE_OFF    EQU (0x0 <<  4) ;- (SUPC) The internal and external supply sources are both deselected and the on-chip charge pump is turned off
AT91C_SUPC_LCDMODE_OFF_AFTER_EOF EQU (0x1 <<  4) ;- (SUPC) At the End Of Frame from LCD controller, the internal and external supply sources are both deselected and the on-chip charge pump is turned off
AT91C_SUPC_LCDMODE_EXTERNAL EQU (0x2 <<  4) ;- (SUPC) The external supply source is selected
AT91C_SUPC_LCDMODE_INTERNAL EQU (0x3 <<  4) ;- (SUPC) The internal supply source is selected and the on-chip charge pump is turned on
AT91C_SUPC_VRDEEP         EQU (0x1 <<  8) ;- (SUPC) Voltage Regulator Deep Mode
AT91C_SUPC_VRVDD          EQU (0x7 <<  9) ;- (SUPC) Voltage Regulator Output Voltage Selection
AT91C_SUPC_VRRSTEN        EQU (0x1 << 12) ;- (SUPC) Voltage Regulation Loss Reset Enable
AT91C_SUPC_GPBRON         EQU (0x1 << 16) ;- (SUPC) GPBR ON
AT91C_SUPC_SRAMON         EQU (0x1 << 17) ;- (SUPC) SRAM ON
AT91C_SUPC_RTCON          EQU (0x1 << 18) ;- (SUPC) Real Time Clock Power switch ON
AT91C_SUPC_FLASHON        EQU (0x1 << 19) ;- (SUPC) Flash Power switch On
AT91C_SUPC_BYPASS         EQU (0x1 << 20) ;- (SUPC) 32kHz oscillator bypass
AT91C_SUPC_MKEY           EQU (0xFF << 24) ;- (SUPC) Supply Controller Writing Protection Key
// - -------- SUPC_WUMR : (SUPC Offset: 0xc) Wake Up Mode Register -------- 
AT91C_SUPC_FWUPEN         EQU (0x1 <<  0) ;- (SUPC) Force Wake Up Enable
AT91C_SUPC_BODEN          EQU (0x1 <<  1) ;- (SUPC) Brown Out Wake Up Enable
AT91C_SUPC_RTTEN          EQU (0x1 <<  2) ;- (SUPC) Real Time Timer Wake Up Enable
AT91C_SUPC_RTCEN          EQU (0x1 <<  3) ;- (SUPC) Real Time Clock Wake Up Enable
AT91C_SUPC_FWUPDBC        EQU (0x7 <<  8) ;- (SUPC) Force Wake Up debouncer
AT91C_SUPC_FWUPDBC_IMMEDIATE EQU (0x0 <<  8) ;- (SUPC) Immediate, No debouncing, detected active at least one Slow clock edge
AT91C_SUPC_FWUPDBC_3_SLCK EQU (0x1 <<  8) ;- (SUPC) An enabled Wake Up input shall be low for at least 3 SLCK periods
AT91C_SUPC_FWUPDBC_32_SLCK EQU (0x2 <<  8) ;- (SUPC) An enabled Wake Up input  shall be low for at least 32 SLCK periods
AT91C_SUPC_FWUPDBC_512_SLCK EQU (0x3 <<  8) ;- (SUPC) An enabled Wake Up input  shall be low for at least 512 SLCK periods
AT91C_SUPC_FWUPDBC_4096_SLCK EQU (0x4 <<  8) ;- (SUPC) An enabled Wake Up input  shall be low for at least 4096 SLCK periods
AT91C_SUPC_FWUPDBC_32768_SLCK EQU (0x5 <<  8) ;- (SUPC) An enabled Wake Up input  shall be low for at least 32768 SLCK periods
AT91C_SUPC_WKUPDBC        EQU (0x7 << 12) ;- (SUPC) Force Wake Up debouncer
AT91C_SUPC_WKUPDBC_IMMEDIATE EQU (0x0 << 12) ;- (SUPC) Immediate, No debouncing, detected active at least one Slow clock edge
AT91C_SUPC_WKUPDBC_3_SLCK EQU (0x1 << 12) ;- (SUPC) FWUP shall be low for at least 3 SLCK periods
AT91C_SUPC_WKUPDBC_32_SLCK EQU (0x2 << 12) ;- (SUPC) FWUP shall be low for at least 32 SLCK periods
AT91C_SUPC_WKUPDBC_512_SLCK EQU (0x3 << 12) ;- (SUPC) FWUP shall be low for at least 512 SLCK periods
AT91C_SUPC_WKUPDBC_4096_SLCK EQU (0x4 << 12) ;- (SUPC) FWUP shall be low for at least 4096 SLCK periods
AT91C_SUPC_WKUPDBC_32768_SLCK EQU (0x5 << 12) ;- (SUPC) FWUP shall be low for at least 32768 SLCK periods
// - -------- SUPC_WUIR : (SUPC Offset: 0x10) Wake Up Inputs Register -------- 
AT91C_SUPC_WKUPEN0        EQU (0x1 <<  0) ;- (SUPC) Wake Up Input Enable 0
AT91C_SUPC_WKUPEN1        EQU (0x1 <<  1) ;- (SUPC) Wake Up Input Enable 1
AT91C_SUPC_WKUPEN2        EQU (0x1 <<  2) ;- (SUPC) Wake Up Input Enable 2
AT91C_SUPC_WKUPEN3        EQU (0x1 <<  3) ;- (SUPC) Wake Up Input Enable 3
AT91C_SUPC_WKUPEN4        EQU (0x1 <<  4) ;- (SUPC) Wake Up Input Enable 4
AT91C_SUPC_WKUPEN5        EQU (0x1 <<  5) ;- (SUPC) Wake Up Input Enable 5
AT91C_SUPC_WKUPEN6        EQU (0x1 <<  6) ;- (SUPC) Wake Up Input Enable 6
AT91C_SUPC_WKUPEN7        EQU (0x1 <<  7) ;- (SUPC) Wake Up Input Enable 7
AT91C_SUPC_WKUPEN8        EQU (0x1 <<  8) ;- (SUPC) Wake Up Input Enable 8
AT91C_SUPC_WKUPEN9        EQU (0x1 <<  9) ;- (SUPC) Wake Up Input Enable 9
AT91C_SUPC_WKUPEN10       EQU (0x1 << 10) ;- (SUPC) Wake Up Input Enable 10
AT91C_SUPC_WKUPEN11       EQU (0x1 << 11) ;- (SUPC) Wake Up Input Enable 11
AT91C_SUPC_WKUPEN12       EQU (0x1 << 12) ;- (SUPC) Wake Up Input Enable 12
AT91C_SUPC_WKUPEN13       EQU (0x1 << 13) ;- (SUPC) Wake Up Input Enable 13
AT91C_SUPC_WKUPEN14       EQU (0x1 << 14) ;- (SUPC) Wake Up Input Enable 14
AT91C_SUPC_WKUPEN15       EQU (0x1 << 15) ;- (SUPC) Wake Up Input Enable 15
AT91C_SUPC_WKUPT0         EQU (0x1 << 16) ;- (SUPC) Wake Up Input Transition 0
AT91C_SUPC_WKUPT1         EQU (0x1 << 17) ;- (SUPC) Wake Up Input Transition 1
AT91C_SUPC_WKUPT2         EQU (0x1 << 18) ;- (SUPC) Wake Up Input Transition 2
AT91C_SUPC_WKUPT3         EQU (0x1 << 19) ;- (SUPC) Wake Up Input Transition 3
AT91C_SUPC_WKUPT4         EQU (0x1 << 20) ;- (SUPC) Wake Up Input Transition 4
AT91C_SUPC_WKUPT5         EQU (0x1 << 21) ;- (SUPC) Wake Up Input Transition 5
AT91C_SUPC_WKUPT6         EQU (0x1 << 22) ;- (SUPC) Wake Up Input Transition 6
AT91C_SUPC_WKUPT7         EQU (0x1 << 23) ;- (SUPC) Wake Up Input Transition 7
AT91C_SUPC_WKUPT8         EQU (0x1 << 24) ;- (SUPC) Wake Up Input Transition 8
AT91C_SUPC_WKUPT9         EQU (0x1 << 25) ;- (SUPC) Wake Up Input Transition 9
AT91C_SUPC_WKUPT10        EQU (0x1 << 26) ;- (SUPC) Wake Up Input Transition 10
AT91C_SUPC_WKUPT11        EQU (0x1 << 27) ;- (SUPC) Wake Up Input Transition 11
AT91C_SUPC_WKUPT12        EQU (0x1 << 28) ;- (SUPC) Wake Up Input Transition 12
AT91C_SUPC_WKUPT13        EQU (0x1 << 29) ;- (SUPC) Wake Up Input Transition 13
AT91C_SUPC_WKUPT14        EQU (0x1 << 30) ;- (SUPC) Wake Up Input Transition 14
AT91C_SUPC_WKUPT15        EQU (0x1 << 31) ;- (SUPC) Wake Up Input Transition 15
// - -------- SUPC_SR : (SUPC Offset: 0x14) Status Register -------- 
AT91C_SUPC_FWUPS          EQU (0x1 <<  0) ;- (SUPC) Force Wake Up Status
AT91C_SUPC_WKUPS          EQU (0x1 <<  1) ;- (SUPC) Wake Up Status
AT91C_SUPC_BODWS          EQU (0x1 <<  2) ;- (SUPC) BOD Detection Wake Up Status
AT91C_SUPC_VRRSTS         EQU (0x1 <<  3) ;- (SUPC) Voltage regulation Loss Reset Status
AT91C_SUPC_BODRSTS        EQU (0x1 <<  4) ;- (SUPC) BOD detection Reset Status
AT91C_SUPC_BODS           EQU (0x1 <<  5) ;- (SUPC) BOD Status
AT91C_SUPC_BROWNOUT       EQU (0x1 <<  6) ;- (SUPC) BOD Output Status
AT91C_SUPC_OSCSEL         EQU (0x1 <<  7) ;- (SUPC) 32kHz Oscillator Selection Status
AT91C_SUPC_LCDS           EQU (0x1 <<  8) ;- (SUPC) LCD Status
AT91C_SUPC_GPBRS          EQU (0x1 <<  9) ;- (SUPC) General Purpose Back-up registers Status
AT91C_SUPC_RTS            EQU (0x1 << 10) ;- (SUPC) Clock Status
AT91C_SUPC_FLASHS         EQU (0x1 << 11) ;- (SUPC) FLASH Memory Status
AT91C_SUPC_FWUPIS         EQU (0x1 << 12) ;- (SUPC) WKUP Input Status
AT91C_SUPC_WKUPIS0        EQU (0x1 << 16) ;- (SUPC) WKUP Input 0 Status
AT91C_SUPC_WKUPIS1        EQU (0x1 << 17) ;- (SUPC) WKUP Input 1 Status
AT91C_SUPC_WKUPIS2        EQU (0x1 << 18) ;- (SUPC) WKUP Input 2 Status
AT91C_SUPC_WKUPIS3        EQU (0x1 << 19) ;- (SUPC) WKUP Input 3 Status
AT91C_SUPC_WKUPIS4        EQU (0x1 << 20) ;- (SUPC) WKUP Input 4 Status
AT91C_SUPC_WKUPIS5        EQU (0x1 << 21) ;- (SUPC) WKUP Input 5 Status
AT91C_SUPC_WKUPIS6        EQU (0x1 << 22) ;- (SUPC) WKUP Input 6 Status
AT91C_SUPC_WKUPIS7        EQU (0x1 << 23) ;- (SUPC) WKUP Input 7 Status
AT91C_SUPC_WKUPIS8        EQU (0x1 << 24) ;- (SUPC) WKUP Input 8 Status
AT91C_SUPC_WKUPIS9        EQU (0x1 << 25) ;- (SUPC) WKUP Input 9 Status
AT91C_SUPC_WKUPIS10       EQU (0x1 << 26) ;- (SUPC) WKUP Input 10 Status
AT91C_SUPC_WKUPIS11       EQU (0x1 << 27) ;- (SUPC) WKUP Input 11 Status
AT91C_SUPC_WKUPIS12       EQU (0x1 << 28) ;- (SUPC) WKUP Input 12 Status
AT91C_SUPC_WKUPIS13       EQU (0x1 << 29) ;- (SUPC) WKUP Input 13 Status
AT91C_SUPC_WKUPIS14       EQU (0x1 << 30) ;- (SUPC) WKUP Input 14 Status
AT91C_SUPC_WKUPIS15       EQU (0x1 << 31) ;- (SUPC) WKUP Input 15 Status
// - -------- SUPC_FWUTR : (SUPC Offset: 0x18) Flash Wake Up Timer Register -------- 
AT91C_SUPC_FWUT           EQU (0x3FF <<  0) ;- (SUPC) Flash Wake Up Timer

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface
// - *****************************************************************************
// - -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- 
AT91C_RTTC_RTPRES         EQU (0xFFFF <<  0) ;- (RTTC) Real-time Timer Prescaler Value
AT91C_RTTC_ALMIEN         EQU (0x1 << 16) ;- (RTTC) Alarm Interrupt Enable
AT91C_RTTC_RTTINCIEN      EQU (0x1 << 17) ;- (RTTC) Real Time Timer Increment Interrupt Enable
AT91C_RTTC_RTTRST         EQU (0x1 << 18) ;- (RTTC) Real Time Timer Restart
// - -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- 
AT91C_RTTC_ALMV           EQU (0x0 <<  0) ;- (RTTC) Alarm Value
// - -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- 
AT91C_RTTC_CRTV           EQU (0x0 <<  0) ;- (RTTC) Current Real-time Value
// - -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- 
AT91C_RTTC_ALMS           EQU (0x1 <<  0) ;- (RTTC) Real-time Alarm Status
AT91C_RTTC_RTTINC         EQU (0x1 <<  1) ;- (RTTC) Real-time Timer Increment

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface
// - *****************************************************************************
// - -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- 
AT91C_WDTC_WDRSTT         EQU (0x1 <<  0) ;- (WDTC) Watchdog Restart
AT91C_WDTC_KEY            EQU (0xFF << 24) ;- (WDTC) Watchdog KEY Password
// - -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- 
AT91C_WDTC_WDV            EQU (0xFFF <<  0) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDFIEN         EQU (0x1 << 12) ;- (WDTC) Watchdog Fault Interrupt Enable
AT91C_WDTC_WDRSTEN        EQU (0x1 << 13) ;- (WDTC) Watchdog Reset Enable
AT91C_WDTC_WDRPROC        EQU (0x1 << 14) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDDIS          EQU (0x1 << 15) ;- (WDTC) Watchdog Disable
AT91C_WDTC_WDD            EQU (0xFFF << 16) ;- (WDTC) Watchdog Delta Value
AT91C_WDTC_WDDBGHLT       EQU (0x1 << 28) ;- (WDTC) Watchdog Debug Halt
AT91C_WDTC_WDIDLEHLT      EQU (0x1 << 29) ;- (WDTC) Watchdog Idle Halt
// - -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- 
AT91C_WDTC_WDUNF          EQU (0x1 <<  0) ;- (WDTC) Watchdog Underflow
AT91C_WDTC_WDERR          EQU (0x1 <<  1) ;- (WDTC) Watchdog Error

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface
// - *****************************************************************************
// - -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- 
AT91C_RTC_UPDTIM          EQU (0x1 <<  0) ;- (RTC) Update Request Time Register
AT91C_RTC_UPDCAL          EQU (0x1 <<  1) ;- (RTC) Update Request Calendar Register
AT91C_RTC_TIMEVSEL        EQU (0x3 <<  8) ;- (RTC) Time Event Selection
AT91C_RTC_TIMEVSEL_MINUTE EQU (0x0 <<  8) ;- (RTC) Minute change.
AT91C_RTC_TIMEVSEL_HOUR   EQU (0x1 <<  8) ;- (RTC) Hour change.
AT91C_RTC_TIMEVSEL_DAY24  EQU (0x2 <<  8) ;- (RTC) Every day at midnight.
AT91C_RTC_TIMEVSEL_DAY12  EQU (0x3 <<  8) ;- (RTC) Every day at noon.
AT91C_RTC_CALEVSEL        EQU (0x3 << 16) ;- (RTC) Calendar Event Selection
AT91C_RTC_CALEVSEL_WEEK   EQU (0x0 << 16) ;- (RTC) Week change (every Monday at time 00:00:00).
AT91C_RTC_CALEVSEL_MONTH  EQU (0x1 << 16) ;- (RTC) Month change (every 01 of each month at time 00:00:00).
AT91C_RTC_CALEVSEL_YEAR   EQU (0x2 << 16) ;- (RTC) Year change (every January 1 at time 00:00:00).
// - -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- 
AT91C_RTC_HRMOD           EQU (0x1 <<  0) ;- (RTC) 12-24 hour Mode
// - -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- 
AT91C_RTC_SEC             EQU (0x7F <<  0) ;- (RTC) Current Second
AT91C_RTC_MIN             EQU (0x7F <<  8) ;- (RTC) Current Minute
AT91C_RTC_HOUR            EQU (0x3F << 16) ;- (RTC) Current Hour
AT91C_RTC_AMPM            EQU (0x1 << 22) ;- (RTC) Ante Meridiem, Post Meridiem Indicator
// - -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- 
AT91C_RTC_CENT            EQU (0x3F <<  0) ;- (RTC) Current Century
AT91C_RTC_YEAR            EQU (0xFF <<  8) ;- (RTC) Current Year
AT91C_RTC_MONTH           EQU (0x1F << 16) ;- (RTC) Current Month
AT91C_RTC_DAY             EQU (0x7 << 21) ;- (RTC) Current Day
AT91C_RTC_DATE            EQU (0x3F << 24) ;- (RTC) Current Date
// - -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- 
AT91C_RTC_SECEN           EQU (0x1 <<  7) ;- (RTC) Second Alarm Enable
AT91C_RTC_MINEN           EQU (0x1 << 15) ;- (RTC) Minute Alarm
AT91C_RTC_HOUREN          EQU (0x1 << 23) ;- (RTC) Current Hour
// - -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- 
AT91C_RTC_MONTHEN         EQU (0x1 << 23) ;- (RTC) Month Alarm Enable
AT91C_RTC_DATEEN          EQU (0x1 << 31) ;- (RTC) Date Alarm Enable
// - -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- 
AT91C_RTC_ACKUPD          EQU (0x1 <<  0) ;- (RTC) Acknowledge for Update
AT91C_RTC_ALARM           EQU (0x1 <<  1) ;- (RTC) Alarm Flag
AT91C_RTC_SECEV           EQU (0x1 <<  2) ;- (RTC) Second Event
AT91C_RTC_TIMEV           EQU (0x1 <<  3) ;- (RTC) Time Event
AT91C_RTC_CALEV           EQU (0x1 <<  4) ;- (RTC) Calendar event
// - -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- 
// - -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- 
// - -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- 
// - -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- 
// - -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- 
AT91C_RTC_NVTIM           EQU (0x1 <<  0) ;- (RTC) Non valid Time
AT91C_RTC_NVCAL           EQU (0x1 <<  1) ;- (RTC) Non valid Calendar
AT91C_RTC_NVTIMALR        EQU (0x1 <<  2) ;- (RTC) Non valid time Alarm
AT91C_RTC_NVCALALR        EQU (0x1 <<  3) ;- (RTC) Nonvalid Calendar Alarm

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Analog to Digital Convertor
// - *****************************************************************************
// - -------- ADC_CR : (ADC Offset: 0x0) ADC Control Register -------- 
AT91C_ADC_SWRST           EQU (0x1 <<  0) ;- (ADC) Software Reset
AT91C_ADC_START           EQU (0x1 <<  1) ;- (ADC) Start Conversion
// - -------- ADC_MR : (ADC Offset: 0x4) ADC Mode Register -------- 
AT91C_ADC_TRGEN           EQU (0x1 <<  0) ;- (ADC) Trigger Enable
AT91C_ADC_TRGEN_DIS       EQU (0x0) ;- (ADC) Hradware triggers are disabled. Starting a conversion is only possible by software
AT91C_ADC_TRGEN_EN        EQU (0x1) ;- (ADC) Hardware trigger selected by TRGSEL field is enabled.
AT91C_ADC_TRGSEL          EQU (0x7 <<  1) ;- (ADC) Trigger Selection
AT91C_ADC_TRGSEL_EXT      EQU (0x0 <<  1) ;- (ADC) Selected TRGSEL = External Trigger
AT91C_ADC_TRGSEL_TIOA0    EQU (0x1 <<  1) ;- (ADC) Selected TRGSEL = TIAO0
AT91C_ADC_TRGSEL_TIOA1    EQU (0x2 <<  1) ;- (ADC) Selected TRGSEL = TIAO1
AT91C_ADC_TRGSEL_TIOA2    EQU (0x3 <<  1) ;- (ADC) Selected TRGSEL = TIAO2
AT91C_ADC_TRGSEL_PWM0_TRIG EQU (0x4 <<  1) ;- (ADC) Selected TRGSEL = PWM trigger
AT91C_ADC_TRGSEL_PWM1_TRIG EQU (0x5 <<  1) ;- (ADC) Selected TRGSEL = PWM Trigger
AT91C_ADC_TRGSEL_RESERVED EQU (0x6 <<  1) ;- (ADC) Selected TRGSEL = Reserved
AT91C_ADC_LOWRES          EQU (0x1 <<  4) ;- (ADC) Resolution.
AT91C_ADC_LOWRES_10_BIT   EQU (0x0 <<  4) ;- (ADC) 10-bit resolution
AT91C_ADC_LOWRES_8_BIT    EQU (0x1 <<  4) ;- (ADC) 8-bit resolution
AT91C_ADC_SLEEP           EQU (0x1 <<  5) ;- (ADC) Sleep Mode
AT91C_ADC_SLEEP_NORMAL_MODE EQU (0x0 <<  5) ;- (ADC) Normal Mode
AT91C_ADC_SLEEP_MODE      EQU (0x1 <<  5) ;- (ADC) Sleep Mode
AT91C_ADC_PRESCAL         EQU (0x3F <<  8) ;- (ADC) Prescaler rate selection
AT91C_ADC_STARTUP         EQU (0x1F << 16) ;- (ADC) Startup Time
AT91C_ADC_SHTIM           EQU (0xF << 24) ;- (ADC) Sample & Hold Time
// - -------- 	ADC_CHER : (ADC Offset: 0x10) ADC Channel Enable Register -------- 
AT91C_ADC_CH0             EQU (0x1 <<  0) ;- (ADC) Channel 0
AT91C_ADC_CH1             EQU (0x1 <<  1) ;- (ADC) Channel 1
AT91C_ADC_CH2             EQU (0x1 <<  2) ;- (ADC) Channel 2
AT91C_ADC_CH3             EQU (0x1 <<  3) ;- (ADC) Channel 3
AT91C_ADC_CH4             EQU (0x1 <<  4) ;- (ADC) Channel 4
AT91C_ADC_CH5             EQU (0x1 <<  5) ;- (ADC) Channel 5
AT91C_ADC_CH6             EQU (0x1 <<  6) ;- (ADC) Channel 6
AT91C_ADC_CH7             EQU (0x1 <<  7) ;- (ADC) Channel 7
// - -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- 
// - -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- 
// - -------- ADC_SR : (ADC Offset: 0x1c) ADC Status Register -------- 
AT91C_ADC_EOC0            EQU (0x1 <<  0) ;- (ADC) End of Conversion
AT91C_ADC_EOC1            EQU (0x1 <<  1) ;- (ADC) End of Conversion
AT91C_ADC_EOC2            EQU (0x1 <<  2) ;- (ADC) End of Conversion
AT91C_ADC_EOC3            EQU (0x1 <<  3) ;- (ADC) End of Conversion
AT91C_ADC_EOC4            EQU (0x1 <<  4) ;- (ADC) End of Conversion
AT91C_ADC_EOC5            EQU (0x1 <<  5) ;- (ADC) End of Conversion
AT91C_ADC_EOC6            EQU (0x1 <<  6) ;- (ADC) End of Conversion
AT91C_ADC_EOC7            EQU (0x1 <<  7) ;- (ADC) End of Conversion
AT91C_ADC_OVRE0           EQU (0x1 <<  8) ;- (ADC) Overrun Error
AT91C_ADC_OVRE1           EQU (0x1 <<  9) ;- (ADC) Overrun Error
AT91C_ADC_OVRE2           EQU (0x1 << 10) ;- (ADC) Overrun Error
AT91C_ADC_OVRE3           EQU (0x1 << 11) ;- (ADC) Overrun Error
AT91C_ADC_OVRE4           EQU (0x1 << 12) ;- (ADC) Overrun Error
AT91C_ADC_OVRE5           EQU (0x1 << 13) ;- (ADC) Overrun Error
AT91C_ADC_OVRE6           EQU (0x1 << 14) ;- (ADC) Overrun Error
AT91C_ADC_OVRE7           EQU (0x1 << 15) ;- (ADC) Overrun Error
AT91C_ADC_DRDY            EQU (0x1 << 16) ;- (ADC) Data Ready
AT91C_ADC_GOVRE           EQU (0x1 << 17) ;- (ADC) General Overrun
AT91C_ADC_ENDRX           EQU (0x1 << 18) ;- (ADC) End of Receiver Transfer
AT91C_ADC_RXBUFF          EQU (0x1 << 19) ;- (ADC) RXBUFF Interrupt
// - -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- 
AT91C_ADC_LDATA           EQU (0x3FF <<  0) ;- (ADC) Last Data Converted
// - -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- 
// - -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- 
// - -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- 
// - -------- ADC_CDR0 : (ADC Offset: 0x30) ADC Channel Data Register 0 -------- 
AT91C_ADC_DATA            EQU (0x3FF <<  0) ;- (ADC) Converted Data
// - -------- ADC_CDR1 : (ADC Offset: 0x34) ADC Channel Data Register 1 -------- 
// - -------- ADC_CDR2 : (ADC Offset: 0x38) ADC Channel Data Register 2 -------- 
// - -------- ADC_CDR3 : (ADC Offset: 0x3c) ADC Channel Data Register 3 -------- 
// - -------- ADC_CDR4 : (ADC Offset: 0x40) ADC Channel Data Register 4 -------- 
// - -------- ADC_CDR5 : (ADC Offset: 0x44) ADC Channel Data Register 5 -------- 
// - -------- ADC_CDR6 : (ADC Offset: 0x48) ADC Channel Data Register 6 -------- 
// - -------- ADC_CDR7 : (ADC Offset: 0x4c) ADC Channel Data Register 7 -------- 
// - -------- ADC_ACR : (ADC Offset: 0x64) ADC Analog Controler Register -------- 
AT91C_ADC_GAIN            EQU (0x3 <<  0) ;- (ADC) Input Gain
AT91C_ADC_IBCTL           EQU (0x3 <<  6) ;- (ADC) Bias Current Control
AT91C_ADC_IBCTL_00        EQU (0x0 <<  6) ;- (ADC) typ - 20%
AT91C_ADC_IBCTL_01        EQU (0x1 <<  6) ;- (ADC) typ
AT91C_ADC_IBCTL_10        EQU (0x2 <<  6) ;- (ADC) typ + 20%
AT91C_ADC_IBCTL_11        EQU (0x3 <<  6) ;- (ADC) typ + 40%
AT91C_ADC_DIFF            EQU (0x1 << 16) ;- (ADC) Differential Mode
AT91C_ADC_OFFSET          EQU (0x1 << 17) ;- (ADC) Input OFFSET
// - -------- ADC_EMR : (ADC Offset: 0x68) ADC Extended Mode Register -------- 
AT91C_OFFMODES            EQU (0x1 <<  0) ;- (ADC) Off Mode if
AT91C_OFF_MODE_STARTUP_TIME EQU (0x1 << 16) ;- (ADC) Startup Time
// - -------- ADC_VER : (ADC Offset: 0xfc) ADC VER -------- 
AT91C_ADC_VER             EQU (0xF <<  0) ;- (ADC) ADC VER

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface
// - *****************************************************************************
// - -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- 
AT91C_TC_CLKEN            EQU (0x1 <<  0) ;- (TC) Counter Clock Enable Command
AT91C_TC_CLKDIS           EQU (0x1 <<  1) ;- (TC) Counter Clock Disable Command
AT91C_TC_SWTRG            EQU (0x1 <<  2) ;- (TC) Software Trigger Command
// - -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- 
AT91C_TC_CLKS             EQU (0x7 <<  0) ;- (TC) Clock Selection
AT91C_TC_CLKS_TIMER_DIV1_CLOCK EQU (0x0) ;- (TC) Clock selected: TIMER_DIV1_CLOCK
AT91C_TC_CLKS_TIMER_DIV2_CLOCK EQU (0x1) ;- (TC) Clock selected: TIMER_DIV2_CLOCK
AT91C_TC_CLKS_TIMER_DIV3_CLOCK EQU (0x2) ;- (TC) Clock selected: TIMER_DIV3_CLOCK
AT91C_TC_CLKS_TIMER_DIV4_CLOCK EQU (0x3) ;- (TC) Clock selected: TIMER_DIV4_CLOCK
AT91C_TC_CLKS_TIMER_DIV5_CLOCK EQU (0x4) ;- (TC) Clock selected: TIMER_DIV5_CLOCK
AT91C_TC_CLKS_XC0         EQU (0x5) ;- (TC) Clock selected: XC0
AT91C_TC_CLKS_XC1         EQU (0x6) ;- (TC) Clock selected: XC1
AT91C_TC_CLKS_XC2         EQU (0x7) ;- (TC) Clock selected: XC2
AT91C_TC_CLKI             EQU (0x1 <<  3) ;- (TC) Clock Invert
AT91C_TC_BURST            EQU (0x3 <<  4) ;- (TC) Burst Signal Selection
AT91C_TC_BURST_NONE       EQU (0x0 <<  4) ;- (TC) The clock is not gated by an external signal
AT91C_TC_BURST_XC0        EQU (0x1 <<  4) ;- (TC) XC0 is ANDed with the selected clock
AT91C_TC_BURST_XC1        EQU (0x2 <<  4) ;- (TC) XC1 is ANDed with the selected clock
AT91C_TC_BURST_XC2        EQU (0x3 <<  4) ;- (TC) XC2 is ANDed with the selected clock
AT91C_TC_CPCSTOP          EQU (0x1 <<  6) ;- (TC) Counter Clock Stopped with RC Compare
AT91C_TC_LDBSTOP          EQU (0x1 <<  6) ;- (TC) Counter Clock Stopped with RB Loading
AT91C_TC_CPCDIS           EQU (0x1 <<  7) ;- (TC) Counter Clock Disable with RC Compare
AT91C_TC_LDBDIS           EQU (0x1 <<  7) ;- (TC) Counter Clock Disabled with RB Loading
AT91C_TC_ETRGEDG          EQU (0x3 <<  8) ;- (TC) External Trigger Edge Selection
AT91C_TC_ETRGEDG_NONE     EQU (0x0 <<  8) ;- (TC) Edge: None
AT91C_TC_ETRGEDG_RISING   EQU (0x1 <<  8) ;- (TC) Edge: rising edge
AT91C_TC_ETRGEDG_FALLING  EQU (0x2 <<  8) ;- (TC) Edge: falling edge
AT91C_TC_ETRGEDG_BOTH     EQU (0x3 <<  8) ;- (TC) Edge: each edge
AT91C_TC_EEVTEDG          EQU (0x3 <<  8) ;- (TC) External Event Edge Selection
AT91C_TC_EEVTEDG_NONE     EQU (0x0 <<  8) ;- (TC) Edge: None
AT91C_TC_EEVTEDG_RISING   EQU (0x1 <<  8) ;- (TC) Edge: rising edge
AT91C_TC_EEVTEDG_FALLING  EQU (0x2 <<  8) ;- (TC) Edge: falling edge
AT91C_TC_EEVTEDG_BOTH     EQU (0x3 <<  8) ;- (TC) Edge: each edge
AT91C_TC_EEVT             EQU (0x3 << 10) ;- (TC) External Event  Selection
AT91C_TC_EEVT_TIOB        EQU (0x0 << 10) ;- (TC) Signal selected as external event: TIOB TIOB direction: input
AT91C_TC_EEVT_XC0         EQU (0x1 << 10) ;- (TC) Signal selected as external event: XC0 TIOB direction: output
AT91C_TC_EEVT_XC1         EQU (0x2 << 10) ;- (TC) Signal selected as external event: XC1 TIOB direction: output
AT91C_TC_EEVT_XC2         EQU (0x3 << 10) ;- (TC) Signal selected as external event: XC2 TIOB direction: output
AT91C_TC_ABETRG           EQU (0x1 << 10) ;- (TC) TIOA or TIOB External Trigger Selection
AT91C_TC_ENETRG           EQU (0x1 << 12) ;- (TC) External Event Trigger enable
AT91C_TC_WAVESEL          EQU (0x3 << 13) ;- (TC) Waveform  Selection
AT91C_TC_WAVESEL_UP       EQU (0x0 << 13) ;- (TC) UP mode without atomatic trigger on RC Compare
AT91C_TC_WAVESEL_UPDOWN   EQU (0x1 << 13) ;- (TC) UPDOWN mode without automatic trigger on RC Compare
AT91C_TC_WAVESEL_UP_AUTO  EQU (0x2 << 13) ;- (TC) UP mode with automatic trigger on RC Compare
AT91C_TC_WAVESEL_UPDOWN_AUTO EQU (0x3 << 13) ;- (TC) UPDOWN mode with automatic trigger on RC Compare
AT91C_TC_CPCTRG           EQU (0x1 << 14) ;- (TC) RC Compare Trigger Enable
AT91C_TC_WAVE             EQU (0x1 << 15) ;- (TC) 
AT91C_TC_ACPA             EQU (0x3 << 16) ;- (TC) RA Compare Effect on TIOA
AT91C_TC_ACPA_NONE        EQU (0x0 << 16) ;- (TC) Effect: none
AT91C_TC_ACPA_SET         EQU (0x1 << 16) ;- (TC) Effect: set
AT91C_TC_ACPA_CLEAR       EQU (0x2 << 16) ;- (TC) Effect: clear
AT91C_TC_ACPA_TOGGLE      EQU (0x3 << 16) ;- (TC) Effect: toggle
AT91C_TC_LDRA             EQU (0x3 << 16) ;- (TC) RA Loading Selection
AT91C_TC_LDRA_NONE        EQU (0x0 << 16) ;- (TC) Edge: None
AT91C_TC_LDRA_RISING      EQU (0x1 << 16) ;- (TC) Edge: rising edge of TIOA
AT91C_TC_LDRA_FALLING     EQU (0x2 << 16) ;- (TC) Edge: falling edge of TIOA
AT91C_TC_LDRA_BOTH        EQU (0x3 << 16) ;- (TC) Edge: each edge of TIOA
AT91C_TC_ACPC             EQU (0x3 << 18) ;- (TC) RC Compare Effect on TIOA
AT91C_TC_ACPC_NONE        EQU (0x0 << 18) ;- (TC) Effect: none
AT91C_TC_ACPC_SET         EQU (0x1 << 18) ;- (TC) Effect: set
AT91C_TC_ACPC_CLEAR       EQU (0x2 << 18) ;- (TC) Effect: clear
AT91C_TC_ACPC_TOGGLE      EQU (0x3 << 18) ;- (TC) Effect: toggle
AT91C_TC_LDRB             EQU (0x3 << 18) ;- (TC) RB Loading Selection
AT91C_TC_LDRB_NONE        EQU (0x0 << 18) ;- (TC) Edge: None
AT91C_TC_LDRB_RISING      EQU (0x1 << 18) ;- (TC) Edge: rising edge of TIOA
AT91C_TC_LDRB_FALLING     EQU (0x2 << 18) ;- (TC) Edge: falling edge of TIOA
AT91C_TC_LDRB_BOTH        EQU (0x3 << 18) ;- (TC) Edge: each edge of TIOA
AT91C_TC_AEEVT            EQU (0x3 << 20) ;- (TC) External Event Effect on TIOA
AT91C_TC_AEEVT_NONE       EQU (0x0 << 20) ;- (TC) Effect: none
AT91C_TC_AEEVT_SET        EQU (0x1 << 20) ;- (TC) Effect: set
AT91C_TC_AEEVT_CLEAR      EQU (0x2 << 20) ;- (TC) Effect: clear
AT91C_TC_AEEVT_TOGGLE     EQU (0x3 << 20) ;- (TC) Effect: toggle
AT91C_TC_ASWTRG           EQU (0x3 << 22) ;- (TC) Software Trigger Effect on TIOA
AT91C_TC_ASWTRG_NONE      EQU (0x0 << 22) ;- (TC) Effect: none
AT91C_TC_ASWTRG_SET       EQU (0x1 << 22) ;- (TC) Effect: set
AT91C_TC_ASWTRG_CLEAR     EQU (0x2 << 22) ;- (TC) Effect: clear
AT91C_TC_ASWTRG_TOGGLE    EQU (0x3 << 22) ;- (TC) Effect: toggle
AT91C_TC_BCPB             EQU (0x3 << 24) ;- (TC) RB Compare Effect on TIOB
AT91C_TC_BCPB_NONE        EQU (0x0 << 24) ;- (TC) Effect: none
AT91C_TC_BCPB_SET         EQU (0x1 << 24) ;- (TC) Effect: set
AT91C_TC_BCPB_CLEAR       EQU (0x2 << 24) ;- (TC) Effect: clear
AT91C_TC_BCPB_TOGGLE      EQU (0x3 << 24) ;- (TC) Effect: toggle
AT91C_TC_BCPC             EQU (0x3 << 26) ;- (TC) RC Compare Effect on TIOB
AT91C_TC_BCPC_NONE        EQU (0x0 << 26) ;- (TC) Effect: none
AT91C_TC_BCPC_SET         EQU (0x1 << 26) ;- (TC) Effect: set
AT91C_TC_BCPC_CLEAR       EQU (0x2 << 26) ;- (TC) Effect: clear
AT91C_TC_BCPC_TOGGLE      EQU (0x3 << 26) ;- (TC) Effect: toggle
AT91C_TC_BEEVT            EQU (0x3 << 28) ;- (TC) External Event Effect on TIOB
AT91C_TC_BEEVT_NONE       EQU (0x0 << 28) ;- (TC) Effect: none
AT91C_TC_BEEVT_SET        EQU (0x1 << 28) ;- (TC) Effect: set
AT91C_TC_BEEVT_CLEAR      EQU (0x2 << 28) ;- (TC) Effect: clear
AT91C_TC_BEEVT_TOGGLE     EQU (0x3 << 28) ;- (TC) Effect: toggle
AT91C_TC_BSWTRG           EQU (0x3 << 30) ;- (TC) Software Trigger Effect on TIOB
AT91C_TC_BSWTRG_NONE      EQU (0x0 << 30) ;- (TC) Effect: none
AT91C_TC_BSWTRG_SET       EQU (0x1 << 30) ;- (TC) Effect: set
AT91C_TC_BSWTRG_CLEAR     EQU (0x2 << 30) ;- (TC) Effect: clear
AT91C_TC_BSWTRG_TOGGLE    EQU (0x3 << 30) ;- (TC) Effect: toggle
// - -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- 
AT91C_TC_COVFS            EQU (0x1 <<  0) ;- (TC) Counter Overflow
AT91C_TC_LOVRS            EQU (0x1 <<  1) ;- (TC) Load Overrun
AT91C_TC_CPAS             EQU (0x1 <<  2) ;- (TC) RA Compare
AT91C_TC_CPBS             EQU (0x1 <<  3) ;- (TC) RB Compare
AT91C_TC_CPCS             EQU (0x1 <<  4) ;- (TC) RC Compare
AT91C_TC_LDRAS            EQU (0x1 <<  5) ;- (TC) RA Loading
AT91C_TC_LDRBS            EQU (0x1 <<  6) ;- (TC) RB Loading
AT91C_TC_ETRGS            EQU (0x1 <<  7) ;- (TC) External Trigger
AT91C_TC_CLKSTA           EQU (0x1 << 16) ;- (TC) Clock Enabling
AT91C_TC_MTIOA            EQU (0x1 << 17) ;- (TC) TIOA Mirror
AT91C_TC_MTIOB            EQU (0x1 << 18) ;- (TC) TIOA Mirror
// - -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- 
// - -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- 
// - -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Timer Counter Interface
// - *****************************************************************************
// - -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- 
AT91C_TCB_SYNC            EQU (0x1 <<  0) ;- (TCB) Synchro Command
// - -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- 
AT91C_TCB_TC0XC0S         EQU (0x3 <<  0) ;- (TCB) External Clock Signal 0 Selection
AT91C_TCB_TC0XC0S_TCLK0   EQU (0x0) ;- (TCB) TCLK0 connected to XC0
AT91C_TCB_TC0XC0S_NONE    EQU (0x1) ;- (TCB) None signal connected to XC0
AT91C_TCB_TC0XC0S_TIOA1   EQU (0x2) ;- (TCB) TIOA1 connected to XC0
AT91C_TCB_TC0XC0S_TIOA2   EQU (0x3) ;- (TCB) TIOA2 connected to XC0
AT91C_TCB_TC1XC1S         EQU (0x3 <<  2) ;- (TCB) External Clock Signal 1 Selection
AT91C_TCB_TC1XC1S_TCLK1   EQU (0x0 <<  2) ;- (TCB) TCLK1 connected to XC1
AT91C_TCB_TC1XC1S_NONE    EQU (0x1 <<  2) ;- (TCB) None signal connected to XC1
AT91C_TCB_TC1XC1S_TIOA0   EQU (0x2 <<  2) ;- (TCB) TIOA0 connected to XC1
AT91C_TCB_TC1XC1S_TIOA2   EQU (0x3 <<  2) ;- (TCB) TIOA2 connected to XC1
AT91C_TCB_TC2XC2S         EQU (0x3 <<  4) ;- (TCB) External Clock Signal 2 Selection
AT91C_TCB_TC2XC2S_TCLK2   EQU (0x0 <<  4) ;- (TCB) TCLK2 connected to XC2
AT91C_TCB_TC2XC2S_NONE    EQU (0x1 <<  4) ;- (TCB) None signal connected to XC2
AT91C_TCB_TC2XC2S_TIOA0   EQU (0x2 <<  4) ;- (TCB) TIOA0 connected to XC2
AT91C_TCB_TC2XC2S_TIOA1   EQU (0x3 <<  4) ;- (TCB) TIOA2 connected to XC2

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Embedded Flash Controller 2.0
// - *****************************************************************************
// - -------- EFC_FMR : (EFC Offset: 0x0) EFC Flash Mode Register -------- 
AT91C_EFC_FRDY            EQU (0x1 <<  0) ;- (EFC) Ready Interrupt Enable
AT91C_EFC_FWS             EQU (0xF <<  8) ;- (EFC) Flash Wait State.
AT91C_EFC_FWS_0WS         EQU (0x0 <<  8) ;- (EFC) 0 Wait State
AT91C_EFC_FWS_1WS         EQU (0x1 <<  8) ;- (EFC) 1 Wait State
AT91C_EFC_FWS_2WS         EQU (0x2 <<  8) ;- (EFC) 2 Wait States
AT91C_EFC_FWS_3WS         EQU (0x3 <<  8) ;- (EFC) 3 Wait States
// - -------- EFC_FCR : (EFC Offset: 0x4) EFC Flash Command Register -------- 
AT91C_EFC_FCMD            EQU (0xFF <<  0) ;- (EFC) Flash Command
AT91C_EFC_FCMD_GETD       EQU (0x0) ;- (EFC) Get Flash Descriptor
AT91C_EFC_FCMD_WP         EQU (0x1) ;- (EFC) Write Page
AT91C_EFC_FCMD_WPL        EQU (0x2) ;- (EFC) Write Page and Lock
AT91C_EFC_FCMD_EWP        EQU (0x3) ;- (EFC) Erase Page and Write Page
AT91C_EFC_FCMD_EWPL       EQU (0x4) ;- (EFC) Erase Page and Write Page then Lock
AT91C_EFC_FCMD_EA         EQU (0x5) ;- (EFC) Erase All
AT91C_EFC_FCMD_EPL        EQU (0x6) ;- (EFC) Erase Plane
AT91C_EFC_FCMD_EPA        EQU (0x7) ;- (EFC) Erase Pages
AT91C_EFC_FCMD_SLB        EQU (0x8) ;- (EFC) Set Lock Bit
AT91C_EFC_FCMD_CLB        EQU (0x9) ;- (EFC) Clear Lock Bit
AT91C_EFC_FCMD_GLB        EQU (0xA) ;- (EFC) Get Lock Bit
AT91C_EFC_FCMD_SFB        EQU (0xB) ;- (EFC) Set Fuse Bit
AT91C_EFC_FCMD_CFB        EQU (0xC) ;- (EFC) Clear Fuse Bit
AT91C_EFC_FCMD_GFB        EQU (0xD) ;- (EFC) Get Fuse Bit
AT91C_EFC_FCMD_STUI       EQU (0xE) ;- (EFC) Start Read Unique ID
AT91C_EFC_FCMD_SPUI       EQU (0xF) ;- (EFC) Stop Read Unique ID
AT91C_EFC_FARG            EQU (0xFFFF <<  8) ;- (EFC) Flash Command Argument
AT91C_EFC_FKEY            EQU (0xFF << 24) ;- (EFC) Flash Writing Protection Key
// - -------- EFC_FSR : (EFC Offset: 0x8) EFC Flash Status Register -------- 
AT91C_EFC_FRDY_S          EQU (0x1 <<  0) ;- (EFC) Flash Ready Status
AT91C_EFC_FCMDE           EQU (0x1 <<  1) ;- (EFC) Flash Command Error Status
AT91C_EFC_LOCKE           EQU (0x1 <<  2) ;- (EFC) Flash Lock Error Status
// - -------- EFC_FRR : (EFC Offset: 0xc) EFC Flash Result Register -------- 
AT91C_EFC_FVALUE          EQU (0x0 <<  0) ;- (EFC) Flash Result Value

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Multimedia Card Interface
// - *****************************************************************************
// - -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- 
AT91C_MCI_MCIEN           EQU (0x1 <<  0) ;- (MCI) Multimedia Interface Enable
AT91C_MCI_MCIEN_0         EQU (0x0) ;- (MCI) No effect
AT91C_MCI_MCIEN_1         EQU (0x1) ;- (MCI) Enable the MultiMedia Interface if MCIDIS is 0
AT91C_MCI_MCIDIS          EQU (0x1 <<  1) ;- (MCI) Multimedia Interface Disable
AT91C_MCI_MCIDIS_0        EQU (0x0 <<  1) ;- (MCI) No effect
AT91C_MCI_MCIDIS_1        EQU (0x1 <<  1) ;- (MCI) Disable the MultiMedia Interface
AT91C_MCI_PWSEN           EQU (0x1 <<  2) ;- (MCI) Power Save Mode Enable
AT91C_MCI_PWSEN_0         EQU (0x0 <<  2) ;- (MCI) No effect
AT91C_MCI_PWSEN_1         EQU (0x1 <<  2) ;- (MCI) Enable the Power-saving mode if PWSDIS is 0.
AT91C_MCI_PWSDIS          EQU (0x1 <<  3) ;- (MCI) Power Save Mode Disable
AT91C_MCI_PWSDIS_0        EQU (0x0 <<  3) ;- (MCI) No effect
AT91C_MCI_PWSDIS_1        EQU (0x1 <<  3) ;- (MCI) Disable the Power-saving mode.
AT91C_MCI_IOWAITEN        EQU (0x1 <<  4) ;- (MCI) SDIO Read Wait Enable
AT91C_MCI_IOWAITEN_0      EQU (0x0 <<  4) ;- (MCI) No effect
AT91C_MCI_IOWAITEN_1      EQU (0x1 <<  4) ;- (MCI) Enables the SDIO Read Wait Operation.
AT91C_MCI_IOWAITDIS       EQU (0x1 <<  5) ;- (MCI) SDIO Read Wait Disable
AT91C_MCI_IOWAITDIS_0     EQU (0x0 <<  5) ;- (MCI) No effect
AT91C_MCI_IOWAITDIS_1     EQU (0x1 <<  5) ;- (MCI) Disables the SDIO Read Wait Operation.
AT91C_MCI_SWRST           EQU (0x1 <<  7) ;- (MCI) MCI Software reset
AT91C_MCI_SWRST_0         EQU (0x0 <<  7) ;- (MCI) No effect
AT91C_MCI_SWRST_1         EQU (0x1 <<  7) ;- (MCI) Resets the MCI
// - -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- 
AT91C_MCI_CLKDIV          EQU (0xFF <<  0) ;- (MCI) Clock Divider
AT91C_MCI_PWSDIV          EQU (0x7 <<  8) ;- (MCI) Power Saving Divider
AT91C_MCI_RDPROOF         EQU (0x1 << 11) ;- (MCI) Read Proof Enable
AT91C_MCI_RDPROOF_DISABLE EQU (0x0 << 11) ;- (MCI) Disables Read Proof
AT91C_MCI_RDPROOF_ENABLE  EQU (0x1 << 11) ;- (MCI) Enables Read Proof
AT91C_MCI_WRPROOF         EQU (0x1 << 12) ;- (MCI) Write Proof Enable
AT91C_MCI_WRPROOF_DISABLE EQU (0x0 << 12) ;- (MCI) Disables Write Proof
AT91C_MCI_WRPROOF_ENABLE  EQU (0x1 << 12) ;- (MCI) Enables Write Proof
AT91C_MCI_PDCFBYTE        EQU (0x1 << 13) ;- (MCI) PDC Force Byte Transfer
AT91C_MCI_PDCFBYTE_DISABLE EQU (0x0 << 13) ;- (MCI) Disables PDC Force Byte Transfer
AT91C_MCI_PDCFBYTE_ENABLE EQU (0x1 << 13) ;- (MCI) Enables PDC Force Byte Transfer
AT91C_MCI_PDCPADV         EQU (0x1 << 14) ;- (MCI) PDC Padding Value
AT91C_MCI_PDCMODE         EQU (0x1 << 15) ;- (MCI) PDC Oriented Mode
AT91C_MCI_PDCMODE_DISABLE EQU (0x0 << 15) ;- (MCI) Disables PDC Transfer
AT91C_MCI_PDCMODE_ENABLE  EQU (0x1 << 15) ;- (MCI) Enables PDC Transfer
AT91C_MCI_BLKLEN          EQU (0xFFFF << 16) ;- (MCI) Data Block Length
// - -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- 
AT91C_MCI_DTOCYC          EQU (0xF <<  0) ;- (MCI) Data Timeout Cycle Number
AT91C_MCI_DTOMUL          EQU (0x7 <<  4) ;- (MCI) Data Timeout Multiplier
AT91C_MCI_DTOMUL_1        EQU (0x0 <<  4) ;- (MCI) DTOCYC x 1
AT91C_MCI_DTOMUL_16       EQU (0x1 <<  4) ;- (MCI) DTOCYC x 16
AT91C_MCI_DTOMUL_128      EQU (0x2 <<  4) ;- (MCI) DTOCYC x 128
AT91C_MCI_DTOMUL_256      EQU (0x3 <<  4) ;- (MCI) DTOCYC x 256
AT91C_MCI_DTOMUL_1024     EQU (0x4 <<  4) ;- (MCI) DTOCYC x 1024
AT91C_MCI_DTOMUL_4096     EQU (0x5 <<  4) ;- (MCI) DTOCYC x 4096
AT91C_MCI_DTOMUL_65536    EQU (0x6 <<  4) ;- (MCI) DTOCYC x 65536
AT91C_MCI_DTOMUL_1048576  EQU (0x7 <<  4) ;- (MCI) DTOCYC x 1048576
// - -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- 
AT91C_MCI_SCDSEL          EQU (0x3 <<  0) ;- (MCI) SD Card/SDIO Selector
AT91C_MCI_SCDSEL_SLOTA    EQU (0x0) ;- (MCI) Slot A selected
AT91C_MCI_SCDSEL_SLOTB    EQU (0x1) ;- (MCI) Slot B selected
AT91C_MCI_SCDSEL_SLOTC    EQU (0x2) ;- (MCI) Slot C selected
AT91C_MCI_SCDSEL_SLOTD    EQU (0x3) ;- (MCI) Slot D selected
AT91C_MCI_SCDBUS          EQU (0x3 <<  6) ;- (MCI) SDCard/SDIO Bus Width
AT91C_MCI_SCDBUS_1BIT     EQU (0x0 <<  6) ;- (MCI) 1-bit data bus
AT91C_MCI_SCDBUS_4BITS    EQU (0x2 <<  6) ;- (MCI) 4-bits data bus
AT91C_MCI_SCDBUS_8BITS    EQU (0x3 <<  6) ;- (MCI) 8-bits data bus
// - -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- 
AT91C_MCI_CMDNB           EQU (0x3F <<  0) ;- (MCI) Command Number
AT91C_MCI_RSPTYP          EQU (0x3 <<  6) ;- (MCI) Response Type
AT91C_MCI_RSPTYP_NO       EQU (0x0 <<  6) ;- (MCI) No response
AT91C_MCI_RSPTYP_48       EQU (0x1 <<  6) ;- (MCI) 48-bit response
AT91C_MCI_RSPTYP_136      EQU (0x2 <<  6) ;- (MCI) 136-bit response
AT91C_MCI_RSPTYP_R1B      EQU (0x3 <<  6) ;- (MCI) R1b response
AT91C_MCI_SPCMD           EQU (0x7 <<  8) ;- (MCI) Special CMD
AT91C_MCI_SPCMD_NONE      EQU (0x0 <<  8) ;- (MCI) Not a special CMD
AT91C_MCI_SPCMD_INIT      EQU (0x1 <<  8) ;- (MCI) Initialization CMD
AT91C_MCI_SPCMD_SYNC      EQU (0x2 <<  8) ;- (MCI) Synchronized CMD
AT91C_MCI_SPCMD_CE_ATA    EQU (0x3 <<  8) ;- (MCI) CE-ATA Completion Signal disable CMD
AT91C_MCI_SPCMD_IT_CMD    EQU (0x4 <<  8) ;- (MCI) Interrupt command
AT91C_MCI_SPCMD_IT_REP    EQU (0x5 <<  8) ;- (MCI) Interrupt response
AT91C_MCI_OPDCMD          EQU (0x1 << 11) ;- (MCI) Open Drain Command
AT91C_MCI_OPDCMD_PUSHPULL EQU (0x0 << 11) ;- (MCI) Push/pull command
AT91C_MCI_OPDCMD_OPENDRAIN EQU (0x1 << 11) ;- (MCI) Open drain command
AT91C_MCI_MAXLAT          EQU (0x1 << 12) ;- (MCI) Maximum Latency for Command to respond
AT91C_MCI_MAXLAT_5        EQU (0x0 << 12) ;- (MCI) 5 cycles maximum latency
AT91C_MCI_MAXLAT_64       EQU (0x1 << 12) ;- (MCI) 64 cycles maximum latency
AT91C_MCI_TRCMD           EQU (0x3 << 16) ;- (MCI) Transfer CMD
AT91C_MCI_TRCMD_NO        EQU (0x0 << 16) ;- (MCI) No transfer
AT91C_MCI_TRCMD_START     EQU (0x1 << 16) ;- (MCI) Start transfer
AT91C_MCI_TRCMD_STOP      EQU (0x2 << 16) ;- (MCI) Stop transfer
AT91C_MCI_TRDIR           EQU (0x1 << 18) ;- (MCI) Transfer Direction
AT91C_MCI_TRDIR_WRITE     EQU (0x0 << 18) ;- (MCI) Write
AT91C_MCI_TRDIR_READ      EQU (0x1 << 18) ;- (MCI) Read
AT91C_MCI_TRTYP           EQU (0x7 << 19) ;- (MCI) Transfer Type
AT91C_MCI_TRTYP_BLOCK     EQU (0x0 << 19) ;- (MCI) MMC/SDCard Single Block Transfer type
AT91C_MCI_TRTYP_MULTIPLE  EQU (0x1 << 19) ;- (MCI) MMC/SDCard Multiple Block transfer type
AT91C_MCI_TRTYP_STREAM    EQU (0x2 << 19) ;- (MCI) MMC Stream transfer type
AT91C_MCI_TRTYP_SDIO_BYTE EQU (0x4 << 19) ;- (MCI) SDIO Byte transfer type
AT91C_MCI_TRTYP_SDIO_BLOCK EQU (0x5 << 19) ;- (MCI) SDIO Block transfer type
AT91C_MCI_IOSPCMD         EQU (0x3 << 24) ;- (MCI) SDIO Special Command
AT91C_MCI_IOSPCMD_NONE    EQU (0x0 << 24) ;- (MCI) NOT a special command
AT91C_MCI_IOSPCMD_SUSPEND EQU (0x1 << 24) ;- (MCI) SDIO Suspend Command
AT91C_MCI_IOSPCMD_RESUME  EQU (0x2 << 24) ;- (MCI) SDIO Resume Command
AT91C_MCI_ATACS           EQU (0x1 << 26) ;- (MCI) ATA with command completion signal
AT91C_MCI_ATACS_NORMAL    EQU (0x0 << 26) ;- (MCI) normal operation mode
AT91C_MCI_ATACS_COMPLETION EQU (0x1 << 26) ;- (MCI) completion signal is expected within MCI_CSTOR
// - -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- 
AT91C_MCI_BCNT            EQU (0xFFFF <<  0) ;- (MCI) MMC/SDIO Block Count / SDIO Byte Count
// - -------- MCI_CSTOR : (MCI Offset: 0x1c) MCI Completion Signal Timeout Register -------- 
AT91C_MCI_CSTOCYC         EQU (0xF <<  0) ;- (MCI) Completion Signal Timeout Cycle Number
AT91C_MCI_CSTOMUL         EQU (0x7 <<  4) ;- (MCI) Completion Signal Timeout Multiplier
AT91C_MCI_CSTOMUL_1       EQU (0x0 <<  4) ;- (MCI) CSTOCYC x 1
AT91C_MCI_CSTOMUL_16      EQU (0x1 <<  4) ;- (MCI) CSTOCYC x  16
AT91C_MCI_CSTOMUL_128     EQU (0x2 <<  4) ;- (MCI) CSTOCYC x  128
AT91C_MCI_CSTOMUL_256     EQU (0x3 <<  4) ;- (MCI) CSTOCYC x  256
AT91C_MCI_CSTOMUL_1024    EQU (0x4 <<  4) ;- (MCI) CSTOCYC x  1024
AT91C_MCI_CSTOMUL_4096    EQU (0x5 <<  4) ;- (MCI) CSTOCYC x  4096
AT91C_MCI_CSTOMUL_65536   EQU (0x6 <<  4) ;- (MCI) CSTOCYC x  65536
AT91C_MCI_CSTOMUL_1048576 EQU (0x7 <<  4) ;- (MCI) CSTOCYC x  1048576
// - -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- 
AT91C_MCI_CMDRDY          EQU (0x1 <<  0) ;- (MCI) Command Ready flag
AT91C_MCI_RXRDY           EQU (0x1 <<  1) ;- (MCI) RX Ready flag
AT91C_MCI_TXRDY           EQU (0x1 <<  2) ;- (MCI) TX Ready flag
AT91C_MCI_BLKE            EQU (0x1 <<  3) ;- (MCI) Data Block Transfer Ended flag
AT91C_MCI_DTIP            EQU (0x1 <<  4) ;- (MCI) Data Transfer in Progress flag
AT91C_MCI_NOTBUSY         EQU (0x1 <<  5) ;- (MCI) Data Line Not Busy flag
AT91C_MCI_ENDRX           EQU (0x1 <<  6) ;- (MCI) End of RX Buffer flag
AT91C_MCI_ENDTX           EQU (0x1 <<  7) ;- (MCI) End of TX Buffer flag
AT91C_MCI_SDIOIRQA        EQU (0x1 <<  8) ;- (MCI) SDIO Interrupt for Slot A
AT91C_MCI_SDIOIRQB        EQU (0x1 <<  9) ;- (MCI) SDIO Interrupt for Slot B
AT91C_MCI_SDIOIRQC        EQU (0x1 << 10) ;- (MCI) SDIO Interrupt for Slot C
AT91C_MCI_SDIOIRQD        EQU (0x1 << 11) ;- (MCI) SDIO Interrupt for Slot D
AT91C_MCI_SDIOWAIT        EQU (0x1 << 12) ;- (MCI) SDIO Read Wait operation flag
AT91C_MCI_CSRCV           EQU (0x1 << 13) ;- (MCI) CE-ATA Completion Signal flag
AT91C_MCI_RXBUFF          EQU (0x1 << 14) ;- (MCI) RX Buffer Full flag
AT91C_MCI_TXBUFE          EQU (0x1 << 15) ;- (MCI) TX Buffer Empty flag
AT91C_MCI_RINDE           EQU (0x1 << 16) ;- (MCI) Response Index Error flag
AT91C_MCI_RDIRE           EQU (0x1 << 17) ;- (MCI) Response Direction Error flag
AT91C_MCI_RCRCE           EQU (0x1 << 18) ;- (MCI) Response CRC Error flag
AT91C_MCI_RENDE           EQU (0x1 << 19) ;- (MCI) Response End Bit Error flag
AT91C_MCI_RTOE            EQU (0x1 << 20) ;- (MCI) Response Time-out Error flag
AT91C_MCI_DCRCE           EQU (0x1 << 21) ;- (MCI) data CRC Error flag
AT91C_MCI_DTOE            EQU (0x1 << 22) ;- (MCI) Data timeout Error flag
AT91C_MCI_CSTOE           EQU (0x1 << 23) ;- (MCI) Completion Signal timeout Error flag
AT91C_MCI_BLKOVRE         EQU (0x1 << 24) ;- (MCI) DMA Block Overrun Error flag
AT91C_MCI_DMADONE         EQU (0x1 << 25) ;- (MCI) DMA Transfer Done flag
AT91C_MCI_FIFOEMPTY       EQU (0x1 << 26) ;- (MCI) FIFO Empty flag
AT91C_MCI_XFRDONE         EQU (0x1 << 27) ;- (MCI) Transfer Done flag
AT91C_MCI_OVRE            EQU (0x1 << 30) ;- (MCI) Overrun flag
AT91C_MCI_UNRE            EQU (0x1 << 31) ;- (MCI) Underrun flag
// - -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- 
// - -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- 
// - -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- 
// - -------- MCI_DMA : (MCI Offset: 0x50) MCI DMA Configuration Register -------- 
AT91C_MCI_OFFSET          EQU (0x3 <<  0) ;- (MCI) DMA Write Buffer Offset
AT91C_MCI_CHKSIZE         EQU (0x7 <<  4) ;- (MCI) DMA Channel Read/Write Chunk Size
AT91C_MCI_CHKSIZE_1       EQU (0x0 <<  4) ;- (MCI) Number of data transferred is 1
AT91C_MCI_CHKSIZE_4       EQU (0x1 <<  4) ;- (MCI) Number of data transferred is 4
AT91C_MCI_CHKSIZE_8       EQU (0x2 <<  4) ;- (MCI) Number of data transferred is 8
AT91C_MCI_CHKSIZE_16      EQU (0x3 <<  4) ;- (MCI) Number of data transferred is 16
AT91C_MCI_CHKSIZE_32      EQU (0x4 <<  4) ;- (MCI) Number of data transferred is 32
AT91C_MCI_DMAEN           EQU (0x1 <<  8) ;- (MCI) DMA Hardware Handshaking Enable
AT91C_MCI_DMAEN_DISABLE   EQU (0x0 <<  8) ;- (MCI) DMA interface is disabled
AT91C_MCI_DMAEN_ENABLE    EQU (0x1 <<  8) ;- (MCI) DMA interface is enabled
// - -------- MCI_CFG : (MCI Offset: 0x54) MCI Configuration Register -------- 
AT91C_MCI_FIFOMODE        EQU (0x1 <<  0) ;- (MCI) MCI Internal FIFO Control Mode
AT91C_MCI_FIFOMODE_AMOUNTDATA EQU (0x0) ;- (MCI) A write transfer starts when a sufficient amount of datas is written into the FIFO
AT91C_MCI_FIFOMODE_ONEDATA EQU (0x1) ;- (MCI) A write transfer starts as soon as one data is written into the FIFO
AT91C_MCI_FERRCTRL        EQU (0x1 <<  4) ;- (MCI) Flow Error Flag Reset Control Mode
AT91C_MCI_FERRCTRL_RWCMD  EQU (0x0 <<  4) ;- (MCI) When an underflow/overflow condition flag is set, a new Write/Read command is needed to reset the flag
AT91C_MCI_FERRCTRL_READSR EQU (0x1 <<  4) ;- (MCI) When an underflow/overflow condition flag is set, a read status resets the flag
AT91C_MCI_HSMODE          EQU (0x1 <<  8) ;- (MCI) High Speed Mode
AT91C_MCI_HSMODE_DISABLE  EQU (0x0 <<  8) ;- (MCI) Default Bus Timing Mode
AT91C_MCI_HSMODE_ENABLE   EQU (0x1 <<  8) ;- (MCI) High Speed Mode
AT91C_MCI_LSYNC           EQU (0x1 << 12) ;- (MCI) Synchronize on last block
AT91C_MCI_LSYNC_CURRENT   EQU (0x0 << 12) ;- (MCI) Pending command sent at end of current data block
AT91C_MCI_LSYNC_INFINITE  EQU (0x1 << 12) ;- (MCI) Pending command sent at end of block transfer when transfer length is not infinite
// - -------- MCI_WPCR : (MCI Offset: 0xe4) Write Protection Control Register -------- 
AT91C_MCI_WP_EN           EQU (0x1 <<  0) ;- (MCI) Write Protection Enable
AT91C_MCI_WP_EN_DISABLE   EQU (0x0) ;- (MCI) Write Operation is disabled (if WP_KEY corresponds)
AT91C_MCI_WP_EN_ENABLE    EQU (0x1) ;- (MCI) Write Operation is enabled (if WP_KEY corresponds)
AT91C_MCI_WP_KEY          EQU (0xFFFFFF <<  8) ;- (MCI) Write Protection Key
// - -------- MCI_WPSR : (MCI Offset: 0xe8) Write Protection Status Register -------- 
AT91C_MCI_WP_VS           EQU (0xF <<  0) ;- (MCI) Write Protection Violation Status
AT91C_MCI_WP_VS_NO_VIOLATION EQU (0x0) ;- (MCI) No Write Protection Violation detected since last read
AT91C_MCI_WP_VS_ON_WRITE  EQU (0x1) ;- (MCI) Write Protection Violation detected since last read
AT91C_MCI_WP_VS_ON_RESET  EQU (0x2) ;- (MCI) Software Reset Violation detected since last read
AT91C_MCI_WP_VS_ON_BOTH   EQU (0x3) ;- (MCI) Write Protection and Software Reset Violation detected since last read
AT91C_MCI_WP_VSRC         EQU (0xF <<  8) ;- (MCI) Write Protection Violation Source
AT91C_MCI_WP_VSRC_NO_VIOLATION EQU (0x0 <<  8) ;- (MCI) No Write Protection Violation detected since last read
AT91C_MCI_WP_VSRC_MCI_MR  EQU (0x1 <<  8) ;- (MCI) Write Protection Violation detected on MCI_MR since last read
AT91C_MCI_WP_VSRC_MCI_DTOR EQU (0x2 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DTOR since last read
AT91C_MCI_WP_VSRC_MCI_SDCR EQU (0x3 <<  8) ;- (MCI) Write Protection Violation detected on MCI_SDCR since last read
AT91C_MCI_WP_VSRC_MCI_CSTOR EQU (0x4 <<  8) ;- (MCI) Write Protection Violation detected on MCI_CSTOR since last read
AT91C_MCI_WP_VSRC_MCI_DMA EQU (0x5 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DMA since last read
AT91C_MCI_WP_VSRC_MCI_CFG EQU (0x6 <<  8) ;- (MCI) Write Protection Violation detected on MCI_CFG since last read
AT91C_MCI_WP_VSRC_MCI_DEL EQU (0x7 <<  8) ;- (MCI) Write Protection Violation detected on MCI_DEL since last read
// - -------- MCI_VER : (MCI Offset: 0xfc)  VERSION  Register -------- 
AT91C_MCI_VER             EQU (0xF <<  0) ;- (MCI)  VERSION  Register

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Two-wire Interface
// - *****************************************************************************
// - -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- 
AT91C_TWI_START           EQU (0x1 <<  0) ;- (TWI) Send a START Condition
AT91C_TWI_STOP            EQU (0x1 <<  1) ;- (TWI) Send a STOP Condition
AT91C_TWI_MSEN            EQU (0x1 <<  2) ;- (TWI) TWI Master Transfer Enabled
AT91C_TWI_MSDIS           EQU (0x1 <<  3) ;- (TWI) TWI Master Transfer Disabled
AT91C_TWI_SVEN            EQU (0x1 <<  4) ;- (TWI) TWI Slave mode Enabled
AT91C_TWI_SVDIS           EQU (0x1 <<  5) ;- (TWI) TWI Slave mode Disabled
AT91C_TWI_SWRST           EQU (0x1 <<  7) ;- (TWI) Software Reset
// - -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- 
AT91C_TWI_IADRSZ          EQU (0x3 <<  8) ;- (TWI) Internal Device Address Size
AT91C_TWI_IADRSZ_NO       EQU (0x0 <<  8) ;- (TWI) No internal device address
AT91C_TWI_IADRSZ_1_BYTE   EQU (0x1 <<  8) ;- (TWI) One-byte internal device address
AT91C_TWI_IADRSZ_2_BYTE   EQU (0x2 <<  8) ;- (TWI) Two-byte internal device address
AT91C_TWI_IADRSZ_3_BYTE   EQU (0x3 <<  8) ;- (TWI) Three-byte internal device address
AT91C_TWI_MREAD           EQU (0x1 << 12) ;- (TWI) Master Read Direction
AT91C_TWI_DADR            EQU (0x7F << 16) ;- (TWI) Device Address
// - -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- 
AT91C_TWI_SADR            EQU (0x7F << 16) ;- (TWI) Slave Address
// - -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- 
AT91C_TWI_CLDIV           EQU (0xFF <<  0) ;- (TWI) Clock Low Divider
AT91C_TWI_CHDIV           EQU (0xFF <<  8) ;- (TWI) Clock High Divider
AT91C_TWI_CKDIV           EQU (0x7 << 16) ;- (TWI) Clock Divider
// - -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- 
AT91C_TWI_TXCOMP_SLAVE    EQU (0x1 <<  0) ;- (TWI) Transmission Completed
AT91C_TWI_TXCOMP_MASTER   EQU (0x1 <<  0) ;- (TWI) Transmission Completed
AT91C_TWI_RXRDY           EQU (0x1 <<  1) ;- (TWI) Receive holding register ReaDY
AT91C_TWI_TXRDY_MASTER    EQU (0x1 <<  2) ;- (TWI) Transmit holding register ReaDY
AT91C_TWI_TXRDY_SLAVE     EQU (0x1 <<  2) ;- (TWI) Transmit holding register ReaDY
AT91C_TWI_SVREAD          EQU (0x1 <<  3) ;- (TWI) Slave READ (used only in Slave mode)
AT91C_TWI_SVACC           EQU (0x1 <<  4) ;- (TWI) Slave ACCess (used only in Slave mode)
AT91C_TWI_GACC            EQU (0x1 <<  5) ;- (TWI) General Call ACcess (used only in Slave mode)
AT91C_TWI_OVRE            EQU (0x1 <<  6) ;- (TWI) Overrun Error (used only in Master and Multi-master mode)
AT91C_TWI_NACK_SLAVE      EQU (0x1 <<  8) ;- (TWI) Not Acknowledged
AT91C_TWI_NACK_MASTER     EQU (0x1 <<  8) ;- (TWI) Not Acknowledged
AT91C_TWI_ARBLST_MULTI_MASTER EQU (0x1 <<  9) ;- (TWI) Arbitration Lost (used only in Multimaster mode)
AT91C_TWI_SCLWS           EQU (0x1 << 10) ;- (TWI) Clock Wait State (used only in Slave mode)
AT91C_TWI_EOSACC          EQU (0x1 << 11) ;- (TWI) End Of Slave ACCess (used only in Slave mode)
AT91C_TWI_ENDRX           EQU (0x1 << 12) ;- (TWI) End of Receiver Transfer
AT91C_TWI_ENDTX           EQU (0x1 << 13) ;- (TWI) End of Receiver Transfer
AT91C_TWI_RXBUFF          EQU (0x1 << 14) ;- (TWI) RXBUFF Interrupt
AT91C_TWI_TXBUFE          EQU (0x1 << 15) ;- (TWI) TXBUFE Interrupt
// - -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- 
// - -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- 
// - -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Usart
// - *****************************************************************************
// - -------- US_CR : (USART Offset: 0x0)  Control Register -------- 
AT91C_US_RSTRX            EQU (0x1 <<  2) ;- (USART) Reset Receiver
AT91C_US_RSTTX            EQU (0x1 <<  3) ;- (USART) Reset Transmitter
AT91C_US_RXEN             EQU (0x1 <<  4) ;- (USART) Receiver Enable
AT91C_US_RXDIS            EQU (0x1 <<  5) ;- (USART) Receiver Disable
AT91C_US_TXEN             EQU (0x1 <<  6) ;- (USART) Transmitter Enable
AT91C_US_TXDIS            EQU (0x1 <<  7) ;- (USART) Transmitter Disable
AT91C_US_RSTSTA           EQU (0x1 <<  8) ;- (USART) Reset Status Bits
AT91C_US_STTBRK           EQU (0x1 <<  9) ;- (USART) Start Break
AT91C_US_STPBRK           EQU (0x1 << 10) ;- (USART) Stop Break
AT91C_US_STTTO            EQU (0x1 << 11) ;- (USART) Start Time-out
AT91C_US_SENDA            EQU (0x1 << 12) ;- (USART) Send Address
AT91C_US_RSTIT            EQU (0x1 << 13) ;- (USART) Reset Iterations
AT91C_US_RSTNACK          EQU (0x1 << 14) ;- (USART) Reset Non Acknowledge
AT91C_US_RETTO            EQU (0x1 << 15) ;- (USART) Rearm Time-out
AT91C_US_DTREN            EQU (0x1 << 16) ;- (USART) Data Terminal ready Enable
AT91C_US_DTRDIS           EQU (0x1 << 17) ;- (USART) Data Terminal ready Disable
AT91C_US_RTSEN            EQU (0x1 << 18) ;- (USART) Request to Send enable
AT91C_US_RTSDIS           EQU (0x1 << 19) ;- (USART) Request to Send Disable
// - -------- US_MR : (USART Offset: 0x4)  Mode Register -------- 
AT91C_US_USMODE           EQU (0xF <<  0) ;- (USART) Usart mode
AT91C_US_USMODE_NORMAL    EQU (0x0) ;- (USART) Normal
AT91C_US_USMODE_RS485     EQU (0x1) ;- (USART) RS485
AT91C_US_USMODE_HWHSH     EQU (0x2) ;- (USART) Hardware Handshaking
AT91C_US_USMODE_MODEM     EQU (0x3) ;- (USART) Modem
AT91C_US_USMODE_ISO7816_0 EQU (0x4) ;- (USART) ISO7816 protocol: T = 0
AT91C_US_USMODE_ISO7816_1 EQU (0x6) ;- (USART) ISO7816 protocol: T = 1
AT91C_US_USMODE_IRDA      EQU (0x8) ;- (USART) IrDA
AT91C_US_CLKS             EQU (0x3 <<  4) ;- (USART) Clock Selection (Baud Rate generator Input Clock
AT91C_US_CLKS_CLOCK       EQU (0x0 <<  4) ;- (USART) Clock
AT91C_US_CLKS_FDIV1       EQU (0x1 <<  4) ;- (USART) fdiv1
AT91C_US_CLKS_SLOW        EQU (0x2 <<  4) ;- (USART) slow_clock (ARM)
AT91C_US_CLKS_EXT         EQU (0x3 <<  4) ;- (USART) External (SCK)
AT91C_US_CHRL             EQU (0x3 <<  6) ;- (USART) Clock Selection (Baud Rate generator Input Clock
AT91C_US_CHRL_5_BITS      EQU (0x0 <<  6) ;- (USART) Character Length: 5 bits
AT91C_US_CHRL_6_BITS      EQU (0x1 <<  6) ;- (USART) Character Length: 6 bits
AT91C_US_CHRL_7_BITS      EQU (0x2 <<  6) ;- (USART) Character Length: 7 bits
AT91C_US_CHRL_8_BITS      EQU (0x3 <<  6) ;- (USART) Character Length: 8 bits
AT91C_US_SYNC             EQU (0x1 <<  8) ;- (USART) Synchronous Mode Select
AT91C_US_PAR              EQU (0x7 <<  9) ;- (USART) Parity type
AT91C_US_PAR_EVEN         EQU (0x0 <<  9) ;- (USART) Even Parity
AT91C_US_PAR_ODD          EQU (0x1 <<  9) ;- (USART) Odd Parity
AT91C_US_PAR_SPACE        EQU (0x2 <<  9) ;- (USART) Parity forced to 0 (Space)
AT91C_US_PAR_MARK         EQU (0x3 <<  9) ;- (USART) Parity forced to 1 (Mark)
AT91C_US_PAR_NONE         EQU (0x4 <<  9) ;- (USART) No Parity
AT91C_US_PAR_MULTI_DROP   EQU (0x6 <<  9) ;- (USART) Multi-drop mode
AT91C_US_NBSTOP           EQU (0x3 << 12) ;- (USART) Number of Stop bits
AT91C_US_NBSTOP_1_BIT     EQU (0x0 << 12) ;- (USART) 1 stop bit
AT91C_US_NBSTOP_15_BIT    EQU (0x1 << 12) ;- (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits
AT91C_US_NBSTOP_2_BIT     EQU (0x2 << 12) ;- (USART) 2 stop bits
AT91C_US_CHMODE           EQU (0x3 << 14) ;- (USART) Channel Mode
AT91C_US_CHMODE_NORMAL    EQU (0x0 << 14) ;- (USART) Normal Mode: The USART channel operates as an RX/TX USART.
AT91C_US_CHMODE_AUTO      EQU (0x1 << 14) ;- (USART) Automatic Echo: Receiver Data Input is connected to the TXD pin.
AT91C_US_CHMODE_LOCAL     EQU (0x2 << 14) ;- (USART) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
AT91C_US_CHMODE_REMOTE    EQU (0x3 << 14) ;- (USART) Remote Loopback: RXD pin is internally connected to TXD pin.
AT91C_US_MSBF             EQU (0x1 << 16) ;- (USART) Bit Order
AT91C_US_MODE9            EQU (0x1 << 17) ;- (USART) 9-bit Character length
AT91C_US_CKLO             EQU (0x1 << 18) ;- (USART) Clock Output Select
AT91C_US_OVER             EQU (0x1 << 19) ;- (USART) Over Sampling Mode
AT91C_US_INACK            EQU (0x1 << 20) ;- (USART) Inhibit Non Acknowledge
AT91C_US_DSNACK           EQU (0x1 << 21) ;- (USART) Disable Successive NACK
AT91C_US_VAR_SYNC         EQU (0x1 << 22) ;- (USART) Variable synchronization of command/data sync Start Frame Delimiter
AT91C_US_MAX_ITER         EQU (0x1 << 24) ;- (USART) Number of Repetitions
AT91C_US_FILTER           EQU (0x1 << 28) ;- (USART) Receive Line Filter
AT91C_US_MANMODE          EQU (0x1 << 29) ;- (USART) Manchester Encoder/Decoder Enable
AT91C_US_MODSYNC          EQU (0x1 << 30) ;- (USART) Manchester Synchronization mode
AT91C_US_ONEBIT           EQU (0x1 << 31) ;- (USART) Start Frame Delimiter selector
// - -------- US_IER : (USART Offset: 0x8)  Interrupt Enable Register -------- 
AT91C_US_RXRDY            EQU (0x1 <<  0) ;- (USART) RXRDY Interrupt
AT91C_US_TXRDY            EQU (0x1 <<  1) ;- (USART) TXRDY Interrupt
AT91C_US_RXBRK            EQU (0x1 <<  2) ;- (USART) Break Received/End of Break
AT91C_US_ENDRX            EQU (0x1 <<  3) ;- (USART) End of Receive Transfer Interrupt
AT91C_US_ENDTX            EQU (0x1 <<  4) ;- (USART) End of Transmit Interrupt
AT91C_US_OVRE             EQU (0x1 <<  5) ;- (USART) Overrun Interrupt
AT91C_US_FRAME            EQU (0x1 <<  6) ;- (USART) Framing Error Interrupt
AT91C_US_PARE             EQU (0x1 <<  7) ;- (USART) Parity Error Interrupt
AT91C_US_TIMEOUT          EQU (0x1 <<  8) ;- (USART) Receiver Time-out
AT91C_US_TXEMPTY          EQU (0x1 <<  9) ;- (USART) TXEMPTY Interrupt
AT91C_US_ITERATION        EQU (0x1 << 10) ;- (USART) Max number of Repetitions Reached
AT91C_US_TXBUFE           EQU (0x1 << 11) ;- (USART) TXBUFE Interrupt
AT91C_US_RXBUFF           EQU (0x1 << 12) ;- (USART) RXBUFF Interrupt
AT91C_US_NACK             EQU (0x1 << 13) ;- (USART) Non Acknowledge
AT91C_US_RIIC             EQU (0x1 << 16) ;- (USART) Ring INdicator Input Change Flag
AT91C_US_DSRIC            EQU (0x1 << 17) ;- (USART) Data Set Ready Input Change Flag
AT91C_US_DCDIC            EQU (0x1 << 18) ;- (USART) Data Carrier Flag
AT91C_US_CTSIC            EQU (0x1 << 19) ;- (USART) Clear To Send Input Change Flag
AT91C_US_MANE             EQU (0x1 << 20) ;- (USART) Manchester Error Interrupt
// - -------- US_IDR : (USART Offset: 0xc)  Interrupt Disable Register -------- 
// - -------- US_IMR : (USART Offset: 0x10)  Interrupt Mask Register -------- 
// - -------- US_CSR : (USART Offset: 0x14)  Channel Status Register -------- 
AT91C_US_RI               EQU (0x1 << 20) ;- (USART) Image of RI Input
AT91C_US_DSR              EQU (0x1 << 21) ;- (USART) Image of DSR Input
AT91C_US_DCD              EQU (0x1 << 22) ;- (USART) Image of DCD Input
AT91C_US_CTS              EQU (0x1 << 23) ;- (USART) Image of CTS Input
AT91C_US_MANERR           EQU (0x1 << 24) ;- (USART) Manchester Error
// - -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- 
AT91C_US_TX_PL            EQU (0xF <<  0) ;- (USART) Transmitter Preamble Length
AT91C_US_TX_PP            EQU (0x3 <<  8) ;- (USART) Transmitter Preamble Pattern
AT91C_US_TX_PP_ALL_ONE    EQU (0x0 <<  8) ;- (USART) ALL_ONE
AT91C_US_TX_PP_ALL_ZERO   EQU (0x1 <<  8) ;- (USART) ALL_ZERO
AT91C_US_TX_PP_ZERO_ONE   EQU (0x2 <<  8) ;- (USART) ZERO_ONE
AT91C_US_TX_PP_ONE_ZERO   EQU (0x3 <<  8) ;- (USART) ONE_ZERO
AT91C_US_TX_MPOL          EQU (0x1 << 12) ;- (USART) Transmitter Manchester Polarity
AT91C_US_RX_PL            EQU (0xF << 16) ;- (USART) Receiver Preamble Length
AT91C_US_RX_PP            EQU (0x3 << 24) ;- (USART) Receiver Preamble Pattern detected
AT91C_US_RX_PP_ALL_ONE    EQU (0x0 << 24) ;- (USART) ALL_ONE
AT91C_US_RX_PP_ALL_ZERO   EQU (0x1 << 24) ;- (USART) ALL_ZERO
AT91C_US_RX_PP_ZERO_ONE   EQU (0x2 << 24) ;- (USART) ZERO_ONE
AT91C_US_RX_PP_ONE_ZERO   EQU (0x3 << 24) ;- (USART) ONE_ZERO
AT91C_US_RX_MPOL          EQU (0x1 << 28) ;- (USART) Receiver Manchester Polarity
AT91C_US_DRIFT            EQU (0x1 << 30) ;- (USART) Drift compensation

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface
// - *****************************************************************************
// - -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- 
AT91C_SSC_RXEN            EQU (0x1 <<  0) ;- (SSC) Receive Enable
AT91C_SSC_RXDIS           EQU (0x1 <<  1) ;- (SSC) Receive Disable
AT91C_SSC_TXEN            EQU (0x1 <<  8) ;- (SSC) Transmit Enable
AT91C_SSC_TXDIS           EQU (0x1 <<  9) ;- (SSC) Transmit Disable
AT91C_SSC_SWRST           EQU (0x1 << 15) ;- (SSC) Software Reset
// - -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- 
AT91C_SSC_CKS             EQU (0x3 <<  0) ;- (SSC) Receive/Transmit Clock Selection
AT91C_SSC_CKS_DIV         EQU (0x0) ;- (SSC) Divided Clock
AT91C_SSC_CKS_TK          EQU (0x1) ;- (SSC) TK Clock signal
AT91C_SSC_CKS_RK          EQU (0x2) ;- (SSC) RK pin
AT91C_SSC_CKO             EQU (0x7 <<  2) ;- (SSC) Receive/Transmit Clock Output Mode Selection
AT91C_SSC_CKO_NONE        EQU (0x0 <<  2) ;- (SSC) Receive/Transmit Clock Output Mode: None RK pin: Input-only
AT91C_SSC_CKO_CONTINOUS   EQU (0x1 <<  2) ;- (SSC) Continuous Receive/Transmit Clock RK pin: Output
AT91C_SSC_CKO_DATA_TX     EQU (0x2 <<  2) ;- (SSC) Receive/Transmit Clock only during data transfers RK pin: Output
AT91C_SSC_CKI             EQU (0x1 <<  5) ;- (SSC) Receive/Transmit Clock Inversion
AT91C_SSC_CKG             EQU (0x3 <<  6) ;- (SSC) Receive/Transmit Clock Gating Selection
AT91C_SSC_CKG_NONE        EQU (0x0 <<  6) ;- (SSC) Receive/Transmit Clock Gating: None, continuous clock
AT91C_SSC_CKG_LOW         EQU (0x1 <<  6) ;- (SSC) Receive/Transmit Clock enabled only if RF Low
AT91C_SSC_CKG_HIGH        EQU (0x2 <<  6) ;- (SSC) Receive/Transmit Clock enabled only if RF High
AT91C_SSC_START           EQU (0xF <<  8) ;- (SSC) Receive/Transmit Start Selection
AT91C_SSC_START_CONTINOUS EQU (0x0 <<  8) ;- (SSC) Continuous, as soon as the receiver is enabled, and immediately after the end of transfer of the previous data.
AT91C_SSC_START_TX        EQU (0x1 <<  8) ;- (SSC) Transmit/Receive start
AT91C_SSC_START_LOW_RF    EQU (0x2 <<  8) ;- (SSC) Detection of a low level on RF input
AT91C_SSC_START_HIGH_RF   EQU (0x3 <<  8) ;- (SSC) Detection of a high level on RF input
AT91C_SSC_START_FALL_RF   EQU (0x4 <<  8) ;- (SSC) Detection of a falling edge on RF input
AT91C_SSC_START_RISE_RF   EQU (0x5 <<  8) ;- (SSC) Detection of a rising edge on RF input
AT91C_SSC_START_LEVEL_RF  EQU (0x6 <<  8) ;- (SSC) Detection of any level change on RF input
AT91C_SSC_START_EDGE_RF   EQU (0x7 <<  8) ;- (SSC) Detection of any edge on RF input
AT91C_SSC_START_0         EQU (0x8 <<  8) ;- (SSC) Compare 0
AT91C_SSC_STOP            EQU (0x1 << 12) ;- (SSC) Receive Stop Selection
AT91C_SSC_STTOUT          EQU (0x1 << 15) ;- (SSC) Receive/Transmit Start Output Selection
AT91C_SSC_STTDLY          EQU (0xFF << 16) ;- (SSC) Receive/Transmit Start Delay
AT91C_SSC_PERIOD          EQU (0xFF << 24) ;- (SSC) Receive/Transmit Period Divider Selection
// - -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- 
AT91C_SSC_DATLEN          EQU (0x1F <<  0) ;- (SSC) Data Length
AT91C_SSC_LOOP            EQU (0x1 <<  5) ;- (SSC) Loop Mode
AT91C_SSC_MSBF            EQU (0x1 <<  7) ;- (SSC) Most Significant Bit First
AT91C_SSC_DATNB           EQU (0xF <<  8) ;- (SSC) Data Number per Frame
AT91C_SSC_FSLEN           EQU (0xF << 16) ;- (SSC) Receive/Transmit Frame Sync length
AT91C_SSC_FSOS            EQU (0x7 << 20) ;- (SSC) Receive/Transmit Frame Sync Output Selection
AT91C_SSC_FSOS_NONE       EQU (0x0 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: None RK pin Input-only
AT91C_SSC_FSOS_NEGATIVE   EQU (0x1 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Negative Pulse
AT91C_SSC_FSOS_POSITIVE   EQU (0x2 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Positive Pulse
AT91C_SSC_FSOS_LOW        EQU (0x3 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Driver Low during data transfer
AT91C_SSC_FSOS_HIGH       EQU (0x4 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Driver High during data transfer
AT91C_SSC_FSOS_TOGGLE     EQU (0x5 << 20) ;- (SSC) Selected Receive/Transmit Frame Sync Signal: Toggling at each start of data transfer
AT91C_SSC_FSEDGE          EQU (0x1 << 24) ;- (SSC) Frame Sync Edge Detection
// - -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- 
// - -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- 
AT91C_SSC_DATDEF          EQU (0x1 <<  5) ;- (SSC) Data Default Value
AT91C_SSC_FSDEN           EQU (0x1 << 23) ;- (SSC) Frame Sync Data Enable
// - -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- 
AT91C_SSC_TXRDY           EQU (0x1 <<  0) ;- (SSC) Transmit Ready
AT91C_SSC_TXEMPTY         EQU (0x1 <<  1) ;- (SSC) Transmit Empty
AT91C_SSC_ENDTX           EQU (0x1 <<  2) ;- (SSC) End Of Transmission
AT91C_SSC_TXBUFE          EQU (0x1 <<  3) ;- (SSC) Transmit Buffer Empty
AT91C_SSC_RXRDY           EQU (0x1 <<  4) ;- (SSC) Receive Ready
AT91C_SSC_OVRUN           EQU (0x1 <<  5) ;- (SSC) Receive Overrun
AT91C_SSC_ENDRX           EQU (0x1 <<  6) ;- (SSC) End of Reception
AT91C_SSC_RXBUFF          EQU (0x1 <<  7) ;- (SSC) Receive Buffer Full
AT91C_SSC_CP0             EQU (0x1 <<  8) ;- (SSC) Compare 0
AT91C_SSC_CP1             EQU (0x1 <<  9) ;- (SSC) Compare 1
AT91C_SSC_TXSYN           EQU (0x1 << 10) ;- (SSC) Transmit Sync
AT91C_SSC_RXSYN           EQU (0x1 << 11) ;- (SSC) Receive Sync
AT91C_SSC_TXENA           EQU (0x1 << 16) ;- (SSC) Transmit Enable
AT91C_SSC_RXENA           EQU (0x1 << 17) ;- (SSC) Receive Enable
// - -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- 
// - -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- 
// - -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR PWMC Channel Interface
// - *****************************************************************************
// - -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- 
AT91C_PWMC_CPRE           EQU (0xF <<  0) ;- (PWMC_CH) Channel Pre-scaler : PWMC_CLKx
AT91C_PWMC_CPRE_MCK       EQU (0x0) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_2 EQU (0x1) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_4 EQU (0x2) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_8 EQU (0x3) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_16 EQU (0x4) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_32 EQU (0x5) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_64 EQU (0x6) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_128 EQU (0x7) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_256 EQU (0x8) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_512 EQU (0x9) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCK_DIV_1024 EQU (0xA) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKA      EQU (0xB) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKB      EQU (0xC) ;- (PWMC_CH) 
AT91C_PWMC_CALG           EQU (0x1 <<  8) ;- (PWMC_CH) Channel Alignment
AT91C_PWMC_CPOL           EQU (0x1 <<  9) ;- (PWMC_CH) Channel Polarity
AT91C_PWMC_CES            EQU (0x1 << 10) ;- (PWMC_CH) Counter Event Selection
AT91C_PWMC_DTE            EQU (0x1 << 16) ;- (PWMC_CH) Dead Time Genrator Enable
AT91C_PWMC_DTHI           EQU (0x1 << 17) ;- (PWMC_CH) Dead Time PWMHx Output Inverted
AT91C_PWMC_DTLI           EQU (0x1 << 18) ;- (PWMC_CH) Dead Time PWMLx Output Inverted
// - -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- 
AT91C_PWMC_CDTY           EQU (0xFFFFFF <<  0) ;- (PWMC_CH) Channel Duty Cycle
// - -------- PWMC_CDTYUPDR : (PWMC_CH Offset: 0x8) PWMC Channel Duty Cycle Update Register -------- 
AT91C_PWMC_CDTYUPD        EQU (0xFFFFFF <<  0) ;- (PWMC_CH) Channel Duty Cycle Update
// - -------- PWMC_CPRDR : (PWMC_CH Offset: 0xc) PWMC Channel Period Register -------- 
AT91C_PWMC_CPRD           EQU (0xFFFFFF <<  0) ;- (PWMC_CH) Channel Period
// - -------- PWMC_CPRDUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Period Update Register -------- 
AT91C_PWMC_CPRDUPD        EQU (0xFFFFFF <<  0) ;- (PWMC_CH) Channel Period Update
// - -------- PWMC_CCNTR : (PWMC_CH Offset: 0x14) PWMC Channel Counter Register -------- 
AT91C_PWMC_CCNT           EQU (0xFFFFFF <<  0) ;- (PWMC_CH) Channel Counter
// - -------- PWMC_DTR : (PWMC_CH Offset: 0x18) Channel Dead Time Value Register -------- 
AT91C_PWMC_DTL            EQU (0xFFFF <<  0) ;- (PWMC_CH) Channel Dead Time for PWML
AT91C_PWMC_DTH            EQU (0xFFFF << 16) ;- (PWMC_CH) Channel Dead Time for PWMH
// - -------- PWMC_DTUPDR : (PWMC_CH Offset: 0x1c) Channel Dead Time Value Register -------- 
AT91C_PWMC_DTLUPD         EQU (0xFFFF <<  0) ;- (PWMC_CH) Channel Dead Time Update for PWML.
AT91C_PWMC_DTHUPD         EQU (0xFFFF << 16) ;- (PWMC_CH) Channel Dead Time Update for PWMH.

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface
// - *****************************************************************************
// - -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- 
AT91C_PWMC_DIVA           EQU (0xFF <<  0) ;- (PWMC) CLKA divide factor.
AT91C_PWMC_PREA           EQU (0xF <<  8) ;- (PWMC) Divider Input Clock Prescaler A
AT91C_PWMC_PREA_MCK       EQU (0x0 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_2 EQU (0x1 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_4 EQU (0x2 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_8 EQU (0x3 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_16 EQU (0x4 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_32 EQU (0x5 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_64 EQU (0x6 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_128 EQU (0x7 <<  8) ;- (PWMC) 
AT91C_PWMC_PREA_MCK_DIV_256 EQU (0x8 <<  8) ;- (PWMC) 
AT91C_PWMC_DIVB           EQU (0xFF << 16) ;- (PWMC) CLKB divide factor.
AT91C_PWMC_PREB           EQU (0xF << 24) ;- (PWMC) Divider Input Clock Prescaler B
AT91C_PWMC_PREB_MCK       EQU (0x0 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_2 EQU (0x1 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_4 EQU (0x2 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_8 EQU (0x3 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_16 EQU (0x4 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_32 EQU (0x5 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_64 EQU (0x6 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_128 EQU (0x7 << 24) ;- (PWMC) 
AT91C_PWMC_PREB_MCK_DIV_256 EQU (0x8 << 24) ;- (PWMC) 
AT91C_PWMC_CLKSEL         EQU (0x1 << 31) ;- (PWMC) CCK Source Clock Selection
// - -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- 
AT91C_PWMC_CHID0          EQU (0x1 <<  0) ;- (PWMC) Channel ID 0
AT91C_PWMC_CHID1          EQU (0x1 <<  1) ;- (PWMC) Channel ID 1
AT91C_PWMC_CHID2          EQU (0x1 <<  2) ;- (PWMC) Channel ID 2
AT91C_PWMC_CHID3          EQU (0x1 <<  3) ;- (PWMC) Channel ID 3
AT91C_PWMC_CHID4          EQU (0x1 <<  4) ;- (PWMC) Channel ID 4
AT91C_PWMC_CHID5          EQU (0x1 <<  5) ;- (PWMC) Channel ID 5
AT91C_PWMC_CHID6          EQU (0x1 <<  6) ;- (PWMC) Channel ID 6
AT91C_PWMC_CHID7          EQU (0x1 <<  7) ;- (PWMC) Channel ID 7
AT91C_PWMC_CHID8          EQU (0x1 <<  8) ;- (PWMC) Channel ID 8
AT91C_PWMC_CHID9          EQU (0x1 <<  9) ;- (PWMC) Channel ID 9
AT91C_PWMC_CHID10         EQU (0x1 << 10) ;- (PWMC) Channel ID 10
AT91C_PWMC_CHID11         EQU (0x1 << 11) ;- (PWMC) Channel ID 11
AT91C_PWMC_CHID12         EQU (0x1 << 12) ;- (PWMC) Channel ID 12
AT91C_PWMC_CHID13         EQU (0x1 << 13) ;- (PWMC) Channel ID 13
AT91C_PWMC_CHID14         EQU (0x1 << 14) ;- (PWMC) Channel ID 14
AT91C_PWMC_CHID15         EQU (0x1 << 15) ;- (PWMC) Channel ID 15
// - -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- 
// - -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- 
// - -------- PWMC_IER1 : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- 
AT91C_PWMC_FCHID0         EQU (0x1 << 16) ;- (PWMC) Fault Event Channel ID 0
AT91C_PWMC_FCHID1         EQU (0x1 << 17) ;- (PWMC) Fault Event Channel ID 1
AT91C_PWMC_FCHID2         EQU (0x1 << 18) ;- (PWMC) Fault Event Channel ID 2
AT91C_PWMC_FCHID3         EQU (0x1 << 19) ;- (PWMC) Fault Event Channel ID 3
AT91C_PWMC_FCHID4         EQU (0x1 << 20) ;- (PWMC) Fault Event Channel ID 4
AT91C_PWMC_FCHID5         EQU (0x1 << 21) ;- (PWMC) Fault Event Channel ID 5
AT91C_PWMC_FCHID6         EQU (0x1 << 22) ;- (PWMC) Fault Event Channel ID 6
AT91C_PWMC_FCHID7         EQU (0x1 << 23) ;- (PWMC) Fault Event Channel ID 7
AT91C_PWMC_FCHID8         EQU (0x1 << 24) ;- (PWMC) Fault Event Channel ID 8
AT91C_PWMC_FCHID9         EQU (0x1 << 25) ;- (PWMC) Fault Event Channel ID 9
AT91C_PWMC_FCHID10        EQU (0x1 << 26) ;- (PWMC) Fault Event Channel ID 10
AT91C_PWMC_FCHID11        EQU (0x1 << 27) ;- (PWMC) Fault Event Channel ID 11
AT91C_PWMC_FCHID12        EQU (0x1 << 28) ;- (PWMC) Fault Event Channel ID 12
AT91C_PWMC_FCHID13        EQU (0x1 << 29) ;- (PWMC) Fault Event Channel ID 13
AT91C_PWMC_FCHID14        EQU (0x1 << 30) ;- (PWMC) Fault Event Channel ID 14
AT91C_PWMC_FCHID15        EQU (0x1 << 31) ;- (PWMC) Fault Event Channel ID 15
// - -------- PWMC_IDR1 : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- 
// - -------- PWMC_IMR1 : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- 
// - -------- PWMC_ISR1 : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- 
// - -------- PWMC_SYNC : (PWMC Offset: 0x20) PWMC Synchronous Channels Register -------- 
AT91C_PWMC_SYNC0          EQU (0x1 <<  0) ;- (PWMC) Synchronous Channel ID 0
AT91C_PWMC_SYNC1          EQU (0x1 <<  1) ;- (PWMC) Synchronous Channel ID 1
AT91C_PWMC_SYNC2          EQU (0x1 <<  2) ;- (PWMC) Synchronous Channel ID 2
AT91C_PWMC_SYNC3          EQU (0x1 <<  3) ;- (PWMC) Synchronous Channel ID 3
AT91C_PWMC_SYNC4          EQU (0x1 <<  4) ;- (PWMC) Synchronous Channel ID 4
AT91C_PWMC_SYNC5          EQU (0x1 <<  5) ;- (PWMC) Synchronous Channel ID 5
AT91C_PWMC_SYNC6          EQU (0x1 <<  6) ;- (PWMC) Synchronous Channel ID 6
AT91C_PWMC_SYNC7          EQU (0x1 <<  7) ;- (PWMC) Synchronous Channel ID 7
AT91C_PWMC_SYNC8          EQU (0x1 <<  8) ;- (PWMC) Synchronous Channel ID 8
AT91C_PWMC_SYNC9          EQU (0x1 <<  9) ;- (PWMC) Synchronous Channel ID 9
AT91C_PWMC_SYNC10         EQU (0x1 << 10) ;- (PWMC) Synchronous Channel ID 10
AT91C_PWMC_SYNC11         EQU (0x1 << 11) ;- (PWMC) Synchronous Channel ID 11
AT91C_PWMC_SYNC12         EQU (0x1 << 12) ;- (PWMC) Synchronous Channel ID 12
AT91C_PWMC_SYNC13         EQU (0x1 << 13) ;- (PWMC) Synchronous Channel ID 13
AT91C_PWMC_SYNC14         EQU (0x1 << 14) ;- (PWMC) Synchronous Channel ID 14
AT91C_PWMC_SYNC15         EQU (0x1 << 15) ;- (PWMC) Synchronous Channel ID 15
AT91C_PWMC_UPDM           EQU (0x3 << 16) ;- (PWMC) Synchronous Channels Update mode
AT91C_PWMC_UPDM_MODE0     EQU (0x0 << 16) ;- (PWMC) Manual write of data and manual trigger of the update
AT91C_PWMC_UPDM_MODE1     EQU (0x1 << 16) ;- (PWMC) Manual write of data and automatic trigger of the update
AT91C_PWMC_UPDM_MODE2     EQU (0x2 << 16) ;- (PWMC) Automatic write of data and automatic trigger of the update
// - -------- PWMC_UPCR : (PWMC Offset: 0x28) PWMC Update Control Register -------- 
AT91C_PWMC_UPDULOCK       EQU (0x1 <<  0) ;- (PWMC) Synchronized Channels Duty Cycle Update Unlock
// - -------- PWMC_SCUP : (PWMC Offset: 0x2c) PWM Update Period Register -------- 
AT91C_PWMC_UPR            EQU (0xF <<  0) ;- (PWMC) PWM Update Period.
AT91C_PWMC_UPRCNT         EQU (0xF <<  4) ;- (PWMC) PWM Update Period Counter.
// - -------- PWMC_SCUPUPD : (PWMC Offset: 0x30) PWM Update Period Update Register -------- 
AT91C_PWMC_UPVUPDAL       EQU (0xF <<  0) ;- (PWMC) PWM Update Period Update.
// - -------- PWMC_IER2 : (PWMC Offset: 0x34) PWMC Interrupt Enable Register -------- 
AT91C_PWMC_WRDY           EQU (0x1 <<  0) ;- (PWMC) PDC Write Ready
AT91C_PWMC_ENDTX          EQU (0x1 <<  1) ;- (PWMC) PDC End of TX Buffer
AT91C_PWMC_TXBUFE         EQU (0x1 <<  2) ;- (PWMC) PDC End of TX Buffer
AT91C_PWMC_UNRE           EQU (0x1 <<  3) ;- (PWMC) PDC End of TX Buffer
// - -------- PWMC_IDR2 : (PWMC Offset: 0x38) PWMC Interrupt Disable Register -------- 
// - -------- PWMC_IMR2 : (PWMC Offset: 0x3c) PWMC Interrupt Mask Register -------- 
// - -------- PWMC_ISR2 : (PWMC Offset: 0x40) PWMC Interrupt Status Register -------- 
AT91C_PWMC_CMPM0          EQU (0x1 <<  8) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM1          EQU (0x1 <<  9) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM2          EQU (0x1 << 10) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM3          EQU (0x1 << 11) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM4          EQU (0x1 << 12) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM5          EQU (0x1 << 13) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM6          EQU (0x1 << 14) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPM7          EQU (0x1 << 15) ;- (PWMC) Comparison x Match
AT91C_PWMC_CMPU0          EQU (0x1 << 16) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU1          EQU (0x1 << 17) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU2          EQU (0x1 << 18) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU3          EQU (0x1 << 19) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU4          EQU (0x1 << 20) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU5          EQU (0x1 << 21) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU6          EQU (0x1 << 22) ;- (PWMC) Comparison x Update
AT91C_PWMC_CMPU7          EQU (0x1 << 23) ;- (PWMC) Comparison x Update
// - -------- PWMC_OOV : (PWMC Offset: 0x44) PWM Output Override Value Register -------- 
AT91C_PWMC_OOVH0          EQU (0x1 <<  0) ;- (PWMC) Output Override Value for PWMH output of the channel 0
AT91C_PWMC_OOVH1          EQU (0x1 <<  1) ;- (PWMC) Output Override Value for PWMH output of the channel 1
AT91C_PWMC_OOVH2          EQU (0x1 <<  2) ;- (PWMC) Output Override Value for PWMH output of the channel 2
AT91C_PWMC_OOVH3          EQU (0x1 <<  3) ;- (PWMC) Output Override Value for PWMH output of the channel 3
AT91C_PWMC_OOVH4          EQU (0x1 <<  4) ;- (PWMC) Output Override Value for PWMH output of the channel 4
AT91C_PWMC_OOVH5          EQU (0x1 <<  5) ;- (PWMC) Output Override Value for PWMH output of the channel 5
AT91C_PWMC_OOVH6          EQU (0x1 <<  6) ;- (PWMC) Output Override Value for PWMH output of the channel 6
AT91C_PWMC_OOVH7          EQU (0x1 <<  7) ;- (PWMC) Output Override Value for PWMH output of the channel 7
AT91C_PWMC_OOVH8          EQU (0x1 <<  8) ;- (PWMC) Output Override Value for PWMH output of the channel 8
AT91C_PWMC_OOVH9          EQU (0x1 <<  9) ;- (PWMC) Output Override Value for PWMH output of the channel 9
AT91C_PWMC_OOVH10         EQU (0x1 << 10) ;- (PWMC) Output Override Value for PWMH output of the channel 10
AT91C_PWMC_OOVH11         EQU (0x1 << 11) ;- (PWMC) Output Override Value for PWMH output of the channel 11
AT91C_PWMC_OOVH12         EQU (0x1 << 12) ;- (PWMC) Output Override Value for PWMH output of the channel 12
AT91C_PWMC_OOVH13         EQU (0x1 << 13) ;- (PWMC) Output Override Value for PWMH output of the channel 13
AT91C_PWMC_OOVH14         EQU (0x1 << 14) ;- (PWMC) Output Override Value for PWMH output of the channel 14
AT91C_PWMC_OOVH15         EQU (0x1 << 15) ;- (PWMC) Output Override Value for PWMH output of the channel 15
AT91C_PWMC_OOVL0          EQU (0x1 << 16) ;- (PWMC) Output Override Value for PWML output of the channel 0
AT91C_PWMC_OOVL1          EQU (0x1 << 17) ;- (PWMC) Output Override Value for PWML output of the channel 1
AT91C_PWMC_OOVL2          EQU (0x1 << 18) ;- (PWMC) Output Override Value for PWML output of the channel 2
AT91C_PWMC_OOVL3          EQU (0x1 << 19) ;- (PWMC) Output Override Value for PWML output of the channel 3
AT91C_PWMC_OOVL4          EQU (0x1 << 20) ;- (PWMC) Output Override Value for PWML output of the channel 4
AT91C_PWMC_OOVL5          EQU (0x1 << 21) ;- (PWMC) Output Override Value for PWML output of the channel 5
AT91C_PWMC_OOVL6          EQU (0x1 << 22) ;- (PWMC) Output Override Value for PWML output of the channel 6
AT91C_PWMC_OOVL7          EQU (0x1 << 23) ;- (PWMC) Output Override Value for PWML output of the channel 7
AT91C_PWMC_OOVL8          EQU (0x1 << 24) ;- (PWMC) Output Override Value for PWML output of the channel 8
AT91C_PWMC_OOVL9          EQU (0x1 << 25) ;- (PWMC) Output Override Value for PWML output of the channel 9
AT91C_PWMC_OOVL10         EQU (0x1 << 26) ;- (PWMC) Output Override Value for PWML output of the channel 10
AT91C_PWMC_OOVL11         EQU (0x1 << 27) ;- (PWMC) Output Override Value for PWML output of the channel 11
AT91C_PWMC_OOVL12         EQU (0x1 << 28) ;- (PWMC) Output Override Value for PWML output of the channel 12
AT91C_PWMC_OOVL13         EQU (0x1 << 29) ;- (PWMC) Output Override Value for PWML output of the channel 13
AT91C_PWMC_OOVL14         EQU (0x1 << 30) ;- (PWMC) Output Override Value for PWML output of the channel 14
AT91C_PWMC_OOVL15         EQU (0x1 << 31) ;- (PWMC) Output Override Value for PWML output of the channel 15
// - -------- PWMC_OS : (PWMC Offset: 0x48) PWM Output Selection Register -------- 
AT91C_PWMC_OSH0           EQU (0x1 <<  0) ;- (PWMC) Output Selection for PWMH output of the channel 0
AT91C_PWMC_OSH1           EQU (0x1 <<  1) ;- (PWMC) Output Selection for PWMH output of the channel 1
AT91C_PWMC_OSH2           EQU (0x1 <<  2) ;- (PWMC) Output Selection for PWMH output of the channel 2
AT91C_PWMC_OSH3           EQU (0x1 <<  3) ;- (PWMC) Output Selection for PWMH output of the channel 3
AT91C_PWMC_OSH4           EQU (0x1 <<  4) ;- (PWMC) Output Selection for PWMH output of the channel 4
AT91C_PWMC_OSH5           EQU (0x1 <<  5) ;- (PWMC) Output Selection for PWMH output of the channel 5
AT91C_PWMC_OSH6           EQU (0x1 <<  6) ;- (PWMC) Output Selection for PWMH output of the channel 6
AT91C_PWMC_OSH7           EQU (0x1 <<  7) ;- (PWMC) Output Selection for PWMH output of the channel 7
AT91C_PWMC_OSH8           EQU (0x1 <<  8) ;- (PWMC) Output Selection for PWMH output of the channel 8
AT91C_PWMC_OSH9           EQU (0x1 <<  9) ;- (PWMC) Output Selection for PWMH output of the channel 9
AT91C_PWMC_OSH10          EQU (0x1 << 10) ;- (PWMC) Output Selection for PWMH output of the channel 10
AT91C_PWMC_OSH11          EQU (0x1 << 11) ;- (PWMC) Output Selection for PWMH output of the channel 11
AT91C_PWMC_OSH12          EQU (0x1 << 12) ;- (PWMC) Output Selection for PWMH output of the channel 12
AT91C_PWMC_OSH13          EQU (0x1 << 13) ;- (PWMC) Output Selection for PWMH output of the channel 13
AT91C_PWMC_OSH14          EQU (0x1 << 14) ;- (PWMC) Output Selection for PWMH output of the channel 14
AT91C_PWMC_OSH15          EQU (0x1 << 15) ;- (PWMC) Output Selection for PWMH output of the channel 15
AT91C_PWMC_OSL0           EQU (0x1 << 16) ;- (PWMC) Output Selection for PWML output of the channel 0
AT91C_PWMC_OSL1           EQU (0x1 << 17) ;- (PWMC) Output Selection for PWML output of the channel 1
AT91C_PWMC_OSL2           EQU (0x1 << 18) ;- (PWMC) Output Selection for PWML output of the channel 2
AT91C_PWMC_OSL3           EQU (0x1 << 19) ;- (PWMC) Output Selection for PWML output of the channel 3
AT91C_PWMC_OSL4           EQU (0x1 << 20) ;- (PWMC) Output Selection for PWML output of the channel 4
AT91C_PWMC_OSL5           EQU (0x1 << 21) ;- (PWMC) Output Selection for PWML output of the channel 5
AT91C_PWMC_OSL6           EQU (0x1 << 22) ;- (PWMC) Output Selection for PWML output of the channel 6
AT91C_PWMC_OSL7           EQU (0x1 << 23) ;- (PWMC) Output Selection for PWML output of the channel 7
AT91C_PWMC_OSL8           EQU (0x1 << 24) ;- (PWMC) Output Selection for PWML output of the channel 8
AT91C_PWMC_OSL9           EQU (0x1 << 25) ;- (PWMC) Output Selection for PWML output of the channel 9
AT91C_PWMC_OSL10          EQU (0x1 << 26) ;- (PWMC) Output Selection for PWML output of the channel 10
AT91C_PWMC_OSL11          EQU (0x1 << 27) ;- (PWMC) Output Selection for PWML output of the channel 11
AT91C_PWMC_OSL12          EQU (0x1 << 28) ;- (PWMC) Output Selection for PWML output of the channel 12
AT91C_PWMC_OSL13          EQU (0x1 << 29) ;- (PWMC) Output Selection for PWML output of the channel 13
AT91C_PWMC_OSL14          EQU (0x1 << 30) ;- (PWMC) Output Selection for PWML output of the channel 14
AT91C_PWMC_OSL15          EQU (0x1 << 31) ;- (PWMC) Output Selection for PWML output of the channel 15
// - -------- PWMC_OSS : (PWMC Offset: 0x4c) PWM Output Selection Set Register -------- 
AT91C_PWMC_OSSH0          EQU (0x1 <<  0) ;- (PWMC) Output Selection Set for PWMH output of the channel 0
AT91C_PWMC_OSSH1          EQU (0x1 <<  1) ;- (PWMC) Output Selection Set for PWMH output of the channel 1
AT91C_PWMC_OSSH2          EQU (0x1 <<  2) ;- (PWMC) Output Selection Set for PWMH output of the channel 2
AT91C_PWMC_OSSH3          EQU (0x1 <<  3) ;- (PWMC) Output Selection Set for PWMH output of the channel 3
AT91C_PWMC_OSSH4          EQU (0x1 <<  4) ;- (PWMC) Output Selection Set for PWMH output of the channel 4
AT91C_PWMC_OSSH5          EQU (0x1 <<  5) ;- (PWMC) Output Selection Set for PWMH output of the channel 5
AT91C_PWMC_OSSH6          EQU (0x1 <<  6) ;- (PWMC) Output Selection Set for PWMH output of the channel 6
AT91C_PWMC_OSSH7          EQU (0x1 <<  7) ;- (PWMC) Output Selection Set for PWMH output of the channel 7
AT91C_PWMC_OSSH8          EQU (0x1 <<  8) ;- (PWMC) Output Selection Set for PWMH output of the channel 8
AT91C_PWMC_OSSH9          EQU (0x1 <<  9) ;- (PWMC) Output Selection Set for PWMH output of the channel 9
AT91C_PWMC_OSSH10         EQU (0x1 << 10) ;- (PWMC) Output Selection Set for PWMH output of the channel 10
AT91C_PWMC_OSSH11         EQU (0x1 << 11) ;- (PWMC) Output Selection Set for PWMH output of the channel 11
AT91C_PWMC_OSSH12         EQU (0x1 << 12) ;- (PWMC) Output Selection Set for PWMH output of the channel 12
AT91C_PWMC_OSSH13         EQU (0x1 << 13) ;- (PWMC) Output Selection Set for PWMH output of the channel 13
AT91C_PWMC_OSSH14         EQU (0x1 << 14) ;- (PWMC) Output Selection Set for PWMH output of the channel 14
AT91C_PWMC_OSSH15         EQU (0x1 << 15) ;- (PWMC) Output Selection Set for PWMH output of the channel 15
AT91C_PWMC_OSSL0          EQU (0x1 << 16) ;- (PWMC) Output Selection Set for PWML output of the channel 0
AT91C_PWMC_OSSL1          EQU (0x1 << 17) ;- (PWMC) Output Selection Set for PWML output of the channel 1
AT91C_PWMC_OSSL2          EQU (0x1 << 18) ;- (PWMC) Output Selection Set for PWML output of the channel 2
AT91C_PWMC_OSSL3          EQU (0x1 << 19) ;- (PWMC) Output Selection Set for PWML output of the channel 3
AT91C_PWMC_OSSL4          EQU (0x1 << 20) ;- (PWMC) Output Selection Set for PWML output of the channel 4
AT91C_PWMC_OSSL5          EQU (0x1 << 21) ;- (PWMC) Output Selection Set for PWML output of the channel 5
AT91C_PWMC_OSSL6          EQU (0x1 << 22) ;- (PWMC) Output Selection Set for PWML output of the channel 6
AT91C_PWMC_OSSL7          EQU (0x1 << 23) ;- (PWMC) Output Selection Set for PWML output of the channel 7
AT91C_PWMC_OSSL8          EQU (0x1 << 24) ;- (PWMC) Output Selection Set for PWML output of the channel 8
AT91C_PWMC_OSSL9          EQU (0x1 << 25) ;- (PWMC) Output Selection Set for PWML output of the channel 9
AT91C_PWMC_OSSL10         EQU (0x1 << 26) ;- (PWMC) Output Selection Set for PWML output of the channel 10
AT91C_PWMC_OSSL11         EQU (0x1 << 27) ;- (PWMC) Output Selection Set for PWML output of the channel 11
AT91C_PWMC_OSSL12         EQU (0x1 << 28) ;- (PWMC) Output Selection Set for PWML output of the channel 12
AT91C_PWMC_OSSL13         EQU (0x1 << 29) ;- (PWMC) Output Selection Set for PWML output of the channel 13
AT91C_PWMC_OSSL14         EQU (0x1 << 30) ;- (PWMC) Output Selection Set for PWML output of the channel 14
AT91C_PWMC_OSSL15         EQU (0x1 << 31) ;- (PWMC) Output Selection Set for PWML output of the channel 15
// - -------- PWMC_OSC : (PWMC Offset: 0x50) PWM Output Selection Clear Register -------- 
AT91C_PWMC_OSCH0          EQU (0x1 <<  0) ;- (PWMC) Output Selection Clear for PWMH output of the channel 0
AT91C_PWMC_OSCH1          EQU (0x1 <<  1) ;- (PWMC) Output Selection Clear for PWMH output of the channel 1
AT91C_PWMC_OSCH2          EQU (0x1 <<  2) ;- (PWMC) Output Selection Clear for PWMH output of the channel 2
AT91C_PWMC_OSCH3          EQU (0x1 <<  3) ;- (PWMC) Output Selection Clear for PWMH output of the channel 3
AT91C_PWMC_OSCH4          EQU (0x1 <<  4) ;- (PWMC) Output Selection Clear for PWMH output of the channel 4
AT91C_PWMC_OSCH5          EQU (0x1 <<  5) ;- (PWMC) Output Selection Clear for PWMH output of the channel 5
AT91C_PWMC_OSCH6          EQU (0x1 <<  6) ;- (PWMC) Output Selection Clear for PWMH output of the channel 6
AT91C_PWMC_OSCH7          EQU (0x1 <<  7) ;- (PWMC) Output Selection Clear for PWMH output of the channel 7
AT91C_PWMC_OSCH8          EQU (0x1 <<  8) ;- (PWMC) Output Selection Clear for PWMH output of the channel 8
AT91C_PWMC_OSCH9          EQU (0x1 <<  9) ;- (PWMC) Output Selection Clear for PWMH output of the channel 9
AT91C_PWMC_OSCH10         EQU (0x1 << 10) ;- (PWMC) Output Selection Clear for PWMH output of the channel 10
AT91C_PWMC_OSCH11         EQU (0x1 << 11) ;- (PWMC) Output Selection Clear for PWMH output of the channel 11
AT91C_PWMC_OSCH12         EQU (0x1 << 12) ;- (PWMC) Output Selection Clear for PWMH output of the channel 12
AT91C_PWMC_OSCH13         EQU (0x1 << 13) ;- (PWMC) Output Selection Clear for PWMH output of the channel 13
AT91C_PWMC_OSCH14         EQU (0x1 << 14) ;- (PWMC) Output Selection Clear for PWMH output of the channel 14
AT91C_PWMC_OSCH15         EQU (0x1 << 15) ;- (PWMC) Output Selection Clear for PWMH output of the channel 15
AT91C_PWMC_OSCL0          EQU (0x1 << 16) ;- (PWMC) Output Selection Clear for PWML output of the channel 0
AT91C_PWMC_OSCL1          EQU (0x1 << 17) ;- (PWMC) Output Selection Clear for PWML output of the channel 1
AT91C_PWMC_OSCL2          EQU (0x1 << 18) ;- (PWMC) Output Selection Clear for PWML output of the channel 2
AT91C_PWMC_OSCL3          EQU (0x1 << 19) ;- (PWMC) Output Selection Clear for PWML output of the channel 3
AT91C_PWMC_OSCL4          EQU (0x1 << 20) ;- (PWMC) Output Selection Clear for PWML output of the channel 4
AT91C_PWMC_OSCL5          EQU (0x1 << 21) ;- (PWMC) Output Selection Clear for PWML output of the channel 5
AT91C_PWMC_OSCL6          EQU (0x1 << 22) ;- (PWMC) Output Selection Clear for PWML output of the channel 6
AT91C_PWMC_OSCL7          EQU (0x1 << 23) ;- (PWMC) Output Selection Clear for PWML output of the channel 7
AT91C_PWMC_OSCL8          EQU (0x1 << 24) ;- (PWMC) Output Selection Clear for PWML output of the channel 8
AT91C_PWMC_OSCL9          EQU (0x1 << 25) ;- (PWMC) Output Selection Clear for PWML output of the channel 9
AT91C_PWMC_OSCL10         EQU (0x1 << 26) ;- (PWMC) Output Selection Clear for PWML output of the channel 10
AT91C_PWMC_OSCL11         EQU (0x1 << 27) ;- (PWMC) Output Selection Clear for PWML output of the channel 11
AT91C_PWMC_OSCL12         EQU (0x1 << 28) ;- (PWMC) Output Selection Clear for PWML output of the channel 12
AT91C_PWMC_OSCL13         EQU (0x1 << 29) ;- (PWMC) Output Selection Clear for PWML output of the channel 13
AT91C_PWMC_OSCL14         EQU (0x1 << 30) ;- (PWMC) Output Selection Clear for PWML output of the channel 14
AT91C_PWMC_OSCL15         EQU (0x1 << 31) ;- (PWMC) Output Selection Clear for PWML output of the channel 15
// - -------- PWMC_OSSUPD : (PWMC Offset: 0x54) Output Selection Set for PWMH / PWML output of the channel x -------- 
AT91C_PWMC_OSSUPDH0       EQU (0x1 <<  0) ;- (PWMC) Output Selection Set for PWMH output of the channel 0
AT91C_PWMC_OSSUPDH1       EQU (0x1 <<  1) ;- (PWMC) Output Selection Set for PWMH output of the channel 1
AT91C_PWMC_OSSUPDH2       EQU (0x1 <<  2) ;- (PWMC) Output Selection Set for PWMH output of the channel 2
AT91C_PWMC_OSSUPDH3       EQU (0x1 <<  3) ;- (PWMC) Output Selection Set for PWMH output of the channel 3
AT91C_PWMC_OSSUPDH4       EQU (0x1 <<  4) ;- (PWMC) Output Selection Set for PWMH output of the channel 4
AT91C_PWMC_OSSUPDH5       EQU (0x1 <<  5) ;- (PWMC) Output Selection Set for PWMH output of the channel 5
AT91C_PWMC_OSSUPDH6       EQU (0x1 <<  6) ;- (PWMC) Output Selection Set for PWMH output of the channel 6
AT91C_PWMC_OSSUPDH7       EQU (0x1 <<  7) ;- (PWMC) Output Selection Set for PWMH output of the channel 7
AT91C_PWMC_OSSUPDH8       EQU (0x1 <<  8) ;- (PWMC) Output Selection Set for PWMH output of the channel 8
AT91C_PWMC_OSSUPDH9       EQU (0x1 <<  9) ;- (PWMC) Output Selection Set for PWMH output of the channel 9
AT91C_PWMC_OSSUPDH10      EQU (0x1 << 10) ;- (PWMC) Output Selection Set for PWMH output of the channel 10
AT91C_PWMC_OSSUPDH11      EQU (0x1 << 11) ;- (PWMC) Output Selection Set for PWMH output of the channel 11
AT91C_PWMC_OSSUPDH12      EQU (0x1 << 12) ;- (PWMC) Output Selection Set for PWMH output of the channel 12
AT91C_PWMC_OSSUPDH13      EQU (0x1 << 13) ;- (PWMC) Output Selection Set for PWMH output of the channel 13
AT91C_PWMC_OSSUPDH14      EQU (0x1 << 14) ;- (PWMC) Output Selection Set for PWMH output of the channel 14
AT91C_PWMC_OSSUPDH15      EQU (0x1 << 15) ;- (PWMC) Output Selection Set for PWMH output of the channel 15
AT91C_PWMC_OSSUPDL0       EQU (0x1 << 16) ;- (PWMC) Output Selection Set for PWML output of the channel 0
AT91C_PWMC_OSSUPDL1       EQU (0x1 << 17) ;- (PWMC) Output Selection Set for PWML output of the channel 1
AT91C_PWMC_OSSUPDL2       EQU (0x1 << 18) ;- (PWMC) Output Selection Set for PWML output of the channel 2
AT91C_PWMC_OSSUPDL3       EQU (0x1 << 19) ;- (PWMC) Output Selection Set for PWML output of the channel 3
AT91C_PWMC_OSSUPDL4       EQU (0x1 << 20) ;- (PWMC) Output Selection Set for PWML output of the channel 4
AT91C_PWMC_OSSUPDL5       EQU (0x1 << 21) ;- (PWMC) Output Selection Set for PWML output of the channel 5
AT91C_PWMC_OSSUPDL6       EQU (0x1 << 22) ;- (PWMC) Output Selection Set for PWML output of the channel 6
AT91C_PWMC_OSSUPDL7       EQU (0x1 << 23) ;- (PWMC) Output Selection Set for PWML output of the channel 7
AT91C_PWMC_OSSUPDL8       EQU (0x1 << 24) ;- (PWMC) Output Selection Set for PWML output of the channel 8
AT91C_PWMC_OSSUPDL9       EQU (0x1 << 25) ;- (PWMC) Output Selection Set for PWML output of the channel 9
AT91C_PWMC_OSSUPDL10      EQU (0x1 << 26) ;- (PWMC) Output Selection Set for PWML output of the channel 10
AT91C_PWMC_OSSUPDL11      EQU (0x1 << 27) ;- (PWMC) Output Selection Set for PWML output of the channel 11
AT91C_PWMC_OSSUPDL12      EQU (0x1 << 28) ;- (PWMC) Output Selection Set for PWML output of the channel 12
AT91C_PWMC_OSSUPDL13      EQU (0x1 << 29) ;- (PWMC) Output Selection Set for PWML output of the channel 13
AT91C_PWMC_OSSUPDL14      EQU (0x1 << 30) ;- (PWMC) Output Selection Set for PWML output of the channel 14
AT91C_PWMC_OSSUPDL15      EQU (0x1 << 31) ;- (PWMC) Output Selection Set for PWML output of the channel 15
// - -------- PWMC_OSCUPD : (PWMC Offset: 0x58) Output Selection Clear for PWMH / PWML output of the channel x -------- 
AT91C_PWMC_OSCUPDH0       EQU (0x1 <<  0) ;- (PWMC) Output Selection Clear for PWMH output of the channel 0
AT91C_PWMC_OSCUPDH1       EQU (0x1 <<  1) ;- (PWMC) Output Selection Clear for PWMH output of the channel 1
AT91C_PWMC_OSCUPDH2       EQU (0x1 <<  2) ;- (PWMC) Output Selection Clear for PWMH output of the channel 2
AT91C_PWMC_OSCUPDH3       EQU (0x1 <<  3) ;- (PWMC) Output Selection Clear for PWMH output of the channel 3
AT91C_PWMC_OSCUPDH4       EQU (0x1 <<  4) ;- (PWMC) Output Selection Clear for PWMH output of the channel 4
AT91C_PWMC_OSCUPDH5       EQU (0x1 <<  5) ;- (PWMC) Output Selection Clear for PWMH output of the channel 5
AT91C_PWMC_OSCUPDH6       EQU (0x1 <<  6) ;- (PWMC) Output Selection Clear for PWMH output of the channel 6
AT91C_PWMC_OSCUPDH7       EQU (0x1 <<  7) ;- (PWMC) Output Selection Clear for PWMH output of the channel 7
AT91C_PWMC_OSCUPDH8       EQU (0x1 <<  8) ;- (PWMC) Output Selection Clear for PWMH output of the channel 8
AT91C_PWMC_OSCUPDH9       EQU (0x1 <<  9) ;- (PWMC) Output Selection Clear for PWMH output of the channel 9
AT91C_PWMC_OSCUPDH10      EQU (0x1 << 10) ;- (PWMC) Output Selection Clear for PWMH output of the channel 10
AT91C_PWMC_OSCUPDH11      EQU (0x1 << 11) ;- (PWMC) Output Selection Clear for PWMH output of the channel 11
AT91C_PWMC_OSCUPDH12      EQU (0x1 << 12) ;- (PWMC) Output Selection Clear for PWMH output of the channel 12
AT91C_PWMC_OSCUPDH13      EQU (0x1 << 13) ;- (PWMC) Output Selection Clear for PWMH output of the channel 13
AT91C_PWMC_OSCUPDH14      EQU (0x1 << 14) ;- (PWMC) Output Selection Clear for PWMH output of the channel 14
AT91C_PWMC_OSCUPDH15      EQU (0x1 << 15) ;- (PWMC) Output Selection Clear for PWMH output of the channel 15
AT91C_PWMC_OSCUPDL0       EQU (0x1 << 16) ;- (PWMC) Output Selection Clear for PWML output of the channel 0
AT91C_PWMC_OSCUPDL1       EQU (0x1 << 17) ;- (PWMC) Output Selection Clear for PWML output of the channel 1
AT91C_PWMC_OSCUPDL2       EQU (0x1 << 18) ;- (PWMC) Output Selection Clear for PWML output of the channel 2
AT91C_PWMC_OSCUPDL3       EQU (0x1 << 19) ;- (PWMC) Output Selection Clear for PWML output of the channel 3
AT91C_PWMC_OSCUPDL4       EQU (0x1 << 20) ;- (PWMC) Output Selection Clear for PWML output of the channel 4
AT91C_PWMC_OSCUPDL5       EQU (0x1 << 21) ;- (PWMC) Output Selection Clear for PWML output of the channel 5
AT91C_PWMC_OSCUPDL6       EQU (0x1 << 22) ;- (PWMC) Output Selection Clear for PWML output of the channel 6
AT91C_PWMC_OSCUPDL7       EQU (0x1 << 23) ;- (PWMC) Output Selection Clear for PWML output of the channel 7
AT91C_PWMC_OSCUPDL8       EQU (0x1 << 24) ;- (PWMC) Output Selection Clear for PWML output of the channel 8
AT91C_PWMC_OSCUPDL9       EQU (0x1 << 25) ;- (PWMC) Output Selection Clear for PWML output of the channel 9
AT91C_PWMC_OSCUPDL10      EQU (0x1 << 26) ;- (PWMC) Output Selection Clear for PWML output of the channel 10
AT91C_PWMC_OSCUPDL11      EQU (0x1 << 27) ;- (PWMC) Output Selection Clear for PWML output of the channel 11
AT91C_PWMC_OSCUPDL12      EQU (0x1 << 28) ;- (PWMC) Output Selection Clear for PWML output of the channel 12
AT91C_PWMC_OSCUPDL13      EQU (0x1 << 29) ;- (PWMC) Output Selection Clear for PWML output of the channel 13
AT91C_PWMC_OSCUPDL14      EQU (0x1 << 30) ;- (PWMC) Output Selection Clear for PWML output of the channel 14
AT91C_PWMC_OSCUPDL15      EQU (0x1 << 31) ;- (PWMC) Output Selection Clear for PWML output of the channel 15
// - -------- PWMC_FMR : (PWMC Offset: 0x5c) PWM Fault Mode Register -------- 
AT91C_PWMC_FPOL0          EQU (0x1 <<  0) ;- (PWMC) Fault Polarity on fault input 0
AT91C_PWMC_FPOL1          EQU (0x1 <<  1) ;- (PWMC) Fault Polarity on fault input 1
AT91C_PWMC_FPOL2          EQU (0x1 <<  2) ;- (PWMC) Fault Polarity on fault input 2
AT91C_PWMC_FPOL3          EQU (0x1 <<  3) ;- (PWMC) Fault Polarity on fault input 3
AT91C_PWMC_FPOL4          EQU (0x1 <<  4) ;- (PWMC) Fault Polarity on fault input 4
AT91C_PWMC_FPOL5          EQU (0x1 <<  5) ;- (PWMC) Fault Polarity on fault input 5
AT91C_PWMC_FPOL6          EQU (0x1 <<  6) ;- (PWMC) Fault Polarity on fault input 6
AT91C_PWMC_FPOL7          EQU (0x1 <<  7) ;- (PWMC) Fault Polarity on fault input 7
AT91C_PWMC_FMOD0          EQU (0x1 <<  8) ;- (PWMC) Fault Activation Mode on fault input 0
AT91C_PWMC_FMOD1          EQU (0x1 <<  9) ;- (PWMC) Fault Activation Mode on fault input 1
AT91C_PWMC_FMOD2          EQU (0x1 << 10) ;- (PWMC) Fault Activation Mode on fault input 2
AT91C_PWMC_FMOD3          EQU (0x1 << 11) ;- (PWMC) Fault Activation Mode on fault input 3
AT91C_PWMC_FMOD4          EQU (0x1 << 12) ;- (PWMC) Fault Activation Mode on fault input 4
AT91C_PWMC_FMOD5          EQU (0x1 << 13) ;- (PWMC) Fault Activation Mode on fault input 5
AT91C_PWMC_FMOD6          EQU (0x1 << 14) ;- (PWMC) Fault Activation Mode on fault input 6
AT91C_PWMC_FMOD7          EQU (0x1 << 15) ;- (PWMC) Fault Activation Mode on fault input 7
AT91C_PWMC_FFIL00         EQU (0x1 << 16) ;- (PWMC) Fault Filtering on fault input 0
AT91C_PWMC_FFIL01         EQU (0x1 << 17) ;- (PWMC) Fault Filtering on fault input 1
AT91C_PWMC_FFIL02         EQU (0x1 << 18) ;- (PWMC) Fault Filtering on fault input 2
AT91C_PWMC_FFIL03         EQU (0x1 << 19) ;- (PWMC) Fault Filtering on fault input 3
AT91C_PWMC_FFIL04         EQU (0x1 << 20) ;- (PWMC) Fault Filtering on fault input 4
AT91C_PWMC_FFIL05         EQU (0x1 << 21) ;- (PWMC) Fault Filtering on fault input 5
AT91C_PWMC_FFIL06         EQU (0x1 << 22) ;- (PWMC) Fault Filtering on fault input 6
AT91C_PWMC_FFIL07         EQU (0x1 << 23) ;- (PWMC) Fault Filtering on fault input 7
// - -------- PWMC_FSR : (PWMC Offset: 0x60) Fault Input x Value -------- 
AT91C_PWMC_FIV0           EQU (0x1 <<  0) ;- (PWMC) Fault Input 0 Value
AT91C_PWMC_FIV1           EQU (0x1 <<  1) ;- (PWMC) Fault Input 1 Value
AT91C_PWMC_FIV2           EQU (0x1 <<  2) ;- (PWMC) Fault Input 2 Value
AT91C_PWMC_FIV3           EQU (0x1 <<  3) ;- (PWMC) Fault Input 3 Value
AT91C_PWMC_FIV4           EQU (0x1 <<  4) ;- (PWMC) Fault Input 4 Value
AT91C_PWMC_FIV5           EQU (0x1 <<  5) ;- (PWMC) Fault Input 5 Value
AT91C_PWMC_FIV6           EQU (0x1 <<  6) ;- (PWMC) Fault Input 6 Value
AT91C_PWMC_FIV7           EQU (0x1 <<  7) ;- (PWMC) Fault Input 7 Value
AT91C_PWMC_FS0            EQU (0x1 <<  8) ;- (PWMC) Fault 0 Status
AT91C_PWMC_FS1            EQU (0x1 <<  9) ;- (PWMC) Fault 1 Status
AT91C_PWMC_FS2            EQU (0x1 << 10) ;- (PWMC) Fault 2 Status
AT91C_PWMC_FS3            EQU (0x1 << 11) ;- (PWMC) Fault 3 Status
AT91C_PWMC_FS4            EQU (0x1 << 12) ;- (PWMC) Fault 4 Status
AT91C_PWMC_FS5            EQU (0x1 << 13) ;- (PWMC) Fault 5 Status
AT91C_PWMC_FS6            EQU (0x1 << 14) ;- (PWMC) Fault 6 Status
AT91C_PWMC_FS7            EQU (0x1 << 15) ;- (PWMC) Fault 7 Status
// - -------- PWMC_FCR : (PWMC Offset: 0x64) Fault y Clear -------- 
AT91C_PWMC_FCLR0          EQU (0x1 <<  0) ;- (PWMC) Fault 0 Clear
AT91C_PWMC_FCLR1          EQU (0x1 <<  1) ;- (PWMC) Fault 1 Clear
AT91C_PWMC_FCLR2          EQU (0x1 <<  2) ;- (PWMC) Fault 2 Clear
AT91C_PWMC_FCLR3          EQU (0x1 <<  3) ;- (PWMC) Fault 3 Clear
AT91C_PWMC_FCLR4          EQU (0x1 <<  4) ;- (PWMC) Fault 4 Clear
AT91C_PWMC_FCLR5          EQU (0x1 <<  5) ;- (PWMC) Fault 5 Clear
AT91C_PWMC_FCLR6          EQU (0x1 <<  6) ;- (PWMC) Fault 6 Clear
AT91C_PWMC_FCLR7          EQU (0x1 <<  7) ;- (PWMC) Fault 7 Clear
// - -------- PWMC_FPV : (PWMC Offset: 0x68) PWM Fault Protection Value -------- 
AT91C_PWMC_FPVH0          EQU (0x1 <<  0) ;- (PWMC) Fault Protection Value for PWMH output on channel 0
AT91C_PWMC_FPVH1          EQU (0x1 <<  1) ;- (PWMC) Fault Protection Value for PWMH output on channel 1
AT91C_PWMC_FPVH2          EQU (0x1 <<  2) ;- (PWMC) Fault Protection Value for PWMH output on channel 2
AT91C_PWMC_FPVH3          EQU (0x1 <<  3) ;- (PWMC) Fault Protection Value for PWMH output on channel 3
AT91C_PWMC_FPVH4          EQU (0x1 <<  4) ;- (PWMC) Fault Protection Value for PWMH output on channel 4
AT91C_PWMC_FPVH5          EQU (0x1 <<  5) ;- (PWMC) Fault Protection Value for PWMH output on channel 5
AT91C_PWMC_FPVH6          EQU (0x1 <<  6) ;- (PWMC) Fault Protection Value for PWMH output on channel 6
AT91C_PWMC_FPVH7          EQU (0x1 <<  7) ;- (PWMC) Fault Protection Value for PWMH output on channel 7
AT91C_PWMC_FPVL0          EQU (0x1 << 16) ;- (PWMC) Fault Protection Value for PWML output on channel 0
AT91C_PWMC_FPVL1          EQU (0x1 << 17) ;- (PWMC) Fault Protection Value for PWML output on channel 1
AT91C_PWMC_FPVL2          EQU (0x1 << 18) ;- (PWMC) Fault Protection Value for PWML output on channel 2
AT91C_PWMC_FPVL3          EQU (0x1 << 19) ;- (PWMC) Fault Protection Value for PWML output on channel 3
AT91C_PWMC_FPVL4          EQU (0x1 << 20) ;- (PWMC) Fault Protection Value for PWML output on channel 4
AT91C_PWMC_FPVL5          EQU (0x1 << 21) ;- (PWMC) Fault Protection Value for PWML output on channel 5
AT91C_PWMC_FPVL6          EQU (0x1 << 22) ;- (PWMC) Fault Protection Value for PWML output on channel 6
AT91C_PWMC_FPVL7          EQU (0x1 << 23) ;- (PWMC) Fault Protection Value for PWML output on channel 7
// - -------- PWMC_FPER1 : (PWMC Offset: 0x6c) PWM Fault Protection Enable Register 1 -------- 
AT91C_PWMC_FPE0           EQU (0xFF <<  0) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 0
AT91C_PWMC_FPE1           EQU (0xFF <<  8) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 1
AT91C_PWMC_FPE2           EQU (0xFF << 16) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 2
AT91C_PWMC_FPE3           EQU (0xFF << 24) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 3
// - -------- PWMC_FPER2 : (PWMC Offset: 0x70) PWM Fault Protection Enable Register 2 -------- 
AT91C_PWMC_FPE4           EQU (0xFF <<  0) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 4
AT91C_PWMC_FPE5           EQU (0xFF <<  8) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 5
AT91C_PWMC_FPE6           EQU (0xFF << 16) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 6
AT91C_PWMC_FPE7           EQU (0xFF << 24) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 7
// - -------- PWMC_FPER3 : (PWMC Offset: 0x74) PWM Fault Protection Enable Register 3 -------- 
AT91C_PWMC_FPE8           EQU (0xFF <<  0) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 8
AT91C_PWMC_FPE9           EQU (0xFF <<  8) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 9
AT91C_PWMC_FPE10          EQU (0xFF << 16) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 10
AT91C_PWMC_FPE11          EQU (0xFF << 24) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 11
// - -------- PWMC_FPER4 : (PWMC Offset: 0x78) PWM Fault Protection Enable Register 4 -------- 
AT91C_PWMC_FPE12          EQU (0xFF <<  0) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 12
AT91C_PWMC_FPE13          EQU (0xFF <<  8) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 13
AT91C_PWMC_FPE14          EQU (0xFF << 16) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 14
AT91C_PWMC_FPE15          EQU (0xFF << 24) ;- (PWMC) Fault Protection Enable with Fault Input y for PWM channel 15
// - -------- PWMC_EL0MR : (PWMC Offset: 0x7c) PWM Event Line 0 Mode Register -------- 
AT91C_PWMC_L0CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L0CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L0CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L0CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L0CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L0CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L0CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L0CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL1MR : (PWMC Offset: 0x80) PWM Event Line 1 Mode Register -------- 
AT91C_PWMC_L1CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L1CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L1CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L1CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L1CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L1CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L1CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L1CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL2MR : (PWMC Offset: 0x84) PWM Event line 2 Mode Register -------- 
AT91C_PWMC_L2CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L2CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L2CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L2CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L2CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L2CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L2CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L2CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL3MR : (PWMC Offset: 0x88) PWM Event line 3 Mode Register -------- 
AT91C_PWMC_L3CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L3CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L3CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L3CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L3CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L3CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L3CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L3CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL4MR : (PWMC Offset: 0x8c) PWM Event line 4 Mode Register -------- 
AT91C_PWMC_L4CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L4CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L4CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L4CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L4CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L4CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L4CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L4CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL5MR : (PWMC Offset: 0x90) PWM Event line 5 Mode Register -------- 
AT91C_PWMC_L5CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L5CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L5CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L5CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L5CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L5CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L5CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L5CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL6MR : (PWMC Offset: 0x94) PWM Event line 6 Mode Register -------- 
AT91C_PWMC_L6CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L6CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L6CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L6CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L6CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L6CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L6CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L6CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_EL7MR : (PWMC Offset: 0x98) PWM Event line 7 Mode Register -------- 
AT91C_PWMC_L7CSEL0        EQU (0x1 <<  0) ;- (PWMC) Comparison 0 Selection
AT91C_PWMC_L7CSEL1        EQU (0x1 <<  1) ;- (PWMC) Comparison 1 Selection
AT91C_PWMC_L7CSEL2        EQU (0x1 <<  2) ;- (PWMC) Comparison 2 Selection
AT91C_PWMC_L7CSEL3        EQU (0x1 <<  3) ;- (PWMC) Comparison 3 Selection
AT91C_PWMC_L7CSEL4        EQU (0x1 <<  4) ;- (PWMC) Comparison 4 Selection
AT91C_PWMC_L7CSEL5        EQU (0x1 <<  5) ;- (PWMC) Comparison 5 Selection
AT91C_PWMC_L7CSEL6        EQU (0x1 <<  6) ;- (PWMC) Comparison 6 Selection
AT91C_PWMC_L7CSEL7        EQU (0x1 <<  7) ;- (PWMC) Comparison 7 Selection
// - -------- PWMC_WPCR : (PWMC Offset: 0xe4) PWM Write Protection Control Register -------- 
AT91C_PWMC_WPCMD          EQU (0x3 <<  0) ;- (PWMC) Write Protection Command
AT91C_PWMC_WPRG0          EQU (0x1 <<  2) ;- (PWMC) Write Protect Register Group 0
AT91C_PWMC_WPRG1          EQU (0x1 <<  3) ;- (PWMC) Write Protect Register Group 1
AT91C_PWMC_WPRG2          EQU (0x1 <<  4) ;- (PWMC) Write Protect Register Group 2
AT91C_PWMC_WPRG3          EQU (0x1 <<  5) ;- (PWMC) Write Protect Register Group 3
AT91C_PWMC_WPRG4          EQU (0x1 <<  6) ;- (PWMC) Write Protect Register Group 4
AT91C_PWMC_WPRG5          EQU (0x1 <<  7) ;- (PWMC) Write Protect Register Group 5
AT91C_PWMC_WPKEY          EQU (0xFFFFFF <<  8) ;- (PWMC) Protection Password
// - -------- PWMC_WPVS : (PWMC Offset: 0xe8) Write Protection Status Register -------- 
AT91C_PWMC_WPSWS0         EQU (0x1 <<  0) ;- (PWMC) Write Protect SW Group 0 Status 
AT91C_PWMC_WPSWS1         EQU (0x1 <<  1) ;- (PWMC) Write Protect SW Group 1 Status 
AT91C_PWMC_WPSWS2         EQU (0x1 <<  2) ;- (PWMC) Write Protect SW Group 2 Status 
AT91C_PWMC_WPSWS3         EQU (0x1 <<  3) ;- (PWMC) Write Protect SW Group 3 Status 
AT91C_PWMC_WPSWS4         EQU (0x1 <<  4) ;- (PWMC) Write Protect SW Group 4 Status 
AT91C_PWMC_WPSWS5         EQU (0x1 <<  5) ;- (PWMC) Write Protect SW Group 5 Status 
AT91C_PWMC_WPVS           EQU (0x1 <<  7) ;- (PWMC) Write Protection Enable
AT91C_PWMC_WPHWS0         EQU (0x1 <<  8) ;- (PWMC) Write Protect HW Group 0 Status 
AT91C_PWMC_WPHWS1         EQU (0x1 <<  9) ;- (PWMC) Write Protect HW Group 1 Status 
AT91C_PWMC_WPHWS2         EQU (0x1 << 10) ;- (PWMC) Write Protect HW Group 2 Status 
AT91C_PWMC_WPHWS3         EQU (0x1 << 11) ;- (PWMC) Write Protect HW Group 3 Status 
AT91C_PWMC_WPHWS4         EQU (0x1 << 12) ;- (PWMC) Write Protect HW Group 4 Status 
AT91C_PWMC_WPHWS5         EQU (0x1 << 13) ;- (PWMC) Write Protect HW Group 5 Status 
AT91C_PWMC_WPVSRC         EQU (0xFFFF << 16) ;- (PWMC) Write Protection Violation Source
// - -------- PWMC_CMP0V : (PWMC Offset: 0x130) PWM Comparison Value 0 Register -------- 
AT91C_PWMC_CV             EQU (0xFFFFFF <<  0) ;- (PWMC) PWM Comparison Value 0.
AT91C_PWMC_CVM            EQU (0x1 << 24) ;- (PWMC) Comparison Value 0 Mode.
// - -------- PWMC_CMP0VUPD : (PWMC Offset: 0x134) PWM Comparison Value 0 Update Register -------- 
AT91C_PWMC_CVUPD          EQU (0xFFFFFF <<  0) ;- (PWMC) PWM Comparison Value Update.
AT91C_PWMC_CVMUPD         EQU (0x1 << 24) ;- (PWMC) Comparison Value Update Mode.
// - -------- PWMC_CMP0M : (PWMC Offset: 0x138) PWM Comparison 0 Mode Register -------- 
AT91C_PWMC_CEN            EQU (0x1 <<  0) ;- (PWMC) Comparison Enable.
AT91C_PWMC_CTR            EQU (0xF <<  4) ;- (PWMC) PWM Comparison Trigger.
AT91C_PWMC_CPR            EQU (0xF <<  8) ;- (PWMC) PWM Comparison Period.
AT91C_PWMC_CPRCNT         EQU (0xF << 12) ;- (PWMC) PWM Comparison Period Counter.
AT91C_PWMC_CUPR           EQU (0xF << 16) ;- (PWMC) PWM Comparison Update Period.
AT91C_PWMC_CUPRCNT        EQU (0xF << 20) ;- (PWMC) PWM Comparison Update Period Counter.
// - -------- PWMC_CMP0MUPD : (PWMC Offset: 0x13c) PWM Comparison 0 Mode Update Register -------- 
AT91C_PWMC_CENUPD         EQU (0x1 <<  0) ;- (PWMC) Comparison Enable Update.
AT91C_PWMC_CTRUPD         EQU (0xF <<  4) ;- (PWMC) PWM Comparison Trigger Update.
AT91C_PWMC_CPRUPD         EQU (0xF <<  8) ;- (PWMC) PWM Comparison Period Update.
AT91C_PWMC_CUPRUPD        EQU (0xF << 16) ;- (PWMC) PWM Comparison Update Period Update.
// - -------- PWMC_CMP1V : (PWMC Offset: 0x140) PWM Comparison Value 1 Register -------- 
// - -------- PWMC_CMP1VUPD : (PWMC Offset: 0x144) PWM Comparison Value 1 Update Register -------- 
// - -------- PWMC_CMP1M : (PWMC Offset: 0x148) PWM Comparison 1 Mode Register -------- 
// - -------- PWMC_CMP1MUPD : (PWMC Offset: 0x14c) PWM Comparison 1 Mode Update Register -------- 
// - -------- PWMC_CMP2V : (PWMC Offset: 0x150) PWM Comparison Value 2 Register -------- 
// - -------- PWMC_CMP2VUPD : (PWMC Offset: 0x154) PWM Comparison Value 2 Update Register -------- 
// - -------- PWMC_CMP2M : (PWMC Offset: 0x158) PWM Comparison 2 Mode Register -------- 
// - -------- PWMC_CMP2MUPD : (PWMC Offset: 0x15c) PWM Comparison 2 Mode Update Register -------- 
// - -------- PWMC_CMP3V : (PWMC Offset: 0x160) PWM Comparison Value 3 Register -------- 
// - -------- PWMC_CMP3VUPD : (PWMC Offset: 0x164) PWM Comparison Value 3 Update Register -------- 
// - -------- PWMC_CMP3M : (PWMC Offset: 0x168) PWM Comparison 3 Mode Register -------- 
// - -------- PWMC_CMP3MUPD : (PWMC Offset: 0x16c) PWM Comparison 3 Mode Update Register -------- 
// - -------- PWMC_CMP4V : (PWMC Offset: 0x170) PWM Comparison Value 4 Register -------- 
// - -------- PWMC_CMP4VUPD : (PWMC Offset: 0x174) PWM Comparison Value 4 Update Register -------- 
// - -------- PWMC_CMP4M : (PWMC Offset: 0x178) PWM Comparison 4 Mode Register -------- 
// - -------- PWMC_CMP4MUPD : (PWMC Offset: 0x17c) PWM Comparison 4 Mode Update Register -------- 
// - -------- PWMC_CMP5V : (PWMC Offset: 0x180) PWM Comparison Value 5 Register -------- 
// - -------- PWMC_CMP5VUPD : (PWMC Offset: 0x184) PWM Comparison Value 5 Update Register -------- 
// - -------- PWMC_CMP5M : (PWMC Offset: 0x188) PWM Comparison 5 Mode Register -------- 
// - -------- PWMC_CMP5MUPD : (PWMC Offset: 0x18c) PWM Comparison 5 Mode Update Register -------- 
// - -------- PWMC_CMP6V : (PWMC Offset: 0x190) PWM Comparison Value 6 Register -------- 
// - -------- PWMC_CMP6VUPD : (PWMC Offset: 0x194) PWM Comparison Value 6 Update Register -------- 
// - -------- PWMC_CMP6M : (PWMC Offset: 0x198) PWM Comparison 6 Mode Register -------- 
// - -------- PWMC_CMP6MUPD : (PWMC Offset: 0x19c) PWM Comparison 6 Mode Update Register -------- 
// - -------- PWMC_CMP7V : (PWMC Offset: 0x1a0) PWM Comparison Value 7 Register -------- 
// - -------- PWMC_CMP7VUPD : (PWMC Offset: 0x1a4) PWM Comparison Value 7 Update Register -------- 
// - -------- PWMC_CMP7M : (PWMC Offset: 0x1a8) PWM Comparison 7 Mode Register -------- 
// - -------- PWMC_CMP7MUPD : (PWMC Offset: 0x1ac) PWM Comparison 7 Mode Update Register -------- 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Serial Parallel Interface
// - *****************************************************************************
// - -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- 
AT91C_SPI_SPIEN           EQU (0x1 <<  0) ;- (SPI) SPI Enable
AT91C_SPI_SPIDIS          EQU (0x1 <<  1) ;- (SPI) SPI Disable
AT91C_SPI_SWRST           EQU (0x1 <<  7) ;- (SPI) SPI Software reset
AT91C_SPI_LASTXFER        EQU (0x1 << 24) ;- (SPI) SPI Last Transfer
// - -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- 
AT91C_SPI_MSTR            EQU (0x1 <<  0) ;- (SPI) Master/Slave Mode
AT91C_SPI_PS              EQU (0x1 <<  1) ;- (SPI) Peripheral Select
AT91C_SPI_PS_FIXED        EQU (0x0 <<  1) ;- (SPI) Fixed Peripheral Select
AT91C_SPI_PS_VARIABLE     EQU (0x1 <<  1) ;- (SPI) Variable Peripheral Select
AT91C_SPI_PCSDEC          EQU (0x1 <<  2) ;- (SPI) Chip Select Decode
AT91C_SPI_FDIV            EQU (0x1 <<  3) ;- (SPI) Clock Selection
AT91C_SPI_MODFDIS         EQU (0x1 <<  4) ;- (SPI) Mode Fault Detection
AT91C_SPI_LLB             EQU (0x1 <<  7) ;- (SPI) Clock Selection
AT91C_SPI_PCS             EQU (0xF << 16) ;- (SPI) Peripheral Chip Select
AT91C_SPI_DLYBCS          EQU (0xFF << 24) ;- (SPI) Delay Between Chip Selects
// - -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- 
AT91C_SPI_RD              EQU (0xFFFF <<  0) ;- (SPI) Receive Data
AT91C_SPI_RPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
// - -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- 
AT91C_SPI_TD              EQU (0xFFFF <<  0) ;- (SPI) Transmit Data
AT91C_SPI_TPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
// - -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- 
AT91C_SPI_RDRF            EQU (0x1 <<  0) ;- (SPI) Receive Data Register Full
AT91C_SPI_TDRE            EQU (0x1 <<  1) ;- (SPI) Transmit Data Register Empty
AT91C_SPI_MODF            EQU (0x1 <<  2) ;- (SPI) Mode Fault Error
AT91C_SPI_OVRES           EQU (0x1 <<  3) ;- (SPI) Overrun Error Status
AT91C_SPI_ENDRX           EQU (0x1 <<  4) ;- (SPI) End of Receiver Transfer
AT91C_SPI_ENDTX           EQU (0x1 <<  5) ;- (SPI) End of Receiver Transfer
AT91C_SPI_RXBUFF          EQU (0x1 <<  6) ;- (SPI) RXBUFF Interrupt
AT91C_SPI_TXBUFE          EQU (0x1 <<  7) ;- (SPI) TXBUFE Interrupt
AT91C_SPI_NSSR            EQU (0x1 <<  8) ;- (SPI) NSSR Interrupt
AT91C_SPI_TXEMPTY         EQU (0x1 <<  9) ;- (SPI) TXEMPTY Interrupt
AT91C_SPI_SPIENS          EQU (0x1 << 16) ;- (SPI) Enable Status
// - -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- 
// - -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- 
// - -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- 
// - -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- 
AT91C_SPI_CPOL            EQU (0x1 <<  0) ;- (SPI) Clock Polarity
AT91C_SPI_NCPHA           EQU (0x1 <<  1) ;- (SPI) Clock Phase
AT91C_SPI_CSNAAT          EQU (0x1 <<  2) ;- (SPI) Chip Select Not Active After Transfer (Ignored if CSAAT = 1)
AT91C_SPI_CSAAT           EQU (0x1 <<  3) ;- (SPI) Chip Select Active After Transfer
AT91C_SPI_BITS            EQU (0xF <<  4) ;- (SPI) Bits Per Transfer
AT91C_SPI_BITS_8          EQU (0x0 <<  4) ;- (SPI) 8 Bits Per transfer
AT91C_SPI_BITS_9          EQU (0x1 <<  4) ;- (SPI) 9 Bits Per transfer
AT91C_SPI_BITS_10         EQU (0x2 <<  4) ;- (SPI) 10 Bits Per transfer
AT91C_SPI_BITS_11         EQU (0x3 <<  4) ;- (SPI) 11 Bits Per transfer
AT91C_SPI_BITS_12         EQU (0x4 <<  4) ;- (SPI) 12 Bits Per transfer
AT91C_SPI_BITS_13         EQU (0x5 <<  4) ;- (SPI) 13 Bits Per transfer
AT91C_SPI_BITS_14         EQU (0x6 <<  4) ;- (SPI) 14 Bits Per transfer
AT91C_SPI_BITS_15         EQU (0x7 <<  4) ;- (SPI) 15 Bits Per transfer
AT91C_SPI_BITS_16         EQU (0x8 <<  4) ;- (SPI) 16 Bits Per transfer
AT91C_SPI_SCBR            EQU (0xFF <<  8) ;- (SPI) Serial Clock Baud Rate
AT91C_SPI_DLYBS           EQU (0xFF << 16) ;- (SPI) Delay Before SPCK
AT91C_SPI_DLYBCT          EQU (0xFF << 24) ;- (SPI) Delay Between Consecutive Transfers

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register
// - *****************************************************************************

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct
// - *****************************************************************************
// - -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- 
AT91C_UDPHS_EPT_SIZE      EQU (0x7 <<  0) ;- (UDPHS_EPT) Endpoint Size
AT91C_UDPHS_EPT_SIZE_8    EQU (0x0) ;- (UDPHS_EPT)    8 bytes
AT91C_UDPHS_EPT_SIZE_16   EQU (0x1) ;- (UDPHS_EPT)   16 bytes
AT91C_UDPHS_EPT_SIZE_32   EQU (0x2) ;- (UDPHS_EPT)   32 bytes
AT91C_UDPHS_EPT_SIZE_64   EQU (0x3) ;- (UDPHS_EPT)   64 bytes
AT91C_UDPHS_EPT_SIZE_128  EQU (0x4) ;- (UDPHS_EPT)  128 bytes
AT91C_UDPHS_EPT_SIZE_256  EQU (0x5) ;- (UDPHS_EPT)  256 bytes (if possible)
AT91C_UDPHS_EPT_SIZE_512  EQU (0x6) ;- (UDPHS_EPT)  512 bytes (if possible)
AT91C_UDPHS_EPT_SIZE_1024 EQU (0x7) ;- (UDPHS_EPT) 1024 bytes (if possible)
AT91C_UDPHS_EPT_DIR       EQU (0x1 <<  3) ;- (UDPHS_EPT) Endpoint Direction 0:OUT, 1:IN
AT91C_UDPHS_EPT_DIR_OUT   EQU (0x0 <<  3) ;- (UDPHS_EPT) Direction OUT
AT91C_UDPHS_EPT_DIR_IN    EQU (0x1 <<  3) ;- (UDPHS_EPT) Direction IN
AT91C_UDPHS_EPT_TYPE      EQU (0x3 <<  4) ;- (UDPHS_EPT) Endpoint Type
AT91C_UDPHS_EPT_TYPE_CTL_EPT EQU (0x0 <<  4) ;- (UDPHS_EPT) Control endpoint
AT91C_UDPHS_EPT_TYPE_ISO_EPT EQU (0x1 <<  4) ;- (UDPHS_EPT) Isochronous endpoint
AT91C_UDPHS_EPT_TYPE_BUL_EPT EQU (0x2 <<  4) ;- (UDPHS_EPT) Bulk endpoint
AT91C_UDPHS_EPT_TYPE_INT_EPT EQU (0x3 <<  4) ;- (UDPHS_EPT) Interrupt endpoint
AT91C_UDPHS_BK_NUMBER     EQU (0x3 <<  6) ;- (UDPHS_EPT) Number of Banks
AT91C_UDPHS_BK_NUMBER_0   EQU (0x0 <<  6) ;- (UDPHS_EPT) Zero Bank, the EndPoint is not mapped in memory
AT91C_UDPHS_BK_NUMBER_1   EQU (0x1 <<  6) ;- (UDPHS_EPT) One Bank (Bank0)
AT91C_UDPHS_BK_NUMBER_2   EQU (0x2 <<  6) ;- (UDPHS_EPT) Double bank (Ping-Pong : Bank0 / Bank1)
AT91C_UDPHS_BK_NUMBER_3   EQU (0x3 <<  6) ;- (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2) (if possible)
AT91C_UDPHS_NB_TRANS      EQU (0x3 <<  8) ;- (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)
AT91C_UDPHS_EPT_MAPD      EQU (0x1 << 31) ;- (UDPHS_EPT) Endpoint Mapped (read only
// - -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- 
AT91C_UDPHS_EPT_ENABL     EQU (0x1 <<  0) ;- (UDPHS_EPT) Endpoint Enable
AT91C_UDPHS_AUTO_VALID    EQU (0x1 <<  1) ;- (UDPHS_EPT) Packet Auto-Valid Enable/Disable
AT91C_UDPHS_INTDIS_DMA    EQU (0x1 <<  3) ;- (UDPHS_EPT) Endpoint Interrupts DMA Request Enable/Disable
AT91C_UDPHS_NYET_DIS      EQU (0x1 <<  4) ;- (UDPHS_EPT) NYET Enable/Disable
AT91C_UDPHS_DATAX_RX      EQU (0x1 <<  6) ;- (UDPHS_EPT) DATAx Interrupt Enable/Disable
AT91C_UDPHS_MDATA_RX      EQU (0x1 <<  7) ;- (UDPHS_EPT) MDATA Interrupt Enabled/Disable
AT91C_UDPHS_ERR_OVFLW     EQU (0x1 <<  8) ;- (UDPHS_EPT) OverFlow Error Interrupt Enable/Disable/Status
AT91C_UDPHS_RX_BK_RDY     EQU (0x1 <<  9) ;- (UDPHS_EPT) Received OUT Data
AT91C_UDPHS_TX_COMPLT     EQU (0x1 << 10) ;- (UDPHS_EPT) Transmitted IN Data Complete Interrupt Enable/Disable or Transmitted IN Data Complete (clear)
AT91C_UDPHS_ERR_TRANS     EQU (0x1 << 11) ;- (UDPHS_EPT) Transaction Error Interrupt Enable/Disable
AT91C_UDPHS_TX_PK_RDY     EQU (0x1 << 11) ;- (UDPHS_EPT) TX Packet Ready Interrupt Enable/Disable
AT91C_UDPHS_RX_SETUP      EQU (0x1 << 12) ;- (UDPHS_EPT) Received SETUP Interrupt Enable/Disable
AT91C_UDPHS_ERR_FL_ISO    EQU (0x1 << 12) ;- (UDPHS_EPT) Error Flow Clear/Interrupt Enable/Disable
AT91C_UDPHS_STALL_SNT     EQU (0x1 << 13) ;- (UDPHS_EPT) Stall Sent Clear
AT91C_UDPHS_ERR_CRISO     EQU (0x1 << 13) ;- (UDPHS_EPT) CRC error / Error NB Trans / Interrupt Enable/Disable
AT91C_UDPHS_NAK_IN        EQU (0x1 << 14) ;- (UDPHS_EPT) NAKIN ERROR FLUSH / Clear / Interrupt Enable/Disable
AT91C_UDPHS_NAK_OUT       EQU (0x1 << 15) ;- (UDPHS_EPT) NAKOUT / Clear / Interrupt Enable/Disable
AT91C_UDPHS_BUSY_BANK     EQU (0x1 << 18) ;- (UDPHS_EPT) Busy Bank Interrupt Enable/Disable
AT91C_UDPHS_SHRT_PCKT     EQU (0x1 << 31) ;- (UDPHS_EPT) Short Packet / Interrupt Enable/Disable
// - -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- 
AT91C_UDPHS_EPT_DISABL    EQU (0x1 <<  0) ;- (UDPHS_EPT) Endpoint Disable
// - -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- 
// - -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- 
AT91C_UDPHS_FRCESTALL     EQU (0x1 <<  5) ;- (UDPHS_EPT) Stall Handshake Request Set/Clear/Status
AT91C_UDPHS_KILL_BANK     EQU (0x1 <<  9) ;- (UDPHS_EPT) KILL Bank
// - -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- 
AT91C_UDPHS_TOGGLESQ      EQU (0x1 <<  6) ;- (UDPHS_EPT) Data Toggle Clear
// - -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- 
AT91C_UDPHS_TOGGLESQ_STA  EQU (0x3 <<  6) ;- (UDPHS_EPT) Toggle Sequencing
AT91C_UDPHS_TOGGLESQ_STA_00 EQU (0x0 <<  6) ;- (UDPHS_EPT) Data0
AT91C_UDPHS_TOGGLESQ_STA_01 EQU (0x1 <<  6) ;- (UDPHS_EPT) Data1
AT91C_UDPHS_TOGGLESQ_STA_10 EQU (0x2 <<  6) ;- (UDPHS_EPT) Data2 (only for High-Bandwidth Isochronous EndPoint)
AT91C_UDPHS_TOGGLESQ_STA_11 EQU (0x3 <<  6) ;- (UDPHS_EPT) MData (only for High-Bandwidth Isochronous EndPoint)
AT91C_UDPHS_CONTROL_DIR   EQU (0x3 << 16) ;- (UDPHS_EPT) 
AT91C_UDPHS_CONTROL_DIR_00 EQU (0x0 << 16) ;- (UDPHS_EPT) Bank 0
AT91C_UDPHS_CONTROL_DIR_01 EQU (0x1 << 16) ;- (UDPHS_EPT) Bank 1
AT91C_UDPHS_CONTROL_DIR_10 EQU (0x2 << 16) ;- (UDPHS_EPT) Bank 2
AT91C_UDPHS_CONTROL_DIR_11 EQU (0x3 << 16) ;- (UDPHS_EPT) Invalid
AT91C_UDPHS_CURRENT_BANK  EQU (0x3 << 16) ;- (UDPHS_EPT) 
AT91C_UDPHS_CURRENT_BANK_00 EQU (0x0 << 16) ;- (UDPHS_EPT) Bank 0
AT91C_UDPHS_CURRENT_BANK_01 EQU (0x1 << 16) ;- (UDPHS_EPT) Bank 1
AT91C_UDPHS_CURRENT_BANK_10 EQU (0x2 << 16) ;- (UDPHS_EPT) Bank 2
AT91C_UDPHS_CURRENT_BANK_11 EQU (0x3 << 16) ;- (UDPHS_EPT) Invalid
AT91C_UDPHS_BUSY_BANK_STA EQU (0x3 << 18) ;- (UDPHS_EPT) Busy Bank Number
AT91C_UDPHS_BUSY_BANK_STA_00 EQU (0x0 << 18) ;- (UDPHS_EPT) All banks are free
AT91C_UDPHS_BUSY_BANK_STA_01 EQU (0x1 << 18) ;- (UDPHS_EPT) 1 busy bank
AT91C_UDPHS_BUSY_BANK_STA_10 EQU (0x2 << 18) ;- (UDPHS_EPT) 2 busy banks
AT91C_UDPHS_BUSY_BANK_STA_11 EQU (0x3 << 18) ;- (UDPHS_EPT) 3 busy banks (if possible)
AT91C_UDPHS_BYTE_COUNT    EQU (0x7FF << 20) ;- (UDPHS_EPT) UDPHS Byte Count

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS DMA struct
// - *****************************************************************************
// - -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- 
AT91C_UDPHS_NXT_DSC_ADD   EQU (0xFFFFFFF <<  4) ;- (UDPHS_DMA) next Channel Descriptor
// - -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- 
AT91C_UDPHS_BUFF_ADD      EQU (0x0 <<  0) ;- (UDPHS_DMA) starting address of a DMA Channel transfer
// - -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- 
AT91C_UDPHS_CHANN_ENB     EQU (0x1 <<  0) ;- (UDPHS_DMA) Channel Enabled
AT91C_UDPHS_LDNXT_DSC     EQU (0x1 <<  1) ;- (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable
AT91C_UDPHS_END_TR_EN     EQU (0x1 <<  2) ;- (UDPHS_DMA) Buffer Close Input Enable
AT91C_UDPHS_END_B_EN      EQU (0x1 <<  3) ;- (UDPHS_DMA) End of DMA Buffer Packet Validation
AT91C_UDPHS_END_TR_IT     EQU (0x1 <<  4) ;- (UDPHS_DMA) End Of Transfer Interrupt Enable
AT91C_UDPHS_END_BUFFIT    EQU (0x1 <<  5) ;- (UDPHS_DMA) End Of Channel Buffer Interrupt Enable
AT91C_UDPHS_DESC_LD_IT    EQU (0x1 <<  6) ;- (UDPHS_DMA) Descriptor Loaded Interrupt Enable
AT91C_UDPHS_BURST_LCK     EQU (0x1 <<  7) ;- (UDPHS_DMA) Burst Lock Enable
AT91C_UDPHS_BUFF_LENGTH   EQU (0xFFFF << 16) ;- (UDPHS_DMA) Buffer Byte Length (write only)
// - -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- 
AT91C_UDPHS_CHANN_ACT     EQU (0x1 <<  1) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_TR_ST     EQU (0x1 <<  4) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_BF_ST     EQU (0x1 <<  5) ;- (UDPHS_DMA) 
AT91C_UDPHS_DESC_LDST     EQU (0x1 <<  6) ;- (UDPHS_DMA) 
AT91C_UDPHS_BUFF_COUNT    EQU (0xFFFF << 16) ;- (UDPHS_DMA) 

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface
// - *****************************************************************************
// - -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- 
AT91C_UDPHS_DEV_ADDR      EQU (0x7F <<  0) ;- (UDPHS) UDPHS Address
AT91C_UDPHS_FADDR_EN      EQU (0x1 <<  7) ;- (UDPHS) Function Address Enable
AT91C_UDPHS_EN_UDPHS      EQU (0x1 <<  8) ;- (UDPHS) UDPHS Enable
AT91C_UDPHS_DETACH        EQU (0x1 <<  9) ;- (UDPHS) Detach Command
AT91C_UDPHS_REWAKEUP      EQU (0x1 << 10) ;- (UDPHS) Send Remote Wake Up
AT91C_UDPHS_PULLD_DIS     EQU (0x1 << 11) ;- (UDPHS) PullDown Disable
// - -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- 
AT91C_UDPHS_MICRO_FRAME_NUM EQU (0x7 <<  0) ;- (UDPHS) Micro Frame Number
AT91C_UDPHS_FRAME_NUMBER  EQU (0x7FF <<  3) ;- (UDPHS) Frame Number as defined in the Packet Field Formats
AT91C_UDPHS_FNUM_ERR      EQU (0x1 << 31) ;- (UDPHS) Frame Number CRC Error
// - -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- 
AT91C_UDPHS_DET_SUSPD     EQU (0x1 <<  1) ;- (UDPHS) Suspend Interrupt Enable/Clear/Status
AT91C_UDPHS_MICRO_SOF     EQU (0x1 <<  2) ;- (UDPHS) Micro-SOF Interrupt Enable/Clear/Status
AT91C_UDPHS_IEN_SOF       EQU (0x1 <<  3) ;- (UDPHS) SOF Interrupt Enable/Clear/Status
AT91C_UDPHS_ENDRESET      EQU (0x1 <<  4) ;- (UDPHS) End Of Reset Interrupt Enable/Clear/Status
AT91C_UDPHS_WAKE_UP       EQU (0x1 <<  5) ;- (UDPHS) Wake Up CPU Interrupt Enable/Clear/Status
AT91C_UDPHS_ENDOFRSM      EQU (0x1 <<  6) ;- (UDPHS) End Of Resume Interrupt Enable/Clear/Status
AT91C_UDPHS_UPSTR_RES     EQU (0x1 <<  7) ;- (UDPHS) Upstream Resume Interrupt Enable/Clear/Status
AT91C_UDPHS_EPT_INT_0     EQU (0x1 <<  8) ;- (UDPHS) Endpoint 0 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_1     EQU (0x1 <<  9) ;- (UDPHS) Endpoint 1 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_2     EQU (0x1 << 10) ;- (UDPHS) Endpoint 2 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_3     EQU (0x1 << 11) ;- (UDPHS) Endpoint 3 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_4     EQU (0x1 << 12) ;- (UDPHS) Endpoint 4 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_5     EQU (0x1 << 13) ;- (UDPHS) Endpoint 5 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_6     EQU (0x1 << 14) ;- (UDPHS) Endpoint 6 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_1     EQU (0x1 << 25) ;- (UDPHS) DMA Channel 1 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_2     EQU (0x1 << 26) ;- (UDPHS) DMA Channel 2 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_3     EQU (0x1 << 27) ;- (UDPHS) DMA Channel 3 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_4     EQU (0x1 << 28) ;- (UDPHS) DMA Channel 4 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_5     EQU (0x1 << 29) ;- (UDPHS) DMA Channel 5 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_6     EQU (0x1 << 30) ;- (UDPHS) DMA Channel 6 Interrupt Enable/Status
// - -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- 
AT91C_UDPHS_SPEED         EQU (0x1 <<  0) ;- (UDPHS) Speed Status
// - -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- 
// - -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- 
AT91C_UDPHS_RST_EPT_0     EQU (0x1 <<  0) ;- (UDPHS) Endpoint Reset 0
AT91C_UDPHS_RST_EPT_1     EQU (0x1 <<  1) ;- (UDPHS) Endpoint Reset 1
AT91C_UDPHS_RST_EPT_2     EQU (0x1 <<  2) ;- (UDPHS) Endpoint Reset 2
AT91C_UDPHS_RST_EPT_3     EQU (0x1 <<  3) ;- (UDPHS) Endpoint Reset 3
AT91C_UDPHS_RST_EPT_4     EQU (0x1 <<  4) ;- (UDPHS) Endpoint Reset 4
AT91C_UDPHS_RST_EPT_5     EQU (0x1 <<  5) ;- (UDPHS) Endpoint Reset 5
AT91C_UDPHS_RST_EPT_6     EQU (0x1 <<  6) ;- (UDPHS) Endpoint Reset 6
// - -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- 
AT91C_UDPHS_SOFCNTMAX     EQU (0x3 <<  0) ;- (UDPHS) SOF Counter Max Value
AT91C_UDPHS_SOFCTLOAD     EQU (0x1 <<  7) ;- (UDPHS) SOF Counter Load
// - -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- 
AT91C_UDPHS_CNTAMAX       EQU (0x7FFF <<  0) ;- (UDPHS) A Counter Max Value
AT91C_UDPHS_CNTALOAD      EQU (0x1 << 15) ;- (UDPHS) A Counter Load
// - -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- 
AT91C_UDPHS_CNTBMAX       EQU (0x7FFF <<  0) ;- (UDPHS) B Counter Max Value
AT91C_UDPHS_CNTBLOAD      EQU (0x1 << 15) ;- (UDPHS) B Counter Load
// - -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- 
AT91C_UDPHS_TSTMODE       EQU (0x1F <<  1) ;- (UDPHS) UDPHS Core TestModeReg
// - -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- 
AT91C_UDPHS_SPEED_CFG     EQU (0x3 <<  0) ;- (UDPHS) Speed Configuration
AT91C_UDPHS_SPEED_CFG_NM  EQU (0x0) ;- (UDPHS) Normal Mode
AT91C_UDPHS_SPEED_CFG_RS  EQU (0x1) ;- (UDPHS) Reserved
AT91C_UDPHS_SPEED_CFG_HS  EQU (0x2) ;- (UDPHS) Force High Speed
AT91C_UDPHS_SPEED_CFG_FS  EQU (0x3) ;- (UDPHS) Force Full-Speed
AT91C_UDPHS_TST_J         EQU (0x1 <<  2) ;- (UDPHS) TestJMode
AT91C_UDPHS_TST_K         EQU (0x1 <<  3) ;- (UDPHS) TestKMode
AT91C_UDPHS_TST_PKT       EQU (0x1 <<  4) ;- (UDPHS) TestPacketMode
AT91C_UDPHS_OPMODE2       EQU (0x1 <<  5) ;- (UDPHS) OpMode2
// - -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- 
AT91C_UDPHS_IPPADDRSIZE   EQU (0x0 <<  0) ;- (UDPHS) 2^UDPHSDEV_PADDR_SIZE
// - -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- 
AT91C_UDPHS_IPNAME1       EQU (0x0 <<  0) ;- (UDPHS) ASCII string HUSB
// - -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- 
AT91C_UDPHS_IPNAME2       EQU (0x0 <<  0) ;- (UDPHS) ASCII string 2DEV
// - -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- 
AT91C_UDPHS_EPT_NBR_MAX   EQU (0xF <<  0) ;- (UDPHS) Max Number of Endpoints
AT91C_UDPHS_DMA_CHANNEL_NBR EQU (0x7 <<  4) ;- (UDPHS) Number of DMA Channels
AT91C_UDPHS_DMA_B_SIZ     EQU (0x1 <<  7) ;- (UDPHS) DMA Buffer Size
AT91C_UDPHS_DMA_FIFO_WORD_DEPTH EQU (0xF <<  8) ;- (UDPHS) DMA FIFO Depth in words
AT91C_UDPHS_FIFO_MAX_SIZE EQU (0x7 << 12) ;- (UDPHS) DPRAM size
AT91C_UDPHS_BW_DPRAM      EQU (0x1 << 15) ;- (UDPHS) DPRAM byte write capability
AT91C_UDPHS_DATAB16_8     EQU (0x1 << 16) ;- (UDPHS) UTMI DataBus16_8
AT91C_UDPHS_ISO_EPT_1     EQU (0x1 << 17) ;- (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_2     EQU (0x1 << 18) ;- (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_5     EQU (0x1 << 21) ;- (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_6     EQU (0x1 << 22) ;- (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability
// - -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- 
AT91C_UDPHS_VERSION_NUM   EQU (0xFFFF <<  0) ;- (UDPHS) Give the IP version
AT91C_UDPHS_METAL_FIX_NUM EQU (0x7 << 16) ;- (UDPHS) Give the number of metal fixes

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR HDMA Channel structure
// - *****************************************************************************
// - -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- 
AT91C_SADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
// - -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- 
AT91C_DADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
// - -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- 
AT91C_HDMA_DSCR           EQU (0x3FFFFFFF <<  2) ;- (HDMA_CH) Buffer Transfer descriptor address. This address is word aligned.
// - -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- 
AT91C_HDMA_BTSIZE         EQU (0xFFFF <<  0) ;- (HDMA_CH) Buffer Transfer Size.
AT91C_HDMA_SCSIZE         EQU (0x1 << 16) ;- (HDMA_CH) Source Chunk Transfer Size.
AT91C_HDMA_SCSIZE_1       EQU (0x0 << 16) ;- (HDMA_CH) 1.
AT91C_HDMA_SCSIZE_4       EQU (0x1 << 16) ;- (HDMA_CH) 4.
AT91C_HDMA_DCSIZE         EQU (0x1 << 20) ;- (HDMA_CH) Destination Chunk Transfer Size
AT91C_HDMA_DCSIZE_1       EQU (0x0 << 20) ;- (HDMA_CH) 1.
AT91C_HDMA_DCSIZE_4       EQU (0x1 << 20) ;- (HDMA_CH) 4.
AT91C_HDMA_SRC_WIDTH      EQU (0x3 << 24) ;- (HDMA_CH) Source Single Transfer Size
AT91C_HDMA_SRC_WIDTH_BYTE EQU (0x0 << 24) ;- (HDMA_CH) BYTE.
AT91C_HDMA_SRC_WIDTH_HALFWORD EQU (0x1 << 24) ;- (HDMA_CH) HALF-WORD.
AT91C_HDMA_SRC_WIDTH_WORD EQU (0x2 << 24) ;- (HDMA_CH) WORD.
AT91C_HDMA_DST_WIDTH      EQU (0x3 << 28) ;- (HDMA_CH) Destination Single Transfer Size
AT91C_HDMA_DST_WIDTH_BYTE EQU (0x0 << 28) ;- (HDMA_CH) BYTE.
AT91C_HDMA_DST_WIDTH_HALFWORD EQU (0x1 << 28) ;- (HDMA_CH) HALF-WORD.
AT91C_HDMA_DST_WIDTH_WORD EQU (0x2 << 28) ;- (HDMA_CH) WORD.
AT91C_HDMA_DONE           EQU (0x1 << 31) ;- (HDMA_CH) 
// - -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- 
AT91C_HDMA_SRC_DSCR       EQU (0x1 << 16) ;- (HDMA_CH) Source Buffer Descriptor Fetch operation
AT91C_HDMA_SRC_DSCR_FETCH_FROM_MEM EQU (0x0 << 16) ;- (HDMA_CH) Source address is updated when the descriptor is fetched from the memory.
AT91C_HDMA_SRC_DSCR_FETCH_DISABLE EQU (0x1 << 16) ;- (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the Source.
AT91C_HDMA_DST_DSCR       EQU (0x1 << 20) ;- (HDMA_CH) Destination Buffer Descriptor operation
AT91C_HDMA_DST_DSCR_FETCH_FROM_MEM EQU (0x0 << 20) ;- (HDMA_CH) Destination address is updated when the descriptor is fetched from the memory.
AT91C_HDMA_DST_DSCR_FETCH_DISABLE EQU (0x1 << 20) ;- (HDMA_CH) Buffer Descriptor Fetch operation is disabled for the destination.
AT91C_HDMA_FC             EQU (0x7 << 21) ;- (HDMA_CH) This field defines which devices controls the size of the buffer transfer, also referred as to the Flow Controller.
AT91C_HDMA_FC_MEM2MEM     EQU (0x0 << 21) ;- (HDMA_CH) Memory-to-Memory (DMA Controller).
AT91C_HDMA_FC_MEM2PER     EQU (0x1 << 21) ;- (HDMA_CH) Memory-to-Peripheral (DMA Controller).
AT91C_HDMA_FC_PER2MEM     EQU (0x2 << 21) ;- (HDMA_CH) Peripheral-to-Memory (DMA Controller).
AT91C_HDMA_FC_PER2PER     EQU (0x3 << 21) ;- (HDMA_CH) Peripheral-to-Peripheral (DMA Controller).
AT91C_HDMA_SRC_ADDRESS_MODE EQU (0x3 << 24) ;- (HDMA_CH) Type of addressing mode
AT91C_HDMA_SRC_ADDRESS_MODE_INCR EQU (0x0 << 24) ;- (HDMA_CH) Incrementing Mode.
AT91C_HDMA_SRC_ADDRESS_MODE_DECR EQU (0x1 << 24) ;- (HDMA_CH) Decrementing Mode.
AT91C_HDMA_SRC_ADDRESS_MODE_FIXED EQU (0x2 << 24) ;- (HDMA_CH) Fixed Mode.
AT91C_HDMA_DST_ADDRESS_MODE EQU (0x3 << 28) ;- (HDMA_CH) Type of addressing mode
AT91C_HDMA_DST_ADDRESS_MODE_INCR EQU (0x0 << 28) ;- (HDMA_CH) Incrementing Mode.
AT91C_HDMA_DST_ADDRESS_MODE_DECR EQU (0x1 << 28) ;- (HDMA_CH) Decrementing Mode.
AT91C_HDMA_DST_ADDRESS_MODE_FIXED EQU (0x2 << 28) ;- (HDMA_CH) Fixed Mode.
AT91C_HDMA_IEN            EQU (0x1 << 30) ;- (HDMA_CH) buffer transfer completed
// - -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- 
AT91C_HDMA_SRC_PER        EQU (0xF <<  0) ;- (HDMA_CH) Channel Source Request is associated with peripheral identifier coded SRC_PER handshaking interface.
AT91C_HDMA_SRC_PER_0      EQU (0x0) ;- (HDMA_CH) HW Handshaking Interface number 0.
AT91C_HDMA_SRC_PER_1      EQU (0x1) ;- (HDMA_CH) HW Handshaking Interface number 1.
AT91C_HDMA_SRC_PER_2      EQU (0x2) ;- (HDMA_CH) HW Handshaking Interface number 2.
AT91C_HDMA_SRC_PER_3      EQU (0x3) ;- (HDMA_CH) HW Handshaking Interface number 3.
AT91C_HDMA_SRC_PER_4      EQU (0x4) ;- (HDMA_CH) HW Handshaking Interface number 4.
AT91C_HDMA_SRC_PER_5      EQU (0x5) ;- (HDMA_CH) HW Handshaking Interface number 5.
AT91C_HDMA_SRC_PER_6      EQU (0x6) ;- (HDMA_CH) HW Handshaking Interface number 6.
AT91C_HDMA_SRC_PER_7      EQU (0x7) ;- (HDMA_CH) HW Handshaking Interface number 7.
AT91C_HDMA_DST_PER        EQU (0xF <<  4) ;- (HDMA_CH) Channel Destination Request is associated with peripheral identifier coded DST_PER handshaking interface.
AT91C_HDMA_DST_PER_0      EQU (0x0 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 0.
AT91C_HDMA_DST_PER_1      EQU (0x1 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 1.
AT91C_HDMA_DST_PER_2      EQU (0x2 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 2.
AT91C_HDMA_DST_PER_3      EQU (0x3 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 3.
AT91C_HDMA_DST_PER_4      EQU (0x4 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 4.
AT91C_HDMA_DST_PER_5      EQU (0x5 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 5.
AT91C_HDMA_DST_PER_6      EQU (0x6 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 6.
AT91C_HDMA_DST_PER_7      EQU (0x7 <<  4) ;- (HDMA_CH) HW Handshaking Interface number 7.
AT91C_HDMA_SRC_H2SEL      EQU (0x1 <<  9) ;- (HDMA_CH) Source Handshaking Mode
AT91C_HDMA_SRC_H2SEL_SW   EQU (0x0 <<  9) ;- (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
AT91C_HDMA_SRC_H2SEL_HW   EQU (0x1 <<  9) ;- (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
AT91C_HDMA_DST_H2SEL      EQU (0x1 << 13) ;- (HDMA_CH) Destination Handshaking Mode
AT91C_HDMA_DST_H2SEL_SW   EQU (0x0 << 13) ;- (HDMA_CH) Software handshaking interface is used to trigger a transfer request.
AT91C_HDMA_DST_H2SEL_HW   EQU (0x1 << 13) ;- (HDMA_CH) Hardware handshaking interface is used to trigger a transfer request.
AT91C_HDMA_SOD            EQU (0x1 << 16) ;- (HDMA_CH) STOP ON DONE
AT91C_HDMA_SOD_DISABLE    EQU (0x0 << 16) ;- (HDMA_CH) STOP ON DONE disabled, the descriptor fetch operation ignores DONE Field of CTRLA register.
AT91C_HDMA_SOD_ENABLE     EQU (0x1 << 16) ;- (HDMA_CH) STOP ON DONE activated, the DMAC module is automatically disabled if DONE FIELD is set to 1.
AT91C_HDMA_LOCK_IF        EQU (0x1 << 20) ;- (HDMA_CH) Interface Lock
AT91C_HDMA_LOCK_IF_DISABLE EQU (0x0 << 20) ;- (HDMA_CH) Interface Lock capability is disabled.
AT91C_HDMA_LOCK_IF_ENABLE EQU (0x1 << 20) ;- (HDMA_CH) Interface Lock capability is enabled.
AT91C_HDMA_LOCK_B         EQU (0x1 << 21) ;- (HDMA_CH) AHB Bus Lock
AT91C_HDMA_LOCK_B_DISABLE EQU (0x0 << 21) ;- (HDMA_CH) AHB Bus Locking capability is disabled.
AT91C_HDMA_LOCK_B_ENABLE  EQU (0x1 << 21) ;- (HDMA_CH) AHB Bus Locking capability is enabled.
AT91C_HDMA_LOCK_IF_L      EQU (0x1 << 22) ;- (HDMA_CH) Master Interface Arbiter Lock
AT91C_HDMA_LOCK_IF_L_CHUNK EQU (0x0 << 22) ;- (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a chunk transfer.
AT91C_HDMA_LOCK_IF_L_BUFFER EQU (0x1 << 22) ;- (HDMA_CH) The Master Interface Arbiter is locked by the channel x for a buffer transfer.
AT91C_HDMA_AHB_PROT       EQU (0x7 << 24) ;- (HDMA_CH) AHB Prot
AT91C_HDMA_FIFOCFG        EQU (0x3 << 28) ;- (HDMA_CH) FIFO Request Configuration
AT91C_HDMA_FIFOCFG_LARGESTBURST EQU (0x0 << 28) ;- (HDMA_CH) The largest defined length AHB burst is performed on the destination AHB interface.
AT91C_HDMA_FIFOCFG_HALFFIFO EQU (0x1 << 28) ;- (HDMA_CH) When half fifo size is available/filled a source/destination request is serviced.
AT91C_HDMA_FIFOCFG_ENOUGHSPACE EQU (0x2 << 28) ;- (HDMA_CH) When there is enough space/data available to perfom a single AHB access then the request is serviced.

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR HDMA controller
// - *****************************************************************************
// - -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- 
AT91C_HDMA_ARB_CFG        EQU (0x1 <<  4) ;- (HDMA) Arbiter mode.
AT91C_HDMA_ARB_CFG_FIXED  EQU (0x0 <<  4) ;- (HDMA) Fixed priority arbiter.
AT91C_HDMA_ARB_CFG_ROUND_ROBIN EQU (0x1 <<  4) ;- (HDMA) Modified round robin arbiter.
// - -------- HDMA_EN : (HDMA Offset: 0x4)  -------- 
AT91C_HDMA_ENABLE         EQU (0x1 <<  0) ;- (HDMA) 
AT91C_HDMA_ENABLE_DISABLE EQU (0x0) ;- (HDMA) Disables HDMA.
AT91C_HDMA_ENABLE_ENABLE  EQU (0x1) ;- (HDMA) Enables HDMA.
// - -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- 
AT91C_HDMA_SSREQ0         EQU (0x1 <<  0) ;- (HDMA) Request a source single transfer on channel 0
AT91C_HDMA_SSREQ0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ0_1       EQU (0x1) ;- (HDMA) Request a source single transfer on channel 0.
AT91C_HDMA_DSREQ0         EQU (0x1 <<  1) ;- (HDMA) Request a destination single transfer on channel 0
AT91C_HDMA_DSREQ0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ0_1       EQU (0x1 <<  1) ;- (HDMA) Request a destination single transfer on channel 0.
AT91C_HDMA_SSREQ1         EQU (0x1 <<  2) ;- (HDMA) Request a source single transfer on channel 1
AT91C_HDMA_SSREQ1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ1_1       EQU (0x1 <<  2) ;- (HDMA) Request a source single transfer on channel 1.
AT91C_HDMA_DSREQ1         EQU (0x1 <<  3) ;- (HDMA) Request a destination single transfer on channel 1
AT91C_HDMA_DSREQ1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ1_1       EQU (0x1 <<  3) ;- (HDMA) Request a destination single transfer on channel 1.
AT91C_HDMA_SSREQ2         EQU (0x1 <<  4) ;- (HDMA) Request a source single transfer on channel 2
AT91C_HDMA_SSREQ2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ2_1       EQU (0x1 <<  4) ;- (HDMA) Request a source single transfer on channel 2.
AT91C_HDMA_DSREQ2         EQU (0x1 <<  5) ;- (HDMA) Request a destination single transfer on channel 2
AT91C_HDMA_DSREQ2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ2_1       EQU (0x1 <<  5) ;- (HDMA) Request a destination single transfer on channel 2.
AT91C_HDMA_SSREQ3         EQU (0x1 <<  6) ;- (HDMA) Request a source single transfer on channel 3
AT91C_HDMA_SSREQ3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SSREQ3_1       EQU (0x1 <<  6) ;- (HDMA) Request a source single transfer on channel 3.
AT91C_HDMA_DSREQ3         EQU (0x1 <<  7) ;- (HDMA) Request a destination single transfer on channel 3
AT91C_HDMA_DSREQ3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DSREQ3_1       EQU (0x1 <<  7) ;- (HDMA) Request a destination single transfer on channel 3.
// - -------- HDMA_CREQ : (HDMA Offset: 0xc)  -------- 
AT91C_HDMA_SCREQ0         EQU (0x1 <<  0) ;- (HDMA) Request a source chunk transfer on channel 0
AT91C_HDMA_SCREQ0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ0_1       EQU (0x1) ;- (HDMA) Request a source chunk transfer on channel 0.
AT91C_HDMA_DCREQ0         EQU (0x1 <<  1) ;- (HDMA) Request a destination chunk transfer on channel 0
AT91C_HDMA_DCREQ0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ0_1       EQU (0x1 <<  1) ;- (HDMA) Request a destination chunk transfer on channel 0.
AT91C_HDMA_SCREQ1         EQU (0x1 <<  2) ;- (HDMA) Request a source chunk transfer on channel 1
AT91C_HDMA_SCREQ1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ1_1       EQU (0x1 <<  2) ;- (HDMA) Request a source chunk transfer on channel 1.
AT91C_HDMA_DCREQ1         EQU (0x1 <<  3) ;- (HDMA) Request a destination chunk transfer on channel 1
AT91C_HDMA_DCREQ1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ1_1       EQU (0x1 <<  3) ;- (HDMA) Request a destination chunk transfer on channel 1.
AT91C_HDMA_SCREQ2         EQU (0x1 <<  4) ;- (HDMA) Request a source chunk transfer on channel 2
AT91C_HDMA_SCREQ2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ2_1       EQU (0x1 <<  4) ;- (HDMA) Request a source chunk transfer on channel 2.
AT91C_HDMA_DCREQ2         EQU (0x1 <<  5) ;- (HDMA) Request a destination chunk transfer on channel 2
AT91C_HDMA_DCREQ2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ2_1       EQU (0x1 <<  5) ;- (HDMA) Request a destination chunk transfer on channel 2.
AT91C_HDMA_SCREQ3         EQU (0x1 <<  6) ;- (HDMA) Request a source chunk transfer on channel 3
AT91C_HDMA_SCREQ3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SCREQ3_1       EQU (0x1 <<  6) ;- (HDMA) Request a source chunk transfer on channel 3.
AT91C_HDMA_DCREQ3         EQU (0x1 <<  7) ;- (HDMA) Request a destination chunk transfer on channel 3
AT91C_HDMA_DCREQ3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DCREQ3_1       EQU (0x1 <<  7) ;- (HDMA) Request a destination chunk transfer on channel 3.
// - -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- 
AT91C_HDMA_SLAST0         EQU (0x1 <<  0) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 0
AT91C_HDMA_SLAST0_0       EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_SLAST0_1       EQU (0x1) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 0.
AT91C_HDMA_DLAST0         EQU (0x1 <<  1) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 0
AT91C_HDMA_DLAST0_0       EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DLAST0_1       EQU (0x1 <<  1) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 0.
AT91C_HDMA_SLAST1         EQU (0x1 <<  2) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 1
AT91C_HDMA_SLAST1_0       EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_SLAST1_1       EQU (0x1 <<  2) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 1.
AT91C_HDMA_DLAST1         EQU (0x1 <<  3) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 1
AT91C_HDMA_DLAST1_0       EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DLAST1_1       EQU (0x1 <<  3) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 1.
AT91C_HDMA_SLAST2         EQU (0x1 <<  4) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 2
AT91C_HDMA_SLAST2_0       EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_SLAST2_1       EQU (0x1 <<  4) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 2.
AT91C_HDMA_DLAST2         EQU (0x1 <<  5) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 2
AT91C_HDMA_DLAST2_0       EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DLAST2_1       EQU (0x1 <<  5) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 2.
AT91C_HDMA_SLAST3         EQU (0x1 <<  6) ;- (HDMA) Indicates that this source request is the last transfer of the buffer on channel 3
AT91C_HDMA_SLAST3_0       EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_SLAST3_1       EQU (0x1 <<  6) ;- (HDMA) Writing one to SLASTx prior to writing one to SSREQx or SCREQx indicates that this source request is the last transfer of the buffer on channel 3.
AT91C_HDMA_DLAST3         EQU (0x1 <<  7) ;- (HDMA) Indicates that this destination request is the last transfer of the buffer on channel 3
AT91C_HDMA_DLAST3_0       EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DLAST3_1       EQU (0x1 <<  7) ;- (HDMA) Writing one to DLASTx prior to writing one to DSREQx or DCREQx indicates that this destination request is the last transfer of the buffer on channel 3.
// - -------- HDMA_EBCIER : (HDMA Offset: 0x18) Buffer Transfer Completed/Chained Buffer Transfer Completed/Access Error Interrupt Enable Register -------- 
AT91C_HDMA_BTC0           EQU (0x1 <<  0) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC1           EQU (0x1 <<  1) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC2           EQU (0x1 <<  2) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC3           EQU (0x1 <<  3) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC4           EQU (0x1 <<  4) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC5           EQU (0x1 <<  5) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC6           EQU (0x1 <<  6) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_BTC7           EQU (0x1 <<  7) ;- (HDMA) Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC0          EQU (0x1 <<  8) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC1          EQU (0x1 <<  9) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC2          EQU (0x1 << 10) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC3          EQU (0x1 << 11) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC4          EQU (0x1 << 12) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC5          EQU (0x1 << 13) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC6          EQU (0x1 << 14) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_CBTC7          EQU (0x1 << 15) ;- (HDMA) Chained Buffer Transfer Completed Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR0           EQU (0x1 << 16) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR1           EQU (0x1 << 17) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR2           EQU (0x1 << 18) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR3           EQU (0x1 << 19) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR4           EQU (0x1 << 20) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR5           EQU (0x1 << 21) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR6           EQU (0x1 << 22) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
AT91C_HDMA_ERR7           EQU (0x1 << 23) ;- (HDMA) Access HDMA_Error Interrupt Enable/Disable/Status Register
// - -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- 
// - -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- 
// - -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- 
// - -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- 
AT91C_HDMA_ENA0           EQU (0x1 <<  0) ;- (HDMA) When set, channel 0 enabled.
AT91C_HDMA_ENA0_0         EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_ENA0_1         EQU (0x1) ;- (HDMA) Channel 0 enabled.
AT91C_HDMA_ENA1           EQU (0x1 <<  1) ;- (HDMA) When set, channel 1 enabled.
AT91C_HDMA_ENA1_0         EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_ENA1_1         EQU (0x1 <<  1) ;- (HDMA) Channel 1 enabled.
AT91C_HDMA_ENA2           EQU (0x1 <<  2) ;- (HDMA) When set, channel 2 enabled.
AT91C_HDMA_ENA2_0         EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_ENA2_1         EQU (0x1 <<  2) ;- (HDMA) Channel 2 enabled.
AT91C_HDMA_ENA3           EQU (0x1 <<  3) ;- (HDMA) When set, channel 3 enabled.
AT91C_HDMA_ENA3_0         EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_ENA3_1         EQU (0x1 <<  3) ;- (HDMA) Channel 3 enabled.
AT91C_HDMA_ENA4           EQU (0x1 <<  4) ;- (HDMA) When set, channel 4 enabled.
AT91C_HDMA_ENA4_0         EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_ENA4_1         EQU (0x1 <<  4) ;- (HDMA) Channel 4 enabled.
AT91C_HDMA_ENA5           EQU (0x1 <<  5) ;- (HDMA) When set, channel 5 enabled.
AT91C_HDMA_ENA5_0         EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_ENA5_1         EQU (0x1 <<  5) ;- (HDMA) Channel 5 enabled.
AT91C_HDMA_ENA6           EQU (0x1 <<  6) ;- (HDMA) When set, channel 6 enabled.
AT91C_HDMA_ENA6_0         EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_ENA6_1         EQU (0x1 <<  6) ;- (HDMA) Channel 6 enabled.
AT91C_HDMA_ENA7           EQU (0x1 <<  7) ;- (HDMA) When set, channel 7 enabled.
AT91C_HDMA_ENA7_0         EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_ENA7_1         EQU (0x1 <<  7) ;- (HDMA) Channel 7 enabled.
AT91C_HDMA_SUSP0          EQU (0x1 <<  8) ;- (HDMA) When set, channel 0 freezed and its current context.
AT91C_HDMA_SUSP0_0        EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_SUSP0_1        EQU (0x1 <<  8) ;- (HDMA) Channel 0 freezed.
AT91C_HDMA_SUSP1          EQU (0x1 <<  9) ;- (HDMA) When set, channel 1 freezed and its current context.
AT91C_HDMA_SUSP1_0        EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_SUSP1_1        EQU (0x1 <<  9) ;- (HDMA) Channel 1 freezed.
AT91C_HDMA_SUSP2          EQU (0x1 << 10) ;- (HDMA) When set, channel 2 freezed and its current context.
AT91C_HDMA_SUSP2_0        EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_SUSP2_1        EQU (0x1 << 10) ;- (HDMA) Channel 2 freezed.
AT91C_HDMA_SUSP3          EQU (0x1 << 11) ;- (HDMA) When set, channel 3 freezed and its current context.
AT91C_HDMA_SUSP3_0        EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_SUSP3_1        EQU (0x1 << 11) ;- (HDMA) Channel 3 freezed.
AT91C_HDMA_SUSP4          EQU (0x1 << 12) ;- (HDMA) When set, channel 4 freezed and its current context.
AT91C_HDMA_SUSP4_0        EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_SUSP4_1        EQU (0x1 << 12) ;- (HDMA) Channel 4 freezed.
AT91C_HDMA_SUSP5          EQU (0x1 << 13) ;- (HDMA) When set, channel 5 freezed and its current context.
AT91C_HDMA_SUSP5_0        EQU (0x0 << 13) ;- (HDMA) No effect.
AT91C_HDMA_SUSP5_1        EQU (0x1 << 13) ;- (HDMA) Channel 5 freezed.
AT91C_HDMA_SUSP6          EQU (0x1 << 14) ;- (HDMA) When set, channel 6 freezed and its current context.
AT91C_HDMA_SUSP6_0        EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_SUSP6_1        EQU (0x1 << 14) ;- (HDMA) Channel 6 freezed.
AT91C_HDMA_SUSP7          EQU (0x1 << 15) ;- (HDMA) When set, channel 7 freezed and its current context.
AT91C_HDMA_SUSP7_0        EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_SUSP7_1        EQU (0x1 << 15) ;- (HDMA) Channel 7 freezed.
AT91C_HDMA_KEEP0          EQU (0x1 << 24) ;- (HDMA) When set, it resumes the channel 0 from an automatic stall state.
AT91C_HDMA_KEEP0_0        EQU (0x0 << 24) ;- (HDMA) No effect.
AT91C_HDMA_KEEP0_1        EQU (0x1 << 24) ;- (HDMA) Resumes the channel 0.
AT91C_HDMA_KEEP1          EQU (0x1 << 25) ;- (HDMA) When set, it resumes the channel 1 from an automatic stall state.
AT91C_HDMA_KEEP1_0        EQU (0x0 << 25) ;- (HDMA) No effect.
AT91C_HDMA_KEEP1_1        EQU (0x1 << 25) ;- (HDMA) Resumes the channel 1.
AT91C_HDMA_KEEP2          EQU (0x1 << 26) ;- (HDMA) When set, it resumes the channel 2 from an automatic stall state.
AT91C_HDMA_KEEP2_0        EQU (0x0 << 26) ;- (HDMA) No effect.
AT91C_HDMA_KEEP2_1        EQU (0x1 << 26) ;- (HDMA) Resumes the channel 2.
AT91C_HDMA_KEEP3          EQU (0x1 << 27) ;- (HDMA) When set, it resumes the channel 3 from an automatic stall state.
AT91C_HDMA_KEEP3_0        EQU (0x0 << 27) ;- (HDMA) No effect.
AT91C_HDMA_KEEP3_1        EQU (0x1 << 27) ;- (HDMA) Resumes the channel 3.
AT91C_HDMA_KEEP4          EQU (0x1 << 28) ;- (HDMA) When set, it resumes the channel 4 from an automatic stall state.
AT91C_HDMA_KEEP4_0        EQU (0x0 << 28) ;- (HDMA) No effect.
AT91C_HDMA_KEEP4_1        EQU (0x1 << 28) ;- (HDMA) Resumes the channel 4.
AT91C_HDMA_KEEP5          EQU (0x1 << 29) ;- (HDMA) When set, it resumes the channel 5 from an automatic stall state.
AT91C_HDMA_KEEP5_0        EQU (0x0 << 29) ;- (HDMA) No effect.
AT91C_HDMA_KEEP5_1        EQU (0x1 << 29) ;- (HDMA) Resumes the channel 5.
AT91C_HDMA_KEEP6          EQU (0x1 << 30) ;- (HDMA) When set, it resumes the channel 6 from an automatic stall state.
AT91C_HDMA_KEEP6_0        EQU (0x0 << 30) ;- (HDMA) No effect.
AT91C_HDMA_KEEP6_1        EQU (0x1 << 30) ;- (HDMA) Resumes the channel 6.
AT91C_HDMA_KEEP7          EQU (0x1 << 31) ;- (HDMA) When set, it resumes the channel 7 from an automatic stall state.
AT91C_HDMA_KEEP7_0        EQU (0x0 << 31) ;- (HDMA) No effect.
AT91C_HDMA_KEEP7_1        EQU (0x1 << 31) ;- (HDMA) Resumes the channel 7.
// - -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- 
AT91C_HDMA_DIS0           EQU (0x1 <<  0) ;- (HDMA) Write one to this field to disable the channel 0.
AT91C_HDMA_DIS0_0         EQU (0x0) ;- (HDMA) No effect.
AT91C_HDMA_DIS0_1         EQU (0x1) ;- (HDMA) Disables the channel 0.
AT91C_HDMA_DIS1           EQU (0x1 <<  1) ;- (HDMA) Write one to this field to disable the channel 1.
AT91C_HDMA_DIS1_0         EQU (0x0 <<  1) ;- (HDMA) No effect.
AT91C_HDMA_DIS1_1         EQU (0x1 <<  1) ;- (HDMA) Disables the channel 1.
AT91C_HDMA_DIS2           EQU (0x1 <<  2) ;- (HDMA) Write one to this field to disable the channel 2.
AT91C_HDMA_DIS2_0         EQU (0x0 <<  2) ;- (HDMA) No effect.
AT91C_HDMA_DIS2_1         EQU (0x1 <<  2) ;- (HDMA) Disables the channel 2.
AT91C_HDMA_DIS3           EQU (0x1 <<  3) ;- (HDMA) Write one to this field to disable the channel 3.
AT91C_HDMA_DIS3_0         EQU (0x0 <<  3) ;- (HDMA) No effect.
AT91C_HDMA_DIS3_1         EQU (0x1 <<  3) ;- (HDMA) Disables the channel 3.
AT91C_HDMA_DIS4           EQU (0x1 <<  4) ;- (HDMA) Write one to this field to disable the channel 4.
AT91C_HDMA_DIS4_0         EQU (0x0 <<  4) ;- (HDMA) No effect.
AT91C_HDMA_DIS4_1         EQU (0x1 <<  4) ;- (HDMA) Disables the channel 4.
AT91C_HDMA_DIS5           EQU (0x1 <<  5) ;- (HDMA) Write one to this field to disable the channel 5.
AT91C_HDMA_DIS5_0         EQU (0x0 <<  5) ;- (HDMA) No effect.
AT91C_HDMA_DIS5_1         EQU (0x1 <<  5) ;- (HDMA) Disables the channel 5.
AT91C_HDMA_DIS6           EQU (0x1 <<  6) ;- (HDMA) Write one to this field to disable the channel 6.
AT91C_HDMA_DIS6_0         EQU (0x0 <<  6) ;- (HDMA) No effect.
AT91C_HDMA_DIS6_1         EQU (0x1 <<  6) ;- (HDMA) Disables the channel 6.
AT91C_HDMA_DIS7           EQU (0x1 <<  7) ;- (HDMA) Write one to this field to disable the channel 7.
AT91C_HDMA_DIS7_0         EQU (0x0 <<  7) ;- (HDMA) No effect.
AT91C_HDMA_DIS7_1         EQU (0x1 <<  7) ;- (HDMA) Disables the channel 7.
AT91C_HDMA_RES0           EQU (0x1 <<  8) ;- (HDMA) Write one to this field to resume the channel 0 transfer restoring its context.
AT91C_HDMA_RES0_0         EQU (0x0 <<  8) ;- (HDMA) No effect.
AT91C_HDMA_RES0_1         EQU (0x1 <<  8) ;- (HDMA) Resumes the channel 0.
AT91C_HDMA_RES1           EQU (0x1 <<  9) ;- (HDMA) Write one to this field to resume the channel 1 transfer restoring its context.
AT91C_HDMA_RES1_0         EQU (0x0 <<  9) ;- (HDMA) No effect.
AT91C_HDMA_RES1_1         EQU (0x1 <<  9) ;- (HDMA) Resumes the channel 1.
AT91C_HDMA_RES2           EQU (0x1 << 10) ;- (HDMA) Write one to this field to resume the channel 2 transfer restoring its context.
AT91C_HDMA_RES2_0         EQU (0x0 << 10) ;- (HDMA) No effect.
AT91C_HDMA_RES2_1         EQU (0x1 << 10) ;- (HDMA) Resumes the channel 2.
AT91C_HDMA_RES3           EQU (0x1 << 11) ;- (HDMA) Write one to this field to resume the channel 3 transfer restoring its context.
AT91C_HDMA_RES3_0         EQU (0x0 << 11) ;- (HDMA) No effect.
AT91C_HDMA_RES3_1         EQU (0x1 << 11) ;- (HDMA) Resumes the channel 3.
AT91C_HDMA_RES4           EQU (0x1 << 12) ;- (HDMA) Write one to this field to resume the channel 4 transfer restoring its context.
AT91C_HDMA_RES4_0         EQU (0x0 << 12) ;- (HDMA) No effect.
AT91C_HDMA_RES4_1         EQU (0x1 << 12) ;- (HDMA) Resumes the channel 4.
AT91C_HDMA_RES5           EQU (0x1 << 13) ;- (HDMA) Write one to this field to resume the channel 5 transfer restoring its context.
AT91C_HDMA_RES5_0         EQU (0x0 << 13) ;- (HDMA) No effect.
AT91C_HDMA_RES5_1         EQU (0x1 << 13) ;- (HDMA) Resumes the channel 5.
AT91C_HDMA_RES6           EQU (0x1 << 14) ;- (HDMA) Write one to this field to resume the channel 6 transfer restoring its context.
AT91C_HDMA_RES6_0         EQU (0x0 << 14) ;- (HDMA) No effect.
AT91C_HDMA_RES6_1         EQU (0x1 << 14) ;- (HDMA) Resumes the channel 6.
AT91C_HDMA_RES7           EQU (0x1 << 15) ;- (HDMA) Write one to this field to resume the channel 7 transfer restoring its context.
AT91C_HDMA_RES7_0         EQU (0x0 << 15) ;- (HDMA) No effect.
AT91C_HDMA_RES7_1         EQU (0x1 << 15) ;- (HDMA) Resumes the channel 7.
// - -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- 
AT91C_HDMA_EMPT0          EQU (0x1 << 16) ;- (HDMA) When set, channel 0 is empty.
AT91C_HDMA_EMPT0_0        EQU (0x0 << 16) ;- (HDMA) No effect.
AT91C_HDMA_EMPT0_1        EQU (0x1 << 16) ;- (HDMA) Channel 0 empty.
AT91C_HDMA_EMPT1          EQU (0x1 << 17) ;- (HDMA) When set, channel 1 is empty.
AT91C_HDMA_EMPT1_0        EQU (0x0 << 17) ;- (HDMA) No effect.
AT91C_HDMA_EMPT1_1        EQU (0x1 << 17) ;- (HDMA) Channel 1 empty.
AT91C_HDMA_EMPT2          EQU (0x1 << 18) ;- (HDMA) When set, channel 2 is empty.
AT91C_HDMA_EMPT2_0        EQU (0x0 << 18) ;- (HDMA) No effect.
AT91C_HDMA_EMPT2_1        EQU (0x1 << 18) ;- (HDMA) Channel 2 empty.
AT91C_HDMA_EMPT3          EQU (0x1 << 19) ;- (HDMA) When set, channel 3 is empty.
AT91C_HDMA_EMPT3_0        EQU (0x0 << 19) ;- (HDMA) No effect.
AT91C_HDMA_EMPT3_1        EQU (0x1 << 19) ;- (HDMA) Channel 3 empty.
AT91C_HDMA_EMPT4          EQU (0x1 << 20) ;- (HDMA) When set, channel 4 is empty.
AT91C_HDMA_EMPT4_0        EQU (0x0 << 20) ;- (HDMA) No effect.
AT91C_HDMA_EMPT4_1        EQU (0x1 << 20) ;- (HDMA) Channel 4 empty.
AT91C_HDMA_EMPT5          EQU (0x1 << 21) ;- (HDMA) When set, channel 5 is empty.
AT91C_HDMA_EMPT5_0        EQU (0x0 << 21) ;- (HDMA) No effect.
AT91C_HDMA_EMPT5_1        EQU (0x1 << 21) ;- (HDMA) Channel 5 empty.
AT91C_HDMA_EMPT6          EQU (0x1 << 22) ;- (HDMA) When set, channel 6 is empty.
AT91C_HDMA_EMPT6_0        EQU (0x0 << 22) ;- (HDMA) No effect.
AT91C_HDMA_EMPT6_1        EQU (0x1 << 22) ;- (HDMA) Channel 6 empty.
AT91C_HDMA_EMPT7          EQU (0x1 << 23) ;- (HDMA) When set, channel 7 is empty.
AT91C_HDMA_EMPT7_0        EQU (0x0 << 23) ;- (HDMA) No effect.
AT91C_HDMA_EMPT7_1        EQU (0x1 << 23) ;- (HDMA) Channel 7 empty.
AT91C_HDMA_STAL0          EQU (0x1 << 24) ;- (HDMA) When set, channel 0 is stalled.
AT91C_HDMA_STAL0_0        EQU (0x0 << 24) ;- (HDMA) No effect.
AT91C_HDMA_STAL0_1        EQU (0x1 << 24) ;- (HDMA) Channel 0 stalled.
AT91C_HDMA_STAL1          EQU (0x1 << 25) ;- (HDMA) When set, channel 1 is stalled.
AT91C_HDMA_STAL1_0        EQU (0x0 << 25) ;- (HDMA) No effect.
AT91C_HDMA_STAL1_1        EQU (0x1 << 25) ;- (HDMA) Channel 1 stalled.
AT91C_HDMA_STAL2          EQU (0x1 << 26) ;- (HDMA) When set, channel 2 is stalled.
AT91C_HDMA_STAL2_0        EQU (0x0 << 26) ;- (HDMA) No effect.
AT91C_HDMA_STAL2_1        EQU (0x1 << 26) ;- (HDMA) Channel 2 stalled.
AT91C_HDMA_STAL3          EQU (0x1 << 27) ;- (HDMA) When set, channel 3 is stalled.
AT91C_HDMA_STAL3_0        EQU (0x0 << 27) ;- (HDMA) No effect.
AT91C_HDMA_STAL3_1        EQU (0x1 << 27) ;- (HDMA) Channel 3 stalled.
AT91C_HDMA_STAL4          EQU (0x1 << 28) ;- (HDMA) When set, channel 4 is stalled.
AT91C_HDMA_STAL4_0        EQU (0x0 << 28) ;- (HDMA) No effect.
AT91C_HDMA_STAL4_1        EQU (0x1 << 28) ;- (HDMA) Channel 4 stalled.
AT91C_HDMA_STAL5          EQU (0x1 << 29) ;- (HDMA) When set, channel 5 is stalled.
AT91C_HDMA_STAL5_0        EQU (0x0 << 29) ;- (HDMA) No effect.
AT91C_HDMA_STAL5_1        EQU (0x1 << 29) ;- (HDMA) Channel 5 stalled.
AT91C_HDMA_STAL6          EQU (0x1 << 30) ;- (HDMA) When set, channel 6 is stalled.
AT91C_HDMA_STAL6_0        EQU (0x0 << 30) ;- (HDMA) No effect.
AT91C_HDMA_STAL6_1        EQU (0x1 << 30) ;- (HDMA) Channel 6 stalled.
AT91C_HDMA_STAL7          EQU (0x1 << 31) ;- (HDMA) When set, channel 7 is stalled.
AT91C_HDMA_STAL7_0        EQU (0x0 << 31) ;- (HDMA) No effect.
AT91C_HDMA_STAL7_1        EQU (0x1 << 31) ;- (HDMA) Channel 7 stalled.
// - -------- HDMA_VER : (HDMA Offset: 0x1fc)  -------- 

// - *****************************************************************************
// -               REGISTER ADDRESS DEFINITION FOR AT91SAM3U1
// - *****************************************************************************
// - ========== Register definition for SYS peripheral ========== 
AT91C_SYS_GPBR            EQU (0x400E1290) ;- (SYS) General Purpose Register
// - ========== Register definition for HSMC4_CS0 peripheral ========== 
AT91C_CS0_MODE            EQU (0x400E0080) ;- (HSMC4_CS0) Mode Register
AT91C_CS0_PULSE           EQU (0x400E0074) ;- (HSMC4_CS0) Pulse Register
AT91C_CS0_CYCLE           EQU (0x400E0078) ;- (HSMC4_CS0) Cycle Register
AT91C_CS0_TIMINGS         EQU (0x400E007C) ;- (HSMC4_CS0) Timmings Register
AT91C_CS0_SETUP           EQU (0x400E0070) ;- (HSMC4_CS0) Setup Register
// - ========== Register definition for HSMC4_CS1 peripheral ========== 
AT91C_CS1_CYCLE           EQU (0x400E008C) ;- (HSMC4_CS1) Cycle Register
AT91C_CS1_PULSE           EQU (0x400E0088) ;- (HSMC4_CS1) Pulse Register
AT91C_CS1_MODE            EQU (0x400E0094) ;- (HSMC4_CS1) Mode Register
AT91C_CS1_SETUP           EQU (0x400E0084) ;- (HSMC4_CS1) Setup Register
AT91C_CS1_TIMINGS         EQU (0x400E0090) ;- (HSMC4_CS1) Timmings Register
// - ========== Register definition for HSMC4_CS2 peripheral ========== 
AT91C_CS2_PULSE           EQU (0x400E009C) ;- (HSMC4_CS2) Pulse Register
AT91C_CS2_TIMINGS         EQU (0x400E00A4) ;- (HSMC4_CS2) Timmings Register
AT91C_CS2_CYCLE           EQU (0x400E00A0) ;- (HSMC4_CS2) Cycle Register
AT91C_CS2_MODE            EQU (0x400E00A8) ;- (HSMC4_CS2) Mode Register
AT91C_CS2_SETUP           EQU (0x400E0098) ;- (HSMC4_CS2) Setup Register
// - ========== Register definition for HSMC4_CS3 peripheral ========== 
AT91C_CS3_MODE            EQU (0x400E00BC) ;- (HSMC4_CS3) Mode Register
AT91C_CS3_TIMINGS         EQU (0x400E00B8) ;- (HSMC4_CS3) Timmings Register
AT91C_CS3_SETUP           EQU (0x400E00AC) ;- (HSMC4_CS3) Setup Register
AT91C_CS3_CYCLE           EQU (0x400E00B4) ;- (HSMC4_CS3) Cycle Register
AT91C_CS3_PULSE           EQU (0x400E00B0) ;- (HSMC4_CS3) Pulse Register
// - ========== Register definition for HSMC4_NFC peripheral ========== 
AT91C_NFC_MODE            EQU (0x400E010C) ;- (HSMC4_NFC) Mode Register
AT91C_NFC_CYCLE           EQU (0x400E0104) ;- (HSMC4_NFC) Cycle Register
AT91C_NFC_PULSE           EQU (0x400E0100) ;- (HSMC4_NFC) Pulse Register
AT91C_NFC_SETUP           EQU (0x400E00FC) ;- (HSMC4_NFC) Setup Register
AT91C_NFC_TIMINGS         EQU (0x400E0108) ;- (HSMC4_NFC) Timmings Register
// - ========== Register definition for HSMC4 peripheral ========== 
AT91C_HSMC4_IPNAME1       EQU (0x400E01F0) ;- (HSMC4) Write Protection Status Register
AT91C_HSMC4_ECCPR6        EQU (0x400E0048) ;- (HSMC4) ECC Parity register 6
AT91C_HSMC4_ADDRSIZE      EQU (0x400E01EC) ;- (HSMC4) Write Protection Status Register
AT91C_HSMC4_ECCPR11       EQU (0x400E005C) ;- (HSMC4) ECC Parity register 11
AT91C_HSMC4_SR            EQU (0x400E0008) ;- (HSMC4) Status Register
AT91C_HSMC4_IMR           EQU (0x400E0014) ;- (HSMC4) Interrupt Mask Register
AT91C_HSMC4_WPSR          EQU (0x400E01E8) ;- (HSMC4) Write Protection Status Register
AT91C_HSMC4_BANK          EQU (0x400E001C) ;- (HSMC4) Bank Register
AT91C_HSMC4_ECCPR8        EQU (0x400E0050) ;- (HSMC4) ECC Parity register 8
AT91C_HSMC4_WPCR          EQU (0x400E01E4) ;- (HSMC4) Write Protection Control register
AT91C_HSMC4_ECCPR2        EQU (0x400E0038) ;- (HSMC4) ECC Parity register 2
AT91C_HSMC4_ECCPR1        EQU (0x400E0030) ;- (HSMC4) ECC Parity register 1
AT91C_HSMC4_ECCSR2        EQU (0x400E0034) ;- (HSMC4) ECC Status register 2
AT91C_HSMC4_OCMS          EQU (0x400E0110) ;- (HSMC4) OCMS MODE register
AT91C_HSMC4_ECCPR9        EQU (0x400E0054) ;- (HSMC4) ECC Parity register 9
AT91C_HSMC4_DUMMY         EQU (0x400E0200) ;- (HSMC4) This rtegister was created only ti have AHB constants
AT91C_HSMC4_ECCPR5        EQU (0x400E0044) ;- (HSMC4) ECC Parity register 5
AT91C_HSMC4_ECCCR         EQU (0x400E0020) ;- (HSMC4) ECC reset register
AT91C_HSMC4_KEY2          EQU (0x400E0118) ;- (HSMC4) KEY2 Register
AT91C_HSMC4_IER           EQU (0x400E000C) ;- (HSMC4) Interrupt Enable Register
AT91C_HSMC4_ECCSR1        EQU (0x400E0028) ;- (HSMC4) ECC Status register 1
AT91C_HSMC4_IDR           EQU (0x400E0010) ;- (HSMC4) Interrupt Disable Register
AT91C_HSMC4_ECCPR0        EQU (0x400E002C) ;- (HSMC4) ECC Parity register 0
AT91C_HSMC4_FEATURES      EQU (0x400E01F8) ;- (HSMC4) Write Protection Status Register
AT91C_HSMC4_ECCPR7        EQU (0x400E004C) ;- (HSMC4) ECC Parity register 7
AT91C_HSMC4_ECCPR12       EQU (0x400E0060) ;- (HSMC4) ECC Parity register 12
AT91C_HSMC4_ECCPR10       EQU (0x400E0058) ;- (HSMC4) ECC Parity register 10
AT91C_HSMC4_KEY1          EQU (0x400E0114) ;- (HSMC4) KEY1 Register
AT91C_HSMC4_VER           EQU (0x400E01FC) ;- (HSMC4) HSMC4 Version Register
AT91C_HSMC4_Eccpr15       EQU (0x400E006C) ;- (HSMC4) ECC Parity register 15
AT91C_HSMC4_ECCPR4        EQU (0x400E0040) ;- (HSMC4) ECC Parity register 4
AT91C_HSMC4_IPNAME2       EQU (0x400E01F4) ;- (HSMC4) Write Protection Status Register
AT91C_HSMC4_ECCCMD        EQU (0x400E0024) ;- (HSMC4) ECC Page size register
AT91C_HSMC4_ADDR          EQU (0x400E0018) ;- (HSMC4) Address Cycle Zero Register
AT91C_HSMC4_ECCPR3        EQU (0x400E003C) ;- (HSMC4) ECC Parity register 3
AT91C_HSMC4_CFG           EQU (0x400E0000) ;- (HSMC4) Configuration Register
AT91C_HSMC4_CTRL          EQU (0x400E0004) ;- (HSMC4) Control Register
AT91C_HSMC4_ECCPR13       EQU (0x400E0064) ;- (HSMC4) ECC Parity register 13
AT91C_HSMC4_ECCPR14       EQU (0x400E0068) ;- (HSMC4) ECC Parity register 14
// - ========== Register definition for MATRIX peripheral ========== 
AT91C_MATRIX_SFR2         EQU (0x400E0318) ;- (MATRIX)  Special Function Register 2
AT91C_MATRIX_SFR3         EQU (0x400E031C) ;- (MATRIX)  Special Function Register 3
AT91C_MATRIX_SCFG8        EQU (0x400E0260) ;- (MATRIX)  Slave Configuration Register 8
AT91C_MATRIX_MCFG2        EQU (0x400E0208) ;- (MATRIX)  Master Configuration Register 2
AT91C_MATRIX_MCFG7        EQU (0x400E021C) ;- (MATRIX)  Master Configuration Register 7
AT91C_MATRIX_SCFG3        EQU (0x400E024C) ;- (MATRIX)  Slave Configuration Register 3
AT91C_MATRIX_SCFG0        EQU (0x400E0240) ;- (MATRIX)  Slave Configuration Register 0
AT91C_MATRIX_SFR12        EQU (0x400E0340) ;- (MATRIX)  Special Function Register 12
AT91C_MATRIX_SCFG1        EQU (0x400E0244) ;- (MATRIX)  Slave Configuration Register 1
AT91C_MATRIX_SFR8         EQU (0x400E0330) ;- (MATRIX)  Special Function Register 8
AT91C_MATRIX_VER          EQU (0x400E03FC) ;- (MATRIX) HMATRIX2 VERSION REGISTER 
AT91C_MATRIX_SFR13        EQU (0x400E0344) ;- (MATRIX)  Special Function Register 13
AT91C_MATRIX_SFR5         EQU (0x400E0324) ;- (MATRIX)  Special Function Register 5
AT91C_MATRIX_MCFG0        EQU (0x400E0200) ;- (MATRIX)  Master Configuration Register 0 : ARM I and D
AT91C_MATRIX_SCFG6        EQU (0x400E0258) ;- (MATRIX)  Slave Configuration Register 6
AT91C_MATRIX_SFR14        EQU (0x400E0348) ;- (MATRIX)  Special Function Register 14
AT91C_MATRIX_SFR1         EQU (0x400E0314) ;- (MATRIX)  Special Function Register 1
AT91C_MATRIX_SFR15        EQU (0x400E034C) ;- (MATRIX)  Special Function Register 15
AT91C_MATRIX_SFR6         EQU (0x400E0328) ;- (MATRIX)  Special Function Register 6
AT91C_MATRIX_SFR11        EQU (0x400E033C) ;- (MATRIX)  Special Function Register 11
AT91C_MATRIX_IPNAME2      EQU (0x400E03F4) ;- (MATRIX) HMATRIX2 IPNAME2 REGISTER 
AT91C_MATRIX_ADDRSIZE     EQU (0x400E03EC) ;- (MATRIX) HMATRIX2 ADDRSIZE REGISTER 
AT91C_MATRIX_MCFG5        EQU (0x400E0214) ;- (MATRIX)  Master Configuration Register 5
AT91C_MATRIX_SFR9         EQU (0x400E0334) ;- (MATRIX)  Special Function Register 9
AT91C_MATRIX_MCFG3        EQU (0x400E020C) ;- (MATRIX)  Master Configuration Register 3
AT91C_MATRIX_SCFG4        EQU (0x400E0250) ;- (MATRIX)  Slave Configuration Register 4
AT91C_MATRIX_MCFG1        EQU (0x400E0204) ;- (MATRIX)  Master Configuration Register 1 : ARM S
AT91C_MATRIX_SCFG7        EQU (0x400E025C) ;- (MATRIX)  Slave Configuration Register 5
AT91C_MATRIX_SFR10        EQU (0x400E0338) ;- (MATRIX)  Special Function Register 10
AT91C_MATRIX_SCFG2        EQU (0x400E0248) ;- (MATRIX)  Slave Configuration Register 2
AT91C_MATRIX_SFR7         EQU (0x400E032C) ;- (MATRIX)  Special Function Register 7
AT91C_MATRIX_IPNAME1      EQU (0x400E03F0) ;- (MATRIX) HMATRIX2 IPNAME1 REGISTER 
AT91C_MATRIX_MCFG4        EQU (0x400E0210) ;- (MATRIX)  Master Configuration Register 4
AT91C_MATRIX_SFR0         EQU (0x400E0310) ;- (MATRIX)  Special Function Register 0
AT91C_MATRIX_FEATURES     EQU (0x400E03F8) ;- (MATRIX) HMATRIX2 FEATURES REGISTER 
AT91C_MATRIX_SCFG5        EQU (0x400E0254) ;- (MATRIX)  Slave Configuration Register 5
AT91C_MATRIX_MCFG6        EQU (0x400E0218) ;- (MATRIX)  Master Configuration Register 6
AT91C_MATRIX_SCFG9        EQU (0x400E0264) ;- (MATRIX)  Slave Configuration Register 9
AT91C_MATRIX_SFR4         EQU (0x400E0320) ;- (MATRIX)  Special Function Register 4
// - ========== Register definition for NVIC peripheral ========== 
AT91C_NVIC_MMAR           EQU (0xE000ED34) ;- (NVIC) Mem Manage Address Register
AT91C_NVIC_STIR           EQU (0xE000EF00) ;- (NVIC) Software Trigger Interrupt Register
AT91C_NVIC_MMFR2          EQU (0xE000ED58) ;- (NVIC) Memory Model Feature register2
AT91C_NVIC_CPUID          EQU (0xE000ED00) ;- (NVIC) CPUID Base Register
AT91C_NVIC_DFSR           EQU (0xE000ED30) ;- (NVIC) Debug Fault Status Register
AT91C_NVIC_HAND4PR        EQU (0xE000ED18) ;- (NVIC) System Handlers 4-7 Priority Register
AT91C_NVIC_HFSR           EQU (0xE000ED2C) ;- (NVIC) Hard Fault Status Register
AT91C_NVIC_PID6           EQU (0xE000EFD8) ;- (NVIC) Peripheral identification register
AT91C_NVIC_PFR0           EQU (0xE000ED40) ;- (NVIC) Processor Feature register0
AT91C_NVIC_VTOFFR         EQU (0xE000ED08) ;- (NVIC) Vector Table Offset Register
AT91C_NVIC_ISPR           EQU (0xE000E200) ;- (NVIC) Set Pending Register
AT91C_NVIC_PID0           EQU (0xE000EFE0) ;- (NVIC) Peripheral identification register b7:0
AT91C_NVIC_PID7           EQU (0xE000EFDC) ;- (NVIC) Peripheral identification register
AT91C_NVIC_STICKRVR       EQU (0xE000E014) ;- (NVIC) SysTick Reload Value Register
AT91C_NVIC_PID2           EQU (0xE000EFE8) ;- (NVIC) Peripheral identification register b23:16
AT91C_NVIC_ISAR0          EQU (0xE000ED60) ;- (NVIC) ISA Feature register0
AT91C_NVIC_SCR            EQU (0xE000ED10) ;- (NVIC) System Control Register
AT91C_NVIC_PID4           EQU (0xE000EFD0) ;- (NVIC) Peripheral identification register
AT91C_NVIC_ISAR2          EQU (0xE000ED68) ;- (NVIC) ISA Feature register2
AT91C_NVIC_ISER           EQU (0xE000E100) ;- (NVIC) Set Enable Register
AT91C_NVIC_IPR            EQU (0xE000E400) ;- (NVIC) Interrupt Mask Register
AT91C_NVIC_AIRCR          EQU (0xE000ED0C) ;- (NVIC) Application Interrupt/Reset Control Reg
AT91C_NVIC_CID2           EQU (0xE000EFF8) ;- (NVIC) Component identification register b23:16
AT91C_NVIC_ICPR           EQU (0xE000E280) ;- (NVIC) Clear Pending Register
AT91C_NVIC_CID3           EQU (0xE000EFFC) ;- (NVIC) Component identification register b31:24
AT91C_NVIC_CFSR           EQU (0xE000ED28) ;- (NVIC) Configurable Fault Status Register
AT91C_NVIC_AFR0           EQU (0xE000ED4C) ;- (NVIC) Auxiliary Feature register0
AT91C_NVIC_ICSR           EQU (0xE000ED04) ;- (NVIC) Interrupt Control State Register
AT91C_NVIC_CCR            EQU (0xE000ED14) ;- (NVIC) Configuration Control Register
AT91C_NVIC_CID0           EQU (0xE000EFF0) ;- (NVIC) Component identification register b7:0
AT91C_NVIC_ISAR1          EQU (0xE000ED64) ;- (NVIC) ISA Feature register1
AT91C_NVIC_STICKCVR       EQU (0xE000E018) ;- (NVIC) SysTick Current Value Register
AT91C_NVIC_STICKCSR       EQU (0xE000E010) ;- (NVIC) SysTick Control and Status Register
AT91C_NVIC_CID1           EQU (0xE000EFF4) ;- (NVIC) Component identification register b15:8
AT91C_NVIC_DFR0           EQU (0xE000ED48) ;- (NVIC) Debug Feature register0
AT91C_NVIC_MMFR3          EQU (0xE000ED5C) ;- (NVIC) Memory Model Feature register3
AT91C_NVIC_MMFR0          EQU (0xE000ED50) ;- (NVIC) Memory Model Feature register0
AT91C_NVIC_STICKCALVR     EQU (0xE000E01C) ;- (NVIC) SysTick Calibration Value Register
AT91C_NVIC_PID1           EQU (0xE000EFE4) ;- (NVIC) Peripheral identification register b15:8
AT91C_NVIC_HAND12PR       EQU (0xE000ED20) ;- (NVIC) System Handlers 12-15 Priority Register
AT91C_NVIC_MMFR1          EQU (0xE000ED54) ;- (NVIC) Memory Model Feature register1
AT91C_NVIC_AFSR           EQU (0xE000ED3C) ;- (NVIC) Auxiliary Fault Status Register
AT91C_NVIC_HANDCSR        EQU (0xE000ED24) ;- (NVIC) System Handler Control and State Register
AT91C_NVIC_ISAR4          EQU (0xE000ED70) ;- (NVIC) ISA Feature register4
AT91C_NVIC_ABR            EQU (0xE000E300) ;- (NVIC) Active Bit Register
AT91C_NVIC_PFR1           EQU (0xE000ED44) ;- (NVIC) Processor Feature register1
AT91C_NVIC_PID5           EQU (0xE000EFD4) ;- (NVIC) Peripheral identification register
AT91C_NVIC_ICTR           EQU (0xE000E004) ;- (NVIC) Interrupt Control Type Register
AT91C_NVIC_ICER           EQU (0xE000E180) ;- (NVIC) Clear enable Register
AT91C_NVIC_PID3           EQU (0xE000EFEC) ;- (NVIC) Peripheral identification register b31:24
AT91C_NVIC_ISAR3          EQU (0xE000ED6C) ;- (NVIC) ISA Feature register3
AT91C_NVIC_HAND8PR        EQU (0xE000ED1C) ;- (NVIC) System Handlers 8-11 Priority Register
AT91C_NVIC_BFAR           EQU (0xE000ED38) ;- (NVIC) Bus Fault Address Register
// - ========== Register definition for MPU peripheral ========== 
AT91C_MPU_REG_BASE_ADDR3  EQU (0xE000EDB4) ;- (MPU) MPU Region Base Address Register alias 3
AT91C_MPU_REG_NB          EQU (0xE000ED98) ;- (MPU) MPU Region Number Register
AT91C_MPU_ATTR_SIZE1      EQU (0xE000EDA8) ;- (MPU) MPU  Attribute and Size Register alias 1
AT91C_MPU_REG_BASE_ADDR1  EQU (0xE000EDA4) ;- (MPU) MPU Region Base Address Register alias 1
AT91C_MPU_ATTR_SIZE3      EQU (0xE000EDB8) ;- (MPU) MPU  Attribute and Size Register alias 3
AT91C_MPU_CTRL            EQU (0xE000ED94) ;- (MPU) MPU Control Register
AT91C_MPU_ATTR_SIZE2      EQU (0xE000EDB0) ;- (MPU) MPU  Attribute and Size Register alias 2
AT91C_MPU_REG_BASE_ADDR   EQU (0xE000ED9C) ;- (MPU) MPU Region Base Address Register
AT91C_MPU_REG_BASE_ADDR2  EQU (0xE000EDAC) ;- (MPU) MPU Region Base Address Register alias 2
AT91C_MPU_ATTR_SIZE       EQU (0xE000EDA0) ;- (MPU) MPU  Attribute and Size Register
AT91C_MPU_TYPE            EQU (0xE000ED90) ;- (MPU) MPU Type Register
// - ========== Register definition for CM3 peripheral ========== 
AT91C_CM3_SHCSR           EQU (0xE000ED24) ;- (CM3) System Handler Control and State Register
AT91C_CM3_CCR             EQU (0xE000ED14) ;- (CM3) Configuration Control Register
AT91C_CM3_ICSR            EQU (0xE000ED04) ;- (CM3) Interrupt Control State Register
AT91C_CM3_CPUID           EQU (0xE000ED00) ;- (CM3) CPU ID Base Register
AT91C_CM3_SCR             EQU (0xE000ED10) ;- (CM3) System Controller Register
AT91C_CM3_AIRCR           EQU (0xE000ED0C) ;- (CM3) Application Interrupt and Reset Control Register
AT91C_CM3_SHPR            EQU (0xE000ED18) ;- (CM3) System Handler Priority Register
AT91C_CM3_VTOR            EQU (0xE000ED08) ;- (CM3) Vector Table Offset Register
// - ========== Register definition for PDC_DBGU peripheral ========== 
AT91C_DBGU_TPR            EQU (0x400E0708) ;- (PDC_DBGU) Transmit Pointer Register
AT91C_DBGU_PTCR           EQU (0x400E0720) ;- (PDC_DBGU) PDC Transfer Control Register
AT91C_DBGU_TNCR           EQU (0x400E071C) ;- (PDC_DBGU) Transmit Next Counter Register
AT91C_DBGU_PTSR           EQU (0x400E0724) ;- (PDC_DBGU) PDC Transfer Status Register
AT91C_DBGU_RNCR           EQU (0x400E0714) ;- (PDC_DBGU) Receive Next Counter Register
AT91C_DBGU_RPR            EQU (0x400E0700) ;- (PDC_DBGU) Receive Pointer Register
AT91C_DBGU_TCR            EQU (0x400E070C) ;- (PDC_DBGU) Transmit Counter Register
AT91C_DBGU_RNPR           EQU (0x400E0710) ;- (PDC_DBGU) Receive Next Pointer Register
AT91C_DBGU_TNPR           EQU (0x400E0718) ;- (PDC_DBGU) Transmit Next Pointer Register
AT91C_DBGU_RCR            EQU (0x400E0704) ;- (PDC_DBGU) Receive Counter Register
// - ========== Register definition for DBGU peripheral ========== 
AT91C_DBGU_CR             EQU (0x400E0600) ;- (DBGU) Control Register
AT91C_DBGU_IDR            EQU (0x400E060C) ;- (DBGU) Interrupt Disable Register
AT91C_DBGU_CIDR           EQU (0x400E0740) ;- (DBGU) Chip ID Register
AT91C_DBGU_IPNAME2        EQU (0x400E06F4) ;- (DBGU) DBGU IPNAME2 REGISTER 
AT91C_DBGU_FEATURES       EQU (0x400E06F8) ;- (DBGU) DBGU FEATURES REGISTER 
AT91C_DBGU_FNTR           EQU (0x400E0648) ;- (DBGU) Force NTRST Register
AT91C_DBGU_RHR            EQU (0x400E0618) ;- (DBGU) Receiver Holding Register
AT91C_DBGU_THR            EQU (0x400E061C) ;- (DBGU) Transmitter Holding Register
AT91C_DBGU_ADDRSIZE       EQU (0x400E06EC) ;- (DBGU) DBGU ADDRSIZE REGISTER 
AT91C_DBGU_MR             EQU (0x400E0604) ;- (DBGU) Mode Register
AT91C_DBGU_IER            EQU (0x400E0608) ;- (DBGU) Interrupt Enable Register
AT91C_DBGU_BRGR           EQU (0x400E0620) ;- (DBGU) Baud Rate Generator Register
AT91C_DBGU_CSR            EQU (0x400E0614) ;- (DBGU) Channel Status Register
AT91C_DBGU_VER            EQU (0x400E06FC) ;- (DBGU) DBGU VERSION REGISTER 
AT91C_DBGU_IMR            EQU (0x400E0610) ;- (DBGU) Interrupt Mask Register
AT91C_DBGU_IPNAME1        EQU (0x400E06F0) ;- (DBGU) DBGU IPNAME1 REGISTER 
AT91C_DBGU_EXID           EQU (0x400E0744) ;- (DBGU) Chip ID Extension Register
// - ========== Register definition for PIOA peripheral ========== 
AT91C_PIOA_PDR            EQU (0x400E0C04) ;- (PIOA) PIO Disable Register
AT91C_PIOA_FRLHSR         EQU (0x400E0CD8) ;- (PIOA) Fall/Rise - Low/High Status Register
AT91C_PIOA_KIMR           EQU (0x400E0D38) ;- (PIOA) Keypad Controller Interrupt Mask Register
AT91C_PIOA_LSR            EQU (0x400E0CC4) ;- (PIOA) Level Select Register
AT91C_PIOA_IFSR           EQU (0x400E0C28) ;- (PIOA) Input Filter Status Register
AT91C_PIOA_KKRR           EQU (0x400E0D44) ;- (PIOA) Keypad Controller Key Release Register
AT91C_PIOA_ODR            EQU (0x400E0C14) ;- (PIOA) Output Disable Registerr
AT91C_PIOA_SCIFSR         EQU (0x400E0C80) ;- (PIOA) System Clock Glitch Input Filter Select Register
AT91C_PIOA_PER            EQU (0x400E0C00) ;- (PIOA) PIO Enable Register
AT91C_PIOA_VER            EQU (0x400E0CFC) ;- (PIOA) PIO VERSION REGISTER 
AT91C_PIOA_OWSR           EQU (0x400E0CA8) ;- (PIOA) Output Write Status Register
AT91C_PIOA_KSR            EQU (0x400E0D3C) ;- (PIOA) Keypad Controller Status Register
AT91C_PIOA_IMR            EQU (0x400E0C48) ;- (PIOA) Interrupt Mask Register
AT91C_PIOA_OWDR           EQU (0x400E0CA4) ;- (PIOA) Output Write Disable Register
AT91C_PIOA_MDSR           EQU (0x400E0C58) ;- (PIOA) Multi-driver Status Register
AT91C_PIOA_IFDR           EQU (0x400E0C24) ;- (PIOA) Input Filter Disable Register
AT91C_PIOA_AIMDR          EQU (0x400E0CB4) ;- (PIOA) Additional Interrupt Modes Disables Register
AT91C_PIOA_CODR           EQU (0x400E0C34) ;- (PIOA) Clear Output Data Register
AT91C_PIOA_SCDR           EQU (0x400E0C8C) ;- (PIOA) Slow Clock Divider Debouncing Register
AT91C_PIOA_KIER           EQU (0x400E0D30) ;- (PIOA) Keypad Controller Interrupt Enable Register
AT91C_PIOA_REHLSR         EQU (0x400E0CD4) ;- (PIOA) Rising Edge/ High Level Select Register
AT91C_PIOA_ISR            EQU (0x400E0C4C) ;- (PIOA) Interrupt Status Register
AT91C_PIOA_ESR            EQU (0x400E0CC0) ;- (PIOA) Edge Select Register
AT91C_PIOA_PPUDR          EQU (0x400E0C60) ;- (PIOA) Pull-up Disable Register
AT91C_PIOA_MDDR           EQU (0x400E0C54) ;- (PIOA) Multi-driver Disable Register
AT91C_PIOA_PSR            EQU (0x400E0C08) ;- (PIOA) PIO Status Register
AT91C_PIOA_PDSR           EQU (0x400E0C3C) ;- (PIOA) Pin Data Status Register
AT91C_PIOA_IFDGSR         EQU (0x400E0C88) ;- (PIOA) Glitch or Debouncing Input Filter Clock Selection Status Register
AT91C_PIOA_FELLSR         EQU (0x400E0CD0) ;- (PIOA) Falling Edge/Low Level Select Register
AT91C_PIOA_PPUSR          EQU (0x400E0C68) ;- (PIOA) Pull-up Status Register
AT91C_PIOA_OER            EQU (0x400E0C10) ;- (PIOA) Output Enable Register
AT91C_PIOA_OSR            EQU (0x400E0C18) ;- (PIOA) Output Status Register
AT91C_PIOA_KKPR           EQU (0x400E0D40) ;- (PIOA) Keypad Controller Key Press Register
AT91C_PIOA_AIMMR          EQU (0x400E0CB8) ;- (PIOA) Additional Interrupt Modes Mask Register
AT91C_PIOA_KRCR           EQU (0x400E0D24) ;- (PIOA) Keypad Controller Row Column Register
AT91C_PIOA_IER            EQU (0x400E0C40) ;- (PIOA) Interrupt Enable Register
AT91C_PIOA_KER            EQU (0x400E0D20) ;- (PIOA) Keypad Controller Enable Register
AT91C_PIOA_PPUER          EQU (0x400E0C64) ;- (PIOA) Pull-up Enable Register
AT91C_PIOA_KIDR           EQU (0x400E0D34) ;- (PIOA) Keypad Controller Interrupt Disable Register
AT91C_PIOA_ABSR           EQU (0x400E0C70) ;- (PIOA) Peripheral AB Select Register
AT91C_PIOA_LOCKSR         EQU (0x400E0CE0) ;- (PIOA) Lock Status Register
AT91C_PIOA_DIFSR          EQU (0x400E0C84) ;- (PIOA) Debouncing Input Filter Select Register
AT91C_PIOA_MDER           EQU (0x400E0C50) ;- (PIOA) Multi-driver Enable Register
AT91C_PIOA_AIMER          EQU (0x400E0CB0) ;- (PIOA) Additional Interrupt Modes Enable Register
AT91C_PIOA_ELSR           EQU (0x400E0CC8) ;- (PIOA) Edge/Level Status Register
AT91C_PIOA_IFER           EQU (0x400E0C20) ;- (PIOA) Input Filter Enable Register
AT91C_PIOA_KDR            EQU (0x400E0D28) ;- (PIOA) Keypad Controller Debouncing Register
AT91C_PIOA_IDR            EQU (0x400E0C44) ;- (PIOA) Interrupt Disable Register
AT91C_PIOA_OWER           EQU (0x400E0CA0) ;- (PIOA) Output Write Enable Register
AT91C_PIOA_ODSR           EQU (0x400E0C38) ;- (PIOA) Output Data Status Register
AT91C_PIOA_SODR           EQU (0x400E0C30) ;- (PIOA) Set Output Data Register
// - ========== Register definition for PIOB peripheral ========== 
AT91C_PIOB_KIDR           EQU (0x400E0F34) ;- (PIOB) Keypad Controller Interrupt Disable Register
AT91C_PIOB_OWSR           EQU (0x400E0EA8) ;- (PIOB) Output Write Status Register
AT91C_PIOB_PSR            EQU (0x400E0E08) ;- (PIOB) PIO Status Register
AT91C_PIOB_MDER           EQU (0x400E0E50) ;- (PIOB) Multi-driver Enable Register
AT91C_PIOB_ODR            EQU (0x400E0E14) ;- (PIOB) Output Disable Registerr
AT91C_PIOB_IDR            EQU (0x400E0E44) ;- (PIOB) Interrupt Disable Register
AT91C_PIOB_AIMER          EQU (0x400E0EB0) ;- (PIOB) Additional Interrupt Modes Enable Register
AT91C_PIOB_DIFSR          EQU (0x400E0E84) ;- (PIOB) Debouncing Input Filter Select Register
AT91C_PIOB_PDR            EQU (0x400E0E04) ;- (PIOB) PIO Disable Register
AT91C_PIOB_REHLSR         EQU (0x400E0ED4) ;- (PIOB) Rising Edge/ High Level Select Register
AT91C_PIOB_PDSR           EQU (0x400E0E3C) ;- (PIOB) Pin Data Status Register
AT91C_PIOB_PPUDR          EQU (0x400E0E60) ;- (PIOB) Pull-up Disable Register
AT91C_PIOB_LSR            EQU (0x400E0EC4) ;- (PIOB) Level Select Register
AT91C_PIOB_OWDR           EQU (0x400E0EA4) ;- (PIOB) Output Write Disable Register
AT91C_PIOB_FELLSR         EQU (0x400E0ED0) ;- (PIOB) Falling Edge/Low Level Select Register
AT91C_PIOB_IFER           EQU (0x400E0E20) ;- (PIOB) Input Filter Enable Register
AT91C_PIOB_ABSR           EQU (0x400E0E70) ;- (PIOB) Peripheral AB Select Register
AT91C_PIOB_KIMR           EQU (0x400E0F38) ;- (PIOB) Keypad Controller Interrupt Mask Register
AT91C_PIOB_KKPR           EQU (0x400E0F40) ;- (PIOB) Keypad Controller Key Press Register
AT91C_PIOB_FRLHSR         EQU (0x400E0ED8) ;- (PIOB) Fall/Rise - Low/High Status Register
AT91C_PIOB_AIMDR          EQU (0x400E0EB4) ;- (PIOB) Additional Interrupt Modes Disables Register
AT91C_PIOB_SCIFSR         EQU (0x400E0E80) ;- (PIOB) System Clock Glitch Input Filter Select Register
AT91C_PIOB_VER            EQU (0x400E0EFC) ;- (PIOB) PIO VERSION REGISTER 
AT91C_PIOB_PER            EQU (0x400E0E00) ;- (PIOB) PIO Enable Register
AT91C_PIOB_ELSR           EQU (0x400E0EC8) ;- (PIOB) Edge/Level Status Register
AT91C_PIOB_IMR            EQU (0x400E0E48) ;- (PIOB) Interrupt Mask Register
AT91C_PIOB_PPUSR          EQU (0x400E0E68) ;- (PIOB) Pull-up Status Register
AT91C_PIOB_SCDR           EQU (0x400E0E8C) ;- (PIOB) Slow Clock Divider Debouncing Register
AT91C_PIOB_KSR            EQU (0x400E0F3C) ;- (PIOB) Keypad Controller Status Register
AT91C_PIOB_IFDGSR         EQU (0x400E0E88) ;- (PIOB) Glitch or Debouncing Input Filter Clock Selection Status Register
AT91C_PIOB_ESR            EQU (0x400E0EC0) ;- (PIOB) Edge Select Register
AT91C_PIOB_ODSR           EQU (0x400E0E38) ;- (PIOB) Output Data Status Register
AT91C_PIOB_IFDR           EQU (0x400E0E24) ;- (PIOB) Input Filter Disable Register
AT91C_PIOB_SODR           EQU (0x400E0E30) ;- (PIOB) Set Output Data Register
AT91C_PIOB_IER            EQU (0x400E0E40) ;- (PIOB) Interrupt Enable Register
AT91C_PIOB_MDSR           EQU (0x400E0E58) ;- (PIOB) Multi-driver Status Register
AT91C_PIOB_ISR            EQU (0x400E0E4C) ;- (PIOB) Interrupt Status Register
AT91C_PIOB_IFSR           EQU (0x400E0E28) ;- (PIOB) Input Filter Status Register
AT91C_PIOB_KER            EQU (0x400E0F20) ;- (PIOB) Keypad Controller Enable Register
AT91C_PIOB_KKRR           EQU (0x400E0F44) ;- (PIOB) Keypad Controller Key Release Register
AT91C_PIOB_PPUER          EQU (0x400E0E64) ;- (PIOB) Pull-up Enable Register
AT91C_PIOB_LOCKSR         EQU (0x400E0EE0) ;- (PIOB) Lock Status Register
AT91C_PIOB_OWER           EQU (0x400E0EA0) ;- (PIOB) Output Write Enable Register
AT91C_PIOB_KIER           EQU (0x400E0F30) ;- (PIOB) Keypad Controller Interrupt Enable Register
AT91C_PIOB_MDDR           EQU (0x400E0E54) ;- (PIOB) Multi-driver Disable Register
AT91C_PIOB_KRCR           EQU (0x400E0F24) ;- (PIOB) Keypad Controller Row Column Register
AT91C_PIOB_CODR           EQU (0x400E0E34) ;- (PIOB) Clear Output Data Register
AT91C_PIOB_KDR            EQU (0x400E0F28) ;- (PIOB) Keypad Controller Debouncing Register
AT91C_PIOB_AIMMR          EQU (0x400E0EB8) ;- (PIOB) Additional Interrupt Modes Mask Register
AT91C_PIOB_OER            EQU (0x400E0E10) ;- (PIOB) Output Enable Register
AT91C_PIOB_OSR            EQU (0x400E0E18) ;- (PIOB) Output Status Register
// - ========== Register definition for PIOC peripheral ========== 
AT91C_PIOC_FELLSR         EQU (0x400E10D0) ;- (PIOC) Falling Edge/Low Level Select Register
AT91C_PIOC_FRLHSR         EQU (0x400E10D8) ;- (PIOC) Fall/Rise - Low/High Status Register
AT91C_PIOC_MDDR           EQU (0x400E1054) ;- (PIOC) Multi-driver Disable Register
AT91C_PIOC_IFDGSR         EQU (0x400E1088) ;- (PIOC) Glitch or Debouncing Input Filter Clock Selection Status Register
AT91C_PIOC_ABSR           EQU (0x400E1070) ;- (PIOC) Peripheral AB Select Register
AT91C_PIOC_KIMR           EQU (0x400E1138) ;- (PIOC) Keypad Controller Interrupt Mask Register
AT91C_PIOC_KRCR           EQU (0x400E1124) ;- (PIOC) Keypad Controller Row Column Register
AT91C_PIOC_ODSR           EQU (0x400E1038) ;- (PIOC) Output Data Status Register
AT91C_PIOC_OSR            EQU (0x400E1018) ;- (PIOC) Output Status Register
AT91C_PIOC_IFER           EQU (0x400E1020) ;- (PIOC) Input Filter Enable Register
AT91C_PIOC_KKPR           EQU (0x400E1140) ;- (PIOC) Keypad Controller Key Press Register
AT91C_PIOC_MDSR           EQU (0x400E1058) ;- (PIOC) Multi-driver Status Register
AT91C_PIOC_IFDR           EQU (0x400E1024) ;- (PIOC) Input Filter Disable Register
AT91C_PIOC_MDER           EQU (0x400E1050) ;- (PIOC) Multi-driver Enable Register
AT91C_PIOC_SCDR           EQU (0x400E108C) ;- (PIOC) Slow Clock Divider Debouncing Register
AT91C_PIOC_SCIFSR         EQU (0x400E1080) ;- (PIOC) System Clock Glitch Input Filter Select Register
AT91C_PIOC_IER            EQU (0x400E1040) ;- (PIOC) Interrupt Enable Register
AT91C_PIOC_KDR            EQU (0x400E1128) ;- (PIOC) Keypad Controller Debouncing Register
AT91C_PIOC_OWDR           EQU (0x400E10A4) ;- (PIOC) Output Write Disable Register
AT91C_PIOC_IFSR           EQU (0x400E1028) ;- (PIOC) Input Filter Status Register
AT91C_PIOC_ISR            EQU (0x400E104C) ;- (PIOC) Interrupt Status Register
AT91C_PIOC_PPUDR          EQU (0x400E1060) ;- (PIOC) Pull-up Disable Register
AT91C_PIOC_PDSR           EQU (0x400E103C) ;- (PIOC) Pin Data Status Register
AT91C_PIOC_KKRR           EQU (0x400E1144) ;- (PIOC) Keypad Controller Key Release Register
AT91C_PIOC_AIMDR          EQU (0x400E10B4) ;- (PIOC) Additional Interrupt Modes Disables Register
AT91C_PIOC_LSR            EQU (0x400E10C4) ;- (PIOC) Level Select Register
AT91C_PIOC_PPUER          EQU (0x400E1064) ;- (PIOC) Pull-up Enable Register
AT91C_PIOC_AIMER          EQU (0x400E10B0) ;- (PIOC) Additional Interrupt Modes Enable Register
AT91C_PIOC_OER            EQU (0x400E1010) ;- (PIOC) Output Enable Register
AT91C_PIOC_CODR           EQU (0x400E1034) ;- (PIOC) Clear Output Data Register
AT91C_PIOC_AIMMR          EQU (0x400E10B8) ;- (PIOC) Additional Interrupt Modes Mask Register
AT91C_PIOC_OWER           EQU (0x400E10A0) ;- (PIOC) Output Write Enable Register
AT91C_PIOC_VER            EQU (0x400E10FC) ;- (PIOC) PIO VERSION REGISTER 
AT91C_PIOC_IMR            EQU (0x400E1048) ;- (PIOC) Interrupt Mask Register
AT91C_PIOC_PPUSR          EQU (0x400E1068) ;- (PIOC) Pull-up Status Register
AT91C_PIOC_IDR            EQU (0x400E1044) ;- (PIOC) Interrupt Disable Register
AT91C_PIOC_DIFSR          EQU (0x400E1084) ;- (PIOC) Debouncing Input Filter Select Register
AT91C_PIOC_KIDR           EQU (0x400E1134) ;- (PIOC) Keypad Controller Interrupt Disable Register
AT91C_PIOC_KSR            EQU (0x400E113C) ;- (PIOC) Keypad Controller Status Register
AT91C_PIOC_REHLSR         EQU (0x400E10D4) ;- (PIOC) Rising Edge/ High Level Select Register
AT91C_PIOC_ESR            EQU (0x400E10C0) ;- (PIOC) Edge Select Register
AT91C_PIOC_KIER           EQU (0x400E1130) ;- (PIOC) Keypad Controller Interrupt Enable Register
AT91C_PIOC_ELSR           EQU (0x400E10C8) ;- (PIOC) Edge/Level Status Register
AT91C_PIOC_SODR           EQU (0x400E1030) ;- (PIOC) Set Output Data Register
AT91C_PIOC_PSR            EQU (0x400E1008) ;- (PIOC) PIO Status Register
AT91C_PIOC_KER            EQU (0x400E1120) ;- (PIOC) Keypad Controller Enable Register
AT91C_PIOC_ODR            EQU (0x400E1014) ;- (PIOC) Output Disable Registerr
AT91C_PIOC_OWSR           EQU (0x400E10A8) ;- (PIOC) Output Write Status Register
AT91C_PIOC_PDR            EQU (0x400E1004) ;- (PIOC) PIO Disable Register
AT91C_PIOC_LOCKSR         EQU (0x400E10E0) ;- (PIOC) Lock Status Register
AT91C_PIOC_PER            EQU (0x400E1000) ;- (PIOC) PIO Enable Register
// - ========== Register definition for PMC peripheral ========== 
AT91C_PMC_PLLAR           EQU (0x400E0428) ;- (PMC) PLL Register
AT91C_PMC_UCKR            EQU (0x400E041C) ;- (PMC) UTMI Clock Configuration Register
AT91C_PMC_FSMR            EQU (0x400E0470) ;- (PMC) Fast Startup Mode Register
AT91C_PMC_MCKR            EQU (0x400E0430) ;- (PMC) Master Clock Register
AT91C_PMC_SCER            EQU (0x400E0400) ;- (PMC) System Clock Enable Register
AT91C_PMC_PCSR            EQU (0x400E0418) ;- (PMC) Peripheral Clock Status Register
AT91C_PMC_MCFR            EQU (0x400E0424) ;- (PMC) Main Clock  Frequency Register
AT91C_PMC_FOCR            EQU (0x400E0478) ;- (PMC) Fault Output Clear Register
AT91C_PMC_FSPR            EQU (0x400E0474) ;- (PMC) Fast Startup Polarity Register
AT91C_PMC_SCSR            EQU (0x400E0408) ;- (PMC) System Clock Status Register
AT91C_PMC_IDR             EQU (0x400E0464) ;- (PMC) Interrupt Disable Register
AT91C_PMC_VER             EQU (0x400E04FC) ;- (PMC) APMC VERSION REGISTER
AT91C_PMC_IMR             EQU (0x400E046C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IPNAME2         EQU (0x400E04F4) ;- (PMC) PMC IPNAME2 REGISTER 
AT91C_PMC_SCDR            EQU (0x400E0404) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCKR            EQU (0x400E0440) ;- (PMC) Programmable Clock Register
AT91C_PMC_ADDRSIZE        EQU (0x400E04EC) ;- (PMC) PMC ADDRSIZE REGISTER 
AT91C_PMC_PCDR            EQU (0x400E0414) ;- (PMC) Peripheral Clock Disable Register
AT91C_PMC_MOR             EQU (0x400E0420) ;- (PMC) Main Oscillator Register
AT91C_PMC_SR              EQU (0x400E0468) ;- (PMC) Status Register
AT91C_PMC_IER             EQU (0x400E0460) ;- (PMC) Interrupt Enable Register
AT91C_PMC_IPNAME1         EQU (0x400E04F0) ;- (PMC) PMC IPNAME1 REGISTER 
AT91C_PMC_PCER            EQU (0x400E0410) ;- (PMC) Peripheral Clock Enable Register
AT91C_PMC_FEATURES        EQU (0x400E04F8) ;- (PMC) PMC FEATURES REGISTER 
// - ========== Register definition for CKGR peripheral ========== 
AT91C_CKGR_PLLAR          EQU (0x400E0428) ;- (CKGR) PLL Register
AT91C_CKGR_UCKR           EQU (0x400E041C) ;- (CKGR) UTMI Clock Configuration Register
AT91C_CKGR_MOR            EQU (0x400E0420) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_MCFR           EQU (0x400E0424) ;- (CKGR) Main Clock  Frequency Register
// - ========== Register definition for RSTC peripheral ========== 
AT91C_RSTC_VER            EQU (0x400E12FC) ;- (RSTC) Version Register
AT91C_RSTC_RCR            EQU (0x400E1200) ;- (RSTC) Reset Control Register
AT91C_RSTC_RMR            EQU (0x400E1208) ;- (RSTC) Reset Mode Register
AT91C_RSTC_RSR            EQU (0x400E1204) ;- (RSTC) Reset Status Register
// - ========== Register definition for SUPC peripheral ========== 
AT91C_SUPC_WUIR           EQU (0x400E1220) ;- (SUPC) Wake Up Inputs Register
AT91C_SUPC_CR             EQU (0x400E1210) ;- (SUPC) Control Register
AT91C_SUPC_MR             EQU (0x400E1218) ;- (SUPC) Mode Register
AT91C_SUPC_FWUTR          EQU (0x400E1228) ;- (SUPC) Flash Wake-up Timer Register
AT91C_SUPC_SR             EQU (0x400E1224) ;- (SUPC) Status Register
AT91C_SUPC_WUMR           EQU (0x400E121C) ;- (SUPC) Wake Up Mode Register
AT91C_SUPC_BOMR           EQU (0x400E1214) ;- (SUPC) Brown Out Mode Register
// - ========== Register definition for RTTC peripheral ========== 
AT91C_RTTC_RTVR           EQU (0x400E1238) ;- (RTTC) Real-time Value Register
AT91C_RTTC_RTAR           EQU (0x400E1234) ;- (RTTC) Real-time Alarm Register
AT91C_RTTC_RTMR           EQU (0x400E1230) ;- (RTTC) Real-time Mode Register
AT91C_RTTC_RTSR           EQU (0x400E123C) ;- (RTTC) Real-time Status Register
// - ========== Register definition for WDTC peripheral ========== 
AT91C_WDTC_WDSR           EQU (0x400E1258) ;- (WDTC) Watchdog Status Register
AT91C_WDTC_WDMR           EQU (0x400E1254) ;- (WDTC) Watchdog Mode Register
AT91C_WDTC_WDCR           EQU (0x400E1250) ;- (WDTC) Watchdog Control Register
// - ========== Register definition for RTC peripheral ========== 
AT91C_RTC_IMR             EQU (0x400E1288) ;- (RTC) Interrupt Mask Register
AT91C_RTC_SCCR            EQU (0x400E127C) ;- (RTC) Status Clear Command Register
AT91C_RTC_CALR            EQU (0x400E126C) ;- (RTC) Calendar Register
AT91C_RTC_MR              EQU (0x400E1264) ;- (RTC) Mode Register
AT91C_RTC_TIMR            EQU (0x400E1268) ;- (RTC) Time Register
AT91C_RTC_CALALR          EQU (0x400E1274) ;- (RTC) Calendar Alarm Register
AT91C_RTC_VER             EQU (0x400E128C) ;- (RTC) Valid Entry Register
AT91C_RTC_CR              EQU (0x400E1260) ;- (RTC) Control Register
AT91C_RTC_IDR             EQU (0x400E1284) ;- (RTC) Interrupt Disable Register
AT91C_RTC_TIMALR          EQU (0x400E1270) ;- (RTC) Time Alarm Register
AT91C_RTC_IER             EQU (0x400E1280) ;- (RTC) Interrupt Enable Register
AT91C_RTC_SR              EQU (0x400E1278) ;- (RTC) Status Register
// - ========== Register definition for ADC0 peripheral ========== 
AT91C_ADC0_IPNAME2        EQU (0x400AC0F4) ;- (ADC0) ADC IPNAME2 REGISTER 
AT91C_ADC0_ADDRSIZE       EQU (0x400AC0EC) ;- (ADC0) ADC ADDRSIZE REGISTER 
AT91C_ADC0_IDR            EQU (0x400AC028) ;- (ADC0) ADC Interrupt Disable Register
AT91C_ADC0_CHSR           EQU (0x400AC018) ;- (ADC0) ADC Channel Status Register
AT91C_ADC0_FEATURES       EQU (0x400AC0F8) ;- (ADC0) ADC FEATURES REGISTER 
AT91C_ADC0_CDR0           EQU (0x400AC030) ;- (ADC0) ADC Channel Data Register 0
AT91C_ADC0_LCDR           EQU (0x400AC020) ;- (ADC0) ADC Last Converted Data Register
AT91C_ADC0_EMR            EQU (0x400AC068) ;- (ADC0) Extended Mode Register
AT91C_ADC0_CDR3           EQU (0x400AC03C) ;- (ADC0) ADC Channel Data Register 3
AT91C_ADC0_CDR7           EQU (0x400AC04C) ;- (ADC0) ADC Channel Data Register 7
AT91C_ADC0_SR             EQU (0x400AC01C) ;- (ADC0) ADC Status Register
AT91C_ADC0_ACR            EQU (0x400AC064) ;- (ADC0) Analog Control Register
AT91C_ADC0_CDR5           EQU (0x400AC044) ;- (ADC0) ADC Channel Data Register 5
AT91C_ADC0_IPNAME1        EQU (0x400AC0F0) ;- (ADC0) ADC IPNAME1 REGISTER 
AT91C_ADC0_CDR6           EQU (0x400AC048) ;- (ADC0) ADC Channel Data Register 6
AT91C_ADC0_MR             EQU (0x400AC004) ;- (ADC0) ADC Mode Register
AT91C_ADC0_CDR1           EQU (0x400AC034) ;- (ADC0) ADC Channel Data Register 1
AT91C_ADC0_CDR2           EQU (0x400AC038) ;- (ADC0) ADC Channel Data Register 2
AT91C_ADC0_CDR4           EQU (0x400AC040) ;- (ADC0) ADC Channel Data Register 4
AT91C_ADC0_CHER           EQU (0x400AC010) ;- (ADC0) ADC Channel Enable Register
AT91C_ADC0_VER            EQU (0x400AC0FC) ;- (ADC0) ADC VERSION REGISTER
AT91C_ADC0_CHDR           EQU (0x400AC014) ;- (ADC0) ADC Channel Disable Register
AT91C_ADC0_CR             EQU (0x400AC000) ;- (ADC0) ADC Control Register
AT91C_ADC0_IMR            EQU (0x400AC02C) ;- (ADC0) ADC Interrupt Mask Register
AT91C_ADC0_IER            EQU (0x400AC024) ;- (ADC0) ADC Interrupt Enable Register
// - ========== Register definition for TC0 peripheral ========== 
AT91C_TC0_IER             EQU (0x40080024) ;- (TC0) Interrupt Enable Register
AT91C_TC0_CV              EQU (0x40080010) ;- (TC0) Counter Value
AT91C_TC0_RA              EQU (0x40080014) ;- (TC0) Register A
AT91C_TC0_RB              EQU (0x40080018) ;- (TC0) Register B
AT91C_TC0_IDR             EQU (0x40080028) ;- (TC0) Interrupt Disable Register
AT91C_TC0_SR              EQU (0x40080020) ;- (TC0) Status Register
AT91C_TC0_IMR             EQU (0x4008002C) ;- (TC0) Interrupt Mask Register
AT91C_TC0_CMR             EQU (0x40080004) ;- (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC0_RC              EQU (0x4008001C) ;- (TC0) Register C
AT91C_TC0_CCR             EQU (0x40080000) ;- (TC0) Channel Control Register
// - ========== Register definition for TC1 peripheral ========== 
AT91C_TC1_SR              EQU (0x40080060) ;- (TC1) Status Register
AT91C_TC1_RA              EQU (0x40080054) ;- (TC1) Register A
AT91C_TC1_IER             EQU (0x40080064) ;- (TC1) Interrupt Enable Register
AT91C_TC1_RB              EQU (0x40080058) ;- (TC1) Register B
AT91C_TC1_IDR             EQU (0x40080068) ;- (TC1) Interrupt Disable Register
AT91C_TC1_CCR             EQU (0x40080040) ;- (TC1) Channel Control Register
AT91C_TC1_IMR             EQU (0x4008006C) ;- (TC1) Interrupt Mask Register
AT91C_TC1_RC              EQU (0x4008005C) ;- (TC1) Register C
AT91C_TC1_CMR             EQU (0x40080044) ;- (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC1_CV              EQU (0x40080050) ;- (TC1) Counter Value
// - ========== Register definition for TC2 peripheral ========== 
AT91C_TC2_RA              EQU (0x40080094) ;- (TC2) Register A
AT91C_TC2_RB              EQU (0x40080098) ;- (TC2) Register B
AT91C_TC2_CMR             EQU (0x40080084) ;- (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC2_SR              EQU (0x400800A0) ;- (TC2) Status Register
AT91C_TC2_CCR             EQU (0x40080080) ;- (TC2) Channel Control Register
AT91C_TC2_IMR             EQU (0x400800AC) ;- (TC2) Interrupt Mask Register
AT91C_TC2_CV              EQU (0x40080090) ;- (TC2) Counter Value
AT91C_TC2_RC              EQU (0x4008009C) ;- (TC2) Register C
AT91C_TC2_IER             EQU (0x400800A4) ;- (TC2) Interrupt Enable Register
AT91C_TC2_IDR             EQU (0x400800A8) ;- (TC2) Interrupt Disable Register
// - ========== Register definition for TCB0 peripheral ========== 
AT91C_TCB0_BCR            EQU (0x400800C0) ;- (TCB0) TC Block Control Register
AT91C_TCB0_IPNAME2        EQU (0x400800F4) ;- (TCB0) TC IPNAME2 REGISTER 
AT91C_TCB0_IPNAME1        EQU (0x400800F0) ;- (TCB0) TC IPNAME1 REGISTER 
AT91C_TCB0_ADDRSIZE       EQU (0x400800EC) ;- (TCB0) TC ADDRSIZE REGISTER 
AT91C_TCB0_FEATURES       EQU (0x400800F8) ;- (TCB0) TC FEATURES REGISTER 
AT91C_TCB0_BMR            EQU (0x400800C4) ;- (TCB0) TC Block Mode Register
AT91C_TCB0_VER            EQU (0x400800FC) ;- (TCB0)  Version Register
// - ========== Register definition for TCB1 peripheral ========== 
AT91C_TCB1_BCR            EQU (0x40080100) ;- (TCB1) TC Block Control Register
AT91C_TCB1_VER            EQU (0x4008013C) ;- (TCB1)  Version Register
AT91C_TCB1_FEATURES       EQU (0x40080138) ;- (TCB1) TC FEATURES REGISTER 
AT91C_TCB1_IPNAME2        EQU (0x40080134) ;- (TCB1) TC IPNAME2 REGISTER 
AT91C_TCB1_BMR            EQU (0x40080104) ;- (TCB1) TC Block Mode Register
AT91C_TCB1_ADDRSIZE       EQU (0x4008012C) ;- (TCB1) TC ADDRSIZE REGISTER 
AT91C_TCB1_IPNAME1        EQU (0x40080130) ;- (TCB1) TC IPNAME1 REGISTER 
// - ========== Register definition for TCB2 peripheral ========== 
AT91C_TCB2_FEATURES       EQU (0x40080178) ;- (TCB2) TC FEATURES REGISTER 
AT91C_TCB2_VER            EQU (0x4008017C) ;- (TCB2)  Version Register
AT91C_TCB2_ADDRSIZE       EQU (0x4008016C) ;- (TCB2) TC ADDRSIZE REGISTER 
AT91C_TCB2_IPNAME1        EQU (0x40080170) ;- (TCB2) TC IPNAME1 REGISTER 
AT91C_TCB2_IPNAME2        EQU (0x40080174) ;- (TCB2) TC IPNAME2 REGISTER 
AT91C_TCB2_BMR            EQU (0x40080144) ;- (TCB2) TC Block Mode Register
AT91C_TCB2_BCR            EQU (0x40080140) ;- (TCB2) TC Block Control Register
// - ========== Register definition for EFC0 peripheral ========== 
AT91C_EFC0_FCR            EQU (0x400E0804) ;- (EFC0) EFC Flash Command Register
AT91C_EFC0_FRR            EQU (0x400E080C) ;- (EFC0) EFC Flash Result Register
AT91C_EFC0_FMR            EQU (0x400E0800) ;- (EFC0) EFC Flash Mode Register
AT91C_EFC0_FSR            EQU (0x400E0808) ;- (EFC0) EFC Flash Status Register
AT91C_EFC0_FVR            EQU (0x400E0814) ;- (EFC0) EFC Flash Version Register
// - ========== Register definition for EFC1 peripheral ========== 
AT91C_EFC1_FMR            EQU (0x400E0A00) ;- (EFC1) EFC Flash Mode Register
AT91C_EFC1_FVR            EQU (0x400E0A14) ;- (EFC1) EFC Flash Version Register
AT91C_EFC1_FSR            EQU (0x400E0A08) ;- (EFC1) EFC Flash Status Register
AT91C_EFC1_FCR            EQU (0x400E0A04) ;- (EFC1) EFC Flash Command Register
AT91C_EFC1_FRR            EQU (0x400E0A0C) ;- (EFC1) EFC Flash Result Register
// - ========== Register definition for MCI0 peripheral ========== 
AT91C_MCI0_DMA            EQU (0x40000050) ;- (MCI0) MCI DMA Configuration Register
AT91C_MCI0_SDCR           EQU (0x4000000C) ;- (MCI0) MCI SD/SDIO Card Register
AT91C_MCI0_IPNAME1        EQU (0x400000F0) ;- (MCI0) MCI IPNAME1 REGISTER 
AT91C_MCI0_CSTOR          EQU (0x4000001C) ;- (MCI0) MCI Completion Signal Timeout Register
AT91C_MCI0_RDR            EQU (0x40000030) ;- (MCI0) MCI Receive Data Register
AT91C_MCI0_CMDR           EQU (0x40000014) ;- (MCI0) MCI Command Register
AT91C_MCI0_IDR            EQU (0x40000048) ;- (MCI0) MCI Interrupt Disable Register
AT91C_MCI0_ADDRSIZE       EQU (0x400000EC) ;- (MCI0) MCI ADDRSIZE REGISTER 
AT91C_MCI0_WPCR           EQU (0x400000E4) ;- (MCI0) MCI Write Protection Control Register
AT91C_MCI0_RSPR           EQU (0x40000020) ;- (MCI0) MCI Response Register
AT91C_MCI0_IPNAME2        EQU (0x400000F4) ;- (MCI0) MCI IPNAME2 REGISTER 
AT91C_MCI0_CR             EQU (0x40000000) ;- (MCI0) MCI Control Register
AT91C_MCI0_IMR            EQU (0x4000004C) ;- (MCI0) MCI Interrupt Mask Register
AT91C_MCI0_WPSR           EQU (0x400000E8) ;- (MCI0) MCI Write Protection Status Register
AT91C_MCI0_DTOR           EQU (0x40000008) ;- (MCI0) MCI Data Timeout Register
AT91C_MCI0_MR             EQU (0x40000004) ;- (MCI0) MCI Mode Register
AT91C_MCI0_SR             EQU (0x40000040) ;- (MCI0) MCI Status Register
AT91C_MCI0_IER            EQU (0x40000044) ;- (MCI0) MCI Interrupt Enable Register
AT91C_MCI0_VER            EQU (0x400000FC) ;- (MCI0) MCI VERSION REGISTER 
AT91C_MCI0_FEATURES       EQU (0x400000F8) ;- (MCI0) MCI FEATURES REGISTER 
AT91C_MCI0_BLKR           EQU (0x40000018) ;- (MCI0) MCI Block Register
AT91C_MCI0_ARGR           EQU (0x40000010) ;- (MCI0) MCI Argument Register
AT91C_MCI0_FIFO           EQU (0x40000200) ;- (MCI0) MCI FIFO Aperture Register
AT91C_MCI0_TDR            EQU (0x40000034) ;- (MCI0) MCI Transmit Data Register
AT91C_MCI0_CFG            EQU (0x40000054) ;- (MCI0) MCI Configuration Register
// - ========== Register definition for PDC_TWI0 peripheral ========== 
AT91C_TWI0_TNCR           EQU (0x4008411C) ;- (PDC_TWI0) Transmit Next Counter Register
AT91C_TWI0_PTCR           EQU (0x40084120) ;- (PDC_TWI0) PDC Transfer Control Register
AT91C_TWI0_PTSR           EQU (0x40084124) ;- (PDC_TWI0) PDC Transfer Status Register
AT91C_TWI0_RCR            EQU (0x40084104) ;- (PDC_TWI0) Receive Counter Register
AT91C_TWI0_TNPR           EQU (0x40084118) ;- (PDC_TWI0) Transmit Next Pointer Register
AT91C_TWI0_RNPR           EQU (0x40084110) ;- (PDC_TWI0) Receive Next Pointer Register
AT91C_TWI0_RPR            EQU (0x40084100) ;- (PDC_TWI0) Receive Pointer Register
AT91C_TWI0_RNCR           EQU (0x40084114) ;- (PDC_TWI0) Receive Next Counter Register
AT91C_TWI0_TPR            EQU (0x40084108) ;- (PDC_TWI0) Transmit Pointer Register
AT91C_TWI0_TCR            EQU (0x4008410C) ;- (PDC_TWI0) Transmit Counter Register
// - ========== Register definition for PDC_TWI1 peripheral ========== 
AT91C_TWI1_TNCR           EQU (0x4008811C) ;- (PDC_TWI1) Transmit Next Counter Register
AT91C_TWI1_PTCR           EQU (0x40088120) ;- (PDC_TWI1) PDC Transfer Control Register
AT91C_TWI1_RNCR           EQU (0x40088114) ;- (PDC_TWI1) Receive Next Counter Register
AT91C_TWI1_RCR            EQU (0x40088104) ;- (PDC_TWI1) Receive Counter Register
AT91C_TWI1_RPR            EQU (0x40088100) ;- (PDC_TWI1) Receive Pointer Register
AT91C_TWI1_TNPR           EQU (0x40088118) ;- (PDC_TWI1) Transmit Next Pointer Register
AT91C_TWI1_RNPR           EQU (0x40088110) ;- (PDC_TWI1) Receive Next Pointer Register
AT91C_TWI1_TCR            EQU (0x4008810C) ;- (PDC_TWI1) Transmit Counter Register
AT91C_TWI1_TPR            EQU (0x40088108) ;- (PDC_TWI1) Transmit Pointer Register
AT91C_TWI1_PTSR           EQU (0x40088124) ;- (PDC_TWI1) PDC Transfer Status Register
// - ========== Register definition for TWI0 peripheral ========== 
AT91C_TWI0_FEATURES       EQU (0x400840F8) ;- (TWI0) TWI FEATURES REGISTER 
AT91C_TWI0_IPNAME1        EQU (0x400840F0) ;- (TWI0) TWI IPNAME1 REGISTER 
AT91C_TWI0_SMR            EQU (0x40084008) ;- (TWI0) Slave Mode Register
AT91C_TWI0_MMR            EQU (0x40084004) ;- (TWI0) Master Mode Register
AT91C_TWI0_SR             EQU (0x40084020) ;- (TWI0) Status Register
AT91C_TWI0_IPNAME2        EQU (0x400840F4) ;- (TWI0) TWI IPNAME2 REGISTER 
AT91C_TWI0_CR             EQU (0x40084000) ;- (TWI0) Control Register
AT91C_TWI0_IER            EQU (0x40084024) ;- (TWI0) Interrupt Enable Register
AT91C_TWI0_RHR            EQU (0x40084030) ;- (TWI0) Receive Holding Register
AT91C_TWI0_ADDRSIZE       EQU (0x400840EC) ;- (TWI0) TWI ADDRSIZE REGISTER 
AT91C_TWI0_THR            EQU (0x40084034) ;- (TWI0) Transmit Holding Register
AT91C_TWI0_VER            EQU (0x400840FC) ;- (TWI0) Version Register
AT91C_TWI0_IADR           EQU (0x4008400C) ;- (TWI0) Internal Address Register
AT91C_TWI0_IMR            EQU (0x4008402C) ;- (TWI0) Interrupt Mask Register
AT91C_TWI0_CWGR           EQU (0x40084010) ;- (TWI0) Clock Waveform Generator Register
AT91C_TWI0_IDR            EQU (0x40084028) ;- (TWI0) Interrupt Disable Register
// - ========== Register definition for TWI1 peripheral ========== 
AT91C_TWI1_VER            EQU (0x400880FC) ;- (TWI1) Version Register
AT91C_TWI1_IDR            EQU (0x40088028) ;- (TWI1) Interrupt Disable Register
AT91C_TWI1_IPNAME2        EQU (0x400880F4) ;- (TWI1) TWI IPNAME2 REGISTER 
AT91C_TWI1_CWGR           EQU (0x40088010) ;- (TWI1) Clock Waveform Generator Register
AT91C_TWI1_CR             EQU (0x40088000) ;- (TWI1) Control Register
AT91C_TWI1_ADDRSIZE       EQU (0x400880EC) ;- (TWI1) TWI ADDRSIZE REGISTER 
AT91C_TWI1_IADR           EQU (0x4008800C) ;- (TWI1) Internal Address Register
AT91C_TWI1_IER            EQU (0x40088024) ;- (TWI1) Interrupt Enable Register
AT91C_TWI1_SMR            EQU (0x40088008) ;- (TWI1) Slave Mode Register
AT91C_TWI1_RHR            EQU (0x40088030) ;- (TWI1) Receive Holding Register
AT91C_TWI1_FEATURES       EQU (0x400880F8) ;- (TWI1) TWI FEATURES REGISTER 
AT91C_TWI1_IMR            EQU (0x4008802C) ;- (TWI1) Interrupt Mask Register
AT91C_TWI1_SR             EQU (0x40088020) ;- (TWI1) Status Register
AT91C_TWI1_THR            EQU (0x40088034) ;- (TWI1) Transmit Holding Register
AT91C_TWI1_MMR            EQU (0x40088004) ;- (TWI1) Master Mode Register
AT91C_TWI1_IPNAME1        EQU (0x400880F0) ;- (TWI1) TWI IPNAME1 REGISTER 
// - ========== Register definition for PDC_US0 peripheral ========== 
AT91C_US0_RNCR            EQU (0x40090114) ;- (PDC_US0) Receive Next Counter Register
AT91C_US0_TNPR            EQU (0x40090118) ;- (PDC_US0) Transmit Next Pointer Register
AT91C_US0_TPR             EQU (0x40090108) ;- (PDC_US0) Transmit Pointer Register
AT91C_US0_RCR             EQU (0x40090104) ;- (PDC_US0) Receive Counter Register
AT91C_US0_RNPR            EQU (0x40090110) ;- (PDC_US0) Receive Next Pointer Register
AT91C_US0_TNCR            EQU (0x4009011C) ;- (PDC_US0) Transmit Next Counter Register
AT91C_US0_PTSR            EQU (0x40090124) ;- (PDC_US0) PDC Transfer Status Register
AT91C_US0_RPR             EQU (0x40090100) ;- (PDC_US0) Receive Pointer Register
AT91C_US0_PTCR            EQU (0x40090120) ;- (PDC_US0) PDC Transfer Control Register
AT91C_US0_TCR             EQU (0x4009010C) ;- (PDC_US0) Transmit Counter Register
// - ========== Register definition for US0 peripheral ========== 
AT91C_US0_NER             EQU (0x40090044) ;- (US0) Nb Errors Register
AT91C_US0_RHR             EQU (0x40090018) ;- (US0) Receiver Holding Register
AT91C_US0_IPNAME1         EQU (0x400900F0) ;- (US0) US IPNAME1 REGISTER 
AT91C_US0_MR              EQU (0x40090004) ;- (US0) Mode Register
AT91C_US0_RTOR            EQU (0x40090024) ;- (US0) Receiver Time-out Register
AT91C_US0_IF              EQU (0x4009004C) ;- (US0) IRDA_FILTER Register
AT91C_US0_ADDRSIZE        EQU (0x400900EC) ;- (US0) US ADDRSIZE REGISTER 
AT91C_US0_IDR             EQU (0x4009000C) ;- (US0) Interrupt Disable Register
AT91C_US0_IMR             EQU (0x40090010) ;- (US0) Interrupt Mask Register
AT91C_US0_IER             EQU (0x40090008) ;- (US0) Interrupt Enable Register
AT91C_US0_TTGR            EQU (0x40090028) ;- (US0) Transmitter Time-guard Register
AT91C_US0_IPNAME2         EQU (0x400900F4) ;- (US0) US IPNAME2 REGISTER 
AT91C_US0_FIDI            EQU (0x40090040) ;- (US0) FI_DI_Ratio Register
AT91C_US0_CR              EQU (0x40090000) ;- (US0) Control Register
AT91C_US0_BRGR            EQU (0x40090020) ;- (US0) Baud Rate Generator Register
AT91C_US0_MAN             EQU (0x40090050) ;- (US0) Manchester Encoder Decoder Register
AT91C_US0_VER             EQU (0x400900FC) ;- (US0) VERSION Register
AT91C_US0_FEATURES        EQU (0x400900F8) ;- (US0) US FEATURES REGISTER 
AT91C_US0_CSR             EQU (0x40090014) ;- (US0) Channel Status Register
AT91C_US0_THR             EQU (0x4009001C) ;- (US0) Transmitter Holding Register
// - ========== Register definition for PDC_US1 peripheral ========== 
AT91C_US1_TNPR            EQU (0x40094118) ;- (PDC_US1) Transmit Next Pointer Register
AT91C_US1_TPR             EQU (0x40094108) ;- (PDC_US1) Transmit Pointer Register
AT91C_US1_RNCR            EQU (0x40094114) ;- (PDC_US1) Receive Next Counter Register
AT91C_US1_TNCR            EQU (0x4009411C) ;- (PDC_US1) Transmit Next Counter Register
AT91C_US1_RNPR            EQU (0x40094110) ;- (PDC_US1) Receive Next Pointer Register
AT91C_US1_TCR             EQU (0x4009410C) ;- (PDC_US1) Transmit Counter Register
AT91C_US1_PTSR            EQU (0x40094124) ;- (PDC_US1) PDC Transfer Status Register
AT91C_US1_RCR             EQU (0x40094104) ;- (PDC_US1) Receive Counter Register
AT91C_US1_RPR             EQU (0x40094100) ;- (PDC_US1) Receive Pointer Register
AT91C_US1_PTCR            EQU (0x40094120) ;- (PDC_US1) PDC Transfer Control Register
// - ========== Register definition for US1 peripheral ========== 
AT91C_US1_IMR             EQU (0x40094010) ;- (US1) Interrupt Mask Register
AT91C_US1_RTOR            EQU (0x40094024) ;- (US1) Receiver Time-out Register
AT91C_US1_RHR             EQU (0x40094018) ;- (US1) Receiver Holding Register
AT91C_US1_IPNAME1         EQU (0x400940F0) ;- (US1) US IPNAME1 REGISTER 
AT91C_US1_VER             EQU (0x400940FC) ;- (US1) VERSION Register
AT91C_US1_MR              EQU (0x40094004) ;- (US1) Mode Register
AT91C_US1_FEATURES        EQU (0x400940F8) ;- (US1) US FEATURES REGISTER 
AT91C_US1_NER             EQU (0x40094044) ;- (US1) Nb Errors Register
AT91C_US1_IPNAME2         EQU (0x400940F4) ;- (US1) US IPNAME2 REGISTER 
AT91C_US1_CR              EQU (0x40094000) ;- (US1) Control Register
AT91C_US1_BRGR            EQU (0x40094020) ;- (US1) Baud Rate Generator Register
AT91C_US1_IF              EQU (0x4009404C) ;- (US1) IRDA_FILTER Register
AT91C_US1_IER             EQU (0x40094008) ;- (US1) Interrupt Enable Register
AT91C_US1_TTGR            EQU (0x40094028) ;- (US1) Transmitter Time-guard Register
AT91C_US1_FIDI            EQU (0x40094040) ;- (US1) FI_DI_Ratio Register
AT91C_US1_MAN             EQU (0x40094050) ;- (US1) Manchester Encoder Decoder Register
AT91C_US1_ADDRSIZE        EQU (0x400940EC) ;- (US1) US ADDRSIZE REGISTER 
AT91C_US1_CSR             EQU (0x40094014) ;- (US1) Channel Status Register
AT91C_US1_THR             EQU (0x4009401C) ;- (US1) Transmitter Holding Register
AT91C_US1_IDR             EQU (0x4009400C) ;- (US1) Interrupt Disable Register
// - ========== Register definition for PDC_US2 peripheral ========== 
AT91C_US2_RPR             EQU (0x40098100) ;- (PDC_US2) Receive Pointer Register
AT91C_US2_TPR             EQU (0x40098108) ;- (PDC_US2) Transmit Pointer Register
AT91C_US2_TCR             EQU (0x4009810C) ;- (PDC_US2) Transmit Counter Register
AT91C_US2_PTSR            EQU (0x40098124) ;- (PDC_US2) PDC Transfer Status Register
AT91C_US2_PTCR            EQU (0x40098120) ;- (PDC_US2) PDC Transfer Control Register
AT91C_US2_RNPR            EQU (0x40098110) ;- (PDC_US2) Receive Next Pointer Register
AT91C_US2_TNCR            EQU (0x4009811C) ;- (PDC_US2) Transmit Next Counter Register
AT91C_US2_RNCR            EQU (0x40098114) ;- (PDC_US2) Receive Next Counter Register
AT91C_US2_TNPR            EQU (0x40098118) ;- (PDC_US2) Transmit Next Pointer Register
AT91C_US2_RCR             EQU (0x40098104) ;- (PDC_US2) Receive Counter Register
// - ========== Register definition for US2 peripheral ========== 
AT91C_US2_MAN             EQU (0x40098050) ;- (US2) Manchester Encoder Decoder Register
AT91C_US2_ADDRSIZE        EQU (0x400980EC) ;- (US2) US ADDRSIZE REGISTER 
AT91C_US2_MR              EQU (0x40098004) ;- (US2) Mode Register
AT91C_US2_IPNAME1         EQU (0x400980F0) ;- (US2) US IPNAME1 REGISTER 
AT91C_US2_IF              EQU (0x4009804C) ;- (US2) IRDA_FILTER Register
AT91C_US2_BRGR            EQU (0x40098020) ;- (US2) Baud Rate Generator Register
AT91C_US2_FIDI            EQU (0x40098040) ;- (US2) FI_DI_Ratio Register
AT91C_US2_IER             EQU (0x40098008) ;- (US2) Interrupt Enable Register
AT91C_US2_RTOR            EQU (0x40098024) ;- (US2) Receiver Time-out Register
AT91C_US2_CR              EQU (0x40098000) ;- (US2) Control Register
AT91C_US2_THR             EQU (0x4009801C) ;- (US2) Transmitter Holding Register
AT91C_US2_CSR             EQU (0x40098014) ;- (US2) Channel Status Register
AT91C_US2_VER             EQU (0x400980FC) ;- (US2) VERSION Register
AT91C_US2_FEATURES        EQU (0x400980F8) ;- (US2) US FEATURES REGISTER 
AT91C_US2_IDR             EQU (0x4009800C) ;- (US2) Interrupt Disable Register
AT91C_US2_TTGR            EQU (0x40098028) ;- (US2) Transmitter Time-guard Register
AT91C_US2_IPNAME2         EQU (0x400980F4) ;- (US2) US IPNAME2 REGISTER 
AT91C_US2_RHR             EQU (0x40098018) ;- (US2) Receiver Holding Register
AT91C_US2_NER             EQU (0x40098044) ;- (US2) Nb Errors Register
AT91C_US2_IMR             EQU (0x40098010) ;- (US2) Interrupt Mask Register
// - ========== Register definition for PDC_US3 peripheral ========== 
AT91C_US3_TPR             EQU (0x4009C108) ;- (PDC_US3) Transmit Pointer Register
AT91C_US3_PTCR            EQU (0x4009C120) ;- (PDC_US3) PDC Transfer Control Register
AT91C_US3_TCR             EQU (0x4009C10C) ;- (PDC_US3) Transmit Counter Register
AT91C_US3_RCR             EQU (0x4009C104) ;- (PDC_US3) Receive Counter Register
AT91C_US3_RNCR            EQU (0x4009C114) ;- (PDC_US3) Receive Next Counter Register
AT91C_US3_RNPR            EQU (0x4009C110) ;- (PDC_US3) Receive Next Pointer Register
AT91C_US3_RPR             EQU (0x4009C100) ;- (PDC_US3) Receive Pointer Register
AT91C_US3_PTSR            EQU (0x4009C124) ;- (PDC_US3) PDC Transfer Status Register
AT91C_US3_TNCR            EQU (0x4009C11C) ;- (PDC_US3) Transmit Next Counter Register
AT91C_US3_TNPR            EQU (0x4009C118) ;- (PDC_US3) Transmit Next Pointer Register
// - ========== Register definition for US3 peripheral ========== 
AT91C_US3_MAN             EQU (0x4009C050) ;- (US3) Manchester Encoder Decoder Register
AT91C_US3_CSR             EQU (0x4009C014) ;- (US3) Channel Status Register
AT91C_US3_BRGR            EQU (0x4009C020) ;- (US3) Baud Rate Generator Register
AT91C_US3_IPNAME2         EQU (0x4009C0F4) ;- (US3) US IPNAME2 REGISTER 
AT91C_US3_RTOR            EQU (0x4009C024) ;- (US3) Receiver Time-out Register
AT91C_US3_ADDRSIZE        EQU (0x4009C0EC) ;- (US3) US ADDRSIZE REGISTER 
AT91C_US3_CR              EQU (0x4009C000) ;- (US3) Control Register
AT91C_US3_IF              EQU (0x4009C04C) ;- (US3) IRDA_FILTER Register
AT91C_US3_FEATURES        EQU (0x4009C0F8) ;- (US3) US FEATURES REGISTER 
AT91C_US3_VER             EQU (0x4009C0FC) ;- (US3) VERSION Register
AT91C_US3_RHR             EQU (0x4009C018) ;- (US3) Receiver Holding Register
AT91C_US3_TTGR            EQU (0x4009C028) ;- (US3) Transmitter Time-guard Register
AT91C_US3_NER             EQU (0x4009C044) ;- (US3) Nb Errors Register
AT91C_US3_IMR             EQU (0x4009C010) ;- (US3) Interrupt Mask Register
AT91C_US3_THR             EQU (0x4009C01C) ;- (US3) Transmitter Holding Register
AT91C_US3_IDR             EQU (0x4009C00C) ;- (US3) Interrupt Disable Register
AT91C_US3_MR              EQU (0x4009C004) ;- (US3) Mode Register
AT91C_US3_IER             EQU (0x4009C008) ;- (US3) Interrupt Enable Register
AT91C_US3_FIDI            EQU (0x4009C040) ;- (US3) FI_DI_Ratio Register
AT91C_US3_IPNAME1         EQU (0x4009C0F0) ;- (US3) US IPNAME1 REGISTER 
// - ========== Register definition for PDC_SSC0 peripheral ========== 
AT91C_SSC0_RNCR           EQU (0x40004114) ;- (PDC_SSC0) Receive Next Counter Register
AT91C_SSC0_TPR            EQU (0x40004108) ;- (PDC_SSC0) Transmit Pointer Register
AT91C_SSC0_TCR            EQU (0x4000410C) ;- (PDC_SSC0) Transmit Counter Register
AT91C_SSC0_PTCR           EQU (0x40004120) ;- (PDC_SSC0) PDC Transfer Control Register
AT91C_SSC0_TNPR           EQU (0x40004118) ;- (PDC_SSC0) Transmit Next Pointer Register
AT91C_SSC0_RPR            EQU (0x40004100) ;- (PDC_SSC0) Receive Pointer Register
AT91C_SSC0_TNCR           EQU (0x4000411C) ;- (PDC_SSC0) Transmit Next Counter Register
AT91C_SSC0_RNPR           EQU (0x40004110) ;- (PDC_SSC0) Receive Next Pointer Register
AT91C_SSC0_RCR            EQU (0x40004104) ;- (PDC_SSC0) Receive Counter Register
AT91C_SSC0_PTSR           EQU (0x40004124) ;- (PDC_SSC0) PDC Transfer Status Register
// - ========== Register definition for SSC0 peripheral ========== 
AT91C_SSC0_CR             EQU (0x40004000) ;- (SSC0) Control Register
AT91C_SSC0_RHR            EQU (0x40004020) ;- (SSC0) Receive Holding Register
AT91C_SSC0_TSHR           EQU (0x40004034) ;- (SSC0) Transmit Sync Holding Register
AT91C_SSC0_RFMR           EQU (0x40004014) ;- (SSC0) Receive Frame Mode Register
AT91C_SSC0_IDR            EQU (0x40004048) ;- (SSC0) Interrupt Disable Register
AT91C_SSC0_TFMR           EQU (0x4000401C) ;- (SSC0) Transmit Frame Mode Register
AT91C_SSC0_RSHR           EQU (0x40004030) ;- (SSC0) Receive Sync Holding Register
AT91C_SSC0_RC1R           EQU (0x4000403C) ;- (SSC0) Receive Compare 1 Register
AT91C_SSC0_TCMR           EQU (0x40004018) ;- (SSC0) Transmit Clock Mode Register
AT91C_SSC0_RCMR           EQU (0x40004010) ;- (SSC0) Receive Clock ModeRegister
AT91C_SSC0_SR             EQU (0x40004040) ;- (SSC0) Status Register
AT91C_SSC0_RC0R           EQU (0x40004038) ;- (SSC0) Receive Compare 0 Register
AT91C_SSC0_THR            EQU (0x40004024) ;- (SSC0) Transmit Holding Register
AT91C_SSC0_CMR            EQU (0x40004004) ;- (SSC0) Clock Mode Register
AT91C_SSC0_IER            EQU (0x40004044) ;- (SSC0) Interrupt Enable Register
AT91C_SSC0_IMR            EQU (0x4000404C) ;- (SSC0) Interrupt Mask Register
// - ========== Register definition for PDC_PWMC peripheral ========== 
AT91C_PWMC_TNCR           EQU (0x4008C11C) ;- (PDC_PWMC) Transmit Next Counter Register
AT91C_PWMC_TPR            EQU (0x4008C108) ;- (PDC_PWMC) Transmit Pointer Register
AT91C_PWMC_RPR            EQU (0x4008C100) ;- (PDC_PWMC) Receive Pointer Register
AT91C_PWMC_TCR            EQU (0x4008C10C) ;- (PDC_PWMC) Transmit Counter Register
AT91C_PWMC_PTSR           EQU (0x4008C124) ;- (PDC_PWMC) PDC Transfer Status Register
AT91C_PWMC_RNPR           EQU (0x4008C110) ;- (PDC_PWMC) Receive Next Pointer Register
AT91C_PWMC_RCR            EQU (0x4008C104) ;- (PDC_PWMC) Receive Counter Register
AT91C_PWMC_RNCR           EQU (0x4008C114) ;- (PDC_PWMC) Receive Next Counter Register
AT91C_PWMC_PTCR           EQU (0x4008C120) ;- (PDC_PWMC) PDC Transfer Control Register
AT91C_PWMC_TNPR           EQU (0x4008C118) ;- (PDC_PWMC) Transmit Next Pointer Register
// - ========== Register definition for PWMC_CH0 peripheral ========== 
AT91C_PWMC_CH0_DTR        EQU (0x4008C218) ;- (PWMC_CH0) Channel Dead Time Value Register
AT91C_PWMC_CH0_CMR        EQU (0x4008C200) ;- (PWMC_CH0) Channel Mode Register
AT91C_PWMC_CH0_CCNTR      EQU (0x4008C214) ;- (PWMC_CH0) Channel Counter Register
AT91C_PWMC_CH0_CPRDR      EQU (0x4008C20C) ;- (PWMC_CH0) Channel Period Register
AT91C_PWMC_CH0_DTUPDR     EQU (0x4008C21C) ;- (PWMC_CH0) Channel Dead Time Update Value Register
AT91C_PWMC_CH0_CPRDUPDR   EQU (0x4008C210) ;- (PWMC_CH0) Channel Period Update Register
AT91C_PWMC_CH0_CDTYUPDR   EQU (0x4008C208) ;- (PWMC_CH0) Channel Duty Cycle Update Register
AT91C_PWMC_CH0_CDTYR      EQU (0x4008C204) ;- (PWMC_CH0) Channel Duty Cycle Register
// - ========== Register definition for PWMC_CH1 peripheral ========== 
AT91C_PWMC_CH1_CCNTR      EQU (0x4008C234) ;- (PWMC_CH1) Channel Counter Register
AT91C_PWMC_CH1_DTR        EQU (0x4008C238) ;- (PWMC_CH1) Channel Dead Time Value Register
AT91C_PWMC_CH1_CDTYUPDR   EQU (0x4008C228) ;- (PWMC_CH1) Channel Duty Cycle Update Register
AT91C_PWMC_CH1_DTUPDR     EQU (0x4008C23C) ;- (PWMC_CH1) Channel Dead Time Update Value Register
AT91C_PWMC_CH1_CDTYR      EQU (0x4008C224) ;- (PWMC_CH1) Channel Duty Cycle Register
AT91C_PWMC_CH1_CPRDR      EQU (0x4008C22C) ;- (PWMC_CH1) Channel Period Register
AT91C_PWMC_CH1_CPRDUPDR   EQU (0x4008C230) ;- (PWMC_CH1) Channel Period Update Register
AT91C_PWMC_CH1_CMR        EQU (0x4008C220) ;- (PWMC_CH1) Channel Mode Register
// - ========== Register definition for PWMC_CH2 peripheral ========== 
AT91C_PWMC_CH2_CDTYR      EQU (0x4008C244) ;- (PWMC_CH2) Channel Duty Cycle Register
AT91C_PWMC_CH2_DTUPDR     EQU (0x4008C25C) ;- (PWMC_CH2) Channel Dead Time Update Value Register
AT91C_PWMC_CH2_CCNTR      EQU (0x4008C254) ;- (PWMC_CH2) Channel Counter Register
AT91C_PWMC_CH2_CMR        EQU (0x4008C240) ;- (PWMC_CH2) Channel Mode Register
AT91C_PWMC_CH2_CPRDR      EQU (0x4008C24C) ;- (PWMC_CH2) Channel Period Register
AT91C_PWMC_CH2_CPRDUPDR   EQU (0x4008C250) ;- (PWMC_CH2) Channel Period Update Register
AT91C_PWMC_CH2_CDTYUPDR   EQU (0x4008C248) ;- (PWMC_CH2) Channel Duty Cycle Update Register
AT91C_PWMC_CH2_DTR        EQU (0x4008C258) ;- (PWMC_CH2) Channel Dead Time Value Register
// - ========== Register definition for PWMC_CH3 peripheral ========== 
AT91C_PWMC_CH3_CPRDUPDR   EQU (0x4008C270) ;- (PWMC_CH3) Channel Period Update Register
AT91C_PWMC_CH3_DTR        EQU (0x4008C278) ;- (PWMC_CH3) Channel Dead Time Value Register
AT91C_PWMC_CH3_CDTYR      EQU (0x4008C264) ;- (PWMC_CH3) Channel Duty Cycle Register
AT91C_PWMC_CH3_DTUPDR     EQU (0x4008C27C) ;- (PWMC_CH3) Channel Dead Time Update Value Register
AT91C_PWMC_CH3_CDTYUPDR   EQU (0x4008C268) ;- (PWMC_CH3) Channel Duty Cycle Update Register
AT91C_PWMC_CH3_CCNTR      EQU (0x4008C274) ;- (PWMC_CH3) Channel Counter Register
AT91C_PWMC_CH3_CMR        EQU (0x4008C260) ;- (PWMC_CH3) Channel Mode Register
AT91C_PWMC_CH3_CPRDR      EQU (0x4008C26C) ;- (PWMC_CH3) Channel Period Register
// - ========== Register definition for PWMC peripheral ========== 
AT91C_PWMC_CMP6MUPD       EQU (0x4008C19C) ;- (PWMC) PWM Comparison Mode 6 Update Register
AT91C_PWMC_ISR1           EQU (0x4008C01C) ;- (PWMC) PWMC Interrupt Status Register 1
AT91C_PWMC_CMP5V          EQU (0x4008C180) ;- (PWMC) PWM Comparison Value 5 Register
AT91C_PWMC_CMP4MUPD       EQU (0x4008C17C) ;- (PWMC) PWM Comparison Mode 4 Update Register
AT91C_PWMC_FMR            EQU (0x4008C05C) ;- (PWMC) PWM Fault Mode Register
AT91C_PWMC_CMP6V          EQU (0x4008C190) ;- (PWMC) PWM Comparison Value 6 Register
AT91C_PWMC_EL4MR          EQU (0x4008C08C) ;- (PWMC) PWM Event Line 4 Mode Register
AT91C_PWMC_UPCR           EQU (0x4008C028) ;- (PWMC) PWM Update Control Register
AT91C_PWMC_CMP1VUPD       EQU (0x4008C144) ;- (PWMC) PWM Comparison Value 1 Update Register
AT91C_PWMC_CMP0M          EQU (0x4008C138) ;- (PWMC) PWM Comparison Mode 0 Register
AT91C_PWMC_CMP5VUPD       EQU (0x4008C184) ;- (PWMC) PWM Comparison Value 5 Update Register
AT91C_PWMC_FPER3          EQU (0x4008C074) ;- (PWMC) PWM Fault Protection Enable Register 3
AT91C_PWMC_OSCUPD         EQU (0x4008C058) ;- (PWMC) PWM Output Selection Clear Update Register
AT91C_PWMC_FPER1          EQU (0x4008C06C) ;- (PWMC) PWM Fault Protection Enable Register 1
AT91C_PWMC_SCUPUPD        EQU (0x4008C030) ;- (PWMC) PWM Update Period Update Register
AT91C_PWMC_DIS            EQU (0x4008C008) ;- (PWMC) PWMC Disable Register
AT91C_PWMC_IER1           EQU (0x4008C010) ;- (PWMC) PWMC Interrupt Enable Register 1
AT91C_PWMC_IMR2           EQU (0x4008C03C) ;- (PWMC) PWMC Interrupt Mask Register 2
AT91C_PWMC_CMP0V          EQU (0x4008C130) ;- (PWMC) PWM Comparison Value 0 Register
AT91C_PWMC_SR             EQU (0x4008C00C) ;- (PWMC) PWMC Status Register
AT91C_PWMC_CMP4M          EQU (0x4008C178) ;- (PWMC) PWM Comparison Mode 4 Register
AT91C_PWMC_CMP3M          EQU (0x4008C168) ;- (PWMC) PWM Comparison Mode 3 Register
AT91C_PWMC_IER2           EQU (0x4008C034) ;- (PWMC) PWMC Interrupt Enable Register 2
AT91C_PWMC_CMP3VUPD       EQU (0x4008C164) ;- (PWMC) PWM Comparison Value 3 Update Register
AT91C_PWMC_CMP2M          EQU (0x4008C158) ;- (PWMC) PWM Comparison Mode 2 Register
AT91C_PWMC_IDR2           EQU (0x4008C038) ;- (PWMC) PWMC Interrupt Disable Register 2
AT91C_PWMC_EL2MR          EQU (0x4008C084) ;- (PWMC) PWM Event Line 2 Mode Register
AT91C_PWMC_CMP7V          EQU (0x4008C1A0) ;- (PWMC) PWM Comparison Value 7 Register
AT91C_PWMC_CMP1M          EQU (0x4008C148) ;- (PWMC) PWM Comparison Mode 1 Register
AT91C_PWMC_CMP0VUPD       EQU (0x4008C134) ;- (PWMC) PWM Comparison Value 0 Update Register
AT91C_PWMC_WPSR           EQU (0x4008C0E8) ;- (PWMC) PWM Write Protection Status Register
AT91C_PWMC_CMP6VUPD       EQU (0x4008C194) ;- (PWMC) PWM Comparison Value 6 Update Register
AT91C_PWMC_CMP1MUPD       EQU (0x4008C14C) ;- (PWMC) PWM Comparison Mode 1 Update Register
AT91C_PWMC_CMP1V          EQU (0x4008C140) ;- (PWMC) PWM Comparison Value 1 Register
AT91C_PWMC_FCR            EQU (0x4008C064) ;- (PWMC) PWM Fault Mode Clear Register
AT91C_PWMC_VER            EQU (0x4008C0FC) ;- (PWMC) PWMC Version Register
AT91C_PWMC_EL1MR          EQU (0x4008C080) ;- (PWMC) PWM Event Line 1 Mode Register
AT91C_PWMC_EL6MR          EQU (0x4008C094) ;- (PWMC) PWM Event Line 6 Mode Register
AT91C_PWMC_ISR2           EQU (0x4008C040) ;- (PWMC) PWMC Interrupt Status Register 2
AT91C_PWMC_CMP4VUPD       EQU (0x4008C174) ;- (PWMC) PWM Comparison Value 4 Update Register
AT91C_PWMC_CMP5MUPD       EQU (0x4008C18C) ;- (PWMC) PWM Comparison Mode 5 Update Register
AT91C_PWMC_OS             EQU (0x4008C048) ;- (PWMC) PWM Output Selection Register
AT91C_PWMC_FPV            EQU (0x4008C068) ;- (PWMC) PWM Fault Protection Value Register
AT91C_PWMC_FPER2          EQU (0x4008C070) ;- (PWMC) PWM Fault Protection Enable Register 2
AT91C_PWMC_EL7MR          EQU (0x4008C098) ;- (PWMC) PWM Event Line 7 Mode Register
AT91C_PWMC_OSSUPD         EQU (0x4008C054) ;- (PWMC) PWM Output Selection Set Update Register
AT91C_PWMC_FEATURES       EQU (0x4008C0F8) ;- (PWMC) PWMC FEATURES REGISTER 
AT91C_PWMC_CMP2V          EQU (0x4008C150) ;- (PWMC) PWM Comparison Value 2 Register
AT91C_PWMC_FSR            EQU (0x4008C060) ;- (PWMC) PWM Fault Mode Status Register
AT91C_PWMC_ADDRSIZE       EQU (0x4008C0EC) ;- (PWMC) PWMC ADDRSIZE REGISTER 
AT91C_PWMC_OSC            EQU (0x4008C050) ;- (PWMC) PWM Output Selection Clear Register
AT91C_PWMC_SCUP           EQU (0x4008C02C) ;- (PWMC) PWM Update Period Register
AT91C_PWMC_CMP7MUPD       EQU (0x4008C1AC) ;- (PWMC) PWM Comparison Mode 7 Update Register
AT91C_PWMC_CMP2VUPD       EQU (0x4008C154) ;- (PWMC) PWM Comparison Value 2 Update Register
AT91C_PWMC_FPER4          EQU (0x4008C078) ;- (PWMC) PWM Fault Protection Enable Register 4
AT91C_PWMC_IMR1           EQU (0x4008C018) ;- (PWMC) PWMC Interrupt Mask Register 1
AT91C_PWMC_EL3MR          EQU (0x4008C088) ;- (PWMC) PWM Event Line 3 Mode Register
AT91C_PWMC_CMP3V          EQU (0x4008C160) ;- (PWMC) PWM Comparison Value 3 Register
AT91C_PWMC_IPNAME1        EQU (0x4008C0F0) ;- (PWMC) PWMC IPNAME1 REGISTER 
AT91C_PWMC_OSS            EQU (0x4008C04C) ;- (PWMC) PWM Output Selection Set Register
AT91C_PWMC_CMP0MUPD       EQU (0x4008C13C) ;- (PWMC) PWM Comparison Mode 0 Update Register
AT91C_PWMC_CMP2MUPD       EQU (0x4008C15C) ;- (PWMC) PWM Comparison Mode 2 Update Register
AT91C_PWMC_CMP4V          EQU (0x4008C170) ;- (PWMC) PWM Comparison Value 4 Register
AT91C_PWMC_ENA            EQU (0x4008C004) ;- (PWMC) PWMC Enable Register
AT91C_PWMC_CMP3MUPD       EQU (0x4008C16C) ;- (PWMC) PWM Comparison Mode 3 Update Register
AT91C_PWMC_EL0MR          EQU (0x4008C07C) ;- (PWMC) PWM Event Line 0 Mode Register
AT91C_PWMC_OOV            EQU (0x4008C044) ;- (PWMC) PWM Output Override Value Register
AT91C_PWMC_WPCR           EQU (0x4008C0E4) ;- (PWMC) PWM Write Protection Enable Register
AT91C_PWMC_CMP7M          EQU (0x4008C1A8) ;- (PWMC) PWM Comparison Mode 7 Register
AT91C_PWMC_CMP6M          EQU (0x4008C198) ;- (PWMC) PWM Comparison Mode 6 Register
AT91C_PWMC_CMP5M          EQU (0x4008C188) ;- (PWMC) PWM Comparison Mode 5 Register
AT91C_PWMC_IPNAME2        EQU (0x4008C0F4) ;- (PWMC) PWMC IPNAME2 REGISTER 
AT91C_PWMC_CMP7VUPD       EQU (0x4008C1A4) ;- (PWMC) PWM Comparison Value 7 Update Register
AT91C_PWMC_SYNC           EQU (0x4008C020) ;- (PWMC) PWM Synchronized Channels Register
AT91C_PWMC_MR             EQU (0x4008C000) ;- (PWMC) PWMC Mode Register
AT91C_PWMC_IDR1           EQU (0x4008C014) ;- (PWMC) PWMC Interrupt Disable Register 1
AT91C_PWMC_EL5MR          EQU (0x4008C090) ;- (PWMC) PWM Event Line 5 Mode Register
// - ========== Register definition for SPI0 peripheral ========== 
AT91C_SPI0_ADDRSIZE       EQU (0x400080EC) ;- (SPI0) SPI ADDRSIZE REGISTER 
AT91C_SPI0_RDR            EQU (0x40008008) ;- (SPI0) Receive Data Register
AT91C_SPI0_FEATURES       EQU (0x400080F8) ;- (SPI0) SPI FEATURES REGISTER 
AT91C_SPI0_CR             EQU (0x40008000) ;- (SPI0) Control Register
AT91C_SPI0_IPNAME1        EQU (0x400080F0) ;- (SPI0) SPI IPNAME1 REGISTER 
AT91C_SPI0_VER            EQU (0x400080FC) ;- (SPI0) Version Register
AT91C_SPI0_IDR            EQU (0x40008018) ;- (SPI0) Interrupt Disable Register
AT91C_SPI0_TDR            EQU (0x4000800C) ;- (SPI0) Transmit Data Register
AT91C_SPI0_MR             EQU (0x40008004) ;- (SPI0) Mode Register
AT91C_SPI0_IER            EQU (0x40008014) ;- (SPI0) Interrupt Enable Register
AT91C_SPI0_IMR            EQU (0x4000801C) ;- (SPI0) Interrupt Mask Register
AT91C_SPI0_IPNAME2        EQU (0x400080F4) ;- (SPI0) SPI IPNAME2 REGISTER 
AT91C_SPI0_CSR            EQU (0x40008030) ;- (SPI0) Chip Select Register
AT91C_SPI0_SR             EQU (0x40008010) ;- (SPI0) Status Register
// - ========== Register definition for UDPHS_EPTFIFO peripheral ========== 
AT91C_UDPHS_EPTFIFO_READEPT6 EQU (0x201E0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6
AT91C_UDPHS_EPTFIFO_READEPT2 EQU (0x201A0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2
AT91C_UDPHS_EPTFIFO_READEPT1 EQU (0x20190000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1
AT91C_UDPHS_EPTFIFO_READEPT0 EQU (0x20180000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0
AT91C_UDPHS_EPTFIFO_READEPT5 EQU (0x201D0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5
AT91C_UDPHS_EPTFIFO_READEPT4 EQU (0x201C0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4
AT91C_UDPHS_EPTFIFO_READEPT3 EQU (0x201B0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3
// - ========== Register definition for UDPHS_EPT_0 peripheral ========== 
AT91C_UDPHS_EPT_0_EPTCTL  EQU (0x400A410C) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_0_EPTSTA  EQU (0x400A411C) ;- (UDPHS_EPT_0) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_0_EPTCLRSTA EQU (0x400A4118) ;- (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_0_EPTCTLDIS EQU (0x400A4108) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_0_EPTCFG  EQU (0x400A4100) ;- (UDPHS_EPT_0) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_0_EPTSETSTA EQU (0x400A4114) ;- (UDPHS_EPT_0) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_0_EPTCTLENB EQU (0x400A4104) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register
// - ========== Register definition for UDPHS_EPT_1 peripheral ========== 
AT91C_UDPHS_EPT_1_EPTSTA  EQU (0x400A413C) ;- (UDPHS_EPT_1) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_1_EPTSETSTA EQU (0x400A4134) ;- (UDPHS_EPT_1) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_1_EPTCTL  EQU (0x400A412C) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_1_EPTCFG  EQU (0x400A4120) ;- (UDPHS_EPT_1) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_1_EPTCTLDIS EQU (0x400A4128) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_1_EPTCLRSTA EQU (0x400A4138) ;- (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_1_EPTCTLENB EQU (0x400A4124) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register
// - ========== Register definition for UDPHS_EPT_2 peripheral ========== 
AT91C_UDPHS_EPT_2_EPTCTLENB EQU (0x400A4144) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_2_EPTCLRSTA EQU (0x400A4158) ;- (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_2_EPTCFG  EQU (0x400A4140) ;- (UDPHS_EPT_2) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_2_EPTCTL  EQU (0x400A414C) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_2_EPTSETSTA EQU (0x400A4154) ;- (UDPHS_EPT_2) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_2_EPTSTA  EQU (0x400A415C) ;- (UDPHS_EPT_2) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_2_EPTCTLDIS EQU (0x400A4148) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register
// - ========== Register definition for UDPHS_EPT_3 peripheral ========== 
AT91C_UDPHS_EPT_3_EPTCTLDIS EQU (0x400A4168) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_3_EPTCTLENB EQU (0x400A4164) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_3_EPTSETSTA EQU (0x400A4174) ;- (UDPHS_EPT_3) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_3_EPTCLRSTA EQU (0x400A4178) ;- (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_3_EPTCFG  EQU (0x400A4160) ;- (UDPHS_EPT_3) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_3_EPTSTA  EQU (0x400A417C) ;- (UDPHS_EPT_3) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_3_EPTCTL  EQU (0x400A416C) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Register
// - ========== Register definition for UDPHS_EPT_4 peripheral ========== 
AT91C_UDPHS_EPT_4_EPTSETSTA EQU (0x400A4194) ;- (UDPHS_EPT_4) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_4_EPTCTLDIS EQU (0x400A4188) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_4_EPTCTL  EQU (0x400A418C) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_4_EPTCFG  EQU (0x400A4180) ;- (UDPHS_EPT_4) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_4_EPTCTLENB EQU (0x400A4184) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_4_EPTSTA  EQU (0x400A419C) ;- (UDPHS_EPT_4) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_4_EPTCLRSTA EQU (0x400A4198) ;- (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register
// - ========== Register definition for UDPHS_EPT_5 peripheral ========== 
AT91C_UDPHS_EPT_5_EPTCFG  EQU (0x400A41A0) ;- (UDPHS_EPT_5) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_5_EPTCTL  EQU (0x400A41AC) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_5_EPTCTLENB EQU (0x400A41A4) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_5_EPTSTA  EQU (0x400A41BC) ;- (UDPHS_EPT_5) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_5_EPTSETSTA EQU (0x400A41B4) ;- (UDPHS_EPT_5) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_5_EPTCTLDIS EQU (0x400A41A8) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_5_EPTCLRSTA EQU (0x400A41B8) ;- (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register
// - ========== Register definition for UDPHS_EPT_6 peripheral ========== 
AT91C_UDPHS_EPT_6_EPTCLRSTA EQU (0x400A41D8) ;- (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_6_EPTCTL  EQU (0x400A41CC) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_6_EPTCFG  EQU (0x400A41C0) ;- (UDPHS_EPT_6) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_6_EPTCTLDIS EQU (0x400A41C8) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_6_EPTSTA  EQU (0x400A41DC) ;- (UDPHS_EPT_6) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_6_EPTCTLENB EQU (0x400A41C4) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_6_EPTSETSTA EQU (0x400A41D4) ;- (UDPHS_EPT_6) UDPHS Endpoint Set Status Register
// - ========== Register definition for UDPHS_DMA_1 peripheral ========== 
AT91C_UDPHS_DMA_1_DMASTATUS EQU (0x400A431C) ;- (UDPHS_DMA_1) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_1_DMACONTROL EQU (0x400A4318) ;- (UDPHS_DMA_1) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_1_DMANXTDSC EQU (0x400A4310) ;- (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_1_DMAADDRESS EQU (0x400A4314) ;- (UDPHS_DMA_1) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS_DMA_2 peripheral ========== 
AT91C_UDPHS_DMA_2_DMASTATUS EQU (0x400A432C) ;- (UDPHS_DMA_2) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_2_DMANXTDSC EQU (0x400A4320) ;- (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_2_DMACONTROL EQU (0x400A4328) ;- (UDPHS_DMA_2) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_2_DMAADDRESS EQU (0x400A4324) ;- (UDPHS_DMA_2) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS_DMA_3 peripheral ========== 
AT91C_UDPHS_DMA_3_DMACONTROL EQU (0x400A4338) ;- (UDPHS_DMA_3) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_3_DMANXTDSC EQU (0x400A4330) ;- (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_3_DMASTATUS EQU (0x400A433C) ;- (UDPHS_DMA_3) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_3_DMAADDRESS EQU (0x400A4334) ;- (UDPHS_DMA_3) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS_DMA_4 peripheral ========== 
AT91C_UDPHS_DMA_4_DMAADDRESS EQU (0x400A4344) ;- (UDPHS_DMA_4) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_4_DMANXTDSC EQU (0x400A4340) ;- (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_4_DMASTATUS EQU (0x400A434C) ;- (UDPHS_DMA_4) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_4_DMACONTROL EQU (0x400A4348) ;- (UDPHS_DMA_4) UDPHS DMA Channel Control Register
// - ========== Register definition for UDPHS_DMA_5 peripheral ========== 
AT91C_UDPHS_DMA_5_DMACONTROL EQU (0x400A4358) ;- (UDPHS_DMA_5) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_5_DMAADDRESS EQU (0x400A4354) ;- (UDPHS_DMA_5) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_5_DMANXTDSC EQU (0x400A4350) ;- (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_5_DMASTATUS EQU (0x400A435C) ;- (UDPHS_DMA_5) UDPHS DMA Channel Status Register
// - ========== Register definition for UDPHS_DMA_6 peripheral ========== 
AT91C_UDPHS_DMA_6_DMASTATUS EQU (0x400A436C) ;- (UDPHS_DMA_6) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_6_DMACONTROL EQU (0x400A4368) ;- (UDPHS_DMA_6) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_6_DMANXTDSC EQU (0x400A4360) ;- (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_6_DMAADDRESS EQU (0x400A4364) ;- (UDPHS_DMA_6) UDPHS DMA Channel Address Register
// - ========== Register definition for UDPHS peripheral ========== 
AT91C_UDPHS_EPTRST        EQU (0x400A401C) ;- (UDPHS) UDPHS Endpoints Reset Register
AT91C_UDPHS_IEN           EQU (0x400A4010) ;- (UDPHS) UDPHS Interrupt Enable Register
AT91C_UDPHS_TSTCNTB       EQU (0x400A40D8) ;- (UDPHS) UDPHS Test B Counter Register
AT91C_UDPHS_RIPNAME2      EQU (0x400A40F4) ;- (UDPHS) UDPHS Name2 Register
AT91C_UDPHS_RIPPADDRSIZE  EQU (0x400A40EC) ;- (UDPHS) UDPHS PADDRSIZE Register
AT91C_UDPHS_TSTMODREG     EQU (0x400A40DC) ;- (UDPHS) UDPHS Test Mode Register
AT91C_UDPHS_TST           EQU (0x400A40E0) ;- (UDPHS) UDPHS Test Register
AT91C_UDPHS_TSTSOFCNT     EQU (0x400A40D0) ;- (UDPHS) UDPHS Test SOF Counter Register
AT91C_UDPHS_FNUM          EQU (0x400A4004) ;- (UDPHS) UDPHS Frame Number Register
AT91C_UDPHS_TSTCNTA       EQU (0x400A40D4) ;- (UDPHS) UDPHS Test A Counter Register
AT91C_UDPHS_INTSTA        EQU (0x400A4014) ;- (UDPHS) UDPHS Interrupt Status Register
AT91C_UDPHS_IPFEATURES    EQU (0x400A40F8) ;- (UDPHS) UDPHS Features Register
AT91C_UDPHS_CLRINT        EQU (0x400A4018) ;- (UDPHS) UDPHS Clear Interrupt Register
AT91C_UDPHS_RIPNAME1      EQU (0x400A40F0) ;- (UDPHS) UDPHS Name1 Register
AT91C_UDPHS_CTRL          EQU (0x400A4000) ;- (UDPHS) UDPHS Control Register
AT91C_UDPHS_IPVERSION     EQU (0x400A40FC) ;- (UDPHS) UDPHS Version Register
// - ========== Register definition for HDMA_CH_0 peripheral ========== 
AT91C_HDMA_CH_0_SADDR     EQU (0x400B003C) ;- (HDMA_CH_0) HDMA Channel Source Address Register
AT91C_HDMA_CH_0_DADDR     EQU (0x400B0040) ;- (HDMA_CH_0) HDMA Channel Destination Address Register
AT91C_HDMA_CH_0_CFG       EQU (0x400B0050) ;- (HDMA_CH_0) HDMA Channel Configuration Register
AT91C_HDMA_CH_0_CTRLB     EQU (0x400B004C) ;- (HDMA_CH_0) HDMA Channel Control B Register
AT91C_HDMA_CH_0_CTRLA     EQU (0x400B0048) ;- (HDMA_CH_0) HDMA Channel Control A Register
AT91C_HDMA_CH_0_DSCR      EQU (0x400B0044) ;- (HDMA_CH_0) HDMA Channel Descriptor Address Register
// - ========== Register definition for HDMA_CH_1 peripheral ========== 
AT91C_HDMA_CH_1_DSCR      EQU (0x400B006C) ;- (HDMA_CH_1) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_1_CTRLB     EQU (0x400B0074) ;- (HDMA_CH_1) HDMA Channel Control B Register
AT91C_HDMA_CH_1_SADDR     EQU (0x400B0064) ;- (HDMA_CH_1) HDMA Channel Source Address Register
AT91C_HDMA_CH_1_CFG       EQU (0x400B0078) ;- (HDMA_CH_1) HDMA Channel Configuration Register
AT91C_HDMA_CH_1_DADDR     EQU (0x400B0068) ;- (HDMA_CH_1) HDMA Channel Destination Address Register
AT91C_HDMA_CH_1_CTRLA     EQU (0x400B0070) ;- (HDMA_CH_1) HDMA Channel Control A Register
// - ========== Register definition for HDMA_CH_2 peripheral ========== 
AT91C_HDMA_CH_2_CTRLA     EQU (0x400B0098) ;- (HDMA_CH_2) HDMA Channel Control A Register
AT91C_HDMA_CH_2_SADDR     EQU (0x400B008C) ;- (HDMA_CH_2) HDMA Channel Source Address Register
AT91C_HDMA_CH_2_CTRLB     EQU (0x400B009C) ;- (HDMA_CH_2) HDMA Channel Control B Register
AT91C_HDMA_CH_2_DADDR     EQU (0x400B0090) ;- (HDMA_CH_2) HDMA Channel Destination Address Register
AT91C_HDMA_CH_2_CFG       EQU (0x400B00A0) ;- (HDMA_CH_2) HDMA Channel Configuration Register
AT91C_HDMA_CH_2_DSCR      EQU (0x400B0094) ;- (HDMA_CH_2) HDMA Channel Descriptor Address Register
// - ========== Register definition for HDMA_CH_3 peripheral ========== 
AT91C_HDMA_CH_3_DSCR      EQU (0x400B00BC) ;- (HDMA_CH_3) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_3_SADDR     EQU (0x400B00B4) ;- (HDMA_CH_3) HDMA Channel Source Address Register
AT91C_HDMA_CH_3_CTRLB     EQU (0x400B00C4) ;- (HDMA_CH_3) HDMA Channel Control B Register
AT91C_HDMA_CH_3_CFG       EQU (0x400B00C8) ;- (HDMA_CH_3) HDMA Channel Configuration Register
AT91C_HDMA_CH_3_DADDR     EQU (0x400B00B8) ;- (HDMA_CH_3) HDMA Channel Destination Address Register
AT91C_HDMA_CH_3_CTRLA     EQU (0x400B00C0) ;- (HDMA_CH_3) HDMA Channel Control A Register
// - ========== Register definition for HDMA peripheral ========== 
AT91C_HDMA_VER            EQU (0x400B01FC) ;- (HDMA) HDMA VERSION REGISTER 
AT91C_HDMA_CHSR           EQU (0x400B0030) ;- (HDMA) HDMA Channel Handler Status Register
AT91C_HDMA_IPNAME2        EQU (0x400B01F4) ;- (HDMA) HDMA IPNAME2 REGISTER 
AT91C_HDMA_EBCIMR         EQU (0x400B0020) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
AT91C_HDMA_CHDR           EQU (0x400B002C) ;- (HDMA) HDMA Channel Handler Disable Register
AT91C_HDMA_EN             EQU (0x400B0004) ;- (HDMA) HDMA Controller Enable Register
AT91C_HDMA_GCFG           EQU (0x400B0000) ;- (HDMA) HDMA Global Configuration Register
AT91C_HDMA_IPNAME1        EQU (0x400B01F0) ;- (HDMA) HDMA IPNAME1 REGISTER 
AT91C_HDMA_LAST           EQU (0x400B0010) ;- (HDMA) HDMA Software Last Transfer Flag Register
AT91C_HDMA_FEATURES       EQU (0x400B01F8) ;- (HDMA) HDMA FEATURES REGISTER 
AT91C_HDMA_CREQ           EQU (0x400B000C) ;- (HDMA) HDMA Software Chunk Transfer Request Register
AT91C_HDMA_EBCIER         EQU (0x400B0018) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
AT91C_HDMA_CHER           EQU (0x400B0028) ;- (HDMA) HDMA Channel Handler Enable Register
AT91C_HDMA_ADDRSIZE       EQU (0x400B01EC) ;- (HDMA) HDMA ADDRSIZE REGISTER 
AT91C_HDMA_EBCISR         EQU (0x400B0024) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
AT91C_HDMA_SREQ           EQU (0x400B0008) ;- (HDMA) HDMA Software Single Request Register
AT91C_HDMA_EBCIDR         EQU (0x400B001C) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register

// - *****************************************************************************
// -               PIO DEFINITIONS FOR AT91SAM3U1
// - *****************************************************************************
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PA0_TIOB0           EQU (AT91C_PIO_PA0) ;-  
AT91C_PA0_SPI0_NPCS1      EQU (AT91C_PIO_PA0) ;-  
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PA1_TIOA0           EQU (AT91C_PIO_PA1) ;-  
AT91C_PA1_SPI0_NPCS2      EQU (AT91C_PIO_PA1) ;-  
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PA10_TWCK0          EQU (AT91C_PIO_PA10) ;-  
AT91C_PA10_PWML3          EQU (AT91C_PIO_PA10) ;-  
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PA11_DRXD           EQU (AT91C_PIO_PA11) ;-  
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PA12_DTXD           EQU (AT91C_PIO_PA12) ;-  
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PA13_SPI0_MISO      EQU (AT91C_PIO_PA13) ;-  
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PA14_SPI0_MOSI      EQU (AT91C_PIO_PA14) ;-  
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PA15_SPI0_SPCK      EQU (AT91C_PIO_PA15) ;-  
AT91C_PA15_PWMH2          EQU (AT91C_PIO_PA15) ;-  
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PA16_SPI0_NPCS0     EQU (AT91C_PIO_PA16) ;-  
AT91C_PA16_NCS1           EQU (AT91C_PIO_PA16) ;-  
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PA17_SCK0           EQU (AT91C_PIO_PA17) ;-  
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PA18_TXD0           EQU (AT91C_PIO_PA18) ;-  
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PA19_RXD0           EQU (AT91C_PIO_PA19) ;-  
AT91C_PA19_SPI0_NPCS3     EQU (AT91C_PIO_PA19) ;-  
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PA2_TCLK0           EQU (AT91C_PIO_PA2) ;-  
AT91C_PA2_ADTRG1          EQU (AT91C_PIO_PA2) ;-  
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PA20_TXD1           EQU (AT91C_PIO_PA20) ;-  
AT91C_PA20_PWMH3          EQU (AT91C_PIO_PA20) ;-  
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PA21_RXD1           EQU (AT91C_PIO_PA21) ;-  
AT91C_PA21_PCK0           EQU (AT91C_PIO_PA21) ;-  
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PA22_TXD2           EQU (AT91C_PIO_PA22) ;-  
AT91C_PA22_RTS1           EQU (AT91C_PIO_PA22) ;-  
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PA23_RXD2           EQU (AT91C_PIO_PA23) ;-  
AT91C_PA23_CTS1           EQU (AT91C_PIO_PA23) ;-  
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PA24_TWD1           EQU (AT91C_PIO_PA24) ;-  
AT91C_PA24_SCK1           EQU (AT91C_PIO_PA24) ;-  
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PA25_TWCK1          EQU (AT91C_PIO_PA25) ;-  
AT91C_PA25_SCK2           EQU (AT91C_PIO_PA25) ;-  
AT91C_PIO_PA26            EQU (1 << 26) ;- Pin Controlled by PA26
AT91C_PA26_TD0            EQU (AT91C_PIO_PA26) ;-  
AT91C_PA26_TCLK2          EQU (AT91C_PIO_PA26) ;-  
AT91C_PIO_PA27            EQU (1 << 27) ;- Pin Controlled by PA27
AT91C_PA27_RD0            EQU (AT91C_PIO_PA27) ;-  
AT91C_PA27_PCK0           EQU (AT91C_PIO_PA27) ;-  
AT91C_PIO_PA28            EQU (1 << 28) ;- Pin Controlled by PA28
AT91C_PA28_TK0            EQU (AT91C_PIO_PA28) ;-  
AT91C_PA28_PWMH0          EQU (AT91C_PIO_PA28) ;-  
AT91C_PIO_PA29            EQU (1 << 29) ;- Pin Controlled by PA29
AT91C_PA29_RK0            EQU (AT91C_PIO_PA29) ;-  
AT91C_PA29_PWMH1          EQU (AT91C_PIO_PA29) ;-  
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PA3_MCI0_CK         EQU (AT91C_PIO_PA3) ;-  
AT91C_PA3_PCK1            EQU (AT91C_PIO_PA3) ;-  
AT91C_PIO_PA30            EQU (1 << 30) ;- Pin Controlled by PA30
AT91C_PA30_TF0            EQU (AT91C_PIO_PA30) ;-  
AT91C_PA30_TIOA2          EQU (AT91C_PIO_PA30) ;-  
AT91C_PIO_PA31            EQU (1 << 31) ;- Pin Controlled by PA31
AT91C_PA31_RF0            EQU (AT91C_PIO_PA31) ;-  
AT91C_PA31_TIOB2          EQU (AT91C_PIO_PA31) ;-  
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PA4_MCI0_CDA        EQU (AT91C_PIO_PA4) ;-  
AT91C_PA4_PWMH0           EQU (AT91C_PIO_PA4) ;-  
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PA5_MCI0_DA0        EQU (AT91C_PIO_PA5) ;-  
AT91C_PA5_PWMH1           EQU (AT91C_PIO_PA5) ;-  
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PA6_MCI0_DA1        EQU (AT91C_PIO_PA6) ;-  
AT91C_PA6_PWMH2           EQU (AT91C_PIO_PA6) ;-  
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PA7_MCI0_DA2        EQU (AT91C_PIO_PA7) ;-  
AT91C_PA7_PWML0           EQU (AT91C_PIO_PA7) ;-  
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PA8_MCI0_DA3        EQU (AT91C_PIO_PA8) ;-  
AT91C_PA8_PWML1           EQU (AT91C_PIO_PA8) ;-  
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PA9_TWD0            EQU (AT91C_PIO_PA9) ;-  
AT91C_PA9_PWML2           EQU (AT91C_PIO_PA9) ;-  
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PB0_PWMH0           EQU (AT91C_PIO_PB0) ;-  
AT91C_PB0_A2              EQU (AT91C_PIO_PB0) ;-  
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PB1_PWMH1           EQU (AT91C_PIO_PB1) ;-  
AT91C_PB1_A3              EQU (AT91C_PIO_PB1) ;-  
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PB10_D1             EQU (AT91C_PIO_PB10) ;-  
AT91C_PB10_DSR0           EQU (AT91C_PIO_PB10) ;-  
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PB11_D2             EQU (AT91C_PIO_PB11) ;-  
AT91C_PB11_DCD0           EQU (AT91C_PIO_PB11) ;-  
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_D3             EQU (AT91C_PIO_PB12) ;-  
AT91C_PB12_RI0            EQU (AT91C_PIO_PB12) ;-  
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_D4             EQU (AT91C_PIO_PB13) ;-  
AT91C_PB13_PWMH0          EQU (AT91C_PIO_PB13) ;-  
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_D5             EQU (AT91C_PIO_PB14) ;-  
AT91C_PB14_PWMH1          EQU (AT91C_PIO_PB14) ;-  
AT91C_PIO_PB15            EQU (1 << 15) ;- Pin Controlled by PB15
AT91C_PB15_D6             EQU (AT91C_PIO_PB15) ;-  
AT91C_PB15_PWMH2          EQU (AT91C_PIO_PB15) ;-  
AT91C_PIO_PB16            EQU (1 << 16) ;- Pin Controlled by PB16
AT91C_PB16_D7             EQU (AT91C_PIO_PB16) ;-  
AT91C_PB16_PWMH3          EQU (AT91C_PIO_PB16) ;-  
AT91C_PIO_PB17            EQU (1 << 17) ;- Pin Controlled by PB17
AT91C_PB17_NANDOE         EQU (AT91C_PIO_PB17) ;-  
AT91C_PB17_PWML0          EQU (AT91C_PIO_PB17) ;-  
AT91C_PIO_PB18            EQU (1 << 18) ;- Pin Controlled by PB18
AT91C_PB18_NANDWE         EQU (AT91C_PIO_PB18) ;-  
AT91C_PB18_PWML1          EQU (AT91C_PIO_PB18) ;-  
AT91C_PIO_PB19            EQU (1 << 19) ;- Pin Controlled by PB19
AT91C_PB19_NRD            EQU (AT91C_PIO_PB19) ;-  
AT91C_PB19_PWML2          EQU (AT91C_PIO_PB19) ;-  
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PB2_PWMH2           EQU (AT91C_PIO_PB2) ;-  
AT91C_PB2_A4              EQU (AT91C_PIO_PB2) ;-  
AT91C_PIO_PB20            EQU (1 << 20) ;- Pin Controlled by PB20
AT91C_PB20_NCS0           EQU (AT91C_PIO_PB20) ;-  
AT91C_PB20_PWML3          EQU (AT91C_PIO_PB20) ;-  
AT91C_PIO_PB21            EQU (1 << 21) ;- Pin Controlled by PB21
AT91C_PB21_A21_NANDALE    EQU (AT91C_PIO_PB21) ;-  
AT91C_PB21_RTS2           EQU (AT91C_PIO_PB21) ;-  
AT91C_PIO_PB22            EQU (1 << 22) ;- Pin Controlled by PB22
AT91C_PB22_A22_NANDCLE    EQU (AT91C_PIO_PB22) ;-  
AT91C_PB22_CTS2           EQU (AT91C_PIO_PB22) ;-  
AT91C_PIO_PB23            EQU (1 << 23) ;- Pin Controlled by PB23
AT91C_PB23_NWR0_NWE       EQU (AT91C_PIO_PB23) ;-  
AT91C_PB23_PCK2           EQU (AT91C_PIO_PB23) ;-  
AT91C_PIO_PB24            EQU (1 << 24) ;- Pin Controlled by PB24
AT91C_PB24_NANDRDY        EQU (AT91C_PIO_PB24) ;-  
AT91C_PB24_PCK1           EQU (AT91C_PIO_PB24) ;-  
AT91C_PIO_PB25            EQU (1 << 25) ;- Pin Controlled by PB25
AT91C_PB25_D8             EQU (AT91C_PIO_PB25) ;-  
AT91C_PB25_PWML0          EQU (AT91C_PIO_PB25) ;-  
AT91C_PIO_PB26            EQU (1 << 26) ;- Pin Controlled by PB26
AT91C_PB26_D9             EQU (AT91C_PIO_PB26) ;-  
AT91C_PB26_PWML1          EQU (AT91C_PIO_PB26) ;-  
AT91C_PIO_PB27            EQU (1 << 27) ;- Pin Controlled by PB27
AT91C_PB27_D10            EQU (AT91C_PIO_PB27) ;-  
AT91C_PB27_PWML2          EQU (AT91C_PIO_PB27) ;-  
AT91C_PIO_PB28            EQU (1 << 28) ;- Pin Controlled by PB28
AT91C_PB28_D11            EQU (AT91C_PIO_PB28) ;-  
AT91C_PB28_PWML3          EQU (AT91C_PIO_PB28) ;-  
AT91C_PIO_PB29            EQU (1 << 29) ;- Pin Controlled by PB29
AT91C_PB29_D12            EQU (AT91C_PIO_PB29) ;-  
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PB3_PWMH3           EQU (AT91C_PIO_PB3) ;-  
AT91C_PB3_A5              EQU (AT91C_PIO_PB3) ;-  
AT91C_PIO_PB30            EQU (1 << 30) ;- Pin Controlled by PB30
AT91C_PB30_D13            EQU (AT91C_PIO_PB30) ;-  
AT91C_PIO_PB31            EQU (1 << 31) ;- Pin Controlled by PB31
AT91C_PB31_D14            EQU (AT91C_PIO_PB31) ;-  
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PB4_TCLK1           EQU (AT91C_PIO_PB4) ;-  
AT91C_PB4_A6              EQU (AT91C_PIO_PB4) ;-  
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PB5_TIOA1           EQU (AT91C_PIO_PB5) ;-  
AT91C_PB5_A7              EQU (AT91C_PIO_PB5) ;-  
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PB6_TIOB1           EQU (AT91C_PIO_PB6) ;-  
AT91C_PB6_D15             EQU (AT91C_PIO_PB6) ;-  
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PB7_RTS0            EQU (AT91C_PIO_PB7) ;-  
AT91C_PB7_A0_NBS0         EQU (AT91C_PIO_PB7) ;-  
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PB8_CTS0            EQU (AT91C_PIO_PB8) ;-  
AT91C_PB8_A1              EQU (AT91C_PIO_PB8) ;-  
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PB9_D0              EQU (AT91C_PIO_PB9) ;-  
AT91C_PB9_DTR0            EQU (AT91C_PIO_PB9) ;-  
AT91C_PIO_PC0             EQU (1 <<  0) ;- Pin Controlled by PC0
AT91C_PC0_A2              EQU (AT91C_PIO_PC0) ;-  
AT91C_PIO_PC1             EQU (1 <<  1) ;- Pin Controlled by PC1
AT91C_PC1_A3              EQU (AT91C_PIO_PC1) ;-  
AT91C_PIO_PC10            EQU (1 << 10) ;- Pin Controlled by PC10
AT91C_PC10_A12            EQU (AT91C_PIO_PC10) ;-  
AT91C_PC10_CTS3           EQU (AT91C_PIO_PC10) ;-  
AT91C_PIO_PC11            EQU (1 << 11) ;- Pin Controlled by PC11
AT91C_PC11_A13            EQU (AT91C_PIO_PC11) ;-  
AT91C_PC11_RTS3           EQU (AT91C_PIO_PC11) ;-  
AT91C_PIO_PC12            EQU (1 << 12) ;- Pin Controlled by PC12
AT91C_PC12_NCS1           EQU (AT91C_PIO_PC12) ;-  
AT91C_PC12_TXD3           EQU (AT91C_PIO_PC12) ;-  
AT91C_PIO_PC13            EQU (1 << 13) ;- Pin Controlled by PC13
AT91C_PC13_A2             EQU (AT91C_PIO_PC13) ;-  
AT91C_PC13_RXD3           EQU (AT91C_PIO_PC13) ;-  
AT91C_PIO_PC14            EQU (1 << 14) ;- Pin Controlled by PC14
AT91C_PC14_A3             EQU (AT91C_PIO_PC14) ;-  
AT91C_PC14_SPI0_NPCS2     EQU (AT91C_PIO_PC14) ;-  
AT91C_PIO_PC15            EQU (1 << 15) ;- Pin Controlled by PC15
AT91C_PC15_NWR1_NBS1      EQU (AT91C_PIO_PC15) ;-  
AT91C_PIO_PC16            EQU (1 << 16) ;- Pin Controlled by PC16
AT91C_PC16_NCS2           EQU (AT91C_PIO_PC16) ;-  
AT91C_PC16_PWML3          EQU (AT91C_PIO_PC16) ;-  
AT91C_PIO_PC17            EQU (1 << 17) ;- Pin Controlled by PC17
AT91C_PC17_NCS3           EQU (AT91C_PIO_PC17) ;-  
AT91C_PC17_A24            EQU (AT91C_PIO_PC17) ;-  
AT91C_PIO_PC18            EQU (1 << 18) ;- Pin Controlled by PC18
AT91C_PC18_NWAIT          EQU (AT91C_PIO_PC18) ;-  
AT91C_PIO_PC19            EQU (1 << 19) ;- Pin Controlled by PC19
AT91C_PC19_SCK3           EQU (AT91C_PIO_PC19) ;-  
AT91C_PC19_NPCS1          EQU (AT91C_PIO_PC19) ;-  
AT91C_PIO_PC2             EQU (1 <<  2) ;- Pin Controlled by PC2
AT91C_PC2_A4              EQU (AT91C_PIO_PC2) ;-  
AT91C_PIO_PC20            EQU (1 << 20) ;- Pin Controlled by PC20
AT91C_PC20_A14            EQU (AT91C_PIO_PC20) ;-  
AT91C_PIO_PC21            EQU (1 << 21) ;- Pin Controlled by PC21
AT91C_PC21_A15            EQU (AT91C_PIO_PC21) ;-  
AT91C_PIO_PC22            EQU (1 << 22) ;- Pin Controlled by PC22
AT91C_PC22_A16            EQU (AT91C_PIO_PC22) ;-  
AT91C_PIO_PC23            EQU (1 << 23) ;- Pin Controlled by PC23
AT91C_PC23_A17            EQU (AT91C_PIO_PC23) ;-  
AT91C_PIO_PC24            EQU (1 << 24) ;- Pin Controlled by PC24
AT91C_PC24_A18            EQU (AT91C_PIO_PC24) ;-  
AT91C_PC24_PWMH0          EQU (AT91C_PIO_PC24) ;-  
AT91C_PIO_PC25            EQU (1 << 25) ;- Pin Controlled by PC25
AT91C_PC25_A19            EQU (AT91C_PIO_PC25) ;-  
AT91C_PC25_PWMH1          EQU (AT91C_PIO_PC25) ;-  
AT91C_PIO_PC26            EQU (1 << 26) ;- Pin Controlled by PC26
AT91C_PC26_A20            EQU (AT91C_PIO_PC26) ;-  
AT91C_PC26_PWMH2          EQU (AT91C_PIO_PC26) ;-  
AT91C_PIO_PC27            EQU (1 << 27) ;- Pin Controlled by PC27
AT91C_PC27_A23            EQU (AT91C_PIO_PC27) ;-  
AT91C_PC27_PWMH3          EQU (AT91C_PIO_PC27) ;-  
AT91C_PIO_PC28            EQU (1 << 28) ;- Pin Controlled by PC28
AT91C_PC28_A24            EQU (AT91C_PIO_PC28) ;-  
AT91C_PC28_MCI0_DA4       EQU (AT91C_PIO_PC28) ;-  
AT91C_PIO_PC29            EQU (1 << 29) ;- Pin Controlled by PC29
AT91C_PC29_PWML0          EQU (AT91C_PIO_PC29) ;-  
AT91C_PC29_MCI0_DA5       EQU (AT91C_PIO_PC29) ;-  
AT91C_PIO_PC3             EQU (1 <<  3) ;- Pin Controlled by PC3
AT91C_PC3_A5              EQU (AT91C_PIO_PC3) ;-  
AT91C_PC3_SPI0_NPCS1      EQU (AT91C_PIO_PC3) ;-  
AT91C_PIO_PC30            EQU (1 << 30) ;- Pin Controlled by PC30
AT91C_PC30_PWML1          EQU (AT91C_PIO_PC30) ;-  
AT91C_PC30_MCI0_DA6       EQU (AT91C_PIO_PC30) ;-  
AT91C_PIO_PC31            EQU (1 << 31) ;- Pin Controlled by PC31
AT91C_PC31_PWML2          EQU (AT91C_PIO_PC31) ;-  
AT91C_PC31_MCI0_DA7       EQU (AT91C_PIO_PC31) ;-  
AT91C_PIO_PC4             EQU (1 <<  4) ;- Pin Controlled by PC4
AT91C_PC4_A6              EQU (AT91C_PIO_PC4) ;-  
AT91C_PC4_SPI0_NPCS2      EQU (AT91C_PIO_PC4) ;-  
AT91C_PIO_PC5             EQU (1 <<  5) ;- Pin Controlled by PC5
AT91C_PC5_A7              EQU (AT91C_PIO_PC5) ;-  
AT91C_PC5_SPI0_NPCS3      EQU (AT91C_PIO_PC5) ;-  
AT91C_PIO_PC6             EQU (1 <<  6) ;- Pin Controlled by PC6
AT91C_PC6_A8              EQU (AT91C_PIO_PC6) ;-  
AT91C_PC6_PWML0           EQU (AT91C_PIO_PC6) ;-  
AT91C_PIO_PC7             EQU (1 <<  7) ;- Pin Controlled by PC7
AT91C_PC7_A9              EQU (AT91C_PIO_PC7) ;-  
AT91C_PC7_PWML1           EQU (AT91C_PIO_PC7) ;-  
AT91C_PIO_PC8             EQU (1 <<  8) ;- Pin Controlled by PC8
AT91C_PC8_A10             EQU (AT91C_PIO_PC8) ;-  
AT91C_PC8_PWML2           EQU (AT91C_PIO_PC8) ;-  
AT91C_PIO_PC9             EQU (1 <<  9) ;- Pin Controlled by PC9
AT91C_PC9_A11             EQU (AT91C_PIO_PC9) ;-  
AT91C_PC9_PWML3           EQU (AT91C_PIO_PC9) ;-  

// - *****************************************************************************
// -               PERIPHERAL ID DEFINITIONS FOR AT91SAM3U1
// - *****************************************************************************
AT91C_ID_SUPC             EQU ( 0) ;- SUPPLY CONTROLLER
AT91C_ID_RSTC             EQU ( 1) ;- RESET CONTROLLER
AT91C_ID_RTC              EQU ( 2) ;- REAL TIME CLOCK
AT91C_ID_RTT              EQU ( 3) ;- REAL TIME TIMER
AT91C_ID_WDG              EQU ( 4) ;- WATCHDOG TIMER
AT91C_ID_PMC              EQU ( 5) ;- PMC
AT91C_ID_EFC0             EQU ( 6) ;- EFC0
AT91C_ID_EFC1             EQU ( 7) ;- EFC1
AT91C_ID_DBGU             EQU ( 8) ;- DBGU
AT91C_ID_HSMC4            EQU ( 9) ;- HSMC4
AT91C_ID_PIOA             EQU (10) ;- Parallel IO Controller A
AT91C_ID_PIOB             EQU (11) ;- Parallel IO Controller B
AT91C_ID_PIOC             EQU (12) ;- Parallel IO Controller C
AT91C_ID_US0              EQU (13) ;- USART 0
AT91C_ID_US1              EQU (14) ;- USART 1
AT91C_ID_US2              EQU (15) ;- USART 2
AT91C_ID_US3              EQU (16) ;- USART 3
AT91C_ID_MCI0             EQU (17) ;- Multimedia Card Interface
AT91C_ID_TWI0             EQU (18) ;- TWI 0
AT91C_ID_TWI1             EQU (19) ;- TWI 1
AT91C_ID_SPI0             EQU (20) ;- Serial Peripheral Interface
AT91C_ID_SSC0             EQU (21) ;- Serial Synchronous Controller 0
AT91C_ID_TC0              EQU (22) ;- Timer Counter 0
AT91C_ID_TC1              EQU (23) ;- Timer Counter 1
AT91C_ID_TC2              EQU (24) ;- Timer Counter 2
AT91C_ID_PWMC             EQU (25) ;- Pulse Width Modulation Controller
AT91C_ID_ADC              EQU (27) ;- ADC controller
AT91C_ID_HDMA             EQU (28) ;- HDMA
AT91C_ID_UDPHS            EQU (29) ;- USB Device High Speed
AT91C_ALL_INT             EQU (0x3BFFFFFF) ;- ALL VALID INTERRUPTS

// - *****************************************************************************
// -               BASE ADDRESS DEFINITIONS FOR AT91SAM3U1
// - *****************************************************************************
AT91C_BASE_SYS            EQU (0x400E0000) ;- (SYS) Base Address
AT91C_BASE_HSMC4_CS0      EQU (0x400E0070) ;- (HSMC4_CS0) Base Address
AT91C_BASE_HSMC4_CS1      EQU (0x400E0084) ;- (HSMC4_CS1) Base Address
AT91C_BASE_HSMC4_CS2      EQU (0x400E0098) ;- (HSMC4_CS2) Base Address
AT91C_BASE_HSMC4_CS3      EQU (0x400E00AC) ;- (HSMC4_CS3) Base Address
AT91C_BASE_HSMC4_NFC      EQU (0x400E00FC) ;- (HSMC4_NFC) Base Address
AT91C_BASE_HSMC4          EQU (0x400E0000) ;- (HSMC4) Base Address
AT91C_BASE_MATRIX         EQU (0x400E0200) ;- (MATRIX) Base Address
AT91C_BASE_NVIC           EQU (0xE000E000) ;- (NVIC) Base Address
AT91C_BASE_MPU            EQU (0xE000ED90) ;- (MPU) Base Address
AT91C_BASE_CM3            EQU (0xE000ED00) ;- (CM3) Base Address
AT91C_BASE_PDC_DBGU       EQU (0x400E0700) ;- (PDC_DBGU) Base Address
AT91C_BASE_DBGU           EQU (0x400E0600) ;- (DBGU) Base Address
AT91C_BASE_PIOA           EQU (0x400E0C00) ;- (PIOA) Base Address
AT91C_BASE_PIOB           EQU (0x400E0E00) ;- (PIOB) Base Address
AT91C_BASE_PIOC           EQU (0x400E1000) ;- (PIOC) Base Address
AT91C_BASE_PMC            EQU (0x400E0400) ;- (PMC) Base Address
AT91C_BASE_CKGR           EQU (0x400E041C) ;- (CKGR) Base Address
AT91C_BASE_RSTC           EQU (0x400E1200) ;- (RSTC) Base Address
AT91C_BASE_SUPC           EQU (0x400E1210) ;- (SUPC) Base Address
AT91C_BASE_RTTC           EQU (0x400E1230) ;- (RTTC) Base Address
AT91C_BASE_WDTC           EQU (0x400E1250) ;- (WDTC) Base Address
AT91C_BASE_RTC            EQU (0x400E1260) ;- (RTC) Base Address
AT91C_BASE_ADC0           EQU (0x400AC000) ;- (ADC0) Base Address
AT91C_BASE_TC0            EQU (0x40080000) ;- (TC0) Base Address
AT91C_BASE_TC1            EQU (0x40080040) ;- (TC1) Base Address
AT91C_BASE_TC2            EQU (0x40080080) ;- (TC2) Base Address
AT91C_BASE_TCB0           EQU (0x40080000) ;- (TCB0) Base Address
AT91C_BASE_TCB1           EQU (0x40080040) ;- (TCB1) Base Address
AT91C_BASE_TCB2           EQU (0x40080080) ;- (TCB2) Base Address
AT91C_BASE_EFC0           EQU (0x400E0800) ;- (EFC0) Base Address
AT91C_BASE_EFC1           EQU (0x400E0A00) ;- (EFC1) Base Address
AT91C_BASE_MCI0           EQU (0x40000000) ;- (MCI0) Base Address
AT91C_BASE_PDC_TWI0       EQU (0x40084100) ;- (PDC_TWI0) Base Address
AT91C_BASE_PDC_TWI1       EQU (0x40088100) ;- (PDC_TWI1) Base Address
AT91C_BASE_TWI0           EQU (0x40084000) ;- (TWI0) Base Address
AT91C_BASE_TWI1           EQU (0x40088000) ;- (TWI1) Base Address
AT91C_BASE_PDC_US0        EQU (0x40090100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0x40090000) ;- (US0) Base Address
AT91C_BASE_PDC_US1        EQU (0x40094100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0x40094000) ;- (US1) Base Address
AT91C_BASE_PDC_US2        EQU (0x40098100) ;- (PDC_US2) Base Address
AT91C_BASE_US2            EQU (0x40098000) ;- (US2) Base Address
AT91C_BASE_PDC_US3        EQU (0x4009C100) ;- (PDC_US3) Base Address
AT91C_BASE_US3            EQU (0x4009C000) ;- (US3) Base Address
AT91C_BASE_PDC_SSC0       EQU (0x40004100) ;- (PDC_SSC0) Base Address
AT91C_BASE_SSC0           EQU (0x40004000) ;- (SSC0) Base Address
AT91C_BASE_PDC_PWMC       EQU (0x4008C100) ;- (PDC_PWMC) Base Address
AT91C_BASE_PWMC_CH0       EQU (0x4008C200) ;- (PWMC_CH0) Base Address
AT91C_BASE_PWMC_CH1       EQU (0x4008C220) ;- (PWMC_CH1) Base Address
AT91C_BASE_PWMC_CH2       EQU (0x4008C240) ;- (PWMC_CH2) Base Address
AT91C_BASE_PWMC_CH3       EQU (0x4008C260) ;- (PWMC_CH3) Base Address
AT91C_BASE_PWMC           EQU (0x4008C000) ;- (PWMC) Base Address
AT91C_BASE_SPI0           EQU (0x40008000) ;- (SPI0) Base Address
AT91C_BASE_UDPHS_EPTFIFO  EQU (0x20180000) ;- (UDPHS_EPTFIFO) Base Address
AT91C_BASE_UDPHS_EPT_0    EQU (0x400A4100) ;- (UDPHS_EPT_0) Base Address
AT91C_BASE_UDPHS_EPT_1    EQU (0x400A4120) ;- (UDPHS_EPT_1) Base Address
AT91C_BASE_UDPHS_EPT_2    EQU (0x400A4140) ;- (UDPHS_EPT_2) Base Address
AT91C_BASE_UDPHS_EPT_3    EQU (0x400A4160) ;- (UDPHS_EPT_3) Base Address
AT91C_BASE_UDPHS_EPT_4    EQU (0x400A4180) ;- (UDPHS_EPT_4) Base Address
AT91C_BASE_UDPHS_EPT_5    EQU (0x400A41A0) ;- (UDPHS_EPT_5) Base Address
AT91C_BASE_UDPHS_EPT_6    EQU (0x400A41C0) ;- (UDPHS_EPT_6) Base Address
AT91C_BASE_UDPHS_DMA_1    EQU (0x400A4310) ;- (UDPHS_DMA_1) Base Address
AT91C_BASE_UDPHS_DMA_2    EQU (0x400A4320) ;- (UDPHS_DMA_2) Base Address
AT91C_BASE_UDPHS_DMA_3    EQU (0x400A4330) ;- (UDPHS_DMA_3) Base Address
AT91C_BASE_UDPHS_DMA_4    EQU (0x400A4340) ;- (UDPHS_DMA_4) Base Address
AT91C_BASE_UDPHS_DMA_5    EQU (0x400A4350) ;- (UDPHS_DMA_5) Base Address
AT91C_BASE_UDPHS_DMA_6    EQU (0x400A4360) ;- (UDPHS_DMA_6) Base Address
AT91C_BASE_UDPHS          EQU (0x400A4000) ;- (UDPHS) Base Address
AT91C_BASE_HDMA_CH_0      EQU (0x400B003C) ;- (HDMA_CH_0) Base Address
AT91C_BASE_HDMA_CH_1      EQU (0x400B0064) ;- (HDMA_CH_1) Base Address
AT91C_BASE_HDMA_CH_2      EQU (0x400B008C) ;- (HDMA_CH_2) Base Address
AT91C_BASE_HDMA_CH_3      EQU (0x400B00B4) ;- (HDMA_CH_3) Base Address
AT91C_BASE_HDMA           EQU (0x400B0000) ;- (HDMA) Base Address

// - *****************************************************************************
// -               MEMORY MAPPING DEFINITIONS FOR AT91SAM3U1
// - *****************************************************************************
// - ITCM
AT91C_ITCM                EQU (0x00100000) ;- Maximum ITCM Area base address
AT91C_ITCM_SIZE           EQU (0x00010000) ;- Maximum ITCM Area size in byte (64 Kbytes)
// - DTCM
AT91C_DTCM                EQU (0x00200000) ;- Maximum DTCM Area base address
AT91C_DTCM_SIZE           EQU (0x00010000) ;- Maximum DTCM Area size in byte (64 Kbytes)
// - IRAM
AT91C_IRAM                EQU (0x20000000) ;- Maximum Internal SRAM base address
AT91C_IRAM_SIZE           EQU (0x00002000) ;- Maximum Internal SRAM size in byte (8 Kbytes)
// - IRAM_MIN
AT91C_IRAM_MIN            EQU (0x00300000) ;- Minimum Internal RAM base address
AT91C_IRAM_MIN_SIZE       EQU (0x00004000) ;- Minimum Internal RAM size in byte (16 Kbytes)
// - IROM
AT91C_IROM                EQU (0x00180000) ;- Internal ROM base address
AT91C_IROM_SIZE           EQU (0x00008000) ;- Internal ROM size in byte (32 Kbytes)
// - IFLASH0
AT91C_IFLASH0             EQU (0x00080000) ;- Maximum IFLASH Area : 64Kbyte base address
AT91C_IFLASH0_SIZE        EQU (0x00010000) ;- Maximum IFLASH Area : 64Kbyte size in byte (64 Kbytes)
AT91C_IFLASH0_PAGE_SIZE   EQU (256) ;- Maximum IFLASH Area : 64Kbyte Page Size: 256 bytes
AT91C_IFLASH0_LOCK_REGION_SIZE EQU (8192) ;- Maximum IFLASH Area : 64Kbyte Lock Region Size: 8 Kbytes
AT91C_IFLASH0_NB_OF_PAGES EQU (512) ;- Maximum IFLASH Area : 64Kbyte Number of Pages: 512 bytes
AT91C_IFLASH0_NB_OF_LOCK_BITS EQU (16) ;- Maximum IFLASH Area : 64Kbyte Number of Lock Bits: 16 bytes

// - EBI_CS0
AT91C_EBI_CS0             EQU (0x10000000) ;- EBI Chip Select 0 base address
AT91C_EBI_CS0_SIZE        EQU (0x10000000) ;- EBI Chip Select 0 size in byte (262144 Kbytes)
// - EBI_CS1
AT91C_EBI_CS1             EQU (0x20000000) ;- EBI Chip Select 1 base address
AT91C_EBI_CS1_SIZE        EQU (0x10000000) ;- EBI Chip Select 1 size in byte (262144 Kbytes)
// - EBI_SDRAM
AT91C_EBI_SDRAM           EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_SIZE      EQU (0x10000000) ;- SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// - EBI_SDRAM_16BIT
AT91C_EBI_SDRAM_16BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_16BIT_SIZE EQU (0x02000000) ;- SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// - EBI_SDRAM_32BIT
AT91C_EBI_SDRAM_32BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_32BIT_SIZE EQU (0x04000000) ;- SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
// - EBI_CS2
AT91C_EBI_CS2             EQU (0x30000000) ;- EBI Chip Select 2 base address
AT91C_EBI_CS2_SIZE        EQU (0x10000000) ;- EBI Chip Select 2 size in byte (262144 Kbytes)
// - EBI_CS3
AT91C_EBI_CS3             EQU (0x40000000) ;- EBI Chip Select 3 base address
AT91C_EBI_CS3_SIZE        EQU (0x10000000) ;- EBI Chip Select 3 size in byte (262144 Kbytes)
// - EBI_SM
AT91C_EBI_SM              EQU (0x40000000) ;- NANDFLASH on EBI Chip Select 3 base address
AT91C_EBI_SM_SIZE         EQU (0x10000000) ;- NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)
// - EBI_CS4
AT91C_EBI_CS4             EQU (0x50000000) ;- EBI Chip Select 4 base address
AT91C_EBI_CS4_SIZE        EQU (0x10000000) ;- EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CF0
AT91C_EBI_CF0             EQU (0x50000000) ;- CompactFlash 0 on EBI Chip Select 4 base address
AT91C_EBI_CF0_SIZE        EQU (0x10000000) ;- CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CS5
AT91C_EBI_CS5             EQU (0x60000000) ;- EBI Chip Select 5 base address
AT91C_EBI_CS5_SIZE        EQU (0x10000000) ;- EBI Chip Select 5 size in byte (262144 Kbytes)
// - EBI_CF1
AT91C_EBI_CF1             EQU (0x60000000) ;- CompactFlash 1 on EBIChip Select 5 base address
AT91C_EBI_CF1_SIZE        EQU (0x10000000) ;- CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91SAM3U1_H */
