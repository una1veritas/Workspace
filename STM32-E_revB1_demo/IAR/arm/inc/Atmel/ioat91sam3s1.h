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
// - File Name           : AT91SAM3S1.h
// - Object              : AT91SAM3S1 definitions
// - Generated           : AT91 SW Application Group  09/30/2009 (16:15:55)
// - 
// - CVS Reference       : /AT91SAM3S1.pl/1.24/Wed Jul 29 14:55:43 2009//
// - CVS Reference       : /ACC_SAM3S4.pl/1.3/Wed Feb 25 13:14:09 2009//
// - CVS Reference       : /ADC_SAM3XE.pl/1.5/Tue Jan 27 10:29:26 2009//
// - CVS Reference       : /CORTEX_M3_MPU.pl/1.3/Mon Sep 15 12:43:55 2008//
// - CVS Reference       : /CORTEX_M3.pl/1.1/Mon Sep 15 15:22:06 2008//
// - CVS Reference       : /CORTEX_M3_NVIC.pl/1.8/Wed Sep 30 14:11:09 2009//
// - CVS Reference       : /DAC_SAM3S4.pl/1.2/Fri Mar 13 08:50:25 2009//
// - CVS Reference       : /DBGU_SAM3UE256.pl/1.3/Wed Sep 30 14:11:10 2009//
// - CVS Reference       : /EFC2_SAM3UE256.pl/1.3/Wed Sep 30 14:11:11 2009//
// - CVS Reference       : /HCBDMA_sam3S4.pl/1.1/Fri Mar 13 09:02:46 2009//
// - CVS Reference       : /HECC_6143A.pl/1.1/Wed Feb  9 17:16:57 2005//
// - CVS Reference       : /HMATRIX2_SAM3UE256.pl/1.5/Wed Sep 30 14:11:13 2009//
// - CVS Reference       : /SFR_SAM3S4.pl/1.3/Wed Jan 14 15:06:20 2009//
// - CVS Reference       : /MCI_6101F.pl/1.3/Wed Sep 30 14:11:15 2009//
// - CVS Reference       : /PDC_6074C.pl/1.2/Thu Feb  3 09:02:11 2005//
// - CVS Reference       : /PIO3_SAM3S4.pl/1.1/Tue Feb 17 08:07:35 2009//
// - CVS Reference       : /PITC_6079A.pl/1.2/Thu Nov  4 13:56:22 2004//
// - CVS Reference       : /PMC_SAM3S4.pl/1.8/Fri Mar 13 08:49:01 2009//
// - CVS Reference       : /PWM_SAM3S4.pl/1.1/Wed Feb 25 13:18:50 2009//
// - CVS Reference       : /RSTC_6098A.pl/1.4/Wed Sep 30 14:11:19 2009//
// - CVS Reference       : /RTC_1245D.pl/1.3/Fri Sep 17 14:01:31 2004//
// - CVS Reference       : /RTTC_6081A.pl/1.2/Thu Nov  4 13:57:22 2004//
// - CVS Reference       : /HSDRAMC1_6100A.pl/1.2/Mon Aug  9 10:52:25 2004//
// - CVS Reference       : /SHDWC_6122A.pl/1.3/Wed Oct  6 14:16:58 2004//
// - CVS Reference       : /HSMC3_SAM3S4.pl/1.1/Fri Mar 13 09:20:27 2009//
// - CVS Reference       : /SPI2.pl/1.6/Wed Sep 30 14:11:20 2009//
// - CVS Reference       : /SSC_SAM3S4.pl/1.1/Thu Apr  9 10:54:32 2009//
// - CVS Reference       : /SUPC_SAM3UE256.pl/1.2/Tue May 27 08:20:16 2008//
// - CVS Reference       : /SYS_SAM3S4.pl/1.4/Thu Mar  5 09:10:26 2009//
// - CVS Reference       : /TC_6082A.pl/1.8/Wed Sep 30 14:11:24 2009//
// - CVS Reference       : /TWI_6061B.pl/1.3/Wed Sep 30 14:11:24 2009//
// - CVS Reference       : /UDP_sam3S4.pl/1.1/Fri Mar 13 09:03:58 2009//
// - CVS Reference       : /US_6089J.pl/1.4/Wed Sep 30 14:11:25 2009//
// - CVS Reference       : /WDTC_6080A.pl/1.3/Thu Nov  4 13:58:52 2004//
// - ----------------------------------------------------------------------------

#ifndef AT91SAM3S1_H
#define AT91SAM3S1_H

#ifdef __IAR_SYSTEMS_ICC__

typedef volatile unsigned int AT91_REG;// Hardware register definition

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR System Peripherals
// *****************************************************************************
typedef struct _AT91S_SYS {
	AT91_REG	 HSMC3_SETUP0; 	//  Setup Register for CS 0
	AT91_REG	 HSMC3_PULSE0; 	//  Pulse Register for CS 0
	AT91_REG	 HSMC3_CYCLE0; 	//  Cycle Register for CS 0
	AT91_REG	 HSMC3_CTRL0; 	//  Control Register for CS 0
	AT91_REG	 HSMC3_SETUP1; 	//  Setup Register for CS 1
	AT91_REG	 HSMC3_PULSE1; 	//  Pulse Register for CS 1
	AT91_REG	 HSMC3_CYCLE1; 	//  Cycle Register for CS 1
	AT91_REG	 HSMC3_CTRL1; 	//  Control Register for CS 1
	AT91_REG	 HSMC3_SETUP2; 	//  Setup Register for CS 2
	AT91_REG	 HSMC3_PULSE2; 	//  Pulse Register for CS 2
	AT91_REG	 HSMC3_CYCLE2; 	//  Cycle Register for CS 2
	AT91_REG	 HSMC3_CTRL2; 	//  Control Register for CS 2
	AT91_REG	 HSMC3_SETUP3; 	//  Setup Register for CS 3
	AT91_REG	 HSMC3_PULSE3; 	//  Pulse Register for CS 3
	AT91_REG	 HSMC3_CYCLE3; 	//  Cycle Register for CS 3
	AT91_REG	 HSMC3_CTRL3; 	//  Control Register for CS 3
	AT91_REG	 HSMC3_SETUP4; 	//  Setup Register for CS 4
	AT91_REG	 HSMC3_PULSE4; 	//  Pulse Register for CS 4
	AT91_REG	 HSMC3_CYCLE4; 	//  Cycle Register for CS 4
	AT91_REG	 HSMC3_CTRL4; 	//  Control Register for CS 4
	AT91_REG	 HSMC3_SETUP5; 	//  Setup Register for CS 5
	AT91_REG	 HSMC3_PULSE5; 	//  Pulse Register for CS 5
	AT91_REG	 HSMC3_CYCLE5; 	//  Cycle Register for CS 5
	AT91_REG	 HSMC3_CTRL5; 	//  Control Register for CS 5
	AT91_REG	 HSMC3_SETUP6; 	//  Setup Register for CS 6
	AT91_REG	 HSMC3_PULSE6; 	//  Pulse Register for CS 6
	AT91_REG	 HSMC3_CYCLE6; 	//  Cycle Register for CS 6
	AT91_REG	 HSMC3_CTRL6; 	//  Control Register for CS 6
	AT91_REG	 HSMC3_SETUP7; 	//  Setup Register for CS 7
	AT91_REG	 HSMC3_PULSE7; 	//  Pulse Register for CS 7
	AT91_REG	 HSMC3_CYCLE7; 	//  Cycle Register for CS 7
	AT91_REG	 HSMC3_CTRL7; 	//  Control Register for CS 7
	AT91_REG	 Reserved0[16]; 	// 
	AT91_REG	 HSMC3_DELAY1; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY2; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY3; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY4; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY5; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY6; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY7; 	// SMC Delay Control Register
	AT91_REG	 HSMC3_DELAY8; 	// SMC Delay Control Register
	AT91_REG	 Reserved1[3]; 	// 
	AT91_REG	 HSMC3_ADDRSIZE; 	// HSMC3 ADDRSIZE REGISTER 
	AT91_REG	 HSMC3_IPNAME1; 	// HSMC3 IPNAME1 REGISTER 
	AT91_REG	 HSMC3_IPNAME2; 	// HSMC3 IPNAME2 REGISTER 
	AT91_REG	 HSMC3_FEATURES; 	// HSMC3 FEATURES REGISTER 
	AT91_REG	 HSMC3_VER; 	// HSMC3 VERSION REGISTER
	AT91_REG	 Reserved2[64]; 	// 
	AT91_REG	 HMATRIX_MCFG0; 	//  Master Configuration Register 0 : ARM I and D
	AT91_REG	 HMATRIX_MCFG1; 	//  Master Configuration Register 1 : ARM S
	AT91_REG	 HMATRIX_MCFG2; 	//  Master Configuration Register 2
	AT91_REG	 HMATRIX_MCFG3; 	//  Master Configuration Register 3
	AT91_REG	 HMATRIX_MCFG4; 	//  Master Configuration Register 4
	AT91_REG	 HMATRIX_MCFG5; 	//  Master Configuration Register 5
	AT91_REG	 HMATRIX_MCFG6; 	//  Master Configuration Register 6
	AT91_REG	 HMATRIX_MCFG7; 	//  Master Configuration Register 7
	AT91_REG	 Reserved3[8]; 	// 
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
	AT91_REG	 Reserved4[42]; 	// 
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
	AT91_REG	 Reserved5[39]; 	// 
	AT91_REG	 HMATRIX_ADDRSIZE; 	// HMATRIX2 ADDRSIZE REGISTER 
	AT91_REG	 HMATRIX_IPNAME1; 	// HMATRIX2 IPNAME1 REGISTER 
	AT91_REG	 HMATRIX_IPNAME2; 	// HMATRIX2 IPNAME2 REGISTER 
	AT91_REG	 HMATRIX_FEATURES; 	// HMATRIX2 FEATURES REGISTER 
	AT91_REG	 HMATRIX_VER; 	// HMATRIX2 VERSION REGISTER 
	AT91_REG	 PMC_SCER; 	// System Clock Enable Register
	AT91_REG	 PMC_SCDR; 	// System Clock Disable Register
	AT91_REG	 PMC_SCSR; 	// System Clock Status Register
	AT91_REG	 Reserved6[1]; 	// 
	AT91_REG	 PMC_PCER; 	// Peripheral Clock Enable Register 0:31 PERI_ID
	AT91_REG	 PMC_PCDR; 	// Peripheral Clock Disable Register 0:31 PERI_ID
	AT91_REG	 PMC_PCSR; 	// Peripheral Clock Status Register 0:31 PERI_ID
	AT91_REG	 PMC_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 PMC_MOR; 	// Main Oscillator Register
	AT91_REG	 PMC_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 PMC_PLLAR; 	// PLL Register
	AT91_REG	 PMC_PLLBR; 	// PLL B Register
	AT91_REG	 PMC_MCKR; 	// Master Clock Register
	AT91_REG	 Reserved7[1]; 	// 
	AT91_REG	 PMC_UDPR; 	// USB DEV Clock Configuration Register
	AT91_REG	 Reserved8[1]; 	// 
	AT91_REG	 PMC_PCKR[8]; 	// Programmable Clock Register
	AT91_REG	 PMC_IER; 	// Interrupt Enable Register
	AT91_REG	 PMC_IDR; 	// Interrupt Disable Register
	AT91_REG	 PMC_SR; 	// Status Register
	AT91_REG	 PMC_IMR; 	// Interrupt Mask Register
	AT91_REG	 PMC_FSMR; 	// Fast Startup Mode Register
	AT91_REG	 PMC_FSPR; 	// Fast Startup Polarity Register
	AT91_REG	 PMC_FOCR; 	// Fault Output Clear Register
	AT91_REG	 Reserved9[28]; 	// 
	AT91_REG	 PMC_ADDRSIZE; 	// PMC ADDRSIZE REGISTER 
	AT91_REG	 PMC_IPNAME1; 	// PMC IPNAME1 REGISTER 
	AT91_REG	 PMC_IPNAME2; 	// PMC IPNAME2 REGISTER 
	AT91_REG	 PMC_FEATURES; 	// PMC FEATURES REGISTER 
	AT91_REG	 PMC_VER; 	// APMC VERSION REGISTER
	AT91_REG	 PMC_PCER1; 	// Peripheral Clock Enable Register 32:63 PERI_ID
	AT91_REG	 PMC_PCDR1; 	// Peripheral Clock Disable Register 32:63 PERI_ID
	AT91_REG	 PMC_PCSR1; 	// Peripheral Clock Status Register 32:63 PERI_ID
	AT91_REG	 PMC_PCR; 	// Peripheral Control Register
	AT91_REG	 Reserved10[60]; 	// 
	AT91_REG	 DBGU0_CR; 	// Control Register
	AT91_REG	 DBGU0_MR; 	// Mode Register
	AT91_REG	 DBGU0_IER; 	// Interrupt Enable Register
	AT91_REG	 DBGU0_IDR; 	// Interrupt Disable Register
	AT91_REG	 DBGU0_IMR; 	// Interrupt Mask Register
	AT91_REG	 DBGU0_CSR; 	// Channel Status Register
	AT91_REG	 DBGU0_RHR; 	// Receiver Holding Register
	AT91_REG	 DBGU0_THR; 	// Transmitter Holding Register
	AT91_REG	 DBGU0_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 Reserved11[9]; 	// 
	AT91_REG	 DBGU0_FNTR; 	// Force NTRST Register
	AT91_REG	 Reserved12[40]; 	// 
	AT91_REG	 DBGU0_ADDRSIZE; 	// DBGU ADDRSIZE REGISTER 
	AT91_REG	 DBGU0_IPNAME1; 	// DBGU IPNAME1 REGISTER 
	AT91_REG	 DBGU0_IPNAME2; 	// DBGU IPNAME2 REGISTER 
	AT91_REG	 DBGU0_FEATURES; 	// DBGU FEATURES REGISTER 
	AT91_REG	 DBGU0_VER; 	// DBGU VERSION REGISTER 
	AT91_REG	 DBGU0_RPR; 	// Receive Pointer Register
	AT91_REG	 DBGU0_RCR; 	// Receive Counter Register
	AT91_REG	 DBGU0_TPR; 	// Transmit Pointer Register
	AT91_REG	 DBGU0_TCR; 	// Transmit Counter Register
	AT91_REG	 DBGU0_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DBGU0_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DBGU0_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DBGU0_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DBGU0_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DBGU0_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved13[6]; 	// 
	AT91_REG	 DBGU0_CIDR; 	// Chip ID Register
	AT91_REG	 DBGU0_EXID; 	// Chip ID Extension Register
	AT91_REG	 Reserved14[46]; 	// 
	AT91_REG	 DBGU1_CR; 	// Control Register
	AT91_REG	 DBGU1_MR; 	// Mode Register
	AT91_REG	 DBGU1_IER; 	// Interrupt Enable Register
	AT91_REG	 DBGU1_IDR; 	// Interrupt Disable Register
	AT91_REG	 DBGU1_IMR; 	// Interrupt Mask Register
	AT91_REG	 DBGU1_CSR; 	// Channel Status Register
	AT91_REG	 DBGU1_RHR; 	// Receiver Holding Register
	AT91_REG	 DBGU1_THR; 	// Transmitter Holding Register
	AT91_REG	 DBGU1_BRGR; 	// Baud Rate Generator Register
	AT91_REG	 Reserved15[9]; 	// 
	AT91_REG	 DBGU1_FNTR; 	// Force NTRST Register
	AT91_REG	 Reserved16[40]; 	// 
	AT91_REG	 DBGU1_ADDRSIZE; 	// DBGU ADDRSIZE REGISTER 
	AT91_REG	 DBGU1_IPNAME1; 	// DBGU IPNAME1 REGISTER 
	AT91_REG	 DBGU1_IPNAME2; 	// DBGU IPNAME2 REGISTER 
	AT91_REG	 DBGU1_FEATURES; 	// DBGU FEATURES REGISTER 
	AT91_REG	 DBGU1_VER; 	// DBGU VERSION REGISTER 
	AT91_REG	 DBGU1_RPR; 	// Receive Pointer Register
	AT91_REG	 DBGU1_RCR; 	// Receive Counter Register
	AT91_REG	 DBGU1_TPR; 	// Transmit Pointer Register
	AT91_REG	 DBGU1_TCR; 	// Transmit Counter Register
	AT91_REG	 DBGU1_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DBGU1_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DBGU1_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DBGU1_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DBGU1_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DBGU1_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved17[6]; 	// 
	AT91_REG	 DBGU1_CIDR; 	// Chip ID Register
	AT91_REG	 DBGU1_EXID; 	// Chip ID Extension Register
	AT91_REG	 Reserved18[46]; 	// 
	AT91_REG	 EFC0_FMR; 	// EFC Flash Mode Register
	AT91_REG	 EFC0_FCR; 	// EFC Flash Command Register
	AT91_REG	 EFC0_FSR; 	// EFC Flash Status Register
	AT91_REG	 EFC0_FRR; 	// EFC Flash Result Register
	AT91_REG	 Reserved19[1]; 	// 
	AT91_REG	 EFC0_FVR; 	// EFC Flash Version Register
	AT91_REG	 Reserved20[250]; 	// 
	AT91_REG	 PIOA_PER; 	// PIO Enable Register
	AT91_REG	 PIOA_PDR; 	// PIO Disable Register
	AT91_REG	 PIOA_PSR; 	// PIO Status Register
	AT91_REG	 Reserved21[1]; 	// 
	AT91_REG	 PIOA_OER; 	// Output Enable Register
	AT91_REG	 PIOA_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOA_OSR; 	// Output Status Register
	AT91_REG	 Reserved22[1]; 	// 
	AT91_REG	 PIOA_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOA_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOA_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved23[1]; 	// 
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
	AT91_REG	 Reserved24[1]; 	// 
	AT91_REG	 PIOA_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOA_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOA_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved25[1]; 	// 
	AT91_REG	 PIOA_SP1; 	// Select B Register
	AT91_REG	 PIOA_SP2; 	// Select B Register
	AT91_REG	 PIOA_ABSR; 	// AB Select Status Register
	AT91_REG	 Reserved26[5]; 	// 
	AT91_REG	 PIOA_PPDDR; 	// Pull-down Disable Register
	AT91_REG	 PIOA_PPDER; 	// Pull-down Enable Register
	AT91_REG	 PIOA_PPDSR; 	// Pull-down Status Register
	AT91_REG	 Reserved27[1]; 	// 
	AT91_REG	 PIOA_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOA_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOA_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved28[16]; 	// 
	AT91_REG	 PIOA_ADDRSIZE; 	// PIO ADDRSIZE REGISTER 
	AT91_REG	 PIOA_IPNAME1; 	// PIO IPNAME1 REGISTER 
	AT91_REG	 PIOA_IPNAME2; 	// PIO IPNAME2 REGISTER 
	AT91_REG	 PIOA_FEATURES; 	// PIO FEATURES REGISTER 
	AT91_REG	 PIOA_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 PIOA_SLEW1; 	// PIO SLEWRATE1 REGISTER 
	AT91_REG	 PIOA_SLEW2; 	// PIO SLEWRATE2 REGISTER 
	AT91_REG	 Reserved29[18]; 	// 
	AT91_REG	 PIOA_SENMR; 	// Sensor Mode Register
	AT91_REG	 PIOA_SENIER; 	// Sensor Interrupt Enable Register
	AT91_REG	 PIOA_SENIDR; 	// Sensor Interrupt Disable Register
	AT91_REG	 PIOA_SENIMR; 	// Sensor Interrupt Mask Register
	AT91_REG	 PIOA_SENISR; 	// Sensor Interrupt Status Register
	AT91_REG	 PIOA_SENDATA; 	// Sensor Data Register
	AT91_REG	 PIOA_RPR; 	// Receive Pointer Register
	AT91_REG	 PIOA_RCR; 	// Receive Counter Register
	AT91_REG	 PIOA_TPR; 	// Transmit Pointer Register
	AT91_REG	 PIOA_TCR; 	// Transmit Counter Register
	AT91_REG	 PIOA_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PIOA_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PIOA_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PIOA_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PIOA_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PIOA_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved30[28]; 	// 
	AT91_REG	 PIOB_PER; 	// PIO Enable Register
	AT91_REG	 PIOB_PDR; 	// PIO Disable Register
	AT91_REG	 PIOB_PSR; 	// PIO Status Register
	AT91_REG	 Reserved31[1]; 	// 
	AT91_REG	 PIOB_OER; 	// Output Enable Register
	AT91_REG	 PIOB_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOB_OSR; 	// Output Status Register
	AT91_REG	 Reserved32[1]; 	// 
	AT91_REG	 PIOB_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOB_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOB_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved33[1]; 	// 
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
	AT91_REG	 Reserved34[1]; 	// 
	AT91_REG	 PIOB_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOB_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOB_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved35[1]; 	// 
	AT91_REG	 PIOB_SP1; 	// Select B Register
	AT91_REG	 PIOB_SP2; 	// Select B Register
	AT91_REG	 PIOB_ABSR; 	// AB Select Status Register
	AT91_REG	 Reserved36[5]; 	// 
	AT91_REG	 PIOB_PPDDR; 	// Pull-down Disable Register
	AT91_REG	 PIOB_PPDER; 	// Pull-down Enable Register
	AT91_REG	 PIOB_PPDSR; 	// Pull-down Status Register
	AT91_REG	 Reserved37[1]; 	// 
	AT91_REG	 PIOB_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOB_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOB_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved38[16]; 	// 
	AT91_REG	 PIOB_ADDRSIZE; 	// PIO ADDRSIZE REGISTER 
	AT91_REG	 PIOB_IPNAME1; 	// PIO IPNAME1 REGISTER 
	AT91_REG	 PIOB_IPNAME2; 	// PIO IPNAME2 REGISTER 
	AT91_REG	 PIOB_FEATURES; 	// PIO FEATURES REGISTER 
	AT91_REG	 PIOB_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 PIOB_SLEW1; 	// PIO SLEWRATE1 REGISTER 
	AT91_REG	 PIOB_SLEW2; 	// PIO SLEWRATE2 REGISTER 
	AT91_REG	 Reserved39[18]; 	// 
	AT91_REG	 PIOB_SENMR; 	// Sensor Mode Register
	AT91_REG	 PIOB_SENIER; 	// Sensor Interrupt Enable Register
	AT91_REG	 PIOB_SENIDR; 	// Sensor Interrupt Disable Register
	AT91_REG	 PIOB_SENIMR; 	// Sensor Interrupt Mask Register
	AT91_REG	 PIOB_SENISR; 	// Sensor Interrupt Status Register
	AT91_REG	 PIOB_SENDATA; 	// Sensor Data Register
	AT91_REG	 PIOB_RPR; 	// Receive Pointer Register
	AT91_REG	 PIOB_RCR; 	// Receive Counter Register
	AT91_REG	 PIOB_TPR; 	// Transmit Pointer Register
	AT91_REG	 PIOB_TCR; 	// Transmit Counter Register
	AT91_REG	 PIOB_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PIOB_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PIOB_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PIOB_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PIOB_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PIOB_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved40[28]; 	// 
	AT91_REG	 PIOC_PER; 	// PIO Enable Register
	AT91_REG	 PIOC_PDR; 	// PIO Disable Register
	AT91_REG	 PIOC_PSR; 	// PIO Status Register
	AT91_REG	 Reserved41[1]; 	// 
	AT91_REG	 PIOC_OER; 	// Output Enable Register
	AT91_REG	 PIOC_ODR; 	// Output Disable Registerr
	AT91_REG	 PIOC_OSR; 	// Output Status Register
	AT91_REG	 Reserved42[1]; 	// 
	AT91_REG	 PIOC_IFER; 	// Input Filter Enable Register
	AT91_REG	 PIOC_IFDR; 	// Input Filter Disable Register
	AT91_REG	 PIOC_IFSR; 	// Input Filter Status Register
	AT91_REG	 Reserved43[1]; 	// 
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
	AT91_REG	 Reserved44[1]; 	// 
	AT91_REG	 PIOC_PPUDR; 	// Pull-up Disable Register
	AT91_REG	 PIOC_PPUER; 	// Pull-up Enable Register
	AT91_REG	 PIOC_PPUSR; 	// Pull-up Status Register
	AT91_REG	 Reserved45[1]; 	// 
	AT91_REG	 PIOC_SP1; 	// Select B Register
	AT91_REG	 PIOC_SP2; 	// Select B Register
	AT91_REG	 PIOC_ABSR; 	// AB Select Status Register
	AT91_REG	 Reserved46[5]; 	// 
	AT91_REG	 PIOC_PPDDR; 	// Pull-down Disable Register
	AT91_REG	 PIOC_PPDER; 	// Pull-down Enable Register
	AT91_REG	 PIOC_PPDSR; 	// Pull-down Status Register
	AT91_REG	 Reserved47[1]; 	// 
	AT91_REG	 PIOC_OWER; 	// Output Write Enable Register
	AT91_REG	 PIOC_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIOC_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved48[16]; 	// 
	AT91_REG	 PIOC_ADDRSIZE; 	// PIO ADDRSIZE REGISTER 
	AT91_REG	 PIOC_IPNAME1; 	// PIO IPNAME1 REGISTER 
	AT91_REG	 PIOC_IPNAME2; 	// PIO IPNAME2 REGISTER 
	AT91_REG	 PIOC_FEATURES; 	// PIO FEATURES REGISTER 
	AT91_REG	 PIOC_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 PIOC_SLEW1; 	// PIO SLEWRATE1 REGISTER 
	AT91_REG	 PIOC_SLEW2; 	// PIO SLEWRATE2 REGISTER 
	AT91_REG	 Reserved49[18]; 	// 
	AT91_REG	 PIOC_SENMR; 	// Sensor Mode Register
	AT91_REG	 PIOC_SENIER; 	// Sensor Interrupt Enable Register
	AT91_REG	 PIOC_SENIDR; 	// Sensor Interrupt Disable Register
	AT91_REG	 PIOC_SENIMR; 	// Sensor Interrupt Mask Register
	AT91_REG	 PIOC_SENISR; 	// Sensor Interrupt Status Register
	AT91_REG	 PIOC_SENDATA; 	// Sensor Data Register
	AT91_REG	 PIOC_RPR; 	// Receive Pointer Register
	AT91_REG	 PIOC_RCR; 	// Receive Counter Register
	AT91_REG	 PIOC_TPR; 	// Transmit Pointer Register
	AT91_REG	 PIOC_TCR; 	// Transmit Counter Register
	AT91_REG	 PIOC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PIOC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PIOC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PIOC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PIOC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PIOC_PTSR; 	// PDC Transfer Status Register
	AT91_REG	 Reserved50[28]; 	// 
	AT91_REG	 RSTC_RCR; 	// Reset Control Register
	AT91_REG	 RSTC_RSR; 	// Reset Status Register
	AT91_REG	 RSTC_RMR; 	// Reset Mode Register
	AT91_REG	 Reserved51[1]; 	// 
	AT91_REG	 SUPC_CR; 	// Control Register
	AT91_REG	 SUPC_BOMR; 	// Brown Out Mode Register
	AT91_REG	 SUPC_MR; 	// Mode Register
	AT91_REG	 SUPC_WUMR; 	// Wake Up Mode Register
	AT91_REG	 SUPC_WUIR; 	// Wake Up Inputs Register
	AT91_REG	 SUPC_SR; 	// Status Register
	AT91_REG	 SUPC_FWUTR; 	// Flash Wake-up Timer Register
	AT91_REG	 Reserved52[1]; 	// 
	AT91_REG	 RTTC_RTMR; 	// Real-time Mode Register
	AT91_REG	 RTTC_RTAR; 	// Real-time Alarm Register
	AT91_REG	 RTTC_RTVR; 	// Real-time Value Register
	AT91_REG	 RTTC_RTSR; 	// Real-time Status Register
	AT91_REG	 Reserved53[4]; 	// 
	AT91_REG	 WDTC_WDCR; 	// Watchdog Control Register
	AT91_REG	 WDTC_WDMR; 	// Watchdog Mode Register
	AT91_REG	 WDTC_WDSR; 	// Watchdog Status Register
	AT91_REG	 Reserved54[1]; 	// 
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
	AT91_REG	 Reserved55[19]; 	// 
	AT91_REG	 RSTC_VER; 	// Version Register
} AT91S_SYS, *AT91PS_SYS;

// -------- GPBR : (SYS Offset: 0x1490) GPBR General Purpose Register -------- 
#define AT91C_GPBR_GPRV       ((unsigned int) 0x0 <<  0) // (SYS) General Purpose Register Value

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface
// *****************************************************************************
typedef struct _AT91S_SMC {
	AT91_REG	 SMC_SETUP0; 	//  Setup Register for CS 0
	AT91_REG	 SMC_PULSE0; 	//  Pulse Register for CS 0
	AT91_REG	 SMC_CYCLE0; 	//  Cycle Register for CS 0
	AT91_REG	 SMC_CTRL0; 	//  Control Register for CS 0
	AT91_REG	 SMC_SETUP1; 	//  Setup Register for CS 1
	AT91_REG	 SMC_PULSE1; 	//  Pulse Register for CS 1
	AT91_REG	 SMC_CYCLE1; 	//  Cycle Register for CS 1
	AT91_REG	 SMC_CTRL1; 	//  Control Register for CS 1
	AT91_REG	 SMC_SETUP2; 	//  Setup Register for CS 2
	AT91_REG	 SMC_PULSE2; 	//  Pulse Register for CS 2
	AT91_REG	 SMC_CYCLE2; 	//  Cycle Register for CS 2
	AT91_REG	 SMC_CTRL2; 	//  Control Register for CS 2
	AT91_REG	 SMC_SETUP3; 	//  Setup Register for CS 3
	AT91_REG	 SMC_PULSE3; 	//  Pulse Register for CS 3
	AT91_REG	 SMC_CYCLE3; 	//  Cycle Register for CS 3
	AT91_REG	 SMC_CTRL3; 	//  Control Register for CS 3
	AT91_REG	 SMC_SETUP4; 	//  Setup Register for CS 4
	AT91_REG	 SMC_PULSE4; 	//  Pulse Register for CS 4
	AT91_REG	 SMC_CYCLE4; 	//  Cycle Register for CS 4
	AT91_REG	 SMC_CTRL4; 	//  Control Register for CS 4
	AT91_REG	 SMC_SETUP5; 	//  Setup Register for CS 5
	AT91_REG	 SMC_PULSE5; 	//  Pulse Register for CS 5
	AT91_REG	 SMC_CYCLE5; 	//  Cycle Register for CS 5
	AT91_REG	 SMC_CTRL5; 	//  Control Register for CS 5
	AT91_REG	 SMC_SETUP6; 	//  Setup Register for CS 6
	AT91_REG	 SMC_PULSE6; 	//  Pulse Register for CS 6
	AT91_REG	 SMC_CYCLE6; 	//  Cycle Register for CS 6
	AT91_REG	 SMC_CTRL6; 	//  Control Register for CS 6
	AT91_REG	 SMC_SETUP7; 	//  Setup Register for CS 7
	AT91_REG	 SMC_PULSE7; 	//  Pulse Register for CS 7
	AT91_REG	 SMC_CYCLE7; 	//  Cycle Register for CS 7
	AT91_REG	 SMC_CTRL7; 	//  Control Register for CS 7
	AT91_REG	 Reserved0[16]; 	// 
	AT91_REG	 SMC_DELAY1; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY2; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY3; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY4; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY5; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY6; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY7; 	// SMC Delay Control Register
	AT91_REG	 SMC_DELAY8; 	// SMC Delay Control Register
	AT91_REG	 Reserved1[3]; 	// 
	AT91_REG	 SMC_ADDRSIZE; 	// HSMC3 ADDRSIZE REGISTER 
	AT91_REG	 SMC_IPNAME1; 	// HSMC3 IPNAME1 REGISTER 
	AT91_REG	 SMC_IPNAME2; 	// HSMC3 IPNAME2 REGISTER 
	AT91_REG	 SMC_FEATURES; 	// HSMC3 FEATURES REGISTER 
	AT91_REG	 SMC_VER; 	// HSMC3 VERSION REGISTER
} AT91S_SMC, *AT91PS_SMC;

// -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- 
#define AT91C_SMC_NWESETUP    ((unsigned int) 0x3F <<  0) // (SMC) NWE Setup Length
#define AT91C_SMC_NCSSETUPWR  ((unsigned int) 0x3F <<  8) // (SMC) NCS Setup Length in WRite Access
#define AT91C_SMC_NRDSETUP    ((unsigned int) 0x3F << 16) // (SMC) NRD Setup Length
#define AT91C_SMC_NCSSETUPRD  ((unsigned int) 0x3F << 24) // (SMC) NCS Setup Length in ReaD Access
// -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- 
#define AT91C_SMC_NWEPULSE    ((unsigned int) 0x7F <<  0) // (SMC) NWE Pulse Length
#define AT91C_SMC_NCSPULSEWR  ((unsigned int) 0x7F <<  8) // (SMC) NCS Pulse Length in WRite Access
#define AT91C_SMC_NRDPULSE    ((unsigned int) 0x7F << 16) // (SMC) NRD Pulse Length
#define AT91C_SMC_NCSPULSERD  ((unsigned int) 0x7F << 24) // (SMC) NCS Pulse Length in ReaD Access
// -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- 
#define AT91C_SMC_NWECYCLE    ((unsigned int) 0x1FF <<  0) // (SMC) Total Write Cycle Length
#define AT91C_SMC_NRDCYCLE    ((unsigned int) 0x1FF << 16) // (SMC) Total Read Cycle Length
// -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- 
#define AT91C_SMC_READMODE    ((unsigned int) 0x1 <<  0) // (SMC) Read Mode
#define AT91C_SMC_WRITEMODE   ((unsigned int) 0x1 <<  1) // (SMC) Write Mode
#define AT91C_SMC_NWAITM      ((unsigned int) 0x3 <<  5) // (SMC) NWAIT Mode
#define 	AT91C_SMC_NWAITM_NWAIT_DISABLE        ((unsigned int) 0x0 <<  5) // (SMC) External NWAIT disabled.
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_FROZEN  ((unsigned int) 0x2 <<  5) // (SMC) External NWAIT enabled in frozen mode.
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_READY   ((unsigned int) 0x3 <<  5) // (SMC) External NWAIT enabled in ready mode.
#define AT91C_SMC_BAT         ((unsigned int) 0x1 <<  8) // (SMC) Byte Access Type
#define 	AT91C_SMC_BAT_BYTE_SELECT          ((unsigned int) 0x0 <<  8) // (SMC) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
#define 	AT91C_SMC_BAT_BYTE_WRITE           ((unsigned int) 0x1 <<  8) // (SMC) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
#define AT91C_SMC_DBW         ((unsigned int) 0x3 << 12) // (SMC) Data Bus Width
#define 	AT91C_SMC_DBW_WIDTH_EIGTH_BITS     ((unsigned int) 0x0 << 12) // (SMC) 8 bits.
#define 	AT91C_SMC_DBW_WIDTH_SIXTEEN_BITS   ((unsigned int) 0x1 << 12) // (SMC) 16 bits.
#define 	AT91C_SMC_DBW_WIDTH_THIRTY_TWO_BITS ((unsigned int) 0x2 << 12) // (SMC) 32 bits.
#define AT91C_SMC_TDF         ((unsigned int) 0xF << 16) // (SMC) Data Float Time.
#define AT91C_SMC_TDFEN       ((unsigned int) 0x1 << 20) // (SMC) TDF Enabled.
#define AT91C_SMC_PMEN        ((unsigned int) 0x1 << 24) // (SMC) Page Mode Enabled.
#define AT91C_SMC_PS          ((unsigned int) 0x3 << 28) // (SMC) Page Size
#define 	AT91C_SMC_PS_SIZE_FOUR_BYTES      ((unsigned int) 0x0 << 28) // (SMC) 4 bytes.
#define 	AT91C_SMC_PS_SIZE_EIGHT_BYTES     ((unsigned int) 0x1 << 28) // (SMC) 8 bytes.
#define 	AT91C_SMC_PS_SIZE_SIXTEEN_BYTES   ((unsigned int) 0x2 << 28) // (SMC) 16 bytes.
#define 	AT91C_SMC_PS_SIZE_THIRTY_TWO_BYTES ((unsigned int) 0x3 << 28) // (SMC) 32 bytes.
// -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- 
// -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- 
// -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- 
// -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- 
// -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- 

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
//              SOFTWARE API DEFINITION  FOR AHB CCFG Interface
// *****************************************************************************
typedef struct _AT91S_CCFG {
	AT91_REG	 CCFG_RAM0; 	//  RAM0 configuration
	AT91_REG	 CCFG_ROM; 	//  ROM  configuration
	AT91_REG	 CCFG_FLASH0; 	//  FLASH0 configuration
	AT91_REG	 CCFG_EBICSA; 	//  EBI Chip Select Assignement Register
	AT91_REG	 CCFG_BRIDGE; 	//  BRIDGE configuration
} AT91S_CCFG, *AT91PS_CCFG;

// -------- CCFG_RAM0 : (CCFG Offset: 0x0) RAM0 Configuration -------- 
// -------- CCFG_ROM : (CCFG Offset: 0x4) ROM configuration -------- 
// -------- CCFG_FLASH0 : (CCFG Offset: 0x8) FLASH0 configuration -------- 
// -------- CCFG_EBICSA : (CCFG Offset: 0xc) EBI Chip Select Assignement Register -------- 
#define AT91C_EBI_CS0A        ((unsigned int) 0x1 <<  0) // (CCFG) Chip Select 0 Assignment
#define 	AT91C_EBI_CS0A_SMC                  ((unsigned int) 0x0) // (CCFG) Chip Select 0 is only assigned to the Static Memory Controller and NCS0 behaves as defined by the SMC.
#define 	AT91C_EBI_CS0A_SM                   ((unsigned int) 0x1) // (CCFG) Chip Select 0 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
#define AT91C_EBI_CS1A        ((unsigned int) 0x1 <<  1) // (CCFG) Chip Select 1 Assignment
#define 	AT91C_EBI_CS1A_SMC                  ((unsigned int) 0x0 <<  1) // (CCFG) Chip Select 1 is only assigned to the Static Memory Controller and NCS1 behaves as defined by the SMC.
#define 	AT91C_EBI_CS1A_SM                   ((unsigned int) 0x1 <<  1) // (CCFG) Chip Select 1 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
#define AT91C_EBI_CS2A        ((unsigned int) 0x1 <<  2) // (CCFG) Chip Select 2 Assignment
#define 	AT91C_EBI_CS2A_SMC                  ((unsigned int) 0x0 <<  2) // (CCFG) Chip Select 2 is only assigned to the Static Memory Controller and NCS2 behaves as defined by the SMC.
#define 	AT91C_EBI_CS2A_SM                   ((unsigned int) 0x1 <<  2) // (CCFG) Chip Select 2 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
#define AT91C_EBI_CS3A        ((unsigned int) 0x1 <<  3) // (CCFG) Chip Select 3 Assignment
#define 	AT91C_EBI_CS3A_SMC                  ((unsigned int) 0x0 <<  3) // (CCFG) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.
#define 	AT91C_EBI_CS3A_SM                   ((unsigned int) 0x1 <<  3) // (CCFG) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
// -------- CCFG_BRIDGE : (CCFG Offset: 0x10) BRIDGE configuration -------- 

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
	AT91_REG	 PIO_SP1; 	// Select B Register
	AT91_REG	 PIO_SP2; 	// Select B Register
	AT91_REG	 PIO_ABSR; 	// AB Select Status Register
	AT91_REG	 Reserved5[5]; 	// 
	AT91_REG	 PIO_PPDDR; 	// Pull-down Disable Register
	AT91_REG	 PIO_PPDER; 	// Pull-down Enable Register
	AT91_REG	 PIO_PPDSR; 	// Pull-down Status Register
	AT91_REG	 Reserved6[1]; 	// 
	AT91_REG	 PIO_OWER; 	// Output Write Enable Register
	AT91_REG	 PIO_OWDR; 	// Output Write Disable Register
	AT91_REG	 PIO_OWSR; 	// Output Write Status Register
	AT91_REG	 Reserved7[16]; 	// 
	AT91_REG	 PIO_ADDRSIZE; 	// PIO ADDRSIZE REGISTER 
	AT91_REG	 PIO_IPNAME1; 	// PIO IPNAME1 REGISTER 
	AT91_REG	 PIO_IPNAME2; 	// PIO IPNAME2 REGISTER 
	AT91_REG	 PIO_FEATURES; 	// PIO FEATURES REGISTER 
	AT91_REG	 PIO_VER; 	// PIO VERSION REGISTER 
	AT91_REG	 PIO_SLEW1; 	// PIO SLEWRATE1 REGISTER 
	AT91_REG	 PIO_SLEW2; 	// PIO SLEWRATE2 REGISTER 
	AT91_REG	 Reserved8[18]; 	// 
	AT91_REG	 PIO_SENMR; 	// Sensor Mode Register
	AT91_REG	 PIO_SENIER; 	// Sensor Interrupt Enable Register
	AT91_REG	 PIO_SENIDR; 	// Sensor Interrupt Disable Register
	AT91_REG	 PIO_SENIMR; 	// Sensor Interrupt Mask Register
	AT91_REG	 PIO_SENISR; 	// Sensor Interrupt Status Register
	AT91_REG	 PIO_SENDATA; 	// Sensor Data Register
	AT91_REG	 PIOA_RPR; 	// Receive Pointer Register
	AT91_REG	 PIOA_RCR; 	// Receive Counter Register
	AT91_REG	 PIOA_TPR; 	// Transmit Pointer Register
	AT91_REG	 PIOA_TCR; 	// Transmit Counter Register
	AT91_REG	 PIOA_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 PIOA_RNCR; 	// Receive Next Counter Register
	AT91_REG	 PIOA_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 PIOA_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 PIOA_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 PIOA_PTSR; 	// PDC Transfer Status Register
} AT91S_PIO, *AT91PS_PIO;


// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Power Management Controler
// *****************************************************************************
typedef struct _AT91S_PMC {
	AT91_REG	 PMC_SCER; 	// System Clock Enable Register
	AT91_REG	 PMC_SCDR; 	// System Clock Disable Register
	AT91_REG	 PMC_SCSR; 	// System Clock Status Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 PMC_PCER; 	// Peripheral Clock Enable Register 0:31 PERI_ID
	AT91_REG	 PMC_PCDR; 	// Peripheral Clock Disable Register 0:31 PERI_ID
	AT91_REG	 PMC_PCSR; 	// Peripheral Clock Status Register 0:31 PERI_ID
	AT91_REG	 PMC_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 PMC_MOR; 	// Main Oscillator Register
	AT91_REG	 PMC_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 PMC_PLLAR; 	// PLL Register
	AT91_REG	 PMC_PLLBR; 	// PLL B Register
	AT91_REG	 PMC_MCKR; 	// Master Clock Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 PMC_UDPR; 	// USB DEV Clock Configuration Register
	AT91_REG	 Reserved2[1]; 	// 
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
	AT91_REG	 PMC_PCER1; 	// Peripheral Clock Enable Register 32:63 PERI_ID
	AT91_REG	 PMC_PCDR1; 	// Peripheral Clock Disable Register 32:63 PERI_ID
	AT91_REG	 PMC_PCSR1; 	// Peripheral Clock Status Register 32:63 PERI_ID
	AT91_REG	 PMC_PCR; 	// Peripheral Control Register
} AT91S_PMC, *AT91PS_PMC;

// -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) // (PMC) Processor Clock
#define AT91C_PMC_UDP         ((unsigned int) 0x1 <<  7) // (PMC) USB Device Port Clock
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
// -------- CKGR_PLLBR : (PMC Offset: 0x2c) PLL B Register -------- 
#define AT91C_CKGR_DIVB       ((unsigned int) 0xFF <<  0) // (PMC) Divider B Selected
#define 	AT91C_CKGR_DIVB_0                    ((unsigned int) 0x0) // (PMC) Divider B output is 0
#define 	AT91C_CKGR_DIVB_BYPASS               ((unsigned int) 0x1) // (PMC) Divider B is bypassed
#define AT91C_CKGR_PLLBCOUNT  ((unsigned int) 0x3F <<  8) // (PMC) PLL B Counter
#define AT91C_CKGR_OUTB       ((unsigned int) 0x3 << 14) // (PMC) PLL B Output Frequency Range
#define 	AT91C_CKGR_OUTB_0                    ((unsigned int) 0x0 << 14) // (PMC) Please refer to the PLLB datasheet
#define 	AT91C_CKGR_OUTB_1                    ((unsigned int) 0x1 << 14) // (PMC) Please refer to the PLLB datasheet
#define 	AT91C_CKGR_OUTB_2                    ((unsigned int) 0x2 << 14) // (PMC) Please refer to the PLLB datasheet
#define 	AT91C_CKGR_OUTB_3                    ((unsigned int) 0x3 << 14) // (PMC) Please refer to the PLLB datasheet
#define AT91C_CKGR_MULB       ((unsigned int) 0x7FF << 16) // (PMC) PLL B Multiplier
// -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
#define AT91C_PMC_CSS         ((unsigned int) 0x7 <<  0) // (PMC) Programmable Clock Selection
#define 	AT91C_PMC_CSS_SLOW_CLK             ((unsigned int) 0x0) // (PMC) Slow Clock is selected
#define 	AT91C_PMC_CSS_MAIN_CLK             ((unsigned int) 0x1) // (PMC) Main Clock is selected
#define 	AT91C_PMC_CSS_PLLA_CLK             ((unsigned int) 0x2) // (PMC) Clock from PLL A is selected
#define 	AT91C_PMC_CSS_PLLB_CLK             ((unsigned int) 0x3) // (PMC) Clock from PLL B is selected
#define 	AT91C_PMC_CSS_SYS_CLK              ((unsigned int) 0x4) // (PMC) System clock is selected
#define AT91C_PMC_PRES        ((unsigned int) 0xF <<  4) // (PMC) Programmable Clock Prescaler
#define 	AT91C_PMC_PRES_CLK                  ((unsigned int) 0x0 <<  4) // (PMC) Selected clock
#define 	AT91C_PMC_PRES_CLK_2                ((unsigned int) 0x1 <<  4) // (PMC) Selected clock divided by 2
#define 	AT91C_PMC_PRES_CLK_4                ((unsigned int) 0x2 <<  4) // (PMC) Selected clock divided by 4
#define 	AT91C_PMC_PRES_CLK_8                ((unsigned int) 0x3 <<  4) // (PMC) Selected clock divided by 8
#define 	AT91C_PMC_PRES_CLK_16               ((unsigned int) 0x4 <<  4) // (PMC) Selected clock divided by 16
#define 	AT91C_PMC_PRES_CLK_32               ((unsigned int) 0x5 <<  4) // (PMC) Selected clock divided by 32
#define 	AT91C_PMC_PRES_CLK_64               ((unsigned int) 0x6 <<  4) // (PMC) Selected clock divided by 64
#define 	AT91C_PMC_PRES_CLK_3                ((unsigned int) 0x7 <<  4) // (PMC) Selected clock divided by 3
#define 	AT91C_PMC_PRES_CLK_1_5              ((unsigned int) 0x8 <<  4) // (PMC) Selected clock divided by 1.5
// -------- PMC_UDPR : (PMC Offset: 0x38) USB DEV Clock Configuration Register -------- 
#define AT91C_PMC_UDP_CLK_SEL ((unsigned int) 0x1 <<  0) // (PMC) UDP Clock Selection
#define 	AT91C_PMC_UDP_CLK_SEL_PLLA                 ((unsigned int) 0x0) // (PMC) PLL A is the selected clock source for UDP DEV
#define 	AT91C_PMC_UDP_CLK_SEL_PLLB                 ((unsigned int) 0x1) // (PMC) PLL B is the selected clock source for UDP DEV
#define AT91C_PMC_UDP_DIV     ((unsigned int) 0xF <<  8) // (PMC) UDP Clock Divider
#define 	AT91C_PMC_UDP_DIV_DIV                  ((unsigned int) 0x0 <<  8) // (PMC) Selected clock
#define 	AT91C_PMC_UDP_DIV_DIV_2                ((unsigned int) 0x1 <<  8) // (PMC) Selected clock divided by 2
// -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- 
// -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
#define AT91C_PMC_MOSCXTS     ((unsigned int) 0x1 <<  0) // (PMC) Main Crystal Oscillator Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKA       ((unsigned int) 0x1 <<  1) // (PMC) PLL A Status/Enable/Disable/Mask
#define AT91C_PMC_LOCKB       ((unsigned int) 0x1 <<  2) // (PMC) PLL B Status/Enable/Disable/Mask
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
#define AT91C_                ((unsigned int) 0x0 <<  7) // (PMC) 
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
// -------- PMC_PCR : (PMC Offset: 0x10c) Peripheral Control Register -------- 
#define AT91C_PMC_PID         ((unsigned int) 0x3F <<  0) // (PMC) Peripheral Identifier
#define AT91C_PMC_CMD         ((unsigned int) 0x1 << 12) // (PMC) Read / Write Command
#define AT91C_PMC_DIV         ((unsigned int) 0x3 << 16) // (PMC) Peripheral clock Divider
#define AT91C_PMC_EN          ((unsigned int) 0x1 << 28) // (PMC) Peripheral Clock Enable

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// *****************************************************************************
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_UCKR; 	// UTMI Clock Configuration Register
	AT91_REG	 CKGR_MOR; 	// Main Oscillator Register
	AT91_REG	 CKGR_MCFR; 	// Main Clock  Frequency Register
	AT91_REG	 CKGR_PLLAR; 	// PLL Register
	AT91_REG	 CKGR_PLLBR; 	// PLL B Register
} AT91S_CKGR, *AT91PS_CKGR;

// -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 
// -------- CKGR_PLLBR : (CKGR Offset: 0x10) PLL B Register -------- 

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
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 ADC_LCDR; 	// ADC Last Converted Data Register
	AT91_REG	 ADC_IER; 	// ADC Interrupt Enable Register
	AT91_REG	 ADC_IDR; 	// ADC Interrupt Disable Register
	AT91_REG	 ADC_IMR; 	// ADC Interrupt Mask Register
	AT91_REG	 ADC_SR; 	// ADC Status Register
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 ADC_OVR; 	// unspecified
	AT91_REG	 ADC_CWR; 	// unspecified
	AT91_REG	 ADC_CWSR; 	// unspecified
	AT91_REG	 ADC_CGR; 	// Control gain register
	AT91_REG	 ADC_COR; 	// unspecified
	AT91_REG	 ADC_CDR0; 	// ADC Channel Data Register 0
	AT91_REG	 ADC_CDR1; 	// ADC Channel Data Register 1
	AT91_REG	 ADC_CDR2; 	// ADC Channel Data Register 2
	AT91_REG	 ADC_CDR3; 	// ADC Channel Data Register 3
	AT91_REG	 ADC_CDR4; 	// ADC Channel Data Register 4
	AT91_REG	 ADC_CDR5; 	// ADC Channel Data Register 5
	AT91_REG	 ADC_CDR6; 	// ADC Channel Data Register 6
	AT91_REG	 ADC_CDR7; 	// ADC Channel Data Register 7
	AT91_REG	 ADC_CDR8; 	// ADC Channel Data Register 8
	AT91_REG	 ADC_CDR9; 	// ADC Channel Data Register 9
	AT91_REG	 ADC_CDR10; 	// ADC Channel Data Register 10
	AT91_REG	 ADC_CDR11; 	// ADC Channel Data Register 11
	AT91_REG	 ADC_CDR12; 	// ADC Channel Data Register 12
	AT91_REG	 ADC_CDR13; 	// ADC Channel Data Register 13
	AT91_REG	 ADC_CDR14; 	// ADC Channel Data Register 14
	AT91_REG	 ADC_CDR15; 	// ADC Channel Data Register 15
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 ADC_ACR; 	// unspecified
	AT91_REG	 Reserved4[21]; 	// 
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
#define AT91C_ADC_CH8         ((unsigned int) 0x1 <<  8) // (ADC) Channel 8
#define AT91C_ADC_CH9         ((unsigned int) 0x1 <<  9) // (ADC) Channel 9
#define AT91C_ADC_CH10        ((unsigned int) 0x1 << 10) // (ADC) Channel 10
#define AT91C_ADC_CH11        ((unsigned int) 0x1 << 11) // (ADC) Channel 11
#define AT91C_ADC_CH12        ((unsigned int) 0x1 << 12) // (ADC) Channel 12
#define AT91C_ADC_CH13        ((unsigned int) 0x1 << 13) // (ADC) Channel 13
#define AT91C_ADC_CH14        ((unsigned int) 0x1 << 14) // (ADC) Channel 14
#define AT91C_ADC_CH15        ((unsigned int) 0x1 << 15) // (ADC) Channel 15
// -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- 
// -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- 
// -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- 
#define AT91C_ADC_LDATA       ((unsigned int) 0x3FF <<  0) // (ADC) Last Data Converted
// -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- 
#define AT91C_ADC_EOC0        ((unsigned int) 0x1 <<  0) // (ADC) End of Conversion
#define AT91C_ADC_EOC1        ((unsigned int) 0x1 <<  1) // (ADC) End of Conversion
#define AT91C_ADC_EOC2        ((unsigned int) 0x1 <<  2) // (ADC) End of Conversion
#define AT91C_ADC_EOC3        ((unsigned int) 0x1 <<  3) // (ADC) End of Conversion
#define AT91C_ADC_EOC4        ((unsigned int) 0x1 <<  4) // (ADC) End of Conversion
#define AT91C_ADC_EOC5        ((unsigned int) 0x1 <<  5) // (ADC) End of Conversion
#define AT91C_ADC_EOC6        ((unsigned int) 0x1 <<  6) // (ADC) End of Conversion
#define AT91C_ADC_EOC7        ((unsigned int) 0x1 <<  7) // (ADC) End of Conversion
#define AT91C_ADC_EOC8        ((unsigned int) 0x1 <<  8) // (ADC) End of Conversion
#define AT91C_ADC_EOC9        ((unsigned int) 0x1 <<  9) // (ADC) End of Conversion
#define AT91C_ADC_EOC10       ((unsigned int) 0x1 << 10) // (ADC) End of Conversion
#define AT91C_ADC_EOC11       ((unsigned int) 0x1 << 11) // (ADC) End of Conversion
#define AT91C_ADC_EOC12       ((unsigned int) 0x1 << 12) // (ADC) End of Conversion
#define AT91C_ADC_EOC13       ((unsigned int) 0x1 << 13) // (ADC) End of Conversion
#define AT91C_ADC_EOC14       ((unsigned int) 0x1 << 14) // (ADC) End of Conversion
#define AT91C_ADC_EOC15       ((unsigned int) 0x1 << 15) // (ADC) End of Conversion
#define AT91C_ADC_DRDY        ((unsigned int) 0x1 << 24) // (ADC) Data Ready
#define AT91C_ADC_GOVRE       ((unsigned int) 0x1 << 25) // (ADC) General Overrun
#define AT91C_ADC_ENDRX       ((unsigned int) 0x1 << 27) // (ADC) End of Receiver Transfer
#define AT91C_ADC_RXBUFF      ((unsigned int) 0x1 << 28) // (ADC) RXBUFF Interrupt
// -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- 
// -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- 
// -------- ADC_SR : (ADC Offset: 0x30) ADC Status Register -------- 
// -------- ADC_OVR : (ADC Offset: 0x3c)  -------- 
#define AT91C_ADC_OVRE2       ((unsigned int) 0x1 << 10) // (ADC) Overrun Error
#define AT91C_ADC_OVRE3       ((unsigned int) 0x1 << 11) // (ADC) Overrun Error
#define AT91C_ADC_OVRE4       ((unsigned int) 0x1 << 12) // (ADC) Overrun Error
#define AT91C_ADC_OVRE5       ((unsigned int) 0x1 << 13) // (ADC) Overrun Error
// -------- ADC_CWR : (ADC Offset: 0x40)  -------- 
// -------- ADC_CWSR : (ADC Offset: 0x44)  -------- 
// -------- ADC_CGR : (ADC Offset: 0x48)  -------- 
// -------- ADC_COR : (ADC Offset: 0x4c)  -------- 
// -------- ADC_CDR0 : (ADC Offset: 0x50) ADC Channel Data Register 0 -------- 
#define AT91C_ADC_DATA        ((unsigned int) 0x3FF <<  0) // (ADC) Converted Data
// -------- ADC_CDR1 : (ADC Offset: 0x54) ADC Channel Data Register 1 -------- 
// -------- ADC_CDR2 : (ADC Offset: 0x58) ADC Channel Data Register 2 -------- 
// -------- ADC_CDR3 : (ADC Offset: 0x5c) ADC Channel Data Register 3 -------- 
// -------- ADC_CDR4 : (ADC Offset: 0x60) ADC Channel Data Register 4 -------- 
// -------- ADC_CDR5 : (ADC Offset: 0x64) ADC Channel Data Register 5 -------- 
// -------- ADC_CDR6 : (ADC Offset: 0x68) ADC Channel Data Register 6 -------- 
// -------- ADC_CDR7 : (ADC Offset: 0x6c) ADC Channel Data Register 7 -------- 
// -------- ADC_CDR8 : (ADC Offset: 0x70) ADC Channel Data Register 8 -------- 
// -------- ADC_CDR9 : (ADC Offset: 0x74) ADC Channel Data Register 9 -------- 
// -------- ADC_CDR10 : (ADC Offset: 0x78) ADC Channel Data Register 10 -------- 
// -------- ADC_CDR11 : (ADC Offset: 0x7c) ADC Channel Data Register 11 -------- 
// -------- ADC_CDR12 : (ADC Offset: 0x80) ADC Channel Data Register 12 -------- 
// -------- ADC_CDR13 : (ADC Offset: 0x84) ADC Channel Data Register 13 -------- 
// -------- ADC_CDR14 : (ADC Offset: 0x88) ADC Channel Data Register 14 -------- 
// -------- ADC_CDR15 : (ADC Offset: 0x8c) ADC Channel Data Register 15 -------- 
// -------- ADC_VER : (ADC Offset: 0xfc) ADC VER -------- 
#define AT91C_ADC_VER         ((unsigned int) 0xF <<  0) // (ADC) ADC VER

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Digital to Analog Convertor
// *****************************************************************************
typedef struct _AT91S_DAC {
	AT91_REG	 DAC_CR; 	// Control Register
	AT91_REG	 DAC_MR; 	// Mode Register
	AT91_REG	 Reserved0[2]; 	// 
	AT91_REG	 DAC_CHER; 	// Channel Enable Register
	AT91_REG	 DAC_CHDR; 	// Channel Disable Register
	AT91_REG	 DAC_CHSR; 	// Channel Status Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 DAC_CDR; 	// Coversion Data Register
	AT91_REG	 DAC_IER; 	// Interrupt Enable Register
	AT91_REG	 DAC_IDR; 	// Interrupt Disable Register
	AT91_REG	 DAC_IMR; 	// Interrupt Mask Register
	AT91_REG	 DAC_ISR; 	// Interrupt Status Register
	AT91_REG	 Reserved2[24]; 	// 
	AT91_REG	 DAC_ACR; 	// Analog Current Register
	AT91_REG	 Reserved3[19]; 	// 
	AT91_REG	 DAC_WPMR; 	// Write Protect Mode Register
	AT91_REG	 DAC_WPSR; 	// Write Protect Status Register
	AT91_REG	 DAC_ADDRSIZE; 	// DAC ADDRSIZE REGISTER 
	AT91_REG	 DAC_IPNAME1; 	// DAC IPNAME1 REGISTER 
	AT91_REG	 DAC_IPNAME2; 	// DAC IPNAME2 REGISTER 
	AT91_REG	 DAC_FEATURES; 	// DAC FEATURES REGISTER 
	AT91_REG	 DAC_VER; 	// DAC VERSION REGISTER
	AT91_REG	 DAC_RPR; 	// Receive Pointer Register
	AT91_REG	 DAC_RCR; 	// Receive Counter Register
	AT91_REG	 DAC_TPR; 	// Transmit Pointer Register
	AT91_REG	 DAC_TCR; 	// Transmit Counter Register
	AT91_REG	 DAC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 DAC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 DAC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 DAC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 DAC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 DAC_PTSR; 	// PDC Transfer Status Register
} AT91S_DAC, *AT91PS_DAC;

// -------- DAC_CR : (DAC Offset: 0x0) Control Register -------- 
#define AT91C_DAC_SWRST       ((unsigned int) 0x1 <<  0) // (DAC) Software Reset
// -------- DAC_MR : (DAC Offset: 0x4) Mode Register -------- 
#define AT91C_DAC_TRGEN       ((unsigned int) 0x1 <<  0) // (DAC) Trigger Enable
#define AT91C_DAC_TRGSEL      ((unsigned int) 0x7 <<  1) // (DAC) Trigger Selection
#define 	AT91C_DAC_TRGSEL_EXTRG0               ((unsigned int) 0x0 <<  1) // (DAC) External Trigger 0
#define 	AT91C_DAC_TRGSEL_EXTRG1               ((unsigned int) 0x1 <<  1) // (DAC) External Trigger 1
#define 	AT91C_DAC_TRGSEL_EXTRG2               ((unsigned int) 0x2 <<  1) // (DAC) External Trigger 2
#define 	AT91C_DAC_TRGSEL_EXTRG3               ((unsigned int) 0x3 <<  1) // (DAC) External Trigger 3
#define 	AT91C_DAC_TRGSEL_EXTRG4               ((unsigned int) 0x4 <<  1) // (DAC) External Trigger 4
#define 	AT91C_DAC_TRGSEL_EXTRG5               ((unsigned int) 0x5 <<  1) // (DAC) External Trigger 5
#define 	AT91C_DAC_TRGSEL_EXTRG6               ((unsigned int) 0x6 <<  1) // (DAC) External Trigger 6
#define AT91C_DAC_WORD        ((unsigned int) 0x1 <<  4) // (DAC) Word Transfer
#define AT91C_DAC_SLEEP       ((unsigned int) 0x1 <<  5) // (DAC) Sleep Mode
#define AT91C_DAC_FASTW       ((unsigned int) 0x1 <<  6) // (DAC) Fast Wake up Mode
#define AT91C_DAC_REFRESH     ((unsigned int) 0xFF <<  8) // (DAC) Refresh period
#define AT91C_DAC_USER_SEL    ((unsigned int) 0x3 << 16) // (DAC) User Channel Selection
#define 	AT91C_DAC_USER_SEL_CH0                  ((unsigned int) 0x0 << 16) // (DAC) Channel 0
#define 	AT91C_DAC_USER_SEL_CH1                  ((unsigned int) 0x1 << 16) // (DAC) Channel 1
#define 	AT91C_DAC_USER_SEL_CH2                  ((unsigned int) 0x2 << 16) // (DAC) Channel 2
#define AT91C_DAC_TAG         ((unsigned int) 0x1 << 20) // (DAC) Tag selection mode
#define AT91C_DAC_MAXSPEED    ((unsigned int) 0x1 << 21) // (DAC) Max speed mode
#define AT91C_DAC_STARTUP     ((unsigned int) 0x3F << 24) // (DAC) Startup Time Selection
// -------- DAC_CHER : (DAC Offset: 0x10) Channel Enable Register -------- 
#define AT91C_DAC_CH0         ((unsigned int) 0x1 <<  0) // (DAC) Channel 0
#define AT91C_DAC_CH1         ((unsigned int) 0x1 <<  1) // (DAC) Channel 1
#define AT91C_DAC_CH2         ((unsigned int) 0x1 <<  2) // (DAC) Channel 2
// -------- DAC_CHDR : (DAC Offset: 0x14) Channel Disable Register -------- 
// -------- DAC_CHSR : (DAC Offset: 0x18) Channel Status Register -------- 
// -------- DAC_CDR : (DAC Offset: 0x20) Conversion Data Register -------- 
#define AT91C_DAC_DATA        ((unsigned int) 0x0 <<  0) // (DAC) Data to convert
// -------- DAC_IER : (DAC Offset: 0x24) DAC Interrupt Enable -------- 
#define AT91C_DAC_TXRDY       ((unsigned int) 0x1 <<  0) // (DAC) Transmission Ready Interrupt
#define AT91C_DAC_EOC         ((unsigned int) 0x1 <<  1) // (DAC) End of Conversion Interrupt
#define AT91C_DAC_TXDMAEND    ((unsigned int) 0x1 <<  2) // (DAC) End of DMA Interrupt
#define AT91C_DAC_TXBUFEMPT   ((unsigned int) 0x1 <<  3) // (DAC) Buffer Empty Interrupt
// -------- DAC_IDR : (DAC Offset: 0x28) DAC Interrupt Disable -------- 
// -------- DAC_IMR : (DAC Offset: 0x2c) DAC Interrupt Mask -------- 
// -------- DAC_ISR : (DAC Offset: 0x30) DAC Interrupt Status -------- 
// -------- DAC_ACR : (DAC Offset: 0x94) Analog Current Register -------- 
#define AT91C_DAC_IBCTL       ((unsigned int) 0x1FF <<  0) // (DAC) Bias current control
// -------- DAC_WPMR : (DAC Offset: 0xe4) Write Protect Mode Register -------- 
#define AT91C_DAC_WPEN        ((unsigned int) 0x1 <<  0) // (DAC) Write Protect Enable
#define AT91C_DAC_WPKEY       ((unsigned int) 0xFFFFFF <<  8) // (DAC) Write Protect KEY
// -------- DAC_WPSR : (DAC Offset: 0xe8) Write Protect Status Register -------- 
#define AT91C_DAC_WPROTERR    ((unsigned int) 0x1 <<  0) // (DAC) Write protection error
#define AT91C_DAC_WPROTADDR   ((unsigned int) 0xFF <<  8) // (DAC) Write protection error address

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Digital to Analog Convertor
// *****************************************************************************
typedef struct _AT91S_ACC {
	AT91_REG	 ACC_CR; 	// Control Register
	AT91_REG	 ACC_MR; 	// Mode Register
	AT91_REG	 Reserved0[7]; 	// 
	AT91_REG	 ACC_IER; 	// Interrupt Enable Register
	AT91_REG	 ACC_IDR; 	// Interrupt Disable Register
	AT91_REG	 ACC_IMR; 	// Interrupt Mask Register
	AT91_REG	 ACC_ISR; 	// Interrupt Status Register
	AT91_REG	 Reserved1[24]; 	// 
	AT91_REG	 ACC_ACR; 	// Analog Control Register
	AT91_REG	 Reserved2[19]; 	// 
	AT91_REG	 ACC_MODE; 	// Write Protection Mode Register
	AT91_REG	 ACC_STATUS; 	// Write Protection Status
	AT91_REG	 ACC_ADDRSIZE; 	// ACC ADDRSIZE REGISTER 
	AT91_REG	 ACC_IPNAME1; 	// ACC IPNAME1 REGISTER 
	AT91_REG	 ACC_IPNAME2; 	// ACC IPNAME2 REGISTER 
	AT91_REG	 ACC_FEATURES; 	// ACC FEATURES REGISTER 
	AT91_REG	 ACC_VER; 	// ACC VERSION REGISTER
} AT91S_ACC, *AT91PS_ACC;

// -------- ACC_IER : (ACC Offset: 0x24) Data Ready for Conversion Interrupt Enable -------- 
#define AT91C_ACC_DATRDY      ((unsigned int) 0x1 <<  0) // (ACC) Data Ready for Conversion
// -------- ACC_IDR : (ACC Offset: 0x28) Data Ready for Conversion Interrupt Disable -------- 
// -------- ACC_IMR : (ACC Offset: 0x2c) Data Ready for Conversion Interrupt Mask -------- 
// -------- ACC_ISR : (ACC Offset: 0x30) Status Register -------- 
// -------- ACC_VER : (ACC Offset: 0xfc) ACC VER -------- 
#define AT91C_ACC_VER         ((unsigned int) 0xF <<  0) // (ACC) ACC VER

// *****************************************************************************
//              SOFTWARE API DEFINITION  FOR Context Based Direct Memory Access Controller Interface
// *****************************************************************************
typedef struct _AT91S_HCBDMA {
	AT91_REG	 HCBDMA_CBDSCR; 	// CB DMA Descriptor Base Register
	AT91_REG	 HCBDMA_CBNXTEN; 	// CB DMA Next Descriptor Enable Register
	AT91_REG	 HCBDMA_CBEN; 	// CB DMA Enable Register
	AT91_REG	 HCBDMA_CBDIS; 	// CB DMA Disable Register
	AT91_REG	 HCBDMA_CBSR; 	// CB DMA Status Register
	AT91_REG	 HCBDMA_CBIER; 	// CB DMA Interrupt Enable Register
	AT91_REG	 HCBDMA_CBIDR; 	// CB DMA Interrupt Disable Register
	AT91_REG	 HCBDMA_CBIMR; 	// CB DMA Interrupt mask Register
	AT91_REG	 HCBDMA_CBISR; 	// CB DMA Interrupt Status Register
	AT91_REG	 HCBDMA_CBDLIER; 	// CB DMA Loaded Interrupt Enable Register
	AT91_REG	 HCBDMA_CBDLIDR; 	// CB DMA Loaded Interrupt Disable Register
	AT91_REG	 HCBDMA_CBDLIMR; 	// CB DMA Loaded Interrupt mask Register
	AT91_REG	 HCBDMA_CBDLISR; 	// CB DMA Loaded Interrupt Status Register
	AT91_REG	 HCBDMA_CBCRCCR; 	// CB DMA CRC Control Resgister
	AT91_REG	 HCBDMA_CBCRCMR; 	// CB DMA CRC Mode Resgister
	AT91_REG	 HCBDMA_CBCRCSR; 	// CB DMA CRC Status Resgister
	AT91_REG	 HCBDMA_CBCRCIER; 	// CB DMA CRC Interrupt Enable Resgister
	AT91_REG	 HCBDMA_CBCRCIDR; 	// CB DMA CRC Interrupt Disable Resgister
	AT91_REG	 HCBDMA_CBCRCIMR; 	// CB DMA CRC Interrupt Mask Resgister
	AT91_REG	 HCBDMA_CBCRCISR; 	// CB DMA CRC Interrupt Status Resgister
	AT91_REG	 Reserved0[39]; 	// 
	AT91_REG	 HCBDMA_ADDRSIZE; 	// HCBDMA ADDRSIZE REGISTER 
	AT91_REG	 HCBDMA_IPNAME1; 	// HCBDMA IPNAME1 REGISTER 
	AT91_REG	 HCBDMA_IPNAME2; 	// HCBDMA IPNAME2 REGISTER 
	AT91_REG	 HCBDMA_FEATURES; 	// HCBDMA FEATURES REGISTER 
	AT91_REG	 HCBDMA_VER; 	// HCBDMA VERSION REGISTER
} AT91S_HCBDMA, *AT91PS_HCBDMA;

// -------- HCBDMA_CBDSCR : (HCBDMA Offset: 0x0) CB DMA Descriptor Base Register -------- 
#define AT91C_HCBDMA_DSCR     ((unsigned int) 0x0 <<  0) // (HCBDMA) Descriptor Base Address
// -------- HCBDMA_CBNXTEN : (HCBDMA Offset: 0x4) Next Descriptor Enable Register -------- 
#define AT91C_HCBDMA_NXTID0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Next Descriptor Identifier for the Channel 0
#define AT91C_HCBDMA_NXTID1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Next Descriptor Identifier for the Channel 1
#define AT91C_HCBDMA_NXTID2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Next Descriptor Identifier for the Channel 2
#define AT91C_HCBDMA_NXTID3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Next Descriptor Identifier for the Channel 3
#define AT91C_HCBDMA_NXTID4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Next Descriptor Identifier for the Channel 4
#define AT91C_HCBDMA_NXTID5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Next Descriptor Identifier for the Channel 5
#define AT91C_HCBDMA_NXTID6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Next Descriptor Identifier for the Channel 6
#define AT91C_HCBDMA_NXTID7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Next Descriptor Identifier for the Channel 7
#define AT91C_HCBDMA_NXTID8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Next Descriptor Identifier for the Channel 8
#define AT91C_HCBDMA_NXTID9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Next Descriptor Identifier for the Channel 9
#define AT91C_HCBDMA_NXTID10  ((unsigned int) 0x1 << 10) // (HCBDMA) Next Descriptor Identifier for the Channel 10
#define AT91C_HCBDMA_NXTID11  ((unsigned int) 0x1 << 11) // (HCBDMA) Next Descriptor Identifier for the Channel 11
#define AT91C_HCBDMA_NXTID12  ((unsigned int) 0x1 << 12) // (HCBDMA) Next Descriptor Identifier for the Channel 12
#define AT91C_HCBDMA_NXTID13  ((unsigned int) 0x1 << 13) // (HCBDMA) Next Descriptor Identifier for the Channel 13
#define AT91C_HCBDMA_NXTID14  ((unsigned int) 0x1 << 14) // (HCBDMA) Next Descriptor Identifier for the Channel 14
#define AT91C_HCBDMA_NXTID15  ((unsigned int) 0x1 << 15) // (HCBDMA) Next Descriptor Identifier for the Channel 15
#define AT91C_HCBDMA_NXTID16  ((unsigned int) 0x1 << 16) // (HCBDMA) Next Descriptor Identifier for the Channel 16
#define AT91C_HCBDMA_NXTID17  ((unsigned int) 0x1 << 17) // (HCBDMA) Next Descriptor Identifier for the Channel 17
#define AT91C_HCBDMA_NXTID18  ((unsigned int) 0x1 << 18) // (HCBDMA) Next Descriptor Identifier for the Channel 18
#define AT91C_HCBDMA_NXTID19  ((unsigned int) 0x1 << 19) // (HCBDMA) Next Descriptor Identifier for the Channel 19
#define AT91C_HCBDMA_NXTID20  ((unsigned int) 0x1 << 20) // (HCBDMA) Next Descriptor Identifier for the Channel 20
#define AT91C_HCBDMA_NXTID21  ((unsigned int) 0x1 << 21) // (HCBDMA) Next Descriptor Identifier for the Channel 21
#define AT91C_HCBDMA_NXTID22  ((unsigned int) 0x1 << 22) // (HCBDMA) Next Descriptor Identifier for the Channel 22
#define AT91C_HCBDMA_NXTID23  ((unsigned int) 0x1 << 23) // (HCBDMA) Next Descriptor Identifier for the Channel 23
#define AT91C_HCBDMA_NXTID24  ((unsigned int) 0x1 << 24) // (HCBDMA) Next Descriptor Identifier for the Channel 24
#define AT91C_HCBDMA_NXTID25  ((unsigned int) 0x1 << 25) // (HCBDMA) Next Descriptor Identifier for the Channel 25
#define AT91C_HCBDMA_NXTID26  ((unsigned int) 0x1 << 26) // (HCBDMA) Next Descriptor Identifier for the Channel 26
#define AT91C_HCBDMA_NXTID27  ((unsigned int) 0x1 << 27) // (HCBDMA) Next Descriptor Identifier for the Channel 27
#define AT91C_HCBDMA_NXTID28  ((unsigned int) 0x1 << 28) // (HCBDMA) Next Descriptor Identifier for the Channel 28
#define AT91C_HCBDMA_NXTID29  ((unsigned int) 0x1 << 29) // (HCBDMA) Next Descriptor Identifier for the Channel 29
#define AT91C_HCBDMA_NXTID30  ((unsigned int) 0x1 << 30) // (HCBDMA) Next Descriptor Identifier for the Channel 30
#define AT91C_HCBDMA_NXTID31  ((unsigned int) 0x1 << 31) // (HCBDMA) Next Descriptor Identifier for the Channel 31
// -------- HCBDMA_CBEN : (HCBDMA Offset: 0x8) CB DMA Enable Register -------- 
#define AT91C_HCBDMA_CBEN0    ((unsigned int) 0x1 <<  0) // (HCBDMA) Enable for the Channel 0
#define AT91C_HCBDMA_CBEN1    ((unsigned int) 0x1 <<  1) // (HCBDMA) Enable for the Channel 1
#define AT91C_HCBDMA_CBEN2    ((unsigned int) 0x1 <<  2) // (HCBDMA) Enable for the Channel 2
#define AT91C_HCBDMA_CBEN3    ((unsigned int) 0x1 <<  3) // (HCBDMA) Enable for the Channel 3
#define AT91C_HCBDMA_CBEN4    ((unsigned int) 0x1 <<  4) // (HCBDMA) Enable for the Channel 4
#define AT91C_HCBDMA_CBEN5    ((unsigned int) 0x1 <<  5) // (HCBDMA) Enable for the Channel 5
#define AT91C_HCBDMA_CBEN6    ((unsigned int) 0x1 <<  6) // (HCBDMA) Enable for the Channel 6
#define AT91C_HCBDMA_CBEN7    ((unsigned int) 0x1 <<  7) // (HCBDMA) Enable for the Channel 7
#define AT91C_HCBDMA_CBEN8    ((unsigned int) 0x1 <<  8) // (HCBDMA) Enable for the Channel 8
#define AT91C_HCBDMA_CBEN9    ((unsigned int) 0x1 <<  9) // (HCBDMA) Enable for the Channel 9
#define AT91C_HCBDMA_CBEN10   ((unsigned int) 0x1 << 10) // (HCBDMA) Enable for the Channel 10
#define AT91C_HCBDMA_CBEN11   ((unsigned int) 0x1 << 11) // (HCBDMA) Enable for the Channel 11
#define AT91C_HCBDMA_CBEN12   ((unsigned int) 0x1 << 12) // (HCBDMA) Enable for the Channel 12
#define AT91C_HCBDMA_CBEN13   ((unsigned int) 0x1 << 13) // (HCBDMA) Enable for the Channel 13
#define AT91C_HCBDMA_CBEN14   ((unsigned int) 0x1 << 14) // (HCBDMA) Enable for the Channel 14
#define AT91C_HCBDMA_CBEN15   ((unsigned int) 0x1 << 15) // (HCBDMA) Enable for the Channel 15
#define AT91C_HCBDMA_CBEN16   ((unsigned int) 0x1 << 16) // (HCBDMA) Enable for the Channel 16
#define AT91C_HCBDMA_CBEN17   ((unsigned int) 0x1 << 17) // (HCBDMA) Enable for the Channel 17
#define AT91C_HCBDMA_CBEN18   ((unsigned int) 0x1 << 18) // (HCBDMA) Enable for the Channel 18
#define AT91C_HCBDMA_CBEN19   ((unsigned int) 0x1 << 19) // (HCBDMA) Enable for the Channel 19
#define AT91C_HCBDMA_CBEN20   ((unsigned int) 0x1 << 20) // (HCBDMA) Enable for the Channel 20
#define AT91C_HCBDMA_CBEN21   ((unsigned int) 0x1 << 21) // (HCBDMA) Enable for the Channel 21
#define AT91C_HCBDMA_CBEN22   ((unsigned int) 0x1 << 22) // (HCBDMA) Enable for the Channel 22
#define AT91C_HCBDMA_CBEN23   ((unsigned int) 0x1 << 23) // (HCBDMA) Enable for the Channel 23
#define AT91C_HCBDMA_CBEN24   ((unsigned int) 0x1 << 24) // (HCBDMA) Enable for the Channel 24
#define AT91C_HCBDMA_CBEN25   ((unsigned int) 0x1 << 25) // (HCBDMA) Enable for the Channel 25
#define AT91C_HCBDMA_CBEN26   ((unsigned int) 0x1 << 26) // (HCBDMA) Enable for the Channel 26
#define AT91C_HCBDMA_CBEN27   ((unsigned int) 0x1 << 27) // (HCBDMA) Enable for the Channel 27
#define AT91C_HCBDMA_CBEN28   ((unsigned int) 0x1 << 28) // (HCBDMA) Enable for the Channel 28
#define AT91C_HCBDMA_CBEN29   ((unsigned int) 0x1 << 29) // (HCBDMA) Enable for the Channel 29
#define AT91C_HCBDMA_CBEN30   ((unsigned int) 0x1 << 30) // (HCBDMA) Enable for the Channel 30
#define AT91C_HCBDMA_CBEN31   ((unsigned int) 0x1 << 31) // (HCBDMA) Enable for the Channel 31
// -------- HCBDMA_CBDIS : (HCBDMA Offset: 0xc) CB DMA Disable Register -------- 
#define AT91C_HCBDMA_CBDIS0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Disable for the Channel 0
#define AT91C_HCBDMA_CBDIS1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Disable for the Channel 1
#define AT91C_HCBDMA_CBDIS2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Disable for the Channel 2
#define AT91C_HCBDMA_CBDIS3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Disable for the Channel 3
#define AT91C_HCBDMA_CBDIS4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Disable for the Channel 4
#define AT91C_HCBDMA_CBDIS5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Disable for the Channel 5
#define AT91C_HCBDMA_CBDIS6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Disable for the Channel 6
#define AT91C_HCBDMA_CBDIS7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Disable for the Channel 7
#define AT91C_HCBDMA_CBDIS8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Disable for the Channel 8
#define AT91C_HCBDMA_CBDIS9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Disable for the Channel 9
#define AT91C_HCBDMA_CBDIS10  ((unsigned int) 0x1 << 10) // (HCBDMA) Disable for the Channel 10
#define AT91C_HCBDMA_CBDIS11  ((unsigned int) 0x1 << 11) // (HCBDMA) Disable for the Channel 11
#define AT91C_HCBDMA_CBDIS12  ((unsigned int) 0x1 << 12) // (HCBDMA) Disable for the Channel 12
#define AT91C_HCBDMA_CBDIS13  ((unsigned int) 0x1 << 13) // (HCBDMA) Disable for the Channel 13
#define AT91C_HCBDMA_CBDIS14  ((unsigned int) 0x1 << 14) // (HCBDMA) Disable for the Channel 14
#define AT91C_HCBDMA_CBDIS15  ((unsigned int) 0x1 << 15) // (HCBDMA) Disable for the Channel 15
#define AT91C_HCBDMA_CBDIS16  ((unsigned int) 0x1 << 16) // (HCBDMA) Disable for the Channel 16
#define AT91C_HCBDMA_CBDIS17  ((unsigned int) 0x1 << 17) // (HCBDMA) Disable for the Channel 17
#define AT91C_HCBDMA_CBDIS18  ((unsigned int) 0x1 << 18) // (HCBDMA) Disable for the Channel 18
#define AT91C_HCBDMA_CBDIS19  ((unsigned int) 0x1 << 19) // (HCBDMA) Disable for the Channel 19
#define AT91C_HCBDMA_CBDIS20  ((unsigned int) 0x1 << 20) // (HCBDMA) Disable for the Channel 20
#define AT91C_HCBDMA_CBDIS21  ((unsigned int) 0x1 << 21) // (HCBDMA) Disable for the Channel 21
#define AT91C_HCBDMA_CBDIS22  ((unsigned int) 0x1 << 22) // (HCBDMA) Disable for the Channel 22
#define AT91C_HCBDMA_CBDIS23  ((unsigned int) 0x1 << 23) // (HCBDMA) Disable for the Channel 23
#define AT91C_HCBDMA_CBDIS24  ((unsigned int) 0x1 << 24) // (HCBDMA) Disable for the Channel 24
#define AT91C_HCBDMA_CBDIS25  ((unsigned int) 0x1 << 25) // (HCBDMA) Disable for the Channel 25
#define AT91C_HCBDMA_CBDIS26  ((unsigned int) 0x1 << 26) // (HCBDMA) Disable for the Channel 26
#define AT91C_HCBDMA_CBDIS27  ((unsigned int) 0x1 << 27) // (HCBDMA) Disable for the Channel 27
#define AT91C_HCBDMA_CBDIS28  ((unsigned int) 0x1 << 28) // (HCBDMA) Disable for the Channel 28
#define AT91C_HCBDMA_CBDIS29  ((unsigned int) 0x1 << 29) // (HCBDMA) Disable for the Channel 29
#define AT91C_HCBDMA_CBDIS30  ((unsigned int) 0x1 << 30) // (HCBDMA) Disable for the Channel 30
#define AT91C_HCBDMA_CBDIS31  ((unsigned int) 0x1 << 31) // (HCBDMA) Disable for the Channel 31
// -------- HCBDMA_CBSR : (HCBDMA Offset: 0x10) CB DMA Status Register -------- 
#define AT91C_HCBDMA_CBSR0    ((unsigned int) 0x1 <<  0) // (HCBDMA) Status for the Channel 0
#define AT91C_HCBDMA_CBSR1    ((unsigned int) 0x1 <<  1) // (HCBDMA) Status for the Channel 1
#define AT91C_HCBDMA_CBSR2    ((unsigned int) 0x1 <<  2) // (HCBDMA) Status for the Channel 2
#define AT91C_HCBDMA_CBSR3    ((unsigned int) 0x1 <<  3) // (HCBDMA) Status for the Channel 3
#define AT91C_HCBDMA_CBSR4    ((unsigned int) 0x1 <<  4) // (HCBDMA) Status for the Channel 4
#define AT91C_HCBDMA_CBSR5    ((unsigned int) 0x1 <<  5) // (HCBDMA) Status for the Channel 5
#define AT91C_HCBDMA_CBSR6    ((unsigned int) 0x1 <<  6) // (HCBDMA) Status for the Channel 6
#define AT91C_HCBDMA_CBSR7    ((unsigned int) 0x1 <<  7) // (HCBDMA) Status for the Channel 7
#define AT91C_HCBDMA_CBSR8    ((unsigned int) 0x1 <<  8) // (HCBDMA) Status for the Channel 8
#define AT91C_HCBDMA_CBSR9    ((unsigned int) 0x1 <<  9) // (HCBDMA) Status for the Channel 9
#define AT91C_HCBDMA_CBSR10   ((unsigned int) 0x1 << 10) // (HCBDMA) Status for the Channel 10
#define AT91C_HCBDMA_CBSR11   ((unsigned int) 0x1 << 11) // (HCBDMA) Status for the Channel 11
#define AT91C_HCBDMA_CBSR12   ((unsigned int) 0x1 << 12) // (HCBDMA) Status for the Channel 12
#define AT91C_HCBDMA_CBSR13   ((unsigned int) 0x1 << 13) // (HCBDMA) Status for the Channel 13
#define AT91C_HCBDMA_CBSR14   ((unsigned int) 0x1 << 14) // (HCBDMA) Status for the Channel 14
#define AT91C_HCBDMA_CBSR15   ((unsigned int) 0x1 << 15) // (HCBDMA) Status for the Channel 15
#define AT91C_HCBDMA_CBSR16   ((unsigned int) 0x1 << 16) // (HCBDMA) Status for the Channel 16
#define AT91C_HCBDMA_CBSR17   ((unsigned int) 0x1 << 17) // (HCBDMA) Status for the Channel 17
#define AT91C_HCBDMA_CBSR18   ((unsigned int) 0x1 << 18) // (HCBDMA) Status for the Channel 18
#define AT91C_HCBDMA_CBSR19   ((unsigned int) 0x1 << 19) // (HCBDMA) Status for the Channel 19
#define AT91C_HCBDMA_CBSR20   ((unsigned int) 0x1 << 20) // (HCBDMA) Status for the Channel 20
#define AT91C_HCBDMA_CBSR21   ((unsigned int) 0x1 << 21) // (HCBDMA) Status for the Channel 21
#define AT91C_HCBDMA_CBSR22   ((unsigned int) 0x1 << 22) // (HCBDMA) Status for the Channel 22
#define AT91C_HCBDMA_CBSR23   ((unsigned int) 0x1 << 23) // (HCBDMA) Status for the Channel 23
#define AT91C_HCBDMA_CBSR24   ((unsigned int) 0x1 << 24) // (HCBDMA) Status for the Channel 24
#define AT91C_HCBDMA_CBSR25   ((unsigned int) 0x1 << 25) // (HCBDMA) Status for the Channel 25
#define AT91C_HCBDMA_CBSR26   ((unsigned int) 0x1 << 26) // (HCBDMA) Status for the Channel 26
#define AT91C_HCBDMA_CBSR27   ((unsigned int) 0x1 << 27) // (HCBDMA) Status for the Channel 27
#define AT91C_HCBDMA_CBSR28   ((unsigned int) 0x1 << 28) // (HCBDMA) Status for the Channel 28
#define AT91C_HCBDMA_CBSR29   ((unsigned int) 0x1 << 29) // (HCBDMA) Status for the Channel 29
#define AT91C_HCBDMA_CBSR30   ((unsigned int) 0x1 << 30) // (HCBDMA) Status for the Channel 30
#define AT91C_HCBDMA_CBSR31   ((unsigned int) 0x1 << 31) // (HCBDMA) Status for the Channel 31
// -------- HCBDMA_CBIER : (HCBDMA Offset: 0x14) CB DMA Interrupt Enable Register -------- 
#define AT91C_HCBDMA_CBIER0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt enable for the Channel 0
#define AT91C_HCBDMA_CBIER1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt enable for the Channel 1
#define AT91C_HCBDMA_CBIER2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt enable for the Channel 2
#define AT91C_HCBDMA_CBIER3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt enable for the Channel 3
#define AT91C_HCBDMA_CBIER4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt enable for the Channel 4
#define AT91C_HCBDMA_CBIER5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt enable for the Channel 5
#define AT91C_HCBDMA_CBIER6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt enable for the Channel 6
#define AT91C_HCBDMA_CBIER7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt enable for the Channel 7
#define AT91C_HCBDMA_CBIER8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt enable for the Channel 8
#define AT91C_HCBDMA_CBIER9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt enable for the Channel 9
#define AT91C_HCBDMA_CBIER10  ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt enable for the Channel 10
#define AT91C_HCBDMA_CBIER11  ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt enable for the Channel 11
#define AT91C_HCBDMA_CBIER12  ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt enable for the Channel 12
#define AT91C_HCBDMA_CBIER13  ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt enable for the Channel 13
#define AT91C_HCBDMA_CBIER14  ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt enable for the Channel 14
#define AT91C_HCBDMA_CBIER15  ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt enable for the Channel 15
#define AT91C_HCBDMA_CBIER16  ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt enable for the Channel 16
#define AT91C_HCBDMA_CBIER17  ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt enable for the Channel 17
#define AT91C_HCBDMA_CBIER18  ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt enable for the Channel 18
#define AT91C_HCBDMA_CBIER19  ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt enable for the Channel 19
#define AT91C_HCBDMA_CBIER20  ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt enable for the Channel 20
#define AT91C_HCBDMA_CBIER21  ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt enable for the Channel 21
#define AT91C_HCBDMA_CBIER22  ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt enable for the Channel 22
#define AT91C_HCBDMA_CBIER23  ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt enable for the Channel 23
#define AT91C_HCBDMA_CBIER24  ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt enable for the Channel 24
#define AT91C_HCBDMA_CBIER25  ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt enable for the Channel 25
#define AT91C_HCBDMA_CBIER26  ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt enable for the Channel 26
#define AT91C_HCBDMA_CBIER27  ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt enable for the Channel 27
#define AT91C_HCBDMA_CBIER28  ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt enable for the Channel 28
#define AT91C_HCBDMA_CBIER29  ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt enable for the Channel 29
#define AT91C_HCBDMA_CBIER30  ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt enable for the Channel 30
#define AT91C_HCBDMA_CBIER31  ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt enable for the Channel 31
// -------- HCBDMA_CBIDR : (HCBDMA Offset: 0x18) CB DMA Interrupt Disable Register -------- 
#define AT91C_HCBDMA_CBIDR0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt disable for the Channel 0
#define AT91C_HCBDMA_CBIDR1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt disable for the Channel 1
#define AT91C_HCBDMA_CBIDR2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt disable for the Channel 2
#define AT91C_HCBDMA_CBIDR3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt disable for the Channel 3
#define AT91C_HCBDMA_CBIDR4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt disable for the Channel 4
#define AT91C_HCBDMA_CBIDR5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt disable for the Channel 5
#define AT91C_HCBDMA_CBIDR6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt disable for the Channel 6
#define AT91C_HCBDMA_CBIDR7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt disable for the Channel 7
#define AT91C_HCBDMA_CBIDR8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt disable for the Channel 8
#define AT91C_HCBDMA_CBIDR9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt disable for the Channel 9
#define AT91C_HCBDMA_CBIDR10  ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt disable for the Channel 10
#define AT91C_HCBDMA_CBIDR11  ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt disable for the Channel 11
#define AT91C_HCBDMA_CBIDR12  ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt disable for the Channel 12
#define AT91C_HCBDMA_CBIDR13  ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt disable for the Channel 13
#define AT91C_HCBDMA_CBIDR14  ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt disable for the Channel 14
#define AT91C_HCBDMA_CBIDR15  ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt disable for the Channel 15
#define AT91C_HCBDMA_CBIDR16  ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt disable for the Channel 16
#define AT91C_HCBDMA_CBIDR17  ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt disable for the Channel 17
#define AT91C_HCBDMA_CBIDR18  ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt disable for the Channel 18
#define AT91C_HCBDMA_CBIDR19  ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt disable for the Channel 19
#define AT91C_HCBDMA_CBIDR20  ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt disable for the Channel 20
#define AT91C_HCBDMA_CBIDR21  ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt disable for the Channel 21
#define AT91C_HCBDMA_CBIDR22  ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt disable for the Channel 22
#define AT91C_HCBDMA_CBIDR23  ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt disable for the Channel 23
#define AT91C_HCBDMA_CBIDR24  ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt disable for the Channel 24
#define AT91C_HCBDMA_CBIDR25  ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt disable for the Channel 25
#define AT91C_HCBDMA_CBIDR26  ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt disable for the Channel 26
#define AT91C_HCBDMA_CBIDR27  ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt disable for the Channel 27
#define AT91C_HCBDMA_CBIDR28  ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt disable for the Channel 28
#define AT91C_HCBDMA_CBIDR29  ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt disable for the Channel 29
#define AT91C_HCBDMA_CBIDR30  ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt disable for the Channel 30
#define AT91C_HCBDMA_CBIDR31  ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt disable for the Channel 31
// -------- HCBDMA_CBIMR : (HCBDMA Offset: 0x1c) CB DMA Interrupt Mask Register -------- 
#define AT91C_HCBDMA_CBIMR0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt mask for the Channel 0
#define AT91C_HCBDMA_CBIMR1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt mask for the Channel 1
#define AT91C_HCBDMA_CBIMR2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt mask for the Channel 2
#define AT91C_HCBDMA_CBIMR3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt mask for the Channel 3
#define AT91C_HCBDMA_CBIMR4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt mask for the Channel 4
#define AT91C_HCBDMA_CBIMR5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt mask for the Channel 5
#define AT91C_HCBDMA_CBIMR6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt mask for the Channel 6
#define AT91C_HCBDMA_CBIMR7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt mask for the Channel 7
#define AT91C_HCBDMA_CBIMR8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt mask for the Channel 8
#define AT91C_HCBDMA_CBIMR9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt mask for the Channel 9
#define AT91C_HCBDMA_CBIMR10  ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt mask for the Channel 10
#define AT91C_HCBDMA_CBIMR11  ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt mask for the Channel 11
#define AT91C_HCBDMA_CBIMR12  ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt mask for the Channel 12
#define AT91C_HCBDMA_CBIMR13  ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt mask for the Channel 13
#define AT91C_HCBDMA_CBIMR14  ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt mask for the Channel 14
#define AT91C_HCBDMA_CBIMR15  ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt mask for the Channel 15
#define AT91C_HCBDMA_CBIMR16  ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt mask for the Channel 16
#define AT91C_HCBDMA_CBIMR17  ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt mask for the Channel 17
#define AT91C_HCBDMA_CBIMR18  ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt mask for the Channel 18
#define AT91C_HCBDMA_CBIMR19  ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt mask for the Channel 19
#define AT91C_HCBDMA_CBIMR20  ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt mask for the Channel 20
#define AT91C_HCBDMA_CBIMR21  ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt mask for the Channel 21
#define AT91C_HCBDMA_CBIMR22  ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt mask for the Channel 22
#define AT91C_HCBDMA_CBIMR23  ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt mask for the Channel 23
#define AT91C_HCBDMA_CBIMR24  ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt mask for the Channel 24
#define AT91C_HCBDMA_CBIMR25  ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt mask for the Channel 25
#define AT91C_HCBDMA_CBIMR26  ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt mask for the Channel 26
#define AT91C_HCBDMA_CBIMR27  ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt mask for the Channel 27
#define AT91C_HCBDMA_CBIMR28  ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt mask for the Channel 28
#define AT91C_HCBDMA_CBIMR29  ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt mask for the Channel 29
#define AT91C_HCBDMA_CBIMR30  ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt mask for the Channel 30
#define AT91C_HCBDMA_CBIMR31  ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt mask for the Channel 31
// -------- HCBDMA_CBISR : (HCBDMA Offset: 0x20) CB DMA Interrupt Satus Register -------- 
#define AT91C_HCBDMA_CBISR0   ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt status for the Channel 0
#define AT91C_HCBDMA_CBISR1   ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt status for the Channel 1
#define AT91C_HCBDMA_CBISR2   ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt status for the Channel 2
#define AT91C_HCBDMA_CBISR3   ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt status for the Channel 3
#define AT91C_HCBDMA_CBISR4   ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt status for the Channel 4
#define AT91C_HCBDMA_CBISR5   ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt status for the Channel 5
#define AT91C_HCBDMA_CBISR6   ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt status for the Channel 6
#define AT91C_HCBDMA_CBISR7   ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt status for the Channel 7
#define AT91C_HCBDMA_CBISR8   ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt status for the Channel 8
#define AT91C_HCBDMA_CBISR9   ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt status for the Channel 9
#define AT91C_HCBDMA_CBISR10  ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt status for the Channel 10
#define AT91C_HCBDMA_CBISR11  ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt status for the Channel 11
#define AT91C_HCBDMA_CBISR12  ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt status for the Channel 12
#define AT91C_HCBDMA_CBISR13  ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt status for the Channel 13
#define AT91C_HCBDMA_CBISR14  ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt status for the Channel 14
#define AT91C_HCBDMA_CBISR15  ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt status for the Channel 15
#define AT91C_HCBDMA_CBISR16  ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt status for the Channel 16
#define AT91C_HCBDMA_CBISR17  ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt status for the Channel 17
#define AT91C_HCBDMA_CBISR18  ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt status for the Channel 18
#define AT91C_HCBDMA_CBISR19  ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt status for the Channel 19
#define AT91C_HCBDMA_CBISR20  ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt status for the Channel 20
#define AT91C_HCBDMA_CBISR21  ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt status for the Channel 21
#define AT91C_HCBDMA_CBISR22  ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt status for the Channel 22
#define AT91C_HCBDMA_CBISR23  ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt status for the Channel 23
#define AT91C_HCBDMA_CBISR24  ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt status for the Channel 24
#define AT91C_HCBDMA_CBISR25  ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt status for the Channel 25
#define AT91C_HCBDMA_CBISR26  ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt status for the Channel 26
#define AT91C_HCBDMA_CBISR27  ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt status for the Channel 27
#define AT91C_HCBDMA_CBISR28  ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt status for the Channel 28
#define AT91C_HCBDMA_CBISR29  ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt status for the Channel 29
#define AT91C_HCBDMA_CBISR30  ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt status for the Channel 30
#define AT91C_HCBDMA_CBISR31  ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt status for the Channel 31
// -------- HCBDMA_CBDLIER : (HCBDMA Offset: 0x24) CB DMA Loaded Interrupt Enable Register -------- 
#define AT91C_HCBDMA_CBDLIER0 ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt enable for the Channel 0
#define AT91C_HCBDMA_CBDLIER1 ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt enable for the Channel 1
#define AT91C_HCBDMA_CBDLIER2 ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt enable for the Channel 2
#define AT91C_HCBDMA_CBDLIER3 ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt enable for the Channel 3
#define AT91C_HCBDMA_CBDLIER4 ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt enable for the Channel 4
#define AT91C_HCBDMA_CBDLIER5 ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt enable for the Channel 5
#define AT91C_HCBDMA_CBDLIER6 ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt enable for the Channel 6
#define AT91C_HCBDMA_CBDLIER7 ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt enable for the Channel 7
#define AT91C_HCBDMA_CBDLIER8 ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt enable for the Channel 8
#define AT91C_HCBDMA_CBDLIER9 ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt enable for the Channel 9
#define AT91C_HCBDMA_CBDLIER10 ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt enable for the Channel 10
#define AT91C_HCBDMA_CBDLIER11 ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt enable for the Channel 11
#define AT91C_HCBDMA_CBDLIER12 ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt enable for the Channel 12
#define AT91C_HCBDMA_CBDLIER13 ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt enable for the Channel 13
#define AT91C_HCBDMA_CBDLIER14 ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt enable for the Channel 14
#define AT91C_HCBDMA_CBDLIER15 ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt enable for the Channel 15
#define AT91C_HCBDMA_CBDLIER16 ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt enable for the Channel 16
#define AT91C_HCBDMA_CBDLIER17 ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt enable for the Channel 17
#define AT91C_HCBDMA_CBDLIER18 ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt enable for the Channel 18
#define AT91C_HCBDMA_CBDLIER19 ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt enable for the Channel 19
#define AT91C_HCBDMA_CBDLIER20 ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt enable for the Channel 20
#define AT91C_HCBDMA_CBDLIER21 ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt enable for the Channel 21
#define AT91C_HCBDMA_CBDLIER22 ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt enable for the Channel 22
#define AT91C_HCBDMA_CBDLIER23 ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt enable for the Channel 23
#define AT91C_HCBDMA_CBDLIER24 ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt enable for the Channel 24
#define AT91C_HCBDMA_CBDLIER25 ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt enable for the Channel 25
#define AT91C_HCBDMA_CBDLIER26 ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt enable for the Channel 26
#define AT91C_HCBDMA_CBDLIER27 ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt enable for the Channel 27
#define AT91C_HCBDMA_CBDLIER28 ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt enable for the Channel 28
#define AT91C_HCBDMA_CBDLIER29 ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt enable for the Channel 29
#define AT91C_HCBDMA_CBDLIER30 ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt enable for the Channel 30
#define AT91C_HCBDMA_CBDLIER31 ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt enable for the Channel 31
// -------- HCBDMA_CBDLIDR : (HCBDMA Offset: 0x28) CB DMA Interrupt Disable Register -------- 
#define AT91C_HCBDMA_CBDLIDR0 ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt disable for the Channel 0
#define AT91C_HCBDMA_CBDLIDR1 ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt disable for the Channel 1
#define AT91C_HCBDMA_CBDLIDR2 ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt disable for the Channel 2
#define AT91C_HCBDMA_CBDLIDR3 ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt disable for the Channel 3
#define AT91C_HCBDMA_CBDLIDR4 ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt disable for the Channel 4
#define AT91C_HCBDMA_CBDLIDR5 ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt disable for the Channel 5
#define AT91C_HCBDMA_CBDLIDR6 ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt disable for the Channel 6
#define AT91C_HCBDMA_CBDLIDR7 ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt disable for the Channel 7
#define AT91C_HCBDMA_CBDLIDR8 ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt disable for the Channel 8
#define AT91C_HCBDMA_CBDLIDR9 ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt disable for the Channel 9
#define AT91C_HCBDMA_CBDLIDR10 ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt disable for the Channel 10
#define AT91C_HCBDMA_CBDLIDR11 ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt disable for the Channel 11
#define AT91C_HCBDMA_CBDLIDR12 ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt disable for the Channel 12
#define AT91C_HCBDMA_CBDLIDR13 ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt disable for the Channel 13
#define AT91C_HCBDMA_CBDLIDR14 ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt disable for the Channel 14
#define AT91C_HCBDMA_CBDLIDR15 ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt disable for the Channel 15
#define AT91C_HCBDMA_CBDLIDR16 ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt disable for the Channel 16
#define AT91C_HCBDMA_CBDLIDR17 ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt disable for the Channel 17
#define AT91C_HCBDMA_CBDLIDR18 ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt disable for the Channel 18
#define AT91C_HCBDMA_CBDLIDR19 ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt disable for the Channel 19
#define AT91C_HCBDMA_CBDLIDR20 ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt disable for the Channel 20
#define AT91C_HCBDMA_CBDLIDR21 ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt disable for the Channel 21
#define AT91C_HCBDMA_CBDLIDR22 ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt disable for the Channel 22
#define AT91C_HCBDMA_CBDLIDR23 ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt disable for the Channel 23
#define AT91C_HCBDMA_CBDLIDR24 ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt disable for the Channel 24
#define AT91C_HCBDMA_CBDLIDR25 ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt disable for the Channel 25
#define AT91C_HCBDMA_CBDLIDR26 ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt disable for the Channel 26
#define AT91C_HCBDMA_CBDLIDR27 ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt disable for the Channel 27
#define AT91C_HCBDMA_CBDLIDR28 ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt disable for the Channel 28
#define AT91C_HCBDMA_CBDLIDR29 ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt disable for the Channel 29
#define AT91C_HCBDMA_CBDLIDR30 ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt disable for the Channel 30
#define AT91C_HCBDMA_CBDLIDR31 ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt disable for the Channel 31
// -------- HCBDMA_CBDLIMR : (HCBDMA Offset: 0x2c) CB DMA Interrupt Mask Register -------- 
#define AT91C_HCBDMA_CBDLIMR0 ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt mask for the Channel 0
#define AT91C_HCBDMA_CBDLIMR1 ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt mask for the Channel 1
#define AT91C_HCBDMA_CBDLIMR2 ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt mask for the Channel 2
#define AT91C_HCBDMA_CBDLIMR3 ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt mask for the Channel 3
#define AT91C_HCBDMA_CBDLIMR4 ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt mask for the Channel 4
#define AT91C_HCBDMA_CBDLIMR5 ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt mask for the Channel 5
#define AT91C_HCBDMA_CBDLIMR6 ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt mask for the Channel 6
#define AT91C_HCBDMA_CBDLIMR7 ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt mask for the Channel 7
#define AT91C_HCBDMA_CBDLIMR8 ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt mask for the Channel 8
#define AT91C_HCBDMA_CBDLIMR9 ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt mask for the Channel 9
#define AT91C_HCBDMA_CBDLIMR10 ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt mask for the Channel 10
#define AT91C_HCBDMA_CBDLIMR11 ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt mask for the Channel 11
#define AT91C_HCBDMA_CBDLIMR12 ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt mask for the Channel 12
#define AT91C_HCBDMA_CBDLIMR13 ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt mask for the Channel 13
#define AT91C_HCBDMA_CBDLIMR14 ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt mask for the Channel 14
#define AT91C_HCBDMA_CBDLIMR15 ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt mask for the Channel 15
#define AT91C_HCBDMA_CBDLIMR16 ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt mask for the Channel 16
#define AT91C_HCBDMA_CBDLIMR17 ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt mask for the Channel 17
#define AT91C_HCBDMA_CBDLIMR18 ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt mask for the Channel 18
#define AT91C_HCBDMA_CBDLIMR19 ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt mask for the Channel 19
#define AT91C_HCBDMA_CBDLIMR20 ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt mask for the Channel 20
#define AT91C_HCBDMA_CBDLIMR21 ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt mask for the Channel 21
#define AT91C_HCBDMA_CBDLIMR22 ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt mask for the Channel 22
#define AT91C_HCBDMA_CBDLIMR23 ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt mask for the Channel 23
#define AT91C_HCBDMA_CBDLIMR24 ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt mask for the Channel 24
#define AT91C_HCBDMA_CBDLIMR25 ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt mask for the Channel 25
#define AT91C_HCBDMA_CBDLIMR26 ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt mask for the Channel 26
#define AT91C_HCBDMA_CBDLIMR27 ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt mask for the Channel 27
#define AT91C_HCBDMA_CBDLIMR28 ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt mask for the Channel 28
#define AT91C_HCBDMA_CBDLIMR29 ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt mask for the Channel 29
#define AT91C_HCBDMA_CBDLIMR30 ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt mask for the Channel 30
#define AT91C_HCBDMA_CBDLIMR31 ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt mask for the Channel 31
// -------- HCBDMA_CBDLISR : (HCBDMA Offset: 0x30) CB DMA Interrupt Satus Register -------- 
#define AT91C_HCBDMA_CBDLISR0 ((unsigned int) 0x1 <<  0) // (HCBDMA) Interrupt status for the Channel 0
#define AT91C_HCBDMA_CBDLISR1 ((unsigned int) 0x1 <<  1) // (HCBDMA) Interrupt status for the Channel 1
#define AT91C_HCBDMA_CBDLISR2 ((unsigned int) 0x1 <<  2) // (HCBDMA) Interrupt status for the Channel 2
#define AT91C_HCBDMA_CBDLISR3 ((unsigned int) 0x1 <<  3) // (HCBDMA) Interrupt status for the Channel 3
#define AT91C_HCBDMA_CBDLISR4 ((unsigned int) 0x1 <<  4) // (HCBDMA) Interrupt status for the Channel 4
#define AT91C_HCBDMA_CBDLISR5 ((unsigned int) 0x1 <<  5) // (HCBDMA) Interrupt status for the Channel 5
#define AT91C_HCBDMA_CBDLISR6 ((unsigned int) 0x1 <<  6) // (HCBDMA) Interrupt status for the Channel 6
#define AT91C_HCBDMA_CBDLISR7 ((unsigned int) 0x1 <<  7) // (HCBDMA) Interrupt status for the Channel 7
#define AT91C_HCBDMA_CBDLISR8 ((unsigned int) 0x1 <<  8) // (HCBDMA) Interrupt status for the Channel 8
#define AT91C_HCBDMA_CBDLISR9 ((unsigned int) 0x1 <<  9) // (HCBDMA) Interrupt status for the Channel 9
#define AT91C_HCBDMA_CBDLISR10 ((unsigned int) 0x1 << 10) // (HCBDMA) Interrupt status for the Channel 10
#define AT91C_HCBDMA_CBDLISR11 ((unsigned int) 0x1 << 11) // (HCBDMA) Interrupt status for the Channel 11
#define AT91C_HCBDMA_CBDLISR12 ((unsigned int) 0x1 << 12) // (HCBDMA) Interrupt status for the Channel 12
#define AT91C_HCBDMA_CBDLISR13 ((unsigned int) 0x1 << 13) // (HCBDMA) Interrupt status for the Channel 13
#define AT91C_HCBDMA_CBDLISR14 ((unsigned int) 0x1 << 14) // (HCBDMA) Interrupt status for the Channel 14
#define AT91C_HCBDMA_CBDLISR15 ((unsigned int) 0x1 << 15) // (HCBDMA) Interrupt status for the Channel 15
#define AT91C_HCBDMA_CBDLISR16 ((unsigned int) 0x1 << 16) // (HCBDMA) Interrupt status for the Channel 16
#define AT91C_HCBDMA_CBDLISR17 ((unsigned int) 0x1 << 17) // (HCBDMA) Interrupt status for the Channel 17
#define AT91C_HCBDMA_CBDLISR18 ((unsigned int) 0x1 << 18) // (HCBDMA) Interrupt status for the Channel 18
#define AT91C_HCBDMA_CBDLISR19 ((unsigned int) 0x1 << 19) // (HCBDMA) Interrupt status for the Channel 19
#define AT91C_HCBDMA_CBDLISR20 ((unsigned int) 0x1 << 20) // (HCBDMA) Interrupt status for the Channel 20
#define AT91C_HCBDMA_CBDLISR21 ((unsigned int) 0x1 << 21) // (HCBDMA) Interrupt status for the Channel 21
#define AT91C_HCBDMA_CBDLISR22 ((unsigned int) 0x1 << 22) // (HCBDMA) Interrupt status for the Channel 22
#define AT91C_HCBDMA_CBDLISR23 ((unsigned int) 0x1 << 23) // (HCBDMA) Interrupt status for the Channel 23
#define AT91C_HCBDMA_CBDLISR24 ((unsigned int) 0x1 << 24) // (HCBDMA) Interrupt status for the Channel 24
#define AT91C_HCBDMA_CBDLISR25 ((unsigned int) 0x1 << 25) // (HCBDMA) Interrupt status for the Channel 25
#define AT91C_HCBDMA_CBDLISR26 ((unsigned int) 0x1 << 26) // (HCBDMA) Interrupt status for the Channel 26
#define AT91C_HCBDMA_CBDLISR27 ((unsigned int) 0x1 << 27) // (HCBDMA) Interrupt status for the Channel 27
#define AT91C_HCBDMA_CBDLISR28 ((unsigned int) 0x1 << 28) // (HCBDMA) Interrupt status for the Channel 28
#define AT91C_HCBDMA_CBDLISR29 ((unsigned int) 0x1 << 29) // (HCBDMA) Interrupt status for the Channel 29
#define AT91C_HCBDMA_CBDLISR30 ((unsigned int) 0x1 << 30) // (HCBDMA) Interrupt status for the Channel 30
#define AT91C_HCBDMA_CBDLISR31 ((unsigned int) 0x1 << 31) // (HCBDMA) Interrupt status for the Channel 31
// -------- HCBDMA_CBCRCCR : (HCBDMA Offset: 0x34) CB DMA CRC Control Resgister -------- 
#define AT91C_CRC_START       ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC compuration initialization
// -------- HCBDMA_CBCRCMR : (HCBDMA Offset: 0x38) CB DMA CRC Mode Resgister -------- 
#define AT91C_CRC_ENABLE      ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC Enable
#define AT91C_CRC_COMPARE     ((unsigned int) 0x1 <<  1) // (HCBDMA) CRC Compare
#define AT91C_CRC_PTYPE       ((unsigned int) 0x3 <<  2) // (HCBDMA) Primitive polynomial type
#define 	AT91C_CRC_PTYPE_CCIT802_3            ((unsigned int) 0x0 <<  2) // (HCBDMA) 
#define 	AT91C_CRC_PTYPE_CASTAGNOLI           ((unsigned int) 0x1 <<  2) // (HCBDMA) 
#define 	AT91C_CRC_PTYPE_CCIT_16              ((unsigned int) 0x2 <<  2) // (HCBDMA) 
#define AT91C_CRC_DIVIDER     ((unsigned int) 0xF <<  4) // (HCBDMA) Request Divider
#define AT91C_CRC_ID          ((unsigned int) 0x1F <<  8) // (HCBDMA) CRC channel Identifier
#define 	AT91C_CRC_ID_CHANNEL_0            ((unsigned int) 0x0 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_1            ((unsigned int) 0x1 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_2            ((unsigned int) 0x2 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_3            ((unsigned int) 0x3 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_4            ((unsigned int) 0x4 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_5            ((unsigned int) 0x5 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_6            ((unsigned int) 0x6 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_7            ((unsigned int) 0x7 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_8            ((unsigned int) 0x8 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_9            ((unsigned int) 0x9 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_10           ((unsigned int) 0xA <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_11           ((unsigned int) 0xB <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_12           ((unsigned int) 0xC <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_13           ((unsigned int) 0xD <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_14           ((unsigned int) 0xE <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_15           ((unsigned int) 0xF <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_16           ((unsigned int) 0x10 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_17           ((unsigned int) 0x11 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_18           ((unsigned int) 0x12 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_19           ((unsigned int) 0x13 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_20           ((unsigned int) 0x14 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_21           ((unsigned int) 0x15 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_22           ((unsigned int) 0x16 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_23           ((unsigned int) 0x17 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_24           ((unsigned int) 0x18 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_25           ((unsigned int) 0x19 <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_26           ((unsigned int) 0x1A <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_27           ((unsigned int) 0x1B <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_28           ((unsigned int) 0x1C <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_29           ((unsigned int) 0x1D <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_30           ((unsigned int) 0x1E <<  8) // (HCBDMA) 
#define 	AT91C_CRC_ID_CHANNEL_31           ((unsigned int) 0x1F <<  8) // (HCBDMA) 
// -------- HCBDMA_CBCRCSR : (HCBDMA Offset: 0x3c) CB DMA CRC Status Resgister -------- 
#define AT91C_HCBDMA_CRCSR    ((unsigned int) 0x0 <<  0) // (HCBDMA) CRC Status Resgister
// -------- HCBDMA_CBCRCIER : (HCBDMA Offset: 0x40) CB DMA CRC Interrupt Enable Resgister -------- 
#define AT91C_CRC_ERRIER      ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC Error Interrupt Enable
// -------- HCBDMA_CBCRCIDR : (HCBDMA Offset: 0x44) CB DMA CRC Interrupt Enable Resgister -------- 
#define AT91C_CRC_ERRIDR      ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC Error Interrupt Disable
// -------- HCBDMA_CBCRCIMR : (HCBDMA Offset: 0x48) CB DMA CRC Interrupt Mask Resgister -------- 
#define AT91C_CRC_ERRIMR      ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC Error Interrupt Mask
// -------- HCBDMA_CBCRCISR : (HCBDMA Offset: 0x4c) CB DMA CRC Interrupt Status Resgister -------- 
#define AT91C_CRC_ERRISR      ((unsigned int) 0x1 <<  0) // (HCBDMA) CRC Error Interrupt Status

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
	AT91_REG	 Reserved2[2]; 	// 
	AT91_REG	 SSC_SR; 	// Status Register
	AT91_REG	 SSC_IER; 	// Interrupt Enable Register
	AT91_REG	 SSC_IDR; 	// Interrupt Disable Register
	AT91_REG	 SSC_IMR; 	// Interrupt Mask Register
	AT91_REG	 Reserved3[40]; 	// 
	AT91_REG	 SSC_ADDRSIZE; 	// SSC ADDRSIZE REGISTER 
	AT91_REG	 SSC_NAME; 	// SSC NAME REGISTER 
	AT91_REG	 SSC_FEATURES; 	// SSC FEATURES REGISTER 
	AT91_REG	 SSC_VER; 	// Version Register
	AT91_REG	 SSC_RPR; 	// Receive Pointer Register
	AT91_REG	 SSC_RCR; 	// Receive Counter Register
	AT91_REG	 SSC_TPR; 	// Transmit Pointer Register
	AT91_REG	 SSC_TCR; 	// Transmit Counter Register
	AT91_REG	 SSC_RNPR; 	// Receive Next Pointer Register
	AT91_REG	 SSC_RNCR; 	// Receive Next Counter Register
	AT91_REG	 SSC_TNPR; 	// Transmit Next Pointer Register
	AT91_REG	 SSC_TNCR; 	// Transmit Next Counter Register
	AT91_REG	 SSC_PTCR; 	// PDC Transfer Control Register
	AT91_REG	 SSC_PTSR; 	// PDC Transfer Status Register
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
	AT91_REG	 Reserved1[5]; 	// 
	AT91_REG	 PWMC_STEP; 	// PWM Stepper Config Register
	AT91_REG	 Reserved2[12]; 	// 
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
	AT91_REG	 Reserved3[2]; 	// 
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
	AT91_REG	 Reserved4[20]; 	// 
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
//              SOFTWARE API DEFINITION  FOR USB Device Interface
// *****************************************************************************
typedef struct _AT91S_UDP {
	AT91_REG	 UDP_NUM; 	// Frame Number Register
	AT91_REG	 UDP_GLBSTATE; 	// Global State Register
	AT91_REG	 UDP_FADDR; 	// Function Address Register
	AT91_REG	 Reserved0[1]; 	// 
	AT91_REG	 UDP_IER; 	// Interrupt Enable Register
	AT91_REG	 UDP_IDR; 	// Interrupt Disable Register
	AT91_REG	 UDP_IMR; 	// Interrupt Mask Register
	AT91_REG	 UDP_ISR; 	// Interrupt Status Register
	AT91_REG	 UDP_ICR; 	// Interrupt Clear Register
	AT91_REG	 Reserved1[1]; 	// 
	AT91_REG	 UDP_RSTEP; 	// Reset Endpoint Register
	AT91_REG	 Reserved2[1]; 	// 
	AT91_REG	 UDP_CSR[8]; 	// Endpoint Control and Status Register
	AT91_REG	 UDP_FDR[8]; 	// Endpoint FIFO Data Register
	AT91_REG	 Reserved3[1]; 	// 
	AT91_REG	 UDP_TXVC; 	// Transceiver Control Register
	AT91_REG	 Reserved4[29]; 	// 
	AT91_REG	 UDP_ADDRSIZE; 	// UDP ADDRSIZE REGISTER 
	AT91_REG	 UDP_IPNAME1; 	// UDP IPNAME1 REGISTER 
	AT91_REG	 UDP_IPNAME2; 	// UDP IPNAME2 REGISTER 
	AT91_REG	 UDP_FEATURES; 	// UDP FEATURES REGISTER 
	AT91_REG	 UDP_VER; 	// UDP VERSION REGISTER
} AT91S_UDP, *AT91PS_UDP;

// -------- UDP_FRM_NUM : (UDP Offset: 0x0) USB Frame Number Register -------- 
#define AT91C_UDP_FRM_NUM     ((unsigned int) 0x7FF <<  0) // (UDP) Frame Number as Defined in the Packet Field Formats
#define AT91C_UDP_FRM_ERR     ((unsigned int) 0x1 << 16) // (UDP) Frame Error
#define AT91C_UDP_FRM_OK      ((unsigned int) 0x1 << 17) // (UDP) Frame OK
// -------- UDP_GLB_STATE : (UDP Offset: 0x4) USB Global State Register -------- 
#define AT91C_UDP_FADDEN      ((unsigned int) 0x1 <<  0) // (UDP) Function Address Enable
#define AT91C_UDP_CONFG       ((unsigned int) 0x1 <<  1) // (UDP) Configured
#define AT91C_UDP_ESR         ((unsigned int) 0x1 <<  2) // (UDP) Enable Send Resume
#define AT91C_UDP_RSMINPR     ((unsigned int) 0x1 <<  3) // (UDP) A Resume Has Been Sent to the Host
#define AT91C_UDP_RMWUPE      ((unsigned int) 0x1 <<  4) // (UDP) Remote Wake Up Enable
// -------- UDP_FADDR : (UDP Offset: 0x8) USB Function Address Register -------- 
#define AT91C_UDP_FADD        ((unsigned int) 0xFF <<  0) // (UDP) Function Address Value
#define AT91C_UDP_FEN         ((unsigned int) 0x1 <<  8) // (UDP) Function Enable
// -------- UDP_IER : (UDP Offset: 0x10) USB Interrupt Enable Register -------- 
#define AT91C_UDP_EPINT0      ((unsigned int) 0x1 <<  0) // (UDP) Endpoint 0 Interrupt
#define AT91C_UDP_EPINT1      ((unsigned int) 0x1 <<  1) // (UDP) Endpoint 0 Interrupt
#define AT91C_UDP_EPINT2      ((unsigned int) 0x1 <<  2) // (UDP) Endpoint 2 Interrupt
#define AT91C_UDP_EPINT3      ((unsigned int) 0x1 <<  3) // (UDP) Endpoint 3 Interrupt
#define AT91C_UDP_EPINT4      ((unsigned int) 0x1 <<  4) // (UDP) Endpoint 4 Interrupt
#define AT91C_UDP_EPINT5      ((unsigned int) 0x1 <<  5) // (UDP) Endpoint 5 Interrupt
#define AT91C_UDP_EPINT6      ((unsigned int) 0x1 <<  6) // (UDP) Endpoint 6 Interrupt
#define AT91C_UDP_EPINT7      ((unsigned int) 0x1 <<  7) // (UDP) Endpoint 7 Interrupt
#define AT91C_UDP_RXSUSP      ((unsigned int) 0x1 <<  8) // (UDP) USB Suspend Interrupt
#define AT91C_UDP_RXRSM       ((unsigned int) 0x1 <<  9) // (UDP) USB Resume Interrupt
#define AT91C_UDP_EXTRSM      ((unsigned int) 0x1 << 10) // (UDP) USB External Resume Interrupt
#define AT91C_UDP_SOFINT      ((unsigned int) 0x1 << 11) // (UDP) USB Start Of frame Interrupt
#define AT91C_UDP_WAKEUP      ((unsigned int) 0x1 << 13) // (UDP) USB Resume Interrupt
// -------- UDP_IDR : (UDP Offset: 0x14) USB Interrupt Disable Register -------- 
// -------- UDP_IMR : (UDP Offset: 0x18) USB Interrupt Mask Register -------- 
// -------- UDP_ISR : (UDP Offset: 0x1c) USB Interrupt Status Register -------- 
#define AT91C_UDP_ENDBUSRES   ((unsigned int) 0x1 << 12) // (UDP) USB End Of Bus Reset Interrupt
// -------- UDP_ICR : (UDP Offset: 0x20) USB Interrupt Clear Register -------- 
// -------- UDP_RST_EP : (UDP Offset: 0x28) USB Reset Endpoint Register -------- 
#define AT91C_UDP_EP0         ((unsigned int) 0x1 <<  0) // (UDP) Reset Endpoint 0
#define AT91C_UDP_EP1         ((unsigned int) 0x1 <<  1) // (UDP) Reset Endpoint 1
#define AT91C_UDP_EP2         ((unsigned int) 0x1 <<  2) // (UDP) Reset Endpoint 2
#define AT91C_UDP_EP3         ((unsigned int) 0x1 <<  3) // (UDP) Reset Endpoint 3
#define AT91C_UDP_EP4         ((unsigned int) 0x1 <<  4) // (UDP) Reset Endpoint 4
#define AT91C_UDP_EP5         ((unsigned int) 0x1 <<  5) // (UDP) Reset Endpoint 5
#define AT91C_UDP_EP6         ((unsigned int) 0x1 <<  6) // (UDP) Reset Endpoint 6
#define AT91C_UDP_EP7         ((unsigned int) 0x1 <<  7) // (UDP) Reset Endpoint 7
// -------- UDP_CSR : (UDP Offset: 0x30) USB Endpoint Control and Status Register -------- 
#define AT91C_UDP_TXCOMP      ((unsigned int) 0x1 <<  0) // (UDP) Generates an IN packet with data previously written in the DPR
#define AT91C_UDP_RX_DATA_BK0 ((unsigned int) 0x1 <<  1) // (UDP) Receive Data Bank 0
#define AT91C_UDP_RXSETUP     ((unsigned int) 0x1 <<  2) // (UDP) Sends STALL to the Host (Control endpoints)
#define AT91C_UDP_ISOERROR    ((unsigned int) 0x1 <<  3) // (UDP) Isochronous error (Isochronous endpoints)
#define AT91C_UDP_STALLSENT   ((unsigned int) 0x1 <<  3) // (UDP) Stall sent (Control, bulk, interrupt endpoints)
#define AT91C_UDP_TXPKTRDY    ((unsigned int) 0x1 <<  4) // (UDP) Transmit Packet Ready
#define AT91C_UDP_FORCESTALL  ((unsigned int) 0x1 <<  5) // (UDP) Force Stall (used by Control, Bulk and Isochronous endpoints).
#define AT91C_UDP_RX_DATA_BK1 ((unsigned int) 0x1 <<  6) // (UDP) Receive Data Bank 1 (only used by endpoints with ping-pong attributes).
#define AT91C_UDP_DIR         ((unsigned int) 0x1 <<  7) // (UDP) Transfer Direction
#define AT91C_UDP_EPTYPE      ((unsigned int) 0x7 <<  8) // (UDP) Endpoint type
#define 	AT91C_UDP_EPTYPE_CTRL                 ((unsigned int) 0x0 <<  8) // (UDP) Control
#define 	AT91C_UDP_EPTYPE_ISO_OUT              ((unsigned int) 0x1 <<  8) // (UDP) Isochronous OUT
#define 	AT91C_UDP_EPTYPE_BULK_OUT             ((unsigned int) 0x2 <<  8) // (UDP) Bulk OUT
#define 	AT91C_UDP_EPTYPE_INT_OUT              ((unsigned int) 0x3 <<  8) // (UDP) Interrupt OUT
#define 	AT91C_UDP_EPTYPE_ISO_IN               ((unsigned int) 0x5 <<  8) // (UDP) Isochronous IN
#define 	AT91C_UDP_EPTYPE_BULK_IN              ((unsigned int) 0x6 <<  8) // (UDP) Bulk IN
#define 	AT91C_UDP_EPTYPE_INT_IN               ((unsigned int) 0x7 <<  8) // (UDP) Interrupt IN
#define AT91C_UDP_DTGLE       ((unsigned int) 0x1 << 11) // (UDP) Data Toggle
#define AT91C_UDP_EPEDS       ((unsigned int) 0x1 << 15) // (UDP) Endpoint Enable Disable
#define AT91C_UDP_RXBYTECNT   ((unsigned int) 0x7FF << 16) // (UDP) Number Of Bytes Available in the FIFO
// -------- UDP_TXVC : (UDP Offset: 0x74) Transceiver Control Register -------- 
#define AT91C_UDP_TXVDIS      ((unsigned int) 0x1 <<  8) // (UDP) 
#define AT91C_UDP_PUON        ((unsigned int) 0x1 <<  9) // (UDP) Pull-up ON

// *****************************************************************************
//               REGISTER ADDRESS DEFINITION FOR AT91SAM3S1
// *****************************************************************************
// ========== Register definition for SYS peripheral ========== 
#define AT91C_SYS_GPBR  ((AT91_REG *) 	0x400E1490) // (SYS) General Purpose Register
// ========== Register definition for SMC peripheral ========== 
#define AT91C_SMC_DELAY2 ((AT91_REG *) 	0x400E00C4) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE4 ((AT91_REG *) 	0x400E0048) // (SMC)  Cycle Register for CS 4
#define AT91C_SMC_CTRL5 ((AT91_REG *) 	0x400E005C) // (SMC)  Control Register for CS 5
#define AT91C_SMC_IPNAME2 ((AT91_REG *) 	0x400E00F4) // (SMC) HSMC3 IPNAME2 REGISTER 
#define AT91C_SMC_DELAY5 ((AT91_REG *) 	0x400E00D0) // (SMC) SMC Delay Control Register
#define AT91C_SMC_DELAY4 ((AT91_REG *) 	0x400E00CC) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE0 ((AT91_REG *) 	0x400E0008) // (SMC)  Cycle Register for CS 0
#define AT91C_SMC_PULSE1 ((AT91_REG *) 	0x400E0014) // (SMC)  Pulse Register for CS 1
#define AT91C_SMC_DELAY6 ((AT91_REG *) 	0x400E00D4) // (SMC) SMC Delay Control Register
#define AT91C_SMC_FEATURES ((AT91_REG *) 	0x400E00F8) // (SMC) HSMC3 FEATURES REGISTER 
#define AT91C_SMC_DELAY3 ((AT91_REG *) 	0x400E00C8) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CTRL1 ((AT91_REG *) 	0x400E001C) // (SMC)  Control Register for CS 1
#define AT91C_SMC_PULSE7 ((AT91_REG *) 	0x400E0074) // (SMC)  Pulse Register for CS 7
#define AT91C_SMC_CTRL7 ((AT91_REG *) 	0x400E007C) // (SMC)  Control Register for CS 7
#define AT91C_SMC_VER   ((AT91_REG *) 	0x400E00FC) // (SMC) HSMC3 VERSION REGISTER
#define AT91C_SMC_SETUP5 ((AT91_REG *) 	0x400E0050) // (SMC)  Setup Register for CS 5
#define AT91C_SMC_CYCLE3 ((AT91_REG *) 	0x400E0038) // (SMC)  Cycle Register for CS 3
#define AT91C_SMC_SETUP3 ((AT91_REG *) 	0x400E0030) // (SMC)  Setup Register for CS 3
#define AT91C_SMC_DELAY1 ((AT91_REG *) 	0x400E00C0) // (SMC) SMC Delay Control Register
#define AT91C_SMC_ADDRSIZE ((AT91_REG *) 	0x400E00EC) // (SMC) HSMC3 ADDRSIZE REGISTER 
#define AT91C_SMC_PULSE3 ((AT91_REG *) 	0x400E0034) // (SMC)  Pulse Register for CS 3
#define AT91C_SMC_PULSE5 ((AT91_REG *) 	0x400E0054) // (SMC)  Pulse Register for CS 5
#define AT91C_SMC_PULSE4 ((AT91_REG *) 	0x400E0044) // (SMC)  Pulse Register for CS 4
#define AT91C_SMC_SETUP2 ((AT91_REG *) 	0x400E0020) // (SMC)  Setup Register for CS 2
#define AT91C_SMC_DELAY8 ((AT91_REG *) 	0x400E00DC) // (SMC) SMC Delay Control Register
#define AT91C_SMC_CYCLE7 ((AT91_REG *) 	0x400E0078) // (SMC)  Cycle Register for CS 7
#define AT91C_SMC_CTRL0 ((AT91_REG *) 	0x400E000C) // (SMC)  Control Register for CS 0
#define AT91C_SMC_CYCLE2 ((AT91_REG *) 	0x400E0028) // (SMC)  Cycle Register for CS 2
#define AT91C_SMC_IPNAME1 ((AT91_REG *) 	0x400E00F0) // (SMC) HSMC3 IPNAME1 REGISTER 
#define AT91C_SMC_SETUP1 ((AT91_REG *) 	0x400E0010) // (SMC)  Setup Register for CS 1
#define AT91C_SMC_CTRL2 ((AT91_REG *) 	0x400E002C) // (SMC)  Control Register for CS 2
#define AT91C_SMC_CTRL4 ((AT91_REG *) 	0x400E004C) // (SMC)  Control Register for CS 4
#define AT91C_SMC_SETUP6 ((AT91_REG *) 	0x400E0060) // (SMC)  Setup Register for CS 6
#define AT91C_SMC_CYCLE5 ((AT91_REG *) 	0x400E0058) // (SMC)  Cycle Register for CS 5
#define AT91C_SMC_CTRL6 ((AT91_REG *) 	0x400E006C) // (SMC)  Control Register for CS 6
#define AT91C_SMC_SETUP4 ((AT91_REG *) 	0x400E0040) // (SMC)  Setup Register for CS 4
#define AT91C_SMC_PULSE2 ((AT91_REG *) 	0x400E0024) // (SMC)  Pulse Register for CS 2
#define AT91C_SMC_DELAY7 ((AT91_REG *) 	0x400E00D8) // (SMC) SMC Delay Control Register
#define AT91C_SMC_SETUP7 ((AT91_REG *) 	0x400E0070) // (SMC)  Setup Register for CS 7
#define AT91C_SMC_CYCLE1 ((AT91_REG *) 	0x400E0018) // (SMC)  Cycle Register for CS 1
#define AT91C_SMC_CTRL3 ((AT91_REG *) 	0x400E003C) // (SMC)  Control Register for CS 3
#define AT91C_SMC_SETUP0 ((AT91_REG *) 	0x400E0000) // (SMC)  Setup Register for CS 0
#define AT91C_SMC_PULSE0 ((AT91_REG *) 	0x400E0004) // (SMC)  Pulse Register for CS 0
#define AT91C_SMC_PULSE6 ((AT91_REG *) 	0x400E0064) // (SMC)  Pulse Register for CS 6
#define AT91C_SMC_CYCLE6 ((AT91_REG *) 	0x400E0068) // (SMC)  Cycle Register for CS 6
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
// ========== Register definition for CCFG peripheral ========== 
#define AT91C_CCFG_FLASH0 ((AT91_REG *) 	0x400E0318) // (CCFG)  FLASH0 configuration
#define AT91C_CCFG_RAM0 ((AT91_REG *) 	0x400E0310) // (CCFG)  RAM0 configuration
#define AT91C_CCFG_ROM  ((AT91_REG *) 	0x400E0314) // (CCFG)  ROM  configuration
#define AT91C_CCFG_EBICSA ((AT91_REG *) 	0x400E031C) // (CCFG)  EBI Chip Select Assignement Register
#define AT91C_CCFG_BRIDGE ((AT91_REG *) 	0x400E0320) // (CCFG)  BRIDGE configuration
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
// ========== Register definition for PDC_DBGU0 peripheral ========== 
#define AT91C_DBGU0_TPR ((AT91_REG *) 	0x400E0708) // (PDC_DBGU0) Transmit Pointer Register
#define AT91C_DBGU0_PTCR ((AT91_REG *) 	0x400E0720) // (PDC_DBGU0) PDC Transfer Control Register
#define AT91C_DBGU0_TNCR ((AT91_REG *) 	0x400E071C) // (PDC_DBGU0) Transmit Next Counter Register
#define AT91C_DBGU0_PTSR ((AT91_REG *) 	0x400E0724) // (PDC_DBGU0) PDC Transfer Status Register
#define AT91C_DBGU0_RNCR ((AT91_REG *) 	0x400E0714) // (PDC_DBGU0) Receive Next Counter Register
#define AT91C_DBGU0_RPR ((AT91_REG *) 	0x400E0700) // (PDC_DBGU0) Receive Pointer Register
#define AT91C_DBGU0_TCR ((AT91_REG *) 	0x400E070C) // (PDC_DBGU0) Transmit Counter Register
#define AT91C_DBGU0_RNPR ((AT91_REG *) 	0x400E0710) // (PDC_DBGU0) Receive Next Pointer Register
#define AT91C_DBGU0_TNPR ((AT91_REG *) 	0x400E0718) // (PDC_DBGU0) Transmit Next Pointer Register
#define AT91C_DBGU0_RCR ((AT91_REG *) 	0x400E0704) // (PDC_DBGU0) Receive Counter Register
// ========== Register definition for DBGU0 peripheral ========== 
#define AT91C_DBGU0_CR  ((AT91_REG *) 	0x400E0600) // (DBGU0) Control Register
#define AT91C_DBGU0_IDR ((AT91_REG *) 	0x400E060C) // (DBGU0) Interrupt Disable Register
#define AT91C_DBGU0_CIDR ((AT91_REG *) 	0x400E0740) // (DBGU0) Chip ID Register
#define AT91C_DBGU0_IPNAME2 ((AT91_REG *) 	0x400E06F4) // (DBGU0) DBGU IPNAME2 REGISTER 
#define AT91C_DBGU0_FEATURES ((AT91_REG *) 	0x400E06F8) // (DBGU0) DBGU FEATURES REGISTER 
#define AT91C_DBGU0_FNTR ((AT91_REG *) 	0x400E0648) // (DBGU0) Force NTRST Register
#define AT91C_DBGU0_RHR ((AT91_REG *) 	0x400E0618) // (DBGU0) Receiver Holding Register
#define AT91C_DBGU0_THR ((AT91_REG *) 	0x400E061C) // (DBGU0) Transmitter Holding Register
#define AT91C_DBGU0_ADDRSIZE ((AT91_REG *) 	0x400E06EC) // (DBGU0) DBGU ADDRSIZE REGISTER 
#define AT91C_DBGU0_MR  ((AT91_REG *) 	0x400E0604) // (DBGU0) Mode Register
#define AT91C_DBGU0_IER ((AT91_REG *) 	0x400E0608) // (DBGU0) Interrupt Enable Register
#define AT91C_DBGU0_BRGR ((AT91_REG *) 	0x400E0620) // (DBGU0) Baud Rate Generator Register
#define AT91C_DBGU0_CSR ((AT91_REG *) 	0x400E0614) // (DBGU0) Channel Status Register
#define AT91C_DBGU0_VER ((AT91_REG *) 	0x400E06FC) // (DBGU0) DBGU VERSION REGISTER 
#define AT91C_DBGU0_IMR ((AT91_REG *) 	0x400E0610) // (DBGU0) Interrupt Mask Register
#define AT91C_DBGU0_IPNAME1 ((AT91_REG *) 	0x400E06F0) // (DBGU0) DBGU IPNAME1 REGISTER 
#define AT91C_DBGU0_EXID ((AT91_REG *) 	0x400E0744) // (DBGU0) Chip ID Extension Register
// ========== Register definition for PDC_DBGU1 peripheral ========== 
#define AT91C_DBGU1_RNCR ((AT91_REG *) 	0x400E0914) // (PDC_DBGU1) Receive Next Counter Register
#define AT91C_DBGU1_RPR ((AT91_REG *) 	0x400E0900) // (PDC_DBGU1) Receive Pointer Register
#define AT91C_DBGU1_TNCR ((AT91_REG *) 	0x400E091C) // (PDC_DBGU1) Transmit Next Counter Register
#define AT91C_DBGU1_TNPR ((AT91_REG *) 	0x400E0918) // (PDC_DBGU1) Transmit Next Pointer Register
#define AT91C_DBGU1_PTSR ((AT91_REG *) 	0x400E0924) // (PDC_DBGU1) PDC Transfer Status Register
#define AT91C_DBGU1_PTCR ((AT91_REG *) 	0x400E0920) // (PDC_DBGU1) PDC Transfer Control Register
#define AT91C_DBGU1_RCR ((AT91_REG *) 	0x400E0904) // (PDC_DBGU1) Receive Counter Register
#define AT91C_DBGU1_RNPR ((AT91_REG *) 	0x400E0910) // (PDC_DBGU1) Receive Next Pointer Register
#define AT91C_DBGU1_TPR ((AT91_REG *) 	0x400E0908) // (PDC_DBGU1) Transmit Pointer Register
#define AT91C_DBGU1_TCR ((AT91_REG *) 	0x400E090C) // (PDC_DBGU1) Transmit Counter Register
// ========== Register definition for DBGU1 peripheral ========== 
#define AT91C_DBGU1_RHR ((AT91_REG *) 	0x400E0818) // (DBGU1) Receiver Holding Register
#define AT91C_DBGU1_IPNAME1 ((AT91_REG *) 	0x400E08F0) // (DBGU1) DBGU IPNAME1 REGISTER 
#define AT91C_DBGU1_CIDR ((AT91_REG *) 	0x400E0940) // (DBGU1) Chip ID Register
#define AT91C_DBGU1_CR  ((AT91_REG *) 	0x400E0800) // (DBGU1) Control Register
#define AT91C_DBGU1_VER ((AT91_REG *) 	0x400E08FC) // (DBGU1) DBGU VERSION REGISTER 
#define AT91C_DBGU1_IPNAME2 ((AT91_REG *) 	0x400E08F4) // (DBGU1) DBGU IPNAME2 REGISTER 
#define AT91C_DBGU1_BRGR ((AT91_REG *) 	0x400E0820) // (DBGU1) Baud Rate Generator Register
#define AT91C_DBGU1_FNTR ((AT91_REG *) 	0x400E0848) // (DBGU1) Force NTRST Register
#define AT91C_DBGU1_MR  ((AT91_REG *) 	0x400E0804) // (DBGU1) Mode Register
#define AT91C_DBGU1_ADDRSIZE ((AT91_REG *) 	0x400E08EC) // (DBGU1) DBGU ADDRSIZE REGISTER 
#define AT91C_DBGU1_CSR ((AT91_REG *) 	0x400E0814) // (DBGU1) Channel Status Register
#define AT91C_DBGU1_IMR ((AT91_REG *) 	0x400E0810) // (DBGU1) Interrupt Mask Register
#define AT91C_DBGU1_EXID ((AT91_REG *) 	0x400E0944) // (DBGU1) Chip ID Extension Register
#define AT91C_DBGU1_IDR ((AT91_REG *) 	0x400E080C) // (DBGU1) Interrupt Disable Register
#define AT91C_DBGU1_FEATURES ((AT91_REG *) 	0x400E08F8) // (DBGU1) DBGU FEATURES REGISTER 
#define AT91C_DBGU1_IER ((AT91_REG *) 	0x400E0808) // (DBGU1) Interrupt Enable Register
#define AT91C_DBGU1_THR ((AT91_REG *) 	0x400E081C) // (DBGU1) Transmitter Holding Register
// ========== Register definition for PIOA peripheral ========== 
#define AT91C_PIOA_SENIDR ((AT91_REG *) 	0x400E0F58) // (PIOA) Sensor Interrupt Disable Register
#define AT91C_PIOA_OWSR ((AT91_REG *) 	0x400E0EA8) // (PIOA) Output Write Status Register
#define AT91C_PIOA_PSR  ((AT91_REG *) 	0x400E0E08) // (PIOA) PIO Status Register
#define AT91C_PIOA_MDER ((AT91_REG *) 	0x400E0E50) // (PIOA) Multi-driver Enable Register
#define AT91C_PIOA_IPNAME1 ((AT91_REG *) 	0x400E0EF0) // (PIOA) PIO IPNAME1 REGISTER 
#define AT91C_PIOA_FEATURES ((AT91_REG *) 	0x400E0EF8) // (PIOA) PIO FEATURES REGISTER 
#define AT91C_PIOA_SP2  ((AT91_REG *) 	0x400E0E74) // (PIOA) Select B Register
#define AT91C_PIOA_ODR  ((AT91_REG *) 	0x400E0E14) // (PIOA) Output Disable Registerr
#define AT91C_PIOA_IDR  ((AT91_REG *) 	0x400E0E44) // (PIOA) Interrupt Disable Register
#define AT91C_PIOA_PDR  ((AT91_REG *) 	0x400E0E04) // (PIOA) PIO Disable Register
#define AT91C_PIOA_PDSR ((AT91_REG *) 	0x400E0E3C) // (PIOA) Pin Data Status Register
#define AT91C_PIOA_PPDER ((AT91_REG *) 	0x400E0E94) // (PIOA) Pull-down Enable Register
#define AT91C_PIOA_SENIER ((AT91_REG *) 	0x400E0F54) // (PIOA) Sensor Interrupt Enable Register
#define AT91C_PIOA_SLEW2 ((AT91_REG *) 	0x400E0F04) // (PIOA) PIO SLEWRATE2 REGISTER 
#define AT91C_PIOA_SENMR ((AT91_REG *) 	0x400E0F50) // (PIOA) Sensor Mode Register
#define AT91C_PIOA_PPUDR ((AT91_REG *) 	0x400E0E60) // (PIOA) Pull-up Disable Register
#define AT91C_PIOA_OWDR ((AT91_REG *) 	0x400E0EA4) // (PIOA) Output Write Disable Register
#define AT91C_PIOA_ADDRSIZE ((AT91_REG *) 	0x400E0EEC) // (PIOA) PIO ADDRSIZE REGISTER 
#define AT91C_PIOA_IFER ((AT91_REG *) 	0x400E0E20) // (PIOA) Input Filter Enable Register
#define AT91C_PIOA_PPDSR ((AT91_REG *) 	0x400E0E98) // (PIOA) Pull-down Status Register
#define AT91C_PIOA_SP1  ((AT91_REG *) 	0x400E0E70) // (PIOA) Select B Register
#define AT91C_PIOA_SENIMR ((AT91_REG *) 	0x400E0F5C) // (PIOA) Sensor Interrupt Mask Register
#define AT91C_PIOA_SENDATA ((AT91_REG *) 	0x400E0F64) // (PIOA) Sensor Data Register
#define AT91C_PIOA_VER  ((AT91_REG *) 	0x400E0EFC) // (PIOA) PIO VERSION REGISTER 
#define AT91C_PIOA_PER  ((AT91_REG *) 	0x400E0E00) // (PIOA) PIO Enable Register
#define AT91C_PIOA_IMR  ((AT91_REG *) 	0x400E0E48) // (PIOA) Interrupt Mask Register
#define AT91C_PIOA_PPUSR ((AT91_REG *) 	0x400E0E68) // (PIOA) Pull-up Status Register
#define AT91C_PIOA_ODSR ((AT91_REG *) 	0x400E0E38) // (PIOA) Output Data Status Register
#define AT91C_PIOA_SENISR ((AT91_REG *) 	0x400E0F60) // (PIOA) Sensor Interrupt Status Register
#define AT91C_PIOA_IFDR ((AT91_REG *) 	0x400E0E24) // (PIOA) Input Filter Disable Register
#define AT91C_PIOA_SODR ((AT91_REG *) 	0x400E0E30) // (PIOA) Set Output Data Register
#define AT91C_PIOA_SLEW1 ((AT91_REG *) 	0x400E0F00) // (PIOA) PIO SLEWRATE1 REGISTER 
#define AT91C_PIOA_IER  ((AT91_REG *) 	0x400E0E40) // (PIOA) Interrupt Enable Register
#define AT91C_PIOA_MDSR ((AT91_REG *) 	0x400E0E58) // (PIOA) Multi-driver Status Register
#define AT91C_PIOA_ISR  ((AT91_REG *) 	0x400E0E4C) // (PIOA) Interrupt Status Register
#define AT91C_PIOA_IFSR ((AT91_REG *) 	0x400E0E28) // (PIOA) Input Filter Status Register
#define AT91C_PIOA_PPDDR ((AT91_REG *) 	0x400E0E90) // (PIOA) Pull-down Disable Register
#define AT91C_PIOA_PPUER ((AT91_REG *) 	0x400E0E64) // (PIOA) Pull-up Enable Register
#define AT91C_PIOA_OWER ((AT91_REG *) 	0x400E0EA0) // (PIOA) Output Write Enable Register
#define AT91C_PIOA_IPNAME2 ((AT91_REG *) 	0x400E0EF4) // (PIOA) PIO IPNAME2 REGISTER 
#define AT91C_PIOA_MDDR ((AT91_REG *) 	0x400E0E54) // (PIOA) Multi-driver Disable Register
#define AT91C_PIOA_CODR ((AT91_REG *) 	0x400E0E34) // (PIOA) Clear Output Data Register
#define AT91C_PIOA_OER  ((AT91_REG *) 	0x400E0E10) // (PIOA) Output Enable Register
#define AT91C_PIOA_OSR  ((AT91_REG *) 	0x400E0E18) // (PIOA) Output Status Register
#define AT91C_PIOA_ABSR ((AT91_REG *) 	0x400E0E78) // (PIOA) AB Select Status Register
// ========== Register definition for PDC_PIOA peripheral ========== 
#define AT91C_PIOA_RPR  ((AT91_REG *) 	0x400E0F68) // (PDC_PIOA) Receive Pointer Register
#define AT91C_PIOA_TPR  ((AT91_REG *) 	0x400E0F70) // (PDC_PIOA) Transmit Pointer Register
#define AT91C_PIOA_RCR  ((AT91_REG *) 	0x400E0F6C) // (PDC_PIOA) Receive Counter Register
#define AT91C_PIOA_PTSR ((AT91_REG *) 	0x400E0F8C) // (PDC_PIOA) PDC Transfer Status Register
#define AT91C_PIOA_TCR  ((AT91_REG *) 	0x400E0F74) // (PDC_PIOA) Transmit Counter Register
#define AT91C_PIOA_PTCR ((AT91_REG *) 	0x400E0F88) // (PDC_PIOA) PDC Transfer Control Register
#define AT91C_PIOA_RNPR ((AT91_REG *) 	0x400E0F78) // (PDC_PIOA) Receive Next Pointer Register
#define AT91C_PIOA_TNCR ((AT91_REG *) 	0x400E0F84) // (PDC_PIOA) Transmit Next Counter Register
#define AT91C_PIOA_RNCR ((AT91_REG *) 	0x400E0F7C) // (PDC_PIOA) Receive Next Counter Register
#define AT91C_PIOA_TNPR ((AT91_REG *) 	0x400E0F80) // (PDC_PIOA) Transmit Next Pointer Register
// ========== Register definition for PIOB peripheral ========== 
#define AT91C_PIOB_MDDR ((AT91_REG *) 	0x400E1054) // (PIOB) Multi-driver Disable Register
#define AT91C_PIOB_ABSR ((AT91_REG *) 	0x400E1078) // (PIOB) AB Select Status Register
#define AT91C_PIOB_SP1  ((AT91_REG *) 	0x400E1070) // (PIOB) Select B Register
#define AT91C_PIOB_ODSR ((AT91_REG *) 	0x400E1038) // (PIOB) Output Data Status Register
#define AT91C_PIOB_SLEW1 ((AT91_REG *) 	0x400E1100) // (PIOB) PIO SLEWRATE1 REGISTER 
#define AT91C_PIOB_SENISR ((AT91_REG *) 	0x400E1160) // (PIOB) Sensor Interrupt Status Register
#define AT91C_PIOB_OSR  ((AT91_REG *) 	0x400E1018) // (PIOB) Output Status Register
#define AT91C_PIOB_IFER ((AT91_REG *) 	0x400E1020) // (PIOB) Input Filter Enable Register
#define AT91C_PIOB_SENDATA ((AT91_REG *) 	0x400E1164) // (PIOB) Sensor Data Register
#define AT91C_PIOB_MDSR ((AT91_REG *) 	0x400E1058) // (PIOB) Multi-driver Status Register
#define AT91C_PIOB_IFDR ((AT91_REG *) 	0x400E1024) // (PIOB) Input Filter Disable Register
#define AT91C_PIOB_MDER ((AT91_REG *) 	0x400E1050) // (PIOB) Multi-driver Enable Register
#define AT91C_PIOB_SENIDR ((AT91_REG *) 	0x400E1158) // (PIOB) Sensor Interrupt Disable Register
#define AT91C_PIOB_IER  ((AT91_REG *) 	0x400E1040) // (PIOB) Interrupt Enable Register
#define AT91C_PIOB_OWDR ((AT91_REG *) 	0x400E10A4) // (PIOB) Output Write Disable Register
#define AT91C_PIOB_IFSR ((AT91_REG *) 	0x400E1028) // (PIOB) Input Filter Status Register
#define AT91C_PIOB_ISR  ((AT91_REG *) 	0x400E104C) // (PIOB) Interrupt Status Register
#define AT91C_PIOB_PPUDR ((AT91_REG *) 	0x400E1060) // (PIOB) Pull-up Disable Register
#define AT91C_PIOB_PDSR ((AT91_REG *) 	0x400E103C) // (PIOB) Pin Data Status Register
#define AT91C_PIOB_IPNAME2 ((AT91_REG *) 	0x400E10F4) // (PIOB) PIO IPNAME2 REGISTER 
#define AT91C_PIOB_PPUER ((AT91_REG *) 	0x400E1064) // (PIOB) Pull-up Enable Register
#define AT91C_PIOB_SLEW2 ((AT91_REG *) 	0x400E1104) // (PIOB) PIO SLEWRATE2 REGISTER 
#define AT91C_PIOB_OER  ((AT91_REG *) 	0x400E1010) // (PIOB) Output Enable Register
#define AT91C_PIOB_CODR ((AT91_REG *) 	0x400E1034) // (PIOB) Clear Output Data Register
#define AT91C_PIOB_PPDDR ((AT91_REG *) 	0x400E1090) // (PIOB) Pull-down Disable Register
#define AT91C_PIOB_OWER ((AT91_REG *) 	0x400E10A0) // (PIOB) Output Write Enable Register
#define AT91C_PIOB_VER  ((AT91_REG *) 	0x400E10FC) // (PIOB) PIO VERSION REGISTER 
#define AT91C_PIOB_PPDER ((AT91_REG *) 	0x400E1094) // (PIOB) Pull-down Enable Register
#define AT91C_PIOB_IMR  ((AT91_REG *) 	0x400E1048) // (PIOB) Interrupt Mask Register
#define AT91C_PIOB_PPUSR ((AT91_REG *) 	0x400E1068) // (PIOB) Pull-up Status Register
#define AT91C_PIOB_IPNAME1 ((AT91_REG *) 	0x400E10F0) // (PIOB) PIO IPNAME1 REGISTER 
#define AT91C_PIOB_ADDRSIZE ((AT91_REG *) 	0x400E10EC) // (PIOB) PIO ADDRSIZE REGISTER 
#define AT91C_PIOB_SP2  ((AT91_REG *) 	0x400E1074) // (PIOB) Select B Register
#define AT91C_PIOB_IDR  ((AT91_REG *) 	0x400E1044) // (PIOB) Interrupt Disable Register
#define AT91C_PIOB_SENMR ((AT91_REG *) 	0x400E1150) // (PIOB) Sensor Mode Register
#define AT91C_PIOB_SODR ((AT91_REG *) 	0x400E1030) // (PIOB) Set Output Data Register
#define AT91C_PIOB_PPDSR ((AT91_REG *) 	0x400E1098) // (PIOB) Pull-down Status Register
#define AT91C_PIOB_PSR  ((AT91_REG *) 	0x400E1008) // (PIOB) PIO Status Register
#define AT91C_PIOB_ODR  ((AT91_REG *) 	0x400E1014) // (PIOB) Output Disable Registerr
#define AT91C_PIOB_OWSR ((AT91_REG *) 	0x400E10A8) // (PIOB) Output Write Status Register
#define AT91C_PIOB_FEATURES ((AT91_REG *) 	0x400E10F8) // (PIOB) PIO FEATURES REGISTER 
#define AT91C_PIOB_PDR  ((AT91_REG *) 	0x400E1004) // (PIOB) PIO Disable Register
#define AT91C_PIOB_SENIMR ((AT91_REG *) 	0x400E115C) // (PIOB) Sensor Interrupt Mask Register
#define AT91C_PIOB_SENIER ((AT91_REG *) 	0x400E1154) // (PIOB) Sensor Interrupt Enable Register
#define AT91C_PIOB_PER  ((AT91_REG *) 	0x400E1000) // (PIOB) PIO Enable Register
// ========== Register definition for PIOC peripheral ========== 
#define AT91C_PIOC_VER  ((AT91_REG *) 	0x400E12FC) // (PIOC) PIO VERSION REGISTER 
#define AT91C_PIOC_IMR  ((AT91_REG *) 	0x400E1248) // (PIOC) Interrupt Mask Register
#define AT91C_PIOC_PSR  ((AT91_REG *) 	0x400E1208) // (PIOC) PIO Status Register
#define AT91C_PIOC_PPDSR ((AT91_REG *) 	0x400E1298) // (PIOC) Pull-down Status Register
#define AT91C_PIOC_OER  ((AT91_REG *) 	0x400E1210) // (PIOC) Output Enable Register
#define AT91C_PIOC_OSR  ((AT91_REG *) 	0x400E1218) // (PIOC) Output Status Register
#define AT91C_PIOC_MDDR ((AT91_REG *) 	0x400E1254) // (PIOC) Multi-driver Disable Register
#define AT91C_PIOC_PPUSR ((AT91_REG *) 	0x400E1268) // (PIOC) Pull-up Status Register
#define AT91C_PIOC_ODSR ((AT91_REG *) 	0x400E1238) // (PIOC) Output Data Status Register
#define AT91C_PIOC_SLEW2 ((AT91_REG *) 	0x400E1304) // (PIOC) PIO SLEWRATE2 REGISTER 
#define AT91C_PIOC_SENMR ((AT91_REG *) 	0x400E1350) // (PIOC) Sensor Mode Register
#define AT91C_PIOC_IFER ((AT91_REG *) 	0x400E1220) // (PIOC) Input Filter Enable Register
#define AT91C_PIOC_PDR  ((AT91_REG *) 	0x400E1204) // (PIOC) PIO Disable Register
#define AT91C_PIOC_MDER ((AT91_REG *) 	0x400E1250) // (PIOC) Multi-driver Enable Register
#define AT91C_PIOC_SP2  ((AT91_REG *) 	0x400E1274) // (PIOC) Select B Register
#define AT91C_PIOC_IPNAME1 ((AT91_REG *) 	0x400E12F0) // (PIOC) PIO IPNAME1 REGISTER 
#define AT91C_PIOC_IER  ((AT91_REG *) 	0x400E1240) // (PIOC) Interrupt Enable Register
#define AT91C_PIOC_OWDR ((AT91_REG *) 	0x400E12A4) // (PIOC) Output Write Disable Register
#define AT91C_PIOC_IDR  ((AT91_REG *) 	0x400E1244) // (PIOC) Interrupt Disable Register
#define AT91C_PIOC_PDSR ((AT91_REG *) 	0x400E123C) // (PIOC) Pin Data Status Register
#define AT91C_PIOC_SENIDR ((AT91_REG *) 	0x400E1358) // (PIOC) Sensor Interrupt Disable Register
#define AT91C_PIOC_SENISR ((AT91_REG *) 	0x400E1360) // (PIOC) Sensor Interrupt Status Register
#define AT91C_PIOC_PER  ((AT91_REG *) 	0x400E1200) // (PIOC) PIO Enable Register
#define AT91C_PIOC_SENDATA ((AT91_REG *) 	0x400E1364) // (PIOC) Sensor Data Register
#define AT91C_PIOC_IPNAME2 ((AT91_REG *) 	0x400E12F4) // (PIOC) PIO IPNAME2 REGISTER 
#define AT91C_PIOC_PPDDR ((AT91_REG *) 	0x400E1290) // (PIOC) Pull-down Disable Register
#define AT91C_PIOC_ADDRSIZE ((AT91_REG *) 	0x400E12EC) // (PIOC) PIO ADDRSIZE REGISTER 
#define AT91C_PIOC_IFDR ((AT91_REG *) 	0x400E1224) // (PIOC) Input Filter Disable Register
#define AT91C_PIOC_ODR  ((AT91_REG *) 	0x400E1214) // (PIOC) Output Disable Registerr
#define AT91C_PIOC_CODR ((AT91_REG *) 	0x400E1234) // (PIOC) Clear Output Data Register
#define AT91C_PIOC_MDSR ((AT91_REG *) 	0x400E1258) // (PIOC) Multi-driver Status Register
#define AT91C_PIOC_FEATURES ((AT91_REG *) 	0x400E12F8) // (PIOC) PIO FEATURES REGISTER 
#define AT91C_PIOC_IFSR ((AT91_REG *) 	0x400E1228) // (PIOC) Input Filter Status Register
#define AT91C_PIOC_PPUER ((AT91_REG *) 	0x400E1264) // (PIOC) Pull-up Enable Register
#define AT91C_PIOC_PPDER ((AT91_REG *) 	0x400E1294) // (PIOC) Pull-down Enable Register
#define AT91C_PIOC_OWSR ((AT91_REG *) 	0x400E12A8) // (PIOC) Output Write Status Register
#define AT91C_PIOC_ISR  ((AT91_REG *) 	0x400E124C) // (PIOC) Interrupt Status Register
#define AT91C_PIOC_OWER ((AT91_REG *) 	0x400E12A0) // (PIOC) Output Write Enable Register
#define AT91C_PIOC_PPUDR ((AT91_REG *) 	0x400E1260) // (PIOC) Pull-up Disable Register
#define AT91C_PIOC_SENIMR ((AT91_REG *) 	0x400E135C) // (PIOC) Sensor Interrupt Mask Register
#define AT91C_PIOC_SLEW1 ((AT91_REG *) 	0x400E1300) // (PIOC) PIO SLEWRATE1 REGISTER 
#define AT91C_PIOC_SENIER ((AT91_REG *) 	0x400E1354) // (PIOC) Sensor Interrupt Enable Register
#define AT91C_PIOC_SODR ((AT91_REG *) 	0x400E1230) // (PIOC) Set Output Data Register
#define AT91C_PIOC_SP1  ((AT91_REG *) 	0x400E1270) // (PIOC) Select B Register
#define AT91C_PIOC_ABSR ((AT91_REG *) 	0x400E1278) // (PIOC) AB Select Status Register
// ========== Register definition for PMC peripheral ========== 
#define AT91C_PMC_PLLAR ((AT91_REG *) 	0x400E0428) // (PMC) PLL Register
#define AT91C_PMC_UCKR  ((AT91_REG *) 	0x400E041C) // (PMC) UTMI Clock Configuration Register
#define AT91C_PMC_FSMR  ((AT91_REG *) 	0x400E0470) // (PMC) Fast Startup Mode Register
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0x400E0430) // (PMC) Master Clock Register
#define AT91C_PMC_SCER  ((AT91_REG *) 	0x400E0400) // (PMC) System Clock Enable Register
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0x400E0418) // (PMC) Peripheral Clock Status Register 0:31 PERI_ID
#define AT91C_PMC_MCFR  ((AT91_REG *) 	0x400E0424) // (PMC) Main Clock  Frequency Register
#define AT91C_PMC_PCER1 ((AT91_REG *) 	0x400E0500) // (PMC) Peripheral Clock Enable Register 32:63 PERI_ID
#define AT91C_PMC_FOCR  ((AT91_REG *) 	0x400E0478) // (PMC) Fault Output Clear Register
#define AT91C_PMC_PCSR1 ((AT91_REG *) 	0x400E0508) // (PMC) Peripheral Clock Status Register 32:63 PERI_ID
#define AT91C_PMC_FSPR  ((AT91_REG *) 	0x400E0474) // (PMC) Fast Startup Polarity Register
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0x400E0408) // (PMC) System Clock Status Register
#define AT91C_PMC_IDR   ((AT91_REG *) 	0x400E0464) // (PMC) Interrupt Disable Register
#define AT91C_PMC_UDPR  ((AT91_REG *) 	0x400E0438) // (PMC) USB DEV Clock Configuration Register
#define AT91C_PMC_PCDR1 ((AT91_REG *) 	0x400E0504) // (PMC) Peripheral Clock Disable Register 32:63 PERI_ID
#define AT91C_PMC_VER   ((AT91_REG *) 	0x400E04FC) // (PMC) APMC VERSION REGISTER
#define AT91C_PMC_IMR   ((AT91_REG *) 	0x400E046C) // (PMC) Interrupt Mask Register
#define AT91C_PMC_IPNAME2 ((AT91_REG *) 	0x400E04F4) // (PMC) PMC IPNAME2 REGISTER 
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0x400E0404) // (PMC) System Clock Disable Register
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0x400E0440) // (PMC) Programmable Clock Register
#define AT91C_PMC_ADDRSIZE ((AT91_REG *) 	0x400E04EC) // (PMC) PMC ADDRSIZE REGISTER 
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0x400E0414) // (PMC) Peripheral Clock Disable Register 0:31 PERI_ID
#define AT91C_PMC_MOR   ((AT91_REG *) 	0x400E0420) // (PMC) Main Oscillator Register
#define AT91C_PMC_SR    ((AT91_REG *) 	0x400E0468) // (PMC) Status Register
#define AT91C_PMC_IER   ((AT91_REG *) 	0x400E0460) // (PMC) Interrupt Enable Register
#define AT91C_PMC_PLLBR ((AT91_REG *) 	0x400E042C) // (PMC) PLL B Register
#define AT91C_PMC_IPNAME1 ((AT91_REG *) 	0x400E04F0) // (PMC) PMC IPNAME1 REGISTER 
#define AT91C_PMC_PCER  ((AT91_REG *) 	0x400E0410) // (PMC) Peripheral Clock Enable Register 0:31 PERI_ID
#define AT91C_PMC_FEATURES ((AT91_REG *) 	0x400E04F8) // (PMC) PMC FEATURES REGISTER 
#define AT91C_PMC_PCR   ((AT91_REG *) 	0x400E050C) // (PMC) Peripheral Control Register
// ========== Register definition for CKGR peripheral ========== 
#define AT91C_CKGR_PLLAR ((AT91_REG *) 	0x400E0428) // (CKGR) PLL Register
#define AT91C_CKGR_UCKR ((AT91_REG *) 	0x400E041C) // (CKGR) UTMI Clock Configuration Register
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0x400E0420) // (CKGR) Main Oscillator Register
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0x400E0424) // (CKGR) Main Clock  Frequency Register
#define AT91C_CKGR_PLLBR ((AT91_REG *) 	0x400E042C) // (CKGR) PLL B Register
// ========== Register definition for RSTC peripheral ========== 
#define AT91C_RSTC_RSR  ((AT91_REG *) 	0x400E1404) // (RSTC) Reset Status Register
#define AT91C_RSTC_RCR  ((AT91_REG *) 	0x400E1400) // (RSTC) Reset Control Register
#define AT91C_RSTC_RMR  ((AT91_REG *) 	0x400E1408) // (RSTC) Reset Mode Register
#define AT91C_RSTC_VER  ((AT91_REG *) 	0x400E14FC) // (RSTC) Version Register
// ========== Register definition for SUPC peripheral ========== 
#define AT91C_SUPC_FWUTR ((AT91_REG *) 	0x400E1428) // (SUPC) Flash Wake-up Timer Register
#define AT91C_SUPC_SR   ((AT91_REG *) 	0x400E1424) // (SUPC) Status Register
#define AT91C_SUPC_BOMR ((AT91_REG *) 	0x400E1414) // (SUPC) Brown Out Mode Register
#define AT91C_SUPC_WUMR ((AT91_REG *) 	0x400E141C) // (SUPC) Wake Up Mode Register
#define AT91C_SUPC_WUIR ((AT91_REG *) 	0x400E1420) // (SUPC) Wake Up Inputs Register
#define AT91C_SUPC_CR   ((AT91_REG *) 	0x400E1410) // (SUPC) Control Register
#define AT91C_SUPC_MR   ((AT91_REG *) 	0x400E1418) // (SUPC) Mode Register
// ========== Register definition for RTTC peripheral ========== 
#define AT91C_RTTC_RTSR ((AT91_REG *) 	0x400E143C) // (RTTC) Real-time Status Register
#define AT91C_RTTC_RTVR ((AT91_REG *) 	0x400E1438) // (RTTC) Real-time Value Register
#define AT91C_RTTC_RTMR ((AT91_REG *) 	0x400E1430) // (RTTC) Real-time Mode Register
#define AT91C_RTTC_RTAR ((AT91_REG *) 	0x400E1434) // (RTTC) Real-time Alarm Register
// ========== Register definition for WDTC peripheral ========== 
#define AT91C_WDTC_WDCR ((AT91_REG *) 	0x400E1450) // (WDTC) Watchdog Control Register
#define AT91C_WDTC_WDMR ((AT91_REG *) 	0x400E1454) // (WDTC) Watchdog Mode Register
#define AT91C_WDTC_WDSR ((AT91_REG *) 	0x400E1458) // (WDTC) Watchdog Status Register
// ========== Register definition for RTC peripheral ========== 
#define AT91C_RTC_VER   ((AT91_REG *) 	0x400E148C) // (RTC) Valid Entry Register
#define AT91C_RTC_TIMR  ((AT91_REG *) 	0x400E1468) // (RTC) Time Register
#define AT91C_RTC_CALALR ((AT91_REG *) 	0x400E1474) // (RTC) Calendar Alarm Register
#define AT91C_RTC_IER   ((AT91_REG *) 	0x400E1480) // (RTC) Interrupt Enable Register
#define AT91C_RTC_MR    ((AT91_REG *) 	0x400E1464) // (RTC) Mode Register
#define AT91C_RTC_CALR  ((AT91_REG *) 	0x400E146C) // (RTC) Calendar Register
#define AT91C_RTC_TIMALR ((AT91_REG *) 	0x400E1470) // (RTC) Time Alarm Register
#define AT91C_RTC_SCCR  ((AT91_REG *) 	0x400E147C) // (RTC) Status Clear Command Register
#define AT91C_RTC_CR    ((AT91_REG *) 	0x400E1460) // (RTC) Control Register
#define AT91C_RTC_IDR   ((AT91_REG *) 	0x400E1484) // (RTC) Interrupt Disable Register
#define AT91C_RTC_IMR   ((AT91_REG *) 	0x400E1488) // (RTC) Interrupt Mask Register
#define AT91C_RTC_SR    ((AT91_REG *) 	0x400E1478) // (RTC) Status Register
// ========== Register definition for ADC0 peripheral ========== 
#define AT91C_ADC0_CDR2 ((AT91_REG *) 	0x40038058) // (ADC0) ADC Channel Data Register 2
#define AT91C_ADC0_CGR  ((AT91_REG *) 	0x40038048) // (ADC0) Control gain register
#define AT91C_ADC0_CDR7 ((AT91_REG *) 	0x4003806C) // (ADC0) ADC Channel Data Register 7
#define AT91C_ADC0_IDR  ((AT91_REG *) 	0x40038028) // (ADC0) ADC Interrupt Disable Register
#define AT91C_ADC0_CR   ((AT91_REG *) 	0x40038000) // (ADC0) ADC Control Register
#define AT91C_ADC0_FEATURES ((AT91_REG *) 	0x400380F8) // (ADC0) ADC FEATURES REGISTER 
#define AT91C_ADC0_CWR  ((AT91_REG *) 	0x40038040) // (ADC0) unspecified
#define AT91C_ADC0_IPNAME1 ((AT91_REG *) 	0x400380F0) // (ADC0) ADC IPNAME1 REGISTER 
#define AT91C_ADC0_CDR9 ((AT91_REG *) 	0x40038074) // (ADC0) ADC Channel Data Register 9
#define AT91C_ADC0_CDR3 ((AT91_REG *) 	0x4003805C) // (ADC0) ADC Channel Data Register 3
#define AT91C_ADC0_SR   ((AT91_REG *) 	0x40038030) // (ADC0) ADC Status Register
#define AT91C_ADC0_CHER ((AT91_REG *) 	0x40038010) // (ADC0) ADC Channel Enable Register
#define AT91C_ADC0_CDR1 ((AT91_REG *) 	0x40038054) // (ADC0) ADC Channel Data Register 1
#define AT91C_ADC0_CDR6 ((AT91_REG *) 	0x40038068) // (ADC0) ADC Channel Data Register 6
#define AT91C_ADC0_MR   ((AT91_REG *) 	0x40038004) // (ADC0) ADC Mode Register
#define AT91C_ADC0_CWSR ((AT91_REG *) 	0x40038044) // (ADC0) unspecified
#define AT91C_ADC0_VER  ((AT91_REG *) 	0x400380FC) // (ADC0) ADC VERSION REGISTER
#define AT91C_ADC0_COR  ((AT91_REG *) 	0x4003804C) // (ADC0) unspecified
#define AT91C_ADC0_CDR8 ((AT91_REG *) 	0x40038070) // (ADC0) ADC Channel Data Register 8
#define AT91C_ADC0_IPNAME2 ((AT91_REG *) 	0x400380F4) // (ADC0) ADC IPNAME2 REGISTER 
#define AT91C_ADC0_CDR0 ((AT91_REG *) 	0x40038050) // (ADC0) ADC Channel Data Register 0
#define AT91C_ADC0_LCDR ((AT91_REG *) 	0x40038020) // (ADC0) ADC Last Converted Data Register
#define AT91C_ADC0_CDR12 ((AT91_REG *) 	0x40038080) // (ADC0) ADC Channel Data Register 12
#define AT91C_ADC0_CHDR ((AT91_REG *) 	0x40038014) // (ADC0) ADC Channel Disable Register
#define AT91C_ADC0_OVR  ((AT91_REG *) 	0x4003803C) // (ADC0) unspecified
#define AT91C_ADC0_CDR15 ((AT91_REG *) 	0x4003808C) // (ADC0) ADC Channel Data Register 15
#define AT91C_ADC0_CDR11 ((AT91_REG *) 	0x4003807C) // (ADC0) ADC Channel Data Register 11
#define AT91C_ADC0_ADDRSIZE ((AT91_REG *) 	0x400380EC) // (ADC0) ADC ADDRSIZE REGISTER 
#define AT91C_ADC0_CDR13 ((AT91_REG *) 	0x40038084) // (ADC0) ADC Channel Data Register 13
#define AT91C_ADC0_ACR  ((AT91_REG *) 	0x40038094) // (ADC0) unspecified
#define AT91C_ADC0_CDR5 ((AT91_REG *) 	0x40038064) // (ADC0) ADC Channel Data Register 5
#define AT91C_ADC0_CDR14 ((AT91_REG *) 	0x40038088) // (ADC0) ADC Channel Data Register 14
#define AT91C_ADC0_IMR  ((AT91_REG *) 	0x4003802C) // (ADC0) ADC Interrupt Mask Register
#define AT91C_ADC0_CHSR ((AT91_REG *) 	0x40038018) // (ADC0) ADC Channel Status Register
#define AT91C_ADC0_CDR10 ((AT91_REG *) 	0x40038078) // (ADC0) ADC Channel Data Register 10
#define AT91C_ADC0_IER  ((AT91_REG *) 	0x40038024) // (ADC0) ADC Interrupt Enable Register
#define AT91C_ADC0_CDR4 ((AT91_REG *) 	0x40038060) // (ADC0) ADC Channel Data Register 4
// ========== Register definition for DAC0 peripheral ========== 
#define AT91C_DAC0_FEATURES ((AT91_REG *) 	0x4003C0F8) // (DAC0) DAC FEATURES REGISTER 
#define AT91C_DAC0_ADDRSIZE ((AT91_REG *) 	0x4003C0EC) // (DAC0) DAC ADDRSIZE REGISTER 
#define AT91C_DAC0_WPMR ((AT91_REG *) 	0x4003C0E4) // (DAC0) Write Protect Mode Register
#define AT91C_DAC0_CHDR ((AT91_REG *) 	0x4003C014) // (DAC0) Channel Disable Register
#define AT91C_DAC0_IPNAME1 ((AT91_REG *) 	0x4003C0F0) // (DAC0) DAC IPNAME1 REGISTER 
#define AT91C_DAC0_IDR  ((AT91_REG *) 	0x4003C028) // (DAC0) Interrupt Disable Register
#define AT91C_DAC0_CR   ((AT91_REG *) 	0x4003C000) // (DAC0) Control Register
#define AT91C_DAC0_IPNAME2 ((AT91_REG *) 	0x4003C0F4) // (DAC0) DAC IPNAME2 REGISTER 
#define AT91C_DAC0_IMR  ((AT91_REG *) 	0x4003C02C) // (DAC0) Interrupt Mask Register
#define AT91C_DAC0_CHSR ((AT91_REG *) 	0x4003C018) // (DAC0) Channel Status Register
#define AT91C_DAC0_ACR  ((AT91_REG *) 	0x4003C094) // (DAC0) Analog Current Register
#define AT91C_DAC0_WPSR ((AT91_REG *) 	0x4003C0E8) // (DAC0) Write Protect Status Register
#define AT91C_DAC0_CHER ((AT91_REG *) 	0x4003C010) // (DAC0) Channel Enable Register
#define AT91C_DAC0_CDR  ((AT91_REG *) 	0x4003C020) // (DAC0) Coversion Data Register
#define AT91C_DAC0_IER  ((AT91_REG *) 	0x4003C024) // (DAC0) Interrupt Enable Register
#define AT91C_DAC0_ISR  ((AT91_REG *) 	0x4003C030) // (DAC0) Interrupt Status Register
#define AT91C_DAC0_VER  ((AT91_REG *) 	0x4003C0FC) // (DAC0) DAC VERSION REGISTER
#define AT91C_DAC0_MR   ((AT91_REG *) 	0x4003C004) // (DAC0) Mode Register
// ========== Register definition for ACC0 peripheral ========== 
#define AT91C_ACC0_IPNAME1 ((AT91_REG *) 	0x400400F0) // (ACC0) ACC IPNAME1 REGISTER 
#define AT91C_ACC0_MR   ((AT91_REG *) 	0x40040004) // (ACC0) Mode Register
#define AT91C_ACC0_FEATURES ((AT91_REG *) 	0x400400F8) // (ACC0) ACC FEATURES REGISTER 
#define AT91C_ACC0_IMR  ((AT91_REG *) 	0x4004002C) // (ACC0) Interrupt Mask Register
#define AT91C_ACC0_ACR  ((AT91_REG *) 	0x40040094) // (ACC0) Analog Control Register
#define AT91C_ACC0_ADDRSIZE ((AT91_REG *) 	0x400400EC) // (ACC0) ACC ADDRSIZE REGISTER 
#define AT91C_ACC0_IER  ((AT91_REG *) 	0x40040024) // (ACC0) Interrupt Enable Register
#define AT91C_ACC0_ISR  ((AT91_REG *) 	0x40040030) // (ACC0) Interrupt Status Register
#define AT91C_ACC0_IDR  ((AT91_REG *) 	0x40040028) // (ACC0) Interrupt Disable Register
#define AT91C_ACC0_MODE ((AT91_REG *) 	0x400400E4) // (ACC0) Write Protection Mode Register
#define AT91C_ACC0_VER  ((AT91_REG *) 	0x400400FC) // (ACC0) ACC VERSION REGISTER
#define AT91C_ACC0_CR   ((AT91_REG *) 	0x40040000) // (ACC0) Control Register
#define AT91C_ACC0_IPNAME2 ((AT91_REG *) 	0x400400F4) // (ACC0) ACC IPNAME2 REGISTER 
#define AT91C_ACC0_STATUS ((AT91_REG *) 	0x400400E8) // (ACC0) Write Protection Status
// ========== Register definition for HCBDMA peripheral ========== 
#define AT91C_HCBDMA_CBIMR ((AT91_REG *) 	0x4004401C) // (HCBDMA) CB DMA Interrupt mask Register
#define AT91C_HCBDMA_CBCRCCR ((AT91_REG *) 	0x40044034) // (HCBDMA) CB DMA CRC Control Resgister
#define AT91C_HCBDMA_CBSR ((AT91_REG *) 	0x40044010) // (HCBDMA) CB DMA Status Register
#define AT91C_HCBDMA_CBCRCISR ((AT91_REG *) 	0x4004404C) // (HCBDMA) CB DMA CRC Interrupt Status Resgister
#define AT91C_HCBDMA_CBCRCSR ((AT91_REG *) 	0x4004403C) // (HCBDMA) CB DMA CRC Status Resgister
#define AT91C_HCBDMA_CBIDR ((AT91_REG *) 	0x40044018) // (HCBDMA) CB DMA Interrupt Disable Register
#define AT91C_HCBDMA_CBCRCIDR ((AT91_REG *) 	0x40044044) // (HCBDMA) CB DMA CRC Interrupt Disable Resgister
#define AT91C_HCBDMA_CBDLIER ((AT91_REG *) 	0x40044024) // (HCBDMA) CB DMA Loaded Interrupt Enable Register
#define AT91C_HCBDMA_CBEN ((AT91_REG *) 	0x40044008) // (HCBDMA) CB DMA Enable Register
#define AT91C_HCBDMA_FEATURES ((AT91_REG *) 	0x400440F8) // (HCBDMA) HCBDMA FEATURES REGISTER 
#define AT91C_HCBDMA_CBDSCR ((AT91_REG *) 	0x40044000) // (HCBDMA) CB DMA Descriptor Base Register
#define AT91C_HCBDMA_ADDRSIZE ((AT91_REG *) 	0x400440EC) // (HCBDMA) HCBDMA ADDRSIZE REGISTER 
#define AT91C_HCBDMA_CBDLISR ((AT91_REG *) 	0x40044030) // (HCBDMA) CB DMA Loaded Interrupt Status Register
#define AT91C_HCBDMA_CBDLIDR ((AT91_REG *) 	0x40044028) // (HCBDMA) CB DMA Loaded Interrupt Disable Register
#define AT91C_HCBDMA_CBCRCIMR ((AT91_REG *) 	0x40044048) // (HCBDMA) CB DMA CRC Interrupt Mask Resgister
#define AT91C_HCBDMA_VER ((AT91_REG *) 	0x400440FC) // (HCBDMA) HCBDMA VERSION REGISTER
#define AT91C_HCBDMA_CBCRCIER ((AT91_REG *) 	0x40044040) // (HCBDMA) CB DMA CRC Interrupt Enable Resgister
#define AT91C_HCBDMA_IPNAME2 ((AT91_REG *) 	0x400440F4) // (HCBDMA) HCBDMA IPNAME2 REGISTER 
#define AT91C_HCBDMA_CBIER ((AT91_REG *) 	0x40044014) // (HCBDMA) CB DMA Interrupt Enable Register
#define AT91C_HCBDMA_CBISR ((AT91_REG *) 	0x40044020) // (HCBDMA) CB DMA Interrupt Status Register
#define AT91C_HCBDMA_IPNAME1 ((AT91_REG *) 	0x400440F0) // (HCBDMA) HCBDMA IPNAME1 REGISTER 
#define AT91C_HCBDMA_CBDIS ((AT91_REG *) 	0x4004400C) // (HCBDMA) CB DMA Disable Register
#define AT91C_HCBDMA_CBNXTEN ((AT91_REG *) 	0x40044004) // (HCBDMA) CB DMA Next Descriptor Enable Register
#define AT91C_HCBDMA_CBDLIMR ((AT91_REG *) 	0x4004402C) // (HCBDMA) CB DMA Loaded Interrupt mask Register
#define AT91C_HCBDMA_CBCRCMR ((AT91_REG *) 	0x40044038) // (HCBDMA) CB DMA CRC Mode Resgister
// ========== Register definition for TC0 peripheral ========== 
#define AT91C_TC0_SR    ((AT91_REG *) 	0x40010020) // (TC0) Status Register
#define AT91C_TC0_CCR   ((AT91_REG *) 	0x40010000) // (TC0) Channel Control Register
#define AT91C_TC0_CMR   ((AT91_REG *) 	0x40010004) // (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC0_IER   ((AT91_REG *) 	0x40010024) // (TC0) Interrupt Enable Register
#define AT91C_TC0_CV    ((AT91_REG *) 	0x40010010) // (TC0) Counter Value
#define AT91C_TC0_RB    ((AT91_REG *) 	0x40010018) // (TC0) Register B
#define AT91C_TC0_IDR   ((AT91_REG *) 	0x40010028) // (TC0) Interrupt Disable Register
#define AT91C_TC0_RA    ((AT91_REG *) 	0x40010014) // (TC0) Register A
#define AT91C_TC0_RC    ((AT91_REG *) 	0x4001001C) // (TC0) Register C
#define AT91C_TC0_IMR   ((AT91_REG *) 	0x4001002C) // (TC0) Interrupt Mask Register
// ========== Register definition for TC1 peripheral ========== 
#define AT91C_TC1_SR    ((AT91_REG *) 	0x40010060) // (TC1) Status Register
#define AT91C_TC1_CV    ((AT91_REG *) 	0x40010050) // (TC1) Counter Value
#define AT91C_TC1_RA    ((AT91_REG *) 	0x40010054) // (TC1) Register A
#define AT91C_TC1_IER   ((AT91_REG *) 	0x40010064) // (TC1) Interrupt Enable Register
#define AT91C_TC1_RB    ((AT91_REG *) 	0x40010058) // (TC1) Register B
#define AT91C_TC1_RC    ((AT91_REG *) 	0x4001005C) // (TC1) Register C
#define AT91C_TC1_CCR   ((AT91_REG *) 	0x40010040) // (TC1) Channel Control Register
#define AT91C_TC1_IMR   ((AT91_REG *) 	0x4001006C) // (TC1) Interrupt Mask Register
#define AT91C_TC1_IDR   ((AT91_REG *) 	0x40010068) // (TC1) Interrupt Disable Register
#define AT91C_TC1_CMR   ((AT91_REG *) 	0x40010044) // (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
// ========== Register definition for TC2 peripheral ========== 
#define AT91C_TC2_SR    ((AT91_REG *) 	0x400100A0) // (TC2) Status Register
#define AT91C_TC2_IER   ((AT91_REG *) 	0x400100A4) // (TC2) Interrupt Enable Register
#define AT91C_TC2_CCR   ((AT91_REG *) 	0x40010080) // (TC2) Channel Control Register
#define AT91C_TC2_IDR   ((AT91_REG *) 	0x400100A8) // (TC2) Interrupt Disable Register
#define AT91C_TC2_RA    ((AT91_REG *) 	0x40010094) // (TC2) Register A
#define AT91C_TC2_RB    ((AT91_REG *) 	0x40010098) // (TC2) Register B
#define AT91C_TC2_IMR   ((AT91_REG *) 	0x400100AC) // (TC2) Interrupt Mask Register
#define AT91C_TC2_CV    ((AT91_REG *) 	0x40010090) // (TC2) Counter Value
#define AT91C_TC2_RC    ((AT91_REG *) 	0x4001009C) // (TC2) Register C
#define AT91C_TC2_CMR   ((AT91_REG *) 	0x40010084) // (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
// ========== Register definition for TC3 peripheral ========== 
#define AT91C_TC3_IDR   ((AT91_REG *) 	0x40014028) // (TC3) Interrupt Disable Register
#define AT91C_TC3_IER   ((AT91_REG *) 	0x40014024) // (TC3) Interrupt Enable Register
#define AT91C_TC3_SR    ((AT91_REG *) 	0x40014020) // (TC3) Status Register
#define AT91C_TC3_CV    ((AT91_REG *) 	0x40014010) // (TC3) Counter Value
#define AT91C_TC3_CMR   ((AT91_REG *) 	0x40014004) // (TC3) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC3_RC    ((AT91_REG *) 	0x4001401C) // (TC3) Register C
#define AT91C_TC3_RA    ((AT91_REG *) 	0x40014014) // (TC3) Register A
#define AT91C_TC3_IMR   ((AT91_REG *) 	0x4001402C) // (TC3) Interrupt Mask Register
#define AT91C_TC3_RB    ((AT91_REG *) 	0x40014018) // (TC3) Register B
#define AT91C_TC3_CCR   ((AT91_REG *) 	0x40014000) // (TC3) Channel Control Register
// ========== Register definition for TC4 peripheral ========== 
#define AT91C_TC4_CV    ((AT91_REG *) 	0x40014050) // (TC4) Counter Value
#define AT91C_TC4_CMR   ((AT91_REG *) 	0x40014044) // (TC4) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC4_RA    ((AT91_REG *) 	0x40014054) // (TC4) Register A
#define AT91C_TC4_IMR   ((AT91_REG *) 	0x4001406C) // (TC4) Interrupt Mask Register
#define AT91C_TC4_RC    ((AT91_REG *) 	0x4001405C) // (TC4) Register C
#define AT91C_TC4_SR    ((AT91_REG *) 	0x40014060) // (TC4) Status Register
#define AT91C_TC4_RB    ((AT91_REG *) 	0x40014058) // (TC4) Register B
#define AT91C_TC4_IDR   ((AT91_REG *) 	0x40014068) // (TC4) Interrupt Disable Register
#define AT91C_TC4_IER   ((AT91_REG *) 	0x40014064) // (TC4) Interrupt Enable Register
#define AT91C_TC4_CCR   ((AT91_REG *) 	0x40014040) // (TC4) Channel Control Register
// ========== Register definition for TC5 peripheral ========== 
#define AT91C_TC5_CV    ((AT91_REG *) 	0x40014090) // (TC5) Counter Value
#define AT91C_TC5_IER   ((AT91_REG *) 	0x400140A4) // (TC5) Interrupt Enable Register
#define AT91C_TC5_CMR   ((AT91_REG *) 	0x40014084) // (TC5) Channel Mode Register (Capture Mode / Waveform Mode)
#define AT91C_TC5_IMR   ((AT91_REG *) 	0x400140AC) // (TC5) Interrupt Mask Register
#define AT91C_TC5_RA    ((AT91_REG *) 	0x40014094) // (TC5) Register A
#define AT91C_TC5_RB    ((AT91_REG *) 	0x40014098) // (TC5) Register B
#define AT91C_TC5_RC    ((AT91_REG *) 	0x4001409C) // (TC5) Register C
#define AT91C_TC5_SR    ((AT91_REG *) 	0x400140A0) // (TC5) Status Register
#define AT91C_TC5_CCR   ((AT91_REG *) 	0x40014080) // (TC5) Channel Control Register
#define AT91C_TC5_IDR   ((AT91_REG *) 	0x400140A8) // (TC5) Interrupt Disable Register
// ========== Register definition for TCB0 peripheral ========== 
#define AT91C_TCB0_BCR  ((AT91_REG *) 	0x400100C0) // (TCB0) TC Block Control Register
#define AT91C_TCB0_VER  ((AT91_REG *) 	0x400100FC) // (TCB0)  Version Register
#define AT91C_TCB0_ADDRSIZE ((AT91_REG *) 	0x400100EC) // (TCB0) TC ADDRSIZE REGISTER 
#define AT91C_TCB0_FEATURES ((AT91_REG *) 	0x400100F8) // (TCB0) TC FEATURES REGISTER 
#define AT91C_TCB0_IPNAME2 ((AT91_REG *) 	0x400100F4) // (TCB0) TC IPNAME2 REGISTER 
#define AT91C_TCB0_BMR  ((AT91_REG *) 	0x400100C4) // (TCB0) TC Block Mode Register
#define AT91C_TCB0_IPNAME1 ((AT91_REG *) 	0x400100F0) // (TCB0) TC IPNAME1 REGISTER 
// ========== Register definition for TCB1 peripheral ========== 
#define AT91C_TCB1_IPNAME1 ((AT91_REG *) 	0x40010130) // (TCB1) TC IPNAME1 REGISTER 
#define AT91C_TCB1_IPNAME2 ((AT91_REG *) 	0x40010134) // (TCB1) TC IPNAME2 REGISTER 
#define AT91C_TCB1_BCR  ((AT91_REG *) 	0x40010100) // (TCB1) TC Block Control Register
#define AT91C_TCB1_VER  ((AT91_REG *) 	0x4001013C) // (TCB1)  Version Register
#define AT91C_TCB1_FEATURES ((AT91_REG *) 	0x40010138) // (TCB1) TC FEATURES REGISTER 
#define AT91C_TCB1_ADDRSIZE ((AT91_REG *) 	0x4001012C) // (TCB1) TC ADDRSIZE REGISTER 
#define AT91C_TCB1_BMR  ((AT91_REG *) 	0x40010104) // (TCB1) TC Block Mode Register
// ========== Register definition for TCB2 peripheral ========== 
#define AT91C_TCB2_VER  ((AT91_REG *) 	0x4001017C) // (TCB2)  Version Register
#define AT91C_TCB2_ADDRSIZE ((AT91_REG *) 	0x4001016C) // (TCB2) TC ADDRSIZE REGISTER 
#define AT91C_TCB2_FEATURES ((AT91_REG *) 	0x40010178) // (TCB2) TC FEATURES REGISTER 
#define AT91C_TCB2_BCR  ((AT91_REG *) 	0x40010140) // (TCB2) TC Block Control Register
#define AT91C_TCB2_IPNAME2 ((AT91_REG *) 	0x40010174) // (TCB2) TC IPNAME2 REGISTER 
#define AT91C_TCB2_BMR  ((AT91_REG *) 	0x40010144) // (TCB2) TC Block Mode Register
#define AT91C_TCB2_IPNAME1 ((AT91_REG *) 	0x40010170) // (TCB2) TC IPNAME1 REGISTER 
// ========== Register definition for TCB3 peripheral ========== 
#define AT91C_TCB3_IPNAME2 ((AT91_REG *) 	0x400140F4) // (TCB3) TC IPNAME2 REGISTER 
#define AT91C_TCB3_BMR  ((AT91_REG *) 	0x400140C4) // (TCB3) TC Block Mode Register
#define AT91C_TCB3_IPNAME1 ((AT91_REG *) 	0x400140F0) // (TCB3) TC IPNAME1 REGISTER 
#define AT91C_TCB3_FEATURES ((AT91_REG *) 	0x400140F8) // (TCB3) TC FEATURES REGISTER 
#define AT91C_TCB3_ADDRSIZE ((AT91_REG *) 	0x400140EC) // (TCB3) TC ADDRSIZE REGISTER 
#define AT91C_TCB3_VER  ((AT91_REG *) 	0x400140FC) // (TCB3)  Version Register
#define AT91C_TCB3_BCR  ((AT91_REG *) 	0x400140C0) // (TCB3) TC Block Control Register
// ========== Register definition for TCB4 peripheral ========== 
#define AT91C_TCB4_BMR  ((AT91_REG *) 	0x40014104) // (TCB4) TC Block Mode Register
#define AT91C_TCB4_BCR  ((AT91_REG *) 	0x40014100) // (TCB4) TC Block Control Register
#define AT91C_TCB4_IPNAME2 ((AT91_REG *) 	0x40014134) // (TCB4) TC IPNAME2 REGISTER 
#define AT91C_TCB4_FEATURES ((AT91_REG *) 	0x40014138) // (TCB4) TC FEATURES REGISTER 
#define AT91C_TCB4_IPNAME1 ((AT91_REG *) 	0x40014130) // (TCB4) TC IPNAME1 REGISTER 
#define AT91C_TCB4_VER  ((AT91_REG *) 	0x4001413C) // (TCB4)  Version Register
#define AT91C_TCB4_ADDRSIZE ((AT91_REG *) 	0x4001412C) // (TCB4) TC ADDRSIZE REGISTER 
// ========== Register definition for TCB5 peripheral ========== 
#define AT91C_TCB5_VER  ((AT91_REG *) 	0x4001417C) // (TCB5)  Version Register
#define AT91C_TCB5_ADDRSIZE ((AT91_REG *) 	0x4001416C) // (TCB5) TC ADDRSIZE REGISTER 
#define AT91C_TCB5_BMR  ((AT91_REG *) 	0x40014144) // (TCB5) TC Block Mode Register
#define AT91C_TCB5_FEATURES ((AT91_REG *) 	0x40014178) // (TCB5) TC FEATURES REGISTER 
#define AT91C_TCB5_IPNAME2 ((AT91_REG *) 	0x40014174) // (TCB5) TC IPNAME2 REGISTER 
#define AT91C_TCB5_IPNAME1 ((AT91_REG *) 	0x40014170) // (TCB5) TC IPNAME1 REGISTER 
#define AT91C_TCB5_BCR  ((AT91_REG *) 	0x40014140) // (TCB5) TC Block Control Register
// ========== Register definition for EFC0 peripheral ========== 
#define AT91C_EFC0_FMR  ((AT91_REG *) 	0x400E0A00) // (EFC0) EFC Flash Mode Register
#define AT91C_EFC0_FVR  ((AT91_REG *) 	0x400E0A14) // (EFC0) EFC Flash Version Register
#define AT91C_EFC0_FSR  ((AT91_REG *) 	0x400E0A08) // (EFC0) EFC Flash Status Register
#define AT91C_EFC0_FCR  ((AT91_REG *) 	0x400E0A04) // (EFC0) EFC Flash Command Register
#define AT91C_EFC0_FRR  ((AT91_REG *) 	0x400E0A0C) // (EFC0) EFC Flash Result Register
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
#define AT91C_TWI0_PTSR ((AT91_REG *) 	0x40018124) // (PDC_TWI0) PDC Transfer Status Register
#define AT91C_TWI0_TPR  ((AT91_REG *) 	0x40018108) // (PDC_TWI0) Transmit Pointer Register
#define AT91C_TWI0_RPR  ((AT91_REG *) 	0x40018100) // (PDC_TWI0) Receive Pointer Register
#define AT91C_TWI0_TNPR ((AT91_REG *) 	0x40018118) // (PDC_TWI0) Transmit Next Pointer Register
#define AT91C_TWI0_PTCR ((AT91_REG *) 	0x40018120) // (PDC_TWI0) PDC Transfer Control Register
#define AT91C_TWI0_RCR  ((AT91_REG *) 	0x40018104) // (PDC_TWI0) Receive Counter Register
#define AT91C_TWI0_RNCR ((AT91_REG *) 	0x40018114) // (PDC_TWI0) Receive Next Counter Register
#define AT91C_TWI0_RNPR ((AT91_REG *) 	0x40018110) // (PDC_TWI0) Receive Next Pointer Register
#define AT91C_TWI0_TNCR ((AT91_REG *) 	0x4001811C) // (PDC_TWI0) Transmit Next Counter Register
#define AT91C_TWI0_TCR  ((AT91_REG *) 	0x4001810C) // (PDC_TWI0) Transmit Counter Register
// ========== Register definition for PDC_TWI1 peripheral ========== 
#define AT91C_TWI1_TPR  ((AT91_REG *) 	0x4001C108) // (PDC_TWI1) Transmit Pointer Register
#define AT91C_TWI1_RNCR ((AT91_REG *) 	0x4001C114) // (PDC_TWI1) Receive Next Counter Register
#define AT91C_TWI1_TNCR ((AT91_REG *) 	0x4001C11C) // (PDC_TWI1) Transmit Next Counter Register
#define AT91C_TWI1_TCR  ((AT91_REG *) 	0x4001C10C) // (PDC_TWI1) Transmit Counter Register
#define AT91C_TWI1_TNPR ((AT91_REG *) 	0x4001C118) // (PDC_TWI1) Transmit Next Pointer Register
#define AT91C_TWI1_PTCR ((AT91_REG *) 	0x4001C120) // (PDC_TWI1) PDC Transfer Control Register
#define AT91C_TWI1_RNPR ((AT91_REG *) 	0x4001C110) // (PDC_TWI1) Receive Next Pointer Register
#define AT91C_TWI1_PTSR ((AT91_REG *) 	0x4001C124) // (PDC_TWI1) PDC Transfer Status Register
#define AT91C_TWI1_RPR  ((AT91_REG *) 	0x4001C100) // (PDC_TWI1) Receive Pointer Register
#define AT91C_TWI1_RCR  ((AT91_REG *) 	0x4001C104) // (PDC_TWI1) Receive Counter Register
// ========== Register definition for TWI0 peripheral ========== 
#define AT91C_TWI0_IMR  ((AT91_REG *) 	0x4001802C) // (TWI0) Interrupt Mask Register
#define AT91C_TWI0_IPNAME1 ((AT91_REG *) 	0x400180F0) // (TWI0) TWI IPNAME1 REGISTER 
#define AT91C_TWI0_CR   ((AT91_REG *) 	0x40018000) // (TWI0) Control Register
#define AT91C_TWI0_IPNAME2 ((AT91_REG *) 	0x400180F4) // (TWI0) TWI IPNAME2 REGISTER 
#define AT91C_TWI0_CWGR ((AT91_REG *) 	0x40018010) // (TWI0) Clock Waveform Generator Register
#define AT91C_TWI0_SMR  ((AT91_REG *) 	0x40018008) // (TWI0) Slave Mode Register
#define AT91C_TWI0_ADDRSIZE ((AT91_REG *) 	0x400180EC) // (TWI0) TWI ADDRSIZE REGISTER 
#define AT91C_TWI0_SR   ((AT91_REG *) 	0x40018020) // (TWI0) Status Register
#define AT91C_TWI0_IER  ((AT91_REG *) 	0x40018024) // (TWI0) Interrupt Enable Register
#define AT91C_TWI0_VER  ((AT91_REG *) 	0x400180FC) // (TWI0) Version Register
#define AT91C_TWI0_RHR  ((AT91_REG *) 	0x40018030) // (TWI0) Receive Holding Register
#define AT91C_TWI0_IADR ((AT91_REG *) 	0x4001800C) // (TWI0) Internal Address Register
#define AT91C_TWI0_IDR  ((AT91_REG *) 	0x40018028) // (TWI0) Interrupt Disable Register
#define AT91C_TWI0_THR  ((AT91_REG *) 	0x40018034) // (TWI0) Transmit Holding Register
#define AT91C_TWI0_FEATURES ((AT91_REG *) 	0x400180F8) // (TWI0) TWI FEATURES REGISTER 
#define AT91C_TWI0_MMR  ((AT91_REG *) 	0x40018004) // (TWI0) Master Mode Register
// ========== Register definition for TWI1 peripheral ========== 
#define AT91C_TWI1_CR   ((AT91_REG *) 	0x4001C000) // (TWI1) Control Register
#define AT91C_TWI1_VER  ((AT91_REG *) 	0x4001C0FC) // (TWI1) Version Register
#define AT91C_TWI1_IMR  ((AT91_REG *) 	0x4001C02C) // (TWI1) Interrupt Mask Register
#define AT91C_TWI1_IADR ((AT91_REG *) 	0x4001C00C) // (TWI1) Internal Address Register
#define AT91C_TWI1_THR  ((AT91_REG *) 	0x4001C034) // (TWI1) Transmit Holding Register
#define AT91C_TWI1_IPNAME2 ((AT91_REG *) 	0x4001C0F4) // (TWI1) TWI IPNAME2 REGISTER 
#define AT91C_TWI1_FEATURES ((AT91_REG *) 	0x4001C0F8) // (TWI1) TWI FEATURES REGISTER 
#define AT91C_TWI1_SMR  ((AT91_REG *) 	0x4001C008) // (TWI1) Slave Mode Register
#define AT91C_TWI1_IDR  ((AT91_REG *) 	0x4001C028) // (TWI1) Interrupt Disable Register
#define AT91C_TWI1_SR   ((AT91_REG *) 	0x4001C020) // (TWI1) Status Register
#define AT91C_TWI1_IPNAME1 ((AT91_REG *) 	0x4001C0F0) // (TWI1) TWI IPNAME1 REGISTER 
#define AT91C_TWI1_IER  ((AT91_REG *) 	0x4001C024) // (TWI1) Interrupt Enable Register
#define AT91C_TWI1_ADDRSIZE ((AT91_REG *) 	0x4001C0EC) // (TWI1) TWI ADDRSIZE REGISTER 
#define AT91C_TWI1_CWGR ((AT91_REG *) 	0x4001C010) // (TWI1) Clock Waveform Generator Register
#define AT91C_TWI1_MMR  ((AT91_REG *) 	0x4001C004) // (TWI1) Master Mode Register
#define AT91C_TWI1_RHR  ((AT91_REG *) 	0x4001C030) // (TWI1) Receive Holding Register
// ========== Register definition for PDC_US0 peripheral ========== 
#define AT91C_US0_RNCR  ((AT91_REG *) 	0x40024114) // (PDC_US0) Receive Next Counter Register
#define AT91C_US0_PTCR  ((AT91_REG *) 	0x40024120) // (PDC_US0) PDC Transfer Control Register
#define AT91C_US0_TCR   ((AT91_REG *) 	0x4002410C) // (PDC_US0) Transmit Counter Register
#define AT91C_US0_RPR   ((AT91_REG *) 	0x40024100) // (PDC_US0) Receive Pointer Register
#define AT91C_US0_RNPR  ((AT91_REG *) 	0x40024110) // (PDC_US0) Receive Next Pointer Register
#define AT91C_US0_TNCR  ((AT91_REG *) 	0x4002411C) // (PDC_US0) Transmit Next Counter Register
#define AT91C_US0_PTSR  ((AT91_REG *) 	0x40024124) // (PDC_US0) PDC Transfer Status Register
#define AT91C_US0_RCR   ((AT91_REG *) 	0x40024104) // (PDC_US0) Receive Counter Register
#define AT91C_US0_TNPR  ((AT91_REG *) 	0x40024118) // (PDC_US0) Transmit Next Pointer Register
#define AT91C_US0_TPR   ((AT91_REG *) 	0x40024108) // (PDC_US0) Transmit Pointer Register
// ========== Register definition for US0 peripheral ========== 
#define AT91C_US0_MAN   ((AT91_REG *) 	0x40024050) // (US0) Manchester Encoder Decoder Register
#define AT91C_US0_IER   ((AT91_REG *) 	0x40024008) // (US0) Interrupt Enable Register
#define AT91C_US0_NER   ((AT91_REG *) 	0x40024044) // (US0) Nb Errors Register
#define AT91C_US0_BRGR  ((AT91_REG *) 	0x40024020) // (US0) Baud Rate Generator Register
#define AT91C_US0_VER   ((AT91_REG *) 	0x400240FC) // (US0) VERSION Register
#define AT91C_US0_IF    ((AT91_REG *) 	0x4002404C) // (US0) IRDA_FILTER Register
#define AT91C_US0_RHR   ((AT91_REG *) 	0x40024018) // (US0) Receiver Holding Register
#define AT91C_US0_CSR   ((AT91_REG *) 	0x40024014) // (US0) Channel Status Register
#define AT91C_US0_FEATURES ((AT91_REG *) 	0x400240F8) // (US0) US FEATURES REGISTER 
#define AT91C_US0_ADDRSIZE ((AT91_REG *) 	0x400240EC) // (US0) US ADDRSIZE REGISTER 
#define AT91C_US0_IMR   ((AT91_REG *) 	0x40024010) // (US0) Interrupt Mask Register
#define AT91C_US0_THR   ((AT91_REG *) 	0x4002401C) // (US0) Transmitter Holding Register
#define AT91C_US0_FIDI  ((AT91_REG *) 	0x40024040) // (US0) FI_DI_Ratio Register
#define AT91C_US0_MR    ((AT91_REG *) 	0x40024004) // (US0) Mode Register
#define AT91C_US0_RTOR  ((AT91_REG *) 	0x40024024) // (US0) Receiver Time-out Register
#define AT91C_US0_IPNAME1 ((AT91_REG *) 	0x400240F0) // (US0) US IPNAME1 REGISTER 
#define AT91C_US0_IDR   ((AT91_REG *) 	0x4002400C) // (US0) Interrupt Disable Register
#define AT91C_US0_IPNAME2 ((AT91_REG *) 	0x400240F4) // (US0) US IPNAME2 REGISTER 
#define AT91C_US0_CR    ((AT91_REG *) 	0x40024000) // (US0) Control Register
#define AT91C_US0_TTGR  ((AT91_REG *) 	0x40024028) // (US0) Transmitter Time-guard Register
// ========== Register definition for PDC_US1 peripheral ========== 
#define AT91C_US1_TNPR  ((AT91_REG *) 	0x40028118) // (PDC_US1) Transmit Next Pointer Register
#define AT91C_US1_RPR   ((AT91_REG *) 	0x40028100) // (PDC_US1) Receive Pointer Register
#define AT91C_US1_TCR   ((AT91_REG *) 	0x4002810C) // (PDC_US1) Transmit Counter Register
#define AT91C_US1_RCR   ((AT91_REG *) 	0x40028104) // (PDC_US1) Receive Counter Register
#define AT91C_US1_TPR   ((AT91_REG *) 	0x40028108) // (PDC_US1) Transmit Pointer Register
#define AT91C_US1_RNPR  ((AT91_REG *) 	0x40028110) // (PDC_US1) Receive Next Pointer Register
#define AT91C_US1_TNCR  ((AT91_REG *) 	0x4002811C) // (PDC_US1) Transmit Next Counter Register
#define AT91C_US1_PTCR  ((AT91_REG *) 	0x40028120) // (PDC_US1) PDC Transfer Control Register
#define AT91C_US1_RNCR  ((AT91_REG *) 	0x40028114) // (PDC_US1) Receive Next Counter Register
#define AT91C_US1_PTSR  ((AT91_REG *) 	0x40028124) // (PDC_US1) PDC Transfer Status Register
// ========== Register definition for US1 peripheral ========== 
#define AT91C_US1_ADDRSIZE ((AT91_REG *) 	0x400280EC) // (US1) US ADDRSIZE REGISTER 
#define AT91C_US1_IDR   ((AT91_REG *) 	0x4002800C) // (US1) Interrupt Disable Register
#define AT91C_US1_FEATURES ((AT91_REG *) 	0x400280F8) // (US1) US FEATURES REGISTER 
#define AT91C_US1_IPNAME2 ((AT91_REG *) 	0x400280F4) // (US1) US IPNAME2 REGISTER 
#define AT91C_US1_MAN   ((AT91_REG *) 	0x40028050) // (US1) Manchester Encoder Decoder Register
#define AT91C_US1_CR    ((AT91_REG *) 	0x40028000) // (US1) Control Register
#define AT91C_US1_TTGR  ((AT91_REG *) 	0x40028028) // (US1) Transmitter Time-guard Register
#define AT91C_US1_IF    ((AT91_REG *) 	0x4002804C) // (US1) IRDA_FILTER Register
#define AT91C_US1_FIDI  ((AT91_REG *) 	0x40028040) // (US1) FI_DI_Ratio Register
#define AT91C_US1_THR   ((AT91_REG *) 	0x4002801C) // (US1) Transmitter Holding Register
#define AT91C_US1_VER   ((AT91_REG *) 	0x400280FC) // (US1) VERSION Register
#define AT91C_US1_MR    ((AT91_REG *) 	0x40028004) // (US1) Mode Register
#define AT91C_US1_CSR   ((AT91_REG *) 	0x40028014) // (US1) Channel Status Register
#define AT91C_US1_IER   ((AT91_REG *) 	0x40028008) // (US1) Interrupt Enable Register
#define AT91C_US1_NER   ((AT91_REG *) 	0x40028044) // (US1) Nb Errors Register
#define AT91C_US1_RHR   ((AT91_REG *) 	0x40028018) // (US1) Receiver Holding Register
#define AT91C_US1_IPNAME1 ((AT91_REG *) 	0x400280F0) // (US1) US IPNAME1 REGISTER 
#define AT91C_US1_IMR   ((AT91_REG *) 	0x40028010) // (US1) Interrupt Mask Register
#define AT91C_US1_BRGR  ((AT91_REG *) 	0x40028020) // (US1) Baud Rate Generator Register
#define AT91C_US1_RTOR  ((AT91_REG *) 	0x40028024) // (US1) Receiver Time-out Register
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
#define AT91C_SSC0_FEATURES ((AT91_REG *) 	0x400040F8) // (SSC0) SSC FEATURES REGISTER 
#define AT91C_SSC0_ADDRSIZE ((AT91_REG *) 	0x400040F0) // (SSC0) SSC ADDRSIZE REGISTER 
#define AT91C_SSC0_CR   ((AT91_REG *) 	0x40004000) // (SSC0) Control Register
#define AT91C_SSC0_RHR  ((AT91_REG *) 	0x40004020) // (SSC0) Receive Holding Register
#define AT91C_SSC0_VER  ((AT91_REG *) 	0x400040FC) // (SSC0) Version Register
#define AT91C_SSC0_TSHR ((AT91_REG *) 	0x40004034) // (SSC0) Transmit Sync Holding Register
#define AT91C_SSC0_RFMR ((AT91_REG *) 	0x40004014) // (SSC0) Receive Frame Mode Register
#define AT91C_SSC0_IDR  ((AT91_REG *) 	0x40004048) // (SSC0) Interrupt Disable Register
#define AT91C_SSC0_TFMR ((AT91_REG *) 	0x4000401C) // (SSC0) Transmit Frame Mode Register
#define AT91C_SSC0_RSHR ((AT91_REG *) 	0x40004030) // (SSC0) Receive Sync Holding Register
#define AT91C_SSC0_TCMR ((AT91_REG *) 	0x40004018) // (SSC0) Transmit Clock Mode Register
#define AT91C_SSC0_RCMR ((AT91_REG *) 	0x40004010) // (SSC0) Receive Clock ModeRegister
#define AT91C_SSC0_SR   ((AT91_REG *) 	0x40004040) // (SSC0) Status Register
#define AT91C_SSC0_NAME ((AT91_REG *) 	0x400040F4) // (SSC0) SSC NAME REGISTER 
#define AT91C_SSC0_THR  ((AT91_REG *) 	0x40004024) // (SSC0) Transmit Holding Register
#define AT91C_SSC0_CMR  ((AT91_REG *) 	0x40004004) // (SSC0) Clock Mode Register
#define AT91C_SSC0_IER  ((AT91_REG *) 	0x40004044) // (SSC0) Interrupt Enable Register
#define AT91C_SSC0_IMR  ((AT91_REG *) 	0x4000404C) // (SSC0) Interrupt Mask Register
// ========== Register definition for PDC_PWMC peripheral ========== 
#define AT91C_PWMC_TNCR ((AT91_REG *) 	0x4002011C) // (PDC_PWMC) Transmit Next Counter Register
#define AT91C_PWMC_RCR  ((AT91_REG *) 	0x40020104) // (PDC_PWMC) Receive Counter Register
#define AT91C_PWMC_TCR  ((AT91_REG *) 	0x4002010C) // (PDC_PWMC) Transmit Counter Register
#define AT91C_PWMC_RNCR ((AT91_REG *) 	0x40020114) // (PDC_PWMC) Receive Next Counter Register
#define AT91C_PWMC_PTSR ((AT91_REG *) 	0x40020124) // (PDC_PWMC) PDC Transfer Status Register
#define AT91C_PWMC_RNPR ((AT91_REG *) 	0x40020110) // (PDC_PWMC) Receive Next Pointer Register
#define AT91C_PWMC_TNPR ((AT91_REG *) 	0x40020118) // (PDC_PWMC) Transmit Next Pointer Register
#define AT91C_PWMC_PTCR ((AT91_REG *) 	0x40020120) // (PDC_PWMC) PDC Transfer Control Register
#define AT91C_PWMC_RPR  ((AT91_REG *) 	0x40020100) // (PDC_PWMC) Receive Pointer Register
#define AT91C_PWMC_TPR  ((AT91_REG *) 	0x40020108) // (PDC_PWMC) Transmit Pointer Register
// ========== Register definition for PWMC_CH0 peripheral ========== 
#define AT91C_PWMC_CH0_CMR ((AT91_REG *) 	0x40020200) // (PWMC_CH0) Channel Mode Register
#define AT91C_PWMC_CH0_DTUPDR ((AT91_REG *) 	0x4002021C) // (PWMC_CH0) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH0_CPRDR ((AT91_REG *) 	0x4002020C) // (PWMC_CH0) Channel Period Register
#define AT91C_PWMC_CH0_CPRDUPDR ((AT91_REG *) 	0x40020210) // (PWMC_CH0) Channel Period Update Register
#define AT91C_PWMC_CH0_CDTYR ((AT91_REG *) 	0x40020204) // (PWMC_CH0) Channel Duty Cycle Register
#define AT91C_PWMC_CH0_DTR ((AT91_REG *) 	0x40020218) // (PWMC_CH0) Channel Dead Time Value Register
#define AT91C_PWMC_CH0_CDTYUPDR ((AT91_REG *) 	0x40020208) // (PWMC_CH0) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH0_CCNTR ((AT91_REG *) 	0x40020214) // (PWMC_CH0) Channel Counter Register
// ========== Register definition for PWMC_CH1 peripheral ========== 
#define AT91C_PWMC_CH1_DTUPDR ((AT91_REG *) 	0x4002023C) // (PWMC_CH1) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH1_DTR ((AT91_REG *) 	0x40020238) // (PWMC_CH1) Channel Dead Time Value Register
#define AT91C_PWMC_CH1_CDTYUPDR ((AT91_REG *) 	0x40020228) // (PWMC_CH1) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH1_CDTYR ((AT91_REG *) 	0x40020224) // (PWMC_CH1) Channel Duty Cycle Register
#define AT91C_PWMC_CH1_CCNTR ((AT91_REG *) 	0x40020234) // (PWMC_CH1) Channel Counter Register
#define AT91C_PWMC_CH1_CPRDR ((AT91_REG *) 	0x4002022C) // (PWMC_CH1) Channel Period Register
#define AT91C_PWMC_CH1_CMR ((AT91_REG *) 	0x40020220) // (PWMC_CH1) Channel Mode Register
#define AT91C_PWMC_CH1_CPRDUPDR ((AT91_REG *) 	0x40020230) // (PWMC_CH1) Channel Period Update Register
// ========== Register definition for PWMC_CH2 peripheral ========== 
#define AT91C_PWMC_CH2_CPRDUPDR ((AT91_REG *) 	0x40020250) // (PWMC_CH2) Channel Period Update Register
#define AT91C_PWMC_CH2_CDTYR ((AT91_REG *) 	0x40020244) // (PWMC_CH2) Channel Duty Cycle Register
#define AT91C_PWMC_CH2_CCNTR ((AT91_REG *) 	0x40020254) // (PWMC_CH2) Channel Counter Register
#define AT91C_PWMC_CH2_CMR ((AT91_REG *) 	0x40020240) // (PWMC_CH2) Channel Mode Register
#define AT91C_PWMC_CH2_CDTYUPDR ((AT91_REG *) 	0x40020248) // (PWMC_CH2) Channel Duty Cycle Update Register
#define AT91C_PWMC_CH2_DTUPDR ((AT91_REG *) 	0x4002025C) // (PWMC_CH2) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH2_DTR ((AT91_REG *) 	0x40020258) // (PWMC_CH2) Channel Dead Time Value Register
#define AT91C_PWMC_CH2_CPRDR ((AT91_REG *) 	0x4002024C) // (PWMC_CH2) Channel Period Register
// ========== Register definition for PWMC_CH3 peripheral ========== 
#define AT91C_PWMC_CH3_CPRDR ((AT91_REG *) 	0x4002026C) // (PWMC_CH3) Channel Period Register
#define AT91C_PWMC_CH3_DTUPDR ((AT91_REG *) 	0x4002027C) // (PWMC_CH3) Channel Dead Time Update Value Register
#define AT91C_PWMC_CH3_DTR ((AT91_REG *) 	0x40020278) // (PWMC_CH3) Channel Dead Time Value Register
#define AT91C_PWMC_CH3_CDTYR ((AT91_REG *) 	0x40020264) // (PWMC_CH3) Channel Duty Cycle Register
#define AT91C_PWMC_CH3_CMR ((AT91_REG *) 	0x40020260) // (PWMC_CH3) Channel Mode Register
#define AT91C_PWMC_CH3_CCNTR ((AT91_REG *) 	0x40020274) // (PWMC_CH3) Channel Counter Register
#define AT91C_PWMC_CH3_CPRDUPDR ((AT91_REG *) 	0x40020270) // (PWMC_CH3) Channel Period Update Register
#define AT91C_PWMC_CH3_CDTYUPDR ((AT91_REG *) 	0x40020268) // (PWMC_CH3) Channel Duty Cycle Update Register
// ========== Register definition for PWMC peripheral ========== 
#define AT91C_PWMC_CMP6M ((AT91_REG *) 	0x40020198) // (PWMC) PWM Comparison Mode 6 Register
#define AT91C_PWMC_ADDRSIZE ((AT91_REG *) 	0x400200EC) // (PWMC) PWMC ADDRSIZE REGISTER 
#define AT91C_PWMC_CMP5V ((AT91_REG *) 	0x40020180) // (PWMC) PWM Comparison Value 5 Register
#define AT91C_PWMC_FMR  ((AT91_REG *) 	0x4002005C) // (PWMC) PWM Fault Mode Register
#define AT91C_PWMC_IER2 ((AT91_REG *) 	0x40020034) // (PWMC) PWMC Interrupt Enable Register 2
#define AT91C_PWMC_EL5MR ((AT91_REG *) 	0x40020090) // (PWMC) PWM Event Line 5 Mode Register
#define AT91C_PWMC_CMP0VUPD ((AT91_REG *) 	0x40020134) // (PWMC) PWM Comparison Value 0 Update Register
#define AT91C_PWMC_FPER1 ((AT91_REG *) 	0x4002006C) // (PWMC) PWM Fault Protection Enable Register 1
#define AT91C_PWMC_SCUPUPD ((AT91_REG *) 	0x40020030) // (PWMC) PWM Update Period Update Register
#define AT91C_PWMC_DIS  ((AT91_REG *) 	0x40020008) // (PWMC) PWMC Disable Register
#define AT91C_PWMC_CMP1M ((AT91_REG *) 	0x40020148) // (PWMC) PWM Comparison Mode 1 Register
#define AT91C_PWMC_CMP2V ((AT91_REG *) 	0x40020150) // (PWMC) PWM Comparison Value 2 Register
#define AT91C_PWMC_WPCR ((AT91_REG *) 	0x400200E4) // (PWMC) PWM Write Protection Enable Register
#define AT91C_PWMC_CMP5MUPD ((AT91_REG *) 	0x4002018C) // (PWMC) PWM Comparison Mode 5 Update Register
#define AT91C_PWMC_FPV  ((AT91_REG *) 	0x40020068) // (PWMC) PWM Fault Protection Value Register
#define AT91C_PWMC_UPCR ((AT91_REG *) 	0x40020028) // (PWMC) PWM Update Control Register
#define AT91C_PWMC_CMP4MUPD ((AT91_REG *) 	0x4002017C) // (PWMC) PWM Comparison Mode 4 Update Register
#define AT91C_PWMC_EL6MR ((AT91_REG *) 	0x40020094) // (PWMC) PWM Event Line 6 Mode Register
#define AT91C_PWMC_OS   ((AT91_REG *) 	0x40020048) // (PWMC) PWM Output Selection Register
#define AT91C_PWMC_OSSUPD ((AT91_REG *) 	0x40020054) // (PWMC) PWM Output Selection Set Update Register
#define AT91C_PWMC_FSR  ((AT91_REG *) 	0x40020060) // (PWMC) PWM Fault Mode Status Register
#define AT91C_PWMC_CMP2M ((AT91_REG *) 	0x40020158) // (PWMC) PWM Comparison Mode 2 Register
#define AT91C_PWMC_EL2MR ((AT91_REG *) 	0x40020084) // (PWMC) PWM Event Line 2 Mode Register
#define AT91C_PWMC_FPER3 ((AT91_REG *) 	0x40020074) // (PWMC) PWM Fault Protection Enable Register 3
#define AT91C_PWMC_CMP4M ((AT91_REG *) 	0x40020178) // (PWMC) PWM Comparison Mode 4 Register
#define AT91C_PWMC_ISR2 ((AT91_REG *) 	0x40020040) // (PWMC) PWMC Interrupt Status Register 2
#define AT91C_PWMC_CMP6VUPD ((AT91_REG *) 	0x40020194) // (PWMC) PWM Comparison Value 6 Update Register
#define AT91C_PWMC_CMP5VUPD ((AT91_REG *) 	0x40020184) // (PWMC) PWM Comparison Value 5 Update Register
#define AT91C_PWMC_EL7MR ((AT91_REG *) 	0x40020098) // (PWMC) PWM Event Line 7 Mode Register
#define AT91C_PWMC_OSC  ((AT91_REG *) 	0x40020050) // (PWMC) PWM Output Selection Clear Register
#define AT91C_PWMC_CMP3MUPD ((AT91_REG *) 	0x4002016C) // (PWMC) PWM Comparison Mode 3 Update Register
#define AT91C_PWMC_CMP2MUPD ((AT91_REG *) 	0x4002015C) // (PWMC) PWM Comparison Mode 2 Update Register
#define AT91C_PWMC_CMP0M ((AT91_REG *) 	0x40020138) // (PWMC) PWM Comparison Mode 0 Register
#define AT91C_PWMC_EL1MR ((AT91_REG *) 	0x40020080) // (PWMC) PWM Event Line 1 Mode Register
#define AT91C_PWMC_CMP0MUPD ((AT91_REG *) 	0x4002013C) // (PWMC) PWM Comparison Mode 0 Update Register
#define AT91C_PWMC_WPSR ((AT91_REG *) 	0x400200E8) // (PWMC) PWM Write Protection Status Register
#define AT91C_PWMC_CMP1MUPD ((AT91_REG *) 	0x4002014C) // (PWMC) PWM Comparison Mode 1 Update Register
#define AT91C_PWMC_IMR2 ((AT91_REG *) 	0x4002003C) // (PWMC) PWMC Interrupt Mask Register 2
#define AT91C_PWMC_CMP3V ((AT91_REG *) 	0x40020160) // (PWMC) PWM Comparison Value 3 Register
#define AT91C_PWMC_CMP3VUPD ((AT91_REG *) 	0x40020164) // (PWMC) PWM Comparison Value 3 Update Register
#define AT91C_PWMC_CMP3M ((AT91_REG *) 	0x40020168) // (PWMC) PWM Comparison Mode 3 Register
#define AT91C_PWMC_FPER4 ((AT91_REG *) 	0x40020078) // (PWMC) PWM Fault Protection Enable Register 4
#define AT91C_PWMC_OSCUPD ((AT91_REG *) 	0x40020058) // (PWMC) PWM Output Selection Clear Update Register
#define AT91C_PWMC_CMP0V ((AT91_REG *) 	0x40020130) // (PWMC) PWM Comparison Value 0 Register
#define AT91C_PWMC_OOV  ((AT91_REG *) 	0x40020044) // (PWMC) PWM Output Override Value Register
#define AT91C_PWMC_ENA  ((AT91_REG *) 	0x40020004) // (PWMC) PWMC Enable Register
#define AT91C_PWMC_CMP6MUPD ((AT91_REG *) 	0x4002019C) // (PWMC) PWM Comparison Mode 6 Update Register
#define AT91C_PWMC_SYNC ((AT91_REG *) 	0x40020020) // (PWMC) PWM Synchronized Channels Register
#define AT91C_PWMC_IPNAME1 ((AT91_REG *) 	0x400200F0) // (PWMC) PWMC IPNAME1 REGISTER 
#define AT91C_PWMC_IDR2 ((AT91_REG *) 	0x40020038) // (PWMC) PWMC Interrupt Disable Register 2
#define AT91C_PWMC_SR   ((AT91_REG *) 	0x4002000C) // (PWMC) PWMC Status Register
#define AT91C_PWMC_FPER2 ((AT91_REG *) 	0x40020070) // (PWMC) PWM Fault Protection Enable Register 2
#define AT91C_PWMC_EL3MR ((AT91_REG *) 	0x40020088) // (PWMC) PWM Event Line 3 Mode Register
#define AT91C_PWMC_IMR1 ((AT91_REG *) 	0x40020018) // (PWMC) PWMC Interrupt Mask Register 1
#define AT91C_PWMC_EL0MR ((AT91_REG *) 	0x4002007C) // (PWMC) PWM Event Line 0 Mode Register
#define AT91C_PWMC_STEP ((AT91_REG *) 	0x400200B0) // (PWMC) PWM Stepper Config Register
#define AT91C_PWMC_FCR  ((AT91_REG *) 	0x40020064) // (PWMC) PWM Fault Mode Clear Register
#define AT91C_PWMC_CMP7MUPD ((AT91_REG *) 	0x400201AC) // (PWMC) PWM Comparison Mode 7 Update Register
#define AT91C_PWMC_ISR1 ((AT91_REG *) 	0x4002001C) // (PWMC) PWMC Interrupt Status Register 1
#define AT91C_PWMC_CMP4VUPD ((AT91_REG *) 	0x40020174) // (PWMC) PWM Comparison Value 4 Update Register
#define AT91C_PWMC_VER  ((AT91_REG *) 	0x400200FC) // (PWMC) PWMC Version Register
#define AT91C_PWMC_CMP5M ((AT91_REG *) 	0x40020188) // (PWMC) PWM Comparison Mode 5 Register
#define AT91C_PWMC_IER1 ((AT91_REG *) 	0x40020010) // (PWMC) PWMC Interrupt Enable Register 1
#define AT91C_PWMC_MR   ((AT91_REG *) 	0x40020000) // (PWMC) PWMC Mode Register
#define AT91C_PWMC_OSS  ((AT91_REG *) 	0x4002004C) // (PWMC) PWM Output Selection Set Register
#define AT91C_PWMC_CMP7V ((AT91_REG *) 	0x400201A0) // (PWMC) PWM Comparison Value 7 Register
#define AT91C_PWMC_FEATURES ((AT91_REG *) 	0x400200F8) // (PWMC) PWMC FEATURES REGISTER 
#define AT91C_PWMC_CMP4V ((AT91_REG *) 	0x40020170) // (PWMC) PWM Comparison Value 4 Register
#define AT91C_PWMC_CMP7M ((AT91_REG *) 	0x400201A8) // (PWMC) PWM Comparison Mode 7 Register
#define AT91C_PWMC_EL4MR ((AT91_REG *) 	0x4002008C) // (PWMC) PWM Event Line 4 Mode Register
#define AT91C_PWMC_CMP2VUPD ((AT91_REG *) 	0x40020154) // (PWMC) PWM Comparison Value 2 Update Register
#define AT91C_PWMC_CMP6V ((AT91_REG *) 	0x40020190) // (PWMC) PWM Comparison Value 6 Register
#define AT91C_PWMC_CMP1V ((AT91_REG *) 	0x40020140) // (PWMC) PWM Comparison Value 1 Register
#define AT91C_PWMC_IDR1 ((AT91_REG *) 	0x40020014) // (PWMC) PWMC Interrupt Disable Register 1
#define AT91C_PWMC_SCUP ((AT91_REG *) 	0x4002002C) // (PWMC) PWM Update Period Register
#define AT91C_PWMC_CMP1VUPD ((AT91_REG *) 	0x40020144) // (PWMC) PWM Comparison Value 1 Update Register
#define AT91C_PWMC_CMP7VUPD ((AT91_REG *) 	0x400201A4) // (PWMC) PWM Comparison Value 7 Update Register
#define AT91C_PWMC_IPNAME2 ((AT91_REG *) 	0x400200F4) // (PWMC) PWMC IPNAME2 REGISTER 
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
// ========== Register definition for UDP peripheral ========== 
#define AT91C_UDP_RSTEP ((AT91_REG *) 	0x40034028) // (UDP) Reset Endpoint Register
#define AT91C_UDP_CSR   ((AT91_REG *) 	0x40034030) // (UDP) Endpoint Control and Status Register
#define AT91C_UDP_IMR   ((AT91_REG *) 	0x40034018) // (UDP) Interrupt Mask Register
#define AT91C_UDP_FDR   ((AT91_REG *) 	0x40034050) // (UDP) Endpoint FIFO Data Register
#define AT91C_UDP_ISR   ((AT91_REG *) 	0x4003401C) // (UDP) Interrupt Status Register
#define AT91C_UDP_IPNAME2 ((AT91_REG *) 	0x400340F4) // (UDP) UDP IPNAME2 REGISTER 
#define AT91C_UDP_ICR   ((AT91_REG *) 	0x40034020) // (UDP) Interrupt Clear Register
#define AT91C_UDP_VER   ((AT91_REG *) 	0x400340FC) // (UDP) UDP VERSION REGISTER
#define AT91C_UDP_IER   ((AT91_REG *) 	0x40034010) // (UDP) Interrupt Enable Register
#define AT91C_UDP_FEATURES ((AT91_REG *) 	0x400340F8) // (UDP) UDP FEATURES REGISTER 
#define AT91C_UDP_IPNAME1 ((AT91_REG *) 	0x400340F0) // (UDP) UDP IPNAME1 REGISTER 
#define AT91C_UDP_GLBSTATE ((AT91_REG *) 	0x40034004) // (UDP) Global State Register
#define AT91C_UDP_ADDRSIZE ((AT91_REG *) 	0x400340EC) // (UDP) UDP ADDRSIZE REGISTER 
#define AT91C_UDP_NUM   ((AT91_REG *) 	0x40034000) // (UDP) Frame Number Register
#define AT91C_UDP_IDR   ((AT91_REG *) 	0x40034014) // (UDP) Interrupt Disable Register
#define AT91C_UDP_TXVC  ((AT91_REG *) 	0x40034074) // (UDP) Transceiver Control Register
#define AT91C_UDP_FADDR ((AT91_REG *) 	0x40034008) // (UDP) Function Address Register

// *****************************************************************************
//               PIO DEFINITIONS FOR AT91SAM3S1
// *****************************************************************************
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) // Pin Controlled by PA0
#define AT91C_PA0_PWMH0    ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PA0_TIOA0    ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PA0_A17      ((unsigned int) AT91C_PIO_PA0) //  
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) // Pin Controlled by PA1
#define AT91C_PA1_PWMH1    ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PA1_TIOB0    ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PA1_A18      ((unsigned int) AT91C_PIO_PA1) //  
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) // Pin Controlled by PA10
#define AT91C_PA10_UTXD0    ((unsigned int) AT91C_PIO_PA10) //  
#define AT91C_PA10_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PA10) //  
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) // Pin Controlled by PA11
#define AT91C_PA11_SPI0_NPCS0 ((unsigned int) AT91C_PIO_PA11) //  
#define AT91C_PA11_PWMH0    ((unsigned int) AT91C_PIO_PA11) //  
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) // Pin Controlled by PA12
#define AT91C_PA12_SPI0_MISO ((unsigned int) AT91C_PIO_PA12) //  
#define AT91C_PA12_PWMH1    ((unsigned int) AT91C_PIO_PA12) //  
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) // Pin Controlled by PA13
#define AT91C_PA13_SPI0_MOSI ((unsigned int) AT91C_PIO_PA13) //  
#define AT91C_PA13_PWMH2    ((unsigned int) AT91C_PIO_PA13) //  
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) // Pin Controlled by PA14
#define AT91C_PA14_SPI0_SPCK ((unsigned int) AT91C_PIO_PA14) //  
#define AT91C_PA14_PWMH3    ((unsigned int) AT91C_PIO_PA14) //  
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) // Pin Controlled by PA15
#define AT91C_PA15_TF       ((unsigned int) AT91C_PIO_PA15) //  
#define AT91C_PA15_TIOA1    ((unsigned int) AT91C_PIO_PA15) //  
#define AT91C_PA15_PWML3    ((unsigned int) AT91C_PIO_PA15) //  
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) // Pin Controlled by PA16
#define AT91C_PA16_TK       ((unsigned int) AT91C_PIO_PA16) //  
#define AT91C_PA16_TIOB1    ((unsigned int) AT91C_PIO_PA16) //  
#define AT91C_PA16_PWML2    ((unsigned int) AT91C_PIO_PA16) //  
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) // Pin Controlled by PA17
#define AT91C_PA17_TD       ((unsigned int) AT91C_PIO_PA17) //  
#define AT91C_PA17_PCK1     ((unsigned int) AT91C_PIO_PA17) //  
#define AT91C_PA17_PWMH3    ((unsigned int) AT91C_PIO_PA17) //  
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) // Pin Controlled by PA18
#define AT91C_PA18_RD       ((unsigned int) AT91C_PIO_PA18) //  
#define AT91C_PA18_PCK2     ((unsigned int) AT91C_PIO_PA18) //  
#define AT91C_PA18_A14      ((unsigned int) AT91C_PIO_PA18) //  
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) // Pin Controlled by PA19
#define AT91C_PA19_RK       ((unsigned int) AT91C_PIO_PA19) //  
#define AT91C_PA19_PWML0    ((unsigned int) AT91C_PIO_PA19) //  
#define AT91C_PA19_A15      ((unsigned int) AT91C_PIO_PA19) //  
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) // Pin Controlled by PA2
#define AT91C_PA2_PWMH2    ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PA2_SCK0     ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PA2_DATRG    ((unsigned int) AT91C_PIO_PA2) //  
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) // Pin Controlled by PA20
#define AT91C_PA20_RF       ((unsigned int) AT91C_PIO_PA20) //  
#define AT91C_PA20_PWML1    ((unsigned int) AT91C_PIO_PA20) //  
#define AT91C_PA20_A16      ((unsigned int) AT91C_PIO_PA20) //  
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) // Pin Controlled by PA21
#define AT91C_PA21_RXD1     ((unsigned int) AT91C_PIO_PA21) //  
#define AT91C_PA21_PCK1     ((unsigned int) AT91C_PIO_PA21) //  
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) // Pin Controlled by PA22
#define AT91C_PA22_TXD1     ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PA22_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PA22_NCS2     ((unsigned int) AT91C_PIO_PA22) //  
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) // Pin Controlled by PA23
#define AT91C_PA23_SCK1     ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PA23_PWMH0    ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PA23_A19      ((unsigned int) AT91C_PIO_PA23) //  
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) // Pin Controlled by PA24
#define AT91C_PA24_RTS1     ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PA24_PWMH1    ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PA24_A20      ((unsigned int) AT91C_PIO_PA24) //  
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) // Pin Controlled by PA25
#define AT91C_PA25_CTS1     ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PA25_PWMH2    ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PA25_A23      ((unsigned int) AT91C_PIO_PA25) //  
#define AT91C_PIO_PA26       ((unsigned int) 1 << 26) // Pin Controlled by PA26
#define AT91C_PA26_DCD1     ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PA26_TIOA2    ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PA26_MCI0_DA2 ((unsigned int) AT91C_PIO_PA26) //  
#define AT91C_PIO_PA27       ((unsigned int) 1 << 27) // Pin Controlled by PA27
#define AT91C_PA27_DTR1     ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PA27_TIOB2    ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PA27_MCI0_DA3 ((unsigned int) AT91C_PIO_PA27) //  
#define AT91C_PIO_PA28       ((unsigned int) 1 << 28) // Pin Controlled by PA28
#define AT91C_PA28_DSR1     ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PA28_TCLK1    ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PA28_MCI0_CDA ((unsigned int) AT91C_PIO_PA28) //  
#define AT91C_PIO_PA29       ((unsigned int) 1 << 29) // Pin Controlled by PA29
#define AT91C_PA29_RI1      ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PA29_TCLK2    ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PA29_MCI0_CK  ((unsigned int) AT91C_PIO_PA29) //  
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) // Pin Controlled by PA3
#define AT91C_PA3_TWD0     ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PA3_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PA3) //  
#define AT91C_PIO_PA30       ((unsigned int) 1 << 30) // Pin Controlled by PA30
#define AT91C_PA30_PWML2    ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PA30_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PA30_MCI0_DA0 ((unsigned int) AT91C_PIO_PA30) //  
#define AT91C_PIO_PA31       ((unsigned int) 1 << 31) // Pin Controlled by PA31
#define AT91C_PA31_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PA31_PCK2     ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PA31_MCI0_DA1 ((unsigned int) AT91C_PIO_PA31) //  
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) // Pin Controlled by PA4
#define AT91C_PA4_TWCK0    ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PA4_TCLK0    ((unsigned int) AT91C_PIO_PA4) //  
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) // Pin Controlled by PA5
#define AT91C_PA5_RXD0     ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PA5_SPI0_NPCS3 ((unsigned int) AT91C_PIO_PA5) //  
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) // Pin Controlled by PA6
#define AT91C_PA6_TXD0     ((unsigned int) AT91C_PIO_PA6) //  
#define AT91C_PA6_PCK0     ((unsigned int) AT91C_PIO_PA6) //  
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) // Pin Controlled by PA7
#define AT91C_PA7_RTS0     ((unsigned int) AT91C_PIO_PA7) //  
#define AT91C_PA7_PWMH3    ((unsigned int) AT91C_PIO_PA7) //  
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) // Pin Controlled by PA8
#define AT91C_PA8_CTS0     ((unsigned int) AT91C_PIO_PA8) //  
#define AT91C_PA8_ADTRG    ((unsigned int) AT91C_PIO_PA8) //  
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) // Pin Controlled by PA9
#define AT91C_PA9_URXD0    ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PA9_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PA9_PWMFI0   ((unsigned int) AT91C_PIO_PA9) //  
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) // Pin Controlled by PB0
#define AT91C_PB0_PWMH0    ((unsigned int) AT91C_PIO_PB0) //  
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) // Pin Controlled by PB1
#define AT91C_PB1_PWMH1    ((unsigned int) AT91C_PIO_PB1) //  
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) // Pin Controlled by PB10
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) // Pin Controlled by PB11
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) // Pin Controlled by PB12
#define AT91C_PB12_PWML1    ((unsigned int) AT91C_PIO_PB12) //  
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) // Pin Controlled by PB13
#define AT91C_PB13_PWML2    ((unsigned int) AT91C_PIO_PB13) //  
#define AT91C_PB13_PCK0     ((unsigned int) AT91C_PIO_PB13) //  
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) // Pin Controlled by PB14
#define AT91C_PB14_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PB14) //  
#define AT91C_PB14_PWMH3    ((unsigned int) AT91C_PIO_PB14) //  
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) // Pin Controlled by PB2
#define AT91C_PB2_URXD1    ((unsigned int) AT91C_PIO_PB2) //  
#define AT91C_PB2_SPI0_NPCS2 ((unsigned int) AT91C_PIO_PB2) //  
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) // Pin Controlled by PB3
#define AT91C_PB3_UTXD1    ((unsigned int) AT91C_PIO_PB3) //  
#define AT91C_PB3_PCK2     ((unsigned int) AT91C_PIO_PB3) //  
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) // Pin Controlled by PB4
#define AT91C_PB4_TWD1     ((unsigned int) AT91C_PIO_PB4) //  
#define AT91C_PB4_PWMH2    ((unsigned int) AT91C_PIO_PB4) //  
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) // Pin Controlled by PB5
#define AT91C_PB5_TWCK1    ((unsigned int) AT91C_PIO_PB5) //  
#define AT91C_PB5_PWML0    ((unsigned int) AT91C_PIO_PB5) //  
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) // Pin Controlled by PB6
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) // Pin Controlled by PB7
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) // Pin Controlled by PB8
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) // Pin Controlled by PB9
#define AT91C_PIO_PC0        ((unsigned int) 1 <<  0) // Pin Controlled by PC0
#define AT91C_PC0_D0       ((unsigned int) AT91C_PIO_PC0) //  
#define AT91C_PC0_PWML0    ((unsigned int) AT91C_PIO_PC0) //  
#define AT91C_PIO_PC1        ((unsigned int) 1 <<  1) // Pin Controlled by PC1
#define AT91C_PC1_D1       ((unsigned int) AT91C_PIO_PC1) //  
#define AT91C_PC1_PWML1    ((unsigned int) AT91C_PIO_PC1) //  
#define AT91C_PIO_PC10       ((unsigned int) 1 << 10) // Pin Controlled by PC10
#define AT91C_PC10_NANDWE   ((unsigned int) AT91C_PIO_PC10) //  
#define AT91C_PIO_PC11       ((unsigned int) 1 << 11) // Pin Controlled by PC11
#define AT91C_PC11_NRD      ((unsigned int) AT91C_PIO_PC11) //  
#define AT91C_PIO_PC12       ((unsigned int) 1 << 12) // Pin Controlled by PC12
#define AT91C_PC12_NCS3     ((unsigned int) AT91C_PIO_PC12) //  
#define AT91C_PIO_PC13       ((unsigned int) 1 << 13) // Pin Controlled by PC13
#define AT91C_PC13_NWAIT    ((unsigned int) AT91C_PIO_PC13) //  
#define AT91C_PC13_PWML0    ((unsigned int) AT91C_PIO_PC13) //  
#define AT91C_PIO_PC14       ((unsigned int) 1 << 14) // Pin Controlled by PC14
#define AT91C_PC14_NCS0     ((unsigned int) AT91C_PIO_PC14) //  
#define AT91C_PIO_PC15       ((unsigned int) 1 << 15) // Pin Controlled by PC15
#define AT91C_PC15_NCS1     ((unsigned int) AT91C_PIO_PC15) //  
#define AT91C_PC15_PWML1    ((unsigned int) AT91C_PIO_PC15) //  
#define AT91C_PIO_PC16       ((unsigned int) 1 << 16) // Pin Controlled by PC16
#define AT91C_PC16_A21_NANDALE ((unsigned int) AT91C_PIO_PC16) //  
#define AT91C_PIO_PC17       ((unsigned int) 1 << 17) // Pin Controlled by PC17
#define AT91C_PC17_A22_NANDCLE ((unsigned int) AT91C_PIO_PC17) //  
#define AT91C_PIO_PC18       ((unsigned int) 1 << 18) // Pin Controlled by PC18
#define AT91C_PC18_A0_NBS0  ((unsigned int) AT91C_PIO_PC18) //  
#define AT91C_PC18_PWMH0    ((unsigned int) AT91C_PIO_PC18) //  
#define AT91C_PIO_PC19       ((unsigned int) 1 << 19) // Pin Controlled by PC19
#define AT91C_PC19_A1       ((unsigned int) AT91C_PIO_PC19) //  
#define AT91C_PC19_PWMH1    ((unsigned int) AT91C_PIO_PC19) //  
#define AT91C_PIO_PC2        ((unsigned int) 1 <<  2) // Pin Controlled by PC2
#define AT91C_PC2_D2       ((unsigned int) AT91C_PIO_PC2) //  
#define AT91C_PC2_PWML2    ((unsigned int) AT91C_PIO_PC2) //  
#define AT91C_PIO_PC20       ((unsigned int) 1 << 20) // Pin Controlled by PC20
#define AT91C_PC20_A2       ((unsigned int) AT91C_PIO_PC20) //  
#define AT91C_PC20_PWMH2    ((unsigned int) AT91C_PIO_PC20) //  
#define AT91C_PIO_PC21       ((unsigned int) 1 << 21) // Pin Controlled by PC21
#define AT91C_PC21_A3       ((unsigned int) AT91C_PIO_PC21) //  
#define AT91C_PC21_PWMH3    ((unsigned int) AT91C_PIO_PC21) //  
#define AT91C_PIO_PC22       ((unsigned int) 1 << 22) // Pin Controlled by PC22
#define AT91C_PC22_A4       ((unsigned int) AT91C_PIO_PC22) //  
#define AT91C_PC22_PWML3    ((unsigned int) AT91C_PIO_PC22) //  
#define AT91C_PIO_PC23       ((unsigned int) 1 << 23) // Pin Controlled by PC23
#define AT91C_PC23_A5       ((unsigned int) AT91C_PIO_PC23) //  
#define AT91C_PC23_TIOA3    ((unsigned int) AT91C_PIO_PC23) //  
#define AT91C_PIO_PC24       ((unsigned int) 1 << 24) // Pin Controlled by PC24
#define AT91C_PC24_A6       ((unsigned int) AT91C_PIO_PC24) //  
#define AT91C_PC24_TIOB3    ((unsigned int) AT91C_PIO_PC24) //  
#define AT91C_PIO_PC25       ((unsigned int) 1 << 25) // Pin Controlled by PC25
#define AT91C_PC25_A7       ((unsigned int) AT91C_PIO_PC25) //  
#define AT91C_PC25_TCLK3    ((unsigned int) AT91C_PIO_PC25) //  
#define AT91C_PIO_PC26       ((unsigned int) 1 << 26) // Pin Controlled by PC26
#define AT91C_PC26_A8       ((unsigned int) AT91C_PIO_PC26) //  
#define AT91C_PC26_TIOA4    ((unsigned int) AT91C_PIO_PC26) //  
#define AT91C_PIO_PC27       ((unsigned int) 1 << 27) // Pin Controlled by PC27
#define AT91C_PC27_A9       ((unsigned int) AT91C_PIO_PC27) //  
#define AT91C_PC27_TIOB4    ((unsigned int) AT91C_PIO_PC27) //  
#define AT91C_PIO_PC28       ((unsigned int) 1 << 28) // Pin Controlled by PC28
#define AT91C_PC28_A10      ((unsigned int) AT91C_PIO_PC28) //  
#define AT91C_PC28_TCLK4    ((unsigned int) AT91C_PIO_PC28) //  
#define AT91C_PIO_PC29       ((unsigned int) 1 << 29) // Pin Controlled by PC29
#define AT91C_PC29_A11      ((unsigned int) AT91C_PIO_PC29) //  
#define AT91C_PC29_TIOA5    ((unsigned int) AT91C_PIO_PC29) //  
#define AT91C_PIO_PC3        ((unsigned int) 1 <<  3) // Pin Controlled by PC3
#define AT91C_PC3_D3       ((unsigned int) AT91C_PIO_PC3) //  
#define AT91C_PC3_PWML3    ((unsigned int) AT91C_PIO_PC3) //  
#define AT91C_PIO_PC30       ((unsigned int) 1 << 30) // Pin Controlled by PC30
#define AT91C_PC30_A12      ((unsigned int) AT91C_PIO_PC30) //  
#define AT91C_PC30_TIOB5    ((unsigned int) AT91C_PIO_PC30) //  
#define AT91C_PIO_PC31       ((unsigned int) 1 << 31) // Pin Controlled by PC31
#define AT91C_PC31_A13      ((unsigned int) AT91C_PIO_PC31) //  
#define AT91C_PC31_TCLK5    ((unsigned int) AT91C_PIO_PC31) //  
#define AT91C_PIO_PC4        ((unsigned int) 1 <<  4) // Pin Controlled by PC4
#define AT91C_PC4_D4       ((unsigned int) AT91C_PIO_PC4) //  
#define AT91C_PC4_SPI0_NPCS1 ((unsigned int) AT91C_PIO_PC4) //  
#define AT91C_PIO_PC5        ((unsigned int) 1 <<  5) // Pin Controlled by PC5
#define AT91C_PC5_D5       ((unsigned int) AT91C_PIO_PC5) //  
#define AT91C_PIO_PC6        ((unsigned int) 1 <<  6) // Pin Controlled by PC6
#define AT91C_PC6_D6       ((unsigned int) AT91C_PIO_PC6) //  
#define AT91C_PIO_PC7        ((unsigned int) 1 <<  7) // Pin Controlled by PC7
#define AT91C_PC7_D7       ((unsigned int) AT91C_PIO_PC7) //  
#define AT91C_PIO_PC8        ((unsigned int) 1 <<  8) // Pin Controlled by PC8
#define AT91C_PC8_NWR0_NWE ((unsigned int) AT91C_PIO_PC8) //  
#define AT91C_PIO_PC9        ((unsigned int) 1 <<  9) // Pin Controlled by PC9
#define AT91C_PC9_NANDOE   ((unsigned int) AT91C_PIO_PC9) //  

// *****************************************************************************
//               PERIPHERAL ID DEFINITIONS FOR AT91SAM3S1
// *****************************************************************************
#define AT91C_ID_SUPC   ((unsigned int)  0) // SUPPLY CONTROLLER
#define AT91C_ID_RSTC   ((unsigned int)  1) // RESET CONTROLLER
#define AT91C_ID_RTC    ((unsigned int)  2) // REAL TIME CLOCK
#define AT91C_ID_RTT    ((unsigned int)  3) // REAL TIME TIMER
#define AT91C_ID_WDG    ((unsigned int)  4) // WATCHDOG TIMER
#define AT91C_ID_PMC    ((unsigned int)  5) // PMC
#define AT91C_ID_EFC0   ((unsigned int)  6) // EFC0
#define AT91C_ID_DBGU0  ((unsigned int)  8) // DBGU0
#define AT91C_ID_DBGU1  ((unsigned int)  9) // DBGU1
#define AT91C_ID_HSMC3  ((unsigned int) 10) // HSMC3
#define AT91C_ID_PIOA   ((unsigned int) 11) // Parallel IO Controller A
#define AT91C_ID_PIOB   ((unsigned int) 12) // Parallel IO Controller B
#define AT91C_ID_PIOC   ((unsigned int) 13) // Parallel IO Controller C
#define AT91C_ID_US0    ((unsigned int) 14) // USART 0
#define AT91C_ID_US1    ((unsigned int) 15) // USART 1
#define AT91C_ID_MCI0   ((unsigned int) 18) // Multimedia Card Interface
#define AT91C_ID_TWI0   ((unsigned int) 19) // TWI 0
#define AT91C_ID_TWI1   ((unsigned int) 20) // TWI 1
#define AT91C_ID_SPI0   ((unsigned int) 21) // Serial Peripheral Interface
#define AT91C_ID_SSC0   ((unsigned int) 22) // Serial Synchronous Controller 0
#define AT91C_ID_TC0    ((unsigned int) 23) // Timer Counter 0
#define AT91C_ID_TC1    ((unsigned int) 24) // Timer Counter 1
#define AT91C_ID_TC2    ((unsigned int) 25) // Timer Counter 2
#define AT91C_ID_TC3    ((unsigned int) 26) // Timer Counter 0
#define AT91C_ID_TC4    ((unsigned int) 27) // Timer Counter 1
#define AT91C_ID_TC5    ((unsigned int) 28) // Timer Counter 2
#define AT91C_ID_ADC0   ((unsigned int) 29) // ADC controller0
#define AT91C_ID_DAC0   ((unsigned int) 30) // Digital to Analog Converter
#define AT91C_ID_PWMC   ((unsigned int) 31) // Pulse Width Modulation Controller
#define AT91C_ID_HCBDMA ((unsigned int) 32) // Context Based Direct Memory Access Controller Interface
#define AT91C_ID_ACC0   ((unsigned int) 33) // Analog Comparator Controller
#define AT91C_ID_UDP    ((unsigned int) 34) // USB Device
#define AT91C_ALL_INT   ((unsigned int) 0xFFFFFFFF) // ALL VALID INTERRUPTS

// *****************************************************************************
//               BASE ADDRESS DEFINITIONS FOR AT91SAM3S1
// *****************************************************************************
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0x400E0000) // (SYS) Base Address
#define AT91C_BASE_SMC       ((AT91PS_SMC) 	0x400E0000) // (SMC) Base Address
#define AT91C_BASE_MATRIX    ((AT91PS_HMATRIX2) 	0x400E0200) // (MATRIX) Base Address
#define AT91C_BASE_CCFG      ((AT91PS_CCFG) 	0x400E0310) // (CCFG) Base Address
#define AT91C_BASE_NVIC      ((AT91PS_NVIC) 	0xE000E000) // (NVIC) Base Address
#define AT91C_BASE_MPU       ((AT91PS_MPU) 	0xE000ED90) // (MPU) Base Address
#define AT91C_BASE_CM3       ((AT91PS_CM3) 	0xE000ED00) // (CM3) Base Address
#define AT91C_BASE_PDC_DBGU0 ((AT91PS_PDC) 	0x400E0700) // (PDC_DBGU0) Base Address
#define AT91C_BASE_DBGU0     ((AT91PS_DBGU) 	0x400E0600) // (DBGU0) Base Address
#define AT91C_BASE_PDC_DBGU1 ((AT91PS_PDC) 	0x400E0900) // (PDC_DBGU1) Base Address
#define AT91C_BASE_DBGU1     ((AT91PS_DBGU) 	0x400E0800) // (DBGU1) Base Address
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0x400E0E00) // (PIOA) Base Address
#define AT91C_BASE_PDC_PIOA  ((AT91PS_PDC) 	0x400E0F68) // (PDC_PIOA) Base Address
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0x400E1000) // (PIOB) Base Address
#define AT91C_BASE_PIOC      ((AT91PS_PIO) 	0x400E1200) // (PIOC) Base Address
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0x400E0400) // (PMC) Base Address
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0x400E041C) // (CKGR) Base Address
#define AT91C_BASE_RSTC      ((AT91PS_RSTC) 	0x400E1400) // (RSTC) Base Address
#define AT91C_BASE_SUPC      ((AT91PS_SUPC) 	0x400E1410) // (SUPC) Base Address
#define AT91C_BASE_RTTC      ((AT91PS_RTTC) 	0x400E1430) // (RTTC) Base Address
#define AT91C_BASE_WDTC      ((AT91PS_WDTC) 	0x400E1450) // (WDTC) Base Address
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0x400E1460) // (RTC) Base Address
#define AT91C_BASE_ADC0      ((AT91PS_ADC) 	0x40038000) // (ADC0) Base Address
#define AT91C_BASE_DAC0      ((AT91PS_DAC) 	0x4003C000) // (DAC0) Base Address
#define AT91C_BASE_ACC0      ((AT91PS_ACC) 	0x40040000) // (ACC0) Base Address
#define AT91C_BASE_HCBDMA    ((AT91PS_HCBDMA) 	0x40044000) // (HCBDMA) Base Address
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0x40010000) // (TC0) Base Address
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0x40010040) // (TC1) Base Address
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0x40010080) // (TC2) Base Address
#define AT91C_BASE_TC3       ((AT91PS_TC) 	0x40014000) // (TC3) Base Address
#define AT91C_BASE_TC4       ((AT91PS_TC) 	0x40014040) // (TC4) Base Address
#define AT91C_BASE_TC5       ((AT91PS_TC) 	0x40014080) // (TC5) Base Address
#define AT91C_BASE_TCB0      ((AT91PS_TCB) 	0x40010000) // (TCB0) Base Address
#define AT91C_BASE_TCB1      ((AT91PS_TCB) 	0x40010040) // (TCB1) Base Address
#define AT91C_BASE_TCB2      ((AT91PS_TCB) 	0x40010080) // (TCB2) Base Address
#define AT91C_BASE_TCB3      ((AT91PS_TCB) 	0x40014000) // (TCB3) Base Address
#define AT91C_BASE_TCB4      ((AT91PS_TCB) 	0x40014040) // (TCB4) Base Address
#define AT91C_BASE_TCB5      ((AT91PS_TCB) 	0x40014080) // (TCB5) Base Address
#define AT91C_BASE_EFC0      ((AT91PS_EFC) 	0x400E0A00) // (EFC0) Base Address
#define AT91C_BASE_MCI0      ((AT91PS_MCI) 	0x40000000) // (MCI0) Base Address
#define AT91C_BASE_PDC_TWI0  ((AT91PS_PDC) 	0x40018100) // (PDC_TWI0) Base Address
#define AT91C_BASE_PDC_TWI1  ((AT91PS_PDC) 	0x4001C100) // (PDC_TWI1) Base Address
#define AT91C_BASE_TWI0      ((AT91PS_TWI) 	0x40018000) // (TWI0) Base Address
#define AT91C_BASE_TWI1      ((AT91PS_TWI) 	0x4001C000) // (TWI1) Base Address
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0x40024100) // (PDC_US0) Base Address
#define AT91C_BASE_US0       ((AT91PS_USART) 	0x40024000) // (US0) Base Address
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0x40028100) // (PDC_US1) Base Address
#define AT91C_BASE_US1       ((AT91PS_USART) 	0x40028000) // (US1) Base Address
#define AT91C_BASE_PDC_SSC0  ((AT91PS_PDC) 	0x40004100) // (PDC_SSC0) Base Address
#define AT91C_BASE_SSC0      ((AT91PS_SSC) 	0x40004000) // (SSC0) Base Address
#define AT91C_BASE_PDC_PWMC  ((AT91PS_PDC) 	0x40020100) // (PDC_PWMC) Base Address
#define AT91C_BASE_PWMC_CH0  ((AT91PS_PWMC_CH) 	0x40020200) // (PWMC_CH0) Base Address
#define AT91C_BASE_PWMC_CH1  ((AT91PS_PWMC_CH) 	0x40020220) // (PWMC_CH1) Base Address
#define AT91C_BASE_PWMC_CH2  ((AT91PS_PWMC_CH) 	0x40020240) // (PWMC_CH2) Base Address
#define AT91C_BASE_PWMC_CH3  ((AT91PS_PWMC_CH) 	0x40020260) // (PWMC_CH3) Base Address
#define AT91C_BASE_PWMC      ((AT91PS_PWMC) 	0x40020000) // (PWMC) Base Address
#define AT91C_BASE_SPI0      ((AT91PS_SPI) 	0x40008000) // (SPI0) Base Address
#define AT91C_BASE_UDP       ((AT91PS_UDP) 	0x40034000) // (UDP) Base Address

// *****************************************************************************
//               MEMORY MAPPING DEFINITIONS FOR AT91SAM3S1
// *****************************************************************************
// IRAM
#define AT91C_IRAM 	 ((char *) 	0x20000000) // Maximum Internal SRAM base address
#define AT91C_IRAM_SIZE	 ((unsigned int) 0x0000C000) // Maximum Internal SRAM size in byte (48 Kbytes)
// IROM
#define AT91C_IROM 	 ((char *) 	0x00800000) // Internal ROM base address
#define AT91C_IROM_SIZE	 ((unsigned int) 0x00008000) // Internal ROM size in byte (32 Kbytes)
// EBI_CS0
#define AT91C_EBI_CS0	 ((char *) 	0x60000000) // EBI Chip Select 0 base address
#define AT91C_EBI_CS0_SIZE	 ((unsigned int) 0x01000000) // EBI Chip Select 0 size in byte (16384 Kbytes)
// EBI_SM0
#define AT91C_EBI_SM0	 ((char *) 	0x60000000) // NANDFLASH on EBI Chip Select 0 base address
#define AT91C_EBI_SM0_SIZE	 ((unsigned int) 0x01000000) // NANDFLASH on EBI Chip Select 0 size in byte (16384 Kbytes)
// EBI_CS1
#define AT91C_EBI_CS1	 ((char *) 	0x61000000) // EBI Chip Select 1 base address
#define AT91C_EBI_CS1_SIZE	 ((unsigned int) 0x01000000) // EBI Chip Select 1 size in byte (16384 Kbytes)
// EBI_SM1
#define AT91C_EBI_SM1	 ((char *) 	0x61000000) // NANDFLASH on EBI Chip Select 1 base address
#define AT91C_EBI_SM1_SIZE	 ((unsigned int) 0x01000000) // NANDFLASH on EBI Chip Select 1 size in byte (16384 Kbytes)
// EBI_CS2
#define AT91C_EBI_CS2	 ((char *) 	0x62000000) // EBI Chip Select 2 base address
#define AT91C_EBI_CS2_SIZE	 ((unsigned int) 0x01000000) // EBI Chip Select 2 size in byte (16384 Kbytes)
// EBI_SM2
#define AT91C_EBI_SM2	 ((char *) 	0x62000000) // NANDFLASH on EBI Chip Select 2 base address
#define AT91C_EBI_SM2_SIZE	 ((unsigned int) 0x01000000) // NANDFLASH on EBI Chip Select 2 size in byte (16384 Kbytes)
// EBI_CS3
#define AT91C_EBI_CS3	 ((char *) 	0x63000000) // EBI Chip Select 3 base address
#define AT91C_EBI_CS3_SIZE	 ((unsigned int) 0x01000000) // EBI Chip Select 3 size in byte (16384 Kbytes)
// EBI_SM3
#define AT91C_EBI_SM3	 ((char *) 	0x63000000) // NANDFLASH on EBI Chip Select 3 base address
#define AT91C_EBI_SM3_SIZE	 ((unsigned int) 0x01000000) // NANDFLASH on EBI Chip Select 3 size in byte (16384 Kbytes)
// EBI_CS4
#define AT91C_EBI_CS4	 ((char *) 	0x64000000) // EBI Chip Select 4 base address
#define AT91C_EBI_CS4_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CF0
#define AT91C_EBI_CF0	 ((char *) 	0x64000000) // CompactFlash 0 on EBI Chip Select 4 base address
#define AT91C_EBI_CF0_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// EBI_CS5
#define AT91C_EBI_CS5	 ((char *) 	0x65000000) // EBI Chip Select 5 base address
#define AT91C_EBI_CS5_SIZE	 ((unsigned int) 0x10000000) // EBI Chip Select 5 size in byte (262144 Kbytes)
// EBI_CF1
#define AT91C_EBI_CF1	 ((char *) 	0x65000000) // CompactFlash 1 on EBIChip Select 5 base address
#define AT91C_EBI_CF1_SIZE	 ((unsigned int) 0x10000000) // CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
// EBI_SDRAM
#define AT91C_EBI_SDRAM	 ((char *) 	0x66000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_SIZE	 ((unsigned int) 0x10000000) // SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// EBI_SDRAM_16BIT
#define AT91C_EBI_SDRAM_16BIT	 ((char *) 	0x67000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_16BIT_SIZE	 ((unsigned int) 0x02000000) // SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// EBI_SDRAM_32BIT
#define AT91C_EBI_SDRAM_32BIT	 ((char *) 	0x68000000) // SDRAM on EBI Chip Select 1 base address
#define AT91C_EBI_SDRAM_32BIT_SIZE	 ((unsigned int) 0x04000000) // SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

// - Hardware register definition

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR System Peripherals
// - *****************************************************************************
// - -------- GPBR : (SYS Offset: 0x1490) GPBR General Purpose Register -------- 
AT91C_GPBR_GPRV           EQU (0x0 <<  0) ;- (SYS) General Purpose Register Value

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface
// - *****************************************************************************
// - -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- 
AT91C_SMC_NWESETUP        EQU (0x3F <<  0) ;- (SMC) NWE Setup Length
AT91C_SMC_NCSSETUPWR      EQU (0x3F <<  8) ;- (SMC) NCS Setup Length in WRite Access
AT91C_SMC_NRDSETUP        EQU (0x3F << 16) ;- (SMC) NRD Setup Length
AT91C_SMC_NCSSETUPRD      EQU (0x3F << 24) ;- (SMC) NCS Setup Length in ReaD Access
// - -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- 
AT91C_SMC_NWEPULSE        EQU (0x7F <<  0) ;- (SMC) NWE Pulse Length
AT91C_SMC_NCSPULSEWR      EQU (0x7F <<  8) ;- (SMC) NCS Pulse Length in WRite Access
AT91C_SMC_NRDPULSE        EQU (0x7F << 16) ;- (SMC) NRD Pulse Length
AT91C_SMC_NCSPULSERD      EQU (0x7F << 24) ;- (SMC) NCS Pulse Length in ReaD Access
// - -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- 
AT91C_SMC_NWECYCLE        EQU (0x1FF <<  0) ;- (SMC) Total Write Cycle Length
AT91C_SMC_NRDCYCLE        EQU (0x1FF << 16) ;- (SMC) Total Read Cycle Length
// - -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- 
AT91C_SMC_READMODE        EQU (0x1 <<  0) ;- (SMC) Read Mode
AT91C_SMC_WRITEMODE       EQU (0x1 <<  1) ;- (SMC) Write Mode
AT91C_SMC_NWAITM          EQU (0x3 <<  5) ;- (SMC) NWAIT Mode
AT91C_SMC_NWAITM_NWAIT_DISABLE EQU (0x0 <<  5) ;- (SMC) External NWAIT disabled.
AT91C_SMC_NWAITM_NWAIT_ENABLE_FROZEN EQU (0x2 <<  5) ;- (SMC) External NWAIT enabled in frozen mode.
AT91C_SMC_NWAITM_NWAIT_ENABLE_READY EQU (0x3 <<  5) ;- (SMC) External NWAIT enabled in ready mode.
AT91C_SMC_BAT             EQU (0x1 <<  8) ;- (SMC) Byte Access Type
AT91C_SMC_BAT_BYTE_SELECT EQU (0x0 <<  8) ;- (SMC) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.
AT91C_SMC_BAT_BYTE_WRITE  EQU (0x1 <<  8) ;- (SMC) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.
AT91C_SMC_DBW             EQU (0x3 << 12) ;- (SMC) Data Bus Width
AT91C_SMC_DBW_WIDTH_EIGTH_BITS EQU (0x0 << 12) ;- (SMC) 8 bits.
AT91C_SMC_DBW_WIDTH_SIXTEEN_BITS EQU (0x1 << 12) ;- (SMC) 16 bits.
AT91C_SMC_DBW_WIDTH_THIRTY_TWO_BITS EQU (0x2 << 12) ;- (SMC) 32 bits.
AT91C_SMC_TDF             EQU (0xF << 16) ;- (SMC) Data Float Time.
AT91C_SMC_TDFEN           EQU (0x1 << 20) ;- (SMC) TDF Enabled.
AT91C_SMC_PMEN            EQU (0x1 << 24) ;- (SMC) Page Mode Enabled.
AT91C_SMC_PS              EQU (0x3 << 28) ;- (SMC) Page Size
AT91C_SMC_PS_SIZE_FOUR_BYTES EQU (0x0 << 28) ;- (SMC) 4 bytes.
AT91C_SMC_PS_SIZE_EIGHT_BYTES EQU (0x1 << 28) ;- (SMC) 8 bytes.
AT91C_SMC_PS_SIZE_SIXTEEN_BYTES EQU (0x2 << 28) ;- (SMC) 16 bytes.
AT91C_SMC_PS_SIZE_THIRTY_TWO_BYTES EQU (0x3 << 28) ;- (SMC) 32 bytes.
// - -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- 
// - -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- 
// - -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- 
// - -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- 
// - -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- 

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
// -              SOFTWARE API DEFINITION  FOR AHB CCFG Interface
// - *****************************************************************************
// - -------- CCFG_RAM0 : (CCFG Offset: 0x0) RAM0 Configuration -------- 
// - -------- CCFG_ROM : (CCFG Offset: 0x4) ROM configuration -------- 
// - -------- CCFG_FLASH0 : (CCFG Offset: 0x8) FLASH0 configuration -------- 
// - -------- CCFG_EBICSA : (CCFG Offset: 0xc) EBI Chip Select Assignement Register -------- 
AT91C_EBI_CS0A            EQU (0x1 <<  0) ;- (CCFG) Chip Select 0 Assignment
AT91C_EBI_CS0A_SMC        EQU (0x0) ;- (CCFG) Chip Select 0 is only assigned to the Static Memory Controller and NCS0 behaves as defined by the SMC.
AT91C_EBI_CS0A_SM         EQU (0x1) ;- (CCFG) Chip Select 0 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
AT91C_EBI_CS1A            EQU (0x1 <<  1) ;- (CCFG) Chip Select 1 Assignment
AT91C_EBI_CS1A_SMC        EQU (0x0 <<  1) ;- (CCFG) Chip Select 1 is only assigned to the Static Memory Controller and NCS1 behaves as defined by the SMC.
AT91C_EBI_CS1A_SM         EQU (0x1 <<  1) ;- (CCFG) Chip Select 1 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
AT91C_EBI_CS2A            EQU (0x1 <<  2) ;- (CCFG) Chip Select 2 Assignment
AT91C_EBI_CS2A_SMC        EQU (0x0 <<  2) ;- (CCFG) Chip Select 2 is only assigned to the Static Memory Controller and NCS2 behaves as defined by the SMC.
AT91C_EBI_CS2A_SM         EQU (0x1 <<  2) ;- (CCFG) Chip Select 2 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
AT91C_EBI_CS3A            EQU (0x1 <<  3) ;- (CCFG) Chip Select 3 Assignment
AT91C_EBI_CS3A_SMC        EQU (0x0 <<  3) ;- (CCFG) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.
AT91C_EBI_CS3A_SM         EQU (0x1 <<  3) ;- (CCFG) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
// - -------- CCFG_BRIDGE : (CCFG Offset: 0x10) BRIDGE configuration -------- 

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

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Power Management Controler
// - *****************************************************************************
// - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- 
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
AT91C_PMC_UDP             EQU (0x1 <<  7) ;- (PMC) USB Device Port Clock
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
// - -------- CKGR_PLLBR : (PMC Offset: 0x2c) PLL B Register -------- 
AT91C_CKGR_DIVB           EQU (0xFF <<  0) ;- (PMC) Divider B Selected
AT91C_CKGR_DIVB_0         EQU (0x0) ;- (PMC) Divider B output is 0
AT91C_CKGR_DIVB_BYPASS    EQU (0x1) ;- (PMC) Divider B is bypassed
AT91C_CKGR_PLLBCOUNT      EQU (0x3F <<  8) ;- (PMC) PLL B Counter
AT91C_CKGR_OUTB           EQU (0x3 << 14) ;- (PMC) PLL B Output Frequency Range
AT91C_CKGR_OUTB_0         EQU (0x0 << 14) ;- (PMC) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_1         EQU (0x1 << 14) ;- (PMC) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_2         EQU (0x2 << 14) ;- (PMC) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_3         EQU (0x3 << 14) ;- (PMC) Please refer to the PLLB datasheet
AT91C_CKGR_MULB           EQU (0x7FF << 16) ;- (PMC) PLL B Multiplier
// - -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- 
AT91C_PMC_CSS             EQU (0x7 <<  0) ;- (PMC) Programmable Clock Selection
AT91C_PMC_CSS_SLOW_CLK    EQU (0x0) ;- (PMC) Slow Clock is selected
AT91C_PMC_CSS_MAIN_CLK    EQU (0x1) ;- (PMC) Main Clock is selected
AT91C_PMC_CSS_PLLA_CLK    EQU (0x2) ;- (PMC) Clock from PLL A is selected
AT91C_PMC_CSS_PLLB_CLK    EQU (0x3) ;- (PMC) Clock from PLL B is selected
AT91C_PMC_CSS_SYS_CLK     EQU (0x4) ;- (PMC) System clock is selected
AT91C_PMC_PRES            EQU (0xF <<  4) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_PRES_CLK        EQU (0x0 <<  4) ;- (PMC) Selected clock
AT91C_PMC_PRES_CLK_2      EQU (0x1 <<  4) ;- (PMC) Selected clock divided by 2
AT91C_PMC_PRES_CLK_4      EQU (0x2 <<  4) ;- (PMC) Selected clock divided by 4
AT91C_PMC_PRES_CLK_8      EQU (0x3 <<  4) ;- (PMC) Selected clock divided by 8
AT91C_PMC_PRES_CLK_16     EQU (0x4 <<  4) ;- (PMC) Selected clock divided by 16
AT91C_PMC_PRES_CLK_32     EQU (0x5 <<  4) ;- (PMC) Selected clock divided by 32
AT91C_PMC_PRES_CLK_64     EQU (0x6 <<  4) ;- (PMC) Selected clock divided by 64
AT91C_PMC_PRES_CLK_3      EQU (0x7 <<  4) ;- (PMC) Selected clock divided by 3
AT91C_PMC_PRES_CLK_1_5    EQU (0x8 <<  4) ;- (PMC) Selected clock divided by 1.5
// - -------- PMC_UDPR : (PMC Offset: 0x38) USB DEV Clock Configuration Register -------- 
AT91C_PMC_UDP_CLK_SEL     EQU (0x1 <<  0) ;- (PMC) UDP Clock Selection
AT91C_PMC_UDP_CLK_SEL_PLLA EQU (0x0) ;- (PMC) PLL A is the selected clock source for UDP DEV
AT91C_PMC_UDP_CLK_SEL_PLLB EQU (0x1) ;- (PMC) PLL B is the selected clock source for UDP DEV
AT91C_PMC_UDP_DIV         EQU (0xF <<  8) ;- (PMC) UDP Clock Divider
AT91C_PMC_UDP_DIV_DIV     EQU (0x0 <<  8) ;- (PMC) Selected clock
AT91C_PMC_UDP_DIV_DIV_2   EQU (0x1 <<  8) ;- (PMC) Selected clock divided by 2
// - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- 
// - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- 
AT91C_PMC_MOSCXTS         EQU (0x1 <<  0) ;- (PMC) Main Crystal Oscillator Status/Enable/Disable/Mask
AT91C_PMC_LOCKA           EQU (0x1 <<  1) ;- (PMC) PLL A Status/Enable/Disable/Mask
AT91C_PMC_LOCKB           EQU (0x1 <<  2) ;- (PMC) PLL B Status/Enable/Disable/Mask
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
AT91C_                    EQU (0x0 <<  7) ;- (PMC) 
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
// - -------- PMC_PCR : (PMC Offset: 0x10c) Peripheral Control Register -------- 
AT91C_PMC_PID             EQU (0x3F <<  0) ;- (PMC) Peripheral Identifier
AT91C_PMC_CMD             EQU (0x1 << 12) ;- (PMC) Read / Write Command
AT91C_PMC_DIV             EQU (0x3 << 16) ;- (PMC) Peripheral clock Divider
AT91C_PMC_EN              EQU (0x1 << 28) ;- (PMC) Peripheral Clock Enable

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Clock Generator Controler
// - *****************************************************************************
// - -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- 
// - -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- 
// - -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- 
// - -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- 
// - -------- CKGR_PLLBR : (CKGR Offset: 0x10) PLL B Register -------- 

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
AT91C_ADC_CH8             EQU (0x1 <<  8) ;- (ADC) Channel 8
AT91C_ADC_CH9             EQU (0x1 <<  9) ;- (ADC) Channel 9
AT91C_ADC_CH10            EQU (0x1 << 10) ;- (ADC) Channel 10
AT91C_ADC_CH11            EQU (0x1 << 11) ;- (ADC) Channel 11
AT91C_ADC_CH12            EQU (0x1 << 12) ;- (ADC) Channel 12
AT91C_ADC_CH13            EQU (0x1 << 13) ;- (ADC) Channel 13
AT91C_ADC_CH14            EQU (0x1 << 14) ;- (ADC) Channel 14
AT91C_ADC_CH15            EQU (0x1 << 15) ;- (ADC) Channel 15
// - -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- 
// - -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- 
// - -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- 
AT91C_ADC_LDATA           EQU (0x3FF <<  0) ;- (ADC) Last Data Converted
// - -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- 
AT91C_ADC_EOC0            EQU (0x1 <<  0) ;- (ADC) End of Conversion
AT91C_ADC_EOC1            EQU (0x1 <<  1) ;- (ADC) End of Conversion
AT91C_ADC_EOC2            EQU (0x1 <<  2) ;- (ADC) End of Conversion
AT91C_ADC_EOC3            EQU (0x1 <<  3) ;- (ADC) End of Conversion
AT91C_ADC_EOC4            EQU (0x1 <<  4) ;- (ADC) End of Conversion
AT91C_ADC_EOC5            EQU (0x1 <<  5) ;- (ADC) End of Conversion
AT91C_ADC_EOC6            EQU (0x1 <<  6) ;- (ADC) End of Conversion
AT91C_ADC_EOC7            EQU (0x1 <<  7) ;- (ADC) End of Conversion
AT91C_ADC_EOC8            EQU (0x1 <<  8) ;- (ADC) End of Conversion
AT91C_ADC_EOC9            EQU (0x1 <<  9) ;- (ADC) End of Conversion
AT91C_ADC_EOC10           EQU (0x1 << 10) ;- (ADC) End of Conversion
AT91C_ADC_EOC11           EQU (0x1 << 11) ;- (ADC) End of Conversion
AT91C_ADC_EOC12           EQU (0x1 << 12) ;- (ADC) End of Conversion
AT91C_ADC_EOC13           EQU (0x1 << 13) ;- (ADC) End of Conversion
AT91C_ADC_EOC14           EQU (0x1 << 14) ;- (ADC) End of Conversion
AT91C_ADC_EOC15           EQU (0x1 << 15) ;- (ADC) End of Conversion
AT91C_ADC_DRDY            EQU (0x1 << 24) ;- (ADC) Data Ready
AT91C_ADC_GOVRE           EQU (0x1 << 25) ;- (ADC) General Overrun
AT91C_ADC_ENDRX           EQU (0x1 << 27) ;- (ADC) End of Receiver Transfer
AT91C_ADC_RXBUFF          EQU (0x1 << 28) ;- (ADC) RXBUFF Interrupt
// - -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- 
// - -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- 
// - -------- ADC_SR : (ADC Offset: 0x30) ADC Status Register -------- 
// - -------- ADC_OVR : (ADC Offset: 0x3c)  -------- 
AT91C_ADC_OVRE2           EQU (0x1 << 10) ;- (ADC) Overrun Error
AT91C_ADC_OVRE3           EQU (0x1 << 11) ;- (ADC) Overrun Error
AT91C_ADC_OVRE4           EQU (0x1 << 12) ;- (ADC) Overrun Error
AT91C_ADC_OVRE5           EQU (0x1 << 13) ;- (ADC) Overrun Error
// - -------- ADC_CWR : (ADC Offset: 0x40)  -------- 
// - -------- ADC_CWSR : (ADC Offset: 0x44)  -------- 
// - -------- ADC_CGR : (ADC Offset: 0x48)  -------- 
// - -------- ADC_COR : (ADC Offset: 0x4c)  -------- 
// - -------- ADC_CDR0 : (ADC Offset: 0x50) ADC Channel Data Register 0 -------- 
AT91C_ADC_DATA            EQU (0x3FF <<  0) ;- (ADC) Converted Data
// - -------- ADC_CDR1 : (ADC Offset: 0x54) ADC Channel Data Register 1 -------- 
// - -------- ADC_CDR2 : (ADC Offset: 0x58) ADC Channel Data Register 2 -------- 
// - -------- ADC_CDR3 : (ADC Offset: 0x5c) ADC Channel Data Register 3 -------- 
// - -------- ADC_CDR4 : (ADC Offset: 0x60) ADC Channel Data Register 4 -------- 
// - -------- ADC_CDR5 : (ADC Offset: 0x64) ADC Channel Data Register 5 -------- 
// - -------- ADC_CDR6 : (ADC Offset: 0x68) ADC Channel Data Register 6 -------- 
// - -------- ADC_CDR7 : (ADC Offset: 0x6c) ADC Channel Data Register 7 -------- 
// - -------- ADC_CDR8 : (ADC Offset: 0x70) ADC Channel Data Register 8 -------- 
// - -------- ADC_CDR9 : (ADC Offset: 0x74) ADC Channel Data Register 9 -------- 
// - -------- ADC_CDR10 : (ADC Offset: 0x78) ADC Channel Data Register 10 -------- 
// - -------- ADC_CDR11 : (ADC Offset: 0x7c) ADC Channel Data Register 11 -------- 
// - -------- ADC_CDR12 : (ADC Offset: 0x80) ADC Channel Data Register 12 -------- 
// - -------- ADC_CDR13 : (ADC Offset: 0x84) ADC Channel Data Register 13 -------- 
// - -------- ADC_CDR14 : (ADC Offset: 0x88) ADC Channel Data Register 14 -------- 
// - -------- ADC_CDR15 : (ADC Offset: 0x8c) ADC Channel Data Register 15 -------- 
// - -------- ADC_VER : (ADC Offset: 0xfc) ADC VER -------- 
AT91C_ADC_VER             EQU (0xF <<  0) ;- (ADC) ADC VER

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Digital to Analog Convertor
// - *****************************************************************************
// - -------- DAC_CR : (DAC Offset: 0x0) Control Register -------- 
AT91C_DAC_SWRST           EQU (0x1 <<  0) ;- (DAC) Software Reset
// - -------- DAC_MR : (DAC Offset: 0x4) Mode Register -------- 
AT91C_DAC_TRGEN           EQU (0x1 <<  0) ;- (DAC) Trigger Enable
AT91C_DAC_TRGSEL          EQU (0x7 <<  1) ;- (DAC) Trigger Selection
AT91C_DAC_TRGSEL_EXTRG0   EQU (0x0 <<  1) ;- (DAC) External Trigger 0
AT91C_DAC_TRGSEL_EXTRG1   EQU (0x1 <<  1) ;- (DAC) External Trigger 1
AT91C_DAC_TRGSEL_EXTRG2   EQU (0x2 <<  1) ;- (DAC) External Trigger 2
AT91C_DAC_TRGSEL_EXTRG3   EQU (0x3 <<  1) ;- (DAC) External Trigger 3
AT91C_DAC_TRGSEL_EXTRG4   EQU (0x4 <<  1) ;- (DAC) External Trigger 4
AT91C_DAC_TRGSEL_EXTRG5   EQU (0x5 <<  1) ;- (DAC) External Trigger 5
AT91C_DAC_TRGSEL_EXTRG6   EQU (0x6 <<  1) ;- (DAC) External Trigger 6
AT91C_DAC_WORD            EQU (0x1 <<  4) ;- (DAC) Word Transfer
AT91C_DAC_SLEEP           EQU (0x1 <<  5) ;- (DAC) Sleep Mode
AT91C_DAC_FASTW           EQU (0x1 <<  6) ;- (DAC) Fast Wake up Mode
AT91C_DAC_REFRESH         EQU (0xFF <<  8) ;- (DAC) Refresh period
AT91C_DAC_USER_SEL        EQU (0x3 << 16) ;- (DAC) User Channel Selection
AT91C_DAC_USER_SEL_CH0    EQU (0x0 << 16) ;- (DAC) Channel 0
AT91C_DAC_USER_SEL_CH1    EQU (0x1 << 16) ;- (DAC) Channel 1
AT91C_DAC_USER_SEL_CH2    EQU (0x2 << 16) ;- (DAC) Channel 2
AT91C_DAC_TAG             EQU (0x1 << 20) ;- (DAC) Tag selection mode
AT91C_DAC_MAXSPEED        EQU (0x1 << 21) ;- (DAC) Max speed mode
AT91C_DAC_STARTUP         EQU (0x3F << 24) ;- (DAC) Startup Time Selection
// - -------- DAC_CHER : (DAC Offset: 0x10) Channel Enable Register -------- 
AT91C_DAC_CH0             EQU (0x1 <<  0) ;- (DAC) Channel 0
AT91C_DAC_CH1             EQU (0x1 <<  1) ;- (DAC) Channel 1
AT91C_DAC_CH2             EQU (0x1 <<  2) ;- (DAC) Channel 2
// - -------- DAC_CHDR : (DAC Offset: 0x14) Channel Disable Register -------- 
// - -------- DAC_CHSR : (DAC Offset: 0x18) Channel Status Register -------- 
// - -------- DAC_CDR : (DAC Offset: 0x20) Conversion Data Register -------- 
AT91C_DAC_DATA            EQU (0x0 <<  0) ;- (DAC) Data to convert
// - -------- DAC_IER : (DAC Offset: 0x24) DAC Interrupt Enable -------- 
AT91C_DAC_TXRDY           EQU (0x1 <<  0) ;- (DAC) Transmission Ready Interrupt
AT91C_DAC_EOC             EQU (0x1 <<  1) ;- (DAC) End of Conversion Interrupt
AT91C_DAC_TXDMAEND        EQU (0x1 <<  2) ;- (DAC) End of DMA Interrupt
AT91C_DAC_TXBUFEMPT       EQU (0x1 <<  3) ;- (DAC) Buffer Empty Interrupt
// - -------- DAC_IDR : (DAC Offset: 0x28) DAC Interrupt Disable -------- 
// - -------- DAC_IMR : (DAC Offset: 0x2c) DAC Interrupt Mask -------- 
// - -------- DAC_ISR : (DAC Offset: 0x30) DAC Interrupt Status -------- 
// - -------- DAC_ACR : (DAC Offset: 0x94) Analog Current Register -------- 
AT91C_DAC_IBCTL           EQU (0x1FF <<  0) ;- (DAC) Bias current control
// - -------- DAC_WPMR : (DAC Offset: 0xe4) Write Protect Mode Register -------- 
AT91C_DAC_WPEN            EQU (0x1 <<  0) ;- (DAC) Write Protect Enable
AT91C_DAC_WPKEY           EQU (0xFFFFFF <<  8) ;- (DAC) Write Protect KEY
// - -------- DAC_WPSR : (DAC Offset: 0xe8) Write Protect Status Register -------- 
AT91C_DAC_WPROTERR        EQU (0x1 <<  0) ;- (DAC) Write protection error
AT91C_DAC_WPROTADDR       EQU (0xFF <<  8) ;- (DAC) Write protection error address

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Digital to Analog Convertor
// - *****************************************************************************
// - -------- ACC_IER : (ACC Offset: 0x24) Data Ready for Conversion Interrupt Enable -------- 
AT91C_ACC_DATRDY          EQU (0x1 <<  0) ;- (ACC) Data Ready for Conversion
// - -------- ACC_IDR : (ACC Offset: 0x28) Data Ready for Conversion Interrupt Disable -------- 
// - -------- ACC_IMR : (ACC Offset: 0x2c) Data Ready for Conversion Interrupt Mask -------- 
// - -------- ACC_ISR : (ACC Offset: 0x30) Status Register -------- 
// - -------- ACC_VER : (ACC Offset: 0xfc) ACC VER -------- 
AT91C_ACC_VER             EQU (0xF <<  0) ;- (ACC) ACC VER

// - *****************************************************************************
// -              SOFTWARE API DEFINITION  FOR Context Based Direct Memory Access Controller Interface
// - *****************************************************************************
// - -------- HCBDMA_CBDSCR : (HCBDMA Offset: 0x0) CB DMA Descriptor Base Register -------- 
AT91C_HCBDMA_DSCR         EQU (0x0 <<  0) ;- (HCBDMA) Descriptor Base Address
// - -------- HCBDMA_CBNXTEN : (HCBDMA Offset: 0x4) Next Descriptor Enable Register -------- 
AT91C_HCBDMA_NXTID0       EQU (0x1 <<  0) ;- (HCBDMA) Next Descriptor Identifier for the Channel 0
AT91C_HCBDMA_NXTID1       EQU (0x1 <<  1) ;- (HCBDMA) Next Descriptor Identifier for the Channel 1
AT91C_HCBDMA_NXTID2       EQU (0x1 <<  2) ;- (HCBDMA) Next Descriptor Identifier for the Channel 2
AT91C_HCBDMA_NXTID3       EQU (0x1 <<  3) ;- (HCBDMA) Next Descriptor Identifier for the Channel 3
AT91C_HCBDMA_NXTID4       EQU (0x1 <<  4) ;- (HCBDMA) Next Descriptor Identifier for the Channel 4
AT91C_HCBDMA_NXTID5       EQU (0x1 <<  5) ;- (HCBDMA) Next Descriptor Identifier for the Channel 5
AT91C_HCBDMA_NXTID6       EQU (0x1 <<  6) ;- (HCBDMA) Next Descriptor Identifier for the Channel 6
AT91C_HCBDMA_NXTID7       EQU (0x1 <<  7) ;- (HCBDMA) Next Descriptor Identifier for the Channel 7
AT91C_HCBDMA_NXTID8       EQU (0x1 <<  8) ;- (HCBDMA) Next Descriptor Identifier for the Channel 8
AT91C_HCBDMA_NXTID9       EQU (0x1 <<  9) ;- (HCBDMA) Next Descriptor Identifier for the Channel 9
AT91C_HCBDMA_NXTID10      EQU (0x1 << 10) ;- (HCBDMA) Next Descriptor Identifier for the Channel 10
AT91C_HCBDMA_NXTID11      EQU (0x1 << 11) ;- (HCBDMA) Next Descriptor Identifier for the Channel 11
AT91C_HCBDMA_NXTID12      EQU (0x1 << 12) ;- (HCBDMA) Next Descriptor Identifier for the Channel 12
AT91C_HCBDMA_NXTID13      EQU (0x1 << 13) ;- (HCBDMA) Next Descriptor Identifier for the Channel 13
AT91C_HCBDMA_NXTID14      EQU (0x1 << 14) ;- (HCBDMA) Next Descriptor Identifier for the Channel 14
AT91C_HCBDMA_NXTID15      EQU (0x1 << 15) ;- (HCBDMA) Next Descriptor Identifier for the Channel 15
AT91C_HCBDMA_NXTID16      EQU (0x1 << 16) ;- (HCBDMA) Next Descriptor Identifier for the Channel 16
AT91C_HCBDMA_NXTID17      EQU (0x1 << 17) ;- (HCBDMA) Next Descriptor Identifier for the Channel 17
AT91C_HCBDMA_NXTID18      EQU (0x1 << 18) ;- (HCBDMA) Next Descriptor Identifier for the Channel 18
AT91C_HCBDMA_NXTID19      EQU (0x1 << 19) ;- (HCBDMA) Next Descriptor Identifier for the Channel 19
AT91C_HCBDMA_NXTID20      EQU (0x1 << 20) ;- (HCBDMA) Next Descriptor Identifier for the Channel 20
AT91C_HCBDMA_NXTID21      EQU (0x1 << 21) ;- (HCBDMA) Next Descriptor Identifier for the Channel 21
AT91C_HCBDMA_NXTID22      EQU (0x1 << 22) ;- (HCBDMA) Next Descriptor Identifier for the Channel 22
AT91C_HCBDMA_NXTID23      EQU (0x1 << 23) ;- (HCBDMA) Next Descriptor Identifier for the Channel 23
AT91C_HCBDMA_NXTID24      EQU (0x1 << 24) ;- (HCBDMA) Next Descriptor Identifier for the Channel 24
AT91C_HCBDMA_NXTID25      EQU (0x1 << 25) ;- (HCBDMA) Next Descriptor Identifier for the Channel 25
AT91C_HCBDMA_NXTID26      EQU (0x1 << 26) ;- (HCBDMA) Next Descriptor Identifier for the Channel 26
AT91C_HCBDMA_NXTID27      EQU (0x1 << 27) ;- (HCBDMA) Next Descriptor Identifier for the Channel 27
AT91C_HCBDMA_NXTID28      EQU (0x1 << 28) ;- (HCBDMA) Next Descriptor Identifier for the Channel 28
AT91C_HCBDMA_NXTID29      EQU (0x1 << 29) ;- (HCBDMA) Next Descriptor Identifier for the Channel 29
AT91C_HCBDMA_NXTID30      EQU (0x1 << 30) ;- (HCBDMA) Next Descriptor Identifier for the Channel 30
AT91C_HCBDMA_NXTID31      EQU (0x1 << 31) ;- (HCBDMA) Next Descriptor Identifier for the Channel 31
// - -------- HCBDMA_CBEN : (HCBDMA Offset: 0x8) CB DMA Enable Register -------- 
AT91C_HCBDMA_CBEN0        EQU (0x1 <<  0) ;- (HCBDMA) Enable for the Channel 0
AT91C_HCBDMA_CBEN1        EQU (0x1 <<  1) ;- (HCBDMA) Enable for the Channel 1
AT91C_HCBDMA_CBEN2        EQU (0x1 <<  2) ;- (HCBDMA) Enable for the Channel 2
AT91C_HCBDMA_CBEN3        EQU (0x1 <<  3) ;- (HCBDMA) Enable for the Channel 3
AT91C_HCBDMA_CBEN4        EQU (0x1 <<  4) ;- (HCBDMA) Enable for the Channel 4
AT91C_HCBDMA_CBEN5        EQU (0x1 <<  5) ;- (HCBDMA) Enable for the Channel 5
AT91C_HCBDMA_CBEN6        EQU (0x1 <<  6) ;- (HCBDMA) Enable for the Channel 6
AT91C_HCBDMA_CBEN7        EQU (0x1 <<  7) ;- (HCBDMA) Enable for the Channel 7
AT91C_HCBDMA_CBEN8        EQU (0x1 <<  8) ;- (HCBDMA) Enable for the Channel 8
AT91C_HCBDMA_CBEN9        EQU (0x1 <<  9) ;- (HCBDMA) Enable for the Channel 9
AT91C_HCBDMA_CBEN10       EQU (0x1 << 10) ;- (HCBDMA) Enable for the Channel 10
AT91C_HCBDMA_CBEN11       EQU (0x1 << 11) ;- (HCBDMA) Enable for the Channel 11
AT91C_HCBDMA_CBEN12       EQU (0x1 << 12) ;- (HCBDMA) Enable for the Channel 12
AT91C_HCBDMA_CBEN13       EQU (0x1 << 13) ;- (HCBDMA) Enable for the Channel 13
AT91C_HCBDMA_CBEN14       EQU (0x1 << 14) ;- (HCBDMA) Enable for the Channel 14
AT91C_HCBDMA_CBEN15       EQU (0x1 << 15) ;- (HCBDMA) Enable for the Channel 15
AT91C_HCBDMA_CBEN16       EQU (0x1 << 16) ;- (HCBDMA) Enable for the Channel 16
AT91C_HCBDMA_CBEN17       EQU (0x1 << 17) ;- (HCBDMA) Enable for the Channel 17
AT91C_HCBDMA_CBEN18       EQU (0x1 << 18) ;- (HCBDMA) Enable for the Channel 18
AT91C_HCBDMA_CBEN19       EQU (0x1 << 19) ;- (HCBDMA) Enable for the Channel 19
AT91C_HCBDMA_CBEN20       EQU (0x1 << 20) ;- (HCBDMA) Enable for the Channel 20
AT91C_HCBDMA_CBEN21       EQU (0x1 << 21) ;- (HCBDMA) Enable for the Channel 21
AT91C_HCBDMA_CBEN22       EQU (0x1 << 22) ;- (HCBDMA) Enable for the Channel 22
AT91C_HCBDMA_CBEN23       EQU (0x1 << 23) ;- (HCBDMA) Enable for the Channel 23
AT91C_HCBDMA_CBEN24       EQU (0x1 << 24) ;- (HCBDMA) Enable for the Channel 24
AT91C_HCBDMA_CBEN25       EQU (0x1 << 25) ;- (HCBDMA) Enable for the Channel 25
AT91C_HCBDMA_CBEN26       EQU (0x1 << 26) ;- (HCBDMA) Enable for the Channel 26
AT91C_HCBDMA_CBEN27       EQU (0x1 << 27) ;- (HCBDMA) Enable for the Channel 27
AT91C_HCBDMA_CBEN28       EQU (0x1 << 28) ;- (HCBDMA) Enable for the Channel 28
AT91C_HCBDMA_CBEN29       EQU (0x1 << 29) ;- (HCBDMA) Enable for the Channel 29
AT91C_HCBDMA_CBEN30       EQU (0x1 << 30) ;- (HCBDMA) Enable for the Channel 30
AT91C_HCBDMA_CBEN31       EQU (0x1 << 31) ;- (HCBDMA) Enable for the Channel 31
// - -------- HCBDMA_CBDIS : (HCBDMA Offset: 0xc) CB DMA Disable Register -------- 
AT91C_HCBDMA_CBDIS0       EQU (0x1 <<  0) ;- (HCBDMA) Disable for the Channel 0
AT91C_HCBDMA_CBDIS1       EQU (0x1 <<  1) ;- (HCBDMA) Disable for the Channel 1
AT91C_HCBDMA_CBDIS2       EQU (0x1 <<  2) ;- (HCBDMA) Disable for the Channel 2
AT91C_HCBDMA_CBDIS3       EQU (0x1 <<  3) ;- (HCBDMA) Disable for the Channel 3
AT91C_HCBDMA_CBDIS4       EQU (0x1 <<  4) ;- (HCBDMA) Disable for the Channel 4
AT91C_HCBDMA_CBDIS5       EQU (0x1 <<  5) ;- (HCBDMA) Disable for the Channel 5
AT91C_HCBDMA_CBDIS6       EQU (0x1 <<  6) ;- (HCBDMA) Disable for the Channel 6
AT91C_HCBDMA_CBDIS7       EQU (0x1 <<  7) ;- (HCBDMA) Disable for the Channel 7
AT91C_HCBDMA_CBDIS8       EQU (0x1 <<  8) ;- (HCBDMA) Disable for the Channel 8
AT91C_HCBDMA_CBDIS9       EQU (0x1 <<  9) ;- (HCBDMA) Disable for the Channel 9
AT91C_HCBDMA_CBDIS10      EQU (0x1 << 10) ;- (HCBDMA) Disable for the Channel 10
AT91C_HCBDMA_CBDIS11      EQU (0x1 << 11) ;- (HCBDMA) Disable for the Channel 11
AT91C_HCBDMA_CBDIS12      EQU (0x1 << 12) ;- (HCBDMA) Disable for the Channel 12
AT91C_HCBDMA_CBDIS13      EQU (0x1 << 13) ;- (HCBDMA) Disable for the Channel 13
AT91C_HCBDMA_CBDIS14      EQU (0x1 << 14) ;- (HCBDMA) Disable for the Channel 14
AT91C_HCBDMA_CBDIS15      EQU (0x1 << 15) ;- (HCBDMA) Disable for the Channel 15
AT91C_HCBDMA_CBDIS16      EQU (0x1 << 16) ;- (HCBDMA) Disable for the Channel 16
AT91C_HCBDMA_CBDIS17      EQU (0x1 << 17) ;- (HCBDMA) Disable for the Channel 17
AT91C_HCBDMA_CBDIS18      EQU (0x1 << 18) ;- (HCBDMA) Disable for the Channel 18
AT91C_HCBDMA_CBDIS19      EQU (0x1 << 19) ;- (HCBDMA) Disable for the Channel 19
AT91C_HCBDMA_CBDIS20      EQU (0x1 << 20) ;- (HCBDMA) Disable for the Channel 20
AT91C_HCBDMA_CBDIS21      EQU (0x1 << 21) ;- (HCBDMA) Disable for the Channel 21
AT91C_HCBDMA_CBDIS22      EQU (0x1 << 22) ;- (HCBDMA) Disable for the Channel 22
AT91C_HCBDMA_CBDIS23      EQU (0x1 << 23) ;- (HCBDMA) Disable for the Channel 23
AT91C_HCBDMA_CBDIS24      EQU (0x1 << 24) ;- (HCBDMA) Disable for the Channel 24
AT91C_HCBDMA_CBDIS25      EQU (0x1 << 25) ;- (HCBDMA) Disable for the Channel 25
AT91C_HCBDMA_CBDIS26      EQU (0x1 << 26) ;- (HCBDMA) Disable for the Channel 26
AT91C_HCBDMA_CBDIS27      EQU (0x1 << 27) ;- (HCBDMA) Disable for the Channel 27
AT91C_HCBDMA_CBDIS28      EQU (0x1 << 28) ;- (HCBDMA) Disable for the Channel 28
AT91C_HCBDMA_CBDIS29      EQU (0x1 << 29) ;- (HCBDMA) Disable for the Channel 29
AT91C_HCBDMA_CBDIS30      EQU (0x1 << 30) ;- (HCBDMA) Disable for the Channel 30
AT91C_HCBDMA_CBDIS31      EQU (0x1 << 31) ;- (HCBDMA) Disable for the Channel 31
// - -------- HCBDMA_CBSR : (HCBDMA Offset: 0x10) CB DMA Status Register -------- 
AT91C_HCBDMA_CBSR0        EQU (0x1 <<  0) ;- (HCBDMA) Status for the Channel 0
AT91C_HCBDMA_CBSR1        EQU (0x1 <<  1) ;- (HCBDMA) Status for the Channel 1
AT91C_HCBDMA_CBSR2        EQU (0x1 <<  2) ;- (HCBDMA) Status for the Channel 2
AT91C_HCBDMA_CBSR3        EQU (0x1 <<  3) ;- (HCBDMA) Status for the Channel 3
AT91C_HCBDMA_CBSR4        EQU (0x1 <<  4) ;- (HCBDMA) Status for the Channel 4
AT91C_HCBDMA_CBSR5        EQU (0x1 <<  5) ;- (HCBDMA) Status for the Channel 5
AT91C_HCBDMA_CBSR6        EQU (0x1 <<  6) ;- (HCBDMA) Status for the Channel 6
AT91C_HCBDMA_CBSR7        EQU (0x1 <<  7) ;- (HCBDMA) Status for the Channel 7
AT91C_HCBDMA_CBSR8        EQU (0x1 <<  8) ;- (HCBDMA) Status for the Channel 8
AT91C_HCBDMA_CBSR9        EQU (0x1 <<  9) ;- (HCBDMA) Status for the Channel 9
AT91C_HCBDMA_CBSR10       EQU (0x1 << 10) ;- (HCBDMA) Status for the Channel 10
AT91C_HCBDMA_CBSR11       EQU (0x1 << 11) ;- (HCBDMA) Status for the Channel 11
AT91C_HCBDMA_CBSR12       EQU (0x1 << 12) ;- (HCBDMA) Status for the Channel 12
AT91C_HCBDMA_CBSR13       EQU (0x1 << 13) ;- (HCBDMA) Status for the Channel 13
AT91C_HCBDMA_CBSR14       EQU (0x1 << 14) ;- (HCBDMA) Status for the Channel 14
AT91C_HCBDMA_CBSR15       EQU (0x1 << 15) ;- (HCBDMA) Status for the Channel 15
AT91C_HCBDMA_CBSR16       EQU (0x1 << 16) ;- (HCBDMA) Status for the Channel 16
AT91C_HCBDMA_CBSR17       EQU (0x1 << 17) ;- (HCBDMA) Status for the Channel 17
AT91C_HCBDMA_CBSR18       EQU (0x1 << 18) ;- (HCBDMA) Status for the Channel 18
AT91C_HCBDMA_CBSR19       EQU (0x1 << 19) ;- (HCBDMA) Status for the Channel 19
AT91C_HCBDMA_CBSR20       EQU (0x1 << 20) ;- (HCBDMA) Status for the Channel 20
AT91C_HCBDMA_CBSR21       EQU (0x1 << 21) ;- (HCBDMA) Status for the Channel 21
AT91C_HCBDMA_CBSR22       EQU (0x1 << 22) ;- (HCBDMA) Status for the Channel 22
AT91C_HCBDMA_CBSR23       EQU (0x1 << 23) ;- (HCBDMA) Status for the Channel 23
AT91C_HCBDMA_CBSR24       EQU (0x1 << 24) ;- (HCBDMA) Status for the Channel 24
AT91C_HCBDMA_CBSR25       EQU (0x1 << 25) ;- (HCBDMA) Status for the Channel 25
AT91C_HCBDMA_CBSR26       EQU (0x1 << 26) ;- (HCBDMA) Status for the Channel 26
AT91C_HCBDMA_CBSR27       EQU (0x1 << 27) ;- (HCBDMA) Status for the Channel 27
AT91C_HCBDMA_CBSR28       EQU (0x1 << 28) ;- (HCBDMA) Status for the Channel 28
AT91C_HCBDMA_CBSR29       EQU (0x1 << 29) ;- (HCBDMA) Status for the Channel 29
AT91C_HCBDMA_CBSR30       EQU (0x1 << 30) ;- (HCBDMA) Status for the Channel 30
AT91C_HCBDMA_CBSR31       EQU (0x1 << 31) ;- (HCBDMA) Status for the Channel 31
// - -------- HCBDMA_CBIER : (HCBDMA Offset: 0x14) CB DMA Interrupt Enable Register -------- 
AT91C_HCBDMA_CBIER0       EQU (0x1 <<  0) ;- (HCBDMA) Interrupt enable for the Channel 0
AT91C_HCBDMA_CBIER1       EQU (0x1 <<  1) ;- (HCBDMA) Interrupt enable for the Channel 1
AT91C_HCBDMA_CBIER2       EQU (0x1 <<  2) ;- (HCBDMA) Interrupt enable for the Channel 2
AT91C_HCBDMA_CBIER3       EQU (0x1 <<  3) ;- (HCBDMA) Interrupt enable for the Channel 3
AT91C_HCBDMA_CBIER4       EQU (0x1 <<  4) ;- (HCBDMA) Interrupt enable for the Channel 4
AT91C_HCBDMA_CBIER5       EQU (0x1 <<  5) ;- (HCBDMA) Interrupt enable for the Channel 5
AT91C_HCBDMA_CBIER6       EQU (0x1 <<  6) ;- (HCBDMA) Interrupt enable for the Channel 6
AT91C_HCBDMA_CBIER7       EQU (0x1 <<  7) ;- (HCBDMA) Interrupt enable for the Channel 7
AT91C_HCBDMA_CBIER8       EQU (0x1 <<  8) ;- (HCBDMA) Interrupt enable for the Channel 8
AT91C_HCBDMA_CBIER9       EQU (0x1 <<  9) ;- (HCBDMA) Interrupt enable for the Channel 9
AT91C_HCBDMA_CBIER10      EQU (0x1 << 10) ;- (HCBDMA) Interrupt enable for the Channel 10
AT91C_HCBDMA_CBIER11      EQU (0x1 << 11) ;- (HCBDMA) Interrupt enable for the Channel 11
AT91C_HCBDMA_CBIER12      EQU (0x1 << 12) ;- (HCBDMA) Interrupt enable for the Channel 12
AT91C_HCBDMA_CBIER13      EQU (0x1 << 13) ;- (HCBDMA) Interrupt enable for the Channel 13
AT91C_HCBDMA_CBIER14      EQU (0x1 << 14) ;- (HCBDMA) Interrupt enable for the Channel 14
AT91C_HCBDMA_CBIER15      EQU (0x1 << 15) ;- (HCBDMA) Interrupt enable for the Channel 15
AT91C_HCBDMA_CBIER16      EQU (0x1 << 16) ;- (HCBDMA) Interrupt enable for the Channel 16
AT91C_HCBDMA_CBIER17      EQU (0x1 << 17) ;- (HCBDMA) Interrupt enable for the Channel 17
AT91C_HCBDMA_CBIER18      EQU (0x1 << 18) ;- (HCBDMA) Interrupt enable for the Channel 18
AT91C_HCBDMA_CBIER19      EQU (0x1 << 19) ;- (HCBDMA) Interrupt enable for the Channel 19
AT91C_HCBDMA_CBIER20      EQU (0x1 << 20) ;- (HCBDMA) Interrupt enable for the Channel 20
AT91C_HCBDMA_CBIER21      EQU (0x1 << 21) ;- (HCBDMA) Interrupt enable for the Channel 21
AT91C_HCBDMA_CBIER22      EQU (0x1 << 22) ;- (HCBDMA) Interrupt enable for the Channel 22
AT91C_HCBDMA_CBIER23      EQU (0x1 << 23) ;- (HCBDMA) Interrupt enable for the Channel 23
AT91C_HCBDMA_CBIER24      EQU (0x1 << 24) ;- (HCBDMA) Interrupt enable for the Channel 24
AT91C_HCBDMA_CBIER25      EQU (0x1 << 25) ;- (HCBDMA) Interrupt enable for the Channel 25
AT91C_HCBDMA_CBIER26      EQU (0x1 << 26) ;- (HCBDMA) Interrupt enable for the Channel 26
AT91C_HCBDMA_CBIER27      EQU (0x1 << 27) ;- (HCBDMA) Interrupt enable for the Channel 27
AT91C_HCBDMA_CBIER28      EQU (0x1 << 28) ;- (HCBDMA) Interrupt enable for the Channel 28
AT91C_HCBDMA_CBIER29      EQU (0x1 << 29) ;- (HCBDMA) Interrupt enable for the Channel 29
AT91C_HCBDMA_CBIER30      EQU (0x1 << 30) ;- (HCBDMA) Interrupt enable for the Channel 30
AT91C_HCBDMA_CBIER31      EQU (0x1 << 31) ;- (HCBDMA) Interrupt enable for the Channel 31
// - -------- HCBDMA_CBIDR : (HCBDMA Offset: 0x18) CB DMA Interrupt Disable Register -------- 
AT91C_HCBDMA_CBIDR0       EQU (0x1 <<  0) ;- (HCBDMA) Interrupt disable for the Channel 0
AT91C_HCBDMA_CBIDR1       EQU (0x1 <<  1) ;- (HCBDMA) Interrupt disable for the Channel 1
AT91C_HCBDMA_CBIDR2       EQU (0x1 <<  2) ;- (HCBDMA) Interrupt disable for the Channel 2
AT91C_HCBDMA_CBIDR3       EQU (0x1 <<  3) ;- (HCBDMA) Interrupt disable for the Channel 3
AT91C_HCBDMA_CBIDR4       EQU (0x1 <<  4) ;- (HCBDMA) Interrupt disable for the Channel 4
AT91C_HCBDMA_CBIDR5       EQU (0x1 <<  5) ;- (HCBDMA) Interrupt disable for the Channel 5
AT91C_HCBDMA_CBIDR6       EQU (0x1 <<  6) ;- (HCBDMA) Interrupt disable for the Channel 6
AT91C_HCBDMA_CBIDR7       EQU (0x1 <<  7) ;- (HCBDMA) Interrupt disable for the Channel 7
AT91C_HCBDMA_CBIDR8       EQU (0x1 <<  8) ;- (HCBDMA) Interrupt disable for the Channel 8
AT91C_HCBDMA_CBIDR9       EQU (0x1 <<  9) ;- (HCBDMA) Interrupt disable for the Channel 9
AT91C_HCBDMA_CBIDR10      EQU (0x1 << 10) ;- (HCBDMA) Interrupt disable for the Channel 10
AT91C_HCBDMA_CBIDR11      EQU (0x1 << 11) ;- (HCBDMA) Interrupt disable for the Channel 11
AT91C_HCBDMA_CBIDR12      EQU (0x1 << 12) ;- (HCBDMA) Interrupt disable for the Channel 12
AT91C_HCBDMA_CBIDR13      EQU (0x1 << 13) ;- (HCBDMA) Interrupt disable for the Channel 13
AT91C_HCBDMA_CBIDR14      EQU (0x1 << 14) ;- (HCBDMA) Interrupt disable for the Channel 14
AT91C_HCBDMA_CBIDR15      EQU (0x1 << 15) ;- (HCBDMA) Interrupt disable for the Channel 15
AT91C_HCBDMA_CBIDR16      EQU (0x1 << 16) ;- (HCBDMA) Interrupt disable for the Channel 16
AT91C_HCBDMA_CBIDR17      EQU (0x1 << 17) ;- (HCBDMA) Interrupt disable for the Channel 17
AT91C_HCBDMA_CBIDR18      EQU (0x1 << 18) ;- (HCBDMA) Interrupt disable for the Channel 18
AT91C_HCBDMA_CBIDR19      EQU (0x1 << 19) ;- (HCBDMA) Interrupt disable for the Channel 19
AT91C_HCBDMA_CBIDR20      EQU (0x1 << 20) ;- (HCBDMA) Interrupt disable for the Channel 20
AT91C_HCBDMA_CBIDR21      EQU (0x1 << 21) ;- (HCBDMA) Interrupt disable for the Channel 21
AT91C_HCBDMA_CBIDR22      EQU (0x1 << 22) ;- (HCBDMA) Interrupt disable for the Channel 22
AT91C_HCBDMA_CBIDR23      EQU (0x1 << 23) ;- (HCBDMA) Interrupt disable for the Channel 23
AT91C_HCBDMA_CBIDR24      EQU (0x1 << 24) ;- (HCBDMA) Interrupt disable for the Channel 24
AT91C_HCBDMA_CBIDR25      EQU (0x1 << 25) ;- (HCBDMA) Interrupt disable for the Channel 25
AT91C_HCBDMA_CBIDR26      EQU (0x1 << 26) ;- (HCBDMA) Interrupt disable for the Channel 26
AT91C_HCBDMA_CBIDR27      EQU (0x1 << 27) ;- (HCBDMA) Interrupt disable for the Channel 27
AT91C_HCBDMA_CBIDR28      EQU (0x1 << 28) ;- (HCBDMA) Interrupt disable for the Channel 28
AT91C_HCBDMA_CBIDR29      EQU (0x1 << 29) ;- (HCBDMA) Interrupt disable for the Channel 29
AT91C_HCBDMA_CBIDR30      EQU (0x1 << 30) ;- (HCBDMA) Interrupt disable for the Channel 30
AT91C_HCBDMA_CBIDR31      EQU (0x1 << 31) ;- (HCBDMA) Interrupt disable for the Channel 31
// - -------- HCBDMA_CBIMR : (HCBDMA Offset: 0x1c) CB DMA Interrupt Mask Register -------- 
AT91C_HCBDMA_CBIMR0       EQU (0x1 <<  0) ;- (HCBDMA) Interrupt mask for the Channel 0
AT91C_HCBDMA_CBIMR1       EQU (0x1 <<  1) ;- (HCBDMA) Interrupt mask for the Channel 1
AT91C_HCBDMA_CBIMR2       EQU (0x1 <<  2) ;- (HCBDMA) Interrupt mask for the Channel 2
AT91C_HCBDMA_CBIMR3       EQU (0x1 <<  3) ;- (HCBDMA) Interrupt mask for the Channel 3
AT91C_HCBDMA_CBIMR4       EQU (0x1 <<  4) ;- (HCBDMA) Interrupt mask for the Channel 4
AT91C_HCBDMA_CBIMR5       EQU (0x1 <<  5) ;- (HCBDMA) Interrupt mask for the Channel 5
AT91C_HCBDMA_CBIMR6       EQU (0x1 <<  6) ;- (HCBDMA) Interrupt mask for the Channel 6
AT91C_HCBDMA_CBIMR7       EQU (0x1 <<  7) ;- (HCBDMA) Interrupt mask for the Channel 7
AT91C_HCBDMA_CBIMR8       EQU (0x1 <<  8) ;- (HCBDMA) Interrupt mask for the Channel 8
AT91C_HCBDMA_CBIMR9       EQU (0x1 <<  9) ;- (HCBDMA) Interrupt mask for the Channel 9
AT91C_HCBDMA_CBIMR10      EQU (0x1 << 10) ;- (HCBDMA) Interrupt mask for the Channel 10
AT91C_HCBDMA_CBIMR11      EQU (0x1 << 11) ;- (HCBDMA) Interrupt mask for the Channel 11
AT91C_HCBDMA_CBIMR12      EQU (0x1 << 12) ;- (HCBDMA) Interrupt mask for the Channel 12
AT91C_HCBDMA_CBIMR13      EQU (0x1 << 13) ;- (HCBDMA) Interrupt mask for the Channel 13
AT91C_HCBDMA_CBIMR14      EQU (0x1 << 14) ;- (HCBDMA) Interrupt mask for the Channel 14
AT91C_HCBDMA_CBIMR15      EQU (0x1 << 15) ;- (HCBDMA) Interrupt mask for the Channel 15
AT91C_HCBDMA_CBIMR16      EQU (0x1 << 16) ;- (HCBDMA) Interrupt mask for the Channel 16
AT91C_HCBDMA_CBIMR17      EQU (0x1 << 17) ;- (HCBDMA) Interrupt mask for the Channel 17
AT91C_HCBDMA_CBIMR18      EQU (0x1 << 18) ;- (HCBDMA) Interrupt mask for the Channel 18
AT91C_HCBDMA_CBIMR19      EQU (0x1 << 19) ;- (HCBDMA) Interrupt mask for the Channel 19
AT91C_HCBDMA_CBIMR20      EQU (0x1 << 20) ;- (HCBDMA) Interrupt mask for the Channel 20
AT91C_HCBDMA_CBIMR21      EQU (0x1 << 21) ;- (HCBDMA) Interrupt mask for the Channel 21
AT91C_HCBDMA_CBIMR22      EQU (0x1 << 22) ;- (HCBDMA) Interrupt mask for the Channel 22
AT91C_HCBDMA_CBIMR23      EQU (0x1 << 23) ;- (HCBDMA) Interrupt mask for the Channel 23
AT91C_HCBDMA_CBIMR24      EQU (0x1 << 24) ;- (HCBDMA) Interrupt mask for the Channel 24
AT91C_HCBDMA_CBIMR25      EQU (0x1 << 25) ;- (HCBDMA) Interrupt mask for the Channel 25
AT91C_HCBDMA_CBIMR26      EQU (0x1 << 26) ;- (HCBDMA) Interrupt mask for the Channel 26
AT91C_HCBDMA_CBIMR27      EQU (0x1 << 27) ;- (HCBDMA) Interrupt mask for the Channel 27
AT91C_HCBDMA_CBIMR28      EQU (0x1 << 28) ;- (HCBDMA) Interrupt mask for the Channel 28
AT91C_HCBDMA_CBIMR29      EQU (0x1 << 29) ;- (HCBDMA) Interrupt mask for the Channel 29
AT91C_HCBDMA_CBIMR30      EQU (0x1 << 30) ;- (HCBDMA) Interrupt mask for the Channel 30
AT91C_HCBDMA_CBIMR31      EQU (0x1 << 31) ;- (HCBDMA) Interrupt mask for the Channel 31
// - -------- HCBDMA_CBISR : (HCBDMA Offset: 0x20) CB DMA Interrupt Satus Register -------- 
AT91C_HCBDMA_CBISR0       EQU (0x1 <<  0) ;- (HCBDMA) Interrupt status for the Channel 0
AT91C_HCBDMA_CBISR1       EQU (0x1 <<  1) ;- (HCBDMA) Interrupt status for the Channel 1
AT91C_HCBDMA_CBISR2       EQU (0x1 <<  2) ;- (HCBDMA) Interrupt status for the Channel 2
AT91C_HCBDMA_CBISR3       EQU (0x1 <<  3) ;- (HCBDMA) Interrupt status for the Channel 3
AT91C_HCBDMA_CBISR4       EQU (0x1 <<  4) ;- (HCBDMA) Interrupt status for the Channel 4
AT91C_HCBDMA_CBISR5       EQU (0x1 <<  5) ;- (HCBDMA) Interrupt status for the Channel 5
AT91C_HCBDMA_CBISR6       EQU (0x1 <<  6) ;- (HCBDMA) Interrupt status for the Channel 6
AT91C_HCBDMA_CBISR7       EQU (0x1 <<  7) ;- (HCBDMA) Interrupt status for the Channel 7
AT91C_HCBDMA_CBISR8       EQU (0x1 <<  8) ;- (HCBDMA) Interrupt status for the Channel 8
AT91C_HCBDMA_CBISR9       EQU (0x1 <<  9) ;- (HCBDMA) Interrupt status for the Channel 9
AT91C_HCBDMA_CBISR10      EQU (0x1 << 10) ;- (HCBDMA) Interrupt status for the Channel 10
AT91C_HCBDMA_CBISR11      EQU (0x1 << 11) ;- (HCBDMA) Interrupt status for the Channel 11
AT91C_HCBDMA_CBISR12      EQU (0x1 << 12) ;- (HCBDMA) Interrupt status for the Channel 12
AT91C_HCBDMA_CBISR13      EQU (0x1 << 13) ;- (HCBDMA) Interrupt status for the Channel 13
AT91C_HCBDMA_CBISR14      EQU (0x1 << 14) ;- (HCBDMA) Interrupt status for the Channel 14
AT91C_HCBDMA_CBISR15      EQU (0x1 << 15) ;- (HCBDMA) Interrupt status for the Channel 15
AT91C_HCBDMA_CBISR16      EQU (0x1 << 16) ;- (HCBDMA) Interrupt status for the Channel 16
AT91C_HCBDMA_CBISR17      EQU (0x1 << 17) ;- (HCBDMA) Interrupt status for the Channel 17
AT91C_HCBDMA_CBISR18      EQU (0x1 << 18) ;- (HCBDMA) Interrupt status for the Channel 18
AT91C_HCBDMA_CBISR19      EQU (0x1 << 19) ;- (HCBDMA) Interrupt status for the Channel 19
AT91C_HCBDMA_CBISR20      EQU (0x1 << 20) ;- (HCBDMA) Interrupt status for the Channel 20
AT91C_HCBDMA_CBISR21      EQU (0x1 << 21) ;- (HCBDMA) Interrupt status for the Channel 21
AT91C_HCBDMA_CBISR22      EQU (0x1 << 22) ;- (HCBDMA) Interrupt status for the Channel 22
AT91C_HCBDMA_CBISR23      EQU (0x1 << 23) ;- (HCBDMA) Interrupt status for the Channel 23
AT91C_HCBDMA_CBISR24      EQU (0x1 << 24) ;- (HCBDMA) Interrupt status for the Channel 24
AT91C_HCBDMA_CBISR25      EQU (0x1 << 25) ;- (HCBDMA) Interrupt status for the Channel 25
AT91C_HCBDMA_CBISR26      EQU (0x1 << 26) ;- (HCBDMA) Interrupt status for the Channel 26
AT91C_HCBDMA_CBISR27      EQU (0x1 << 27) ;- (HCBDMA) Interrupt status for the Channel 27
AT91C_HCBDMA_CBISR28      EQU (0x1 << 28) ;- (HCBDMA) Interrupt status for the Channel 28
AT91C_HCBDMA_CBISR29      EQU (0x1 << 29) ;- (HCBDMA) Interrupt status for the Channel 29
AT91C_HCBDMA_CBISR30      EQU (0x1 << 30) ;- (HCBDMA) Interrupt status for the Channel 30
AT91C_HCBDMA_CBISR31      EQU (0x1 << 31) ;- (HCBDMA) Interrupt status for the Channel 31
// - -------- HCBDMA_CBDLIER : (HCBDMA Offset: 0x24) CB DMA Loaded Interrupt Enable Register -------- 
AT91C_HCBDMA_CBDLIER0     EQU (0x1 <<  0) ;- (HCBDMA) Interrupt enable for the Channel 0
AT91C_HCBDMA_CBDLIER1     EQU (0x1 <<  1) ;- (HCBDMA) Interrupt enable for the Channel 1
AT91C_HCBDMA_CBDLIER2     EQU (0x1 <<  2) ;- (HCBDMA) Interrupt enable for the Channel 2
AT91C_HCBDMA_CBDLIER3     EQU (0x1 <<  3) ;- (HCBDMA) Interrupt enable for the Channel 3
AT91C_HCBDMA_CBDLIER4     EQU (0x1 <<  4) ;- (HCBDMA) Interrupt enable for the Channel 4
AT91C_HCBDMA_CBDLIER5     EQU (0x1 <<  5) ;- (HCBDMA) Interrupt enable for the Channel 5
AT91C_HCBDMA_CBDLIER6     EQU (0x1 <<  6) ;- (HCBDMA) Interrupt enable for the Channel 6
AT91C_HCBDMA_CBDLIER7     EQU (0x1 <<  7) ;- (HCBDMA) Interrupt enable for the Channel 7
AT91C_HCBDMA_CBDLIER8     EQU (0x1 <<  8) ;- (HCBDMA) Interrupt enable for the Channel 8
AT91C_HCBDMA_CBDLIER9     EQU (0x1 <<  9) ;- (HCBDMA) Interrupt enable for the Channel 9
AT91C_HCBDMA_CBDLIER10    EQU (0x1 << 10) ;- (HCBDMA) Interrupt enable for the Channel 10
AT91C_HCBDMA_CBDLIER11    EQU (0x1 << 11) ;- (HCBDMA) Interrupt enable for the Channel 11
AT91C_HCBDMA_CBDLIER12    EQU (0x1 << 12) ;- (HCBDMA) Interrupt enable for the Channel 12
AT91C_HCBDMA_CBDLIER13    EQU (0x1 << 13) ;- (HCBDMA) Interrupt enable for the Channel 13
AT91C_HCBDMA_CBDLIER14    EQU (0x1 << 14) ;- (HCBDMA) Interrupt enable for the Channel 14
AT91C_HCBDMA_CBDLIER15    EQU (0x1 << 15) ;- (HCBDMA) Interrupt enable for the Channel 15
AT91C_HCBDMA_CBDLIER16    EQU (0x1 << 16) ;- (HCBDMA) Interrupt enable for the Channel 16
AT91C_HCBDMA_CBDLIER17    EQU (0x1 << 17) ;- (HCBDMA) Interrupt enable for the Channel 17
AT91C_HCBDMA_CBDLIER18    EQU (0x1 << 18) ;- (HCBDMA) Interrupt enable for the Channel 18
AT91C_HCBDMA_CBDLIER19    EQU (0x1 << 19) ;- (HCBDMA) Interrupt enable for the Channel 19
AT91C_HCBDMA_CBDLIER20    EQU (0x1 << 20) ;- (HCBDMA) Interrupt enable for the Channel 20
AT91C_HCBDMA_CBDLIER21    EQU (0x1 << 21) ;- (HCBDMA) Interrupt enable for the Channel 21
AT91C_HCBDMA_CBDLIER22    EQU (0x1 << 22) ;- (HCBDMA) Interrupt enable for the Channel 22
AT91C_HCBDMA_CBDLIER23    EQU (0x1 << 23) ;- (HCBDMA) Interrupt enable for the Channel 23
AT91C_HCBDMA_CBDLIER24    EQU (0x1 << 24) ;- (HCBDMA) Interrupt enable for the Channel 24
AT91C_HCBDMA_CBDLIER25    EQU (0x1 << 25) ;- (HCBDMA) Interrupt enable for the Channel 25
AT91C_HCBDMA_CBDLIER26    EQU (0x1 << 26) ;- (HCBDMA) Interrupt enable for the Channel 26
AT91C_HCBDMA_CBDLIER27    EQU (0x1 << 27) ;- (HCBDMA) Interrupt enable for the Channel 27
AT91C_HCBDMA_CBDLIER28    EQU (0x1 << 28) ;- (HCBDMA) Interrupt enable for the Channel 28
AT91C_HCBDMA_CBDLIER29    EQU (0x1 << 29) ;- (HCBDMA) Interrupt enable for the Channel 29
AT91C_HCBDMA_CBDLIER30    EQU (0x1 << 30) ;- (HCBDMA) Interrupt enable for the Channel 30
AT91C_HCBDMA_CBDLIER31    EQU (0x1 << 31) ;- (HCBDMA) Interrupt enable for the Channel 31
// - -------- HCBDMA_CBDLIDR : (HCBDMA Offset: 0x28) CB DMA Interrupt Disable Register -------- 
AT91C_HCBDMA_CBDLIDR0     EQU (0x1 <<  0) ;- (HCBDMA) Interrupt disable for the Channel 0
AT91C_HCBDMA_CBDLIDR1     EQU (0x1 <<  1) ;- (HCBDMA) Interrupt disable for the Channel 1
AT91C_HCBDMA_CBDLIDR2     EQU (0x1 <<  2) ;- (HCBDMA) Interrupt disable for the Channel 2
AT91C_HCBDMA_CBDLIDR3     EQU (0x1 <<  3) ;- (HCBDMA) Interrupt disable for the Channel 3
AT91C_HCBDMA_CBDLIDR4     EQU (0x1 <<  4) ;- (HCBDMA) Interrupt disable for the Channel 4
AT91C_HCBDMA_CBDLIDR5     EQU (0x1 <<  5) ;- (HCBDMA) Interrupt disable for the Channel 5
AT91C_HCBDMA_CBDLIDR6     EQU (0x1 <<  6) ;- (HCBDMA) Interrupt disable for the Channel 6
AT91C_HCBDMA_CBDLIDR7     EQU (0x1 <<  7) ;- (HCBDMA) Interrupt disable for the Channel 7
AT91C_HCBDMA_CBDLIDR8     EQU (0x1 <<  8) ;- (HCBDMA) Interrupt disable for the Channel 8
AT91C_HCBDMA_CBDLIDR9     EQU (0x1 <<  9) ;- (HCBDMA) Interrupt disable for the Channel 9
AT91C_HCBDMA_CBDLIDR10    EQU (0x1 << 10) ;- (HCBDMA) Interrupt disable for the Channel 10
AT91C_HCBDMA_CBDLIDR11    EQU (0x1 << 11) ;- (HCBDMA) Interrupt disable for the Channel 11
AT91C_HCBDMA_CBDLIDR12    EQU (0x1 << 12) ;- (HCBDMA) Interrupt disable for the Channel 12
AT91C_HCBDMA_CBDLIDR13    EQU (0x1 << 13) ;- (HCBDMA) Interrupt disable for the Channel 13
AT91C_HCBDMA_CBDLIDR14    EQU (0x1 << 14) ;- (HCBDMA) Interrupt disable for the Channel 14
AT91C_HCBDMA_CBDLIDR15    EQU (0x1 << 15) ;- (HCBDMA) Interrupt disable for the Channel 15
AT91C_HCBDMA_CBDLIDR16    EQU (0x1 << 16) ;- (HCBDMA) Interrupt disable for the Channel 16
AT91C_HCBDMA_CBDLIDR17    EQU (0x1 << 17) ;- (HCBDMA) Interrupt disable for the Channel 17
AT91C_HCBDMA_CBDLIDR18    EQU (0x1 << 18) ;- (HCBDMA) Interrupt disable for the Channel 18
AT91C_HCBDMA_CBDLIDR19    EQU (0x1 << 19) ;- (HCBDMA) Interrupt disable for the Channel 19
AT91C_HCBDMA_CBDLIDR20    EQU (0x1 << 20) ;- (HCBDMA) Interrupt disable for the Channel 20
AT91C_HCBDMA_CBDLIDR21    EQU (0x1 << 21) ;- (HCBDMA) Interrupt disable for the Channel 21
AT91C_HCBDMA_CBDLIDR22    EQU (0x1 << 22) ;- (HCBDMA) Interrupt disable for the Channel 22
AT91C_HCBDMA_CBDLIDR23    EQU (0x1 << 23) ;- (HCBDMA) Interrupt disable for the Channel 23
AT91C_HCBDMA_CBDLIDR24    EQU (0x1 << 24) ;- (HCBDMA) Interrupt disable for the Channel 24
AT91C_HCBDMA_CBDLIDR25    EQU (0x1 << 25) ;- (HCBDMA) Interrupt disable for the Channel 25
AT91C_HCBDMA_CBDLIDR26    EQU (0x1 << 26) ;- (HCBDMA) Interrupt disable for the Channel 26
AT91C_HCBDMA_CBDLIDR27    EQU (0x1 << 27) ;- (HCBDMA) Interrupt disable for the Channel 27
AT91C_HCBDMA_CBDLIDR28    EQU (0x1 << 28) ;- (HCBDMA) Interrupt disable for the Channel 28
AT91C_HCBDMA_CBDLIDR29    EQU (0x1 << 29) ;- (HCBDMA) Interrupt disable for the Channel 29
AT91C_HCBDMA_CBDLIDR30    EQU (0x1 << 30) ;- (HCBDMA) Interrupt disable for the Channel 30
AT91C_HCBDMA_CBDLIDR31    EQU (0x1 << 31) ;- (HCBDMA) Interrupt disable for the Channel 31
// - -------- HCBDMA_CBDLIMR : (HCBDMA Offset: 0x2c) CB DMA Interrupt Mask Register -------- 
AT91C_HCBDMA_CBDLIMR0     EQU (0x1 <<  0) ;- (HCBDMA) Interrupt mask for the Channel 0
AT91C_HCBDMA_CBDLIMR1     EQU (0x1 <<  1) ;- (HCBDMA) Interrupt mask for the Channel 1
AT91C_HCBDMA_CBDLIMR2     EQU (0x1 <<  2) ;- (HCBDMA) Interrupt mask for the Channel 2
AT91C_HCBDMA_CBDLIMR3     EQU (0x1 <<  3) ;- (HCBDMA) Interrupt mask for the Channel 3
AT91C_HCBDMA_CBDLIMR4     EQU (0x1 <<  4) ;- (HCBDMA) Interrupt mask for the Channel 4
AT91C_HCBDMA_CBDLIMR5     EQU (0x1 <<  5) ;- (HCBDMA) Interrupt mask for the Channel 5
AT91C_HCBDMA_CBDLIMR6     EQU (0x1 <<  6) ;- (HCBDMA) Interrupt mask for the Channel 6
AT91C_HCBDMA_CBDLIMR7     EQU (0x1 <<  7) ;- (HCBDMA) Interrupt mask for the Channel 7
AT91C_HCBDMA_CBDLIMR8     EQU (0x1 <<  8) ;- (HCBDMA) Interrupt mask for the Channel 8
AT91C_HCBDMA_CBDLIMR9     EQU (0x1 <<  9) ;- (HCBDMA) Interrupt mask for the Channel 9
AT91C_HCBDMA_CBDLIMR10    EQU (0x1 << 10) ;- (HCBDMA) Interrupt mask for the Channel 10
AT91C_HCBDMA_CBDLIMR11    EQU (0x1 << 11) ;- (HCBDMA) Interrupt mask for the Channel 11
AT91C_HCBDMA_CBDLIMR12    EQU (0x1 << 12) ;- (HCBDMA) Interrupt mask for the Channel 12
AT91C_HCBDMA_CBDLIMR13    EQU (0x1 << 13) ;- (HCBDMA) Interrupt mask for the Channel 13
AT91C_HCBDMA_CBDLIMR14    EQU (0x1 << 14) ;- (HCBDMA) Interrupt mask for the Channel 14
AT91C_HCBDMA_CBDLIMR15    EQU (0x1 << 15) ;- (HCBDMA) Interrupt mask for the Channel 15
AT91C_HCBDMA_CBDLIMR16    EQU (0x1 << 16) ;- (HCBDMA) Interrupt mask for the Channel 16
AT91C_HCBDMA_CBDLIMR17    EQU (0x1 << 17) ;- (HCBDMA) Interrupt mask for the Channel 17
AT91C_HCBDMA_CBDLIMR18    EQU (0x1 << 18) ;- (HCBDMA) Interrupt mask for the Channel 18
AT91C_HCBDMA_CBDLIMR19    EQU (0x1 << 19) ;- (HCBDMA) Interrupt mask for the Channel 19
AT91C_HCBDMA_CBDLIMR20    EQU (0x1 << 20) ;- (HCBDMA) Interrupt mask for the Channel 20
AT91C_HCBDMA_CBDLIMR21    EQU (0x1 << 21) ;- (HCBDMA) Interrupt mask for the Channel 21
AT91C_HCBDMA_CBDLIMR22    EQU (0x1 << 22) ;- (HCBDMA) Interrupt mask for the Channel 22
AT91C_HCBDMA_CBDLIMR23    EQU (0x1 << 23) ;- (HCBDMA) Interrupt mask for the Channel 23
AT91C_HCBDMA_CBDLIMR24    EQU (0x1 << 24) ;- (HCBDMA) Interrupt mask for the Channel 24
AT91C_HCBDMA_CBDLIMR25    EQU (0x1 << 25) ;- (HCBDMA) Interrupt mask for the Channel 25
AT91C_HCBDMA_CBDLIMR26    EQU (0x1 << 26) ;- (HCBDMA) Interrupt mask for the Channel 26
AT91C_HCBDMA_CBDLIMR27    EQU (0x1 << 27) ;- (HCBDMA) Interrupt mask for the Channel 27
AT91C_HCBDMA_CBDLIMR28    EQU (0x1 << 28) ;- (HCBDMA) Interrupt mask for the Channel 28
AT91C_HCBDMA_CBDLIMR29    EQU (0x1 << 29) ;- (HCBDMA) Interrupt mask for the Channel 29
AT91C_HCBDMA_CBDLIMR30    EQU (0x1 << 30) ;- (HCBDMA) Interrupt mask for the Channel 30
AT91C_HCBDMA_CBDLIMR31    EQU (0x1 << 31) ;- (HCBDMA) Interrupt mask for the Channel 31
// - -------- HCBDMA_CBDLISR : (HCBDMA Offset: 0x30) CB DMA Interrupt Satus Register -------- 
AT91C_HCBDMA_CBDLISR0     EQU (0x1 <<  0) ;- (HCBDMA) Interrupt status for the Channel 0
AT91C_HCBDMA_CBDLISR1     EQU (0x1 <<  1) ;- (HCBDMA) Interrupt status for the Channel 1
AT91C_HCBDMA_CBDLISR2     EQU (0x1 <<  2) ;- (HCBDMA) Interrupt status for the Channel 2
AT91C_HCBDMA_CBDLISR3     EQU (0x1 <<  3) ;- (HCBDMA) Interrupt status for the Channel 3
AT91C_HCBDMA_CBDLISR4     EQU (0x1 <<  4) ;- (HCBDMA) Interrupt status for the Channel 4
AT91C_HCBDMA_CBDLISR5     EQU (0x1 <<  5) ;- (HCBDMA) Interrupt status for the Channel 5
AT91C_HCBDMA_CBDLISR6     EQU (0x1 <<  6) ;- (HCBDMA) Interrupt status for the Channel 6
AT91C_HCBDMA_CBDLISR7     EQU (0x1 <<  7) ;- (HCBDMA) Interrupt status for the Channel 7
AT91C_HCBDMA_CBDLISR8     EQU (0x1 <<  8) ;- (HCBDMA) Interrupt status for the Channel 8
AT91C_HCBDMA_CBDLISR9     EQU (0x1 <<  9) ;- (HCBDMA) Interrupt status for the Channel 9
AT91C_HCBDMA_CBDLISR10    EQU (0x1 << 10) ;- (HCBDMA) Interrupt status for the Channel 10
AT91C_HCBDMA_CBDLISR11    EQU (0x1 << 11) ;- (HCBDMA) Interrupt status for the Channel 11
AT91C_HCBDMA_CBDLISR12    EQU (0x1 << 12) ;- (HCBDMA) Interrupt status for the Channel 12
AT91C_HCBDMA_CBDLISR13    EQU (0x1 << 13) ;- (HCBDMA) Interrupt status for the Channel 13
AT91C_HCBDMA_CBDLISR14    EQU (0x1 << 14) ;- (HCBDMA) Interrupt status for the Channel 14
AT91C_HCBDMA_CBDLISR15    EQU (0x1 << 15) ;- (HCBDMA) Interrupt status for the Channel 15
AT91C_HCBDMA_CBDLISR16    EQU (0x1 << 16) ;- (HCBDMA) Interrupt status for the Channel 16
AT91C_HCBDMA_CBDLISR17    EQU (0x1 << 17) ;- (HCBDMA) Interrupt status for the Channel 17
AT91C_HCBDMA_CBDLISR18    EQU (0x1 << 18) ;- (HCBDMA) Interrupt status for the Channel 18
AT91C_HCBDMA_CBDLISR19    EQU (0x1 << 19) ;- (HCBDMA) Interrupt status for the Channel 19
AT91C_HCBDMA_CBDLISR20    EQU (0x1 << 20) ;- (HCBDMA) Interrupt status for the Channel 20
AT91C_HCBDMA_CBDLISR21    EQU (0x1 << 21) ;- (HCBDMA) Interrupt status for the Channel 21
AT91C_HCBDMA_CBDLISR22    EQU (0x1 << 22) ;- (HCBDMA) Interrupt status for the Channel 22
AT91C_HCBDMA_CBDLISR23    EQU (0x1 << 23) ;- (HCBDMA) Interrupt status for the Channel 23
AT91C_HCBDMA_CBDLISR24    EQU (0x1 << 24) ;- (HCBDMA) Interrupt status for the Channel 24
AT91C_HCBDMA_CBDLISR25    EQU (0x1 << 25) ;- (HCBDMA) Interrupt status for the Channel 25
AT91C_HCBDMA_CBDLISR26    EQU (0x1 << 26) ;- (HCBDMA) Interrupt status for the Channel 26
AT91C_HCBDMA_CBDLISR27    EQU (0x1 << 27) ;- (HCBDMA) Interrupt status for the Channel 27
AT91C_HCBDMA_CBDLISR28    EQU (0x1 << 28) ;- (HCBDMA) Interrupt status for the Channel 28
AT91C_HCBDMA_CBDLISR29    EQU (0x1 << 29) ;- (HCBDMA) Interrupt status for the Channel 29
AT91C_HCBDMA_CBDLISR30    EQU (0x1 << 30) ;- (HCBDMA) Interrupt status for the Channel 30
AT91C_HCBDMA_CBDLISR31    EQU (0x1 << 31) ;- (HCBDMA) Interrupt status for the Channel 31
// - -------- HCBDMA_CBCRCCR : (HCBDMA Offset: 0x34) CB DMA CRC Control Resgister -------- 
AT91C_CRC_START           EQU (0x1 <<  0) ;- (HCBDMA) CRC compuration initialization
// - -------- HCBDMA_CBCRCMR : (HCBDMA Offset: 0x38) CB DMA CRC Mode Resgister -------- 
AT91C_CRC_ENABLE          EQU (0x1 <<  0) ;- (HCBDMA) CRC Enable
AT91C_CRC_COMPARE         EQU (0x1 <<  1) ;- (HCBDMA) CRC Compare
AT91C_CRC_PTYPE           EQU (0x3 <<  2) ;- (HCBDMA) Primitive polynomial type
AT91C_CRC_PTYPE_CCIT802_3 EQU (0x0 <<  2) ;- (HCBDMA) 
AT91C_CRC_PTYPE_CASTAGNOLI EQU (0x1 <<  2) ;- (HCBDMA) 
AT91C_CRC_PTYPE_CCIT_16   EQU (0x2 <<  2) ;- (HCBDMA) 
AT91C_CRC_DIVIDER         EQU (0xF <<  4) ;- (HCBDMA) Request Divider
AT91C_CRC_ID              EQU (0x1F <<  8) ;- (HCBDMA) CRC channel Identifier
AT91C_CRC_ID_CHANNEL_0    EQU (0x0 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_1    EQU (0x1 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_2    EQU (0x2 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_3    EQU (0x3 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_4    EQU (0x4 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_5    EQU (0x5 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_6    EQU (0x6 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_7    EQU (0x7 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_8    EQU (0x8 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_9    EQU (0x9 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_10   EQU (0xA <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_11   EQU (0xB <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_12   EQU (0xC <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_13   EQU (0xD <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_14   EQU (0xE <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_15   EQU (0xF <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_16   EQU (0x10 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_17   EQU (0x11 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_18   EQU (0x12 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_19   EQU (0x13 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_20   EQU (0x14 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_21   EQU (0x15 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_22   EQU (0x16 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_23   EQU (0x17 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_24   EQU (0x18 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_25   EQU (0x19 <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_26   EQU (0x1A <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_27   EQU (0x1B <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_28   EQU (0x1C <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_29   EQU (0x1D <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_30   EQU (0x1E <<  8) ;- (HCBDMA) 
AT91C_CRC_ID_CHANNEL_31   EQU (0x1F <<  8) ;- (HCBDMA) 
// - -------- HCBDMA_CBCRCSR : (HCBDMA Offset: 0x3c) CB DMA CRC Status Resgister -------- 
AT91C_HCBDMA_CRCSR        EQU (0x0 <<  0) ;- (HCBDMA) CRC Status Resgister
// - -------- HCBDMA_CBCRCIER : (HCBDMA Offset: 0x40) CB DMA CRC Interrupt Enable Resgister -------- 
AT91C_CRC_ERRIER          EQU (0x1 <<  0) ;- (HCBDMA) CRC Error Interrupt Enable
// - -------- HCBDMA_CBCRCIDR : (HCBDMA Offset: 0x44) CB DMA CRC Interrupt Enable Resgister -------- 
AT91C_CRC_ERRIDR          EQU (0x1 <<  0) ;- (HCBDMA) CRC Error Interrupt Disable
// - -------- HCBDMA_CBCRCIMR : (HCBDMA Offset: 0x48) CB DMA CRC Interrupt Mask Resgister -------- 
AT91C_CRC_ERRIMR          EQU (0x1 <<  0) ;- (HCBDMA) CRC Error Interrupt Mask
// - -------- HCBDMA_CBCRCISR : (HCBDMA Offset: 0x4c) CB DMA CRC Interrupt Status Resgister -------- 
AT91C_CRC_ERRISR          EQU (0x1 <<  0) ;- (HCBDMA) CRC Error Interrupt Status

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
// -              SOFTWARE API DEFINITION  FOR USB Device Interface
// - *****************************************************************************
// - -------- UDP_FRM_NUM : (UDP Offset: 0x0) USB Frame Number Register -------- 
AT91C_UDP_FRM_NUM         EQU (0x7FF <<  0) ;- (UDP) Frame Number as Defined in the Packet Field Formats
AT91C_UDP_FRM_ERR         EQU (0x1 << 16) ;- (UDP) Frame Error
AT91C_UDP_FRM_OK          EQU (0x1 << 17) ;- (UDP) Frame OK
// - -------- UDP_GLB_STATE : (UDP Offset: 0x4) USB Global State Register -------- 
AT91C_UDP_FADDEN          EQU (0x1 <<  0) ;- (UDP) Function Address Enable
AT91C_UDP_CONFG           EQU (0x1 <<  1) ;- (UDP) Configured
AT91C_UDP_ESR             EQU (0x1 <<  2) ;- (UDP) Enable Send Resume
AT91C_UDP_RSMINPR         EQU (0x1 <<  3) ;- (UDP) A Resume Has Been Sent to the Host
AT91C_UDP_RMWUPE          EQU (0x1 <<  4) ;- (UDP) Remote Wake Up Enable
// - -------- UDP_FADDR : (UDP Offset: 0x8) USB Function Address Register -------- 
AT91C_UDP_FADD            EQU (0xFF <<  0) ;- (UDP) Function Address Value
AT91C_UDP_FEN             EQU (0x1 <<  8) ;- (UDP) Function Enable
// - -------- UDP_IER : (UDP Offset: 0x10) USB Interrupt Enable Register -------- 
AT91C_UDP_EPINT0          EQU (0x1 <<  0) ;- (UDP) Endpoint 0 Interrupt
AT91C_UDP_EPINT1          EQU (0x1 <<  1) ;- (UDP) Endpoint 0 Interrupt
AT91C_UDP_EPINT2          EQU (0x1 <<  2) ;- (UDP) Endpoint 2 Interrupt
AT91C_UDP_EPINT3          EQU (0x1 <<  3) ;- (UDP) Endpoint 3 Interrupt
AT91C_UDP_EPINT4          EQU (0x1 <<  4) ;- (UDP) Endpoint 4 Interrupt
AT91C_UDP_EPINT5          EQU (0x1 <<  5) ;- (UDP) Endpoint 5 Interrupt
AT91C_UDP_EPINT6          EQU (0x1 <<  6) ;- (UDP) Endpoint 6 Interrupt
AT91C_UDP_EPINT7          EQU (0x1 <<  7) ;- (UDP) Endpoint 7 Interrupt
AT91C_UDP_RXSUSP          EQU (0x1 <<  8) ;- (UDP) USB Suspend Interrupt
AT91C_UDP_RXRSM           EQU (0x1 <<  9) ;- (UDP) USB Resume Interrupt
AT91C_UDP_EXTRSM          EQU (0x1 << 10) ;- (UDP) USB External Resume Interrupt
AT91C_UDP_SOFINT          EQU (0x1 << 11) ;- (UDP) USB Start Of frame Interrupt
AT91C_UDP_WAKEUP          EQU (0x1 << 13) ;- (UDP) USB Resume Interrupt
// - -------- UDP_IDR : (UDP Offset: 0x14) USB Interrupt Disable Register -------- 
// - -------- UDP_IMR : (UDP Offset: 0x18) USB Interrupt Mask Register -------- 
// - -------- UDP_ISR : (UDP Offset: 0x1c) USB Interrupt Status Register -------- 
AT91C_UDP_ENDBUSRES       EQU (0x1 << 12) ;- (UDP) USB End Of Bus Reset Interrupt
// - -------- UDP_ICR : (UDP Offset: 0x20) USB Interrupt Clear Register -------- 
// - -------- UDP_RST_EP : (UDP Offset: 0x28) USB Reset Endpoint Register -------- 
AT91C_UDP_EP0             EQU (0x1 <<  0) ;- (UDP) Reset Endpoint 0
AT91C_UDP_EP1             EQU (0x1 <<  1) ;- (UDP) Reset Endpoint 1
AT91C_UDP_EP2             EQU (0x1 <<  2) ;- (UDP) Reset Endpoint 2
AT91C_UDP_EP3             EQU (0x1 <<  3) ;- (UDP) Reset Endpoint 3
AT91C_UDP_EP4             EQU (0x1 <<  4) ;- (UDP) Reset Endpoint 4
AT91C_UDP_EP5             EQU (0x1 <<  5) ;- (UDP) Reset Endpoint 5
AT91C_UDP_EP6             EQU (0x1 <<  6) ;- (UDP) Reset Endpoint 6
AT91C_UDP_EP7             EQU (0x1 <<  7) ;- (UDP) Reset Endpoint 7
// - -------- UDP_CSR : (UDP Offset: 0x30) USB Endpoint Control and Status Register -------- 
AT91C_UDP_TXCOMP          EQU (0x1 <<  0) ;- (UDP) Generates an IN packet with data previously written in the DPR
AT91C_UDP_RX_DATA_BK0     EQU (0x1 <<  1) ;- (UDP) Receive Data Bank 0
AT91C_UDP_RXSETUP         EQU (0x1 <<  2) ;- (UDP) Sends STALL to the Host (Control endpoints)
AT91C_UDP_ISOERROR        EQU (0x1 <<  3) ;- (UDP) Isochronous error (Isochronous endpoints)
AT91C_UDP_STALLSENT       EQU (0x1 <<  3) ;- (UDP) Stall sent (Control, bulk, interrupt endpoints)
AT91C_UDP_TXPKTRDY        EQU (0x1 <<  4) ;- (UDP) Transmit Packet Ready
AT91C_UDP_FORCESTALL      EQU (0x1 <<  5) ;- (UDP) Force Stall (used by Control, Bulk and Isochronous endpoints).
AT91C_UDP_RX_DATA_BK1     EQU (0x1 <<  6) ;- (UDP) Receive Data Bank 1 (only used by endpoints with ping-pong attributes).
AT91C_UDP_DIR             EQU (0x1 <<  7) ;- (UDP) Transfer Direction
AT91C_UDP_EPTYPE          EQU (0x7 <<  8) ;- (UDP) Endpoint type
AT91C_UDP_EPTYPE_CTRL     EQU (0x0 <<  8) ;- (UDP) Control
AT91C_UDP_EPTYPE_ISO_OUT  EQU (0x1 <<  8) ;- (UDP) Isochronous OUT
AT91C_UDP_EPTYPE_BULK_OUT EQU (0x2 <<  8) ;- (UDP) Bulk OUT
AT91C_UDP_EPTYPE_INT_OUT  EQU (0x3 <<  8) ;- (UDP) Interrupt OUT
AT91C_UDP_EPTYPE_ISO_IN   EQU (0x5 <<  8) ;- (UDP) Isochronous IN
AT91C_UDP_EPTYPE_BULK_IN  EQU (0x6 <<  8) ;- (UDP) Bulk IN
AT91C_UDP_EPTYPE_INT_IN   EQU (0x7 <<  8) ;- (UDP) Interrupt IN
AT91C_UDP_DTGLE           EQU (0x1 << 11) ;- (UDP) Data Toggle
AT91C_UDP_EPEDS           EQU (0x1 << 15) ;- (UDP) Endpoint Enable Disable
AT91C_UDP_RXBYTECNT       EQU (0x7FF << 16) ;- (UDP) Number Of Bytes Available in the FIFO
// - -------- UDP_TXVC : (UDP Offset: 0x74) Transceiver Control Register -------- 
AT91C_UDP_TXVDIS          EQU (0x1 <<  8) ;- (UDP) 
AT91C_UDP_PUON            EQU (0x1 <<  9) ;- (UDP) Pull-up ON

// - *****************************************************************************
// -               REGISTER ADDRESS DEFINITION FOR AT91SAM3S1
// - *****************************************************************************
// - ========== Register definition for SYS peripheral ========== 
AT91C_SYS_GPBR            EQU (0x400E1490) ;- (SYS) General Purpose Register
// - ========== Register definition for SMC peripheral ========== 
AT91C_SMC_DELAY2          EQU (0x400E00C4) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE4          EQU (0x400E0048) ;- (SMC)  Cycle Register for CS 4
AT91C_SMC_CTRL5           EQU (0x400E005C) ;- (SMC)  Control Register for CS 5
AT91C_SMC_IPNAME2         EQU (0x400E00F4) ;- (SMC) HSMC3 IPNAME2 REGISTER 
AT91C_SMC_DELAY5          EQU (0x400E00D0) ;- (SMC) SMC Delay Control Register
AT91C_SMC_DELAY4          EQU (0x400E00CC) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE0          EQU (0x400E0008) ;- (SMC)  Cycle Register for CS 0
AT91C_SMC_PULSE1          EQU (0x400E0014) ;- (SMC)  Pulse Register for CS 1
AT91C_SMC_DELAY6          EQU (0x400E00D4) ;- (SMC) SMC Delay Control Register
AT91C_SMC_FEATURES        EQU (0x400E00F8) ;- (SMC) HSMC3 FEATURES REGISTER 
AT91C_SMC_DELAY3          EQU (0x400E00C8) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CTRL1           EQU (0x400E001C) ;- (SMC)  Control Register for CS 1
AT91C_SMC_PULSE7          EQU (0x400E0074) ;- (SMC)  Pulse Register for CS 7
AT91C_SMC_CTRL7           EQU (0x400E007C) ;- (SMC)  Control Register for CS 7
AT91C_SMC_VER             EQU (0x400E00FC) ;- (SMC) HSMC3 VERSION REGISTER
AT91C_SMC_SETUP5          EQU (0x400E0050) ;- (SMC)  Setup Register for CS 5
AT91C_SMC_CYCLE3          EQU (0x400E0038) ;- (SMC)  Cycle Register for CS 3
AT91C_SMC_SETUP3          EQU (0x400E0030) ;- (SMC)  Setup Register for CS 3
AT91C_SMC_DELAY1          EQU (0x400E00C0) ;- (SMC) SMC Delay Control Register
AT91C_SMC_ADDRSIZE        EQU (0x400E00EC) ;- (SMC) HSMC3 ADDRSIZE REGISTER 
AT91C_SMC_PULSE3          EQU (0x400E0034) ;- (SMC)  Pulse Register for CS 3
AT91C_SMC_PULSE5          EQU (0x400E0054) ;- (SMC)  Pulse Register for CS 5
AT91C_SMC_PULSE4          EQU (0x400E0044) ;- (SMC)  Pulse Register for CS 4
AT91C_SMC_SETUP2          EQU (0x400E0020) ;- (SMC)  Setup Register for CS 2
AT91C_SMC_DELAY8          EQU (0x400E00DC) ;- (SMC) SMC Delay Control Register
AT91C_SMC_CYCLE7          EQU (0x400E0078) ;- (SMC)  Cycle Register for CS 7
AT91C_SMC_CTRL0           EQU (0x400E000C) ;- (SMC)  Control Register for CS 0
AT91C_SMC_CYCLE2          EQU (0x400E0028) ;- (SMC)  Cycle Register for CS 2
AT91C_SMC_IPNAME1         EQU (0x400E00F0) ;- (SMC) HSMC3 IPNAME1 REGISTER 
AT91C_SMC_SETUP1          EQU (0x400E0010) ;- (SMC)  Setup Register for CS 1
AT91C_SMC_CTRL2           EQU (0x400E002C) ;- (SMC)  Control Register for CS 2
AT91C_SMC_CTRL4           EQU (0x400E004C) ;- (SMC)  Control Register for CS 4
AT91C_SMC_SETUP6          EQU (0x400E0060) ;- (SMC)  Setup Register for CS 6
AT91C_SMC_CYCLE5          EQU (0x400E0058) ;- (SMC)  Cycle Register for CS 5
AT91C_SMC_CTRL6           EQU (0x400E006C) ;- (SMC)  Control Register for CS 6
AT91C_SMC_SETUP4          EQU (0x400E0040) ;- (SMC)  Setup Register for CS 4
AT91C_SMC_PULSE2          EQU (0x400E0024) ;- (SMC)  Pulse Register for CS 2
AT91C_SMC_DELAY7          EQU (0x400E00D8) ;- (SMC) SMC Delay Control Register
AT91C_SMC_SETUP7          EQU (0x400E0070) ;- (SMC)  Setup Register for CS 7
AT91C_SMC_CYCLE1          EQU (0x400E0018) ;- (SMC)  Cycle Register for CS 1
AT91C_SMC_CTRL3           EQU (0x400E003C) ;- (SMC)  Control Register for CS 3
AT91C_SMC_SETUP0          EQU (0x400E0000) ;- (SMC)  Setup Register for CS 0
AT91C_SMC_PULSE0          EQU (0x400E0004) ;- (SMC)  Pulse Register for CS 0
AT91C_SMC_PULSE6          EQU (0x400E0064) ;- (SMC)  Pulse Register for CS 6
AT91C_SMC_CYCLE6          EQU (0x400E0068) ;- (SMC)  Cycle Register for CS 6
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
// - ========== Register definition for CCFG peripheral ========== 
AT91C_CCFG_FLASH0         EQU (0x400E0318) ;- (CCFG)  FLASH0 configuration
AT91C_CCFG_RAM0           EQU (0x400E0310) ;- (CCFG)  RAM0 configuration
AT91C_CCFG_ROM            EQU (0x400E0314) ;- (CCFG)  ROM  configuration
AT91C_CCFG_EBICSA         EQU (0x400E031C) ;- (CCFG)  EBI Chip Select Assignement Register
AT91C_CCFG_BRIDGE         EQU (0x400E0320) ;- (CCFG)  BRIDGE configuration
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
// - ========== Register definition for PDC_DBGU0 peripheral ========== 
AT91C_DBGU0_TPR           EQU (0x400E0708) ;- (PDC_DBGU0) Transmit Pointer Register
AT91C_DBGU0_PTCR          EQU (0x400E0720) ;- (PDC_DBGU0) PDC Transfer Control Register
AT91C_DBGU0_TNCR          EQU (0x400E071C) ;- (PDC_DBGU0) Transmit Next Counter Register
AT91C_DBGU0_PTSR          EQU (0x400E0724) ;- (PDC_DBGU0) PDC Transfer Status Register
AT91C_DBGU0_RNCR          EQU (0x400E0714) ;- (PDC_DBGU0) Receive Next Counter Register
AT91C_DBGU0_RPR           EQU (0x400E0700) ;- (PDC_DBGU0) Receive Pointer Register
AT91C_DBGU0_TCR           EQU (0x400E070C) ;- (PDC_DBGU0) Transmit Counter Register
AT91C_DBGU0_RNPR          EQU (0x400E0710) ;- (PDC_DBGU0) Receive Next Pointer Register
AT91C_DBGU0_TNPR          EQU (0x400E0718) ;- (PDC_DBGU0) Transmit Next Pointer Register
AT91C_DBGU0_RCR           EQU (0x400E0704) ;- (PDC_DBGU0) Receive Counter Register
// - ========== Register definition for DBGU0 peripheral ========== 
AT91C_DBGU0_CR            EQU (0x400E0600) ;- (DBGU0) Control Register
AT91C_DBGU0_IDR           EQU (0x400E060C) ;- (DBGU0) Interrupt Disable Register
AT91C_DBGU0_CIDR          EQU (0x400E0740) ;- (DBGU0) Chip ID Register
AT91C_DBGU0_IPNAME2       EQU (0x400E06F4) ;- (DBGU0) DBGU IPNAME2 REGISTER 
AT91C_DBGU0_FEATURES      EQU (0x400E06F8) ;- (DBGU0) DBGU FEATURES REGISTER 
AT91C_DBGU0_FNTR          EQU (0x400E0648) ;- (DBGU0) Force NTRST Register
AT91C_DBGU0_RHR           EQU (0x400E0618) ;- (DBGU0) Receiver Holding Register
AT91C_DBGU0_THR           EQU (0x400E061C) ;- (DBGU0) Transmitter Holding Register
AT91C_DBGU0_ADDRSIZE      EQU (0x400E06EC) ;- (DBGU0) DBGU ADDRSIZE REGISTER 
AT91C_DBGU0_MR            EQU (0x400E0604) ;- (DBGU0) Mode Register
AT91C_DBGU0_IER           EQU (0x400E0608) ;- (DBGU0) Interrupt Enable Register
AT91C_DBGU0_BRGR          EQU (0x400E0620) ;- (DBGU0) Baud Rate Generator Register
AT91C_DBGU0_CSR           EQU (0x400E0614) ;- (DBGU0) Channel Status Register
AT91C_DBGU0_VER           EQU (0x400E06FC) ;- (DBGU0) DBGU VERSION REGISTER 
AT91C_DBGU0_IMR           EQU (0x400E0610) ;- (DBGU0) Interrupt Mask Register
AT91C_DBGU0_IPNAME1       EQU (0x400E06F0) ;- (DBGU0) DBGU IPNAME1 REGISTER 
AT91C_DBGU0_EXID          EQU (0x400E0744) ;- (DBGU0) Chip ID Extension Register
// - ========== Register definition for PDC_DBGU1 peripheral ========== 
AT91C_DBGU1_RNCR          EQU (0x400E0914) ;- (PDC_DBGU1) Receive Next Counter Register
AT91C_DBGU1_RPR           EQU (0x400E0900) ;- (PDC_DBGU1) Receive Pointer Register
AT91C_DBGU1_TNCR          EQU (0x400E091C) ;- (PDC_DBGU1) Transmit Next Counter Register
AT91C_DBGU1_TNPR          EQU (0x400E0918) ;- (PDC_DBGU1) Transmit Next Pointer Register
AT91C_DBGU1_PTSR          EQU (0x400E0924) ;- (PDC_DBGU1) PDC Transfer Status Register
AT91C_DBGU1_PTCR          EQU (0x400E0920) ;- (PDC_DBGU1) PDC Transfer Control Register
AT91C_DBGU1_RCR           EQU (0x400E0904) ;- (PDC_DBGU1) Receive Counter Register
AT91C_DBGU1_RNPR          EQU (0x400E0910) ;- (PDC_DBGU1) Receive Next Pointer Register
AT91C_DBGU1_TPR           EQU (0x400E0908) ;- (PDC_DBGU1) Transmit Pointer Register
AT91C_DBGU1_TCR           EQU (0x400E090C) ;- (PDC_DBGU1) Transmit Counter Register
// - ========== Register definition for DBGU1 peripheral ========== 
AT91C_DBGU1_RHR           EQU (0x400E0818) ;- (DBGU1) Receiver Holding Register
AT91C_DBGU1_IPNAME1       EQU (0x400E08F0) ;- (DBGU1) DBGU IPNAME1 REGISTER 
AT91C_DBGU1_CIDR          EQU (0x400E0940) ;- (DBGU1) Chip ID Register
AT91C_DBGU1_CR            EQU (0x400E0800) ;- (DBGU1) Control Register
AT91C_DBGU1_VER           EQU (0x400E08FC) ;- (DBGU1) DBGU VERSION REGISTER 
AT91C_DBGU1_IPNAME2       EQU (0x400E08F4) ;- (DBGU1) DBGU IPNAME2 REGISTER 
AT91C_DBGU1_BRGR          EQU (0x400E0820) ;- (DBGU1) Baud Rate Generator Register
AT91C_DBGU1_FNTR          EQU (0x400E0848) ;- (DBGU1) Force NTRST Register
AT91C_DBGU1_MR            EQU (0x400E0804) ;- (DBGU1) Mode Register
AT91C_DBGU1_ADDRSIZE      EQU (0x400E08EC) ;- (DBGU1) DBGU ADDRSIZE REGISTER 
AT91C_DBGU1_CSR           EQU (0x400E0814) ;- (DBGU1) Channel Status Register
AT91C_DBGU1_IMR           EQU (0x400E0810) ;- (DBGU1) Interrupt Mask Register
AT91C_DBGU1_EXID          EQU (0x400E0944) ;- (DBGU1) Chip ID Extension Register
AT91C_DBGU1_IDR           EQU (0x400E080C) ;- (DBGU1) Interrupt Disable Register
AT91C_DBGU1_FEATURES      EQU (0x400E08F8) ;- (DBGU1) DBGU FEATURES REGISTER 
AT91C_DBGU1_IER           EQU (0x400E0808) ;- (DBGU1) Interrupt Enable Register
AT91C_DBGU1_THR           EQU (0x400E081C) ;- (DBGU1) Transmitter Holding Register
// - ========== Register definition for PIOA peripheral ========== 
AT91C_PIOA_SENIDR         EQU (0x400E0F58) ;- (PIOA) Sensor Interrupt Disable Register
AT91C_PIOA_OWSR           EQU (0x400E0EA8) ;- (PIOA) Output Write Status Register
AT91C_PIOA_PSR            EQU (0x400E0E08) ;- (PIOA) PIO Status Register
AT91C_PIOA_MDER           EQU (0x400E0E50) ;- (PIOA) Multi-driver Enable Register
AT91C_PIOA_IPNAME1        EQU (0x400E0EF0) ;- (PIOA) PIO IPNAME1 REGISTER 
AT91C_PIOA_FEATURES       EQU (0x400E0EF8) ;- (PIOA) PIO FEATURES REGISTER 
AT91C_PIOA_SP2            EQU (0x400E0E74) ;- (PIOA) Select B Register
AT91C_PIOA_ODR            EQU (0x400E0E14) ;- (PIOA) Output Disable Registerr
AT91C_PIOA_IDR            EQU (0x400E0E44) ;- (PIOA) Interrupt Disable Register
AT91C_PIOA_PDR            EQU (0x400E0E04) ;- (PIOA) PIO Disable Register
AT91C_PIOA_PDSR           EQU (0x400E0E3C) ;- (PIOA) Pin Data Status Register
AT91C_PIOA_PPDER          EQU (0x400E0E94) ;- (PIOA) Pull-down Enable Register
AT91C_PIOA_SENIER         EQU (0x400E0F54) ;- (PIOA) Sensor Interrupt Enable Register
AT91C_PIOA_SLEW2          EQU (0x400E0F04) ;- (PIOA) PIO SLEWRATE2 REGISTER 
AT91C_PIOA_SENMR          EQU (0x400E0F50) ;- (PIOA) Sensor Mode Register
AT91C_PIOA_PPUDR          EQU (0x400E0E60) ;- (PIOA) Pull-up Disable Register
AT91C_PIOA_OWDR           EQU (0x400E0EA4) ;- (PIOA) Output Write Disable Register
AT91C_PIOA_ADDRSIZE       EQU (0x400E0EEC) ;- (PIOA) PIO ADDRSIZE REGISTER 
AT91C_PIOA_IFER           EQU (0x400E0E20) ;- (PIOA) Input Filter Enable Register
AT91C_PIOA_PPDSR          EQU (0x400E0E98) ;- (PIOA) Pull-down Status Register
AT91C_PIOA_SP1            EQU (0x400E0E70) ;- (PIOA) Select B Register
AT91C_PIOA_SENIMR         EQU (0x400E0F5C) ;- (PIOA) Sensor Interrupt Mask Register
AT91C_PIOA_SENDATA        EQU (0x400E0F64) ;- (PIOA) Sensor Data Register
AT91C_PIOA_VER            EQU (0x400E0EFC) ;- (PIOA) PIO VERSION REGISTER 
AT91C_PIOA_PER            EQU (0x400E0E00) ;- (PIOA) PIO Enable Register
AT91C_PIOA_IMR            EQU (0x400E0E48) ;- (PIOA) Interrupt Mask Register
AT91C_PIOA_PPUSR          EQU (0x400E0E68) ;- (PIOA) Pull-up Status Register
AT91C_PIOA_ODSR           EQU (0x400E0E38) ;- (PIOA) Output Data Status Register
AT91C_PIOA_SENISR         EQU (0x400E0F60) ;- (PIOA) Sensor Interrupt Status Register
AT91C_PIOA_IFDR           EQU (0x400E0E24) ;- (PIOA) Input Filter Disable Register
AT91C_PIOA_SODR           EQU (0x400E0E30) ;- (PIOA) Set Output Data Register
AT91C_PIOA_SLEW1          EQU (0x400E0F00) ;- (PIOA) PIO SLEWRATE1 REGISTER 
AT91C_PIOA_IER            EQU (0x400E0E40) ;- (PIOA) Interrupt Enable Register
AT91C_PIOA_MDSR           EQU (0x400E0E58) ;- (PIOA) Multi-driver Status Register
AT91C_PIOA_ISR            EQU (0x400E0E4C) ;- (PIOA) Interrupt Status Register
AT91C_PIOA_IFSR           EQU (0x400E0E28) ;- (PIOA) Input Filter Status Register
AT91C_PIOA_PPDDR          EQU (0x400E0E90) ;- (PIOA) Pull-down Disable Register
AT91C_PIOA_PPUER          EQU (0x400E0E64) ;- (PIOA) Pull-up Enable Register
AT91C_PIOA_OWER           EQU (0x400E0EA0) ;- (PIOA) Output Write Enable Register
AT91C_PIOA_IPNAME2        EQU (0x400E0EF4) ;- (PIOA) PIO IPNAME2 REGISTER 
AT91C_PIOA_MDDR           EQU (0x400E0E54) ;- (PIOA) Multi-driver Disable Register
AT91C_PIOA_CODR           EQU (0x400E0E34) ;- (PIOA) Clear Output Data Register
AT91C_PIOA_OER            EQU (0x400E0E10) ;- (PIOA) Output Enable Register
AT91C_PIOA_OSR            EQU (0x400E0E18) ;- (PIOA) Output Status Register
AT91C_PIOA_ABSR           EQU (0x400E0E78) ;- (PIOA) AB Select Status Register
// - ========== Register definition for PDC_PIOA peripheral ========== 
AT91C_PIOA_RPR            EQU (0x400E0F68) ;- (PDC_PIOA) Receive Pointer Register
AT91C_PIOA_TPR            EQU (0x400E0F70) ;- (PDC_PIOA) Transmit Pointer Register
AT91C_PIOA_RCR            EQU (0x400E0F6C) ;- (PDC_PIOA) Receive Counter Register
AT91C_PIOA_PTSR           EQU (0x400E0F8C) ;- (PDC_PIOA) PDC Transfer Status Register
AT91C_PIOA_TCR            EQU (0x400E0F74) ;- (PDC_PIOA) Transmit Counter Register
AT91C_PIOA_PTCR           EQU (0x400E0F88) ;- (PDC_PIOA) PDC Transfer Control Register
AT91C_PIOA_RNPR           EQU (0x400E0F78) ;- (PDC_PIOA) Receive Next Pointer Register
AT91C_PIOA_TNCR           EQU (0x400E0F84) ;- (PDC_PIOA) Transmit Next Counter Register
AT91C_PIOA_RNCR           EQU (0x400E0F7C) ;- (PDC_PIOA) Receive Next Counter Register
AT91C_PIOA_TNPR           EQU (0x400E0F80) ;- (PDC_PIOA) Transmit Next Pointer Register
// - ========== Register definition for PIOB peripheral ========== 
AT91C_PIOB_MDDR           EQU (0x400E1054) ;- (PIOB) Multi-driver Disable Register
AT91C_PIOB_ABSR           EQU (0x400E1078) ;- (PIOB) AB Select Status Register
AT91C_PIOB_SP1            EQU (0x400E1070) ;- (PIOB) Select B Register
AT91C_PIOB_ODSR           EQU (0x400E1038) ;- (PIOB) Output Data Status Register
AT91C_PIOB_SLEW1          EQU (0x400E1100) ;- (PIOB) PIO SLEWRATE1 REGISTER 
AT91C_PIOB_SENISR         EQU (0x400E1160) ;- (PIOB) Sensor Interrupt Status Register
AT91C_PIOB_OSR            EQU (0x400E1018) ;- (PIOB) Output Status Register
AT91C_PIOB_IFER           EQU (0x400E1020) ;- (PIOB) Input Filter Enable Register
AT91C_PIOB_SENDATA        EQU (0x400E1164) ;- (PIOB) Sensor Data Register
AT91C_PIOB_MDSR           EQU (0x400E1058) ;- (PIOB) Multi-driver Status Register
AT91C_PIOB_IFDR           EQU (0x400E1024) ;- (PIOB) Input Filter Disable Register
AT91C_PIOB_MDER           EQU (0x400E1050) ;- (PIOB) Multi-driver Enable Register
AT91C_PIOB_SENIDR         EQU (0x400E1158) ;- (PIOB) Sensor Interrupt Disable Register
AT91C_PIOB_IER            EQU (0x400E1040) ;- (PIOB) Interrupt Enable Register
AT91C_PIOB_OWDR           EQU (0x400E10A4) ;- (PIOB) Output Write Disable Register
AT91C_PIOB_IFSR           EQU (0x400E1028) ;- (PIOB) Input Filter Status Register
AT91C_PIOB_ISR            EQU (0x400E104C) ;- (PIOB) Interrupt Status Register
AT91C_PIOB_PPUDR          EQU (0x400E1060) ;- (PIOB) Pull-up Disable Register
AT91C_PIOB_PDSR           EQU (0x400E103C) ;- (PIOB) Pin Data Status Register
AT91C_PIOB_IPNAME2        EQU (0x400E10F4) ;- (PIOB) PIO IPNAME2 REGISTER 
AT91C_PIOB_PPUER          EQU (0x400E1064) ;- (PIOB) Pull-up Enable Register
AT91C_PIOB_SLEW2          EQU (0x400E1104) ;- (PIOB) PIO SLEWRATE2 REGISTER 
AT91C_PIOB_OER            EQU (0x400E1010) ;- (PIOB) Output Enable Register
AT91C_PIOB_CODR           EQU (0x400E1034) ;- (PIOB) Clear Output Data Register
AT91C_PIOB_PPDDR          EQU (0x400E1090) ;- (PIOB) Pull-down Disable Register
AT91C_PIOB_OWER           EQU (0x400E10A0) ;- (PIOB) Output Write Enable Register
AT91C_PIOB_VER            EQU (0x400E10FC) ;- (PIOB) PIO VERSION REGISTER 
AT91C_PIOB_PPDER          EQU (0x400E1094) ;- (PIOB) Pull-down Enable Register
AT91C_PIOB_IMR            EQU (0x400E1048) ;- (PIOB) Interrupt Mask Register
AT91C_PIOB_PPUSR          EQU (0x400E1068) ;- (PIOB) Pull-up Status Register
AT91C_PIOB_IPNAME1        EQU (0x400E10F0) ;- (PIOB) PIO IPNAME1 REGISTER 
AT91C_PIOB_ADDRSIZE       EQU (0x400E10EC) ;- (PIOB) PIO ADDRSIZE REGISTER 
AT91C_PIOB_SP2            EQU (0x400E1074) ;- (PIOB) Select B Register
AT91C_PIOB_IDR            EQU (0x400E1044) ;- (PIOB) Interrupt Disable Register
AT91C_PIOB_SENMR          EQU (0x400E1150) ;- (PIOB) Sensor Mode Register
AT91C_PIOB_SODR           EQU (0x400E1030) ;- (PIOB) Set Output Data Register
AT91C_PIOB_PPDSR          EQU (0x400E1098) ;- (PIOB) Pull-down Status Register
AT91C_PIOB_PSR            EQU (0x400E1008) ;- (PIOB) PIO Status Register
AT91C_PIOB_ODR            EQU (0x400E1014) ;- (PIOB) Output Disable Registerr
AT91C_PIOB_OWSR           EQU (0x400E10A8) ;- (PIOB) Output Write Status Register
AT91C_PIOB_FEATURES       EQU (0x400E10F8) ;- (PIOB) PIO FEATURES REGISTER 
AT91C_PIOB_PDR            EQU (0x400E1004) ;- (PIOB) PIO Disable Register
AT91C_PIOB_SENIMR         EQU (0x400E115C) ;- (PIOB) Sensor Interrupt Mask Register
AT91C_PIOB_SENIER         EQU (0x400E1154) ;- (PIOB) Sensor Interrupt Enable Register
AT91C_PIOB_PER            EQU (0x400E1000) ;- (PIOB) PIO Enable Register
// - ========== Register definition for PIOC peripheral ========== 
AT91C_PIOC_VER            EQU (0x400E12FC) ;- (PIOC) PIO VERSION REGISTER 
AT91C_PIOC_IMR            EQU (0x400E1248) ;- (PIOC) Interrupt Mask Register
AT91C_PIOC_PSR            EQU (0x400E1208) ;- (PIOC) PIO Status Register
AT91C_PIOC_PPDSR          EQU (0x400E1298) ;- (PIOC) Pull-down Status Register
AT91C_PIOC_OER            EQU (0x400E1210) ;- (PIOC) Output Enable Register
AT91C_PIOC_OSR            EQU (0x400E1218) ;- (PIOC) Output Status Register
AT91C_PIOC_MDDR           EQU (0x400E1254) ;- (PIOC) Multi-driver Disable Register
AT91C_PIOC_PPUSR          EQU (0x400E1268) ;- (PIOC) Pull-up Status Register
AT91C_PIOC_ODSR           EQU (0x400E1238) ;- (PIOC) Output Data Status Register
AT91C_PIOC_SLEW2          EQU (0x400E1304) ;- (PIOC) PIO SLEWRATE2 REGISTER 
AT91C_PIOC_SENMR          EQU (0x400E1350) ;- (PIOC) Sensor Mode Register
AT91C_PIOC_IFER           EQU (0x400E1220) ;- (PIOC) Input Filter Enable Register
AT91C_PIOC_PDR            EQU (0x400E1204) ;- (PIOC) PIO Disable Register
AT91C_PIOC_MDER           EQU (0x400E1250) ;- (PIOC) Multi-driver Enable Register
AT91C_PIOC_SP2            EQU (0x400E1274) ;- (PIOC) Select B Register
AT91C_PIOC_IPNAME1        EQU (0x400E12F0) ;- (PIOC) PIO IPNAME1 REGISTER 
AT91C_PIOC_IER            EQU (0x400E1240) ;- (PIOC) Interrupt Enable Register
AT91C_PIOC_OWDR           EQU (0x400E12A4) ;- (PIOC) Output Write Disable Register
AT91C_PIOC_IDR            EQU (0x400E1244) ;- (PIOC) Interrupt Disable Register
AT91C_PIOC_PDSR           EQU (0x400E123C) ;- (PIOC) Pin Data Status Register
AT91C_PIOC_SENIDR         EQU (0x400E1358) ;- (PIOC) Sensor Interrupt Disable Register
AT91C_PIOC_SENISR         EQU (0x400E1360) ;- (PIOC) Sensor Interrupt Status Register
AT91C_PIOC_PER            EQU (0x400E1200) ;- (PIOC) PIO Enable Register
AT91C_PIOC_SENDATA        EQU (0x400E1364) ;- (PIOC) Sensor Data Register
AT91C_PIOC_IPNAME2        EQU (0x400E12F4) ;- (PIOC) PIO IPNAME2 REGISTER 
AT91C_PIOC_PPDDR          EQU (0x400E1290) ;- (PIOC) Pull-down Disable Register
AT91C_PIOC_ADDRSIZE       EQU (0x400E12EC) ;- (PIOC) PIO ADDRSIZE REGISTER 
AT91C_PIOC_IFDR           EQU (0x400E1224) ;- (PIOC) Input Filter Disable Register
AT91C_PIOC_ODR            EQU (0x400E1214) ;- (PIOC) Output Disable Registerr
AT91C_PIOC_CODR           EQU (0x400E1234) ;- (PIOC) Clear Output Data Register
AT91C_PIOC_MDSR           EQU (0x400E1258) ;- (PIOC) Multi-driver Status Register
AT91C_PIOC_FEATURES       EQU (0x400E12F8) ;- (PIOC) PIO FEATURES REGISTER 
AT91C_PIOC_IFSR           EQU (0x400E1228) ;- (PIOC) Input Filter Status Register
AT91C_PIOC_PPUER          EQU (0x400E1264) ;- (PIOC) Pull-up Enable Register
AT91C_PIOC_PPDER          EQU (0x400E1294) ;- (PIOC) Pull-down Enable Register
AT91C_PIOC_OWSR           EQU (0x400E12A8) ;- (PIOC) Output Write Status Register
AT91C_PIOC_ISR            EQU (0x400E124C) ;- (PIOC) Interrupt Status Register
AT91C_PIOC_OWER           EQU (0x400E12A0) ;- (PIOC) Output Write Enable Register
AT91C_PIOC_PPUDR          EQU (0x400E1260) ;- (PIOC) Pull-up Disable Register
AT91C_PIOC_SENIMR         EQU (0x400E135C) ;- (PIOC) Sensor Interrupt Mask Register
AT91C_PIOC_SLEW1          EQU (0x400E1300) ;- (PIOC) PIO SLEWRATE1 REGISTER 
AT91C_PIOC_SENIER         EQU (0x400E1354) ;- (PIOC) Sensor Interrupt Enable Register
AT91C_PIOC_SODR           EQU (0x400E1230) ;- (PIOC) Set Output Data Register
AT91C_PIOC_SP1            EQU (0x400E1270) ;- (PIOC) Select B Register
AT91C_PIOC_ABSR           EQU (0x400E1278) ;- (PIOC) AB Select Status Register
// - ========== Register definition for PMC peripheral ========== 
AT91C_PMC_PLLAR           EQU (0x400E0428) ;- (PMC) PLL Register
AT91C_PMC_UCKR            EQU (0x400E041C) ;- (PMC) UTMI Clock Configuration Register
AT91C_PMC_FSMR            EQU (0x400E0470) ;- (PMC) Fast Startup Mode Register
AT91C_PMC_MCKR            EQU (0x400E0430) ;- (PMC) Master Clock Register
AT91C_PMC_SCER            EQU (0x400E0400) ;- (PMC) System Clock Enable Register
AT91C_PMC_PCSR            EQU (0x400E0418) ;- (PMC) Peripheral Clock Status Register 0:31 PERI_ID
AT91C_PMC_MCFR            EQU (0x400E0424) ;- (PMC) Main Clock  Frequency Register
AT91C_PMC_PCER1           EQU (0x400E0500) ;- (PMC) Peripheral Clock Enable Register 32:63 PERI_ID
AT91C_PMC_FOCR            EQU (0x400E0478) ;- (PMC) Fault Output Clear Register
AT91C_PMC_PCSR1           EQU (0x400E0508) ;- (PMC) Peripheral Clock Status Register 32:63 PERI_ID
AT91C_PMC_FSPR            EQU (0x400E0474) ;- (PMC) Fast Startup Polarity Register
AT91C_PMC_SCSR            EQU (0x400E0408) ;- (PMC) System Clock Status Register
AT91C_PMC_IDR             EQU (0x400E0464) ;- (PMC) Interrupt Disable Register
AT91C_PMC_UDPR            EQU (0x400E0438) ;- (PMC) USB DEV Clock Configuration Register
AT91C_PMC_PCDR1           EQU (0x400E0504) ;- (PMC) Peripheral Clock Disable Register 32:63 PERI_ID
AT91C_PMC_VER             EQU (0x400E04FC) ;- (PMC) APMC VERSION REGISTER
AT91C_PMC_IMR             EQU (0x400E046C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IPNAME2         EQU (0x400E04F4) ;- (PMC) PMC IPNAME2 REGISTER 
AT91C_PMC_SCDR            EQU (0x400E0404) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCKR            EQU (0x400E0440) ;- (PMC) Programmable Clock Register
AT91C_PMC_ADDRSIZE        EQU (0x400E04EC) ;- (PMC) PMC ADDRSIZE REGISTER 
AT91C_PMC_PCDR            EQU (0x400E0414) ;- (PMC) Peripheral Clock Disable Register 0:31 PERI_ID
AT91C_PMC_MOR             EQU (0x400E0420) ;- (PMC) Main Oscillator Register
AT91C_PMC_SR              EQU (0x400E0468) ;- (PMC) Status Register
AT91C_PMC_IER             EQU (0x400E0460) ;- (PMC) Interrupt Enable Register
AT91C_PMC_PLLBR           EQU (0x400E042C) ;- (PMC) PLL B Register
AT91C_PMC_IPNAME1         EQU (0x400E04F0) ;- (PMC) PMC IPNAME1 REGISTER 
AT91C_PMC_PCER            EQU (0x400E0410) ;- (PMC) Peripheral Clock Enable Register 0:31 PERI_ID
AT91C_PMC_FEATURES        EQU (0x400E04F8) ;- (PMC) PMC FEATURES REGISTER 
AT91C_PMC_PCR             EQU (0x400E050C) ;- (PMC) Peripheral Control Register
// - ========== Register definition for CKGR peripheral ========== 
AT91C_CKGR_PLLAR          EQU (0x400E0428) ;- (CKGR) PLL Register
AT91C_CKGR_UCKR           EQU (0x400E041C) ;- (CKGR) UTMI Clock Configuration Register
AT91C_CKGR_MOR            EQU (0x400E0420) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_MCFR           EQU (0x400E0424) ;- (CKGR) Main Clock  Frequency Register
AT91C_CKGR_PLLBR          EQU (0x400E042C) ;- (CKGR) PLL B Register
// - ========== Register definition for RSTC peripheral ========== 
AT91C_RSTC_RSR            EQU (0x400E1404) ;- (RSTC) Reset Status Register
AT91C_RSTC_RCR            EQU (0x400E1400) ;- (RSTC) Reset Control Register
AT91C_RSTC_RMR            EQU (0x400E1408) ;- (RSTC) Reset Mode Register
AT91C_RSTC_VER            EQU (0x400E14FC) ;- (RSTC) Version Register
// - ========== Register definition for SUPC peripheral ========== 
AT91C_SUPC_FWUTR          EQU (0x400E1428) ;- (SUPC) Flash Wake-up Timer Register
AT91C_SUPC_SR             EQU (0x400E1424) ;- (SUPC) Status Register
AT91C_SUPC_BOMR           EQU (0x400E1414) ;- (SUPC) Brown Out Mode Register
AT91C_SUPC_WUMR           EQU (0x400E141C) ;- (SUPC) Wake Up Mode Register
AT91C_SUPC_WUIR           EQU (0x400E1420) ;- (SUPC) Wake Up Inputs Register
AT91C_SUPC_CR             EQU (0x400E1410) ;- (SUPC) Control Register
AT91C_SUPC_MR             EQU (0x400E1418) ;- (SUPC) Mode Register
// - ========== Register definition for RTTC peripheral ========== 
AT91C_RTTC_RTSR           EQU (0x400E143C) ;- (RTTC) Real-time Status Register
AT91C_RTTC_RTVR           EQU (0x400E1438) ;- (RTTC) Real-time Value Register
AT91C_RTTC_RTMR           EQU (0x400E1430) ;- (RTTC) Real-time Mode Register
AT91C_RTTC_RTAR           EQU (0x400E1434) ;- (RTTC) Real-time Alarm Register
// - ========== Register definition for WDTC peripheral ========== 
AT91C_WDTC_WDCR           EQU (0x400E1450) ;- (WDTC) Watchdog Control Register
AT91C_WDTC_WDMR           EQU (0x400E1454) ;- (WDTC) Watchdog Mode Register
AT91C_WDTC_WDSR           EQU (0x400E1458) ;- (WDTC) Watchdog Status Register
// - ========== Register definition for RTC peripheral ========== 
AT91C_RTC_VER             EQU (0x400E148C) ;- (RTC) Valid Entry Register
AT91C_RTC_TIMR            EQU (0x400E1468) ;- (RTC) Time Register
AT91C_RTC_CALALR          EQU (0x400E1474) ;- (RTC) Calendar Alarm Register
AT91C_RTC_IER             EQU (0x400E1480) ;- (RTC) Interrupt Enable Register
AT91C_RTC_MR              EQU (0x400E1464) ;- (RTC) Mode Register
AT91C_RTC_CALR            EQU (0x400E146C) ;- (RTC) Calendar Register
AT91C_RTC_TIMALR          EQU (0x400E1470) ;- (RTC) Time Alarm Register
AT91C_RTC_SCCR            EQU (0x400E147C) ;- (RTC) Status Clear Command Register
AT91C_RTC_CR              EQU (0x400E1460) ;- (RTC) Control Register
AT91C_RTC_IDR             EQU (0x400E1484) ;- (RTC) Interrupt Disable Register
AT91C_RTC_IMR             EQU (0x400E1488) ;- (RTC) Interrupt Mask Register
AT91C_RTC_SR              EQU (0x400E1478) ;- (RTC) Status Register
// - ========== Register definition for ADC0 peripheral ========== 
AT91C_ADC0_CDR2           EQU (0x40038058) ;- (ADC0) ADC Channel Data Register 2
AT91C_ADC0_CGR            EQU (0x40038048) ;- (ADC0) Control gain register
AT91C_ADC0_CDR7           EQU (0x4003806C) ;- (ADC0) ADC Channel Data Register 7
AT91C_ADC0_IDR            EQU (0x40038028) ;- (ADC0) ADC Interrupt Disable Register
AT91C_ADC0_CR             EQU (0x40038000) ;- (ADC0) ADC Control Register
AT91C_ADC0_FEATURES       EQU (0x400380F8) ;- (ADC0) ADC FEATURES REGISTER 
AT91C_ADC0_CWR            EQU (0x40038040) ;- (ADC0) unspecified
AT91C_ADC0_IPNAME1        EQU (0x400380F0) ;- (ADC0) ADC IPNAME1 REGISTER 
AT91C_ADC0_CDR9           EQU (0x40038074) ;- (ADC0) ADC Channel Data Register 9
AT91C_ADC0_CDR3           EQU (0x4003805C) ;- (ADC0) ADC Channel Data Register 3
AT91C_ADC0_SR             EQU (0x40038030) ;- (ADC0) ADC Status Register
AT91C_ADC0_CHER           EQU (0x40038010) ;- (ADC0) ADC Channel Enable Register
AT91C_ADC0_CDR1           EQU (0x40038054) ;- (ADC0) ADC Channel Data Register 1
AT91C_ADC0_CDR6           EQU (0x40038068) ;- (ADC0) ADC Channel Data Register 6
AT91C_ADC0_MR             EQU (0x40038004) ;- (ADC0) ADC Mode Register
AT91C_ADC0_CWSR           EQU (0x40038044) ;- (ADC0) unspecified
AT91C_ADC0_VER            EQU (0x400380FC) ;- (ADC0) ADC VERSION REGISTER
AT91C_ADC0_COR            EQU (0x4003804C) ;- (ADC0) unspecified
AT91C_ADC0_CDR8           EQU (0x40038070) ;- (ADC0) ADC Channel Data Register 8
AT91C_ADC0_IPNAME2        EQU (0x400380F4) ;- (ADC0) ADC IPNAME2 REGISTER 
AT91C_ADC0_CDR0           EQU (0x40038050) ;- (ADC0) ADC Channel Data Register 0
AT91C_ADC0_LCDR           EQU (0x40038020) ;- (ADC0) ADC Last Converted Data Register
AT91C_ADC0_CDR12          EQU (0x40038080) ;- (ADC0) ADC Channel Data Register 12
AT91C_ADC0_CHDR           EQU (0x40038014) ;- (ADC0) ADC Channel Disable Register
AT91C_ADC0_OVR            EQU (0x4003803C) ;- (ADC0) unspecified
AT91C_ADC0_CDR15          EQU (0x4003808C) ;- (ADC0) ADC Channel Data Register 15
AT91C_ADC0_CDR11          EQU (0x4003807C) ;- (ADC0) ADC Channel Data Register 11
AT91C_ADC0_ADDRSIZE       EQU (0x400380EC) ;- (ADC0) ADC ADDRSIZE REGISTER 
AT91C_ADC0_CDR13          EQU (0x40038084) ;- (ADC0) ADC Channel Data Register 13
AT91C_ADC0_ACR            EQU (0x40038094) ;- (ADC0) unspecified
AT91C_ADC0_CDR5           EQU (0x40038064) ;- (ADC0) ADC Channel Data Register 5
AT91C_ADC0_CDR14          EQU (0x40038088) ;- (ADC0) ADC Channel Data Register 14
AT91C_ADC0_IMR            EQU (0x4003802C) ;- (ADC0) ADC Interrupt Mask Register
AT91C_ADC0_CHSR           EQU (0x40038018) ;- (ADC0) ADC Channel Status Register
AT91C_ADC0_CDR10          EQU (0x40038078) ;- (ADC0) ADC Channel Data Register 10
AT91C_ADC0_IER            EQU (0x40038024) ;- (ADC0) ADC Interrupt Enable Register
AT91C_ADC0_CDR4           EQU (0x40038060) ;- (ADC0) ADC Channel Data Register 4
// - ========== Register definition for DAC0 peripheral ========== 
AT91C_DAC0_FEATURES       EQU (0x4003C0F8) ;- (DAC0) DAC FEATURES REGISTER 
AT91C_DAC0_ADDRSIZE       EQU (0x4003C0EC) ;- (DAC0) DAC ADDRSIZE REGISTER 
AT91C_DAC0_WPMR           EQU (0x4003C0E4) ;- (DAC0) Write Protect Mode Register
AT91C_DAC0_CHDR           EQU (0x4003C014) ;- (DAC0) Channel Disable Register
AT91C_DAC0_IPNAME1        EQU (0x4003C0F0) ;- (DAC0) DAC IPNAME1 REGISTER 
AT91C_DAC0_IDR            EQU (0x4003C028) ;- (DAC0) Interrupt Disable Register
AT91C_DAC0_CR             EQU (0x4003C000) ;- (DAC0) Control Register
AT91C_DAC0_IPNAME2        EQU (0x4003C0F4) ;- (DAC0) DAC IPNAME2 REGISTER 
AT91C_DAC0_IMR            EQU (0x4003C02C) ;- (DAC0) Interrupt Mask Register
AT91C_DAC0_CHSR           EQU (0x4003C018) ;- (DAC0) Channel Status Register
AT91C_DAC0_ACR            EQU (0x4003C094) ;- (DAC0) Analog Current Register
AT91C_DAC0_WPSR           EQU (0x4003C0E8) ;- (DAC0) Write Protect Status Register
AT91C_DAC0_CHER           EQU (0x4003C010) ;- (DAC0) Channel Enable Register
AT91C_DAC0_CDR            EQU (0x4003C020) ;- (DAC0) Coversion Data Register
AT91C_DAC0_IER            EQU (0x4003C024) ;- (DAC0) Interrupt Enable Register
AT91C_DAC0_ISR            EQU (0x4003C030) ;- (DAC0) Interrupt Status Register
AT91C_DAC0_VER            EQU (0x4003C0FC) ;- (DAC0) DAC VERSION REGISTER
AT91C_DAC0_MR             EQU (0x4003C004) ;- (DAC0) Mode Register
// - ========== Register definition for ACC0 peripheral ========== 
AT91C_ACC0_IPNAME1        EQU (0x400400F0) ;- (ACC0) ACC IPNAME1 REGISTER 
AT91C_ACC0_MR             EQU (0x40040004) ;- (ACC0) Mode Register
AT91C_ACC0_FEATURES       EQU (0x400400F8) ;- (ACC0) ACC FEATURES REGISTER 
AT91C_ACC0_IMR            EQU (0x4004002C) ;- (ACC0) Interrupt Mask Register
AT91C_ACC0_ACR            EQU (0x40040094) ;- (ACC0) Analog Control Register
AT91C_ACC0_ADDRSIZE       EQU (0x400400EC) ;- (ACC0) ACC ADDRSIZE REGISTER 
AT91C_ACC0_IER            EQU (0x40040024) ;- (ACC0) Interrupt Enable Register
AT91C_ACC0_ISR            EQU (0x40040030) ;- (ACC0) Interrupt Status Register
AT91C_ACC0_IDR            EQU (0x40040028) ;- (ACC0) Interrupt Disable Register
AT91C_ACC0_MODE           EQU (0x400400E4) ;- (ACC0) Write Protection Mode Register
AT91C_ACC0_VER            EQU (0x400400FC) ;- (ACC0) ACC VERSION REGISTER
AT91C_ACC0_CR             EQU (0x40040000) ;- (ACC0) Control Register
AT91C_ACC0_IPNAME2        EQU (0x400400F4) ;- (ACC0) ACC IPNAME2 REGISTER 
AT91C_ACC0_STATUS         EQU (0x400400E8) ;- (ACC0) Write Protection Status
// - ========== Register definition for HCBDMA peripheral ========== 
AT91C_HCBDMA_CBIMR        EQU (0x4004401C) ;- (HCBDMA) CB DMA Interrupt mask Register
AT91C_HCBDMA_CBCRCCR      EQU (0x40044034) ;- (HCBDMA) CB DMA CRC Control Resgister
AT91C_HCBDMA_CBSR         EQU (0x40044010) ;- (HCBDMA) CB DMA Status Register
AT91C_HCBDMA_CBCRCISR     EQU (0x4004404C) ;- (HCBDMA) CB DMA CRC Interrupt Status Resgister
AT91C_HCBDMA_CBCRCSR      EQU (0x4004403C) ;- (HCBDMA) CB DMA CRC Status Resgister
AT91C_HCBDMA_CBIDR        EQU (0x40044018) ;- (HCBDMA) CB DMA Interrupt Disable Register
AT91C_HCBDMA_CBCRCIDR     EQU (0x40044044) ;- (HCBDMA) CB DMA CRC Interrupt Disable Resgister
AT91C_HCBDMA_CBDLIER      EQU (0x40044024) ;- (HCBDMA) CB DMA Loaded Interrupt Enable Register
AT91C_HCBDMA_CBEN         EQU (0x40044008) ;- (HCBDMA) CB DMA Enable Register
AT91C_HCBDMA_FEATURES     EQU (0x400440F8) ;- (HCBDMA) HCBDMA FEATURES REGISTER 
AT91C_HCBDMA_CBDSCR       EQU (0x40044000) ;- (HCBDMA) CB DMA Descriptor Base Register
AT91C_HCBDMA_ADDRSIZE     EQU (0x400440EC) ;- (HCBDMA) HCBDMA ADDRSIZE REGISTER 
AT91C_HCBDMA_CBDLISR      EQU (0x40044030) ;- (HCBDMA) CB DMA Loaded Interrupt Status Register
AT91C_HCBDMA_CBDLIDR      EQU (0x40044028) ;- (HCBDMA) CB DMA Loaded Interrupt Disable Register
AT91C_HCBDMA_CBCRCIMR     EQU (0x40044048) ;- (HCBDMA) CB DMA CRC Interrupt Mask Resgister
AT91C_HCBDMA_VER          EQU (0x400440FC) ;- (HCBDMA) HCBDMA VERSION REGISTER
AT91C_HCBDMA_CBCRCIER     EQU (0x40044040) ;- (HCBDMA) CB DMA CRC Interrupt Enable Resgister
AT91C_HCBDMA_IPNAME2      EQU (0x400440F4) ;- (HCBDMA) HCBDMA IPNAME2 REGISTER 
AT91C_HCBDMA_CBIER        EQU (0x40044014) ;- (HCBDMA) CB DMA Interrupt Enable Register
AT91C_HCBDMA_CBISR        EQU (0x40044020) ;- (HCBDMA) CB DMA Interrupt Status Register
AT91C_HCBDMA_IPNAME1      EQU (0x400440F0) ;- (HCBDMA) HCBDMA IPNAME1 REGISTER 
AT91C_HCBDMA_CBDIS        EQU (0x4004400C) ;- (HCBDMA) CB DMA Disable Register
AT91C_HCBDMA_CBNXTEN      EQU (0x40044004) ;- (HCBDMA) CB DMA Next Descriptor Enable Register
AT91C_HCBDMA_CBDLIMR      EQU (0x4004402C) ;- (HCBDMA) CB DMA Loaded Interrupt mask Register
AT91C_HCBDMA_CBCRCMR      EQU (0x40044038) ;- (HCBDMA) CB DMA CRC Mode Resgister
// - ========== Register definition for TC0 peripheral ========== 
AT91C_TC0_SR              EQU (0x40010020) ;- (TC0) Status Register
AT91C_TC0_CCR             EQU (0x40010000) ;- (TC0) Channel Control Register
AT91C_TC0_CMR             EQU (0x40010004) ;- (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC0_IER             EQU (0x40010024) ;- (TC0) Interrupt Enable Register
AT91C_TC0_CV              EQU (0x40010010) ;- (TC0) Counter Value
AT91C_TC0_RB              EQU (0x40010018) ;- (TC0) Register B
AT91C_TC0_IDR             EQU (0x40010028) ;- (TC0) Interrupt Disable Register
AT91C_TC0_RA              EQU (0x40010014) ;- (TC0) Register A
AT91C_TC0_RC              EQU (0x4001001C) ;- (TC0) Register C
AT91C_TC0_IMR             EQU (0x4001002C) ;- (TC0) Interrupt Mask Register
// - ========== Register definition for TC1 peripheral ========== 
AT91C_TC1_SR              EQU (0x40010060) ;- (TC1) Status Register
AT91C_TC1_CV              EQU (0x40010050) ;- (TC1) Counter Value
AT91C_TC1_RA              EQU (0x40010054) ;- (TC1) Register A
AT91C_TC1_IER             EQU (0x40010064) ;- (TC1) Interrupt Enable Register
AT91C_TC1_RB              EQU (0x40010058) ;- (TC1) Register B
AT91C_TC1_RC              EQU (0x4001005C) ;- (TC1) Register C
AT91C_TC1_CCR             EQU (0x40010040) ;- (TC1) Channel Control Register
AT91C_TC1_IMR             EQU (0x4001006C) ;- (TC1) Interrupt Mask Register
AT91C_TC1_IDR             EQU (0x40010068) ;- (TC1) Interrupt Disable Register
AT91C_TC1_CMR             EQU (0x40010044) ;- (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
// - ========== Register definition for TC2 peripheral ========== 
AT91C_TC2_SR              EQU (0x400100A0) ;- (TC2) Status Register
AT91C_TC2_IER             EQU (0x400100A4) ;- (TC2) Interrupt Enable Register
AT91C_TC2_CCR             EQU (0x40010080) ;- (TC2) Channel Control Register
AT91C_TC2_IDR             EQU (0x400100A8) ;- (TC2) Interrupt Disable Register
AT91C_TC2_RA              EQU (0x40010094) ;- (TC2) Register A
AT91C_TC2_RB              EQU (0x40010098) ;- (TC2) Register B
AT91C_TC2_IMR             EQU (0x400100AC) ;- (TC2) Interrupt Mask Register
AT91C_TC2_CV              EQU (0x40010090) ;- (TC2) Counter Value
AT91C_TC2_RC              EQU (0x4001009C) ;- (TC2) Register C
AT91C_TC2_CMR             EQU (0x40010084) ;- (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
// - ========== Register definition for TC3 peripheral ========== 
AT91C_TC3_IDR             EQU (0x40014028) ;- (TC3) Interrupt Disable Register
AT91C_TC3_IER             EQU (0x40014024) ;- (TC3) Interrupt Enable Register
AT91C_TC3_SR              EQU (0x40014020) ;- (TC3) Status Register
AT91C_TC3_CV              EQU (0x40014010) ;- (TC3) Counter Value
AT91C_TC3_CMR             EQU (0x40014004) ;- (TC3) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC3_RC              EQU (0x4001401C) ;- (TC3) Register C
AT91C_TC3_RA              EQU (0x40014014) ;- (TC3) Register A
AT91C_TC3_IMR             EQU (0x4001402C) ;- (TC3) Interrupt Mask Register
AT91C_TC3_RB              EQU (0x40014018) ;- (TC3) Register B
AT91C_TC3_CCR             EQU (0x40014000) ;- (TC3) Channel Control Register
// - ========== Register definition for TC4 peripheral ========== 
AT91C_TC4_CV              EQU (0x40014050) ;- (TC4) Counter Value
AT91C_TC4_CMR             EQU (0x40014044) ;- (TC4) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC4_RA              EQU (0x40014054) ;- (TC4) Register A
AT91C_TC4_IMR             EQU (0x4001406C) ;- (TC4) Interrupt Mask Register
AT91C_TC4_RC              EQU (0x4001405C) ;- (TC4) Register C
AT91C_TC4_SR              EQU (0x40014060) ;- (TC4) Status Register
AT91C_TC4_RB              EQU (0x40014058) ;- (TC4) Register B
AT91C_TC4_IDR             EQU (0x40014068) ;- (TC4) Interrupt Disable Register
AT91C_TC4_IER             EQU (0x40014064) ;- (TC4) Interrupt Enable Register
AT91C_TC4_CCR             EQU (0x40014040) ;- (TC4) Channel Control Register
// - ========== Register definition for TC5 peripheral ========== 
AT91C_TC5_CV              EQU (0x40014090) ;- (TC5) Counter Value
AT91C_TC5_IER             EQU (0x400140A4) ;- (TC5) Interrupt Enable Register
AT91C_TC5_CMR             EQU (0x40014084) ;- (TC5) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC5_IMR             EQU (0x400140AC) ;- (TC5) Interrupt Mask Register
AT91C_TC5_RA              EQU (0x40014094) ;- (TC5) Register A
AT91C_TC5_RB              EQU (0x40014098) ;- (TC5) Register B
AT91C_TC5_RC              EQU (0x4001409C) ;- (TC5) Register C
AT91C_TC5_SR              EQU (0x400140A0) ;- (TC5) Status Register
AT91C_TC5_CCR             EQU (0x40014080) ;- (TC5) Channel Control Register
AT91C_TC5_IDR             EQU (0x400140A8) ;- (TC5) Interrupt Disable Register
// - ========== Register definition for TCB0 peripheral ========== 
AT91C_TCB0_BCR            EQU (0x400100C0) ;- (TCB0) TC Block Control Register
AT91C_TCB0_VER            EQU (0x400100FC) ;- (TCB0)  Version Register
AT91C_TCB0_ADDRSIZE       EQU (0x400100EC) ;- (TCB0) TC ADDRSIZE REGISTER 
AT91C_TCB0_FEATURES       EQU (0x400100F8) ;- (TCB0) TC FEATURES REGISTER 
AT91C_TCB0_IPNAME2        EQU (0x400100F4) ;- (TCB0) TC IPNAME2 REGISTER 
AT91C_TCB0_BMR            EQU (0x400100C4) ;- (TCB0) TC Block Mode Register
AT91C_TCB0_IPNAME1        EQU (0x400100F0) ;- (TCB0) TC IPNAME1 REGISTER 
// - ========== Register definition for TCB1 peripheral ========== 
AT91C_TCB1_IPNAME1        EQU (0x40010130) ;- (TCB1) TC IPNAME1 REGISTER 
AT91C_TCB1_IPNAME2        EQU (0x40010134) ;- (TCB1) TC IPNAME2 REGISTER 
AT91C_TCB1_BCR            EQU (0x40010100) ;- (TCB1) TC Block Control Register
AT91C_TCB1_VER            EQU (0x4001013C) ;- (TCB1)  Version Register
AT91C_TCB1_FEATURES       EQU (0x40010138) ;- (TCB1) TC FEATURES REGISTER 
AT91C_TCB1_ADDRSIZE       EQU (0x4001012C) ;- (TCB1) TC ADDRSIZE REGISTER 
AT91C_TCB1_BMR            EQU (0x40010104) ;- (TCB1) TC Block Mode Register
// - ========== Register definition for TCB2 peripheral ========== 
AT91C_TCB2_VER            EQU (0x4001017C) ;- (TCB2)  Version Register
AT91C_TCB2_ADDRSIZE       EQU (0x4001016C) ;- (TCB2) TC ADDRSIZE REGISTER 
AT91C_TCB2_FEATURES       EQU (0x40010178) ;- (TCB2) TC FEATURES REGISTER 
AT91C_TCB2_BCR            EQU (0x40010140) ;- (TCB2) TC Block Control Register
AT91C_TCB2_IPNAME2        EQU (0x40010174) ;- (TCB2) TC IPNAME2 REGISTER 
AT91C_TCB2_BMR            EQU (0x40010144) ;- (TCB2) TC Block Mode Register
AT91C_TCB2_IPNAME1        EQU (0x40010170) ;- (TCB2) TC IPNAME1 REGISTER 
// - ========== Register definition for TCB3 peripheral ========== 
AT91C_TCB3_IPNAME2        EQU (0x400140F4) ;- (TCB3) TC IPNAME2 REGISTER 
AT91C_TCB3_BMR            EQU (0x400140C4) ;- (TCB3) TC Block Mode Register
AT91C_TCB3_IPNAME1        EQU (0x400140F0) ;- (TCB3) TC IPNAME1 REGISTER 
AT91C_TCB3_FEATURES       EQU (0x400140F8) ;- (TCB3) TC FEATURES REGISTER 
AT91C_TCB3_ADDRSIZE       EQU (0x400140EC) ;- (TCB3) TC ADDRSIZE REGISTER 
AT91C_TCB3_VER            EQU (0x400140FC) ;- (TCB3)  Version Register
AT91C_TCB3_BCR            EQU (0x400140C0) ;- (TCB3) TC Block Control Register
// - ========== Register definition for TCB4 peripheral ========== 
AT91C_TCB4_BMR            EQU (0x40014104) ;- (TCB4) TC Block Mode Register
AT91C_TCB4_BCR            EQU (0x40014100) ;- (TCB4) TC Block Control Register
AT91C_TCB4_IPNAME2        EQU (0x40014134) ;- (TCB4) TC IPNAME2 REGISTER 
AT91C_TCB4_FEATURES       EQU (0x40014138) ;- (TCB4) TC FEATURES REGISTER 
AT91C_TCB4_IPNAME1        EQU (0x40014130) ;- (TCB4) TC IPNAME1 REGISTER 
AT91C_TCB4_VER            EQU (0x4001413C) ;- (TCB4)  Version Register
AT91C_TCB4_ADDRSIZE       EQU (0x4001412C) ;- (TCB4) TC ADDRSIZE REGISTER 
// - ========== Register definition for TCB5 peripheral ========== 
AT91C_TCB5_VER            EQU (0x4001417C) ;- (TCB5)  Version Register
AT91C_TCB5_ADDRSIZE       EQU (0x4001416C) ;- (TCB5) TC ADDRSIZE REGISTER 
AT91C_TCB5_BMR            EQU (0x40014144) ;- (TCB5) TC Block Mode Register
AT91C_TCB5_FEATURES       EQU (0x40014178) ;- (TCB5) TC FEATURES REGISTER 
AT91C_TCB5_IPNAME2        EQU (0x40014174) ;- (TCB5) TC IPNAME2 REGISTER 
AT91C_TCB5_IPNAME1        EQU (0x40014170) ;- (TCB5) TC IPNAME1 REGISTER 
AT91C_TCB5_BCR            EQU (0x40014140) ;- (TCB5) TC Block Control Register
// - ========== Register definition for EFC0 peripheral ========== 
AT91C_EFC0_FMR            EQU (0x400E0A00) ;- (EFC0) EFC Flash Mode Register
AT91C_EFC0_FVR            EQU (0x400E0A14) ;- (EFC0) EFC Flash Version Register
AT91C_EFC0_FSR            EQU (0x400E0A08) ;- (EFC0) EFC Flash Status Register
AT91C_EFC0_FCR            EQU (0x400E0A04) ;- (EFC0) EFC Flash Command Register
AT91C_EFC0_FRR            EQU (0x400E0A0C) ;- (EFC0) EFC Flash Result Register
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
AT91C_TWI0_PTSR           EQU (0x40018124) ;- (PDC_TWI0) PDC Transfer Status Register
AT91C_TWI0_TPR            EQU (0x40018108) ;- (PDC_TWI0) Transmit Pointer Register
AT91C_TWI0_RPR            EQU (0x40018100) ;- (PDC_TWI0) Receive Pointer Register
AT91C_TWI0_TNPR           EQU (0x40018118) ;- (PDC_TWI0) Transmit Next Pointer Register
AT91C_TWI0_PTCR           EQU (0x40018120) ;- (PDC_TWI0) PDC Transfer Control Register
AT91C_TWI0_RCR            EQU (0x40018104) ;- (PDC_TWI0) Receive Counter Register
AT91C_TWI0_RNCR           EQU (0x40018114) ;- (PDC_TWI0) Receive Next Counter Register
AT91C_TWI0_RNPR           EQU (0x40018110) ;- (PDC_TWI0) Receive Next Pointer Register
AT91C_TWI0_TNCR           EQU (0x4001811C) ;- (PDC_TWI0) Transmit Next Counter Register
AT91C_TWI0_TCR            EQU (0x4001810C) ;- (PDC_TWI0) Transmit Counter Register
// - ========== Register definition for PDC_TWI1 peripheral ========== 
AT91C_TWI1_TPR            EQU (0x4001C108) ;- (PDC_TWI1) Transmit Pointer Register
AT91C_TWI1_RNCR           EQU (0x4001C114) ;- (PDC_TWI1) Receive Next Counter Register
AT91C_TWI1_TNCR           EQU (0x4001C11C) ;- (PDC_TWI1) Transmit Next Counter Register
AT91C_TWI1_TCR            EQU (0x4001C10C) ;- (PDC_TWI1) Transmit Counter Register
AT91C_TWI1_TNPR           EQU (0x4001C118) ;- (PDC_TWI1) Transmit Next Pointer Register
AT91C_TWI1_PTCR           EQU (0x4001C120) ;- (PDC_TWI1) PDC Transfer Control Register
AT91C_TWI1_RNPR           EQU (0x4001C110) ;- (PDC_TWI1) Receive Next Pointer Register
AT91C_TWI1_PTSR           EQU (0x4001C124) ;- (PDC_TWI1) PDC Transfer Status Register
AT91C_TWI1_RPR            EQU (0x4001C100) ;- (PDC_TWI1) Receive Pointer Register
AT91C_TWI1_RCR            EQU (0x4001C104) ;- (PDC_TWI1) Receive Counter Register
// - ========== Register definition for TWI0 peripheral ========== 
AT91C_TWI0_IMR            EQU (0x4001802C) ;- (TWI0) Interrupt Mask Register
AT91C_TWI0_IPNAME1        EQU (0x400180F0) ;- (TWI0) TWI IPNAME1 REGISTER 
AT91C_TWI0_CR             EQU (0x40018000) ;- (TWI0) Control Register
AT91C_TWI0_IPNAME2        EQU (0x400180F4) ;- (TWI0) TWI IPNAME2 REGISTER 
AT91C_TWI0_CWGR           EQU (0x40018010) ;- (TWI0) Clock Waveform Generator Register
AT91C_TWI0_SMR            EQU (0x40018008) ;- (TWI0) Slave Mode Register
AT91C_TWI0_ADDRSIZE       EQU (0x400180EC) ;- (TWI0) TWI ADDRSIZE REGISTER 
AT91C_TWI0_SR             EQU (0x40018020) ;- (TWI0) Status Register
AT91C_TWI0_IER            EQU (0x40018024) ;- (TWI0) Interrupt Enable Register
AT91C_TWI0_VER            EQU (0x400180FC) ;- (TWI0) Version Register
AT91C_TWI0_RHR            EQU (0x40018030) ;- (TWI0) Receive Holding Register
AT91C_TWI0_IADR           EQU (0x4001800C) ;- (TWI0) Internal Address Register
AT91C_TWI0_IDR            EQU (0x40018028) ;- (TWI0) Interrupt Disable Register
AT91C_TWI0_THR            EQU (0x40018034) ;- (TWI0) Transmit Holding Register
AT91C_TWI0_FEATURES       EQU (0x400180F8) ;- (TWI0) TWI FEATURES REGISTER 
AT91C_TWI0_MMR            EQU (0x40018004) ;- (TWI0) Master Mode Register
// - ========== Register definition for TWI1 peripheral ========== 
AT91C_TWI1_CR             EQU (0x4001C000) ;- (TWI1) Control Register
AT91C_TWI1_VER            EQU (0x4001C0FC) ;- (TWI1) Version Register
AT91C_TWI1_IMR            EQU (0x4001C02C) ;- (TWI1) Interrupt Mask Register
AT91C_TWI1_IADR           EQU (0x4001C00C) ;- (TWI1) Internal Address Register
AT91C_TWI1_THR            EQU (0x4001C034) ;- (TWI1) Transmit Holding Register
AT91C_TWI1_IPNAME2        EQU (0x4001C0F4) ;- (TWI1) TWI IPNAME2 REGISTER 
AT91C_TWI1_FEATURES       EQU (0x4001C0F8) ;- (TWI1) TWI FEATURES REGISTER 
AT91C_TWI1_SMR            EQU (0x4001C008) ;- (TWI1) Slave Mode Register
AT91C_TWI1_IDR            EQU (0x4001C028) ;- (TWI1) Interrupt Disable Register
AT91C_TWI1_SR             EQU (0x4001C020) ;- (TWI1) Status Register
AT91C_TWI1_IPNAME1        EQU (0x4001C0F0) ;- (TWI1) TWI IPNAME1 REGISTER 
AT91C_TWI1_IER            EQU (0x4001C024) ;- (TWI1) Interrupt Enable Register
AT91C_TWI1_ADDRSIZE       EQU (0x4001C0EC) ;- (TWI1) TWI ADDRSIZE REGISTER 
AT91C_TWI1_CWGR           EQU (0x4001C010) ;- (TWI1) Clock Waveform Generator Register
AT91C_TWI1_MMR            EQU (0x4001C004) ;- (TWI1) Master Mode Register
AT91C_TWI1_RHR            EQU (0x4001C030) ;- (TWI1) Receive Holding Register
// - ========== Register definition for PDC_US0 peripheral ========== 
AT91C_US0_RNCR            EQU (0x40024114) ;- (PDC_US0) Receive Next Counter Register
AT91C_US0_PTCR            EQU (0x40024120) ;- (PDC_US0) PDC Transfer Control Register
AT91C_US0_TCR             EQU (0x4002410C) ;- (PDC_US0) Transmit Counter Register
AT91C_US0_RPR             EQU (0x40024100) ;- (PDC_US0) Receive Pointer Register
AT91C_US0_RNPR            EQU (0x40024110) ;- (PDC_US0) Receive Next Pointer Register
AT91C_US0_TNCR            EQU (0x4002411C) ;- (PDC_US0) Transmit Next Counter Register
AT91C_US0_PTSR            EQU (0x40024124) ;- (PDC_US0) PDC Transfer Status Register
AT91C_US0_RCR             EQU (0x40024104) ;- (PDC_US0) Receive Counter Register
AT91C_US0_TNPR            EQU (0x40024118) ;- (PDC_US0) Transmit Next Pointer Register
AT91C_US0_TPR             EQU (0x40024108) ;- (PDC_US0) Transmit Pointer Register
// - ========== Register definition for US0 peripheral ========== 
AT91C_US0_MAN             EQU (0x40024050) ;- (US0) Manchester Encoder Decoder Register
AT91C_US0_IER             EQU (0x40024008) ;- (US0) Interrupt Enable Register
AT91C_US0_NER             EQU (0x40024044) ;- (US0) Nb Errors Register
AT91C_US0_BRGR            EQU (0x40024020) ;- (US0) Baud Rate Generator Register
AT91C_US0_VER             EQU (0x400240FC) ;- (US0) VERSION Register
AT91C_US0_IF              EQU (0x4002404C) ;- (US0) IRDA_FILTER Register
AT91C_US0_RHR             EQU (0x40024018) ;- (US0) Receiver Holding Register
AT91C_US0_CSR             EQU (0x40024014) ;- (US0) Channel Status Register
AT91C_US0_FEATURES        EQU (0x400240F8) ;- (US0) US FEATURES REGISTER 
AT91C_US0_ADDRSIZE        EQU (0x400240EC) ;- (US0) US ADDRSIZE REGISTER 
AT91C_US0_IMR             EQU (0x40024010) ;- (US0) Interrupt Mask Register
AT91C_US0_THR             EQU (0x4002401C) ;- (US0) Transmitter Holding Register
AT91C_US0_FIDI            EQU (0x40024040) ;- (US0) FI_DI_Ratio Register
AT91C_US0_MR              EQU (0x40024004) ;- (US0) Mode Register
AT91C_US0_RTOR            EQU (0x40024024) ;- (US0) Receiver Time-out Register
AT91C_US0_IPNAME1         EQU (0x400240F0) ;- (US0) US IPNAME1 REGISTER 
AT91C_US0_IDR             EQU (0x4002400C) ;- (US0) Interrupt Disable Register
AT91C_US0_IPNAME2         EQU (0x400240F4) ;- (US0) US IPNAME2 REGISTER 
AT91C_US0_CR              EQU (0x40024000) ;- (US0) Control Register
AT91C_US0_TTGR            EQU (0x40024028) ;- (US0) Transmitter Time-guard Register
// - ========== Register definition for PDC_US1 peripheral ========== 
AT91C_US1_TNPR            EQU (0x40028118) ;- (PDC_US1) Transmit Next Pointer Register
AT91C_US1_RPR             EQU (0x40028100) ;- (PDC_US1) Receive Pointer Register
AT91C_US1_TCR             EQU (0x4002810C) ;- (PDC_US1) Transmit Counter Register
AT91C_US1_RCR             EQU (0x40028104) ;- (PDC_US1) Receive Counter Register
AT91C_US1_TPR             EQU (0x40028108) ;- (PDC_US1) Transmit Pointer Register
AT91C_US1_RNPR            EQU (0x40028110) ;- (PDC_US1) Receive Next Pointer Register
AT91C_US1_TNCR            EQU (0x4002811C) ;- (PDC_US1) Transmit Next Counter Register
AT91C_US1_PTCR            EQU (0x40028120) ;- (PDC_US1) PDC Transfer Control Register
AT91C_US1_RNCR            EQU (0x40028114) ;- (PDC_US1) Receive Next Counter Register
AT91C_US1_PTSR            EQU (0x40028124) ;- (PDC_US1) PDC Transfer Status Register
// - ========== Register definition for US1 peripheral ========== 
AT91C_US1_ADDRSIZE        EQU (0x400280EC) ;- (US1) US ADDRSIZE REGISTER 
AT91C_US1_IDR             EQU (0x4002800C) ;- (US1) Interrupt Disable Register
AT91C_US1_FEATURES        EQU (0x400280F8) ;- (US1) US FEATURES REGISTER 
AT91C_US1_IPNAME2         EQU (0x400280F4) ;- (US1) US IPNAME2 REGISTER 
AT91C_US1_MAN             EQU (0x40028050) ;- (US1) Manchester Encoder Decoder Register
AT91C_US1_CR              EQU (0x40028000) ;- (US1) Control Register
AT91C_US1_TTGR            EQU (0x40028028) ;- (US1) Transmitter Time-guard Register
AT91C_US1_IF              EQU (0x4002804C) ;- (US1) IRDA_FILTER Register
AT91C_US1_FIDI            EQU (0x40028040) ;- (US1) FI_DI_Ratio Register
AT91C_US1_THR             EQU (0x4002801C) ;- (US1) Transmitter Holding Register
AT91C_US1_VER             EQU (0x400280FC) ;- (US1) VERSION Register
AT91C_US1_MR              EQU (0x40028004) ;- (US1) Mode Register
AT91C_US1_CSR             EQU (0x40028014) ;- (US1) Channel Status Register
AT91C_US1_IER             EQU (0x40028008) ;- (US1) Interrupt Enable Register
AT91C_US1_NER             EQU (0x40028044) ;- (US1) Nb Errors Register
AT91C_US1_RHR             EQU (0x40028018) ;- (US1) Receiver Holding Register
AT91C_US1_IPNAME1         EQU (0x400280F0) ;- (US1) US IPNAME1 REGISTER 
AT91C_US1_IMR             EQU (0x40028010) ;- (US1) Interrupt Mask Register
AT91C_US1_BRGR            EQU (0x40028020) ;- (US1) Baud Rate Generator Register
AT91C_US1_RTOR            EQU (0x40028024) ;- (US1) Receiver Time-out Register
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
AT91C_SSC0_FEATURES       EQU (0x400040F8) ;- (SSC0) SSC FEATURES REGISTER 
AT91C_SSC0_ADDRSIZE       EQU (0x400040F0) ;- (SSC0) SSC ADDRSIZE REGISTER 
AT91C_SSC0_CR             EQU (0x40004000) ;- (SSC0) Control Register
AT91C_SSC0_RHR            EQU (0x40004020) ;- (SSC0) Receive Holding Register
AT91C_SSC0_VER            EQU (0x400040FC) ;- (SSC0) Version Register
AT91C_SSC0_TSHR           EQU (0x40004034) ;- (SSC0) Transmit Sync Holding Register
AT91C_SSC0_RFMR           EQU (0x40004014) ;- (SSC0) Receive Frame Mode Register
AT91C_SSC0_IDR            EQU (0x40004048) ;- (SSC0) Interrupt Disable Register
AT91C_SSC0_TFMR           EQU (0x4000401C) ;- (SSC0) Transmit Frame Mode Register
AT91C_SSC0_RSHR           EQU (0x40004030) ;- (SSC0) Receive Sync Holding Register
AT91C_SSC0_TCMR           EQU (0x40004018) ;- (SSC0) Transmit Clock Mode Register
AT91C_SSC0_RCMR           EQU (0x40004010) ;- (SSC0) Receive Clock ModeRegister
AT91C_SSC0_SR             EQU (0x40004040) ;- (SSC0) Status Register
AT91C_SSC0_NAME           EQU (0x400040F4) ;- (SSC0) SSC NAME REGISTER 
AT91C_SSC0_THR            EQU (0x40004024) ;- (SSC0) Transmit Holding Register
AT91C_SSC0_CMR            EQU (0x40004004) ;- (SSC0) Clock Mode Register
AT91C_SSC0_IER            EQU (0x40004044) ;- (SSC0) Interrupt Enable Register
AT91C_SSC0_IMR            EQU (0x4000404C) ;- (SSC0) Interrupt Mask Register
// - ========== Register definition for PDC_PWMC peripheral ========== 
AT91C_PWMC_TNCR           EQU (0x4002011C) ;- (PDC_PWMC) Transmit Next Counter Register
AT91C_PWMC_RCR            EQU (0x40020104) ;- (PDC_PWMC) Receive Counter Register
AT91C_PWMC_TCR            EQU (0x4002010C) ;- (PDC_PWMC) Transmit Counter Register
AT91C_PWMC_RNCR           EQU (0x40020114) ;- (PDC_PWMC) Receive Next Counter Register
AT91C_PWMC_PTSR           EQU (0x40020124) ;- (PDC_PWMC) PDC Transfer Status Register
AT91C_PWMC_RNPR           EQU (0x40020110) ;- (PDC_PWMC) Receive Next Pointer Register
AT91C_PWMC_TNPR           EQU (0x40020118) ;- (PDC_PWMC) Transmit Next Pointer Register
AT91C_PWMC_PTCR           EQU (0x40020120) ;- (PDC_PWMC) PDC Transfer Control Register
AT91C_PWMC_RPR            EQU (0x40020100) ;- (PDC_PWMC) Receive Pointer Register
AT91C_PWMC_TPR            EQU (0x40020108) ;- (PDC_PWMC) Transmit Pointer Register
// - ========== Register definition for PWMC_CH0 peripheral ========== 
AT91C_PWMC_CH0_CMR        EQU (0x40020200) ;- (PWMC_CH0) Channel Mode Register
AT91C_PWMC_CH0_DTUPDR     EQU (0x4002021C) ;- (PWMC_CH0) Channel Dead Time Update Value Register
AT91C_PWMC_CH0_CPRDR      EQU (0x4002020C) ;- (PWMC_CH0) Channel Period Register
AT91C_PWMC_CH0_CPRDUPDR   EQU (0x40020210) ;- (PWMC_CH0) Channel Period Update Register
AT91C_PWMC_CH0_CDTYR      EQU (0x40020204) ;- (PWMC_CH0) Channel Duty Cycle Register
AT91C_PWMC_CH0_DTR        EQU (0x40020218) ;- (PWMC_CH0) Channel Dead Time Value Register
AT91C_PWMC_CH0_CDTYUPDR   EQU (0x40020208) ;- (PWMC_CH0) Channel Duty Cycle Update Register
AT91C_PWMC_CH0_CCNTR      EQU (0x40020214) ;- (PWMC_CH0) Channel Counter Register
// - ========== Register definition for PWMC_CH1 peripheral ========== 
AT91C_PWMC_CH1_DTUPDR     EQU (0x4002023C) ;- (PWMC_CH1) Channel Dead Time Update Value Register
AT91C_PWMC_CH1_DTR        EQU (0x40020238) ;- (PWMC_CH1) Channel Dead Time Value Register
AT91C_PWMC_CH1_CDTYUPDR   EQU (0x40020228) ;- (PWMC_CH1) Channel Duty Cycle Update Register
AT91C_PWMC_CH1_CDTYR      EQU (0x40020224) ;- (PWMC_CH1) Channel Duty Cycle Register
AT91C_PWMC_CH1_CCNTR      EQU (0x40020234) ;- (PWMC_CH1) Channel Counter Register
AT91C_PWMC_CH1_CPRDR      EQU (0x4002022C) ;- (PWMC_CH1) Channel Period Register
AT91C_PWMC_CH1_CMR        EQU (0x40020220) ;- (PWMC_CH1) Channel Mode Register
AT91C_PWMC_CH1_CPRDUPDR   EQU (0x40020230) ;- (PWMC_CH1) Channel Period Update Register
// - ========== Register definition for PWMC_CH2 peripheral ========== 
AT91C_PWMC_CH2_CPRDUPDR   EQU (0x40020250) ;- (PWMC_CH2) Channel Period Update Register
AT91C_PWMC_CH2_CDTYR      EQU (0x40020244) ;- (PWMC_CH2) Channel Duty Cycle Register
AT91C_PWMC_CH2_CCNTR      EQU (0x40020254) ;- (PWMC_CH2) Channel Counter Register
AT91C_PWMC_CH2_CMR        EQU (0x40020240) ;- (PWMC_CH2) Channel Mode Register
AT91C_PWMC_CH2_CDTYUPDR   EQU (0x40020248) ;- (PWMC_CH2) Channel Duty Cycle Update Register
AT91C_PWMC_CH2_DTUPDR     EQU (0x4002025C) ;- (PWMC_CH2) Channel Dead Time Update Value Register
AT91C_PWMC_CH2_DTR        EQU (0x40020258) ;- (PWMC_CH2) Channel Dead Time Value Register
AT91C_PWMC_CH2_CPRDR      EQU (0x4002024C) ;- (PWMC_CH2) Channel Period Register
// - ========== Register definition for PWMC_CH3 peripheral ========== 
AT91C_PWMC_CH3_CPRDR      EQU (0x4002026C) ;- (PWMC_CH3) Channel Period Register
AT91C_PWMC_CH3_DTUPDR     EQU (0x4002027C) ;- (PWMC_CH3) Channel Dead Time Update Value Register
AT91C_PWMC_CH3_DTR        EQU (0x40020278) ;- (PWMC_CH3) Channel Dead Time Value Register
AT91C_PWMC_CH3_CDTYR      EQU (0x40020264) ;- (PWMC_CH3) Channel Duty Cycle Register
AT91C_PWMC_CH3_CMR        EQU (0x40020260) ;- (PWMC_CH3) Channel Mode Register
AT91C_PWMC_CH3_CCNTR      EQU (0x40020274) ;- (PWMC_CH3) Channel Counter Register
AT91C_PWMC_CH3_CPRDUPDR   EQU (0x40020270) ;- (PWMC_CH3) Channel Period Update Register
AT91C_PWMC_CH3_CDTYUPDR   EQU (0x40020268) ;- (PWMC_CH3) Channel Duty Cycle Update Register
// - ========== Register definition for PWMC peripheral ========== 
AT91C_PWMC_CMP6M          EQU (0x40020198) ;- (PWMC) PWM Comparison Mode 6 Register
AT91C_PWMC_ADDRSIZE       EQU (0x400200EC) ;- (PWMC) PWMC ADDRSIZE REGISTER 
AT91C_PWMC_CMP5V          EQU (0x40020180) ;- (PWMC) PWM Comparison Value 5 Register
AT91C_PWMC_FMR            EQU (0x4002005C) ;- (PWMC) PWM Fault Mode Register
AT91C_PWMC_IER2           EQU (0x40020034) ;- (PWMC) PWMC Interrupt Enable Register 2
AT91C_PWMC_EL5MR          EQU (0x40020090) ;- (PWMC) PWM Event Line 5 Mode Register
AT91C_PWMC_CMP0VUPD       EQU (0x40020134) ;- (PWMC) PWM Comparison Value 0 Update Register
AT91C_PWMC_FPER1          EQU (0x4002006C) ;- (PWMC) PWM Fault Protection Enable Register 1
AT91C_PWMC_SCUPUPD        EQU (0x40020030) ;- (PWMC) PWM Update Period Update Register
AT91C_PWMC_DIS            EQU (0x40020008) ;- (PWMC) PWMC Disable Register
AT91C_PWMC_CMP1M          EQU (0x40020148) ;- (PWMC) PWM Comparison Mode 1 Register
AT91C_PWMC_CMP2V          EQU (0x40020150) ;- (PWMC) PWM Comparison Value 2 Register
AT91C_PWMC_WPCR           EQU (0x400200E4) ;- (PWMC) PWM Write Protection Enable Register
AT91C_PWMC_CMP5MUPD       EQU (0x4002018C) ;- (PWMC) PWM Comparison Mode 5 Update Register
AT91C_PWMC_FPV            EQU (0x40020068) ;- (PWMC) PWM Fault Protection Value Register
AT91C_PWMC_UPCR           EQU (0x40020028) ;- (PWMC) PWM Update Control Register
AT91C_PWMC_CMP4MUPD       EQU (0x4002017C) ;- (PWMC) PWM Comparison Mode 4 Update Register
AT91C_PWMC_EL6MR          EQU (0x40020094) ;- (PWMC) PWM Event Line 6 Mode Register
AT91C_PWMC_OS             EQU (0x40020048) ;- (PWMC) PWM Output Selection Register
AT91C_PWMC_OSSUPD         EQU (0x40020054) ;- (PWMC) PWM Output Selection Set Update Register
AT91C_PWMC_FSR            EQU (0x40020060) ;- (PWMC) PWM Fault Mode Status Register
AT91C_PWMC_CMP2M          EQU (0x40020158) ;- (PWMC) PWM Comparison Mode 2 Register
AT91C_PWMC_EL2MR          EQU (0x40020084) ;- (PWMC) PWM Event Line 2 Mode Register
AT91C_PWMC_FPER3          EQU (0x40020074) ;- (PWMC) PWM Fault Protection Enable Register 3
AT91C_PWMC_CMP4M          EQU (0x40020178) ;- (PWMC) PWM Comparison Mode 4 Register
AT91C_PWMC_ISR2           EQU (0x40020040) ;- (PWMC) PWMC Interrupt Status Register 2
AT91C_PWMC_CMP6VUPD       EQU (0x40020194) ;- (PWMC) PWM Comparison Value 6 Update Register
AT91C_PWMC_CMP5VUPD       EQU (0x40020184) ;- (PWMC) PWM Comparison Value 5 Update Register
AT91C_PWMC_EL7MR          EQU (0x40020098) ;- (PWMC) PWM Event Line 7 Mode Register
AT91C_PWMC_OSC            EQU (0x40020050) ;- (PWMC) PWM Output Selection Clear Register
AT91C_PWMC_CMP3MUPD       EQU (0x4002016C) ;- (PWMC) PWM Comparison Mode 3 Update Register
AT91C_PWMC_CMP2MUPD       EQU (0x4002015C) ;- (PWMC) PWM Comparison Mode 2 Update Register
AT91C_PWMC_CMP0M          EQU (0x40020138) ;- (PWMC) PWM Comparison Mode 0 Register
AT91C_PWMC_EL1MR          EQU (0x40020080) ;- (PWMC) PWM Event Line 1 Mode Register
AT91C_PWMC_CMP0MUPD       EQU (0x4002013C) ;- (PWMC) PWM Comparison Mode 0 Update Register
AT91C_PWMC_WPSR           EQU (0x400200E8) ;- (PWMC) PWM Write Protection Status Register
AT91C_PWMC_CMP1MUPD       EQU (0x4002014C) ;- (PWMC) PWM Comparison Mode 1 Update Register
AT91C_PWMC_IMR2           EQU (0x4002003C) ;- (PWMC) PWMC Interrupt Mask Register 2
AT91C_PWMC_CMP3V          EQU (0x40020160) ;- (PWMC) PWM Comparison Value 3 Register
AT91C_PWMC_CMP3VUPD       EQU (0x40020164) ;- (PWMC) PWM Comparison Value 3 Update Register
AT91C_PWMC_CMP3M          EQU (0x40020168) ;- (PWMC) PWM Comparison Mode 3 Register
AT91C_PWMC_FPER4          EQU (0x40020078) ;- (PWMC) PWM Fault Protection Enable Register 4
AT91C_PWMC_OSCUPD         EQU (0x40020058) ;- (PWMC) PWM Output Selection Clear Update Register
AT91C_PWMC_CMP0V          EQU (0x40020130) ;- (PWMC) PWM Comparison Value 0 Register
AT91C_PWMC_OOV            EQU (0x40020044) ;- (PWMC) PWM Output Override Value Register
AT91C_PWMC_ENA            EQU (0x40020004) ;- (PWMC) PWMC Enable Register
AT91C_PWMC_CMP6MUPD       EQU (0x4002019C) ;- (PWMC) PWM Comparison Mode 6 Update Register
AT91C_PWMC_SYNC           EQU (0x40020020) ;- (PWMC) PWM Synchronized Channels Register
AT91C_PWMC_IPNAME1        EQU (0x400200F0) ;- (PWMC) PWMC IPNAME1 REGISTER 
AT91C_PWMC_IDR2           EQU (0x40020038) ;- (PWMC) PWMC Interrupt Disable Register 2
AT91C_PWMC_SR             EQU (0x4002000C) ;- (PWMC) PWMC Status Register
AT91C_PWMC_FPER2          EQU (0x40020070) ;- (PWMC) PWM Fault Protection Enable Register 2
AT91C_PWMC_EL3MR          EQU (0x40020088) ;- (PWMC) PWM Event Line 3 Mode Register
AT91C_PWMC_IMR1           EQU (0x40020018) ;- (PWMC) PWMC Interrupt Mask Register 1
AT91C_PWMC_EL0MR          EQU (0x4002007C) ;- (PWMC) PWM Event Line 0 Mode Register
AT91C_PWMC_STEP           EQU (0x400200B0) ;- (PWMC) PWM Stepper Config Register
AT91C_PWMC_FCR            EQU (0x40020064) ;- (PWMC) PWM Fault Mode Clear Register
AT91C_PWMC_CMP7MUPD       EQU (0x400201AC) ;- (PWMC) PWM Comparison Mode 7 Update Register
AT91C_PWMC_ISR1           EQU (0x4002001C) ;- (PWMC) PWMC Interrupt Status Register 1
AT91C_PWMC_CMP4VUPD       EQU (0x40020174) ;- (PWMC) PWM Comparison Value 4 Update Register
AT91C_PWMC_VER            EQU (0x400200FC) ;- (PWMC) PWMC Version Register
AT91C_PWMC_CMP5M          EQU (0x40020188) ;- (PWMC) PWM Comparison Mode 5 Register
AT91C_PWMC_IER1           EQU (0x40020010) ;- (PWMC) PWMC Interrupt Enable Register 1
AT91C_PWMC_MR             EQU (0x40020000) ;- (PWMC) PWMC Mode Register
AT91C_PWMC_OSS            EQU (0x4002004C) ;- (PWMC) PWM Output Selection Set Register
AT91C_PWMC_CMP7V          EQU (0x400201A0) ;- (PWMC) PWM Comparison Value 7 Register
AT91C_PWMC_FEATURES       EQU (0x400200F8) ;- (PWMC) PWMC FEATURES REGISTER 
AT91C_PWMC_CMP4V          EQU (0x40020170) ;- (PWMC) PWM Comparison Value 4 Register
AT91C_PWMC_CMP7M          EQU (0x400201A8) ;- (PWMC) PWM Comparison Mode 7 Register
AT91C_PWMC_EL4MR          EQU (0x4002008C) ;- (PWMC) PWM Event Line 4 Mode Register
AT91C_PWMC_CMP2VUPD       EQU (0x40020154) ;- (PWMC) PWM Comparison Value 2 Update Register
AT91C_PWMC_CMP6V          EQU (0x40020190) ;- (PWMC) PWM Comparison Value 6 Register
AT91C_PWMC_CMP1V          EQU (0x40020140) ;- (PWMC) PWM Comparison Value 1 Register
AT91C_PWMC_IDR1           EQU (0x40020014) ;- (PWMC) PWMC Interrupt Disable Register 1
AT91C_PWMC_SCUP           EQU (0x4002002C) ;- (PWMC) PWM Update Period Register
AT91C_PWMC_CMP1VUPD       EQU (0x40020144) ;- (PWMC) PWM Comparison Value 1 Update Register
AT91C_PWMC_CMP7VUPD       EQU (0x400201A4) ;- (PWMC) PWM Comparison Value 7 Update Register
AT91C_PWMC_IPNAME2        EQU (0x400200F4) ;- (PWMC) PWMC IPNAME2 REGISTER 
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
// - ========== Register definition for UDP peripheral ========== 
AT91C_UDP_RSTEP           EQU (0x40034028) ;- (UDP) Reset Endpoint Register
AT91C_UDP_CSR             EQU (0x40034030) ;- (UDP) Endpoint Control and Status Register
AT91C_UDP_IMR             EQU (0x40034018) ;- (UDP) Interrupt Mask Register
AT91C_UDP_FDR             EQU (0x40034050) ;- (UDP) Endpoint FIFO Data Register
AT91C_UDP_ISR             EQU (0x4003401C) ;- (UDP) Interrupt Status Register
AT91C_UDP_IPNAME2         EQU (0x400340F4) ;- (UDP) UDP IPNAME2 REGISTER 
AT91C_UDP_ICR             EQU (0x40034020) ;- (UDP) Interrupt Clear Register
AT91C_UDP_VER             EQU (0x400340FC) ;- (UDP) UDP VERSION REGISTER
AT91C_UDP_IER             EQU (0x40034010) ;- (UDP) Interrupt Enable Register
AT91C_UDP_FEATURES        EQU (0x400340F8) ;- (UDP) UDP FEATURES REGISTER 
AT91C_UDP_IPNAME1         EQU (0x400340F0) ;- (UDP) UDP IPNAME1 REGISTER 
AT91C_UDP_GLBSTATE        EQU (0x40034004) ;- (UDP) Global State Register
AT91C_UDP_ADDRSIZE        EQU (0x400340EC) ;- (UDP) UDP ADDRSIZE REGISTER 
AT91C_UDP_NUM             EQU (0x40034000) ;- (UDP) Frame Number Register
AT91C_UDP_IDR             EQU (0x40034014) ;- (UDP) Interrupt Disable Register
AT91C_UDP_TXVC            EQU (0x40034074) ;- (UDP) Transceiver Control Register
AT91C_UDP_FADDR           EQU (0x40034008) ;- (UDP) Function Address Register

// - *****************************************************************************
// -               PIO DEFINITIONS FOR AT91SAM3S1
// - *****************************************************************************
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PA0_PWMH0           EQU (AT91C_PIO_PA0) ;-  
AT91C_PA0_TIOA0           EQU (AT91C_PIO_PA0) ;-  
AT91C_PA0_A17             EQU (AT91C_PIO_PA0) ;-  
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PA1_PWMH1           EQU (AT91C_PIO_PA1) ;-  
AT91C_PA1_TIOB0           EQU (AT91C_PIO_PA1) ;-  
AT91C_PA1_A18             EQU (AT91C_PIO_PA1) ;-  
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PA10_UTXD0          EQU (AT91C_PIO_PA10) ;-  
AT91C_PA10_SPI0_NPCS2     EQU (AT91C_PIO_PA10) ;-  
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PA11_SPI0_NPCS0     EQU (AT91C_PIO_PA11) ;-  
AT91C_PA11_PWMH0          EQU (AT91C_PIO_PA11) ;-  
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PA12_SPI0_MISO      EQU (AT91C_PIO_PA12) ;-  
AT91C_PA12_PWMH1          EQU (AT91C_PIO_PA12) ;-  
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PA13_SPI0_MOSI      EQU (AT91C_PIO_PA13) ;-  
AT91C_PA13_PWMH2          EQU (AT91C_PIO_PA13) ;-  
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PA14_SPI0_SPCK      EQU (AT91C_PIO_PA14) ;-  
AT91C_PA14_PWMH3          EQU (AT91C_PIO_PA14) ;-  
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PA15_TF             EQU (AT91C_PIO_PA15) ;-  
AT91C_PA15_TIOA1          EQU (AT91C_PIO_PA15) ;-  
AT91C_PA15_PWML3          EQU (AT91C_PIO_PA15) ;-  
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PA16_TK             EQU (AT91C_PIO_PA16) ;-  
AT91C_PA16_TIOB1          EQU (AT91C_PIO_PA16) ;-  
AT91C_PA16_PWML2          EQU (AT91C_PIO_PA16) ;-  
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PA17_TD             EQU (AT91C_PIO_PA17) ;-  
AT91C_PA17_PCK1           EQU (AT91C_PIO_PA17) ;-  
AT91C_PA17_PWMH3          EQU (AT91C_PIO_PA17) ;-  
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PA18_RD             EQU (AT91C_PIO_PA18) ;-  
AT91C_PA18_PCK2           EQU (AT91C_PIO_PA18) ;-  
AT91C_PA18_A14            EQU (AT91C_PIO_PA18) ;-  
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PA19_RK             EQU (AT91C_PIO_PA19) ;-  
AT91C_PA19_PWML0          EQU (AT91C_PIO_PA19) ;-  
AT91C_PA19_A15            EQU (AT91C_PIO_PA19) ;-  
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PA2_PWMH2           EQU (AT91C_PIO_PA2) ;-  
AT91C_PA2_SCK0            EQU (AT91C_PIO_PA2) ;-  
AT91C_PA2_DATRG           EQU (AT91C_PIO_PA2) ;-  
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PA20_RF             EQU (AT91C_PIO_PA20) ;-  
AT91C_PA20_PWML1          EQU (AT91C_PIO_PA20) ;-  
AT91C_PA20_A16            EQU (AT91C_PIO_PA20) ;-  
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PA21_RXD1           EQU (AT91C_PIO_PA21) ;-  
AT91C_PA21_PCK1           EQU (AT91C_PIO_PA21) ;-  
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PA22_TXD1           EQU (AT91C_PIO_PA22) ;-  
AT91C_PA22_SPI0_NPCS3     EQU (AT91C_PIO_PA22) ;-  
AT91C_PA22_NCS2           EQU (AT91C_PIO_PA22) ;-  
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PA23_SCK1           EQU (AT91C_PIO_PA23) ;-  
AT91C_PA23_PWMH0          EQU (AT91C_PIO_PA23) ;-  
AT91C_PA23_A19            EQU (AT91C_PIO_PA23) ;-  
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PA24_RTS1           EQU (AT91C_PIO_PA24) ;-  
AT91C_PA24_PWMH1          EQU (AT91C_PIO_PA24) ;-  
AT91C_PA24_A20            EQU (AT91C_PIO_PA24) ;-  
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PA25_CTS1           EQU (AT91C_PIO_PA25) ;-  
AT91C_PA25_PWMH2          EQU (AT91C_PIO_PA25) ;-  
AT91C_PA25_A23            EQU (AT91C_PIO_PA25) ;-  
AT91C_PIO_PA26            EQU (1 << 26) ;- Pin Controlled by PA26
AT91C_PA26_DCD1           EQU (AT91C_PIO_PA26) ;-  
AT91C_PA26_TIOA2          EQU (AT91C_PIO_PA26) ;-  
AT91C_PA26_MCI0_DA2       EQU (AT91C_PIO_PA26) ;-  
AT91C_PIO_PA27            EQU (1 << 27) ;- Pin Controlled by PA27
AT91C_PA27_DTR1           EQU (AT91C_PIO_PA27) ;-  
AT91C_PA27_TIOB2          EQU (AT91C_PIO_PA27) ;-  
AT91C_PA27_MCI0_DA3       EQU (AT91C_PIO_PA27) ;-  
AT91C_PIO_PA28            EQU (1 << 28) ;- Pin Controlled by PA28
AT91C_PA28_DSR1           EQU (AT91C_PIO_PA28) ;-  
AT91C_PA28_TCLK1          EQU (AT91C_PIO_PA28) ;-  
AT91C_PA28_MCI0_CDA       EQU (AT91C_PIO_PA28) ;-  
AT91C_PIO_PA29            EQU (1 << 29) ;- Pin Controlled by PA29
AT91C_PA29_RI1            EQU (AT91C_PIO_PA29) ;-  
AT91C_PA29_TCLK2          EQU (AT91C_PIO_PA29) ;-  
AT91C_PA29_MCI0_CK        EQU (AT91C_PIO_PA29) ;-  
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PA3_TWD0            EQU (AT91C_PIO_PA3) ;-  
AT91C_PA3_SPI0_NPCS3      EQU (AT91C_PIO_PA3) ;-  
AT91C_PIO_PA30            EQU (1 << 30) ;- Pin Controlled by PA30
AT91C_PA30_PWML2          EQU (AT91C_PIO_PA30) ;-  
AT91C_PA30_SPI0_NPCS2     EQU (AT91C_PIO_PA30) ;-  
AT91C_PA30_MCI0_DA0       EQU (AT91C_PIO_PA30) ;-  
AT91C_PIO_PA31            EQU (1 << 31) ;- Pin Controlled by PA31
AT91C_PA31_SPI0_NPCS1     EQU (AT91C_PIO_PA31) ;-  
AT91C_PA31_PCK2           EQU (AT91C_PIO_PA31) ;-  
AT91C_PA31_MCI0_DA1       EQU (AT91C_PIO_PA31) ;-  
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PA4_TWCK0           EQU (AT91C_PIO_PA4) ;-  
AT91C_PA4_TCLK0           EQU (AT91C_PIO_PA4) ;-  
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PA5_RXD0            EQU (AT91C_PIO_PA5) ;-  
AT91C_PA5_SPI0_NPCS3      EQU (AT91C_PIO_PA5) ;-  
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PA6_TXD0            EQU (AT91C_PIO_PA6) ;-  
AT91C_PA6_PCK0            EQU (AT91C_PIO_PA6) ;-  
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PA7_RTS0            EQU (AT91C_PIO_PA7) ;-  
AT91C_PA7_PWMH3           EQU (AT91C_PIO_PA7) ;-  
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PA8_CTS0            EQU (AT91C_PIO_PA8) ;-  
AT91C_PA8_ADTRG           EQU (AT91C_PIO_PA8) ;-  
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PA9_URXD0           EQU (AT91C_PIO_PA9) ;-  
AT91C_PA9_SPI0_NPCS1      EQU (AT91C_PIO_PA9) ;-  
AT91C_PA9_PWMFI0          EQU (AT91C_PIO_PA9) ;-  
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PB0_PWMH0           EQU (AT91C_PIO_PB0) ;-  
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PB1_PWMH1           EQU (AT91C_PIO_PB1) ;-  
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_PWML1          EQU (AT91C_PIO_PB12) ;-  
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_PWML2          EQU (AT91C_PIO_PB13) ;-  
AT91C_PB13_PCK0           EQU (AT91C_PIO_PB13) ;-  
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_SPI0_NPCS1     EQU (AT91C_PIO_PB14) ;-  
AT91C_PB14_PWMH3          EQU (AT91C_PIO_PB14) ;-  
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PB2_URXD1           EQU (AT91C_PIO_PB2) ;-  
AT91C_PB2_SPI0_NPCS2      EQU (AT91C_PIO_PB2) ;-  
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PB3_UTXD1           EQU (AT91C_PIO_PB3) ;-  
AT91C_PB3_PCK2            EQU (AT91C_PIO_PB3) ;-  
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PB4_TWD1            EQU (AT91C_PIO_PB4) ;-  
AT91C_PB4_PWMH2           EQU (AT91C_PIO_PB4) ;-  
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PB5_TWCK1           EQU (AT91C_PIO_PB5) ;-  
AT91C_PB5_PWML0           EQU (AT91C_PIO_PB5) ;-  
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PIO_PC0             EQU (1 <<  0) ;- Pin Controlled by PC0
AT91C_PC0_D0              EQU (AT91C_PIO_PC0) ;-  
AT91C_PC0_PWML0           EQU (AT91C_PIO_PC0) ;-  
AT91C_PIO_PC1             EQU (1 <<  1) ;- Pin Controlled by PC1
AT91C_PC1_D1              EQU (AT91C_PIO_PC1) ;-  
AT91C_PC1_PWML1           EQU (AT91C_PIO_PC1) ;-  
AT91C_PIO_PC10            EQU (1 << 10) ;- Pin Controlled by PC10
AT91C_PC10_NANDWE         EQU (AT91C_PIO_PC10) ;-  
AT91C_PIO_PC11            EQU (1 << 11) ;- Pin Controlled by PC11
AT91C_PC11_NRD            EQU (AT91C_PIO_PC11) ;-  
AT91C_PIO_PC12            EQU (1 << 12) ;- Pin Controlled by PC12
AT91C_PC12_NCS3           EQU (AT91C_PIO_PC12) ;-  
AT91C_PIO_PC13            EQU (1 << 13) ;- Pin Controlled by PC13
AT91C_PC13_NWAIT          EQU (AT91C_PIO_PC13) ;-  
AT91C_PC13_PWML0          EQU (AT91C_PIO_PC13) ;-  
AT91C_PIO_PC14            EQU (1 << 14) ;- Pin Controlled by PC14
AT91C_PC14_NCS0           EQU (AT91C_PIO_PC14) ;-  
AT91C_PIO_PC15            EQU (1 << 15) ;- Pin Controlled by PC15
AT91C_PC15_NCS1           EQU (AT91C_PIO_PC15) ;-  
AT91C_PC15_PWML1          EQU (AT91C_PIO_PC15) ;-  
AT91C_PIO_PC16            EQU (1 << 16) ;- Pin Controlled by PC16
AT91C_PC16_A21_NANDALE    EQU (AT91C_PIO_PC16) ;-  
AT91C_PIO_PC17            EQU (1 << 17) ;- Pin Controlled by PC17
AT91C_PC17_A22_NANDCLE    EQU (AT91C_PIO_PC17) ;-  
AT91C_PIO_PC18            EQU (1 << 18) ;- Pin Controlled by PC18
AT91C_PC18_A0_NBS0        EQU (AT91C_PIO_PC18) ;-  
AT91C_PC18_PWMH0          EQU (AT91C_PIO_PC18) ;-  
AT91C_PIO_PC19            EQU (1 << 19) ;- Pin Controlled by PC19
AT91C_PC19_A1             EQU (AT91C_PIO_PC19) ;-  
AT91C_PC19_PWMH1          EQU (AT91C_PIO_PC19) ;-  
AT91C_PIO_PC2             EQU (1 <<  2) ;- Pin Controlled by PC2
AT91C_PC2_D2              EQU (AT91C_PIO_PC2) ;-  
AT91C_PC2_PWML2           EQU (AT91C_PIO_PC2) ;-  
AT91C_PIO_PC20            EQU (1 << 20) ;- Pin Controlled by PC20
AT91C_PC20_A2             EQU (AT91C_PIO_PC20) ;-  
AT91C_PC20_PWMH2          EQU (AT91C_PIO_PC20) ;-  
AT91C_PIO_PC21            EQU (1 << 21) ;- Pin Controlled by PC21
AT91C_PC21_A3             EQU (AT91C_PIO_PC21) ;-  
AT91C_PC21_PWMH3          EQU (AT91C_PIO_PC21) ;-  
AT91C_PIO_PC22            EQU (1 << 22) ;- Pin Controlled by PC22
AT91C_PC22_A4             EQU (AT91C_PIO_PC22) ;-  
AT91C_PC22_PWML3          EQU (AT91C_PIO_PC22) ;-  
AT91C_PIO_PC23            EQU (1 << 23) ;- Pin Controlled by PC23
AT91C_PC23_A5             EQU (AT91C_PIO_PC23) ;-  
AT91C_PC23_TIOA3          EQU (AT91C_PIO_PC23) ;-  
AT91C_PIO_PC24            EQU (1 << 24) ;- Pin Controlled by PC24
AT91C_PC24_A6             EQU (AT91C_PIO_PC24) ;-  
AT91C_PC24_TIOB3          EQU (AT91C_PIO_PC24) ;-  
AT91C_PIO_PC25            EQU (1 << 25) ;- Pin Controlled by PC25
AT91C_PC25_A7             EQU (AT91C_PIO_PC25) ;-  
AT91C_PC25_TCLK3          EQU (AT91C_PIO_PC25) ;-  
AT91C_PIO_PC26            EQU (1 << 26) ;- Pin Controlled by PC26
AT91C_PC26_A8             EQU (AT91C_PIO_PC26) ;-  
AT91C_PC26_TIOA4          EQU (AT91C_PIO_PC26) ;-  
AT91C_PIO_PC27            EQU (1 << 27) ;- Pin Controlled by PC27
AT91C_PC27_A9             EQU (AT91C_PIO_PC27) ;-  
AT91C_PC27_TIOB4          EQU (AT91C_PIO_PC27) ;-  
AT91C_PIO_PC28            EQU (1 << 28) ;- Pin Controlled by PC28
AT91C_PC28_A10            EQU (AT91C_PIO_PC28) ;-  
AT91C_PC28_TCLK4          EQU (AT91C_PIO_PC28) ;-  
AT91C_PIO_PC29            EQU (1 << 29) ;- Pin Controlled by PC29
AT91C_PC29_A11            EQU (AT91C_PIO_PC29) ;-  
AT91C_PC29_TIOA5          EQU (AT91C_PIO_PC29) ;-  
AT91C_PIO_PC3             EQU (1 <<  3) ;- Pin Controlled by PC3
AT91C_PC3_D3              EQU (AT91C_PIO_PC3) ;-  
AT91C_PC3_PWML3           EQU (AT91C_PIO_PC3) ;-  
AT91C_PIO_PC30            EQU (1 << 30) ;- Pin Controlled by PC30
AT91C_PC30_A12            EQU (AT91C_PIO_PC30) ;-  
AT91C_PC30_TIOB5          EQU (AT91C_PIO_PC30) ;-  
AT91C_PIO_PC31            EQU (1 << 31) ;- Pin Controlled by PC31
AT91C_PC31_A13            EQU (AT91C_PIO_PC31) ;-  
AT91C_PC31_TCLK5          EQU (AT91C_PIO_PC31) ;-  
AT91C_PIO_PC4             EQU (1 <<  4) ;- Pin Controlled by PC4
AT91C_PC4_D4              EQU (AT91C_PIO_PC4) ;-  
AT91C_PC4_SPI0_NPCS1      EQU (AT91C_PIO_PC4) ;-  
AT91C_PIO_PC5             EQU (1 <<  5) ;- Pin Controlled by PC5
AT91C_PC5_D5              EQU (AT91C_PIO_PC5) ;-  
AT91C_PIO_PC6             EQU (1 <<  6) ;- Pin Controlled by PC6
AT91C_PC6_D6              EQU (AT91C_PIO_PC6) ;-  
AT91C_PIO_PC7             EQU (1 <<  7) ;- Pin Controlled by PC7
AT91C_PC7_D7              EQU (AT91C_PIO_PC7) ;-  
AT91C_PIO_PC8             EQU (1 <<  8) ;- Pin Controlled by PC8
AT91C_PC8_NWR0_NWE        EQU (AT91C_PIO_PC8) ;-  
AT91C_PIO_PC9             EQU (1 <<  9) ;- Pin Controlled by PC9
AT91C_PC9_NANDOE          EQU (AT91C_PIO_PC9) ;-  

// - *****************************************************************************
// -               PERIPHERAL ID DEFINITIONS FOR AT91SAM3S1
// - *****************************************************************************
AT91C_ID_SUPC             EQU ( 0) ;- SUPPLY CONTROLLER
AT91C_ID_RSTC             EQU ( 1) ;- RESET CONTROLLER
AT91C_ID_RTC              EQU ( 2) ;- REAL TIME CLOCK
AT91C_ID_RTT              EQU ( 3) ;- REAL TIME TIMER
AT91C_ID_WDG              EQU ( 4) ;- WATCHDOG TIMER
AT91C_ID_PMC              EQU ( 5) ;- PMC
AT91C_ID_EFC0             EQU ( 6) ;- EFC0
AT91C_ID_DBGU0            EQU ( 8) ;- DBGU0
AT91C_ID_DBGU1            EQU ( 9) ;- DBGU1
AT91C_ID_HSMC3            EQU (10) ;- HSMC3
AT91C_ID_PIOA             EQU (11) ;- Parallel IO Controller A
AT91C_ID_PIOB             EQU (12) ;- Parallel IO Controller B
AT91C_ID_PIOC             EQU (13) ;- Parallel IO Controller C
AT91C_ID_US0              EQU (14) ;- USART 0
AT91C_ID_US1              EQU (15) ;- USART 1
AT91C_ID_MCI0             EQU (18) ;- Multimedia Card Interface
AT91C_ID_TWI0             EQU (19) ;- TWI 0
AT91C_ID_TWI1             EQU (20) ;- TWI 1
AT91C_ID_SPI0             EQU (21) ;- Serial Peripheral Interface
AT91C_ID_SSC0             EQU (22) ;- Serial Synchronous Controller 0
AT91C_ID_TC0              EQU (23) ;- Timer Counter 0
AT91C_ID_TC1              EQU (24) ;- Timer Counter 1
AT91C_ID_TC2              EQU (25) ;- Timer Counter 2
AT91C_ID_TC3              EQU (26) ;- Timer Counter 0
AT91C_ID_TC4              EQU (27) ;- Timer Counter 1
AT91C_ID_TC5              EQU (28) ;- Timer Counter 2
AT91C_ID_ADC0             EQU (29) ;- ADC controller0
AT91C_ID_DAC0             EQU (30) ;- Digital to Analog Converter
AT91C_ID_PWMC             EQU (31) ;- Pulse Width Modulation Controller
AT91C_ID_HCBDMA           EQU (32) ;- Context Based Direct Memory Access Controller Interface
AT91C_ID_ACC0             EQU (33) ;- Analog Comparator Controller
AT91C_ID_UDP              EQU (34) ;- USB Device
AT91C_ALL_INT             EQU (0xFFFFFFFF) ;- ALL VALID INTERRUPTS

// - *****************************************************************************
// -               BASE ADDRESS DEFINITIONS FOR AT91SAM3S1
// - *****************************************************************************
AT91C_BASE_SYS            EQU (0x400E0000) ;- (SYS) Base Address
AT91C_BASE_SMC            EQU (0x400E0000) ;- (SMC) Base Address
AT91C_BASE_MATRIX         EQU (0x400E0200) ;- (MATRIX) Base Address
AT91C_BASE_CCFG           EQU (0x400E0310) ;- (CCFG) Base Address
AT91C_BASE_NVIC           EQU (0xE000E000) ;- (NVIC) Base Address
AT91C_BASE_MPU            EQU (0xE000ED90) ;- (MPU) Base Address
AT91C_BASE_CM3            EQU (0xE000ED00) ;- (CM3) Base Address
AT91C_BASE_PDC_DBGU0      EQU (0x400E0700) ;- (PDC_DBGU0) Base Address
AT91C_BASE_DBGU0          EQU (0x400E0600) ;- (DBGU0) Base Address
AT91C_BASE_PDC_DBGU1      EQU (0x400E0900) ;- (PDC_DBGU1) Base Address
AT91C_BASE_DBGU1          EQU (0x400E0800) ;- (DBGU1) Base Address
AT91C_BASE_PIOA           EQU (0x400E0E00) ;- (PIOA) Base Address
AT91C_BASE_PDC_PIOA       EQU (0x400E0F68) ;- (PDC_PIOA) Base Address
AT91C_BASE_PIOB           EQU (0x400E1000) ;- (PIOB) Base Address
AT91C_BASE_PIOC           EQU (0x400E1200) ;- (PIOC) Base Address
AT91C_BASE_PMC            EQU (0x400E0400) ;- (PMC) Base Address
AT91C_BASE_CKGR           EQU (0x400E041C) ;- (CKGR) Base Address
AT91C_BASE_RSTC           EQU (0x400E1400) ;- (RSTC) Base Address
AT91C_BASE_SUPC           EQU (0x400E1410) ;- (SUPC) Base Address
AT91C_BASE_RTTC           EQU (0x400E1430) ;- (RTTC) Base Address
AT91C_BASE_WDTC           EQU (0x400E1450) ;- (WDTC) Base Address
AT91C_BASE_RTC            EQU (0x400E1460) ;- (RTC) Base Address
AT91C_BASE_ADC0           EQU (0x40038000) ;- (ADC0) Base Address
AT91C_BASE_DAC0           EQU (0x4003C000) ;- (DAC0) Base Address
AT91C_BASE_ACC0           EQU (0x40040000) ;- (ACC0) Base Address
AT91C_BASE_HCBDMA         EQU (0x40044000) ;- (HCBDMA) Base Address
AT91C_BASE_TC0            EQU (0x40010000) ;- (TC0) Base Address
AT91C_BASE_TC1            EQU (0x40010040) ;- (TC1) Base Address
AT91C_BASE_TC2            EQU (0x40010080) ;- (TC2) Base Address
AT91C_BASE_TC3            EQU (0x40014000) ;- (TC3) Base Address
AT91C_BASE_TC4            EQU (0x40014040) ;- (TC4) Base Address
AT91C_BASE_TC5            EQU (0x40014080) ;- (TC5) Base Address
AT91C_BASE_TCB0           EQU (0x40010000) ;- (TCB0) Base Address
AT91C_BASE_TCB1           EQU (0x40010040) ;- (TCB1) Base Address
AT91C_BASE_TCB2           EQU (0x40010080) ;- (TCB2) Base Address
AT91C_BASE_TCB3           EQU (0x40014000) ;- (TCB3) Base Address
AT91C_BASE_TCB4           EQU (0x40014040) ;- (TCB4) Base Address
AT91C_BASE_TCB5           EQU (0x40014080) ;- (TCB5) Base Address
AT91C_BASE_EFC0           EQU (0x400E0A00) ;- (EFC0) Base Address
AT91C_BASE_MCI0           EQU (0x40000000) ;- (MCI0) Base Address
AT91C_BASE_PDC_TWI0       EQU (0x40018100) ;- (PDC_TWI0) Base Address
AT91C_BASE_PDC_TWI1       EQU (0x4001C100) ;- (PDC_TWI1) Base Address
AT91C_BASE_TWI0           EQU (0x40018000) ;- (TWI0) Base Address
AT91C_BASE_TWI1           EQU (0x4001C000) ;- (TWI1) Base Address
AT91C_BASE_PDC_US0        EQU (0x40024100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0x40024000) ;- (US0) Base Address
AT91C_BASE_PDC_US1        EQU (0x40028100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0x40028000) ;- (US1) Base Address
AT91C_BASE_PDC_SSC0       EQU (0x40004100) ;- (PDC_SSC0) Base Address
AT91C_BASE_SSC0           EQU (0x40004000) ;- (SSC0) Base Address
AT91C_BASE_PDC_PWMC       EQU (0x40020100) ;- (PDC_PWMC) Base Address
AT91C_BASE_PWMC_CH0       EQU (0x40020200) ;- (PWMC_CH0) Base Address
AT91C_BASE_PWMC_CH1       EQU (0x40020220) ;- (PWMC_CH1) Base Address
AT91C_BASE_PWMC_CH2       EQU (0x40020240) ;- (PWMC_CH2) Base Address
AT91C_BASE_PWMC_CH3       EQU (0x40020260) ;- (PWMC_CH3) Base Address
AT91C_BASE_PWMC           EQU (0x40020000) ;- (PWMC) Base Address
AT91C_BASE_SPI0           EQU (0x40008000) ;- (SPI0) Base Address
AT91C_BASE_UDP            EQU (0x40034000) ;- (UDP) Base Address

// - *****************************************************************************
// -               MEMORY MAPPING DEFINITIONS FOR AT91SAM3S1
// - *****************************************************************************
// - IRAM
AT91C_IRAM                EQU (0x20000000) ;- Maximum Internal SRAM base address
AT91C_IRAM_SIZE           EQU (0x00004000) ;- Maximum Internal SRAM size in byte (16 Kbytes)
// - IROM
AT91C_IROM                EQU (0x00800000) ;- Internal ROM base address
AT91C_IROM_SIZE           EQU (0x00008000) ;- Internal ROM size in byte (32 Kbytes)
// - EBI_CS0
AT91C_EBI_CS0             EQU (0x60000000) ;- EBI Chip Select 0 base address
AT91C_EBI_CS0_SIZE        EQU (0x01000000) ;- EBI Chip Select 0 size in byte (16384 Kbytes)
// - EBI_SM0
AT91C_EBI_SM0             EQU (0x60000000) ;- NANDFLASH on EBI Chip Select 0 base address
AT91C_EBI_SM0_SIZE        EQU (0x01000000) ;- NANDFLASH on EBI Chip Select 0 size in byte (16384 Kbytes)
// - EBI_CS1
AT91C_EBI_CS1             EQU (0x61000000) ;- EBI Chip Select 1 base address
AT91C_EBI_CS1_SIZE        EQU (0x01000000) ;- EBI Chip Select 1 size in byte (16384 Kbytes)
// - EBI_SM1
AT91C_EBI_SM1             EQU (0x61000000) ;- NANDFLASH on EBI Chip Select 1 base address
AT91C_EBI_SM1_SIZE        EQU (0x01000000) ;- NANDFLASH on EBI Chip Select 1 size in byte (16384 Kbytes)
// - EBI_CS2
AT91C_EBI_CS2             EQU (0x62000000) ;- EBI Chip Select 2 base address
AT91C_EBI_CS2_SIZE        EQU (0x01000000) ;- EBI Chip Select 2 size in byte (16384 Kbytes)
// - EBI_SM2
AT91C_EBI_SM2             EQU (0x62000000) ;- NANDFLASH on EBI Chip Select 2 base address
AT91C_EBI_SM2_SIZE        EQU (0x01000000) ;- NANDFLASH on EBI Chip Select 2 size in byte (16384 Kbytes)
// - EBI_CS3
AT91C_EBI_CS3             EQU (0x63000000) ;- EBI Chip Select 3 base address
AT91C_EBI_CS3_SIZE        EQU (0x01000000) ;- EBI Chip Select 3 size in byte (16384 Kbytes)
// - EBI_SM3
AT91C_EBI_SM3             EQU (0x63000000) ;- NANDFLASH on EBI Chip Select 3 base address
AT91C_EBI_SM3_SIZE        EQU (0x01000000) ;- NANDFLASH on EBI Chip Select 3 size in byte (16384 Kbytes)
// - EBI_CS4
AT91C_EBI_CS4             EQU (0x64000000) ;- EBI Chip Select 4 base address
AT91C_EBI_CS4_SIZE        EQU (0x10000000) ;- EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CF0
AT91C_EBI_CF0             EQU (0x64000000) ;- CompactFlash 0 on EBI Chip Select 4 base address
AT91C_EBI_CF0_SIZE        EQU (0x10000000) ;- CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
// - EBI_CS5
AT91C_EBI_CS5             EQU (0x65000000) ;- EBI Chip Select 5 base address
AT91C_EBI_CS5_SIZE        EQU (0x10000000) ;- EBI Chip Select 5 size in byte (262144 Kbytes)
// - EBI_CF1
AT91C_EBI_CF1             EQU (0x65000000) ;- CompactFlash 1 on EBIChip Select 5 base address
AT91C_EBI_CF1_SIZE        EQU (0x10000000) ;- CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
// - EBI_SDRAM
AT91C_EBI_SDRAM           EQU (0x66000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_SIZE      EQU (0x10000000) ;- SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
// - EBI_SDRAM_16BIT
AT91C_EBI_SDRAM_16BIT     EQU (0x67000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_16BIT_SIZE EQU (0x02000000) ;- SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
// - EBI_SDRAM_32BIT
AT91C_EBI_SDRAM_32BIT     EQU (0x68000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_32BIT_SIZE EQU (0x04000000) ;- SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91SAM3S1_H */
