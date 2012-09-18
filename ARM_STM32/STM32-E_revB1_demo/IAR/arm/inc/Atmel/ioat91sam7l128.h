/* - ----------------------------------------------------------------------------*/
/* -          ATMEL Microcontroller Software Support  -  ROUSSET  -*/
/* - ----------------------------------------------------------------------------*/
/* -  DISCLAIMER:  THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR*/
/* -  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF*/
/* -  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE*/
/* -  DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR ANY DIRECT, INDIRECT,*/
/* -  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT*/
/* -  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,*/
/* -  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF*/
/* -  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING*/
/* -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,*/
/* -  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
/* - ----------------------------------------------------------------------------*/
/* - File Name           : AT91SAM7L128.h*/
/* - Object              : AT91SAM7L128 definitions*/
/* - Generated           : AT91 SW Application Group  06/19/2007 (15:58:20)*/
/* - */
/* - CVS Reference       : /AT91SAM7L128.pl/1.3/Tue Jun 19 13:54:27 2007  */
/* - CVS Reference       : /SYS_SAM7L64.pl/1.3/Fri Feb  2 13:37:40 2007  */
/* - CVS Reference       : /MC_SAM7L64.pl/1.4/Fri Mar 16 08:22:15 2007  */
/* - CVS Reference       : /PMC_SAM7L_NO_USB.pl/1.5/Fri Mar 16 07:50:08 2007  */
/* - CVS Reference       : /RSTC_SAM7X.pl/1.2/Wed Jul 13 15:25:17 2005  */
/* - CVS Reference       : /PWM_SAM7X.pl/1.1/Tue May 10 12:38:54 2005  */
/* - CVS Reference       : /AIC_6075B.pl/1.3/Fri May 20 14:21:42 2005  */
/* - CVS Reference       : /PIO_6057A.pl/1.2/Thu Feb  3 10:29:42 2005  */
/* - CVS Reference       : /RTC_1366C.pl/1.2/Mon Nov  4 17:50:59 2002  */
/* - CVS Reference       : /PITC_6079A.pl/1.2/Thu Nov  4 13:56:22 2004  */
/* - CVS Reference       : /WDTC_6080A.pl/1.3/Thu Nov  4 13:58:52 2004  */
/* - CVS Reference       : /VREG_6085B.pl/1.1/Tue Feb  1 16:40:38 2005  */
/* - CVS Reference       : /PDC_6074C.pl/1.2/Thu Feb  3 09:02:11 2005  */
/* - CVS Reference       : /DBGU_6059D.pl/1.1/Mon Jan 31 13:54:41 2005  */
/* - CVS Reference       : /SPI_6088D.pl/1.3/Fri May 20 14:23:02 2005  */
/* - CVS Reference       : /US_6089C.pl/1.1/Mon Jan 31 13:56:02 2005  */
/* - CVS Reference       : /TWI_6061B.pl/1.2/Fri Aug  4 08:53:02 2006  */
/* - CVS Reference       : /SLCDC_60.pl/1.5/Fri Mar 16 08:23:45 2007  */
/* - CVS Reference       : /TC_6082A.pl/1.7/Wed Mar  9 16:31:51 2005  */
/* - CVS Reference       : /ADC_6051C.pl/1.1/Mon Jan 31 13:12:40 2005  */
/* - ----------------------------------------------------------------------------*/

#ifndef AT91SAM7L128_H
#define AT91SAM7L128_H

#ifdef __IAR_SYSTEMS_ICC__

typedef volatile unsigned int AT91_REG;/* Hardware register definition*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR System Peripherals*/
/* ******************************************************************************/
typedef struct _AT91S_SYS {
	AT91_REG	 AIC_SMR[32]; 	/* Source Mode Register*/
	AT91_REG	 AIC_SVR[32]; 	/* Source Vector Register*/
	AT91_REG	 AIC_IVR; 	/* IRQ Vector Register*/
	AT91_REG	 AIC_FVR; 	/* FIQ Vector Register*/
	AT91_REG	 AIC_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 AIC_IPR; 	/* Interrupt Pending Register*/
	AT91_REG	 AIC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 AIC_CISR; 	/* Core Interrupt Status Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 AIC_IECR; 	/* Interrupt Enable Command Register*/
	AT91_REG	 AIC_IDCR; 	/* Interrupt Disable Command Register*/
	AT91_REG	 AIC_ICCR; 	/* Interrupt Clear Command Register*/
	AT91_REG	 AIC_ISCR; 	/* Interrupt Set Command Register*/
	AT91_REG	 AIC_EOICR; 	/* End of Interrupt Command Register*/
	AT91_REG	 AIC_SPU; 	/* Spurious Vector Register*/
	AT91_REG	 AIC_DCR; 	/* Debug Control Register (Protect)*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 AIC_FFER; 	/* Fast Forcing Enable Register*/
	AT91_REG	 AIC_FFDR; 	/* Fast Forcing Disable Register*/
	AT91_REG	 AIC_FFSR; 	/* Fast Forcing Status Register*/
	AT91_REG	 Reserved2[45]; 	/* */
	AT91_REG	 DBGU_CR; 	/* Control Register*/
	AT91_REG	 DBGU_MR; 	/* Mode Register*/
	AT91_REG	 DBGU_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 DBGU_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 DBGU_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 DBGU_CSR; 	/* Channel Status Register*/
	AT91_REG	 DBGU_RHR; 	/* Receiver Holding Register*/
	AT91_REG	 DBGU_THR; 	/* Transmitter Holding Register*/
	AT91_REG	 DBGU_BRGR; 	/* Baud Rate Generator Register*/
	AT91_REG	 Reserved3[7]; 	/* */
	AT91_REG	 DBGU_CIDR; 	/* Chip ID Register*/
	AT91_REG	 DBGU_EXID; 	/* Chip ID Extension Register*/
	AT91_REG	 DBGU_FNTR; 	/* Force NTRST Register*/
	AT91_REG	 Reserved4[45]; 	/* */
	AT91_REG	 DBGU_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 DBGU_RCR; 	/* Receive Counter Register*/
	AT91_REG	 DBGU_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 DBGU_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 DBGU_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 DBGU_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 DBGU_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 DBGU_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 DBGU_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 DBGU_PTSR; 	/* PDC Transfer Status Register*/
	AT91_REG	 Reserved5[54]; 	/* */
	AT91_REG	 PIOA_PER; 	/* PIO Enable Register*/
	AT91_REG	 PIOA_PDR; 	/* PIO Disable Register*/
	AT91_REG	 PIOA_PSR; 	/* PIO Status Register*/
	AT91_REG	 Reserved6[1]; 	/* */
	AT91_REG	 PIOA_OER; 	/* Output Enable Register*/
	AT91_REG	 PIOA_ODR; 	/* Output Disable Registerr*/
	AT91_REG	 PIOA_OSR; 	/* Output Status Register*/
	AT91_REG	 Reserved7[1]; 	/* */
	AT91_REG	 PIOA_IFER; 	/* Input Filter Enable Register*/
	AT91_REG	 PIOA_IFDR; 	/* Input Filter Disable Register*/
	AT91_REG	 PIOA_IFSR; 	/* Input Filter Status Register*/
	AT91_REG	 Reserved8[1]; 	/* */
	AT91_REG	 PIOA_SODR; 	/* Set Output Data Register*/
	AT91_REG	 PIOA_CODR; 	/* Clear Output Data Register*/
	AT91_REG	 PIOA_ODSR; 	/* Output Data Status Register*/
	AT91_REG	 PIOA_PDSR; 	/* Pin Data Status Register*/
	AT91_REG	 PIOA_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PIOA_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PIOA_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PIOA_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 PIOA_MDER; 	/* Multi-driver Enable Register*/
	AT91_REG	 PIOA_MDDR; 	/* Multi-driver Disable Register*/
	AT91_REG	 PIOA_MDSR; 	/* Multi-driver Status Register*/
	AT91_REG	 Reserved9[1]; 	/* */
	AT91_REG	 PIOA_PPUDR; 	/* Pull-up Disable Register*/
	AT91_REG	 PIOA_PPUER; 	/* Pull-up Enable Register*/
	AT91_REG	 PIOA_PPUSR; 	/* Pull-up Status Register*/
	AT91_REG	 Reserved10[1]; 	/* */
	AT91_REG	 PIOA_ASR; 	/* Select A Register*/
	AT91_REG	 PIOA_BSR; 	/* Select B Register*/
	AT91_REG	 PIOA_ABSR; 	/* AB Select Status Register*/
	AT91_REG	 Reserved11[9]; 	/* */
	AT91_REG	 PIOA_OWER; 	/* Output Write Enable Register*/
	AT91_REG	 PIOA_OWDR; 	/* Output Write Disable Register*/
	AT91_REG	 PIOA_OWSR; 	/* Output Write Status Register*/
	AT91_REG	 Reserved12[85]; 	/* */
	AT91_REG	 PIOB_PER; 	/* PIO Enable Register*/
	AT91_REG	 PIOB_PDR; 	/* PIO Disable Register*/
	AT91_REG	 PIOB_PSR; 	/* PIO Status Register*/
	AT91_REG	 Reserved13[1]; 	/* */
	AT91_REG	 PIOB_OER; 	/* Output Enable Register*/
	AT91_REG	 PIOB_ODR; 	/* Output Disable Registerr*/
	AT91_REG	 PIOB_OSR; 	/* Output Status Register*/
	AT91_REG	 Reserved14[1]; 	/* */
	AT91_REG	 PIOB_IFER; 	/* Input Filter Enable Register*/
	AT91_REG	 PIOB_IFDR; 	/* Input Filter Disable Register*/
	AT91_REG	 PIOB_IFSR; 	/* Input Filter Status Register*/
	AT91_REG	 Reserved15[1]; 	/* */
	AT91_REG	 PIOB_SODR; 	/* Set Output Data Register*/
	AT91_REG	 PIOB_CODR; 	/* Clear Output Data Register*/
	AT91_REG	 PIOB_ODSR; 	/* Output Data Status Register*/
	AT91_REG	 PIOB_PDSR; 	/* Pin Data Status Register*/
	AT91_REG	 PIOB_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PIOB_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PIOB_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PIOB_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 PIOB_MDER; 	/* Multi-driver Enable Register*/
	AT91_REG	 PIOB_MDDR; 	/* Multi-driver Disable Register*/
	AT91_REG	 PIOB_MDSR; 	/* Multi-driver Status Register*/
	AT91_REG	 Reserved16[1]; 	/* */
	AT91_REG	 PIOB_PPUDR; 	/* Pull-up Disable Register*/
	AT91_REG	 PIOB_PPUER; 	/* Pull-up Enable Register*/
	AT91_REG	 PIOB_PPUSR; 	/* Pull-up Status Register*/
	AT91_REG	 Reserved17[1]; 	/* */
	AT91_REG	 PIOB_ASR; 	/* Select A Register*/
	AT91_REG	 PIOB_BSR; 	/* Select B Register*/
	AT91_REG	 PIOB_ABSR; 	/* AB Select Status Register*/
	AT91_REG	 Reserved18[9]; 	/* */
	AT91_REG	 PIOB_OWER; 	/* Output Write Enable Register*/
	AT91_REG	 PIOB_OWDR; 	/* Output Write Disable Register*/
	AT91_REG	 PIOB_OWSR; 	/* Output Write Status Register*/
	AT91_REG	 Reserved19[85]; 	/* */
	AT91_REG	 PIOC_PER; 	/* PIO Enable Register*/
	AT91_REG	 PIOC_PDR; 	/* PIO Disable Register*/
	AT91_REG	 PIOC_PSR; 	/* PIO Status Register*/
	AT91_REG	 Reserved20[1]; 	/* */
	AT91_REG	 PIOC_OER; 	/* Output Enable Register*/
	AT91_REG	 PIOC_ODR; 	/* Output Disable Registerr*/
	AT91_REG	 PIOC_OSR; 	/* Output Status Register*/
	AT91_REG	 Reserved21[1]; 	/* */
	AT91_REG	 PIOC_IFER; 	/* Input Filter Enable Register*/
	AT91_REG	 PIOC_IFDR; 	/* Input Filter Disable Register*/
	AT91_REG	 PIOC_IFSR; 	/* Input Filter Status Register*/
	AT91_REG	 Reserved22[1]; 	/* */
	AT91_REG	 PIOC_SODR; 	/* Set Output Data Register*/
	AT91_REG	 PIOC_CODR; 	/* Clear Output Data Register*/
	AT91_REG	 PIOC_ODSR; 	/* Output Data Status Register*/
	AT91_REG	 PIOC_PDSR; 	/* Pin Data Status Register*/
	AT91_REG	 PIOC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PIOC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PIOC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PIOC_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 PIOC_MDER; 	/* Multi-driver Enable Register*/
	AT91_REG	 PIOC_MDDR; 	/* Multi-driver Disable Register*/
	AT91_REG	 PIOC_MDSR; 	/* Multi-driver Status Register*/
	AT91_REG	 Reserved23[1]; 	/* */
	AT91_REG	 PIOC_PPUDR; 	/* Pull-up Disable Register*/
	AT91_REG	 PIOC_PPUER; 	/* Pull-up Enable Register*/
	AT91_REG	 PIOC_PPUSR; 	/* Pull-up Status Register*/
	AT91_REG	 Reserved24[1]; 	/* */
	AT91_REG	 PIOC_ASR; 	/* Select A Register*/
	AT91_REG	 PIOC_BSR; 	/* Select B Register*/
	AT91_REG	 PIOC_ABSR; 	/* AB Select Status Register*/
	AT91_REG	 Reserved25[9]; 	/* */
	AT91_REG	 PIOC_OWER; 	/* Output Write Enable Register*/
	AT91_REG	 PIOC_OWDR; 	/* Output Write Disable Register*/
	AT91_REG	 PIOC_OWSR; 	/* Output Write Status Register*/
	AT91_REG	 Reserved26[213]; 	/* */
	AT91_REG	 PMC_SCER; 	/* System Clock Enable Register*/
	AT91_REG	 PMC_SCDR; 	/* System Clock Disable Register*/
	AT91_REG	 PMC_SCSR; 	/* System Clock Status Register*/
	AT91_REG	 Reserved27[1]; 	/* */
	AT91_REG	 PMC_PCER; 	/* Peripheral Clock Enable Register*/
	AT91_REG	 PMC_PCDR; 	/* Peripheral Clock Disable Register*/
	AT91_REG	 PMC_PCSR; 	/* Peripheral Clock Status Register*/
	AT91_REG	 Reserved28[1]; 	/* */
	AT91_REG	 PMC_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 PMC_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 PMC_PLLR; 	/* PLL Register*/
	AT91_REG	 Reserved29[1]; 	/* */
	AT91_REG	 PMC_MCKR; 	/* Master Clock Register*/
	AT91_REG	 Reserved30[3]; 	/* */
	AT91_REG	 PMC_PCKR[3]; 	/* Programmable Clock Register*/
	AT91_REG	 Reserved31[5]; 	/* */
	AT91_REG	 PMC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PMC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PMC_SR; 	/* Status Register*/
	AT91_REG	 PMC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PMC_FSMR; 	/* Fast Startup Mode Register*/
	AT91_REG	 Reserved32[35]; 	/* */
	AT91_REG	 RSTC_RCR; 	/* Reset Control Register*/
	AT91_REG	 RSTC_RSR; 	/* Reset Status Register*/
	AT91_REG	 RSTC_RMR; 	/* Reset Mode Register*/
	AT91_REG	 Reserved33[1]; 	/* */
	AT91_REG	 SUPC_CR; 	/* Control Register*/
	AT91_REG	 SUPC_BOMR; 	/* Brown Out Mode Register*/
	AT91_REG	 SUPC_MR; 	/* Mode Register*/
	AT91_REG	 SUPC_WUMR; 	/* Wake Up Mode Register*/
	AT91_REG	 SUPC_WUIR; 	/* Wake Up Inputs Register*/
	AT91_REG	 SUPC_SR; 	/* Status Register*/
	AT91_REG	 SUPC_FWUTR; 	/* Flash Wake-up Timer Register*/
	AT91_REG	 Reserved34[5]; 	/* */
	AT91_REG	 PITC_PIMR; 	/* Period Interval Mode Register*/
	AT91_REG	 PITC_PISR; 	/* Period Interval Status Register*/
	AT91_REG	 PITC_PIVR; 	/* Period Interval Value Register*/
	AT91_REG	 PITC_PIIR; 	/* Period Interval Image Register*/
	AT91_REG	 WDTC_WDCR; 	/* Watchdog Control Register*/
	AT91_REG	 WDTC_WDMR; 	/* Watchdog Mode Register*/
	AT91_REG	 WDTC_WDSR; 	/* Watchdog Status Register*/
	AT91_REG	 Reserved35[1]; 	/* */
	AT91_REG	 RTC_MR; 	/* Mode Register*/
	AT91_REG	 RTC_HMR; 	/* Hour Mode Register*/
	AT91_REG	 RTC_TIMR; 	/* Time Register*/
	AT91_REG	 RTC_CALR; 	/* Calendar Register*/
	AT91_REG	 RTC_TAR; 	/* Time Alarm Register*/
	AT91_REG	 RTC_CAR; 	/* Calendar Alarm Register*/
	AT91_REG	 RTC_SR; 	/* Status Register*/
	AT91_REG	 RTC_SCR; 	/* Status Clear Register*/
	AT91_REG	 RTC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 RTC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 RTC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 RTC_VER; 	/* Valid Entry Register*/
} AT91S_SYS, *AT91PS_SYS;


/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Advanced Interrupt Controller*/
/* ******************************************************************************/
typedef struct _AT91S_AIC {
	AT91_REG	 AIC_SMR[32]; 	/* Source Mode Register*/
	AT91_REG	 AIC_SVR[32]; 	/* Source Vector Register*/
	AT91_REG	 AIC_IVR; 	/* IRQ Vector Register*/
	AT91_REG	 AIC_FVR; 	/* FIQ Vector Register*/
	AT91_REG	 AIC_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 AIC_IPR; 	/* Interrupt Pending Register*/
	AT91_REG	 AIC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 AIC_CISR; 	/* Core Interrupt Status Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 AIC_IECR; 	/* Interrupt Enable Command Register*/
	AT91_REG	 AIC_IDCR; 	/* Interrupt Disable Command Register*/
	AT91_REG	 AIC_ICCR; 	/* Interrupt Clear Command Register*/
	AT91_REG	 AIC_ISCR; 	/* Interrupt Set Command Register*/
	AT91_REG	 AIC_EOICR; 	/* End of Interrupt Command Register*/
	AT91_REG	 AIC_SPU; 	/* Spurious Vector Register*/
	AT91_REG	 AIC_DCR; 	/* Debug Control Register (Protect)*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 AIC_FFER; 	/* Fast Forcing Enable Register*/
	AT91_REG	 AIC_FFDR; 	/* Fast Forcing Disable Register*/
	AT91_REG	 AIC_FFSR; 	/* Fast Forcing Status Register*/
} AT91S_AIC, *AT91PS_AIC;

/* -------- AIC_SMR : (AIC Offset: 0x0) Control Register -------- */
#define AT91C_AIC_PRIOR       ((unsigned int) 0x7 <<  0) /* (AIC) Priority Level*/
#define 	AT91C_AIC_PRIOR_LOWEST               ((unsigned int) 0x0) /* (AIC) Lowest priority level*/
#define 	AT91C_AIC_PRIOR_HIGHEST              ((unsigned int) 0x7) /* (AIC) Highest priority level*/
#define AT91C_AIC_SRCTYPE     ((unsigned int) 0x3 <<  5) /* (AIC) Interrupt Source Type*/
#define 	AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL       ((unsigned int) 0x0 <<  5) /* (AIC) Internal Sources Code Label High-level Sensitive*/
#define 	AT91C_AIC_SRCTYPE_EXT_LOW_LEVEL        ((unsigned int) 0x0 <<  5) /* (AIC) External Sources Code Label Low-level Sensitive*/
#define 	AT91C_AIC_SRCTYPE_INT_POSITIVE_EDGE    ((unsigned int) 0x1 <<  5) /* (AIC) Internal Sources Code Label Positive Edge triggered*/
#define 	AT91C_AIC_SRCTYPE_EXT_NEGATIVE_EDGE    ((unsigned int) 0x1 <<  5) /* (AIC) External Sources Code Label Negative Edge triggered*/
#define 	AT91C_AIC_SRCTYPE_HIGH_LEVEL           ((unsigned int) 0x2 <<  5) /* (AIC) Internal Or External Sources Code Label High-level Sensitive*/
#define 	AT91C_AIC_SRCTYPE_POSITIVE_EDGE        ((unsigned int) 0x3 <<  5) /* (AIC) Internal Or External Sources Code Label Positive Edge triggered*/
/* -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- */
#define AT91C_AIC_NFIQ        ((unsigned int) 0x1 <<  0) /* (AIC) NFIQ Status*/
#define AT91C_AIC_NIRQ        ((unsigned int) 0x1 <<  1) /* (AIC) NIRQ Status*/
/* -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- */
#define AT91C_AIC_DCR_PROT    ((unsigned int) 0x1 <<  0) /* (AIC) Protection Mode*/
#define AT91C_AIC_DCR_GMSK    ((unsigned int) 0x1 <<  1) /* (AIC) General Mask*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller*/
/* ******************************************************************************/
typedef struct _AT91S_PDC {
	AT91_REG	 PDC_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 PDC_RCR; 	/* Receive Counter Register*/
	AT91_REG	 PDC_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 PDC_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 PDC_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 PDC_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 PDC_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 PDC_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 PDC_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 PDC_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_PDC, *AT91PS_PDC;

/* -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- */
#define AT91C_PDC_RXTEN       ((unsigned int) 0x1 <<  0) /* (PDC) Receiver Transfer Enable*/
#define AT91C_PDC_RXTDIS      ((unsigned int) 0x1 <<  1) /* (PDC) Receiver Transfer Disable*/
#define AT91C_PDC_TXTEN       ((unsigned int) 0x1 <<  8) /* (PDC) Transmitter Transfer Enable*/
#define AT91C_PDC_TXTDIS      ((unsigned int) 0x1 <<  9) /* (PDC) Transmitter Transfer Disable*/
/* -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Debug Unit*/
/* ******************************************************************************/
typedef struct _AT91S_DBGU {
	AT91_REG	 DBGU_CR; 	/* Control Register*/
	AT91_REG	 DBGU_MR; 	/* Mode Register*/
	AT91_REG	 DBGU_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 DBGU_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 DBGU_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 DBGU_CSR; 	/* Channel Status Register*/
	AT91_REG	 DBGU_RHR; 	/* Receiver Holding Register*/
	AT91_REG	 DBGU_THR; 	/* Transmitter Holding Register*/
	AT91_REG	 DBGU_BRGR; 	/* Baud Rate Generator Register*/
	AT91_REG	 Reserved0[7]; 	/* */
	AT91_REG	 DBGU_CIDR; 	/* Chip ID Register*/
	AT91_REG	 DBGU_EXID; 	/* Chip ID Extension Register*/
	AT91_REG	 DBGU_FNTR; 	/* Force NTRST Register*/
	AT91_REG	 Reserved1[45]; 	/* */
	AT91_REG	 DBGU_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 DBGU_RCR; 	/* Receive Counter Register*/
	AT91_REG	 DBGU_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 DBGU_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 DBGU_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 DBGU_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 DBGU_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 DBGU_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 DBGU_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 DBGU_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_DBGU, *AT91PS_DBGU;

/* -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- */
#define AT91C_US_RSTRX        ((unsigned int) 0x1 <<  2) /* (DBGU) Reset Receiver*/
#define AT91C_US_RSTTX        ((unsigned int) 0x1 <<  3) /* (DBGU) Reset Transmitter*/
#define AT91C_US_RXEN         ((unsigned int) 0x1 <<  4) /* (DBGU) Receiver Enable*/
#define AT91C_US_RXDIS        ((unsigned int) 0x1 <<  5) /* (DBGU) Receiver Disable*/
#define AT91C_US_TXEN         ((unsigned int) 0x1 <<  6) /* (DBGU) Transmitter Enable*/
#define AT91C_US_TXDIS        ((unsigned int) 0x1 <<  7) /* (DBGU) Transmitter Disable*/
#define AT91C_US_RSTSTA       ((unsigned int) 0x1 <<  8) /* (DBGU) Reset Status Bits*/
/* -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- */
#define AT91C_US_PAR          ((unsigned int) 0x7 <<  9) /* (DBGU) Parity type*/
#define 	AT91C_US_PAR_EVEN                 ((unsigned int) 0x0 <<  9) /* (DBGU) Even Parity*/
#define 	AT91C_US_PAR_ODD                  ((unsigned int) 0x1 <<  9) /* (DBGU) Odd Parity*/
#define 	AT91C_US_PAR_SPACE                ((unsigned int) 0x2 <<  9) /* (DBGU) Parity forced to 0 (Space)*/
#define 	AT91C_US_PAR_MARK                 ((unsigned int) 0x3 <<  9) /* (DBGU) Parity forced to 1 (Mark)*/
#define 	AT91C_US_PAR_NONE                 ((unsigned int) 0x4 <<  9) /* (DBGU) No Parity*/
#define 	AT91C_US_PAR_MULTI_DROP           ((unsigned int) 0x6 <<  9) /* (DBGU) Multi-drop mode*/
#define AT91C_US_CHMODE       ((unsigned int) 0x3 << 14) /* (DBGU) Channel Mode*/
#define 	AT91C_US_CHMODE_NORMAL               ((unsigned int) 0x0 << 14) /* (DBGU) Normal Mode: The USART channel operates as an RX/TX USART.*/
#define 	AT91C_US_CHMODE_AUTO                 ((unsigned int) 0x1 << 14) /* (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.*/
#define 	AT91C_US_CHMODE_LOCAL                ((unsigned int) 0x2 << 14) /* (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.*/
#define 	AT91C_US_CHMODE_REMOTE               ((unsigned int) 0x3 << 14) /* (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.*/
/* -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
#define AT91C_US_RXRDY        ((unsigned int) 0x1 <<  0) /* (DBGU) RXRDY Interrupt*/
#define AT91C_US_TXRDY        ((unsigned int) 0x1 <<  1) /* (DBGU) TXRDY Interrupt*/
#define AT91C_US_ENDRX        ((unsigned int) 0x1 <<  3) /* (DBGU) End of Receive Transfer Interrupt*/
#define AT91C_US_ENDTX        ((unsigned int) 0x1 <<  4) /* (DBGU) End of Transmit Interrupt*/
#define AT91C_US_OVRE         ((unsigned int) 0x1 <<  5) /* (DBGU) Overrun Interrupt*/
#define AT91C_US_FRAME        ((unsigned int) 0x1 <<  6) /* (DBGU) Framing Error Interrupt*/
#define AT91C_US_PARE         ((unsigned int) 0x1 <<  7) /* (DBGU) Parity Error Interrupt*/
#define AT91C_US_TXEMPTY      ((unsigned int) 0x1 <<  9) /* (DBGU) TXEMPTY Interrupt*/
#define AT91C_US_TXBUFE       ((unsigned int) 0x1 << 11) /* (DBGU) TXBUFE Interrupt*/
#define AT91C_US_RXBUFF       ((unsigned int) 0x1 << 12) /* (DBGU) RXBUFF Interrupt*/
#define AT91C_US_COMM_TX      ((unsigned int) 0x1 << 30) /* (DBGU) COMM_TX Interrupt*/
#define AT91C_US_COMM_RX      ((unsigned int) 0x1 << 31) /* (DBGU) COMM_RX Interrupt*/
/* -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- */
/* -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- */
#define AT91C_US_FORCE_NTRST  ((unsigned int) 0x1 <<  0) /* (DBGU) Force NTRST in JTAG*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler*/
/* ******************************************************************************/
typedef struct _AT91S_PIO {
	AT91_REG	 PIO_PER; 	/* PIO Enable Register*/
	AT91_REG	 PIO_PDR; 	/* PIO Disable Register*/
	AT91_REG	 PIO_PSR; 	/* PIO Status Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 PIO_OER; 	/* Output Enable Register*/
	AT91_REG	 PIO_ODR; 	/* Output Disable Registerr*/
	AT91_REG	 PIO_OSR; 	/* Output Status Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 PIO_IFER; 	/* Input Filter Enable Register*/
	AT91_REG	 PIO_IFDR; 	/* Input Filter Disable Register*/
	AT91_REG	 PIO_IFSR; 	/* Input Filter Status Register*/
	AT91_REG	 Reserved2[1]; 	/* */
	AT91_REG	 PIO_SODR; 	/* Set Output Data Register*/
	AT91_REG	 PIO_CODR; 	/* Clear Output Data Register*/
	AT91_REG	 PIO_ODSR; 	/* Output Data Status Register*/
	AT91_REG	 PIO_PDSR; 	/* Pin Data Status Register*/
	AT91_REG	 PIO_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PIO_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PIO_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PIO_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 PIO_MDER; 	/* Multi-driver Enable Register*/
	AT91_REG	 PIO_MDDR; 	/* Multi-driver Disable Register*/
	AT91_REG	 PIO_MDSR; 	/* Multi-driver Status Register*/
	AT91_REG	 Reserved3[1]; 	/* */
	AT91_REG	 PIO_PPUDR; 	/* Pull-up Disable Register*/
	AT91_REG	 PIO_PPUER; 	/* Pull-up Enable Register*/
	AT91_REG	 PIO_PPUSR; 	/* Pull-up Status Register*/
	AT91_REG	 Reserved4[1]; 	/* */
	AT91_REG	 PIO_ASR; 	/* Select A Register*/
	AT91_REG	 PIO_BSR; 	/* Select B Register*/
	AT91_REG	 PIO_ABSR; 	/* AB Select Status Register*/
	AT91_REG	 Reserved5[9]; 	/* */
	AT91_REG	 PIO_OWER; 	/* Output Write Enable Register*/
	AT91_REG	 PIO_OWDR; 	/* Output Write Disable Register*/
	AT91_REG	 PIO_OWSR; 	/* Output Write Status Register*/
} AT91S_PIO, *AT91PS_PIO;


/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* ******************************************************************************/
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 CKGR_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 CKGR_PLLR; 	/* PLL Register*/
} AT91S_CKGR, *AT91PS_CKGR;

/* -------- CKGR_MOR : (CKGR Offset: 0x0) Main Oscillator Register -------- */
#define AT91C_CKGR_MAINCKON   ((unsigned int) 0x1 <<  0) /* (CKGR) RC 2 MHz Oscillator Enable (RC 2 MHz oscillator enabled at startup)*/
#define AT91C_CKGR_FKEY       ((unsigned int) 0xFF << 16) /* (CKGR) Clock Generator Controller Writing Protection Key*/
#define AT91C_CKGR_MCKSEL     ((unsigned int) 0x1 << 24) /* (CKGR) */
/* -------- CKGR_MCFR : (CKGR Offset: 0x4) Main Clock Frequency Register -------- */
#define AT91C_CKGR_MAINF      ((unsigned int) 0xFFFF <<  0) /* (CKGR) Main Clock Frequency*/
#define AT91C_CKGR_MAINRDY    ((unsigned int) 0x1 << 16) /* (CKGR) Main Clock Ready*/
/* -------- CKGR_PLLR : (CKGR Offset: 0x8) PLL A Register -------- */
#define AT91C_CKGR_DIV        ((unsigned int) 0xFF <<  0) /* (CKGR) Divider Selected*/
#define 	AT91C_CKGR_DIV_0                    ((unsigned int) 0x0) /* (CKGR) Divider output is 0*/
#define 	AT91C_CKGR_DIV_BYPASS               ((unsigned int) 0x1) /* (CKGR) Divider is bypassed*/
#define AT91C_CKGR_PLLCOUNT   ((unsigned int) 0x3F <<  8) /* (CKGR) PLL Counter*/
#define AT91C_CKGR_OUT        ((unsigned int) 0x3 << 14) /* (CKGR) PLL Output Frequency Range*/
#define 	AT91C_CKGR_OUT_0                    ((unsigned int) 0x0 << 14) /* (CKGR) Please refer to the PLL datasheet*/
#define 	AT91C_CKGR_OUT_1                    ((unsigned int) 0x1 << 14) /* (CKGR) Please refer to the PLL datasheet*/
#define 	AT91C_CKGR_OUT_2                    ((unsigned int) 0x2 << 14) /* (CKGR) Please refer to the PLL datasheet*/
#define 	AT91C_CKGR_OUT_3                    ((unsigned int) 0x3 << 14) /* (CKGR) Please refer to the PLL datasheet*/
#define AT91C_CKGR_MUL        ((unsigned int) 0x7FF << 16) /* (CKGR) PLL Multiplier*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Power Management Controler*/
/* ******************************************************************************/
typedef struct _AT91S_PMC {
	AT91_REG	 PMC_SCER; 	/* System Clock Enable Register*/
	AT91_REG	 PMC_SCDR; 	/* System Clock Disable Register*/
	AT91_REG	 PMC_SCSR; 	/* System Clock Status Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 PMC_PCER; 	/* Peripheral Clock Enable Register*/
	AT91_REG	 PMC_PCDR; 	/* Peripheral Clock Disable Register*/
	AT91_REG	 PMC_PCSR; 	/* Peripheral Clock Status Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 PMC_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 PMC_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 PMC_PLLR; 	/* PLL Register*/
	AT91_REG	 Reserved2[1]; 	/* */
	AT91_REG	 PMC_MCKR; 	/* Master Clock Register*/
	AT91_REG	 Reserved3[3]; 	/* */
	AT91_REG	 PMC_PCKR[3]; 	/* Programmable Clock Register*/
	AT91_REG	 Reserved4[5]; 	/* */
	AT91_REG	 PMC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PMC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PMC_SR; 	/* Status Register*/
	AT91_REG	 PMC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 PMC_FSMR; 	/* Fast Startup Mode Register*/
} AT91S_PMC, *AT91PS_PMC;

/* -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) /* (PMC) Processor Clock*/
#define AT91C_PMC_PCK0        ((unsigned int) 0x1 <<  8) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK1        ((unsigned int) 0x1 <<  9) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK2        ((unsigned int) 0x1 << 10) /* (PMC) Programmable Clock Output*/
/* -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- */
/* -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- */
/* -------- CKGR_PLLR : (PMC Offset: 0x28) PLL A Register -------- */
/* -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- */
#define AT91C_PMC_CSS         ((unsigned int) 0x3 <<  0) /* (PMC) Programmable Clock Selection*/
#define 	AT91C_PMC_CSS_SLOW_CLK             ((unsigned int) 0x0) /* (PMC) Slow Clock is selected*/
#define 	AT91C_PMC_CSS_MAIN_CLK             ((unsigned int) 0x1) /* (PMC) Main Clock is selected*/
#define 	AT91C_PMC_CSS_PLL_CLK              ((unsigned int) 0x2) /* (PMC) Clock from PLL is selected*/
#define AT91C_PMC_PRES        ((unsigned int) 0x7 <<  2) /* (PMC) Programmable Clock Prescaler*/
#define 	AT91C_PMC_PRES_CLK                  ((unsigned int) 0x0 <<  2) /* (PMC) Selected clock*/
#define 	AT91C_PMC_PRES_CLK_2                ((unsigned int) 0x1 <<  2) /* (PMC) Selected clock divided by 2*/
#define 	AT91C_PMC_PRES_CLK_4                ((unsigned int) 0x2 <<  2) /* (PMC) Selected clock divided by 4*/
#define 	AT91C_PMC_PRES_CLK_8                ((unsigned int) 0x3 <<  2) /* (PMC) Selected clock divided by 8*/
#define 	AT91C_PMC_PRES_CLK_16               ((unsigned int) 0x4 <<  2) /* (PMC) Selected clock divided by 16*/
#define 	AT91C_PMC_PRES_CLK_32               ((unsigned int) 0x5 <<  2) /* (PMC) Selected clock divided by 32*/
#define 	AT91C_PMC_PRES_CLK_64               ((unsigned int) 0x6 <<  2) /* (PMC) Selected clock divided by 64*/
/* -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
#define AT91C_PMC_MAINSELS    ((unsigned int) 0x1 <<  0) /* (PMC) Main Clock Selection Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCK        ((unsigned int) 0x1 <<  1) /* (PMC) PLL Status/Enable/Disable/Mask*/
#define AT91C_PMC_MCKRDY      ((unsigned int) 0x1 <<  3) /* (PMC) MCK_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK0RDY     ((unsigned int) 0x1 <<  8) /* (PMC) PCK0_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK1RDY     ((unsigned int) 0x1 <<  9) /* (PMC) PCK1_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK2RDY     ((unsigned int) 0x1 << 10) /* (PMC) PCK2_RDY Status/Enable/Disable/Mask*/
/* -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Reset Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_RSTC {
	AT91_REG	 RSTC_RCR; 	/* Reset Control Register*/
	AT91_REG	 RSTC_RSR; 	/* Reset Status Register*/
	AT91_REG	 RSTC_RMR; 	/* Reset Mode Register*/
} AT91S_RSTC, *AT91PS_RSTC;

/* -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- */
#define AT91C_RSTC_PROCRST    ((unsigned int) 0x1 <<  0) /* (RSTC) Processor Reset*/
#define AT91C_RSTC_PERRST     ((unsigned int) 0x1 <<  2) /* (RSTC) Peripheral Reset*/
#define AT91C_RSTC_EXTRST     ((unsigned int) 0x1 <<  3) /* (RSTC) External Reset*/
#define AT91C_RSTC_KEY        ((unsigned int) 0xFF << 24) /* (RSTC) Password*/
/* -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- */
#define AT91C_RSTC_URSTS      ((unsigned int) 0x1 <<  0) /* (RSTC) User Reset Status*/
#define AT91C_RSTC_BODSTS     ((unsigned int) 0x1 <<  1) /* (RSTC) Brownout Detection Status*/
#define AT91C_RSTC_RSTTYP     ((unsigned int) 0x7 <<  8) /* (RSTC) Reset Type*/
#define 	AT91C_RSTC_RSTTYP_POWERUP              ((unsigned int) 0x0 <<  8) /* (RSTC) Power-up Reset. VDDCORE rising.*/
#define 	AT91C_RSTC_RSTTYP_WAKEUP               ((unsigned int) 0x1 <<  8) /* (RSTC) WakeUp Reset. VDDCORE rising.*/
#define 	AT91C_RSTC_RSTTYP_WATCHDOG             ((unsigned int) 0x2 <<  8) /* (RSTC) Watchdog Reset. Watchdog overflow occured.*/
#define 	AT91C_RSTC_RSTTYP_SOFTWARE             ((unsigned int) 0x3 <<  8) /* (RSTC) Software Reset. Processor reset required by the software.*/
#define 	AT91C_RSTC_RSTTYP_USER                 ((unsigned int) 0x4 <<  8) /* (RSTC) User Reset. NRST pin detected low.*/
#define 	AT91C_RSTC_RSTTYP_BROWNOUT             ((unsigned int) 0x5 <<  8) /* (RSTC) Brownout Reset occured.*/
#define AT91C_RSTC_NRSTL      ((unsigned int) 0x1 << 16) /* (RSTC) NRST pin level*/
#define AT91C_RSTC_SRCMP      ((unsigned int) 0x1 << 17) /* (RSTC) Software Reset Command in Progress.*/
/* -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- */
#define AT91C_RSTC_URSTEN     ((unsigned int) 0x1 <<  0) /* (RSTC) User Reset Enable*/
#define AT91C_RSTC_URSTIEN    ((unsigned int) 0x1 <<  4) /* (RSTC) User Reset Interrupt Enable*/
#define AT91C_RSTC_ERSTL      ((unsigned int) 0xF <<  8) /* (RSTC) User Reset Length*/
#define AT91C_RSTC_BODIEN     ((unsigned int) 0x1 << 16) /* (RSTC) Brownout Detection Interrupt Enable*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm*/
/* ******************************************************************************/
typedef struct _AT91S_RTC {
	AT91_REG	 RTC_MR; 	/* Mode Register*/
	AT91_REG	 RTC_HMR; 	/* Hour Mode Register*/
	AT91_REG	 RTC_TIMR; 	/* Time Register*/
	AT91_REG	 RTC_CALR; 	/* Calendar Register*/
	AT91_REG	 RTC_TAR; 	/* Time Alarm Register*/
	AT91_REG	 RTC_CAR; 	/* Calendar Alarm Register*/
	AT91_REG	 RTC_SR; 	/* Status Register*/
	AT91_REG	 RTC_SCR; 	/* Status Clear Register*/
	AT91_REG	 RTC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 RTC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 RTC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 RTC_VER; 	/* Valid Entry Register*/
} AT91S_RTC, *AT91PS_RTC;

/* -------- RTC_MR : (RTC Offset: 0x0) RTC Mode Register -------- */
#define AT91C_RTC_UPDTIM      ((unsigned int) 0x1 <<  0) /* (RTC) Update Request Time Register*/
#define AT91C_RTC_UPDCAL      ((unsigned int) 0x1 <<  1) /* (RTC) Update Request Calendar Register*/
#define AT91C_RTC_TEVSEL      ((unsigned int) 0x3 <<  8) /* (RTC) Time Event Selection*/
#define 	AT91C_RTC_TEVSEL_MN_CHG               ((unsigned int) 0x0 <<  8) /* (RTC) Minute change.*/
#define 	AT91C_RTC_TEVSEL_HR_CHG               ((unsigned int) 0x1 <<  8) /* (RTC) Hour change.*/
#define 	AT91C_RTC_TEVSEL_EVDAY_MD             ((unsigned int) 0x2 <<  8) /* (RTC) Every day at midnight.*/
#define 	AT91C_RTC_TEVSEL_EVDAY_NOON           ((unsigned int) 0x3 <<  8) /* (RTC) Every day at noon.*/
#define AT91C_RTC_CEVSEL      ((unsigned int) 0x3 << 16) /* (RTC) Calendar Event Selection*/
#define 	AT91C_RTC_CEVSEL_WEEK_CHG             ((unsigned int) 0x0 << 16) /* (RTC) Week change (every Monday at time 00:00:00).*/
#define 	AT91C_RTC_CEVSEL_MONTH_CHG            ((unsigned int) 0x1 << 16) /* (RTC) Month change (every 01 of each month at time 00:00:00).*/
#define 	AT91C_RTC_CEVSEL_YEAR_CHG             ((unsigned int) 0x2 << 16) /* (RTC) Year change (every January 1 at time 00:00:00).*/
/* -------- RTC_HMR : (RTC Offset: 0x4) RTC Hour Mode Register -------- */
#define AT91C_RTC_HRMOD       ((unsigned int) 0x1 <<  0) /* (RTC) 12-24 hour Mode*/
/* -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- */
#define AT91C_RTC_SEC         ((unsigned int) 0x7F <<  0) /* (RTC) Current Second*/
#define AT91C_RTC_MIN         ((unsigned int) 0x7F <<  8) /* (RTC) Current Minute*/
#define AT91C_RTC_HOUR        ((unsigned int) 0x3F << 16) /* (RTC) Current Hour*/
#define AT91C_RTC_AMPM        ((unsigned int) 0x1 << 22) /* (RTC) Ante Meridiem, Post Meridiem Indicator*/
/* -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- */
#define AT91C_RTC_CENT        ((unsigned int) 0x3F <<  0) /* (RTC) Current Century*/
#define AT91C_RTC_YEAR        ((unsigned int) 0xFF <<  8) /* (RTC) Current Year*/
#define AT91C_RTC_MONTH       ((unsigned int) 0x1F << 16) /* (RTC) Current Month*/
#define AT91C_RTC_DAY         ((unsigned int) 0x7 << 21) /* (RTC) Current Day*/
#define AT91C_RTC_DATE        ((unsigned int) 0x3F << 24) /* (RTC) Current Date*/
/* -------- RTC_TAR : (RTC Offset: 0x10) RTC Time Alarm Register -------- */
#define AT91C_RTC_SECEN       ((unsigned int) 0x1 <<  7) /* (RTC) Second Alarm Enable*/
#define AT91C_RTC_MINEN       ((unsigned int) 0x1 << 15) /* (RTC) Minute Alarm*/
#define AT91C_RTC_HOUREN      ((unsigned int) 0x1 << 23) /* (RTC) Current Hour*/
/* -------- RTC_CAR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- */
#define AT91C_RTC_MTHEN       ((unsigned int) 0x1 << 23) /* (RTC) Month Alarm Enable*/
#define AT91C_RTC_DATEN       ((unsigned int) 0x1 << 31) /* (RTC) Date Alarm Enable*/
/* -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- */
#define AT91C_RTC_ACKUPD      ((unsigned int) 0x1 <<  0) /* (RTC) Acknowledge for Update*/
#define AT91C_RTC_ALARM       ((unsigned int) 0x1 <<  1) /* (RTC) Alarm Flag*/
#define AT91C_RTC_SECEV       ((unsigned int) 0x1 <<  2) /* (RTC) Second Event*/
#define AT91C_RTC_TIMEV       ((unsigned int) 0x1 <<  3) /* (RTC) Time Event*/
#define AT91C_RTC_CALEV       ((unsigned int) 0x1 <<  4) /* (RTC) Calendar event*/
/* -------- RTC_SCR : (RTC Offset: 0x1c) RTC Status Clear Register -------- */
/* -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- */
/* -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- */
/* -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- */
/* -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- */
#define AT91C_RTC_NVT         ((unsigned int) 0x1 <<  0) /* (RTC) Non valid Time*/
#define AT91C_RTC_NVC         ((unsigned int) 0x1 <<  1) /* (RTC) Non valid Calendar*/
#define AT91C_RTC_NVTAL       ((unsigned int) 0x1 <<  2) /* (RTC) Non valid time Alarm*/
#define AT91C_RTC_NVCAL       ((unsigned int) 0x1 <<  3) /* (RTC) Nonvalid Calendar Alarm*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Periodic Interval Timer Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_PITC {
	AT91_REG	 PITC_PIMR; 	/* Period Interval Mode Register*/
	AT91_REG	 PITC_PISR; 	/* Period Interval Status Register*/
	AT91_REG	 PITC_PIVR; 	/* Period Interval Value Register*/
	AT91_REG	 PITC_PIIR; 	/* Period Interval Image Register*/
} AT91S_PITC, *AT91PS_PITC;

/* -------- PITC_PIMR : (PITC Offset: 0x0) Periodic Interval Mode Register -------- */
#define AT91C_PITC_PIV        ((unsigned int) 0xFFFFF <<  0) /* (PITC) Periodic Interval Value*/
#define AT91C_PITC_PITEN      ((unsigned int) 0x1 << 24) /* (PITC) Periodic Interval Timer Enabled*/
#define AT91C_PITC_PITIEN     ((unsigned int) 0x1 << 25) /* (PITC) Periodic Interval Timer Interrupt Enable*/
/* -------- PITC_PISR : (PITC Offset: 0x4) Periodic Interval Status Register -------- */
#define AT91C_PITC_PITS       ((unsigned int) 0x1 <<  0) /* (PITC) Periodic Interval Timer Status*/
/* -------- PITC_PIVR : (PITC Offset: 0x8) Periodic Interval Value Register -------- */
#define AT91C_PITC_CPIV       ((unsigned int) 0xFFFFF <<  0) /* (PITC) Current Periodic Interval Value*/
#define AT91C_PITC_PICNT      ((unsigned int) 0xFFF << 20) /* (PITC) Periodic Interval Counter*/
/* -------- PITC_PIIR : (PITC Offset: 0xc) Periodic Interval Image Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_WDTC {
	AT91_REG	 WDTC_WDCR; 	/* Watchdog Control Register*/
	AT91_REG	 WDTC_WDMR; 	/* Watchdog Mode Register*/
	AT91_REG	 WDTC_WDSR; 	/* Watchdog Status Register*/
} AT91S_WDTC, *AT91PS_WDTC;

/* -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- */
#define AT91C_WDTC_WDRSTT     ((unsigned int) 0x1 <<  0) /* (WDTC) Watchdog Restart*/
#define AT91C_WDTC_KEY        ((unsigned int) 0xFF << 24) /* (WDTC) Watchdog KEY Password*/
/* -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- */
#define AT91C_WDTC_WDV        ((unsigned int) 0xFFF <<  0) /* (WDTC) Watchdog Timer Restart*/
#define AT91C_WDTC_WDFIEN     ((unsigned int) 0x1 << 12) /* (WDTC) Watchdog Fault Interrupt Enable*/
#define AT91C_WDTC_WDRSTEN    ((unsigned int) 0x1 << 13) /* (WDTC) Watchdog Reset Enable*/
#define AT91C_WDTC_WDRPROC    ((unsigned int) 0x1 << 14) /* (WDTC) Watchdog Timer Restart*/
#define AT91C_WDTC_WDDIS      ((unsigned int) 0x1 << 15) /* (WDTC) Watchdog Disable*/
#define AT91C_WDTC_WDD        ((unsigned int) 0xFFF << 16) /* (WDTC) Watchdog Delta Value*/
#define AT91C_WDTC_WDDBGHLT   ((unsigned int) 0x1 << 28) /* (WDTC) Watchdog Debug Halt*/
#define AT91C_WDTC_WDIDLEHLT  ((unsigned int) 0x1 << 29) /* (WDTC) Watchdog Idle Halt*/
/* -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- */
#define AT91C_WDTC_WDUNF      ((unsigned int) 0x1 <<  0) /* (WDTC) Watchdog Underflow*/
#define AT91C_WDTC_WDERR      ((unsigned int) 0x1 <<  1) /* (WDTC) Watchdog Error*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Voltage Regulator Mode Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_VREG {
	AT91_REG	 VREG_MR; 	/* Voltage Regulator Mode Register*/
} AT91S_VREG, *AT91PS_VREG;

/* -------- VREG_MR : (VREG Offset: 0x0) Voltage Regulator Mode Register -------- */
#define AT91C_VREG_PSTDBY     ((unsigned int) 0x1 <<  0) /* (VREG) Voltage Regulator Power Standby Mode*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Memory Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_MC {
	AT91_REG	 MC_RCR; 	/* MC Remap Control Register*/
	AT91_REG	 MC_ASR; 	/* MC Abort Status Register*/
	AT91_REG	 MC_AASR; 	/* MC Abort Address Status Register*/
	AT91_REG	 Reserved0[21]; 	/* */
	AT91_REG	 MC_FMR; 	/* MC Flash Mode Register*/
	AT91_REG	 MC_FCR; 	/* MC Flash Command Register*/
	AT91_REG	 MC_FSR; 	/* MC Flash Status Register*/
	AT91_REG	 MC_FRR; 	/* MC Flash Result Register*/
} AT91S_MC, *AT91PS_MC;

/* -------- MC_RCR : (MC Offset: 0x0) MC Remap Control Register -------- */
#define AT91C_MC_RCB          ((unsigned int) 0x1 <<  0) /* (MC) Remap Command Bit*/
/* -------- MC_ASR : (MC Offset: 0x4) MC Abort Status Register -------- */
#define AT91C_MC_UNDADD       ((unsigned int) 0x1 <<  0) /* (MC) Undefined Addess Abort Status*/
#define AT91C_MC_MISADD       ((unsigned int) 0x1 <<  1) /* (MC) Misaligned Addess Abort Status*/
#define AT91C_MC_ABTSZ        ((unsigned int) 0x3 <<  8) /* (MC) Abort Size Status*/
#define 	AT91C_MC_ABTSZ_BYTE                 ((unsigned int) 0x0 <<  8) /* (MC) Byte*/
#define 	AT91C_MC_ABTSZ_HWORD                ((unsigned int) 0x1 <<  8) /* (MC) Half-word*/
#define 	AT91C_MC_ABTSZ_WORD                 ((unsigned int) 0x2 <<  8) /* (MC) Word*/
#define AT91C_MC_ABTTYP       ((unsigned int) 0x3 << 10) /* (MC) Abort Type Status*/
#define 	AT91C_MC_ABTTYP_DATAR                ((unsigned int) 0x0 << 10) /* (MC) Data Read*/
#define 	AT91C_MC_ABTTYP_DATAW                ((unsigned int) 0x1 << 10) /* (MC) Data Write*/
#define 	AT91C_MC_ABTTYP_FETCH                ((unsigned int) 0x2 << 10) /* (MC) Code Fetch*/
#define AT91C_MC_MST0         ((unsigned int) 0x1 << 16) /* (MC) Master 0 Abort Source*/
#define AT91C_MC_MST1         ((unsigned int) 0x1 << 17) /* (MC) Master 1 Abort Source*/
#define AT91C_MC_SVMST0       ((unsigned int) 0x1 << 24) /* (MC) Saved Master 0 Abort Source*/
#define AT91C_MC_SVMST1       ((unsigned int) 0x1 << 25) /* (MC) Saved Master 1 Abort Source*/
/* -------- MC_FMR : (MC Offset: 0x60) MC Flash Mode Register -------- */
#define AT91C_MC_FRDY         ((unsigned int) 0x1 <<  0) /* (MC) Ready Interrupt Enable*/
#define AT91C_MC_FWS          ((unsigned int) 0xF <<  8) /* (MC) Flash Wait State*/
#define 	AT91C_MC_FWS_0FWS                 ((unsigned int) 0x0 <<  8) /* (MC) 0 Wait State*/
#define 	AT91C_MC_FWS_1FWS                 ((unsigned int) 0x1 <<  8) /* (MC) 1 Wait State*/
#define 	AT91C_MC_FWS_2FWS                 ((unsigned int) 0x2 <<  8) /* (MC) 2 Wait States*/
#define 	AT91C_MC_FWS_3FWS                 ((unsigned int) 0x3 <<  8) /* (MC) 3 Wait States*/
/* -------- MC_FCR : (MC Offset: 0x64) MC Flash Command Register -------- */
#define AT91C_MC_FCMD         ((unsigned int) 0xFF <<  0) /* (MC) Flash Command*/
#define 	AT91C_MC_FCMD_GETD                 ((unsigned int) 0x0) /* (MC) Get Flash Descriptor*/
#define 	AT91C_MC_FCMD_WP                   ((unsigned int) 0x1) /* (MC) Write Page*/
#define 	AT91C_MC_FCMD_WPL                  ((unsigned int) 0x2) /* (MC) Write Page and Lock*/
#define 	AT91C_MC_FCMD_EWP                  ((unsigned int) 0x3) /* (MC) Erase Page and Write Page*/
#define 	AT91C_MC_FCMD_EWPL                 ((unsigned int) 0x4) /* (MC) Erase Page and Write Page then Lock*/
#define 	AT91C_MC_FCMD_EA                   ((unsigned int) 0x5) /* (MC) Erase All*/
#define 	AT91C_MC_FCMD_EPL                  ((unsigned int) 0x6) /* (MC) Erase Plane*/
#define 	AT91C_MC_FCMD_EPA                  ((unsigned int) 0x7) /* (MC) Erase Pages*/
#define 	AT91C_MC_FCMD_SLB                  ((unsigned int) 0x8) /* (MC) Set Lock Bit*/
#define 	AT91C_MC_FCMD_CLB                  ((unsigned int) 0x9) /* (MC) Clear Lock Bit*/
#define 	AT91C_MC_FCMD_GLB                  ((unsigned int) 0xA) /* (MC) Get Lock Bit*/
#define 	AT91C_MC_FCMD_SFB                  ((unsigned int) 0xB) /* (MC) Set Fuse Bit*/
#define 	AT91C_MC_FCMD_CFB                  ((unsigned int) 0xC) /* (MC) Clear Fuse Bit*/
#define 	AT91C_MC_FCMD_GFB                  ((unsigned int) 0xD) /* (MC) Get Fuse Bit*/
#define AT91C_MC_FARG         ((unsigned int) 0xFFFF <<  8) /* (MC) Flash Command Argument*/
#define AT91C_MC_KEY          ((unsigned int) 0xFF << 24) /* (MC) Writing Protect Key*/
/* -------- MC_FSR : (MC Offset: 0x68) MC Flash Command Register -------- */
#define AT91C_MC_FRDY_S       ((unsigned int) 0x1 <<  0) /* (MC) Flash Ready Status*/
#define AT91C_MC_FCMDE        ((unsigned int) 0x1 <<  1) /* (MC) Flash Command Error Status*/
#define AT91C_MC_LOCKE        ((unsigned int) 0x1 <<  2) /* (MC) Flash Lock Error Status*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Serial Parallel Interface*/
/* ******************************************************************************/
typedef struct _AT91S_SPI {
	AT91_REG	 SPI_CR; 	/* Control Register*/
	AT91_REG	 SPI_MR; 	/* Mode Register*/
	AT91_REG	 SPI_RDR; 	/* Receive Data Register*/
	AT91_REG	 SPI_TDR; 	/* Transmit Data Register*/
	AT91_REG	 SPI_SR; 	/* Status Register*/
	AT91_REG	 SPI_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 SPI_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 SPI_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 Reserved0[4]; 	/* */
	AT91_REG	 SPI_CSR[4]; 	/* Chip Select Register*/
	AT91_REG	 Reserved1[48]; 	/* */
	AT91_REG	 SPI_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 SPI_RCR; 	/* Receive Counter Register*/
	AT91_REG	 SPI_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 SPI_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 SPI_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 SPI_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 SPI_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 SPI_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 SPI_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 SPI_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_SPI, *AT91PS_SPI;

/* -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- */
#define AT91C_SPI_SPIEN       ((unsigned int) 0x1 <<  0) /* (SPI) SPI Enable*/
#define AT91C_SPI_SPIDIS      ((unsigned int) 0x1 <<  1) /* (SPI) SPI Disable*/
#define AT91C_SPI_SWRST       ((unsigned int) 0x1 <<  7) /* (SPI) SPI Software reset*/
#define AT91C_SPI_LASTXFER    ((unsigned int) 0x1 << 24) /* (SPI) SPI Last Transfer*/
/* -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- */
#define AT91C_SPI_MSTR        ((unsigned int) 0x1 <<  0) /* (SPI) Master/Slave Mode*/
#define AT91C_SPI_PS          ((unsigned int) 0x1 <<  1) /* (SPI) Peripheral Select*/
#define 	AT91C_SPI_PS_FIXED                ((unsigned int) 0x0 <<  1) /* (SPI) Fixed Peripheral Select*/
#define 	AT91C_SPI_PS_VARIABLE             ((unsigned int) 0x1 <<  1) /* (SPI) Variable Peripheral Select*/
#define AT91C_SPI_PCSDEC      ((unsigned int) 0x1 <<  2) /* (SPI) Chip Select Decode*/
#define AT91C_SPI_FDIV        ((unsigned int) 0x1 <<  3) /* (SPI) Clock Selection*/
#define AT91C_SPI_MODFDIS     ((unsigned int) 0x1 <<  4) /* (SPI) Mode Fault Detection*/
#define AT91C_SPI_LLB         ((unsigned int) 0x1 <<  7) /* (SPI) Clock Selection*/
#define AT91C_SPI_PCS         ((unsigned int) 0xF << 16) /* (SPI) Peripheral Chip Select*/
#define AT91C_SPI_DLYBCS      ((unsigned int) 0xFF << 24) /* (SPI) Delay Between Chip Selects*/
/* -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- */
#define AT91C_SPI_RD          ((unsigned int) 0xFFFF <<  0) /* (SPI) Receive Data*/
#define AT91C_SPI_RPCS        ((unsigned int) 0xF << 16) /* (SPI) Peripheral Chip Select Status*/
/* -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- */
#define AT91C_SPI_TD          ((unsigned int) 0xFFFF <<  0) /* (SPI) Transmit Data*/
#define AT91C_SPI_TPCS        ((unsigned int) 0xF << 16) /* (SPI) Peripheral Chip Select Status*/
/* -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- */
#define AT91C_SPI_RDRF        ((unsigned int) 0x1 <<  0) /* (SPI) Receive Data Register Full*/
#define AT91C_SPI_TDRE        ((unsigned int) 0x1 <<  1) /* (SPI) Transmit Data Register Empty*/
#define AT91C_SPI_MODF        ((unsigned int) 0x1 <<  2) /* (SPI) Mode Fault Error*/
#define AT91C_SPI_OVRES       ((unsigned int) 0x1 <<  3) /* (SPI) Overrun Error Status*/
#define AT91C_SPI_ENDRX       ((unsigned int) 0x1 <<  4) /* (SPI) End of Receiver Transfer*/
#define AT91C_SPI_ENDTX       ((unsigned int) 0x1 <<  5) /* (SPI) End of Receiver Transfer*/
#define AT91C_SPI_RXBUFF      ((unsigned int) 0x1 <<  6) /* (SPI) RXBUFF Interrupt*/
#define AT91C_SPI_TXBUFE      ((unsigned int) 0x1 <<  7) /* (SPI) TXBUFE Interrupt*/
#define AT91C_SPI_NSSR        ((unsigned int) 0x1 <<  8) /* (SPI) NSSR Interrupt*/
#define AT91C_SPI_TXEMPTY     ((unsigned int) 0x1 <<  9) /* (SPI) TXEMPTY Interrupt*/
#define AT91C_SPI_SPIENS      ((unsigned int) 0x1 << 16) /* (SPI) Enable Status*/
/* -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- */
/* -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- */
/* -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- */
/* -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- */
#define AT91C_SPI_CPOL        ((unsigned int) 0x1 <<  0) /* (SPI) Clock Polarity*/
#define AT91C_SPI_NCPHA       ((unsigned int) 0x1 <<  1) /* (SPI) Clock Phase*/
#define AT91C_SPI_CSAAT       ((unsigned int) 0x1 <<  3) /* (SPI) Chip Select Active After Transfer*/
#define AT91C_SPI_BITS        ((unsigned int) 0xF <<  4) /* (SPI) Bits Per Transfer*/
#define 	AT91C_SPI_BITS_8                    ((unsigned int) 0x0 <<  4) /* (SPI) 8 Bits Per transfer*/
#define 	AT91C_SPI_BITS_9                    ((unsigned int) 0x1 <<  4) /* (SPI) 9 Bits Per transfer*/
#define 	AT91C_SPI_BITS_10                   ((unsigned int) 0x2 <<  4) /* (SPI) 10 Bits Per transfer*/
#define 	AT91C_SPI_BITS_11                   ((unsigned int) 0x3 <<  4) /* (SPI) 11 Bits Per transfer*/
#define 	AT91C_SPI_BITS_12                   ((unsigned int) 0x4 <<  4) /* (SPI) 12 Bits Per transfer*/
#define 	AT91C_SPI_BITS_13                   ((unsigned int) 0x5 <<  4) /* (SPI) 13 Bits Per transfer*/
#define 	AT91C_SPI_BITS_14                   ((unsigned int) 0x6 <<  4) /* (SPI) 14 Bits Per transfer*/
#define 	AT91C_SPI_BITS_15                   ((unsigned int) 0x7 <<  4) /* (SPI) 15 Bits Per transfer*/
#define 	AT91C_SPI_BITS_16                   ((unsigned int) 0x8 <<  4) /* (SPI) 16 Bits Per transfer*/
#define AT91C_SPI_SCBR        ((unsigned int) 0xFF <<  8) /* (SPI) Serial Clock Baud Rate*/
#define AT91C_SPI_DLYBS       ((unsigned int) 0xFF << 16) /* (SPI) Delay Before SPCK*/
#define AT91C_SPI_DLYBCT      ((unsigned int) 0xFF << 24) /* (SPI) Delay Between Consecutive Transfers*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Usart*/
/* ******************************************************************************/
typedef struct _AT91S_USART {
	AT91_REG	 US_CR; 	/* Control Register*/
	AT91_REG	 US_MR; 	/* Mode Register*/
	AT91_REG	 US_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 US_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 US_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 US_CSR; 	/* Channel Status Register*/
	AT91_REG	 US_RHR; 	/* Receiver Holding Register*/
	AT91_REG	 US_THR; 	/* Transmitter Holding Register*/
	AT91_REG	 US_BRGR; 	/* Baud Rate Generator Register*/
	AT91_REG	 US_RTOR; 	/* Receiver Time-out Register*/
	AT91_REG	 US_TTGR; 	/* Transmitter Time-guard Register*/
	AT91_REG	 Reserved0[5]; 	/* */
	AT91_REG	 US_FIDI; 	/* FI_DI_Ratio Register*/
	AT91_REG	 US_NER; 	/* Nb Errors Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 US_IF; 	/* IRDA_FILTER Register*/
	AT91_REG	 Reserved2[44]; 	/* */
	AT91_REG	 US_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 US_RCR; 	/* Receive Counter Register*/
	AT91_REG	 US_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 US_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 US_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 US_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 US_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 US_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 US_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 US_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_USART, *AT91PS_USART;

/* -------- US_CR : (USART Offset: 0x0) Debug Unit Control Register -------- */
#define AT91C_US_STTBRK       ((unsigned int) 0x1 <<  9) /* (USART) Start Break*/
#define AT91C_US_STPBRK       ((unsigned int) 0x1 << 10) /* (USART) Stop Break*/
#define AT91C_US_STTTO        ((unsigned int) 0x1 << 11) /* (USART) Start Time-out*/
#define AT91C_US_SENDA        ((unsigned int) 0x1 << 12) /* (USART) Send Address*/
#define AT91C_US_RSTIT        ((unsigned int) 0x1 << 13) /* (USART) Reset Iterations*/
#define AT91C_US_RSTNACK      ((unsigned int) 0x1 << 14) /* (USART) Reset Non Acknowledge*/
#define AT91C_US_RETTO        ((unsigned int) 0x1 << 15) /* (USART) Rearm Time-out*/
#define AT91C_US_DTREN        ((unsigned int) 0x1 << 16) /* (USART) Data Terminal ready Enable*/
#define AT91C_US_DTRDIS       ((unsigned int) 0x1 << 17) /* (USART) Data Terminal ready Disable*/
#define AT91C_US_RTSEN        ((unsigned int) 0x1 << 18) /* (USART) Request to Send enable*/
#define AT91C_US_RTSDIS       ((unsigned int) 0x1 << 19) /* (USART) Request to Send Disable*/
/* -------- US_MR : (USART Offset: 0x4) Debug Unit Mode Register -------- */
#define AT91C_US_USMODE       ((unsigned int) 0xF <<  0) /* (USART) Usart mode*/
#define 	AT91C_US_USMODE_NORMAL               ((unsigned int) 0x0) /* (USART) Normal*/
#define 	AT91C_US_USMODE_RS485                ((unsigned int) 0x1) /* (USART) RS485*/
#define 	AT91C_US_USMODE_HWHSH                ((unsigned int) 0x2) /* (USART) Hardware Handshaking*/
#define 	AT91C_US_USMODE_MODEM                ((unsigned int) 0x3) /* (USART) Modem*/
#define 	AT91C_US_USMODE_ISO7816_0            ((unsigned int) 0x4) /* (USART) ISO7816 protocol: T = 0*/
#define 	AT91C_US_USMODE_ISO7816_1            ((unsigned int) 0x6) /* (USART) ISO7816 protocol: T = 1*/
#define 	AT91C_US_USMODE_IRDA                 ((unsigned int) 0x8) /* (USART) IrDA*/
#define 	AT91C_US_USMODE_SWHSH                ((unsigned int) 0xC) /* (USART) Software Handshaking*/
#define AT91C_US_CLKS         ((unsigned int) 0x3 <<  4) /* (USART) Clock Selection (Baud Rate generator Input Clock*/
#define 	AT91C_US_CLKS_CLOCK                ((unsigned int) 0x0 <<  4) /* (USART) Clock*/
#define 	AT91C_US_CLKS_FDIV1                ((unsigned int) 0x1 <<  4) /* (USART) fdiv1*/
#define 	AT91C_US_CLKS_SLOW                 ((unsigned int) 0x2 <<  4) /* (USART) slow_clock (ARM)*/
#define 	AT91C_US_CLKS_EXT                  ((unsigned int) 0x3 <<  4) /* (USART) External (SCK)*/
#define AT91C_US_CHRL         ((unsigned int) 0x3 <<  6) /* (USART) Clock Selection (Baud Rate generator Input Clock*/
#define 	AT91C_US_CHRL_5_BITS               ((unsigned int) 0x0 <<  6) /* (USART) Character Length: 5 bits*/
#define 	AT91C_US_CHRL_6_BITS               ((unsigned int) 0x1 <<  6) /* (USART) Character Length: 6 bits*/
#define 	AT91C_US_CHRL_7_BITS               ((unsigned int) 0x2 <<  6) /* (USART) Character Length: 7 bits*/
#define 	AT91C_US_CHRL_8_BITS               ((unsigned int) 0x3 <<  6) /* (USART) Character Length: 8 bits*/
#define AT91C_US_SYNC         ((unsigned int) 0x1 <<  8) /* (USART) Synchronous Mode Select*/
#define AT91C_US_NBSTOP       ((unsigned int) 0x3 << 12) /* (USART) Number of Stop bits*/
#define 	AT91C_US_NBSTOP_1_BIT                ((unsigned int) 0x0 << 12) /* (USART) 1 stop bit*/
#define 	AT91C_US_NBSTOP_15_BIT               ((unsigned int) 0x1 << 12) /* (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits*/
#define 	AT91C_US_NBSTOP_2_BIT                ((unsigned int) 0x2 << 12) /* (USART) 2 stop bits*/
#define AT91C_US_MSBF         ((unsigned int) 0x1 << 16) /* (USART) Bit Order*/
#define AT91C_US_MODE9        ((unsigned int) 0x1 << 17) /* (USART) 9-bit Character length*/
#define AT91C_US_CKLO         ((unsigned int) 0x1 << 18) /* (USART) Clock Output Select*/
#define AT91C_US_OVER         ((unsigned int) 0x1 << 19) /* (USART) Over Sampling Mode*/
#define AT91C_US_INACK        ((unsigned int) 0x1 << 20) /* (USART) Inhibit Non Acknowledge*/
#define AT91C_US_DSNACK       ((unsigned int) 0x1 << 21) /* (USART) Disable Successive NACK*/
#define AT91C_US_MAX_ITER     ((unsigned int) 0x1 << 24) /* (USART) Number of Repetitions*/
#define AT91C_US_FILTER       ((unsigned int) 0x1 << 28) /* (USART) Receive Line Filter*/
/* -------- US_IER : (USART Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
#define AT91C_US_RXBRK        ((unsigned int) 0x1 <<  2) /* (USART) Break Received/End of Break*/
#define AT91C_US_TIMEOUT      ((unsigned int) 0x1 <<  8) /* (USART) Receiver Time-out*/
#define AT91C_US_ITERATION    ((unsigned int) 0x1 << 10) /* (USART) Max number of Repetitions Reached*/
#define AT91C_US_NACK         ((unsigned int) 0x1 << 13) /* (USART) Non Acknowledge*/
#define AT91C_US_RIIC         ((unsigned int) 0x1 << 16) /* (USART) Ring INdicator Input Change Flag*/
#define AT91C_US_DSRIC        ((unsigned int) 0x1 << 17) /* (USART) Data Set Ready Input Change Flag*/
#define AT91C_US_DCDIC        ((unsigned int) 0x1 << 18) /* (USART) Data Carrier Flag*/
#define AT91C_US_CTSIC        ((unsigned int) 0x1 << 19) /* (USART) Clear To Send Input Change Flag*/
/* -------- US_IDR : (USART Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* -------- US_IMR : (USART Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* -------- US_CSR : (USART Offset: 0x14) Debug Unit Channel Status Register -------- */
#define AT91C_US_RI           ((unsigned int) 0x1 << 20) /* (USART) Image of RI Input*/
#define AT91C_US_DSR          ((unsigned int) 0x1 << 21) /* (USART) Image of DSR Input*/
#define AT91C_US_DCD          ((unsigned int) 0x1 << 22) /* (USART) Image of DCD Input*/
#define AT91C_US_CTS          ((unsigned int) 0x1 << 23) /* (USART) Image of CTS Input*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Two-wire Interface*/
/* ******************************************************************************/
typedef struct _AT91S_TWI {
	AT91_REG	 TWI_CR; 	/* Control Register*/
	AT91_REG	 TWI_MMR; 	/* Master Mode Register*/
	AT91_REG	 TWI_SMR; 	/* Slave Mode Register*/
	AT91_REG	 TWI_IADR; 	/* Internal Address Register*/
	AT91_REG	 TWI_CWGR; 	/* Clock Waveform Generator Register*/
	AT91_REG	 Reserved0[3]; 	/* */
	AT91_REG	 TWI_SR; 	/* Status Register*/
	AT91_REG	 TWI_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 TWI_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 TWI_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 TWI_RHR; 	/* Receive Holding Register*/
	AT91_REG	 TWI_THR; 	/* Transmit Holding Register*/
	AT91_REG	 Reserved1[50]; 	/* */
	AT91_REG	 TWI_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 TWI_RCR; 	/* Receive Counter Register*/
	AT91_REG	 TWI_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 TWI_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 TWI_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 TWI_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 TWI_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 TWI_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 TWI_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 TWI_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_TWI, *AT91PS_TWI;

/* -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- */
#define AT91C_TWI_START       ((unsigned int) 0x1 <<  0) /* (TWI) Send a START Condition*/
#define AT91C_TWI_STOP        ((unsigned int) 0x1 <<  1) /* (TWI) Send a STOP Condition*/
#define AT91C_TWI_MSEN        ((unsigned int) 0x1 <<  2) /* (TWI) TWI Master Transfer Enabled*/
#define AT91C_TWI_MSDIS       ((unsigned int) 0x1 <<  3) /* (TWI) TWI Master Transfer Disabled*/
#define AT91C_TWI_SVEN        ((unsigned int) 0x1 <<  4) /* (TWI) TWI Slave mode Enabled*/
#define AT91C_TWI_SVDIS       ((unsigned int) 0x1 <<  5) /* (TWI) TWI Slave mode Disabled*/
#define AT91C_TWI_SWRST       ((unsigned int) 0x1 <<  7) /* (TWI) Software Reset*/
/* -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- */
#define AT91C_TWI_IADRSZ      ((unsigned int) 0x3 <<  8) /* (TWI) Internal Device Address Size*/
#define 	AT91C_TWI_IADRSZ_NO                   ((unsigned int) 0x0 <<  8) /* (TWI) No internal device address*/
#define 	AT91C_TWI_IADRSZ_1_BYTE               ((unsigned int) 0x1 <<  8) /* (TWI) One-byte internal device address*/
#define 	AT91C_TWI_IADRSZ_2_BYTE               ((unsigned int) 0x2 <<  8) /* (TWI) Two-byte internal device address*/
#define 	AT91C_TWI_IADRSZ_3_BYTE               ((unsigned int) 0x3 <<  8) /* (TWI) Three-byte internal device address*/
#define AT91C_TWI_MREAD       ((unsigned int) 0x1 << 12) /* (TWI) Master Read Direction*/
#define AT91C_TWI_DADR        ((unsigned int) 0x7F << 16) /* (TWI) Device Address*/
/* -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- */
#define AT91C_TWI_SADR        ((unsigned int) 0x7F << 16) /* (TWI) Slave Address*/
/* -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- */
#define AT91C_TWI_CLDIV       ((unsigned int) 0xFF <<  0) /* (TWI) Clock Low Divider*/
#define AT91C_TWI_CHDIV       ((unsigned int) 0xFF <<  8) /* (TWI) Clock High Divider*/
#define AT91C_TWI_CKDIV       ((unsigned int) 0x7 << 16) /* (TWI) Clock Divider*/
/* -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- */
#define AT91C_TWI_TXCOMP_SLAVE ((unsigned int) 0x1 <<  0) /* (TWI) Transmission Completed*/
#define AT91C_TWI_TXCOMP_MASTER ((unsigned int) 0x1 <<  0) /* (TWI) Transmission Completed*/
#define AT91C_TWI_RXRDY       ((unsigned int) 0x1 <<  1) /* (TWI) Receive holding register ReaDY*/
#define AT91C_TWI_TXRDY_MASTER ((unsigned int) 0x1 <<  2) /* (TWI) Transmit holding register ReaDY*/
#define AT91C_TWI_TXRDY_SLAVE ((unsigned int) 0x1 <<  2) /* (TWI) Transmit holding register ReaDY*/
#define AT91C_TWI_SVREAD      ((unsigned int) 0x1 <<  3) /* (TWI) Slave READ (used only in Slave mode)*/
#define AT91C_TWI_SVACC       ((unsigned int) 0x1 <<  4) /* (TWI) Slave ACCess (used only in Slave mode)*/
#define AT91C_TWI_GACC        ((unsigned int) 0x1 <<  5) /* (TWI) General Call ACcess (used only in Slave mode)*/
#define AT91C_TWI_OVRE        ((unsigned int) 0x1 <<  6) /* (TWI) Overrun Error (used only in Master and Multi-master mode)*/
#define AT91C_TWI_NACK_SLAVE  ((unsigned int) 0x1 <<  8) /* (TWI) Not Acknowledged*/
#define AT91C_TWI_NACK_MASTER ((unsigned int) 0x1 <<  8) /* (TWI) Not Acknowledged*/
#define AT91C_TWI_ARBLST_MULTI_MASTER ((unsigned int) 0x1 <<  9) /* (TWI) Arbitration Lost (used only in Multimaster mode)*/
#define AT91C_TWI_SCLWS       ((unsigned int) 0x1 << 10) /* (TWI) Clock Wait State (used only in Slave mode)*/
#define AT91C_TWI_EOSACC      ((unsigned int) 0x1 << 11) /* (TWI) End Of Slave ACCess (used only in Slave mode)*/
#define AT91C_TWI_ENDRX       ((unsigned int) 0x1 << 12) /* (TWI) End of Receiver Transfer*/
#define AT91C_TWI_ENDTX       ((unsigned int) 0x1 << 13) /* (TWI) End of Receiver Transfer*/
#define AT91C_TWI_RXBUFF      ((unsigned int) 0x1 << 14) /* (TWI) RXBUFF Interrupt*/
#define AT91C_TWI_TXBUFE      ((unsigned int) 0x1 << 15) /* (TWI) TXBUFE Interrupt*/
/* -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- */
/* -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- */
/* -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR PWMC Channel Interface*/
/* ******************************************************************************/
typedef struct _AT91S_PWMC_CH {
	AT91_REG	 PWMC_CMR; 	/* Channel Mode Register*/
	AT91_REG	 PWMC_CDTYR; 	/* Channel Duty Cycle Register*/
	AT91_REG	 PWMC_CPRDR; 	/* Channel Period Register*/
	AT91_REG	 PWMC_CCNTR; 	/* Channel Counter Register*/
	AT91_REG	 PWMC_CUPDR; 	/* Channel Update Register*/
	AT91_REG	 PWMC_Reserved[3]; 	/* Reserved*/
} AT91S_PWMC_CH, *AT91PS_PWMC_CH;

/* -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- */
#define AT91C_PWMC_CPRE       ((unsigned int) 0xF <<  0) /* (PWMC_CH) Channel Pre-scaler : PWMC_CLKx*/
#define 	AT91C_PWMC_CPRE_MCK                  ((unsigned int) 0x0) /* (PWMC_CH) */
#define 	AT91C_PWMC_CPRE_MCKA                 ((unsigned int) 0xB) /* (PWMC_CH) */
#define 	AT91C_PWMC_CPRE_MCKB                 ((unsigned int) 0xC) /* (PWMC_CH) */
#define AT91C_PWMC_CALG       ((unsigned int) 0x1 <<  8) /* (PWMC_CH) Channel Alignment*/
#define AT91C_PWMC_CPOL       ((unsigned int) 0x1 <<  9) /* (PWMC_CH) Channel Polarity*/
#define AT91C_PWMC_CPD        ((unsigned int) 0x1 << 10) /* (PWMC_CH) Channel Update Period*/
/* -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- */
#define AT91C_PWMC_CDTY       ((unsigned int) 0x0 <<  0) /* (PWMC_CH) Channel Duty Cycle*/
/* -------- PWMC_CPRDR : (PWMC_CH Offset: 0x8) PWMC Channel Period Register -------- */
#define AT91C_PWMC_CPRD       ((unsigned int) 0x0 <<  0) /* (PWMC_CH) Channel Period*/
/* -------- PWMC_CCNTR : (PWMC_CH Offset: 0xc) PWMC Channel Counter Register -------- */
#define AT91C_PWMC_CCNT       ((unsigned int) 0x0 <<  0) /* (PWMC_CH) Channel Counter*/
/* -------- PWMC_CUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Update Register -------- */
#define AT91C_PWMC_CUPD       ((unsigned int) 0x0 <<  0) /* (PWMC_CH) Channel Update*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_PWMC {
	AT91_REG	 PWMC_MR; 	/* PWMC Mode Register*/
	AT91_REG	 PWMC_ENA; 	/* PWMC Enable Register*/
	AT91_REG	 PWMC_DIS; 	/* PWMC Disable Register*/
	AT91_REG	 PWMC_SR; 	/* PWMC Status Register*/
	AT91_REG	 PWMC_IER; 	/* PWMC Interrupt Enable Register*/
	AT91_REG	 PWMC_IDR; 	/* PWMC Interrupt Disable Register*/
	AT91_REG	 PWMC_IMR; 	/* PWMC Interrupt Mask Register*/
	AT91_REG	 PWMC_ISR; 	/* PWMC Interrupt Status Register*/
	AT91_REG	 Reserved0[55]; 	/* */
	AT91_REG	 PWMC_VR; 	/* PWMC Version Register*/
	AT91_REG	 Reserved1[64]; 	/* */
	AT91S_PWMC_CH	 PWMC_CH[4]; 	/* PWMC Channel*/
} AT91S_PWMC, *AT91PS_PWMC;

/* -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- */
#define AT91C_PWMC_DIVA       ((unsigned int) 0xFF <<  0) /* (PWMC) CLKA divide factor.*/
#define AT91C_PWMC_PREA       ((unsigned int) 0xF <<  8) /* (PWMC) Divider Input Clock Prescaler A*/
#define 	AT91C_PWMC_PREA_MCK                  ((unsigned int) 0x0 <<  8) /* (PWMC) */
#define AT91C_PWMC_DIVB       ((unsigned int) 0xFF << 16) /* (PWMC) CLKB divide factor.*/
#define AT91C_PWMC_PREB       ((unsigned int) 0xF << 24) /* (PWMC) Divider Input Clock Prescaler B*/
#define 	AT91C_PWMC_PREB_MCK                  ((unsigned int) 0x0 << 24) /* (PWMC) */
/* -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- */
#define AT91C_PWMC_CHID0      ((unsigned int) 0x1 <<  0) /* (PWMC) Channel ID 0*/
#define AT91C_PWMC_CHID1      ((unsigned int) 0x1 <<  1) /* (PWMC) Channel ID 1*/
#define AT91C_PWMC_CHID2      ((unsigned int) 0x1 <<  2) /* (PWMC) Channel ID 2*/
#define AT91C_PWMC_CHID3      ((unsigned int) 0x1 <<  3) /* (PWMC) Channel ID 3*/
/* -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- */
/* -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- */
/* -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- */
/* -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- */
/* -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- */
/* -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface*/
/* ******************************************************************************/
typedef struct _AT91S_TC {
	AT91_REG	 TC_CCR; 	/* Channel Control Register*/
	AT91_REG	 TC_CMR; 	/* Channel Mode Register (Capture Mode / Waveform Mode)*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 TC_CV; 	/* Counter Value*/
	AT91_REG	 TC_RA; 	/* Register A*/
	AT91_REG	 TC_RB; 	/* Register B*/
	AT91_REG	 TC_RC; 	/* Register C*/
	AT91_REG	 TC_SR; 	/* Status Register*/
	AT91_REG	 TC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 TC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 TC_IMR; 	/* Interrupt Mask Register*/
} AT91S_TC, *AT91PS_TC;

/* -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- */
#define AT91C_TC_CLKEN        ((unsigned int) 0x1 <<  0) /* (TC) Counter Clock Enable Command*/
#define AT91C_TC_CLKDIS       ((unsigned int) 0x1 <<  1) /* (TC) Counter Clock Disable Command*/
#define AT91C_TC_SWTRG        ((unsigned int) 0x1 <<  2) /* (TC) Software Trigger Command*/
/* -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- */
#define AT91C_TC_CLKS         ((unsigned int) 0x7 <<  0) /* (TC) Clock Selection*/
#define 	AT91C_TC_CLKS_TIMER_DIV1_CLOCK     ((unsigned int) 0x0) /* (TC) Clock selected: TIMER_DIV1_CLOCK*/
#define 	AT91C_TC_CLKS_TIMER_DIV2_CLOCK     ((unsigned int) 0x1) /* (TC) Clock selected: TIMER_DIV2_CLOCK*/
#define 	AT91C_TC_CLKS_TIMER_DIV3_CLOCK     ((unsigned int) 0x2) /* (TC) Clock selected: TIMER_DIV3_CLOCK*/
#define 	AT91C_TC_CLKS_TIMER_DIV4_CLOCK     ((unsigned int) 0x3) /* (TC) Clock selected: TIMER_DIV4_CLOCK*/
#define 	AT91C_TC_CLKS_TIMER_DIV5_CLOCK     ((unsigned int) 0x4) /* (TC) Clock selected: TIMER_DIV5_CLOCK*/
#define 	AT91C_TC_CLKS_XC0                  ((unsigned int) 0x5) /* (TC) Clock selected: XC0*/
#define 	AT91C_TC_CLKS_XC1                  ((unsigned int) 0x6) /* (TC) Clock selected: XC1*/
#define 	AT91C_TC_CLKS_XC2                  ((unsigned int) 0x7) /* (TC) Clock selected: XC2*/
#define AT91C_TC_CLKI         ((unsigned int) 0x1 <<  3) /* (TC) Clock Invert*/
#define AT91C_TC_BURST        ((unsigned int) 0x3 <<  4) /* (TC) Burst Signal Selection*/
#define 	AT91C_TC_BURST_NONE                 ((unsigned int) 0x0 <<  4) /* (TC) The clock is not gated by an external signal*/
#define 	AT91C_TC_BURST_XC0                  ((unsigned int) 0x1 <<  4) /* (TC) XC0 is ANDed with the selected clock*/
#define 	AT91C_TC_BURST_XC1                  ((unsigned int) 0x2 <<  4) /* (TC) XC1 is ANDed with the selected clock*/
#define 	AT91C_TC_BURST_XC2                  ((unsigned int) 0x3 <<  4) /* (TC) XC2 is ANDed with the selected clock*/
#define AT91C_TC_CPCSTOP      ((unsigned int) 0x1 <<  6) /* (TC) Counter Clock Stopped with RC Compare*/
#define AT91C_TC_LDBSTOP      ((unsigned int) 0x1 <<  6) /* (TC) Counter Clock Stopped with RB Loading*/
#define AT91C_TC_CPCDIS       ((unsigned int) 0x1 <<  7) /* (TC) Counter Clock Disable with RC Compare*/
#define AT91C_TC_LDBDIS       ((unsigned int) 0x1 <<  7) /* (TC) Counter Clock Disabled with RB Loading*/
#define AT91C_TC_ETRGEDG      ((unsigned int) 0x3 <<  8) /* (TC) External Trigger Edge Selection*/
#define 	AT91C_TC_ETRGEDG_NONE                 ((unsigned int) 0x0 <<  8) /* (TC) Edge: None*/
#define 	AT91C_TC_ETRGEDG_RISING               ((unsigned int) 0x1 <<  8) /* (TC) Edge: rising edge*/
#define 	AT91C_TC_ETRGEDG_FALLING              ((unsigned int) 0x2 <<  8) /* (TC) Edge: falling edge*/
#define 	AT91C_TC_ETRGEDG_BOTH                 ((unsigned int) 0x3 <<  8) /* (TC) Edge: each edge*/
#define AT91C_TC_EEVTEDG      ((unsigned int) 0x3 <<  8) /* (TC) External Event Edge Selection*/
#define 	AT91C_TC_EEVTEDG_NONE                 ((unsigned int) 0x0 <<  8) /* (TC) Edge: None*/
#define 	AT91C_TC_EEVTEDG_RISING               ((unsigned int) 0x1 <<  8) /* (TC) Edge: rising edge*/
#define 	AT91C_TC_EEVTEDG_FALLING              ((unsigned int) 0x2 <<  8) /* (TC) Edge: falling edge*/
#define 	AT91C_TC_EEVTEDG_BOTH                 ((unsigned int) 0x3 <<  8) /* (TC) Edge: each edge*/
#define AT91C_TC_EEVT         ((unsigned int) 0x3 << 10) /* (TC) External Event  Selection*/
#define 	AT91C_TC_EEVT_TIOB                 ((unsigned int) 0x0 << 10) /* (TC) Signal selected as external event: TIOB TIOB direction: input*/
#define 	AT91C_TC_EEVT_XC0                  ((unsigned int) 0x1 << 10) /* (TC) Signal selected as external event: XC0 TIOB direction: output*/
#define 	AT91C_TC_EEVT_XC1                  ((unsigned int) 0x2 << 10) /* (TC) Signal selected as external event: XC1 TIOB direction: output*/
#define 	AT91C_TC_EEVT_XC2                  ((unsigned int) 0x3 << 10) /* (TC) Signal selected as external event: XC2 TIOB direction: output*/
#define AT91C_TC_ABETRG       ((unsigned int) 0x1 << 10) /* (TC) TIOA or TIOB External Trigger Selection*/
#define AT91C_TC_ENETRG       ((unsigned int) 0x1 << 12) /* (TC) External Event Trigger enable*/
#define AT91C_TC_WAVESEL      ((unsigned int) 0x3 << 13) /* (TC) Waveform  Selection*/
#define 	AT91C_TC_WAVESEL_UP                   ((unsigned int) 0x0 << 13) /* (TC) UP mode without atomatic trigger on RC Compare*/
#define 	AT91C_TC_WAVESEL_UPDOWN               ((unsigned int) 0x1 << 13) /* (TC) UPDOWN mode without automatic trigger on RC Compare*/
#define 	AT91C_TC_WAVESEL_UP_AUTO              ((unsigned int) 0x2 << 13) /* (TC) UP mode with automatic trigger on RC Compare*/
#define 	AT91C_TC_WAVESEL_UPDOWN_AUTO          ((unsigned int) 0x3 << 13) /* (TC) UPDOWN mode with automatic trigger on RC Compare*/
#define AT91C_TC_CPCTRG       ((unsigned int) 0x1 << 14) /* (TC) RC Compare Trigger Enable*/
#define AT91C_TC_WAVE         ((unsigned int) 0x1 << 15) /* (TC) */
#define AT91C_TC_ACPA         ((unsigned int) 0x3 << 16) /* (TC) RA Compare Effect on TIOA*/
#define 	AT91C_TC_ACPA_NONE                 ((unsigned int) 0x0 << 16) /* (TC) Effect: none*/
#define 	AT91C_TC_ACPA_SET                  ((unsigned int) 0x1 << 16) /* (TC) Effect: set*/
#define 	AT91C_TC_ACPA_CLEAR                ((unsigned int) 0x2 << 16) /* (TC) Effect: clear*/
#define 	AT91C_TC_ACPA_TOGGLE               ((unsigned int) 0x3 << 16) /* (TC) Effect: toggle*/
#define AT91C_TC_LDRA         ((unsigned int) 0x3 << 16) /* (TC) RA Loading Selection*/
#define 	AT91C_TC_LDRA_NONE                 ((unsigned int) 0x0 << 16) /* (TC) Edge: None*/
#define 	AT91C_TC_LDRA_RISING               ((unsigned int) 0x1 << 16) /* (TC) Edge: rising edge of TIOA*/
#define 	AT91C_TC_LDRA_FALLING              ((unsigned int) 0x2 << 16) /* (TC) Edge: falling edge of TIOA*/
#define 	AT91C_TC_LDRA_BOTH                 ((unsigned int) 0x3 << 16) /* (TC) Edge: each edge of TIOA*/
#define AT91C_TC_ACPC         ((unsigned int) 0x3 << 18) /* (TC) RC Compare Effect on TIOA*/
#define 	AT91C_TC_ACPC_NONE                 ((unsigned int) 0x0 << 18) /* (TC) Effect: none*/
#define 	AT91C_TC_ACPC_SET                  ((unsigned int) 0x1 << 18) /* (TC) Effect: set*/
#define 	AT91C_TC_ACPC_CLEAR                ((unsigned int) 0x2 << 18) /* (TC) Effect: clear*/
#define 	AT91C_TC_ACPC_TOGGLE               ((unsigned int) 0x3 << 18) /* (TC) Effect: toggle*/
#define AT91C_TC_LDRB         ((unsigned int) 0x3 << 18) /* (TC) RB Loading Selection*/
#define 	AT91C_TC_LDRB_NONE                 ((unsigned int) 0x0 << 18) /* (TC) Edge: None*/
#define 	AT91C_TC_LDRB_RISING               ((unsigned int) 0x1 << 18) /* (TC) Edge: rising edge of TIOA*/
#define 	AT91C_TC_LDRB_FALLING              ((unsigned int) 0x2 << 18) /* (TC) Edge: falling edge of TIOA*/
#define 	AT91C_TC_LDRB_BOTH                 ((unsigned int) 0x3 << 18) /* (TC) Edge: each edge of TIOA*/
#define AT91C_TC_AEEVT        ((unsigned int) 0x3 << 20) /* (TC) External Event Effect on TIOA*/
#define 	AT91C_TC_AEEVT_NONE                 ((unsigned int) 0x0 << 20) /* (TC) Effect: none*/
#define 	AT91C_TC_AEEVT_SET                  ((unsigned int) 0x1 << 20) /* (TC) Effect: set*/
#define 	AT91C_TC_AEEVT_CLEAR                ((unsigned int) 0x2 << 20) /* (TC) Effect: clear*/
#define 	AT91C_TC_AEEVT_TOGGLE               ((unsigned int) 0x3 << 20) /* (TC) Effect: toggle*/
#define AT91C_TC_ASWTRG       ((unsigned int) 0x3 << 22) /* (TC) Software Trigger Effect on TIOA*/
#define 	AT91C_TC_ASWTRG_NONE                 ((unsigned int) 0x0 << 22) /* (TC) Effect: none*/
#define 	AT91C_TC_ASWTRG_SET                  ((unsigned int) 0x1 << 22) /* (TC) Effect: set*/
#define 	AT91C_TC_ASWTRG_CLEAR                ((unsigned int) 0x2 << 22) /* (TC) Effect: clear*/
#define 	AT91C_TC_ASWTRG_TOGGLE               ((unsigned int) 0x3 << 22) /* (TC) Effect: toggle*/
#define AT91C_TC_BCPB         ((unsigned int) 0x3 << 24) /* (TC) RB Compare Effect on TIOB*/
#define 	AT91C_TC_BCPB_NONE                 ((unsigned int) 0x0 << 24) /* (TC) Effect: none*/
#define 	AT91C_TC_BCPB_SET                  ((unsigned int) 0x1 << 24) /* (TC) Effect: set*/
#define 	AT91C_TC_BCPB_CLEAR                ((unsigned int) 0x2 << 24) /* (TC) Effect: clear*/
#define 	AT91C_TC_BCPB_TOGGLE               ((unsigned int) 0x3 << 24) /* (TC) Effect: toggle*/
#define AT91C_TC_BCPC         ((unsigned int) 0x3 << 26) /* (TC) RC Compare Effect on TIOB*/
#define 	AT91C_TC_BCPC_NONE                 ((unsigned int) 0x0 << 26) /* (TC) Effect: none*/
#define 	AT91C_TC_BCPC_SET                  ((unsigned int) 0x1 << 26) /* (TC) Effect: set*/
#define 	AT91C_TC_BCPC_CLEAR                ((unsigned int) 0x2 << 26) /* (TC) Effect: clear*/
#define 	AT91C_TC_BCPC_TOGGLE               ((unsigned int) 0x3 << 26) /* (TC) Effect: toggle*/
#define AT91C_TC_BEEVT        ((unsigned int) 0x3 << 28) /* (TC) External Event Effect on TIOB*/
#define 	AT91C_TC_BEEVT_NONE                 ((unsigned int) 0x0 << 28) /* (TC) Effect: none*/
#define 	AT91C_TC_BEEVT_SET                  ((unsigned int) 0x1 << 28) /* (TC) Effect: set*/
#define 	AT91C_TC_BEEVT_CLEAR                ((unsigned int) 0x2 << 28) /* (TC) Effect: clear*/
#define 	AT91C_TC_BEEVT_TOGGLE               ((unsigned int) 0x3 << 28) /* (TC) Effect: toggle*/
#define AT91C_TC_BSWTRG       ((unsigned int) 0x3 << 30) /* (TC) Software Trigger Effect on TIOB*/
#define 	AT91C_TC_BSWTRG_NONE                 ((unsigned int) 0x0 << 30) /* (TC) Effect: none*/
#define 	AT91C_TC_BSWTRG_SET                  ((unsigned int) 0x1 << 30) /* (TC) Effect: set*/
#define 	AT91C_TC_BSWTRG_CLEAR                ((unsigned int) 0x2 << 30) /* (TC) Effect: clear*/
#define 	AT91C_TC_BSWTRG_TOGGLE               ((unsigned int) 0x3 << 30) /* (TC) Effect: toggle*/
/* -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- */
#define AT91C_TC_COVFS        ((unsigned int) 0x1 <<  0) /* (TC) Counter Overflow*/
#define AT91C_TC_LOVRS        ((unsigned int) 0x1 <<  1) /* (TC) Load Overrun*/
#define AT91C_TC_CPAS         ((unsigned int) 0x1 <<  2) /* (TC) RA Compare*/
#define AT91C_TC_CPBS         ((unsigned int) 0x1 <<  3) /* (TC) RB Compare*/
#define AT91C_TC_CPCS         ((unsigned int) 0x1 <<  4) /* (TC) RC Compare*/
#define AT91C_TC_LDRAS        ((unsigned int) 0x1 <<  5) /* (TC) RA Loading*/
#define AT91C_TC_LDRBS        ((unsigned int) 0x1 <<  6) /* (TC) RB Loading*/
#define AT91C_TC_ETRGS        ((unsigned int) 0x1 <<  7) /* (TC) External Trigger*/
#define AT91C_TC_CLKSTA       ((unsigned int) 0x1 << 16) /* (TC) Clock Enabling*/
#define AT91C_TC_MTIOA        ((unsigned int) 0x1 << 17) /* (TC) TIOA Mirror*/
#define AT91C_TC_MTIOB        ((unsigned int) 0x1 << 18) /* (TC) TIOA Mirror*/
/* -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- */
/* -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- */
/* -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Timer Counter Interface*/
/* ******************************************************************************/
typedef struct _AT91S_TCB {
	AT91S_TC	 TCB_TC0; 	/* TC Channel 0*/
	AT91_REG	 Reserved0[4]; 	/* */
	AT91S_TC	 TCB_TC1; 	/* TC Channel 1*/
	AT91_REG	 Reserved1[4]; 	/* */
	AT91S_TC	 TCB_TC2; 	/* TC Channel 2*/
	AT91_REG	 Reserved2[4]; 	/* */
	AT91_REG	 TCB_BCR; 	/* TC Block Control Register*/
	AT91_REG	 TCB_BMR; 	/* TC Block Mode Register*/
} AT91S_TCB, *AT91PS_TCB;

/* -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- */
#define AT91C_TCB_SYNC        ((unsigned int) 0x1 <<  0) /* (TCB) Synchro Command*/
/* -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- */
#define AT91C_TCB_TC0XC0S     ((unsigned int) 0x3 <<  0) /* (TCB) External Clock Signal 0 Selection*/
#define 	AT91C_TCB_TC0XC0S_TCLK0                ((unsigned int) 0x0) /* (TCB) TCLK0 connected to XC0*/
#define 	AT91C_TCB_TC0XC0S_NONE                 ((unsigned int) 0x1) /* (TCB) None signal connected to XC0*/
#define 	AT91C_TCB_TC0XC0S_TIOA1                ((unsigned int) 0x2) /* (TCB) TIOA1 connected to XC0*/
#define 	AT91C_TCB_TC0XC0S_TIOA2                ((unsigned int) 0x3) /* (TCB) TIOA2 connected to XC0*/
#define AT91C_TCB_TC1XC1S     ((unsigned int) 0x3 <<  2) /* (TCB) External Clock Signal 1 Selection*/
#define 	AT91C_TCB_TC1XC1S_TCLK1                ((unsigned int) 0x0 <<  2) /* (TCB) TCLK1 connected to XC1*/
#define 	AT91C_TCB_TC1XC1S_NONE                 ((unsigned int) 0x1 <<  2) /* (TCB) None signal connected to XC1*/
#define 	AT91C_TCB_TC1XC1S_TIOA0                ((unsigned int) 0x2 <<  2) /* (TCB) TIOA0 connected to XC1*/
#define 	AT91C_TCB_TC1XC1S_TIOA2                ((unsigned int) 0x3 <<  2) /* (TCB) TIOA2 connected to XC1*/
#define AT91C_TCB_TC2XC2S     ((unsigned int) 0x3 <<  4) /* (TCB) External Clock Signal 2 Selection*/
#define 	AT91C_TCB_TC2XC2S_TCLK2                ((unsigned int) 0x0 <<  4) /* (TCB) TCLK2 connected to XC2*/
#define 	AT91C_TCB_TC2XC2S_NONE                 ((unsigned int) 0x1 <<  4) /* (TCB) None signal connected to XC2*/
#define 	AT91C_TCB_TC2XC2S_TIOA0                ((unsigned int) 0x2 <<  4) /* (TCB) TIOA0 connected to XC2*/
#define 	AT91C_TCB_TC2XC2S_TIOA1                ((unsigned int) 0x3 <<  4) /* (TCB) TIOA2 connected to XC2*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Segment LCD Controller*/
/* ******************************************************************************/
typedef struct _AT91S_SLCDC {
	AT91_REG	 SLCDC_CR; 	/* Control Register*/
	AT91_REG	 SLCDC_MR; 	/* Mode Register*/
	AT91_REG	 SLCDC_FRR; 	/* Frame Rate Register*/
	AT91_REG	 SLCDC_DR; 	/* Display Register*/
	AT91_REG	 SLCDC_SR; 	/* Status Register*/
	AT91_REG	 Reserved0[3]; 	/* */
	AT91_REG	 SLCDC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 SLCDC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 SLCDC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 SLCDC_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 Reserved1[116]; 	/* */
	AT91_REG	 SLCDC_MEM[80]; 	/* Memory Register*/
} AT91S_SLCDC, *AT91PS_SLCDC;

/* -------- SLCDC_CR : (SLCDC Offset: 0x0) SLCDC Control Register -------- */
#define AT91C_SLCDC_LCDEN     ((unsigned int) 0x1 <<  0) /* (SLCDC) Enable the LCDC*/
#define AT91C_SLCDC_LCDDIS    ((unsigned int) 0x1 <<  1) /* (SLCDC) Disable the LCDC*/
#define AT91C_SLCDC_SWRST     ((unsigned int) 0x1 <<  3) /* (SLCDC) Software Reset*/
/* -------- SLCDC_MR : (SLCDC Offset: 0x4) SLCDC Control Register -------- */
#define AT91C_SLCDC_COMSEL    ((unsigned int) 0xF <<  0) /* (SLCDC) Selection of the number of common*/
#define 	AT91C_SLCDC_COMSEL_0                    ((unsigned int) 0x0) /* (SLCDC) COM0 selected*/
#define 	AT91C_SLCDC_COMSEL_1                    ((unsigned int) 0x1) /* (SLCDC) COM1 selected*/
#define 	AT91C_SLCDC_COMSEL_2                    ((unsigned int) 0x2) /* (SLCDC) COM2 selected*/
#define 	AT91C_SLCDC_COMSEL_3                    ((unsigned int) 0x3) /* (SLCDC) COM3 selected*/
#define 	AT91C_SLCDC_COMSEL_4                    ((unsigned int) 0x4) /* (SLCDC) COM4 selected*/
#define 	AT91C_SLCDC_COMSEL_5                    ((unsigned int) 0x5) /* (SLCDC) COM5 selected*/
#define 	AT91C_SLCDC_COMSEL_6                    ((unsigned int) 0x6) /* (SLCDC) COM6 selected*/
#define 	AT91C_SLCDC_COMSEL_7                    ((unsigned int) 0x7) /* (SLCDC) COM7 selected*/
#define 	AT91C_SLCDC_COMSEL_8                    ((unsigned int) 0x8) /* (SLCDC) COM8 selected*/
#define 	AT91C_SLCDC_COMSEL_9                    ((unsigned int) 0x9) /* (SLCDC) COM9 selected*/
#define AT91C_SLCDC_SEGSEL    ((unsigned int) 0x1F <<  8) /* (SLCDC) Selection of the number of segment*/
#define AT91C_SLCDC_BUFTIME   ((unsigned int) 0xF << 16) /* (SLCDC) Buffer on time*/
#define 	AT91C_SLCDC_BUFTIME_0_percent            ((unsigned int) 0x0 << 16) /* (SLCDC) Buffer aren't driven*/
#define 	AT91C_SLCDC_BUFTIME_2_Tsclk              ((unsigned int) 0x1 << 16) /* (SLCDC) Buffer are driven during 2 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_4_Tsclk              ((unsigned int) 0x2 << 16) /* (SLCDC) Buffer are driven during 4 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_8_Tsclk              ((unsigned int) 0x3 << 16) /* (SLCDC) Buffer are driven during 8 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_16_Tsclk             ((unsigned int) 0x4 << 16) /* (SLCDC) Buffer are driven during 16 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_32_Tsclk             ((unsigned int) 0x5 << 16) /* (SLCDC) Buffer are driven during 32 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_64_Tsclk             ((unsigned int) 0x6 << 16) /* (SLCDC) Buffer are driven during 64 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_128_Tsclk            ((unsigned int) 0x7 << 16) /* (SLCDC) Buffer are driven during 128 SLCDC clock periods*/
#define 	AT91C_SLCDC_BUFTIME_50_percent           ((unsigned int) 0x8 << 16) /* (SLCDC) Buffer are driven during 50 percent of the frame*/
#define 	AT91C_SLCDC_BUFTIME_100_percent          ((unsigned int) 0x9 << 16) /* (SLCDC) Buffer are driven during 100 percent of the frame*/
#define AT91C_SLCDC_BIAS      ((unsigned int) 0x3 << 20) /* (SLCDC) Bias setting*/
#define 	AT91C_SLCDC_BIAS_1                    ((unsigned int) 0x0 << 20) /* (SLCDC) Vbias is VDD */
#define 	AT91C_SLCDC_BIAS_1_2                  ((unsigned int) 0x1 << 20) /* (SLCDC) Vbias is 1/2 VDD*/
#define 	AT91C_SLCDC_BIAS_1_3                  ((unsigned int) 0x2 << 20) /* (SLCDC) Vbias is 1/3 VDD*/
#define 	AT91C_SLCDC_BIAS_1_4                  ((unsigned int) 0x3 << 20) /* (SLCDC) Vbias is 1/4 VDD*/
#define AT91C_SLCDC_LPMODE    ((unsigned int) 0x1 << 24) /* (SLCDC) Low Power mode*/
/* -------- SLCDC_FRR : (SLCDC Offset: 0x8) SLCDC Frame Rate Register -------- */
#define AT91C_SLCDC_PRESC     ((unsigned int) 0x7 <<  0) /* (SLCDC) Clock prescaler*/
#define 	AT91C_SLCDC_PRESC_SCLK_8               ((unsigned int) 0x0) /* (SLCDC) Clock Prescaler is 8*/
#define 	AT91C_SLCDC_PRESC_SCLK_16              ((unsigned int) 0x1) /* (SLCDC) Clock Prescaler is 16*/
#define 	AT91C_SLCDC_PRESC_SCLK_32              ((unsigned int) 0x2) /* (SLCDC) Clock Prescaler is 32*/
#define 	AT91C_SLCDC_PRESC_SCLK_64              ((unsigned int) 0x3) /* (SLCDC) Clock Prescaler is 64*/
#define 	AT91C_SLCDC_PRESC_SCLK_128             ((unsigned int) 0x4) /* (SLCDC) Clock Prescaler is 128*/
#define 	AT91C_SLCDC_PRESC_SCLK_256             ((unsigned int) 0x5) /* (SLCDC) Clock Prescaler is 256*/
#define 	AT91C_SLCDC_PRESC_SCLK_512             ((unsigned int) 0x6) /* (SLCDC) Clock Prescaler is 512*/
#define 	AT91C_SLCDC_PRESC_SCLK_1024            ((unsigned int) 0x7) /* (SLCDC) Clock Prescaler is 1024*/
#define AT91C_SLCDC_DIV       ((unsigned int) 0x7 <<  8) /* (SLCDC) Clock division*/
#define 	AT91C_SLCDC_DIV_1                    ((unsigned int) 0x0 <<  8) /* (SLCDC) Clock division is 1*/
#define 	AT91C_SLCDC_DIV_2                    ((unsigned int) 0x1 <<  8) /* (SLCDC) Clock division is 2*/
#define 	AT91C_SLCDC_DIV_3                    ((unsigned int) 0x2 <<  8) /* (SLCDC) Clock division is 3*/
#define 	AT91C_SLCDC_DIV_4                    ((unsigned int) 0x3 <<  8) /* (SLCDC) Clock division is 4*/
#define 	AT91C_SLCDC_DIV_5                    ((unsigned int) 0x4 <<  8) /* (SLCDC) Clock division is 5*/
#define 	AT91C_SLCDC_DIV_6                    ((unsigned int) 0x5 <<  8) /* (SLCDC) Clock division is 6*/
#define 	AT91C_SLCDC_DIV_7                    ((unsigned int) 0x6 <<  8) /* (SLCDC) Clock division is 7*/
#define 	AT91C_SLCDC_DIV_8                    ((unsigned int) 0x7 <<  8) /* (SLCDC) Clock division is 8*/
/* -------- SLCDC_DR : (SLCDC Offset: 0xc) SLCDC Display Register -------- */
#define AT91C_SLCDC_DISPMODE  ((unsigned int) 0x7 <<  0) /* (SLCDC) Display mode*/
#define 	AT91C_SLCDC_DISPMODE_NORMAL               ((unsigned int) 0x0) /* (SLCDC) Latched datas are displayed*/
#define 	AT91C_SLCDC_DISPMODE_FORCE_OFF            ((unsigned int) 0x1) /* (SLCDC) All pixel are unvisible*/
#define 	AT91C_SLCDC_DISPMODE_FORCE_ON             ((unsigned int) 0x2) /* (SLCDC) All pixel are visible*/
#define 	AT91C_SLCDC_DISPMODE_BLINK                ((unsigned int) 0x3) /* (SLCDC) Turn all pixel alternatively off to the state defined in SCLD memory at LCDBLKFREQ frequency*/
#define 	AT91C_SLCDC_DISPMODE_INVERTED             ((unsigned int) 0x4) /* (SLCDC) All pixel are set in the inverted state as defined in SCLD memory*/
#define 	AT91C_SLCDC_DISPMODE_INVERTED_BLINK       ((unsigned int) 0x5) /* (SLCDC) Turn all pixel alternatively off to the opposite state defined in SCLD memory at LCDBLKFREQ frequency*/
#define AT91C_SLCDC_BLKFREQ   ((unsigned int) 0xFF <<  8) /* (SLCDC) Blinking frequency*/
/* -------- SLCDC_SR : (SLCDC Offset: 0x10) SLCDC Status  Register -------- */
#define AT91C_SLCDC_ENA       ((unsigned int) 0x1 <<  0) /* (SLCDC) Enable status*/
/* -------- SLCDC_IER : (SLCDC Offset: 0x20) SLCDC Interrupt Enable Register -------- */
#define AT91C_SLCDC_ENDFRAME  ((unsigned int) 0x1 <<  0) /* (SLCDC) End of Frame*/
#define AT91C_SLCDC_BKPER     ((unsigned int) 0x1 <<  1) /* (SLCDC) Blank Periode*/
#define AT91C_SLCDC_DIS       ((unsigned int) 0x1 <<  2) /* (SLCDC) Disable*/
/* -------- SLCDC_IDR : (SLCDC Offset: 0x24) SLCDC Interrupt Disable Register -------- */
/* -------- SLCDC_IMR : (SLCDC Offset: 0x28) SLCDC Interrupt Mask Register -------- */
/* -------- SLCDC_ISR : (SLCDC Offset: 0x2c) SLCDC Interrupt Status Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Analog to Digital Convertor*/
/* ******************************************************************************/
typedef struct _AT91S_ADC {
	AT91_REG	 ADC_CR; 	/* ADC Control Register*/
	AT91_REG	 ADC_MR; 	/* ADC Mode Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 ADC_CHER; 	/* ADC Channel Enable Register*/
	AT91_REG	 ADC_CHDR; 	/* ADC Channel Disable Register*/
	AT91_REG	 ADC_CHSR; 	/* ADC Channel Status Register*/
	AT91_REG	 ADC_SR; 	/* ADC Status Register*/
	AT91_REG	 ADC_LCDR; 	/* ADC Last Converted Data Register*/
	AT91_REG	 ADC_IER; 	/* ADC Interrupt Enable Register*/
	AT91_REG	 ADC_IDR; 	/* ADC Interrupt Disable Register*/
	AT91_REG	 ADC_IMR; 	/* ADC Interrupt Mask Register*/
	AT91_REG	 ADC_CDR0; 	/* ADC Channel Data Register 0*/
	AT91_REG	 ADC_CDR1; 	/* ADC Channel Data Register 1*/
	AT91_REG	 ADC_CDR2; 	/* ADC Channel Data Register 2*/
	AT91_REG	 ADC_CDR3; 	/* ADC Channel Data Register 3*/
	AT91_REG	 ADC_CDR4; 	/* ADC Channel Data Register 4*/
	AT91_REG	 ADC_CDR5; 	/* ADC Channel Data Register 5*/
	AT91_REG	 ADC_CDR6; 	/* ADC Channel Data Register 6*/
	AT91_REG	 ADC_CDR7; 	/* ADC Channel Data Register 7*/
	AT91_REG	 Reserved1[44]; 	/* */
	AT91_REG	 ADC_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 ADC_RCR; 	/* Receive Counter Register*/
	AT91_REG	 ADC_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 ADC_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 ADC_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 ADC_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 ADC_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 ADC_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 ADC_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 ADC_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_ADC, *AT91PS_ADC;

/* -------- ADC_CR : (ADC Offset: 0x0) ADC Control Register -------- */
#define AT91C_ADC_SWRST       ((unsigned int) 0x1 <<  0) /* (ADC) Software Reset*/
#define AT91C_ADC_START       ((unsigned int) 0x1 <<  1) /* (ADC) Start Conversion*/
/* -------- ADC_MR : (ADC Offset: 0x4) ADC Mode Register -------- */
#define AT91C_ADC_TRGEN       ((unsigned int) 0x1 <<  0) /* (ADC) Trigger Enable*/
#define 	AT91C_ADC_TRGEN_DIS                  ((unsigned int) 0x0) /* (ADC) Hradware triggers are disabled. Starting a conversion is only possible by software*/
#define 	AT91C_ADC_TRGEN_EN                   ((unsigned int) 0x1) /* (ADC) Hardware trigger selected by TRGSEL field is enabled.*/
#define AT91C_ADC_TRGSEL      ((unsigned int) 0x7 <<  1) /* (ADC) Trigger Selection*/
#define 	AT91C_ADC_TRGSEL_TIOA0                ((unsigned int) 0x0 <<  1) /* (ADC) Selected TRGSEL = TIAO0*/
#define 	AT91C_ADC_TRGSEL_TIOA1                ((unsigned int) 0x1 <<  1) /* (ADC) Selected TRGSEL = TIAO1*/
#define 	AT91C_ADC_TRGSEL_TIOA2                ((unsigned int) 0x2 <<  1) /* (ADC) Selected TRGSEL = TIAO2*/
#define 	AT91C_ADC_TRGSEL_TIOA3                ((unsigned int) 0x3 <<  1) /* (ADC) Selected TRGSEL = TIAO3*/
#define 	AT91C_ADC_TRGSEL_TIOA4                ((unsigned int) 0x4 <<  1) /* (ADC) Selected TRGSEL = TIAO4*/
#define 	AT91C_ADC_TRGSEL_TIOA5                ((unsigned int) 0x5 <<  1) /* (ADC) Selected TRGSEL = TIAO5*/
#define 	AT91C_ADC_TRGSEL_EXT                  ((unsigned int) 0x6 <<  1) /* (ADC) Selected TRGSEL = External Trigger*/
#define AT91C_ADC_LOWRES      ((unsigned int) 0x1 <<  4) /* (ADC) Resolution.*/
#define 	AT91C_ADC_LOWRES_10_BIT               ((unsigned int) 0x0 <<  4) /* (ADC) 10-bit resolution*/
#define 	AT91C_ADC_LOWRES_8_BIT                ((unsigned int) 0x1 <<  4) /* (ADC) 8-bit resolution*/
#define AT91C_ADC_SLEEP       ((unsigned int) 0x1 <<  5) /* (ADC) Sleep Mode*/
#define 	AT91C_ADC_SLEEP_NORMAL_MODE          ((unsigned int) 0x0 <<  5) /* (ADC) Normal Mode*/
#define 	AT91C_ADC_SLEEP_MODE                 ((unsigned int) 0x1 <<  5) /* (ADC) Sleep Mode*/
#define AT91C_ADC_PRESCAL     ((unsigned int) 0x3F <<  8) /* (ADC) Prescaler rate selection*/
#define AT91C_ADC_STARTUP     ((unsigned int) 0x1F << 16) /* (ADC) Startup Time*/
#define AT91C_ADC_SHTIM       ((unsigned int) 0xF << 24) /* (ADC) Sample & Hold Time*/
/* -------- 	ADC_CHER : (ADC Offset: 0x10) ADC Channel Enable Register -------- */
#define AT91C_ADC_CH0         ((unsigned int) 0x1 <<  0) /* (ADC) Channel 0*/
#define AT91C_ADC_CH1         ((unsigned int) 0x1 <<  1) /* (ADC) Channel 1*/
#define AT91C_ADC_CH2         ((unsigned int) 0x1 <<  2) /* (ADC) Channel 2*/
#define AT91C_ADC_CH3         ((unsigned int) 0x1 <<  3) /* (ADC) Channel 3*/
#define AT91C_ADC_CH4         ((unsigned int) 0x1 <<  4) /* (ADC) Channel 4*/
#define AT91C_ADC_CH5         ((unsigned int) 0x1 <<  5) /* (ADC) Channel 5*/
#define AT91C_ADC_CH6         ((unsigned int) 0x1 <<  6) /* (ADC) Channel 6*/
#define AT91C_ADC_CH7         ((unsigned int) 0x1 <<  7) /* (ADC) Channel 7*/
/* -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- */
/* -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- */
/* -------- ADC_SR : (ADC Offset: 0x1c) ADC Status Register -------- */
#define AT91C_ADC_EOC0        ((unsigned int) 0x1 <<  0) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC1        ((unsigned int) 0x1 <<  1) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC2        ((unsigned int) 0x1 <<  2) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC3        ((unsigned int) 0x1 <<  3) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC4        ((unsigned int) 0x1 <<  4) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC5        ((unsigned int) 0x1 <<  5) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC6        ((unsigned int) 0x1 <<  6) /* (ADC) End of Conversion*/
#define AT91C_ADC_EOC7        ((unsigned int) 0x1 <<  7) /* (ADC) End of Conversion*/
#define AT91C_ADC_OVRE0       ((unsigned int) 0x1 <<  8) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE1       ((unsigned int) 0x1 <<  9) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE2       ((unsigned int) 0x1 << 10) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE3       ((unsigned int) 0x1 << 11) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE4       ((unsigned int) 0x1 << 12) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE5       ((unsigned int) 0x1 << 13) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE6       ((unsigned int) 0x1 << 14) /* (ADC) Overrun Error*/
#define AT91C_ADC_OVRE7       ((unsigned int) 0x1 << 15) /* (ADC) Overrun Error*/
#define AT91C_ADC_DRDY        ((unsigned int) 0x1 << 16) /* (ADC) Data Ready*/
#define AT91C_ADC_GOVRE       ((unsigned int) 0x1 << 17) /* (ADC) General Overrun*/
#define AT91C_ADC_ENDRX       ((unsigned int) 0x1 << 18) /* (ADC) End of Receiver Transfer*/
#define AT91C_ADC_RXBUFF      ((unsigned int) 0x1 << 19) /* (ADC) RXBUFF Interrupt*/
/* -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- */
#define AT91C_ADC_LDATA       ((unsigned int) 0x3FF <<  0) /* (ADC) Last Data Converted*/
/* -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- */
/* -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- */
/* -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- */
/* -------- ADC_CDR0 : (ADC Offset: 0x30) ADC Channel Data Register 0 -------- */
#define AT91C_ADC_DATA        ((unsigned int) 0x3FF <<  0) /* (ADC) Converted Data*/
/* -------- ADC_CDR1 : (ADC Offset: 0x34) ADC Channel Data Register 1 -------- */
/* -------- ADC_CDR2 : (ADC Offset: 0x38) ADC Channel Data Register 2 -------- */
/* -------- ADC_CDR3 : (ADC Offset: 0x3c) ADC Channel Data Register 3 -------- */
/* -------- ADC_CDR4 : (ADC Offset: 0x40) ADC Channel Data Register 4 -------- */
/* -------- ADC_CDR5 : (ADC Offset: 0x44) ADC Channel Data Register 5 -------- */
/* -------- ADC_CDR6 : (ADC Offset: 0x48) ADC Channel Data Register 6 -------- */
/* -------- ADC_CDR7 : (ADC Offset: 0x4c) ADC Channel Data Register 7 -------- */

/* ******************************************************************************/
/*               REGISTER ADDRESS DEFINITION FOR AT91SAM7L128*/
/* ******************************************************************************/
/* ========== Register definition for SYS peripheral ========== */
/* ========== Register definition for AIC peripheral ========== */
#define AT91C_AIC_IVR   ((AT91_REG *) 	0xFFFFF100) /* (AIC) IRQ Vector Register*/
#define AT91C_AIC_SMR   ((AT91_REG *) 	0xFFFFF000) /* (AIC) Source Mode Register*/
#define AT91C_AIC_FVR   ((AT91_REG *) 	0xFFFFF104) /* (AIC) FIQ Vector Register*/
#define AT91C_AIC_DCR   ((AT91_REG *) 	0xFFFFF138) /* (AIC) Debug Control Register (Protect)*/
#define AT91C_AIC_EOICR ((AT91_REG *) 	0xFFFFF130) /* (AIC) End of Interrupt Command Register*/
#define AT91C_AIC_SVR   ((AT91_REG *) 	0xFFFFF080) /* (AIC) Source Vector Register*/
#define AT91C_AIC_FFSR  ((AT91_REG *) 	0xFFFFF148) /* (AIC) Fast Forcing Status Register*/
#define AT91C_AIC_ICCR  ((AT91_REG *) 	0xFFFFF128) /* (AIC) Interrupt Clear Command Register*/
#define AT91C_AIC_ISR   ((AT91_REG *) 	0xFFFFF108) /* (AIC) Interrupt Status Register*/
#define AT91C_AIC_IMR   ((AT91_REG *) 	0xFFFFF110) /* (AIC) Interrupt Mask Register*/
#define AT91C_AIC_IPR   ((AT91_REG *) 	0xFFFFF10C) /* (AIC) Interrupt Pending Register*/
#define AT91C_AIC_FFER  ((AT91_REG *) 	0xFFFFF140) /* (AIC) Fast Forcing Enable Register*/
#define AT91C_AIC_IECR  ((AT91_REG *) 	0xFFFFF120) /* (AIC) Interrupt Enable Command Register*/
#define AT91C_AIC_ISCR  ((AT91_REG *) 	0xFFFFF12C) /* (AIC) Interrupt Set Command Register*/
#define AT91C_AIC_FFDR  ((AT91_REG *) 	0xFFFFF144) /* (AIC) Fast Forcing Disable Register*/
#define AT91C_AIC_CISR  ((AT91_REG *) 	0xFFFFF114) /* (AIC) Core Interrupt Status Register*/
#define AT91C_AIC_IDCR  ((AT91_REG *) 	0xFFFFF124) /* (AIC) Interrupt Disable Command Register*/
#define AT91C_AIC_SPU   ((AT91_REG *) 	0xFFFFF134) /* (AIC) Spurious Vector Register*/
/* ========== Register definition for PDC_DBGU peripheral ========== */
#define AT91C_DBGU_TCR  ((AT91_REG *) 	0xFFFFF30C) /* (PDC_DBGU) Transmit Counter Register*/
#define AT91C_DBGU_RNPR ((AT91_REG *) 	0xFFFFF310) /* (PDC_DBGU) Receive Next Pointer Register*/
#define AT91C_DBGU_TNPR ((AT91_REG *) 	0xFFFFF318) /* (PDC_DBGU) Transmit Next Pointer Register*/
#define AT91C_DBGU_TPR  ((AT91_REG *) 	0xFFFFF308) /* (PDC_DBGU) Transmit Pointer Register*/
#define AT91C_DBGU_RPR  ((AT91_REG *) 	0xFFFFF300) /* (PDC_DBGU) Receive Pointer Register*/
#define AT91C_DBGU_RCR  ((AT91_REG *) 	0xFFFFF304) /* (PDC_DBGU) Receive Counter Register*/
#define AT91C_DBGU_RNCR ((AT91_REG *) 	0xFFFFF314) /* (PDC_DBGU) Receive Next Counter Register*/
#define AT91C_DBGU_PTCR ((AT91_REG *) 	0xFFFFF320) /* (PDC_DBGU) PDC Transfer Control Register*/
#define AT91C_DBGU_PTSR ((AT91_REG *) 	0xFFFFF324) /* (PDC_DBGU) PDC Transfer Status Register*/
#define AT91C_DBGU_TNCR ((AT91_REG *) 	0xFFFFF31C) /* (PDC_DBGU) Transmit Next Counter Register*/
/* ========== Register definition for DBGU peripheral ========== */
#define AT91C_DBGU_EXID ((AT91_REG *) 	0xFFFFF244) /* (DBGU) Chip ID Extension Register*/
#define AT91C_DBGU_BRGR ((AT91_REG *) 	0xFFFFF220) /* (DBGU) Baud Rate Generator Register*/
#define AT91C_DBGU_IDR  ((AT91_REG *) 	0xFFFFF20C) /* (DBGU) Interrupt Disable Register*/
#define AT91C_DBGU_CSR  ((AT91_REG *) 	0xFFFFF214) /* (DBGU) Channel Status Register*/
#define AT91C_DBGU_CIDR ((AT91_REG *) 	0xFFFFF240) /* (DBGU) Chip ID Register*/
#define AT91C_DBGU_MR   ((AT91_REG *) 	0xFFFFF204) /* (DBGU) Mode Register*/
#define AT91C_DBGU_IMR  ((AT91_REG *) 	0xFFFFF210) /* (DBGU) Interrupt Mask Register*/
#define AT91C_DBGU_CR   ((AT91_REG *) 	0xFFFFF200) /* (DBGU) Control Register*/
#define AT91C_DBGU_FNTR ((AT91_REG *) 	0xFFFFF248) /* (DBGU) Force NTRST Register*/
#define AT91C_DBGU_THR  ((AT91_REG *) 	0xFFFFF21C) /* (DBGU) Transmitter Holding Register*/
#define AT91C_DBGU_RHR  ((AT91_REG *) 	0xFFFFF218) /* (DBGU) Receiver Holding Register*/
#define AT91C_DBGU_IER  ((AT91_REG *) 	0xFFFFF208) /* (DBGU) Interrupt Enable Register*/
/* ========== Register definition for PIOA peripheral ========== */
#define AT91C_PIOA_ODR  ((AT91_REG *) 	0xFFFFF414) /* (PIOA) Output Disable Registerr*/
#define AT91C_PIOA_SODR ((AT91_REG *) 	0xFFFFF430) /* (PIOA) Set Output Data Register*/
#define AT91C_PIOA_ISR  ((AT91_REG *) 	0xFFFFF44C) /* (PIOA) Interrupt Status Register*/
#define AT91C_PIOA_ABSR ((AT91_REG *) 	0xFFFFF478) /* (PIOA) AB Select Status Register*/
#define AT91C_PIOA_IER  ((AT91_REG *) 	0xFFFFF440) /* (PIOA) Interrupt Enable Register*/
#define AT91C_PIOA_PPUDR ((AT91_REG *) 	0xFFFFF460) /* (PIOA) Pull-up Disable Register*/
#define AT91C_PIOA_IMR  ((AT91_REG *) 	0xFFFFF448) /* (PIOA) Interrupt Mask Register*/
#define AT91C_PIOA_PER  ((AT91_REG *) 	0xFFFFF400) /* (PIOA) PIO Enable Register*/
#define AT91C_PIOA_IFDR ((AT91_REG *) 	0xFFFFF424) /* (PIOA) Input Filter Disable Register*/
#define AT91C_PIOA_OWDR ((AT91_REG *) 	0xFFFFF4A4) /* (PIOA) Output Write Disable Register*/
#define AT91C_PIOA_MDSR ((AT91_REG *) 	0xFFFFF458) /* (PIOA) Multi-driver Status Register*/
#define AT91C_PIOA_IDR  ((AT91_REG *) 	0xFFFFF444) /* (PIOA) Interrupt Disable Register*/
#define AT91C_PIOA_ODSR ((AT91_REG *) 	0xFFFFF438) /* (PIOA) Output Data Status Register*/
#define AT91C_PIOA_PPUSR ((AT91_REG *) 	0xFFFFF468) /* (PIOA) Pull-up Status Register*/
#define AT91C_PIOA_OWSR ((AT91_REG *) 	0xFFFFF4A8) /* (PIOA) Output Write Status Register*/
#define AT91C_PIOA_BSR  ((AT91_REG *) 	0xFFFFF474) /* (PIOA) Select B Register*/
#define AT91C_PIOA_OWER ((AT91_REG *) 	0xFFFFF4A0) /* (PIOA) Output Write Enable Register*/
#define AT91C_PIOA_IFER ((AT91_REG *) 	0xFFFFF420) /* (PIOA) Input Filter Enable Register*/
#define AT91C_PIOA_PDSR ((AT91_REG *) 	0xFFFFF43C) /* (PIOA) Pin Data Status Register*/
#define AT91C_PIOA_PPUER ((AT91_REG *) 	0xFFFFF464) /* (PIOA) Pull-up Enable Register*/
#define AT91C_PIOA_OSR  ((AT91_REG *) 	0xFFFFF418) /* (PIOA) Output Status Register*/
#define AT91C_PIOA_ASR  ((AT91_REG *) 	0xFFFFF470) /* (PIOA) Select A Register*/
#define AT91C_PIOA_MDDR ((AT91_REG *) 	0xFFFFF454) /* (PIOA) Multi-driver Disable Register*/
#define AT91C_PIOA_CODR ((AT91_REG *) 	0xFFFFF434) /* (PIOA) Clear Output Data Register*/
#define AT91C_PIOA_MDER ((AT91_REG *) 	0xFFFFF450) /* (PIOA) Multi-driver Enable Register*/
#define AT91C_PIOA_PDR  ((AT91_REG *) 	0xFFFFF404) /* (PIOA) PIO Disable Register*/
#define AT91C_PIOA_IFSR ((AT91_REG *) 	0xFFFFF428) /* (PIOA) Input Filter Status Register*/
#define AT91C_PIOA_OER  ((AT91_REG *) 	0xFFFFF410) /* (PIOA) Output Enable Register*/
#define AT91C_PIOA_PSR  ((AT91_REG *) 	0xFFFFF408) /* (PIOA) PIO Status Register*/
/* ========== Register definition for PIOB peripheral ========== */
#define AT91C_PIOB_OWDR ((AT91_REG *) 	0xFFFFF6A4) /* (PIOB) Output Write Disable Register*/
#define AT91C_PIOB_MDER ((AT91_REG *) 	0xFFFFF650) /* (PIOB) Multi-driver Enable Register*/
#define AT91C_PIOB_PPUSR ((AT91_REG *) 	0xFFFFF668) /* (PIOB) Pull-up Status Register*/
#define AT91C_PIOB_IMR  ((AT91_REG *) 	0xFFFFF648) /* (PIOB) Interrupt Mask Register*/
#define AT91C_PIOB_ASR  ((AT91_REG *) 	0xFFFFF670) /* (PIOB) Select A Register*/
#define AT91C_PIOB_PPUDR ((AT91_REG *) 	0xFFFFF660) /* (PIOB) Pull-up Disable Register*/
#define AT91C_PIOB_PSR  ((AT91_REG *) 	0xFFFFF608) /* (PIOB) PIO Status Register*/
#define AT91C_PIOB_IER  ((AT91_REG *) 	0xFFFFF640) /* (PIOB) Interrupt Enable Register*/
#define AT91C_PIOB_CODR ((AT91_REG *) 	0xFFFFF634) /* (PIOB) Clear Output Data Register*/
#define AT91C_PIOB_OWER ((AT91_REG *) 	0xFFFFF6A0) /* (PIOB) Output Write Enable Register*/
#define AT91C_PIOB_ABSR ((AT91_REG *) 	0xFFFFF678) /* (PIOB) AB Select Status Register*/
#define AT91C_PIOB_IFDR ((AT91_REG *) 	0xFFFFF624) /* (PIOB) Input Filter Disable Register*/
#define AT91C_PIOB_PDSR ((AT91_REG *) 	0xFFFFF63C) /* (PIOB) Pin Data Status Register*/
#define AT91C_PIOB_IDR  ((AT91_REG *) 	0xFFFFF644) /* (PIOB) Interrupt Disable Register*/
#define AT91C_PIOB_OWSR ((AT91_REG *) 	0xFFFFF6A8) /* (PIOB) Output Write Status Register*/
#define AT91C_PIOB_PDR  ((AT91_REG *) 	0xFFFFF604) /* (PIOB) PIO Disable Register*/
#define AT91C_PIOB_ODR  ((AT91_REG *) 	0xFFFFF614) /* (PIOB) Output Disable Registerr*/
#define AT91C_PIOB_IFSR ((AT91_REG *) 	0xFFFFF628) /* (PIOB) Input Filter Status Register*/
#define AT91C_PIOB_PPUER ((AT91_REG *) 	0xFFFFF664) /* (PIOB) Pull-up Enable Register*/
#define AT91C_PIOB_SODR ((AT91_REG *) 	0xFFFFF630) /* (PIOB) Set Output Data Register*/
#define AT91C_PIOB_ISR  ((AT91_REG *) 	0xFFFFF64C) /* (PIOB) Interrupt Status Register*/
#define AT91C_PIOB_ODSR ((AT91_REG *) 	0xFFFFF638) /* (PIOB) Output Data Status Register*/
#define AT91C_PIOB_OSR  ((AT91_REG *) 	0xFFFFF618) /* (PIOB) Output Status Register*/
#define AT91C_PIOB_MDSR ((AT91_REG *) 	0xFFFFF658) /* (PIOB) Multi-driver Status Register*/
#define AT91C_PIOB_IFER ((AT91_REG *) 	0xFFFFF620) /* (PIOB) Input Filter Enable Register*/
#define AT91C_PIOB_BSR  ((AT91_REG *) 	0xFFFFF674) /* (PIOB) Select B Register*/
#define AT91C_PIOB_MDDR ((AT91_REG *) 	0xFFFFF654) /* (PIOB) Multi-driver Disable Register*/
#define AT91C_PIOB_OER  ((AT91_REG *) 	0xFFFFF610) /* (PIOB) Output Enable Register*/
#define AT91C_PIOB_PER  ((AT91_REG *) 	0xFFFFF600) /* (PIOB) PIO Enable Register*/
/* ========== Register definition for PIOC peripheral ========== */
#define AT91C_PIOC_OWDR ((AT91_REG *) 	0xFFFFF8A4) /* (PIOC) Output Write Disable Register*/
#define AT91C_PIOC_SODR ((AT91_REG *) 	0xFFFFF830) /* (PIOC) Set Output Data Register*/
#define AT91C_PIOC_PPUER ((AT91_REG *) 	0xFFFFF864) /* (PIOC) Pull-up Enable Register*/
#define AT91C_PIOC_CODR ((AT91_REG *) 	0xFFFFF834) /* (PIOC) Clear Output Data Register*/
#define AT91C_PIOC_PSR  ((AT91_REG *) 	0xFFFFF808) /* (PIOC) PIO Status Register*/
#define AT91C_PIOC_PDR  ((AT91_REG *) 	0xFFFFF804) /* (PIOC) PIO Disable Register*/
#define AT91C_PIOC_ODR  ((AT91_REG *) 	0xFFFFF814) /* (PIOC) Output Disable Registerr*/
#define AT91C_PIOC_PPUSR ((AT91_REG *) 	0xFFFFF868) /* (PIOC) Pull-up Status Register*/
#define AT91C_PIOC_ABSR ((AT91_REG *) 	0xFFFFF878) /* (PIOC) AB Select Status Register*/
#define AT91C_PIOC_IFSR ((AT91_REG *) 	0xFFFFF828) /* (PIOC) Input Filter Status Register*/
#define AT91C_PIOC_OER  ((AT91_REG *) 	0xFFFFF810) /* (PIOC) Output Enable Register*/
#define AT91C_PIOC_IMR  ((AT91_REG *) 	0xFFFFF848) /* (PIOC) Interrupt Mask Register*/
#define AT91C_PIOC_ASR  ((AT91_REG *) 	0xFFFFF870) /* (PIOC) Select A Register*/
#define AT91C_PIOC_MDDR ((AT91_REG *) 	0xFFFFF854) /* (PIOC) Multi-driver Disable Register*/
#define AT91C_PIOC_OWSR ((AT91_REG *) 	0xFFFFF8A8) /* (PIOC) Output Write Status Register*/
#define AT91C_PIOC_PER  ((AT91_REG *) 	0xFFFFF800) /* (PIOC) PIO Enable Register*/
#define AT91C_PIOC_IDR  ((AT91_REG *) 	0xFFFFF844) /* (PIOC) Interrupt Disable Register*/
#define AT91C_PIOC_MDER ((AT91_REG *) 	0xFFFFF850) /* (PIOC) Multi-driver Enable Register*/
#define AT91C_PIOC_PDSR ((AT91_REG *) 	0xFFFFF83C) /* (PIOC) Pin Data Status Register*/
#define AT91C_PIOC_MDSR ((AT91_REG *) 	0xFFFFF858) /* (PIOC) Multi-driver Status Register*/
#define AT91C_PIOC_OWER ((AT91_REG *) 	0xFFFFF8A0) /* (PIOC) Output Write Enable Register*/
#define AT91C_PIOC_BSR  ((AT91_REG *) 	0xFFFFF874) /* (PIOC) Select B Register*/
#define AT91C_PIOC_PPUDR ((AT91_REG *) 	0xFFFFF860) /* (PIOC) Pull-up Disable Register*/
#define AT91C_PIOC_IFDR ((AT91_REG *) 	0xFFFFF824) /* (PIOC) Input Filter Disable Register*/
#define AT91C_PIOC_IER  ((AT91_REG *) 	0xFFFFF840) /* (PIOC) Interrupt Enable Register*/
#define AT91C_PIOC_OSR  ((AT91_REG *) 	0xFFFFF818) /* (PIOC) Output Status Register*/
#define AT91C_PIOC_ODSR ((AT91_REG *) 	0xFFFFF838) /* (PIOC) Output Data Status Register*/
#define AT91C_PIOC_ISR  ((AT91_REG *) 	0xFFFFF84C) /* (PIOC) Interrupt Status Register*/
#define AT91C_PIOC_IFER ((AT91_REG *) 	0xFFFFF820) /* (PIOC) Input Filter Enable Register*/
/* ========== Register definition for CKGR peripheral ========== */
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0xFFFFFC20) /* (CKGR) Main Oscillator Register*/
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0xFFFFFC24) /* (CKGR) Main Clock  Frequency Register*/
#define AT91C_CKGR_PLLR ((AT91_REG *) 	0xFFFFFC28) /* (CKGR) PLL Register*/
/* ========== Register definition for PMC peripheral ========== */
#define AT91C_PMC_PCER  ((AT91_REG *) 	0xFFFFFC10) /* (PMC) Peripheral Clock Enable Register*/
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0xFFFFFC40) /* (PMC) Programmable Clock Register*/
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0xFFFFFC30) /* (PMC) Master Clock Register*/
#define AT91C_PMC_PLLR  ((AT91_REG *) 	0xFFFFFC28) /* (PMC) PLL Register*/
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0xFFFFFC14) /* (PMC) Peripheral Clock Disable Register*/
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0xFFFFFC08) /* (PMC) System Clock Status Register*/
#define AT91C_PMC_MCFR  ((AT91_REG *) 	0xFFFFFC24) /* (PMC) Main Clock  Frequency Register*/
#define AT91C_PMC_IMR   ((AT91_REG *) 	0xFFFFFC6C) /* (PMC) Interrupt Mask Register*/
#define AT91C_PMC_IER   ((AT91_REG *) 	0xFFFFFC60) /* (PMC) Interrupt Enable Register*/
#define AT91C_PMC_MOR   ((AT91_REG *) 	0xFFFFFC20) /* (PMC) Main Oscillator Register*/
#define AT91C_PMC_IDR   ((AT91_REG *) 	0xFFFFFC64) /* (PMC) Interrupt Disable Register*/
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0xFFFFFC04) /* (PMC) System Clock Disable Register*/
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0xFFFFFC18) /* (PMC) Peripheral Clock Status Register*/
#define AT91C_PMC_FSMR  ((AT91_REG *) 	0xFFFFFC70) /* (PMC) Fast Startup Mode Register*/
#define AT91C_PMC_SCER  ((AT91_REG *) 	0xFFFFFC00) /* (PMC) System Clock Enable Register*/
#define AT91C_PMC_SR    ((AT91_REG *) 	0xFFFFFC68) /* (PMC) Status Register*/
/* ========== Register definition for RSTC peripheral ========== */
#define AT91C_RSTC_RCR  ((AT91_REG *) 	0xFFFFFD00) /* (RSTC) Reset Control Register*/
#define AT91C_RSTC_RMR  ((AT91_REG *) 	0xFFFFFD08) /* (RSTC) Reset Mode Register*/
#define AT91C_RSTC_RSR  ((AT91_REG *) 	0xFFFFFD04) /* (RSTC) Reset Status Register*/
/* ========== Register definition for RTC peripheral ========== */
#define AT91C_RTC_IMR   ((AT91_REG *) 	0xFFFFFD88) /* (RTC) Interrupt Mask Register*/
#define AT91C_RTC_TAR   ((AT91_REG *) 	0xFFFFFD70) /* (RTC) Time Alarm Register*/
#define AT91C_RTC_IDR   ((AT91_REG *) 	0xFFFFFD84) /* (RTC) Interrupt Disable Register*/
#define AT91C_RTC_HMR   ((AT91_REG *) 	0xFFFFFD64) /* (RTC) Hour Mode Register*/
#define AT91C_RTC_SR    ((AT91_REG *) 	0xFFFFFD78) /* (RTC) Status Register*/
#define AT91C_RTC_VER   ((AT91_REG *) 	0xFFFFFD8C) /* (RTC) Valid Entry Register*/
#define AT91C_RTC_TIMR  ((AT91_REG *) 	0xFFFFFD68) /* (RTC) Time Register*/
#define AT91C_RTC_CAR   ((AT91_REG *) 	0xFFFFFD74) /* (RTC) Calendar Alarm Register*/
#define AT91C_RTC_IER   ((AT91_REG *) 	0xFFFFFD80) /* (RTC) Interrupt Enable Register*/
#define AT91C_RTC_SCR   ((AT91_REG *) 	0xFFFFFD7C) /* (RTC) Status Clear Register*/
#define AT91C_RTC_MR    ((AT91_REG *) 	0xFFFFFD60) /* (RTC) Mode Register*/
#define AT91C_RTC_CALR  ((AT91_REG *) 	0xFFFFFD6C) /* (RTC) Calendar Register*/
/* ========== Register definition for PITC peripheral ========== */
#define AT91C_PITC_PIMR ((AT91_REG *) 	0xFFFFFD40) /* (PITC) Period Interval Mode Register*/
#define AT91C_PITC_PIVR ((AT91_REG *) 	0xFFFFFD48) /* (PITC) Period Interval Value Register*/
#define AT91C_PITC_PISR ((AT91_REG *) 	0xFFFFFD44) /* (PITC) Period Interval Status Register*/
#define AT91C_PITC_PIIR ((AT91_REG *) 	0xFFFFFD4C) /* (PITC) Period Interval Image Register*/
/* ========== Register definition for WDTC peripheral ========== */
#define AT91C_WDTC_WDMR ((AT91_REG *) 	0xFFFFFD54) /* (WDTC) Watchdog Mode Register*/
#define AT91C_WDTC_WDSR ((AT91_REG *) 	0xFFFFFD58) /* (WDTC) Watchdog Status Register*/
#define AT91C_WDTC_WDCR ((AT91_REG *) 	0xFFFFFD50) /* (WDTC) Watchdog Control Register*/
/* ========== Register definition for VREG peripheral ========== */
#define AT91C_VREG_MR   ((AT91_REG *) 	0xFFFFFD60) /* (VREG) Voltage Regulator Mode Register*/
/* ========== Register definition for MC peripheral ========== */
#define AT91C_MC_ASR    ((AT91_REG *) 	0xFFFFFF04) /* (MC) MC Abort Status Register*/
#define AT91C_MC_RCR    ((AT91_REG *) 	0xFFFFFF00) /* (MC) MC Remap Control Register*/
#define AT91C_MC_FCR    ((AT91_REG *) 	0xFFFFFF64) /* (MC) MC Flash Command Register*/
#define AT91C_MC_AASR   ((AT91_REG *) 	0xFFFFFF08) /* (MC) MC Abort Address Status Register*/
#define AT91C_MC_FSR    ((AT91_REG *) 	0xFFFFFF68) /* (MC) MC Flash Status Register*/
#define AT91C_MC_FRR    ((AT91_REG *) 	0xFFFFFF6C) /* (MC) MC Flash Result Register*/
#define AT91C_MC_FMR    ((AT91_REG *) 	0xFFFFFF60) /* (MC) MC Flash Mode Register*/
/* ========== Register definition for PDC_SPI peripheral ========== */
#define AT91C_SPI_PTCR  ((AT91_REG *) 	0xFFFE0120) /* (PDC_SPI) PDC Transfer Control Register*/
#define AT91C_SPI_TPR   ((AT91_REG *) 	0xFFFE0108) /* (PDC_SPI) Transmit Pointer Register*/
#define AT91C_SPI_TCR   ((AT91_REG *) 	0xFFFE010C) /* (PDC_SPI) Transmit Counter Register*/
#define AT91C_SPI_RCR   ((AT91_REG *) 	0xFFFE0104) /* (PDC_SPI) Receive Counter Register*/
#define AT91C_SPI_PTSR  ((AT91_REG *) 	0xFFFE0124) /* (PDC_SPI) PDC Transfer Status Register*/
#define AT91C_SPI_RNPR  ((AT91_REG *) 	0xFFFE0110) /* (PDC_SPI) Receive Next Pointer Register*/
#define AT91C_SPI_RPR   ((AT91_REG *) 	0xFFFE0100) /* (PDC_SPI) Receive Pointer Register*/
#define AT91C_SPI_TNCR  ((AT91_REG *) 	0xFFFE011C) /* (PDC_SPI) Transmit Next Counter Register*/
#define AT91C_SPI_RNCR  ((AT91_REG *) 	0xFFFE0114) /* (PDC_SPI) Receive Next Counter Register*/
#define AT91C_SPI_TNPR  ((AT91_REG *) 	0xFFFE0118) /* (PDC_SPI) Transmit Next Pointer Register*/
/* ========== Register definition for SPI peripheral ========== */
#define AT91C_SPI_IER   ((AT91_REG *) 	0xFFFE0014) /* (SPI) Interrupt Enable Register*/
#define AT91C_SPI_SR    ((AT91_REG *) 	0xFFFE0010) /* (SPI) Status Register*/
#define AT91C_SPI_IDR   ((AT91_REG *) 	0xFFFE0018) /* (SPI) Interrupt Disable Register*/
#define AT91C_SPI_CR    ((AT91_REG *) 	0xFFFE0000) /* (SPI) Control Register*/
#define AT91C_SPI_MR    ((AT91_REG *) 	0xFFFE0004) /* (SPI) Mode Register*/
#define AT91C_SPI_IMR   ((AT91_REG *) 	0xFFFE001C) /* (SPI) Interrupt Mask Register*/
#define AT91C_SPI_TDR   ((AT91_REG *) 	0xFFFE000C) /* (SPI) Transmit Data Register*/
#define AT91C_SPI_RDR   ((AT91_REG *) 	0xFFFE0008) /* (SPI) Receive Data Register*/
#define AT91C_SPI_CSR   ((AT91_REG *) 	0xFFFE0030) /* (SPI) Chip Select Register*/
/* ========== Register definition for PDC_US1 peripheral ========== */
#define AT91C_US1_RNCR  ((AT91_REG *) 	0xFFFC4114) /* (PDC_US1) Receive Next Counter Register*/
#define AT91C_US1_PTCR  ((AT91_REG *) 	0xFFFC4120) /* (PDC_US1) PDC Transfer Control Register*/
#define AT91C_US1_TCR   ((AT91_REG *) 	0xFFFC410C) /* (PDC_US1) Transmit Counter Register*/
#define AT91C_US1_PTSR  ((AT91_REG *) 	0xFFFC4124) /* (PDC_US1) PDC Transfer Status Register*/
#define AT91C_US1_TNPR  ((AT91_REG *) 	0xFFFC4118) /* (PDC_US1) Transmit Next Pointer Register*/
#define AT91C_US1_RCR   ((AT91_REG *) 	0xFFFC4104) /* (PDC_US1) Receive Counter Register*/
#define AT91C_US1_RNPR  ((AT91_REG *) 	0xFFFC4110) /* (PDC_US1) Receive Next Pointer Register*/
#define AT91C_US1_RPR   ((AT91_REG *) 	0xFFFC4100) /* (PDC_US1) Receive Pointer Register*/
#define AT91C_US1_TNCR  ((AT91_REG *) 	0xFFFC411C) /* (PDC_US1) Transmit Next Counter Register*/
#define AT91C_US1_TPR   ((AT91_REG *) 	0xFFFC4108) /* (PDC_US1) Transmit Pointer Register*/
/* ========== Register definition for US1 peripheral ========== */
#define AT91C_US1_IF    ((AT91_REG *) 	0xFFFC404C) /* (US1) IRDA_FILTER Register*/
#define AT91C_US1_NER   ((AT91_REG *) 	0xFFFC4044) /* (US1) Nb Errors Register*/
#define AT91C_US1_RTOR  ((AT91_REG *) 	0xFFFC4024) /* (US1) Receiver Time-out Register*/
#define AT91C_US1_CSR   ((AT91_REG *) 	0xFFFC4014) /* (US1) Channel Status Register*/
#define AT91C_US1_IDR   ((AT91_REG *) 	0xFFFC400C) /* (US1) Interrupt Disable Register*/
#define AT91C_US1_IER   ((AT91_REG *) 	0xFFFC4008) /* (US1) Interrupt Enable Register*/
#define AT91C_US1_THR   ((AT91_REG *) 	0xFFFC401C) /* (US1) Transmitter Holding Register*/
#define AT91C_US1_TTGR  ((AT91_REG *) 	0xFFFC4028) /* (US1) Transmitter Time-guard Register*/
#define AT91C_US1_RHR   ((AT91_REG *) 	0xFFFC4018) /* (US1) Receiver Holding Register*/
#define AT91C_US1_BRGR  ((AT91_REG *) 	0xFFFC4020) /* (US1) Baud Rate Generator Register*/
#define AT91C_US1_IMR   ((AT91_REG *) 	0xFFFC4010) /* (US1) Interrupt Mask Register*/
#define AT91C_US1_FIDI  ((AT91_REG *) 	0xFFFC4040) /* (US1) FI_DI_Ratio Register*/
#define AT91C_US1_CR    ((AT91_REG *) 	0xFFFC4000) /* (US1) Control Register*/
#define AT91C_US1_MR    ((AT91_REG *) 	0xFFFC4004) /* (US1) Mode Register*/
/* ========== Register definition for PDC_US0 peripheral ========== */
#define AT91C_US0_TNPR  ((AT91_REG *) 	0xFFFC0118) /* (PDC_US0) Transmit Next Pointer Register*/
#define AT91C_US0_RNPR  ((AT91_REG *) 	0xFFFC0110) /* (PDC_US0) Receive Next Pointer Register*/
#define AT91C_US0_TCR   ((AT91_REG *) 	0xFFFC010C) /* (PDC_US0) Transmit Counter Register*/
#define AT91C_US0_PTCR  ((AT91_REG *) 	0xFFFC0120) /* (PDC_US0) PDC Transfer Control Register*/
#define AT91C_US0_PTSR  ((AT91_REG *) 	0xFFFC0124) /* (PDC_US0) PDC Transfer Status Register*/
#define AT91C_US0_TNCR  ((AT91_REG *) 	0xFFFC011C) /* (PDC_US0) Transmit Next Counter Register*/
#define AT91C_US0_TPR   ((AT91_REG *) 	0xFFFC0108) /* (PDC_US0) Transmit Pointer Register*/
#define AT91C_US0_RCR   ((AT91_REG *) 	0xFFFC0104) /* (PDC_US0) Receive Counter Register*/
#define AT91C_US0_RPR   ((AT91_REG *) 	0xFFFC0100) /* (PDC_US0) Receive Pointer Register*/
#define AT91C_US0_RNCR  ((AT91_REG *) 	0xFFFC0114) /* (PDC_US0) Receive Next Counter Register*/
/* ========== Register definition for US0 peripheral ========== */
#define AT91C_US0_BRGR  ((AT91_REG *) 	0xFFFC0020) /* (US0) Baud Rate Generator Register*/
#define AT91C_US0_NER   ((AT91_REG *) 	0xFFFC0044) /* (US0) Nb Errors Register*/
#define AT91C_US0_CR    ((AT91_REG *) 	0xFFFC0000) /* (US0) Control Register*/
#define AT91C_US0_IMR   ((AT91_REG *) 	0xFFFC0010) /* (US0) Interrupt Mask Register*/
#define AT91C_US0_FIDI  ((AT91_REG *) 	0xFFFC0040) /* (US0) FI_DI_Ratio Register*/
#define AT91C_US0_TTGR  ((AT91_REG *) 	0xFFFC0028) /* (US0) Transmitter Time-guard Register*/
#define AT91C_US0_MR    ((AT91_REG *) 	0xFFFC0004) /* (US0) Mode Register*/
#define AT91C_US0_RTOR  ((AT91_REG *) 	0xFFFC0024) /* (US0) Receiver Time-out Register*/
#define AT91C_US0_CSR   ((AT91_REG *) 	0xFFFC0014) /* (US0) Channel Status Register*/
#define AT91C_US0_RHR   ((AT91_REG *) 	0xFFFC0018) /* (US0) Receiver Holding Register*/
#define AT91C_US0_IDR   ((AT91_REG *) 	0xFFFC000C) /* (US0) Interrupt Disable Register*/
#define AT91C_US0_THR   ((AT91_REG *) 	0xFFFC001C) /* (US0) Transmitter Holding Register*/
#define AT91C_US0_IF    ((AT91_REG *) 	0xFFFC004C) /* (US0) IRDA_FILTER Register*/
#define AT91C_US0_IER   ((AT91_REG *) 	0xFFFC0008) /* (US0) Interrupt Enable Register*/
/* ========== Register definition for PDC_TWI peripheral ========== */
#define AT91C_TWI_TNCR  ((AT91_REG *) 	0xFFFB811C) /* (PDC_TWI) Transmit Next Counter Register*/
#define AT91C_TWI_RNCR  ((AT91_REG *) 	0xFFFB8114) /* (PDC_TWI) Receive Next Counter Register*/
#define AT91C_TWI_TNPR  ((AT91_REG *) 	0xFFFB8118) /* (PDC_TWI) Transmit Next Pointer Register*/
#define AT91C_TWI_PTCR  ((AT91_REG *) 	0xFFFB8120) /* (PDC_TWI) PDC Transfer Control Register*/
#define AT91C_TWI_TCR   ((AT91_REG *) 	0xFFFB810C) /* (PDC_TWI) Transmit Counter Register*/
#define AT91C_TWI_RPR   ((AT91_REG *) 	0xFFFB8100) /* (PDC_TWI) Receive Pointer Register*/
#define AT91C_TWI_TPR   ((AT91_REG *) 	0xFFFB8108) /* (PDC_TWI) Transmit Pointer Register*/
#define AT91C_TWI_RCR   ((AT91_REG *) 	0xFFFB8104) /* (PDC_TWI) Receive Counter Register*/
#define AT91C_TWI_PTSR  ((AT91_REG *) 	0xFFFB8124) /* (PDC_TWI) PDC Transfer Status Register*/
#define AT91C_TWI_RNPR  ((AT91_REG *) 	0xFFFB8110) /* (PDC_TWI) Receive Next Pointer Register*/
/* ========== Register definition for TWI peripheral ========== */
#define AT91C_TWI_IER   ((AT91_REG *) 	0xFFFB8024) /* (TWI) Interrupt Enable Register*/
#define AT91C_TWI_CR    ((AT91_REG *) 	0xFFFB8000) /* (TWI) Control Register*/
#define AT91C_TWI_SR    ((AT91_REG *) 	0xFFFB8020) /* (TWI) Status Register*/
#define AT91C_TWI_IMR   ((AT91_REG *) 	0xFFFB802C) /* (TWI) Interrupt Mask Register*/
#define AT91C_TWI_THR   ((AT91_REG *) 	0xFFFB8034) /* (TWI) Transmit Holding Register*/
#define AT91C_TWI_IDR   ((AT91_REG *) 	0xFFFB8028) /* (TWI) Interrupt Disable Register*/
#define AT91C_TWI_IADR  ((AT91_REG *) 	0xFFFB800C) /* (TWI) Internal Address Register*/
#define AT91C_TWI_MMR   ((AT91_REG *) 	0xFFFB8004) /* (TWI) Master Mode Register*/
#define AT91C_TWI_CWGR  ((AT91_REG *) 	0xFFFB8010) /* (TWI) Clock Waveform Generator Register*/
#define AT91C_TWI_RHR   ((AT91_REG *) 	0xFFFB8030) /* (TWI) Receive Holding Register*/
#define AT91C_TWI_SMR   ((AT91_REG *) 	0xFFFB8008) /* (TWI) Slave Mode Register*/
/* ========== Register definition for PWMC_CH3 peripheral ========== */
#define AT91C_CH3_CUPDR ((AT91_REG *) 	0xFFFCC270) /* (PWMC_CH3) Channel Update Register*/
#define AT91C_CH3_Reserved ((AT91_REG *) 	0xFFFCC274) /* (PWMC_CH3) Reserved*/
#define AT91C_CH3_CPRDR ((AT91_REG *) 	0xFFFCC268) /* (PWMC_CH3) Channel Period Register*/
#define AT91C_CH3_CDTYR ((AT91_REG *) 	0xFFFCC264) /* (PWMC_CH3) Channel Duty Cycle Register*/
#define AT91C_CH3_CCNTR ((AT91_REG *) 	0xFFFCC26C) /* (PWMC_CH3) Channel Counter Register*/
#define AT91C_CH3_CMR   ((AT91_REG *) 	0xFFFCC260) /* (PWMC_CH3) Channel Mode Register*/
/* ========== Register definition for PWMC_CH2 peripheral ========== */
#define AT91C_CH2_Reserved ((AT91_REG *) 	0xFFFCC254) /* (PWMC_CH2) Reserved*/
#define AT91C_CH2_CMR   ((AT91_REG *) 	0xFFFCC240) /* (PWMC_CH2) Channel Mode Register*/
#define AT91C_CH2_CCNTR ((AT91_REG *) 	0xFFFCC24C) /* (PWMC_CH2) Channel Counter Register*/
#define AT91C_CH2_CPRDR ((AT91_REG *) 	0xFFFCC248) /* (PWMC_CH2) Channel Period Register*/
#define AT91C_CH2_CUPDR ((AT91_REG *) 	0xFFFCC250) /* (PWMC_CH2) Channel Update Register*/
#define AT91C_CH2_CDTYR ((AT91_REG *) 	0xFFFCC244) /* (PWMC_CH2) Channel Duty Cycle Register*/
/* ========== Register definition for PWMC_CH1 peripheral ========== */
#define AT91C_CH1_Reserved ((AT91_REG *) 	0xFFFCC234) /* (PWMC_CH1) Reserved*/
#define AT91C_CH1_CUPDR ((AT91_REG *) 	0xFFFCC230) /* (PWMC_CH1) Channel Update Register*/
#define AT91C_CH1_CPRDR ((AT91_REG *) 	0xFFFCC228) /* (PWMC_CH1) Channel Period Register*/
#define AT91C_CH1_CCNTR ((AT91_REG *) 	0xFFFCC22C) /* (PWMC_CH1) Channel Counter Register*/
#define AT91C_CH1_CDTYR ((AT91_REG *) 	0xFFFCC224) /* (PWMC_CH1) Channel Duty Cycle Register*/
#define AT91C_CH1_CMR   ((AT91_REG *) 	0xFFFCC220) /* (PWMC_CH1) Channel Mode Register*/
/* ========== Register definition for PWMC_CH0 peripheral ========== */
#define AT91C_CH0_Reserved ((AT91_REG *) 	0xFFFCC214) /* (PWMC_CH0) Reserved*/
#define AT91C_CH0_CPRDR ((AT91_REG *) 	0xFFFCC208) /* (PWMC_CH0) Channel Period Register*/
#define AT91C_CH0_CDTYR ((AT91_REG *) 	0xFFFCC204) /* (PWMC_CH0) Channel Duty Cycle Register*/
#define AT91C_CH0_CMR   ((AT91_REG *) 	0xFFFCC200) /* (PWMC_CH0) Channel Mode Register*/
#define AT91C_CH0_CUPDR ((AT91_REG *) 	0xFFFCC210) /* (PWMC_CH0) Channel Update Register*/
#define AT91C_CH0_CCNTR ((AT91_REG *) 	0xFFFCC20C) /* (PWMC_CH0) Channel Counter Register*/
/* ========== Register definition for PWMC peripheral ========== */
#define AT91C_PWMC_IDR  ((AT91_REG *) 	0xFFFCC014) /* (PWMC) PWMC Interrupt Disable Register*/
#define AT91C_PWMC_DIS  ((AT91_REG *) 	0xFFFCC008) /* (PWMC) PWMC Disable Register*/
#define AT91C_PWMC_IER  ((AT91_REG *) 	0xFFFCC010) /* (PWMC) PWMC Interrupt Enable Register*/
#define AT91C_PWMC_VR   ((AT91_REG *) 	0xFFFCC0FC) /* (PWMC) PWMC Version Register*/
#define AT91C_PWMC_ISR  ((AT91_REG *) 	0xFFFCC01C) /* (PWMC) PWMC Interrupt Status Register*/
#define AT91C_PWMC_SR   ((AT91_REG *) 	0xFFFCC00C) /* (PWMC) PWMC Status Register*/
#define AT91C_PWMC_IMR  ((AT91_REG *) 	0xFFFCC018) /* (PWMC) PWMC Interrupt Mask Register*/
#define AT91C_PWMC_MR   ((AT91_REG *) 	0xFFFCC000) /* (PWMC) PWMC Mode Register*/
#define AT91C_PWMC_ENA  ((AT91_REG *) 	0xFFFCC004) /* (PWMC) PWMC Enable Register*/
/* ========== Register definition for TC0 peripheral ========== */
#define AT91C_TC0_SR    ((AT91_REG *) 	0xFFFA0020) /* (TC0) Status Register*/
#define AT91C_TC0_RC    ((AT91_REG *) 	0xFFFA001C) /* (TC0) Register C*/
#define AT91C_TC0_RB    ((AT91_REG *) 	0xFFFA0018) /* (TC0) Register B*/
#define AT91C_TC0_CCR   ((AT91_REG *) 	0xFFFA0000) /* (TC0) Channel Control Register*/
#define AT91C_TC0_CMR   ((AT91_REG *) 	0xFFFA0004) /* (TC0) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC0_IER   ((AT91_REG *) 	0xFFFA0024) /* (TC0) Interrupt Enable Register*/
#define AT91C_TC0_RA    ((AT91_REG *) 	0xFFFA0014) /* (TC0) Register A*/
#define AT91C_TC0_IDR   ((AT91_REG *) 	0xFFFA0028) /* (TC0) Interrupt Disable Register*/
#define AT91C_TC0_CV    ((AT91_REG *) 	0xFFFA0010) /* (TC0) Counter Value*/
#define AT91C_TC0_IMR   ((AT91_REG *) 	0xFFFA002C) /* (TC0) Interrupt Mask Register*/
/* ========== Register definition for TC1 peripheral ========== */
#define AT91C_TC1_RB    ((AT91_REG *) 	0xFFFA0058) /* (TC1) Register B*/
#define AT91C_TC1_CCR   ((AT91_REG *) 	0xFFFA0040) /* (TC1) Channel Control Register*/
#define AT91C_TC1_IER   ((AT91_REG *) 	0xFFFA0064) /* (TC1) Interrupt Enable Register*/
#define AT91C_TC1_IDR   ((AT91_REG *) 	0xFFFA0068) /* (TC1) Interrupt Disable Register*/
#define AT91C_TC1_SR    ((AT91_REG *) 	0xFFFA0060) /* (TC1) Status Register*/
#define AT91C_TC1_CMR   ((AT91_REG *) 	0xFFFA0044) /* (TC1) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC1_RA    ((AT91_REG *) 	0xFFFA0054) /* (TC1) Register A*/
#define AT91C_TC1_RC    ((AT91_REG *) 	0xFFFA005C) /* (TC1) Register C*/
#define AT91C_TC1_IMR   ((AT91_REG *) 	0xFFFA006C) /* (TC1) Interrupt Mask Register*/
#define AT91C_TC1_CV    ((AT91_REG *) 	0xFFFA0050) /* (TC1) Counter Value*/
/* ========== Register definition for TC2 peripheral ========== */
#define AT91C_TC2_CMR   ((AT91_REG *) 	0xFFFA0084) /* (TC2) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC2_CCR   ((AT91_REG *) 	0xFFFA0080) /* (TC2) Channel Control Register*/
#define AT91C_TC2_CV    ((AT91_REG *) 	0xFFFA0090) /* (TC2) Counter Value*/
#define AT91C_TC2_RA    ((AT91_REG *) 	0xFFFA0094) /* (TC2) Register A*/
#define AT91C_TC2_RB    ((AT91_REG *) 	0xFFFA0098) /* (TC2) Register B*/
#define AT91C_TC2_IDR   ((AT91_REG *) 	0xFFFA00A8) /* (TC2) Interrupt Disable Register*/
#define AT91C_TC2_IMR   ((AT91_REG *) 	0xFFFA00AC) /* (TC2) Interrupt Mask Register*/
#define AT91C_TC2_RC    ((AT91_REG *) 	0xFFFA009C) /* (TC2) Register C*/
#define AT91C_TC2_IER   ((AT91_REG *) 	0xFFFA00A4) /* (TC2) Interrupt Enable Register*/
#define AT91C_TC2_SR    ((AT91_REG *) 	0xFFFA00A0) /* (TC2) Status Register*/
/* ========== Register definition for TCB peripheral ========== */
#define AT91C_TCB_BMR   ((AT91_REG *) 	0xFFFA00C4) /* (TCB) TC Block Mode Register*/
#define AT91C_TCB_BCR   ((AT91_REG *) 	0xFFFA00C0) /* (TCB) TC Block Control Register*/
/* ========== Register definition for SLCDC peripheral ========== */
#define AT91C_SLCDC_IMR ((AT91_REG *) 	0xFFFB4028) /* (SLCDC) Interrupt Mask Register*/
#define AT91C_SLCDC_IER ((AT91_REG *) 	0xFFFB4020) /* (SLCDC) Interrupt Enable Register*/
#define AT91C_SLCDC_MEM ((AT91_REG *) 	0xFFFB4200) /* (SLCDC) Memory Register*/
#define AT91C_SLCDC_DR  ((AT91_REG *) 	0xFFFB400C) /* (SLCDC) Display Register*/
#define AT91C_SLCDC_MR  ((AT91_REG *) 	0xFFFB4004) /* (SLCDC) Mode Register*/
#define AT91C_SLCDC_ISR ((AT91_REG *) 	0xFFFB402C) /* (SLCDC) Interrupt Status Register*/
#define AT91C_SLCDC_IDR ((AT91_REG *) 	0xFFFB4024) /* (SLCDC) Interrupt Disable Register*/
#define AT91C_SLCDC_CR  ((AT91_REG *) 	0xFFFB4000) /* (SLCDC) Control Register*/
#define AT91C_SLCDC_SR  ((AT91_REG *) 	0xFFFB4010) /* (SLCDC) Status Register*/
#define AT91C_SLCDC_FRR ((AT91_REG *) 	0xFFFB4008) /* (SLCDC) Frame Rate Register*/
/* ========== Register definition for PDC_ADC peripheral ========== */
#define AT91C_ADC_PTSR  ((AT91_REG *) 	0xFFFD8124) /* (PDC_ADC) PDC Transfer Status Register*/
#define AT91C_ADC_PTCR  ((AT91_REG *) 	0xFFFD8120) /* (PDC_ADC) PDC Transfer Control Register*/
#define AT91C_ADC_TNPR  ((AT91_REG *) 	0xFFFD8118) /* (PDC_ADC) Transmit Next Pointer Register*/
#define AT91C_ADC_TNCR  ((AT91_REG *) 	0xFFFD811C) /* (PDC_ADC) Transmit Next Counter Register*/
#define AT91C_ADC_RNPR  ((AT91_REG *) 	0xFFFD8110) /* (PDC_ADC) Receive Next Pointer Register*/
#define AT91C_ADC_RNCR  ((AT91_REG *) 	0xFFFD8114) /* (PDC_ADC) Receive Next Counter Register*/
#define AT91C_ADC_RPR   ((AT91_REG *) 	0xFFFD8100) /* (PDC_ADC) Receive Pointer Register*/
#define AT91C_ADC_TCR   ((AT91_REG *) 	0xFFFD810C) /* (PDC_ADC) Transmit Counter Register*/
#define AT91C_ADC_TPR   ((AT91_REG *) 	0xFFFD8108) /* (PDC_ADC) Transmit Pointer Register*/
#define AT91C_ADC_RCR   ((AT91_REG *) 	0xFFFD8104) /* (PDC_ADC) Receive Counter Register*/
/* ========== Register definition for ADC peripheral ========== */
#define AT91C_ADC_CDR2  ((AT91_REG *) 	0xFFFD8038) /* (ADC) ADC Channel Data Register 2*/
#define AT91C_ADC_CDR3  ((AT91_REG *) 	0xFFFD803C) /* (ADC) ADC Channel Data Register 3*/
#define AT91C_ADC_CDR0  ((AT91_REG *) 	0xFFFD8030) /* (ADC) ADC Channel Data Register 0*/
#define AT91C_ADC_CDR5  ((AT91_REG *) 	0xFFFD8044) /* (ADC) ADC Channel Data Register 5*/
#define AT91C_ADC_CHDR  ((AT91_REG *) 	0xFFFD8014) /* (ADC) ADC Channel Disable Register*/
#define AT91C_ADC_SR    ((AT91_REG *) 	0xFFFD801C) /* (ADC) ADC Status Register*/
#define AT91C_ADC_CDR4  ((AT91_REG *) 	0xFFFD8040) /* (ADC) ADC Channel Data Register 4*/
#define AT91C_ADC_CDR1  ((AT91_REG *) 	0xFFFD8034) /* (ADC) ADC Channel Data Register 1*/
#define AT91C_ADC_LCDR  ((AT91_REG *) 	0xFFFD8020) /* (ADC) ADC Last Converted Data Register*/
#define AT91C_ADC_IDR   ((AT91_REG *) 	0xFFFD8028) /* (ADC) ADC Interrupt Disable Register*/
#define AT91C_ADC_CR    ((AT91_REG *) 	0xFFFD8000) /* (ADC) ADC Control Register*/
#define AT91C_ADC_CDR7  ((AT91_REG *) 	0xFFFD804C) /* (ADC) ADC Channel Data Register 7*/
#define AT91C_ADC_CDR6  ((AT91_REG *) 	0xFFFD8048) /* (ADC) ADC Channel Data Register 6*/
#define AT91C_ADC_IER   ((AT91_REG *) 	0xFFFD8024) /* (ADC) ADC Interrupt Enable Register*/
#define AT91C_ADC_CHER  ((AT91_REG *) 	0xFFFD8010) /* (ADC) ADC Channel Enable Register*/
#define AT91C_ADC_CHSR  ((AT91_REG *) 	0xFFFD8018) /* (ADC) ADC Channel Status Register*/
#define AT91C_ADC_MR    ((AT91_REG *) 	0xFFFD8004) /* (ADC) ADC Mode Register*/
#define AT91C_ADC_IMR   ((AT91_REG *) 	0xFFFD802C) /* (ADC) ADC Interrupt Mask Register*/

/* ******************************************************************************/
/*               PIO DEFINITIONS FOR AT91SAM7L128*/
/* ******************************************************************************/
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) /* Pin Controlled by PA0*/
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) /* Pin Controlled by PA1*/
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) /* Pin Controlled by PA10*/
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) /* Pin Controlled by PA11*/
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) /* Pin Controlled by PA12*/
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) /* Pin Controlled by PA13*/
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) /* Pin Controlled by PA14*/
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) /* Pin Controlled by PA15*/
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) /* Pin Controlled by PA16*/
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) /* Pin Controlled by PA17*/
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) /* Pin Controlled by PA18*/
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) /* Pin Controlled by PA19*/
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) /* Pin Controlled by PA2*/
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) /* Pin Controlled by PA20*/
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) /* Pin Controlled by PA21*/
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) /* Pin Controlled by PA22*/
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) /* Pin Controlled by PA23*/
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) /* Pin Controlled by PA24*/
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) /* Pin Controlled by PA25*/
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) /* Pin Controlled by PA3*/
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) /* Pin Controlled by PA4*/
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) /* Pin Controlled by PA5*/
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) /* Pin Controlled by PA6*/
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) /* Pin Controlled by PA7*/
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) /* Pin Controlled by PA8*/
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) /* Pin Controlled by PA9*/
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) /* Pin Controlled by PB0*/
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) /* Pin Controlled by PB1*/
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) /* Pin Controlled by PB10*/
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) /* Pin Controlled by PB11*/
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) /* Pin Controlled by PB12*/
#define AT91C_PB12_NPCS3    ((unsigned int) AT91C_PIO_PB12) /*  */
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) /* Pin Controlled by PB13*/
#define AT91C_PB13_NPCS2    ((unsigned int) AT91C_PIO_PB13) /*  */
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) /* Pin Controlled by PB14*/
#define AT91C_PB14_NPCS1    ((unsigned int) AT91C_PIO_PB14) /*  */
#define AT91C_PIO_PB15       ((unsigned int) 1 << 15) /* Pin Controlled by PB15*/
#define AT91C_PB15_RTS1     ((unsigned int) AT91C_PIO_PB15) /*  */
#define AT91C_PIO_PB16       ((unsigned int) 1 << 16) /* Pin Controlled by PB16*/
#define AT91C_PB16_RTS0     ((unsigned int) AT91C_PIO_PB16) /*  */
#define AT91C_PIO_PB17       ((unsigned int) 1 << 17) /* Pin Controlled by PB17*/
#define AT91C_PB17_DTR1     ((unsigned int) AT91C_PIO_PB17) /*  */
#define AT91C_PIO_PB18       ((unsigned int) 1 << 18) /* Pin Controlled by PB18*/
#define AT91C_PB18_PWM0     ((unsigned int) AT91C_PIO_PB18) /*  */
#define AT91C_PIO_PB19       ((unsigned int) 1 << 19) /* Pin Controlled by PB19*/
#define AT91C_PB19_PWM1     ((unsigned int) AT91C_PIO_PB19) /*  */
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) /* Pin Controlled by PB2*/
#define AT91C_PIO_PB20       ((unsigned int) 1 << 20) /* Pin Controlled by PB20*/
#define AT91C_PB20_PWM2     ((unsigned int) AT91C_PIO_PB20) /*  */
#define AT91C_PIO_PB21       ((unsigned int) 1 << 21) /* Pin Controlled by PB21*/
#define AT91C_PB21_PWM3     ((unsigned int) AT91C_PIO_PB21) /*  */
#define AT91C_PIO_PB22       ((unsigned int) 1 << 22) /* Pin Controlled by PB22*/
#define AT91C_PB22_NPCS1    ((unsigned int) AT91C_PIO_PB22) /*  */
#define AT91C_PB22_PCK1     ((unsigned int) AT91C_PIO_PB22) /*  */
#define AT91C_PIO_PB23       ((unsigned int) 1 << 23) /* Pin Controlled by PB23*/
#define AT91C_PB23_PCK0     ((unsigned int) AT91C_PIO_PB23) /*  */
#define AT91C_PB23_NPCS3    ((unsigned int) AT91C_PIO_PB23) /*  */
#define AT91C_PIO_PB24       ((unsigned int) 1 << 24) /* Pin Controlled by PB24*/
#define AT91C_PIO_PB25       ((unsigned int) 1 << 25) /* Pin Controlled by PB25*/
#define AT91C_PB25_PCK0     ((unsigned int) AT91C_PIO_PB25) /*  */
#define AT91C_PB25_NPCS3    ((unsigned int) AT91C_PIO_PB25) /*  */
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) /* Pin Controlled by PB3*/
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) /* Pin Controlled by PB4*/
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) /* Pin Controlled by PB5*/
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) /* Pin Controlled by PB6*/
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) /* Pin Controlled by PB7*/
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) /* Pin Controlled by PB8*/
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) /* Pin Controlled by PB9*/
#define AT91C_PIO_PC0        ((unsigned int) 1 <<  0) /* Pin Controlled by PC0*/
#define AT91C_PC0_CTS1     ((unsigned int) AT91C_PIO_PC0) /*  */
#define AT91C_PC0_PWM2     ((unsigned int) AT91C_PIO_PC0) /*  */
#define AT91C_PIO_PC1        ((unsigned int) 1 <<  1) /* Pin Controlled by PC1*/
#define AT91C_PC1_DCD1     ((unsigned int) AT91C_PIO_PC1) /*  */
#define AT91C_PC1_TIOA2    ((unsigned int) AT91C_PIO_PC1) /*  */
#define AT91C_PIO_PC10       ((unsigned int) 1 << 10) /* Pin Controlled by PC10*/
#define AT91C_PC10_TWD      ((unsigned int) AT91C_PIO_PC10) /*  */
#define AT91C_PC10_NPCS3    ((unsigned int) AT91C_PIO_PC10) /*  */
#define AT91C_PIO_PC11       ((unsigned int) 1 << 11) /* Pin Controlled by PC11*/
#define AT91C_PC11_TWCK     ((unsigned int) AT91C_PIO_PC11) /*  */
#define AT91C_PC11_TCLK0    ((unsigned int) AT91C_PIO_PC11) /*  */
#define AT91C_PIO_PC12       ((unsigned int) 1 << 12) /* Pin Controlled by PC12*/
#define AT91C_PC12_RXD0     ((unsigned int) AT91C_PIO_PC12) /*  */
#define AT91C_PC12_NPCS3    ((unsigned int) AT91C_PIO_PC12) /*  */
#define AT91C_PIO_PC13       ((unsigned int) 1 << 13) /* Pin Controlled by PC13*/
#define AT91C_PC13_TXD0     ((unsigned int) AT91C_PIO_PC13) /*  */
#define AT91C_PC13_PCK0     ((unsigned int) AT91C_PIO_PC13) /*  */
#define AT91C_PIO_PC14       ((unsigned int) 1 << 14) /* Pin Controlled by PC14*/
#define AT91C_PC14_RTS0     ((unsigned int) AT91C_PIO_PC14) /*  */
#define AT91C_PC14_ADTRG    ((unsigned int) AT91C_PIO_PC14) /*  */
#define AT91C_PIO_PC15       ((unsigned int) 1 << 15) /* Pin Controlled by PC15*/
#define AT91C_PC15_CTS0     ((unsigned int) AT91C_PIO_PC15) /*  */
#define AT91C_PC15_PWM3     ((unsigned int) AT91C_PIO_PC15) /*  */
#define AT91C_PIO_PC16       ((unsigned int) 1 << 16) /* Pin Controlled by PC16*/
#define AT91C_PC16_DRXD     ((unsigned int) AT91C_PIO_PC16) /*  */
#define AT91C_PC16_NPCS1    ((unsigned int) AT91C_PIO_PC16) /*  */
#define AT91C_PIO_PC17       ((unsigned int) 1 << 17) /* Pin Controlled by PC17*/
#define AT91C_PC17_DTXD     ((unsigned int) AT91C_PIO_PC17) /*  */
#define AT91C_PC17_NPCS2    ((unsigned int) AT91C_PIO_PC17) /*  */
#define AT91C_PIO_PC18       ((unsigned int) 1 << 18) /* Pin Controlled by PC18*/
#define AT91C_PC18_NPCS0    ((unsigned int) AT91C_PIO_PC18) /*  */
#define AT91C_PC18_PWM0     ((unsigned int) AT91C_PIO_PC18) /*  */
#define AT91C_PIO_PC19       ((unsigned int) 1 << 19) /* Pin Controlled by PC19*/
#define AT91C_PC19_MISO     ((unsigned int) AT91C_PIO_PC19) /*  */
#define AT91C_PC19_PWM1     ((unsigned int) AT91C_PIO_PC19) /*  */
#define AT91C_PIO_PC2        ((unsigned int) 1 <<  2) /* Pin Controlled by PC2*/
#define AT91C_PC2_DTR1     ((unsigned int) AT91C_PIO_PC2) /*  */
#define AT91C_PC2_TIOB2    ((unsigned int) AT91C_PIO_PC2) /*  */
#define AT91C_PIO_PC20       ((unsigned int) 1 << 20) /* Pin Controlled by PC20*/
#define AT91C_PC20_MOSI     ((unsigned int) AT91C_PIO_PC20) /*  */
#define AT91C_PC20_PWM2     ((unsigned int) AT91C_PIO_PC20) /*  */
#define AT91C_PIO_PC21       ((unsigned int) 1 << 21) /* Pin Controlled by PC21*/
#define AT91C_PC21_SPCK     ((unsigned int) AT91C_PIO_PC21) /*  */
#define AT91C_PC21_PWM3     ((unsigned int) AT91C_PIO_PC21) /*  */
#define AT91C_PIO_PC22       ((unsigned int) 1 << 22) /* Pin Controlled by PC22*/
#define AT91C_PC22_NPCS3    ((unsigned int) AT91C_PIO_PC22) /*  */
#define AT91C_PC22_TIOA1    ((unsigned int) AT91C_PIO_PC22) /*  */
#define AT91C_PIO_PC23       ((unsigned int) 1 << 23) /* Pin Controlled by PC23*/
#define AT91C_PC23_PCK0     ((unsigned int) AT91C_PIO_PC23) /*  */
#define AT91C_PC23_TIOB1    ((unsigned int) AT91C_PIO_PC23) /*  */
#define AT91C_PIO_PC24       ((unsigned int) 1 << 24) /* Pin Controlled by PC24*/
#define AT91C_PC24_RXD1     ((unsigned int) AT91C_PIO_PC24) /*  */
#define AT91C_PC24_PCK1     ((unsigned int) AT91C_PIO_PC24) /*  */
#define AT91C_PIO_PC25       ((unsigned int) 1 << 25) /* Pin Controlled by PC25*/
#define AT91C_PC25_TXD1     ((unsigned int) AT91C_PIO_PC25) /*  */
#define AT91C_PC25_PCK2     ((unsigned int) AT91C_PIO_PC25) /*  */
#define AT91C_PIO_PC26       ((unsigned int) 1 << 26) /* Pin Controlled by PC26*/
#define AT91C_PC26_RTS0     ((unsigned int) AT91C_PIO_PC26) /*  */
#define AT91C_PC26_FIQ      ((unsigned int) AT91C_PIO_PC26) /*  */
#define AT91C_PIO_PC27       ((unsigned int) 1 << 27) /* Pin Controlled by PC27*/
#define AT91C_PC27_NPCS2    ((unsigned int) AT91C_PIO_PC27) /*  */
#define AT91C_PC27_IRQ0     ((unsigned int) AT91C_PIO_PC27) /*  */
#define AT91C_PIO_PC28       ((unsigned int) 1 << 28) /* Pin Controlled by PC28*/
#define AT91C_PC28_SCK1     ((unsigned int) AT91C_PIO_PC28) /*  */
#define AT91C_PC28_PWM0     ((unsigned int) AT91C_PIO_PC28) /*  */
#define AT91C_PIO_PC29       ((unsigned int) 1 << 29) /* Pin Controlled by PC29*/
#define AT91C_PC29_RTS1     ((unsigned int) AT91C_PIO_PC29) /*  */
#define AT91C_PC29_PWM1     ((unsigned int) AT91C_PIO_PC29) /*  */
#define AT91C_PIO_PC3        ((unsigned int) 1 <<  3) /* Pin Controlled by PC3*/
#define AT91C_PC3_DSR1     ((unsigned int) AT91C_PIO_PC3) /*  */
#define AT91C_PC3_TCLK1    ((unsigned int) AT91C_PIO_PC3) /*  */
#define AT91C_PIO_PC4        ((unsigned int) 1 <<  4) /* Pin Controlled by PC4*/
#define AT91C_PC4_RI1      ((unsigned int) AT91C_PIO_PC4) /*  */
#define AT91C_PC4_TCLK2    ((unsigned int) AT91C_PIO_PC4) /*  */
#define AT91C_PIO_PC5        ((unsigned int) 1 <<  5) /* Pin Controlled by PC5*/
#define AT91C_PC5_IRQ1     ((unsigned int) AT91C_PIO_PC5) /*  */
#define AT91C_PC5_NPCS2    ((unsigned int) AT91C_PIO_PC5) /*  */
#define AT91C_PIO_PC6        ((unsigned int) 1 <<  6) /* Pin Controlled by PC6*/
#define AT91C_PC6_NPCS1    ((unsigned int) AT91C_PIO_PC6) /*  */
#define AT91C_PC6_PCK2     ((unsigned int) AT91C_PIO_PC6) /*  */
#define AT91C_PIO_PC7        ((unsigned int) 1 <<  7) /* Pin Controlled by PC7*/
#define AT91C_PC7_PWM0     ((unsigned int) AT91C_PIO_PC7) /*  */
#define AT91C_PC7_TIOA0    ((unsigned int) AT91C_PIO_PC7) /*  */
#define AT91C_PIO_PC8        ((unsigned int) 1 <<  8) /* Pin Controlled by PC8*/
#define AT91C_PC8_PWM1     ((unsigned int) AT91C_PIO_PC8) /*  */
#define AT91C_PC8_TIOB0    ((unsigned int) AT91C_PIO_PC8) /*  */
#define AT91C_PIO_PC9        ((unsigned int) 1 <<  9) /* Pin Controlled by PC9*/
#define AT91C_PC9_PWM2     ((unsigned int) AT91C_PIO_PC9) /*  */
#define AT91C_PC9_SCK0     ((unsigned int) AT91C_PIO_PC9) /*  */

/* ******************************************************************************/
/*               PERIPHERAL ID DEFINITIONS FOR AT91SAM7L128*/
/* ******************************************************************************/
#define AT91C_ID_FIQ    ((unsigned int)  0) /* Advanced Interrupt Controller (FIQ)*/
#define AT91C_ID_SYS    ((unsigned int)  1) /* System Peripheral*/
#define AT91C_ID_PIOA   ((unsigned int)  2) /* Parallel IO Controller A*/
#define AT91C_ID_PIOB   ((unsigned int)  3) /* Parallel IO Controller B*/
#define AT91C_ID_PIOC   ((unsigned int)  4) /* Parallel IO Controller C*/
#define AT91C_ID_SPI    ((unsigned int)  5) /* Serial Peripheral Interface 0*/
#define AT91C_ID_US0    ((unsigned int)  6) /* USART 0*/
#define AT91C_ID_US1    ((unsigned int)  7) /* USART 1*/
#define AT91C_ID_8_Reserved ((unsigned int)  8) /* Reserved*/
#define AT91C_ID_TWI    ((unsigned int)  9) /* Two-Wire Interface*/
#define AT91C_ID_PWMC   ((unsigned int) 10) /* PWM Controller*/
#define AT91C_ID_SLCD   ((unsigned int) 11) /* Segmented LCD Controller*/
#define AT91C_ID_TC0    ((unsigned int) 12) /* Timer Counter 0*/
#define AT91C_ID_TC1    ((unsigned int) 13) /* Timer Counter 1*/
#define AT91C_ID_TC2    ((unsigned int) 14) /* Timer Counter 2*/
#define AT91C_ID_ADC    ((unsigned int) 15) /* Analog-to-Digital Converter*/
#define AT91C_ID_16_Reserved ((unsigned int) 16) /* Reserved*/
#define AT91C_ID_17_Reserved ((unsigned int) 17) /* Reserved*/
#define AT91C_ID_18_Reserved ((unsigned int) 18) /* Reserved*/
#define AT91C_ID_19_Reserved ((unsigned int) 19) /* Reserved*/
#define AT91C_ID_20_Reserved ((unsigned int) 20) /* Reserved*/
#define AT91C_ID_21_Reserved ((unsigned int) 21) /* Reserved*/
#define AT91C_ID_22_Reserved ((unsigned int) 22) /* Reserved*/
#define AT91C_ID_23_Reserved ((unsigned int) 23) /* Reserved*/
#define AT91C_ID_24_Reserved ((unsigned int) 24) /* Reserved*/
#define AT91C_ID_25_Reserved ((unsigned int) 25) /* Reserved*/
#define AT91C_ID_26_Reserved ((unsigned int) 26) /* Reserved*/
#define AT91C_ID_27_Reserved ((unsigned int) 27) /* Reserved*/
#define AT91C_ID_28_Reserved ((unsigned int) 28) /* Reserved*/
#define AT91C_ID_IRQ0   ((unsigned int) 30) /* Advanced Interrupt Controller (IRQ0)*/
#define AT91C_ID_IRQ1   ((unsigned int) 31) /* Advanced Interrupt Controller (IRQ1)*/
#define AT91C_ALL_INT   ((unsigned int) 0xC000FEFF) /* ALL VALID INTERRUPTS*/

/* ******************************************************************************/
/*               BASE ADDRESS DEFINITIONS FOR AT91SAM7L128*/
/* ******************************************************************************/
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0xFFFFF000) /* (SYS) Base Address*/
#define AT91C_BASE_AIC       ((AT91PS_AIC) 	0xFFFFF000) /* (AIC) Base Address*/
#define AT91C_BASE_PDC_DBGU  ((AT91PS_PDC) 	0xFFFFF300) /* (PDC_DBGU) Base Address*/
#define AT91C_BASE_DBGU      ((AT91PS_DBGU) 	0xFFFFF200) /* (DBGU) Base Address*/
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0xFFFFF400) /* (PIOA) Base Address*/
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0xFFFFF600) /* (PIOB) Base Address*/
#define AT91C_BASE_PIOC      ((AT91PS_PIO) 	0xFFFFF800) /* (PIOC) Base Address*/
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0xFFFFFC20) /* (CKGR) Base Address*/
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0xFFFFFC00) /* (PMC) Base Address*/
#define AT91C_BASE_RSTC      ((AT91PS_RSTC) 	0xFFFFFD00) /* (RSTC) Base Address*/
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0xFFFFFD60) /* (RTC) Base Address*/
#define AT91C_BASE_PITC      ((AT91PS_PITC) 	0xFFFFFD40) /* (PITC) Base Address*/
#define AT91C_BASE_WDTC      ((AT91PS_WDTC) 	0xFFFFFD50) /* (WDTC) Base Address*/
#define AT91C_BASE_VREG      ((AT91PS_VREG) 	0xFFFFFD60) /* (VREG) Base Address*/
#define AT91C_BASE_MC        ((AT91PS_MC) 	0xFFFFFF00) /* (MC) Base Address*/
#define AT91C_BASE_PDC_SPI   ((AT91PS_PDC) 	0xFFFE0100) /* (PDC_SPI) Base Address*/
#define AT91C_BASE_SPI       ((AT91PS_SPI) 	0xFFFE0000) /* (SPI) Base Address*/
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0xFFFC4100) /* (PDC_US1) Base Address*/
#define AT91C_BASE_US1       ((AT91PS_USART) 	0xFFFC4000) /* (US1) Base Address*/
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0xFFFC0100) /* (PDC_US0) Base Address*/
#define AT91C_BASE_US0       ((AT91PS_USART) 	0xFFFC0000) /* (US0) Base Address*/
#define AT91C_BASE_PDC_TWI   ((AT91PS_PDC) 	0xFFFB8100) /* (PDC_TWI) Base Address*/
#define AT91C_BASE_TWI       ((AT91PS_TWI) 	0xFFFB8000) /* (TWI) Base Address*/
#define AT91C_BASE_PWMC_CH3  ((AT91PS_PWMC_CH) 	0xFFFCC260) /* (PWMC_CH3) Base Address*/
#define AT91C_BASE_PWMC_CH2  ((AT91PS_PWMC_CH) 	0xFFFCC240) /* (PWMC_CH2) Base Address*/
#define AT91C_BASE_PWMC_CH1  ((AT91PS_PWMC_CH) 	0xFFFCC220) /* (PWMC_CH1) Base Address*/
#define AT91C_BASE_PWMC_CH0  ((AT91PS_PWMC_CH) 	0xFFFCC200) /* (PWMC_CH0) Base Address*/
#define AT91C_BASE_PWMC      ((AT91PS_PWMC) 	0xFFFCC000) /* (PWMC) Base Address*/
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0xFFFA0000) /* (TC0) Base Address*/
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0xFFFA0040) /* (TC1) Base Address*/
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0xFFFA0080) /* (TC2) Base Address*/
#define AT91C_BASE_TCB       ((AT91PS_TCB) 	0xFFFA0000) /* (TCB) Base Address*/
#define AT91C_BASE_SLCDC     ((AT91PS_SLCDC) 	0xFFFB4000) /* (SLCDC) Base Address*/
#define AT91C_BASE_PDC_ADC   ((AT91PS_PDC) 	0xFFFD8100) /* (PDC_ADC) Base Address*/
#define AT91C_BASE_ADC       ((AT91PS_ADC) 	0xFFFD8000) /* (ADC) Base Address*/

/* ******************************************************************************/
/*               MEMORY MAPPING DEFINITIONS FOR AT91SAM7L128*/
/* ******************************************************************************/
/* ISRAM_1*/
#define AT91C_ISRAM_1	 ((char *) 	0x00200000) /* Internal SRAM base address*/
#define AT91C_ISRAM_1_SIZE	 ((unsigned int) 0x00001000) /* Internal SRAM size in byte (4 Kbytes)*/
/* ISRAM_2*/
#define AT91C_ISRAM_2	 ((char *) 	0x00300000) /* Internal SRAM base address*/
#define AT91C_ISRAM_2_SIZE	 ((unsigned int) 0x00000800) /* Internal SRAM size in byte (2 Kbytes)*/
/* IFLASH*/
#define AT91C_IFLASH	 ((char *) 	0x00100000) /* Internal FLASH base address*/
#define AT91C_IFLASH_SIZE	 ((unsigned int) 0x00020000) /* Internal FLASH size in byte (128 Kbytes)*/
#define AT91C_IFLASH_PAGE_SIZE	 ((unsigned int) 256) /* Internal FLASH Page Size: 256 bytes*/
#define AT91C_IFLASH_LOCK_REGION_SIZE	 ((unsigned int) 8192) /* Internal FLASH Lock Region Size: 8 Kbytes*/
#define AT91C_IFLASH_NB_OF_PAGES	 ((unsigned int) 512) /* Internal FLASH Number of Pages: 512 bytes*/
#define AT91C_IFLASH_NB_OF_LOCK_BITS	 ((unsigned int) 16) /* Internal FLASH Number of Lock Bits: 16 bytes*/
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

/* - Hardware register definition*/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR System Peripherals*/
/* - ******************************************************************************/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Advanced Interrupt Controller*/
/* - ******************************************************************************/
/* - -------- AIC_SMR : (AIC Offset: 0x0) Control Register -------- */
AT91C_AIC_PRIOR           EQU (0x7 <<  0) ;- (AIC) Priority Level
AT91C_AIC_PRIOR_LOWEST    EQU (0x0) ;- (AIC) Lowest priority level
AT91C_AIC_PRIOR_HIGHEST   EQU (0x7) ;- (AIC) Highest priority level
AT91C_AIC_SRCTYPE         EQU (0x3 <<  5) ;- (AIC) Interrupt Source Type
AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL EQU (0x0 <<  5) ;- (AIC) Internal Sources Code Label High-level Sensitive
AT91C_AIC_SRCTYPE_EXT_LOW_LEVEL EQU (0x0 <<  5) ;- (AIC) External Sources Code Label Low-level Sensitive
AT91C_AIC_SRCTYPE_INT_POSITIVE_EDGE EQU (0x1 <<  5) ;- (AIC) Internal Sources Code Label Positive Edge triggered
AT91C_AIC_SRCTYPE_EXT_NEGATIVE_EDGE EQU (0x1 <<  5) ;- (AIC) External Sources Code Label Negative Edge triggered
AT91C_AIC_SRCTYPE_HIGH_LEVEL EQU (0x2 <<  5) ;- (AIC) Internal Or External Sources Code Label High-level Sensitive
AT91C_AIC_SRCTYPE_POSITIVE_EDGE EQU (0x3 <<  5) ;- (AIC) Internal Or External Sources Code Label Positive Edge triggered
/* - -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- */
AT91C_AIC_NFIQ            EQU (0x1 <<  0) ;- (AIC) NFIQ Status
AT91C_AIC_NIRQ            EQU (0x1 <<  1) ;- (AIC) NIRQ Status
/* - -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- */
AT91C_AIC_DCR_PROT        EQU (0x1 <<  0) ;- (AIC) Protection Mode
AT91C_AIC_DCR_GMSK        EQU (0x1 <<  1) ;- (AIC) General Mask

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Peripheral DMA Controller*/
/* - ******************************************************************************/
/* - -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- */
AT91C_PDC_RXTEN           EQU (0x1 <<  0) ;- (PDC) Receiver Transfer Enable
AT91C_PDC_RXTDIS          EQU (0x1 <<  1) ;- (PDC) Receiver Transfer Disable
AT91C_PDC_TXTEN           EQU (0x1 <<  8) ;- (PDC) Transmitter Transfer Enable
AT91C_PDC_TXTDIS          EQU (0x1 <<  9) ;- (PDC) Transmitter Transfer Disable
/* - -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Debug Unit*/
/* - ******************************************************************************/
/* - -------- DBGU_CR : (DBGU Offset: 0x0) Debug Unit Control Register -------- */
AT91C_US_RSTRX            EQU (0x1 <<  2) ;- (DBGU) Reset Receiver
AT91C_US_RSTTX            EQU (0x1 <<  3) ;- (DBGU) Reset Transmitter
AT91C_US_RXEN             EQU (0x1 <<  4) ;- (DBGU) Receiver Enable
AT91C_US_RXDIS            EQU (0x1 <<  5) ;- (DBGU) Receiver Disable
AT91C_US_TXEN             EQU (0x1 <<  6) ;- (DBGU) Transmitter Enable
AT91C_US_TXDIS            EQU (0x1 <<  7) ;- (DBGU) Transmitter Disable
AT91C_US_RSTSTA           EQU (0x1 <<  8) ;- (DBGU) Reset Status Bits
/* - -------- DBGU_MR : (DBGU Offset: 0x4) Debug Unit Mode Register -------- */
AT91C_US_PAR              EQU (0x7 <<  9) ;- (DBGU) Parity type
AT91C_US_PAR_EVEN         EQU (0x0 <<  9) ;- (DBGU) Even Parity
AT91C_US_PAR_ODD          EQU (0x1 <<  9) ;- (DBGU) Odd Parity
AT91C_US_PAR_SPACE        EQU (0x2 <<  9) ;- (DBGU) Parity forced to 0 (Space)
AT91C_US_PAR_MARK         EQU (0x3 <<  9) ;- (DBGU) Parity forced to 1 (Mark)
AT91C_US_PAR_NONE         EQU (0x4 <<  9) ;- (DBGU) No Parity
AT91C_US_PAR_MULTI_DROP   EQU (0x6 <<  9) ;- (DBGU) Multi-drop mode
AT91C_US_CHMODE           EQU (0x3 << 14) ;- (DBGU) Channel Mode
AT91C_US_CHMODE_NORMAL    EQU (0x0 << 14) ;- (DBGU) Normal Mode: The USART channel operates as an RX/TX USART.
AT91C_US_CHMODE_AUTO      EQU (0x1 << 14) ;- (DBGU) Automatic Echo: Receiver Data Input is connected to the TXD pin.
AT91C_US_CHMODE_LOCAL     EQU (0x2 << 14) ;- (DBGU) Local Loopback: Transmitter Output Signal is connected to Receiver Input Signal.
AT91C_US_CHMODE_REMOTE    EQU (0x3 << 14) ;- (DBGU) Remote Loopback: RXD pin is internally connected to TXD pin.
/* - -------- DBGU_IER : (DBGU Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
AT91C_US_RXRDY            EQU (0x1 <<  0) ;- (DBGU) RXRDY Interrupt
AT91C_US_TXRDY            EQU (0x1 <<  1) ;- (DBGU) TXRDY Interrupt
AT91C_US_ENDRX            EQU (0x1 <<  3) ;- (DBGU) End of Receive Transfer Interrupt
AT91C_US_ENDTX            EQU (0x1 <<  4) ;- (DBGU) End of Transmit Interrupt
AT91C_US_OVRE             EQU (0x1 <<  5) ;- (DBGU) Overrun Interrupt
AT91C_US_FRAME            EQU (0x1 <<  6) ;- (DBGU) Framing Error Interrupt
AT91C_US_PARE             EQU (0x1 <<  7) ;- (DBGU) Parity Error Interrupt
AT91C_US_TXEMPTY          EQU (0x1 <<  9) ;- (DBGU) TXEMPTY Interrupt
AT91C_US_TXBUFE           EQU (0x1 << 11) ;- (DBGU) TXBUFE Interrupt
AT91C_US_RXBUFF           EQU (0x1 << 12) ;- (DBGU) RXBUFF Interrupt
AT91C_US_COMM_TX          EQU (0x1 << 30) ;- (DBGU) COMM_TX Interrupt
AT91C_US_COMM_RX          EQU (0x1 << 31) ;- (DBGU) COMM_RX Interrupt
/* - -------- DBGU_IDR : (DBGU Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* - -------- DBGU_IMR : (DBGU Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* - -------- DBGU_CSR : (DBGU Offset: 0x14) Debug Unit Channel Status Register -------- */
/* - -------- DBGU_FNTR : (DBGU Offset: 0x48) Debug Unit FORCE_NTRST Register -------- */
AT91C_US_FORCE_NTRST      EQU (0x1 <<  0) ;- (DBGU) Force NTRST in JTAG

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler*/
/* - ******************************************************************************/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* - ******************************************************************************/
/* - -------- CKGR_MOR : (CKGR Offset: 0x0) Main Oscillator Register -------- */
AT91C_CKGR_MAINCKON       EQU (0x1 <<  0) ;- (CKGR) RC 2 MHz Oscillator Enable (RC 2 MHz oscillator enabled at startup)
AT91C_CKGR_FKEY           EQU (0xFF << 16) ;- (CKGR) Clock Generator Controller Writing Protection Key
AT91C_CKGR_MCKSEL         EQU (0x1 << 24) ;- (CKGR) 
/* - -------- CKGR_MCFR : (CKGR Offset: 0x4) Main Clock Frequency Register -------- */
AT91C_CKGR_MAINF          EQU (0xFFFF <<  0) ;- (CKGR) Main Clock Frequency
AT91C_CKGR_MAINRDY        EQU (0x1 << 16) ;- (CKGR) Main Clock Ready
/* - -------- CKGR_PLLR : (CKGR Offset: 0x8) PLL A Register -------- */
AT91C_CKGR_DIV            EQU (0xFF <<  0) ;- (CKGR) Divider Selected
AT91C_CKGR_DIV_0          EQU (0x0) ;- (CKGR) Divider output is 0
AT91C_CKGR_DIV_BYPASS     EQU (0x1) ;- (CKGR) Divider is bypassed
AT91C_CKGR_PLLCOUNT       EQU (0x3F <<  8) ;- (CKGR) PLL Counter
AT91C_CKGR_OUT            EQU (0x3 << 14) ;- (CKGR) PLL Output Frequency Range
AT91C_CKGR_OUT_0          EQU (0x0 << 14) ;- (CKGR) Please refer to the PLL datasheet
AT91C_CKGR_OUT_1          EQU (0x1 << 14) ;- (CKGR) Please refer to the PLL datasheet
AT91C_CKGR_OUT_2          EQU (0x2 << 14) ;- (CKGR) Please refer to the PLL datasheet
AT91C_CKGR_OUT_3          EQU (0x3 << 14) ;- (CKGR) Please refer to the PLL datasheet
AT91C_CKGR_MUL            EQU (0x7FF << 16) ;- (CKGR) PLL Multiplier

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Power Management Controler*/
/* - ******************************************************************************/
/* - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
AT91C_PMC_PCK0            EQU (0x1 <<  8) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK1            EQU (0x1 <<  9) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK2            EQU (0x1 << 10) ;- (PMC) Programmable Clock Output
/* - -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* - -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* - -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- */
/* - -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- */
/* - -------- CKGR_PLLR : (PMC Offset: 0x28) PLL A Register -------- */
/* - -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- */
AT91C_PMC_CSS             EQU (0x3 <<  0) ;- (PMC) Programmable Clock Selection
AT91C_PMC_CSS_SLOW_CLK    EQU (0x0) ;- (PMC) Slow Clock is selected
AT91C_PMC_CSS_MAIN_CLK    EQU (0x1) ;- (PMC) Main Clock is selected
AT91C_PMC_CSS_PLL_CLK     EQU (0x2) ;- (PMC) Clock from PLL is selected
AT91C_PMC_PRES            EQU (0x7 <<  2) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_PRES_CLK        EQU (0x0 <<  2) ;- (PMC) Selected clock
AT91C_PMC_PRES_CLK_2      EQU (0x1 <<  2) ;- (PMC) Selected clock divided by 2
AT91C_PMC_PRES_CLK_4      EQU (0x2 <<  2) ;- (PMC) Selected clock divided by 4
AT91C_PMC_PRES_CLK_8      EQU (0x3 <<  2) ;- (PMC) Selected clock divided by 8
AT91C_PMC_PRES_CLK_16     EQU (0x4 <<  2) ;- (PMC) Selected clock divided by 16
AT91C_PMC_PRES_CLK_32     EQU (0x5 <<  2) ;- (PMC) Selected clock divided by 32
AT91C_PMC_PRES_CLK_64     EQU (0x6 <<  2) ;- (PMC) Selected clock divided by 64
/* - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
AT91C_PMC_MAINSELS        EQU (0x1 <<  0) ;- (PMC) Main Clock Selection Status/Enable/Disable/Mask
AT91C_PMC_LOCK            EQU (0x1 <<  1) ;- (PMC) PLL Status/Enable/Disable/Mask
AT91C_PMC_MCKRDY          EQU (0x1 <<  3) ;- (PMC) MCK_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK0RDY         EQU (0x1 <<  8) ;- (PMC) PCK0_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK1RDY         EQU (0x1 <<  9) ;- (PMC) PCK1_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK2RDY         EQU (0x1 << 10) ;- (PMC) PCK2_RDY Status/Enable/Disable/Mask
/* - -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* - -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* - -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Reset Controller Interface*/
/* - ******************************************************************************/
/* - -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- */
AT91C_RSTC_PROCRST        EQU (0x1 <<  0) ;- (RSTC) Processor Reset
AT91C_RSTC_PERRST         EQU (0x1 <<  2) ;- (RSTC) Peripheral Reset
AT91C_RSTC_EXTRST         EQU (0x1 <<  3) ;- (RSTC) External Reset
AT91C_RSTC_KEY            EQU (0xFF << 24) ;- (RSTC) Password
/* - -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- */
AT91C_RSTC_URSTS          EQU (0x1 <<  0) ;- (RSTC) User Reset Status
AT91C_RSTC_BODSTS         EQU (0x1 <<  1) ;- (RSTC) Brownout Detection Status
AT91C_RSTC_RSTTYP         EQU (0x7 <<  8) ;- (RSTC) Reset Type
AT91C_RSTC_RSTTYP_POWERUP EQU (0x0 <<  8) ;- (RSTC) Power-up Reset. VDDCORE rising.
AT91C_RSTC_RSTTYP_WAKEUP  EQU (0x1 <<  8) ;- (RSTC) WakeUp Reset. VDDCORE rising.
AT91C_RSTC_RSTTYP_WATCHDOG EQU (0x2 <<  8) ;- (RSTC) Watchdog Reset. Watchdog overflow occured.
AT91C_RSTC_RSTTYP_SOFTWARE EQU (0x3 <<  8) ;- (RSTC) Software Reset. Processor reset required by the software.
AT91C_RSTC_RSTTYP_USER    EQU (0x4 <<  8) ;- (RSTC) User Reset. NRST pin detected low.
AT91C_RSTC_RSTTYP_BROWNOUT EQU (0x5 <<  8) ;- (RSTC) Brownout Reset occured.
AT91C_RSTC_NRSTL          EQU (0x1 << 16) ;- (RSTC) NRST pin level
AT91C_RSTC_SRCMP          EQU (0x1 << 17) ;- (RSTC) Software Reset Command in Progress.
/* - -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- */
AT91C_RSTC_URSTEN         EQU (0x1 <<  0) ;- (RSTC) User Reset Enable
AT91C_RSTC_URSTIEN        EQU (0x1 <<  4) ;- (RSTC) User Reset Interrupt Enable
AT91C_RSTC_ERSTL          EQU (0xF <<  8) ;- (RSTC) User Reset Length
AT91C_RSTC_BODIEN         EQU (0x1 << 16) ;- (RSTC) Brownout Detection Interrupt Enable

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm*/
/* - ******************************************************************************/
/* - -------- RTC_MR : (RTC Offset: 0x0) RTC Mode Register -------- */
AT91C_RTC_UPDTIM          EQU (0x1 <<  0) ;- (RTC) Update Request Time Register
AT91C_RTC_UPDCAL          EQU (0x1 <<  1) ;- (RTC) Update Request Calendar Register
AT91C_RTC_TEVSEL          EQU (0x3 <<  8) ;- (RTC) Time Event Selection
AT91C_RTC_TEVSEL_MN_CHG   EQU (0x0 <<  8) ;- (RTC) Minute change.
AT91C_RTC_TEVSEL_HR_CHG   EQU (0x1 <<  8) ;- (RTC) Hour change.
AT91C_RTC_TEVSEL_EVDAY_MD EQU (0x2 <<  8) ;- (RTC) Every day at midnight.
AT91C_RTC_TEVSEL_EVDAY_NOON EQU (0x3 <<  8) ;- (RTC) Every day at noon.
AT91C_RTC_CEVSEL          EQU (0x3 << 16) ;- (RTC) Calendar Event Selection
AT91C_RTC_CEVSEL_WEEK_CHG EQU (0x0 << 16) ;- (RTC) Week change (every Monday at time 00:00:00).
AT91C_RTC_CEVSEL_MONTH_CHG EQU (0x1 << 16) ;- (RTC) Month change (every 01 of each month at time 00:00:00).
AT91C_RTC_CEVSEL_YEAR_CHG EQU (0x2 << 16) ;- (RTC) Year change (every January 1 at time 00:00:00).
/* - -------- RTC_HMR : (RTC Offset: 0x4) RTC Hour Mode Register -------- */
AT91C_RTC_HRMOD           EQU (0x1 <<  0) ;- (RTC) 12-24 hour Mode
/* - -------- RTC_TIMR : (RTC Offset: 0x8) RTC Time Register -------- */
AT91C_RTC_SEC             EQU (0x7F <<  0) ;- (RTC) Current Second
AT91C_RTC_MIN             EQU (0x7F <<  8) ;- (RTC) Current Minute
AT91C_RTC_HOUR            EQU (0x3F << 16) ;- (RTC) Current Hour
AT91C_RTC_AMPM            EQU (0x1 << 22) ;- (RTC) Ante Meridiem, Post Meridiem Indicator
/* - -------- RTC_CALR : (RTC Offset: 0xc) RTC Calendar Register -------- */
AT91C_RTC_CENT            EQU (0x3F <<  0) ;- (RTC) Current Century
AT91C_RTC_YEAR            EQU (0xFF <<  8) ;- (RTC) Current Year
AT91C_RTC_MONTH           EQU (0x1F << 16) ;- (RTC) Current Month
AT91C_RTC_DAY             EQU (0x7 << 21) ;- (RTC) Current Day
AT91C_RTC_DATE            EQU (0x3F << 24) ;- (RTC) Current Date
/* - -------- RTC_TAR : (RTC Offset: 0x10) RTC Time Alarm Register -------- */
AT91C_RTC_SECEN           EQU (0x1 <<  7) ;- (RTC) Second Alarm Enable
AT91C_RTC_MINEN           EQU (0x1 << 15) ;- (RTC) Minute Alarm
AT91C_RTC_HOUREN          EQU (0x1 << 23) ;- (RTC) Current Hour
/* - -------- RTC_CAR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- */
AT91C_RTC_MTHEN           EQU (0x1 << 23) ;- (RTC) Month Alarm Enable
AT91C_RTC_DATEN           EQU (0x1 << 31) ;- (RTC) Date Alarm Enable
/* - -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- */
AT91C_RTC_ACKUPD          EQU (0x1 <<  0) ;- (RTC) Acknowledge for Update
AT91C_RTC_ALARM           EQU (0x1 <<  1) ;- (RTC) Alarm Flag
AT91C_RTC_SECEV           EQU (0x1 <<  2) ;- (RTC) Second Event
AT91C_RTC_TIMEV           EQU (0x1 <<  3) ;- (RTC) Time Event
AT91C_RTC_CALEV           EQU (0x1 <<  4) ;- (RTC) Calendar event
/* - -------- RTC_SCR : (RTC Offset: 0x1c) RTC Status Clear Register -------- */
/* - -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- */
/* - -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- */
/* - -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- */
/* - -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- */
AT91C_RTC_NVT             EQU (0x1 <<  0) ;- (RTC) Non valid Time
AT91C_RTC_NVC             EQU (0x1 <<  1) ;- (RTC) Non valid Calendar
AT91C_RTC_NVTAL           EQU (0x1 <<  2) ;- (RTC) Non valid time Alarm
AT91C_RTC_NVCAL           EQU (0x1 <<  3) ;- (RTC) Nonvalid Calendar Alarm

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Periodic Interval Timer Controller Interface*/
/* - ******************************************************************************/
/* - -------- PITC_PIMR : (PITC Offset: 0x0) Periodic Interval Mode Register -------- */
AT91C_PITC_PIV            EQU (0xFFFFF <<  0) ;- (PITC) Periodic Interval Value
AT91C_PITC_PITEN          EQU (0x1 << 24) ;- (PITC) Periodic Interval Timer Enabled
AT91C_PITC_PITIEN         EQU (0x1 << 25) ;- (PITC) Periodic Interval Timer Interrupt Enable
/* - -------- PITC_PISR : (PITC Offset: 0x4) Periodic Interval Status Register -------- */
AT91C_PITC_PITS           EQU (0x1 <<  0) ;- (PITC) Periodic Interval Timer Status
/* - -------- PITC_PIVR : (PITC Offset: 0x8) Periodic Interval Value Register -------- */
AT91C_PITC_CPIV           EQU (0xFFFFF <<  0) ;- (PITC) Current Periodic Interval Value
AT91C_PITC_PICNT          EQU (0xFFF << 20) ;- (PITC) Periodic Interval Counter
/* - -------- PITC_PIIR : (PITC Offset: 0xc) Periodic Interval Image Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Watchdog Timer Controller Interface*/
/* - ******************************************************************************/
/* - -------- WDTC_WDCR : (WDTC Offset: 0x0) Periodic Interval Image Register -------- */
AT91C_WDTC_WDRSTT         EQU (0x1 <<  0) ;- (WDTC) Watchdog Restart
AT91C_WDTC_KEY            EQU (0xFF << 24) ;- (WDTC) Watchdog KEY Password
/* - -------- WDTC_WDMR : (WDTC Offset: 0x4) Watchdog Mode Register -------- */
AT91C_WDTC_WDV            EQU (0xFFF <<  0) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDFIEN         EQU (0x1 << 12) ;- (WDTC) Watchdog Fault Interrupt Enable
AT91C_WDTC_WDRSTEN        EQU (0x1 << 13) ;- (WDTC) Watchdog Reset Enable
AT91C_WDTC_WDRPROC        EQU (0x1 << 14) ;- (WDTC) Watchdog Timer Restart
AT91C_WDTC_WDDIS          EQU (0x1 << 15) ;- (WDTC) Watchdog Disable
AT91C_WDTC_WDD            EQU (0xFFF << 16) ;- (WDTC) Watchdog Delta Value
AT91C_WDTC_WDDBGHLT       EQU (0x1 << 28) ;- (WDTC) Watchdog Debug Halt
AT91C_WDTC_WDIDLEHLT      EQU (0x1 << 29) ;- (WDTC) Watchdog Idle Halt
/* - -------- WDTC_WDSR : (WDTC Offset: 0x8) Watchdog Status Register -------- */
AT91C_WDTC_WDUNF          EQU (0x1 <<  0) ;- (WDTC) Watchdog Underflow
AT91C_WDTC_WDERR          EQU (0x1 <<  1) ;- (WDTC) Watchdog Error

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Voltage Regulator Mode Controller Interface*/
/* - ******************************************************************************/
/* - -------- VREG_MR : (VREG Offset: 0x0) Voltage Regulator Mode Register -------- */
AT91C_VREG_PSTDBY         EQU (0x1 <<  0) ;- (VREG) Voltage Regulator Power Standby Mode

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Memory Controller Interface*/
/* - ******************************************************************************/
/* - -------- MC_RCR : (MC Offset: 0x0) MC Remap Control Register -------- */
AT91C_MC_RCB              EQU (0x1 <<  0) ;- (MC) Remap Command Bit
/* - -------- MC_ASR : (MC Offset: 0x4) MC Abort Status Register -------- */
AT91C_MC_UNDADD           EQU (0x1 <<  0) ;- (MC) Undefined Addess Abort Status
AT91C_MC_MISADD           EQU (0x1 <<  1) ;- (MC) Misaligned Addess Abort Status
AT91C_MC_ABTSZ            EQU (0x3 <<  8) ;- (MC) Abort Size Status
AT91C_MC_ABTSZ_BYTE       EQU (0x0 <<  8) ;- (MC) Byte
AT91C_MC_ABTSZ_HWORD      EQU (0x1 <<  8) ;- (MC) Half-word
AT91C_MC_ABTSZ_WORD       EQU (0x2 <<  8) ;- (MC) Word
AT91C_MC_ABTTYP           EQU (0x3 << 10) ;- (MC) Abort Type Status
AT91C_MC_ABTTYP_DATAR     EQU (0x0 << 10) ;- (MC) Data Read
AT91C_MC_ABTTYP_DATAW     EQU (0x1 << 10) ;- (MC) Data Write
AT91C_MC_ABTTYP_FETCH     EQU (0x2 << 10) ;- (MC) Code Fetch
AT91C_MC_MST0             EQU (0x1 << 16) ;- (MC) Master 0 Abort Source
AT91C_MC_MST1             EQU (0x1 << 17) ;- (MC) Master 1 Abort Source
AT91C_MC_SVMST0           EQU (0x1 << 24) ;- (MC) Saved Master 0 Abort Source
AT91C_MC_SVMST1           EQU (0x1 << 25) ;- (MC) Saved Master 1 Abort Source
/* - -------- MC_FMR : (MC Offset: 0x60) MC Flash Mode Register -------- */
AT91C_MC_FRDY             EQU (0x1 <<  0) ;- (MC) Ready Interrupt Enable
AT91C_MC_FWS              EQU (0xF <<  8) ;- (MC) Flash Wait State
AT91C_MC_FWS_0FWS         EQU (0x0 <<  8) ;- (MC) 0 Wait State
AT91C_MC_FWS_1FWS         EQU (0x1 <<  8) ;- (MC) 1 Wait State
AT91C_MC_FWS_2FWS         EQU (0x2 <<  8) ;- (MC) 2 Wait States
AT91C_MC_FWS_3FWS         EQU (0x3 <<  8) ;- (MC) 3 Wait States
/* - -------- MC_FCR : (MC Offset: 0x64) MC Flash Command Register -------- */
AT91C_MC_FCMD             EQU (0xFF <<  0) ;- (MC) Flash Command
AT91C_MC_FCMD_GETD        EQU (0x0) ;- (MC) Get Flash Descriptor
AT91C_MC_FCMD_WP          EQU (0x1) ;- (MC) Write Page
AT91C_MC_FCMD_WPL         EQU (0x2) ;- (MC) Write Page and Lock
AT91C_MC_FCMD_EWP         EQU (0x3) ;- (MC) Erase Page and Write Page
AT91C_MC_FCMD_EWPL        EQU (0x4) ;- (MC) Erase Page and Write Page then Lock
AT91C_MC_FCMD_EA          EQU (0x5) ;- (MC) Erase All
AT91C_MC_FCMD_EPL         EQU (0x6) ;- (MC) Erase Plane
AT91C_MC_FCMD_EPA         EQU (0x7) ;- (MC) Erase Pages
AT91C_MC_FCMD_SLB         EQU (0x8) ;- (MC) Set Lock Bit
AT91C_MC_FCMD_CLB         EQU (0x9) ;- (MC) Clear Lock Bit
AT91C_MC_FCMD_GLB         EQU (0xA) ;- (MC) Get Lock Bit
AT91C_MC_FCMD_SFB         EQU (0xB) ;- (MC) Set Fuse Bit
AT91C_MC_FCMD_CFB         EQU (0xC) ;- (MC) Clear Fuse Bit
AT91C_MC_FCMD_GFB         EQU (0xD) ;- (MC) Get Fuse Bit
AT91C_MC_FARG             EQU (0xFFFF <<  8) ;- (MC) Flash Command Argument
AT91C_MC_KEY              EQU (0xFF << 24) ;- (MC) Writing Protect Key
/* - -------- MC_FSR : (MC Offset: 0x68) MC Flash Command Register -------- */
AT91C_MC_FRDY_S           EQU (0x1 <<  0) ;- (MC) Flash Ready Status
AT91C_MC_FCMDE            EQU (0x1 <<  1) ;- (MC) Flash Command Error Status
AT91C_MC_LOCKE            EQU (0x1 <<  2) ;- (MC) Flash Lock Error Status

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Serial Parallel Interface*/
/* - ******************************************************************************/
/* - -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- */
AT91C_SPI_SPIEN           EQU (0x1 <<  0) ;- (SPI) SPI Enable
AT91C_SPI_SPIDIS          EQU (0x1 <<  1) ;- (SPI) SPI Disable
AT91C_SPI_SWRST           EQU (0x1 <<  7) ;- (SPI) SPI Software reset
AT91C_SPI_LASTXFER        EQU (0x1 << 24) ;- (SPI) SPI Last Transfer
/* - -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- */
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
/* - -------- SPI_RDR : (SPI Offset: 0x8) Receive Data Register -------- */
AT91C_SPI_RD              EQU (0xFFFF <<  0) ;- (SPI) Receive Data
AT91C_SPI_RPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
/* - -------- SPI_TDR : (SPI Offset: 0xc) Transmit Data Register -------- */
AT91C_SPI_TD              EQU (0xFFFF <<  0) ;- (SPI) Transmit Data
AT91C_SPI_TPCS            EQU (0xF << 16) ;- (SPI) Peripheral Chip Select Status
/* - -------- SPI_SR : (SPI Offset: 0x10) Status Register -------- */
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
/* - -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- */
/* - -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- */
/* - -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- */
/* - -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- */
AT91C_SPI_CPOL            EQU (0x1 <<  0) ;- (SPI) Clock Polarity
AT91C_SPI_NCPHA           EQU (0x1 <<  1) ;- (SPI) Clock Phase
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

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Usart*/
/* - ******************************************************************************/
/* - -------- US_CR : (USART Offset: 0x0) Debug Unit Control Register -------- */
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
/* - -------- US_MR : (USART Offset: 0x4) Debug Unit Mode Register -------- */
AT91C_US_USMODE           EQU (0xF <<  0) ;- (USART) Usart mode
AT91C_US_USMODE_NORMAL    EQU (0x0) ;- (USART) Normal
AT91C_US_USMODE_RS485     EQU (0x1) ;- (USART) RS485
AT91C_US_USMODE_HWHSH     EQU (0x2) ;- (USART) Hardware Handshaking
AT91C_US_USMODE_MODEM     EQU (0x3) ;- (USART) Modem
AT91C_US_USMODE_ISO7816_0 EQU (0x4) ;- (USART) ISO7816 protocol: T = 0
AT91C_US_USMODE_ISO7816_1 EQU (0x6) ;- (USART) ISO7816 protocol: T = 1
AT91C_US_USMODE_IRDA      EQU (0x8) ;- (USART) IrDA
AT91C_US_USMODE_SWHSH     EQU (0xC) ;- (USART) Software Handshaking
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
AT91C_US_NBSTOP           EQU (0x3 << 12) ;- (USART) Number of Stop bits
AT91C_US_NBSTOP_1_BIT     EQU (0x0 << 12) ;- (USART) 1 stop bit
AT91C_US_NBSTOP_15_BIT    EQU (0x1 << 12) ;- (USART) Asynchronous (SYNC=0) 2 stop bits Synchronous (SYNC=1) 2 stop bits
AT91C_US_NBSTOP_2_BIT     EQU (0x2 << 12) ;- (USART) 2 stop bits
AT91C_US_MSBF             EQU (0x1 << 16) ;- (USART) Bit Order
AT91C_US_MODE9            EQU (0x1 << 17) ;- (USART) 9-bit Character length
AT91C_US_CKLO             EQU (0x1 << 18) ;- (USART) Clock Output Select
AT91C_US_OVER             EQU (0x1 << 19) ;- (USART) Over Sampling Mode
AT91C_US_INACK            EQU (0x1 << 20) ;- (USART) Inhibit Non Acknowledge
AT91C_US_DSNACK           EQU (0x1 << 21) ;- (USART) Disable Successive NACK
AT91C_US_MAX_ITER         EQU (0x1 << 24) ;- (USART) Number of Repetitions
AT91C_US_FILTER           EQU (0x1 << 28) ;- (USART) Receive Line Filter
/* - -------- US_IER : (USART Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
AT91C_US_RXBRK            EQU (0x1 <<  2) ;- (USART) Break Received/End of Break
AT91C_US_TIMEOUT          EQU (0x1 <<  8) ;- (USART) Receiver Time-out
AT91C_US_ITERATION        EQU (0x1 << 10) ;- (USART) Max number of Repetitions Reached
AT91C_US_NACK             EQU (0x1 << 13) ;- (USART) Non Acknowledge
AT91C_US_RIIC             EQU (0x1 << 16) ;- (USART) Ring INdicator Input Change Flag
AT91C_US_DSRIC            EQU (0x1 << 17) ;- (USART) Data Set Ready Input Change Flag
AT91C_US_DCDIC            EQU (0x1 << 18) ;- (USART) Data Carrier Flag
AT91C_US_CTSIC            EQU (0x1 << 19) ;- (USART) Clear To Send Input Change Flag
/* - -------- US_IDR : (USART Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* - -------- US_IMR : (USART Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* - -------- US_CSR : (USART Offset: 0x14) Debug Unit Channel Status Register -------- */
AT91C_US_RI               EQU (0x1 << 20) ;- (USART) Image of RI Input
AT91C_US_DSR              EQU (0x1 << 21) ;- (USART) Image of DSR Input
AT91C_US_DCD              EQU (0x1 << 22) ;- (USART) Image of DCD Input
AT91C_US_CTS              EQU (0x1 << 23) ;- (USART) Image of CTS Input

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Two-wire Interface*/
/* - ******************************************************************************/
/* - -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- */
AT91C_TWI_START           EQU (0x1 <<  0) ;- (TWI) Send a START Condition
AT91C_TWI_STOP            EQU (0x1 <<  1) ;- (TWI) Send a STOP Condition
AT91C_TWI_MSEN            EQU (0x1 <<  2) ;- (TWI) TWI Master Transfer Enabled
AT91C_TWI_MSDIS           EQU (0x1 <<  3) ;- (TWI) TWI Master Transfer Disabled
AT91C_TWI_SVEN            EQU (0x1 <<  4) ;- (TWI) TWI Slave mode Enabled
AT91C_TWI_SVDIS           EQU (0x1 <<  5) ;- (TWI) TWI Slave mode Disabled
AT91C_TWI_SWRST           EQU (0x1 <<  7) ;- (TWI) Software Reset
/* - -------- TWI_MMR : (TWI Offset: 0x4) TWI Master Mode Register -------- */
AT91C_TWI_IADRSZ          EQU (0x3 <<  8) ;- (TWI) Internal Device Address Size
AT91C_TWI_IADRSZ_NO       EQU (0x0 <<  8) ;- (TWI) No internal device address
AT91C_TWI_IADRSZ_1_BYTE   EQU (0x1 <<  8) ;- (TWI) One-byte internal device address
AT91C_TWI_IADRSZ_2_BYTE   EQU (0x2 <<  8) ;- (TWI) Two-byte internal device address
AT91C_TWI_IADRSZ_3_BYTE   EQU (0x3 <<  8) ;- (TWI) Three-byte internal device address
AT91C_TWI_MREAD           EQU (0x1 << 12) ;- (TWI) Master Read Direction
AT91C_TWI_DADR            EQU (0x7F << 16) ;- (TWI) Device Address
/* - -------- TWI_SMR : (TWI Offset: 0x8) TWI Slave Mode Register -------- */
AT91C_TWI_SADR            EQU (0x7F << 16) ;- (TWI) Slave Address
/* - -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- */
AT91C_TWI_CLDIV           EQU (0xFF <<  0) ;- (TWI) Clock Low Divider
AT91C_TWI_CHDIV           EQU (0xFF <<  8) ;- (TWI) Clock High Divider
AT91C_TWI_CKDIV           EQU (0x7 << 16) ;- (TWI) Clock Divider
/* - -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- */
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
/* - -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- */
/* - -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- */
/* - -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR PWMC Channel Interface*/
/* - ******************************************************************************/
/* - -------- PWMC_CMR : (PWMC_CH Offset: 0x0) PWMC Channel Mode Register -------- */
AT91C_PWMC_CPRE           EQU (0xF <<  0) ;- (PWMC_CH) Channel Pre-scaler : PWMC_CLKx
AT91C_PWMC_CPRE_MCK       EQU (0x0) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKA      EQU (0xB) ;- (PWMC_CH) 
AT91C_PWMC_CPRE_MCKB      EQU (0xC) ;- (PWMC_CH) 
AT91C_PWMC_CALG           EQU (0x1 <<  8) ;- (PWMC_CH) Channel Alignment
AT91C_PWMC_CPOL           EQU (0x1 <<  9) ;- (PWMC_CH) Channel Polarity
AT91C_PWMC_CPD            EQU (0x1 << 10) ;- (PWMC_CH) Channel Update Period
/* - -------- PWMC_CDTYR : (PWMC_CH Offset: 0x4) PWMC Channel Duty Cycle Register -------- */
AT91C_PWMC_CDTY           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Duty Cycle
/* - -------- PWMC_CPRDR : (PWMC_CH Offset: 0x8) PWMC Channel Period Register -------- */
AT91C_PWMC_CPRD           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Period
/* - -------- PWMC_CCNTR : (PWMC_CH Offset: 0xc) PWMC Channel Counter Register -------- */
AT91C_PWMC_CCNT           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Counter
/* - -------- PWMC_CUPDR : (PWMC_CH Offset: 0x10) PWMC Channel Update Register -------- */
AT91C_PWMC_CUPD           EQU (0x0 <<  0) ;- (PWMC_CH) Channel Update

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Pulse Width Modulation Controller Interface*/
/* - ******************************************************************************/
/* - -------- PWMC_MR : (PWMC Offset: 0x0) PWMC Mode Register -------- */
AT91C_PWMC_DIVA           EQU (0xFF <<  0) ;- (PWMC) CLKA divide factor.
AT91C_PWMC_PREA           EQU (0xF <<  8) ;- (PWMC) Divider Input Clock Prescaler A
AT91C_PWMC_PREA_MCK       EQU (0x0 <<  8) ;- (PWMC) 
AT91C_PWMC_DIVB           EQU (0xFF << 16) ;- (PWMC) CLKB divide factor.
AT91C_PWMC_PREB           EQU (0xF << 24) ;- (PWMC) Divider Input Clock Prescaler B
AT91C_PWMC_PREB_MCK       EQU (0x0 << 24) ;- (PWMC) 
/* - -------- PWMC_ENA : (PWMC Offset: 0x4) PWMC Enable Register -------- */
AT91C_PWMC_CHID0          EQU (0x1 <<  0) ;- (PWMC) Channel ID 0
AT91C_PWMC_CHID1          EQU (0x1 <<  1) ;- (PWMC) Channel ID 1
AT91C_PWMC_CHID2          EQU (0x1 <<  2) ;- (PWMC) Channel ID 2
AT91C_PWMC_CHID3          EQU (0x1 <<  3) ;- (PWMC) Channel ID 3
/* - -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- */
/* - -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- */
/* - -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- */
/* - -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- */
/* - -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- */
/* - -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Timer Counter Channel Interface*/
/* - ******************************************************************************/
/* - -------- TC_CCR : (TC Offset: 0x0) TC Channel Control Register -------- */
AT91C_TC_CLKEN            EQU (0x1 <<  0) ;- (TC) Counter Clock Enable Command
AT91C_TC_CLKDIS           EQU (0x1 <<  1) ;- (TC) Counter Clock Disable Command
AT91C_TC_SWTRG            EQU (0x1 <<  2) ;- (TC) Software Trigger Command
/* - -------- TC_CMR : (TC Offset: 0x4) TC Channel Mode Register: Capture Mode / Waveform Mode -------- */
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
/* - -------- TC_SR : (TC Offset: 0x20) TC Channel Status Register -------- */
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
/* - -------- TC_IER : (TC Offset: 0x24) TC Channel Interrupt Enable Register -------- */
/* - -------- TC_IDR : (TC Offset: 0x28) TC Channel Interrupt Disable Register -------- */
/* - -------- TC_IMR : (TC Offset: 0x2c) TC Channel Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Timer Counter Interface*/
/* - ******************************************************************************/
/* - -------- TCB_BCR : (TCB Offset: 0xc0) TC Block Control Register -------- */
AT91C_TCB_SYNC            EQU (0x1 <<  0) ;- (TCB) Synchro Command
/* - -------- TCB_BMR : (TCB Offset: 0xc4) TC Block Mode Register -------- */
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

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Segment LCD Controller*/
/* - ******************************************************************************/
/* - -------- SLCDC_CR : (SLCDC Offset: 0x0) SLCDC Control Register -------- */
AT91C_SLCDC_LCDEN         EQU (0x1 <<  0) ;- (SLCDC) Enable the LCDC
AT91C_SLCDC_LCDDIS        EQU (0x1 <<  1) ;- (SLCDC) Disable the LCDC
AT91C_SLCDC_SWRST         EQU (0x1 <<  3) ;- (SLCDC) Software Reset
/* - -------- SLCDC_MR : (SLCDC Offset: 0x4) SLCDC Control Register -------- */
AT91C_SLCDC_COMSEL        EQU (0xF <<  0) ;- (SLCDC) Selection of the number of common
AT91C_SLCDC_COMSEL_0      EQU (0x0) ;- (SLCDC) COM0 selected
AT91C_SLCDC_COMSEL_1      EQU (0x1) ;- (SLCDC) COM1 selected
AT91C_SLCDC_COMSEL_2      EQU (0x2) ;- (SLCDC) COM2 selected
AT91C_SLCDC_COMSEL_3      EQU (0x3) ;- (SLCDC) COM3 selected
AT91C_SLCDC_COMSEL_4      EQU (0x4) ;- (SLCDC) COM4 selected
AT91C_SLCDC_COMSEL_5      EQU (0x5) ;- (SLCDC) COM5 selected
AT91C_SLCDC_COMSEL_6      EQU (0x6) ;- (SLCDC) COM6 selected
AT91C_SLCDC_COMSEL_7      EQU (0x7) ;- (SLCDC) COM7 selected
AT91C_SLCDC_COMSEL_8      EQU (0x8) ;- (SLCDC) COM8 selected
AT91C_SLCDC_COMSEL_9      EQU (0x9) ;- (SLCDC) COM9 selected
AT91C_SLCDC_SEGSEL        EQU (0x1F <<  8) ;- (SLCDC) Selection of the number of segment
AT91C_SLCDC_BUFTIME       EQU (0xF << 16) ;- (SLCDC) Buffer on time
AT91C_SLCDC_BUFTIME_0_percent EQU (0x0 << 16) ;- (SLCDC) Buffer aren't driven
AT91C_SLCDC_BUFTIME_2_Tsclk EQU (0x1 << 16) ;- (SLCDC) Buffer are driven during 2 SLCDC clock periods
AT91C_SLCDC_BUFTIME_4_Tsclk EQU (0x2 << 16) ;- (SLCDC) Buffer are driven during 4 SLCDC clock periods
AT91C_SLCDC_BUFTIME_8_Tsclk EQU (0x3 << 16) ;- (SLCDC) Buffer are driven during 8 SLCDC clock periods
AT91C_SLCDC_BUFTIME_16_Tsclk EQU (0x4 << 16) ;- (SLCDC) Buffer are driven during 16 SLCDC clock periods
AT91C_SLCDC_BUFTIME_32_Tsclk EQU (0x5 << 16) ;- (SLCDC) Buffer are driven during 32 SLCDC clock periods
AT91C_SLCDC_BUFTIME_64_Tsclk EQU (0x6 << 16) ;- (SLCDC) Buffer are driven during 64 SLCDC clock periods
AT91C_SLCDC_BUFTIME_128_Tsclk EQU (0x7 << 16) ;- (SLCDC) Buffer are driven during 128 SLCDC clock periods
AT91C_SLCDC_BUFTIME_50_percent EQU (0x8 << 16) ;- (SLCDC) Buffer are driven during 50 percent of the frame
AT91C_SLCDC_BUFTIME_100_percent EQU (0x9 << 16) ;- (SLCDC) Buffer are driven during 100 percent of the frame
AT91C_SLCDC_BIAS          EQU (0x3 << 20) ;- (SLCDC) Bias setting
AT91C_SLCDC_BIAS_1        EQU (0x0 << 20) ;- (SLCDC) Vbias is VDD 
AT91C_SLCDC_BIAS_1_2      EQU (0x1 << 20) ;- (SLCDC) Vbias is 1/2 VDD
AT91C_SLCDC_BIAS_1_3      EQU (0x2 << 20) ;- (SLCDC) Vbias is 1/3 VDD
AT91C_SLCDC_BIAS_1_4      EQU (0x3 << 20) ;- (SLCDC) Vbias is 1/4 VDD
AT91C_SLCDC_LPMODE        EQU (0x1 << 24) ;- (SLCDC) Low Power mode
/* - -------- SLCDC_FRR : (SLCDC Offset: 0x8) SLCDC Frame Rate Register -------- */
AT91C_SLCDC_PRESC         EQU (0x7 <<  0) ;- (SLCDC) Clock prescaler
AT91C_SLCDC_PRESC_SCLK_8  EQU (0x0) ;- (SLCDC) Clock Prescaler is 8
AT91C_SLCDC_PRESC_SCLK_16 EQU (0x1) ;- (SLCDC) Clock Prescaler is 16
AT91C_SLCDC_PRESC_SCLK_32 EQU (0x2) ;- (SLCDC) Clock Prescaler is 32
AT91C_SLCDC_PRESC_SCLK_64 EQU (0x3) ;- (SLCDC) Clock Prescaler is 64
AT91C_SLCDC_PRESC_SCLK_128 EQU (0x4) ;- (SLCDC) Clock Prescaler is 128
AT91C_SLCDC_PRESC_SCLK_256 EQU (0x5) ;- (SLCDC) Clock Prescaler is 256
AT91C_SLCDC_PRESC_SCLK_512 EQU (0x6) ;- (SLCDC) Clock Prescaler is 512
AT91C_SLCDC_PRESC_SCLK_1024 EQU (0x7) ;- (SLCDC) Clock Prescaler is 1024
AT91C_SLCDC_DIV           EQU (0x7 <<  8) ;- (SLCDC) Clock division
AT91C_SLCDC_DIV_1         EQU (0x0 <<  8) ;- (SLCDC) Clock division is 1
AT91C_SLCDC_DIV_2         EQU (0x1 <<  8) ;- (SLCDC) Clock division is 2
AT91C_SLCDC_DIV_3         EQU (0x2 <<  8) ;- (SLCDC) Clock division is 3
AT91C_SLCDC_DIV_4         EQU (0x3 <<  8) ;- (SLCDC) Clock division is 4
AT91C_SLCDC_DIV_5         EQU (0x4 <<  8) ;- (SLCDC) Clock division is 5
AT91C_SLCDC_DIV_6         EQU (0x5 <<  8) ;- (SLCDC) Clock division is 6
AT91C_SLCDC_DIV_7         EQU (0x6 <<  8) ;- (SLCDC) Clock division is 7
AT91C_SLCDC_DIV_8         EQU (0x7 <<  8) ;- (SLCDC) Clock division is 8
/* - -------- SLCDC_DR : (SLCDC Offset: 0xc) SLCDC Display Register -------- */
AT91C_SLCDC_DISPMODE      EQU (0x7 <<  0) ;- (SLCDC) Display mode
AT91C_SLCDC_DISPMODE_NORMAL EQU (0x0) ;- (SLCDC) Latched datas are displayed
AT91C_SLCDC_DISPMODE_FORCE_OFF EQU (0x1) ;- (SLCDC) All pixel are unvisible
AT91C_SLCDC_DISPMODE_FORCE_ON EQU (0x2) ;- (SLCDC) All pixel are visible
AT91C_SLCDC_DISPMODE_BLINK EQU (0x3) ;- (SLCDC) Turn all pixel alternatively off to the state defined in SCLD memory at LCDBLKFREQ frequency
AT91C_SLCDC_DISPMODE_INVERTED EQU (0x4) ;- (SLCDC) All pixel are set in the inverted state as defined in SCLD memory
AT91C_SLCDC_DISPMODE_INVERTED_BLINK EQU (0x5) ;- (SLCDC) Turn all pixel alternatively off to the opposite state defined in SCLD memory at LCDBLKFREQ frequency
AT91C_SLCDC_BLKFREQ       EQU (0xFF <<  8) ;- (SLCDC) Blinking frequency
/* - -------- SLCDC_SR : (SLCDC Offset: 0x10) SLCDC Status  Register -------- */
AT91C_SLCDC_ENA           EQU (0x1 <<  0) ;- (SLCDC) Enable status
/* - -------- SLCDC_IER : (SLCDC Offset: 0x20) SLCDC Interrupt Enable Register -------- */
AT91C_SLCDC_ENDFRAME      EQU (0x1 <<  0) ;- (SLCDC) End of Frame
AT91C_SLCDC_BKPER         EQU (0x1 <<  1) ;- (SLCDC) Blank Periode
AT91C_SLCDC_DIS           EQU (0x1 <<  2) ;- (SLCDC) Disable
/* - -------- SLCDC_IDR : (SLCDC Offset: 0x24) SLCDC Interrupt Disable Register -------- */
/* - -------- SLCDC_IMR : (SLCDC Offset: 0x28) SLCDC Interrupt Mask Register -------- */
/* - -------- SLCDC_ISR : (SLCDC Offset: 0x2c) SLCDC Interrupt Status Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Analog to Digital Convertor*/
/* - ******************************************************************************/
/* - -------- ADC_CR : (ADC Offset: 0x0) ADC Control Register -------- */
AT91C_ADC_SWRST           EQU (0x1 <<  0) ;- (ADC) Software Reset
AT91C_ADC_START           EQU (0x1 <<  1) ;- (ADC) Start Conversion
/* - -------- ADC_MR : (ADC Offset: 0x4) ADC Mode Register -------- */
AT91C_ADC_TRGEN           EQU (0x1 <<  0) ;- (ADC) Trigger Enable
AT91C_ADC_TRGEN_DIS       EQU (0x0) ;- (ADC) Hradware triggers are disabled. Starting a conversion is only possible by software
AT91C_ADC_TRGEN_EN        EQU (0x1) ;- (ADC) Hardware trigger selected by TRGSEL field is enabled.
AT91C_ADC_TRGSEL          EQU (0x7 <<  1) ;- (ADC) Trigger Selection
AT91C_ADC_TRGSEL_TIOA0    EQU (0x0 <<  1) ;- (ADC) Selected TRGSEL = TIAO0
AT91C_ADC_TRGSEL_TIOA1    EQU (0x1 <<  1) ;- (ADC) Selected TRGSEL = TIAO1
AT91C_ADC_TRGSEL_TIOA2    EQU (0x2 <<  1) ;- (ADC) Selected TRGSEL = TIAO2
AT91C_ADC_TRGSEL_TIOA3    EQU (0x3 <<  1) ;- (ADC) Selected TRGSEL = TIAO3
AT91C_ADC_TRGSEL_TIOA4    EQU (0x4 <<  1) ;- (ADC) Selected TRGSEL = TIAO4
AT91C_ADC_TRGSEL_TIOA5    EQU (0x5 <<  1) ;- (ADC) Selected TRGSEL = TIAO5
AT91C_ADC_TRGSEL_EXT      EQU (0x6 <<  1) ;- (ADC) Selected TRGSEL = External Trigger
AT91C_ADC_LOWRES          EQU (0x1 <<  4) ;- (ADC) Resolution.
AT91C_ADC_LOWRES_10_BIT   EQU (0x0 <<  4) ;- (ADC) 10-bit resolution
AT91C_ADC_LOWRES_8_BIT    EQU (0x1 <<  4) ;- (ADC) 8-bit resolution
AT91C_ADC_SLEEP           EQU (0x1 <<  5) ;- (ADC) Sleep Mode
AT91C_ADC_SLEEP_NORMAL_MODE EQU (0x0 <<  5) ;- (ADC) Normal Mode
AT91C_ADC_SLEEP_MODE      EQU (0x1 <<  5) ;- (ADC) Sleep Mode
AT91C_ADC_PRESCAL         EQU (0x3F <<  8) ;- (ADC) Prescaler rate selection
AT91C_ADC_STARTUP         EQU (0x1F << 16) ;- (ADC) Startup Time
AT91C_ADC_SHTIM           EQU (0xF << 24) ;- (ADC) Sample & Hold Time
/* - -------- 	ADC_CHER : (ADC Offset: 0x10) ADC Channel Enable Register -------- */
AT91C_ADC_CH0             EQU (0x1 <<  0) ;- (ADC) Channel 0
AT91C_ADC_CH1             EQU (0x1 <<  1) ;- (ADC) Channel 1
AT91C_ADC_CH2             EQU (0x1 <<  2) ;- (ADC) Channel 2
AT91C_ADC_CH3             EQU (0x1 <<  3) ;- (ADC) Channel 3
AT91C_ADC_CH4             EQU (0x1 <<  4) ;- (ADC) Channel 4
AT91C_ADC_CH5             EQU (0x1 <<  5) ;- (ADC) Channel 5
AT91C_ADC_CH6             EQU (0x1 <<  6) ;- (ADC) Channel 6
AT91C_ADC_CH7             EQU (0x1 <<  7) ;- (ADC) Channel 7
/* - -------- 	ADC_CHDR : (ADC Offset: 0x14) ADC Channel Disable Register -------- */
/* - -------- 	ADC_CHSR : (ADC Offset: 0x18) ADC Channel Status Register -------- */
/* - -------- ADC_SR : (ADC Offset: 0x1c) ADC Status Register -------- */
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
/* - -------- ADC_LCDR : (ADC Offset: 0x20) ADC Last Converted Data Register -------- */
AT91C_ADC_LDATA           EQU (0x3FF <<  0) ;- (ADC) Last Data Converted
/* - -------- ADC_IER : (ADC Offset: 0x24) ADC Interrupt Enable Register -------- */
/* - -------- ADC_IDR : (ADC Offset: 0x28) ADC Interrupt Disable Register -------- */
/* - -------- ADC_IMR : (ADC Offset: 0x2c) ADC Interrupt Mask Register -------- */
/* - -------- ADC_CDR0 : (ADC Offset: 0x30) ADC Channel Data Register 0 -------- */
AT91C_ADC_DATA            EQU (0x3FF <<  0) ;- (ADC) Converted Data
/* - -------- ADC_CDR1 : (ADC Offset: 0x34) ADC Channel Data Register 1 -------- */
/* - -------- ADC_CDR2 : (ADC Offset: 0x38) ADC Channel Data Register 2 -------- */
/* - -------- ADC_CDR3 : (ADC Offset: 0x3c) ADC Channel Data Register 3 -------- */
/* - -------- ADC_CDR4 : (ADC Offset: 0x40) ADC Channel Data Register 4 -------- */
/* - -------- ADC_CDR5 : (ADC Offset: 0x44) ADC Channel Data Register 5 -------- */
/* - -------- ADC_CDR6 : (ADC Offset: 0x48) ADC Channel Data Register 6 -------- */
/* - -------- ADC_CDR7 : (ADC Offset: 0x4c) ADC Channel Data Register 7 -------- */

/* - ******************************************************************************/
/* -               REGISTER ADDRESS DEFINITION FOR AT91SAM7L128*/
/* - ******************************************************************************/
/* - ========== Register definition for SYS peripheral ========== */
/* - ========== Register definition for AIC peripheral ========== */
AT91C_AIC_IVR             EQU (0xFFFFF100) ;- (AIC) IRQ Vector Register
AT91C_AIC_SMR             EQU (0xFFFFF000) ;- (AIC) Source Mode Register
AT91C_AIC_FVR             EQU (0xFFFFF104) ;- (AIC) FIQ Vector Register
AT91C_AIC_DCR             EQU (0xFFFFF138) ;- (AIC) Debug Control Register (Protect)
AT91C_AIC_EOICR           EQU (0xFFFFF130) ;- (AIC) End of Interrupt Command Register
AT91C_AIC_SVR             EQU (0xFFFFF080) ;- (AIC) Source Vector Register
AT91C_AIC_FFSR            EQU (0xFFFFF148) ;- (AIC) Fast Forcing Status Register
AT91C_AIC_ICCR            EQU (0xFFFFF128) ;- (AIC) Interrupt Clear Command Register
AT91C_AIC_ISR             EQU (0xFFFFF108) ;- (AIC) Interrupt Status Register
AT91C_AIC_IMR             EQU (0xFFFFF110) ;- (AIC) Interrupt Mask Register
AT91C_AIC_IPR             EQU (0xFFFFF10C) ;- (AIC) Interrupt Pending Register
AT91C_AIC_FFER            EQU (0xFFFFF140) ;- (AIC) Fast Forcing Enable Register
AT91C_AIC_IECR            EQU (0xFFFFF120) ;- (AIC) Interrupt Enable Command Register
AT91C_AIC_ISCR            EQU (0xFFFFF12C) ;- (AIC) Interrupt Set Command Register
AT91C_AIC_FFDR            EQU (0xFFFFF144) ;- (AIC) Fast Forcing Disable Register
AT91C_AIC_CISR            EQU (0xFFFFF114) ;- (AIC) Core Interrupt Status Register
AT91C_AIC_IDCR            EQU (0xFFFFF124) ;- (AIC) Interrupt Disable Command Register
AT91C_AIC_SPU             EQU (0xFFFFF134) ;- (AIC) Spurious Vector Register
/* - ========== Register definition for PDC_DBGU peripheral ========== */
AT91C_DBGU_TCR            EQU (0xFFFFF30C) ;- (PDC_DBGU) Transmit Counter Register
AT91C_DBGU_RNPR           EQU (0xFFFFF310) ;- (PDC_DBGU) Receive Next Pointer Register
AT91C_DBGU_TNPR           EQU (0xFFFFF318) ;- (PDC_DBGU) Transmit Next Pointer Register
AT91C_DBGU_TPR            EQU (0xFFFFF308) ;- (PDC_DBGU) Transmit Pointer Register
AT91C_DBGU_RPR            EQU (0xFFFFF300) ;- (PDC_DBGU) Receive Pointer Register
AT91C_DBGU_RCR            EQU (0xFFFFF304) ;- (PDC_DBGU) Receive Counter Register
AT91C_DBGU_RNCR           EQU (0xFFFFF314) ;- (PDC_DBGU) Receive Next Counter Register
AT91C_DBGU_PTCR           EQU (0xFFFFF320) ;- (PDC_DBGU) PDC Transfer Control Register
AT91C_DBGU_PTSR           EQU (0xFFFFF324) ;- (PDC_DBGU) PDC Transfer Status Register
AT91C_DBGU_TNCR           EQU (0xFFFFF31C) ;- (PDC_DBGU) Transmit Next Counter Register
/* - ========== Register definition for DBGU peripheral ========== */
AT91C_DBGU_EXID           EQU (0xFFFFF244) ;- (DBGU) Chip ID Extension Register
AT91C_DBGU_BRGR           EQU (0xFFFFF220) ;- (DBGU) Baud Rate Generator Register
AT91C_DBGU_IDR            EQU (0xFFFFF20C) ;- (DBGU) Interrupt Disable Register
AT91C_DBGU_CSR            EQU (0xFFFFF214) ;- (DBGU) Channel Status Register
AT91C_DBGU_CIDR           EQU (0xFFFFF240) ;- (DBGU) Chip ID Register
AT91C_DBGU_MR             EQU (0xFFFFF204) ;- (DBGU) Mode Register
AT91C_DBGU_IMR            EQU (0xFFFFF210) ;- (DBGU) Interrupt Mask Register
AT91C_DBGU_CR             EQU (0xFFFFF200) ;- (DBGU) Control Register
AT91C_DBGU_FNTR           EQU (0xFFFFF248) ;- (DBGU) Force NTRST Register
AT91C_DBGU_THR            EQU (0xFFFFF21C) ;- (DBGU) Transmitter Holding Register
AT91C_DBGU_RHR            EQU (0xFFFFF218) ;- (DBGU) Receiver Holding Register
AT91C_DBGU_IER            EQU (0xFFFFF208) ;- (DBGU) Interrupt Enable Register
/* - ========== Register definition for PIOA peripheral ========== */
AT91C_PIOA_ODR            EQU (0xFFFFF414) ;- (PIOA) Output Disable Registerr
AT91C_PIOA_SODR           EQU (0xFFFFF430) ;- (PIOA) Set Output Data Register
AT91C_PIOA_ISR            EQU (0xFFFFF44C) ;- (PIOA) Interrupt Status Register
AT91C_PIOA_ABSR           EQU (0xFFFFF478) ;- (PIOA) AB Select Status Register
AT91C_PIOA_IER            EQU (0xFFFFF440) ;- (PIOA) Interrupt Enable Register
AT91C_PIOA_PPUDR          EQU (0xFFFFF460) ;- (PIOA) Pull-up Disable Register
AT91C_PIOA_IMR            EQU (0xFFFFF448) ;- (PIOA) Interrupt Mask Register
AT91C_PIOA_PER            EQU (0xFFFFF400) ;- (PIOA) PIO Enable Register
AT91C_PIOA_IFDR           EQU (0xFFFFF424) ;- (PIOA) Input Filter Disable Register
AT91C_PIOA_OWDR           EQU (0xFFFFF4A4) ;- (PIOA) Output Write Disable Register
AT91C_PIOA_MDSR           EQU (0xFFFFF458) ;- (PIOA) Multi-driver Status Register
AT91C_PIOA_IDR            EQU (0xFFFFF444) ;- (PIOA) Interrupt Disable Register
AT91C_PIOA_ODSR           EQU (0xFFFFF438) ;- (PIOA) Output Data Status Register
AT91C_PIOA_PPUSR          EQU (0xFFFFF468) ;- (PIOA) Pull-up Status Register
AT91C_PIOA_OWSR           EQU (0xFFFFF4A8) ;- (PIOA) Output Write Status Register
AT91C_PIOA_BSR            EQU (0xFFFFF474) ;- (PIOA) Select B Register
AT91C_PIOA_OWER           EQU (0xFFFFF4A0) ;- (PIOA) Output Write Enable Register
AT91C_PIOA_IFER           EQU (0xFFFFF420) ;- (PIOA) Input Filter Enable Register
AT91C_PIOA_PDSR           EQU (0xFFFFF43C) ;- (PIOA) Pin Data Status Register
AT91C_PIOA_PPUER          EQU (0xFFFFF464) ;- (PIOA) Pull-up Enable Register
AT91C_PIOA_OSR            EQU (0xFFFFF418) ;- (PIOA) Output Status Register
AT91C_PIOA_ASR            EQU (0xFFFFF470) ;- (PIOA) Select A Register
AT91C_PIOA_MDDR           EQU (0xFFFFF454) ;- (PIOA) Multi-driver Disable Register
AT91C_PIOA_CODR           EQU (0xFFFFF434) ;- (PIOA) Clear Output Data Register
AT91C_PIOA_MDER           EQU (0xFFFFF450) ;- (PIOA) Multi-driver Enable Register
AT91C_PIOA_PDR            EQU (0xFFFFF404) ;- (PIOA) PIO Disable Register
AT91C_PIOA_IFSR           EQU (0xFFFFF428) ;- (PIOA) Input Filter Status Register
AT91C_PIOA_OER            EQU (0xFFFFF410) ;- (PIOA) Output Enable Register
AT91C_PIOA_PSR            EQU (0xFFFFF408) ;- (PIOA) PIO Status Register
/* - ========== Register definition for PIOB peripheral ========== */
AT91C_PIOB_OWDR           EQU (0xFFFFF6A4) ;- (PIOB) Output Write Disable Register
AT91C_PIOB_MDER           EQU (0xFFFFF650) ;- (PIOB) Multi-driver Enable Register
AT91C_PIOB_PPUSR          EQU (0xFFFFF668) ;- (PIOB) Pull-up Status Register
AT91C_PIOB_IMR            EQU (0xFFFFF648) ;- (PIOB) Interrupt Mask Register
AT91C_PIOB_ASR            EQU (0xFFFFF670) ;- (PIOB) Select A Register
AT91C_PIOB_PPUDR          EQU (0xFFFFF660) ;- (PIOB) Pull-up Disable Register
AT91C_PIOB_PSR            EQU (0xFFFFF608) ;- (PIOB) PIO Status Register
AT91C_PIOB_IER            EQU (0xFFFFF640) ;- (PIOB) Interrupt Enable Register
AT91C_PIOB_CODR           EQU (0xFFFFF634) ;- (PIOB) Clear Output Data Register
AT91C_PIOB_OWER           EQU (0xFFFFF6A0) ;- (PIOB) Output Write Enable Register
AT91C_PIOB_ABSR           EQU (0xFFFFF678) ;- (PIOB) AB Select Status Register
AT91C_PIOB_IFDR           EQU (0xFFFFF624) ;- (PIOB) Input Filter Disable Register
AT91C_PIOB_PDSR           EQU (0xFFFFF63C) ;- (PIOB) Pin Data Status Register
AT91C_PIOB_IDR            EQU (0xFFFFF644) ;- (PIOB) Interrupt Disable Register
AT91C_PIOB_OWSR           EQU (0xFFFFF6A8) ;- (PIOB) Output Write Status Register
AT91C_PIOB_PDR            EQU (0xFFFFF604) ;- (PIOB) PIO Disable Register
AT91C_PIOB_ODR            EQU (0xFFFFF614) ;- (PIOB) Output Disable Registerr
AT91C_PIOB_IFSR           EQU (0xFFFFF628) ;- (PIOB) Input Filter Status Register
AT91C_PIOB_PPUER          EQU (0xFFFFF664) ;- (PIOB) Pull-up Enable Register
AT91C_PIOB_SODR           EQU (0xFFFFF630) ;- (PIOB) Set Output Data Register
AT91C_PIOB_ISR            EQU (0xFFFFF64C) ;- (PIOB) Interrupt Status Register
AT91C_PIOB_ODSR           EQU (0xFFFFF638) ;- (PIOB) Output Data Status Register
AT91C_PIOB_OSR            EQU (0xFFFFF618) ;- (PIOB) Output Status Register
AT91C_PIOB_MDSR           EQU (0xFFFFF658) ;- (PIOB) Multi-driver Status Register
AT91C_PIOB_IFER           EQU (0xFFFFF620) ;- (PIOB) Input Filter Enable Register
AT91C_PIOB_BSR            EQU (0xFFFFF674) ;- (PIOB) Select B Register
AT91C_PIOB_MDDR           EQU (0xFFFFF654) ;- (PIOB) Multi-driver Disable Register
AT91C_PIOB_OER            EQU (0xFFFFF610) ;- (PIOB) Output Enable Register
AT91C_PIOB_PER            EQU (0xFFFFF600) ;- (PIOB) PIO Enable Register
/* - ========== Register definition for PIOC peripheral ========== */
AT91C_PIOC_OWDR           EQU (0xFFFFF8A4) ;- (PIOC) Output Write Disable Register
AT91C_PIOC_SODR           EQU (0xFFFFF830) ;- (PIOC) Set Output Data Register
AT91C_PIOC_PPUER          EQU (0xFFFFF864) ;- (PIOC) Pull-up Enable Register
AT91C_PIOC_CODR           EQU (0xFFFFF834) ;- (PIOC) Clear Output Data Register
AT91C_PIOC_PSR            EQU (0xFFFFF808) ;- (PIOC) PIO Status Register
AT91C_PIOC_PDR            EQU (0xFFFFF804) ;- (PIOC) PIO Disable Register
AT91C_PIOC_ODR            EQU (0xFFFFF814) ;- (PIOC) Output Disable Registerr
AT91C_PIOC_PPUSR          EQU (0xFFFFF868) ;- (PIOC) Pull-up Status Register
AT91C_PIOC_ABSR           EQU (0xFFFFF878) ;- (PIOC) AB Select Status Register
AT91C_PIOC_IFSR           EQU (0xFFFFF828) ;- (PIOC) Input Filter Status Register
AT91C_PIOC_OER            EQU (0xFFFFF810) ;- (PIOC) Output Enable Register
AT91C_PIOC_IMR            EQU (0xFFFFF848) ;- (PIOC) Interrupt Mask Register
AT91C_PIOC_ASR            EQU (0xFFFFF870) ;- (PIOC) Select A Register
AT91C_PIOC_MDDR           EQU (0xFFFFF854) ;- (PIOC) Multi-driver Disable Register
AT91C_PIOC_OWSR           EQU (0xFFFFF8A8) ;- (PIOC) Output Write Status Register
AT91C_PIOC_PER            EQU (0xFFFFF800) ;- (PIOC) PIO Enable Register
AT91C_PIOC_IDR            EQU (0xFFFFF844) ;- (PIOC) Interrupt Disable Register
AT91C_PIOC_MDER           EQU (0xFFFFF850) ;- (PIOC) Multi-driver Enable Register
AT91C_PIOC_PDSR           EQU (0xFFFFF83C) ;- (PIOC) Pin Data Status Register
AT91C_PIOC_MDSR           EQU (0xFFFFF858) ;- (PIOC) Multi-driver Status Register
AT91C_PIOC_OWER           EQU (0xFFFFF8A0) ;- (PIOC) Output Write Enable Register
AT91C_PIOC_BSR            EQU (0xFFFFF874) ;- (PIOC) Select B Register
AT91C_PIOC_PPUDR          EQU (0xFFFFF860) ;- (PIOC) Pull-up Disable Register
AT91C_PIOC_IFDR           EQU (0xFFFFF824) ;- (PIOC) Input Filter Disable Register
AT91C_PIOC_IER            EQU (0xFFFFF840) ;- (PIOC) Interrupt Enable Register
AT91C_PIOC_OSR            EQU (0xFFFFF818) ;- (PIOC) Output Status Register
AT91C_PIOC_ODSR           EQU (0xFFFFF838) ;- (PIOC) Output Data Status Register
AT91C_PIOC_ISR            EQU (0xFFFFF84C) ;- (PIOC) Interrupt Status Register
AT91C_PIOC_IFER           EQU (0xFFFFF820) ;- (PIOC) Input Filter Enable Register
/* - ========== Register definition for CKGR peripheral ========== */
AT91C_CKGR_MOR            EQU (0xFFFFFC20) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_MCFR           EQU (0xFFFFFC24) ;- (CKGR) Main Clock  Frequency Register
AT91C_CKGR_PLLR           EQU (0xFFFFFC28) ;- (CKGR) PLL Register
/* - ========== Register definition for PMC peripheral ========== */
AT91C_PMC_PCER            EQU (0xFFFFFC10) ;- (PMC) Peripheral Clock Enable Register
AT91C_PMC_PCKR            EQU (0xFFFFFC40) ;- (PMC) Programmable Clock Register
AT91C_PMC_MCKR            EQU (0xFFFFFC30) ;- (PMC) Master Clock Register
AT91C_PMC_PLLR            EQU (0xFFFFFC28) ;- (PMC) PLL Register
AT91C_PMC_PCDR            EQU (0xFFFFFC14) ;- (PMC) Peripheral Clock Disable Register
AT91C_PMC_SCSR            EQU (0xFFFFFC08) ;- (PMC) System Clock Status Register
AT91C_PMC_MCFR            EQU (0xFFFFFC24) ;- (PMC) Main Clock  Frequency Register
AT91C_PMC_IMR             EQU (0xFFFFFC6C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IER             EQU (0xFFFFFC60) ;- (PMC) Interrupt Enable Register
AT91C_PMC_MOR             EQU (0xFFFFFC20) ;- (PMC) Main Oscillator Register
AT91C_PMC_IDR             EQU (0xFFFFFC64) ;- (PMC) Interrupt Disable Register
AT91C_PMC_SCDR            EQU (0xFFFFFC04) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCSR            EQU (0xFFFFFC18) ;- (PMC) Peripheral Clock Status Register
AT91C_PMC_FSMR            EQU (0xFFFFFC70) ;- (PMC) Fast Startup Mode Register
AT91C_PMC_SCER            EQU (0xFFFFFC00) ;- (PMC) System Clock Enable Register
AT91C_PMC_SR              EQU (0xFFFFFC68) ;- (PMC) Status Register
/* - ========== Register definition for RSTC peripheral ========== */
AT91C_RSTC_RCR            EQU (0xFFFFFD00) ;- (RSTC) Reset Control Register
AT91C_RSTC_RMR            EQU (0xFFFFFD08) ;- (RSTC) Reset Mode Register
AT91C_RSTC_RSR            EQU (0xFFFFFD04) ;- (RSTC) Reset Status Register
/* - ========== Register definition for RTC peripheral ========== */
AT91C_RTC_IMR             EQU (0xFFFFFD88) ;- (RTC) Interrupt Mask Register
AT91C_RTC_TAR             EQU (0xFFFFFD70) ;- (RTC) Time Alarm Register
AT91C_RTC_IDR             EQU (0xFFFFFD84) ;- (RTC) Interrupt Disable Register
AT91C_RTC_HMR             EQU (0xFFFFFD64) ;- (RTC) Hour Mode Register
AT91C_RTC_SR              EQU (0xFFFFFD78) ;- (RTC) Status Register
AT91C_RTC_VER             EQU (0xFFFFFD8C) ;- (RTC) Valid Entry Register
AT91C_RTC_TIMR            EQU (0xFFFFFD68) ;- (RTC) Time Register
AT91C_RTC_CAR             EQU (0xFFFFFD74) ;- (RTC) Calendar Alarm Register
AT91C_RTC_IER             EQU (0xFFFFFD80) ;- (RTC) Interrupt Enable Register
AT91C_RTC_SCR             EQU (0xFFFFFD7C) ;- (RTC) Status Clear Register
AT91C_RTC_MR              EQU (0xFFFFFD60) ;- (RTC) Mode Register
AT91C_RTC_CALR            EQU (0xFFFFFD6C) ;- (RTC) Calendar Register
/* - ========== Register definition for PITC peripheral ========== */
AT91C_PITC_PIMR           EQU (0xFFFFFD40) ;- (PITC) Period Interval Mode Register
AT91C_PITC_PIVR           EQU (0xFFFFFD48) ;- (PITC) Period Interval Value Register
AT91C_PITC_PISR           EQU (0xFFFFFD44) ;- (PITC) Period Interval Status Register
AT91C_PITC_PIIR           EQU (0xFFFFFD4C) ;- (PITC) Period Interval Image Register
/* - ========== Register definition for WDTC peripheral ========== */
AT91C_WDTC_WDMR           EQU (0xFFFFFD54) ;- (WDTC) Watchdog Mode Register
AT91C_WDTC_WDSR           EQU (0xFFFFFD58) ;- (WDTC) Watchdog Status Register
AT91C_WDTC_WDCR           EQU (0xFFFFFD50) ;- (WDTC) Watchdog Control Register
/* - ========== Register definition for VREG peripheral ========== */
AT91C_VREG_MR             EQU (0xFFFFFD60) ;- (VREG) Voltage Regulator Mode Register
/* - ========== Register definition for MC peripheral ========== */
AT91C_MC_ASR              EQU (0xFFFFFF04) ;- (MC) MC Abort Status Register
AT91C_MC_RCR              EQU (0xFFFFFF00) ;- (MC) MC Remap Control Register
AT91C_MC_FCR              EQU (0xFFFFFF64) ;- (MC) MC Flash Command Register
AT91C_MC_AASR             EQU (0xFFFFFF08) ;- (MC) MC Abort Address Status Register
AT91C_MC_FSR              EQU (0xFFFFFF68) ;- (MC) MC Flash Status Register
AT91C_MC_FRR              EQU (0xFFFFFF6C) ;- (MC) MC Flash Result Register
AT91C_MC_FMR              EQU (0xFFFFFF60) ;- (MC) MC Flash Mode Register
/* - ========== Register definition for PDC_SPI peripheral ========== */
AT91C_SPI_PTCR            EQU (0xFFFE0120) ;- (PDC_SPI) PDC Transfer Control Register
AT91C_SPI_TPR             EQU (0xFFFE0108) ;- (PDC_SPI) Transmit Pointer Register
AT91C_SPI_TCR             EQU (0xFFFE010C) ;- (PDC_SPI) Transmit Counter Register
AT91C_SPI_RCR             EQU (0xFFFE0104) ;- (PDC_SPI) Receive Counter Register
AT91C_SPI_PTSR            EQU (0xFFFE0124) ;- (PDC_SPI) PDC Transfer Status Register
AT91C_SPI_RNPR            EQU (0xFFFE0110) ;- (PDC_SPI) Receive Next Pointer Register
AT91C_SPI_RPR             EQU (0xFFFE0100) ;- (PDC_SPI) Receive Pointer Register
AT91C_SPI_TNCR            EQU (0xFFFE011C) ;- (PDC_SPI) Transmit Next Counter Register
AT91C_SPI_RNCR            EQU (0xFFFE0114) ;- (PDC_SPI) Receive Next Counter Register
AT91C_SPI_TNPR            EQU (0xFFFE0118) ;- (PDC_SPI) Transmit Next Pointer Register
/* - ========== Register definition for SPI peripheral ========== */
AT91C_SPI_IER             EQU (0xFFFE0014) ;- (SPI) Interrupt Enable Register
AT91C_SPI_SR              EQU (0xFFFE0010) ;- (SPI) Status Register
AT91C_SPI_IDR             EQU (0xFFFE0018) ;- (SPI) Interrupt Disable Register
AT91C_SPI_CR              EQU (0xFFFE0000) ;- (SPI) Control Register
AT91C_SPI_MR              EQU (0xFFFE0004) ;- (SPI) Mode Register
AT91C_SPI_IMR             EQU (0xFFFE001C) ;- (SPI) Interrupt Mask Register
AT91C_SPI_TDR             EQU (0xFFFE000C) ;- (SPI) Transmit Data Register
AT91C_SPI_RDR             EQU (0xFFFE0008) ;- (SPI) Receive Data Register
AT91C_SPI_CSR             EQU (0xFFFE0030) ;- (SPI) Chip Select Register
/* - ========== Register definition for PDC_US1 peripheral ========== */
AT91C_US1_RNCR            EQU (0xFFFC4114) ;- (PDC_US1) Receive Next Counter Register
AT91C_US1_PTCR            EQU (0xFFFC4120) ;- (PDC_US1) PDC Transfer Control Register
AT91C_US1_TCR             EQU (0xFFFC410C) ;- (PDC_US1) Transmit Counter Register
AT91C_US1_PTSR            EQU (0xFFFC4124) ;- (PDC_US1) PDC Transfer Status Register
AT91C_US1_TNPR            EQU (0xFFFC4118) ;- (PDC_US1) Transmit Next Pointer Register
AT91C_US1_RCR             EQU (0xFFFC4104) ;- (PDC_US1) Receive Counter Register
AT91C_US1_RNPR            EQU (0xFFFC4110) ;- (PDC_US1) Receive Next Pointer Register
AT91C_US1_RPR             EQU (0xFFFC4100) ;- (PDC_US1) Receive Pointer Register
AT91C_US1_TNCR            EQU (0xFFFC411C) ;- (PDC_US1) Transmit Next Counter Register
AT91C_US1_TPR             EQU (0xFFFC4108) ;- (PDC_US1) Transmit Pointer Register
/* - ========== Register definition for US1 peripheral ========== */
AT91C_US1_IF              EQU (0xFFFC404C) ;- (US1) IRDA_FILTER Register
AT91C_US1_NER             EQU (0xFFFC4044) ;- (US1) Nb Errors Register
AT91C_US1_RTOR            EQU (0xFFFC4024) ;- (US1) Receiver Time-out Register
AT91C_US1_CSR             EQU (0xFFFC4014) ;- (US1) Channel Status Register
AT91C_US1_IDR             EQU (0xFFFC400C) ;- (US1) Interrupt Disable Register
AT91C_US1_IER             EQU (0xFFFC4008) ;- (US1) Interrupt Enable Register
AT91C_US1_THR             EQU (0xFFFC401C) ;- (US1) Transmitter Holding Register
AT91C_US1_TTGR            EQU (0xFFFC4028) ;- (US1) Transmitter Time-guard Register
AT91C_US1_RHR             EQU (0xFFFC4018) ;- (US1) Receiver Holding Register
AT91C_US1_BRGR            EQU (0xFFFC4020) ;- (US1) Baud Rate Generator Register
AT91C_US1_IMR             EQU (0xFFFC4010) ;- (US1) Interrupt Mask Register
AT91C_US1_FIDI            EQU (0xFFFC4040) ;- (US1) FI_DI_Ratio Register
AT91C_US1_CR              EQU (0xFFFC4000) ;- (US1) Control Register
AT91C_US1_MR              EQU (0xFFFC4004) ;- (US1) Mode Register
/* - ========== Register definition for PDC_US0 peripheral ========== */
AT91C_US0_TNPR            EQU (0xFFFC0118) ;- (PDC_US0) Transmit Next Pointer Register
AT91C_US0_RNPR            EQU (0xFFFC0110) ;- (PDC_US0) Receive Next Pointer Register
AT91C_US0_TCR             EQU (0xFFFC010C) ;- (PDC_US0) Transmit Counter Register
AT91C_US0_PTCR            EQU (0xFFFC0120) ;- (PDC_US0) PDC Transfer Control Register
AT91C_US0_PTSR            EQU (0xFFFC0124) ;- (PDC_US0) PDC Transfer Status Register
AT91C_US0_TNCR            EQU (0xFFFC011C) ;- (PDC_US0) Transmit Next Counter Register
AT91C_US0_TPR             EQU (0xFFFC0108) ;- (PDC_US0) Transmit Pointer Register
AT91C_US0_RCR             EQU (0xFFFC0104) ;- (PDC_US0) Receive Counter Register
AT91C_US0_RPR             EQU (0xFFFC0100) ;- (PDC_US0) Receive Pointer Register
AT91C_US0_RNCR            EQU (0xFFFC0114) ;- (PDC_US0) Receive Next Counter Register
/* - ========== Register definition for US0 peripheral ========== */
AT91C_US0_BRGR            EQU (0xFFFC0020) ;- (US0) Baud Rate Generator Register
AT91C_US0_NER             EQU (0xFFFC0044) ;- (US0) Nb Errors Register
AT91C_US0_CR              EQU (0xFFFC0000) ;- (US0) Control Register
AT91C_US0_IMR             EQU (0xFFFC0010) ;- (US0) Interrupt Mask Register
AT91C_US0_FIDI            EQU (0xFFFC0040) ;- (US0) FI_DI_Ratio Register
AT91C_US0_TTGR            EQU (0xFFFC0028) ;- (US0) Transmitter Time-guard Register
AT91C_US0_MR              EQU (0xFFFC0004) ;- (US0) Mode Register
AT91C_US0_RTOR            EQU (0xFFFC0024) ;- (US0) Receiver Time-out Register
AT91C_US0_CSR             EQU (0xFFFC0014) ;- (US0) Channel Status Register
AT91C_US0_RHR             EQU (0xFFFC0018) ;- (US0) Receiver Holding Register
AT91C_US0_IDR             EQU (0xFFFC000C) ;- (US0) Interrupt Disable Register
AT91C_US0_THR             EQU (0xFFFC001C) ;- (US0) Transmitter Holding Register
AT91C_US0_IF              EQU (0xFFFC004C) ;- (US0) IRDA_FILTER Register
AT91C_US0_IER             EQU (0xFFFC0008) ;- (US0) Interrupt Enable Register
/* - ========== Register definition for PDC_TWI peripheral ========== */
AT91C_TWI_TNCR            EQU (0xFFFB811C) ;- (PDC_TWI) Transmit Next Counter Register
AT91C_TWI_RNCR            EQU (0xFFFB8114) ;- (PDC_TWI) Receive Next Counter Register
AT91C_TWI_TNPR            EQU (0xFFFB8118) ;- (PDC_TWI) Transmit Next Pointer Register
AT91C_TWI_PTCR            EQU (0xFFFB8120) ;- (PDC_TWI) PDC Transfer Control Register
AT91C_TWI_TCR             EQU (0xFFFB810C) ;- (PDC_TWI) Transmit Counter Register
AT91C_TWI_RPR             EQU (0xFFFB8100) ;- (PDC_TWI) Receive Pointer Register
AT91C_TWI_TPR             EQU (0xFFFB8108) ;- (PDC_TWI) Transmit Pointer Register
AT91C_TWI_RCR             EQU (0xFFFB8104) ;- (PDC_TWI) Receive Counter Register
AT91C_TWI_PTSR            EQU (0xFFFB8124) ;- (PDC_TWI) PDC Transfer Status Register
AT91C_TWI_RNPR            EQU (0xFFFB8110) ;- (PDC_TWI) Receive Next Pointer Register
/* - ========== Register definition for TWI peripheral ========== */
AT91C_TWI_IER             EQU (0xFFFB8024) ;- (TWI) Interrupt Enable Register
AT91C_TWI_CR              EQU (0xFFFB8000) ;- (TWI) Control Register
AT91C_TWI_SR              EQU (0xFFFB8020) ;- (TWI) Status Register
AT91C_TWI_IMR             EQU (0xFFFB802C) ;- (TWI) Interrupt Mask Register
AT91C_TWI_THR             EQU (0xFFFB8034) ;- (TWI) Transmit Holding Register
AT91C_TWI_IDR             EQU (0xFFFB8028) ;- (TWI) Interrupt Disable Register
AT91C_TWI_IADR            EQU (0xFFFB800C) ;- (TWI) Internal Address Register
AT91C_TWI_MMR             EQU (0xFFFB8004) ;- (TWI) Master Mode Register
AT91C_TWI_CWGR            EQU (0xFFFB8010) ;- (TWI) Clock Waveform Generator Register
AT91C_TWI_RHR             EQU (0xFFFB8030) ;- (TWI) Receive Holding Register
AT91C_TWI_SMR             EQU (0xFFFB8008) ;- (TWI) Slave Mode Register
/* - ========== Register definition for PWMC_CH3 peripheral ========== */
AT91C_CH3_CUPDR           EQU (0xFFFCC270) ;- (PWMC_CH3) Channel Update Register
AT91C_CH3_Reserved        EQU (0xFFFCC274) ;- (PWMC_CH3) Reserved
AT91C_CH3_CPRDR           EQU (0xFFFCC268) ;- (PWMC_CH3) Channel Period Register
AT91C_CH3_CDTYR           EQU (0xFFFCC264) ;- (PWMC_CH3) Channel Duty Cycle Register
AT91C_CH3_CCNTR           EQU (0xFFFCC26C) ;- (PWMC_CH3) Channel Counter Register
AT91C_CH3_CMR             EQU (0xFFFCC260) ;- (PWMC_CH3) Channel Mode Register
/* - ========== Register definition for PWMC_CH2 peripheral ========== */
AT91C_CH2_Reserved        EQU (0xFFFCC254) ;- (PWMC_CH2) Reserved
AT91C_CH2_CMR             EQU (0xFFFCC240) ;- (PWMC_CH2) Channel Mode Register
AT91C_CH2_CCNTR           EQU (0xFFFCC24C) ;- (PWMC_CH2) Channel Counter Register
AT91C_CH2_CPRDR           EQU (0xFFFCC248) ;- (PWMC_CH2) Channel Period Register
AT91C_CH2_CUPDR           EQU (0xFFFCC250) ;- (PWMC_CH2) Channel Update Register
AT91C_CH2_CDTYR           EQU (0xFFFCC244) ;- (PWMC_CH2) Channel Duty Cycle Register
/* - ========== Register definition for PWMC_CH1 peripheral ========== */
AT91C_CH1_Reserved        EQU (0xFFFCC234) ;- (PWMC_CH1) Reserved
AT91C_CH1_CUPDR           EQU (0xFFFCC230) ;- (PWMC_CH1) Channel Update Register
AT91C_CH1_CPRDR           EQU (0xFFFCC228) ;- (PWMC_CH1) Channel Period Register
AT91C_CH1_CCNTR           EQU (0xFFFCC22C) ;- (PWMC_CH1) Channel Counter Register
AT91C_CH1_CDTYR           EQU (0xFFFCC224) ;- (PWMC_CH1) Channel Duty Cycle Register
AT91C_CH1_CMR             EQU (0xFFFCC220) ;- (PWMC_CH1) Channel Mode Register
/* - ========== Register definition for PWMC_CH0 peripheral ========== */
AT91C_CH0_Reserved        EQU (0xFFFCC214) ;- (PWMC_CH0) Reserved
AT91C_CH0_CPRDR           EQU (0xFFFCC208) ;- (PWMC_CH0) Channel Period Register
AT91C_CH0_CDTYR           EQU (0xFFFCC204) ;- (PWMC_CH0) Channel Duty Cycle Register
AT91C_CH0_CMR             EQU (0xFFFCC200) ;- (PWMC_CH0) Channel Mode Register
AT91C_CH0_CUPDR           EQU (0xFFFCC210) ;- (PWMC_CH0) Channel Update Register
AT91C_CH0_CCNTR           EQU (0xFFFCC20C) ;- (PWMC_CH0) Channel Counter Register
/* - ========== Register definition for PWMC peripheral ========== */
AT91C_PWMC_IDR            EQU (0xFFFCC014) ;- (PWMC) PWMC Interrupt Disable Register
AT91C_PWMC_DIS            EQU (0xFFFCC008) ;- (PWMC) PWMC Disable Register
AT91C_PWMC_IER            EQU (0xFFFCC010) ;- (PWMC) PWMC Interrupt Enable Register
AT91C_PWMC_VR             EQU (0xFFFCC0FC) ;- (PWMC) PWMC Version Register
AT91C_PWMC_ISR            EQU (0xFFFCC01C) ;- (PWMC) PWMC Interrupt Status Register
AT91C_PWMC_SR             EQU (0xFFFCC00C) ;- (PWMC) PWMC Status Register
AT91C_PWMC_IMR            EQU (0xFFFCC018) ;- (PWMC) PWMC Interrupt Mask Register
AT91C_PWMC_MR             EQU (0xFFFCC000) ;- (PWMC) PWMC Mode Register
AT91C_PWMC_ENA            EQU (0xFFFCC004) ;- (PWMC) PWMC Enable Register
/* - ========== Register definition for TC0 peripheral ========== */
AT91C_TC0_SR              EQU (0xFFFA0020) ;- (TC0) Status Register
AT91C_TC0_RC              EQU (0xFFFA001C) ;- (TC0) Register C
AT91C_TC0_RB              EQU (0xFFFA0018) ;- (TC0) Register B
AT91C_TC0_CCR             EQU (0xFFFA0000) ;- (TC0) Channel Control Register
AT91C_TC0_CMR             EQU (0xFFFA0004) ;- (TC0) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC0_IER             EQU (0xFFFA0024) ;- (TC0) Interrupt Enable Register
AT91C_TC0_RA              EQU (0xFFFA0014) ;- (TC0) Register A
AT91C_TC0_IDR             EQU (0xFFFA0028) ;- (TC0) Interrupt Disable Register
AT91C_TC0_CV              EQU (0xFFFA0010) ;- (TC0) Counter Value
AT91C_TC0_IMR             EQU (0xFFFA002C) ;- (TC0) Interrupt Mask Register
/* - ========== Register definition for TC1 peripheral ========== */
AT91C_TC1_RB              EQU (0xFFFA0058) ;- (TC1) Register B
AT91C_TC1_CCR             EQU (0xFFFA0040) ;- (TC1) Channel Control Register
AT91C_TC1_IER             EQU (0xFFFA0064) ;- (TC1) Interrupt Enable Register
AT91C_TC1_IDR             EQU (0xFFFA0068) ;- (TC1) Interrupt Disable Register
AT91C_TC1_SR              EQU (0xFFFA0060) ;- (TC1) Status Register
AT91C_TC1_CMR             EQU (0xFFFA0044) ;- (TC1) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC1_RA              EQU (0xFFFA0054) ;- (TC1) Register A
AT91C_TC1_RC              EQU (0xFFFA005C) ;- (TC1) Register C
AT91C_TC1_IMR             EQU (0xFFFA006C) ;- (TC1) Interrupt Mask Register
AT91C_TC1_CV              EQU (0xFFFA0050) ;- (TC1) Counter Value
/* - ========== Register definition for TC2 peripheral ========== */
AT91C_TC2_CMR             EQU (0xFFFA0084) ;- (TC2) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC2_CCR             EQU (0xFFFA0080) ;- (TC2) Channel Control Register
AT91C_TC2_CV              EQU (0xFFFA0090) ;- (TC2) Counter Value
AT91C_TC2_RA              EQU (0xFFFA0094) ;- (TC2) Register A
AT91C_TC2_RB              EQU (0xFFFA0098) ;- (TC2) Register B
AT91C_TC2_IDR             EQU (0xFFFA00A8) ;- (TC2) Interrupt Disable Register
AT91C_TC2_IMR             EQU (0xFFFA00AC) ;- (TC2) Interrupt Mask Register
AT91C_TC2_RC              EQU (0xFFFA009C) ;- (TC2) Register C
AT91C_TC2_IER             EQU (0xFFFA00A4) ;- (TC2) Interrupt Enable Register
AT91C_TC2_SR              EQU (0xFFFA00A0) ;- (TC2) Status Register
/* - ========== Register definition for TCB peripheral ========== */
AT91C_TCB_BMR             EQU (0xFFFA00C4) ;- (TCB) TC Block Mode Register
AT91C_TCB_BCR             EQU (0xFFFA00C0) ;- (TCB) TC Block Control Register
/* - ========== Register definition for SLCDC peripheral ========== */
AT91C_SLCDC_IMR           EQU (0xFFFB4028) ;- (SLCDC) Interrupt Mask Register
AT91C_SLCDC_IER           EQU (0xFFFB4020) ;- (SLCDC) Interrupt Enable Register
AT91C_SLCDC_MEM           EQU (0xFFFB4200) ;- (SLCDC) Memory Register
AT91C_SLCDC_DR            EQU (0xFFFB400C) ;- (SLCDC) Display Register
AT91C_SLCDC_MR            EQU (0xFFFB4004) ;- (SLCDC) Mode Register
AT91C_SLCDC_ISR           EQU (0xFFFB402C) ;- (SLCDC) Interrupt Status Register
AT91C_SLCDC_IDR           EQU (0xFFFB4024) ;- (SLCDC) Interrupt Disable Register
AT91C_SLCDC_CR            EQU (0xFFFB4000) ;- (SLCDC) Control Register
AT91C_SLCDC_SR            EQU (0xFFFB4010) ;- (SLCDC) Status Register
AT91C_SLCDC_FRR           EQU (0xFFFB4008) ;- (SLCDC) Frame Rate Register
/* - ========== Register definition for PDC_ADC peripheral ========== */
AT91C_ADC_PTSR            EQU (0xFFFD8124) ;- (PDC_ADC) PDC Transfer Status Register
AT91C_ADC_PTCR            EQU (0xFFFD8120) ;- (PDC_ADC) PDC Transfer Control Register
AT91C_ADC_TNPR            EQU (0xFFFD8118) ;- (PDC_ADC) Transmit Next Pointer Register
AT91C_ADC_TNCR            EQU (0xFFFD811C) ;- (PDC_ADC) Transmit Next Counter Register
AT91C_ADC_RNPR            EQU (0xFFFD8110) ;- (PDC_ADC) Receive Next Pointer Register
AT91C_ADC_RNCR            EQU (0xFFFD8114) ;- (PDC_ADC) Receive Next Counter Register
AT91C_ADC_RPR             EQU (0xFFFD8100) ;- (PDC_ADC) Receive Pointer Register
AT91C_ADC_TCR             EQU (0xFFFD810C) ;- (PDC_ADC) Transmit Counter Register
AT91C_ADC_TPR             EQU (0xFFFD8108) ;- (PDC_ADC) Transmit Pointer Register
AT91C_ADC_RCR             EQU (0xFFFD8104) ;- (PDC_ADC) Receive Counter Register
/* - ========== Register definition for ADC peripheral ========== */
AT91C_ADC_CDR2            EQU (0xFFFD8038) ;- (ADC) ADC Channel Data Register 2
AT91C_ADC_CDR3            EQU (0xFFFD803C) ;- (ADC) ADC Channel Data Register 3
AT91C_ADC_CDR0            EQU (0xFFFD8030) ;- (ADC) ADC Channel Data Register 0
AT91C_ADC_CDR5            EQU (0xFFFD8044) ;- (ADC) ADC Channel Data Register 5
AT91C_ADC_CHDR            EQU (0xFFFD8014) ;- (ADC) ADC Channel Disable Register
AT91C_ADC_SR              EQU (0xFFFD801C) ;- (ADC) ADC Status Register
AT91C_ADC_CDR4            EQU (0xFFFD8040) ;- (ADC) ADC Channel Data Register 4
AT91C_ADC_CDR1            EQU (0xFFFD8034) ;- (ADC) ADC Channel Data Register 1
AT91C_ADC_LCDR            EQU (0xFFFD8020) ;- (ADC) ADC Last Converted Data Register
AT91C_ADC_IDR             EQU (0xFFFD8028) ;- (ADC) ADC Interrupt Disable Register
AT91C_ADC_CR              EQU (0xFFFD8000) ;- (ADC) ADC Control Register
AT91C_ADC_CDR7            EQU (0xFFFD804C) ;- (ADC) ADC Channel Data Register 7
AT91C_ADC_CDR6            EQU (0xFFFD8048) ;- (ADC) ADC Channel Data Register 6
AT91C_ADC_IER             EQU (0xFFFD8024) ;- (ADC) ADC Interrupt Enable Register
AT91C_ADC_CHER            EQU (0xFFFD8010) ;- (ADC) ADC Channel Enable Register
AT91C_ADC_CHSR            EQU (0xFFFD8018) ;- (ADC) ADC Channel Status Register
AT91C_ADC_MR              EQU (0xFFFD8004) ;- (ADC) ADC Mode Register
AT91C_ADC_IMR             EQU (0xFFFD802C) ;- (ADC) ADC Interrupt Mask Register

/* - ******************************************************************************/
/* -               PIO DEFINITIONS FOR AT91SAM7L128*/
/* - ******************************************************************************/
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_NPCS3          EQU (AT91C_PIO_PB12) ;-  
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_NPCS2          EQU (AT91C_PIO_PB13) ;-  
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_NPCS1          EQU (AT91C_PIO_PB14) ;-  
AT91C_PIO_PB15            EQU (1 << 15) ;- Pin Controlled by PB15
AT91C_PB15_RTS1           EQU (AT91C_PIO_PB15) ;-  
AT91C_PIO_PB16            EQU (1 << 16) ;- Pin Controlled by PB16
AT91C_PB16_RTS0           EQU (AT91C_PIO_PB16) ;-  
AT91C_PIO_PB17            EQU (1 << 17) ;- Pin Controlled by PB17
AT91C_PB17_DTR1           EQU (AT91C_PIO_PB17) ;-  
AT91C_PIO_PB18            EQU (1 << 18) ;- Pin Controlled by PB18
AT91C_PB18_PWM0           EQU (AT91C_PIO_PB18) ;-  
AT91C_PIO_PB19            EQU (1 << 19) ;- Pin Controlled by PB19
AT91C_PB19_PWM1           EQU (AT91C_PIO_PB19) ;-  
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PIO_PB20            EQU (1 << 20) ;- Pin Controlled by PB20
AT91C_PB20_PWM2           EQU (AT91C_PIO_PB20) ;-  
AT91C_PIO_PB21            EQU (1 << 21) ;- Pin Controlled by PB21
AT91C_PB21_PWM3           EQU (AT91C_PIO_PB21) ;-  
AT91C_PIO_PB22            EQU (1 << 22) ;- Pin Controlled by PB22
AT91C_PB22_NPCS1          EQU (AT91C_PIO_PB22) ;-  
AT91C_PB22_PCK1           EQU (AT91C_PIO_PB22) ;-  
AT91C_PIO_PB23            EQU (1 << 23) ;- Pin Controlled by PB23
AT91C_PB23_PCK0           EQU (AT91C_PIO_PB23) ;-  
AT91C_PB23_NPCS3          EQU (AT91C_PIO_PB23) ;-  
AT91C_PIO_PB24            EQU (1 << 24) ;- Pin Controlled by PB24
AT91C_PIO_PB25            EQU (1 << 25) ;- Pin Controlled by PB25
AT91C_PB25_PCK0           EQU (AT91C_PIO_PB25) ;-  
AT91C_PB25_NPCS3          EQU (AT91C_PIO_PB25) ;-  
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PIO_PC0             EQU (1 <<  0) ;- Pin Controlled by PC0
AT91C_PC0_CTS1            EQU (AT91C_PIO_PC0) ;-  
AT91C_PC0_PWM2            EQU (AT91C_PIO_PC0) ;-  
AT91C_PIO_PC1             EQU (1 <<  1) ;- Pin Controlled by PC1
AT91C_PC1_DCD1            EQU (AT91C_PIO_PC1) ;-  
AT91C_PC1_TIOA2           EQU (AT91C_PIO_PC1) ;-  
AT91C_PIO_PC10            EQU (1 << 10) ;- Pin Controlled by PC10
AT91C_PC10_TWD            EQU (AT91C_PIO_PC10) ;-  
AT91C_PC10_NPCS3          EQU (AT91C_PIO_PC10) ;-  
AT91C_PIO_PC11            EQU (1 << 11) ;- Pin Controlled by PC11
AT91C_PC11_TWCK           EQU (AT91C_PIO_PC11) ;-  
AT91C_PC11_TCLK0          EQU (AT91C_PIO_PC11) ;-  
AT91C_PIO_PC12            EQU (1 << 12) ;- Pin Controlled by PC12
AT91C_PC12_RXD0           EQU (AT91C_PIO_PC12) ;-  
AT91C_PC12_NPCS3          EQU (AT91C_PIO_PC12) ;-  
AT91C_PIO_PC13            EQU (1 << 13) ;- Pin Controlled by PC13
AT91C_PC13_TXD0           EQU (AT91C_PIO_PC13) ;-  
AT91C_PC13_PCK0           EQU (AT91C_PIO_PC13) ;-  
AT91C_PIO_PC14            EQU (1 << 14) ;- Pin Controlled by PC14
AT91C_PC14_RTS0           EQU (AT91C_PIO_PC14) ;-  
AT91C_PC14_ADTRG          EQU (AT91C_PIO_PC14) ;-  
AT91C_PIO_PC15            EQU (1 << 15) ;- Pin Controlled by PC15
AT91C_PC15_CTS0           EQU (AT91C_PIO_PC15) ;-  
AT91C_PC15_PWM3           EQU (AT91C_PIO_PC15) ;-  
AT91C_PIO_PC16            EQU (1 << 16) ;- Pin Controlled by PC16
AT91C_PC16_DRXD           EQU (AT91C_PIO_PC16) ;-  
AT91C_PC16_NPCS1          EQU (AT91C_PIO_PC16) ;-  
AT91C_PIO_PC17            EQU (1 << 17) ;- Pin Controlled by PC17
AT91C_PC17_DTXD           EQU (AT91C_PIO_PC17) ;-  
AT91C_PC17_NPCS2          EQU (AT91C_PIO_PC17) ;-  
AT91C_PIO_PC18            EQU (1 << 18) ;- Pin Controlled by PC18
AT91C_PC18_NPCS0          EQU (AT91C_PIO_PC18) ;-  
AT91C_PC18_PWM0           EQU (AT91C_PIO_PC18) ;-  
AT91C_PIO_PC19            EQU (1 << 19) ;- Pin Controlled by PC19
AT91C_PC19_MISO           EQU (AT91C_PIO_PC19) ;-  
AT91C_PC19_PWM1           EQU (AT91C_PIO_PC19) ;-  
AT91C_PIO_PC2             EQU (1 <<  2) ;- Pin Controlled by PC2
AT91C_PC2_DTR1            EQU (AT91C_PIO_PC2) ;-  
AT91C_PC2_TIOB2           EQU (AT91C_PIO_PC2) ;-  
AT91C_PIO_PC20            EQU (1 << 20) ;- Pin Controlled by PC20
AT91C_PC20_MOSI           EQU (AT91C_PIO_PC20) ;-  
AT91C_PC20_PWM2           EQU (AT91C_PIO_PC20) ;-  
AT91C_PIO_PC21            EQU (1 << 21) ;- Pin Controlled by PC21
AT91C_PC21_SPCK           EQU (AT91C_PIO_PC21) ;-  
AT91C_PC21_PWM3           EQU (AT91C_PIO_PC21) ;-  
AT91C_PIO_PC22            EQU (1 << 22) ;- Pin Controlled by PC22
AT91C_PC22_NPCS3          EQU (AT91C_PIO_PC22) ;-  
AT91C_PC22_TIOA1          EQU (AT91C_PIO_PC22) ;-  
AT91C_PIO_PC23            EQU (1 << 23) ;- Pin Controlled by PC23
AT91C_PC23_PCK0           EQU (AT91C_PIO_PC23) ;-  
AT91C_PC23_TIOB1          EQU (AT91C_PIO_PC23) ;-  
AT91C_PIO_PC24            EQU (1 << 24) ;- Pin Controlled by PC24
AT91C_PC24_RXD1           EQU (AT91C_PIO_PC24) ;-  
AT91C_PC24_PCK1           EQU (AT91C_PIO_PC24) ;-  
AT91C_PIO_PC25            EQU (1 << 25) ;- Pin Controlled by PC25
AT91C_PC25_TXD1           EQU (AT91C_PIO_PC25) ;-  
AT91C_PC25_PCK2           EQU (AT91C_PIO_PC25) ;-  
AT91C_PIO_PC26            EQU (1 << 26) ;- Pin Controlled by PC26
AT91C_PC26_RTS0           EQU (AT91C_PIO_PC26) ;-  
AT91C_PC26_FIQ            EQU (AT91C_PIO_PC26) ;-  
AT91C_PIO_PC27            EQU (1 << 27) ;- Pin Controlled by PC27
AT91C_PC27_NPCS2          EQU (AT91C_PIO_PC27) ;-  
AT91C_PC27_IRQ0           EQU (AT91C_PIO_PC27) ;-  
AT91C_PIO_PC28            EQU (1 << 28) ;- Pin Controlled by PC28
AT91C_PC28_SCK1           EQU (AT91C_PIO_PC28) ;-  
AT91C_PC28_PWM0           EQU (AT91C_PIO_PC28) ;-  
AT91C_PIO_PC29            EQU (1 << 29) ;- Pin Controlled by PC29
AT91C_PC29_RTS1           EQU (AT91C_PIO_PC29) ;-  
AT91C_PC29_PWM1           EQU (AT91C_PIO_PC29) ;-  
AT91C_PIO_PC3             EQU (1 <<  3) ;- Pin Controlled by PC3
AT91C_PC3_DSR1            EQU (AT91C_PIO_PC3) ;-  
AT91C_PC3_TCLK1           EQU (AT91C_PIO_PC3) ;-  
AT91C_PIO_PC4             EQU (1 <<  4) ;- Pin Controlled by PC4
AT91C_PC4_RI1             EQU (AT91C_PIO_PC4) ;-  
AT91C_PC4_TCLK2           EQU (AT91C_PIO_PC4) ;-  
AT91C_PIO_PC5             EQU (1 <<  5) ;- Pin Controlled by PC5
AT91C_PC5_IRQ1            EQU (AT91C_PIO_PC5) ;-  
AT91C_PC5_NPCS2           EQU (AT91C_PIO_PC5) ;-  
AT91C_PIO_PC6             EQU (1 <<  6) ;- Pin Controlled by PC6
AT91C_PC6_NPCS1           EQU (AT91C_PIO_PC6) ;-  
AT91C_PC6_PCK2            EQU (AT91C_PIO_PC6) ;-  
AT91C_PIO_PC7             EQU (1 <<  7) ;- Pin Controlled by PC7
AT91C_PC7_PWM0            EQU (AT91C_PIO_PC7) ;-  
AT91C_PC7_TIOA0           EQU (AT91C_PIO_PC7) ;-  
AT91C_PIO_PC8             EQU (1 <<  8) ;- Pin Controlled by PC8
AT91C_PC8_PWM1            EQU (AT91C_PIO_PC8) ;-  
AT91C_PC8_TIOB0           EQU (AT91C_PIO_PC8) ;-  
AT91C_PIO_PC9             EQU (1 <<  9) ;- Pin Controlled by PC9
AT91C_PC9_PWM2            EQU (AT91C_PIO_PC9) ;-  
AT91C_PC9_SCK0            EQU (AT91C_PIO_PC9) ;-  

/* - ******************************************************************************/
/* -               PERIPHERAL ID DEFINITIONS FOR AT91SAM7L128*/
/* - ******************************************************************************/
AT91C_ID_FIQ              EQU ( 0) ;- Advanced Interrupt Controller (FIQ)
AT91C_ID_SYS              EQU ( 1) ;- System Peripheral
AT91C_ID_PIOA             EQU ( 2) ;- Parallel IO Controller A
AT91C_ID_PIOB             EQU ( 3) ;- Parallel IO Controller B
AT91C_ID_PIOC             EQU ( 4) ;- Parallel IO Controller C
AT91C_ID_SPI              EQU ( 5) ;- Serial Peripheral Interface 0
AT91C_ID_US0              EQU ( 6) ;- USART 0
AT91C_ID_US1              EQU ( 7) ;- USART 1
AT91C_ID_8_Reserved       EQU ( 8) ;- Reserved
AT91C_ID_TWI              EQU ( 9) ;- Two-Wire Interface
AT91C_ID_PWMC             EQU (10) ;- PWM Controller
AT91C_ID_SLCD             EQU (11) ;- Segmented LCD Controller
AT91C_ID_TC0              EQU (12) ;- Timer Counter 0
AT91C_ID_TC1              EQU (13) ;- Timer Counter 1
AT91C_ID_TC2              EQU (14) ;- Timer Counter 2
AT91C_ID_ADC              EQU (15) ;- Analog-to-Digital Converter
AT91C_ID_16_Reserved      EQU (16) ;- Reserved
AT91C_ID_17_Reserved      EQU (17) ;- Reserved
AT91C_ID_18_Reserved      EQU (18) ;- Reserved
AT91C_ID_19_Reserved      EQU (19) ;- Reserved
AT91C_ID_20_Reserved      EQU (20) ;- Reserved
AT91C_ID_21_Reserved      EQU (21) ;- Reserved
AT91C_ID_22_Reserved      EQU (22) ;- Reserved
AT91C_ID_23_Reserved      EQU (23) ;- Reserved
AT91C_ID_24_Reserved      EQU (24) ;- Reserved
AT91C_ID_25_Reserved      EQU (25) ;- Reserved
AT91C_ID_26_Reserved      EQU (26) ;- Reserved
AT91C_ID_27_Reserved      EQU (27) ;- Reserved
AT91C_ID_28_Reserved      EQU (28) ;- Reserved
AT91C_ID_IRQ0             EQU (30) ;- Advanced Interrupt Controller (IRQ0)
AT91C_ID_IRQ1             EQU (31) ;- Advanced Interrupt Controller (IRQ1)
AT91C_ALL_INT             EQU (0xC000FEFF) ;- ALL VALID INTERRUPTS

/* - ******************************************************************************/
/* -               BASE ADDRESS DEFINITIONS FOR AT91SAM7L128*/
/* - ******************************************************************************/
AT91C_BASE_SYS            EQU (0xFFFFF000) ;- (SYS) Base Address
AT91C_BASE_AIC            EQU (0xFFFFF000) ;- (AIC) Base Address
AT91C_BASE_PDC_DBGU       EQU (0xFFFFF300) ;- (PDC_DBGU) Base Address
AT91C_BASE_DBGU           EQU (0xFFFFF200) ;- (DBGU) Base Address
AT91C_BASE_PIOA           EQU (0xFFFFF400) ;- (PIOA) Base Address
AT91C_BASE_PIOB           EQU (0xFFFFF600) ;- (PIOB) Base Address
AT91C_BASE_PIOC           EQU (0xFFFFF800) ;- (PIOC) Base Address
AT91C_BASE_CKGR           EQU (0xFFFFFC20) ;- (CKGR) Base Address
AT91C_BASE_PMC            EQU (0xFFFFFC00) ;- (PMC) Base Address
AT91C_BASE_RSTC           EQU (0xFFFFFD00) ;- (RSTC) Base Address
AT91C_BASE_RTC            EQU (0xFFFFFD60) ;- (RTC) Base Address
AT91C_BASE_PITC           EQU (0xFFFFFD40) ;- (PITC) Base Address
AT91C_BASE_WDTC           EQU (0xFFFFFD50) ;- (WDTC) Base Address
AT91C_BASE_VREG           EQU (0xFFFFFD60) ;- (VREG) Base Address
AT91C_BASE_MC             EQU (0xFFFFFF00) ;- (MC) Base Address
AT91C_BASE_PDC_SPI        EQU (0xFFFE0100) ;- (PDC_SPI) Base Address
AT91C_BASE_SPI            EQU (0xFFFE0000) ;- (SPI) Base Address
AT91C_BASE_PDC_US1        EQU (0xFFFC4100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0xFFFC4000) ;- (US1) Base Address
AT91C_BASE_PDC_US0        EQU (0xFFFC0100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0xFFFC0000) ;- (US0) Base Address
AT91C_BASE_PDC_TWI        EQU (0xFFFB8100) ;- (PDC_TWI) Base Address
AT91C_BASE_TWI            EQU (0xFFFB8000) ;- (TWI) Base Address
AT91C_BASE_PWMC_CH3       EQU (0xFFFCC260) ;- (PWMC_CH3) Base Address
AT91C_BASE_PWMC_CH2       EQU (0xFFFCC240) ;- (PWMC_CH2) Base Address
AT91C_BASE_PWMC_CH1       EQU (0xFFFCC220) ;- (PWMC_CH1) Base Address
AT91C_BASE_PWMC_CH0       EQU (0xFFFCC200) ;- (PWMC_CH0) Base Address
AT91C_BASE_PWMC           EQU (0xFFFCC000) ;- (PWMC) Base Address
AT91C_BASE_TC0            EQU (0xFFFA0000) ;- (TC0) Base Address
AT91C_BASE_TC1            EQU (0xFFFA0040) ;- (TC1) Base Address
AT91C_BASE_TC2            EQU (0xFFFA0080) ;- (TC2) Base Address
AT91C_BASE_TCB            EQU (0xFFFA0000) ;- (TCB) Base Address
AT91C_BASE_SLCDC          EQU (0xFFFB4000) ;- (SLCDC) Base Address
AT91C_BASE_PDC_ADC        EQU (0xFFFD8100) ;- (PDC_ADC) Base Address
AT91C_BASE_ADC            EQU (0xFFFD8000) ;- (ADC) Base Address

/* - ******************************************************************************/
/* -               MEMORY MAPPING DEFINITIONS FOR AT91SAM7L128*/
/* - ******************************************************************************/
/* - ISRAM_1*/
AT91C_ISRAM_1             EQU (0x00200000) ;- Internal SRAM base address
AT91C_ISRAM_1_SIZE        EQU (0x00001000) ;- Internal SRAM size in byte (4 Kbytes)
/* - ISRAM_2*/
AT91C_ISRAM_2             EQU (0x00300000) ;- Internal SRAM base address
AT91C_ISRAM_2_SIZE        EQU (0x00000800) ;- Internal SRAM size in byte (2 Kbytes)
/* - IFLASH*/
AT91C_IFLASH              EQU (0x00100000) ;- Internal FLASH base address
AT91C_IFLASH_SIZE         EQU (0x00020000) ;- Internal FLASH size in byte (128 Kbytes)
AT91C_IFLASH_PAGE_SIZE    EQU (256) ;- Internal FLASH Page Size: 256 bytes
AT91C_IFLASH_LOCK_REGION_SIZE EQU (8192) ;- Internal FLASH Lock Region Size: 8 Kbytes
AT91C_IFLASH_NB_OF_PAGES  EQU (512) ;- Internal FLASH Number of Pages: 512 bytes
AT91C_IFLASH_NB_OF_LOCK_BITS EQU (16) ;- Internal FLASH Number of Lock Bits: 16 bytes
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91SAM7L128_H */
