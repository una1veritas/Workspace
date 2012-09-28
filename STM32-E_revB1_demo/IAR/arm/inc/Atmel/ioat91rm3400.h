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
/* - File Name           : AT91RM3400.h*/
/* - Object              : AT91RM3400 definitions*/
/* - Generated           : AT91 SW Application Group  09/05/2005 (14:05:57)*/
/* - */
/* - CVS Reference       : /AT91RM3400.pl/1.11/Fri Feb  7 10:29:52 2003  */
/* - CVS Reference       : /SYS_0000A.pl/1.1.1.1/Wed Apr 28 12:01:18 2004  */
/* - CVS Reference       : /MC_1760A.pl/1.1/Fri Aug 23 15:38:22 2002  */
/* - CVS Reference       : /AIC_1796B.pl/1.1.1.1/Fri Jun 28 10:36:48 2002  */
/* - CVS Reference       : /PMC_2636A.pl/1.1.1.1/Fri Jun 28 10:36:48 2002  */
/* - CVS Reference       : /ST_1763B.pl/1.1/Fri Aug 23 15:41:42 2002  */
/* - CVS Reference       : /RTC_1245D.pl/1.3/Fri Sep 17 15:10:20 2004  */
/* - CVS Reference       : /PIO_1725D.pl/1.1.1.1/Fri Jun 28 10:36:48 2002  */
/* - CVS Reference       : /DBGU_1754A.pl/1.4/Fri Jan 31 12:18:24 2003  */
/* - CVS Reference       : /UDP_1765B.pl/1.3/Fri Aug  2 15:45:38 2002  */
/* - CVS Reference       : /MCI_1764A.pl/1.4/Tue May 18 13:48:46 2004  */
/* - CVS Reference       : /US_1739C.pl/1.2/Mon Jul 12 18:26:24 2004  */
/* - CVS Reference       : /SPI_AT91RMxxxx.pl/1.3/Tue Nov 26 10:20:30 2002  */
/* - CVS Reference       : /SSC_1762A.pl/1.2/Fri Nov  8 13:26:40 2002  */
/* - CVS Reference       : /TC_1753B.pl/1.7/Fri Feb 18 13:53:44 2005  */
/* - CVS Reference       : /TWI_1761B.pl/1.4/Fri Feb  7 10:30:08 2003  */
/* - CVS Reference       : /PDC_1734B.pl/1.2/Thu Nov 21 16:38:24 2002  */
/* - ----------------------------------------------------------------------------*/

#ifndef AT91RM3400_H
#define AT91RM3400_H

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
	AT91_REG	 DBGU_C1R; 	/* Chip ID1 Register*/
	AT91_REG	 DBGU_C2R; 	/* Chip ID2 Register*/
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
	AT91_REG	 PIOA_PPUSR; 	/* Pad Pull-up Status Register*/
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
	AT91_REG	 PIOB_PPUSR; 	/* Pad Pull-up Status Register*/
	AT91_REG	 Reserved17[1]; 	/* */
	AT91_REG	 PIOB_ASR; 	/* Select A Register*/
	AT91_REG	 PIOB_BSR; 	/* Select B Register*/
	AT91_REG	 PIOB_ABSR; 	/* AB Select Status Register*/
	AT91_REG	 Reserved18[9]; 	/* */
	AT91_REG	 PIOB_OWER; 	/* Output Write Enable Register*/
	AT91_REG	 PIOB_OWDR; 	/* Output Write Disable Register*/
	AT91_REG	 PIOB_OWSR; 	/* Output Write Status Register*/
	AT91_REG	 Reserved19[341]; 	/* */
	AT91_REG	 PMC_SCER; 	/* System Clock Enable Register*/
	AT91_REG	 PMC_SCDR; 	/* System Clock Disable Register*/
	AT91_REG	 PMC_SCSR; 	/* System Clock Status Register*/
	AT91_REG	 Reserved20[1]; 	/* */
	AT91_REG	 PMC_PCER; 	/* Peripheral Clock Enable Register*/
	AT91_REG	 PMC_PCDR; 	/* Peripheral Clock Disable Register*/
	AT91_REG	 PMC_PCSR; 	/* Peripheral Clock Status Register*/
	AT91_REG	 Reserved21[1]; 	/* */
	AT91_REG	 CKGR_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 CKGR_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 CKGR_PLLAR; 	/* PLL A Register*/
	AT91_REG	 CKGR_PLLBR; 	/* PLL B Register*/
	AT91_REG	 PMC_MCKR; 	/* Master Clock Register*/
	AT91_REG	 Reserved22[3]; 	/* */
	AT91_REG	 PMC_PCKR[8]; 	/* Programmable Clock Register*/
	AT91_REG	 PMC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PMC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PMC_SR; 	/* Status Register*/
	AT91_REG	 PMC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 Reserved23[36]; 	/* */
	AT91_REG	 ST_CR; 	/* Control Register*/
	AT91_REG	 ST_PIMR; 	/* Period Interval Mode Register*/
	AT91_REG	 ST_WDMR; 	/* Watchdog Mode Register*/
	AT91_REG	 ST_RTMR; 	/* Real-time Mode Register*/
	AT91_REG	 ST_SR; 	/* Status Register*/
	AT91_REG	 ST_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 ST_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 ST_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 ST_RTAR; 	/* Real-time Alarm Register*/
	AT91_REG	 ST_CRTR; 	/* Current Real-time Register*/
	AT91_REG	 Reserved24[54]; 	/* */
	AT91_REG	 RTC_CR; 	/* Control Register*/
	AT91_REG	 RTC_MR; 	/* Mode Register*/
	AT91_REG	 RTC_TIMR; 	/* Time Register*/
	AT91_REG	 RTC_CALR; 	/* Calendar Register*/
	AT91_REG	 RTC_TIMALR; 	/* Time Alarm Register*/
	AT91_REG	 RTC_CALALR; 	/* Calendar Alarm Register*/
	AT91_REG	 RTC_SR; 	/* Status Register*/
	AT91_REG	 RTC_SCCR; 	/* Status Clear Command Register*/
	AT91_REG	 RTC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 RTC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 RTC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 RTC_VER; 	/* Valid Entry Register*/
	AT91_REG	 Reserved25[52]; 	/* */
	AT91_REG	 MC_RCR; 	/* MC Remap Control Register*/
	AT91_REG	 MC_ASR; 	/* MC Abort Status Register*/
	AT91_REG	 MC_AASR; 	/* MC Abort Address Status Register*/
	AT91_REG	 Reserved26[1]; 	/* */
	AT91_REG	 MC_PUIA[16]; 	/* MC Protection Unit Area*/
	AT91_REG	 MC_PUP; 	/* MC Protection Unit Peripherals*/
	AT91_REG	 MC_PUER; 	/* MC Protection Unit Enable Register*/
} AT91S_SYS, *AT91PS_SYS;


/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Memory Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_MC {
	AT91_REG	 MC_RCR; 	/* MC Remap Control Register*/
	AT91_REG	 MC_ASR; 	/* MC Abort Status Register*/
	AT91_REG	 MC_AASR; 	/* MC Abort Address Status Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 MC_PUIA[16]; 	/* MC Protection Unit Area*/
	AT91_REG	 MC_PUP; 	/* MC Protection Unit Peripherals*/
	AT91_REG	 MC_PUER; 	/* MC Protection Unit Enable Register*/
} AT91S_MC, *AT91PS_MC;

/* -------- MC_RCR : (MC Offset: 0x0) MC Remap Control Register -------- */
#define AT91C_MC_RCB          ((unsigned int) 0x1 <<  0) /* (MC) Remap Command Bit*/
/* -------- MC_ASR : (MC Offset: 0x4) MC Abort Status Register -------- */
#define AT91C_MC_UNDADD       ((unsigned int) 0x1 <<  0) /* (MC) Undefined Addess Abort Status*/
#define AT91C_MC_MISADD       ((unsigned int) 0x1 <<  1) /* (MC) Misaligned Addess Abort Status*/
#define AT91C_MC_MPU          ((unsigned int) 0x1 <<  2) /* (MC) Memory protection Unit Abort Status*/
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
/* -------- MC_PUIA : (MC Offset: 0x10) MC Protection Unit Area -------- */
#define AT91C_MC_PROT         ((unsigned int) 0x3 <<  0) /* (MC) Protection*/
#define 	AT91C_MC_PROT_PNAUNA               ((unsigned int) 0x0) /* (MC) Privilege: No Access, User: No Access*/
#define 	AT91C_MC_PROT_PRWUNA               ((unsigned int) 0x1) /* (MC) Privilege: Read/Write, User: No Access*/
#define 	AT91C_MC_PROT_PRWURO               ((unsigned int) 0x2) /* (MC) Privilege: Read/Write, User: Read Only*/
#define 	AT91C_MC_PROT_PRWURW               ((unsigned int) 0x3) /* (MC) Privilege: Read/Write, User: Read/Write*/
#define AT91C_MC_SIZE         ((unsigned int) 0xF <<  4) /* (MC) Internal Area Size*/
#define 	AT91C_MC_SIZE_1KB                  ((unsigned int) 0x0 <<  4) /* (MC) Area size 1KByte*/
#define 	AT91C_MC_SIZE_2KB                  ((unsigned int) 0x1 <<  4) /* (MC) Area size 2KByte*/
#define 	AT91C_MC_SIZE_4KB                  ((unsigned int) 0x2 <<  4) /* (MC) Area size 4KByte*/
#define 	AT91C_MC_SIZE_8KB                  ((unsigned int) 0x3 <<  4) /* (MC) Area size 8KByte*/
#define 	AT91C_MC_SIZE_16KB                 ((unsigned int) 0x4 <<  4) /* (MC) Area size 16KByte*/
#define 	AT91C_MC_SIZE_32KB                 ((unsigned int) 0x5 <<  4) /* (MC) Area size 32KByte*/
#define 	AT91C_MC_SIZE_64KB                 ((unsigned int) 0x6 <<  4) /* (MC) Area size 64KByte*/
#define 	AT91C_MC_SIZE_128KB                ((unsigned int) 0x7 <<  4) /* (MC) Area size 128KByte*/
#define 	AT91C_MC_SIZE_256KB                ((unsigned int) 0x8 <<  4) /* (MC) Area size 256KByte*/
#define 	AT91C_MC_SIZE_512KB                ((unsigned int) 0x9 <<  4) /* (MC) Area size 512KByte*/
#define 	AT91C_MC_SIZE_1MB                  ((unsigned int) 0xA <<  4) /* (MC) Area size 1MByte*/
#define 	AT91C_MC_SIZE_2MB                  ((unsigned int) 0xB <<  4) /* (MC) Area size 2MByte*/
#define 	AT91C_MC_SIZE_4MB                  ((unsigned int) 0xC <<  4) /* (MC) Area size 4MByte*/
#define 	AT91C_MC_SIZE_8MB                  ((unsigned int) 0xD <<  4) /* (MC) Area size 8MByte*/
#define 	AT91C_MC_SIZE_16MB                 ((unsigned int) 0xE <<  4) /* (MC) Area size 16MByte*/
#define 	AT91C_MC_SIZE_64MB                 ((unsigned int) 0xF <<  4) /* (MC) Area size 64MByte*/
#define AT91C_MC_BA           ((unsigned int) 0x3FFFF << 10) /* (MC) Internal Area Base Address*/
/* -------- MC_PUP : (MC Offset: 0x50) MC Protection Unit Peripheral -------- */
/* -------- MC_PUER : (MC Offset: 0x54) MC Protection Unit Area -------- */
#define AT91C_MC_PUEB         ((unsigned int) 0x1 <<  0) /* (MC) Protection Unit enable Bit*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface*/
/* ******************************************************************************/
typedef struct _AT91S_RTC {
	AT91_REG	 RTC_CR; 	/* Control Register*/
	AT91_REG	 RTC_MR; 	/* Mode Register*/
	AT91_REG	 RTC_TIMR; 	/* Time Register*/
	AT91_REG	 RTC_CALR; 	/* Calendar Register*/
	AT91_REG	 RTC_TIMALR; 	/* Time Alarm Register*/
	AT91_REG	 RTC_CALALR; 	/* Calendar Alarm Register*/
	AT91_REG	 RTC_SR; 	/* Status Register*/
	AT91_REG	 RTC_SCCR; 	/* Status Clear Command Register*/
	AT91_REG	 RTC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 RTC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 RTC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 RTC_VER; 	/* Valid Entry Register*/
} AT91S_RTC, *AT91PS_RTC;

/* -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- */
#define AT91C_RTC_UPDTIM      ((unsigned int) 0x1 <<  0) /* (RTC) Update Request Time Register*/
#define AT91C_RTC_UPDCAL      ((unsigned int) 0x1 <<  1) /* (RTC) Update Request Calendar Register*/
#define AT91C_RTC_TIMEVSEL    ((unsigned int) 0x3 <<  8) /* (RTC) Time Event Selection*/
#define 	AT91C_RTC_TIMEVSEL_MINUTE               ((unsigned int) 0x0 <<  8) /* (RTC) Minute change.*/
#define 	AT91C_RTC_TIMEVSEL_HOUR                 ((unsigned int) 0x1 <<  8) /* (RTC) Hour change.*/
#define 	AT91C_RTC_TIMEVSEL_DAY24                ((unsigned int) 0x2 <<  8) /* (RTC) Every day at midnight.*/
#define 	AT91C_RTC_TIMEVSEL_DAY12                ((unsigned int) 0x3 <<  8) /* (RTC) Every day at noon.*/
#define AT91C_RTC_CALEVSEL    ((unsigned int) 0x3 << 16) /* (RTC) Calendar Event Selection*/
#define 	AT91C_RTC_CALEVSEL_WEEK                 ((unsigned int) 0x0 << 16) /* (RTC) Week change (every Monday at time 00:00:00).*/
#define 	AT91C_RTC_CALEVSEL_MONTH                ((unsigned int) 0x1 << 16) /* (RTC) Month change (every 01 of each month at time 00:00:00).*/
#define 	AT91C_RTC_CALEVSEL_YEAR                 ((unsigned int) 0x2 << 16) /* (RTC) Year change (every January 1 at time 00:00:00).*/
/* -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- */
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
/* -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- */
#define AT91C_RTC_SECEN       ((unsigned int) 0x1 <<  7) /* (RTC) Second Alarm Enable*/
#define AT91C_RTC_MINEN       ((unsigned int) 0x1 << 15) /* (RTC) Minute Alarm*/
#define AT91C_RTC_HOUREN      ((unsigned int) 0x1 << 23) /* (RTC) Current Hour*/
/* -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- */
#define AT91C_RTC_MONTHEN     ((unsigned int) 0x1 << 23) /* (RTC) Month Alarm Enable*/
#define AT91C_RTC_DATEEN      ((unsigned int) 0x1 << 31) /* (RTC) Date Alarm Enable*/
/* -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- */
#define AT91C_RTC_ACKUPD      ((unsigned int) 0x1 <<  0) /* (RTC) Acknowledge for Update*/
#define AT91C_RTC_ALARM       ((unsigned int) 0x1 <<  1) /* (RTC) Alarm Flag*/
#define AT91C_RTC_SECEV       ((unsigned int) 0x1 <<  2) /* (RTC) Second Event*/
#define AT91C_RTC_TIMEV       ((unsigned int) 0x1 <<  3) /* (RTC) Time Event*/
#define AT91C_RTC_CALEV       ((unsigned int) 0x1 <<  4) /* (RTC) Calendar event*/
/* -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- */
/* -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- */
/* -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- */
/* -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- */
/* -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- */
#define AT91C_RTC_NVTIM       ((unsigned int) 0x1 <<  0) /* (RTC) Non valid Time*/
#define AT91C_RTC_NVCAL       ((unsigned int) 0x1 <<  1) /* (RTC) Non valid Calendar*/
#define AT91C_RTC_NVTIMALR    ((unsigned int) 0x1 <<  2) /* (RTC) Non valid time Alarm*/
#define AT91C_RTC_NVCALALR    ((unsigned int) 0x1 <<  3) /* (RTC) Nonvalid Calendar Alarm*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR System Timer Interface*/
/* ******************************************************************************/
typedef struct _AT91S_ST {
	AT91_REG	 ST_CR; 	/* Control Register*/
	AT91_REG	 ST_PIMR; 	/* Period Interval Mode Register*/
	AT91_REG	 ST_WDMR; 	/* Watchdog Mode Register*/
	AT91_REG	 ST_RTMR; 	/* Real-time Mode Register*/
	AT91_REG	 ST_SR; 	/* Status Register*/
	AT91_REG	 ST_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 ST_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 ST_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 ST_RTAR; 	/* Real-time Alarm Register*/
	AT91_REG	 ST_CRTR; 	/* Current Real-time Register*/
} AT91S_ST, *AT91PS_ST;

/* -------- ST_CR : (ST Offset: 0x0) System Timer Control Register -------- */
#define AT91C_ST_WDRST        ((unsigned int) 0x1 <<  0) /* (ST) Watchdog Timer Restart*/
/* -------- ST_PIMR : (ST Offset: 0x4) System Timer Period Interval Mode Register -------- */
#define AT91C_ST_PIV          ((unsigned int) 0xFFFF <<  0) /* (ST) Watchdog Timer Restart*/
/* -------- ST_WDMR : (ST Offset: 0x8) System Timer Watchdog Mode Register -------- */
#define AT91C_ST_WDV          ((unsigned int) 0xFFFF <<  0) /* (ST) Watchdog Timer Restart*/
#define AT91C_ST_RSTEN        ((unsigned int) 0x1 << 16) /* (ST) Reset Enable*/
#define AT91C_ST_EXTEN        ((unsigned int) 0x1 << 17) /* (ST) External Signal Assertion Enable*/
/* -------- ST_RTMR : (ST Offset: 0xc) System Timer Real-time Mode Register -------- */
#define AT91C_ST_RTPRES       ((unsigned int) 0xFFFF <<  0) /* (ST) Real-time Timer Prescaler Value*/
/* -------- ST_SR : (ST Offset: 0x10) System Timer Status Register -------- */
#define AT91C_ST_PITS         ((unsigned int) 0x1 <<  0) /* (ST) Period Interval Timer Interrupt*/
#define AT91C_ST_WDOVF        ((unsigned int) 0x1 <<  1) /* (ST) Watchdog Overflow*/
#define AT91C_ST_RTTINC       ((unsigned int) 0x1 <<  2) /* (ST) Real-time Timer Increment*/
#define AT91C_ST_ALMS         ((unsigned int) 0x1 <<  3) /* (ST) Alarm Status*/
/* -------- ST_IER : (ST Offset: 0x14) System Timer Interrupt Enable Register -------- */
/* -------- ST_IDR : (ST Offset: 0x18) System Timer Interrupt Disable Register -------- */
/* -------- ST_IMR : (ST Offset: 0x1c) System Timer Interrupt Mask Register -------- */
/* -------- ST_RTAR : (ST Offset: 0x20) System Timer Real-time Alarm Register -------- */
#define AT91C_ST_ALMV         ((unsigned int) 0xFFFFF <<  0) /* (ST) Alarm Value Value*/
/* -------- ST_CRTR : (ST Offset: 0x24) System Timer Current Real-time Register -------- */
#define AT91C_ST_CRTV         ((unsigned int) 0xFFFFF <<  0) /* (ST) Current Real-time Value*/

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
	AT91_REG	 Reserved1[5]; 	/* */
	AT91_REG	 PMC_MCKR; 	/* Master Clock Register*/
	AT91_REG	 Reserved2[3]; 	/* */
	AT91_REG	 PMC_PCKR[8]; 	/* Programmable Clock Register*/
	AT91_REG	 PMC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PMC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PMC_SR; 	/* Status Register*/
	AT91_REG	 PMC_IMR; 	/* Interrupt Mask Register*/
} AT91S_PMC, *AT91PS_PMC;

/* -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) /* (PMC) Processor Clock*/
#define AT91C_PMC_UDP         ((unsigned int) 0x1 <<  1) /* (PMC) USB Device Port Clock*/
#define AT91C_PMC_MCKUDP      ((unsigned int) 0x1 <<  2) /* (PMC) USB Device Port Master Clock Automatic Disable on Suspend*/
#define AT91C_PMC_UHP         ((unsigned int) 0x1 <<  4) /* (PMC) USB Host Port Clock*/
#define AT91C_PMC_PCK0        ((unsigned int) 0x1 <<  8) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK1        ((unsigned int) 0x1 <<  9) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK2        ((unsigned int) 0x1 << 10) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK3        ((unsigned int) 0x1 << 11) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK4        ((unsigned int) 0x1 << 12) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK5        ((unsigned int) 0x1 << 13) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK6        ((unsigned int) 0x1 << 14) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK7        ((unsigned int) 0x1 << 15) /* (PMC) Programmable Clock Output*/
/* -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- */
#define AT91C_PMC_CSS         ((unsigned int) 0x3 <<  0) /* (PMC) Programmable Clock Selection*/
#define 	AT91C_PMC_CSS_SLOW_CLK             ((unsigned int) 0x0) /* (PMC) Slow Clock is selected*/
#define 	AT91C_PMC_CSS_MAIN_CLK             ((unsigned int) 0x1) /* (PMC) Main Clock is selected*/
#define 	AT91C_PMC_CSS_PLLA_CLK             ((unsigned int) 0x2) /* (PMC) Clock from PLL A is selected*/
#define 	AT91C_PMC_CSS_PLLB_CLK             ((unsigned int) 0x3) /* (PMC) Clock from PLL B is selected*/
#define AT91C_PMC_PRES        ((unsigned int) 0x7 <<  2) /* (PMC) Programmable Clock Prescaler*/
#define 	AT91C_PMC_PRES_CLK                  ((unsigned int) 0x0 <<  2) /* (PMC) Selected clock*/
#define 	AT91C_PMC_PRES_CLK_2                ((unsigned int) 0x1 <<  2) /* (PMC) Selected clock divided by 2*/
#define 	AT91C_PMC_PRES_CLK_4                ((unsigned int) 0x2 <<  2) /* (PMC) Selected clock divided by 4*/
#define 	AT91C_PMC_PRES_CLK_8                ((unsigned int) 0x3 <<  2) /* (PMC) Selected clock divided by 8*/
#define 	AT91C_PMC_PRES_CLK_16               ((unsigned int) 0x4 <<  2) /* (PMC) Selected clock divided by 16*/
#define 	AT91C_PMC_PRES_CLK_32               ((unsigned int) 0x5 <<  2) /* (PMC) Selected clock divided by 32*/
#define 	AT91C_PMC_PRES_CLK_64               ((unsigned int) 0x6 <<  2) /* (PMC) Selected clock divided by 64*/
#define AT91C_PMC_MDIV        ((unsigned int) 0x3 <<  8) /* (PMC) Master Clock Division*/
#define 	AT91C_PMC_MDIV_1                    ((unsigned int) 0x0 <<  8) /* (PMC) The master clock and the processor clock are the same*/
#define 	AT91C_PMC_MDIV_2                    ((unsigned int) 0x1 <<  8) /* (PMC) The processor clock is twice as fast as the master clock*/
#define 	AT91C_PMC_MDIV_3                    ((unsigned int) 0x2 <<  8) /* (PMC) The processor clock is three times faster than the master clock*/
#define 	AT91C_PMC_MDIV_4                    ((unsigned int) 0x3 <<  8) /* (PMC) The processor clock is four times faster than the master clock*/
/* -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
#define AT91C_PMC_MOSCS       ((unsigned int) 0x1 <<  0) /* (PMC) MOSC Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCKA       ((unsigned int) 0x1 <<  1) /* (PMC) PLL A Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCKB       ((unsigned int) 0x1 <<  2) /* (PMC) PLL B Status/Enable/Disable/Mask*/
#define AT91C_PMC_MCKRDY      ((unsigned int) 0x1 <<  3) /* (PMC) MCK_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK0RDY     ((unsigned int) 0x1 <<  8) /* (PMC) PCK0_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK1RDY     ((unsigned int) 0x1 <<  9) /* (PMC) PCK1_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK2RDY     ((unsigned int) 0x1 << 10) /* (PMC) PCK2_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK3RDY     ((unsigned int) 0x1 << 11) /* (PMC) PCK3_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK4RDY     ((unsigned int) 0x1 << 12) /* (PMC) PCK4_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK5RDY     ((unsigned int) 0x1 << 13) /* (PMC) PCK5_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK6RDY     ((unsigned int) 0x1 << 14) /* (PMC) PCK6_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK7RDY     ((unsigned int) 0x1 << 15) /* (PMC) PCK7_RDY Status/Enable/Disable/Mask*/
/* -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* ******************************************************************************/
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 CKGR_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 CKGR_PLLAR; 	/* PLL A Register*/
	AT91_REG	 CKGR_PLLBR; 	/* PLL B Register*/
} AT91S_CKGR, *AT91PS_CKGR;

/* -------- CKGR_MOR : (CKGR Offset: 0x0) Main Oscillator Register -------- */
#define AT91C_CKGR_MOSCEN     ((unsigned int) 0x1 <<  0) /* (CKGR) Main Oscillator Enable*/
#define AT91C_CKGR_OSCTEST    ((unsigned int) 0x1 <<  1) /* (CKGR) Oscillator Test*/
#define AT91C_CKGR_OSCOUNT    ((unsigned int) 0xFF <<  8) /* (CKGR) Main Oscillator Start-up Time*/
/* -------- CKGR_MCFR : (CKGR Offset: 0x4) Main Clock Frequency Register -------- */
#define AT91C_CKGR_MAINF      ((unsigned int) 0xFFFF <<  0) /* (CKGR) Main Clock Frequency*/
#define AT91C_CKGR_MAINRDY    ((unsigned int) 0x1 << 16) /* (CKGR) Main Clock Ready*/
/* -------- CKGR_PLLAR : (CKGR Offset: 0x8) PLL A Register -------- */
#define AT91C_CKGR_DIVA       ((unsigned int) 0xFF <<  0) /* (CKGR) Divider Selected*/
#define 	AT91C_CKGR_DIVA_0                    ((unsigned int) 0x0) /* (CKGR) Divider output is 0*/
#define 	AT91C_CKGR_DIVA_BYPASS               ((unsigned int) 0x1) /* (CKGR) Divider is bypassed*/
#define AT91C_CKGR_PLLACOUNT  ((unsigned int) 0x3F <<  8) /* (CKGR) PLL A Counter*/
#define AT91C_CKGR_OUTA       ((unsigned int) 0x3 << 14) /* (CKGR) PLL A Output Frequency Range*/
#define 	AT91C_CKGR_OUTA_0                    ((unsigned int) 0x0 << 14) /* (CKGR) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_1                    ((unsigned int) 0x1 << 14) /* (CKGR) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_2                    ((unsigned int) 0x2 << 14) /* (CKGR) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_3                    ((unsigned int) 0x3 << 14) /* (CKGR) Please refer to the PLLA datasheet*/
#define AT91C_CKGR_MULA       ((unsigned int) 0x7FF << 16) /* (CKGR) PLL A Multiplier*/
#define AT91C_CKGR_SRCA       ((unsigned int) 0x1 << 29) /* (CKGR) PLL A Source*/
/* -------- CKGR_PLLBR : (CKGR Offset: 0xc) PLL B Register -------- */
#define AT91C_CKGR_DIVB       ((unsigned int) 0xFF <<  0) /* (CKGR) Divider Selected*/
#define 	AT91C_CKGR_DIVB_0                    ((unsigned int) 0x0) /* (CKGR) Divider output is 0*/
#define 	AT91C_CKGR_DIVB_BYPASS               ((unsigned int) 0x1) /* (CKGR) Divider is bypassed*/
#define AT91C_CKGR_PLLBCOUNT  ((unsigned int) 0x3F <<  8) /* (CKGR) PLL B Counter*/
#define AT91C_CKGR_OUTB       ((unsigned int) 0x3 << 14) /* (CKGR) PLL B Output Frequency Range*/
#define 	AT91C_CKGR_OUTB_0                    ((unsigned int) 0x0 << 14) /* (CKGR) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_1                    ((unsigned int) 0x1 << 14) /* (CKGR) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_2                    ((unsigned int) 0x2 << 14) /* (CKGR) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_3                    ((unsigned int) 0x3 << 14) /* (CKGR) Please refer to the PLLB datasheet*/
#define AT91C_CKGR_MULB       ((unsigned int) 0x7FF << 16) /* (CKGR) PLL B Multiplier*/
#define AT91C_CKGR_USB_96M    ((unsigned int) 0x1 << 28) /* (CKGR) Divider for USB Ports*/
#define AT91C_CKGR_USB_PLL    ((unsigned int) 0x1 << 29) /* (CKGR) PLL Use*/

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
	AT91_REG	 PIO_PPUSR; 	/* Pad Pull-up Status Register*/
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
	AT91_REG	 DBGU_C1R; 	/* Chip ID1 Register*/
	AT91_REG	 DBGU_C2R; 	/* Chip ID2 Register*/
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
/*              SOFTWARE API DEFINITION  FOR Peripheral Data Controller*/
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
#define 	AT91C_AIC_SRCTYPE_INT_LEVEL_SENSITIVE  ((unsigned int) 0x0 <<  5) /* (AIC) Internal Sources Code Label Level Sensitive*/
#define 	AT91C_AIC_SRCTYPE_INT_EDGE_TRIGGERED   ((unsigned int) 0x1 <<  5) /* (AIC) Internal Sources Code Label Edge triggered*/
#define 	AT91C_AIC_SRCTYPE_EXT_HIGH_LEVEL       ((unsigned int) 0x2 <<  5) /* (AIC) External Sources Code Label High-level Sensitive*/
#define 	AT91C_AIC_SRCTYPE_EXT_POSITIVE_EDGE    ((unsigned int) 0x3 <<  5) /* (AIC) External Sources Code Label Positive Edge triggered*/
/* -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- */
#define AT91C_AIC_NFIQ        ((unsigned int) 0x1 <<  0) /* (AIC) NFIQ Status*/
#define AT91C_AIC_NIRQ        ((unsigned int) 0x1 <<  1) /* (AIC) NIRQ Status*/
/* -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- */
#define AT91C_AIC_DCR_PROT    ((unsigned int) 0x1 <<  0) /* (AIC) Protection Mode*/
#define AT91C_AIC_DCR_GMSK    ((unsigned int) 0x1 <<  1) /* (AIC) General Mask*/

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
/* -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- */
#define AT91C_SPI_MSTR        ((unsigned int) 0x1 <<  0) /* (SPI) Master/Slave Mode*/
#define AT91C_SPI_PS          ((unsigned int) 0x1 <<  1) /* (SPI) Peripheral Select*/
#define 	AT91C_SPI_PS_FIXED                ((unsigned int) 0x0 <<  1) /* (SPI) Fixed Peripheral Select*/
#define 	AT91C_SPI_PS_VARIABLE             ((unsigned int) 0x1 <<  1) /* (SPI) Variable Peripheral Select*/
#define AT91C_SPI_PCSDEC      ((unsigned int) 0x1 <<  2) /* (SPI) Chip Select Decode*/
#define AT91C_SPI_DIV32       ((unsigned int) 0x1 <<  3) /* (SPI) Clock Selection*/
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
#define AT91C_SPI_SPENDRX     ((unsigned int) 0x1 <<  4) /* (SPI) End of Receiver Transfer*/
#define AT91C_SPI_SPENDTX     ((unsigned int) 0x1 <<  5) /* (SPI) End of Receiver Transfer*/
#define AT91C_SPI_RXBUFF      ((unsigned int) 0x1 <<  6) /* (SPI) RXBUFF Interrupt*/
#define AT91C_SPI_TXBUFE      ((unsigned int) 0x1 <<  7) /* (SPI) TXBUFE Interrupt*/
#define AT91C_SPI_SPIENS      ((unsigned int) 0x1 << 16) /* (SPI) Enable Status*/
/* -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- */
/* -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- */
/* -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- */
/* -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- */
#define AT91C_SPI_CPOL        ((unsigned int) 0x1 <<  0) /* (SPI) Clock Polarity*/
#define AT91C_SPI_NCPHA       ((unsigned int) 0x1 <<  1) /* (SPI) Clock Phase*/
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
#define AT91C_SPI_DLYBS       ((unsigned int) 0xFF << 16) /* (SPI) Serial Clock Baud Rate*/
#define AT91C_SPI_DLYBCT      ((unsigned int) 0xFF << 24) /* (SPI) Delay Between Consecutive Transfers*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_SSC {
	AT91_REG	 SSC_CR; 	/* Control Register*/
	AT91_REG	 SSC_CMR; 	/* Clock Mode Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 SSC_RCMR; 	/* Receive Clock ModeRegister*/
	AT91_REG	 SSC_RFMR; 	/* Receive Frame Mode Register*/
	AT91_REG	 SSC_TCMR; 	/* Transmit Clock Mode Register*/
	AT91_REG	 SSC_TFMR; 	/* Transmit Frame Mode Register*/
	AT91_REG	 SSC_RHR; 	/* Receive Holding Register*/
	AT91_REG	 SSC_THR; 	/* Transmit Holding Register*/
	AT91_REG	 Reserved1[2]; 	/* */
	AT91_REG	 SSC_RSHR; 	/* Receive Sync Holding Register*/
	AT91_REG	 SSC_TSHR; 	/* Transmit Sync Holding Register*/
	AT91_REG	 SSC_RC0R; 	/* Receive Compare 0 Register*/
	AT91_REG	 SSC_RC1R; 	/* Receive Compare 1 Register*/
	AT91_REG	 SSC_SR; 	/* Status Register*/
	AT91_REG	 SSC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 SSC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 SSC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 Reserved2[44]; 	/* */
	AT91_REG	 SSC_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 SSC_RCR; 	/* Receive Counter Register*/
	AT91_REG	 SSC_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 SSC_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 SSC_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 SSC_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 SSC_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 SSC_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 SSC_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 SSC_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_SSC, *AT91PS_SSC;

/* -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- */
#define AT91C_SSC_RXEN        ((unsigned int) 0x1 <<  0) /* (SSC) Receive Enable*/
#define AT91C_SSC_RXDIS       ((unsigned int) 0x1 <<  1) /* (SSC) Receive Disable*/
#define AT91C_SSC_TXEN        ((unsigned int) 0x1 <<  8) /* (SSC) Transmit Enable*/
#define AT91C_SSC_TXDIS       ((unsigned int) 0x1 <<  9) /* (SSC) Transmit Disable*/
#define AT91C_SSC_SWRST       ((unsigned int) 0x1 << 15) /* (SSC) Software Reset*/
/* -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- */
#define AT91C_SSC_CKS         ((unsigned int) 0x3 <<  0) /* (SSC) Receive/Transmit Clock Selection*/
#define 	AT91C_SSC_CKS_DIV                  ((unsigned int) 0x0) /* (SSC) Divided Clock*/
#define 	AT91C_SSC_CKS_TK                   ((unsigned int) 0x1) /* (SSC) TK Clock signal*/
#define 	AT91C_SSC_CKS_RK                   ((unsigned int) 0x2) /* (SSC) RK pin*/
#define AT91C_SSC_CKO         ((unsigned int) 0x7 <<  2) /* (SSC) Receive/Transmit Clock Output Mode Selection*/
#define 	AT91C_SSC_CKO_NONE                 ((unsigned int) 0x0 <<  2) /* (SSC) Receive/Transmit Clock Output Mode: None RK pin: Input-only*/
#define 	AT91C_SSC_CKO_CONTINOUS            ((unsigned int) 0x1 <<  2) /* (SSC) Continuous Receive/Transmit Clock RK pin: Output*/
#define 	AT91C_SSC_CKO_DATA_TX              ((unsigned int) 0x2 <<  2) /* (SSC) Receive/Transmit Clock only during data transfers RK pin: Output*/
#define AT91C_SSC_CKI         ((unsigned int) 0x1 <<  5) /* (SSC) Receive/Transmit Clock Inversion*/
#define AT91C_SSC_CKG         ((unsigned int) 0x3 <<  6) /* (SSC) Receive/Transmit Clock Gating Selection*/
#define 	AT91C_SSC_CKG_NONE                 ((unsigned int) 0x0 <<  6) /* (SSC) Receive/Transmit Clock Gating: None, continuous clock*/
#define 	AT91C_SSC_CKG_LOW                  ((unsigned int) 0x1 <<  6) /* (SSC) Receive/Transmit Clock enabled only if RF Low*/
#define 	AT91C_SSC_CKG_HIGH                 ((unsigned int) 0x2 <<  6) /* (SSC) Receive/Transmit Clock enabled only if RF High*/
#define AT91C_SSC_START       ((unsigned int) 0xF <<  8) /* (SSC) Receive/Transmit Start Selection*/
#define 	AT91C_SSC_START_CONTINOUS            ((unsigned int) 0x0 <<  8) /* (SSC) Continuous, as soon as the receiver is enabled, and immediately after the end of transfer of the previous data.*/
#define 	AT91C_SSC_START_TX                   ((unsigned int) 0x1 <<  8) /* (SSC) Transmit/Receive start*/
#define 	AT91C_SSC_START_LOW_RF               ((unsigned int) 0x2 <<  8) /* (SSC) Detection of a low level on RF input*/
#define 	AT91C_SSC_START_HIGH_RF              ((unsigned int) 0x3 <<  8) /* (SSC) Detection of a high level on RF input*/
#define 	AT91C_SSC_START_FALL_RF              ((unsigned int) 0x4 <<  8) /* (SSC) Detection of a falling edge on RF input*/
#define 	AT91C_SSC_START_RISE_RF              ((unsigned int) 0x5 <<  8) /* (SSC) Detection of a rising edge on RF input*/
#define 	AT91C_SSC_START_LEVEL_RF             ((unsigned int) 0x6 <<  8) /* (SSC) Detection of any level change on RF input*/
#define 	AT91C_SSC_START_EDGE_RF              ((unsigned int) 0x7 <<  8) /* (SSC) Detection of any edge on RF input*/
#define 	AT91C_SSC_START_0                    ((unsigned int) 0x8 <<  8) /* (SSC) Compare 0*/
#define AT91C_SSC_STOP        ((unsigned int) 0x1 << 12) /* (SSC) Receive Stop Selection*/
#define AT91C_SSC_STTOUT      ((unsigned int) 0x1 << 15) /* (SSC) Receive/Transmit Start Output Selection*/
#define AT91C_SSC_STTDLY      ((unsigned int) 0xFF << 16) /* (SSC) Receive/Transmit Start Delay*/
#define AT91C_SSC_PERIOD      ((unsigned int) 0xFF << 24) /* (SSC) Receive/Transmit Period Divider Selection*/
/* -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- */
#define AT91C_SSC_DATLEN      ((unsigned int) 0x1F <<  0) /* (SSC) Data Length*/
#define AT91C_SSC_LOOP        ((unsigned int) 0x1 <<  5) /* (SSC) Loop Mode*/
#define AT91C_SSC_MSBF        ((unsigned int) 0x1 <<  7) /* (SSC) Most Significant Bit First*/
#define AT91C_SSC_DATNB       ((unsigned int) 0xF <<  8) /* (SSC) Data Number per Frame*/
#define AT91C_SSC_FSLEN       ((unsigned int) 0xF << 16) /* (SSC) Receive/Transmit Frame Sync length*/
#define AT91C_SSC_FSOS        ((unsigned int) 0x7 << 20) /* (SSC) Receive/Transmit Frame Sync Output Selection*/
#define 	AT91C_SSC_FSOS_NONE                 ((unsigned int) 0x0 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: None RK pin Input-only*/
#define 	AT91C_SSC_FSOS_NEGATIVE             ((unsigned int) 0x1 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: Negative Pulse*/
#define 	AT91C_SSC_FSOS_POSITIVE             ((unsigned int) 0x2 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: Positive Pulse*/
#define 	AT91C_SSC_FSOS_LOW                  ((unsigned int) 0x3 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: Driver Low during data transfer*/
#define 	AT91C_SSC_FSOS_HIGH                 ((unsigned int) 0x4 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: Driver High during data transfer*/
#define 	AT91C_SSC_FSOS_TOGGLE               ((unsigned int) 0x5 << 20) /* (SSC) Selected Receive/Transmit Frame Sync Signal: Toggling at each start of data transfer*/
#define AT91C_SSC_FSEDGE      ((unsigned int) 0x1 << 24) /* (SSC) Frame Sync Edge Detection*/
/* -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- */
/* -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- */
#define AT91C_SSC_DATDEF      ((unsigned int) 0x1 <<  5) /* (SSC) Data Default Value*/
#define AT91C_SSC_FSDEN       ((unsigned int) 0x1 << 23) /* (SSC) Frame Sync Data Enable*/
/* -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- */
#define AT91C_SSC_TXRDY       ((unsigned int) 0x1 <<  0) /* (SSC) Transmit Ready*/
#define AT91C_SSC_TXEMPTY     ((unsigned int) 0x1 <<  1) /* (SSC) Transmit Empty*/
#define AT91C_SSC_ENDTX       ((unsigned int) 0x1 <<  2) /* (SSC) End Of Transmission*/
#define AT91C_SSC_TXBUFE      ((unsigned int) 0x1 <<  3) /* (SSC) Transmit Buffer Empty*/
#define AT91C_SSC_RXRDY       ((unsigned int) 0x1 <<  4) /* (SSC) Receive Ready*/
#define AT91C_SSC_OVRUN       ((unsigned int) 0x1 <<  5) /* (SSC) Receive Overrun*/
#define AT91C_SSC_ENDRX       ((unsigned int) 0x1 <<  6) /* (SSC) End of Reception*/
#define AT91C_SSC_RXBUFF      ((unsigned int) 0x1 <<  7) /* (SSC) Receive Buffer Full*/
#define AT91C_SSC_CP0         ((unsigned int) 0x1 <<  8) /* (SSC) Compare 0*/
#define AT91C_SSC_CP1         ((unsigned int) 0x1 <<  9) /* (SSC) Compare 1*/
#define AT91C_SSC_TXSYN       ((unsigned int) 0x1 << 10) /* (SSC) Transmit Sync*/
#define AT91C_SSC_RXSYN       ((unsigned int) 0x1 << 11) /* (SSC) Receive Sync*/
#define AT91C_SSC_TXENA       ((unsigned int) 0x1 << 16) /* (SSC) Transmit Enable*/
#define AT91C_SSC_RXENA       ((unsigned int) 0x1 << 17) /* (SSC) Receive Enable*/
/* -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- */
/* -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- */
/* -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- */

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
	AT91_REG	 US_XXR; 	/* XON_XOFF Register*/
	AT91_REG	 US_IF; 	/* IRDA_FILTER Register*/
	AT91_REG	 Reserved1[44]; 	/* */
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
#define AT91C_US_RSTSTA       ((unsigned int) 0x1 <<  8) /* (USART) Reset Status Bits*/
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
} AT91S_TWI, *AT91PS_TWI;

/* -------- TWI_CR : (TWI Offset: 0x0) TWI Control Register -------- */
#define AT91C_TWI_START       ((unsigned int) 0x1 <<  0) /* (TWI) Send a START Condition*/
#define AT91C_TWI_STOP        ((unsigned int) 0x1 <<  1) /* (TWI) Send a STOP Condition*/
#define AT91C_TWI_MSEN        ((unsigned int) 0x1 <<  2) /* (TWI) TWI Master Transfer Enabled*/
#define AT91C_TWI_MSDIS       ((unsigned int) 0x1 <<  3) /* (TWI) TWI Master Transfer Disabled*/
#define AT91C_TWI_SVEN        ((unsigned int) 0x1 <<  4) /* (TWI) TWI Slave Transfer Enabled*/
#define AT91C_TWI_SVDIS       ((unsigned int) 0x1 <<  5) /* (TWI) TWI Slave Transfer Disabled*/
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
#define AT91C_TWI_SADR        ((unsigned int) 0x7F << 16) /* (TWI) Slave Device Address*/
/* -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- */
#define AT91C_TWI_CLDIV       ((unsigned int) 0xFF <<  0) /* (TWI) Clock Low Divider*/
#define AT91C_TWI_CHDIV       ((unsigned int) 0xFF <<  8) /* (TWI) Clock High Divider*/
#define AT91C_TWI_CKDIV       ((unsigned int) 0x7 << 16) /* (TWI) Clock Divider*/
/* -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- */
#define AT91C_TWI_TXCOMP      ((unsigned int) 0x1 <<  0) /* (TWI) Transmission Completed*/
#define AT91C_TWI_RXRDY       ((unsigned int) 0x1 <<  1) /* (TWI) Receive holding register ReaDY*/
#define AT91C_TWI_TXRDY       ((unsigned int) 0x1 <<  2) /* (TWI) Transmit holding register ReaDY*/
#define AT91C_TWI_SVREAD      ((unsigned int) 0x1 <<  3) /* (TWI) Slave Read*/
#define AT91C_TWI_SVACC       ((unsigned int) 0x1 <<  4) /* (TWI) Slave Access*/
#define AT91C_TWI_GCACC       ((unsigned int) 0x1 <<  5) /* (TWI) General Call Access*/
#define AT91C_TWI_OVRE        ((unsigned int) 0x1 <<  6) /* (TWI) Overrun Error*/
#define AT91C_TWI_UNRE        ((unsigned int) 0x1 <<  7) /* (TWI) Underrun Error*/
#define AT91C_TWI_NACK        ((unsigned int) 0x1 <<  8) /* (TWI) Not Acknowledged*/
#define AT91C_TWI_ARBLST      ((unsigned int) 0x1 <<  9) /* (TWI) Arbitration Lost*/
/* -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- */
/* -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- */
/* -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Multimedia Card Interface*/
/* ******************************************************************************/
typedef struct _AT91S_MCI {
	AT91_REG	 MCI_CR; 	/* MCI Control Register*/
	AT91_REG	 MCI_MR; 	/* MCI Mode Register*/
	AT91_REG	 MCI_DTOR; 	/* MCI Data Timeout Register*/
	AT91_REG	 MCI_SDCR; 	/* MCI SD Card Register*/
	AT91_REG	 MCI_ARGR; 	/* MCI Argument Register*/
	AT91_REG	 MCI_CMDR; 	/* MCI Command Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 MCI_RSPR[4]; 	/* MCI Response Register*/
	AT91_REG	 MCI_RDR; 	/* MCI Receive Data Register*/
	AT91_REG	 MCI_TDR; 	/* MCI Transmit Data Register*/
	AT91_REG	 Reserved1[2]; 	/* */
	AT91_REG	 MCI_SR; 	/* MCI Status Register*/
	AT91_REG	 MCI_IER; 	/* MCI Interrupt Enable Register*/
	AT91_REG	 MCI_IDR; 	/* MCI Interrupt Disable Register*/
	AT91_REG	 MCI_IMR; 	/* MCI Interrupt Mask Register*/
	AT91_REG	 Reserved2[44]; 	/* */
	AT91_REG	 MCI_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 MCI_RCR; 	/* Receive Counter Register*/
	AT91_REG	 MCI_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 MCI_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 MCI_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 MCI_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 MCI_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 MCI_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 MCI_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 MCI_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_MCI, *AT91PS_MCI;

/* -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- */
#define AT91C_MCI_MCIEN       ((unsigned int) 0x1 <<  0) /* (MCI) Multimedia Interface Enable*/
#define AT91C_MCI_MCIDIS      ((unsigned int) 0x1 <<  1) /* (MCI) Multimedia Interface Disable*/
#define AT91C_MCI_PWSEN       ((unsigned int) 0x1 <<  2) /* (MCI) Power Save Mode Enable*/
#define AT91C_MCI_PWSDIS      ((unsigned int) 0x1 <<  3) /* (MCI) Power Save Mode Disable*/
#define AT91C_MCI_SWRST       ((unsigned int) 0x1 <<  7) /* (MCI) MCI Software reset*/
/* -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- */
#define AT91C_MCI_CLKDIV      ((unsigned int) 0xFF <<  0) /* (MCI) Clock Divider*/
#define AT91C_MCI_PWSDIV      ((unsigned int) 0x7 <<  8) /* (MCI) Power Saving Divider*/
#define AT91C_MCI_PDCPADV     ((unsigned int) 0x1 << 14) /* (MCI) PDC Padding Value*/
#define AT91C_MCI_PDCMODE     ((unsigned int) 0x1 << 15) /* (MCI) PDC Oriented Mode*/
#define AT91C_MCI_BLKLEN      ((unsigned int) 0xFFF << 18) /* (MCI) Data Block Length*/
/* -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- */
#define AT91C_MCI_DTOCYC      ((unsigned int) 0xF <<  0) /* (MCI) Data Timeout Cycle Number*/
#define AT91C_MCI_DTOMUL      ((unsigned int) 0x7 <<  4) /* (MCI) Data Timeout Multiplier*/
#define 	AT91C_MCI_DTOMUL_1                    ((unsigned int) 0x0 <<  4) /* (MCI) DTOCYC x 1*/
#define 	AT91C_MCI_DTOMUL_16                   ((unsigned int) 0x1 <<  4) /* (MCI) DTOCYC x 16*/
#define 	AT91C_MCI_DTOMUL_128                  ((unsigned int) 0x2 <<  4) /* (MCI) DTOCYC x 128*/
#define 	AT91C_MCI_DTOMUL_256                  ((unsigned int) 0x3 <<  4) /* (MCI) DTOCYC x 256*/
#define 	AT91C_MCI_DTOMUL_1024                 ((unsigned int) 0x4 <<  4) /* (MCI) DTOCYC x 1024*/
#define 	AT91C_MCI_DTOMUL_4096                 ((unsigned int) 0x5 <<  4) /* (MCI) DTOCYC x 4096*/
#define 	AT91C_MCI_DTOMUL_65536                ((unsigned int) 0x6 <<  4) /* (MCI) DTOCYC x 65536*/
#define 	AT91C_MCI_DTOMUL_1048576              ((unsigned int) 0x7 <<  4) /* (MCI) DTOCYC x 1048576*/
/* -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- */
#define AT91C_MCI_SCDSEL      ((unsigned int) 0xF <<  0) /* (MCI) SD Card Selector*/
#define AT91C_MCI_SCDBUS      ((unsigned int) 0x1 <<  7) /* (MCI) SD Card Bus Width*/
/* -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- */
#define AT91C_MCI_CMDNB       ((unsigned int) 0x3F <<  0) /* (MCI) Command Number*/
#define AT91C_MCI_RSPTYP      ((unsigned int) 0x3 <<  6) /* (MCI) Response Type*/
#define 	AT91C_MCI_RSPTYP_NO                   ((unsigned int) 0x0 <<  6) /* (MCI) No response*/
#define 	AT91C_MCI_RSPTYP_48                   ((unsigned int) 0x1 <<  6) /* (MCI) 48-bit response*/
#define 	AT91C_MCI_RSPTYP_136                  ((unsigned int) 0x2 <<  6) /* (MCI) 136-bit response*/
#define AT91C_MCI_SPCMD       ((unsigned int) 0x7 <<  8) /* (MCI) Special CMD*/
#define 	AT91C_MCI_SPCMD_NONE                 ((unsigned int) 0x0 <<  8) /* (MCI) Not a special CMD*/
#define 	AT91C_MCI_SPCMD_INIT                 ((unsigned int) 0x1 <<  8) /* (MCI) Initialization CMD*/
#define 	AT91C_MCI_SPCMD_SYNC                 ((unsigned int) 0x2 <<  8) /* (MCI) Synchronized CMD*/
#define 	AT91C_MCI_SPCMD_IT_CMD               ((unsigned int) 0x4 <<  8) /* (MCI) Interrupt command*/
#define 	AT91C_MCI_SPCMD_IT_REP               ((unsigned int) 0x5 <<  8) /* (MCI) Interrupt response*/
#define AT91C_MCI_OPDCMD      ((unsigned int) 0x1 << 11) /* (MCI) Open Drain Command*/
#define AT91C_MCI_MAXLAT      ((unsigned int) 0x1 << 12) /* (MCI) Maximum Latency for Command to respond*/
#define AT91C_MCI_TRCMD       ((unsigned int) 0x3 << 16) /* (MCI) Transfer CMD*/
#define 	AT91C_MCI_TRCMD_NO                   ((unsigned int) 0x0 << 16) /* (MCI) No transfer*/
#define 	AT91C_MCI_TRCMD_START                ((unsigned int) 0x1 << 16) /* (MCI) Start transfer*/
#define 	AT91C_MCI_TRCMD_STOP                 ((unsigned int) 0x2 << 16) /* (MCI) Stop transfer*/
#define AT91C_MCI_TRDIR       ((unsigned int) 0x1 << 18) /* (MCI) Transfer Direction*/
#define AT91C_MCI_TRTYP       ((unsigned int) 0x3 << 19) /* (MCI) Transfer Type*/
#define 	AT91C_MCI_TRTYP_BLOCK                ((unsigned int) 0x0 << 19) /* (MCI) Block Transfer type*/
#define 	AT91C_MCI_TRTYP_MULTIPLE             ((unsigned int) 0x1 << 19) /* (MCI) Multiple Block transfer type*/
#define 	AT91C_MCI_TRTYP_STREAM               ((unsigned int) 0x2 << 19) /* (MCI) Stream transfer type*/
/* -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- */
#define AT91C_MCI_CMDRDY      ((unsigned int) 0x1 <<  0) /* (MCI) Command Ready flag*/
#define AT91C_MCI_RXRDY       ((unsigned int) 0x1 <<  1) /* (MCI) RX Ready flag*/
#define AT91C_MCI_TXRDY       ((unsigned int) 0x1 <<  2) /* (MCI) TX Ready flag*/
#define AT91C_MCI_BLKE        ((unsigned int) 0x1 <<  3) /* (MCI) Data Block Transfer Ended flag*/
#define AT91C_MCI_DTIP        ((unsigned int) 0x1 <<  4) /* (MCI) Data Transfer in Progress flag*/
#define AT91C_MCI_NOTBUSY     ((unsigned int) 0x1 <<  5) /* (MCI) Data Line Not Busy flag*/
#define AT91C_MCI_ENDRX       ((unsigned int) 0x1 <<  6) /* (MCI) End of RX Buffer flag*/
#define AT91C_MCI_ENDTX       ((unsigned int) 0x1 <<  7) /* (MCI) End of TX Buffer flag*/
#define AT91C_MCI_RXBUFF      ((unsigned int) 0x1 << 14) /* (MCI) RX Buffer Full flag*/
#define AT91C_MCI_TXBUFE      ((unsigned int) 0x1 << 15) /* (MCI) TX Buffer Empty flag*/
#define AT91C_MCI_RINDE       ((unsigned int) 0x1 << 16) /* (MCI) Response Index Error flag*/
#define AT91C_MCI_RDIRE       ((unsigned int) 0x1 << 17) /* (MCI) Response Direction Error flag*/
#define AT91C_MCI_RCRCE       ((unsigned int) 0x1 << 18) /* (MCI) Response CRC Error flag*/
#define AT91C_MCI_RENDE       ((unsigned int) 0x1 << 19) /* (MCI) Response End Bit Error flag*/
#define AT91C_MCI_RTOE        ((unsigned int) 0x1 << 20) /* (MCI) Response Time-out Error flag*/
#define AT91C_MCI_DCRCE       ((unsigned int) 0x1 << 21) /* (MCI) data CRC Error flag*/
#define AT91C_MCI_DTOE        ((unsigned int) 0x1 << 22) /* (MCI) Data timeout Error flag*/
#define AT91C_MCI_OVRE        ((unsigned int) 0x1 << 30) /* (MCI) Overrun flag*/
#define AT91C_MCI_UNRE        ((unsigned int) 0x1 << 31) /* (MCI) Underrun flag*/
/* -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- */
/* -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- */
/* -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR USB Device Interface*/
/* ******************************************************************************/
typedef struct _AT91S_UDP {
	AT91_REG	 UDP_NUM; 	/* Frame Number Register*/
	AT91_REG	 UDP_GLBSTATE; 	/* Global State Register*/
	AT91_REG	 UDP_FADDR; 	/* Function Address Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 UDP_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 UDP_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 UDP_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 UDP_ISR; 	/* Interrupt Status Register*/
	AT91_REG	 UDP_ICR; 	/* Interrupt Clear Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 UDP_RSTEP; 	/* Reset Endpoint Register*/
	AT91_REG	 Reserved2[1]; 	/* */
	AT91_REG	 UDP_CSR[8]; 	/* Endpoint Control and Status Register*/
	AT91_REG	 UDP_FDR[8]; 	/* Endpoint FIFO Data Register*/
} AT91S_UDP, *AT91PS_UDP;

/* -------- UDP_FRM_NUM : (UDP Offset: 0x0) USB Frame Number Register -------- */
#define AT91C_UDP_FRM_NUM     ((unsigned int) 0x7FF <<  0) /* (UDP) Frame Number as Defined in the Packet Field Formats*/
#define AT91C_UDP_FRM_ERR     ((unsigned int) 0x1 << 16) /* (UDP) Frame Error*/
#define AT91C_UDP_FRM_OK      ((unsigned int) 0x1 << 17) /* (UDP) Frame OK*/
/* -------- UDP_GLB_STATE : (UDP Offset: 0x4) USB Global State Register -------- */
#define AT91C_UDP_FADDEN      ((unsigned int) 0x1 <<  0) /* (UDP) Function Address Enable*/
#define AT91C_UDP_CONFG       ((unsigned int) 0x1 <<  1) /* (UDP) Configured*/
#define AT91C_UDP_RMWUPE      ((unsigned int) 0x1 <<  2) /* (UDP) Remote Wake Up Enable*/
#define AT91C_UDP_RSMINPR     ((unsigned int) 0x1 <<  3) /* (UDP) A Resume Has Been Sent to the Host*/
/* -------- UDP_FADDR : (UDP Offset: 0x8) USB Function Address Register -------- */
#define AT91C_UDP_FADD        ((unsigned int) 0xFF <<  0) /* (UDP) Function Address Value*/
#define AT91C_UDP_FEN         ((unsigned int) 0x1 <<  8) /* (UDP) Function Enable*/
/* -------- UDP_IER : (UDP Offset: 0x10) USB Interrupt Enable Register -------- */
#define AT91C_UDP_EPINT0      ((unsigned int) 0x1 <<  0) /* (UDP) Endpoint 0 Interrupt*/
#define AT91C_UDP_EPINT1      ((unsigned int) 0x1 <<  1) /* (UDP) Endpoint 0 Interrupt*/
#define AT91C_UDP_EPINT2      ((unsigned int) 0x1 <<  2) /* (UDP) Endpoint 2 Interrupt*/
#define AT91C_UDP_EPINT3      ((unsigned int) 0x1 <<  3) /* (UDP) Endpoint 3 Interrupt*/
#define AT91C_UDP_EPINT4      ((unsigned int) 0x1 <<  4) /* (UDP) Endpoint 4 Interrupt*/
#define AT91C_UDP_EPINT5      ((unsigned int) 0x1 <<  5) /* (UDP) Endpoint 5 Interrupt*/
#define AT91C_UDP_EPINT6      ((unsigned int) 0x1 <<  6) /* (UDP) Endpoint 6 Interrupt*/
#define AT91C_UDP_EPINT7      ((unsigned int) 0x1 <<  7) /* (UDP) Endpoint 7 Interrupt*/
#define AT91C_UDP_RXSUSP      ((unsigned int) 0x1 <<  8) /* (UDP) USB Suspend Interrupt*/
#define AT91C_UDP_RXRSM       ((unsigned int) 0x1 <<  9) /* (UDP) USB Resume Interrupt*/
#define AT91C_UDP_EXTRSM      ((unsigned int) 0x1 << 10) /* (UDP) USB External Resume Interrupt*/
#define AT91C_UDP_SOFINT      ((unsigned int) 0x1 << 11) /* (UDP) USB Start Of frame Interrupt*/
#define AT91C_UDP_WAKEUP      ((unsigned int) 0x1 << 13) /* (UDP) USB Resume Interrupt*/
/* -------- UDP_IDR : (UDP Offset: 0x14) USB Interrupt Disable Register -------- */
/* -------- UDP_IMR : (UDP Offset: 0x18) USB Interrupt Mask Register -------- */
/* -------- UDP_ISR : (UDP Offset: 0x1c) USB Interrupt Status Register -------- */
#define AT91C_UDP_ENDBUSRES   ((unsigned int) 0x1 << 12) /* (UDP) USB End Of Bus Reset Interrupt*/
/* -------- UDP_ICR : (UDP Offset: 0x20) USB Interrupt Clear Register -------- */
/* -------- UDP_RST_EP : (UDP Offset: 0x28) USB Reset Endpoint Register -------- */
#define AT91C_UDP_EP0         ((unsigned int) 0x1 <<  0) /* (UDP) Reset Endpoint 0*/
#define AT91C_UDP_EP1         ((unsigned int) 0x1 <<  1) /* (UDP) Reset Endpoint 1*/
#define AT91C_UDP_EP2         ((unsigned int) 0x1 <<  2) /* (UDP) Reset Endpoint 2*/
#define AT91C_UDP_EP3         ((unsigned int) 0x1 <<  3) /* (UDP) Reset Endpoint 3*/
#define AT91C_UDP_EP4         ((unsigned int) 0x1 <<  4) /* (UDP) Reset Endpoint 4*/
#define AT91C_UDP_EP5         ((unsigned int) 0x1 <<  5) /* (UDP) Reset Endpoint 5*/
#define AT91C_UDP_EP6         ((unsigned int) 0x1 <<  6) /* (UDP) Reset Endpoint 6*/
#define AT91C_UDP_EP7         ((unsigned int) 0x1 <<  7) /* (UDP) Reset Endpoint 7*/
/* -------- UDP_CSR : (UDP Offset: 0x30) USB Endpoint Control and Status Register -------- */
#define AT91C_UDP_TXCOMP      ((unsigned int) 0x1 <<  0) /* (UDP) Generates an IN packet with data previously written in the DPR*/
#define AT91C_UDP_RX_DATA_BK0 ((unsigned int) 0x1 <<  1) /* (UDP) Receive Data Bank 0*/
#define AT91C_UDP_RXSETUP     ((unsigned int) 0x1 <<  2) /* (UDP) Sends STALL to the Host (Control endpoints)*/
#define AT91C_UDP_ISOERROR    ((unsigned int) 0x1 <<  3) /* (UDP) Isochronous error (Isochronous endpoints)*/
#define AT91C_UDP_TXPKTRDY    ((unsigned int) 0x1 <<  4) /* (UDP) Transmit Packet Ready*/
#define AT91C_UDP_FORCESTALL  ((unsigned int) 0x1 <<  5) /* (UDP) Force Stall (used by Control, Bulk and Isochronous endpoints).*/
#define AT91C_UDP_RX_DATA_BK1 ((unsigned int) 0x1 <<  6) /* (UDP) Receive Data Bank 1 (only used by endpoints with ping-pong attributes).*/
#define AT91C_UDP_DIR         ((unsigned int) 0x1 <<  7) /* (UDP) Transfer Direction*/
#define AT91C_UDP_EPTYPE      ((unsigned int) 0x7 <<  8) /* (UDP) Endpoint type*/
#define 	AT91C_UDP_EPTYPE_CTRL                 ((unsigned int) 0x0 <<  8) /* (UDP) Control*/
#define 	AT91C_UDP_EPTYPE_ISO_OUT              ((unsigned int) 0x1 <<  8) /* (UDP) Isochronous OUT*/
#define 	AT91C_UDP_EPTYPE_BULK_OUT             ((unsigned int) 0x2 <<  8) /* (UDP) Bulk OUT*/
#define 	AT91C_UDP_EPTYPE_INT_OUT              ((unsigned int) 0x3 <<  8) /* (UDP) Interrupt OUT*/
#define 	AT91C_UDP_EPTYPE_ISO_IN               ((unsigned int) 0x5 <<  8) /* (UDP) Isochronous IN*/
#define 	AT91C_UDP_EPTYPE_BULK_IN              ((unsigned int) 0x6 <<  8) /* (UDP) Bulk IN*/
#define 	AT91C_UDP_EPTYPE_INT_IN               ((unsigned int) 0x7 <<  8) /* (UDP) Interrupt IN*/
#define AT91C_UDP_DTGLE       ((unsigned int) 0x1 << 11) /* (UDP) Data Toggle*/
#define AT91C_UDP_EPEDS       ((unsigned int) 0x1 << 15) /* (UDP) Endpoint Enable Disable*/
#define AT91C_UDP_RXBYTECNT   ((unsigned int) 0x7FF << 16) /* (UDP) Number Of Bytes Available in the FIFO*/

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
/*               REGISTER ADDRESS DEFINITION FOR AT91RM3400*/
/* ******************************************************************************/
/* ========== Register definition for SYS peripheral ========== */
/* ========== Register definition for MC peripheral ========== */
#define AT91C_MC_ASR    ((AT91_REG *) 	0xFFFFFF04) /* (MC) MC Abort Status Register*/
#define AT91C_MC_RCR    ((AT91_REG *) 	0xFFFFFF00) /* (MC) MC Remap Control Register*/
#define AT91C_MC_PUP    ((AT91_REG *) 	0xFFFFFF50) /* (MC) MC Protection Unit Peripherals*/
#define AT91C_MC_PUIA   ((AT91_REG *) 	0xFFFFFF10) /* (MC) MC Protection Unit Area*/
#define AT91C_MC_AASR   ((AT91_REG *) 	0xFFFFFF08) /* (MC) MC Abort Address Status Register*/
#define AT91C_MC_PUER   ((AT91_REG *) 	0xFFFFFF54) /* (MC) MC Protection Unit Enable Register*/
/* ========== Register definition for RTC peripheral ========== */
#define AT91C_RTC_VER   ((AT91_REG *) 	0xFFFFFE2C) /* (RTC) Valid Entry Register*/
#define AT91C_RTC_MR    ((AT91_REG *) 	0xFFFFFE04) /* (RTC) Mode Register*/
#define AT91C_RTC_IDR   ((AT91_REG *) 	0xFFFFFE24) /* (RTC) Interrupt Disable Register*/
#define AT91C_RTC_CALALR ((AT91_REG *) 	0xFFFFFE14) /* (RTC) Calendar Alarm Register*/
#define AT91C_RTC_IER   ((AT91_REG *) 	0xFFFFFE20) /* (RTC) Interrupt Enable Register*/
#define AT91C_RTC_TIMALR ((AT91_REG *) 	0xFFFFFE10) /* (RTC) Time Alarm Register*/
#define AT91C_RTC_IMR   ((AT91_REG *) 	0xFFFFFE28) /* (RTC) Interrupt Mask Register*/
#define AT91C_RTC_CR    ((AT91_REG *) 	0xFFFFFE00) /* (RTC) Control Register*/
#define AT91C_RTC_SCCR  ((AT91_REG *) 	0xFFFFFE1C) /* (RTC) Status Clear Command Register*/
#define AT91C_RTC_TIMR  ((AT91_REG *) 	0xFFFFFE08) /* (RTC) Time Register*/
#define AT91C_RTC_SR    ((AT91_REG *) 	0xFFFFFE18) /* (RTC) Status Register*/
#define AT91C_RTC_CALR  ((AT91_REG *) 	0xFFFFFE0C) /* (RTC) Calendar Register*/
/* ========== Register definition for ST peripheral ========== */
#define AT91C_ST_CR     ((AT91_REG *) 	0xFFFFFD00) /* (ST) Control Register*/
#define AT91C_ST_RTAR   ((AT91_REG *) 	0xFFFFFD20) /* (ST) Real-time Alarm Register*/
#define AT91C_ST_IDR    ((AT91_REG *) 	0xFFFFFD18) /* (ST) Interrupt Disable Register*/
#define AT91C_ST_PIMR   ((AT91_REG *) 	0xFFFFFD04) /* (ST) Period Interval Mode Register*/
#define AT91C_ST_IER    ((AT91_REG *) 	0xFFFFFD14) /* (ST) Interrupt Enable Register*/
#define AT91C_ST_CRTR   ((AT91_REG *) 	0xFFFFFD24) /* (ST) Current Real-time Register*/
#define AT91C_ST_WDMR   ((AT91_REG *) 	0xFFFFFD08) /* (ST) Watchdog Mode Register*/
#define AT91C_ST_SR     ((AT91_REG *) 	0xFFFFFD10) /* (ST) Status Register*/
#define AT91C_ST_RTMR   ((AT91_REG *) 	0xFFFFFD0C) /* (ST) Real-time Mode Register*/
#define AT91C_ST_IMR    ((AT91_REG *) 	0xFFFFFD1C) /* (ST) Interrupt Mask Register*/
/* ========== Register definition for PMC peripheral ========== */
#define AT91C_PMC_IDR   ((AT91_REG *) 	0xFFFFFC64) /* (PMC) Interrupt Disable Register*/
#define AT91C_PMC_PCER  ((AT91_REG *) 	0xFFFFFC10) /* (PMC) Peripheral Clock Enable Register*/
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0xFFFFFC40) /* (PMC) Programmable Clock Register*/
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0xFFFFFC30) /* (PMC) Master Clock Register*/
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0xFFFFFC04) /* (PMC) System Clock Disable Register*/
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0xFFFFFC14) /* (PMC) Peripheral Clock Disable Register*/
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0xFFFFFC08) /* (PMC) System Clock Status Register*/
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0xFFFFFC18) /* (PMC) Peripheral Clock Status Register*/
#define AT91C_PMC_SCER  ((AT91_REG *) 	0xFFFFFC00) /* (PMC) System Clock Enable Register*/
#define AT91C_PMC_IMR   ((AT91_REG *) 	0xFFFFFC6C) /* (PMC) Interrupt Mask Register*/
#define AT91C_PMC_IER   ((AT91_REG *) 	0xFFFFFC60) /* (PMC) Interrupt Enable Register*/
#define AT91C_PMC_SR    ((AT91_REG *) 	0xFFFFFC68) /* (PMC) Status Register*/
/* ========== Register definition for CKGR peripheral ========== */
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0xFFFFFC20) /* (CKGR) Main Oscillator Register*/
#define AT91C_CKGR_PLLBR ((AT91_REG *) 	0xFFFFFC2C) /* (CKGR) PLL B Register*/
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0xFFFFFC24) /* (CKGR) Main Clock  Frequency Register*/
#define AT91C_CKGR_PLLAR ((AT91_REG *) 	0xFFFFFC28) /* (CKGR) PLL A Register*/
/* ========== Register definition for PIOB peripheral ========== */
#define AT91C_PIOB_OWDR ((AT91_REG *) 	0xFFFFF6A4) /* (PIOB) Output Write Disable Register*/
#define AT91C_PIOB_MDER ((AT91_REG *) 	0xFFFFF650) /* (PIOB) Multi-driver Enable Register*/
#define AT91C_PIOB_PPUSR ((AT91_REG *) 	0xFFFFF668) /* (PIOB) Pad Pull-up Status Register*/
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
#define AT91C_PIOA_PPUSR ((AT91_REG *) 	0xFFFFF468) /* (PIOA) Pad Pull-up Status Register*/
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
/* ========== Register definition for DBGU peripheral ========== */
#define AT91C_DBGU_C2R  ((AT91_REG *) 	0xFFFFF244) /* (DBGU) Chip ID2 Register*/
#define AT91C_DBGU_BRGR ((AT91_REG *) 	0xFFFFF220) /* (DBGU) Baud Rate Generator Register*/
#define AT91C_DBGU_IDR  ((AT91_REG *) 	0xFFFFF20C) /* (DBGU) Interrupt Disable Register*/
#define AT91C_DBGU_CSR  ((AT91_REG *) 	0xFFFFF214) /* (DBGU) Channel Status Register*/
#define AT91C_DBGU_C1R  ((AT91_REG *) 	0xFFFFF240) /* (DBGU) Chip ID1 Register*/
#define AT91C_DBGU_MR   ((AT91_REG *) 	0xFFFFF204) /* (DBGU) Mode Register*/
#define AT91C_DBGU_IMR  ((AT91_REG *) 	0xFFFFF210) /* (DBGU) Interrupt Mask Register*/
#define AT91C_DBGU_CR   ((AT91_REG *) 	0xFFFFF200) /* (DBGU) Control Register*/
#define AT91C_DBGU_FNTR ((AT91_REG *) 	0xFFFFF248) /* (DBGU) Force NTRST Register*/
#define AT91C_DBGU_THR  ((AT91_REG *) 	0xFFFFF21C) /* (DBGU) Transmitter Holding Register*/
#define AT91C_DBGU_RHR  ((AT91_REG *) 	0xFFFFF218) /* (DBGU) Receiver Holding Register*/
#define AT91C_DBGU_IER  ((AT91_REG *) 	0xFFFFF208) /* (DBGU) Interrupt Enable Register*/
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
/* ========== Register definition for PDC_SSC2 peripheral ========== */
#define AT91C_SSC2_PTSR ((AT91_REG *) 	0xFFFD8124) /* (PDC_SSC2) PDC Transfer Status Register*/
#define AT91C_SSC2_PTCR ((AT91_REG *) 	0xFFFD8120) /* (PDC_SSC2) PDC Transfer Control Register*/
#define AT91C_SSC2_TNPR ((AT91_REG *) 	0xFFFD8118) /* (PDC_SSC2) Transmit Next Pointer Register*/
#define AT91C_SSC2_TNCR ((AT91_REG *) 	0xFFFD811C) /* (PDC_SSC2) Transmit Next Counter Register*/
#define AT91C_SSC2_RNPR ((AT91_REG *) 	0xFFFD8110) /* (PDC_SSC2) Receive Next Pointer Register*/
#define AT91C_SSC2_RNCR ((AT91_REG *) 	0xFFFD8114) /* (PDC_SSC2) Receive Next Counter Register*/
#define AT91C_SSC2_RPR  ((AT91_REG *) 	0xFFFD8100) /* (PDC_SSC2) Receive Pointer Register*/
#define AT91C_SSC2_TCR  ((AT91_REG *) 	0xFFFD810C) /* (PDC_SSC2) Transmit Counter Register*/
#define AT91C_SSC2_TPR  ((AT91_REG *) 	0xFFFD8108) /* (PDC_SSC2) Transmit Pointer Register*/
#define AT91C_SSC2_RCR  ((AT91_REG *) 	0xFFFD8104) /* (PDC_SSC2) Receive Counter Register*/
/* ========== Register definition for SSC2 peripheral ========== */
#define AT91C_SSC2_RC0R ((AT91_REG *) 	0xFFFD8038) /* (SSC2) Receive Compare 0 Register*/
#define AT91C_SSC2_RC1R ((AT91_REG *) 	0xFFFD803C) /* (SSC2) Receive Compare 1 Register*/
#define AT91C_SSC2_RSHR ((AT91_REG *) 	0xFFFD8030) /* (SSC2) Receive Sync Holding Register*/
#define AT91C_SSC2_IER  ((AT91_REG *) 	0xFFFD8044) /* (SSC2) Interrupt Enable Register*/
#define AT91C_SSC2_RFMR ((AT91_REG *) 	0xFFFD8014) /* (SSC2) Receive Frame Mode Register*/
#define AT91C_SSC2_TFMR ((AT91_REG *) 	0xFFFD801C) /* (SSC2) Transmit Frame Mode Register*/
#define AT91C_SSC2_SR   ((AT91_REG *) 	0xFFFD8040) /* (SSC2) Status Register*/
#define AT91C_SSC2_TSHR ((AT91_REG *) 	0xFFFD8034) /* (SSC2) Transmit Sync Holding Register*/
#define AT91C_SSC2_RHR  ((AT91_REG *) 	0xFFFD8020) /* (SSC2) Receive Holding Register*/
#define AT91C_SSC2_CR   ((AT91_REG *) 	0xFFFD8000) /* (SSC2) Control Register*/
#define AT91C_SSC2_IDR  ((AT91_REG *) 	0xFFFD8048) /* (SSC2) Interrupt Disable Register*/
#define AT91C_SSC2_IMR  ((AT91_REG *) 	0xFFFD804C) /* (SSC2) Interrupt Mask Register*/
#define AT91C_SSC2_THR  ((AT91_REG *) 	0xFFFD8024) /* (SSC2) Transmit Holding Register*/
#define AT91C_SSC2_RCMR ((AT91_REG *) 	0xFFFD8010) /* (SSC2) Receive Clock ModeRegister*/
#define AT91C_SSC2_TCMR ((AT91_REG *) 	0xFFFD8018) /* (SSC2) Transmit Clock Mode Register*/
#define AT91C_SSC2_CMR  ((AT91_REG *) 	0xFFFD8004) /* (SSC2) Clock Mode Register*/
/* ========== Register definition for PDC_SSC1 peripheral ========== */
#define AT91C_SSC1_TNCR ((AT91_REG *) 	0xFFFD411C) /* (PDC_SSC1) Transmit Next Counter Register*/
#define AT91C_SSC1_RPR  ((AT91_REG *) 	0xFFFD4100) /* (PDC_SSC1) Receive Pointer Register*/
#define AT91C_SSC1_RNCR ((AT91_REG *) 	0xFFFD4114) /* (PDC_SSC1) Receive Next Counter Register*/
#define AT91C_SSC1_TPR  ((AT91_REG *) 	0xFFFD4108) /* (PDC_SSC1) Transmit Pointer Register*/
#define AT91C_SSC1_PTCR ((AT91_REG *) 	0xFFFD4120) /* (PDC_SSC1) PDC Transfer Control Register*/
#define AT91C_SSC1_TCR  ((AT91_REG *) 	0xFFFD410C) /* (PDC_SSC1) Transmit Counter Register*/
#define AT91C_SSC1_RCR  ((AT91_REG *) 	0xFFFD4104) /* (PDC_SSC1) Receive Counter Register*/
#define AT91C_SSC1_RNPR ((AT91_REG *) 	0xFFFD4110) /* (PDC_SSC1) Receive Next Pointer Register*/
#define AT91C_SSC1_TNPR ((AT91_REG *) 	0xFFFD4118) /* (PDC_SSC1) Transmit Next Pointer Register*/
#define AT91C_SSC1_PTSR ((AT91_REG *) 	0xFFFD4124) /* (PDC_SSC1) PDC Transfer Status Register*/
/* ========== Register definition for SSC1 peripheral ========== */
#define AT91C_SSC1_RC1R ((AT91_REG *) 	0xFFFD403C) /* (SSC1) Receive Compare 1 Register*/
#define AT91C_SSC1_RHR  ((AT91_REG *) 	0xFFFD4020) /* (SSC1) Receive Holding Register*/
#define AT91C_SSC1_RSHR ((AT91_REG *) 	0xFFFD4030) /* (SSC1) Receive Sync Holding Register*/
#define AT91C_SSC1_TFMR ((AT91_REG *) 	0xFFFD401C) /* (SSC1) Transmit Frame Mode Register*/
#define AT91C_SSC1_IDR  ((AT91_REG *) 	0xFFFD4048) /* (SSC1) Interrupt Disable Register*/
#define AT91C_SSC1_RC0R ((AT91_REG *) 	0xFFFD4038) /* (SSC1) Receive Compare 0 Register*/
#define AT91C_SSC1_THR  ((AT91_REG *) 	0xFFFD4024) /* (SSC1) Transmit Holding Register*/
#define AT91C_SSC1_RCMR ((AT91_REG *) 	0xFFFD4010) /* (SSC1) Receive Clock ModeRegister*/
#define AT91C_SSC1_IER  ((AT91_REG *) 	0xFFFD4044) /* (SSC1) Interrupt Enable Register*/
#define AT91C_SSC1_TSHR ((AT91_REG *) 	0xFFFD4034) /* (SSC1) Transmit Sync Holding Register*/
#define AT91C_SSC1_SR   ((AT91_REG *) 	0xFFFD4040) /* (SSC1) Status Register*/
#define AT91C_SSC1_CMR  ((AT91_REG *) 	0xFFFD4004) /* (SSC1) Clock Mode Register*/
#define AT91C_SSC1_TCMR ((AT91_REG *) 	0xFFFD4018) /* (SSC1) Transmit Clock Mode Register*/
#define AT91C_SSC1_CR   ((AT91_REG *) 	0xFFFD4000) /* (SSC1) Control Register*/
#define AT91C_SSC1_IMR  ((AT91_REG *) 	0xFFFD404C) /* (SSC1) Interrupt Mask Register*/
#define AT91C_SSC1_RFMR ((AT91_REG *) 	0xFFFD4014) /* (SSC1) Receive Frame Mode Register*/
/* ========== Register definition for PDC_SSC0 peripheral ========== */
#define AT91C_SSC0_RNPR ((AT91_REG *) 	0xFFFD0110) /* (PDC_SSC0) Receive Next Pointer Register*/
#define AT91C_SSC0_RNCR ((AT91_REG *) 	0xFFFD0114) /* (PDC_SSC0) Receive Next Counter Register*/
#define AT91C_SSC0_PTSR ((AT91_REG *) 	0xFFFD0124) /* (PDC_SSC0) PDC Transfer Status Register*/
#define AT91C_SSC0_PTCR ((AT91_REG *) 	0xFFFD0120) /* (PDC_SSC0) PDC Transfer Control Register*/
#define AT91C_SSC0_TCR  ((AT91_REG *) 	0xFFFD010C) /* (PDC_SSC0) Transmit Counter Register*/
#define AT91C_SSC0_TNPR ((AT91_REG *) 	0xFFFD0118) /* (PDC_SSC0) Transmit Next Pointer Register*/
#define AT91C_SSC0_RCR  ((AT91_REG *) 	0xFFFD0104) /* (PDC_SSC0) Receive Counter Register*/
#define AT91C_SSC0_TPR  ((AT91_REG *) 	0xFFFD0108) /* (PDC_SSC0) Transmit Pointer Register*/
#define AT91C_SSC0_TNCR ((AT91_REG *) 	0xFFFD011C) /* (PDC_SSC0) Transmit Next Counter Register*/
#define AT91C_SSC0_RPR  ((AT91_REG *) 	0xFFFD0100) /* (PDC_SSC0) Receive Pointer Register*/
/* ========== Register definition for SSC0 peripheral ========== */
#define AT91C_SSC0_THR  ((AT91_REG *) 	0xFFFD0024) /* (SSC0) Transmit Holding Register*/
#define AT91C_SSC0_IDR  ((AT91_REG *) 	0xFFFD0048) /* (SSC0) Interrupt Disable Register*/
#define AT91C_SSC0_CMR  ((AT91_REG *) 	0xFFFD0004) /* (SSC0) Clock Mode Register*/
#define AT91C_SSC0_SR   ((AT91_REG *) 	0xFFFD0040) /* (SSC0) Status Register*/
#define AT91C_SSC0_RC0R ((AT91_REG *) 	0xFFFD0038) /* (SSC0) Receive Compare 0 Register*/
#define AT91C_SSC0_RCMR ((AT91_REG *) 	0xFFFD0010) /* (SSC0) Receive Clock ModeRegister*/
#define AT91C_SSC0_RC1R ((AT91_REG *) 	0xFFFD003C) /* (SSC0) Receive Compare 1 Register*/
#define AT91C_SSC0_IER  ((AT91_REG *) 	0xFFFD0044) /* (SSC0) Interrupt Enable Register*/
#define AT91C_SSC0_CR   ((AT91_REG *) 	0xFFFD0000) /* (SSC0) Control Register*/
#define AT91C_SSC0_TFMR ((AT91_REG *) 	0xFFFD001C) /* (SSC0) Transmit Frame Mode Register*/
#define AT91C_SSC0_RHR  ((AT91_REG *) 	0xFFFD0020) /* (SSC0) Receive Holding Register*/
#define AT91C_SSC0_IMR  ((AT91_REG *) 	0xFFFD004C) /* (SSC0) Interrupt Mask Register*/
#define AT91C_SSC0_TCMR ((AT91_REG *) 	0xFFFD0018) /* (SSC0) Transmit Clock Mode Register*/
#define AT91C_SSC0_RSHR ((AT91_REG *) 	0xFFFD0030) /* (SSC0) Receive Sync Holding Register*/
#define AT91C_SSC0_RFMR ((AT91_REG *) 	0xFFFD0014) /* (SSC0) Receive Frame Mode Register*/
#define AT91C_SSC0_TSHR ((AT91_REG *) 	0xFFFD0034) /* (SSC0) Transmit Sync Holding Register*/
/* ========== Register definition for PDC_US3 peripheral ========== */
#define AT91C_US3_PTCR  ((AT91_REG *) 	0xFFFCC120) /* (PDC_US3) PDC Transfer Control Register*/
#define AT91C_US3_RNPR  ((AT91_REG *) 	0xFFFCC110) /* (PDC_US3) Receive Next Pointer Register*/
#define AT91C_US3_RCR   ((AT91_REG *) 	0xFFFCC104) /* (PDC_US3) Receive Counter Register*/
#define AT91C_US3_TPR   ((AT91_REG *) 	0xFFFCC108) /* (PDC_US3) Transmit Pointer Register*/
#define AT91C_US3_PTSR  ((AT91_REG *) 	0xFFFCC124) /* (PDC_US3) PDC Transfer Status Register*/
#define AT91C_US3_TNCR  ((AT91_REG *) 	0xFFFCC11C) /* (PDC_US3) Transmit Next Counter Register*/
#define AT91C_US3_RPR   ((AT91_REG *) 	0xFFFCC100) /* (PDC_US3) Receive Pointer Register*/
#define AT91C_US3_TCR   ((AT91_REG *) 	0xFFFCC10C) /* (PDC_US3) Transmit Counter Register*/
#define AT91C_US3_RNCR  ((AT91_REG *) 	0xFFFCC114) /* (PDC_US3) Receive Next Counter Register*/
#define AT91C_US3_TNPR  ((AT91_REG *) 	0xFFFCC118) /* (PDC_US3) Transmit Next Pointer Register*/
/* ========== Register definition for US3 peripheral ========== */
#define AT91C_US3_CSR   ((AT91_REG *) 	0xFFFCC014) /* (US3) Channel Status Register*/
#define AT91C_US3_IER   ((AT91_REG *) 	0xFFFCC008) /* (US3) Interrupt Enable Register*/
#define AT91C_US3_IMR   ((AT91_REG *) 	0xFFFCC010) /* (US3) Interrupt Mask Register*/
#define AT91C_US3_THR   ((AT91_REG *) 	0xFFFCC01C) /* (US3) Transmitter Holding Register*/
#define AT91C_US3_BRGR  ((AT91_REG *) 	0xFFFCC020) /* (US3) Baud Rate Generator Register*/
#define AT91C_US3_RTOR  ((AT91_REG *) 	0xFFFCC024) /* (US3) Receiver Time-out Register*/
#define AT91C_US3_IDR   ((AT91_REG *) 	0xFFFCC00C) /* (US3) Interrupt Disable Register*/
#define AT91C_US3_FIDI  ((AT91_REG *) 	0xFFFCC040) /* (US3) FI_DI_Ratio Register*/
#define AT91C_US3_RHR   ((AT91_REG *) 	0xFFFCC018) /* (US3) Receiver Holding Register*/
#define AT91C_US3_NER   ((AT91_REG *) 	0xFFFCC044) /* (US3) Nb Errors Register*/
#define AT91C_US3_XXR   ((AT91_REG *) 	0xFFFCC048) /* (US3) XON_XOFF Register*/
#define AT91C_US3_IF    ((AT91_REG *) 	0xFFFCC04C) /* (US3) IRDA_FILTER Register*/
#define AT91C_US3_TTGR  ((AT91_REG *) 	0xFFFCC028) /* (US3) Transmitter Time-guard Register*/
#define AT91C_US3_CR    ((AT91_REG *) 	0xFFFCC000) /* (US3) Control Register*/
#define AT91C_US3_MR    ((AT91_REG *) 	0xFFFCC004) /* (US3) Mode Register*/
/* ========== Register definition for PDC_US2 peripheral ========== */
#define AT91C_US2_PTCR  ((AT91_REG *) 	0xFFFC8120) /* (PDC_US2) PDC Transfer Control Register*/
#define AT91C_US2_TCR   ((AT91_REG *) 	0xFFFC810C) /* (PDC_US2) Transmit Counter Register*/
#define AT91C_US2_RPR   ((AT91_REG *) 	0xFFFC8100) /* (PDC_US2) Receive Pointer Register*/
#define AT91C_US2_TPR   ((AT91_REG *) 	0xFFFC8108) /* (PDC_US2) Transmit Pointer Register*/
#define AT91C_US2_PTSR  ((AT91_REG *) 	0xFFFC8124) /* (PDC_US2) PDC Transfer Status Register*/
#define AT91C_US2_RNCR  ((AT91_REG *) 	0xFFFC8114) /* (PDC_US2) Receive Next Counter Register*/
#define AT91C_US2_TNPR  ((AT91_REG *) 	0xFFFC8118) /* (PDC_US2) Transmit Next Pointer Register*/
#define AT91C_US2_RCR   ((AT91_REG *) 	0xFFFC8104) /* (PDC_US2) Receive Counter Register*/
#define AT91C_US2_RNPR  ((AT91_REG *) 	0xFFFC8110) /* (PDC_US2) Receive Next Pointer Register*/
#define AT91C_US2_TNCR  ((AT91_REG *) 	0xFFFC811C) /* (PDC_US2) Transmit Next Counter Register*/
/* ========== Register definition for US2 peripheral ========== */
#define AT91C_US2_RHR   ((AT91_REG *) 	0xFFFC8018) /* (US2) Receiver Holding Register*/
#define AT91C_US2_BRGR  ((AT91_REG *) 	0xFFFC8020) /* (US2) Baud Rate Generator Register*/
#define AT91C_US2_IF    ((AT91_REG *) 	0xFFFC804C) /* (US2) IRDA_FILTER Register*/
#define AT91C_US2_IDR   ((AT91_REG *) 	0xFFFC800C) /* (US2) Interrupt Disable Register*/
#define AT91C_US2_IMR   ((AT91_REG *) 	0xFFFC8010) /* (US2) Interrupt Mask Register*/
#define AT91C_US2_CR    ((AT91_REG *) 	0xFFFC8000) /* (US2) Control Register*/
#define AT91C_US2_IER   ((AT91_REG *) 	0xFFFC8008) /* (US2) Interrupt Enable Register*/
#define AT91C_US2_NER   ((AT91_REG *) 	0xFFFC8044) /* (US2) Nb Errors Register*/
#define AT91C_US2_RTOR  ((AT91_REG *) 	0xFFFC8024) /* (US2) Receiver Time-out Register*/
#define AT91C_US2_TTGR  ((AT91_REG *) 	0xFFFC8028) /* (US2) Transmitter Time-guard Register*/
#define AT91C_US2_MR    ((AT91_REG *) 	0xFFFC8004) /* (US2) Mode Register*/
#define AT91C_US2_CSR   ((AT91_REG *) 	0xFFFC8014) /* (US2) Channel Status Register*/
#define AT91C_US2_XXR   ((AT91_REG *) 	0xFFFC8048) /* (US2) XON_XOFF Register*/
#define AT91C_US2_THR   ((AT91_REG *) 	0xFFFC801C) /* (US2) Transmitter Holding Register*/
#define AT91C_US2_FIDI  ((AT91_REG *) 	0xFFFC8040) /* (US2) FI_DI_Ratio Register*/
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
#define AT91C_US1_XXR   ((AT91_REG *) 	0xFFFC4048) /* (US1) XON_XOFF Register*/
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
#define AT91C_US0_XXR   ((AT91_REG *) 	0xFFFC0048) /* (US0) XON_XOFF Register*/
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
/* ========== Register definition for PDC_MCI peripheral ========== */
#define AT91C_MCI_PTCR  ((AT91_REG *) 	0xFFFB4120) /* (PDC_MCI) PDC Transfer Control Register*/
#define AT91C_MCI_RCR   ((AT91_REG *) 	0xFFFB4104) /* (PDC_MCI) Receive Counter Register*/
#define AT91C_MCI_RPR   ((AT91_REG *) 	0xFFFB4100) /* (PDC_MCI) Receive Pointer Register*/
#define AT91C_MCI_PTSR  ((AT91_REG *) 	0xFFFB4124) /* (PDC_MCI) PDC Transfer Status Register*/
#define AT91C_MCI_TPR   ((AT91_REG *) 	0xFFFB4108) /* (PDC_MCI) Transmit Pointer Register*/
#define AT91C_MCI_TCR   ((AT91_REG *) 	0xFFFB410C) /* (PDC_MCI) Transmit Counter Register*/
#define AT91C_MCI_RNPR  ((AT91_REG *) 	0xFFFB4110) /* (PDC_MCI) Receive Next Pointer Register*/
#define AT91C_MCI_TNCR  ((AT91_REG *) 	0xFFFB411C) /* (PDC_MCI) Transmit Next Counter Register*/
#define AT91C_MCI_RNCR  ((AT91_REG *) 	0xFFFB4114) /* (PDC_MCI) Receive Next Counter Register*/
#define AT91C_MCI_TNPR  ((AT91_REG *) 	0xFFFB4118) /* (PDC_MCI) Transmit Next Pointer Register*/
/* ========== Register definition for MCI peripheral ========== */
#define AT91C_MCI_TDR   ((AT91_REG *) 	0xFFFB4034) /* (MCI) MCI Transmit Data Register*/
#define AT91C_MCI_RSPR  ((AT91_REG *) 	0xFFFB4020) /* (MCI) MCI Response Register*/
#define AT91C_MCI_SDCR  ((AT91_REG *) 	0xFFFB400C) /* (MCI) MCI SD Card Register*/
#define AT91C_MCI_MR    ((AT91_REG *) 	0xFFFB4004) /* (MCI) MCI Mode Register*/
#define AT91C_MCI_CR    ((AT91_REG *) 	0xFFFB4000) /* (MCI) MCI Control Register*/
#define AT91C_MCI_ARGR  ((AT91_REG *) 	0xFFFB4010) /* (MCI) MCI Argument Register*/
#define AT91C_MCI_SR    ((AT91_REG *) 	0xFFFB4040) /* (MCI) MCI Status Register*/
#define AT91C_MCI_RDR   ((AT91_REG *) 	0xFFFB4030) /* (MCI) MCI Receive Data Register*/
#define AT91C_MCI_DTOR  ((AT91_REG *) 	0xFFFB4008) /* (MCI) MCI Data Timeout Register*/
#define AT91C_MCI_CMDR  ((AT91_REG *) 	0xFFFB4014) /* (MCI) MCI Command Register*/
#define AT91C_MCI_IMR   ((AT91_REG *) 	0xFFFB404C) /* (MCI) MCI Interrupt Mask Register*/
#define AT91C_MCI_IER   ((AT91_REG *) 	0xFFFB4044) /* (MCI) MCI Interrupt Enable Register*/
#define AT91C_MCI_IDR   ((AT91_REG *) 	0xFFFB4048) /* (MCI) MCI Interrupt Disable Register*/
/* ========== Register definition for UDP peripheral ========== */
#define AT91C_UDP_IMR   ((AT91_REG *) 	0xFFFB0018) /* (UDP) Interrupt Mask Register*/
#define AT91C_UDP_FADDR ((AT91_REG *) 	0xFFFB0008) /* (UDP) Function Address Register*/
#define AT91C_UDP_NUM   ((AT91_REG *) 	0xFFFB0000) /* (UDP) Frame Number Register*/
#define AT91C_UDP_FDR   ((AT91_REG *) 	0xFFFB0050) /* (UDP) Endpoint FIFO Data Register*/
#define AT91C_UDP_ISR   ((AT91_REG *) 	0xFFFB001C) /* (UDP) Interrupt Status Register*/
#define AT91C_UDP_CSR   ((AT91_REG *) 	0xFFFB0030) /* (UDP) Endpoint Control and Status Register*/
#define AT91C_UDP_IDR   ((AT91_REG *) 	0xFFFB0014) /* (UDP) Interrupt Disable Register*/
#define AT91C_UDP_ICR   ((AT91_REG *) 	0xFFFB0020) /* (UDP) Interrupt Clear Register*/
#define AT91C_UDP_RSTEP ((AT91_REG *) 	0xFFFB0028) /* (UDP) Reset Endpoint Register*/
#define AT91C_UDP_GLBSTATE ((AT91_REG *) 	0xFFFB0004) /* (UDP) Global State Register*/
#define AT91C_UDP_IER   ((AT91_REG *) 	0xFFFB0010) /* (UDP) Interrupt Enable Register*/
/* ========== Register definition for TC5 peripheral ========== */
#define AT91C_TC5_CMR   ((AT91_REG *) 	0xFFFA4084) /* (TC5) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC5_IDR   ((AT91_REG *) 	0xFFFA40A8) /* (TC5) Interrupt Disable Register*/
#define AT91C_TC5_CCR   ((AT91_REG *) 	0xFFFA4080) /* (TC5) Channel Control Register*/
#define AT91C_TC5_RB    ((AT91_REG *) 	0xFFFA4098) /* (TC5) Register B*/
#define AT91C_TC5_IMR   ((AT91_REG *) 	0xFFFA40AC) /* (TC5) Interrupt Mask Register*/
#define AT91C_TC5_CV    ((AT91_REG *) 	0xFFFA4090) /* (TC5) Counter Value*/
#define AT91C_TC5_RC    ((AT91_REG *) 	0xFFFA409C) /* (TC5) Register C*/
#define AT91C_TC5_SR    ((AT91_REG *) 	0xFFFA40A0) /* (TC5) Status Register*/
#define AT91C_TC5_IER   ((AT91_REG *) 	0xFFFA40A4) /* (TC5) Interrupt Enable Register*/
#define AT91C_TC5_RA    ((AT91_REG *) 	0xFFFA4094) /* (TC5) Register A*/
/* ========== Register definition for TC4 peripheral ========== */
#define AT91C_TC4_SR    ((AT91_REG *) 	0xFFFA4060) /* (TC4) Status Register*/
#define AT91C_TC4_RA    ((AT91_REG *) 	0xFFFA4054) /* (TC4) Register A*/
#define AT91C_TC4_CV    ((AT91_REG *) 	0xFFFA4050) /* (TC4) Counter Value*/
#define AT91C_TC4_CMR   ((AT91_REG *) 	0xFFFA4044) /* (TC4) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC4_RB    ((AT91_REG *) 	0xFFFA4058) /* (TC4) Register B*/
#define AT91C_TC4_CCR   ((AT91_REG *) 	0xFFFA4040) /* (TC4) Channel Control Register*/
#define AT91C_TC4_IER   ((AT91_REG *) 	0xFFFA4064) /* (TC4) Interrupt Enable Register*/
#define AT91C_TC4_IMR   ((AT91_REG *) 	0xFFFA406C) /* (TC4) Interrupt Mask Register*/
#define AT91C_TC4_RC    ((AT91_REG *) 	0xFFFA405C) /* (TC4) Register C*/
#define AT91C_TC4_IDR   ((AT91_REG *) 	0xFFFA4068) /* (TC4) Interrupt Disable Register*/
/* ========== Register definition for TC3 peripheral ========== */
#define AT91C_TC3_IMR   ((AT91_REG *) 	0xFFFA402C) /* (TC3) Interrupt Mask Register*/
#define AT91C_TC3_CMR   ((AT91_REG *) 	0xFFFA4004) /* (TC3) Channel Mode Register (Capture Mode / Waveform Mode)*/
#define AT91C_TC3_IDR   ((AT91_REG *) 	0xFFFA4028) /* (TC3) Interrupt Disable Register*/
#define AT91C_TC3_CCR   ((AT91_REG *) 	0xFFFA4000) /* (TC3) Channel Control Register*/
#define AT91C_TC3_RA    ((AT91_REG *) 	0xFFFA4014) /* (TC3) Register A*/
#define AT91C_TC3_RB    ((AT91_REG *) 	0xFFFA4018) /* (TC3) Register B*/
#define AT91C_TC3_CV    ((AT91_REG *) 	0xFFFA4010) /* (TC3) Counter Value*/
#define AT91C_TC3_SR    ((AT91_REG *) 	0xFFFA4020) /* (TC3) Status Register*/
#define AT91C_TC3_IER   ((AT91_REG *) 	0xFFFA4024) /* (TC3) Interrupt Enable Register*/
#define AT91C_TC3_RC    ((AT91_REG *) 	0xFFFA401C) /* (TC3) Register C*/
/* ========== Register definition for TCB1 peripheral ========== */
#define AT91C_TCB1_BMR  ((AT91_REG *) 	0xFFFA40C4) /* (TCB1) TC Block Mode Register*/
#define AT91C_TCB1_BCR  ((AT91_REG *) 	0xFFFA40C0) /* (TCB1) TC Block Control Register*/
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
/* ========== Register definition for TCB0 peripheral ========== */
#define AT91C_TCB0_BMR  ((AT91_REG *) 	0xFFFA00C4) /* (TCB0) TC Block Mode Register*/
#define AT91C_TCB0_BCR  ((AT91_REG *) 	0xFFFA00C0) /* (TCB0) TC Block Control Register*/

/* ******************************************************************************/
/*               PIO DEFINITIONS FOR AT91RM3400*/
/* ******************************************************************************/
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) /* Pin Controlled by PA0*/
#define AT91C_PA0_MISO     ((unsigned int) AT91C_PIO_PA0) /*  SPI Master In Slave*/
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) /* Pin Controlled by PA1*/
#define AT91C_PA1_MOSI     ((unsigned int) AT91C_PIO_PA1) /*  SPI Master Out Slave*/
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) /* Pin Controlled by PA10*/
#define AT91C_PA10_RXD0     ((unsigned int) AT91C_PIO_PA10) /*  USART 0 Receive Data*/
#define AT91C_PA10_UTXOEN   ((unsigned int) AT91C_PIO_PA10) /*  USB device TXOEN*/
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) /* Pin Controlled by PA11*/
#define AT91C_PA11_SCK0     ((unsigned int) AT91C_PIO_PA11) /*  USART 0 Serial Clock*/
#define AT91C_PA11_TCLK0    ((unsigned int) AT91C_PIO_PA11) /*  Timer Counter 0 external clock input*/
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) /* Pin Controlled by PA12*/
#define AT91C_PA12_CTS0     ((unsigned int) AT91C_PIO_PA12) /*  USART 0 Clear To Send*/
#define AT91C_PA12_TCLK1    ((unsigned int) AT91C_PIO_PA12) /*  Timer Counter 1 external clock input*/
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) /* Pin Controlled by PA13*/
#define AT91C_PA13_RTS0     ((unsigned int) AT91C_PIO_PA13) /*  USART 0 Ready To Send*/
#define AT91C_PA13_TCLK2    ((unsigned int) AT91C_PIO_PA13) /*  Timer Counter 2 external clock input*/
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) /* Pin Controlled by PA14*/
#define AT91C_PA14_RXD1     ((unsigned int) AT91C_PIO_PA14) /*  USART 1 Receive Data*/
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) /* Pin Controlled by PA15*/
#define AT91C_PA15_TXD1     ((unsigned int) AT91C_PIO_PA15) /*  USART 1 Transmit Data*/
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) /* Pin Controlled by PA16*/
#define AT91C_PA16_RTS1     ((unsigned int) AT91C_PIO_PA16) /*  USART 1 Ready To Send*/
#define AT91C_PA16_TIOA0    ((unsigned int) AT91C_PIO_PA16) /*  Timer Counter 0 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) /* Pin Controlled by PA17*/
#define AT91C_PA17_CTS1     ((unsigned int) AT91C_PIO_PA17) /*  USART 1 Clear To Send*/
#define AT91C_PA17_TIOB0    ((unsigned int) AT91C_PIO_PA17) /*  Timer Counter 0 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) /* Pin Controlled by PA18*/
#define AT91C_PA18_DTR1     ((unsigned int) AT91C_PIO_PA18) /*  USART 1 Data Terminal ready*/
#define AT91C_PA18_TIOA1    ((unsigned int) AT91C_PIO_PA18) /*  Timer Counter 1 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) /* Pin Controlled by PA19*/
#define AT91C_PA19_DSR1     ((unsigned int) AT91C_PIO_PA19) /*  USART 1 Data Set ready*/
#define AT91C_PA19_TIOB1    ((unsigned int) AT91C_PIO_PA19) /*  Timer Counter 1 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) /* Pin Controlled by PA2*/
#define AT91C_PA2_SPCK     ((unsigned int) AT91C_PIO_PA2) /*  SPI Serial Clock*/
#define AT91C_PA2_PCK0     ((unsigned int) AT91C_PIO_PA2) /*  PMC Programmable clock Output 0*/
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) /* Pin Controlled by PA20*/
#define AT91C_PA20_DCD1     ((unsigned int) AT91C_PIO_PA20) /*  USART 1 Data Carrier Detect*/
#define AT91C_PA20_TIOA2    ((unsigned int) AT91C_PIO_PA20) /*  Timer Counter 2 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) /* Pin Controlled by PA21*/
#define AT91C_PA21_RI1      ((unsigned int) AT91C_PIO_PA21) /*  USART 1 Ring Indicator*/
#define AT91C_PA21_TIOB2    ((unsigned int) AT91C_PIO_PA21) /*  Timer Counter 2 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) /* Pin Controlled by PA22*/
#define AT91C_PA22_RXD2     ((unsigned int) AT91C_PIO_PA22) /*  USART 2 Receive Data*/
#define AT91C_PA22_URXD     ((unsigned int) AT91C_PIO_PA22) /*  USB device RXD*/
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) /* Pin Controlled by PA23*/
#define AT91C_PA23_TXD2     ((unsigned int) AT91C_PIO_PA23) /*  USART 2 Transmit Data*/
#define AT91C_PA23_UTXD     ((unsigned int) AT91C_PIO_PA23) /*  USB device TXD*/
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) /* Pin Controlled by PA24*/
#define AT91C_PA24_MCCK     ((unsigned int) AT91C_PIO_PA24) /*  Multimedia Card Clock*/
#define AT91C_PA24_RTS0     ((unsigned int) AT91C_PIO_PA24) /*  Usart 0 Ready To Send*/
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) /* Pin Controlled by PA25*/
#define AT91C_PA25_MCCDA    ((unsigned int) AT91C_PIO_PA25) /*  Multimedia Card A Command*/
#define AT91C_PA25_RTS1     ((unsigned int) AT91C_PIO_PA25) /*  Usart 0 Ready To Send*/
#define AT91C_PIO_PA26       ((unsigned int) 1 << 26) /* Pin Controlled by PA26*/
#define AT91C_PA26_MCDA0    ((unsigned int) AT91C_PIO_PA26) /*  Multimedia Card A Data 0*/
#define AT91C_PIO_PA27       ((unsigned int) 1 << 27) /* Pin Controlled by PA27*/
#define AT91C_PA27_MCDA1    ((unsigned int) AT91C_PIO_PA27) /*  Multimedia Card A Data 1*/
#define AT91C_PA27_UEON     ((unsigned int) AT91C_PIO_PA27) /*  USB Device UEON*/
#define AT91C_PIO_PA28       ((unsigned int) 1 << 28) /* Pin Controlled by PA28*/
#define AT91C_PA28_MCDA2    ((unsigned int) AT91C_PIO_PA28) /*  Multimedia Card A Data 2*/
#define AT91C_PA28_RTS2     ((unsigned int) AT91C_PIO_PA28) /*  Usart 2 Ready To Send*/
#define AT91C_PIO_PA29       ((unsigned int) 1 << 29) /* Pin Controlled by PA29*/
#define AT91C_PA29_MCDA3    ((unsigned int) AT91C_PIO_PA29) /*  Multimedia Card A Data 3*/
#define AT91C_PA29_CTS2     ((unsigned int) AT91C_PIO_PA29) /*  Usart 2 Clear To Send*/
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) /* Pin Controlled by PA3*/
#define AT91C_PA3_NPCS0    ((unsigned int) AT91C_PIO_PA3) /*  SPI Peripheral Chip Select 0*/
#define AT91C_PA3_PCK1     ((unsigned int) AT91C_PIO_PA3) /*  PMC Programmable clock Output 1*/
#define AT91C_PIO_PA30       ((unsigned int) 1 << 30) /* Pin Controlled by PA30*/
#define AT91C_PA30_DRXD     ((unsigned int) AT91C_PIO_PA30) /*  DBGU Debug Receive Data*/
#define AT91C_PIO_PA31       ((unsigned int) 1 << 31) /* Pin Controlled by PA31*/
#define AT91C_PA31_DTXD     ((unsigned int) AT91C_PIO_PA31) /*  DBGU Debug Transmit Data*/
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) /* Pin Controlled by PA4*/
#define AT91C_PA4_NPCS1    ((unsigned int) AT91C_PIO_PA4) /*  SPI Peripheral Chip Select 1*/
#define AT91C_PA4_USUSPEND ((unsigned int) AT91C_PIO_PA4) /*  USB device suspend*/
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) /* Pin Controlled by PA5*/
#define AT91C_PA5_NPCS2    ((unsigned int) AT91C_PIO_PA5) /*  SPI Peripheral Chip Select 2*/
#define AT91C_PA5_SCK1     ((unsigned int) AT91C_PIO_PA5) /*  USART1 Serial Clock*/
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) /* Pin Controlled by PA6*/
#define AT91C_PA6_NPCS3    ((unsigned int) AT91C_PIO_PA6) /*  SPI Peripheral Chip Select 3*/
#define AT91C_PA6_SCK2     ((unsigned int) AT91C_PIO_PA6) /*  USART2 Serial Clock*/
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) /* Pin Controlled by PA7*/
#define AT91C_PA7_TWD      ((unsigned int) AT91C_PIO_PA7) /*  TWI Two-wire Serial Data*/
#define AT91C_PA7_PCK2     ((unsigned int) AT91C_PIO_PA7) /*  PMC Programmable Clock 2*/
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) /* Pin Controlled by PA8*/
#define AT91C_PA8_TWCK     ((unsigned int) AT91C_PIO_PA8) /*  TWI Two-wire Serial Clock*/
#define AT91C_PA8_PCK3     ((unsigned int) AT91C_PIO_PA8) /*  PMC Programmable Clock 3*/
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) /* Pin Controlled by PA9*/
#define AT91C_PA9_TXD0     ((unsigned int) AT91C_PIO_PA9) /*  USART 0 Transmit Data*/
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) /* Pin Controlled by PB0*/
#define AT91C_PB0_TF0      ((unsigned int) AT91C_PIO_PB0) /*  SSC Transmit Frame Sync 0*/
#define AT91C_PB0_TIOB3    ((unsigned int) AT91C_PIO_PB0) /*  Timer Counter 3 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) /* Pin Controlled by PB1*/
#define AT91C_PB1_TK0      ((unsigned int) AT91C_PIO_PB1) /*  SSC Transmit Clock 0*/
#define AT91C_PB1_TCLK3    ((unsigned int) AT91C_PIO_PB1) /*  Timer Counter 3 External Clock Input*/
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) /* Pin Controlled by PB10*/
#define AT91C_PB10_RK1      ((unsigned int) AT91C_PIO_PB10) /*  SSC Receive Clock 1*/
#define AT91C_PB10_PCK1     ((unsigned int) AT91C_PIO_PB10) /*  PMC Programmable Clock Output 1*/
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) /* Pin Controlled by PB11*/
#define AT91C_PB11_RF1      ((unsigned int) AT91C_PIO_PB11) /*  SSC Receive Frame Sync 1*/
#define AT91C_PB11_TIOA4    ((unsigned int) AT91C_PIO_PB11) /*  Timer Counter 4 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) /* Pin Controlled by PB12*/
#define AT91C_PB12_TF2      ((unsigned int) AT91C_PIO_PB12) /*  SSC Transmit Frame Sync 2*/
#define AT91C_PB12_TIOB5    ((unsigned int) AT91C_PIO_PB12) /*  Timer Counter 5 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) /* Pin Controlled by PB13*/
#define AT91C_PB13_TK2      ((unsigned int) AT91C_PIO_PB13) /*  SSC Transmit Clock 2*/
#define AT91C_PB13_TCLK5    ((unsigned int) AT91C_PIO_PB13) /*  Timer Counter 5 external clock input*/
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) /* Pin Controlled by PB14*/
#define AT91C_PB14_TD2      ((unsigned int) AT91C_PIO_PB14) /*  SSC Transmit Data 2*/
#define AT91C_PB14_NPCS3    ((unsigned int) AT91C_PIO_PB14) /*  SPI Peripheral Chip Select 3*/
#define AT91C_PIO_PB15       ((unsigned int) 1 << 15) /* Pin Controlled by PB15*/
#define AT91C_PB15_RD2      ((unsigned int) AT91C_PIO_PB15) /*  SSC Receive Data 2*/
#define AT91C_PB15_PCK1     ((unsigned int) AT91C_PIO_PB15) /*  PMC Programmable Clock Output 1*/
#define AT91C_PIO_PB16       ((unsigned int) 1 << 16) /* Pin Controlled by PB16*/
#define AT91C_PB16_RK2      ((unsigned int) AT91C_PIO_PB16) /*  SSC Receive Clock 2*/
#define AT91C_PB16_PCK2     ((unsigned int) AT91C_PIO_PB16) /*  PMC Programmable Clock Output 2*/
#define AT91C_PIO_PB17       ((unsigned int) 1 << 17) /* Pin Controlled by PB17*/
#define AT91C_PB17_RF2      ((unsigned int) AT91C_PIO_PB17) /*  SSC Receive Frame Sync 2*/
#define AT91C_PB17_TIOA5    ((unsigned int) AT91C_PIO_PB17) /*  Timer Counter 5 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PB18       ((unsigned int) 1 << 18) /* Pin Controlled by PB18*/
#define AT91C_PB18_RTS3     ((unsigned int) AT91C_PIO_PB18) /*  USART 3 Ready To Send*/
#define AT91C_PB18_MCCDB    ((unsigned int) AT91C_PIO_PB18) /*  Multimedia Card B Command*/
#define AT91C_PIO_PB19       ((unsigned int) 1 << 19) /* Pin Controlled by PB19*/
#define AT91C_PB19_CTS3     ((unsigned int) AT91C_PIO_PB19) /*  USART 3 Clear To Send*/
#define AT91C_PB19_MCDB0    ((unsigned int) AT91C_PIO_PB19) /*  Multimedia Card B Data 0*/
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) /* Pin Controlled by PB2*/
#define AT91C_PB2_TD0      ((unsigned int) AT91C_PIO_PB2) /*  SSC Transmit data*/
#define AT91C_PB2_RTS2     ((unsigned int) AT91C_PIO_PB2) /*  USART 2 Ready To Send*/
#define AT91C_PIO_PB20       ((unsigned int) 1 << 20) /* Pin Controlled by PB20*/
#define AT91C_PB20_TXD3     ((unsigned int) AT91C_PIO_PB20) /*  USART 3 Transmit Data*/
#define AT91C_PB20_DTR1     ((unsigned int) AT91C_PIO_PB20) /*  USART 1 Data Terminal Ready*/
#define AT91C_PIO_PB21       ((unsigned int) 1 << 21) /* Pin Controlled by PB21*/
#define AT91C_PB21_RXD3     ((unsigned int) AT91C_PIO_PB21) /*  USART 3 Receive Data*/
#define AT91C_PIO_PB22       ((unsigned int) 1 << 22) /* Pin Controlled by PB22*/
#define AT91C_PB22_SCK3     ((unsigned int) AT91C_PIO_PB22) /*  USART 3 Serial Clock*/
#define AT91C_PB22_PCK3     ((unsigned int) AT91C_PIO_PB22) /*  PMC Programmable Clock Output 3*/
#define AT91C_PIO_PB23       ((unsigned int) 1 << 23) /* Pin Controlled by PB23*/
#define AT91C_PB23_FIQ      ((unsigned int) AT91C_PIO_PB23) /*  AIC Fast Interrupt Input*/
#define AT91C_PIO_PB24       ((unsigned int) 1 << 24) /* Pin Controlled by PB24*/
#define AT91C_PB24_IRQ0     ((unsigned int) AT91C_PIO_PB24) /*  Interrupt input 0*/
#define AT91C_PB24_TD0      ((unsigned int) AT91C_PIO_PB24) /*  SSC Transmit Data 0*/
#define AT91C_PIO_PB25       ((unsigned int) 1 << 25) /* Pin Controlled by PB25*/
#define AT91C_PB25_IRQ1     ((unsigned int) AT91C_PIO_PB25) /*  Interrupt input 1*/
#define AT91C_PB25_TD1      ((unsigned int) AT91C_PIO_PB25) /*  SSC Transmit Data 1*/
#define AT91C_PIO_PB26       ((unsigned int) 1 << 26) /* Pin Controlled by PB26*/
#define AT91C_PB26_IRQ2     ((unsigned int) AT91C_PIO_PB26) /*  Interrupt input 2*/
#define AT91C_PB26_TD2      ((unsigned int) AT91C_PIO_PB26) /*  SSC Transmit Data 2*/
#define AT91C_PIO_PB27       ((unsigned int) 1 << 27) /* Pin Controlled by PB27*/
#define AT91C_PB27_IRQ3     ((unsigned int) AT91C_PIO_PB27) /*  Interrupt input 3*/
#define AT91C_PB27_DTXD     ((unsigned int) AT91C_PIO_PB27) /*  Debug Unit Transmit Data*/
#define AT91C_PIO_PB28       ((unsigned int) 1 << 28) /* Pin Controlled by PB28*/
#define AT91C_PB28_IRQ4     ((unsigned int) AT91C_PIO_PB28) /*  Interrupt input 4*/
#define AT91C_PB28_MCDB1    ((unsigned int) AT91C_PIO_PB28) /*  Multimedia Card B Data 1*/
#define AT91C_PIO_PB29       ((unsigned int) 1 << 29) /* Pin Controlled by PB29*/
#define AT91C_PB29_IRQ5     ((unsigned int) AT91C_PIO_PB29) /*  Interrupt input 5*/
#define AT91C_PB29_MCDB2    ((unsigned int) AT91C_PIO_PB29) /*  Multimedia Card B Data 2*/
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) /* Pin Controlled by PB3*/
#define AT91C_PB3_RD0      ((unsigned int) AT91C_PIO_PB3) /*  SSC Receive Data*/
#define AT91C_PB3_RTS3     ((unsigned int) AT91C_PIO_PB3) /*  USART 3 Ready To Send*/
#define AT91C_PIO_PB30       ((unsigned int) 1 << 30) /* Pin Controlled by PB30*/
#define AT91C_PB30_IRQ6     ((unsigned int) AT91C_PIO_PB30) /*  Interrupt input 6*/
#define AT91C_PB30_MCDB3    ((unsigned int) AT91C_PIO_PB30) /*  Multimedia Card B Data 3*/
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) /* Pin Controlled by PB4*/
#define AT91C_PB4_RK0      ((unsigned int) AT91C_PIO_PB4) /*  SSC Receive Clock*/
#define AT91C_PB4_PCK0     ((unsigned int) AT91C_PIO_PB4) /*  PMC Programmable Clock Output 0*/
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) /* Pin Controlled by PB5*/
#define AT91C_PB5_RF0      ((unsigned int) AT91C_PIO_PB5) /*  SSC Receive Frame Sync 0*/
#define AT91C_PB5_TIOA3    ((unsigned int) AT91C_PIO_PB5) /*  Timer Counter 4 Multipurpose Timer I/O Pin A*/
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) /* Pin Controlled by PB6*/
#define AT91C_PB6_TF1      ((unsigned int) AT91C_PIO_PB6) /*  SSC Transmit Frame Sync 1*/
#define AT91C_PB6_TIOB4    ((unsigned int) AT91C_PIO_PB6) /*  Timer Counter 4 Multipurpose Timer I/O Pin B*/
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) /* Pin Controlled by PB7*/
#define AT91C_PB7_TK1      ((unsigned int) AT91C_PIO_PB7) /*  SSC Transmit Clock 1*/
#define AT91C_PB7_TCLK4    ((unsigned int) AT91C_PIO_PB7) /*  Timer Counter 4 external Clock Input*/
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) /* Pin Controlled by PB8*/
#define AT91C_PB8_TD1      ((unsigned int) AT91C_PIO_PB8) /*  SSC Transmit Data 1*/
#define AT91C_PB8_NPCS1    ((unsigned int) AT91C_PIO_PB8) /*  SPI Peripheral Chip Select 1*/
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) /* Pin Controlled by PB9*/
#define AT91C_PB9_RD1      ((unsigned int) AT91C_PIO_PB9) /*  SSC Receive Data 1*/
#define AT91C_PB9_NPCS2    ((unsigned int) AT91C_PIO_PB9) /*  SPI Peripheral Chip Select 2*/

/* ******************************************************************************/
/*               PERIPHERAL ID DEFINITIONS FOR AT91RM3400*/
/* ******************************************************************************/
#define AT91C_ID_FIQ    ((unsigned int)  0) /* Advanced Interrupt Controller (FIQ)*/
#define AT91C_ID_SYS    ((unsigned int)  1) /* System Peripheral*/
#define AT91C_ID_PIOA   ((unsigned int)  2) /* Parallel IO Controller A */
#define AT91C_ID_PIOB   ((unsigned int)  3) /* Parallel IO Controller B*/
#define AT91C_ID_US0    ((unsigned int)  6) /* USART 0*/
#define AT91C_ID_US1    ((unsigned int)  7) /* USART 1*/
#define AT91C_ID_US2    ((unsigned int)  8) /* USART 2*/
#define AT91C_ID_US3    ((unsigned int)  9) /* USART 3*/
#define AT91C_ID_MCI    ((unsigned int) 10) /* Multimedia Card Interface*/
#define AT91C_ID_UDP    ((unsigned int) 11) /* USB Device Port*/
#define AT91C_ID_TWI    ((unsigned int) 12) /* Two-Wire Interface*/
#define AT91C_ID_SPI    ((unsigned int) 13) /* Serial Peripheral Interface*/
#define AT91C_ID_SSC0   ((unsigned int) 14) /* Serial Synchronous Controller 0*/
#define AT91C_ID_SSC1   ((unsigned int) 15) /* Serial Synchronous Controller 1*/
#define AT91C_ID_SSC2   ((unsigned int) 16) /* Serial Synchronous Controller 2*/
#define AT91C_ID_TC0    ((unsigned int) 17) /* Timer Counter 0*/
#define AT91C_ID_TC1    ((unsigned int) 18) /* Timer Counter 1*/
#define AT91C_ID_TC2    ((unsigned int) 19) /* Timer Counter 2*/
#define AT91C_ID_TC3    ((unsigned int) 20) /* Timer Counter 3*/
#define AT91C_ID_TC4    ((unsigned int) 21) /* Timer Counter 4*/
#define AT91C_ID_TC5    ((unsigned int) 22) /* Timer Counter 5*/
#define AT91C_ID_IRQ0   ((unsigned int) 25) /* Advanced Interrupt Controller (IRQ0)*/
#define AT91C_ID_IRQ1   ((unsigned int) 26) /* Advanced Interrupt Controller (IRQ1)*/
#define AT91C_ID_IRQ2   ((unsigned int) 27) /* Advanced Interrupt Controller (IRQ2)*/
#define AT91C_ID_IRQ3   ((unsigned int) 28) /* Advanced Interrupt Controller (IRQ3)*/
#define AT91C_ID_IRQ4   ((unsigned int) 29) /* Advanced Interrupt Controller (IRQ4)*/
#define AT91C_ID_IRQ5   ((unsigned int) 30) /* Advanced Interrupt Controller (IRQ5)*/
#define AT91C_ID_IRQ6   ((unsigned int) 31) /* Advanced Interrupt Controller (IRQ6)*/
#define AT91C_ALL_INT   ((unsigned int) 0xFE7FFFCF) /* ALL VALID INTERRUPTS*/

/* ******************************************************************************/
/*               BASE ADDRESS DEFINITIONS FOR AT91RM3400*/
/* ******************************************************************************/
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0xFFFFF000) /* (SYS) Base Address*/
#define AT91C_BASE_MC        ((AT91PS_MC) 	0xFFFFFF00) /* (MC) Base Address*/
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0xFFFFFE00) /* (RTC) Base Address*/
#define AT91C_BASE_ST        ((AT91PS_ST) 	0xFFFFFD00) /* (ST) Base Address*/
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0xFFFFFC00) /* (PMC) Base Address*/
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0xFFFFFC20) /* (CKGR) Base Address*/
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0xFFFFF600) /* (PIOB) Base Address*/
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0xFFFFF400) /* (PIOA) Base Address*/
#define AT91C_BASE_DBGU      ((AT91PS_DBGU) 	0xFFFFF200) /* (DBGU) Base Address*/
#define AT91C_BASE_PDC_DBGU  ((AT91PS_PDC) 	0xFFFFF300) /* (PDC_DBGU) Base Address*/
#define AT91C_BASE_AIC       ((AT91PS_AIC) 	0xFFFFF000) /* (AIC) Base Address*/
#define AT91C_BASE_PDC_SPI   ((AT91PS_PDC) 	0xFFFE0100) /* (PDC_SPI) Base Address*/
#define AT91C_BASE_SPI       ((AT91PS_SPI) 	0xFFFE0000) /* (SPI) Base Address*/
#define AT91C_BASE_PDC_SSC2  ((AT91PS_PDC) 	0xFFFD8100) /* (PDC_SSC2) Base Address*/
#define AT91C_BASE_SSC2      ((AT91PS_SSC) 	0xFFFD8000) /* (SSC2) Base Address*/
#define AT91C_BASE_PDC_SSC1  ((AT91PS_PDC) 	0xFFFD4100) /* (PDC_SSC1) Base Address*/
#define AT91C_BASE_SSC1      ((AT91PS_SSC) 	0xFFFD4000) /* (SSC1) Base Address*/
#define AT91C_BASE_PDC_SSC0  ((AT91PS_PDC) 	0xFFFD0100) /* (PDC_SSC0) Base Address*/
#define AT91C_BASE_SSC0      ((AT91PS_SSC) 	0xFFFD0000) /* (SSC0) Base Address*/
#define AT91C_BASE_PDC_US3   ((AT91PS_PDC) 	0xFFFCC100) /* (PDC_US3) Base Address*/
#define AT91C_BASE_US3       ((AT91PS_USART) 	0xFFFCC000) /* (US3) Base Address*/
#define AT91C_BASE_PDC_US2   ((AT91PS_PDC) 	0xFFFC8100) /* (PDC_US2) Base Address*/
#define AT91C_BASE_US2       ((AT91PS_USART) 	0xFFFC8000) /* (US2) Base Address*/
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0xFFFC4100) /* (PDC_US1) Base Address*/
#define AT91C_BASE_US1       ((AT91PS_USART) 	0xFFFC4000) /* (US1) Base Address*/
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0xFFFC0100) /* (PDC_US0) Base Address*/
#define AT91C_BASE_US0       ((AT91PS_USART) 	0xFFFC0000) /* (US0) Base Address*/
#define AT91C_BASE_TWI       ((AT91PS_TWI) 	0xFFFB8000) /* (TWI) Base Address*/
#define AT91C_BASE_PDC_MCI   ((AT91PS_PDC) 	0xFFFB4100) /* (PDC_MCI) Base Address*/
#define AT91C_BASE_MCI       ((AT91PS_MCI) 	0xFFFB4000) /* (MCI) Base Address*/
#define AT91C_BASE_UDP       ((AT91PS_UDP) 	0xFFFB0000) /* (UDP) Base Address*/
#define AT91C_BASE_TC5       ((AT91PS_TC) 	0xFFFA4080) /* (TC5) Base Address*/
#define AT91C_BASE_TC4       ((AT91PS_TC) 	0xFFFA4040) /* (TC4) Base Address*/
#define AT91C_BASE_TC3       ((AT91PS_TC) 	0xFFFA4000) /* (TC3) Base Address*/
#define AT91C_BASE_TCB1      ((AT91PS_TCB) 	0xFFFA4000) /* (TCB1) Base Address*/
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0xFFFA0080) /* (TC2) Base Address*/
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0xFFFA0040) /* (TC1) Base Address*/
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0xFFFA0000) /* (TC0) Base Address*/
#define AT91C_BASE_TCB0      ((AT91PS_TCB) 	0xFFFA0000) /* (TCB0) Base Address*/

/* ******************************************************************************/
/*               MEMORY MAPPING DEFINITIONS FOR AT91RM3400*/
/* ******************************************************************************/
/* ISRAM*/
#define AT91C_ISRAM	 ((char *) 	0x00200000) /* Internal SRAM base address*/
#define AT91C_ISRAM_SIZE	 ((unsigned int) 0x00018000) /* Internal SRAM size in byte (96 Kbytes)*/
/* IROM*/
#define AT91C_IROM 	 ((char *) 	0x00100000) /* Internal ROM base address*/
#define AT91C_IROM_SIZE	 ((unsigned int) 0x00040000) /* Internal ROM size in byte (256 Kbytes)*/
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

/* - Hardware register definition*/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR System Peripherals*/
/* - ******************************************************************************/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Memory Controller Interface*/
/* - ******************************************************************************/
/* - -------- MC_RCR : (MC Offset: 0x0) MC Remap Control Register -------- */
AT91C_MC_RCB              EQU (0x1 <<  0) ;- (MC) Remap Command Bit
/* - -------- MC_ASR : (MC Offset: 0x4) MC Abort Status Register -------- */
AT91C_MC_UNDADD           EQU (0x1 <<  0) ;- (MC) Undefined Addess Abort Status
AT91C_MC_MISADD           EQU (0x1 <<  1) ;- (MC) Misaligned Addess Abort Status
AT91C_MC_MPU              EQU (0x1 <<  2) ;- (MC) Memory protection Unit Abort Status
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
/* - -------- MC_PUIA : (MC Offset: 0x10) MC Protection Unit Area -------- */
AT91C_MC_PROT             EQU (0x3 <<  0) ;- (MC) Protection
AT91C_MC_PROT_PNAUNA      EQU (0x0) ;- (MC) Privilege: No Access, User: No Access
AT91C_MC_PROT_PRWUNA      EQU (0x1) ;- (MC) Privilege: Read/Write, User: No Access
AT91C_MC_PROT_PRWURO      EQU (0x2) ;- (MC) Privilege: Read/Write, User: Read Only
AT91C_MC_PROT_PRWURW      EQU (0x3) ;- (MC) Privilege: Read/Write, User: Read/Write
AT91C_MC_SIZE             EQU (0xF <<  4) ;- (MC) Internal Area Size
AT91C_MC_SIZE_1KB         EQU (0x0 <<  4) ;- (MC) Area size 1KByte
AT91C_MC_SIZE_2KB         EQU (0x1 <<  4) ;- (MC) Area size 2KByte
AT91C_MC_SIZE_4KB         EQU (0x2 <<  4) ;- (MC) Area size 4KByte
AT91C_MC_SIZE_8KB         EQU (0x3 <<  4) ;- (MC) Area size 8KByte
AT91C_MC_SIZE_16KB        EQU (0x4 <<  4) ;- (MC) Area size 16KByte
AT91C_MC_SIZE_32KB        EQU (0x5 <<  4) ;- (MC) Area size 32KByte
AT91C_MC_SIZE_64KB        EQU (0x6 <<  4) ;- (MC) Area size 64KByte
AT91C_MC_SIZE_128KB       EQU (0x7 <<  4) ;- (MC) Area size 128KByte
AT91C_MC_SIZE_256KB       EQU (0x8 <<  4) ;- (MC) Area size 256KByte
AT91C_MC_SIZE_512KB       EQU (0x9 <<  4) ;- (MC) Area size 512KByte
AT91C_MC_SIZE_1MB         EQU (0xA <<  4) ;- (MC) Area size 1MByte
AT91C_MC_SIZE_2MB         EQU (0xB <<  4) ;- (MC) Area size 2MByte
AT91C_MC_SIZE_4MB         EQU (0xC <<  4) ;- (MC) Area size 4MByte
AT91C_MC_SIZE_8MB         EQU (0xD <<  4) ;- (MC) Area size 8MByte
AT91C_MC_SIZE_16MB        EQU (0xE <<  4) ;- (MC) Area size 16MByte
AT91C_MC_SIZE_64MB        EQU (0xF <<  4) ;- (MC) Area size 64MByte
AT91C_MC_BA               EQU (0x3FFFF << 10) ;- (MC) Internal Area Base Address
/* - -------- MC_PUP : (MC Offset: 0x50) MC Protection Unit Peripheral -------- */
/* - -------- MC_PUER : (MC Offset: 0x54) MC Protection Unit Area -------- */
AT91C_MC_PUEB             EQU (0x1 <<  0) ;- (MC) Protection Unit enable Bit

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Real-time Clock Alarm and Parallel Load Interface*/
/* - ******************************************************************************/
/* - -------- RTC_CR : (RTC Offset: 0x0) RTC Control Register -------- */
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
/* - -------- RTC_MR : (RTC Offset: 0x4) RTC Mode Register -------- */
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
/* - -------- RTC_TIMALR : (RTC Offset: 0x10) RTC Time Alarm Register -------- */
AT91C_RTC_SECEN           EQU (0x1 <<  7) ;- (RTC) Second Alarm Enable
AT91C_RTC_MINEN           EQU (0x1 << 15) ;- (RTC) Minute Alarm
AT91C_RTC_HOUREN          EQU (0x1 << 23) ;- (RTC) Current Hour
/* - -------- RTC_CALALR : (RTC Offset: 0x14) RTC Calendar Alarm Register -------- */
AT91C_RTC_MONTHEN         EQU (0x1 << 23) ;- (RTC) Month Alarm Enable
AT91C_RTC_DATEEN          EQU (0x1 << 31) ;- (RTC) Date Alarm Enable
/* - -------- RTC_SR : (RTC Offset: 0x18) RTC Status Register -------- */
AT91C_RTC_ACKUPD          EQU (0x1 <<  0) ;- (RTC) Acknowledge for Update
AT91C_RTC_ALARM           EQU (0x1 <<  1) ;- (RTC) Alarm Flag
AT91C_RTC_SECEV           EQU (0x1 <<  2) ;- (RTC) Second Event
AT91C_RTC_TIMEV           EQU (0x1 <<  3) ;- (RTC) Time Event
AT91C_RTC_CALEV           EQU (0x1 <<  4) ;- (RTC) Calendar event
/* - -------- RTC_SCCR : (RTC Offset: 0x1c) RTC Status Clear Command Register -------- */
/* - -------- RTC_IER : (RTC Offset: 0x20) RTC Interrupt Enable Register -------- */
/* - -------- RTC_IDR : (RTC Offset: 0x24) RTC Interrupt Disable Register -------- */
/* - -------- RTC_IMR : (RTC Offset: 0x28) RTC Interrupt Mask Register -------- */
/* - -------- RTC_VER : (RTC Offset: 0x2c) RTC Valid Entry Register -------- */
AT91C_RTC_NVTIM           EQU (0x1 <<  0) ;- (RTC) Non valid Time
AT91C_RTC_NVCAL           EQU (0x1 <<  1) ;- (RTC) Non valid Calendar
AT91C_RTC_NVTIMALR        EQU (0x1 <<  2) ;- (RTC) Non valid time Alarm
AT91C_RTC_NVCALALR        EQU (0x1 <<  3) ;- (RTC) Nonvalid Calendar Alarm

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR System Timer Interface*/
/* - ******************************************************************************/
/* - -------- ST_CR : (ST Offset: 0x0) System Timer Control Register -------- */
AT91C_ST_WDRST            EQU (0x1 <<  0) ;- (ST) Watchdog Timer Restart
/* - -------- ST_PIMR : (ST Offset: 0x4) System Timer Period Interval Mode Register -------- */
AT91C_ST_PIV              EQU (0xFFFF <<  0) ;- (ST) Watchdog Timer Restart
/* - -------- ST_WDMR : (ST Offset: 0x8) System Timer Watchdog Mode Register -------- */
AT91C_ST_WDV              EQU (0xFFFF <<  0) ;- (ST) Watchdog Timer Restart
AT91C_ST_RSTEN            EQU (0x1 << 16) ;- (ST) Reset Enable
AT91C_ST_EXTEN            EQU (0x1 << 17) ;- (ST) External Signal Assertion Enable
/* - -------- ST_RTMR : (ST Offset: 0xc) System Timer Real-time Mode Register -------- */
AT91C_ST_RTPRES           EQU (0xFFFF <<  0) ;- (ST) Real-time Timer Prescaler Value
/* - -------- ST_SR : (ST Offset: 0x10) System Timer Status Register -------- */
AT91C_ST_PITS             EQU (0x1 <<  0) ;- (ST) Period Interval Timer Interrupt
AT91C_ST_WDOVF            EQU (0x1 <<  1) ;- (ST) Watchdog Overflow
AT91C_ST_RTTINC           EQU (0x1 <<  2) ;- (ST) Real-time Timer Increment
AT91C_ST_ALMS             EQU (0x1 <<  3) ;- (ST) Alarm Status
/* - -------- ST_IER : (ST Offset: 0x14) System Timer Interrupt Enable Register -------- */
/* - -------- ST_IDR : (ST Offset: 0x18) System Timer Interrupt Disable Register -------- */
/* - -------- ST_IMR : (ST Offset: 0x1c) System Timer Interrupt Mask Register -------- */
/* - -------- ST_RTAR : (ST Offset: 0x20) System Timer Real-time Alarm Register -------- */
AT91C_ST_ALMV             EQU (0xFFFFF <<  0) ;- (ST) Alarm Value Value
/* - -------- ST_CRTR : (ST Offset: 0x24) System Timer Current Real-time Register -------- */
AT91C_ST_CRTV             EQU (0xFFFFF <<  0) ;- (ST) Current Real-time Value

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Power Management Controler*/
/* - ******************************************************************************/
/* - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
AT91C_PMC_UDP             EQU (0x1 <<  1) ;- (PMC) USB Device Port Clock
AT91C_PMC_MCKUDP          EQU (0x1 <<  2) ;- (PMC) USB Device Port Master Clock Automatic Disable on Suspend
AT91C_PMC_UHP             EQU (0x1 <<  4) ;- (PMC) USB Host Port Clock
AT91C_PMC_PCK0            EQU (0x1 <<  8) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK1            EQU (0x1 <<  9) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK2            EQU (0x1 << 10) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK3            EQU (0x1 << 11) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK4            EQU (0x1 << 12) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK5            EQU (0x1 << 13) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK6            EQU (0x1 << 14) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK7            EQU (0x1 << 15) ;- (PMC) Programmable Clock Output
/* - -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* - -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* - -------- PMC_MCKR : (PMC Offset: 0x30) Master Clock Register -------- */
AT91C_PMC_CSS             EQU (0x3 <<  0) ;- (PMC) Programmable Clock Selection
AT91C_PMC_CSS_SLOW_CLK    EQU (0x0) ;- (PMC) Slow Clock is selected
AT91C_PMC_CSS_MAIN_CLK    EQU (0x1) ;- (PMC) Main Clock is selected
AT91C_PMC_CSS_PLLA_CLK    EQU (0x2) ;- (PMC) Clock from PLL A is selected
AT91C_PMC_CSS_PLLB_CLK    EQU (0x3) ;- (PMC) Clock from PLL B is selected
AT91C_PMC_PRES            EQU (0x7 <<  2) ;- (PMC) Programmable Clock Prescaler
AT91C_PMC_PRES_CLK        EQU (0x0 <<  2) ;- (PMC) Selected clock
AT91C_PMC_PRES_CLK_2      EQU (0x1 <<  2) ;- (PMC) Selected clock divided by 2
AT91C_PMC_PRES_CLK_4      EQU (0x2 <<  2) ;- (PMC) Selected clock divided by 4
AT91C_PMC_PRES_CLK_8      EQU (0x3 <<  2) ;- (PMC) Selected clock divided by 8
AT91C_PMC_PRES_CLK_16     EQU (0x4 <<  2) ;- (PMC) Selected clock divided by 16
AT91C_PMC_PRES_CLK_32     EQU (0x5 <<  2) ;- (PMC) Selected clock divided by 32
AT91C_PMC_PRES_CLK_64     EQU (0x6 <<  2) ;- (PMC) Selected clock divided by 64
AT91C_PMC_MDIV            EQU (0x3 <<  8) ;- (PMC) Master Clock Division
AT91C_PMC_MDIV_1          EQU (0x0 <<  8) ;- (PMC) The master clock and the processor clock are the same
AT91C_PMC_MDIV_2          EQU (0x1 <<  8) ;- (PMC) The processor clock is twice as fast as the master clock
AT91C_PMC_MDIV_3          EQU (0x2 <<  8) ;- (PMC) The processor clock is three times faster than the master clock
AT91C_PMC_MDIV_4          EQU (0x3 <<  8) ;- (PMC) The processor clock is four times faster than the master clock
/* - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
AT91C_PMC_MOSCS           EQU (0x1 <<  0) ;- (PMC) MOSC Status/Enable/Disable/Mask
AT91C_PMC_LOCKA           EQU (0x1 <<  1) ;- (PMC) PLL A Status/Enable/Disable/Mask
AT91C_PMC_LOCKB           EQU (0x1 <<  2) ;- (PMC) PLL B Status/Enable/Disable/Mask
AT91C_PMC_MCKRDY          EQU (0x1 <<  3) ;- (PMC) MCK_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK0RDY         EQU (0x1 <<  8) ;- (PMC) PCK0_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK1RDY         EQU (0x1 <<  9) ;- (PMC) PCK1_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK2RDY         EQU (0x1 << 10) ;- (PMC) PCK2_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK3RDY         EQU (0x1 << 11) ;- (PMC) PCK3_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK4RDY         EQU (0x1 << 12) ;- (PMC) PCK4_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK5RDY         EQU (0x1 << 13) ;- (PMC) PCK5_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK6RDY         EQU (0x1 << 14) ;- (PMC) PCK6_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK7RDY         EQU (0x1 << 15) ;- (PMC) PCK7_RDY Status/Enable/Disable/Mask
/* - -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* - -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* - -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* - ******************************************************************************/
/* - -------- CKGR_MOR : (CKGR Offset: 0x0) Main Oscillator Register -------- */
AT91C_CKGR_MOSCEN         EQU (0x1 <<  0) ;- (CKGR) Main Oscillator Enable
AT91C_CKGR_OSCTEST        EQU (0x1 <<  1) ;- (CKGR) Oscillator Test
AT91C_CKGR_OSCOUNT        EQU (0xFF <<  8) ;- (CKGR) Main Oscillator Start-up Time
/* - -------- CKGR_MCFR : (CKGR Offset: 0x4) Main Clock Frequency Register -------- */
AT91C_CKGR_MAINF          EQU (0xFFFF <<  0) ;- (CKGR) Main Clock Frequency
AT91C_CKGR_MAINRDY        EQU (0x1 << 16) ;- (CKGR) Main Clock Ready
/* - -------- CKGR_PLLAR : (CKGR Offset: 0x8) PLL A Register -------- */
AT91C_CKGR_DIVA           EQU (0xFF <<  0) ;- (CKGR) Divider Selected
AT91C_CKGR_DIVA_0         EQU (0x0) ;- (CKGR) Divider output is 0
AT91C_CKGR_DIVA_BYPASS    EQU (0x1) ;- (CKGR) Divider is bypassed
AT91C_CKGR_PLLACOUNT      EQU (0x3F <<  8) ;- (CKGR) PLL A Counter
AT91C_CKGR_OUTA           EQU (0x3 << 14) ;- (CKGR) PLL A Output Frequency Range
AT91C_CKGR_OUTA_0         EQU (0x0 << 14) ;- (CKGR) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_1         EQU (0x1 << 14) ;- (CKGR) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_2         EQU (0x2 << 14) ;- (CKGR) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_3         EQU (0x3 << 14) ;- (CKGR) Please refer to the PLLA datasheet
AT91C_CKGR_MULA           EQU (0x7FF << 16) ;- (CKGR) PLL A Multiplier
AT91C_CKGR_SRCA           EQU (0x1 << 29) ;- (CKGR) PLL A Source
/* - -------- CKGR_PLLBR : (CKGR Offset: 0xc) PLL B Register -------- */
AT91C_CKGR_DIVB           EQU (0xFF <<  0) ;- (CKGR) Divider Selected
AT91C_CKGR_DIVB_0         EQU (0x0) ;- (CKGR) Divider output is 0
AT91C_CKGR_DIVB_BYPASS    EQU (0x1) ;- (CKGR) Divider is bypassed
AT91C_CKGR_PLLBCOUNT      EQU (0x3F <<  8) ;- (CKGR) PLL B Counter
AT91C_CKGR_OUTB           EQU (0x3 << 14) ;- (CKGR) PLL B Output Frequency Range
AT91C_CKGR_OUTB_0         EQU (0x0 << 14) ;- (CKGR) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_1         EQU (0x1 << 14) ;- (CKGR) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_2         EQU (0x2 << 14) ;- (CKGR) Please refer to the PLLB datasheet
AT91C_CKGR_OUTB_3         EQU (0x3 << 14) ;- (CKGR) Please refer to the PLLB datasheet
AT91C_CKGR_MULB           EQU (0x7FF << 16) ;- (CKGR) PLL B Multiplier
AT91C_CKGR_USB_96M        EQU (0x1 << 28) ;- (CKGR) Divider for USB Ports
AT91C_CKGR_USB_PLL        EQU (0x1 << 29) ;- (CKGR) PLL Use

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Parallel Input Output Controler*/
/* - ******************************************************************************/

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
/* -              SOFTWARE API DEFINITION  FOR Peripheral Data Controller*/
/* - ******************************************************************************/
/* - -------- PDC_PTCR : (PDC Offset: 0x20) PDC Transfer Control Register -------- */
AT91C_PDC_RXTEN           EQU (0x1 <<  0) ;- (PDC) Receiver Transfer Enable
AT91C_PDC_RXTDIS          EQU (0x1 <<  1) ;- (PDC) Receiver Transfer Disable
AT91C_PDC_TXTEN           EQU (0x1 <<  8) ;- (PDC) Transmitter Transfer Enable
AT91C_PDC_TXTDIS          EQU (0x1 <<  9) ;- (PDC) Transmitter Transfer Disable
/* - -------- PDC_PTSR : (PDC Offset: 0x24) PDC Transfer Status Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Advanced Interrupt Controller*/
/* - ******************************************************************************/
/* - -------- AIC_SMR : (AIC Offset: 0x0) Control Register -------- */
AT91C_AIC_PRIOR           EQU (0x7 <<  0) ;- (AIC) Priority Level
AT91C_AIC_PRIOR_LOWEST    EQU (0x0) ;- (AIC) Lowest priority level
AT91C_AIC_PRIOR_HIGHEST   EQU (0x7) ;- (AIC) Highest priority level
AT91C_AIC_SRCTYPE         EQU (0x3 <<  5) ;- (AIC) Interrupt Source Type
AT91C_AIC_SRCTYPE_INT_LEVEL_SENSITIVE EQU (0x0 <<  5) ;- (AIC) Internal Sources Code Label Level Sensitive
AT91C_AIC_SRCTYPE_INT_EDGE_TRIGGERED EQU (0x1 <<  5) ;- (AIC) Internal Sources Code Label Edge triggered
AT91C_AIC_SRCTYPE_EXT_HIGH_LEVEL EQU (0x2 <<  5) ;- (AIC) External Sources Code Label High-level Sensitive
AT91C_AIC_SRCTYPE_EXT_POSITIVE_EDGE EQU (0x3 <<  5) ;- (AIC) External Sources Code Label Positive Edge triggered
/* - -------- AIC_CISR : (AIC Offset: 0x114) AIC Core Interrupt Status Register -------- */
AT91C_AIC_NFIQ            EQU (0x1 <<  0) ;- (AIC) NFIQ Status
AT91C_AIC_NIRQ            EQU (0x1 <<  1) ;- (AIC) NIRQ Status
/* - -------- AIC_DCR : (AIC Offset: 0x138) AIC Debug Control Register (Protect) -------- */
AT91C_AIC_DCR_PROT        EQU (0x1 <<  0) ;- (AIC) Protection Mode
AT91C_AIC_DCR_GMSK        EQU (0x1 <<  1) ;- (AIC) General Mask

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Serial Parallel Interface*/
/* - ******************************************************************************/
/* - -------- SPI_CR : (SPI Offset: 0x0) SPI Control Register -------- */
AT91C_SPI_SPIEN           EQU (0x1 <<  0) ;- (SPI) SPI Enable
AT91C_SPI_SPIDIS          EQU (0x1 <<  1) ;- (SPI) SPI Disable
AT91C_SPI_SWRST           EQU (0x1 <<  7) ;- (SPI) SPI Software reset
/* - -------- SPI_MR : (SPI Offset: 0x4) SPI Mode Register -------- */
AT91C_SPI_MSTR            EQU (0x1 <<  0) ;- (SPI) Master/Slave Mode
AT91C_SPI_PS              EQU (0x1 <<  1) ;- (SPI) Peripheral Select
AT91C_SPI_PS_FIXED        EQU (0x0 <<  1) ;- (SPI) Fixed Peripheral Select
AT91C_SPI_PS_VARIABLE     EQU (0x1 <<  1) ;- (SPI) Variable Peripheral Select
AT91C_SPI_PCSDEC          EQU (0x1 <<  2) ;- (SPI) Chip Select Decode
AT91C_SPI_DIV32           EQU (0x1 <<  3) ;- (SPI) Clock Selection
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
AT91C_SPI_SPENDRX         EQU (0x1 <<  4) ;- (SPI) End of Receiver Transfer
AT91C_SPI_SPENDTX         EQU (0x1 <<  5) ;- (SPI) End of Receiver Transfer
AT91C_SPI_RXBUFF          EQU (0x1 <<  6) ;- (SPI) RXBUFF Interrupt
AT91C_SPI_TXBUFE          EQU (0x1 <<  7) ;- (SPI) TXBUFE Interrupt
AT91C_SPI_SPIENS          EQU (0x1 << 16) ;- (SPI) Enable Status
/* - -------- SPI_IER : (SPI Offset: 0x14) Interrupt Enable Register -------- */
/* - -------- SPI_IDR : (SPI Offset: 0x18) Interrupt Disable Register -------- */
/* - -------- SPI_IMR : (SPI Offset: 0x1c) Interrupt Mask Register -------- */
/* - -------- SPI_CSR : (SPI Offset: 0x30) Chip Select Register -------- */
AT91C_SPI_CPOL            EQU (0x1 <<  0) ;- (SPI) Clock Polarity
AT91C_SPI_NCPHA           EQU (0x1 <<  1) ;- (SPI) Clock Phase
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
AT91C_SPI_DLYBS           EQU (0xFF << 16) ;- (SPI) Serial Clock Baud Rate
AT91C_SPI_DLYBCT          EQU (0xFF << 24) ;- (SPI) Delay Between Consecutive Transfers

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Synchronous Serial Controller Interface*/
/* - ******************************************************************************/
/* - -------- SSC_CR : (SSC Offset: 0x0) SSC Control Register -------- */
AT91C_SSC_RXEN            EQU (0x1 <<  0) ;- (SSC) Receive Enable
AT91C_SSC_RXDIS           EQU (0x1 <<  1) ;- (SSC) Receive Disable
AT91C_SSC_TXEN            EQU (0x1 <<  8) ;- (SSC) Transmit Enable
AT91C_SSC_TXDIS           EQU (0x1 <<  9) ;- (SSC) Transmit Disable
AT91C_SSC_SWRST           EQU (0x1 << 15) ;- (SSC) Software Reset
/* - -------- SSC_RCMR : (SSC Offset: 0x10) SSC Receive Clock Mode Register -------- */
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
/* - -------- SSC_RFMR : (SSC Offset: 0x14) SSC Receive Frame Mode Register -------- */
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
/* - -------- SSC_TCMR : (SSC Offset: 0x18) SSC Transmit Clock Mode Register -------- */
/* - -------- SSC_TFMR : (SSC Offset: 0x1c) SSC Transmit Frame Mode Register -------- */
AT91C_SSC_DATDEF          EQU (0x1 <<  5) ;- (SSC) Data Default Value
AT91C_SSC_FSDEN           EQU (0x1 << 23) ;- (SSC) Frame Sync Data Enable
/* - -------- SSC_SR : (SSC Offset: 0x40) SSC Status Register -------- */
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
/* - -------- SSC_IER : (SSC Offset: 0x44) SSC Interrupt Enable Register -------- */
/* - -------- SSC_IDR : (SSC Offset: 0x48) SSC Interrupt Disable Register -------- */
/* - -------- SSC_IMR : (SSC Offset: 0x4c) SSC Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Usart*/
/* - ******************************************************************************/
/* - -------- US_CR : (USART Offset: 0x0) Debug Unit Control Register -------- */
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
AT91C_TWI_SVEN            EQU (0x1 <<  4) ;- (TWI) TWI Slave Transfer Enabled
AT91C_TWI_SVDIS           EQU (0x1 <<  5) ;- (TWI) TWI Slave Transfer Disabled
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
AT91C_TWI_SADR            EQU (0x7F << 16) ;- (TWI) Slave Device Address
/* - -------- TWI_CWGR : (TWI Offset: 0x10) TWI Clock Waveform Generator Register -------- */
AT91C_TWI_CLDIV           EQU (0xFF <<  0) ;- (TWI) Clock Low Divider
AT91C_TWI_CHDIV           EQU (0xFF <<  8) ;- (TWI) Clock High Divider
AT91C_TWI_CKDIV           EQU (0x7 << 16) ;- (TWI) Clock Divider
/* - -------- TWI_SR : (TWI Offset: 0x20) TWI Status Register -------- */
AT91C_TWI_TXCOMP          EQU (0x1 <<  0) ;- (TWI) Transmission Completed
AT91C_TWI_RXRDY           EQU (0x1 <<  1) ;- (TWI) Receive holding register ReaDY
AT91C_TWI_TXRDY           EQU (0x1 <<  2) ;- (TWI) Transmit holding register ReaDY
AT91C_TWI_SVREAD          EQU (0x1 <<  3) ;- (TWI) Slave Read
AT91C_TWI_SVACC           EQU (0x1 <<  4) ;- (TWI) Slave Access
AT91C_TWI_GCACC           EQU (0x1 <<  5) ;- (TWI) General Call Access
AT91C_TWI_OVRE            EQU (0x1 <<  6) ;- (TWI) Overrun Error
AT91C_TWI_UNRE            EQU (0x1 <<  7) ;- (TWI) Underrun Error
AT91C_TWI_NACK            EQU (0x1 <<  8) ;- (TWI) Not Acknowledged
AT91C_TWI_ARBLST          EQU (0x1 <<  9) ;- (TWI) Arbitration Lost
/* - -------- TWI_IER : (TWI Offset: 0x24) TWI Interrupt Enable Register -------- */
/* - -------- TWI_IDR : (TWI Offset: 0x28) TWI Interrupt Disable Register -------- */
/* - -------- TWI_IMR : (TWI Offset: 0x2c) TWI Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Multimedia Card Interface*/
/* - ******************************************************************************/
/* - -------- MCI_CR : (MCI Offset: 0x0) MCI Control Register -------- */
AT91C_MCI_MCIEN           EQU (0x1 <<  0) ;- (MCI) Multimedia Interface Enable
AT91C_MCI_MCIDIS          EQU (0x1 <<  1) ;- (MCI) Multimedia Interface Disable
AT91C_MCI_PWSEN           EQU (0x1 <<  2) ;- (MCI) Power Save Mode Enable
AT91C_MCI_PWSDIS          EQU (0x1 <<  3) ;- (MCI) Power Save Mode Disable
AT91C_MCI_SWRST           EQU (0x1 <<  7) ;- (MCI) MCI Software reset
/* - -------- MCI_MR : (MCI Offset: 0x4) MCI Mode Register -------- */
AT91C_MCI_CLKDIV          EQU (0xFF <<  0) ;- (MCI) Clock Divider
AT91C_MCI_PWSDIV          EQU (0x7 <<  8) ;- (MCI) Power Saving Divider
AT91C_MCI_PDCPADV         EQU (0x1 << 14) ;- (MCI) PDC Padding Value
AT91C_MCI_PDCMODE         EQU (0x1 << 15) ;- (MCI) PDC Oriented Mode
AT91C_MCI_BLKLEN          EQU (0xFFF << 18) ;- (MCI) Data Block Length
/* - -------- MCI_DTOR : (MCI Offset: 0x8) MCI Data Timeout Register -------- */
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
/* - -------- MCI_SDCR : (MCI Offset: 0xc) MCI SD Card Register -------- */
AT91C_MCI_SCDSEL          EQU (0xF <<  0) ;- (MCI) SD Card Selector
AT91C_MCI_SCDBUS          EQU (0x1 <<  7) ;- (MCI) SD Card Bus Width
/* - -------- MCI_CMDR : (MCI Offset: 0x14) MCI Command Register -------- */
AT91C_MCI_CMDNB           EQU (0x3F <<  0) ;- (MCI) Command Number
AT91C_MCI_RSPTYP          EQU (0x3 <<  6) ;- (MCI) Response Type
AT91C_MCI_RSPTYP_NO       EQU (0x0 <<  6) ;- (MCI) No response
AT91C_MCI_RSPTYP_48       EQU (0x1 <<  6) ;- (MCI) 48-bit response
AT91C_MCI_RSPTYP_136      EQU (0x2 <<  6) ;- (MCI) 136-bit response
AT91C_MCI_SPCMD           EQU (0x7 <<  8) ;- (MCI) Special CMD
AT91C_MCI_SPCMD_NONE      EQU (0x0 <<  8) ;- (MCI) Not a special CMD
AT91C_MCI_SPCMD_INIT      EQU (0x1 <<  8) ;- (MCI) Initialization CMD
AT91C_MCI_SPCMD_SYNC      EQU (0x2 <<  8) ;- (MCI) Synchronized CMD
AT91C_MCI_SPCMD_IT_CMD    EQU (0x4 <<  8) ;- (MCI) Interrupt command
AT91C_MCI_SPCMD_IT_REP    EQU (0x5 <<  8) ;- (MCI) Interrupt response
AT91C_MCI_OPDCMD          EQU (0x1 << 11) ;- (MCI) Open Drain Command
AT91C_MCI_MAXLAT          EQU (0x1 << 12) ;- (MCI) Maximum Latency for Command to respond
AT91C_MCI_TRCMD           EQU (0x3 << 16) ;- (MCI) Transfer CMD
AT91C_MCI_TRCMD_NO        EQU (0x0 << 16) ;- (MCI) No transfer
AT91C_MCI_TRCMD_START     EQU (0x1 << 16) ;- (MCI) Start transfer
AT91C_MCI_TRCMD_STOP      EQU (0x2 << 16) ;- (MCI) Stop transfer
AT91C_MCI_TRDIR           EQU (0x1 << 18) ;- (MCI) Transfer Direction
AT91C_MCI_TRTYP           EQU (0x3 << 19) ;- (MCI) Transfer Type
AT91C_MCI_TRTYP_BLOCK     EQU (0x0 << 19) ;- (MCI) Block Transfer type
AT91C_MCI_TRTYP_MULTIPLE  EQU (0x1 << 19) ;- (MCI) Multiple Block transfer type
AT91C_MCI_TRTYP_STREAM    EQU (0x2 << 19) ;- (MCI) Stream transfer type
/* - -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- */
AT91C_MCI_CMDRDY          EQU (0x1 <<  0) ;- (MCI) Command Ready flag
AT91C_MCI_RXRDY           EQU (0x1 <<  1) ;- (MCI) RX Ready flag
AT91C_MCI_TXRDY           EQU (0x1 <<  2) ;- (MCI) TX Ready flag
AT91C_MCI_BLKE            EQU (0x1 <<  3) ;- (MCI) Data Block Transfer Ended flag
AT91C_MCI_DTIP            EQU (0x1 <<  4) ;- (MCI) Data Transfer in Progress flag
AT91C_MCI_NOTBUSY         EQU (0x1 <<  5) ;- (MCI) Data Line Not Busy flag
AT91C_MCI_ENDRX           EQU (0x1 <<  6) ;- (MCI) End of RX Buffer flag
AT91C_MCI_ENDTX           EQU (0x1 <<  7) ;- (MCI) End of TX Buffer flag
AT91C_MCI_RXBUFF          EQU (0x1 << 14) ;- (MCI) RX Buffer Full flag
AT91C_MCI_TXBUFE          EQU (0x1 << 15) ;- (MCI) TX Buffer Empty flag
AT91C_MCI_RINDE           EQU (0x1 << 16) ;- (MCI) Response Index Error flag
AT91C_MCI_RDIRE           EQU (0x1 << 17) ;- (MCI) Response Direction Error flag
AT91C_MCI_RCRCE           EQU (0x1 << 18) ;- (MCI) Response CRC Error flag
AT91C_MCI_RENDE           EQU (0x1 << 19) ;- (MCI) Response End Bit Error flag
AT91C_MCI_RTOE            EQU (0x1 << 20) ;- (MCI) Response Time-out Error flag
AT91C_MCI_DCRCE           EQU (0x1 << 21) ;- (MCI) data CRC Error flag
AT91C_MCI_DTOE            EQU (0x1 << 22) ;- (MCI) Data timeout Error flag
AT91C_MCI_OVRE            EQU (0x1 << 30) ;- (MCI) Overrun flag
AT91C_MCI_UNRE            EQU (0x1 << 31) ;- (MCI) Underrun flag
/* - -------- MCI_IER : (MCI Offset: 0x44) MCI Interrupt Enable Register -------- */
/* - -------- MCI_IDR : (MCI Offset: 0x48) MCI Interrupt Disable Register -------- */
/* - -------- MCI_IMR : (MCI Offset: 0x4c) MCI Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR USB Device Interface*/
/* - ******************************************************************************/
/* - -------- UDP_FRM_NUM : (UDP Offset: 0x0) USB Frame Number Register -------- */
AT91C_UDP_FRM_NUM         EQU (0x7FF <<  0) ;- (UDP) Frame Number as Defined in the Packet Field Formats
AT91C_UDP_FRM_ERR         EQU (0x1 << 16) ;- (UDP) Frame Error
AT91C_UDP_FRM_OK          EQU (0x1 << 17) ;- (UDP) Frame OK
/* - -------- UDP_GLB_STATE : (UDP Offset: 0x4) USB Global State Register -------- */
AT91C_UDP_FADDEN          EQU (0x1 <<  0) ;- (UDP) Function Address Enable
AT91C_UDP_CONFG           EQU (0x1 <<  1) ;- (UDP) Configured
AT91C_UDP_RMWUPE          EQU (0x1 <<  2) ;- (UDP) Remote Wake Up Enable
AT91C_UDP_RSMINPR         EQU (0x1 <<  3) ;- (UDP) A Resume Has Been Sent to the Host
/* - -------- UDP_FADDR : (UDP Offset: 0x8) USB Function Address Register -------- */
AT91C_UDP_FADD            EQU (0xFF <<  0) ;- (UDP) Function Address Value
AT91C_UDP_FEN             EQU (0x1 <<  8) ;- (UDP) Function Enable
/* - -------- UDP_IER : (UDP Offset: 0x10) USB Interrupt Enable Register -------- */
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
/* - -------- UDP_IDR : (UDP Offset: 0x14) USB Interrupt Disable Register -------- */
/* - -------- UDP_IMR : (UDP Offset: 0x18) USB Interrupt Mask Register -------- */
/* - -------- UDP_ISR : (UDP Offset: 0x1c) USB Interrupt Status Register -------- */
AT91C_UDP_ENDBUSRES       EQU (0x1 << 12) ;- (UDP) USB End Of Bus Reset Interrupt
/* - -------- UDP_ICR : (UDP Offset: 0x20) USB Interrupt Clear Register -------- */
/* - -------- UDP_RST_EP : (UDP Offset: 0x28) USB Reset Endpoint Register -------- */
AT91C_UDP_EP0             EQU (0x1 <<  0) ;- (UDP) Reset Endpoint 0
AT91C_UDP_EP1             EQU (0x1 <<  1) ;- (UDP) Reset Endpoint 1
AT91C_UDP_EP2             EQU (0x1 <<  2) ;- (UDP) Reset Endpoint 2
AT91C_UDP_EP3             EQU (0x1 <<  3) ;- (UDP) Reset Endpoint 3
AT91C_UDP_EP4             EQU (0x1 <<  4) ;- (UDP) Reset Endpoint 4
AT91C_UDP_EP5             EQU (0x1 <<  5) ;- (UDP) Reset Endpoint 5
AT91C_UDP_EP6             EQU (0x1 <<  6) ;- (UDP) Reset Endpoint 6
AT91C_UDP_EP7             EQU (0x1 <<  7) ;- (UDP) Reset Endpoint 7
/* - -------- UDP_CSR : (UDP Offset: 0x30) USB Endpoint Control and Status Register -------- */
AT91C_UDP_TXCOMP          EQU (0x1 <<  0) ;- (UDP) Generates an IN packet with data previously written in the DPR
AT91C_UDP_RX_DATA_BK0     EQU (0x1 <<  1) ;- (UDP) Receive Data Bank 0
AT91C_UDP_RXSETUP         EQU (0x1 <<  2) ;- (UDP) Sends STALL to the Host (Control endpoints)
AT91C_UDP_ISOERROR        EQU (0x1 <<  3) ;- (UDP) Isochronous error (Isochronous endpoints)
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
/* -               REGISTER ADDRESS DEFINITION FOR AT91RM3400*/
/* - ******************************************************************************/
/* - ========== Register definition for SYS peripheral ========== */
/* - ========== Register definition for MC peripheral ========== */
AT91C_MC_ASR              EQU (0xFFFFFF04) ;- (MC) MC Abort Status Register
AT91C_MC_RCR              EQU (0xFFFFFF00) ;- (MC) MC Remap Control Register
AT91C_MC_PUP              EQU (0xFFFFFF50) ;- (MC) MC Protection Unit Peripherals
AT91C_MC_PUIA             EQU (0xFFFFFF10) ;- (MC) MC Protection Unit Area
AT91C_MC_AASR             EQU (0xFFFFFF08) ;- (MC) MC Abort Address Status Register
AT91C_MC_PUER             EQU (0xFFFFFF54) ;- (MC) MC Protection Unit Enable Register
/* - ========== Register definition for RTC peripheral ========== */
AT91C_RTC_VER             EQU (0xFFFFFE2C) ;- (RTC) Valid Entry Register
AT91C_RTC_MR              EQU (0xFFFFFE04) ;- (RTC) Mode Register
AT91C_RTC_IDR             EQU (0xFFFFFE24) ;- (RTC) Interrupt Disable Register
AT91C_RTC_CALALR          EQU (0xFFFFFE14) ;- (RTC) Calendar Alarm Register
AT91C_RTC_IER             EQU (0xFFFFFE20) ;- (RTC) Interrupt Enable Register
AT91C_RTC_TIMALR          EQU (0xFFFFFE10) ;- (RTC) Time Alarm Register
AT91C_RTC_IMR             EQU (0xFFFFFE28) ;- (RTC) Interrupt Mask Register
AT91C_RTC_CR              EQU (0xFFFFFE00) ;- (RTC) Control Register
AT91C_RTC_SCCR            EQU (0xFFFFFE1C) ;- (RTC) Status Clear Command Register
AT91C_RTC_TIMR            EQU (0xFFFFFE08) ;- (RTC) Time Register
AT91C_RTC_SR              EQU (0xFFFFFE18) ;- (RTC) Status Register
AT91C_RTC_CALR            EQU (0xFFFFFE0C) ;- (RTC) Calendar Register
/* - ========== Register definition for ST peripheral ========== */
AT91C_ST_CR               EQU (0xFFFFFD00) ;- (ST) Control Register
AT91C_ST_RTAR             EQU (0xFFFFFD20) ;- (ST) Real-time Alarm Register
AT91C_ST_IDR              EQU (0xFFFFFD18) ;- (ST) Interrupt Disable Register
AT91C_ST_PIMR             EQU (0xFFFFFD04) ;- (ST) Period Interval Mode Register
AT91C_ST_IER              EQU (0xFFFFFD14) ;- (ST) Interrupt Enable Register
AT91C_ST_CRTR             EQU (0xFFFFFD24) ;- (ST) Current Real-time Register
AT91C_ST_WDMR             EQU (0xFFFFFD08) ;- (ST) Watchdog Mode Register
AT91C_ST_SR               EQU (0xFFFFFD10) ;- (ST) Status Register
AT91C_ST_RTMR             EQU (0xFFFFFD0C) ;- (ST) Real-time Mode Register
AT91C_ST_IMR              EQU (0xFFFFFD1C) ;- (ST) Interrupt Mask Register
/* - ========== Register definition for PMC peripheral ========== */
AT91C_PMC_IDR             EQU (0xFFFFFC64) ;- (PMC) Interrupt Disable Register
AT91C_PMC_PCER            EQU (0xFFFFFC10) ;- (PMC) Peripheral Clock Enable Register
AT91C_PMC_PCKR            EQU (0xFFFFFC40) ;- (PMC) Programmable Clock Register
AT91C_PMC_MCKR            EQU (0xFFFFFC30) ;- (PMC) Master Clock Register
AT91C_PMC_SCDR            EQU (0xFFFFFC04) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCDR            EQU (0xFFFFFC14) ;- (PMC) Peripheral Clock Disable Register
AT91C_PMC_SCSR            EQU (0xFFFFFC08) ;- (PMC) System Clock Status Register
AT91C_PMC_PCSR            EQU (0xFFFFFC18) ;- (PMC) Peripheral Clock Status Register
AT91C_PMC_SCER            EQU (0xFFFFFC00) ;- (PMC) System Clock Enable Register
AT91C_PMC_IMR             EQU (0xFFFFFC6C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IER             EQU (0xFFFFFC60) ;- (PMC) Interrupt Enable Register
AT91C_PMC_SR              EQU (0xFFFFFC68) ;- (PMC) Status Register
/* - ========== Register definition for CKGR peripheral ========== */
AT91C_CKGR_MOR            EQU (0xFFFFFC20) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_PLLBR          EQU (0xFFFFFC2C) ;- (CKGR) PLL B Register
AT91C_CKGR_MCFR           EQU (0xFFFFFC24) ;- (CKGR) Main Clock  Frequency Register
AT91C_CKGR_PLLAR          EQU (0xFFFFFC28) ;- (CKGR) PLL A Register
/* - ========== Register definition for PIOB peripheral ========== */
AT91C_PIOB_OWDR           EQU (0xFFFFF6A4) ;- (PIOB) Output Write Disable Register
AT91C_PIOB_MDER           EQU (0xFFFFF650) ;- (PIOB) Multi-driver Enable Register
AT91C_PIOB_PPUSR          EQU (0xFFFFF668) ;- (PIOB) Pad Pull-up Status Register
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
AT91C_PIOA_PPUSR          EQU (0xFFFFF468) ;- (PIOA) Pad Pull-up Status Register
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
/* - ========== Register definition for DBGU peripheral ========== */
AT91C_DBGU_C2R            EQU (0xFFFFF244) ;- (DBGU) Chip ID2 Register
AT91C_DBGU_BRGR           EQU (0xFFFFF220) ;- (DBGU) Baud Rate Generator Register
AT91C_DBGU_IDR            EQU (0xFFFFF20C) ;- (DBGU) Interrupt Disable Register
AT91C_DBGU_CSR            EQU (0xFFFFF214) ;- (DBGU) Channel Status Register
AT91C_DBGU_C1R            EQU (0xFFFFF240) ;- (DBGU) Chip ID1 Register
AT91C_DBGU_MR             EQU (0xFFFFF204) ;- (DBGU) Mode Register
AT91C_DBGU_IMR            EQU (0xFFFFF210) ;- (DBGU) Interrupt Mask Register
AT91C_DBGU_CR             EQU (0xFFFFF200) ;- (DBGU) Control Register
AT91C_DBGU_FNTR           EQU (0xFFFFF248) ;- (DBGU) Force NTRST Register
AT91C_DBGU_THR            EQU (0xFFFFF21C) ;- (DBGU) Transmitter Holding Register
AT91C_DBGU_RHR            EQU (0xFFFFF218) ;- (DBGU) Receiver Holding Register
AT91C_DBGU_IER            EQU (0xFFFFF208) ;- (DBGU) Interrupt Enable Register
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
/* - ========== Register definition for PDC_SSC2 peripheral ========== */
AT91C_SSC2_PTSR           EQU (0xFFFD8124) ;- (PDC_SSC2) PDC Transfer Status Register
AT91C_SSC2_PTCR           EQU (0xFFFD8120) ;- (PDC_SSC2) PDC Transfer Control Register
AT91C_SSC2_TNPR           EQU (0xFFFD8118) ;- (PDC_SSC2) Transmit Next Pointer Register
AT91C_SSC2_TNCR           EQU (0xFFFD811C) ;- (PDC_SSC2) Transmit Next Counter Register
AT91C_SSC2_RNPR           EQU (0xFFFD8110) ;- (PDC_SSC2) Receive Next Pointer Register
AT91C_SSC2_RNCR           EQU (0xFFFD8114) ;- (PDC_SSC2) Receive Next Counter Register
AT91C_SSC2_RPR            EQU (0xFFFD8100) ;- (PDC_SSC2) Receive Pointer Register
AT91C_SSC2_TCR            EQU (0xFFFD810C) ;- (PDC_SSC2) Transmit Counter Register
AT91C_SSC2_TPR            EQU (0xFFFD8108) ;- (PDC_SSC2) Transmit Pointer Register
AT91C_SSC2_RCR            EQU (0xFFFD8104) ;- (PDC_SSC2) Receive Counter Register
/* - ========== Register definition for SSC2 peripheral ========== */
AT91C_SSC2_RC0R           EQU (0xFFFD8038) ;- (SSC2) Receive Compare 0 Register
AT91C_SSC2_RC1R           EQU (0xFFFD803C) ;- (SSC2) Receive Compare 1 Register
AT91C_SSC2_RSHR           EQU (0xFFFD8030) ;- (SSC2) Receive Sync Holding Register
AT91C_SSC2_IER            EQU (0xFFFD8044) ;- (SSC2) Interrupt Enable Register
AT91C_SSC2_RFMR           EQU (0xFFFD8014) ;- (SSC2) Receive Frame Mode Register
AT91C_SSC2_TFMR           EQU (0xFFFD801C) ;- (SSC2) Transmit Frame Mode Register
AT91C_SSC2_SR             EQU (0xFFFD8040) ;- (SSC2) Status Register
AT91C_SSC2_TSHR           EQU (0xFFFD8034) ;- (SSC2) Transmit Sync Holding Register
AT91C_SSC2_RHR            EQU (0xFFFD8020) ;- (SSC2) Receive Holding Register
AT91C_SSC2_CR             EQU (0xFFFD8000) ;- (SSC2) Control Register
AT91C_SSC2_IDR            EQU (0xFFFD8048) ;- (SSC2) Interrupt Disable Register
AT91C_SSC2_IMR            EQU (0xFFFD804C) ;- (SSC2) Interrupt Mask Register
AT91C_SSC2_THR            EQU (0xFFFD8024) ;- (SSC2) Transmit Holding Register
AT91C_SSC2_RCMR           EQU (0xFFFD8010) ;- (SSC2) Receive Clock ModeRegister
AT91C_SSC2_TCMR           EQU (0xFFFD8018) ;- (SSC2) Transmit Clock Mode Register
AT91C_SSC2_CMR            EQU (0xFFFD8004) ;- (SSC2) Clock Mode Register
/* - ========== Register definition for PDC_SSC1 peripheral ========== */
AT91C_SSC1_TNCR           EQU (0xFFFD411C) ;- (PDC_SSC1) Transmit Next Counter Register
AT91C_SSC1_RPR            EQU (0xFFFD4100) ;- (PDC_SSC1) Receive Pointer Register
AT91C_SSC1_RNCR           EQU (0xFFFD4114) ;- (PDC_SSC1) Receive Next Counter Register
AT91C_SSC1_TPR            EQU (0xFFFD4108) ;- (PDC_SSC1) Transmit Pointer Register
AT91C_SSC1_PTCR           EQU (0xFFFD4120) ;- (PDC_SSC1) PDC Transfer Control Register
AT91C_SSC1_TCR            EQU (0xFFFD410C) ;- (PDC_SSC1) Transmit Counter Register
AT91C_SSC1_RCR            EQU (0xFFFD4104) ;- (PDC_SSC1) Receive Counter Register
AT91C_SSC1_RNPR           EQU (0xFFFD4110) ;- (PDC_SSC1) Receive Next Pointer Register
AT91C_SSC1_TNPR           EQU (0xFFFD4118) ;- (PDC_SSC1) Transmit Next Pointer Register
AT91C_SSC1_PTSR           EQU (0xFFFD4124) ;- (PDC_SSC1) PDC Transfer Status Register
/* - ========== Register definition for SSC1 peripheral ========== */
AT91C_SSC1_RC1R           EQU (0xFFFD403C) ;- (SSC1) Receive Compare 1 Register
AT91C_SSC1_RHR            EQU (0xFFFD4020) ;- (SSC1) Receive Holding Register
AT91C_SSC1_RSHR           EQU (0xFFFD4030) ;- (SSC1) Receive Sync Holding Register
AT91C_SSC1_TFMR           EQU (0xFFFD401C) ;- (SSC1) Transmit Frame Mode Register
AT91C_SSC1_IDR            EQU (0xFFFD4048) ;- (SSC1) Interrupt Disable Register
AT91C_SSC1_RC0R           EQU (0xFFFD4038) ;- (SSC1) Receive Compare 0 Register
AT91C_SSC1_THR            EQU (0xFFFD4024) ;- (SSC1) Transmit Holding Register
AT91C_SSC1_RCMR           EQU (0xFFFD4010) ;- (SSC1) Receive Clock ModeRegister
AT91C_SSC1_IER            EQU (0xFFFD4044) ;- (SSC1) Interrupt Enable Register
AT91C_SSC1_TSHR           EQU (0xFFFD4034) ;- (SSC1) Transmit Sync Holding Register
AT91C_SSC1_SR             EQU (0xFFFD4040) ;- (SSC1) Status Register
AT91C_SSC1_CMR            EQU (0xFFFD4004) ;- (SSC1) Clock Mode Register
AT91C_SSC1_TCMR           EQU (0xFFFD4018) ;- (SSC1) Transmit Clock Mode Register
AT91C_SSC1_CR             EQU (0xFFFD4000) ;- (SSC1) Control Register
AT91C_SSC1_IMR            EQU (0xFFFD404C) ;- (SSC1) Interrupt Mask Register
AT91C_SSC1_RFMR           EQU (0xFFFD4014) ;- (SSC1) Receive Frame Mode Register
/* - ========== Register definition for PDC_SSC0 peripheral ========== */
AT91C_SSC0_RNPR           EQU (0xFFFD0110) ;- (PDC_SSC0) Receive Next Pointer Register
AT91C_SSC0_RNCR           EQU (0xFFFD0114) ;- (PDC_SSC0) Receive Next Counter Register
AT91C_SSC0_PTSR           EQU (0xFFFD0124) ;- (PDC_SSC0) PDC Transfer Status Register
AT91C_SSC0_PTCR           EQU (0xFFFD0120) ;- (PDC_SSC0) PDC Transfer Control Register
AT91C_SSC0_TCR            EQU (0xFFFD010C) ;- (PDC_SSC0) Transmit Counter Register
AT91C_SSC0_TNPR           EQU (0xFFFD0118) ;- (PDC_SSC0) Transmit Next Pointer Register
AT91C_SSC0_RCR            EQU (0xFFFD0104) ;- (PDC_SSC0) Receive Counter Register
AT91C_SSC0_TPR            EQU (0xFFFD0108) ;- (PDC_SSC0) Transmit Pointer Register
AT91C_SSC0_TNCR           EQU (0xFFFD011C) ;- (PDC_SSC0) Transmit Next Counter Register
AT91C_SSC0_RPR            EQU (0xFFFD0100) ;- (PDC_SSC0) Receive Pointer Register
/* - ========== Register definition for SSC0 peripheral ========== */
AT91C_SSC0_THR            EQU (0xFFFD0024) ;- (SSC0) Transmit Holding Register
AT91C_SSC0_IDR            EQU (0xFFFD0048) ;- (SSC0) Interrupt Disable Register
AT91C_SSC0_CMR            EQU (0xFFFD0004) ;- (SSC0) Clock Mode Register
AT91C_SSC0_SR             EQU (0xFFFD0040) ;- (SSC0) Status Register
AT91C_SSC0_RC0R           EQU (0xFFFD0038) ;- (SSC0) Receive Compare 0 Register
AT91C_SSC0_RCMR           EQU (0xFFFD0010) ;- (SSC0) Receive Clock ModeRegister
AT91C_SSC0_RC1R           EQU (0xFFFD003C) ;- (SSC0) Receive Compare 1 Register
AT91C_SSC0_IER            EQU (0xFFFD0044) ;- (SSC0) Interrupt Enable Register
AT91C_SSC0_CR             EQU (0xFFFD0000) ;- (SSC0) Control Register
AT91C_SSC0_TFMR           EQU (0xFFFD001C) ;- (SSC0) Transmit Frame Mode Register
AT91C_SSC0_RHR            EQU (0xFFFD0020) ;- (SSC0) Receive Holding Register
AT91C_SSC0_IMR            EQU (0xFFFD004C) ;- (SSC0) Interrupt Mask Register
AT91C_SSC0_TCMR           EQU (0xFFFD0018) ;- (SSC0) Transmit Clock Mode Register
AT91C_SSC0_RSHR           EQU (0xFFFD0030) ;- (SSC0) Receive Sync Holding Register
AT91C_SSC0_RFMR           EQU (0xFFFD0014) ;- (SSC0) Receive Frame Mode Register
AT91C_SSC0_TSHR           EQU (0xFFFD0034) ;- (SSC0) Transmit Sync Holding Register
/* - ========== Register definition for PDC_US3 peripheral ========== */
AT91C_US3_PTCR            EQU (0xFFFCC120) ;- (PDC_US3) PDC Transfer Control Register
AT91C_US3_RNPR            EQU (0xFFFCC110) ;- (PDC_US3) Receive Next Pointer Register
AT91C_US3_RCR             EQU (0xFFFCC104) ;- (PDC_US3) Receive Counter Register
AT91C_US3_TPR             EQU (0xFFFCC108) ;- (PDC_US3) Transmit Pointer Register
AT91C_US3_PTSR            EQU (0xFFFCC124) ;- (PDC_US3) PDC Transfer Status Register
AT91C_US3_TNCR            EQU (0xFFFCC11C) ;- (PDC_US3) Transmit Next Counter Register
AT91C_US3_RPR             EQU (0xFFFCC100) ;- (PDC_US3) Receive Pointer Register
AT91C_US3_TCR             EQU (0xFFFCC10C) ;- (PDC_US3) Transmit Counter Register
AT91C_US3_RNCR            EQU (0xFFFCC114) ;- (PDC_US3) Receive Next Counter Register
AT91C_US3_TNPR            EQU (0xFFFCC118) ;- (PDC_US3) Transmit Next Pointer Register
/* - ========== Register definition for US3 peripheral ========== */
AT91C_US3_CSR             EQU (0xFFFCC014) ;- (US3) Channel Status Register
AT91C_US3_IER             EQU (0xFFFCC008) ;- (US3) Interrupt Enable Register
AT91C_US3_IMR             EQU (0xFFFCC010) ;- (US3) Interrupt Mask Register
AT91C_US3_THR             EQU (0xFFFCC01C) ;- (US3) Transmitter Holding Register
AT91C_US3_BRGR            EQU (0xFFFCC020) ;- (US3) Baud Rate Generator Register
AT91C_US3_RTOR            EQU (0xFFFCC024) ;- (US3) Receiver Time-out Register
AT91C_US3_IDR             EQU (0xFFFCC00C) ;- (US3) Interrupt Disable Register
AT91C_US3_FIDI            EQU (0xFFFCC040) ;- (US3) FI_DI_Ratio Register
AT91C_US3_RHR             EQU (0xFFFCC018) ;- (US3) Receiver Holding Register
AT91C_US3_NER             EQU (0xFFFCC044) ;- (US3) Nb Errors Register
AT91C_US3_XXR             EQU (0xFFFCC048) ;- (US3) XON_XOFF Register
AT91C_US3_IF              EQU (0xFFFCC04C) ;- (US3) IRDA_FILTER Register
AT91C_US3_TTGR            EQU (0xFFFCC028) ;- (US3) Transmitter Time-guard Register
AT91C_US3_CR              EQU (0xFFFCC000) ;- (US3) Control Register
AT91C_US3_MR              EQU (0xFFFCC004) ;- (US3) Mode Register
/* - ========== Register definition for PDC_US2 peripheral ========== */
AT91C_US2_PTCR            EQU (0xFFFC8120) ;- (PDC_US2) PDC Transfer Control Register
AT91C_US2_TCR             EQU (0xFFFC810C) ;- (PDC_US2) Transmit Counter Register
AT91C_US2_RPR             EQU (0xFFFC8100) ;- (PDC_US2) Receive Pointer Register
AT91C_US2_TPR             EQU (0xFFFC8108) ;- (PDC_US2) Transmit Pointer Register
AT91C_US2_PTSR            EQU (0xFFFC8124) ;- (PDC_US2) PDC Transfer Status Register
AT91C_US2_RNCR            EQU (0xFFFC8114) ;- (PDC_US2) Receive Next Counter Register
AT91C_US2_TNPR            EQU (0xFFFC8118) ;- (PDC_US2) Transmit Next Pointer Register
AT91C_US2_RCR             EQU (0xFFFC8104) ;- (PDC_US2) Receive Counter Register
AT91C_US2_RNPR            EQU (0xFFFC8110) ;- (PDC_US2) Receive Next Pointer Register
AT91C_US2_TNCR            EQU (0xFFFC811C) ;- (PDC_US2) Transmit Next Counter Register
/* - ========== Register definition for US2 peripheral ========== */
AT91C_US2_RHR             EQU (0xFFFC8018) ;- (US2) Receiver Holding Register
AT91C_US2_BRGR            EQU (0xFFFC8020) ;- (US2) Baud Rate Generator Register
AT91C_US2_IF              EQU (0xFFFC804C) ;- (US2) IRDA_FILTER Register
AT91C_US2_IDR             EQU (0xFFFC800C) ;- (US2) Interrupt Disable Register
AT91C_US2_IMR             EQU (0xFFFC8010) ;- (US2) Interrupt Mask Register
AT91C_US2_CR              EQU (0xFFFC8000) ;- (US2) Control Register
AT91C_US2_IER             EQU (0xFFFC8008) ;- (US2) Interrupt Enable Register
AT91C_US2_NER             EQU (0xFFFC8044) ;- (US2) Nb Errors Register
AT91C_US2_RTOR            EQU (0xFFFC8024) ;- (US2) Receiver Time-out Register
AT91C_US2_TTGR            EQU (0xFFFC8028) ;- (US2) Transmitter Time-guard Register
AT91C_US2_MR              EQU (0xFFFC8004) ;- (US2) Mode Register
AT91C_US2_CSR             EQU (0xFFFC8014) ;- (US2) Channel Status Register
AT91C_US2_XXR             EQU (0xFFFC8048) ;- (US2) XON_XOFF Register
AT91C_US2_THR             EQU (0xFFFC801C) ;- (US2) Transmitter Holding Register
AT91C_US2_FIDI            EQU (0xFFFC8040) ;- (US2) FI_DI_Ratio Register
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
AT91C_US1_XXR             EQU (0xFFFC4048) ;- (US1) XON_XOFF Register
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
AT91C_US0_XXR             EQU (0xFFFC0048) ;- (US0) XON_XOFF Register
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
/* - ========== Register definition for PDC_MCI peripheral ========== */
AT91C_MCI_PTCR            EQU (0xFFFB4120) ;- (PDC_MCI) PDC Transfer Control Register
AT91C_MCI_RCR             EQU (0xFFFB4104) ;- (PDC_MCI) Receive Counter Register
AT91C_MCI_RPR             EQU (0xFFFB4100) ;- (PDC_MCI) Receive Pointer Register
AT91C_MCI_PTSR            EQU (0xFFFB4124) ;- (PDC_MCI) PDC Transfer Status Register
AT91C_MCI_TPR             EQU (0xFFFB4108) ;- (PDC_MCI) Transmit Pointer Register
AT91C_MCI_TCR             EQU (0xFFFB410C) ;- (PDC_MCI) Transmit Counter Register
AT91C_MCI_RNPR            EQU (0xFFFB4110) ;- (PDC_MCI) Receive Next Pointer Register
AT91C_MCI_TNCR            EQU (0xFFFB411C) ;- (PDC_MCI) Transmit Next Counter Register
AT91C_MCI_RNCR            EQU (0xFFFB4114) ;- (PDC_MCI) Receive Next Counter Register
AT91C_MCI_TNPR            EQU (0xFFFB4118) ;- (PDC_MCI) Transmit Next Pointer Register
/* - ========== Register definition for MCI peripheral ========== */
AT91C_MCI_TDR             EQU (0xFFFB4034) ;- (MCI) MCI Transmit Data Register
AT91C_MCI_RSPR            EQU (0xFFFB4020) ;- (MCI) MCI Response Register
AT91C_MCI_SDCR            EQU (0xFFFB400C) ;- (MCI) MCI SD Card Register
AT91C_MCI_MR              EQU (0xFFFB4004) ;- (MCI) MCI Mode Register
AT91C_MCI_CR              EQU (0xFFFB4000) ;- (MCI) MCI Control Register
AT91C_MCI_ARGR            EQU (0xFFFB4010) ;- (MCI) MCI Argument Register
AT91C_MCI_SR              EQU (0xFFFB4040) ;- (MCI) MCI Status Register
AT91C_MCI_RDR             EQU (0xFFFB4030) ;- (MCI) MCI Receive Data Register
AT91C_MCI_DTOR            EQU (0xFFFB4008) ;- (MCI) MCI Data Timeout Register
AT91C_MCI_CMDR            EQU (0xFFFB4014) ;- (MCI) MCI Command Register
AT91C_MCI_IMR             EQU (0xFFFB404C) ;- (MCI) MCI Interrupt Mask Register
AT91C_MCI_IER             EQU (0xFFFB4044) ;- (MCI) MCI Interrupt Enable Register
AT91C_MCI_IDR             EQU (0xFFFB4048) ;- (MCI) MCI Interrupt Disable Register
/* - ========== Register definition for UDP peripheral ========== */
AT91C_UDP_IMR             EQU (0xFFFB0018) ;- (UDP) Interrupt Mask Register
AT91C_UDP_FADDR           EQU (0xFFFB0008) ;- (UDP) Function Address Register
AT91C_UDP_NUM             EQU (0xFFFB0000) ;- (UDP) Frame Number Register
AT91C_UDP_FDR             EQU (0xFFFB0050) ;- (UDP) Endpoint FIFO Data Register
AT91C_UDP_ISR             EQU (0xFFFB001C) ;- (UDP) Interrupt Status Register
AT91C_UDP_CSR             EQU (0xFFFB0030) ;- (UDP) Endpoint Control and Status Register
AT91C_UDP_IDR             EQU (0xFFFB0014) ;- (UDP) Interrupt Disable Register
AT91C_UDP_ICR             EQU (0xFFFB0020) ;- (UDP) Interrupt Clear Register
AT91C_UDP_RSTEP           EQU (0xFFFB0028) ;- (UDP) Reset Endpoint Register
AT91C_UDP_GLBSTATE        EQU (0xFFFB0004) ;- (UDP) Global State Register
AT91C_UDP_IER             EQU (0xFFFB0010) ;- (UDP) Interrupt Enable Register
/* - ========== Register definition for TC5 peripheral ========== */
AT91C_TC5_CMR             EQU (0xFFFA4084) ;- (TC5) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC5_IDR             EQU (0xFFFA40A8) ;- (TC5) Interrupt Disable Register
AT91C_TC5_CCR             EQU (0xFFFA4080) ;- (TC5) Channel Control Register
AT91C_TC5_RB              EQU (0xFFFA4098) ;- (TC5) Register B
AT91C_TC5_IMR             EQU (0xFFFA40AC) ;- (TC5) Interrupt Mask Register
AT91C_TC5_CV              EQU (0xFFFA4090) ;- (TC5) Counter Value
AT91C_TC5_RC              EQU (0xFFFA409C) ;- (TC5) Register C
AT91C_TC5_SR              EQU (0xFFFA40A0) ;- (TC5) Status Register
AT91C_TC5_IER             EQU (0xFFFA40A4) ;- (TC5) Interrupt Enable Register
AT91C_TC5_RA              EQU (0xFFFA4094) ;- (TC5) Register A
/* - ========== Register definition for TC4 peripheral ========== */
AT91C_TC4_SR              EQU (0xFFFA4060) ;- (TC4) Status Register
AT91C_TC4_RA              EQU (0xFFFA4054) ;- (TC4) Register A
AT91C_TC4_CV              EQU (0xFFFA4050) ;- (TC4) Counter Value
AT91C_TC4_CMR             EQU (0xFFFA4044) ;- (TC4) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC4_RB              EQU (0xFFFA4058) ;- (TC4) Register B
AT91C_TC4_CCR             EQU (0xFFFA4040) ;- (TC4) Channel Control Register
AT91C_TC4_IER             EQU (0xFFFA4064) ;- (TC4) Interrupt Enable Register
AT91C_TC4_IMR             EQU (0xFFFA406C) ;- (TC4) Interrupt Mask Register
AT91C_TC4_RC              EQU (0xFFFA405C) ;- (TC4) Register C
AT91C_TC4_IDR             EQU (0xFFFA4068) ;- (TC4) Interrupt Disable Register
/* - ========== Register definition for TC3 peripheral ========== */
AT91C_TC3_IMR             EQU (0xFFFA402C) ;- (TC3) Interrupt Mask Register
AT91C_TC3_CMR             EQU (0xFFFA4004) ;- (TC3) Channel Mode Register (Capture Mode / Waveform Mode)
AT91C_TC3_IDR             EQU (0xFFFA4028) ;- (TC3) Interrupt Disable Register
AT91C_TC3_CCR             EQU (0xFFFA4000) ;- (TC3) Channel Control Register
AT91C_TC3_RA              EQU (0xFFFA4014) ;- (TC3) Register A
AT91C_TC3_RB              EQU (0xFFFA4018) ;- (TC3) Register B
AT91C_TC3_CV              EQU (0xFFFA4010) ;- (TC3) Counter Value
AT91C_TC3_SR              EQU (0xFFFA4020) ;- (TC3) Status Register
AT91C_TC3_IER             EQU (0xFFFA4024) ;- (TC3) Interrupt Enable Register
AT91C_TC3_RC              EQU (0xFFFA401C) ;- (TC3) Register C
/* - ========== Register definition for TCB1 peripheral ========== */
AT91C_TCB1_BMR            EQU (0xFFFA40C4) ;- (TCB1) TC Block Mode Register
AT91C_TCB1_BCR            EQU (0xFFFA40C0) ;- (TCB1) TC Block Control Register
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
/* - ========== Register definition for TCB0 peripheral ========== */
AT91C_TCB0_BMR            EQU (0xFFFA00C4) ;- (TCB0) TC Block Mode Register
AT91C_TCB0_BCR            EQU (0xFFFA00C0) ;- (TCB0) TC Block Control Register

/* - ******************************************************************************/
/* -               PIO DEFINITIONS FOR AT91RM3400*/
/* - ******************************************************************************/
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PA0_MISO            EQU (AT91C_PIO_PA0) ;-  SPI Master In Slave
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PA1_MOSI            EQU (AT91C_PIO_PA1) ;-  SPI Master Out Slave
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PA10_RXD0           EQU (AT91C_PIO_PA10) ;-  USART 0 Receive Data
AT91C_PA10_UTXOEN         EQU (AT91C_PIO_PA10) ;-  USB device TXOEN
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PA11_SCK0           EQU (AT91C_PIO_PA11) ;-  USART 0 Serial Clock
AT91C_PA11_TCLK0          EQU (AT91C_PIO_PA11) ;-  Timer Counter 0 external clock input
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PA12_CTS0           EQU (AT91C_PIO_PA12) ;-  USART 0 Clear To Send
AT91C_PA12_TCLK1          EQU (AT91C_PIO_PA12) ;-  Timer Counter 1 external clock input
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PA13_RTS0           EQU (AT91C_PIO_PA13) ;-  USART 0 Ready To Send
AT91C_PA13_TCLK2          EQU (AT91C_PIO_PA13) ;-  Timer Counter 2 external clock input
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PA14_RXD1           EQU (AT91C_PIO_PA14) ;-  USART 1 Receive Data
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PA15_TXD1           EQU (AT91C_PIO_PA15) ;-  USART 1 Transmit Data
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PA16_RTS1           EQU (AT91C_PIO_PA16) ;-  USART 1 Ready To Send
AT91C_PA16_TIOA0          EQU (AT91C_PIO_PA16) ;-  Timer Counter 0 Multipurpose Timer I/O Pin A
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PA17_CTS1           EQU (AT91C_PIO_PA17) ;-  USART 1 Clear To Send
AT91C_PA17_TIOB0          EQU (AT91C_PIO_PA17) ;-  Timer Counter 0 Multipurpose Timer I/O Pin B
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PA18_DTR1           EQU (AT91C_PIO_PA18) ;-  USART 1 Data Terminal ready
AT91C_PA18_TIOA1          EQU (AT91C_PIO_PA18) ;-  Timer Counter 1 Multipurpose Timer I/O Pin A
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PA19_DSR1           EQU (AT91C_PIO_PA19) ;-  USART 1 Data Set ready
AT91C_PA19_TIOB1          EQU (AT91C_PIO_PA19) ;-  Timer Counter 1 Multipurpose Timer I/O Pin B
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PA2_SPCK            EQU (AT91C_PIO_PA2) ;-  SPI Serial Clock
AT91C_PA2_PCK0            EQU (AT91C_PIO_PA2) ;-  PMC Programmable clock Output 0
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PA20_DCD1           EQU (AT91C_PIO_PA20) ;-  USART 1 Data Carrier Detect
AT91C_PA20_TIOA2          EQU (AT91C_PIO_PA20) ;-  Timer Counter 2 Multipurpose Timer I/O Pin A
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PA21_RI1            EQU (AT91C_PIO_PA21) ;-  USART 1 Ring Indicator
AT91C_PA21_TIOB2          EQU (AT91C_PIO_PA21) ;-  Timer Counter 2 Multipurpose Timer I/O Pin B
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PA22_RXD2           EQU (AT91C_PIO_PA22) ;-  USART 2 Receive Data
AT91C_PA22_URXD           EQU (AT91C_PIO_PA22) ;-  USB device RXD
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PA23_TXD2           EQU (AT91C_PIO_PA23) ;-  USART 2 Transmit Data
AT91C_PA23_UTXD           EQU (AT91C_PIO_PA23) ;-  USB device TXD
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PA24_MCCK           EQU (AT91C_PIO_PA24) ;-  Multimedia Card Clock
AT91C_PA24_RTS0           EQU (AT91C_PIO_PA24) ;-  Usart 0 Ready To Send
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PA25_MCCDA          EQU (AT91C_PIO_PA25) ;-  Multimedia Card A Command
AT91C_PA25_RTS1           EQU (AT91C_PIO_PA25) ;-  Usart 0 Ready To Send
AT91C_PIO_PA26            EQU (1 << 26) ;- Pin Controlled by PA26
AT91C_PA26_MCDA0          EQU (AT91C_PIO_PA26) ;-  Multimedia Card A Data 0
AT91C_PIO_PA27            EQU (1 << 27) ;- Pin Controlled by PA27
AT91C_PA27_MCDA1          EQU (AT91C_PIO_PA27) ;-  Multimedia Card A Data 1
AT91C_PA27_UEON           EQU (AT91C_PIO_PA27) ;-  USB Device UEON
AT91C_PIO_PA28            EQU (1 << 28) ;- Pin Controlled by PA28
AT91C_PA28_MCDA2          EQU (AT91C_PIO_PA28) ;-  Multimedia Card A Data 2
AT91C_PA28_RTS2           EQU (AT91C_PIO_PA28) ;-  Usart 2 Ready To Send
AT91C_PIO_PA29            EQU (1 << 29) ;- Pin Controlled by PA29
AT91C_PA29_MCDA3          EQU (AT91C_PIO_PA29) ;-  Multimedia Card A Data 3
AT91C_PA29_CTS2           EQU (AT91C_PIO_PA29) ;-  Usart 2 Clear To Send
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PA3_NPCS0           EQU (AT91C_PIO_PA3) ;-  SPI Peripheral Chip Select 0
AT91C_PA3_PCK1            EQU (AT91C_PIO_PA3) ;-  PMC Programmable clock Output 1
AT91C_PIO_PA30            EQU (1 << 30) ;- Pin Controlled by PA30
AT91C_PA30_DRXD           EQU (AT91C_PIO_PA30) ;-  DBGU Debug Receive Data
AT91C_PIO_PA31            EQU (1 << 31) ;- Pin Controlled by PA31
AT91C_PA31_DTXD           EQU (AT91C_PIO_PA31) ;-  DBGU Debug Transmit Data
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PA4_NPCS1           EQU (AT91C_PIO_PA4) ;-  SPI Peripheral Chip Select 1
AT91C_PA4_USUSPEND        EQU (AT91C_PIO_PA4) ;-  USB device suspend
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PA5_NPCS2           EQU (AT91C_PIO_PA5) ;-  SPI Peripheral Chip Select 2
AT91C_PA5_SCK1            EQU (AT91C_PIO_PA5) ;-  USART1 Serial Clock
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PA6_NPCS3           EQU (AT91C_PIO_PA6) ;-  SPI Peripheral Chip Select 3
AT91C_PA6_SCK2            EQU (AT91C_PIO_PA6) ;-  USART2 Serial Clock
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PA7_TWD             EQU (AT91C_PIO_PA7) ;-  TWI Two-wire Serial Data
AT91C_PA7_PCK2            EQU (AT91C_PIO_PA7) ;-  PMC Programmable Clock 2
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PA8_TWCK            EQU (AT91C_PIO_PA8) ;-  TWI Two-wire Serial Clock
AT91C_PA8_PCK3            EQU (AT91C_PIO_PA8) ;-  PMC Programmable Clock 3
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PA9_TXD0            EQU (AT91C_PIO_PA9) ;-  USART 0 Transmit Data
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PB0_TF0             EQU (AT91C_PIO_PB0) ;-  SSC Transmit Frame Sync 0
AT91C_PB0_TIOB3           EQU (AT91C_PIO_PB0) ;-  Timer Counter 3 Multipurpose Timer I/O Pin B
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PB1_TK0             EQU (AT91C_PIO_PB1) ;-  SSC Transmit Clock 0
AT91C_PB1_TCLK3           EQU (AT91C_PIO_PB1) ;-  Timer Counter 3 External Clock Input
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PB10_RK1            EQU (AT91C_PIO_PB10) ;-  SSC Receive Clock 1
AT91C_PB10_PCK1           EQU (AT91C_PIO_PB10) ;-  PMC Programmable Clock Output 1
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PB11_RF1            EQU (AT91C_PIO_PB11) ;-  SSC Receive Frame Sync 1
AT91C_PB11_TIOA4          EQU (AT91C_PIO_PB11) ;-  Timer Counter 4 Multipurpose Timer I/O Pin A
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_TF2            EQU (AT91C_PIO_PB12) ;-  SSC Transmit Frame Sync 2
AT91C_PB12_TIOB5          EQU (AT91C_PIO_PB12) ;-  Timer Counter 5 Multipurpose Timer I/O Pin B
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_TK2            EQU (AT91C_PIO_PB13) ;-  SSC Transmit Clock 2
AT91C_PB13_TCLK5          EQU (AT91C_PIO_PB13) ;-  Timer Counter 5 external clock input
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_TD2            EQU (AT91C_PIO_PB14) ;-  SSC Transmit Data 2
AT91C_PB14_NPCS3          EQU (AT91C_PIO_PB14) ;-  SPI Peripheral Chip Select 3
AT91C_PIO_PB15            EQU (1 << 15) ;- Pin Controlled by PB15
AT91C_PB15_RD2            EQU (AT91C_PIO_PB15) ;-  SSC Receive Data 2
AT91C_PB15_PCK1           EQU (AT91C_PIO_PB15) ;-  PMC Programmable Clock Output 1
AT91C_PIO_PB16            EQU (1 << 16) ;- Pin Controlled by PB16
AT91C_PB16_RK2            EQU (AT91C_PIO_PB16) ;-  SSC Receive Clock 2
AT91C_PB16_PCK2           EQU (AT91C_PIO_PB16) ;-  PMC Programmable Clock Output 2
AT91C_PIO_PB17            EQU (1 << 17) ;- Pin Controlled by PB17
AT91C_PB17_RF2            EQU (AT91C_PIO_PB17) ;-  SSC Receive Frame Sync 2
AT91C_PB17_TIOA5          EQU (AT91C_PIO_PB17) ;-  Timer Counter 5 Multipurpose Timer I/O Pin A
AT91C_PIO_PB18            EQU (1 << 18) ;- Pin Controlled by PB18
AT91C_PB18_RTS3           EQU (AT91C_PIO_PB18) ;-  USART 3 Ready To Send
AT91C_PB18_MCCDB          EQU (AT91C_PIO_PB18) ;-  Multimedia Card B Command
AT91C_PIO_PB19            EQU (1 << 19) ;- Pin Controlled by PB19
AT91C_PB19_CTS3           EQU (AT91C_PIO_PB19) ;-  USART 3 Clear To Send
AT91C_PB19_MCDB0          EQU (AT91C_PIO_PB19) ;-  Multimedia Card B Data 0
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PB2_TD0             EQU (AT91C_PIO_PB2) ;-  SSC Transmit data
AT91C_PB2_RTS2            EQU (AT91C_PIO_PB2) ;-  USART 2 Ready To Send
AT91C_PIO_PB20            EQU (1 << 20) ;- Pin Controlled by PB20
AT91C_PB20_TXD3           EQU (AT91C_PIO_PB20) ;-  USART 3 Transmit Data
AT91C_PB20_DTR1           EQU (AT91C_PIO_PB20) ;-  USART 1 Data Terminal Ready
AT91C_PIO_PB21            EQU (1 << 21) ;- Pin Controlled by PB21
AT91C_PB21_RXD3           EQU (AT91C_PIO_PB21) ;-  USART 3 Receive Data
AT91C_PIO_PB22            EQU (1 << 22) ;- Pin Controlled by PB22
AT91C_PB22_SCK3           EQU (AT91C_PIO_PB22) ;-  USART 3 Serial Clock
AT91C_PB22_PCK3           EQU (AT91C_PIO_PB22) ;-  PMC Programmable Clock Output 3
AT91C_PIO_PB23            EQU (1 << 23) ;- Pin Controlled by PB23
AT91C_PB23_FIQ            EQU (AT91C_PIO_PB23) ;-  AIC Fast Interrupt Input
AT91C_PIO_PB24            EQU (1 << 24) ;- Pin Controlled by PB24
AT91C_PB24_IRQ0           EQU (AT91C_PIO_PB24) ;-  Interrupt input 0
AT91C_PB24_TD0            EQU (AT91C_PIO_PB24) ;-  SSC Transmit Data 0
AT91C_PIO_PB25            EQU (1 << 25) ;- Pin Controlled by PB25
AT91C_PB25_IRQ1           EQU (AT91C_PIO_PB25) ;-  Interrupt input 1
AT91C_PB25_TD1            EQU (AT91C_PIO_PB25) ;-  SSC Transmit Data 1
AT91C_PIO_PB26            EQU (1 << 26) ;- Pin Controlled by PB26
AT91C_PB26_IRQ2           EQU (AT91C_PIO_PB26) ;-  Interrupt input 2
AT91C_PB26_TD2            EQU (AT91C_PIO_PB26) ;-  SSC Transmit Data 2
AT91C_PIO_PB27            EQU (1 << 27) ;- Pin Controlled by PB27
AT91C_PB27_IRQ3           EQU (AT91C_PIO_PB27) ;-  Interrupt input 3
AT91C_PB27_DTXD           EQU (AT91C_PIO_PB27) ;-  Debug Unit Transmit Data
AT91C_PIO_PB28            EQU (1 << 28) ;- Pin Controlled by PB28
AT91C_PB28_IRQ4           EQU (AT91C_PIO_PB28) ;-  Interrupt input 4
AT91C_PB28_MCDB1          EQU (AT91C_PIO_PB28) ;-  Multimedia Card B Data 1
AT91C_PIO_PB29            EQU (1 << 29) ;- Pin Controlled by PB29
AT91C_PB29_IRQ5           EQU (AT91C_PIO_PB29) ;-  Interrupt input 5
AT91C_PB29_MCDB2          EQU (AT91C_PIO_PB29) ;-  Multimedia Card B Data 2
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PB3_RD0             EQU (AT91C_PIO_PB3) ;-  SSC Receive Data
AT91C_PB3_RTS3            EQU (AT91C_PIO_PB3) ;-  USART 3 Ready To Send
AT91C_PIO_PB30            EQU (1 << 30) ;- Pin Controlled by PB30
AT91C_PB30_IRQ6           EQU (AT91C_PIO_PB30) ;-  Interrupt input 6
AT91C_PB30_MCDB3          EQU (AT91C_PIO_PB30) ;-  Multimedia Card B Data 3
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PB4_RK0             EQU (AT91C_PIO_PB4) ;-  SSC Receive Clock
AT91C_PB4_PCK0            EQU (AT91C_PIO_PB4) ;-  PMC Programmable Clock Output 0
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PB5_RF0             EQU (AT91C_PIO_PB5) ;-  SSC Receive Frame Sync 0
AT91C_PB5_TIOA3           EQU (AT91C_PIO_PB5) ;-  Timer Counter 4 Multipurpose Timer I/O Pin A
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PB6_TF1             EQU (AT91C_PIO_PB6) ;-  SSC Transmit Frame Sync 1
AT91C_PB6_TIOB4           EQU (AT91C_PIO_PB6) ;-  Timer Counter 4 Multipurpose Timer I/O Pin B
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PB7_TK1             EQU (AT91C_PIO_PB7) ;-  SSC Transmit Clock 1
AT91C_PB7_TCLK4           EQU (AT91C_PIO_PB7) ;-  Timer Counter 4 external Clock Input
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PB8_TD1             EQU (AT91C_PIO_PB8) ;-  SSC Transmit Data 1
AT91C_PB8_NPCS1           EQU (AT91C_PIO_PB8) ;-  SPI Peripheral Chip Select 1
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PB9_RD1             EQU (AT91C_PIO_PB9) ;-  SSC Receive Data 1
AT91C_PB9_NPCS2           EQU (AT91C_PIO_PB9) ;-  SPI Peripheral Chip Select 2

/* - ******************************************************************************/
/* -               PERIPHERAL ID DEFINITIONS FOR AT91RM3400*/
/* - ******************************************************************************/
AT91C_ID_FIQ              EQU ( 0) ;- Advanced Interrupt Controller (FIQ)
AT91C_ID_SYS              EQU ( 1) ;- System Peripheral
AT91C_ID_PIOA             EQU ( 2) ;- Parallel IO Controller A 
AT91C_ID_PIOB             EQU ( 3) ;- Parallel IO Controller B
AT91C_ID_US0              EQU ( 6) ;- USART 0
AT91C_ID_US1              EQU ( 7) ;- USART 1
AT91C_ID_US2              EQU ( 8) ;- USART 2
AT91C_ID_US3              EQU ( 9) ;- USART 3
AT91C_ID_MCI              EQU (10) ;- Multimedia Card Interface
AT91C_ID_UDP              EQU (11) ;- USB Device Port
AT91C_ID_TWI              EQU (12) ;- Two-Wire Interface
AT91C_ID_SPI              EQU (13) ;- Serial Peripheral Interface
AT91C_ID_SSC0             EQU (14) ;- Serial Synchronous Controller 0
AT91C_ID_SSC1             EQU (15) ;- Serial Synchronous Controller 1
AT91C_ID_SSC2             EQU (16) ;- Serial Synchronous Controller 2
AT91C_ID_TC0              EQU (17) ;- Timer Counter 0
AT91C_ID_TC1              EQU (18) ;- Timer Counter 1
AT91C_ID_TC2              EQU (19) ;- Timer Counter 2
AT91C_ID_TC3              EQU (20) ;- Timer Counter 3
AT91C_ID_TC4              EQU (21) ;- Timer Counter 4
AT91C_ID_TC5              EQU (22) ;- Timer Counter 5
AT91C_ID_IRQ0             EQU (25) ;- Advanced Interrupt Controller (IRQ0)
AT91C_ID_IRQ1             EQU (26) ;- Advanced Interrupt Controller (IRQ1)
AT91C_ID_IRQ2             EQU (27) ;- Advanced Interrupt Controller (IRQ2)
AT91C_ID_IRQ3             EQU (28) ;- Advanced Interrupt Controller (IRQ3)
AT91C_ID_IRQ4             EQU (29) ;- Advanced Interrupt Controller (IRQ4)
AT91C_ID_IRQ5             EQU (30) ;- Advanced Interrupt Controller (IRQ5)
AT91C_ID_IRQ6             EQU (31) ;- Advanced Interrupt Controller (IRQ6)
AT91C_ALL_INT             EQU (0xFE7FFFCF) ;- ALL VALID INTERRUPTS

/* - ******************************************************************************/
/* -               BASE ADDRESS DEFINITIONS FOR AT91RM3400*/
/* - ******************************************************************************/
AT91C_BASE_SYS            EQU (0xFFFFF000) ;- (SYS) Base Address
AT91C_BASE_MC             EQU (0xFFFFFF00) ;- (MC) Base Address
AT91C_BASE_RTC            EQU (0xFFFFFE00) ;- (RTC) Base Address
AT91C_BASE_ST             EQU (0xFFFFFD00) ;- (ST) Base Address
AT91C_BASE_PMC            EQU (0xFFFFFC00) ;- (PMC) Base Address
AT91C_BASE_CKGR           EQU (0xFFFFFC20) ;- (CKGR) Base Address
AT91C_BASE_PIOB           EQU (0xFFFFF600) ;- (PIOB) Base Address
AT91C_BASE_PIOA           EQU (0xFFFFF400) ;- (PIOA) Base Address
AT91C_BASE_DBGU           EQU (0xFFFFF200) ;- (DBGU) Base Address
AT91C_BASE_PDC_DBGU       EQU (0xFFFFF300) ;- (PDC_DBGU) Base Address
AT91C_BASE_AIC            EQU (0xFFFFF000) ;- (AIC) Base Address
AT91C_BASE_PDC_SPI        EQU (0xFFFE0100) ;- (PDC_SPI) Base Address
AT91C_BASE_SPI            EQU (0xFFFE0000) ;- (SPI) Base Address
AT91C_BASE_PDC_SSC2       EQU (0xFFFD8100) ;- (PDC_SSC2) Base Address
AT91C_BASE_SSC2           EQU (0xFFFD8000) ;- (SSC2) Base Address
AT91C_BASE_PDC_SSC1       EQU (0xFFFD4100) ;- (PDC_SSC1) Base Address
AT91C_BASE_SSC1           EQU (0xFFFD4000) ;- (SSC1) Base Address
AT91C_BASE_PDC_SSC0       EQU (0xFFFD0100) ;- (PDC_SSC0) Base Address
AT91C_BASE_SSC0           EQU (0xFFFD0000) ;- (SSC0) Base Address
AT91C_BASE_PDC_US3        EQU (0xFFFCC100) ;- (PDC_US3) Base Address
AT91C_BASE_US3            EQU (0xFFFCC000) ;- (US3) Base Address
AT91C_BASE_PDC_US2        EQU (0xFFFC8100) ;- (PDC_US2) Base Address
AT91C_BASE_US2            EQU (0xFFFC8000) ;- (US2) Base Address
AT91C_BASE_PDC_US1        EQU (0xFFFC4100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0xFFFC4000) ;- (US1) Base Address
AT91C_BASE_PDC_US0        EQU (0xFFFC0100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0xFFFC0000) ;- (US0) Base Address
AT91C_BASE_TWI            EQU (0xFFFB8000) ;- (TWI) Base Address
AT91C_BASE_PDC_MCI        EQU (0xFFFB4100) ;- (PDC_MCI) Base Address
AT91C_BASE_MCI            EQU (0xFFFB4000) ;- (MCI) Base Address
AT91C_BASE_UDP            EQU (0xFFFB0000) ;- (UDP) Base Address
AT91C_BASE_TC5            EQU (0xFFFA4080) ;- (TC5) Base Address
AT91C_BASE_TC4            EQU (0xFFFA4040) ;- (TC4) Base Address
AT91C_BASE_TC3            EQU (0xFFFA4000) ;- (TC3) Base Address
AT91C_BASE_TCB1           EQU (0xFFFA4000) ;- (TCB1) Base Address
AT91C_BASE_TC2            EQU (0xFFFA0080) ;- (TC2) Base Address
AT91C_BASE_TC1            EQU (0xFFFA0040) ;- (TC1) Base Address
AT91C_BASE_TC0            EQU (0xFFFA0000) ;- (TC0) Base Address
AT91C_BASE_TCB0           EQU (0xFFFA0000) ;- (TCB0) Base Address

/* - ******************************************************************************/
/* -               MEMORY MAPPING DEFINITIONS FOR AT91RM3400*/
/* - ******************************************************************************/
/* - ISRAM*/
AT91C_ISRAM               EQU (0x00200000) ;- Internal SRAM base address
AT91C_ISRAM_SIZE          EQU (0x00018000) ;- Internal SRAM size in byte (96 Kbytes)
/* - IROM*/
AT91C_IROM                EQU (0x00100000) ;- Internal ROM base address
AT91C_IROM_SIZE           EQU (0x00040000) ;- Internal ROM size in byte (256 Kbytes)
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91RM3400_H */
