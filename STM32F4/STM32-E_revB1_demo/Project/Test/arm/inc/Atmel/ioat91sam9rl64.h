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
/* - File Name           : AT91SAM9RL64.h*/
/* - Object              : AT91SAM9RL64 definitions*/
/* - Generated           : AT91 SW Application Group  07/13/2007 (09:03:05)*/
/* - */
/* - CVS Reference       : /AT91SAM9RL64.pl/1.20/Tue Nov 14 10:59:03 2006  */
/* - CVS Reference       : /SYS_SAM9RL64.pl/1.3/Tue Nov 14 10:57:49 2006  */
/* - CVS Reference       : /HMATRIX1_SAM9RL64.pl/1.1/Wed Sep 13 16:29:30 2006  */
/* - CVS Reference       : /CCR_SAM9RL64.pl/1.1/Wed Sep 13 16:29:30 2006  */
/* - CVS Reference       : /PMC_CAP9.pl/1.2/Mon Oct 23 15:19:41 2006  */
/* - CVS Reference       : /EBI_SAM9260.pl/1.1/Fri Sep 30 12:12:14 2005  */
/* - CVS Reference       : /HSDRAMC1_6100A.pl/1.2/Mon Aug  9 10:52:25 2004  */
/* - CVS Reference       : /HSMC3_6105A.pl/1.4/Tue Nov 16 09:16:23 2004  */
/* - CVS Reference       : /HECC_6143A.pl/1.1/Wed Feb  9 17:16:57 2005  */
/* - CVS Reference       : /AIC_6075A.pl/1.1/Mon Jul 12 17:04:01 2004  */
/* - CVS Reference       : /PDC_6074C.pl/1.2/Thu Feb  3 09:02:11 2005  */
/* - CVS Reference       : /DBGU_6059D.pl/1.1/Mon Jan 31 13:54:41 2005  */
/* - CVS Reference       : /PIO_6057A.pl/1.2/Thu Feb  3 10:29:42 2005  */
/* - CVS Reference       : /RSTC_6098A.pl/1.3/Thu Nov  4 13:57:00 2004  */
/* - CVS Reference       : /SHDWC_6122A.pl/1.3/Wed Oct  6 14:16:58 2004  */
/* - CVS Reference       : /RTTC_6081A.pl/1.2/Thu Nov  4 13:57:22 2004  */
/* - CVS Reference       : /PITC_6079A.pl/1.2/Thu Nov  4 13:56:22 2004  */
/* - CVS Reference       : /WDTC_6080A.pl/1.3/Thu Nov  4 13:58:52 2004  */
/* - CVS Reference       : /TC_6082A.pl/1.7/Wed Mar  9 16:31:51 2005  */
/* - CVS Reference       : /MCI_6101E.pl/1.1/Fri Jun  3 13:20:23 2005  */
/* - CVS Reference       : /TWI_6061B.pl/1.2/Fri Aug  4 08:53:02 2006  */
/* - CVS Reference       : /US_6089J.pl/1.2/Wed Oct 11 13:26:02 2006  */
/* - CVS Reference       : /SSC_6078B.pl/1.1/Wed Jul 13 15:25:46 2005  */
/* - CVS Reference       : /SPI_6088D.pl/1.3/Fri May 20 14:23:02 2005  */
/* - CVS Reference       : /AC97C_XXXX.pl/1.3/Tue Feb 22 17:08:27 2005  */
/* - CVS Reference       : /PWM_6044D.pl/1.2/Tue May 10 12:39:09 2005  */
/* - CVS Reference       : /LCDC_6063A.pl/1.3/Fri Dec  9 10:59:26 2005  */
/* - CVS Reference       : /HDMA_SAM9RL64.pl/1.2/Wed Sep  6 16:25:21 2006  */
/* - CVS Reference       : /UDPHS_SAM9265.pl/1.8/Fri Aug 18 13:39:09 2006  */
/* - CVS Reference       : /TSC_XXXX.pl/1.2/Tue Nov 14 10:58:20 2006  */
/* - CVS Reference       : /RTC_1245D.pl/1.3/Fri Sep 17 14:01:31 2004  */
/* - ----------------------------------------------------------------------------*/

#ifndef AT91SAM9RL64_H
#define AT91SAM9RL64_H

#ifdef __IAR_SYSTEMS_ICC__

typedef volatile unsigned int AT91_REG;/* Hardware register definition*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR System Peripherals*/
/* ******************************************************************************/
typedef struct _AT91S_SYS {
	AT91_REG	 Reserved0[3904]; 	/* */
	AT91_REG	 SYS_RSTC_RCR; 	/* Reset Control Register*/
	AT91_REG	 SYS_RSTC_RSR; 	/* Reset Status Register*/
	AT91_REG	 SYS_RSTC_RMR; 	/* Reset Mode Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 SYS_SHDWC_SHCR; 	/* Shut Down Control Register*/
	AT91_REG	 SYS_SHDWC_SHMR; 	/* Shut Down Mode Register*/
	AT91_REG	 SYS_SHDWC_SHSR; 	/* Shut Down Status Register*/
	AT91_REG	 Reserved2[1]; 	/* */
	AT91_REG	 SYS_RTTC0_RTMR; 	/* Real-time Mode Register*/
	AT91_REG	 SYS_RTTC0_RTAR; 	/* Real-time Alarm Register*/
	AT91_REG	 SYS_RTTC0_RTVR; 	/* Real-time Value Register*/
	AT91_REG	 SYS_RTTC0_RTSR; 	/* Real-time Status Register*/
	AT91_REG	 SYS_PITC_PIMR; 	/* Period Interval Mode Register*/
	AT91_REG	 SYS_PITC_PISR; 	/* Period Interval Status Register*/
	AT91_REG	 SYS_PITC_PIVR; 	/* Period Interval Value Register*/
	AT91_REG	 SYS_PITC_PIIR; 	/* Period Interval Image Register*/
	AT91_REG	 SYS_WDTC_WDCR; 	/* Watchdog Control Register*/
	AT91_REG	 SYS_WDTC_WDMR; 	/* Watchdog Mode Register*/
	AT91_REG	 SYS_WDTC_WDSR; 	/* Watchdog Status Register*/
	AT91_REG	 Reserved3[1]; 	/* */
	AT91_REG	 SYS_SLCKSEL; 	/* Slow Clock Selection Register*/
	AT91_REG	 Reserved4[3]; 	/* */
	AT91_REG	 SYS_GPBR[4]; 	/* General Purpose Register*/
} AT91S_SYS, *AT91PS_SYS;

/* -------- SLCKSEL : (SYS Offset: 0x3d50) Slow Clock Selection Register -------- */
#define AT91C_SLCKSEL_RCEN    ((unsigned int) 0x1 <<  0) /* (SYS) Enable Internal RC Oscillator*/
#define AT91C_SLCKSEL_OSC32EN ((unsigned int) 0x1 <<  1) /* (SYS) Enable External Oscillator*/
#define AT91C_SLCKSEL_OSC32BYP ((unsigned int) 0x1 <<  2) /* (SYS) Bypass External Oscillator*/
#define AT91C_SLCKSEL_OSCSEL  ((unsigned int) 0x1 <<  3) /* (SYS) OSC Selection*/
/* -------- GPBR : (SYS Offset: 0x3d60) GPBR General Purpose Register -------- */
#define AT91C_GPBR_GPRV       ((unsigned int) 0x0 <<  0) /* (SYS) General Purpose Register Value*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR External Bus Interface*/
/* ******************************************************************************/
typedef struct _AT91S_EBI {
	AT91_REG	 EBI_DUMMY; 	/* Dummy register - Do not use*/
} AT91S_EBI, *AT91PS_EBI;


/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR SDRAM Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_SDRAMC {
	AT91_REG	 SDRAMC_MR; 	/* SDRAM Controller Mode Register*/
	AT91_REG	 SDRAMC_TR; 	/* SDRAM Controller Refresh Timer Register*/
	AT91_REG	 SDRAMC_CR; 	/* SDRAM Controller Configuration Register*/
	AT91_REG	 SDRAMC_HSR; 	/* SDRAM Controller High Speed Register*/
	AT91_REG	 SDRAMC_LPR; 	/* SDRAM Controller Low Power Register*/
	AT91_REG	 SDRAMC_IER; 	/* SDRAM Controller Interrupt Enable Register*/
	AT91_REG	 SDRAMC_IDR; 	/* SDRAM Controller Interrupt Disable Register*/
	AT91_REG	 SDRAMC_IMR; 	/* SDRAM Controller Interrupt Mask Register*/
	AT91_REG	 SDRAMC_ISR; 	/* SDRAM Controller Interrupt Mask Register*/
	AT91_REG	 SDRAMC_MDR; 	/* SDRAM Memory Device Register*/
} AT91S_SDRAMC, *AT91PS_SDRAMC;

/* -------- SDRAMC_MR : (SDRAMC Offset: 0x0) SDRAM Controller Mode Register -------- */
#define AT91C_SDRAMC_MODE     ((unsigned int) 0xF <<  0) /* (SDRAMC) Mode*/
#define 	AT91C_SDRAMC_MODE_NORMAL_CMD           ((unsigned int) 0x0) /* (SDRAMC) Normal Mode*/
#define 	AT91C_SDRAMC_MODE_NOP_CMD              ((unsigned int) 0x1) /* (SDRAMC) Issue a NOP Command at every access*/
#define 	AT91C_SDRAMC_MODE_PRCGALL_CMD          ((unsigned int) 0x2) /* (SDRAMC) Issue a All Banks Precharge Command at every access*/
#define 	AT91C_SDRAMC_MODE_LMR_CMD              ((unsigned int) 0x3) /* (SDRAMC) Issue a Load Mode Register at every access*/
#define 	AT91C_SDRAMC_MODE_RFSH_CMD             ((unsigned int) 0x4) /* (SDRAMC) Issue a Refresh*/
#define 	AT91C_SDRAMC_MODE_EXT_LMR_CMD          ((unsigned int) 0x5) /* (SDRAMC) Issue an Extended Load Mode Register*/
#define 	AT91C_SDRAMC_MODE_DEEP_CMD             ((unsigned int) 0x6) /* (SDRAMC) Enter Deep Power Mode*/
/* -------- SDRAMC_TR : (SDRAMC Offset: 0x4) SDRAMC Refresh Timer Register -------- */
#define AT91C_SDRAMC_COUNT    ((unsigned int) 0xFFF <<  0) /* (SDRAMC) Refresh Counter*/
/* -------- SDRAMC_CR : (SDRAMC Offset: 0x8) SDRAM Configuration Register -------- */
#define AT91C_SDRAMC_NC       ((unsigned int) 0x3 <<  0) /* (SDRAMC) Number of Column Bits*/
#define 	AT91C_SDRAMC_NC_8                    ((unsigned int) 0x0) /* (SDRAMC) 8 Bits*/
#define 	AT91C_SDRAMC_NC_9                    ((unsigned int) 0x1) /* (SDRAMC) 9 Bits*/
#define 	AT91C_SDRAMC_NC_10                   ((unsigned int) 0x2) /* (SDRAMC) 10 Bits*/
#define 	AT91C_SDRAMC_NC_11                   ((unsigned int) 0x3) /* (SDRAMC) 11 Bits*/
#define AT91C_SDRAMC_NR       ((unsigned int) 0x3 <<  2) /* (SDRAMC) Number of Row Bits*/
#define 	AT91C_SDRAMC_NR_11                   ((unsigned int) 0x0 <<  2) /* (SDRAMC) 11 Bits*/
#define 	AT91C_SDRAMC_NR_12                   ((unsigned int) 0x1 <<  2) /* (SDRAMC) 12 Bits*/
#define 	AT91C_SDRAMC_NR_13                   ((unsigned int) 0x2 <<  2) /* (SDRAMC) 13 Bits*/
#define AT91C_SDRAMC_NB       ((unsigned int) 0x1 <<  4) /* (SDRAMC) Number of Banks*/
#define 	AT91C_SDRAMC_NB_2_BANKS              ((unsigned int) 0x0 <<  4) /* (SDRAMC) 2 banks*/
#define 	AT91C_SDRAMC_NB_4_BANKS              ((unsigned int) 0x1 <<  4) /* (SDRAMC) 4 banks*/
#define AT91C_SDRAMC_CAS      ((unsigned int) 0x3 <<  5) /* (SDRAMC) CAS Latency*/
#define 	AT91C_SDRAMC_CAS_2                    ((unsigned int) 0x2 <<  5) /* (SDRAMC) 2 cycles*/
#define 	AT91C_SDRAMC_CAS_3                    ((unsigned int) 0x3 <<  5) /* (SDRAMC) 3 cycles*/
#define AT91C_SDRAMC_DBW      ((unsigned int) 0x1 <<  7) /* (SDRAMC) Data Bus Width*/
#define 	AT91C_SDRAMC_DBW_32_BITS              ((unsigned int) 0x0 <<  7) /* (SDRAMC) 32 Bits datas bus*/
#define 	AT91C_SDRAMC_DBW_16_BITS              ((unsigned int) 0x1 <<  7) /* (SDRAMC) 16 Bits datas bus*/
#define AT91C_SDRAMC_TWR      ((unsigned int) 0xF <<  8) /* (SDRAMC) Number of Write Recovery Time Cycles*/
#define 	AT91C_SDRAMC_TWR_0                    ((unsigned int) 0x0 <<  8) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TWR_1                    ((unsigned int) 0x1 <<  8) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TWR_2                    ((unsigned int) 0x2 <<  8) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TWR_3                    ((unsigned int) 0x3 <<  8) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TWR_4                    ((unsigned int) 0x4 <<  8) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TWR_5                    ((unsigned int) 0x5 <<  8) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TWR_6                    ((unsigned int) 0x6 <<  8) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TWR_7                    ((unsigned int) 0x7 <<  8) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TWR_8                    ((unsigned int) 0x8 <<  8) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TWR_9                    ((unsigned int) 0x9 <<  8) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TWR_10                   ((unsigned int) 0xA <<  8) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TWR_11                   ((unsigned int) 0xB <<  8) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TWR_12                   ((unsigned int) 0xC <<  8) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TWR_13                   ((unsigned int) 0xD <<  8) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TWR_14                   ((unsigned int) 0xE <<  8) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TWR_15                   ((unsigned int) 0xF <<  8) /* (SDRAMC) Value : 15*/
#define AT91C_SDRAMC_TRC      ((unsigned int) 0xF << 12) /* (SDRAMC) Number of RAS Cycle Time Cycles*/
#define 	AT91C_SDRAMC_TRC_0                    ((unsigned int) 0x0 << 12) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TRC_1                    ((unsigned int) 0x1 << 12) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TRC_2                    ((unsigned int) 0x2 << 12) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TRC_3                    ((unsigned int) 0x3 << 12) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TRC_4                    ((unsigned int) 0x4 << 12) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TRC_5                    ((unsigned int) 0x5 << 12) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TRC_6                    ((unsigned int) 0x6 << 12) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TRC_7                    ((unsigned int) 0x7 << 12) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TRC_8                    ((unsigned int) 0x8 << 12) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TRC_9                    ((unsigned int) 0x9 << 12) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TRC_10                   ((unsigned int) 0xA << 12) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TRC_11                   ((unsigned int) 0xB << 12) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TRC_12                   ((unsigned int) 0xC << 12) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TRC_13                   ((unsigned int) 0xD << 12) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TRC_14                   ((unsigned int) 0xE << 12) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TRC_15                   ((unsigned int) 0xF << 12) /* (SDRAMC) Value : 15*/
#define AT91C_SDRAMC_TRP      ((unsigned int) 0xF << 16) /* (SDRAMC) Number of RAS Precharge Time Cycles*/
#define 	AT91C_SDRAMC_TRP_0                    ((unsigned int) 0x0 << 16) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TRP_1                    ((unsigned int) 0x1 << 16) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TRP_2                    ((unsigned int) 0x2 << 16) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TRP_3                    ((unsigned int) 0x3 << 16) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TRP_4                    ((unsigned int) 0x4 << 16) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TRP_5                    ((unsigned int) 0x5 << 16) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TRP_6                    ((unsigned int) 0x6 << 16) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TRP_7                    ((unsigned int) 0x7 << 16) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TRP_8                    ((unsigned int) 0x8 << 16) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TRP_9                    ((unsigned int) 0x9 << 16) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TRP_10                   ((unsigned int) 0xA << 16) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TRP_11                   ((unsigned int) 0xB << 16) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TRP_12                   ((unsigned int) 0xC << 16) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TRP_13                   ((unsigned int) 0xD << 16) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TRP_14                   ((unsigned int) 0xE << 16) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TRP_15                   ((unsigned int) 0xF << 16) /* (SDRAMC) Value : 15*/
#define AT91C_SDRAMC_TRCD     ((unsigned int) 0xF << 20) /* (SDRAMC) Number of RAS to CAS Delay Cycles*/
#define 	AT91C_SDRAMC_TRCD_0                    ((unsigned int) 0x0 << 20) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TRCD_1                    ((unsigned int) 0x1 << 20) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TRCD_2                    ((unsigned int) 0x2 << 20) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TRCD_3                    ((unsigned int) 0x3 << 20) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TRCD_4                    ((unsigned int) 0x4 << 20) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TRCD_5                    ((unsigned int) 0x5 << 20) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TRCD_6                    ((unsigned int) 0x6 << 20) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TRCD_7                    ((unsigned int) 0x7 << 20) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TRCD_8                    ((unsigned int) 0x8 << 20) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TRCD_9                    ((unsigned int) 0x9 << 20) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TRCD_10                   ((unsigned int) 0xA << 20) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TRCD_11                   ((unsigned int) 0xB << 20) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TRCD_12                   ((unsigned int) 0xC << 20) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TRCD_13                   ((unsigned int) 0xD << 20) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TRCD_14                   ((unsigned int) 0xE << 20) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TRCD_15                   ((unsigned int) 0xF << 20) /* (SDRAMC) Value : 15*/
#define AT91C_SDRAMC_TRAS     ((unsigned int) 0xF << 24) /* (SDRAMC) Number of RAS Active Time Cycles*/
#define 	AT91C_SDRAMC_TRAS_0                    ((unsigned int) 0x0 << 24) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TRAS_1                    ((unsigned int) 0x1 << 24) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TRAS_2                    ((unsigned int) 0x2 << 24) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TRAS_3                    ((unsigned int) 0x3 << 24) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TRAS_4                    ((unsigned int) 0x4 << 24) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TRAS_5                    ((unsigned int) 0x5 << 24) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TRAS_6                    ((unsigned int) 0x6 << 24) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TRAS_7                    ((unsigned int) 0x7 << 24) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TRAS_8                    ((unsigned int) 0x8 << 24) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TRAS_9                    ((unsigned int) 0x9 << 24) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TRAS_10                   ((unsigned int) 0xA << 24) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TRAS_11                   ((unsigned int) 0xB << 24) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TRAS_12                   ((unsigned int) 0xC << 24) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TRAS_13                   ((unsigned int) 0xD << 24) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TRAS_14                   ((unsigned int) 0xE << 24) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TRAS_15                   ((unsigned int) 0xF << 24) /* (SDRAMC) Value : 15*/
#define AT91C_SDRAMC_TXSR     ((unsigned int) 0xF << 28) /* (SDRAMC) Number of Command Recovery Time Cycles*/
#define 	AT91C_SDRAMC_TXSR_0                    ((unsigned int) 0x0 << 28) /* (SDRAMC) Value :  0*/
#define 	AT91C_SDRAMC_TXSR_1                    ((unsigned int) 0x1 << 28) /* (SDRAMC) Value :  1*/
#define 	AT91C_SDRAMC_TXSR_2                    ((unsigned int) 0x2 << 28) /* (SDRAMC) Value :  2*/
#define 	AT91C_SDRAMC_TXSR_3                    ((unsigned int) 0x3 << 28) /* (SDRAMC) Value :  3*/
#define 	AT91C_SDRAMC_TXSR_4                    ((unsigned int) 0x4 << 28) /* (SDRAMC) Value :  4*/
#define 	AT91C_SDRAMC_TXSR_5                    ((unsigned int) 0x5 << 28) /* (SDRAMC) Value :  5*/
#define 	AT91C_SDRAMC_TXSR_6                    ((unsigned int) 0x6 << 28) /* (SDRAMC) Value :  6*/
#define 	AT91C_SDRAMC_TXSR_7                    ((unsigned int) 0x7 << 28) /* (SDRAMC) Value :  7*/
#define 	AT91C_SDRAMC_TXSR_8                    ((unsigned int) 0x8 << 28) /* (SDRAMC) Value :  8*/
#define 	AT91C_SDRAMC_TXSR_9                    ((unsigned int) 0x9 << 28) /* (SDRAMC) Value :  9*/
#define 	AT91C_SDRAMC_TXSR_10                   ((unsigned int) 0xA << 28) /* (SDRAMC) Value : 10*/
#define 	AT91C_SDRAMC_TXSR_11                   ((unsigned int) 0xB << 28) /* (SDRAMC) Value : 11*/
#define 	AT91C_SDRAMC_TXSR_12                   ((unsigned int) 0xC << 28) /* (SDRAMC) Value : 12*/
#define 	AT91C_SDRAMC_TXSR_13                   ((unsigned int) 0xD << 28) /* (SDRAMC) Value : 13*/
#define 	AT91C_SDRAMC_TXSR_14                   ((unsigned int) 0xE << 28) /* (SDRAMC) Value : 14*/
#define 	AT91C_SDRAMC_TXSR_15                   ((unsigned int) 0xF << 28) /* (SDRAMC) Value : 15*/
/* -------- SDRAMC_HSR : (SDRAMC Offset: 0xc) SDRAM Controller High Speed Register -------- */
#define AT91C_SDRAMC_DA       ((unsigned int) 0x1 <<  0) /* (SDRAMC) Decode Cycle Enable Bit*/
#define 	AT91C_SDRAMC_DA_DISABLE              ((unsigned int) 0x0) /* (SDRAMC) Disable Decode Cycle*/
#define 	AT91C_SDRAMC_DA_ENABLE               ((unsigned int) 0x1) /* (SDRAMC) Enable Decode Cycle*/
/* -------- SDRAMC_LPR : (SDRAMC Offset: 0x10) SDRAM Controller Low-power Register -------- */
#define AT91C_SDRAMC_LPCB     ((unsigned int) 0x3 <<  0) /* (SDRAMC) Low-power Configurations*/
#define 	AT91C_SDRAMC_LPCB_DISABLE              ((unsigned int) 0x0) /* (SDRAMC) Disable Low Power Features*/
#define 	AT91C_SDRAMC_LPCB_SELF_REFRESH         ((unsigned int) 0x1) /* (SDRAMC) Enable SELF_REFRESH*/
#define 	AT91C_SDRAMC_LPCB_POWER_DOWN           ((unsigned int) 0x2) /* (SDRAMC) Enable POWER_DOWN*/
#define 	AT91C_SDRAMC_LPCB_DEEP_POWER_DOWN      ((unsigned int) 0x3) /* (SDRAMC) Enable DEEP_POWER_DOWN*/
#define AT91C_SDRAMC_PASR     ((unsigned int) 0x7 <<  4) /* (SDRAMC) Partial Array Self Refresh (only for Low Power SDRAM)*/
#define AT91C_SDRAMC_TCSR     ((unsigned int) 0x3 <<  8) /* (SDRAMC) Temperature Compensated Self Refresh (only for Low Power SDRAM)*/
#define AT91C_SDRAMC_DS       ((unsigned int) 0x3 << 10) /* (SDRAMC) Drive Strenght (only for Low Power SDRAM)*/
#define AT91C_SDRAMC_TIMEOUT  ((unsigned int) 0x3 << 12) /* (SDRAMC) Time to define when Low Power Mode is enabled*/
#define 	AT91C_SDRAMC_TIMEOUT_0_CLK_CYCLES         ((unsigned int) 0x0 << 12) /* (SDRAMC) Activate SDRAM Low Power Mode Immediately*/
#define 	AT91C_SDRAMC_TIMEOUT_64_CLK_CYCLES        ((unsigned int) 0x1 << 12) /* (SDRAMC) Activate SDRAM Low Power Mode after 64 clock cycles after the end of the last transfer*/
#define 	AT91C_SDRAMC_TIMEOUT_128_CLK_CYCLES       ((unsigned int) 0x2 << 12) /* (SDRAMC) Activate SDRAM Low Power Mode after 64 clock cycles after the end of the last transfer*/
/* -------- SDRAMC_IER : (SDRAMC Offset: 0x14) SDRAM Controller Interrupt Enable Register -------- */
#define AT91C_SDRAMC_RES      ((unsigned int) 0x1 <<  0) /* (SDRAMC) Refresh Error Status*/
/* -------- SDRAMC_IDR : (SDRAMC Offset: 0x18) SDRAM Controller Interrupt Disable Register -------- */
/* -------- SDRAMC_IMR : (SDRAMC Offset: 0x1c) SDRAM Controller Interrupt Mask Register -------- */
/* -------- SDRAMC_ISR : (SDRAMC Offset: 0x20) SDRAM Controller Interrupt Status Register -------- */
/* -------- SDRAMC_MDR : (SDRAMC Offset: 0x24) SDRAM Controller Memory Device Register -------- */
#define AT91C_SDRAMC_MD       ((unsigned int) 0x3 <<  0) /* (SDRAMC) Memory Device Type*/
#define 	AT91C_SDRAMC_MD_SDRAM                ((unsigned int) 0x0) /* (SDRAMC) SDRAM Mode*/
#define 	AT91C_SDRAMC_MD_LOW_POWER_SDRAM      ((unsigned int) 0x1) /* (SDRAMC) SDRAM Low Power Mode*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_SMC {
	AT91_REG	 SMC_SETUP0; 	/*  Setup Register for CS 0*/
	AT91_REG	 SMC_PULSE0; 	/*  Pulse Register for CS 0*/
	AT91_REG	 SMC_CYCLE0; 	/*  Cycle Register for CS 0*/
	AT91_REG	 SMC_CTRL0; 	/*  Control Register for CS 0*/
	AT91_REG	 SMC_SETUP1; 	/*  Setup Register for CS 1*/
	AT91_REG	 SMC_PULSE1; 	/*  Pulse Register for CS 1*/
	AT91_REG	 SMC_CYCLE1; 	/*  Cycle Register for CS 1*/
	AT91_REG	 SMC_CTRL1; 	/*  Control Register for CS 1*/
	AT91_REG	 SMC_SETUP2; 	/*  Setup Register for CS 2*/
	AT91_REG	 SMC_PULSE2; 	/*  Pulse Register for CS 2*/
	AT91_REG	 SMC_CYCLE2; 	/*  Cycle Register for CS 2*/
	AT91_REG	 SMC_CTRL2; 	/*  Control Register for CS 2*/
	AT91_REG	 SMC_SETUP3; 	/*  Setup Register for CS 3*/
	AT91_REG	 SMC_PULSE3; 	/*  Pulse Register for CS 3*/
	AT91_REG	 SMC_CYCLE3; 	/*  Cycle Register for CS 3*/
	AT91_REG	 SMC_CTRL3; 	/*  Control Register for CS 3*/
	AT91_REG	 SMC_SETUP4; 	/*  Setup Register for CS 4*/
	AT91_REG	 SMC_PULSE4; 	/*  Pulse Register for CS 4*/
	AT91_REG	 SMC_CYCLE4; 	/*  Cycle Register for CS 4*/
	AT91_REG	 SMC_CTRL4; 	/*  Control Register for CS 4*/
	AT91_REG	 SMC_SETUP5; 	/*  Setup Register for CS 5*/
	AT91_REG	 SMC_PULSE5; 	/*  Pulse Register for CS 5*/
	AT91_REG	 SMC_CYCLE5; 	/*  Cycle Register for CS 5*/
	AT91_REG	 SMC_CTRL5; 	/*  Control Register for CS 5*/
	AT91_REG	 SMC_SETUP6; 	/*  Setup Register for CS 6*/
	AT91_REG	 SMC_PULSE6; 	/*  Pulse Register for CS 6*/
	AT91_REG	 SMC_CYCLE6; 	/*  Cycle Register for CS 6*/
	AT91_REG	 SMC_CTRL6; 	/*  Control Register for CS 6*/
	AT91_REG	 SMC_SETUP7; 	/*  Setup Register for CS 7*/
	AT91_REG	 SMC_PULSE7; 	/*  Pulse Register for CS 7*/
	AT91_REG	 SMC_CYCLE7; 	/*  Cycle Register for CS 7*/
	AT91_REG	 SMC_CTRL7; 	/*  Control Register for CS 7*/
} AT91S_SMC, *AT91PS_SMC;

/* -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- */
#define AT91C_SMC_NWESETUP    ((unsigned int) 0x3F <<  0) /* (SMC) NWE Setup Length*/
#define AT91C_SMC_NCSSETUPWR  ((unsigned int) 0x3F <<  8) /* (SMC) NCS Setup Length in WRite Access*/
#define AT91C_SMC_NRDSETUP    ((unsigned int) 0x3F << 16) /* (SMC) NRD Setup Length*/
#define AT91C_SMC_NCSSETUPRD  ((unsigned int) 0x3F << 24) /* (SMC) NCS Setup Length in ReaD Access*/
/* -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- */
#define AT91C_SMC_NWEPULSE    ((unsigned int) 0x7F <<  0) /* (SMC) NWE Pulse Length*/
#define AT91C_SMC_NCSPULSEWR  ((unsigned int) 0x7F <<  8) /* (SMC) NCS Pulse Length in WRite Access*/
#define AT91C_SMC_NRDPULSE    ((unsigned int) 0x7F << 16) /* (SMC) NRD Pulse Length*/
#define AT91C_SMC_NCSPULSERD  ((unsigned int) 0x7F << 24) /* (SMC) NCS Pulse Length in ReaD Access*/
/* -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- */
#define AT91C_SMC_NWECYCLE    ((unsigned int) 0x1FF <<  0) /* (SMC) Total Write Cycle Length*/
#define AT91C_SMC_NRDCYCLE    ((unsigned int) 0x1FF << 16) /* (SMC) Total Read Cycle Length*/
/* -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- */
#define AT91C_SMC_READMODE    ((unsigned int) 0x1 <<  0) /* (SMC) Read Mode*/
#define AT91C_SMC_WRITEMODE   ((unsigned int) 0x1 <<  1) /* (SMC) Write Mode*/
#define AT91C_SMC_NWAITM      ((unsigned int) 0x3 <<  5) /* (SMC) NWAIT Mode*/
#define 	AT91C_SMC_NWAITM_NWAIT_DISABLE        ((unsigned int) 0x0 <<  5) /* (SMC) External NWAIT disabled.*/
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_FROZEN  ((unsigned int) 0x2 <<  5) /* (SMC) External NWAIT enabled in frozen mode.*/
#define 	AT91C_SMC_NWAITM_NWAIT_ENABLE_READY   ((unsigned int) 0x3 <<  5) /* (SMC) External NWAIT enabled in ready mode.*/
#define AT91C_SMC_BAT         ((unsigned int) 0x1 <<  8) /* (SMC) Byte Access Type*/
#define 	AT91C_SMC_BAT_BYTE_SELECT          ((unsigned int) 0x0 <<  8) /* (SMC) Write controled by ncs, nbs0, nbs1, nbs2, nbs3. Read controled by ncs, nrd, nbs0, nbs1, nbs2, nbs3.*/
#define 	AT91C_SMC_BAT_BYTE_WRITE           ((unsigned int) 0x1 <<  8) /* (SMC) Write controled by ncs, nwe0, nwe1, nwe2, nwe3. Read controled by ncs and nrd.*/
#define AT91C_SMC_DBW         ((unsigned int) 0x3 << 12) /* (SMC) Data Bus Width*/
#define 	AT91C_SMC_DBW_WIDTH_EIGTH_BITS     ((unsigned int) 0x0 << 12) /* (SMC) 8 bits.*/
#define 	AT91C_SMC_DBW_WIDTH_SIXTEEN_BITS   ((unsigned int) 0x1 << 12) /* (SMC) 16 bits.*/
#define 	AT91C_SMC_DBW_WIDTH_THIRTY_TWO_BITS ((unsigned int) 0x2 << 12) /* (SMC) 32 bits.*/
#define AT91C_SMC_TDF         ((unsigned int) 0xF << 16) /* (SMC) Data Float Time.*/
#define AT91C_SMC_TDFEN       ((unsigned int) 0x1 << 20) /* (SMC) TDF Enabled.*/
#define AT91C_SMC_PMEN        ((unsigned int) 0x1 << 24) /* (SMC) Page Mode Enabled.*/
#define AT91C_SMC_PS          ((unsigned int) 0x3 << 28) /* (SMC) Page Size*/
#define 	AT91C_SMC_PS_SIZE_FOUR_BYTES      ((unsigned int) 0x0 << 28) /* (SMC) 4 bytes.*/
#define 	AT91C_SMC_PS_SIZE_EIGHT_BYTES     ((unsigned int) 0x1 << 28) /* (SMC) 8 bytes.*/
#define 	AT91C_SMC_PS_SIZE_SIXTEEN_BYTES   ((unsigned int) 0x2 << 28) /* (SMC) 16 bytes.*/
#define 	AT91C_SMC_PS_SIZE_THIRTY_TWO_BYTES ((unsigned int) 0x3 << 28) /* (SMC) 32 bytes.*/
/* -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- */
/* -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- */
/* -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- */
/* -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- */
/* -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR AHB Matrix Interface*/
/* ******************************************************************************/
typedef struct _AT91S_MATRIX {
	AT91_REG	 MATRIX_MCFG0; 	/*  Master Configuration Register 0 : rom*/
	AT91_REG	 MATRIX_MCFG1; 	/*  Master Configuration Register 1 ; htcm*/
	AT91_REG	 MATRIX_MCFG2; 	/*  Master Configuration Register 2 : lcdc*/
	AT91_REG	 MATRIX_MCFG3; 	/*  Master Configuration Register 3 : usb_dev_hs*/
	AT91_REG	 MATRIX_MCFG4; 	/*  Master Configuration Register 4 : ebi*/
	AT91_REG	 MATRIX_MCFG5; 	/*  Master Configuration Register 5 : bridge*/
	AT91_REG	 MATRIX_MCFG6; 	/*  Master Configuration Register 6 */
	AT91_REG	 MATRIX_MCFG7; 	/*  Master Configuration Register 7 */
	AT91_REG	 MATRIX_MCFG8; 	/*  Master Configuration Register 8 */
	AT91_REG	 Reserved0[7]; 	/* */
	AT91_REG	 MATRIX_SCFG0; 	/*  Slave Configuration Register 0 : rom*/
	AT91_REG	 MATRIX_SCFG1; 	/*  Slave Configuration Register 1 : htcm*/
	AT91_REG	 MATRIX_SCFG2; 	/*  Slave Configuration Register 2 : lcdc*/
	AT91_REG	 MATRIX_SCFG3; 	/*  Slave Configuration Register 3 : usb_dev_hs*/
	AT91_REG	 MATRIX_SCFG4; 	/*  Slave Configuration Register 4 ; ebi*/
	AT91_REG	 MATRIX_SCFG5; 	/*  Slave Configuration Register 5 : bridge*/
	AT91_REG	 MATRIX_SCFG6; 	/*  Slave Configuration Register 6*/
	AT91_REG	 MATRIX_SCFG7; 	/*  Slave Configuration Register 7*/
	AT91_REG	 Reserved1[8]; 	/* */
	AT91_REG	 MATRIX_PRAS0; 	/*  PRAS0 : rom*/
	AT91_REG	 MATRIX_PRBS0; 	/*  PRBS0 : rom*/
	AT91_REG	 MATRIX_PRAS1; 	/*  PRAS1 : htcm*/
	AT91_REG	 MATRIX_PRBS1; 	/*  PRBS1 : htcm*/
	AT91_REG	 MATRIX_PRAS2; 	/*  PRAS2 : lcdc*/
	AT91_REG	 MATRIX_PRBS2; 	/*  PRBS2 : lcdc*/
	AT91_REG	 MATRIX_PRAS3; 	/*  PRAS3 : usb_dev_hs*/
	AT91_REG	 MATRIX_PRBS3; 	/*  PRBS3 : usb_dev_hs*/
	AT91_REG	 MATRIX_PRAS4; 	/*  PRAS4 : ebi*/
	AT91_REG	 MATRIX_PRBS4; 	/*  PRBS4 : ebi*/
	AT91_REG	 MATRIX_PRAS5; 	/*  PRAS5 : bridge*/
	AT91_REG	 MATRIX_PRBS5; 	/*  PRBS5 : bridge*/
	AT91_REG	 MATRIX_PRAS6; 	/*  PRAS6*/
	AT91_REG	 MATRIX_PRBS6; 	/*  PRBS6*/
	AT91_REG	 MATRIX_PRAS7; 	/*  PRAS7*/
	AT91_REG	 MATRIX_PRBS7; 	/*  PRBS7*/
	AT91_REG	 Reserved2[16]; 	/* */
	AT91_REG	 MATRIX_MRCR; 	/*  Master Remp Control Register */
} AT91S_MATRIX, *AT91PS_MATRIX;

/* -------- MATRIX_MCFG0 : (MATRIX Offset: 0x0) Master Configuration Register rom -------- */
#define AT91C_MATRIX_ULBT     ((unsigned int) 0x7 <<  0) /* (MATRIX) Undefined Length Burst Type*/
/* -------- MATRIX_MCFG1 : (MATRIX Offset: 0x4) Master Configuration Register htcm -------- */
/* -------- MATRIX_MCFG2 : (MATRIX Offset: 0x8) Master Configuration Register gps_tcm -------- */
/* -------- MATRIX_MCFG3 : (MATRIX Offset: 0xc) Master Configuration Register hperiphs -------- */
/* -------- MATRIX_MCFG4 : (MATRIX Offset: 0x10) Master Configuration Register ebi0 -------- */
/* -------- MATRIX_MCFG5 : (MATRIX Offset: 0x14) Master Configuration Register ebi1 -------- */
/* -------- MATRIX_MCFG6 : (MATRIX Offset: 0x18) Master Configuration Register bridge -------- */
/* -------- MATRIX_MCFG7 : (MATRIX Offset: 0x1c) Master Configuration Register gps -------- */
/* -------- MATRIX_MCFG8 : (MATRIX Offset: 0x20) Master Configuration Register gps -------- */
/* -------- MATRIX_SCFG0 : (MATRIX Offset: 0x40) Slave Configuration Register 0 -------- */
#define AT91C_MATRIX_SLOT_CYCLE ((unsigned int) 0xFF <<  0) /* (MATRIX) Maximum Number of Allowed Cycles for a Burst*/
#define AT91C_MATRIX_DEFMSTR_TYPE ((unsigned int) 0x3 << 16) /* (MATRIX) Default Master Type*/
#define 	AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR           ((unsigned int) 0x0 << 16) /* (MATRIX) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.*/
#define 	AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR         ((unsigned int) 0x1 << 16) /* (MATRIX) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.*/
#define 	AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR        ((unsigned int) 0x2 << 16) /* (MATRIX) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.*/
#define AT91C_MATRIX_FIXED_DEFMSTR0 ((unsigned int) 0x7 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_LCDC                 ((unsigned int) 0x3 << 18) /* (MATRIX) LCDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_2DGC                 ((unsigned int) 0x4 << 18) /* (MATRIX) 2DGC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_ISI                  ((unsigned int) 0x5 << 18) /* (MATRIX) ISI Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_EMAC                 ((unsigned int) 0x7 << 18) /* (MATRIX) EMAC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR0_USB                  ((unsigned int) 0x8 << 18) /* (MATRIX) USB Master is Default Master*/
#define AT91C_MATRIX_ARBT     ((unsigned int) 0x3 << 24) /* (MATRIX) Arbitration Type*/
/* -------- MATRIX_SCFG1 : (MATRIX Offset: 0x44) Slave Configuration Register 1 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR1 ((unsigned int) 0x7 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_LCDC                 ((unsigned int) 0x3 << 18) /* (MATRIX) LCDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_2DGC                 ((unsigned int) 0x4 << 18) /* (MATRIX) 2DGC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_ISI                  ((unsigned int) 0x5 << 18) /* (MATRIX) ISI Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_EMAC                 ((unsigned int) 0x7 << 18) /* (MATRIX) EMAC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR1_USB                  ((unsigned int) 0x8 << 18) /* (MATRIX) USB Master is Default Master*/
/* -------- MATRIX_SCFG2 : (MATRIX Offset: 0x48) Slave Configuration Register 2 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR2 ((unsigned int) 0x1 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR2_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR2_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR2_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
/* -------- MATRIX_SCFG3 : (MATRIX Offset: 0x4c) Slave Configuration Register 3 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR3 ((unsigned int) 0x7 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_LCDC                 ((unsigned int) 0x3 << 18) /* (MATRIX) LCDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_2DGC                 ((unsigned int) 0x4 << 18) /* (MATRIX) 2DGC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_ISI                  ((unsigned int) 0x5 << 18) /* (MATRIX) ISI Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_EMAC                 ((unsigned int) 0x7 << 18) /* (MATRIX) EMAC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR3_USB                  ((unsigned int) 0x8 << 18) /* (MATRIX) USB Master is Default Master*/
/* -------- MATRIX_SCFG4 : (MATRIX Offset: 0x50) Slave Configuration Register 4 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR4 ((unsigned int) 0x3 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR4_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
/* -------- MATRIX_SCFG5 : (MATRIX Offset: 0x54) Slave Configuration Register 5 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR5 ((unsigned int) 0x3 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_LCDC                 ((unsigned int) 0x3 << 18) /* (MATRIX) LCDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_2DGC                 ((unsigned int) 0x4 << 18) /* (MATRIX) 2DGC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_ISI                  ((unsigned int) 0x5 << 18) /* (MATRIX) ISI Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_EMAC                 ((unsigned int) 0x7 << 18) /* (MATRIX) EMAC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR5_USB                  ((unsigned int) 0x8 << 18) /* (MATRIX) USB Master is Default Master*/
/* -------- MATRIX_SCFG6 : (MATRIX Offset: 0x58) Slave Configuration Register 6 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR6 ((unsigned int) 0x3 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_LCDC                 ((unsigned int) 0x3 << 18) /* (MATRIX) LCDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_2DGC                 ((unsigned int) 0x4 << 18) /* (MATRIX) 2DGC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_ISI                  ((unsigned int) 0x5 << 18) /* (MATRIX) ISI Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_EMAC                 ((unsigned int) 0x7 << 18) /* (MATRIX) EMAC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR6_USB                  ((unsigned int) 0x8 << 18) /* (MATRIX) USB Master is Default Master*/
/* -------- MATRIX_SCFG7 : (MATRIX Offset: 0x5c) Slave Configuration Register 7 -------- */
#define AT91C_MATRIX_FIXED_DEFMSTR7 ((unsigned int) 0x3 << 18) /* (MATRIX) Fixed Index of Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_ARM926I              ((unsigned int) 0x0 << 18) /* (MATRIX) ARM926EJ-S Instruction Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_ARM926D              ((unsigned int) 0x1 << 18) /* (MATRIX) ARM926EJ-S Data Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_PDC                  ((unsigned int) 0x2 << 18) /* (MATRIX) PDC Master is Default Master*/
#define 	AT91C_MATRIX_FIXED_DEFMSTR7_DMA                  ((unsigned int) 0x6 << 18) /* (MATRIX) DMA Controller Master is Default Master*/
/* -------- MATRIX_PRAS0 : (MATRIX Offset: 0x80) PRAS0 Register -------- */
#define AT91C_MATRIX_M0PR     ((unsigned int) 0x3 <<  0) /* (MATRIX) ARM926EJ-S Instruction priority*/
#define AT91C_MATRIX_M1PR     ((unsigned int) 0x3 <<  4) /* (MATRIX) ARM926EJ-S Data priority*/
#define AT91C_MATRIX_M2PR     ((unsigned int) 0x3 <<  8) /* (MATRIX) PDC priority*/
#define AT91C_MATRIX_M3PR     ((unsigned int) 0x3 << 12) /* (MATRIX) LCDC priority*/
#define AT91C_MATRIX_M4PR     ((unsigned int) 0x3 << 16) /* (MATRIX) 2DGC priority*/
#define AT91C_MATRIX_M5PR     ((unsigned int) 0x3 << 20) /* (MATRIX) ISI priority*/
#define AT91C_MATRIX_M6PR     ((unsigned int) 0x3 << 24) /* (MATRIX) DMA priority*/
#define AT91C_MATRIX_M7PR     ((unsigned int) 0x3 << 28) /* (MATRIX) EMAC priority*/
/* -------- MATRIX_PRBS0 : (MATRIX Offset: 0x84) PRBS0 Register -------- */
#define AT91C_MATRIX_M8PR     ((unsigned int) 0x3 <<  0) /* (MATRIX) USB priority*/
/* -------- MATRIX_PRAS1 : (MATRIX Offset: 0x88) PRAS1 Register -------- */
/* -------- MATRIX_PRBS1 : (MATRIX Offset: 0x8c) PRBS1 Register -------- */
/* -------- MATRIX_PRAS2 : (MATRIX Offset: 0x90) PRAS2 Register -------- */
/* -------- MATRIX_PRBS2 : (MATRIX Offset: 0x94) PRBS2 Register -------- */
/* -------- MATRIX_PRAS3 : (MATRIX Offset: 0x98) PRAS3 Register -------- */
/* -------- MATRIX_PRBS3 : (MATRIX Offset: 0x9c) PRBS3 Register -------- */
/* -------- MATRIX_PRAS4 : (MATRIX Offset: 0xa0) PRAS4 Register -------- */
/* -------- MATRIX_PRBS4 : (MATRIX Offset: 0xa4) PRBS4 Register -------- */
/* -------- MATRIX_PRAS5 : (MATRIX Offset: 0xa8) PRAS5 Register -------- */
/* -------- MATRIX_PRBS5 : (MATRIX Offset: 0xac) PRBS5 Register -------- */
/* -------- MATRIX_PRAS6 : (MATRIX Offset: 0xb0) PRAS6 Register -------- */
/* -------- MATRIX_PRBS6 : (MATRIX Offset: 0xb4) PRBS6 Register -------- */
/* -------- MATRIX_PRAS7 : (MATRIX Offset: 0xb8) PRAS7 Register -------- */
/* -------- MATRIX_PRBS7 : (MATRIX Offset: 0xbc) PRBS7 Register -------- */
/* -------- MATRIX_MRCR : (MATRIX Offset: 0x100) MRCR Register -------- */
#define AT91C_MATRIX_RCA926I  ((unsigned int) 0x1 <<  0) /* (MATRIX) Remap Command Bit for ARM926EJ-S Instruction*/
#define AT91C_MATRIX_RCA926D  ((unsigned int) 0x1 <<  1) /* (MATRIX) Remap Command Bit for ARM926EJ-S Data*/
#define AT91C_MATRIX_RCB2     ((unsigned int) 0x1 <<  2) /* (MATRIX) Remap Command Bit for PDC*/
#define AT91C_MATRIX_RCB3     ((unsigned int) 0x1 <<  3) /* (MATRIX) Remap Command Bit for LCD*/
#define AT91C_MATRIX_RCB4     ((unsigned int) 0x1 <<  4) /* (MATRIX) Remap Command Bit for 2DGC*/
#define AT91C_MATRIX_RCB5     ((unsigned int) 0x1 <<  5) /* (MATRIX) Remap Command Bit for ISI*/
#define AT91C_MATRIX_RCB6     ((unsigned int) 0x1 <<  6) /* (MATRIX) Remap Command Bit for DMA*/
#define AT91C_MATRIX_RCB7     ((unsigned int) 0x1 <<  7) /* (MATRIX) Remap Command Bit for EMAC*/
#define AT91C_MATRIX_RCB8     ((unsigned int) 0x1 <<  8) /* (MATRIX) Remap Command Bit for USB*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR AHB CCFG Interface*/
/* ******************************************************************************/
typedef struct _AT91S_CCFG {
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 CCFG_TCMR; 	/*  TCM configuration*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 CCFG_UDPHS; 	/*  USB Device HS configuration*/
	AT91_REG	 CCFG_EBICSA; 	/*  EBI Chip Select Assignement Register*/
	AT91_REG	 Reserved2[54]; 	/* */
	AT91_REG	 CCFG_MATRIXVERSION; 	/*  Version Register*/
} AT91S_CCFG, *AT91PS_CCFG;

/* -------- CCFG_TCMR : (CCFG Offset: 0x4) TCM Configuration -------- */
#define AT91C_CCFG_ITCM_SIZE  ((unsigned int) 0xF <<  0) /* (CCFG) Size of ITCM enabled memory block*/
#define 	AT91C_CCFG_ITCM_SIZE_0KB                  ((unsigned int) 0x0) /* (CCFG) 0 KB (No ITCM Memory)*/
#define 	AT91C_CCFG_ITCM_SIZE_16KB                 ((unsigned int) 0x5) /* (CCFG) 16 KB*/
#define 	AT91C_CCFG_ITCM_SIZE_32KB                 ((unsigned int) 0x6) /* (CCFG) 32 KB*/
#define AT91C_CCFG_DTCM_SIZE  ((unsigned int) 0xF <<  4) /* (CCFG) Size of DTCM enabled memory block*/
#define 	AT91C_CCFG_DTCM_SIZE_0KB                  ((unsigned int) 0x0 <<  4) /* (CCFG) 0 KB (No DTCM Memory)*/
#define 	AT91C_CCFG_DTCM_SIZE_16KB                 ((unsigned int) 0x5 <<  4) /* (CCFG) 16 KB*/
#define 	AT91C_CCFG_DTCM_SIZE_32KB                 ((unsigned int) 0x6 <<  4) /* (CCFG) 32 KB*/
#define AT91C_CCFG_RM         ((unsigned int) 0xF <<  8) /* (CCFG) Read Margin registers*/
/* -------- CCFG_UDPHS : (CCFG Offset: 0xc) USB Device HS configuration -------- */
#define AT91C_CCFG_DONT_USE_UTMI_LOCK ((unsigned int) 0x1 <<  0) /* (CCFG) */
#define 	AT91C_CCFG_DONT_USE_UTMI_LOCK_DONT_USE_LOCK        ((unsigned int) 0x0) /* (CCFG) */
/* -------- CCFG_EBICSA : (CCFG Offset: 0x10) EBI Chip Select Assignement Register -------- */
#define AT91C_EBI_CS1A        ((unsigned int) 0x1 <<  1) /* (CCFG) Chip Select 1 Assignment*/
#define 	AT91C_EBI_CS1A_SMC                  ((unsigned int) 0x0 <<  1) /* (CCFG) Chip Select 1 is assigned to the Static Memory Controller.*/
#define 	AT91C_EBI_CS1A_SDRAMC               ((unsigned int) 0x1 <<  1) /* (CCFG) Chip Select 1 is assigned to the SDRAM Controller.*/
#define AT91C_EBI_CS3A        ((unsigned int) 0x1 <<  3) /* (CCFG) Chip Select 3 Assignment*/
#define 	AT91C_EBI_CS3A_SMC                  ((unsigned int) 0x0 <<  3) /* (CCFG) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.*/
#define 	AT91C_EBI_CS3A_SM                   ((unsigned int) 0x1 <<  3) /* (CCFG) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.*/
#define AT91C_EBI_CS4A        ((unsigned int) 0x1 <<  4) /* (CCFG) Chip Select 4 Assignment*/
#define 	AT91C_EBI_CS4A_SMC                  ((unsigned int) 0x0 <<  4) /* (CCFG) Chip Select 4 is only assigned to the Static Memory Controller and NCS4 behaves as defined by the SMC.*/
#define 	AT91C_EBI_CS4A_CF                   ((unsigned int) 0x1 <<  4) /* (CCFG) Chip Select 4 is assigned to the Static Memory Controller and the CompactFlash Logic (first slot) is activated.*/
#define AT91C_EBI_CS5A        ((unsigned int) 0x1 <<  5) /* (CCFG) Chip Select 5 Assignment*/
#define 	AT91C_EBI_CS5A_SMC                  ((unsigned int) 0x0 <<  5) /* (CCFG) Chip Select 5 is only assigned to the Static Memory Controller and NCS5 behaves as defined by the SMC*/
#define 	AT91C_EBI_CS5A_CF                   ((unsigned int) 0x1 <<  5) /* (CCFG) Chip Select 5 is assigned to the Static Memory Controller and the CompactFlash Logic (second slot) is activated.*/
#define AT91C_EBI_DBPUC       ((unsigned int) 0x1 <<  8) /* (CCFG) Data Bus Pull-up Configuration*/
#define AT91C_EBI_SUPPLY      ((unsigned int) 0x1 << 16) /* (CCFG) EBI supply set to 1.8*/

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
	AT91_REG	 PMC_UCKR; 	/* UTMI Clock Configuration Register*/
	AT91_REG	 PMC_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 PMC_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 PMC_PLLAR; 	/* PLL A Register*/
	AT91_REG	 PMC_PLLBR; 	/* PLL B Register*/
	AT91_REG	 PMC_MCKR; 	/* Master Clock Register*/
	AT91_REG	 Reserved1[3]; 	/* */
	AT91_REG	 PMC_PCKR[8]; 	/* Programmable Clock Register*/
	AT91_REG	 PMC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 PMC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 PMC_SR; 	/* Status Register*/
	AT91_REG	 PMC_IMR; 	/* Interrupt Mask Register*/
} AT91S_PMC, *AT91PS_PMC;

/* -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
#define AT91C_PMC_PCK         ((unsigned int) 0x1 <<  0) /* (PMC) Processor Clock*/
#define AT91C_PMC_OTG         ((unsigned int) 0x1 <<  5) /* (PMC) USB OTG Clock*/
#define AT91C_PMC_UHP         ((unsigned int) 0x1 <<  6) /* (PMC) USB Host Port Clock*/
#define AT91C_PMC_UDP         ((unsigned int) 0x1 <<  7) /* (PMC) USB Device Port Clock*/
#define AT91C_PMC_PCK0        ((unsigned int) 0x1 <<  8) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK1        ((unsigned int) 0x1 <<  9) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK2        ((unsigned int) 0x1 << 10) /* (PMC) Programmable Clock Output*/
#define AT91C_PMC_PCK3        ((unsigned int) 0x1 << 11) /* (PMC) Programmable Clock Output*/
/* -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- */
#define AT91C_CKGR_UPLLEN     ((unsigned int) 0x1 << 16) /* (PMC) UTMI PLL Enable*/
#define 	AT91C_CKGR_UPLLEN_DISABLED             ((unsigned int) 0x0 << 16) /* (PMC) The UTMI PLL is disabled*/
#define 	AT91C_CKGR_UPLLEN_ENABLED              ((unsigned int) 0x1 << 16) /* (PMC) The UTMI PLL is enabled*/
#define AT91C_CKGR_PLLCOUNT   ((unsigned int) 0xF << 20) /* (PMC) UTMI Oscillator Start-up Time*/
#define AT91C_CKGR_BIASEN     ((unsigned int) 0x1 << 24) /* (PMC) UTMI BIAS Enable*/
#define 	AT91C_CKGR_BIASEN_DISABLED             ((unsigned int) 0x0 << 24) /* (PMC) The UTMI BIAS is disabled*/
#define 	AT91C_CKGR_BIASEN_ENABLED              ((unsigned int) 0x1 << 24) /* (PMC) The UTMI BIAS is enabled*/
#define AT91C_CKGR_BIASCOUNT  ((unsigned int) 0xF << 28) /* (PMC) UTMI BIAS Start-up Time*/
/* -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- */
#define AT91C_CKGR_MOSCEN     ((unsigned int) 0x1 <<  0) /* (PMC) Main Oscillator Enable*/
#define AT91C_CKGR_OSCBYPASS  ((unsigned int) 0x1 <<  1) /* (PMC) Main Oscillator Bypass*/
#define AT91C_CKGR_OSCOUNT    ((unsigned int) 0xFF <<  8) /* (PMC) Main Oscillator Start-up Time*/
/* -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- */
#define AT91C_CKGR_MAINF      ((unsigned int) 0xFFFF <<  0) /* (PMC) Main Clock Frequency*/
#define AT91C_CKGR_MAINRDY    ((unsigned int) 0x1 << 16) /* (PMC) Main Clock Ready*/
/* -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- */
#define AT91C_CKGR_DIVA       ((unsigned int) 0xFF <<  0) /* (PMC) Divider A Selected*/
#define 	AT91C_CKGR_DIVA_0                    ((unsigned int) 0x0) /* (PMC) Divider A output is 0*/
#define 	AT91C_CKGR_DIVA_BYPASS               ((unsigned int) 0x1) /* (PMC) Divider A is bypassed*/
#define AT91C_CKGR_PLLACOUNT  ((unsigned int) 0x3F <<  8) /* (PMC) PLL A Counter*/
#define AT91C_CKGR_OUTA       ((unsigned int) 0x3 << 14) /* (PMC) PLL A Output Frequency Range*/
#define 	AT91C_CKGR_OUTA_0                    ((unsigned int) 0x0 << 14) /* (PMC) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_1                    ((unsigned int) 0x1 << 14) /* (PMC) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_2                    ((unsigned int) 0x2 << 14) /* (PMC) Please refer to the PLLA datasheet*/
#define 	AT91C_CKGR_OUTA_3                    ((unsigned int) 0x3 << 14) /* (PMC) Please refer to the PLLA datasheet*/
#define AT91C_CKGR_MULA       ((unsigned int) 0x7FF << 16) /* (PMC) PLL A Multiplier*/
#define AT91C_CKGR_SRCA       ((unsigned int) 0x1 << 29) /* (PMC) */
/* -------- CKGR_PLLBR : (PMC Offset: 0x2c) PLL B Register -------- */
#define AT91C_CKGR_DIVB       ((unsigned int) 0xFF <<  0) /* (PMC) Divider B Selected*/
#define 	AT91C_CKGR_DIVB_0                    ((unsigned int) 0x0) /* (PMC) Divider B output is 0*/
#define 	AT91C_CKGR_DIVB_BYPASS               ((unsigned int) 0x1) /* (PMC) Divider B is bypassed*/
#define AT91C_CKGR_PLLBCOUNT  ((unsigned int) 0x3F <<  8) /* (PMC) PLL B Counter*/
#define AT91C_CKGR_OUTB       ((unsigned int) 0x3 << 14) /* (PMC) PLL B Output Frequency Range*/
#define 	AT91C_CKGR_OUTB_0                    ((unsigned int) 0x0 << 14) /* (PMC) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_1                    ((unsigned int) 0x1 << 14) /* (PMC) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_2                    ((unsigned int) 0x2 << 14) /* (PMC) Please refer to the PLLB datasheet*/
#define 	AT91C_CKGR_OUTB_3                    ((unsigned int) 0x3 << 14) /* (PMC) Please refer to the PLLB datasheet*/
#define AT91C_CKGR_MULB       ((unsigned int) 0x7FF << 16) /* (PMC) PLL B Multiplier*/
#define AT91C_CKGR_USBDIV     ((unsigned int) 0x3 << 28) /* (PMC) Divider for USB Clocks*/
#define 	AT91C_CKGR_USBDIV_0                    ((unsigned int) 0x0 << 28) /* (PMC) Divider output is PLL clock output*/
#define 	AT91C_CKGR_USBDIV_1                    ((unsigned int) 0x1 << 28) /* (PMC) Divider output is PLL clock output divided by 2*/
#define 	AT91C_CKGR_USBDIV_2                    ((unsigned int) 0x2 << 28) /* (PMC) Divider output is PLL clock output divided by 4*/
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
#define 	AT91C_PMC_MDIV_4                    ((unsigned int) 0x2 <<  8) /* (PMC) The processor clock is four times faster than the master clock*/
/* -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
#define AT91C_PMC_MOSCS       ((unsigned int) 0x1 <<  0) /* (PMC) MOSC Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCKA       ((unsigned int) 0x1 <<  1) /* (PMC) PLL A Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCKB       ((unsigned int) 0x1 <<  2) /* (PMC) PLL B Status/Enable/Disable/Mask*/
#define AT91C_PMC_MCKRDY      ((unsigned int) 0x1 <<  3) /* (PMC) Master Clock Status/Enable/Disable/Mask*/
#define AT91C_PMC_LOCKU       ((unsigned int) 0x1 <<  6) /* (PMC) PLL UTMI Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK0RDY     ((unsigned int) 0x1 <<  8) /* (PMC) PCK0_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK1RDY     ((unsigned int) 0x1 <<  9) /* (PMC) PCK1_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK2RDY     ((unsigned int) 0x1 << 10) /* (PMC) PCK2_RDY Status/Enable/Disable/Mask*/
#define AT91C_PMC_PCK3RDY     ((unsigned int) 0x1 << 11) /* (PMC) PCK3_RDY Status/Enable/Disable/Mask*/
/* -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* ******************************************************************************/
typedef struct _AT91S_CKGR {
	AT91_REG	 CKGR_UCKR; 	/* UTMI Clock Configuration Register*/
	AT91_REG	 CKGR_MOR; 	/* Main Oscillator Register*/
	AT91_REG	 CKGR_MCFR; 	/* Main Clock  Frequency Register*/
	AT91_REG	 CKGR_PLLAR; 	/* PLL A Register*/
	AT91_REG	 CKGR_PLLBR; 	/* PLL B Register*/
} AT91S_CKGR, *AT91PS_CKGR;

/* -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- */
/* -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- */
/* -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- */
/* -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- */
/* -------- CKGR_PLLBR : (CKGR Offset: 0x10) PLL B Register -------- */

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
#define AT91C_RSTC_ICERST     ((unsigned int) 0x1 <<  1) /* (RSTC) ICE Interface Reset*/
#define AT91C_RSTC_PERRST     ((unsigned int) 0x1 <<  2) /* (RSTC) Peripheral Reset*/
#define AT91C_RSTC_EXTRST     ((unsigned int) 0x1 <<  3) /* (RSTC) External Reset*/
#define AT91C_RSTC_KEY        ((unsigned int) 0xFF << 24) /* (RSTC) Password*/
/* -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- */
#define AT91C_RSTC_URSTS      ((unsigned int) 0x1 <<  0) /* (RSTC) User Reset Status*/
#define AT91C_RSTC_RSTTYP     ((unsigned int) 0x7 <<  8) /* (RSTC) Reset Type*/
#define 	AT91C_RSTC_RSTTYP_GENERAL              ((unsigned int) 0x0 <<  8) /* (RSTC) General reset. Both VDDCORE and VDDBU rising.*/
#define 	AT91C_RSTC_RSTTYP_WAKEUP               ((unsigned int) 0x1 <<  8) /* (RSTC) WakeUp Reset. VDDCORE rising.*/
#define 	AT91C_RSTC_RSTTYP_WATCHDOG             ((unsigned int) 0x2 <<  8) /* (RSTC) Watchdog Reset. Watchdog overflow occured.*/
#define 	AT91C_RSTC_RSTTYP_SOFTWARE             ((unsigned int) 0x3 <<  8) /* (RSTC) Software Reset. Processor reset required by the software.*/
#define 	AT91C_RSTC_RSTTYP_USER                 ((unsigned int) 0x4 <<  8) /* (RSTC) User Reset. NRST pin detected low.*/
#define AT91C_RSTC_NRSTL      ((unsigned int) 0x1 << 16) /* (RSTC) NRST pin level*/
#define AT91C_RSTC_SRCMP      ((unsigned int) 0x1 << 17) /* (RSTC) Software Reset Command in Progress.*/
/* -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- */
#define AT91C_RSTC_URSTEN     ((unsigned int) 0x1 <<  0) /* (RSTC) User Reset Enable*/
#define AT91C_RSTC_URSTIEN    ((unsigned int) 0x1 <<  4) /* (RSTC) User Reset Interrupt Enable*/
#define AT91C_RSTC_ERSTL      ((unsigned int) 0xF <<  8) /* (RSTC) User Reset Enable*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Shut Down Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_SHDWC {
	AT91_REG	 SHDWC_SHCR; 	/* Shut Down Control Register*/
	AT91_REG	 SHDWC_SHMR; 	/* Shut Down Mode Register*/
	AT91_REG	 SHDWC_SHSR; 	/* Shut Down Status Register*/
} AT91S_SHDWC, *AT91PS_SHDWC;

/* -------- SHDWC_SHCR : (SHDWC Offset: 0x0) Shut Down Control Register -------- */
#define AT91C_SHDWC_SHDW      ((unsigned int) 0x1 <<  0) /* (SHDWC) Processor Reset*/
#define AT91C_SHDWC_KEY       ((unsigned int) 0xFF << 24) /* (SHDWC) Shut down KEY Password*/
/* -------- SHDWC_SHMR : (SHDWC Offset: 0x4) Shut Down Mode Register -------- */
#define AT91C_SHDWC_WKMODE0   ((unsigned int) 0x3 <<  0) /* (SHDWC) Wake Up 0 Mode Selection*/
#define 	AT91C_SHDWC_WKMODE0_NONE                 ((unsigned int) 0x0) /* (SHDWC) None. No detection is performed on the wake up input.*/
#define 	AT91C_SHDWC_WKMODE0_HIGH                 ((unsigned int) 0x1) /* (SHDWC) High Level.*/
#define 	AT91C_SHDWC_WKMODE0_LOW                  ((unsigned int) 0x2) /* (SHDWC) Low Level.*/
#define 	AT91C_SHDWC_WKMODE0_ANYLEVEL             ((unsigned int) 0x3) /* (SHDWC) Any level change.*/
#define AT91C_SHDWC_CPTWK0    ((unsigned int) 0xF <<  4) /* (SHDWC) Counter On Wake Up 0*/
#define AT91C_SHDWC_WKMODE1   ((unsigned int) 0x3 <<  8) /* (SHDWC) Wake Up 1 Mode Selection*/
#define 	AT91C_SHDWC_WKMODE1_NONE                 ((unsigned int) 0x0 <<  8) /* (SHDWC) None. No detection is performed on the wake up input.*/
#define 	AT91C_SHDWC_WKMODE1_HIGH                 ((unsigned int) 0x1 <<  8) /* (SHDWC) High Level.*/
#define 	AT91C_SHDWC_WKMODE1_LOW                  ((unsigned int) 0x2 <<  8) /* (SHDWC) Low Level.*/
#define 	AT91C_SHDWC_WKMODE1_ANYLEVEL             ((unsigned int) 0x3 <<  8) /* (SHDWC) Any level change.*/
#define AT91C_SHDWC_CPTWK1    ((unsigned int) 0xF << 12) /* (SHDWC) Counter On Wake Up 1*/
#define AT91C_SHDWC_RTTWKEN   ((unsigned int) 0x1 << 16) /* (SHDWC) Real Time Timer Wake Up Enable*/
#define AT91C_SHDWC_RTCWKEN   ((unsigned int) 0x1 << 17) /* (SHDWC) Real Time Clock Wake Up Enable*/
/* -------- SHDWC_SHSR : (SHDWC Offset: 0x8) Shut Down Status Register -------- */
#define AT91C_SHDWC_WAKEUP0   ((unsigned int) 0x1 <<  0) /* (SHDWC) Wake Up 0 Status*/
#define AT91C_SHDWC_WAKEUP1   ((unsigned int) 0x1 <<  1) /* (SHDWC) Wake Up 1 Status*/
#define AT91C_SHDWC_FWKUP     ((unsigned int) 0x1 <<  2) /* (SHDWC) Force Wake Up Status*/
#define AT91C_SHDWC_RTTWK     ((unsigned int) 0x1 << 16) /* (SHDWC) Real Time Timer wake Up*/
#define AT91C_SHDWC_RTCWK     ((unsigned int) 0x1 << 17) /* (SHDWC) Real Time Clock wake Up*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_RTTC {
	AT91_REG	 RTTC_RTMR; 	/* Real-time Mode Register*/
	AT91_REG	 RTTC_RTAR; 	/* Real-time Alarm Register*/
	AT91_REG	 RTTC_RTVR; 	/* Real-time Value Register*/
	AT91_REG	 RTTC_RTSR; 	/* Real-time Status Register*/
} AT91S_RTTC, *AT91PS_RTTC;

/* -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- */
#define AT91C_RTTC_RTPRES     ((unsigned int) 0xFFFF <<  0) /* (RTTC) Real-time Timer Prescaler Value*/
#define AT91C_RTTC_ALMIEN     ((unsigned int) 0x1 << 16) /* (RTTC) Alarm Interrupt Enable*/
#define AT91C_RTTC_RTTINCIEN  ((unsigned int) 0x1 << 17) /* (RTTC) Real Time Timer Increment Interrupt Enable*/
#define AT91C_RTTC_RTTRST     ((unsigned int) 0x1 << 18) /* (RTTC) Real Time Timer Restart*/
/* -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- */
#define AT91C_RTTC_ALMV       ((unsigned int) 0x0 <<  0) /* (RTTC) Alarm Value*/
/* -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- */
#define AT91C_RTTC_CRTV       ((unsigned int) 0x0 <<  0) /* (RTTC) Current Real-time Value*/
/* -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- */
#define AT91C_RTTC_ALMS       ((unsigned int) 0x1 <<  0) /* (RTTC) Real-time Alarm Status*/
#define AT91C_RTTC_RTTINC     ((unsigned int) 0x1 <<  1) /* (RTTC) Real-time Timer Increment*/

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
/*              SOFTWARE API DEFINITION  FOR Multimedia Card Interface*/
/* ******************************************************************************/
typedef struct _AT91S_MCI {
	AT91_REG	 MCI_CR; 	/* MCI Control Register*/
	AT91_REG	 MCI_MR; 	/* MCI Mode Register*/
	AT91_REG	 MCI_DTOR; 	/* MCI Data Timeout Register*/
	AT91_REG	 MCI_SDCR; 	/* MCI SD Card Register*/
	AT91_REG	 MCI_ARGR; 	/* MCI Argument Register*/
	AT91_REG	 MCI_CMDR; 	/* MCI Command Register*/
	AT91_REG	 MCI_BLKR; 	/* MCI Block Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 MCI_RSPR[4]; 	/* MCI Response Register*/
	AT91_REG	 MCI_RDR; 	/* MCI Receive Data Register*/
	AT91_REG	 MCI_TDR; 	/* MCI Transmit Data Register*/
	AT91_REG	 Reserved1[2]; 	/* */
	AT91_REG	 MCI_SR; 	/* MCI Status Register*/
	AT91_REG	 MCI_IER; 	/* MCI Interrupt Enable Register*/
	AT91_REG	 MCI_IDR; 	/* MCI Interrupt Disable Register*/
	AT91_REG	 MCI_IMR; 	/* MCI Interrupt Mask Register*/
	AT91_REG	 Reserved2[43]; 	/* */
	AT91_REG	 MCI_VR; 	/* MCI Version Register*/
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
#define AT91C_MCI_RDPROOF     ((unsigned int) 0x1 << 11) /* (MCI) Read Proof Enable*/
#define AT91C_MCI_WRPROOF     ((unsigned int) 0x1 << 12) /* (MCI) Write Proof Enable*/
#define AT91C_MCI_PDCFBYTE    ((unsigned int) 0x1 << 13) /* (MCI) PDC Force Byte Transfer*/
#define AT91C_MCI_PDCPADV     ((unsigned int) 0x1 << 14) /* (MCI) PDC Padding Value*/
#define AT91C_MCI_PDCMODE     ((unsigned int) 0x1 << 15) /* (MCI) PDC Oriented Mode*/
#define AT91C_MCI_BLKLEN      ((unsigned int) 0xFFFF << 16) /* (MCI) Data Block Length*/
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
#define AT91C_MCI_SCDSEL      ((unsigned int) 0x3 <<  0) /* (MCI) SD Card Selector*/
#define AT91C_MCI_SCDBUS      ((unsigned int) 0x1 <<  7) /* (MCI) SDCard/SDIO Bus Width*/
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
#define AT91C_MCI_TRTYP       ((unsigned int) 0x7 << 19) /* (MCI) Transfer Type*/
#define 	AT91C_MCI_TRTYP_BLOCK                ((unsigned int) 0x0 << 19) /* (MCI) MMC/SDCard Single Block Transfer type*/
#define 	AT91C_MCI_TRTYP_MULTIPLE             ((unsigned int) 0x1 << 19) /* (MCI) MMC/SDCard Multiple Block transfer type*/
#define 	AT91C_MCI_TRTYP_STREAM               ((unsigned int) 0x2 << 19) /* (MCI) MMC Stream transfer type*/
#define 	AT91C_MCI_TRTYP_SDIO_BYTE            ((unsigned int) 0x4 << 19) /* (MCI) SDIO Byte transfer type*/
#define 	AT91C_MCI_TRTYP_SDIO_BLOCK           ((unsigned int) 0x5 << 19) /* (MCI) SDIO Block transfer type*/
#define AT91C_MCI_IOSPCMD     ((unsigned int) 0x3 << 24) /* (MCI) SDIO Special Command*/
#define 	AT91C_MCI_IOSPCMD_NONE                 ((unsigned int) 0x0 << 24) /* (MCI) NOT a special command*/
#define 	AT91C_MCI_IOSPCMD_SUSPEND              ((unsigned int) 0x1 << 24) /* (MCI) SDIO Suspend Command*/
#define 	AT91C_MCI_IOSPCMD_RESUME               ((unsigned int) 0x2 << 24) /* (MCI) SDIO Resume Command*/
/* -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- */
#define AT91C_MCI_BCNT        ((unsigned int) 0xFFFF <<  0) /* (MCI) MMC/SDIO Block Count / SDIO Byte Count*/
/* -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- */
#define AT91C_MCI_CMDRDY      ((unsigned int) 0x1 <<  0) /* (MCI) Command Ready flag*/
#define AT91C_MCI_RXRDY       ((unsigned int) 0x1 <<  1) /* (MCI) RX Ready flag*/
#define AT91C_MCI_TXRDY       ((unsigned int) 0x1 <<  2) /* (MCI) TX Ready flag*/
#define AT91C_MCI_BLKE        ((unsigned int) 0x1 <<  3) /* (MCI) Data Block Transfer Ended flag*/
#define AT91C_MCI_DTIP        ((unsigned int) 0x1 <<  4) /* (MCI) Data Transfer in Progress flag*/
#define AT91C_MCI_NOTBUSY     ((unsigned int) 0x1 <<  5) /* (MCI) Data Line Not Busy flag*/
#define AT91C_MCI_ENDRX       ((unsigned int) 0x1 <<  6) /* (MCI) End of RX Buffer flag*/
#define AT91C_MCI_ENDTX       ((unsigned int) 0x1 <<  7) /* (MCI) End of TX Buffer flag*/
#define AT91C_MCI_SDIOIRQA    ((unsigned int) 0x1 <<  8) /* (MCI) SDIO Interrupt for Slot A*/
#define AT91C_MCI_SDIOIRQB    ((unsigned int) 0x1 <<  9) /* (MCI) SDIO Interrupt for Slot B*/
#define AT91C_MCI_SDIOIRQC    ((unsigned int) 0x1 << 10) /* (MCI) SDIO Interrupt for Slot C*/
#define AT91C_MCI_SDIOIRQD    ((unsigned int) 0x1 << 11) /* (MCI) SDIO Interrupt for Slot D*/
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
	AT91_REG	 US_MAN; 	/* Manchester Encoder Decoder Register*/
	AT91_REG	 Reserved2[43]; 	/* */
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
#define AT91C_US_VAR_SYNC     ((unsigned int) 0x1 << 22) /* (USART) Variable synchronization of command/data sync Start Frame Delimiter*/
#define AT91C_US_MAX_ITER     ((unsigned int) 0x1 << 24) /* (USART) Number of Repetitions*/
#define AT91C_US_FILTER       ((unsigned int) 0x1 << 28) /* (USART) Receive Line Filter*/
#define AT91C_US_MANMODE      ((unsigned int) 0x1 << 29) /* (USART) Manchester Encoder/Decoder Enable*/
#define AT91C_US_MODSYNC      ((unsigned int) 0x1 << 30) /* (USART) Manchester Synchronization mode*/
#define AT91C_US_ONEBIT       ((unsigned int) 0x1 << 31) /* (USART) Start Frame Delimiter selector*/
/* -------- US_IER : (USART Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
#define AT91C_US_RXBRK        ((unsigned int) 0x1 <<  2) /* (USART) Break Received/End of Break*/
#define AT91C_US_TIMEOUT      ((unsigned int) 0x1 <<  8) /* (USART) Receiver Time-out*/
#define AT91C_US_ITERATION    ((unsigned int) 0x1 << 10) /* (USART) Max number of Repetitions Reached*/
#define AT91C_US_NACK         ((unsigned int) 0x1 << 13) /* (USART) Non Acknowledge*/
#define AT91C_US_RIIC         ((unsigned int) 0x1 << 16) /* (USART) Ring INdicator Input Change Flag*/
#define AT91C_US_DSRIC        ((unsigned int) 0x1 << 17) /* (USART) Data Set Ready Input Change Flag*/
#define AT91C_US_DCDIC        ((unsigned int) 0x1 << 18) /* (USART) Data Carrier Flag*/
#define AT91C_US_CTSIC        ((unsigned int) 0x1 << 19) /* (USART) Clear To Send Input Change Flag*/
#define AT91C_US_MANE         ((unsigned int) 0x1 << 20) /* (USART) Manchester Error Interrupt*/
/* -------- US_IDR : (USART Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* -------- US_IMR : (USART Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* -------- US_CSR : (USART Offset: 0x14) Debug Unit Channel Status Register -------- */
#define AT91C_US_RI           ((unsigned int) 0x1 << 20) /* (USART) Image of RI Input*/
#define AT91C_US_DSR          ((unsigned int) 0x1 << 21) /* (USART) Image of DSR Input*/
#define AT91C_US_DCD          ((unsigned int) 0x1 << 22) /* (USART) Image of DCD Input*/
#define AT91C_US_CTS          ((unsigned int) 0x1 << 23) /* (USART) Image of CTS Input*/
#define AT91C_US_MANERR       ((unsigned int) 0x1 << 24) /* (USART) Manchester Error*/
/* -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- */
#define AT91C_US_TX_PL        ((unsigned int) 0xF <<  0) /* (USART) Transmitter Preamble Length*/
#define AT91C_US_TX_PP        ((unsigned int) 0x3 <<  8) /* (USART) Transmitter Preamble Pattern*/
#define 	AT91C_US_TX_PP_ALL_ONE              ((unsigned int) 0x0 <<  8) /* (USART) ALL_ONE*/
#define 	AT91C_US_TX_PP_ALL_ZERO             ((unsigned int) 0x1 <<  8) /* (USART) ALL_ZERO*/
#define 	AT91C_US_TX_PP_ZERO_ONE             ((unsigned int) 0x2 <<  8) /* (USART) ZERO_ONE*/
#define 	AT91C_US_TX_PP_ONE_ZERO             ((unsigned int) 0x3 <<  8) /* (USART) ONE_ZERO*/
#define AT91C_US_TX_MPOL      ((unsigned int) 0x1 << 12) /* (USART) Transmitter Manchester Polarity*/
#define AT91C_US_RX_PL        ((unsigned int) 0xF << 16) /* (USART) Receiver Preamble Length*/
#define AT91C_US_RX_PP        ((unsigned int) 0x3 << 24) /* (USART) Receiver Preamble Pattern detected*/
#define 	AT91C_US_RX_PP_ALL_ONE              ((unsigned int) 0x0 << 24) /* (USART) ALL_ONE*/
#define 	AT91C_US_RX_PP_ALL_ZERO             ((unsigned int) 0x1 << 24) /* (USART) ALL_ZERO*/
#define 	AT91C_US_RX_PP_ZERO_ONE             ((unsigned int) 0x2 << 24) /* (USART) ZERO_ONE*/
#define 	AT91C_US_RX_PP_ONE_ZERO             ((unsigned int) 0x3 << 24) /* (USART) ONE_ZERO*/
#define AT91C_US_RX_MPOL      ((unsigned int) 0x1 << 28) /* (USART) Receiver Manchester Polarity*/
#define AT91C_US_DRIFT        ((unsigned int) 0x1 << 30) /* (USART) Drift compensation*/

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
	AT91_REG	 Reserved2[2]; 	/* */
	AT91_REG	 SSC_SR; 	/* Status Register*/
	AT91_REG	 SSC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 SSC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 SSC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 Reserved3[44]; 	/* */
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
	AT91S_PWMC_CH	 PWMC_CH[32]; 	/* PWMC Channel*/
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
#define AT91C_PWMC_CHID4      ((unsigned int) 0x1 <<  4) /* (PWMC) Channel ID 4*/
#define AT91C_PWMC_CHID5      ((unsigned int) 0x1 <<  5) /* (PWMC) Channel ID 5*/
#define AT91C_PWMC_CHID6      ((unsigned int) 0x1 <<  6) /* (PWMC) Channel ID 6*/
#define AT91C_PWMC_CHID7      ((unsigned int) 0x1 <<  7) /* (PWMC) Channel ID 7*/
/* -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- */
/* -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- */
/* -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- */
/* -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- */
/* -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- */
/* -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- */

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
/*              SOFTWARE API DEFINITION  FOR TSADC*/
/* ******************************************************************************/
typedef struct _AT91S_TSADC {
	AT91_REG	 TSADC_CR; 	/* Control Register*/
	AT91_REG	 TSADC_MR; 	/* Mode Register*/
	AT91_REG	 TSADC_TRGR; 	/* Trigger Register*/
	AT91_REG	 TSADC_TSR; 	/* Touch Screen Register*/
	AT91_REG	 TSADC_CHER; 	/* Channel Enable Register*/
	AT91_REG	 TSADC_CHDR; 	/* Channel Disable Register*/
	AT91_REG	 TSADC_CHSR; 	/* Channel Status Register*/
	AT91_REG	 TSADC_SR; 	/* Status Register*/
	AT91_REG	 TSADC_LCDR; 	/* Last Converted Register*/
	AT91_REG	 TSADC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 TSADC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 TSADC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 TSADC_CDR0; 	/* Channel Data Register 0*/
	AT91_REG	 TSADC_CDR1; 	/* Channel Data Register 1*/
	AT91_REG	 TSADC_CDR2; 	/* Channel Data Register 2*/
	AT91_REG	 TSADC_CDR3; 	/* Channel Data Register 3*/
	AT91_REG	 TSADC_CDR4; 	/* Channel Data Register 4*/
	AT91_REG	 TSADC_CDR5; 	/* Channel Data Register 5*/
	AT91_REG	 Reserved0[46]; 	/* */
	AT91_REG	 TSADC_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 TSADC_RCR; 	/* Receive Counter Register*/
	AT91_REG	 TSADC_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 TSADC_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 TSADC_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 TSADC_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 TSADC_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 TSADC_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 TSADC_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 TSADC_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_TSADC, *AT91PS_TSADC;

/* -------- TSADC_CR : (TSADC Offset: 0x0) Control Register -------- */
#define AT91C_TSADC_SWRST     ((unsigned int) 0x1 <<  0) /* (TSADC) Software Reset*/
#define AT91C_TSADC_START     ((unsigned int) 0x1 <<  1) /* (TSADC) Start Conversion*/
/* -------- TSADC_MR : (TSADC Offset: 0x4) Mode Register -------- */
#define AT91C_TSADC_TSAMOD    ((unsigned int) 0x3 <<  0) /* (TSADC) Touch Screen ADC Mode*/
#define 	AT91C_TSADC_TSAMOD_ADC_ONLY_MODE        ((unsigned int) 0x0) /* (TSADC) ADC Mode*/
#define 	AT91C_TSADC_TSAMOD_TS_ONLY_MODE         ((unsigned int) 0x1) /* (TSADC) Touch Screen Only Mode*/
#define AT91C_TSADC_LOWRES    ((unsigned int) 0x1 <<  4) /* (TSADC) ADC Resolution*/
#define AT91C_TSADC_SLEEP     ((unsigned int) 0x1 <<  5) /* (TSADC) Sleep Mode*/
#define AT91C_TSADC_PENDET    ((unsigned int) 0x1 <<  6) /* (TSADC) Pen Detect Selection*/
#define AT91C_TSADC_PRESCAL   ((unsigned int) 0x3F <<  8) /* (TSADC) Prescaler Rate Selection*/
#define AT91C_TSADC_STARTUP   ((unsigned int) 0x7F << 16) /* (TSADC) Startup Time*/
#define AT91C_TSADC_SHTIM     ((unsigned int) 0xF << 24) /* (TSADC) Sample and Hold Time for ADC Channels*/
#define AT91C_TSADC_PENDBC    ((unsigned int) 0xF << 28) /* (TSADC) Pen Detect Debouncing Period*/
/* -------- TSADC_TRGR : (TSADC Offset: 0x8) Trigger Register -------- */
#define AT91C_TSADC_TRGMOD    ((unsigned int) 0x7 <<  0) /* (TSADC) Trigger Mode*/
#define 	AT91C_TSADC_TRGMOD_NO_TRIGGER           ((unsigned int) 0x0) /* (TSADC) No Trigger*/
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_RE  ((unsigned int) 0x1) /* (TSADC) External Trigger Rising Edge*/
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_FE  ((unsigned int) 0x2) /* (TSADC) External Trigger Falling Edge*/
#define 	AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_AE  ((unsigned int) 0x3) /* (TSADC) External Trigger Any Edge*/
#define 	AT91C_TSADC_TRGMOD_PENDET_TRIGGER       ((unsigned int) 0x4) /* (TSADC) Pen Detect Trigger (only if PENDET is set and in Touch Screen mode only)*/
#define 	AT91C_TSADC_TRGMOD_PERIODIC_TRIGGER     ((unsigned int) 0x5) /* (TSADC) Periodic Trigger (wrt TRGPER)*/
#define 	AT91C_TSADC_TRGMOD_CONT_TRIGGER         ((unsigned int) 0x6) /* (TSADC) Continuous Trigger*/
#define AT91C_TSADC_TRGPER    ((unsigned int) 0xFFFF << 16) /* (TSADC) Trigger Period*/
/* -------- TSADC_TSR : (TSADC Offset: 0xc) Touch Screen Register -------- */
#define AT91C_TSADC_TSSHTIM   ((unsigned int) 0xF << 24) /* (TSADC) Sample and Hold Time for Touch Screen Channels*/
/* -------- TSADC_CHER : (TSADC Offset: 0x10) Channel Enable Register -------- */
#define AT91C_TSADC_CHENA0    ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 Enable*/
#define AT91C_TSADC_CHENA1    ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 Enable*/
#define AT91C_TSADC_CHENA2    ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 Enable*/
#define AT91C_TSADC_CHENA3    ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 Enable*/
#define AT91C_TSADC_CHENA4    ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 Enable*/
#define AT91C_TSADC_CHENA5    ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 Enable*/
/* -------- TSADC_CHDR : (TSADC Offset: 0x14) Channel Disable Register -------- */
#define AT91C_TSADC_CHDIS0    ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 Disable*/
#define AT91C_TSADC_CHDIS1    ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 Disable*/
#define AT91C_TSADC_CHDIS2    ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 Disable*/
#define AT91C_TSADC_CHDIS3    ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 Disable*/
#define AT91C_TSADC_CHDIS4    ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 Disable*/
#define AT91C_TSADC_CHDIS5    ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 Disable*/
/* -------- TSADC_CHSR : (TSADC Offset: 0x18) Channel Status Register -------- */
#define AT91C_TSADC_CHS0      ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 Status*/
#define AT91C_TSADC_CHS1      ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 Status*/
#define AT91C_TSADC_CHS2      ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 Status*/
#define AT91C_TSADC_CHS3      ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 Status*/
#define AT91C_TSADC_CHS4      ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 Status*/
#define AT91C_TSADC_CHS5      ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 Status*/
/* -------- TSADC_SR : (TSADC Offset: 0x1c) Status Register -------- */
#define AT91C_TSADC_EOC0      ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 End Of Conversion*/
#define AT91C_TSADC_EOC1      ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 End Of Conversion*/
#define AT91C_TSADC_EOC2      ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 End Of Conversion*/
#define AT91C_TSADC_EOC3      ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 End Of Conversion*/
#define AT91C_TSADC_EOC4      ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 End Of Conversion*/
#define AT91C_TSADC_EOC5      ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 End Of Conversion*/
#define AT91C_TSADC_OVRE0     ((unsigned int) 0x1 <<  8) /* (TSADC) Channel 0 Overrun Error*/
#define AT91C_TSADC_OVRE1     ((unsigned int) 0x1 <<  9) /* (TSADC) Channel 1 Overrun Error*/
#define AT91C_TSADC_OVRE2     ((unsigned int) 0x1 << 10) /* (TSADC) Channel 2 Overrun Error*/
#define AT91C_TSADC_OVRE3     ((unsigned int) 0x1 << 11) /* (TSADC) Channel 3 Overrun Error*/
#define AT91C_TSADC_OVRE4     ((unsigned int) 0x1 << 12) /* (TSADC) Channel 4 Overrun Error*/
#define AT91C_TSADC_OVRE5     ((unsigned int) 0x1 << 13) /* (TSADC) Channel 5 Overrun Error*/
#define AT91C_TSADC_DRDY      ((unsigned int) 0x1 << 16) /* (TSADC) Data Ready*/
#define AT91C_TSADC_GOVRE     ((unsigned int) 0x1 << 17) /* (TSADC) General Overrun Error*/
#define AT91C_TSADC_ENDRX     ((unsigned int) 0x1 << 18) /* (TSADC) End of RX Buffer*/
#define AT91C_TSADC_RXBUFF    ((unsigned int) 0x1 << 19) /* (TSADC) RX Buffer Full*/
#define AT91C_TSADC_PENCNT    ((unsigned int) 0x1 << 20) /* (TSADC) Pen Contact*/
#define AT91C_TSADC_NOCNT     ((unsigned int) 0x1 << 21) /* (TSADC) No Contact*/
/* -------- TSADC_LCDR : (TSADC Offset: 0x20) Last Converted Data Register -------- */
#define AT91C_TSADC_LDATA     ((unsigned int) 0x3FF <<  0) /* (TSADC) Last Converted Data*/
/* -------- TSADC_IER : (TSADC Offset: 0x24) Interrupt Enable Register -------- */
#define AT91C_TSADC_IENAEOC0  ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAEOC1  ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAEOC2  ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAEOC3  ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAEOC4  ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAEOC5  ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 End Of Conversion Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE0 ((unsigned int) 0x1 <<  8) /* (TSADC) Channel 0 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE1 ((unsigned int) 0x1 <<  9) /* (TSADC) Channel 1 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE2 ((unsigned int) 0x1 << 10) /* (TSADC) Channel 2 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE3 ((unsigned int) 0x1 << 11) /* (TSADC) Channel 3 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE4 ((unsigned int) 0x1 << 12) /* (TSADC) Channel 4 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAOVRE5 ((unsigned int) 0x1 << 13) /* (TSADC) Channel 5 Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENADRDY  ((unsigned int) 0x1 << 16) /* (TSADC) Data Ready Interrupt Enable*/
#define AT91C_TSADC_IENAGOVRE ((unsigned int) 0x1 << 17) /* (TSADC) General Overrun Error Interrupt Enable*/
#define AT91C_TSADC_IENAENDRX ((unsigned int) 0x1 << 18) /* (TSADC) End of RX Buffer Interrupt Enable*/
#define AT91C_TSADC_IENARXBUFF ((unsigned int) 0x1 << 19) /* (TSADC) RX Buffer Full Interrupt Enable*/
#define AT91C_TSADC_IENAPENCNT ((unsigned int) 0x1 << 20) /* (TSADC) Pen Contact Interrupt Enable*/
#define AT91C_TSADC_IENANOCNT ((unsigned int) 0x1 << 21) /* (TSADC) No Contact Interrupt Enable*/
/* -------- TSADC_IDR : (TSADC Offset: 0x28) Interrupt Disable Register -------- */
#define AT91C_TSADC_IDISEOC0  ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISEOC1  ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISEOC2  ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISEOC3  ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISEOC4  ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISEOC5  ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 End Of Conversion Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE0 ((unsigned int) 0x1 <<  8) /* (TSADC) Channel 0 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE1 ((unsigned int) 0x1 <<  9) /* (TSADC) Channel 1 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE2 ((unsigned int) 0x1 << 10) /* (TSADC) Channel 2 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE3 ((unsigned int) 0x1 << 11) /* (TSADC) Channel 3 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE4 ((unsigned int) 0x1 << 12) /* (TSADC) Channel 4 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISOVRE5 ((unsigned int) 0x1 << 13) /* (TSADC) Channel 5 Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISDRDY  ((unsigned int) 0x1 << 16) /* (TSADC) Data Ready Interrupt Disable*/
#define AT91C_TSADC_IDISGOVRE ((unsigned int) 0x1 << 17) /* (TSADC) General Overrun Error Interrupt Disable*/
#define AT91C_TSADC_IDISENDRX ((unsigned int) 0x1 << 18) /* (TSADC) End of RX Buffer Interrupt Disable*/
#define AT91C_TSADC_IDISRXBUFF ((unsigned int) 0x1 << 19) /* (TSADC) RX Buffer Full Interrupt Disable*/
#define AT91C_TSADC_IDISPENCNT ((unsigned int) 0x1 << 20) /* (TSADC) Pen Contact Interrupt Disable*/
#define AT91C_TSADC_IDISNOCNT ((unsigned int) 0x1 << 21) /* (TSADC) No Contact Interrupt Disable*/
/* -------- TSADC_IMR : (TSADC Offset: 0x2c) Interrupt Mask Register -------- */
#define AT91C_TSADC_IMSKEOC0  ((unsigned int) 0x1 <<  0) /* (TSADC) Channel 0 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKEOC1  ((unsigned int) 0x1 <<  1) /* (TSADC) Channel 1 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKEOC2  ((unsigned int) 0x1 <<  2) /* (TSADC) Channel 2 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKEOC3  ((unsigned int) 0x1 <<  3) /* (TSADC) Channel 3 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKEOC4  ((unsigned int) 0x1 <<  4) /* (TSADC) Channel 4 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKEOC5  ((unsigned int) 0x1 <<  5) /* (TSADC) Channel 5 End Of Conversion Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE0 ((unsigned int) 0x1 <<  8) /* (TSADC) Channel 0 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE1 ((unsigned int) 0x1 <<  9) /* (TSADC) Channel 1 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE2 ((unsigned int) 0x1 << 10) /* (TSADC) Channel 2 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE3 ((unsigned int) 0x1 << 11) /* (TSADC) Channel 3 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE4 ((unsigned int) 0x1 << 12) /* (TSADC) Channel 4 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKOVRE5 ((unsigned int) 0x1 << 13) /* (TSADC) Channel 5 Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKDRDY  ((unsigned int) 0x1 << 16) /* (TSADC) Data Ready Interrupt Mask*/
#define AT91C_TSADC_IMSKGOVRE ((unsigned int) 0x1 << 17) /* (TSADC) General Overrun Error Interrupt Mask*/
#define AT91C_TSADC_IMSKENDRX ((unsigned int) 0x1 << 18) /* (TSADC) End of RX Buffer Interrupt Mask*/
#define AT91C_TSADC_IMSKRXBUFF ((unsigned int) 0x1 << 19) /* (TSADC) RX Buffer Full Interrupt Mask*/
#define AT91C_TSADC_IMSKPENCNT ((unsigned int) 0x1 << 20) /* (TSADC) Pen Contact Interrupt Mask*/
#define AT91C_TSADC_IMSKNOCNT ((unsigned int) 0x1 << 21) /* (TSADC) No Contact Interrupt Mask*/
/* -------- TSADC_CDR0 : (TSADC Offset: 0x30) Channel 0 Data Register -------- */
#define AT91C_TSADC_DATA0     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 0 Data*/
/* -------- TSADC_CDR1 : (TSADC Offset: 0x34) Channel 1 Data Register -------- */
#define AT91C_TSADC_DATA1     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 1 Data*/
/* -------- TSADC_CDR2 : (TSADC Offset: 0x38) Channel 2 Data Register -------- */
#define AT91C_TSADC_DATA2     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 2 Data*/
/* -------- TSADC_CDR3 : (TSADC Offset: 0x3c) Channel 3 Data Register -------- */
#define AT91C_TSADC_DATA3     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 3 Data*/
/* -------- TSADC_CDR4 : (TSADC Offset: 0x40) Channel 4 Data Register -------- */
#define AT91C_TSADC_DATA4     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 4 Data*/
/* -------- TSADC_CDR5 : (TSADC Offset: 0x44) Channel 5 Data Register -------- */
#define AT91C_TSADC_DATA5     ((unsigned int) 0x3FF <<  0) /* (TSADC) Channel 5 Data*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register*/
/* ******************************************************************************/
typedef struct _AT91S_UDPHS_EPTFIFO {
	AT91_REG	 UDPHS_READEPT0[16384]; 	/* FIFO Endpoint Data Register 0*/
	AT91_REG	 UDPHS_READEPT1[16384]; 	/* FIFO Endpoint Data Register 1*/
	AT91_REG	 UDPHS_READEPT2[16384]; 	/* FIFO Endpoint Data Register 2*/
	AT91_REG	 UDPHS_READEPT3[16384]; 	/* FIFO Endpoint Data Register 3*/
	AT91_REG	 UDPHS_READEPT4[16384]; 	/* FIFO Endpoint Data Register 4*/
	AT91_REG	 UDPHS_READEPT5[16384]; 	/* FIFO Endpoint Data Register 5*/
	AT91_REG	 UDPHS_READEPT6[16384]; 	/* FIFO Endpoint Data Register 6*/
	AT91_REG	 UDPHS_READEPT7[16384]; 	/* FIFO Endpoint Data Register 7*/
	AT91_REG	 UDPHS_READEPT8[16384]; 	/* FIFO Endpoint Data Register 8*/
	AT91_REG	 UDPHS_READEPT9[16384]; 	/* FIFO Endpoint Data Register 9*/
	AT91_REG	 UDPHS_READEPTA[16384]; 	/* FIFO Endpoint Data Register 10*/
	AT91_REG	 UDPHS_READEPTB[16384]; 	/* FIFO Endpoint Data Register 11*/
	AT91_REG	 UDPHS_READEPTC[16384]; 	/* FIFO Endpoint Data Register 12*/
	AT91_REG	 UDPHS_READEPTD[16384]; 	/* FIFO Endpoint Data Register 13*/
	AT91_REG	 UDPHS_READEPTE[16384]; 	/* FIFO Endpoint Data Register 14*/
	AT91_REG	 UDPHS_READEPTF[16384]; 	/* FIFO Endpoint Data Register 15*/
} AT91S_UDPHS_EPTFIFO, *AT91PS_UDPHS_EPTFIFO;


/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct*/
/* ******************************************************************************/
typedef struct _AT91S_UDPHS_EPT {
	AT91_REG	 UDPHS_EPTCFG; 	/* UDPHS Endpoint Config Register*/
	AT91_REG	 UDPHS_EPTCTLENB; 	/* UDPHS Endpoint Control Enable Register*/
	AT91_REG	 UDPHS_EPTCTLDIS; 	/* UDPHS Endpoint Control Disable Register*/
	AT91_REG	 UDPHS_EPTCTL; 	/* UDPHS Endpoint Control Register*/
	AT91_REG	 Reserved0[1]; 	/* */
	AT91_REG	 UDPHS_EPTSETSTA; 	/* UDPHS Endpoint Set Status Register*/
	AT91_REG	 UDPHS_EPTCLRSTA; 	/* UDPHS Endpoint Clear Status Register*/
	AT91_REG	 UDPHS_EPTSTA; 	/* UDPHS Endpoint Status Register*/
} AT91S_UDPHS_EPT, *AT91PS_UDPHS_EPT;

/* -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- */
#define AT91C_UDPHS_EPT_SIZE  ((unsigned int) 0x7 <<  0) /* (UDPHS_EPT) Endpoint Size*/
#define 	AT91C_UDPHS_EPT_SIZE_8                    ((unsigned int) 0x0) /* (UDPHS_EPT)    8 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_16                   ((unsigned int) 0x1) /* (UDPHS_EPT)   16 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_32                   ((unsigned int) 0x2) /* (UDPHS_EPT)   32 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_64                   ((unsigned int) 0x3) /* (UDPHS_EPT)   64 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_128                  ((unsigned int) 0x4) /* (UDPHS_EPT)  128 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_256                  ((unsigned int) 0x5) /* (UDPHS_EPT)  256 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_512                  ((unsigned int) 0x6) /* (UDPHS_EPT)  512 bytes*/
#define 	AT91C_UDPHS_EPT_SIZE_1024                 ((unsigned int) 0x7) /* (UDPHS_EPT) 1024 bytes*/
#define AT91C_UDPHS_EPT_DIR   ((unsigned int) 0x1 <<  3) /* (UDPHS_EPT) Endpoint Direction 0:OUT, 1:IN*/
#define 	AT91C_UDPHS_EPT_DIR_OUT                  ((unsigned int) 0x0 <<  3) /* (UDPHS_EPT) Direction OUT*/
#define 	AT91C_UDPHS_EPT_DIR_IN                   ((unsigned int) 0x1 <<  3) /* (UDPHS_EPT) Direction IN*/
#define AT91C_UDPHS_EPT_TYPE  ((unsigned int) 0x3 <<  4) /* (UDPHS_EPT) Endpoint Type*/
#define 	AT91C_UDPHS_EPT_TYPE_CTL_EPT              ((unsigned int) 0x0 <<  4) /* (UDPHS_EPT) Control endpoint*/
#define 	AT91C_UDPHS_EPT_TYPE_ISO_EPT              ((unsigned int) 0x1 <<  4) /* (UDPHS_EPT) Isochronous endpoint*/
#define 	AT91C_UDPHS_EPT_TYPE_BUL_EPT              ((unsigned int) 0x2 <<  4) /* (UDPHS_EPT) Bulk endpoint*/
#define 	AT91C_UDPHS_EPT_TYPE_INT_EPT              ((unsigned int) 0x3 <<  4) /* (UDPHS_EPT) Interrupt endpoint*/
#define AT91C_UDPHS_BK_NUMBER ((unsigned int) 0x3 <<  6) /* (UDPHS_EPT) Number of Banks*/
#define 	AT91C_UDPHS_BK_NUMBER_0                    ((unsigned int) 0x0 <<  6) /* (UDPHS_EPT) Zero Bank, the EndPoint is not mapped in memory*/
#define 	AT91C_UDPHS_BK_NUMBER_1                    ((unsigned int) 0x1 <<  6) /* (UDPHS_EPT) One Bank (Bank0)*/
#define 	AT91C_UDPHS_BK_NUMBER_2                    ((unsigned int) 0x2 <<  6) /* (UDPHS_EPT) Double bank (Ping-Pong : Bank0 / Bank1)*/
#define 	AT91C_UDPHS_BK_NUMBER_3                    ((unsigned int) 0x3 <<  6) /* (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2)*/
#define AT91C_UDPHS_NB_TRANS  ((unsigned int) 0x3 <<  8) /* (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)*/
#define AT91C_UDPHS_EPT_MAPD  ((unsigned int) 0x1 << 31) /* (UDPHS_EPT) Endpoint Mapped (read only*/
/* -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- */
#define AT91C_UDPHS_EPT_ENABL ((unsigned int) 0x1 <<  0) /* (UDPHS_EPT) Endpoint Enable*/
#define AT91C_UDPHS_AUTO_VALID ((unsigned int) 0x1 <<  1) /* (UDPHS_EPT) Packet Auto-Valid Enable/Disable*/
#define AT91C_UDPHS_INTDIS_DMA ((unsigned int) 0x1 <<  3) /* (UDPHS_EPT) Endpoint Interrupts DMA Request Enable/Disable*/
#define AT91C_UDPHS_NYET_DIS  ((unsigned int) 0x1 <<  4) /* (UDPHS_EPT) NYET Enable/Disable*/
#define AT91C_UDPHS_DATAX_RX  ((unsigned int) 0x1 <<  6) /* (UDPHS_EPT) DATAx Interrupt Enable/Disable*/
#define AT91C_UDPHS_MDATA_RX  ((unsigned int) 0x1 <<  7) /* (UDPHS_EPT) MDATA Interrupt Enabled/Disable*/
#define AT91C_UDPHS_ERR_OVFLW ((unsigned int) 0x1 <<  8) /* (UDPHS_EPT) OverFlow Error Interrupt Enable/Disable/Status*/
#define AT91C_UDPHS_RX_BK_RDY ((unsigned int) 0x1 <<  9) /* (UDPHS_EPT) Received OUT Data*/
#define AT91C_UDPHS_TX_COMPLT ((unsigned int) 0x1 << 10) /* (UDPHS_EPT) Transmitted IN Data Complete Interrupt Enable/Disable or Transmitted IN Data Complete (clear)*/
#define AT91C_UDPHS_ERR_TRANS ((unsigned int) 0x1 << 11) /* (UDPHS_EPT) Transaction Error Interrupt Enable/Disable*/
#define AT91C_UDPHS_TX_PK_RDY ((unsigned int) 0x1 << 11) /* (UDPHS_EPT) TX Packet Ready Interrupt Enable/Disable*/
#define AT91C_UDPHS_RX_SETUP  ((unsigned int) 0x1 << 12) /* (UDPHS_EPT) Received SETUP Interrupt Enable/Disable*/
#define AT91C_UDPHS_ERR_FL_ISO ((unsigned int) 0x1 << 12) /* (UDPHS_EPT) Error Flow Clear/Interrupt Enable/Disable*/
#define AT91C_UDPHS_STALL_SNT ((unsigned int) 0x1 << 13) /* (UDPHS_EPT) Stall Sent Clear*/
#define AT91C_UDPHS_ERR_CRISO ((unsigned int) 0x1 << 13) /* (UDPHS_EPT) CRC error / Error NB Trans / Interrupt Enable/Disable*/
#define AT91C_UDPHS_NAK_IN    ((unsigned int) 0x1 << 14) /* (UDPHS_EPT) NAKIN ERROR FLUSH / Clear / Interrupt Enable/Disable*/
#define AT91C_UDPHS_NAK_OUT   ((unsigned int) 0x1 << 15) /* (UDPHS_EPT) NAKOUT / Clear / Interrupt Enable/Disable*/
#define AT91C_UDPHS_BUSY_BANK ((unsigned int) 0x1 << 18) /* (UDPHS_EPT) Busy Bank Interrupt Enable/Disable*/
#define AT91C_UDPHS_SHRT_PCKT ((unsigned int) 0x1 << 31) /* (UDPHS_EPT) Short Packet / Interrupt Enable/Disable*/
/* -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- */
#define AT91C_UDPHS_EPT_DISABL ((unsigned int) 0x1 <<  0) /* (UDPHS_EPT) Endpoint Disable*/
/* -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- */
/* -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- */
#define AT91C_UDPHS_FRCESTALL ((unsigned int) 0x1 <<  5) /* (UDPHS_EPT) Stall Handshake Request Set/Clear/Status*/
#define AT91C_UDPHS_KILL_BANK ((unsigned int) 0x1 <<  9) /* (UDPHS_EPT) KILL Bank*/
/* -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- */
#define AT91C_UDPHS_TOGGLESQ  ((unsigned int) 0x1 <<  6) /* (UDPHS_EPT) Data Toggle Clear*/
/* -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- */
#define AT91C_UDPHS_TOGGLESQ_STA ((unsigned int) 0x3 <<  6) /* (UDPHS_EPT) Toggle Sequencing*/
#define 	AT91C_UDPHS_TOGGLESQ_STA_00                   ((unsigned int) 0x0 <<  6) /* (UDPHS_EPT) Data0*/
#define 	AT91C_UDPHS_TOGGLESQ_STA_01                   ((unsigned int) 0x1 <<  6) /* (UDPHS_EPT) Data1*/
#define 	AT91C_UDPHS_TOGGLESQ_STA_10                   ((unsigned int) 0x2 <<  6) /* (UDPHS_EPT) Data2 (only for High-Bandwidth Isochronous EndPoint)*/
#define 	AT91C_UDPHS_TOGGLESQ_STA_11                   ((unsigned int) 0x3 <<  6) /* (UDPHS_EPT) MData (only for High-Bandwidth Isochronous EndPoint)*/
#define AT91C_UDPHS_CONTROL_DIR ((unsigned int) 0x3 << 16) /* (UDPHS_EPT) */
#define 	AT91C_UDPHS_CONTROL_DIR_00                   ((unsigned int) 0x0 << 16) /* (UDPHS_EPT) Bank 0*/
#define 	AT91C_UDPHS_CONTROL_DIR_01                   ((unsigned int) 0x1 << 16) /* (UDPHS_EPT) Bank 1*/
#define 	AT91C_UDPHS_CONTROL_DIR_10                   ((unsigned int) 0x2 << 16) /* (UDPHS_EPT) Bank 2*/
#define 	AT91C_UDPHS_CONTROL_DIR_11                   ((unsigned int) 0x3 << 16) /* (UDPHS_EPT) Invalid*/
#define AT91C_UDPHS_CURRENT_BANK ((unsigned int) 0x3 << 16) /* (UDPHS_EPT) */
#define 	AT91C_UDPHS_CURRENT_BANK_00                   ((unsigned int) 0x0 << 16) /* (UDPHS_EPT) Bank 0*/
#define 	AT91C_UDPHS_CURRENT_BANK_01                   ((unsigned int) 0x1 << 16) /* (UDPHS_EPT) Bank 1*/
#define 	AT91C_UDPHS_CURRENT_BANK_10                   ((unsigned int) 0x2 << 16) /* (UDPHS_EPT) Bank 2*/
#define 	AT91C_UDPHS_CURRENT_BANK_11                   ((unsigned int) 0x3 << 16) /* (UDPHS_EPT) Invalid*/
#define AT91C_UDPHS_BUSY_BANK_STA ((unsigned int) 0x3 << 18) /* (UDPHS_EPT) Busy Bank Number*/
#define 	AT91C_UDPHS_BUSY_BANK_STA_00                   ((unsigned int) 0x0 << 18) /* (UDPHS_EPT) All banks are free*/
#define 	AT91C_UDPHS_BUSY_BANK_STA_01                   ((unsigned int) 0x1 << 18) /* (UDPHS_EPT) 1 busy bank*/
#define 	AT91C_UDPHS_BUSY_BANK_STA_10                   ((unsigned int) 0x2 << 18) /* (UDPHS_EPT) 2 busy banks*/
#define 	AT91C_UDPHS_BUSY_BANK_STA_11                   ((unsigned int) 0x3 << 18) /* (UDPHS_EPT) 3 busy banks*/
#define AT91C_UDPHS_BYTE_COUNT ((unsigned int) 0x7FF << 20) /* (UDPHS_EPT) UDPHS Byte Count*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR UDPHS DMA struct*/
/* ******************************************************************************/
typedef struct _AT91S_UDPHS_DMA {
	AT91_REG	 UDPHS_DMANXTDSC; 	/* UDPHS DMA Channel Next Descriptor Address*/
	AT91_REG	 UDPHS_DMAADDRESS; 	/* UDPHS DMA Channel Address Register*/
	AT91_REG	 UDPHS_DMACONTROL; 	/* UDPHS DMA Channel Control Register*/
	AT91_REG	 UDPHS_DMASTATUS; 	/* UDPHS DMA Channel Status Register*/
} AT91S_UDPHS_DMA, *AT91PS_UDPHS_DMA;

/* -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- */
#define AT91C_UDPHS_NXT_DSC_ADD ((unsigned int) 0xFFFFFFF <<  4) /* (UDPHS_DMA) next Channel Descriptor*/
/* -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- */
#define AT91C_UDPHS_BUFF_ADD  ((unsigned int) 0x0 <<  0) /* (UDPHS_DMA) starting address of a DMA Channel transfer*/
/* -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- */
#define AT91C_UDPHS_CHANN_ENB ((unsigned int) 0x1 <<  0) /* (UDPHS_DMA) Channel Enabled*/
#define AT91C_UDPHS_LDNXT_DSC ((unsigned int) 0x1 <<  1) /* (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable*/
#define AT91C_UDPHS_END_TR_EN ((unsigned int) 0x1 <<  2) /* (UDPHS_DMA) Buffer Close Input Enable*/
#define AT91C_UDPHS_END_B_EN  ((unsigned int) 0x1 <<  3) /* (UDPHS_DMA) End of DMA Buffer Packet Validation*/
#define AT91C_UDPHS_END_TR_IT ((unsigned int) 0x1 <<  4) /* (UDPHS_DMA) End Of Transfer Interrupt Enable*/
#define AT91C_UDPHS_END_BUFFIT ((unsigned int) 0x1 <<  5) /* (UDPHS_DMA) End Of Channel Buffer Interrupt Enable*/
#define AT91C_UDPHS_DESC_LD_IT ((unsigned int) 0x1 <<  6) /* (UDPHS_DMA) Descriptor Loaded Interrupt Enable*/
#define AT91C_UDPHS_BURST_LCK ((unsigned int) 0x1 <<  7) /* (UDPHS_DMA) Burst Lock Enable*/
#define AT91C_UDPHS_BUFF_LENGTH ((unsigned int) 0xFFFF << 16) /* (UDPHS_DMA) Buffer Byte Length (write only)*/
/* -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- */
#define AT91C_UDPHS_CHANN_ACT ((unsigned int) 0x1 <<  1) /* (UDPHS_DMA) */
#define AT91C_UDPHS_END_TR_ST ((unsigned int) 0x1 <<  4) /* (UDPHS_DMA) */
#define AT91C_UDPHS_END_BF_ST ((unsigned int) 0x1 <<  5) /* (UDPHS_DMA) */
#define AT91C_UDPHS_DESC_LDST ((unsigned int) 0x1 <<  6) /* (UDPHS_DMA) */
#define AT91C_UDPHS_BUFF_COUNT ((unsigned int) 0xFFFF << 16) /* (UDPHS_DMA) */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface*/
/* ******************************************************************************/
typedef struct _AT91S_UDPHS {
	AT91_REG	 UDPHS_CTRL; 	/* UDPHS Control Register*/
	AT91_REG	 UDPHS_FNUM; 	/* UDPHS Frame Number Register*/
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 UDPHS_IEN; 	/* UDPHS Interrupt Enable Register*/
	AT91_REG	 UDPHS_INTSTA; 	/* UDPHS Interrupt Status Register*/
	AT91_REG	 UDPHS_CLRINT; 	/* UDPHS Clear Interrupt Register*/
	AT91_REG	 UDPHS_EPTRST; 	/* UDPHS Endpoints Reset Register*/
	AT91_REG	 Reserved1[44]; 	/* */
	AT91_REG	 UDPHS_TSTSOFCNT; 	/* UDPHS Test SOF Counter Register*/
	AT91_REG	 UDPHS_TSTCNTA; 	/* UDPHS Test A Counter Register*/
	AT91_REG	 UDPHS_TSTCNTB; 	/* UDPHS Test B Counter Register*/
	AT91_REG	 UDPHS_TSTMODREG; 	/* UDPHS Test Mode Register*/
	AT91_REG	 UDPHS_TST; 	/* UDPHS Test Register*/
	AT91_REG	 Reserved2[2]; 	/* */
	AT91_REG	 UDPHS_RIPPADDRSIZE; 	/* UDPHS PADDRSIZE Register*/
	AT91_REG	 UDPHS_RIPNAME1; 	/* UDPHS Name1 Register*/
	AT91_REG	 UDPHS_RIPNAME2; 	/* UDPHS Name2 Register*/
	AT91_REG	 UDPHS_IPFEATURES; 	/* UDPHS Features Register*/
	AT91_REG	 UDPHS_IPVERSION; 	/* UDPHS Version Register*/
	AT91S_UDPHS_EPT	 UDPHS_EPT[16]; 	/* UDPHS Endpoint struct*/
	AT91S_UDPHS_DMA	 UDPHS_DMA[8]; 	/* UDPHS DMA channel struct (not use [0])*/
} AT91S_UDPHS, *AT91PS_UDPHS;

/* -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- */
#define AT91C_UDPHS_DEV_ADDR  ((unsigned int) 0x7F <<  0) /* (UDPHS) UDPHS Address*/
#define AT91C_UDPHS_FADDR_EN  ((unsigned int) 0x1 <<  7) /* (UDPHS) Function Address Enable*/
#define AT91C_UDPHS_EN_UDPHS  ((unsigned int) 0x1 <<  8) /* (UDPHS) UDPHS Enable*/
#define AT91C_UDPHS_DETACH    ((unsigned int) 0x1 <<  9) /* (UDPHS) Detach Command*/
#define AT91C_UDPHS_REWAKEUP  ((unsigned int) 0x1 << 10) /* (UDPHS) Send Remote Wake Up*/
#define AT91C_UDPHS_PULLD_DIS ((unsigned int) 0x1 << 11) /* (UDPHS) PullDown Disable*/
/* -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- */
#define AT91C_UDPHS_MICRO_FRAME_NUM ((unsigned int) 0x7 <<  0) /* (UDPHS) Micro Frame Number*/
#define AT91C_UDPHS_FRAME_NUMBER ((unsigned int) 0x7FF <<  3) /* (UDPHS) Frame Number as defined in the Packet Field Formats*/
#define AT91C_UDPHS_FNUM_ERR  ((unsigned int) 0x1 << 31) /* (UDPHS) Frame Number CRC Error*/
/* -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- */
#define AT91C_UDPHS_DET_SUSPD ((unsigned int) 0x1 <<  1) /* (UDPHS) Suspend Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_MICRO_SOF ((unsigned int) 0x1 <<  2) /* (UDPHS) Micro-SOF Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_IEN_SOF   ((unsigned int) 0x1 <<  3) /* (UDPHS) SOF Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_ENDRESET  ((unsigned int) 0x1 <<  4) /* (UDPHS) End Of Reset Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_WAKE_UP   ((unsigned int) 0x1 <<  5) /* (UDPHS) Wake Up CPU Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_ENDOFRSM  ((unsigned int) 0x1 <<  6) /* (UDPHS) End Of Resume Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_UPSTR_RES ((unsigned int) 0x1 <<  7) /* (UDPHS) Upstream Resume Interrupt Enable/Clear/Status*/
#define AT91C_UDPHS_EPT_INT_0 ((unsigned int) 0x1 <<  8) /* (UDPHS) Endpoint 0 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_1 ((unsigned int) 0x1 <<  9) /* (UDPHS) Endpoint 1 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_2 ((unsigned int) 0x1 << 10) /* (UDPHS) Endpoint 2 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_3 ((unsigned int) 0x1 << 11) /* (UDPHS) Endpoint 3 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_4 ((unsigned int) 0x1 << 12) /* (UDPHS) Endpoint 4 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_5 ((unsigned int) 0x1 << 13) /* (UDPHS) Endpoint 5 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_6 ((unsigned int) 0x1 << 14) /* (UDPHS) Endpoint 6 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_7 ((unsigned int) 0x1 << 15) /* (UDPHS) Endpoint 7 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_8 ((unsigned int) 0x1 << 16) /* (UDPHS) Endpoint 8 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_9 ((unsigned int) 0x1 << 17) /* (UDPHS) Endpoint 9 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_10 ((unsigned int) 0x1 << 18) /* (UDPHS) Endpoint 10 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_11 ((unsigned int) 0x1 << 19) /* (UDPHS) Endpoint 11 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_12 ((unsigned int) 0x1 << 20) /* (UDPHS) Endpoint 12 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_13 ((unsigned int) 0x1 << 21) /* (UDPHS) Endpoint 13 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_14 ((unsigned int) 0x1 << 22) /* (UDPHS) Endpoint 14 Interrupt Enable/Status*/
#define AT91C_UDPHS_EPT_INT_15 ((unsigned int) 0x1 << 23) /* (UDPHS) Endpoint 15 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_1 ((unsigned int) 0x1 << 25) /* (UDPHS) DMA Channel 1 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_2 ((unsigned int) 0x1 << 26) /* (UDPHS) DMA Channel 2 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_3 ((unsigned int) 0x1 << 27) /* (UDPHS) DMA Channel 3 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_4 ((unsigned int) 0x1 << 28) /* (UDPHS) DMA Channel 4 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_5 ((unsigned int) 0x1 << 29) /* (UDPHS) DMA Channel 5 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_6 ((unsigned int) 0x1 << 30) /* (UDPHS) DMA Channel 6 Interrupt Enable/Status*/
#define AT91C_UDPHS_DMA_INT_7 ((unsigned int) 0x1 << 31) /* (UDPHS) DMA Channel 7 Interrupt Enable/Status*/
/* -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- */
#define AT91C_UDPHS_SPEED     ((unsigned int) 0x1 <<  0) /* (UDPHS) Speed Status*/
/* -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- */
/* -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- */
#define AT91C_UDPHS_RST_EPT_0 ((unsigned int) 0x1 <<  0) /* (UDPHS) Endpoint Reset 0*/
#define AT91C_UDPHS_RST_EPT_1 ((unsigned int) 0x1 <<  1) /* (UDPHS) Endpoint Reset 1*/
#define AT91C_UDPHS_RST_EPT_2 ((unsigned int) 0x1 <<  2) /* (UDPHS) Endpoint Reset 2*/
#define AT91C_UDPHS_RST_EPT_3 ((unsigned int) 0x1 <<  3) /* (UDPHS) Endpoint Reset 3*/
#define AT91C_UDPHS_RST_EPT_4 ((unsigned int) 0x1 <<  4) /* (UDPHS) Endpoint Reset 4*/
#define AT91C_UDPHS_RST_EPT_5 ((unsigned int) 0x1 <<  5) /* (UDPHS) Endpoint Reset 5*/
#define AT91C_UDPHS_RST_EPT_6 ((unsigned int) 0x1 <<  6) /* (UDPHS) Endpoint Reset 6*/
#define AT91C_UDPHS_RST_EPT_7 ((unsigned int) 0x1 <<  7) /* (UDPHS) Endpoint Reset 7*/
#define AT91C_UDPHS_RST_EPT_8 ((unsigned int) 0x1 <<  8) /* (UDPHS) Endpoint Reset 8*/
#define AT91C_UDPHS_RST_EPT_9 ((unsigned int) 0x1 <<  9) /* (UDPHS) Endpoint Reset 9*/
#define AT91C_UDPHS_RST_EPT_10 ((unsigned int) 0x1 << 10) /* (UDPHS) Endpoint Reset 10*/
#define AT91C_UDPHS_RST_EPT_11 ((unsigned int) 0x1 << 11) /* (UDPHS) Endpoint Reset 11*/
#define AT91C_UDPHS_RST_EPT_12 ((unsigned int) 0x1 << 12) /* (UDPHS) Endpoint Reset 12*/
#define AT91C_UDPHS_RST_EPT_13 ((unsigned int) 0x1 << 13) /* (UDPHS) Endpoint Reset 13*/
#define AT91C_UDPHS_RST_EPT_14 ((unsigned int) 0x1 << 14) /* (UDPHS) Endpoint Reset 14*/
#define AT91C_UDPHS_RST_EPT_15 ((unsigned int) 0x1 << 15) /* (UDPHS) Endpoint Reset 15*/
/* -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- */
#define AT91C_UDPHS_SOFCNTMAX ((unsigned int) 0x3 <<  0) /* (UDPHS) SOF Counter Max Value*/
#define AT91C_UDPHS_SOFCTLOAD ((unsigned int) 0x1 <<  7) /* (UDPHS) SOF Counter Load*/
/* -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- */
#define AT91C_UDPHS_CNTAMAX   ((unsigned int) 0x7FFF <<  0) /* (UDPHS) A Counter Max Value*/
#define AT91C_UDPHS_CNTALOAD  ((unsigned int) 0x1 << 15) /* (UDPHS) A Counter Load*/
/* -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- */
#define AT91C_UDPHS_CNTBMAX   ((unsigned int) 0x7FFF <<  0) /* (UDPHS) B Counter Max Value*/
#define AT91C_UDPHS_CNTBLOAD  ((unsigned int) 0x1 << 15) /* (UDPHS) B Counter Load*/
/* -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- */
#define AT91C_UDPHS_TSTMODE   ((unsigned int) 0x1F <<  1) /* (UDPHS) UDPHS Core TestModeReg*/
/* -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- */
#define AT91C_UDPHS_SPEED_CFG ((unsigned int) 0x3 <<  0) /* (UDPHS) Speed Configuration*/
#define 	AT91C_UDPHS_SPEED_CFG_NM                   ((unsigned int) 0x0) /* (UDPHS) Normal Mode*/
#define 	AT91C_UDPHS_SPEED_CFG_RS                   ((unsigned int) 0x1) /* (UDPHS) Reserved*/
#define 	AT91C_UDPHS_SPEED_CFG_HS                   ((unsigned int) 0x2) /* (UDPHS) Force High Speed*/
#define 	AT91C_UDPHS_SPEED_CFG_FS                   ((unsigned int) 0x3) /* (UDPHS) Force Full-Speed*/
#define AT91C_UDPHS_TST_J     ((unsigned int) 0x1 <<  2) /* (UDPHS) TestJMode*/
#define AT91C_UDPHS_TST_K     ((unsigned int) 0x1 <<  3) /* (UDPHS) TestKMode*/
#define AT91C_UDPHS_TST_PKT   ((unsigned int) 0x1 <<  4) /* (UDPHS) TestPacketMode*/
#define AT91C_UDPHS_OPMODE2   ((unsigned int) 0x1 <<  5) /* (UDPHS) OpMode2*/
/* -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- */
#define AT91C_UDPHS_IPPADDRSIZE ((unsigned int) 0x0 <<  0) /* (UDPHS) 2^UDPHSDEV_PADDR_SIZE*/
/* -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- */
#define AT91C_UDPHS_IPNAME1   ((unsigned int) 0x0 <<  0) /* (UDPHS) ASCII string HUSB*/
/* -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- */
#define AT91C_UDPHS_IPNAME2   ((unsigned int) 0x0 <<  0) /* (UDPHS) ASCII string 2DEV*/
/* -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- */
#define AT91C_UDPHS_EPT_NBR_MAX ((unsigned int) 0xF <<  0) /* (UDPHS) Max Number of Endpoints*/
#define AT91C_UDPHS_DMA_CHANNEL_NBR ((unsigned int) 0x7 <<  4) /* (UDPHS) Number of DMA Channels*/
#define AT91C_UDPHS_DMA_B_SIZ ((unsigned int) 0x1 <<  7) /* (UDPHS) DMA Buffer Size*/
#define AT91C_UDPHS_DMA_FIFO_WORD_DEPTH ((unsigned int) 0xF <<  8) /* (UDPHS) DMA FIFO Depth in words*/
#define AT91C_UDPHS_FIFO_MAX_SIZE ((unsigned int) 0x7 << 12) /* (UDPHS) DPRAM size*/
#define AT91C_UDPHS_BW_DPRAM  ((unsigned int) 0x1 << 15) /* (UDPHS) DPRAM byte write capability*/
#define AT91C_UDPHS_DATAB16_8 ((unsigned int) 0x1 << 16) /* (UDPHS) UTMI DataBus16_8*/
#define AT91C_UDPHS_ISO_EPT_1 ((unsigned int) 0x1 << 17) /* (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_2 ((unsigned int) 0x1 << 18) /* (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_3 ((unsigned int) 0x1 << 19) /* (UDPHS) Endpoint 3 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_4 ((unsigned int) 0x1 << 20) /* (UDPHS) Endpoint 4 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_5 ((unsigned int) 0x1 << 21) /* (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_6 ((unsigned int) 0x1 << 22) /* (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_7 ((unsigned int) 0x1 << 23) /* (UDPHS) Endpoint 7 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_8 ((unsigned int) 0x1 << 24) /* (UDPHS) Endpoint 8 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_9 ((unsigned int) 0x1 << 25) /* (UDPHS) Endpoint 9 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_10 ((unsigned int) 0x1 << 26) /* (UDPHS) Endpoint 10 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_11 ((unsigned int) 0x1 << 27) /* (UDPHS) Endpoint 11 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_12 ((unsigned int) 0x1 << 28) /* (UDPHS) Endpoint 12 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_13 ((unsigned int) 0x1 << 29) /* (UDPHS) Endpoint 13 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_14 ((unsigned int) 0x1 << 30) /* (UDPHS) Endpoint 14 High Bandwidth Isochronous Capability*/
#define AT91C_UDPHS_ISO_EPT_15 ((unsigned int) 0x1 << 31) /* (UDPHS) Endpoint 15 High Bandwidth Isochronous Capability*/
/* -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- */
#define AT91C_UDPHS_VERSION_NUM ((unsigned int) 0xFFFF <<  0) /* (UDPHS) Give the IP version*/
#define AT91C_UDPHS_METAL_FIX_NUM ((unsigned int) 0x7 << 16) /* (UDPHS) Give the number of metal fixes*/

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR AC97 Controller Interface*/
/* ******************************************************************************/
typedef struct _AT91S_AC97C {
	AT91_REG	 Reserved0[2]; 	/* */
	AT91_REG	 AC97C_MR; 	/* Mode Register*/
	AT91_REG	 Reserved1[1]; 	/* */
	AT91_REG	 AC97C_ICA; 	/* Input Channel AssignementRegister*/
	AT91_REG	 AC97C_OCA; 	/* Output Channel Assignement Register*/
	AT91_REG	 Reserved2[2]; 	/* */
	AT91_REG	 AC97C_CARHR; 	/* Channel A Receive Holding Register*/
	AT91_REG	 AC97C_CATHR; 	/* Channel A Transmit Holding Register*/
	AT91_REG	 AC97C_CASR; 	/* Channel A Status Register*/
	AT91_REG	 AC97C_CAMR; 	/* Channel A Mode Register*/
	AT91_REG	 AC97C_CBRHR; 	/* Channel B Receive Holding Register (optional)*/
	AT91_REG	 AC97C_CBTHR; 	/* Channel B Transmit Holding Register (optional)*/
	AT91_REG	 AC97C_CBSR; 	/* Channel B Status Register*/
	AT91_REG	 AC97C_CBMR; 	/* Channel B Mode Register*/
	AT91_REG	 AC97C_CORHR; 	/* COdec Transmit Holding Register*/
	AT91_REG	 AC97C_COTHR; 	/* COdec Transmit Holding Register*/
	AT91_REG	 AC97C_COSR; 	/* CODEC Status Register*/
	AT91_REG	 AC97C_COMR; 	/* CODEC Mask Status Register*/
	AT91_REG	 AC97C_SR; 	/* Status Register*/
	AT91_REG	 AC97C_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 AC97C_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 AC97C_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 Reserved3[39]; 	/* */
	AT91_REG	 AC97C_VERSION; 	/* Version Register*/
	AT91_REG	 AC97C_RPR; 	/* Receive Pointer Register*/
	AT91_REG	 AC97C_RCR; 	/* Receive Counter Register*/
	AT91_REG	 AC97C_TPR; 	/* Transmit Pointer Register*/
	AT91_REG	 AC97C_TCR; 	/* Transmit Counter Register*/
	AT91_REG	 AC97C_RNPR; 	/* Receive Next Pointer Register*/
	AT91_REG	 AC97C_RNCR; 	/* Receive Next Counter Register*/
	AT91_REG	 AC97C_TNPR; 	/* Transmit Next Pointer Register*/
	AT91_REG	 AC97C_TNCR; 	/* Transmit Next Counter Register*/
	AT91_REG	 AC97C_PTCR; 	/* PDC Transfer Control Register*/
	AT91_REG	 AC97C_PTSR; 	/* PDC Transfer Status Register*/
} AT91S_AC97C, *AT91PS_AC97C;

/* -------- AC97C_MR : (AC97C Offset: 0x8) AC97C Mode Register -------- */
#define AT91C_AC97C_ENA       ((unsigned int) 0x1 <<  0) /* (AC97C) AC97 Controller Global Enable*/
#define AT91C_AC97C_WRST      ((unsigned int) 0x1 <<  1) /* (AC97C) Warm Reset*/
#define AT91C_AC97C_VRA       ((unsigned int) 0x1 <<  2) /* (AC97C) Variable RAte (for Data Slots)*/
/* -------- AC97C_ICA : (AC97C Offset: 0x10) AC97C Input Channel Assignement Register -------- */
#define AT91C_AC97C_CHID3     ((unsigned int) 0x7 <<  0) /* (AC97C) Channel Id for the input slot 3*/
#define 	AT91C_AC97C_CHID3_NONE                 ((unsigned int) 0x0) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID3_CA                   ((unsigned int) 0x1) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID3_CB                   ((unsigned int) 0x2) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID3_CC                   ((unsigned int) 0x3) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID4     ((unsigned int) 0x7 <<  3) /* (AC97C) Channel Id for the input slot 4*/
#define 	AT91C_AC97C_CHID4_NONE                 ((unsigned int) 0x0 <<  3) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID4_CA                   ((unsigned int) 0x1 <<  3) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID4_CB                   ((unsigned int) 0x2 <<  3) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID4_CC                   ((unsigned int) 0x3 <<  3) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID5     ((unsigned int) 0x7 <<  6) /* (AC97C) Channel Id for the input slot 5*/
#define 	AT91C_AC97C_CHID5_NONE                 ((unsigned int) 0x0 <<  6) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID5_CA                   ((unsigned int) 0x1 <<  6) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID5_CB                   ((unsigned int) 0x2 <<  6) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID5_CC                   ((unsigned int) 0x3 <<  6) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID6     ((unsigned int) 0x7 <<  9) /* (AC97C) Channel Id for the input slot 6*/
#define 	AT91C_AC97C_CHID6_NONE                 ((unsigned int) 0x0 <<  9) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID6_CA                   ((unsigned int) 0x1 <<  9) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID6_CB                   ((unsigned int) 0x2 <<  9) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID6_CC                   ((unsigned int) 0x3 <<  9) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID7     ((unsigned int) 0x7 << 12) /* (AC97C) Channel Id for the input slot 7*/
#define 	AT91C_AC97C_CHID7_NONE                 ((unsigned int) 0x0 << 12) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID7_CA                   ((unsigned int) 0x1 << 12) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID7_CB                   ((unsigned int) 0x2 << 12) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID7_CC                   ((unsigned int) 0x3 << 12) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID8     ((unsigned int) 0x7 << 15) /* (AC97C) Channel Id for the input slot 8*/
#define 	AT91C_AC97C_CHID8_NONE                 ((unsigned int) 0x0 << 15) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID8_CA                   ((unsigned int) 0x1 << 15) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID8_CB                   ((unsigned int) 0x2 << 15) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID8_CC                   ((unsigned int) 0x3 << 15) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID9     ((unsigned int) 0x7 << 18) /* (AC97C) Channel Id for the input slot 9*/
#define 	AT91C_AC97C_CHID9_NONE                 ((unsigned int) 0x0 << 18) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID9_CA                   ((unsigned int) 0x1 << 18) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID9_CB                   ((unsigned int) 0x2 << 18) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID9_CC                   ((unsigned int) 0x3 << 18) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID10    ((unsigned int) 0x7 << 21) /* (AC97C) Channel Id for the input slot 10*/
#define 	AT91C_AC97C_CHID10_NONE                 ((unsigned int) 0x0 << 21) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID10_CA                   ((unsigned int) 0x1 << 21) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID10_CB                   ((unsigned int) 0x2 << 21) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID10_CC                   ((unsigned int) 0x3 << 21) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID11    ((unsigned int) 0x7 << 24) /* (AC97C) Channel Id for the input slot 11*/
#define 	AT91C_AC97C_CHID11_NONE                 ((unsigned int) 0x0 << 24) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID11_CA                   ((unsigned int) 0x1 << 24) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID11_CB                   ((unsigned int) 0x2 << 24) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID11_CC                   ((unsigned int) 0x3 << 24) /* (AC97C) Channel C data will be transmitted during this slot*/
#define AT91C_AC97C_CHID12    ((unsigned int) 0x7 << 27) /* (AC97C) Channel Id for the input slot 12*/
#define 	AT91C_AC97C_CHID12_NONE                 ((unsigned int) 0x0 << 27) /* (AC97C) No data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID12_CA                   ((unsigned int) 0x1 << 27) /* (AC97C) Channel A data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID12_CB                   ((unsigned int) 0x2 << 27) /* (AC97C) Channel B data will be transmitted during this slot*/
#define 	AT91C_AC97C_CHID12_CC                   ((unsigned int) 0x3 << 27) /* (AC97C) Channel C data will be transmitted during this slot*/
/* -------- AC97C_OCA : (AC97C Offset: 0x14) AC97C Output Channel Assignement Register -------- */
/* -------- AC97C_CARHR : (AC97C Offset: 0x20) AC97C Channel A Receive Holding Register -------- */
#define AT91C_AC97C_RDATA     ((unsigned int) 0xFFFFF <<  0) /* (AC97C) Receive data*/
/* -------- AC97C_CATHR : (AC97C Offset: 0x24) AC97C Channel A Transmit Holding Register -------- */
#define AT91C_AC97C_TDATA     ((unsigned int) 0xFFFFF <<  0) /* (AC97C) Transmit data*/
/* -------- AC97C_CASR : (AC97C Offset: 0x28) AC97C Channel A Status Register -------- */
#define AT91C_AC97C_TXRDY     ((unsigned int) 0x1 <<  0) /* (AC97C) */
#define AT91C_AC97C_TXEMPTY   ((unsigned int) 0x1 <<  1) /* (AC97C) */
#define AT91C_AC97C_UNRUN     ((unsigned int) 0x1 <<  2) /* (AC97C) */
#define AT91C_AC97C_RXRDY     ((unsigned int) 0x1 <<  4) /* (AC97C) */
#define AT91C_AC97C_OVRUN     ((unsigned int) 0x1 <<  5) /* (AC97C) */
#define AT91C_AC97C_ENDTX     ((unsigned int) 0x1 << 10) /* (AC97C) */
#define AT91C_AC97C_TXBUFE    ((unsigned int) 0x1 << 11) /* (AC97C) */
#define AT91C_AC97C_ENDRX     ((unsigned int) 0x1 << 14) /* (AC97C) */
#define AT91C_AC97C_RXBUFF    ((unsigned int) 0x1 << 15) /* (AC97C) */
/* -------- AC97C_CAMR : (AC97C Offset: 0x2c) AC97C Channel A Mode Register -------- */
#define AT91C_AC97C_SIZE      ((unsigned int) 0x3 << 16) /* (AC97C) */
#define 	AT91C_AC97C_SIZE_20_BITS              ((unsigned int) 0x0 << 16) /* (AC97C) Data size is 20 bits*/
#define 	AT91C_AC97C_SIZE_18_BITS              ((unsigned int) 0x1 << 16) /* (AC97C) Data size is 18 bits*/
#define 	AT91C_AC97C_SIZE_16_BITS              ((unsigned int) 0x2 << 16) /* (AC97C) Data size is 16 bits*/
#define 	AT91C_AC97C_SIZE_10_BITS              ((unsigned int) 0x3 << 16) /* (AC97C) Data size is 10 bits*/
#define AT91C_AC97C_CEM       ((unsigned int) 0x1 << 18) /* (AC97C) */
#define AT91C_AC97C_CEN       ((unsigned int) 0x1 << 21) /* (AC97C) */
#define AT91C_AC97C_PDCEN     ((unsigned int) 0x1 << 22) /* (AC97C) */
/* -------- AC97C_CBRHR : (AC97C Offset: 0x30) AC97C Channel B Receive Holding Register -------- */
/* -------- AC97C_CBTHR : (AC97C Offset: 0x34) AC97C Channel B Transmit Holding Register -------- */
/* -------- AC97C_CBSR : (AC97C Offset: 0x38) AC97C Channel B Status Register -------- */
/* -------- AC97C_CBMR : (AC97C Offset: 0x3c) AC97C Channel B Mode Register -------- */
/* -------- AC97C_CORHR : (AC97C Offset: 0x40) AC97C Codec Channel Receive Holding Register -------- */
#define AT91C_AC97C_SDATA     ((unsigned int) 0xFFFF <<  0) /* (AC97C) Status Data*/
/* -------- AC97C_COTHR : (AC97C Offset: 0x44) AC97C Codec Channel Transmit Holding Register -------- */
#define AT91C_AC97C_CDATA     ((unsigned int) 0xFFFF <<  0) /* (AC97C) Command Data*/
#define AT91C_AC97C_CADDR     ((unsigned int) 0x7F << 16) /* (AC97C) COdec control register index*/
#define AT91C_AC97C_READ      ((unsigned int) 0x1 << 23) /* (AC97C) Read/Write command*/
/* -------- AC97C_COSR : (AC97C Offset: 0x48) AC97C CODEC Status Register -------- */
/* -------- AC97C_COMR : (AC97C Offset: 0x4c) AC97C CODEC Mode Register -------- */
/* -------- AC97C_SR : (AC97C Offset: 0x50) AC97C Status Register -------- */
#define AT91C_AC97C_SOF       ((unsigned int) 0x1 <<  0) /* (AC97C) */
#define AT91C_AC97C_WKUP      ((unsigned int) 0x1 <<  1) /* (AC97C) */
#define AT91C_AC97C_COEVT     ((unsigned int) 0x1 <<  2) /* (AC97C) */
#define AT91C_AC97C_CAEVT     ((unsigned int) 0x1 <<  3) /* (AC97C) */
#define AT91C_AC97C_CBEVT     ((unsigned int) 0x1 <<  4) /* (AC97C) */
/* -------- AC97C_IER : (AC97C Offset: 0x54) AC97C Interrupt Enable Register -------- */
/* -------- AC97C_IDR : (AC97C Offset: 0x58) AC97C Interrupt Disable Register -------- */
/* -------- AC97C_IMR : (AC97C Offset: 0x5c) AC97C Interrupt Mask Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR LCD Controller*/
/* ******************************************************************************/
typedef struct _AT91S_LCDC {
	AT91_REG	 LCDC_BA1; 	/* DMA Base Address Register 1*/
	AT91_REG	 LCDC_BA2; 	/* DMA Base Address Register 2*/
	AT91_REG	 LCDC_FRMP1; 	/* DMA Frame Pointer Register 1*/
	AT91_REG	 LCDC_FRMP2; 	/* DMA Frame Pointer Register 2*/
	AT91_REG	 LCDC_FRMA1; 	/* DMA Frame Address Register 1*/
	AT91_REG	 LCDC_FRMA2; 	/* DMA Frame Address Register 2*/
	AT91_REG	 LCDC_FRMCFG; 	/* DMA Frame Configuration Register*/
	AT91_REG	 LCDC_DMACON; 	/* DMA Control Register*/
	AT91_REG	 LCDC_DMA2DCFG; 	/* DMA 2D addressing configuration*/
	AT91_REG	 Reserved0[503]; 	/* */
	AT91_REG	 LCDC_LCDCON1; 	/* LCD Control 1 Register*/
	AT91_REG	 LCDC_LCDCON2; 	/* LCD Control 2 Register*/
	AT91_REG	 LCDC_TIM1; 	/* LCD Timing Config 1 Register*/
	AT91_REG	 LCDC_TIM2; 	/* LCD Timing Config 2 Register*/
	AT91_REG	 LCDC_LCDFRCFG; 	/* LCD Frame Config Register*/
	AT91_REG	 LCDC_FIFO; 	/* LCD FIFO Register*/
	AT91_REG	 LCDC_MVAL; 	/* LCD Mode Toggle Rate Value Register*/
	AT91_REG	 LCDC_DP1_2; 	/* Dithering Pattern DP1_2 Register*/
	AT91_REG	 LCDC_DP4_7; 	/* Dithering Pattern DP4_7 Register*/
	AT91_REG	 LCDC_DP3_5; 	/* Dithering Pattern DP3_5 Register*/
	AT91_REG	 LCDC_DP2_3; 	/* Dithering Pattern DP2_3 Register*/
	AT91_REG	 LCDC_DP5_7; 	/* Dithering Pattern DP5_7 Register*/
	AT91_REG	 LCDC_DP3_4; 	/* Dithering Pattern DP3_4 Register*/
	AT91_REG	 LCDC_DP4_5; 	/* Dithering Pattern DP4_5 Register*/
	AT91_REG	 LCDC_DP6_7; 	/* Dithering Pattern DP6_7 Register*/
	AT91_REG	 LCDC_PWRCON; 	/* Power Control Register*/
	AT91_REG	 LCDC_CTRSTCON; 	/* Contrast Control Register*/
	AT91_REG	 LCDC_CTRSTVAL; 	/* Contrast Value Register*/
	AT91_REG	 LCDC_IER; 	/* Interrupt Enable Register*/
	AT91_REG	 LCDC_IDR; 	/* Interrupt Disable Register*/
	AT91_REG	 LCDC_IMR; 	/* Interrupt Mask Register*/
	AT91_REG	 LCDC_ISR; 	/* Interrupt Enable Register*/
	AT91_REG	 LCDC_ICR; 	/* Interrupt Clear Register*/
	AT91_REG	 LCDC_GPR; 	/* General Purpose Register*/
	AT91_REG	 LCDC_ITR; 	/* Interrupts Test Register*/
	AT91_REG	 LCDC_IRR; 	/* Interrupts Raw Status Register*/
	AT91_REG	 Reserved1[230]; 	/* */
	AT91_REG	 LCDC_LUT_ENTRY[256]; 	/* LUT Entries Register*/
} AT91S_LCDC, *AT91PS_LCDC;

/* -------- LCDC_FRMP1 : (LCDC Offset: 0x8) DMA Frame Pointer 1 Register -------- */
#define AT91C_LCDC_FRMPT1     ((unsigned int) 0x3FFFFF <<  0) /* (LCDC) Frame Pointer Address 1*/
/* -------- LCDC_FRMP2 : (LCDC Offset: 0xc) DMA Frame Pointer 2 Register -------- */
#define AT91C_LCDC_FRMPT2     ((unsigned int) 0x1FFFFF <<  0) /* (LCDC) Frame Pointer Address 2*/
/* -------- LCDC_FRMCFG : (LCDC Offset: 0x18) DMA Frame Config Register -------- */
#define AT91C_LCDC_FRSIZE     ((unsigned int) 0x3FFFFF <<  0) /* (LCDC) FRAME SIZE*/
#define AT91C_LCDC_BLENGTH    ((unsigned int) 0xF << 24) /* (LCDC) BURST LENGTH*/
/* -------- LCDC_DMACON : (LCDC Offset: 0x1c) DMA Control Register -------- */
#define AT91C_LCDC_DMAEN      ((unsigned int) 0x1 <<  0) /* (LCDC) DAM Enable*/
#define AT91C_LCDC_DMARST     ((unsigned int) 0x1 <<  1) /* (LCDC) DMA Reset (WO)*/
#define AT91C_LCDC_DMABUSY    ((unsigned int) 0x1 <<  2) /* (LCDC) DMA Reset (WO)*/
#define AT91C_LCDC_DMAUPDT    ((unsigned int) 0x1 <<  3) /* (LCDC) DMA Configuration Update*/
#define AT91C_LCDC_DMA2DEN    ((unsigned int) 0x1 <<  4) /* (LCDC) 2D Addressing Enable*/
/* -------- LCDC_DMA2DCFG : (LCDC Offset: 0x20) DMA 2D addressing configuration Register -------- */
#define AT91C_LCDC_ADDRINC    ((unsigned int) 0xFFFF <<  0) /* (LCDC) Number of 32b words that the DMA must jump when going to the next line*/
#define AT91C_LCDC_PIXELOFF   ((unsigned int) 0x1F << 24) /* (LCDC) Offset (in bits) of the first pixel of the screen in the memory word which contain it*/
/* -------- LCDC_LCDCON1 : (LCDC Offset: 0x800) LCD Control 1 Register -------- */
#define AT91C_LCDC_BYPASS     ((unsigned int) 0x1 <<  0) /* (LCDC) Bypass lcd_pccklk divider*/
#define AT91C_LCDC_CLKVAL     ((unsigned int) 0x1FF << 12) /* (LCDC) 9-bit Divider for pixel clock frequency*/
#define AT91C_LCDC_LINCNT     ((unsigned int) 0x7FF << 21) /* (LCDC) Line Counter (RO)*/
/* -------- LCDC_LCDCON2 : (LCDC Offset: 0x804) LCD Control 2 Register -------- */
#define AT91C_LCDC_DISTYPE    ((unsigned int) 0x3 <<  0) /* (LCDC) Display Type*/
#define 	AT91C_LCDC_DISTYPE_STNMONO              ((unsigned int) 0x0) /* (LCDC) STN Mono*/
#define 	AT91C_LCDC_DISTYPE_STNCOLOR             ((unsigned int) 0x1) /* (LCDC) STN Color*/
#define 	AT91C_LCDC_DISTYPE_TFT                  ((unsigned int) 0x2) /* (LCDC) TFT*/
#define AT91C_LCDC_SCANMOD    ((unsigned int) 0x1 <<  2) /* (LCDC) Scan Mode*/
#define 	AT91C_LCDC_SCANMOD_SINGLESCAN           ((unsigned int) 0x0 <<  2) /* (LCDC) Single Scan*/
#define 	AT91C_LCDC_SCANMOD_DUALSCAN             ((unsigned int) 0x1 <<  2) /* (LCDC) Dual Scan*/
#define AT91C_LCDC_IFWIDTH    ((unsigned int) 0x3 <<  3) /* (LCDC) Interface Width*/
#define 	AT91C_LCDC_IFWIDTH_FOURBITSWIDTH        ((unsigned int) 0x0 <<  3) /* (LCDC) 4 Bits*/
#define 	AT91C_LCDC_IFWIDTH_EIGTHBITSWIDTH       ((unsigned int) 0x1 <<  3) /* (LCDC) 8 Bits*/
#define 	AT91C_LCDC_IFWIDTH_SIXTEENBITSWIDTH     ((unsigned int) 0x2 <<  3) /* (LCDC) 16 Bits*/
#define AT91C_LCDC_PIXELSIZE  ((unsigned int) 0x7 <<  5) /* (LCDC) Bits per pixel*/
#define 	AT91C_LCDC_PIXELSIZE_ONEBITSPERPIXEL      ((unsigned int) 0x0 <<  5) /* (LCDC) 1 Bits*/
#define 	AT91C_LCDC_PIXELSIZE_TWOBITSPERPIXEL      ((unsigned int) 0x1 <<  5) /* (LCDC) 2 Bits*/
#define 	AT91C_LCDC_PIXELSIZE_FOURBITSPERPIXEL     ((unsigned int) 0x2 <<  5) /* (LCDC) 4 Bits*/
#define 	AT91C_LCDC_PIXELSIZE_EIGTHBITSPERPIXEL    ((unsigned int) 0x3 <<  5) /* (LCDC) 8 Bits*/
#define 	AT91C_LCDC_PIXELSIZE_SIXTEENBITSPERPIXEL  ((unsigned int) 0x4 <<  5) /* (LCDC) 16 Bits*/
#define 	AT91C_LCDC_PIXELSIZE_TWENTYFOURBITSPERPIXEL ((unsigned int) 0x5 <<  5) /* (LCDC) 24 Bits*/
#define AT91C_LCDC_INVVD      ((unsigned int) 0x1 <<  8) /* (LCDC) lcd datas polarity*/
#define 	AT91C_LCDC_INVVD_NORMALPOL            ((unsigned int) 0x0 <<  8) /* (LCDC) Normal Polarity*/
#define 	AT91C_LCDC_INVVD_INVERTEDPOL          ((unsigned int) 0x1 <<  8) /* (LCDC) Inverted Polarity*/
#define AT91C_LCDC_INVFRAME   ((unsigned int) 0x1 <<  9) /* (LCDC) lcd vsync polarity*/
#define 	AT91C_LCDC_INVFRAME_NORMALPOL            ((unsigned int) 0x0 <<  9) /* (LCDC) Normal Polarity*/
#define 	AT91C_LCDC_INVFRAME_INVERTEDPOL          ((unsigned int) 0x1 <<  9) /* (LCDC) Inverted Polarity*/
#define AT91C_LCDC_INVLINE    ((unsigned int) 0x1 << 10) /* (LCDC) lcd hsync polarity*/
#define 	AT91C_LCDC_INVLINE_NORMALPOL            ((unsigned int) 0x0 << 10) /* (LCDC) Normal Polarity*/
#define 	AT91C_LCDC_INVLINE_INVERTEDPOL          ((unsigned int) 0x1 << 10) /* (LCDC) Inverted Polarity*/
#define AT91C_LCDC_INVCLK     ((unsigned int) 0x1 << 11) /* (LCDC) lcd pclk polarity*/
#define 	AT91C_LCDC_INVCLK_NORMALPOL            ((unsigned int) 0x0 << 11) /* (LCDC) Normal Polarity*/
#define 	AT91C_LCDC_INVCLK_INVERTEDPOL          ((unsigned int) 0x1 << 11) /* (LCDC) Inverted Polarity*/
#define AT91C_LCDC_INVDVAL    ((unsigned int) 0x1 << 12) /* (LCDC) lcd dval polarity*/
#define 	AT91C_LCDC_INVDVAL_NORMALPOL            ((unsigned int) 0x0 << 12) /* (LCDC) Normal Polarity*/
#define 	AT91C_LCDC_INVDVAL_INVERTEDPOL          ((unsigned int) 0x1 << 12) /* (LCDC) Inverted Polarity*/
#define AT91C_LCDC_CLKMOD     ((unsigned int) 0x1 << 15) /* (LCDC) lcd pclk Mode*/
#define 	AT91C_LCDC_CLKMOD_ACTIVEONLYDISP       ((unsigned int) 0x0 << 15) /* (LCDC) Active during display period*/
#define 	AT91C_LCDC_CLKMOD_ALWAYSACTIVE         ((unsigned int) 0x1 << 15) /* (LCDC) Always Active*/
#define AT91C_LCDC_MEMOR      ((unsigned int) 0x1 << 31) /* (LCDC) lcd pclk Mode*/
#define 	AT91C_LCDC_MEMOR_BIGIND               ((unsigned int) 0x0 << 31) /* (LCDC) Big Endian*/
#define 	AT91C_LCDC_MEMOR_LITTLEIND            ((unsigned int) 0x1 << 31) /* (LCDC) Little Endian*/
/* -------- LCDC_TIM1 : (LCDC Offset: 0x808) LCDC Timing Config 1 Register -------- */
#define AT91C_LCDC_VFP        ((unsigned int) 0xFF <<  0) /* (LCDC) Vertical Front Porch*/
#define AT91C_LCDC_VBP        ((unsigned int) 0xFF <<  8) /* (LCDC) Vertical Back Porch*/
#define AT91C_LCDC_VPW        ((unsigned int) 0x3F << 16) /* (LCDC) Vertical Synchronization Pulse Width*/
#define AT91C_LCDC_VHDLY      ((unsigned int) 0xF << 24) /* (LCDC) Vertical to Horizontal Delay*/
/* -------- LCDC_TIM2 : (LCDC Offset: 0x80c) LCDC Timing Config 2 Register -------- */
#define AT91C_LCDC_HBP        ((unsigned int) 0xFF <<  0) /* (LCDC) Horizontal Back Porch*/
#define AT91C_LCDC_HPW        ((unsigned int) 0x3F <<  8) /* (LCDC) Horizontal Synchronization Pulse Width*/
#define AT91C_LCDC_HFP        ((unsigned int) 0x3FF << 22) /* (LCDC) Horizontal Front Porch*/
/* -------- LCDC_LCDFRCFG : (LCDC Offset: 0x810) LCD Frame Config Register -------- */
#define AT91C_LCDC_LINEVAL    ((unsigned int) 0x7FF <<  0) /* (LCDC) Vertical Size of LCD Module*/
#define AT91C_LCDC_HOZVAL     ((unsigned int) 0x7FF << 21) /* (LCDC) Horizontal Size of LCD Module*/
/* -------- LCDC_FIFO : (LCDC Offset: 0x814) LCD FIFO Register -------- */
#define AT91C_LCDC_FIFOTH     ((unsigned int) 0xFFFF <<  0) /* (LCDC) FIFO Threshold*/
/* -------- LCDC_MVAL : (LCDC Offset: 0x818) LCD Mode Toggle Rate Value Register -------- */
#define AT91C_LCDC_MVALUE     ((unsigned int) 0xFF <<  0) /* (LCDC) Toggle Rate Value*/
#define AT91C_LCDC_MMODE      ((unsigned int) 0x1 << 31) /* (LCDC) Toggle Rate Sel*/
#define 	AT91C_LCDC_MMODE_EACHFRAME            ((unsigned int) 0x0 << 31) /* (LCDC) Each Frame*/
#define 	AT91C_LCDC_MMODE_MVALDEFINED          ((unsigned int) 0x1 << 31) /* (LCDC) Defined by MVAL*/
/* -------- LCDC_DP1_2 : (LCDC Offset: 0x81c) Dithering Pattern 1/2 -------- */
#define AT91C_LCDC_DP1_2_FIELD ((unsigned int) 0xFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP4_7 : (LCDC Offset: 0x820) Dithering Pattern 4/7 -------- */
#define AT91C_LCDC_DP4_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP3_5 : (LCDC Offset: 0x824) Dithering Pattern 3/5 -------- */
#define AT91C_LCDC_DP3_5_FIELD ((unsigned int) 0xFFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP2_3 : (LCDC Offset: 0x828) Dithering Pattern 2/3 -------- */
#define AT91C_LCDC_DP2_3_FIELD ((unsigned int) 0xFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP5_7 : (LCDC Offset: 0x82c) Dithering Pattern 5/7 -------- */
#define AT91C_LCDC_DP5_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP3_4 : (LCDC Offset: 0x830) Dithering Pattern 3/4 -------- */
#define AT91C_LCDC_DP3_4_FIELD ((unsigned int) 0xFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP4_5 : (LCDC Offset: 0x834) Dithering Pattern 4/5 -------- */
#define AT91C_LCDC_DP4_5_FIELD ((unsigned int) 0xFFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_DP6_7 : (LCDC Offset: 0x838) Dithering Pattern 6/7 -------- */
#define AT91C_LCDC_DP6_7_FIELD ((unsigned int) 0xFFFFFFF <<  0) /* (LCDC) Ratio*/
/* -------- LCDC_PWRCON : (LCDC Offset: 0x83c) LCDC Power Control Register -------- */
#define AT91C_LCDC_PWR        ((unsigned int) 0x1 <<  0) /* (LCDC) LCD Module Power Control*/
#define AT91C_LCDC_GUARDT     ((unsigned int) 0x7F <<  1) /* (LCDC) Delay in Frame Period*/
#define AT91C_LCDC_BUSY       ((unsigned int) 0x1 << 31) /* (LCDC) Read Only : 1 indicates that LCDC is busy*/
#define 	AT91C_LCDC_BUSY_LCDNOTBUSY           ((unsigned int) 0x0 << 31) /* (LCDC) LCD is Not Busy*/
#define 	AT91C_LCDC_BUSY_LCDBUSY              ((unsigned int) 0x1 << 31) /* (LCDC) LCD is Busy*/
/* -------- LCDC_CTRSTCON : (LCDC Offset: 0x840) LCDC Contrast Control Register -------- */
#define AT91C_LCDC_PS         ((unsigned int) 0x3 <<  0) /* (LCDC) LCD Contrast Counter Prescaler*/
#define 	AT91C_LCDC_PS_NOTDIVIDED           ((unsigned int) 0x0) /* (LCDC) Counter Freq is System Freq.*/
#define 	AT91C_LCDC_PS_DIVIDEDBYTWO         ((unsigned int) 0x1) /* (LCDC) Counter Freq is System Freq divided by 2.*/
#define 	AT91C_LCDC_PS_DIVIDEDBYFOUR        ((unsigned int) 0x2) /* (LCDC) Counter Freq is System Freq divided by 4.*/
#define 	AT91C_LCDC_PS_DIVIDEDBYEIGHT       ((unsigned int) 0x3) /* (LCDC) Counter Freq is System Freq divided by 8.*/
#define AT91C_LCDC_POL        ((unsigned int) 0x1 <<  2) /* (LCDC) Polarity of output Pulse*/
#define 	AT91C_LCDC_POL_NEGATIVEPULSE        ((unsigned int) 0x0 <<  2) /* (LCDC) Negative Pulse*/
#define 	AT91C_LCDC_POL_POSITIVEPULSE        ((unsigned int) 0x1 <<  2) /* (LCDC) Positive Pulse*/
#define AT91C_LCDC_ENA        ((unsigned int) 0x1 <<  3) /* (LCDC) PWM generator Control*/
#define 	AT91C_LCDC_ENA_PWMGEMDISABLED       ((unsigned int) 0x0 <<  3) /* (LCDC) PWM Generator Disabled*/
#define 	AT91C_LCDC_ENA_PWMGEMENABLED        ((unsigned int) 0x1 <<  3) /* (LCDC) PWM Generator Disabled*/
/* -------- LCDC_CTRSTVAL : (LCDC Offset: 0x844) Contrast Value Register -------- */
#define AT91C_LCDC_CVAL       ((unsigned int) 0xFF <<  0) /* (LCDC) PWM Compare Value*/
/* -------- LCDC_IER : (LCDC Offset: 0x848) LCDC Interrupt Enable Register -------- */
#define AT91C_LCDC_LNI        ((unsigned int) 0x1 <<  0) /* (LCDC) Line Interrupt*/
#define AT91C_LCDC_LSTLNI     ((unsigned int) 0x1 <<  1) /* (LCDC) Last Line Interrupt*/
#define AT91C_LCDC_EOFI       ((unsigned int) 0x1 <<  2) /* (LCDC) End Of Frame Interrupt*/
#define AT91C_LCDC_UFLWI      ((unsigned int) 0x1 <<  4) /* (LCDC) FIFO Underflow Interrupt*/
#define AT91C_LCDC_OWRI       ((unsigned int) 0x1 <<  5) /* (LCDC) Over Write Interrupt*/
#define AT91C_LCDC_MERI       ((unsigned int) 0x1 <<  6) /* (LCDC) Memory Error  Interrupt*/
/* -------- LCDC_IDR : (LCDC Offset: 0x84c) LCDC Interrupt Disable Register -------- */
/* -------- LCDC_IMR : (LCDC Offset: 0x850) LCDC Interrupt Mask Register -------- */
/* -------- LCDC_ISR : (LCDC Offset: 0x854) LCDC Interrupt Status Register -------- */
/* -------- LCDC_ICR : (LCDC Offset: 0x858) LCDC Interrupt Clear Register -------- */
/* -------- LCDC_GPR : (LCDC Offset: 0x85c) LCDC General Purpose Register -------- */
#define AT91C_LCDC_GPRBUS     ((unsigned int) 0xFF <<  0) /* (LCDC) 8 bits available*/
/* -------- LCDC_ITR : (LCDC Offset: 0x860) Interrupts Test Register -------- */
/* -------- LCDC_IRR : (LCDC Offset: 0x864) Interrupts Raw Status Register -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR HDMA Channel structure*/
/* ******************************************************************************/
typedef struct _AT91S_HDMA_CH {
	AT91_REG	 HDMA_SADDR; 	/* HDMA Channel Source Address Register*/
	AT91_REG	 HDMA_DADDR; 	/* HDMA Channel Destination Address Register*/
	AT91_REG	 HDMA_DSCR; 	/* HDMA Channel Descriptor Address Register*/
	AT91_REG	 HDMA_CTRLA; 	/* HDMA Channel Control A Register*/
	AT91_REG	 HDMA_CTRLB; 	/* HDMA Channel Control B Register*/
	AT91_REG	 HDMA_CFG; 	/* HDMA Channel Configuration Register*/
	AT91_REG	 HDMA_SPIP; 	/* HDMA Channel Source Picture in Picture Configuration Register*/
	AT91_REG	 HDMA_DPIP; 	/* HDMA Channel Destination Picture in Picture Configuration Register*/
	AT91_REG	 HDMA_BDSCR; 	/* HDMA Reserved*/
	AT91_REG	 HDMA_CADDR; 	/* HDMA Reserved*/
} AT91S_HDMA_CH, *AT91PS_HDMA_CH;

/* -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- */
#define AT91C_SADDR           ((unsigned int) 0x0 <<  0) /* (HDMA_CH) */
/* -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- */
#define AT91C_DADDR           ((unsigned int) 0x0 <<  0) /* (HDMA_CH) */
/* -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- */
#define AT91C_DSCR_IF         ((unsigned int) 0x3 <<  0) /* (HDMA_CH) */
#define AT91C_DSCR            ((unsigned int) 0x3FFFFFFF <<  2) /* (HDMA_CH) */
/* -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- */
#define AT91C_BTSIZE          ((unsigned int) 0xFFF <<  0) /* (HDMA_CH) */
#define AT91C_FC              ((unsigned int) 0x7 << 12) /* (HDMA_CH) */
#define AT91C_AUTO            ((unsigned int) 0x1 << 15) /* (HDMA_CH) */
#define AT91C_SCSIZE          ((unsigned int) 0x7 << 16) /* (HDMA_CH) */
#define AT91C_DCSIZE          ((unsigned int) 0x7 << 20) /* (HDMA_CH) */
#define AT91C_SRC_WIDTH       ((unsigned int) 0x3 << 24) /* (HDMA_CH) */
#define AT91C_DST_WIDTH       ((unsigned int) 0x3 << 28) /* (HDMA_CH) */
/* -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- */
#define AT91C_SIF             ((unsigned int) 0x3 <<  0) /* (HDMA_CH) */
#define AT91C_DIF             ((unsigned int) 0x3 <<  4) /* (HDMA_CH) */
#define AT91C_SRC_PIP         ((unsigned int) 0x1 <<  8) /* (HDMA_CH) */
#define AT91C_DST_PIP         ((unsigned int) 0x1 << 12) /* (HDMA_CH) */
#define AT91C_SRC_DSCR        ((unsigned int) 0x1 << 16) /* (HDMA_CH) */
#define AT91C_DST_DSCR        ((unsigned int) 0x1 << 20) /* (HDMA_CH) */
#define AT91C_SRC_INCR        ((unsigned int) 0x3 << 24) /* (HDMA_CH) */
#define AT91C_DST_INCR        ((unsigned int) 0x3 << 28) /* (HDMA_CH) */
/* -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- */
#define AT91C_SRC_PER         ((unsigned int) 0xF <<  0) /* (HDMA_CH) */
#define AT91C_DST_PER         ((unsigned int) 0xF <<  4) /* (HDMA_CH) */
#define AT91C_SRC_REP         ((unsigned int) 0x1 <<  8) /* (HDMA_CH) */
#define AT91C_SRC_H2SEL       ((unsigned int) 0x1 <<  9) /* (HDMA_CH) */
#define AT91C_DST_REP         ((unsigned int) 0x1 << 12) /* (HDMA_CH) */
#define AT91C_DST_H2SEL       ((unsigned int) 0x1 << 13) /* (HDMA_CH) */
#define AT91C_LOCK_IF         ((unsigned int) 0x1 << 20) /* (HDMA_CH) */
#define AT91C_LOCK_B          ((unsigned int) 0x1 << 21) /* (HDMA_CH) */
#define AT91C_LOCK_IF_L       ((unsigned int) 0x1 << 22) /* (HDMA_CH) */
#define AT91C_AHB_PROT        ((unsigned int) 0x7 << 24) /* (HDMA_CH) */
/* -------- HDMA_SPIP : (HDMA_CH Offset: 0x18)  -------- */
#define AT91C_SPIP_HOLE       ((unsigned int) 0xFFFF <<  0) /* (HDMA_CH) */
#define AT91C_SPIP_BOUNDARY   ((unsigned int) 0x3FF << 16) /* (HDMA_CH) */
/* -------- HDMA_DPIP : (HDMA_CH Offset: 0x1c)  -------- */
#define AT91C_DPIP_HOLE       ((unsigned int) 0xFFFF <<  0) /* (HDMA_CH) */
#define AT91C_DPIP_BOUNDARY   ((unsigned int) 0x3FF << 16) /* (HDMA_CH) */
/* -------- HDMA_BDSCR : (HDMA_CH Offset: 0x20)  -------- */
/* -------- HDMA_CADDR : (HDMA_CH Offset: 0x24)  -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR HDMA controller*/
/* ******************************************************************************/
typedef struct _AT91S_HDMA {
	AT91_REG	 HDMA_GCFG; 	/* HDMA Global Configuration Register*/
	AT91_REG	 HDMA_EN; 	/* HDMA Controller Enable Register*/
	AT91_REG	 HDMA_SREQ; 	/* HDMA Software Single Request Register*/
	AT91_REG	 HDMA_BREQ; 	/* HDMA Software Chunk Transfer Request Register*/
	AT91_REG	 HDMA_LAST; 	/* HDMA Software Last Transfer Flag Register*/
	AT91_REG	 HDMA_SYNC; 	/* HDMA Request Synchronization Register*/
	AT91_REG	 HDMA_EBCIER; 	/* HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register*/
	AT91_REG	 HDMA_EBCIDR; 	/* HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register*/
	AT91_REG	 HDMA_EBCIMR; 	/* HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register*/
	AT91_REG	 HDMA_EBCISR; 	/* HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register*/
	AT91_REG	 HDMA_CHER; 	/* HDMA Channel Handler Enable Register*/
	AT91_REG	 HDMA_CHDR; 	/* HDMA Channel Handler Disable Register*/
	AT91_REG	 HDMA_CHSR; 	/* HDMA Channel Handler Status Register*/
	AT91_REG	 HDMA_RSVD0; 	/* HDMA Reserved*/
	AT91_REG	 HDMA_RSVD1; 	/* HDMA Reserved*/
	AT91S_HDMA_CH	 HDMA_CH[2]; 	/* HDMA Channel structure*/
} AT91S_HDMA, *AT91PS_HDMA;

/* -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- */
#define AT91C_IF0_BIGEND      ((unsigned int) 0x1 <<  0) /* (HDMA) */
#define AT91C_IF1_BIGEND      ((unsigned int) 0x1 <<  1) /* (HDMA) */
#define AT91C_IF2_BIGEND      ((unsigned int) 0x1 <<  2) /* (HDMA) */
#define AT91C_IF3_BIGEND      ((unsigned int) 0x1 <<  3) /* (HDMA) */
#define AT91C_ARB_CFG         ((unsigned int) 0x1 <<  4) /* (HDMA) */
/* -------- HDMA_EN : (HDMA Offset: 0x4)  -------- */
#define AT91C_HDMA_ENABLE     ((unsigned int) 0x1 <<  0) /* (HDMA) */
/* -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- */
#define AT91C_SOFT_SREQ       ((unsigned int) 0xFFFF <<  0) /* (HDMA) */
/* -------- HDMA_BREQ : (HDMA Offset: 0xc)  -------- */
#define AT91C_SOFT_BREQ       ((unsigned int) 0xFFFF <<  0) /* (HDMA) */
/* -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- */
#define AT91C_SOFT_LAST       ((unsigned int) 0xFFFF <<  0) /* (HDMA) */
/* -------- HDMA_SYNC : (HDMA Offset: 0x14)  -------- */
#define AT91C_SYNC_REQ        ((unsigned int) 0xFFFF <<  0) /* (HDMA) */
/* -------- HDMA_EBCIER : (HDMA Offset: 0x18)  -------- */
#define AT91C_BTC             ((unsigned int) 0xFF <<  0) /* (HDMA) */
#define AT91C_CBTC            ((unsigned int) 0xFF <<  8) /* (HDMA) */
#define AT91C_ERR             ((unsigned int) 0xFF << 16) /* (HDMA) */
/* -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- */
/* -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- */
/* -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- */
/* -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- */
#define AT91C_ENABLE          ((unsigned int) 0xFF <<  0) /* (HDMA) */
#define AT91C_SUSPEND         ((unsigned int) 0xFF <<  8) /* (HDMA) */
#define AT91C_KEEPON          ((unsigned int) 0xFF << 24) /* (HDMA) */
/* -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- */
#define AT91C_RESUME          ((unsigned int) 0xFF <<  8) /* (HDMA) */
/* -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- */
#define AT91C_STALLED         ((unsigned int) 0xFF << 14) /* (HDMA) */
#define AT91C_EMPTY           ((unsigned int) 0xFF << 16) /* (HDMA) */
/* -------- HDMA_RSVD : (HDMA Offset: 0x34)  -------- */
/* -------- HDMA_RSVD : (HDMA Offset: 0x38)  -------- */

/* ******************************************************************************/
/*              SOFTWARE API DEFINITION  FOR Error Correction Code controller*/
/* ******************************************************************************/
typedef struct _AT91S_ECC {
	AT91_REG	 ECC_CR; 	/*  ECC reset register*/
	AT91_REG	 ECC_MR; 	/*  ECC Page size register*/
	AT91_REG	 ECC_SR; 	/*  ECC Status register*/
	AT91_REG	 ECC_PR; 	/*  ECC Parity register*/
	AT91_REG	 ECC_NPR; 	/*  ECC Parity N register*/
	AT91_REG	 Reserved0[58]; 	/* */
	AT91_REG	 ECC_VR; 	/*  ECC Version register*/
} AT91S_ECC, *AT91PS_ECC;

/* -------- ECC_CR : (ECC Offset: 0x0) ECC reset register -------- */
#define AT91C_ECC_RST         ((unsigned int) 0x1 <<  0) /* (ECC) ECC reset parity*/
/* -------- ECC_MR : (ECC Offset: 0x4) ECC page size register -------- */
#define AT91C_ECC_PAGE_SIZE   ((unsigned int) 0x3 <<  0) /* (ECC) Nand Flash page size*/
/* -------- ECC_SR : (ECC Offset: 0x8) ECC status register -------- */
#define AT91C_ECC_RECERR      ((unsigned int) 0x1 <<  0) /* (ECC) ECC error*/
#define AT91C_ECC_ECCERR      ((unsigned int) 0x1 <<  1) /* (ECC) ECC single error*/
#define AT91C_ECC_MULERR      ((unsigned int) 0x1 <<  2) /* (ECC) ECC_MULERR*/
/* -------- ECC_PR : (ECC Offset: 0xc) ECC parity register -------- */
#define AT91C_ECC_BITADDR     ((unsigned int) 0xF <<  0) /* (ECC) Bit address error*/
#define AT91C_ECC_WORDADDR    ((unsigned int) 0xFFF <<  4) /* (ECC) address of the failing bit*/
/* -------- ECC_NPR : (ECC Offset: 0x10) ECC N parity register -------- */
#define AT91C_ECC_NPARITY     ((unsigned int) 0xFFFF <<  0) /* (ECC) ECC parity N */
/* -------- ECC_VR : (ECC Offset: 0xfc) ECC version register -------- */
#define AT91C_ECC_VR          ((unsigned int) 0xF <<  0) /* (ECC) ECC version register*/

/* ******************************************************************************/
/*               REGISTER ADDRESS DEFINITION FOR AT91SAM9RL64*/
/* ******************************************************************************/
/* ========== Register definition for SYS peripheral ========== */
#define AT91C_SYS_SLCKSEL ((AT91_REG *) 	0xFFFFFD50) /* (SYS) Slow Clock Selection Register*/
#define AT91C_SYS_GPBR  ((AT91_REG *) 	0xFFFFFD60) /* (SYS) General Purpose Register*/
/* ========== Register definition for EBI peripheral ========== */
#define AT91C_EBI_DUMMY ((AT91_REG *) 	0xFFFFE800) /* (EBI) Dummy register - Do not use*/
/* ========== Register definition for SDRAMC peripheral ========== */
#define AT91C_SDRAMC_MR ((AT91_REG *) 	0xFFFFEA00) /* (SDRAMC) SDRAM Controller Mode Register*/
#define AT91C_SDRAMC_IMR ((AT91_REG *) 	0xFFFFEA1C) /* (SDRAMC) SDRAM Controller Interrupt Mask Register*/
#define AT91C_SDRAMC_LPR ((AT91_REG *) 	0xFFFFEA10) /* (SDRAMC) SDRAM Controller Low Power Register*/
#define AT91C_SDRAMC_ISR ((AT91_REG *) 	0xFFFFEA20) /* (SDRAMC) SDRAM Controller Interrupt Mask Register*/
#define AT91C_SDRAMC_IDR ((AT91_REG *) 	0xFFFFEA18) /* (SDRAMC) SDRAM Controller Interrupt Disable Register*/
#define AT91C_SDRAMC_CR ((AT91_REG *) 	0xFFFFEA08) /* (SDRAMC) SDRAM Controller Configuration Register*/
#define AT91C_SDRAMC_TR ((AT91_REG *) 	0xFFFFEA04) /* (SDRAMC) SDRAM Controller Refresh Timer Register*/
#define AT91C_SDRAMC_MDR ((AT91_REG *) 	0xFFFFEA24) /* (SDRAMC) SDRAM Memory Device Register*/
#define AT91C_SDRAMC_HSR ((AT91_REG *) 	0xFFFFEA0C) /* (SDRAMC) SDRAM Controller High Speed Register*/
#define AT91C_SDRAMC_IER ((AT91_REG *) 	0xFFFFEA14) /* (SDRAMC) SDRAM Controller Interrupt Enable Register*/
/* ========== Register definition for SMC peripheral ========== */
#define AT91C_SMC_CTRL1 ((AT91_REG *) 	0xFFFFEC1C) /* (SMC)  Control Register for CS 1*/
#define AT91C_SMC_PULSE7 ((AT91_REG *) 	0xFFFFEC74) /* (SMC)  Pulse Register for CS 7*/
#define AT91C_SMC_PULSE6 ((AT91_REG *) 	0xFFFFEC64) /* (SMC)  Pulse Register for CS 6*/
#define AT91C_SMC_SETUP4 ((AT91_REG *) 	0xFFFFEC40) /* (SMC)  Setup Register for CS 4*/
#define AT91C_SMC_PULSE3 ((AT91_REG *) 	0xFFFFEC34) /* (SMC)  Pulse Register for CS 3*/
#define AT91C_SMC_CYCLE5 ((AT91_REG *) 	0xFFFFEC58) /* (SMC)  Cycle Register for CS 5*/
#define AT91C_SMC_CYCLE2 ((AT91_REG *) 	0xFFFFEC28) /* (SMC)  Cycle Register for CS 2*/
#define AT91C_SMC_CTRL2 ((AT91_REG *) 	0xFFFFEC2C) /* (SMC)  Control Register for CS 2*/
#define AT91C_SMC_CTRL0 ((AT91_REG *) 	0xFFFFEC0C) /* (SMC)  Control Register for CS 0*/
#define AT91C_SMC_PULSE5 ((AT91_REG *) 	0xFFFFEC54) /* (SMC)  Pulse Register for CS 5*/
#define AT91C_SMC_PULSE1 ((AT91_REG *) 	0xFFFFEC14) /* (SMC)  Pulse Register for CS 1*/
#define AT91C_SMC_PULSE0 ((AT91_REG *) 	0xFFFFEC04) /* (SMC)  Pulse Register for CS 0*/
#define AT91C_SMC_CYCLE7 ((AT91_REG *) 	0xFFFFEC78) /* (SMC)  Cycle Register for CS 7*/
#define AT91C_SMC_CTRL4 ((AT91_REG *) 	0xFFFFEC4C) /* (SMC)  Control Register for CS 4*/
#define AT91C_SMC_CTRL3 ((AT91_REG *) 	0xFFFFEC3C) /* (SMC)  Control Register for CS 3*/
#define AT91C_SMC_SETUP7 ((AT91_REG *) 	0xFFFFEC70) /* (SMC)  Setup Register for CS 7*/
#define AT91C_SMC_CTRL7 ((AT91_REG *) 	0xFFFFEC7C) /* (SMC)  Control Register for CS 7*/
#define AT91C_SMC_SETUP1 ((AT91_REG *) 	0xFFFFEC10) /* (SMC)  Setup Register for CS 1*/
#define AT91C_SMC_CYCLE0 ((AT91_REG *) 	0xFFFFEC08) /* (SMC)  Cycle Register for CS 0*/
#define AT91C_SMC_CTRL5 ((AT91_REG *) 	0xFFFFEC5C) /* (SMC)  Control Register for CS 5*/
#define AT91C_SMC_CYCLE1 ((AT91_REG *) 	0xFFFFEC18) /* (SMC)  Cycle Register for CS 1*/
#define AT91C_SMC_CTRL6 ((AT91_REG *) 	0xFFFFEC6C) /* (SMC)  Control Register for CS 6*/
#define AT91C_SMC_SETUP0 ((AT91_REG *) 	0xFFFFEC00) /* (SMC)  Setup Register for CS 0*/
#define AT91C_SMC_PULSE4 ((AT91_REG *) 	0xFFFFEC44) /* (SMC)  Pulse Register for CS 4*/
#define AT91C_SMC_SETUP5 ((AT91_REG *) 	0xFFFFEC50) /* (SMC)  Setup Register for CS 5*/
#define AT91C_SMC_SETUP2 ((AT91_REG *) 	0xFFFFEC20) /* (SMC)  Setup Register for CS 2*/
#define AT91C_SMC_CYCLE3 ((AT91_REG *) 	0xFFFFEC38) /* (SMC)  Cycle Register for CS 3*/
#define AT91C_SMC_CYCLE6 ((AT91_REG *) 	0xFFFFEC68) /* (SMC)  Cycle Register for CS 6*/
#define AT91C_SMC_SETUP6 ((AT91_REG *) 	0xFFFFEC60) /* (SMC)  Setup Register for CS 6*/
#define AT91C_SMC_CYCLE4 ((AT91_REG *) 	0xFFFFEC48) /* (SMC)  Cycle Register for CS 4*/
#define AT91C_SMC_PULSE2 ((AT91_REG *) 	0xFFFFEC24) /* (SMC)  Pulse Register for CS 2*/
#define AT91C_SMC_SETUP3 ((AT91_REG *) 	0xFFFFEC30) /* (SMC)  Setup Register for CS 3*/
/* ========== Register definition for MATRIX peripheral ========== */
#define AT91C_MATRIX_PRBS4 ((AT91_REG *) 	0xFFFFEEA4) /* (MATRIX)  PRBS4 : ebi*/
#define AT91C_MATRIX_SCFG3 ((AT91_REG *) 	0xFFFFEE4C) /* (MATRIX)  Slave Configuration Register 3 : usb_dev_hs*/
#define AT91C_MATRIX_MCFG6 ((AT91_REG *) 	0xFFFFEE18) /* (MATRIX)  Master Configuration Register 6 */
#define AT91C_MATRIX_PRAS3 ((AT91_REG *) 	0xFFFFEE98) /* (MATRIX)  PRAS3 : usb_dev_hs*/
#define AT91C_MATRIX_PRBS7 ((AT91_REG *) 	0xFFFFEEBC) /* (MATRIX)  PRBS7*/
#define AT91C_MATRIX_PRAS5 ((AT91_REG *) 	0xFFFFEEA8) /* (MATRIX)  PRAS5 : bridge*/
#define AT91C_MATRIX_PRAS0 ((AT91_REG *) 	0xFFFFEE80) /* (MATRIX)  PRAS0 : rom*/
#define AT91C_MATRIX_PRAS2 ((AT91_REG *) 	0xFFFFEE90) /* (MATRIX)  PRAS2 : lcdc*/
#define AT91C_MATRIX_MCFG5 ((AT91_REG *) 	0xFFFFEE14) /* (MATRIX)  Master Configuration Register 5 : bridge*/
#define AT91C_MATRIX_MCFG1 ((AT91_REG *) 	0xFFFFEE04) /* (MATRIX)  Master Configuration Register 1 ; htcm*/
#define AT91C_MATRIX_MRCR ((AT91_REG *) 	0xFFFFEF00) /* (MATRIX)  Master Remp Control Register */
#define AT91C_MATRIX_PRBS2 ((AT91_REG *) 	0xFFFFEE94) /* (MATRIX)  PRBS2 : lcdc*/
#define AT91C_MATRIX_SCFG4 ((AT91_REG *) 	0xFFFFEE50) /* (MATRIX)  Slave Configuration Register 4 ; ebi*/
#define AT91C_MATRIX_SCFG6 ((AT91_REG *) 	0xFFFFEE58) /* (MATRIX)  Slave Configuration Register 6*/
#define AT91C_MATRIX_MCFG0 ((AT91_REG *) 	0xFFFFEE00) /* (MATRIX)  Master Configuration Register 0 : rom*/
#define AT91C_MATRIX_MCFG8 ((AT91_REG *) 	0xFFFFEE20) /* (MATRIX)  Master Configuration Register 8 */
#define AT91C_MATRIX_MCFG7 ((AT91_REG *) 	0xFFFFEE1C) /* (MATRIX)  Master Configuration Register 7 */
#define AT91C_MATRIX_MCFG4 ((AT91_REG *) 	0xFFFFEE10) /* (MATRIX)  Master Configuration Register 4 : ebi*/
#define AT91C_MATRIX_SCFG1 ((AT91_REG *) 	0xFFFFEE44) /* (MATRIX)  Slave Configuration Register 1 : htcm*/
#define AT91C_MATRIX_SCFG7 ((AT91_REG *) 	0xFFFFEE5C) /* (MATRIX)  Slave Configuration Register 7*/
#define AT91C_MATRIX_MCFG2 ((AT91_REG *) 	0xFFFFEE08) /* (MATRIX)  Master Configuration Register 2 : lcdc*/
#define AT91C_MATRIX_PRBS0 ((AT91_REG *) 	0xFFFFEE84) /* (MATRIX)  PRBS0 : rom*/
#define AT91C_MATRIX_PRAS7 ((AT91_REG *) 	0xFFFFEEB8) /* (MATRIX)  PRAS7*/
#define AT91C_MATRIX_SCFG0 ((AT91_REG *) 	0xFFFFEE40) /* (MATRIX)  Slave Configuration Register 0 : rom*/
#define AT91C_MATRIX_PRBS5 ((AT91_REG *) 	0xFFFFEEAC) /* (MATRIX)  PRBS5 : bridge*/
#define AT91C_MATRIX_PRBS3 ((AT91_REG *) 	0xFFFFEE9C) /* (MATRIX)  PRBS3 : usb_dev_hs*/
#define AT91C_MATRIX_PRAS6 ((AT91_REG *) 	0xFFFFEEB0) /* (MATRIX)  PRAS6*/
#define AT91C_MATRIX_MCFG3 ((AT91_REG *) 	0xFFFFEE0C) /* (MATRIX)  Master Configuration Register 3 : usb_dev_hs*/
#define AT91C_MATRIX_PRAS1 ((AT91_REG *) 	0xFFFFEE88) /* (MATRIX)  PRAS1 : htcm*/
#define AT91C_MATRIX_SCFG2 ((AT91_REG *) 	0xFFFFEE48) /* (MATRIX)  Slave Configuration Register 2 : lcdc*/
#define AT91C_MATRIX_PRBS6 ((AT91_REG *) 	0xFFFFEEB4) /* (MATRIX)  PRBS6*/
#define AT91C_MATRIX_PRAS4 ((AT91_REG *) 	0xFFFFEEA0) /* (MATRIX)  PRAS4 : ebi*/
#define AT91C_MATRIX_SCFG5 ((AT91_REG *) 	0xFFFFEE54) /* (MATRIX)  Slave Configuration Register 5 : bridge*/
#define AT91C_MATRIX_PRBS1 ((AT91_REG *) 	0xFFFFEE8C) /* (MATRIX)  PRBS1 : htcm*/
/* ========== Register definition for CCFG peripheral ========== */
#define AT91C_CCFG_MATRIXVERSION ((AT91_REG *) 	0xFFFFEFFC) /* (CCFG)  Version Register*/
#define AT91C_CCFG_TCMR ((AT91_REG *) 	0xFFFFEF14) /* (CCFG)  TCM configuration*/
#define AT91C_CCFG_EBICSA ((AT91_REG *) 	0xFFFFEF20) /* (CCFG)  EBI Chip Select Assignement Register*/
#define AT91C_CCFG_UDPHS ((AT91_REG *) 	0xFFFFEF1C) /* (CCFG)  USB Device HS configuration*/
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
/* ========== Register definition for PIOD peripheral ========== */
#define AT91C_PIOD_ODSR ((AT91_REG *) 	0xFFFFFA38) /* (PIOD) Output Data Status Register*/
#define AT91C_PIOD_ABSR ((AT91_REG *) 	0xFFFFFA78) /* (PIOD) AB Select Status Register*/
#define AT91C_PIOD_PSR  ((AT91_REG *) 	0xFFFFFA08) /* (PIOD) PIO Status Register*/
#define AT91C_PIOD_PPUDR ((AT91_REG *) 	0xFFFFFA60) /* (PIOD) Pull-up Disable Register*/
#define AT91C_PIOD_OER  ((AT91_REG *) 	0xFFFFFA10) /* (PIOD) Output Enable Register*/
#define AT91C_PIOD_OWDR ((AT91_REG *) 	0xFFFFFAA4) /* (PIOD) Output Write Disable Register*/
#define AT91C_PIOD_PER  ((AT91_REG *) 	0xFFFFFA00) /* (PIOD) PIO Enable Register*/
#define AT91C_PIOD_IFSR ((AT91_REG *) 	0xFFFFFA28) /* (PIOD) Input Filter Status Register*/
#define AT91C_PIOD_IFER ((AT91_REG *) 	0xFFFFFA20) /* (PIOD) Input Filter Enable Register*/
#define AT91C_PIOD_ODR  ((AT91_REG *) 	0xFFFFFA14) /* (PIOD) Output Disable Registerr*/
#define AT91C_PIOD_PPUSR ((AT91_REG *) 	0xFFFFFA68) /* (PIOD) Pull-up Status Register*/
#define AT91C_PIOD_IFDR ((AT91_REG *) 	0xFFFFFA24) /* (PIOD) Input Filter Disable Register*/
#define AT91C_PIOD_PDSR ((AT91_REG *) 	0xFFFFFA3C) /* (PIOD) Pin Data Status Register*/
#define AT91C_PIOD_PPUER ((AT91_REG *) 	0xFFFFFA64) /* (PIOD) Pull-up Enable Register*/
#define AT91C_PIOD_IDR  ((AT91_REG *) 	0xFFFFFA44) /* (PIOD) Interrupt Disable Register*/
#define AT91C_PIOD_MDDR ((AT91_REG *) 	0xFFFFFA54) /* (PIOD) Multi-driver Disable Register*/
#define AT91C_PIOD_ISR  ((AT91_REG *) 	0xFFFFFA4C) /* (PIOD) Interrupt Status Register*/
#define AT91C_PIOD_OSR  ((AT91_REG *) 	0xFFFFFA18) /* (PIOD) Output Status Register*/
#define AT91C_PIOD_CODR ((AT91_REG *) 	0xFFFFFA34) /* (PIOD) Clear Output Data Register*/
#define AT91C_PIOD_MDSR ((AT91_REG *) 	0xFFFFFA58) /* (PIOD) Multi-driver Status Register*/
#define AT91C_PIOD_PDR  ((AT91_REG *) 	0xFFFFFA04) /* (PIOD) PIO Disable Register*/
#define AT91C_PIOD_IER  ((AT91_REG *) 	0xFFFFFA40) /* (PIOD) Interrupt Enable Register*/
#define AT91C_PIOD_OWSR ((AT91_REG *) 	0xFFFFFAA8) /* (PIOD) Output Write Status Register*/
#define AT91C_PIOD_BSR  ((AT91_REG *) 	0xFFFFFA74) /* (PIOD) Select B Register*/
#define AT91C_PIOD_ASR  ((AT91_REG *) 	0xFFFFFA70) /* (PIOD) Select A Register*/
#define AT91C_PIOD_SODR ((AT91_REG *) 	0xFFFFFA30) /* (PIOD) Set Output Data Register*/
#define AT91C_PIOD_IMR  ((AT91_REG *) 	0xFFFFFA48) /* (PIOD) Interrupt Mask Register*/
#define AT91C_PIOD_OWER ((AT91_REG *) 	0xFFFFFAA0) /* (PIOD) Output Write Enable Register*/
#define AT91C_PIOD_MDER ((AT91_REG *) 	0xFFFFFA50) /* (PIOD) Multi-driver Enable Register*/
/* ========== Register definition for PMC peripheral ========== */
#define AT91C_PMC_PCER  ((AT91_REG *) 	0xFFFFFC10) /* (PMC) Peripheral Clock Enable Register*/
#define AT91C_PMC_PCKR  ((AT91_REG *) 	0xFFFFFC40) /* (PMC) Programmable Clock Register*/
#define AT91C_PMC_MCKR  ((AT91_REG *) 	0xFFFFFC30) /* (PMC) Master Clock Register*/
#define AT91C_PMC_PLLAR ((AT91_REG *) 	0xFFFFFC28) /* (PMC) PLL A Register*/
#define AT91C_PMC_PCDR  ((AT91_REG *) 	0xFFFFFC14) /* (PMC) Peripheral Clock Disable Register*/
#define AT91C_PMC_SCSR  ((AT91_REG *) 	0xFFFFFC08) /* (PMC) System Clock Status Register*/
#define AT91C_PMC_MCFR  ((AT91_REG *) 	0xFFFFFC24) /* (PMC) Main Clock  Frequency Register*/
#define AT91C_PMC_IMR   ((AT91_REG *) 	0xFFFFFC6C) /* (PMC) Interrupt Mask Register*/
#define AT91C_PMC_IER   ((AT91_REG *) 	0xFFFFFC60) /* (PMC) Interrupt Enable Register*/
#define AT91C_PMC_UCKR  ((AT91_REG *) 	0xFFFFFC1C) /* (PMC) UTMI Clock Configuration Register*/
#define AT91C_PMC_MOR   ((AT91_REG *) 	0xFFFFFC20) /* (PMC) Main Oscillator Register*/
#define AT91C_PMC_IDR   ((AT91_REG *) 	0xFFFFFC64) /* (PMC) Interrupt Disable Register*/
#define AT91C_PMC_PLLBR ((AT91_REG *) 	0xFFFFFC2C) /* (PMC) PLL B Register*/
#define AT91C_PMC_SCDR  ((AT91_REG *) 	0xFFFFFC04) /* (PMC) System Clock Disable Register*/
#define AT91C_PMC_PCSR  ((AT91_REG *) 	0xFFFFFC18) /* (PMC) Peripheral Clock Status Register*/
#define AT91C_PMC_SCER  ((AT91_REG *) 	0xFFFFFC00) /* (PMC) System Clock Enable Register*/
#define AT91C_PMC_SR    ((AT91_REG *) 	0xFFFFFC68) /* (PMC) Status Register*/
/* ========== Register definition for CKGR peripheral ========== */
#define AT91C_CKGR_MOR  ((AT91_REG *) 	0xFFFFFC20) /* (CKGR) Main Oscillator Register*/
#define AT91C_CKGR_PLLBR ((AT91_REG *) 	0xFFFFFC2C) /* (CKGR) PLL B Register*/
#define AT91C_CKGR_MCFR ((AT91_REG *) 	0xFFFFFC24) /* (CKGR) Main Clock  Frequency Register*/
#define AT91C_CKGR_PLLAR ((AT91_REG *) 	0xFFFFFC28) /* (CKGR) PLL A Register*/
#define AT91C_CKGR_UCKR ((AT91_REG *) 	0xFFFFFC1C) /* (CKGR) UTMI Clock Configuration Register*/
/* ========== Register definition for RSTC peripheral ========== */
#define AT91C_RSTC_RCR  ((AT91_REG *) 	0xFFFFFD00) /* (RSTC) Reset Control Register*/
#define AT91C_RSTC_RMR  ((AT91_REG *) 	0xFFFFFD08) /* (RSTC) Reset Mode Register*/
#define AT91C_RSTC_RSR  ((AT91_REG *) 	0xFFFFFD04) /* (RSTC) Reset Status Register*/
/* ========== Register definition for SHDWC peripheral ========== */
#define AT91C_SHDWC_SHSR ((AT91_REG *) 	0xFFFFFD18) /* (SHDWC) Shut Down Status Register*/
#define AT91C_SHDWC_SHMR ((AT91_REG *) 	0xFFFFFD14) /* (SHDWC) Shut Down Mode Register*/
#define AT91C_SHDWC_SHCR ((AT91_REG *) 	0xFFFFFD10) /* (SHDWC) Shut Down Control Register*/
/* ========== Register definition for RTTC peripheral ========== */
#define AT91C_RTTC_RTSR ((AT91_REG *) 	0xFFFFFD2C) /* (RTTC) Real-time Status Register*/
#define AT91C_RTTC_RTMR ((AT91_REG *) 	0xFFFFFD20) /* (RTTC) Real-time Mode Register*/
#define AT91C_RTTC_RTVR ((AT91_REG *) 	0xFFFFFD28) /* (RTTC) Real-time Value Register*/
#define AT91C_RTTC_RTAR ((AT91_REG *) 	0xFFFFFD24) /* (RTTC) Real-time Alarm Register*/
/* ========== Register definition for PITC peripheral ========== */
#define AT91C_PITC_PIVR ((AT91_REG *) 	0xFFFFFD38) /* (PITC) Period Interval Value Register*/
#define AT91C_PITC_PISR ((AT91_REG *) 	0xFFFFFD34) /* (PITC) Period Interval Status Register*/
#define AT91C_PITC_PIIR ((AT91_REG *) 	0xFFFFFD3C) /* (PITC) Period Interval Image Register*/
#define AT91C_PITC_PIMR ((AT91_REG *) 	0xFFFFFD30) /* (PITC) Period Interval Mode Register*/
/* ========== Register definition for WDTC peripheral ========== */
#define AT91C_WDTC_WDCR ((AT91_REG *) 	0xFFFFFD40) /* (WDTC) Watchdog Control Register*/
#define AT91C_WDTC_WDSR ((AT91_REG *) 	0xFFFFFD48) /* (WDTC) Watchdog Status Register*/
#define AT91C_WDTC_WDMR ((AT91_REG *) 	0xFFFFFD44) /* (WDTC) Watchdog Mode Register*/
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
/* ========== Register definition for TCB0 peripheral ========== */
#define AT91C_TCB0_BMR  ((AT91_REG *) 	0xFFFA00C4) /* (TCB0) TC Block Mode Register*/
#define AT91C_TCB0_BCR  ((AT91_REG *) 	0xFFFA00C0) /* (TCB0) TC Block Control Register*/
/* ========== Register definition for TCB1 peripheral ========== */
#define AT91C_TCB1_BMR  ((AT91_REG *) 	0xFFFA0104) /* (TCB1) TC Block Mode Register*/
#define AT91C_TCB1_BCR  ((AT91_REG *) 	0xFFFA0100) /* (TCB1) TC Block Control Register*/
/* ========== Register definition for TCB2 peripheral ========== */
#define AT91C_TCB2_BCR  ((AT91_REG *) 	0xFFFA0140) /* (TCB2) TC Block Control Register*/
#define AT91C_TCB2_BMR  ((AT91_REG *) 	0xFFFA0144) /* (TCB2) TC Block Mode Register*/
/* ========== Register definition for PDC_MCI peripheral ========== */
#define AT91C_MCI_TPR   ((AT91_REG *) 	0xFFFA4108) /* (PDC_MCI) Transmit Pointer Register*/
#define AT91C_MCI_PTCR  ((AT91_REG *) 	0xFFFA4120) /* (PDC_MCI) PDC Transfer Control Register*/
#define AT91C_MCI_RNPR  ((AT91_REG *) 	0xFFFA4110) /* (PDC_MCI) Receive Next Pointer Register*/
#define AT91C_MCI_TNCR  ((AT91_REG *) 	0xFFFA411C) /* (PDC_MCI) Transmit Next Counter Register*/
#define AT91C_MCI_TCR   ((AT91_REG *) 	0xFFFA410C) /* (PDC_MCI) Transmit Counter Register*/
#define AT91C_MCI_RCR   ((AT91_REG *) 	0xFFFA4104) /* (PDC_MCI) Receive Counter Register*/
#define AT91C_MCI_RNCR  ((AT91_REG *) 	0xFFFA4114) /* (PDC_MCI) Receive Next Counter Register*/
#define AT91C_MCI_TNPR  ((AT91_REG *) 	0xFFFA4118) /* (PDC_MCI) Transmit Next Pointer Register*/
#define AT91C_MCI_RPR   ((AT91_REG *) 	0xFFFA4100) /* (PDC_MCI) Receive Pointer Register*/
#define AT91C_MCI_PTSR  ((AT91_REG *) 	0xFFFA4124) /* (PDC_MCI) PDC Transfer Status Register*/
/* ========== Register definition for MCI peripheral ========== */
#define AT91C_MCI_IDR   ((AT91_REG *) 	0xFFFA4048) /* (MCI) MCI Interrupt Disable Register*/
#define AT91C_MCI_MR    ((AT91_REG *) 	0xFFFA4004) /* (MCI) MCI Mode Register*/
#define AT91C_MCI_VR    ((AT91_REG *) 	0xFFFA40FC) /* (MCI) MCI Version Register*/
#define AT91C_MCI_IER   ((AT91_REG *) 	0xFFFA4044) /* (MCI) MCI Interrupt Enable Register*/
#define AT91C_MCI_IMR   ((AT91_REG *) 	0xFFFA404C) /* (MCI) MCI Interrupt Mask Register*/
#define AT91C_MCI_SR    ((AT91_REG *) 	0xFFFA4040) /* (MCI) MCI Status Register*/
#define AT91C_MCI_DTOR  ((AT91_REG *) 	0xFFFA4008) /* (MCI) MCI Data Timeout Register*/
#define AT91C_MCI_CR    ((AT91_REG *) 	0xFFFA4000) /* (MCI) MCI Control Register*/
#define AT91C_MCI_CMDR  ((AT91_REG *) 	0xFFFA4014) /* (MCI) MCI Command Register*/
#define AT91C_MCI_SDCR  ((AT91_REG *) 	0xFFFA400C) /* (MCI) MCI SD Card Register*/
#define AT91C_MCI_BLKR  ((AT91_REG *) 	0xFFFA4018) /* (MCI) MCI Block Register*/
#define AT91C_MCI_RDR   ((AT91_REG *) 	0xFFFA4030) /* (MCI) MCI Receive Data Register*/
#define AT91C_MCI_ARGR  ((AT91_REG *) 	0xFFFA4010) /* (MCI) MCI Argument Register*/
#define AT91C_MCI_TDR   ((AT91_REG *) 	0xFFFA4034) /* (MCI) MCI Transmit Data Register*/
#define AT91C_MCI_RSPR  ((AT91_REG *) 	0xFFFA4020) /* (MCI) MCI Response Register*/
/* ========== Register definition for PDC_TWI0 peripheral ========== */
#define AT91C_TWI0_RNCR ((AT91_REG *) 	0xFFFA8114) /* (PDC_TWI0) Receive Next Counter Register*/
#define AT91C_TWI0_TCR  ((AT91_REG *) 	0xFFFA810C) /* (PDC_TWI0) Transmit Counter Register*/
#define AT91C_TWI0_RCR  ((AT91_REG *) 	0xFFFA8104) /* (PDC_TWI0) Receive Counter Register*/
#define AT91C_TWI0_TNPR ((AT91_REG *) 	0xFFFA8118) /* (PDC_TWI0) Transmit Next Pointer Register*/
#define AT91C_TWI0_RNPR ((AT91_REG *) 	0xFFFA8110) /* (PDC_TWI0) Receive Next Pointer Register*/
#define AT91C_TWI0_RPR  ((AT91_REG *) 	0xFFFA8100) /* (PDC_TWI0) Receive Pointer Register*/
#define AT91C_TWI0_TNCR ((AT91_REG *) 	0xFFFA811C) /* (PDC_TWI0) Transmit Next Counter Register*/
#define AT91C_TWI0_TPR  ((AT91_REG *) 	0xFFFA8108) /* (PDC_TWI0) Transmit Pointer Register*/
#define AT91C_TWI0_PTSR ((AT91_REG *) 	0xFFFA8124) /* (PDC_TWI0) PDC Transfer Status Register*/
#define AT91C_TWI0_PTCR ((AT91_REG *) 	0xFFFA8120) /* (PDC_TWI0) PDC Transfer Control Register*/
/* ========== Register definition for TWI0 peripheral ========== */
#define AT91C_TWI0_IDR  ((AT91_REG *) 	0xFFFA8028) /* (TWI0) Interrupt Disable Register*/
#define AT91C_TWI0_RHR  ((AT91_REG *) 	0xFFFA8030) /* (TWI0) Receive Holding Register*/
#define AT91C_TWI0_SMR  ((AT91_REG *) 	0xFFFA8008) /* (TWI0) Slave Mode Register*/
#define AT91C_TWI0_IER  ((AT91_REG *) 	0xFFFA8024) /* (TWI0) Interrupt Enable Register*/
#define AT91C_TWI0_THR  ((AT91_REG *) 	0xFFFA8034) /* (TWI0) Transmit Holding Register*/
#define AT91C_TWI0_MMR  ((AT91_REG *) 	0xFFFA8004) /* (TWI0) Master Mode Register*/
#define AT91C_TWI0_CR   ((AT91_REG *) 	0xFFFA8000) /* (TWI0) Control Register*/
#define AT91C_TWI0_CWGR ((AT91_REG *) 	0xFFFA8010) /* (TWI0) Clock Waveform Generator Register*/
#define AT91C_TWI0_IADR ((AT91_REG *) 	0xFFFA800C) /* (TWI0) Internal Address Register*/
#define AT91C_TWI0_IMR  ((AT91_REG *) 	0xFFFA802C) /* (TWI0) Interrupt Mask Register*/
#define AT91C_TWI0_SR   ((AT91_REG *) 	0xFFFA8020) /* (TWI0) Status Register*/
/* ========== Register definition for TWI1 peripheral ========== */
#define AT91C_TWI1_THR  ((AT91_REG *) 	0xFFFAC034) /* (TWI1) Transmit Holding Register*/
#define AT91C_TWI1_IDR  ((AT91_REG *) 	0xFFFAC028) /* (TWI1) Interrupt Disable Register*/
#define AT91C_TWI1_SMR  ((AT91_REG *) 	0xFFFAC008) /* (TWI1) Slave Mode Register*/
#define AT91C_TWI1_CWGR ((AT91_REG *) 	0xFFFAC010) /* (TWI1) Clock Waveform Generator Register*/
#define AT91C_TWI1_IADR ((AT91_REG *) 	0xFFFAC00C) /* (TWI1) Internal Address Register*/
#define AT91C_TWI1_RHR  ((AT91_REG *) 	0xFFFAC030) /* (TWI1) Receive Holding Register*/
#define AT91C_TWI1_IER  ((AT91_REG *) 	0xFFFAC024) /* (TWI1) Interrupt Enable Register*/
#define AT91C_TWI1_MMR  ((AT91_REG *) 	0xFFFAC004) /* (TWI1) Master Mode Register*/
#define AT91C_TWI1_SR   ((AT91_REG *) 	0xFFFAC020) /* (TWI1) Status Register*/
#define AT91C_TWI1_IMR  ((AT91_REG *) 	0xFFFAC02C) /* (TWI1) Interrupt Mask Register*/
#define AT91C_TWI1_CR   ((AT91_REG *) 	0xFFFAC000) /* (TWI1) Control Register*/
/* ========== Register definition for PDC_US0 peripheral ========== */
#define AT91C_US0_TCR   ((AT91_REG *) 	0xFFFB010C) /* (PDC_US0) Transmit Counter Register*/
#define AT91C_US0_PTCR  ((AT91_REG *) 	0xFFFB0120) /* (PDC_US0) PDC Transfer Control Register*/
#define AT91C_US0_RNCR  ((AT91_REG *) 	0xFFFB0114) /* (PDC_US0) Receive Next Counter Register*/
#define AT91C_US0_PTSR  ((AT91_REG *) 	0xFFFB0124) /* (PDC_US0) PDC Transfer Status Register*/
#define AT91C_US0_TNCR  ((AT91_REG *) 	0xFFFB011C) /* (PDC_US0) Transmit Next Counter Register*/
#define AT91C_US0_RNPR  ((AT91_REG *) 	0xFFFB0110) /* (PDC_US0) Receive Next Pointer Register*/
#define AT91C_US0_RCR   ((AT91_REG *) 	0xFFFB0104) /* (PDC_US0) Receive Counter Register*/
#define AT91C_US0_TPR   ((AT91_REG *) 	0xFFFB0108) /* (PDC_US0) Transmit Pointer Register*/
#define AT91C_US0_TNPR  ((AT91_REG *) 	0xFFFB0118) /* (PDC_US0) Transmit Next Pointer Register*/
#define AT91C_US0_RPR   ((AT91_REG *) 	0xFFFB0100) /* (PDC_US0) Receive Pointer Register*/
/* ========== Register definition for US0 peripheral ========== */
#define AT91C_US0_RHR   ((AT91_REG *) 	0xFFFB0018) /* (US0) Receiver Holding Register*/
#define AT91C_US0_NER   ((AT91_REG *) 	0xFFFB0044) /* (US0) Nb Errors Register*/
#define AT91C_US0_IER   ((AT91_REG *) 	0xFFFB0008) /* (US0) Interrupt Enable Register*/
#define AT91C_US0_CR    ((AT91_REG *) 	0xFFFB0000) /* (US0) Control Register*/
#define AT91C_US0_MAN   ((AT91_REG *) 	0xFFFB0050) /* (US0) Manchester Encoder Decoder Register*/
#define AT91C_US0_THR   ((AT91_REG *) 	0xFFFB001C) /* (US0) Transmitter Holding Register*/
#define AT91C_US0_CSR   ((AT91_REG *) 	0xFFFB0014) /* (US0) Channel Status Register*/
#define AT91C_US0_BRGR  ((AT91_REG *) 	0xFFFB0020) /* (US0) Baud Rate Generator Register*/
#define AT91C_US0_RTOR  ((AT91_REG *) 	0xFFFB0024) /* (US0) Receiver Time-out Register*/
#define AT91C_US0_TTGR  ((AT91_REG *) 	0xFFFB0028) /* (US0) Transmitter Time-guard Register*/
#define AT91C_US0_IDR   ((AT91_REG *) 	0xFFFB000C) /* (US0) Interrupt Disable Register*/
#define AT91C_US0_MR    ((AT91_REG *) 	0xFFFB0004) /* (US0) Mode Register*/
#define AT91C_US0_IF    ((AT91_REG *) 	0xFFFB004C) /* (US0) IRDA_FILTER Register*/
#define AT91C_US0_FIDI  ((AT91_REG *) 	0xFFFB0040) /* (US0) FI_DI_Ratio Register*/
#define AT91C_US0_IMR   ((AT91_REG *) 	0xFFFB0010) /* (US0) Interrupt Mask Register*/
/* ========== Register definition for PDC_US1 peripheral ========== */
#define AT91C_US1_PTCR  ((AT91_REG *) 	0xFFFB4120) /* (PDC_US1) PDC Transfer Control Register*/
#define AT91C_US1_RCR   ((AT91_REG *) 	0xFFFB4104) /* (PDC_US1) Receive Counter Register*/
#define AT91C_US1_RPR   ((AT91_REG *) 	0xFFFB4100) /* (PDC_US1) Receive Pointer Register*/
#define AT91C_US1_PTSR  ((AT91_REG *) 	0xFFFB4124) /* (PDC_US1) PDC Transfer Status Register*/
#define AT91C_US1_TPR   ((AT91_REG *) 	0xFFFB4108) /* (PDC_US1) Transmit Pointer Register*/
#define AT91C_US1_TCR   ((AT91_REG *) 	0xFFFB410C) /* (PDC_US1) Transmit Counter Register*/
#define AT91C_US1_RNPR  ((AT91_REG *) 	0xFFFB4110) /* (PDC_US1) Receive Next Pointer Register*/
#define AT91C_US1_TNCR  ((AT91_REG *) 	0xFFFB411C) /* (PDC_US1) Transmit Next Counter Register*/
#define AT91C_US1_RNCR  ((AT91_REG *) 	0xFFFB4114) /* (PDC_US1) Receive Next Counter Register*/
#define AT91C_US1_TNPR  ((AT91_REG *) 	0xFFFB4118) /* (PDC_US1) Transmit Next Pointer Register*/
/* ========== Register definition for US1 peripheral ========== */
#define AT91C_US1_THR   ((AT91_REG *) 	0xFFFB401C) /* (US1) Transmitter Holding Register*/
#define AT91C_US1_TTGR  ((AT91_REG *) 	0xFFFB4028) /* (US1) Transmitter Time-guard Register*/
#define AT91C_US1_BRGR  ((AT91_REG *) 	0xFFFB4020) /* (US1) Baud Rate Generator Register*/
#define AT91C_US1_IDR   ((AT91_REG *) 	0xFFFB400C) /* (US1) Interrupt Disable Register*/
#define AT91C_US1_MR    ((AT91_REG *) 	0xFFFB4004) /* (US1) Mode Register*/
#define AT91C_US1_RTOR  ((AT91_REG *) 	0xFFFB4024) /* (US1) Receiver Time-out Register*/
#define AT91C_US1_MAN   ((AT91_REG *) 	0xFFFB4050) /* (US1) Manchester Encoder Decoder Register*/
#define AT91C_US1_CR    ((AT91_REG *) 	0xFFFB4000) /* (US1) Control Register*/
#define AT91C_US1_IMR   ((AT91_REG *) 	0xFFFB4010) /* (US1) Interrupt Mask Register*/
#define AT91C_US1_FIDI  ((AT91_REG *) 	0xFFFB4040) /* (US1) FI_DI_Ratio Register*/
#define AT91C_US1_RHR   ((AT91_REG *) 	0xFFFB4018) /* (US1) Receiver Holding Register*/
#define AT91C_US1_IER   ((AT91_REG *) 	0xFFFB4008) /* (US1) Interrupt Enable Register*/
#define AT91C_US1_CSR   ((AT91_REG *) 	0xFFFB4014) /* (US1) Channel Status Register*/
#define AT91C_US1_IF    ((AT91_REG *) 	0xFFFB404C) /* (US1) IRDA_FILTER Register*/
#define AT91C_US1_NER   ((AT91_REG *) 	0xFFFB4044) /* (US1) Nb Errors Register*/
/* ========== Register definition for PDC_US2 peripheral ========== */
#define AT91C_US2_TNCR  ((AT91_REG *) 	0xFFFB811C) /* (PDC_US2) Transmit Next Counter Register*/
#define AT91C_US2_RNCR  ((AT91_REG *) 	0xFFFB8114) /* (PDC_US2) Receive Next Counter Register*/
#define AT91C_US2_TNPR  ((AT91_REG *) 	0xFFFB8118) /* (PDC_US2) Transmit Next Pointer Register*/
#define AT91C_US2_PTCR  ((AT91_REG *) 	0xFFFB8120) /* (PDC_US2) PDC Transfer Control Register*/
#define AT91C_US2_TCR   ((AT91_REG *) 	0xFFFB810C) /* (PDC_US2) Transmit Counter Register*/
#define AT91C_US2_RPR   ((AT91_REG *) 	0xFFFB8100) /* (PDC_US2) Receive Pointer Register*/
#define AT91C_US2_TPR   ((AT91_REG *) 	0xFFFB8108) /* (PDC_US2) Transmit Pointer Register*/
#define AT91C_US2_RCR   ((AT91_REG *) 	0xFFFB8104) /* (PDC_US2) Receive Counter Register*/
#define AT91C_US2_PTSR  ((AT91_REG *) 	0xFFFB8124) /* (PDC_US2) PDC Transfer Status Register*/
#define AT91C_US2_RNPR  ((AT91_REG *) 	0xFFFB8110) /* (PDC_US2) Receive Next Pointer Register*/
/* ========== Register definition for US2 peripheral ========== */
#define AT91C_US2_RTOR  ((AT91_REG *) 	0xFFFB8024) /* (US2) Receiver Time-out Register*/
#define AT91C_US2_CSR   ((AT91_REG *) 	0xFFFB8014) /* (US2) Channel Status Register*/
#define AT91C_US2_CR    ((AT91_REG *) 	0xFFFB8000) /* (US2) Control Register*/
#define AT91C_US2_BRGR  ((AT91_REG *) 	0xFFFB8020) /* (US2) Baud Rate Generator Register*/
#define AT91C_US2_NER   ((AT91_REG *) 	0xFFFB8044) /* (US2) Nb Errors Register*/
#define AT91C_US2_MAN   ((AT91_REG *) 	0xFFFB8050) /* (US2) Manchester Encoder Decoder Register*/
#define AT91C_US2_FIDI  ((AT91_REG *) 	0xFFFB8040) /* (US2) FI_DI_Ratio Register*/
#define AT91C_US2_TTGR  ((AT91_REG *) 	0xFFFB8028) /* (US2) Transmitter Time-guard Register*/
#define AT91C_US2_RHR   ((AT91_REG *) 	0xFFFB8018) /* (US2) Receiver Holding Register*/
#define AT91C_US2_IDR   ((AT91_REG *) 	0xFFFB800C) /* (US2) Interrupt Disable Register*/
#define AT91C_US2_THR   ((AT91_REG *) 	0xFFFB801C) /* (US2) Transmitter Holding Register*/
#define AT91C_US2_MR    ((AT91_REG *) 	0xFFFB8004) /* (US2) Mode Register*/
#define AT91C_US2_IMR   ((AT91_REG *) 	0xFFFB8010) /* (US2) Interrupt Mask Register*/
#define AT91C_US2_IF    ((AT91_REG *) 	0xFFFB804C) /* (US2) IRDA_FILTER Register*/
#define AT91C_US2_IER   ((AT91_REG *) 	0xFFFB8008) /* (US2) Interrupt Enable Register*/
/* ========== Register definition for PDC_US3 peripheral ========== */
#define AT91C_US3_TNPR  ((AT91_REG *) 	0xFFFBC118) /* (PDC_US3) Transmit Next Pointer Register*/
#define AT91C_US3_TCR   ((AT91_REG *) 	0xFFFBC10C) /* (PDC_US3) Transmit Counter Register*/
#define AT91C_US3_RNCR  ((AT91_REG *) 	0xFFFBC114) /* (PDC_US3) Receive Next Counter Register*/
#define AT91C_US3_RPR   ((AT91_REG *) 	0xFFFBC100) /* (PDC_US3) Receive Pointer Register*/
#define AT91C_US3_TPR   ((AT91_REG *) 	0xFFFBC108) /* (PDC_US3) Transmit Pointer Register*/
#define AT91C_US3_RCR   ((AT91_REG *) 	0xFFFBC104) /* (PDC_US3) Receive Counter Register*/
#define AT91C_US3_RNPR  ((AT91_REG *) 	0xFFFBC110) /* (PDC_US3) Receive Next Pointer Register*/
#define AT91C_US3_PTCR  ((AT91_REG *) 	0xFFFBC120) /* (PDC_US3) PDC Transfer Control Register*/
#define AT91C_US3_TNCR  ((AT91_REG *) 	0xFFFBC11C) /* (PDC_US3) Transmit Next Counter Register*/
#define AT91C_US3_PTSR  ((AT91_REG *) 	0xFFFBC124) /* (PDC_US3) PDC Transfer Status Register*/
/* ========== Register definition for US3 peripheral ========== */
#define AT91C_US3_IF    ((AT91_REG *) 	0xFFFBC04C) /* (US3) IRDA_FILTER Register*/
#define AT91C_US3_CSR   ((AT91_REG *) 	0xFFFBC014) /* (US3) Channel Status Register*/
#define AT91C_US3_TTGR  ((AT91_REG *) 	0xFFFBC028) /* (US3) Transmitter Time-guard Register*/
#define AT91C_US3_MAN   ((AT91_REG *) 	0xFFFBC050) /* (US3) Manchester Encoder Decoder Register*/
#define AT91C_US3_CR    ((AT91_REG *) 	0xFFFBC000) /* (US3) Control Register*/
#define AT91C_US3_THR   ((AT91_REG *) 	0xFFFBC01C) /* (US3) Transmitter Holding Register*/
#define AT91C_US3_MR    ((AT91_REG *) 	0xFFFBC004) /* (US3) Mode Register*/
#define AT91C_US3_NER   ((AT91_REG *) 	0xFFFBC044) /* (US3) Nb Errors Register*/
#define AT91C_US3_IDR   ((AT91_REG *) 	0xFFFBC00C) /* (US3) Interrupt Disable Register*/
#define AT91C_US3_BRGR  ((AT91_REG *) 	0xFFFBC020) /* (US3) Baud Rate Generator Register*/
#define AT91C_US3_IMR   ((AT91_REG *) 	0xFFFBC010) /* (US3) Interrupt Mask Register*/
#define AT91C_US3_FIDI  ((AT91_REG *) 	0xFFFBC040) /* (US3) FI_DI_Ratio Register*/
#define AT91C_US3_RTOR  ((AT91_REG *) 	0xFFFBC024) /* (US3) Receiver Time-out Register*/
#define AT91C_US3_IER   ((AT91_REG *) 	0xFFFBC008) /* (US3) Interrupt Enable Register*/
#define AT91C_US3_RHR   ((AT91_REG *) 	0xFFFBC018) /* (US3) Receiver Holding Register*/
/* ========== Register definition for PDC_SSC0 peripheral ========== */
#define AT91C_SSC0_TNPR ((AT91_REG *) 	0xFFFC0118) /* (PDC_SSC0) Transmit Next Pointer Register*/
#define AT91C_SSC0_RNPR ((AT91_REG *) 	0xFFFC0110) /* (PDC_SSC0) Receive Next Pointer Register*/
#define AT91C_SSC0_TCR  ((AT91_REG *) 	0xFFFC010C) /* (PDC_SSC0) Transmit Counter Register*/
#define AT91C_SSC0_PTCR ((AT91_REG *) 	0xFFFC0120) /* (PDC_SSC0) PDC Transfer Control Register*/
#define AT91C_SSC0_PTSR ((AT91_REG *) 	0xFFFC0124) /* (PDC_SSC0) PDC Transfer Status Register*/
#define AT91C_SSC0_TNCR ((AT91_REG *) 	0xFFFC011C) /* (PDC_SSC0) Transmit Next Counter Register*/
#define AT91C_SSC0_TPR  ((AT91_REG *) 	0xFFFC0108) /* (PDC_SSC0) Transmit Pointer Register*/
#define AT91C_SSC0_RCR  ((AT91_REG *) 	0xFFFC0104) /* (PDC_SSC0) Receive Counter Register*/
#define AT91C_SSC0_RPR  ((AT91_REG *) 	0xFFFC0100) /* (PDC_SSC0) Receive Pointer Register*/
#define AT91C_SSC0_RNCR ((AT91_REG *) 	0xFFFC0114) /* (PDC_SSC0) Receive Next Counter Register*/
/* ========== Register definition for SSC0 peripheral ========== */
#define AT91C_SSC0_IDR  ((AT91_REG *) 	0xFFFC0048) /* (SSC0) Interrupt Disable Register*/
#define AT91C_SSC0_RHR  ((AT91_REG *) 	0xFFFC0020) /* (SSC0) Receive Holding Register*/
#define AT91C_SSC0_IER  ((AT91_REG *) 	0xFFFC0044) /* (SSC0) Interrupt Enable Register*/
#define AT91C_SSC0_CR   ((AT91_REG *) 	0xFFFC0000) /* (SSC0) Control Register*/
#define AT91C_SSC0_RCMR ((AT91_REG *) 	0xFFFC0010) /* (SSC0) Receive Clock ModeRegister*/
#define AT91C_SSC0_SR   ((AT91_REG *) 	0xFFFC0040) /* (SSC0) Status Register*/
#define AT91C_SSC0_TSHR ((AT91_REG *) 	0xFFFC0034) /* (SSC0) Transmit Sync Holding Register*/
#define AT91C_SSC0_CMR  ((AT91_REG *) 	0xFFFC0004) /* (SSC0) Clock Mode Register*/
#define AT91C_SSC0_RSHR ((AT91_REG *) 	0xFFFC0030) /* (SSC0) Receive Sync Holding Register*/
#define AT91C_SSC0_THR  ((AT91_REG *) 	0xFFFC0024) /* (SSC0) Transmit Holding Register*/
#define AT91C_SSC0_RFMR ((AT91_REG *) 	0xFFFC0014) /* (SSC0) Receive Frame Mode Register*/
#define AT91C_SSC0_TCMR ((AT91_REG *) 	0xFFFC0018) /* (SSC0) Transmit Clock Mode Register*/
#define AT91C_SSC0_TFMR ((AT91_REG *) 	0xFFFC001C) /* (SSC0) Transmit Frame Mode Register*/
#define AT91C_SSC0_IMR  ((AT91_REG *) 	0xFFFC004C) /* (SSC0) Interrupt Mask Register*/
/* ========== Register definition for PDC_SSC1 peripheral ========== */
#define AT91C_SSC1_RNCR ((AT91_REG *) 	0xFFFC4114) /* (PDC_SSC1) Receive Next Counter Register*/
#define AT91C_SSC1_PTCR ((AT91_REG *) 	0xFFFC4120) /* (PDC_SSC1) PDC Transfer Control Register*/
#define AT91C_SSC1_TCR  ((AT91_REG *) 	0xFFFC410C) /* (PDC_SSC1) Transmit Counter Register*/
#define AT91C_SSC1_PTSR ((AT91_REG *) 	0xFFFC4124) /* (PDC_SSC1) PDC Transfer Status Register*/
#define AT91C_SSC1_TNPR ((AT91_REG *) 	0xFFFC4118) /* (PDC_SSC1) Transmit Next Pointer Register*/
#define AT91C_SSC1_RCR  ((AT91_REG *) 	0xFFFC4104) /* (PDC_SSC1) Receive Counter Register*/
#define AT91C_SSC1_RNPR ((AT91_REG *) 	0xFFFC4110) /* (PDC_SSC1) Receive Next Pointer Register*/
#define AT91C_SSC1_RPR  ((AT91_REG *) 	0xFFFC4100) /* (PDC_SSC1) Receive Pointer Register*/
#define AT91C_SSC1_TNCR ((AT91_REG *) 	0xFFFC411C) /* (PDC_SSC1) Transmit Next Counter Register*/
#define AT91C_SSC1_TPR  ((AT91_REG *) 	0xFFFC4108) /* (PDC_SSC1) Transmit Pointer Register*/
/* ========== Register definition for SSC1 peripheral ========== */
#define AT91C_SSC1_IMR  ((AT91_REG *) 	0xFFFC404C) /* (SSC1) Interrupt Mask Register*/
#define AT91C_SSC1_IER  ((AT91_REG *) 	0xFFFC4044) /* (SSC1) Interrupt Enable Register*/
#define AT91C_SSC1_THR  ((AT91_REG *) 	0xFFFC4024) /* (SSC1) Transmit Holding Register*/
#define AT91C_SSC1_RFMR ((AT91_REG *) 	0xFFFC4014) /* (SSC1) Receive Frame Mode Register*/
#define AT91C_SSC1_TFMR ((AT91_REG *) 	0xFFFC401C) /* (SSC1) Transmit Frame Mode Register*/
#define AT91C_SSC1_IDR  ((AT91_REG *) 	0xFFFC4048) /* (SSC1) Interrupt Disable Register*/
#define AT91C_SSC1_RSHR ((AT91_REG *) 	0xFFFC4030) /* (SSC1) Receive Sync Holding Register*/
#define AT91C_SSC1_TCMR ((AT91_REG *) 	0xFFFC4018) /* (SSC1) Transmit Clock Mode Register*/
#define AT91C_SSC1_RHR  ((AT91_REG *) 	0xFFFC4020) /* (SSC1) Receive Holding Register*/
#define AT91C_SSC1_RCMR ((AT91_REG *) 	0xFFFC4010) /* (SSC1) Receive Clock ModeRegister*/
#define AT91C_SSC1_CR   ((AT91_REG *) 	0xFFFC4000) /* (SSC1) Control Register*/
#define AT91C_SSC1_SR   ((AT91_REG *) 	0xFFFC4040) /* (SSC1) Status Register*/
#define AT91C_SSC1_CMR  ((AT91_REG *) 	0xFFFC4004) /* (SSC1) Clock Mode Register*/
#define AT91C_SSC1_TSHR ((AT91_REG *) 	0xFFFC4034) /* (SSC1) Transmit Sync Holding Register*/
/* ========== Register definition for PWMC_CH0 peripheral ========== */
#define AT91C_PWMC_CH0_Reserved ((AT91_REG *) 	0xFFFC8214) /* (PWMC_CH0) Reserved*/
#define AT91C_PWMC_CH0_CUPDR ((AT91_REG *) 	0xFFFC8210) /* (PWMC_CH0) Channel Update Register*/
#define AT91C_PWMC_CH0_CCNTR ((AT91_REG *) 	0xFFFC820C) /* (PWMC_CH0) Channel Counter Register*/
#define AT91C_PWMC_CH0_CDTYR ((AT91_REG *) 	0xFFFC8204) /* (PWMC_CH0) Channel Duty Cycle Register*/
#define AT91C_PWMC_CH0_CPRDR ((AT91_REG *) 	0xFFFC8208) /* (PWMC_CH0) Channel Period Register*/
#define AT91C_PWMC_CH0_CMR ((AT91_REG *) 	0xFFFC8200) /* (PWMC_CH0) Channel Mode Register*/
/* ========== Register definition for PWMC_CH1 peripheral ========== */
#define AT91C_PWMC_CH1_CPRDR ((AT91_REG *) 	0xFFFC8228) /* (PWMC_CH1) Channel Period Register*/
#define AT91C_PWMC_CH1_CMR ((AT91_REG *) 	0xFFFC8220) /* (PWMC_CH1) Channel Mode Register*/
#define AT91C_PWMC_CH1_Reserved ((AT91_REG *) 	0xFFFC8234) /* (PWMC_CH1) Reserved*/
#define AT91C_PWMC_CH1_CUPDR ((AT91_REG *) 	0xFFFC8230) /* (PWMC_CH1) Channel Update Register*/
#define AT91C_PWMC_CH1_CDTYR ((AT91_REG *) 	0xFFFC8224) /* (PWMC_CH1) Channel Duty Cycle Register*/
#define AT91C_PWMC_CH1_CCNTR ((AT91_REG *) 	0xFFFC822C) /* (PWMC_CH1) Channel Counter Register*/
/* ========== Register definition for PWMC_CH2 peripheral ========== */
#define AT91C_PWMC_CH2_CCNTR ((AT91_REG *) 	0xFFFC824C) /* (PWMC_CH2) Channel Counter Register*/
#define AT91C_PWMC_CH2_CUPDR ((AT91_REG *) 	0xFFFC8250) /* (PWMC_CH2) Channel Update Register*/
#define AT91C_PWMC_CH2_Reserved ((AT91_REG *) 	0xFFFC8254) /* (PWMC_CH2) Reserved*/
#define AT91C_PWMC_CH2_CDTYR ((AT91_REG *) 	0xFFFC8244) /* (PWMC_CH2) Channel Duty Cycle Register*/
#define AT91C_PWMC_CH2_CMR ((AT91_REG *) 	0xFFFC8240) /* (PWMC_CH2) Channel Mode Register*/
#define AT91C_PWMC_CH2_CPRDR ((AT91_REG *) 	0xFFFC8248) /* (PWMC_CH2) Channel Period Register*/
/* ========== Register definition for PWMC_CH3 peripheral ========== */
#define AT91C_PWMC_CH3_CPRDR ((AT91_REG *) 	0xFFFC8268) /* (PWMC_CH3) Channel Period Register*/
#define AT91C_PWMC_CH3_CCNTR ((AT91_REG *) 	0xFFFC826C) /* (PWMC_CH3) Channel Counter Register*/
#define AT91C_PWMC_CH3_CDTYR ((AT91_REG *) 	0xFFFC8264) /* (PWMC_CH3) Channel Duty Cycle Register*/
#define AT91C_PWMC_CH3_CUPDR ((AT91_REG *) 	0xFFFC8270) /* (PWMC_CH3) Channel Update Register*/
#define AT91C_PWMC_CH3_Reserved ((AT91_REG *) 	0xFFFC8274) /* (PWMC_CH3) Reserved*/
#define AT91C_PWMC_CH3_CMR ((AT91_REG *) 	0xFFFC8260) /* (PWMC_CH3) Channel Mode Register*/
/* ========== Register definition for PWMC peripheral ========== */
#define AT91C_PWMC_IMR  ((AT91_REG *) 	0xFFFC8018) /* (PWMC) PWMC Interrupt Mask Register*/
#define AT91C_PWMC_SR   ((AT91_REG *) 	0xFFFC800C) /* (PWMC) PWMC Status Register*/
#define AT91C_PWMC_IER  ((AT91_REG *) 	0xFFFC8010) /* (PWMC) PWMC Interrupt Enable Register*/
#define AT91C_PWMC_VR   ((AT91_REG *) 	0xFFFC80FC) /* (PWMC) PWMC Version Register*/
#define AT91C_PWMC_MR   ((AT91_REG *) 	0xFFFC8000) /* (PWMC) PWMC Mode Register*/
#define AT91C_PWMC_DIS  ((AT91_REG *) 	0xFFFC8008) /* (PWMC) PWMC Disable Register*/
#define AT91C_PWMC_ENA  ((AT91_REG *) 	0xFFFC8004) /* (PWMC) PWMC Enable Register*/
#define AT91C_PWMC_IDR  ((AT91_REG *) 	0xFFFC8014) /* (PWMC) PWMC Interrupt Disable Register*/
#define AT91C_PWMC_ISR  ((AT91_REG *) 	0xFFFC801C) /* (PWMC) PWMC Interrupt Status Register*/
/* ========== Register definition for PDC_SPI peripheral ========== */
#define AT91C_SPI_PTCR  ((AT91_REG *) 	0xFFFCC120) /* (PDC_SPI) PDC Transfer Control Register*/
#define AT91C_SPI_RNPR  ((AT91_REG *) 	0xFFFCC110) /* (PDC_SPI) Receive Next Pointer Register*/
#define AT91C_SPI_RCR   ((AT91_REG *) 	0xFFFCC104) /* (PDC_SPI) Receive Counter Register*/
#define AT91C_SPI_TPR   ((AT91_REG *) 	0xFFFCC108) /* (PDC_SPI) Transmit Pointer Register*/
#define AT91C_SPI_PTSR  ((AT91_REG *) 	0xFFFCC124) /* (PDC_SPI) PDC Transfer Status Register*/
#define AT91C_SPI_TNCR  ((AT91_REG *) 	0xFFFCC11C) /* (PDC_SPI) Transmit Next Counter Register*/
#define AT91C_SPI_RPR   ((AT91_REG *) 	0xFFFCC100) /* (PDC_SPI) Receive Pointer Register*/
#define AT91C_SPI_TCR   ((AT91_REG *) 	0xFFFCC10C) /* (PDC_SPI) Transmit Counter Register*/
#define AT91C_SPI_RNCR  ((AT91_REG *) 	0xFFFCC114) /* (PDC_SPI) Receive Next Counter Register*/
#define AT91C_SPI_TNPR  ((AT91_REG *) 	0xFFFCC118) /* (PDC_SPI) Transmit Next Pointer Register*/
/* ========== Register definition for SPI peripheral ========== */
#define AT91C_SPI_IER   ((AT91_REG *) 	0xFFFCC014) /* (SPI) Interrupt Enable Register*/
#define AT91C_SPI_RDR   ((AT91_REG *) 	0xFFFCC008) /* (SPI) Receive Data Register*/
#define AT91C_SPI_SR    ((AT91_REG *) 	0xFFFCC010) /* (SPI) Status Register*/
#define AT91C_SPI_IMR   ((AT91_REG *) 	0xFFFCC01C) /* (SPI) Interrupt Mask Register*/
#define AT91C_SPI_TDR   ((AT91_REG *) 	0xFFFCC00C) /* (SPI) Transmit Data Register*/
#define AT91C_SPI_IDR   ((AT91_REG *) 	0xFFFCC018) /* (SPI) Interrupt Disable Register*/
#define AT91C_SPI_CSR   ((AT91_REG *) 	0xFFFCC030) /* (SPI) Chip Select Register*/
#define AT91C_SPI_CR    ((AT91_REG *) 	0xFFFCC000) /* (SPI) Control Register*/
#define AT91C_SPI_MR    ((AT91_REG *) 	0xFFFCC004) /* (SPI) Mode Register*/
/* ========== Register definition for PDC_TSADC peripheral ========== */
#define AT91C_TSADC_RNPR ((AT91_REG *) 	0xFFFD0110) /* (PDC_TSADC) Receive Next Pointer Register*/
#define AT91C_TSADC_RNCR ((AT91_REG *) 	0xFFFD0114) /* (PDC_TSADC) Receive Next Counter Register*/
#define AT91C_TSADC_PTSR ((AT91_REG *) 	0xFFFD0124) /* (PDC_TSADC) PDC Transfer Status Register*/
#define AT91C_TSADC_PTCR ((AT91_REG *) 	0xFFFD0120) /* (PDC_TSADC) PDC Transfer Control Register*/
#define AT91C_TSADC_TCR ((AT91_REG *) 	0xFFFD010C) /* (PDC_TSADC) Transmit Counter Register*/
#define AT91C_TSADC_TNPR ((AT91_REG *) 	0xFFFD0118) /* (PDC_TSADC) Transmit Next Pointer Register*/
#define AT91C_TSADC_RCR ((AT91_REG *) 	0xFFFD0104) /* (PDC_TSADC) Receive Counter Register*/
#define AT91C_TSADC_TPR ((AT91_REG *) 	0xFFFD0108) /* (PDC_TSADC) Transmit Pointer Register*/
#define AT91C_TSADC_TNCR ((AT91_REG *) 	0xFFFD011C) /* (PDC_TSADC) Transmit Next Counter Register*/
#define AT91C_TSADC_RPR ((AT91_REG *) 	0xFFFD0100) /* (PDC_TSADC) Receive Pointer Register*/
/* ========== Register definition for TSADC peripheral ========== */
#define AT91C_TSADC_IER ((AT91_REG *) 	0xFFFD0024) /* (TSADC) Interrupt Enable Register*/
#define AT91C_TSADC_MR  ((AT91_REG *) 	0xFFFD0004) /* (TSADC) Mode Register*/
#define AT91C_TSADC_CDR4 ((AT91_REG *) 	0xFFFD0040) /* (TSADC) Channel Data Register 4*/
#define AT91C_TSADC_CDR2 ((AT91_REG *) 	0xFFFD0038) /* (TSADC) Channel Data Register 2*/
#define AT91C_TSADC_TRGR ((AT91_REG *) 	0xFFFD0008) /* (TSADC) Trigger Register*/
#define AT91C_TSADC_IDR ((AT91_REG *) 	0xFFFD0028) /* (TSADC) Interrupt Disable Register*/
#define AT91C_TSADC_CHER ((AT91_REG *) 	0xFFFD0010) /* (TSADC) Channel Enable Register*/
#define AT91C_TSADC_CDR5 ((AT91_REG *) 	0xFFFD0044) /* (TSADC) Channel Data Register 5*/
#define AT91C_TSADC_CDR3 ((AT91_REG *) 	0xFFFD003C) /* (TSADC) Channel Data Register 3*/
#define AT91C_TSADC_TSR ((AT91_REG *) 	0xFFFD000C) /* (TSADC) Touch Screen Register*/
#define AT91C_TSADC_IMR ((AT91_REG *) 	0xFFFD002C) /* (TSADC) Interrupt Mask Register*/
#define AT91C_TSADC_CR  ((AT91_REG *) 	0xFFFD0000) /* (TSADC) Control Register*/
#define AT91C_TSADC_SR  ((AT91_REG *) 	0xFFFD001C) /* (TSADC) Status Register*/
#define AT91C_TSADC_LCDR ((AT91_REG *) 	0xFFFD0020) /* (TSADC) Last Converted Register*/
#define AT91C_TSADC_CHSR ((AT91_REG *) 	0xFFFD0018) /* (TSADC) Channel Status Register*/
#define AT91C_TSADC_CDR0 ((AT91_REG *) 	0xFFFD0030) /* (TSADC) Channel Data Register 0*/
#define AT91C_TSADC_CHDR ((AT91_REG *) 	0xFFFD0014) /* (TSADC) Channel Disable Register*/
#define AT91C_TSADC_CDR1 ((AT91_REG *) 	0xFFFD0034) /* (TSADC) Channel Data Register 1*/
/* ========== Register definition for UDPHS_EPTFIFO peripheral ========== */
#define AT91C_UDPHS_EPTFIFO_READEPTF ((AT91_REG *) 	0x006F0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 15*/
#define AT91C_UDPHS_EPTFIFO_READEPT5 ((AT91_REG *) 	0x00650000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5*/
#define AT91C_UDPHS_EPTFIFO_READEPT1 ((AT91_REG *) 	0x00610000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1*/
#define AT91C_UDPHS_EPTFIFO_READEPTE ((AT91_REG *) 	0x006E0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 14*/
#define AT91C_UDPHS_EPTFIFO_READEPT4 ((AT91_REG *) 	0x00640000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4*/
#define AT91C_UDPHS_EPTFIFO_READEPTD ((AT91_REG *) 	0x006D0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 13*/
#define AT91C_UDPHS_EPTFIFO_READEPT2 ((AT91_REG *) 	0x00620000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2*/
#define AT91C_UDPHS_EPTFIFO_READEPT6 ((AT91_REG *) 	0x00660000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6*/
#define AT91C_UDPHS_EPTFIFO_READEPT9 ((AT91_REG *) 	0x00690000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 9*/
#define AT91C_UDPHS_EPTFIFO_READEPT0 ((AT91_REG *) 	0x00600000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0*/
#define AT91C_UDPHS_EPTFIFO_READEPTA ((AT91_REG *) 	0x006A0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 10*/
#define AT91C_UDPHS_EPTFIFO_READEPT3 ((AT91_REG *) 	0x00630000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3*/
#define AT91C_UDPHS_EPTFIFO_READEPTC ((AT91_REG *) 	0x006C0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 12*/
#define AT91C_UDPHS_EPTFIFO_READEPTB ((AT91_REG *) 	0x006B0000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 11*/
#define AT91C_UDPHS_EPTFIFO_READEPT8 ((AT91_REG *) 	0x00680000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 8*/
#define AT91C_UDPHS_EPTFIFO_READEPT7 ((AT91_REG *) 	0x00670000) /* (UDPHS_EPTFIFO) FIFO Endpoint Data Register 7*/
/* ========== Register definition for UDPHS_EPT_0 peripheral ========== */
#define AT91C_UDPHS_EPT_0_EPTSTA ((AT91_REG *) 	0xFFFD411C) /* (UDPHS_EPT_0) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_0_EPTCTLENB ((AT91_REG *) 	0xFFFD4104) /* (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_0_EPTCFG ((AT91_REG *) 	0xFFFD4100) /* (UDPHS_EPT_0) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_0_EPTSETSTA ((AT91_REG *) 	0xFFFD4114) /* (UDPHS_EPT_0) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_0_EPTCTLDIS ((AT91_REG *) 	0xFFFD4108) /* (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_0_EPTCLRSTA ((AT91_REG *) 	0xFFFD4118) /* (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_0_EPTCTL ((AT91_REG *) 	0xFFFD410C) /* (UDPHS_EPT_0) UDPHS Endpoint Control Register*/
/* ========== Register definition for UDPHS_EPT_1 peripheral ========== */
#define AT91C_UDPHS_EPT_1_EPTCTL ((AT91_REG *) 	0xFFFD412C) /* (UDPHS_EPT_1) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_1_EPTSTA ((AT91_REG *) 	0xFFFD413C) /* (UDPHS_EPT_1) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_1_EPTCTLDIS ((AT91_REG *) 	0xFFFD4128) /* (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_1_EPTCFG ((AT91_REG *) 	0xFFFD4120) /* (UDPHS_EPT_1) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_1_EPTSETSTA ((AT91_REG *) 	0xFFFD4134) /* (UDPHS_EPT_1) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_1_EPTCLRSTA ((AT91_REG *) 	0xFFFD4138) /* (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_1_EPTCTLENB ((AT91_REG *) 	0xFFFD4124) /* (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register*/
/* ========== Register definition for UDPHS_EPT_2 peripheral ========== */
#define AT91C_UDPHS_EPT_2_EPTCLRSTA ((AT91_REG *) 	0xFFFD4158) /* (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_2_EPTCTLDIS ((AT91_REG *) 	0xFFFD4148) /* (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_2_EPTSETSTA ((AT91_REG *) 	0xFFFD4154) /* (UDPHS_EPT_2) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_2_EPTCFG ((AT91_REG *) 	0xFFFD4140) /* (UDPHS_EPT_2) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_2_EPTCTL ((AT91_REG *) 	0xFFFD414C) /* (UDPHS_EPT_2) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_2_EPTSTA ((AT91_REG *) 	0xFFFD415C) /* (UDPHS_EPT_2) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_2_EPTCTLENB ((AT91_REG *) 	0xFFFD4144) /* (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register*/
/* ========== Register definition for UDPHS_EPT_3 peripheral ========== */
#define AT91C_UDPHS_EPT_3_EPTSTA ((AT91_REG *) 	0xFFFD417C) /* (UDPHS_EPT_3) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_3_EPTSETSTA ((AT91_REG *) 	0xFFFD4174) /* (UDPHS_EPT_3) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_3_EPTCTL ((AT91_REG *) 	0xFFFD416C) /* (UDPHS_EPT_3) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_3_EPTCTLENB ((AT91_REG *) 	0xFFFD4164) /* (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_3_EPTCLRSTA ((AT91_REG *) 	0xFFFD4178) /* (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_3_EPTCFG ((AT91_REG *) 	0xFFFD4160) /* (UDPHS_EPT_3) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_3_EPTCTLDIS ((AT91_REG *) 	0xFFFD4168) /* (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register*/
/* ========== Register definition for UDPHS_EPT_4 peripheral ========== */
#define AT91C_UDPHS_EPT_4_EPTCLRSTA ((AT91_REG *) 	0xFFFD4198) /* (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_4_EPTCTL ((AT91_REG *) 	0xFFFD418C) /* (UDPHS_EPT_4) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_4_EPTSTA ((AT91_REG *) 	0xFFFD419C) /* (UDPHS_EPT_4) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_4_EPTCTLENB ((AT91_REG *) 	0xFFFD4184) /* (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_4_EPTCFG ((AT91_REG *) 	0xFFFD4180) /* (UDPHS_EPT_4) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_4_EPTSETSTA ((AT91_REG *) 	0xFFFD4194) /* (UDPHS_EPT_4) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_4_EPTCTLDIS ((AT91_REG *) 	0xFFFD4188) /* (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register*/
/* ========== Register definition for UDPHS_EPT_5 peripheral ========== */
#define AT91C_UDPHS_EPT_5_EPTSETSTA ((AT91_REG *) 	0xFFFD41B4) /* (UDPHS_EPT_5) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_5_EPTCTLDIS ((AT91_REG *) 	0xFFFD41A8) /* (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_5_EPTCTL ((AT91_REG *) 	0xFFFD41AC) /* (UDPHS_EPT_5) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_5_EPTCTLENB ((AT91_REG *) 	0xFFFD41A4) /* (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_5_EPTCFG ((AT91_REG *) 	0xFFFD41A0) /* (UDPHS_EPT_5) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_5_EPTCLRSTA ((AT91_REG *) 	0xFFFD41B8) /* (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_5_EPTSTA ((AT91_REG *) 	0xFFFD41BC) /* (UDPHS_EPT_5) UDPHS Endpoint Status Register*/
/* ========== Register definition for UDPHS_EPT_6 peripheral ========== */
#define AT91C_UDPHS_EPT_6_EPTCFG ((AT91_REG *) 	0xFFFD41C0) /* (UDPHS_EPT_6) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_6_EPTCLRSTA ((AT91_REG *) 	0xFFFD41D8) /* (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_6_EPTCTL ((AT91_REG *) 	0xFFFD41CC) /* (UDPHS_EPT_6) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_6_EPTCTLDIS ((AT91_REG *) 	0xFFFD41C8) /* (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_6_EPTSTA ((AT91_REG *) 	0xFFFD41DC) /* (UDPHS_EPT_6) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_6_EPTSETSTA ((AT91_REG *) 	0xFFFD41D4) /* (UDPHS_EPT_6) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_6_EPTCTLENB ((AT91_REG *) 	0xFFFD41C4) /* (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register*/
/* ========== Register definition for UDPHS_EPT_7 peripheral ========== */
#define AT91C_UDPHS_EPT_7_EPTSETSTA ((AT91_REG *) 	0xFFFD41F4) /* (UDPHS_EPT_7) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_7_EPTCTLDIS ((AT91_REG *) 	0xFFFD41E8) /* (UDPHS_EPT_7) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_7_EPTCTLENB ((AT91_REG *) 	0xFFFD41E4) /* (UDPHS_EPT_7) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_7_EPTSTA ((AT91_REG *) 	0xFFFD41FC) /* (UDPHS_EPT_7) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_7_EPTCLRSTA ((AT91_REG *) 	0xFFFD41F8) /* (UDPHS_EPT_7) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_7_EPTCTL ((AT91_REG *) 	0xFFFD41EC) /* (UDPHS_EPT_7) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_7_EPTCFG ((AT91_REG *) 	0xFFFD41E0) /* (UDPHS_EPT_7) UDPHS Endpoint Config Register*/
/* ========== Register definition for UDPHS_EPT_8 peripheral ========== */
#define AT91C_UDPHS_EPT_8_EPTCLRSTA ((AT91_REG *) 	0xFFFD4218) /* (UDPHS_EPT_8) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_8_EPTCTLDIS ((AT91_REG *) 	0xFFFD4208) /* (UDPHS_EPT_8) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_8_EPTSTA ((AT91_REG *) 	0xFFFD421C) /* (UDPHS_EPT_8) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_8_EPTCFG ((AT91_REG *) 	0xFFFD4200) /* (UDPHS_EPT_8) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_8_EPTCTL ((AT91_REG *) 	0xFFFD420C) /* (UDPHS_EPT_8) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_8_EPTSETSTA ((AT91_REG *) 	0xFFFD4214) /* (UDPHS_EPT_8) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_8_EPTCTLENB ((AT91_REG *) 	0xFFFD4204) /* (UDPHS_EPT_8) UDPHS Endpoint Control Enable Register*/
/* ========== Register definition for UDPHS_EPT_9 peripheral ========== */
#define AT91C_UDPHS_EPT_9_EPTCLRSTA ((AT91_REG *) 	0xFFFD4238) /* (UDPHS_EPT_9) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_9_EPTCTLENB ((AT91_REG *) 	0xFFFD4224) /* (UDPHS_EPT_9) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_9_EPTSTA ((AT91_REG *) 	0xFFFD423C) /* (UDPHS_EPT_9) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_9_EPTSETSTA ((AT91_REG *) 	0xFFFD4234) /* (UDPHS_EPT_9) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_9_EPTCTL ((AT91_REG *) 	0xFFFD422C) /* (UDPHS_EPT_9) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_9_EPTCFG ((AT91_REG *) 	0xFFFD4220) /* (UDPHS_EPT_9) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_9_EPTCTLDIS ((AT91_REG *) 	0xFFFD4228) /* (UDPHS_EPT_9) UDPHS Endpoint Control Disable Register*/
/* ========== Register definition for UDPHS_EPT_10 peripheral ========== */
#define AT91C_UDPHS_EPT_10_EPTCTL ((AT91_REG *) 	0xFFFD424C) /* (UDPHS_EPT_10) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_10_EPTSETSTA ((AT91_REG *) 	0xFFFD4254) /* (UDPHS_EPT_10) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_10_EPTCFG ((AT91_REG *) 	0xFFFD4240) /* (UDPHS_EPT_10) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_10_EPTCLRSTA ((AT91_REG *) 	0xFFFD4258) /* (UDPHS_EPT_10) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_10_EPTSTA ((AT91_REG *) 	0xFFFD425C) /* (UDPHS_EPT_10) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_10_EPTCTLDIS ((AT91_REG *) 	0xFFFD4248) /* (UDPHS_EPT_10) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_10_EPTCTLENB ((AT91_REG *) 	0xFFFD4244) /* (UDPHS_EPT_10) UDPHS Endpoint Control Enable Register*/
/* ========== Register definition for UDPHS_EPT_11 peripheral ========== */
#define AT91C_UDPHS_EPT_11_EPTCTL ((AT91_REG *) 	0xFFFD426C) /* (UDPHS_EPT_11) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_11_EPTCFG ((AT91_REG *) 	0xFFFD4260) /* (UDPHS_EPT_11) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_11_EPTSTA ((AT91_REG *) 	0xFFFD427C) /* (UDPHS_EPT_11) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_11_EPTCTLENB ((AT91_REG *) 	0xFFFD4264) /* (UDPHS_EPT_11) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_11_EPTCLRSTA ((AT91_REG *) 	0xFFFD4278) /* (UDPHS_EPT_11) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_11_EPTSETSTA ((AT91_REG *) 	0xFFFD4274) /* (UDPHS_EPT_11) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_11_EPTCTLDIS ((AT91_REG *) 	0xFFFD4268) /* (UDPHS_EPT_11) UDPHS Endpoint Control Disable Register*/
/* ========== Register definition for UDPHS_EPT_12 peripheral ========== */
#define AT91C_UDPHS_EPT_12_EPTCTLENB ((AT91_REG *) 	0xFFFD4284) /* (UDPHS_EPT_12) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_12_EPTSTA ((AT91_REG *) 	0xFFFD429C) /* (UDPHS_EPT_12) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_12_EPTCTLDIS ((AT91_REG *) 	0xFFFD4288) /* (UDPHS_EPT_12) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_12_EPTSETSTA ((AT91_REG *) 	0xFFFD4294) /* (UDPHS_EPT_12) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_12_EPTCLRSTA ((AT91_REG *) 	0xFFFD4298) /* (UDPHS_EPT_12) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_12_EPTCTL ((AT91_REG *) 	0xFFFD428C) /* (UDPHS_EPT_12) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_12_EPTCFG ((AT91_REG *) 	0xFFFD4280) /* (UDPHS_EPT_12) UDPHS Endpoint Config Register*/
/* ========== Register definition for UDPHS_EPT_13 peripheral ========== */
#define AT91C_UDPHS_EPT_13_EPTSETSTA ((AT91_REG *) 	0xFFFD42B4) /* (UDPHS_EPT_13) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_13_EPTCTLENB ((AT91_REG *) 	0xFFFD42A4) /* (UDPHS_EPT_13) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_13_EPTCFG ((AT91_REG *) 	0xFFFD42A0) /* (UDPHS_EPT_13) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_13_EPTSTA ((AT91_REG *) 	0xFFFD42BC) /* (UDPHS_EPT_13) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_13_EPTCLRSTA ((AT91_REG *) 	0xFFFD42B8) /* (UDPHS_EPT_13) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_13_EPTCTLDIS ((AT91_REG *) 	0xFFFD42A8) /* (UDPHS_EPT_13) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_13_EPTCTL ((AT91_REG *) 	0xFFFD42AC) /* (UDPHS_EPT_13) UDPHS Endpoint Control Register*/
/* ========== Register definition for UDPHS_EPT_14 peripheral ========== */
#define AT91C_UDPHS_EPT_14_EPTCFG ((AT91_REG *) 	0xFFFD42C0) /* (UDPHS_EPT_14) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_14_EPTCLRSTA ((AT91_REG *) 	0xFFFD42D8) /* (UDPHS_EPT_14) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_14_EPTCTLENB ((AT91_REG *) 	0xFFFD42C4) /* (UDPHS_EPT_14) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_14_EPTCTL ((AT91_REG *) 	0xFFFD42CC) /* (UDPHS_EPT_14) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_14_EPTSTA ((AT91_REG *) 	0xFFFD42DC) /* (UDPHS_EPT_14) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_14_EPTSETSTA ((AT91_REG *) 	0xFFFD42D4) /* (UDPHS_EPT_14) UDPHS Endpoint Set Status Register*/
#define AT91C_UDPHS_EPT_14_EPTCTLDIS ((AT91_REG *) 	0xFFFD42C8) /* (UDPHS_EPT_14) UDPHS Endpoint Control Disable Register*/
/* ========== Register definition for UDPHS_EPT_15 peripheral ========== */
#define AT91C_UDPHS_EPT_15_EPTCLRSTA ((AT91_REG *) 	0xFFFD42F8) /* (UDPHS_EPT_15) UDPHS Endpoint Clear Status Register*/
#define AT91C_UDPHS_EPT_15_EPTCTLDIS ((AT91_REG *) 	0xFFFD42E8) /* (UDPHS_EPT_15) UDPHS Endpoint Control Disable Register*/
#define AT91C_UDPHS_EPT_15_EPTSTA ((AT91_REG *) 	0xFFFD42FC) /* (UDPHS_EPT_15) UDPHS Endpoint Status Register*/
#define AT91C_UDPHS_EPT_15_EPTCFG ((AT91_REG *) 	0xFFFD42E0) /* (UDPHS_EPT_15) UDPHS Endpoint Config Register*/
#define AT91C_UDPHS_EPT_15_EPTCTLENB ((AT91_REG *) 	0xFFFD42E4) /* (UDPHS_EPT_15) UDPHS Endpoint Control Enable Register*/
#define AT91C_UDPHS_EPT_15_EPTCTL ((AT91_REG *) 	0xFFFD42EC) /* (UDPHS_EPT_15) UDPHS Endpoint Control Register*/
#define AT91C_UDPHS_EPT_15_EPTSETSTA ((AT91_REG *) 	0xFFFD42F4) /* (UDPHS_EPT_15) UDPHS Endpoint Set Status Register*/
/* ========== Register definition for UDPHS_DMA_1 peripheral ========== */
#define AT91C_UDPHS_DMA_1_DMACONTROL ((AT91_REG *) 	0xFFFD4318) /* (UDPHS_DMA_1) UDPHS DMA Channel Control Register*/
#define AT91C_UDPHS_DMA_1_DMAADDRESS ((AT91_REG *) 	0xFFFD4314) /* (UDPHS_DMA_1) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_1_DMASTATUS ((AT91_REG *) 	0xFFFD431C) /* (UDPHS_DMA_1) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_1_DMANXTDSC ((AT91_REG *) 	0xFFFD4310) /* (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address*/
/* ========== Register definition for UDPHS_DMA_2 peripheral ========== */
#define AT91C_UDPHS_DMA_2_DMACONTROL ((AT91_REG *) 	0xFFFD4328) /* (UDPHS_DMA_2) UDPHS DMA Channel Control Register*/
#define AT91C_UDPHS_DMA_2_DMASTATUS ((AT91_REG *) 	0xFFFD432C) /* (UDPHS_DMA_2) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_2_DMANXTDSC ((AT91_REG *) 	0xFFFD4320) /* (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address*/
#define AT91C_UDPHS_DMA_2_DMAADDRESS ((AT91_REG *) 	0xFFFD4324) /* (UDPHS_DMA_2) UDPHS DMA Channel Address Register*/
/* ========== Register definition for UDPHS_DMA_3 peripheral ========== */
#define AT91C_UDPHS_DMA_3_DMACONTROL ((AT91_REG *) 	0xFFFD4338) /* (UDPHS_DMA_3) UDPHS DMA Channel Control Register*/
#define AT91C_UDPHS_DMA_3_DMAADDRESS ((AT91_REG *) 	0xFFFD4334) /* (UDPHS_DMA_3) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_3_DMANXTDSC ((AT91_REG *) 	0xFFFD4330) /* (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address*/
#define AT91C_UDPHS_DMA_3_DMASTATUS ((AT91_REG *) 	0xFFFD433C) /* (UDPHS_DMA_3) UDPHS DMA Channel Status Register*/
/* ========== Register definition for UDPHS_DMA_4 peripheral ========== */
#define AT91C_UDPHS_DMA_4_DMASTATUS ((AT91_REG *) 	0xFFFD434C) /* (UDPHS_DMA_4) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_4_DMANXTDSC ((AT91_REG *) 	0xFFFD4340) /* (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address*/
#define AT91C_UDPHS_DMA_4_DMAADDRESS ((AT91_REG *) 	0xFFFD4344) /* (UDPHS_DMA_4) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_4_DMACONTROL ((AT91_REG *) 	0xFFFD4348) /* (UDPHS_DMA_4) UDPHS DMA Channel Control Register*/
/* ========== Register definition for UDPHS_DMA_5 peripheral ========== */
#define AT91C_UDPHS_DMA_5_DMASTATUS ((AT91_REG *) 	0xFFFD435C) /* (UDPHS_DMA_5) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_5_DMACONTROL ((AT91_REG *) 	0xFFFD4358) /* (UDPHS_DMA_5) UDPHS DMA Channel Control Register*/
#define AT91C_UDPHS_DMA_5_DMAADDRESS ((AT91_REG *) 	0xFFFD4354) /* (UDPHS_DMA_5) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_5_DMANXTDSC ((AT91_REG *) 	0xFFFD4350) /* (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address*/
/* ========== Register definition for UDPHS_DMA_6 peripheral ========== */
#define AT91C_UDPHS_DMA_6_DMAADDRESS ((AT91_REG *) 	0xFFFD4364) /* (UDPHS_DMA_6) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_6_DMACONTROL ((AT91_REG *) 	0xFFFD4368) /* (UDPHS_DMA_6) UDPHS DMA Channel Control Register*/
#define AT91C_UDPHS_DMA_6_DMASTATUS ((AT91_REG *) 	0xFFFD436C) /* (UDPHS_DMA_6) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_6_DMANXTDSC ((AT91_REG *) 	0xFFFD4360) /* (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address*/
/* ========== Register definition for UDPHS_DMA_7 peripheral ========== */
#define AT91C_UDPHS_DMA_7_DMANXTDSC ((AT91_REG *) 	0xFFFD4370) /* (UDPHS_DMA_7) UDPHS DMA Channel Next Descriptor Address*/
#define AT91C_UDPHS_DMA_7_DMAADDRESS ((AT91_REG *) 	0xFFFD4374) /* (UDPHS_DMA_7) UDPHS DMA Channel Address Register*/
#define AT91C_UDPHS_DMA_7_DMASTATUS ((AT91_REG *) 	0xFFFD437C) /* (UDPHS_DMA_7) UDPHS DMA Channel Status Register*/
#define AT91C_UDPHS_DMA_7_DMACONTROL ((AT91_REG *) 	0xFFFD4378) /* (UDPHS_DMA_7) UDPHS DMA Channel Control Register*/
/* ========== Register definition for UDPHS peripheral ========== */
#define AT91C_UDPHS_TSTMODREG ((AT91_REG *) 	0xFFFD40DC) /* (UDPHS) UDPHS Test Mode Register*/
#define AT91C_UDPHS_RIPNAME2 ((AT91_REG *) 	0xFFFD40F4) /* (UDPHS) UDPHS Name2 Register*/
#define AT91C_UDPHS_TSTSOFCNT ((AT91_REG *) 	0xFFFD40D0) /* (UDPHS) UDPHS Test SOF Counter Register*/
#define AT91C_UDPHS_EPTRST ((AT91_REG *) 	0xFFFD401C) /* (UDPHS) UDPHS Endpoints Reset Register*/
#define AT91C_UDPHS_TSTCNTA ((AT91_REG *) 	0xFFFD40D4) /* (UDPHS) UDPHS Test A Counter Register*/
#define AT91C_UDPHS_IEN ((AT91_REG *) 	0xFFFD4010) /* (UDPHS) UDPHS Interrupt Enable Register*/
#define AT91C_UDPHS_TSTCNTB ((AT91_REG *) 	0xFFFD40D8) /* (UDPHS) UDPHS Test B Counter Register*/
#define AT91C_UDPHS_TST ((AT91_REG *) 	0xFFFD40E0) /* (UDPHS) UDPHS Test Register*/
#define AT91C_UDPHS_IPFEATURES ((AT91_REG *) 	0xFFFD40F8) /* (UDPHS) UDPHS Features Register*/
#define AT91C_UDPHS_RIPNAME1 ((AT91_REG *) 	0xFFFD40F0) /* (UDPHS) UDPHS Name1 Register*/
#define AT91C_UDPHS_FNUM ((AT91_REG *) 	0xFFFD4004) /* (UDPHS) UDPHS Frame Number Register*/
#define AT91C_UDPHS_CLRINT ((AT91_REG *) 	0xFFFD4018) /* (UDPHS) UDPHS Clear Interrupt Register*/
#define AT91C_UDPHS_IPVERSION ((AT91_REG *) 	0xFFFD40FC) /* (UDPHS) UDPHS Version Register*/
#define AT91C_UDPHS_RIPPADDRSIZE ((AT91_REG *) 	0xFFFD40EC) /* (UDPHS) UDPHS PADDRSIZE Register*/
#define AT91C_UDPHS_CTRL ((AT91_REG *) 	0xFFFD4000) /* (UDPHS) UDPHS Control Register*/
#define AT91C_UDPHS_INTSTA ((AT91_REG *) 	0xFFFD4014) /* (UDPHS) UDPHS Interrupt Status Register*/
/* ========== Register definition for PDC_AC97C peripheral ========== */
#define AT91C_AC97C_PTSR ((AT91_REG *) 	0xFFFD8124) /* (PDC_AC97C) PDC Transfer Status Register*/
#define AT91C_AC97C_PTCR ((AT91_REG *) 	0xFFFD8120) /* (PDC_AC97C) PDC Transfer Control Register*/
#define AT91C_AC97C_TNPR ((AT91_REG *) 	0xFFFD8118) /* (PDC_AC97C) Transmit Next Pointer Register*/
#define AT91C_AC97C_TNCR ((AT91_REG *) 	0xFFFD811C) /* (PDC_AC97C) Transmit Next Counter Register*/
#define AT91C_AC97C_RNPR ((AT91_REG *) 	0xFFFD8110) /* (PDC_AC97C) Receive Next Pointer Register*/
#define AT91C_AC97C_RNCR ((AT91_REG *) 	0xFFFD8114) /* (PDC_AC97C) Receive Next Counter Register*/
#define AT91C_AC97C_RPR ((AT91_REG *) 	0xFFFD8100) /* (PDC_AC97C) Receive Pointer Register*/
#define AT91C_AC97C_TCR ((AT91_REG *) 	0xFFFD810C) /* (PDC_AC97C) Transmit Counter Register*/
#define AT91C_AC97C_TPR ((AT91_REG *) 	0xFFFD8108) /* (PDC_AC97C) Transmit Pointer Register*/
#define AT91C_AC97C_RCR ((AT91_REG *) 	0xFFFD8104) /* (PDC_AC97C) Receive Counter Register*/
/* ========== Register definition for AC97C peripheral ========== */
#define AT91C_AC97C_CBSR ((AT91_REG *) 	0xFFFD8038) /* (AC97C) Channel B Status Register*/
#define AT91C_AC97C_CBMR ((AT91_REG *) 	0xFFFD803C) /* (AC97C) Channel B Mode Register*/
#define AT91C_AC97C_CBRHR ((AT91_REG *) 	0xFFFD8030) /* (AC97C) Channel B Receive Holding Register (optional)*/
#define AT91C_AC97C_COTHR ((AT91_REG *) 	0xFFFD8044) /* (AC97C) COdec Transmit Holding Register*/
#define AT91C_AC97C_OCA ((AT91_REG *) 	0xFFFD8014) /* (AC97C) Output Channel Assignement Register*/
#define AT91C_AC97C_IMR ((AT91_REG *) 	0xFFFD805C) /* (AC97C) Interrupt Mask Register*/
#define AT91C_AC97C_CORHR ((AT91_REG *) 	0xFFFD8040) /* (AC97C) COdec Transmit Holding Register*/
#define AT91C_AC97C_CBTHR ((AT91_REG *) 	0xFFFD8034) /* (AC97C) Channel B Transmit Holding Register (optional)*/
#define AT91C_AC97C_CARHR ((AT91_REG *) 	0xFFFD8020) /* (AC97C) Channel A Receive Holding Register*/
#define AT91C_AC97C_CASR ((AT91_REG *) 	0xFFFD8028) /* (AC97C) Channel A Status Register*/
#define AT91C_AC97C_IER ((AT91_REG *) 	0xFFFD8054) /* (AC97C) Interrupt Enable Register*/
#define AT91C_AC97C_MR  ((AT91_REG *) 	0xFFFD8008) /* (AC97C) Mode Register*/
#define AT91C_AC97C_COSR ((AT91_REG *) 	0xFFFD8048) /* (AC97C) CODEC Status Register*/
#define AT91C_AC97C_COMR ((AT91_REG *) 	0xFFFD804C) /* (AC97C) CODEC Mask Status Register*/
#define AT91C_AC97C_CATHR ((AT91_REG *) 	0xFFFD8024) /* (AC97C) Channel A Transmit Holding Register*/
#define AT91C_AC97C_ICA ((AT91_REG *) 	0xFFFD8010) /* (AC97C) Input Channel AssignementRegister*/
#define AT91C_AC97C_IDR ((AT91_REG *) 	0xFFFD8058) /* (AC97C) Interrupt Disable Register*/
#define AT91C_AC97C_CAMR ((AT91_REG *) 	0xFFFD802C) /* (AC97C) Channel A Mode Register*/
#define AT91C_AC97C_VERSION ((AT91_REG *) 	0xFFFD80FC) /* (AC97C) Version Register*/
#define AT91C_AC97C_SR  ((AT91_REG *) 	0xFFFD8050) /* (AC97C) Status Register*/
/* ========== Register definition for LCDC peripheral ========== */
#define AT91C_LCDC_MVAL ((AT91_REG *) 	0x00500818) /* (LCDC) LCD Mode Toggle Rate Value Register*/
#define AT91C_LCDC_PWRCON ((AT91_REG *) 	0x0050083C) /* (LCDC) Power Control Register*/
#define AT91C_LCDC_ISR  ((AT91_REG *) 	0x00500854) /* (LCDC) Interrupt Enable Register*/
#define AT91C_LCDC_FRMP1 ((AT91_REG *) 	0x00500008) /* (LCDC) DMA Frame Pointer Register 1*/
#define AT91C_LCDC_CTRSTVAL ((AT91_REG *) 	0x00500844) /* (LCDC) Contrast Value Register*/
#define AT91C_LCDC_ICR  ((AT91_REG *) 	0x00500858) /* (LCDC) Interrupt Clear Register*/
#define AT91C_LCDC_TIM1 ((AT91_REG *) 	0x00500808) /* (LCDC) LCD Timing Config 1 Register*/
#define AT91C_LCDC_DMACON ((AT91_REG *) 	0x0050001C) /* (LCDC) DMA Control Register*/
#define AT91C_LCDC_ITR  ((AT91_REG *) 	0x00500860) /* (LCDC) Interrupts Test Register*/
#define AT91C_LCDC_IDR  ((AT91_REG *) 	0x0050084C) /* (LCDC) Interrupt Disable Register*/
#define AT91C_LCDC_DP4_7 ((AT91_REG *) 	0x00500820) /* (LCDC) Dithering Pattern DP4_7 Register*/
#define AT91C_LCDC_DP5_7 ((AT91_REG *) 	0x0050082C) /* (LCDC) Dithering Pattern DP5_7 Register*/
#define AT91C_LCDC_IRR  ((AT91_REG *) 	0x00500864) /* (LCDC) Interrupts Raw Status Register*/
#define AT91C_LCDC_DP3_4 ((AT91_REG *) 	0x00500830) /* (LCDC) Dithering Pattern DP3_4 Register*/
#define AT91C_LCDC_IMR  ((AT91_REG *) 	0x00500850) /* (LCDC) Interrupt Mask Register*/
#define AT91C_LCDC_LCDFRCFG ((AT91_REG *) 	0x00500810) /* (LCDC) LCD Frame Config Register*/
#define AT91C_LCDC_CTRSTCON ((AT91_REG *) 	0x00500840) /* (LCDC) Contrast Control Register*/
#define AT91C_LCDC_DP1_2 ((AT91_REG *) 	0x0050081C) /* (LCDC) Dithering Pattern DP1_2 Register*/
#define AT91C_LCDC_FRMP2 ((AT91_REG *) 	0x0050000C) /* (LCDC) DMA Frame Pointer Register 2*/
#define AT91C_LCDC_LCDCON1 ((AT91_REG *) 	0x00500800) /* (LCDC) LCD Control 1 Register*/
#define AT91C_LCDC_DP4_5 ((AT91_REG *) 	0x00500834) /* (LCDC) Dithering Pattern DP4_5 Register*/
#define AT91C_LCDC_FRMA2 ((AT91_REG *) 	0x00500014) /* (LCDC) DMA Frame Address Register 2*/
#define AT91C_LCDC_BA1  ((AT91_REG *) 	0x00500000) /* (LCDC) DMA Base Address Register 1*/
#define AT91C_LCDC_DMA2DCFG ((AT91_REG *) 	0x00500020) /* (LCDC) DMA 2D addressing configuration*/
#define AT91C_LCDC_LUT_ENTRY ((AT91_REG *) 	0x00500C00) /* (LCDC) LUT Entries Register*/
#define AT91C_LCDC_DP6_7 ((AT91_REG *) 	0x00500838) /* (LCDC) Dithering Pattern DP6_7 Register*/
#define AT91C_LCDC_FRMCFG ((AT91_REG *) 	0x00500018) /* (LCDC) DMA Frame Configuration Register*/
#define AT91C_LCDC_TIM2 ((AT91_REG *) 	0x0050080C) /* (LCDC) LCD Timing Config 2 Register*/
#define AT91C_LCDC_DP3_5 ((AT91_REG *) 	0x00500824) /* (LCDC) Dithering Pattern DP3_5 Register*/
#define AT91C_LCDC_FRMA1 ((AT91_REG *) 	0x00500010) /* (LCDC) DMA Frame Address Register 1*/
#define AT91C_LCDC_IER  ((AT91_REG *) 	0x00500848) /* (LCDC) Interrupt Enable Register*/
#define AT91C_LCDC_DP2_3 ((AT91_REG *) 	0x00500828) /* (LCDC) Dithering Pattern DP2_3 Register*/
#define AT91C_LCDC_FIFO ((AT91_REG *) 	0x00500814) /* (LCDC) LCD FIFO Register*/
#define AT91C_LCDC_BA2  ((AT91_REG *) 	0x00500004) /* (LCDC) DMA Base Address Register 2*/
#define AT91C_LCDC_LCDCON2 ((AT91_REG *) 	0x00500804) /* (LCDC) LCD Control 2 Register*/
#define AT91C_LCDC_GPR  ((AT91_REG *) 	0x0050085C) /* (LCDC) General Purpose Register*/
/* ========== Register definition for LCDC_16B_TFT peripheral ========== */
#define AT91C_TFT_MVAL  ((AT91_REG *) 	0x00500818) /* (LCDC_16B_TFT) LCD Mode Toggle Rate Value Register*/
#define AT91C_TFT_PWRCON ((AT91_REG *) 	0x0050083C) /* (LCDC_16B_TFT) Power Control Register*/
#define AT91C_TFT_ISR   ((AT91_REG *) 	0x00500854) /* (LCDC_16B_TFT) Interrupt Enable Register*/
#define AT91C_TFT_FRMP1 ((AT91_REG *) 	0x00500008) /* (LCDC_16B_TFT) DMA Frame Pointer Register 1*/
#define AT91C_TFT_CTRSTVAL ((AT91_REG *) 	0x00500844) /* (LCDC_16B_TFT) Contrast Value Register*/
#define AT91C_TFT_ICR   ((AT91_REG *) 	0x00500858) /* (LCDC_16B_TFT) Interrupt Clear Register*/
#define AT91C_TFT_TIM1  ((AT91_REG *) 	0x00500808) /* (LCDC_16B_TFT) LCD Timing Config 1 Register*/
#define AT91C_TFT_DMACON ((AT91_REG *) 	0x0050001C) /* (LCDC_16B_TFT) DMA Control Register*/
#define AT91C_TFT_ITR   ((AT91_REG *) 	0x00500860) /* (LCDC_16B_TFT) Interrupts Test Register*/
#define AT91C_TFT_IDR   ((AT91_REG *) 	0x0050084C) /* (LCDC_16B_TFT) Interrupt Disable Register*/
#define AT91C_TFT_DP4_7 ((AT91_REG *) 	0x00500820) /* (LCDC_16B_TFT) Dithering Pattern DP4_7 Register*/
#define AT91C_TFT_DP5_7 ((AT91_REG *) 	0x0050082C) /* (LCDC_16B_TFT) Dithering Pattern DP5_7 Register*/
#define AT91C_TFT_IRR   ((AT91_REG *) 	0x00500864) /* (LCDC_16B_TFT) Interrupts Raw Status Register*/
#define AT91C_TFT_DP3_4 ((AT91_REG *) 	0x00500830) /* (LCDC_16B_TFT) Dithering Pattern DP3_4 Register*/
#define AT91C_TFT_IMR   ((AT91_REG *) 	0x00500850) /* (LCDC_16B_TFT) Interrupt Mask Register*/
#define AT91C_TFT_LCDFRCFG ((AT91_REG *) 	0x00500810) /* (LCDC_16B_TFT) LCD Frame Config Register*/
#define AT91C_TFT_CTRSTCON ((AT91_REG *) 	0x00500840) /* (LCDC_16B_TFT) Contrast Control Register*/
#define AT91C_TFT_DP1_2 ((AT91_REG *) 	0x0050081C) /* (LCDC_16B_TFT) Dithering Pattern DP1_2 Register*/
#define AT91C_TFT_FRMP2 ((AT91_REG *) 	0x0050000C) /* (LCDC_16B_TFT) DMA Frame Pointer Register 2*/
#define AT91C_TFT_LCDCON1 ((AT91_REG *) 	0x00500800) /* (LCDC_16B_TFT) LCD Control 1 Register*/
#define AT91C_TFT_DP4_5 ((AT91_REG *) 	0x00500834) /* (LCDC_16B_TFT) Dithering Pattern DP4_5 Register*/
#define AT91C_TFT_FRMA2 ((AT91_REG *) 	0x00500014) /* (LCDC_16B_TFT) DMA Frame Address Register 2*/
#define AT91C_TFT_BA1   ((AT91_REG *) 	0x00500000) /* (LCDC_16B_TFT) DMA Base Address Register 1*/
#define AT91C_TFT_DMA2DCFG ((AT91_REG *) 	0x00500020) /* (LCDC_16B_TFT) DMA 2D addressing configuration*/
#define AT91C_TFT_LUT_ENTRY ((AT91_REG *) 	0x00500C00) /* (LCDC_16B_TFT) LUT Entries Register*/
#define AT91C_TFT_DP6_7 ((AT91_REG *) 	0x00500838) /* (LCDC_16B_TFT) Dithering Pattern DP6_7 Register*/
#define AT91C_TFT_FRMCFG ((AT91_REG *) 	0x00500018) /* (LCDC_16B_TFT) DMA Frame Configuration Register*/
#define AT91C_TFT_TIM2  ((AT91_REG *) 	0x0050080C) /* (LCDC_16B_TFT) LCD Timing Config 2 Register*/
#define AT91C_TFT_DP3_5 ((AT91_REG *) 	0x00500824) /* (LCDC_16B_TFT) Dithering Pattern DP3_5 Register*/
#define AT91C_TFT_FRMA1 ((AT91_REG *) 	0x00500010) /* (LCDC_16B_TFT) DMA Frame Address Register 1*/
#define AT91C_TFT_IER   ((AT91_REG *) 	0x00500848) /* (LCDC_16B_TFT) Interrupt Enable Register*/
#define AT91C_TFT_DP2_3 ((AT91_REG *) 	0x00500828) /* (LCDC_16B_TFT) Dithering Pattern DP2_3 Register*/
#define AT91C_TFT_FIFO  ((AT91_REG *) 	0x00500814) /* (LCDC_16B_TFT) LCD FIFO Register*/
#define AT91C_TFT_BA2   ((AT91_REG *) 	0x00500004) /* (LCDC_16B_TFT) DMA Base Address Register 2*/
#define AT91C_TFT_LCDCON2 ((AT91_REG *) 	0x00500804) /* (LCDC_16B_TFT) LCD Control 2 Register*/
#define AT91C_TFT_GPR   ((AT91_REG *) 	0x0050085C) /* (LCDC_16B_TFT) General Purpose Register*/
/* ========== Register definition for HDMA_CH_0 peripheral ========== */
#define AT91C_HDMA_CH_0_CFG ((AT91_REG *) 	0xFFFFE650) /* (HDMA_CH_0) HDMA Channel Configuration Register*/
#define AT91C_HDMA_CH_0_CTRLB ((AT91_REG *) 	0xFFFFE64C) /* (HDMA_CH_0) HDMA Channel Control B Register*/
#define AT91C_HDMA_CH_0_DADDR ((AT91_REG *) 	0xFFFFE640) /* (HDMA_CH_0) HDMA Channel Destination Address Register*/
#define AT91C_HDMA_CH_0_DPIP ((AT91_REG *) 	0xFFFFE658) /* (HDMA_CH_0) HDMA Channel Destination Picture in Picture Configuration Register*/
#define AT91C_HDMA_CH_0_SPIP ((AT91_REG *) 	0xFFFFE654) /* (HDMA_CH_0) HDMA Channel Source Picture in Picture Configuration Register*/
#define AT91C_HDMA_CH_0_DSCR ((AT91_REG *) 	0xFFFFE644) /* (HDMA_CH_0) HDMA Channel Descriptor Address Register*/
#define AT91C_HDMA_CH_0_BDSCR ((AT91_REG *) 	0xFFFFE65C) /* (HDMA_CH_0) HDMA Reserved*/
#define AT91C_HDMA_CH_0_SADDR ((AT91_REG *) 	0xFFFFE63C) /* (HDMA_CH_0) HDMA Channel Source Address Register*/
#define AT91C_HDMA_CH_0_CTRLA ((AT91_REG *) 	0xFFFFE648) /* (HDMA_CH_0) HDMA Channel Control A Register*/
#define AT91C_HDMA_CH_0_CADDR ((AT91_REG *) 	0xFFFFE660) /* (HDMA_CH_0) HDMA Reserved*/
/* ========== Register definition for HDMA_CH_1 peripheral ========== */
#define AT91C_HDMA_CH_1_DADDR ((AT91_REG *) 	0xFFFFE668) /* (HDMA_CH_1) HDMA Channel Destination Address Register*/
#define AT91C_HDMA_CH_1_BDSCR ((AT91_REG *) 	0xFFFFE684) /* (HDMA_CH_1) HDMA Reserved*/
#define AT91C_HDMA_CH_1_CFG ((AT91_REG *) 	0xFFFFE678) /* (HDMA_CH_1) HDMA Channel Configuration Register*/
#define AT91C_HDMA_CH_1_CTRLB ((AT91_REG *) 	0xFFFFE674) /* (HDMA_CH_1) HDMA Channel Control B Register*/
#define AT91C_HDMA_CH_1_SADDR ((AT91_REG *) 	0xFFFFE664) /* (HDMA_CH_1) HDMA Channel Source Address Register*/
#define AT91C_HDMA_CH_1_CTRLA ((AT91_REG *) 	0xFFFFE670) /* (HDMA_CH_1) HDMA Channel Control A Register*/
#define AT91C_HDMA_CH_1_CADDR ((AT91_REG *) 	0xFFFFE688) /* (HDMA_CH_1) HDMA Reserved*/
#define AT91C_HDMA_CH_1_SPIP ((AT91_REG *) 	0xFFFFE67C) /* (HDMA_CH_1) HDMA Channel Source Picture in Picture Configuration Register*/
#define AT91C_HDMA_CH_1_DSCR ((AT91_REG *) 	0xFFFFE66C) /* (HDMA_CH_1) HDMA Channel Descriptor Address Register*/
#define AT91C_HDMA_CH_1_DPIP ((AT91_REG *) 	0xFFFFE680) /* (HDMA_CH_1) HDMA Channel Destination Picture in Picture Configuration Register*/
/* ========== Register definition for HDMA peripheral ========== */
#define AT91C_HDMA_CHER ((AT91_REG *) 	0xFFFFE628) /* (HDMA) HDMA Channel Handler Enable Register*/
#define AT91C_HDMA_EN   ((AT91_REG *) 	0xFFFFE604) /* (HDMA) HDMA Controller Enable Register*/
#define AT91C_HDMA_EBCIMR ((AT91_REG *) 	0xFFFFE620) /* (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register*/
#define AT91C_HDMA_BREQ ((AT91_REG *) 	0xFFFFE60C) /* (HDMA) HDMA Software Chunk Transfer Request Register*/
#define AT91C_HDMA_EBCIDR ((AT91_REG *) 	0xFFFFE61C) /* (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register*/
#define AT91C_HDMA_CHDR ((AT91_REG *) 	0xFFFFE62C) /* (HDMA) HDMA Channel Handler Disable Register*/
#define AT91C_HDMA_RSVD0 ((AT91_REG *) 	0xFFFFE634) /* (HDMA) HDMA Reserved*/
#define AT91C_HDMA_SYNC ((AT91_REG *) 	0xFFFFE614) /* (HDMA) HDMA Request Synchronization Register*/
#define AT91C_HDMA_GCFG ((AT91_REG *) 	0xFFFFE600) /* (HDMA) HDMA Global Configuration Register*/
#define AT91C_HDMA_EBCISR ((AT91_REG *) 	0xFFFFE624) /* (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register*/
#define AT91C_HDMA_CHSR ((AT91_REG *) 	0xFFFFE630) /* (HDMA) HDMA Channel Handler Status Register*/
#define AT91C_HDMA_LAST ((AT91_REG *) 	0xFFFFE610) /* (HDMA) HDMA Software Last Transfer Flag Register*/
#define AT91C_HDMA_RSVD1 ((AT91_REG *) 	0xFFFFE638) /* (HDMA) HDMA Reserved*/
#define AT91C_HDMA_SREQ ((AT91_REG *) 	0xFFFFE608) /* (HDMA) HDMA Software Single Request Register*/
#define AT91C_HDMA_EBCIER ((AT91_REG *) 	0xFFFFE618) /* (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register*/
/* ========== Register definition for HECC peripheral ========== */
#define AT91C_HECC_VR   ((AT91_REG *) 	0xFFFFE8FC) /* (HECC)  ECC Version register*/
#define AT91C_HECC_NPR  ((AT91_REG *) 	0xFFFFE810) /* (HECC)  ECC Parity N register*/
#define AT91C_HECC_SR   ((AT91_REG *) 	0xFFFFE808) /* (HECC)  ECC Status register*/
#define AT91C_HECC_PR   ((AT91_REG *) 	0xFFFFE80C) /* (HECC)  ECC Parity register*/
#define AT91C_HECC_MR   ((AT91_REG *) 	0xFFFFE804) /* (HECC)  ECC Page size register*/
#define AT91C_HECC_CR   ((AT91_REG *) 	0xFFFFE800) /* (HECC)  ECC reset register*/

/* ******************************************************************************/
/*               PIO DEFINITIONS FOR AT91SAM9RL64*/
/* ******************************************************************************/
#define AT91C_PIO_PA0        ((unsigned int) 1 <<  0) /* Pin Controlled by PA0*/
#define AT91C_PA0_MC_DA0   ((unsigned int) AT91C_PIO_PA0) /*  */
#define AT91C_PIO_PA1        ((unsigned int) 1 <<  1) /* Pin Controlled by PA1*/
#define AT91C_PA1_MC_CDA   ((unsigned int) AT91C_PIO_PA1) /*  */
#define AT91C_PIO_PA10       ((unsigned int) 1 << 10) /* Pin Controlled by PA10*/
#define AT91C_PA10_CTS0     ((unsigned int) AT91C_PIO_PA10) /*  */
#define AT91C_PA10_RK0      ((unsigned int) AT91C_PIO_PA10) /*  */
#define AT91C_PIO_PA11       ((unsigned int) 1 << 11) /* Pin Controlled by PA11*/
#define AT91C_PA11_TXD1     ((unsigned int) AT91C_PIO_PA11) /*  */
#define AT91C_PIO_PA12       ((unsigned int) 1 << 12) /* Pin Controlled by PA12*/
#define AT91C_PA12_RXD1     ((unsigned int) AT91C_PIO_PA12) /*  */
#define AT91C_PIO_PA13       ((unsigned int) 1 << 13) /* Pin Controlled by PA13*/
#define AT91C_PA13_TXD2     ((unsigned int) AT91C_PIO_PA13) /*  */
#define AT91C_PA13_TD1      ((unsigned int) AT91C_PIO_PA13) /*  */
#define AT91C_PIO_PA14       ((unsigned int) 1 << 14) /* Pin Controlled by PA14*/
#define AT91C_PA14_RXD2     ((unsigned int) AT91C_PIO_PA14) /*  */
#define AT91C_PA14_RD1      ((unsigned int) AT91C_PIO_PA14) /*  */
#define AT91C_PIO_PA15       ((unsigned int) 1 << 15) /* Pin Controlled by PA15*/
#define AT91C_PA15_TD0      ((unsigned int) AT91C_PIO_PA15) /*  */
#define AT91C_PIO_PA16       ((unsigned int) 1 << 16) /* Pin Controlled by PA16*/
#define AT91C_PA16_RD0      ((unsigned int) AT91C_PIO_PA16) /*  */
#define AT91C_PIO_PA17       ((unsigned int) 1 << 17) /* Pin Controlled by PA17*/
#define AT91C_PA17_AD0      ((unsigned int) AT91C_PIO_PA17) /*  */
#define AT91C_PIO_PA18       ((unsigned int) 1 << 18) /* Pin Controlled by PA18*/
#define AT91C_PA18_AD1      ((unsigned int) AT91C_PIO_PA18) /*  */
#define AT91C_PA18_RTS1     ((unsigned int) AT91C_PIO_PA18) /*  */
#define AT91C_PIO_PA19       ((unsigned int) 1 << 19) /* Pin Controlled by PA19*/
#define AT91C_PA19_AD2      ((unsigned int) AT91C_PIO_PA19) /*  */
#define AT91C_PA19_CTS1     ((unsigned int) AT91C_PIO_PA19) /*  */
#define AT91C_PIO_PA2        ((unsigned int) 1 <<  2) /* Pin Controlled by PA2*/
#define AT91C_PA2_MC_CK    ((unsigned int) AT91C_PIO_PA2) /*  */
#define AT91C_PIO_PA20       ((unsigned int) 1 << 20) /* Pin Controlled by PA20*/
#define AT91C_PA20_AD3      ((unsigned int) AT91C_PIO_PA20) /*  */
#define AT91C_PA20_SCK3     ((unsigned int) AT91C_PIO_PA20) /*  */
#define AT91C_PIO_PA21       ((unsigned int) 1 << 21) /* Pin Controlled by PA21*/
#define AT91C_PA21_DRXD     ((unsigned int) AT91C_PIO_PA21) /*  */
#define AT91C_PIO_PA22       ((unsigned int) 1 << 22) /* Pin Controlled by PA22*/
#define AT91C_PA22_DTXD     ((unsigned int) AT91C_PIO_PA22) /*  */
#define AT91C_PA22_RF0      ((unsigned int) AT91C_PIO_PA22) /*  */
#define AT91C_PIO_PA23       ((unsigned int) 1 << 23) /* Pin Controlled by PA23*/
#define AT91C_PA23_TWD0     ((unsigned int) AT91C_PIO_PA23) /*  */
#define AT91C_PIO_PA24       ((unsigned int) 1 << 24) /* Pin Controlled by PA24*/
#define AT91C_PA24_TWCK0    ((unsigned int) AT91C_PIO_PA24) /*  */
#define AT91C_PIO_PA25       ((unsigned int) 1 << 25) /* Pin Controlled by PA25*/
#define AT91C_PA25_MISO     ((unsigned int) AT91C_PIO_PA25) /*  */
#define AT91C_PIO_PA26       ((unsigned int) 1 << 26) /* Pin Controlled by PA26*/
#define AT91C_PA26_MOSI     ((unsigned int) AT91C_PIO_PA26) /*  */
#define AT91C_PIO_PA27       ((unsigned int) 1 << 27) /* Pin Controlled by PA27*/
#define AT91C_PA27_SPCK     ((unsigned int) AT91C_PIO_PA27) /*  */
#define AT91C_PIO_PA28       ((unsigned int) 1 << 28) /* Pin Controlled by PA28*/
#define AT91C_PA28_NPCS0    ((unsigned int) AT91C_PIO_PA28) /*  */
#define AT91C_PIO_PA29       ((unsigned int) 1 << 29) /* Pin Controlled by PA29*/
#define AT91C_PA29_RTS2     ((unsigned int) AT91C_PIO_PA29) /*  */
#define AT91C_PA29_TF1      ((unsigned int) AT91C_PIO_PA29) /*  */
#define AT91C_PIO_PA3        ((unsigned int) 1 <<  3) /* Pin Controlled by PA3*/
#define AT91C_PA3_MC_DA1   ((unsigned int) AT91C_PIO_PA3) /*  */
#define AT91C_PA3_TCLK0    ((unsigned int) AT91C_PIO_PA3) /*  */
#define AT91C_PIO_PA30       ((unsigned int) 1 << 30) /* Pin Controlled by PA30*/
#define AT91C_PA30_CTS2     ((unsigned int) AT91C_PIO_PA30) /*  */
#define AT91C_PA30_TK1      ((unsigned int) AT91C_PIO_PA30) /*  */
#define AT91C_PIO_PA31       ((unsigned int) 1 << 31) /* Pin Controlled by PA31*/
#define AT91C_PA31_NWAIT    ((unsigned int) AT91C_PIO_PA31) /*  */
#define AT91C_PA31_IRQ      ((unsigned int) AT91C_PIO_PA31) /*  */
#define AT91C_PIO_PA4        ((unsigned int) 1 <<  4) /* Pin Controlled by PA4*/
#define AT91C_PA4_MC_DA2   ((unsigned int) AT91C_PIO_PA4) /*  */
#define AT91C_PA4_TIOA0    ((unsigned int) AT91C_PIO_PA4) /*  */
#define AT91C_PIO_PA5        ((unsigned int) 1 <<  5) /* Pin Controlled by PA5*/
#define AT91C_PA5_MC_DA3   ((unsigned int) AT91C_PIO_PA5) /*  */
#define AT91C_PA5_TIOB0    ((unsigned int) AT91C_PIO_PA5) /*  */
#define AT91C_PIO_PA6        ((unsigned int) 1 <<  6) /* Pin Controlled by PA6*/
#define AT91C_PA6_TXD0     ((unsigned int) AT91C_PIO_PA6) /*  */
#define AT91C_PIO_PA7        ((unsigned int) 1 <<  7) /* Pin Controlled by PA7*/
#define AT91C_PA7_RXD0     ((unsigned int) AT91C_PIO_PA7) /*  */
#define AT91C_PIO_PA8        ((unsigned int) 1 <<  8) /* Pin Controlled by PA8*/
#define AT91C_PA8_SCK0     ((unsigned int) AT91C_PIO_PA8) /*  */
#define AT91C_PA8_RF1      ((unsigned int) AT91C_PIO_PA8) /*  */
#define AT91C_PIO_PA9        ((unsigned int) 1 <<  9) /* Pin Controlled by PA9*/
#define AT91C_PA9_RTS0     ((unsigned int) AT91C_PIO_PA9) /*  */
#define AT91C_PA9_RK1      ((unsigned int) AT91C_PIO_PA9) /*  */
#define AT91C_PIO_PB0        ((unsigned int) 1 <<  0) /* Pin Controlled by PB0*/
#define AT91C_PB0_TXD3     ((unsigned int) AT91C_PIO_PB0) /*  */
#define AT91C_PIO_PB1        ((unsigned int) 1 <<  1) /* Pin Controlled by PB1*/
#define AT91C_PB1_RXD3     ((unsigned int) AT91C_PIO_PB1) /*  */
#define AT91C_PIO_PB10       ((unsigned int) 1 << 10) /* Pin Controlled by PB10*/
#define AT91C_PB10_A25_CFRNW ((unsigned int) AT91C_PIO_PB10) /*  */
#define AT91C_PB10_FIQ      ((unsigned int) AT91C_PIO_PB10) /*  */
#define AT91C_PIO_PB11       ((unsigned int) 1 << 11) /* Pin Controlled by PB11*/
#define AT91C_PB11_A18      ((unsigned int) AT91C_PIO_PB11) /*  */
#define AT91C_PIO_PB12       ((unsigned int) 1 << 12) /* Pin Controlled by PB12*/
#define AT91C_PB12_A19      ((unsigned int) AT91C_PIO_PB12) /*  */
#define AT91C_PIO_PB13       ((unsigned int) 1 << 13) /* Pin Controlled by PB13*/
#define AT91C_PB13_A20      ((unsigned int) AT91C_PIO_PB13) /*  */
#define AT91C_PIO_PB14       ((unsigned int) 1 << 14) /* Pin Controlled by PB14*/
#define AT91C_PB14_A23      ((unsigned int) AT91C_PIO_PB14) /*  */
#define AT91C_PB14_PCK0     ((unsigned int) AT91C_PIO_PB14) /*  */
#define AT91C_PIO_PB15       ((unsigned int) 1 << 15) /* Pin Controlled by PB15*/
#define AT91C_PB15_A24      ((unsigned int) AT91C_PIO_PB15) /*  */
#define AT91C_PB15_ADTRG    ((unsigned int) AT91C_PIO_PB15) /*  */
#define AT91C_PIO_PB16       ((unsigned int) 1 << 16) /* Pin Controlled by PB16*/
#define AT91C_PB16_D16      ((unsigned int) AT91C_PIO_PB16) /*  */
#define AT91C_PIO_PB17       ((unsigned int) 1 << 17) /* Pin Controlled by PB17*/
#define AT91C_PB17_D17      ((unsigned int) AT91C_PIO_PB17) /*  */
#define AT91C_PIO_PB18       ((unsigned int) 1 << 18) /* Pin Controlled by PB18*/
#define AT91C_PB18_D18      ((unsigned int) AT91C_PIO_PB18) /*  */
#define AT91C_PIO_PB19       ((unsigned int) 1 << 19) /* Pin Controlled by PB19*/
#define AT91C_PB19_D19      ((unsigned int) AT91C_PIO_PB19) /*  */
#define AT91C_PIO_PB2        ((unsigned int) 1 <<  2) /* Pin Controlled by PB2*/
#define AT91C_PB2_A21_NANDALE ((unsigned int) AT91C_PIO_PB2) /*  */
#define AT91C_PIO_PB20       ((unsigned int) 1 << 20) /* Pin Controlled by PB20*/
#define AT91C_PB20_D20      ((unsigned int) AT91C_PIO_PB20) /*  */
#define AT91C_PIO_PB21       ((unsigned int) 1 << 21) /* Pin Controlled by PB21*/
#define AT91C_PB21_D21      ((unsigned int) AT91C_PIO_PB21) /*  */
#define AT91C_PIO_PB22       ((unsigned int) 1 << 22) /* Pin Controlled by PB22*/
#define AT91C_PB22_D22      ((unsigned int) AT91C_PIO_PB22) /*  */
#define AT91C_PIO_PB23       ((unsigned int) 1 << 23) /* Pin Controlled by PB23*/
#define AT91C_PB23_D23      ((unsigned int) AT91C_PIO_PB23) /*  */
#define AT91C_PIO_PB24       ((unsigned int) 1 << 24) /* Pin Controlled by PB24*/
#define AT91C_PB24_D24      ((unsigned int) AT91C_PIO_PB24) /*  */
#define AT91C_PIO_PB25       ((unsigned int) 1 << 25) /* Pin Controlled by PB25*/
#define AT91C_PB25_D25      ((unsigned int) AT91C_PIO_PB25) /*  */
#define AT91C_PIO_PB26       ((unsigned int) 1 << 26) /* Pin Controlled by PB26*/
#define AT91C_PB26_D26      ((unsigned int) AT91C_PIO_PB26) /*  */
#define AT91C_PIO_PB27       ((unsigned int) 1 << 27) /* Pin Controlled by PB27*/
#define AT91C_PB27_D27      ((unsigned int) AT91C_PIO_PB27) /*  */
#define AT91C_PIO_PB28       ((unsigned int) 1 << 28) /* Pin Controlled by PB28*/
#define AT91C_PB28_D28      ((unsigned int) AT91C_PIO_PB28) /*  */
#define AT91C_PIO_PB29       ((unsigned int) 1 << 29) /* Pin Controlled by PB29*/
#define AT91C_PB29_D29      ((unsigned int) AT91C_PIO_PB29) /*  */
#define AT91C_PIO_PB3        ((unsigned int) 1 <<  3) /* Pin Controlled by PB3*/
#define AT91C_PB3_A22_NANDCLE ((unsigned int) AT91C_PIO_PB3) /*  */
#define AT91C_PIO_PB30       ((unsigned int) 1 << 30) /* Pin Controlled by PB30*/
#define AT91C_PB30_D30      ((unsigned int) AT91C_PIO_PB30) /*  */
#define AT91C_PIO_PB31       ((unsigned int) 1 << 31) /* Pin Controlled by PB31*/
#define AT91C_PB31_D31      ((unsigned int) AT91C_PIO_PB31) /*  */
#define AT91C_PIO_PB4        ((unsigned int) 1 <<  4) /* Pin Controlled by PB4*/
#define AT91C_PB4_NANDOE   ((unsigned int) AT91C_PIO_PB4) /*  */
#define AT91C_PIO_PB5        ((unsigned int) 1 <<  5) /* Pin Controlled by PB5*/
#define AT91C_PB5_NANDWE   ((unsigned int) AT91C_PIO_PB5) /*  */
#define AT91C_PIO_PB6        ((unsigned int) 1 <<  6) /* Pin Controlled by PB6*/
#define AT91C_PB6_NCS3_NANDCS ((unsigned int) AT91C_PIO_PB6) /*  */
#define AT91C_PIO_PB7        ((unsigned int) 1 <<  7) /* Pin Controlled by PB7*/
#define AT91C_PB7_NCS4_CFCS0 ((unsigned int) AT91C_PIO_PB7) /*  */
#define AT91C_PB7_NPCS1    ((unsigned int) AT91C_PIO_PB7) /*  */
#define AT91C_PIO_PB8        ((unsigned int) 1 <<  8) /* Pin Controlled by PB8*/
#define AT91C_PB8_CFE1     ((unsigned int) AT91C_PIO_PB8) /*  */
#define AT91C_PB8_PWM0     ((unsigned int) AT91C_PIO_PB8) /*  */
#define AT91C_PIO_PB9        ((unsigned int) 1 <<  9) /* Pin Controlled by PB9*/
#define AT91C_PB9_CFE2     ((unsigned int) AT91C_PIO_PB9) /*  */
#define AT91C_PB9_PWM1     ((unsigned int) AT91C_PIO_PB9) /*  */
#define AT91C_PIO_PC0        ((unsigned int) 1 <<  0) /* Pin Controlled by PC0*/
#define AT91C_PC0_TF0      ((unsigned int) AT91C_PIO_PC0) /*  */
#define AT91C_PIO_PC1        ((unsigned int) 1 <<  1) /* Pin Controlled by PC1*/
#define AT91C_PC1_TK0      ((unsigned int) AT91C_PIO_PC1) /*  */
#define AT91C_PC1_LCDPWR   ((unsigned int) AT91C_PIO_PC1) /*  */
#define AT91C_PIO_PC10       ((unsigned int) 1 << 10) /* Pin Controlled by PC10*/
#define AT91C_PC10_LCDD2    ((unsigned int) AT91C_PIO_PC10) /*  */
#define AT91C_PC10_LCDD4    ((unsigned int) AT91C_PIO_PC10) /*  */
#define AT91C_PIO_PC11       ((unsigned int) 1 << 11) /* Pin Controlled by PC11*/
#define AT91C_PC11_LCDD3    ((unsigned int) AT91C_PIO_PC11) /*  */
#define AT91C_PC11_LCDD5    ((unsigned int) AT91C_PIO_PC11) /*  */
#define AT91C_PIO_PC12       ((unsigned int) 1 << 12) /* Pin Controlled by PC12*/
#define AT91C_PC12_LCDD4    ((unsigned int) AT91C_PIO_PC12) /*  */
#define AT91C_PC12_LCDD6    ((unsigned int) AT91C_PIO_PC12) /*  */
#define AT91C_PIO_PC13       ((unsigned int) 1 << 13) /* Pin Controlled by PC13*/
#define AT91C_PC13_LCDD5    ((unsigned int) AT91C_PIO_PC13) /*  */
#define AT91C_PC13_LCDD7    ((unsigned int) AT91C_PIO_PC13) /*  */
#define AT91C_PIO_PC14       ((unsigned int) 1 << 14) /* Pin Controlled by PC14*/
#define AT91C_PC14_LCDD6    ((unsigned int) AT91C_PIO_PC14) /*  */
#define AT91C_PC14_LCDD10   ((unsigned int) AT91C_PIO_PC14) /*  */
#define AT91C_PIO_PC15       ((unsigned int) 1 << 15) /* Pin Controlled by PC15*/
#define AT91C_PC15_LCDD7    ((unsigned int) AT91C_PIO_PC15) /*  */
#define AT91C_PC15_LCDD11   ((unsigned int) AT91C_PIO_PC15) /*  */
#define AT91C_PIO_PC16       ((unsigned int) 1 << 16) /* Pin Controlled by PC16*/
#define AT91C_PC16_LCDD8    ((unsigned int) AT91C_PIO_PC16) /*  */
#define AT91C_PC16_LCDD12   ((unsigned int) AT91C_PIO_PC16) /*  */
#define AT91C_PIO_PC17       ((unsigned int) 1 << 17) /* Pin Controlled by PC17*/
#define AT91C_PC17_LCDD9    ((unsigned int) AT91C_PIO_PC17) /*  */
#define AT91C_PC17_LCDD13   ((unsigned int) AT91C_PIO_PC17) /*  */
#define AT91C_PIO_PC18       ((unsigned int) 1 << 18) /* Pin Controlled by PC18*/
#define AT91C_PC18_LCDD10   ((unsigned int) AT91C_PIO_PC18) /*  */
#define AT91C_PC18_LCDD14   ((unsigned int) AT91C_PIO_PC18) /*  */
#define AT91C_PIO_PC19       ((unsigned int) 1 << 19) /* Pin Controlled by PC19*/
#define AT91C_PC19_LCDD11   ((unsigned int) AT91C_PIO_PC19) /*  */
#define AT91C_PC19_LCDD15   ((unsigned int) AT91C_PIO_PC19) /*  */
#define AT91C_PIO_PC2        ((unsigned int) 1 <<  2) /* Pin Controlled by PC2*/
#define AT91C_PC2_LCDMOD   ((unsigned int) AT91C_PIO_PC2) /*  */
#define AT91C_PC2_PWM0     ((unsigned int) AT91C_PIO_PC2) /*  */
#define AT91C_PIO_PC20       ((unsigned int) 1 << 20) /* Pin Controlled by PC20*/
#define AT91C_PC20_LCDD12   ((unsigned int) AT91C_PIO_PC20) /*  */
#define AT91C_PC20_LCDD18   ((unsigned int) AT91C_PIO_PC20) /*  */
#define AT91C_PIO_PC21       ((unsigned int) 1 << 21) /* Pin Controlled by PC21*/
#define AT91C_PC21_LCDD13   ((unsigned int) AT91C_PIO_PC21) /*  */
#define AT91C_PC21_LCDD19   ((unsigned int) AT91C_PIO_PC21) /*  */
#define AT91C_PIO_PC22       ((unsigned int) 1 << 22) /* Pin Controlled by PC22*/
#define AT91C_PC22_LCDD14   ((unsigned int) AT91C_PIO_PC22) /*  */
#define AT91C_PC22_LCDD20   ((unsigned int) AT91C_PIO_PC22) /*  */
#define AT91C_PIO_PC23       ((unsigned int) 1 << 23) /* Pin Controlled by PC23*/
#define AT91C_PC23_LCDD15   ((unsigned int) AT91C_PIO_PC23) /*  */
#define AT91C_PC23_LCDD21   ((unsigned int) AT91C_PIO_PC23) /*  */
#define AT91C_PIO_PC24       ((unsigned int) 1 << 24) /* Pin Controlled by PC24*/
#define AT91C_PC24_LCDD16   ((unsigned int) AT91C_PIO_PC24) /*  */
#define AT91C_PC24_LCDD22   ((unsigned int) AT91C_PIO_PC24) /*  */
#define AT91C_PIO_PC25       ((unsigned int) 1 << 25) /* Pin Controlled by PC25*/
#define AT91C_PC25_LCDD17   ((unsigned int) AT91C_PIO_PC25) /*  */
#define AT91C_PC25_LCDD23   ((unsigned int) AT91C_PIO_PC25) /*  */
#define AT91C_PIO_PC26       ((unsigned int) 1 << 26) /* Pin Controlled by PC26*/
#define AT91C_PC26_LCDD18   ((unsigned int) AT91C_PIO_PC26) /*  */
#define AT91C_PIO_PC27       ((unsigned int) 1 << 27) /* Pin Controlled by PC27*/
#define AT91C_PC27_LCDD19   ((unsigned int) AT91C_PIO_PC27) /*  */
#define AT91C_PIO_PC28       ((unsigned int) 1 << 28) /* Pin Controlled by PC28*/
#define AT91C_PC28_LCDD20   ((unsigned int) AT91C_PIO_PC28) /*  */
#define AT91C_PIO_PC29       ((unsigned int) 1 << 29) /* Pin Controlled by PC29*/
#define AT91C_PC29_LCDD21   ((unsigned int) AT91C_PIO_PC29) /*  */
#define AT91C_PC29_TIOA1    ((unsigned int) AT91C_PIO_PC29) /*  */
#define AT91C_PIO_PC3        ((unsigned int) 1 <<  3) /* Pin Controlled by PC3*/
#define AT91C_PC3_LCDCC    ((unsigned int) AT91C_PIO_PC3) /*  */
#define AT91C_PC3_PWM1     ((unsigned int) AT91C_PIO_PC3) /*  */
#define AT91C_PIO_PC30       ((unsigned int) 1 << 30) /* Pin Controlled by PC30*/
#define AT91C_PC30_LCDD22   ((unsigned int) AT91C_PIO_PC30) /*  */
#define AT91C_PC30_TIOB1    ((unsigned int) AT91C_PIO_PC30) /*  */
#define AT91C_PIO_PC31       ((unsigned int) 1 << 31) /* Pin Controlled by PC31*/
#define AT91C_PC31_LCDD23   ((unsigned int) AT91C_PIO_PC31) /*  */
#define AT91C_PC31_TCLK1    ((unsigned int) AT91C_PIO_PC31) /*  */
#define AT91C_PIO_PC4        ((unsigned int) 1 <<  4) /* Pin Controlled by PC4*/
#define AT91C_PC4_LCDVSYNC ((unsigned int) AT91C_PIO_PC4) /*  */
#define AT91C_PIO_PC5        ((unsigned int) 1 <<  5) /* Pin Controlled by PC5*/
#define AT91C_PC5_LCDHSYNC ((unsigned int) AT91C_PIO_PC5) /*  */
#define AT91C_PIO_PC6        ((unsigned int) 1 <<  6) /* Pin Controlled by PC6*/
#define AT91C_PC6_LCDDOTCK ((unsigned int) AT91C_PIO_PC6) /*  */
#define AT91C_PIO_PC7        ((unsigned int) 1 <<  7) /* Pin Controlled by PC7*/
#define AT91C_PC7_LCDDEN   ((unsigned int) AT91C_PIO_PC7) /*  */
#define AT91C_PIO_PC8        ((unsigned int) 1 <<  8) /* Pin Controlled by PC8*/
#define AT91C_PC8_LCDD0    ((unsigned int) AT91C_PIO_PC8) /*  */
#define AT91C_PC8_LCDD2    ((unsigned int) AT91C_PIO_PC8) /*  */
#define AT91C_PIO_PC9        ((unsigned int) 1 <<  9) /* Pin Controlled by PC9*/
#define AT91C_PC9_LCDD1    ((unsigned int) AT91C_PIO_PC9) /*  */
#define AT91C_PC9_LCDD3    ((unsigned int) AT91C_PIO_PC9) /*  */
#define AT91C_PIO_PD0        ((unsigned int) 1 <<  0) /* Pin Controlled by PD0*/
#define AT91C_PD0_NCS2     ((unsigned int) AT91C_PIO_PD0) /*  */
#define AT91C_PIO_PD1        ((unsigned int) 1 <<  1) /* Pin Controlled by PD1*/
#define AT91C_PD1_AC97_FS  ((unsigned int) AT91C_PIO_PD1) /*  */
#define AT91C_PIO_PD10       ((unsigned int) 1 << 10) /* Pin Controlled by PD10*/
#define AT91C_PD10_TWD1     ((unsigned int) AT91C_PIO_PD10) /*  */
#define AT91C_PD10_TIOA2    ((unsigned int) AT91C_PIO_PD10) /*  */
#define AT91C_PIO_PD11       ((unsigned int) 1 << 11) /* Pin Controlled by PD11*/
#define AT91C_PD11_TWCK1    ((unsigned int) AT91C_PIO_PD11) /*  */
#define AT91C_PD11_TIOB2    ((unsigned int) AT91C_PIO_PD11) /*  */
#define AT91C_PIO_PD12       ((unsigned int) 1 << 12) /* Pin Controlled by PD12*/
#define AT91C_PD12_PWM2     ((unsigned int) AT91C_PIO_PD12) /*  */
#define AT91C_PD12_PCK1     ((unsigned int) AT91C_PIO_PD12) /*  */
#define AT91C_PIO_PD13       ((unsigned int) 1 << 13) /* Pin Controlled by PD13*/
#define AT91C_PD13_NCS5_CFCS1 ((unsigned int) AT91C_PIO_PD13) /*  */
#define AT91C_PD13_NPCS3    ((unsigned int) AT91C_PIO_PD13) /*  */
#define AT91C_PIO_PD14       ((unsigned int) 1 << 14) /* Pin Controlled by PD14*/
#define AT91C_PD14_DSR0     ((unsigned int) AT91C_PIO_PD14) /*  */
#define AT91C_PD14_PWM0     ((unsigned int) AT91C_PIO_PD14) /*  */
#define AT91C_PIO_PD15       ((unsigned int) 1 << 15) /* Pin Controlled by PD15*/
#define AT91C_PD15_DTR0     ((unsigned int) AT91C_PIO_PD15) /*  */
#define AT91C_PD15_PWM1     ((unsigned int) AT91C_PIO_PD15) /*  */
#define AT91C_PIO_PD16       ((unsigned int) 1 << 16) /* Pin Controlled by PD16*/
#define AT91C_PD16_DCD0     ((unsigned int) AT91C_PIO_PD16) /*  */
#define AT91C_PD16_PWM2     ((unsigned int) AT91C_PIO_PD16) /*  */
#define AT91C_PIO_PD17       ((unsigned int) 1 << 17) /* Pin Controlled by PD17*/
#define AT91C_PD17_RI0      ((unsigned int) AT91C_PIO_PD17) /*  */
#define AT91C_PIO_PD18       ((unsigned int) 1 << 18) /* Pin Controlled by PD18*/
#define AT91C_PD18_PWM3     ((unsigned int) AT91C_PIO_PD18) /*  */
#define AT91C_PIO_PD19       ((unsigned int) 1 << 19) /* Pin Controlled by PD19*/
#define AT91C_PD19_PCK0     ((unsigned int) AT91C_PIO_PD19) /*  */
#define AT91C_PIO_PD2        ((unsigned int) 1 <<  2) /* Pin Controlled by PD2*/
#define AT91C_PD2_AC97_CK  ((unsigned int) AT91C_PIO_PD2) /*  */
#define AT91C_PD2_SCK1     ((unsigned int) AT91C_PIO_PD2) /*  */
#define AT91C_PIO_PD20       ((unsigned int) 1 << 20) /* Pin Controlled by PD20*/
#define AT91C_PD20_PCK1     ((unsigned int) AT91C_PIO_PD20) /*  */
#define AT91C_PIO_PD21       ((unsigned int) 1 << 21) /* Pin Controlled by PD21*/
#define AT91C_PD21_TCLK2    ((unsigned int) AT91C_PIO_PD21) /*  */
#define AT91C_PIO_PD3        ((unsigned int) 1 <<  3) /* Pin Controlled by PD3*/
#define AT91C_PD3_AC97_TX  ((unsigned int) AT91C_PIO_PD3) /*  */
#define AT91C_PD3_CTS3     ((unsigned int) AT91C_PIO_PD3) /*  */
#define AT91C_PIO_PD4        ((unsigned int) 1 <<  4) /* Pin Controlled by PD4*/
#define AT91C_PD4_AC97_RX  ((unsigned int) AT91C_PIO_PD4) /*  */
#define AT91C_PD4_RTS3     ((unsigned int) AT91C_PIO_PD4) /*  */
#define AT91C_PIO_PD5        ((unsigned int) 1 <<  5) /* Pin Controlled by PD5*/
#define AT91C_PD5_DTXD     ((unsigned int) AT91C_PIO_PD5) /*  */
#define AT91C_PD5_PWM2     ((unsigned int) AT91C_PIO_PD5) /*  */
#define AT91C_PIO_PD6        ((unsigned int) 1 <<  6) /* Pin Controlled by PD6*/
#define AT91C_PD6_AD4      ((unsigned int) AT91C_PIO_PD6) /*  */
#define AT91C_PIO_PD7        ((unsigned int) 1 <<  7) /* Pin Controlled by PD7*/
#define AT91C_PD7_AD5      ((unsigned int) AT91C_PIO_PD7) /*  */
#define AT91C_PIO_PD8        ((unsigned int) 1 <<  8) /* Pin Controlled by PD8*/
#define AT91C_PD8_NPCS2    ((unsigned int) AT91C_PIO_PD8) /*  */
#define AT91C_PD8_PWM3     ((unsigned int) AT91C_PIO_PD8) /*  */
#define AT91C_PIO_PD9        ((unsigned int) 1 <<  9) /* Pin Controlled by PD9*/
#define AT91C_PD9_SCK2     ((unsigned int) AT91C_PIO_PD9) /*  */
#define AT91C_PD9_NPCS3    ((unsigned int) AT91C_PIO_PD9) /*  */

/* ******************************************************************************/
/*               PERIPHERAL ID DEFINITIONS FOR AT91SAM9RL64*/
/* ******************************************************************************/
#define AT91C_ID_FIQ    ((unsigned int)  0) /* Advanced Interrupt Controller (FIQ)*/
#define AT91C_ID_SYS    ((unsigned int)  1) /* System Controller*/
#define AT91C_ID_PIOA   ((unsigned int)  2) /* Parallel IO Controller A*/
#define AT91C_ID_PIOB   ((unsigned int)  3) /* Parallel IO Controller B*/
#define AT91C_ID_PIOC   ((unsigned int)  4) /* Parallel IO Controller C*/
#define AT91C_ID_PIOD   ((unsigned int)  5) /* Parallel IO Controller D*/
#define AT91C_ID_US0    ((unsigned int)  6) /* USART 0*/
#define AT91C_ID_US1    ((unsigned int)  7) /* USART 1*/
#define AT91C_ID_US2    ((unsigned int)  8) /* USART 2*/
#define AT91C_ID_US3    ((unsigned int)  9) /* USART 2*/
#define AT91C_ID_MCI    ((unsigned int) 10) /* Multimedia Card Interface*/
#define AT91C_ID_TWI0   ((unsigned int) 11) /* TWI 0*/
#define AT91C_ID_TWI1   ((unsigned int) 12) /* TWI 1*/
#define AT91C_ID_SPI    ((unsigned int) 13) /* Serial Peripheral Interface*/
#define AT91C_ID_SSC0   ((unsigned int) 14) /* Serial Synchronous Controller 0*/
#define AT91C_ID_SSC1   ((unsigned int) 15) /* Serial Synchronous Controller 1*/
#define AT91C_ID_TC0    ((unsigned int) 16) /* Timer Counter 0*/
#define AT91C_ID_TC1    ((unsigned int) 17) /* Timer Counter 1*/
#define AT91C_ID_TC2    ((unsigned int) 18) /* Timer Counter 2*/
#define AT91C_ID_PWMC   ((unsigned int) 19) /* Pulse Width Modulation Controller*/
#define AT91C_ID_TSADC  ((unsigned int) 20) /* Touch Screen Controller*/
#define AT91C_ID_HDMA   ((unsigned int) 21) /* HDMA*/
#define AT91C_ID_UDPHS  ((unsigned int) 22) /* USB Device HS*/
#define AT91C_ID_LCDC   ((unsigned int) 23) /* LCD Controller*/
#define AT91C_ID_AC97C  ((unsigned int) 24) /* AC97 Controller*/
#define AT91C_ID_IRQ0   ((unsigned int) 31) /* Advanced Interrupt Controller (IRQ0)*/
#define AT91C_ALL_INT   ((unsigned int) 0x81FFFFFF) /* ALL VALID INTERRUPTS*/

/* ******************************************************************************/
/*               BASE ADDRESS DEFINITIONS FOR AT91SAM9RL64*/
/* ******************************************************************************/
#define AT91C_BASE_SYS       ((AT91PS_SYS) 	0xFFFFC000) /* (SYS) Base Address*/
#define AT91C_BASE_EBI       ((AT91PS_EBI) 	0xFFFFE800) /* (EBI) Base Address*/
#define AT91C_BASE_SDRAMC    ((AT91PS_SDRAMC) 	0xFFFFEA00) /* (SDRAMC) Base Address*/
#define AT91C_BASE_SMC       ((AT91PS_SMC) 	0xFFFFEC00) /* (SMC) Base Address*/
#define AT91C_BASE_MATRIX    ((AT91PS_MATRIX) 	0xFFFFEE00) /* (MATRIX) Base Address*/
#define AT91C_BASE_CCFG      ((AT91PS_CCFG) 	0xFFFFEF10) /* (CCFG) Base Address*/
#define AT91C_BASE_AIC       ((AT91PS_AIC) 	0xFFFFF000) /* (AIC) Base Address*/
#define AT91C_BASE_PDC_DBGU  ((AT91PS_PDC) 	0xFFFFF300) /* (PDC_DBGU) Base Address*/
#define AT91C_BASE_DBGU      ((AT91PS_DBGU) 	0xFFFFF200) /* (DBGU) Base Address*/
#define AT91C_BASE_PIOA      ((AT91PS_PIO) 	0xFFFFF400) /* (PIOA) Base Address*/
#define AT91C_BASE_PIOB      ((AT91PS_PIO) 	0xFFFFF600) /* (PIOB) Base Address*/
#define AT91C_BASE_PIOC      ((AT91PS_PIO) 	0xFFFFF800) /* (PIOC) Base Address*/
#define AT91C_BASE_PIOD      ((AT91PS_PIO) 	0xFFFFFA00) /* (PIOD) Base Address*/
#define AT91C_BASE_PMC       ((AT91PS_PMC) 	0xFFFFFC00) /* (PMC) Base Address*/
#define AT91C_BASE_CKGR      ((AT91PS_CKGR) 	0xFFFFFC1C) /* (CKGR) Base Address*/
#define AT91C_BASE_RSTC      ((AT91PS_RSTC) 	0xFFFFFD00) /* (RSTC) Base Address*/
#define AT91C_BASE_SHDWC     ((AT91PS_SHDWC) 	0xFFFFFD10) /* (SHDWC) Base Address*/
#define AT91C_BASE_RTTC      ((AT91PS_RTTC) 	0xFFFFFD20) /* (RTTC) Base Address*/
#define AT91C_BASE_PITC      ((AT91PS_PITC) 	0xFFFFFD30) /* (PITC) Base Address*/
#define AT91C_BASE_WDTC      ((AT91PS_WDTC) 	0xFFFFFD40) /* (WDTC) Base Address*/
#define AT91C_BASE_RTC       ((AT91PS_RTC) 	0xFFFFFE00) /* (RTC) Base Address*/
#define AT91C_BASE_TC0       ((AT91PS_TC) 	0xFFFA0000) /* (TC0) Base Address*/
#define AT91C_BASE_TC1       ((AT91PS_TC) 	0xFFFA0040) /* (TC1) Base Address*/
#define AT91C_BASE_TC2       ((AT91PS_TC) 	0xFFFA0080) /* (TC2) Base Address*/
#define AT91C_BASE_TCB0      ((AT91PS_TCB) 	0xFFFA0000) /* (TCB0) Base Address*/
#define AT91C_BASE_TCB1      ((AT91PS_TCB) 	0xFFFA0040) /* (TCB1) Base Address*/
#define AT91C_BASE_TCB2      ((AT91PS_TCB) 	0xFFFA0080) /* (TCB2) Base Address*/
#define AT91C_BASE_PDC_MCI   ((AT91PS_PDC) 	0xFFFA4100) /* (PDC_MCI) Base Address*/
#define AT91C_BASE_MCI       ((AT91PS_MCI) 	0xFFFA4000) /* (MCI) Base Address*/
#define AT91C_BASE_PDC_TWI0  ((AT91PS_PDC) 	0xFFFA8100) /* (PDC_TWI0) Base Address*/
#define AT91C_BASE_TWI0      ((AT91PS_TWI) 	0xFFFA8000) /* (TWI0) Base Address*/
#define AT91C_BASE_TWI1      ((AT91PS_TWI) 	0xFFFAC000) /* (TWI1) Base Address*/
#define AT91C_BASE_PDC_US0   ((AT91PS_PDC) 	0xFFFB0100) /* (PDC_US0) Base Address*/
#define AT91C_BASE_US0       ((AT91PS_USART) 	0xFFFB0000) /* (US0) Base Address*/
#define AT91C_BASE_PDC_US1   ((AT91PS_PDC) 	0xFFFB4100) /* (PDC_US1) Base Address*/
#define AT91C_BASE_US1       ((AT91PS_USART) 	0xFFFB4000) /* (US1) Base Address*/
#define AT91C_BASE_PDC_US2   ((AT91PS_PDC) 	0xFFFB8100) /* (PDC_US2) Base Address*/
#define AT91C_BASE_US2       ((AT91PS_USART) 	0xFFFB8000) /* (US2) Base Address*/
#define AT91C_BASE_PDC_US3   ((AT91PS_PDC) 	0xFFFBC100) /* (PDC_US3) Base Address*/
#define AT91C_BASE_US3       ((AT91PS_USART) 	0xFFFBC000) /* (US3) Base Address*/
#define AT91C_BASE_PDC_SSC0  ((AT91PS_PDC) 	0xFFFC0100) /* (PDC_SSC0) Base Address*/
#define AT91C_BASE_SSC0      ((AT91PS_SSC) 	0xFFFC0000) /* (SSC0) Base Address*/
#define AT91C_BASE_PDC_SSC1  ((AT91PS_PDC) 	0xFFFC4100) /* (PDC_SSC1) Base Address*/
#define AT91C_BASE_SSC1      ((AT91PS_SSC) 	0xFFFC4000) /* (SSC1) Base Address*/
#define AT91C_BASE_PWMC_CH0  ((AT91PS_PWMC_CH) 	0xFFFC8200) /* (PWMC_CH0) Base Address*/
#define AT91C_BASE_PWMC_CH1  ((AT91PS_PWMC_CH) 	0xFFFC8220) /* (PWMC_CH1) Base Address*/
#define AT91C_BASE_PWMC_CH2  ((AT91PS_PWMC_CH) 	0xFFFC8240) /* (PWMC_CH2) Base Address*/
#define AT91C_BASE_PWMC_CH3  ((AT91PS_PWMC_CH) 	0xFFFC8260) /* (PWMC_CH3) Base Address*/
#define AT91C_BASE_PWMC      ((AT91PS_PWMC) 	0xFFFC8000) /* (PWMC) Base Address*/
#define AT91C_BASE_PDC_SPI   ((AT91PS_PDC) 	0xFFFCC100) /* (PDC_SPI) Base Address*/
#define AT91C_BASE_SPI       ((AT91PS_SPI) 	0xFFFCC000) /* (SPI) Base Address*/
#define AT91C_BASE_PDC_TSADC ((AT91PS_PDC) 	0xFFFD0100) /* (PDC_TSADC) Base Address*/
#define AT91C_BASE_TSADC     ((AT91PS_TSADC) 	0xFFFD0000) /* (TSADC) Base Address*/
#define AT91C_BASE_UDPHS_EPTFIFO ((AT91PS_UDPHS_EPTFIFO) 	0x00600000) /* (UDPHS_EPTFIFO) Base Address*/
#define AT91C_BASE_UDPHS_EPT_0 ((AT91PS_UDPHS_EPT) 	0xFFFD4100) /* (UDPHS_EPT_0) Base Address*/
#define AT91C_BASE_UDPHS_EPT_1 ((AT91PS_UDPHS_EPT) 	0xFFFD4120) /* (UDPHS_EPT_1) Base Address*/
#define AT91C_BASE_UDPHS_EPT_2 ((AT91PS_UDPHS_EPT) 	0xFFFD4140) /* (UDPHS_EPT_2) Base Address*/
#define AT91C_BASE_UDPHS_EPT_3 ((AT91PS_UDPHS_EPT) 	0xFFFD4160) /* (UDPHS_EPT_3) Base Address*/
#define AT91C_BASE_UDPHS_EPT_4 ((AT91PS_UDPHS_EPT) 	0xFFFD4180) /* (UDPHS_EPT_4) Base Address*/
#define AT91C_BASE_UDPHS_EPT_5 ((AT91PS_UDPHS_EPT) 	0xFFFD41A0) /* (UDPHS_EPT_5) Base Address*/
#define AT91C_BASE_UDPHS_EPT_6 ((AT91PS_UDPHS_EPT) 	0xFFFD41C0) /* (UDPHS_EPT_6) Base Address*/
#define AT91C_BASE_UDPHS_EPT_7 ((AT91PS_UDPHS_EPT) 	0xFFFD41E0) /* (UDPHS_EPT_7) Base Address*/
#define AT91C_BASE_UDPHS_EPT_8 ((AT91PS_UDPHS_EPT) 	0xFFFD4200) /* (UDPHS_EPT_8) Base Address*/
#define AT91C_BASE_UDPHS_EPT_9 ((AT91PS_UDPHS_EPT) 	0xFFFD4220) /* (UDPHS_EPT_9) Base Address*/
#define AT91C_BASE_UDPHS_EPT_10 ((AT91PS_UDPHS_EPT) 	0xFFFD4240) /* (UDPHS_EPT_10) Base Address*/
#define AT91C_BASE_UDPHS_EPT_11 ((AT91PS_UDPHS_EPT) 	0xFFFD4260) /* (UDPHS_EPT_11) Base Address*/
#define AT91C_BASE_UDPHS_EPT_12 ((AT91PS_UDPHS_EPT) 	0xFFFD4280) /* (UDPHS_EPT_12) Base Address*/
#define AT91C_BASE_UDPHS_EPT_13 ((AT91PS_UDPHS_EPT) 	0xFFFD42A0) /* (UDPHS_EPT_13) Base Address*/
#define AT91C_BASE_UDPHS_EPT_14 ((AT91PS_UDPHS_EPT) 	0xFFFD42C0) /* (UDPHS_EPT_14) Base Address*/
#define AT91C_BASE_UDPHS_EPT_15 ((AT91PS_UDPHS_EPT) 	0xFFFD42E0) /* (UDPHS_EPT_15) Base Address*/
#define AT91C_BASE_UDPHS_DMA_1 ((AT91PS_UDPHS_DMA) 	0xFFFD4310) /* (UDPHS_DMA_1) Base Address*/
#define AT91C_BASE_UDPHS_DMA_2 ((AT91PS_UDPHS_DMA) 	0xFFFD4320) /* (UDPHS_DMA_2) Base Address*/
#define AT91C_BASE_UDPHS_DMA_3 ((AT91PS_UDPHS_DMA) 	0xFFFD4330) /* (UDPHS_DMA_3) Base Address*/
#define AT91C_BASE_UDPHS_DMA_4 ((AT91PS_UDPHS_DMA) 	0xFFFD4340) /* (UDPHS_DMA_4) Base Address*/
#define AT91C_BASE_UDPHS_DMA_5 ((AT91PS_UDPHS_DMA) 	0xFFFD4350) /* (UDPHS_DMA_5) Base Address*/
#define AT91C_BASE_UDPHS_DMA_6 ((AT91PS_UDPHS_DMA) 	0xFFFD4360) /* (UDPHS_DMA_6) Base Address*/
#define AT91C_BASE_UDPHS_DMA_7 ((AT91PS_UDPHS_DMA) 	0xFFFD4370) /* (UDPHS_DMA_7) Base Address*/
#define AT91C_BASE_UDPHS     ((AT91PS_UDPHS) 	0xFFFD4000) /* (UDPHS) Base Address*/
#define AT91C_BASE_PDC_AC97C ((AT91PS_PDC) 	0xFFFD8100) /* (PDC_AC97C) Base Address*/
#define AT91C_BASE_AC97C     ((AT91PS_AC97C) 	0xFFFD8000) /* (AC97C) Base Address*/
#define AT91C_BASE_LCDC      ((AT91PS_LCDC) 	0x00500000) /* (LCDC) Base Address*/
#define AT91C_BASE_LCDC_16B_TFT ((AT91PS_LCDC) 	0x00500000) /* (LCDC_16B_TFT) Base Address*/
#define AT91C_BASE_HDMA_CH_0 ((AT91PS_HDMA_CH) 	0xFFFFE63C) /* (HDMA_CH_0) Base Address*/
#define AT91C_BASE_HDMA_CH_1 ((AT91PS_HDMA_CH) 	0xFFFFE664) /* (HDMA_CH_1) Base Address*/
#define AT91C_BASE_HDMA      ((AT91PS_HDMA) 	0xFFFFE600) /* (HDMA) Base Address*/
#define AT91C_BASE_HECC      ((AT91PS_ECC) 	0xFFFFE800) /* (HECC) Base Address*/

/* ******************************************************************************/
/*               MEMORY MAPPING DEFINITIONS FOR AT91SAM9RL64*/
/* ******************************************************************************/
/* ITCM*/
#define AT91C_ITCM 	 ((char *) 	0x00100000) /* Maximum ITCM Area base address*/
#define AT91C_ITCM_SIZE	 ((unsigned int) 0x00010000) /* Maximum ITCM Area size in byte (64 Kbytes)*/
/* DTCM*/
#define AT91C_DTCM 	 ((char *) 	0x00200000) /* Maximum DTCM Area base address*/
#define AT91C_DTCM_SIZE	 ((unsigned int) 0x00010000) /* Maximum DTCM Area size in byte (64 Kbytes)*/
/* IRAM*/
#define AT91C_IRAM 	 ((char *) 	0x00300000) /* Maximum Internal SRAM base address*/
#define AT91C_IRAM_SIZE	 ((unsigned int) 0x00010000) /* Maximum Internal SRAM size in byte (64 Kbytes)*/
/* IRAM_MIN*/
#define AT91C_IRAM_MIN	 ((char *) 	0x00300000) /* Minimum Internal RAM base address*/
#define AT91C_IRAM_MIN_SIZE	 ((unsigned int) 0x00004000) /* Minimum Internal RAM size in byte (16 Kbytes)*/
/* IROM*/
#define AT91C_IROM 	 ((char *) 	0x00400000) /* Internal ROM base address*/
#define AT91C_IROM_SIZE	 ((unsigned int) 0x00008000) /* Internal ROM size in byte (32 Kbytes)*/
/* EBI_CS0*/
#define AT91C_EBI_CS0	 ((char *) 	0x10000000) /* EBI Chip Select 0 base address*/
#define AT91C_EBI_CS0_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 0 size in byte (262144 Kbytes)*/
/* EBI_CS1*/
#define AT91C_EBI_CS1	 ((char *) 	0x20000000) /* EBI Chip Select 1 base address*/
#define AT91C_EBI_CS1_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 1 size in byte (262144 Kbytes)*/
/* EBI_SDRAM*/
#define AT91C_EBI_SDRAM	 ((char *) 	0x20000000) /* SDRAM on EBI Chip Select 1 base address*/
#define AT91C_EBI_SDRAM_SIZE	 ((unsigned int) 0x10000000) /* SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)*/
/* EBI_SDRAM_16BIT*/
#define AT91C_EBI_SDRAM_16BIT	 ((char *) 	0x20000000) /* SDRAM on EBI Chip Select 1 base address*/
#define AT91C_EBI_SDRAM_16BIT_SIZE	 ((unsigned int) 0x02000000) /* SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)*/
/* EBI_SDRAM_32BIT*/
#define AT91C_EBI_SDRAM_32BIT	 ((char *) 	0x20000000) /* SDRAM on EBI Chip Select 1 base address*/
#define AT91C_EBI_SDRAM_32BIT_SIZE	 ((unsigned int) 0x04000000) /* SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)*/
/* EBI_CS2*/
#define AT91C_EBI_CS2	 ((char *) 	0x30000000) /* EBI Chip Select 2 base address*/
#define AT91C_EBI_CS2_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 2 size in byte (262144 Kbytes)*/
/* EBI_CS3*/
#define AT91C_EBI_CS3	 ((char *) 	0x40000000) /* EBI Chip Select 3 base address*/
#define AT91C_EBI_CS3_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 3 size in byte (262144 Kbytes)*/
/* EBI_SM*/
#define AT91C_EBI_SM	 ((char *) 	0x40000000) /* NANDFLASH on EBI Chip Select 3 base address*/
#define AT91C_EBI_SM_SIZE	 ((unsigned int) 0x10000000) /* NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)*/
/* EBI_CS4*/
#define AT91C_EBI_CS4	 ((char *) 	0x50000000) /* EBI Chip Select 4 base address*/
#define AT91C_EBI_CS4_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 4 size in byte (262144 Kbytes)*/
/* EBI_CF0*/
#define AT91C_EBI_CF0	 ((char *) 	0x50000000) /* CompactFlash 0 on EBI Chip Select 4 base address*/
#define AT91C_EBI_CF0_SIZE	 ((unsigned int) 0x10000000) /* CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)*/
/* EBI_CS5*/
#define AT91C_EBI_CS5	 ((char *) 	0x60000000) /* EBI Chip Select 5 base address*/
#define AT91C_EBI_CS5_SIZE	 ((unsigned int) 0x10000000) /* EBI Chip Select 5 size in byte (262144 Kbytes)*/
/* EBI_CF1*/
#define AT91C_EBI_CF1	 ((char *) 	0x60000000) /* CompactFlash 1 on EBIChip Select 5 base address*/
#define AT91C_EBI_CF1_SIZE	 ((unsigned int) 0x10000000) /* CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)*/
#endif /* __IAR_SYSTEMS_ICC__ */

#ifdef __IAR_SYSTEMS_ASM__

/* - Hardware register definition*/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR System Peripherals*/
/* - ******************************************************************************/
/* - -------- SLCKSEL : (SYS Offset: 0x3d50) Slow Clock Selection Register -------- */
AT91C_SLCKSEL_RCEN        EQU (0x1 <<  0) ;- (SYS) Enable Internal RC Oscillator
AT91C_SLCKSEL_OSC32EN     EQU (0x1 <<  1) ;- (SYS) Enable External Oscillator
AT91C_SLCKSEL_OSC32BYP    EQU (0x1 <<  2) ;- (SYS) Bypass External Oscillator
AT91C_SLCKSEL_OSCSEL      EQU (0x1 <<  3) ;- (SYS) OSC Selection
/* - -------- GPBR : (SYS Offset: 0x3d60) GPBR General Purpose Register -------- */
AT91C_GPBR_GPRV           EQU (0x0 <<  0) ;- (SYS) General Purpose Register Value

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR External Bus Interface*/
/* - ******************************************************************************/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR SDRAM Controller Interface*/
/* - ******************************************************************************/
/* - -------- SDRAMC_MR : (SDRAMC Offset: 0x0) SDRAM Controller Mode Register -------- */
AT91C_SDRAMC_MODE         EQU (0xF <<  0) ;- (SDRAMC) Mode
AT91C_SDRAMC_MODE_NORMAL_CMD EQU (0x0) ;- (SDRAMC) Normal Mode
AT91C_SDRAMC_MODE_NOP_CMD EQU (0x1) ;- (SDRAMC) Issue a NOP Command at every access
AT91C_SDRAMC_MODE_PRCGALL_CMD EQU (0x2) ;- (SDRAMC) Issue a All Banks Precharge Command at every access
AT91C_SDRAMC_MODE_LMR_CMD EQU (0x3) ;- (SDRAMC) Issue a Load Mode Register at every access
AT91C_SDRAMC_MODE_RFSH_CMD EQU (0x4) ;- (SDRAMC) Issue a Refresh
AT91C_SDRAMC_MODE_EXT_LMR_CMD EQU (0x5) ;- (SDRAMC) Issue an Extended Load Mode Register
AT91C_SDRAMC_MODE_DEEP_CMD EQU (0x6) ;- (SDRAMC) Enter Deep Power Mode
/* - -------- SDRAMC_TR : (SDRAMC Offset: 0x4) SDRAMC Refresh Timer Register -------- */
AT91C_SDRAMC_COUNT        EQU (0xFFF <<  0) ;- (SDRAMC) Refresh Counter
/* - -------- SDRAMC_CR : (SDRAMC Offset: 0x8) SDRAM Configuration Register -------- */
AT91C_SDRAMC_NC           EQU (0x3 <<  0) ;- (SDRAMC) Number of Column Bits
AT91C_SDRAMC_NC_8         EQU (0x0) ;- (SDRAMC) 8 Bits
AT91C_SDRAMC_NC_9         EQU (0x1) ;- (SDRAMC) 9 Bits
AT91C_SDRAMC_NC_10        EQU (0x2) ;- (SDRAMC) 10 Bits
AT91C_SDRAMC_NC_11        EQU (0x3) ;- (SDRAMC) 11 Bits
AT91C_SDRAMC_NR           EQU (0x3 <<  2) ;- (SDRAMC) Number of Row Bits
AT91C_SDRAMC_NR_11        EQU (0x0 <<  2) ;- (SDRAMC) 11 Bits
AT91C_SDRAMC_NR_12        EQU (0x1 <<  2) ;- (SDRAMC) 12 Bits
AT91C_SDRAMC_NR_13        EQU (0x2 <<  2) ;- (SDRAMC) 13 Bits
AT91C_SDRAMC_NB           EQU (0x1 <<  4) ;- (SDRAMC) Number of Banks
AT91C_SDRAMC_NB_2_BANKS   EQU (0x0 <<  4) ;- (SDRAMC) 2 banks
AT91C_SDRAMC_NB_4_BANKS   EQU (0x1 <<  4) ;- (SDRAMC) 4 banks
AT91C_SDRAMC_CAS          EQU (0x3 <<  5) ;- (SDRAMC) CAS Latency
AT91C_SDRAMC_CAS_2        EQU (0x2 <<  5) ;- (SDRAMC) 2 cycles
AT91C_SDRAMC_CAS_3        EQU (0x3 <<  5) ;- (SDRAMC) 3 cycles
AT91C_SDRAMC_DBW          EQU (0x1 <<  7) ;- (SDRAMC) Data Bus Width
AT91C_SDRAMC_DBW_32_BITS  EQU (0x0 <<  7) ;- (SDRAMC) 32 Bits datas bus
AT91C_SDRAMC_DBW_16_BITS  EQU (0x1 <<  7) ;- (SDRAMC) 16 Bits datas bus
AT91C_SDRAMC_TWR          EQU (0xF <<  8) ;- (SDRAMC) Number of Write Recovery Time Cycles
AT91C_SDRAMC_TWR_0        EQU (0x0 <<  8) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TWR_1        EQU (0x1 <<  8) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TWR_2        EQU (0x2 <<  8) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TWR_3        EQU (0x3 <<  8) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TWR_4        EQU (0x4 <<  8) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TWR_5        EQU (0x5 <<  8) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TWR_6        EQU (0x6 <<  8) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TWR_7        EQU (0x7 <<  8) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TWR_8        EQU (0x8 <<  8) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TWR_9        EQU (0x9 <<  8) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TWR_10       EQU (0xA <<  8) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TWR_11       EQU (0xB <<  8) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TWR_12       EQU (0xC <<  8) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TWR_13       EQU (0xD <<  8) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TWR_14       EQU (0xE <<  8) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TWR_15       EQU (0xF <<  8) ;- (SDRAMC) Value : 15
AT91C_SDRAMC_TRC          EQU (0xF << 12) ;- (SDRAMC) Number of RAS Cycle Time Cycles
AT91C_SDRAMC_TRC_0        EQU (0x0 << 12) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TRC_1        EQU (0x1 << 12) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TRC_2        EQU (0x2 << 12) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TRC_3        EQU (0x3 << 12) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TRC_4        EQU (0x4 << 12) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TRC_5        EQU (0x5 << 12) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TRC_6        EQU (0x6 << 12) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TRC_7        EQU (0x7 << 12) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TRC_8        EQU (0x8 << 12) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TRC_9        EQU (0x9 << 12) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TRC_10       EQU (0xA << 12) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TRC_11       EQU (0xB << 12) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TRC_12       EQU (0xC << 12) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TRC_13       EQU (0xD << 12) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TRC_14       EQU (0xE << 12) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TRC_15       EQU (0xF << 12) ;- (SDRAMC) Value : 15
AT91C_SDRAMC_TRP          EQU (0xF << 16) ;- (SDRAMC) Number of RAS Precharge Time Cycles
AT91C_SDRAMC_TRP_0        EQU (0x0 << 16) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TRP_1        EQU (0x1 << 16) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TRP_2        EQU (0x2 << 16) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TRP_3        EQU (0x3 << 16) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TRP_4        EQU (0x4 << 16) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TRP_5        EQU (0x5 << 16) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TRP_6        EQU (0x6 << 16) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TRP_7        EQU (0x7 << 16) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TRP_8        EQU (0x8 << 16) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TRP_9        EQU (0x9 << 16) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TRP_10       EQU (0xA << 16) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TRP_11       EQU (0xB << 16) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TRP_12       EQU (0xC << 16) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TRP_13       EQU (0xD << 16) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TRP_14       EQU (0xE << 16) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TRP_15       EQU (0xF << 16) ;- (SDRAMC) Value : 15
AT91C_SDRAMC_TRCD         EQU (0xF << 20) ;- (SDRAMC) Number of RAS to CAS Delay Cycles
AT91C_SDRAMC_TRCD_0       EQU (0x0 << 20) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TRCD_1       EQU (0x1 << 20) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TRCD_2       EQU (0x2 << 20) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TRCD_3       EQU (0x3 << 20) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TRCD_4       EQU (0x4 << 20) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TRCD_5       EQU (0x5 << 20) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TRCD_6       EQU (0x6 << 20) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TRCD_7       EQU (0x7 << 20) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TRCD_8       EQU (0x8 << 20) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TRCD_9       EQU (0x9 << 20) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TRCD_10      EQU (0xA << 20) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TRCD_11      EQU (0xB << 20) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TRCD_12      EQU (0xC << 20) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TRCD_13      EQU (0xD << 20) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TRCD_14      EQU (0xE << 20) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TRCD_15      EQU (0xF << 20) ;- (SDRAMC) Value : 15
AT91C_SDRAMC_TRAS         EQU (0xF << 24) ;- (SDRAMC) Number of RAS Active Time Cycles
AT91C_SDRAMC_TRAS_0       EQU (0x0 << 24) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TRAS_1       EQU (0x1 << 24) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TRAS_2       EQU (0x2 << 24) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TRAS_3       EQU (0x3 << 24) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TRAS_4       EQU (0x4 << 24) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TRAS_5       EQU (0x5 << 24) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TRAS_6       EQU (0x6 << 24) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TRAS_7       EQU (0x7 << 24) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TRAS_8       EQU (0x8 << 24) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TRAS_9       EQU (0x9 << 24) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TRAS_10      EQU (0xA << 24) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TRAS_11      EQU (0xB << 24) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TRAS_12      EQU (0xC << 24) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TRAS_13      EQU (0xD << 24) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TRAS_14      EQU (0xE << 24) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TRAS_15      EQU (0xF << 24) ;- (SDRAMC) Value : 15
AT91C_SDRAMC_TXSR         EQU (0xF << 28) ;- (SDRAMC) Number of Command Recovery Time Cycles
AT91C_SDRAMC_TXSR_0       EQU (0x0 << 28) ;- (SDRAMC) Value :  0
AT91C_SDRAMC_TXSR_1       EQU (0x1 << 28) ;- (SDRAMC) Value :  1
AT91C_SDRAMC_TXSR_2       EQU (0x2 << 28) ;- (SDRAMC) Value :  2
AT91C_SDRAMC_TXSR_3       EQU (0x3 << 28) ;- (SDRAMC) Value :  3
AT91C_SDRAMC_TXSR_4       EQU (0x4 << 28) ;- (SDRAMC) Value :  4
AT91C_SDRAMC_TXSR_5       EQU (0x5 << 28) ;- (SDRAMC) Value :  5
AT91C_SDRAMC_TXSR_6       EQU (0x6 << 28) ;- (SDRAMC) Value :  6
AT91C_SDRAMC_TXSR_7       EQU (0x7 << 28) ;- (SDRAMC) Value :  7
AT91C_SDRAMC_TXSR_8       EQU (0x8 << 28) ;- (SDRAMC) Value :  8
AT91C_SDRAMC_TXSR_9       EQU (0x9 << 28) ;- (SDRAMC) Value :  9
AT91C_SDRAMC_TXSR_10      EQU (0xA << 28) ;- (SDRAMC) Value : 10
AT91C_SDRAMC_TXSR_11      EQU (0xB << 28) ;- (SDRAMC) Value : 11
AT91C_SDRAMC_TXSR_12      EQU (0xC << 28) ;- (SDRAMC) Value : 12
AT91C_SDRAMC_TXSR_13      EQU (0xD << 28) ;- (SDRAMC) Value : 13
AT91C_SDRAMC_TXSR_14      EQU (0xE << 28) ;- (SDRAMC) Value : 14
AT91C_SDRAMC_TXSR_15      EQU (0xF << 28) ;- (SDRAMC) Value : 15
/* - -------- SDRAMC_HSR : (SDRAMC Offset: 0xc) SDRAM Controller High Speed Register -------- */
AT91C_SDRAMC_DA           EQU (0x1 <<  0) ;- (SDRAMC) Decode Cycle Enable Bit
AT91C_SDRAMC_DA_DISABLE   EQU (0x0) ;- (SDRAMC) Disable Decode Cycle
AT91C_SDRAMC_DA_ENABLE    EQU (0x1) ;- (SDRAMC) Enable Decode Cycle
/* - -------- SDRAMC_LPR : (SDRAMC Offset: 0x10) SDRAM Controller Low-power Register -------- */
AT91C_SDRAMC_LPCB         EQU (0x3 <<  0) ;- (SDRAMC) Low-power Configurations
AT91C_SDRAMC_LPCB_DISABLE EQU (0x0) ;- (SDRAMC) Disable Low Power Features
AT91C_SDRAMC_LPCB_SELF_REFRESH EQU (0x1) ;- (SDRAMC) Enable SELF_REFRESH
AT91C_SDRAMC_LPCB_POWER_DOWN EQU (0x2) ;- (SDRAMC) Enable POWER_DOWN
AT91C_SDRAMC_LPCB_DEEP_POWER_DOWN EQU (0x3) ;- (SDRAMC) Enable DEEP_POWER_DOWN
AT91C_SDRAMC_PASR         EQU (0x7 <<  4) ;- (SDRAMC) Partial Array Self Refresh (only for Low Power SDRAM)
AT91C_SDRAMC_TCSR         EQU (0x3 <<  8) ;- (SDRAMC) Temperature Compensated Self Refresh (only for Low Power SDRAM)
AT91C_SDRAMC_DS           EQU (0x3 << 10) ;- (SDRAMC) Drive Strenght (only for Low Power SDRAM)
AT91C_SDRAMC_TIMEOUT      EQU (0x3 << 12) ;- (SDRAMC) Time to define when Low Power Mode is enabled
AT91C_SDRAMC_TIMEOUT_0_CLK_CYCLES EQU (0x0 << 12) ;- (SDRAMC) Activate SDRAM Low Power Mode Immediately
AT91C_SDRAMC_TIMEOUT_64_CLK_CYCLES EQU (0x1 << 12) ;- (SDRAMC) Activate SDRAM Low Power Mode after 64 clock cycles after the end of the last transfer
AT91C_SDRAMC_TIMEOUT_128_CLK_CYCLES EQU (0x2 << 12) ;- (SDRAMC) Activate SDRAM Low Power Mode after 64 clock cycles after the end of the last transfer
/* - -------- SDRAMC_IER : (SDRAMC Offset: 0x14) SDRAM Controller Interrupt Enable Register -------- */
AT91C_SDRAMC_RES          EQU (0x1 <<  0) ;- (SDRAMC) Refresh Error Status
/* - -------- SDRAMC_IDR : (SDRAMC Offset: 0x18) SDRAM Controller Interrupt Disable Register -------- */
/* - -------- SDRAMC_IMR : (SDRAMC Offset: 0x1c) SDRAM Controller Interrupt Mask Register -------- */
/* - -------- SDRAMC_ISR : (SDRAMC Offset: 0x20) SDRAM Controller Interrupt Status Register -------- */
/* - -------- SDRAMC_MDR : (SDRAMC Offset: 0x24) SDRAM Controller Memory Device Register -------- */
AT91C_SDRAMC_MD           EQU (0x3 <<  0) ;- (SDRAMC) Memory Device Type
AT91C_SDRAMC_MD_SDRAM     EQU (0x0) ;- (SDRAMC) SDRAM Mode
AT91C_SDRAMC_MD_LOW_POWER_SDRAM EQU (0x1) ;- (SDRAMC) SDRAM Low Power Mode

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Static Memory Controller Interface*/
/* - ******************************************************************************/
/* - -------- SMC_SETUP : (SMC Offset: 0x0) Setup Register for CS x -------- */
AT91C_SMC_NWESETUP        EQU (0x3F <<  0) ;- (SMC) NWE Setup Length
AT91C_SMC_NCSSETUPWR      EQU (0x3F <<  8) ;- (SMC) NCS Setup Length in WRite Access
AT91C_SMC_NRDSETUP        EQU (0x3F << 16) ;- (SMC) NRD Setup Length
AT91C_SMC_NCSSETUPRD      EQU (0x3F << 24) ;- (SMC) NCS Setup Length in ReaD Access
/* - -------- SMC_PULSE : (SMC Offset: 0x4) Pulse Register for CS x -------- */
AT91C_SMC_NWEPULSE        EQU (0x7F <<  0) ;- (SMC) NWE Pulse Length
AT91C_SMC_NCSPULSEWR      EQU (0x7F <<  8) ;- (SMC) NCS Pulse Length in WRite Access
AT91C_SMC_NRDPULSE        EQU (0x7F << 16) ;- (SMC) NRD Pulse Length
AT91C_SMC_NCSPULSERD      EQU (0x7F << 24) ;- (SMC) NCS Pulse Length in ReaD Access
/* - -------- SMC_CYC : (SMC Offset: 0x8) Cycle Register for CS x -------- */
AT91C_SMC_NWECYCLE        EQU (0x1FF <<  0) ;- (SMC) Total Write Cycle Length
AT91C_SMC_NRDCYCLE        EQU (0x1FF << 16) ;- (SMC) Total Read Cycle Length
/* - -------- SMC_CTRL : (SMC Offset: 0xc) Control Register for CS x -------- */
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
/* - -------- SMC_SETUP : (SMC Offset: 0x10) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x14) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x18) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x1c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x20) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x24) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x28) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x2c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x30) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x34) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x38) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x3c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x40) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x44) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x48) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x4c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x50) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x54) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x58) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x5c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x60) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x64) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x68) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x6c) Control Register for CS x -------- */
/* - -------- SMC_SETUP : (SMC Offset: 0x70) Setup Register for CS x -------- */
/* - -------- SMC_PULSE : (SMC Offset: 0x74) Pulse Register for CS x -------- */
/* - -------- SMC_CYC : (SMC Offset: 0x78) Cycle Register for CS x -------- */
/* - -------- SMC_CTRL : (SMC Offset: 0x7c) Control Register for CS x -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR AHB Matrix Interface*/
/* - ******************************************************************************/
/* - -------- MATRIX_MCFG0 : (MATRIX Offset: 0x0) Master Configuration Register rom -------- */
AT91C_MATRIX_ULBT         EQU (0x7 <<  0) ;- (MATRIX) Undefined Length Burst Type
/* - -------- MATRIX_MCFG1 : (MATRIX Offset: 0x4) Master Configuration Register htcm -------- */
/* - -------- MATRIX_MCFG2 : (MATRIX Offset: 0x8) Master Configuration Register gps_tcm -------- */
/* - -------- MATRIX_MCFG3 : (MATRIX Offset: 0xc) Master Configuration Register hperiphs -------- */
/* - -------- MATRIX_MCFG4 : (MATRIX Offset: 0x10) Master Configuration Register ebi0 -------- */
/* - -------- MATRIX_MCFG5 : (MATRIX Offset: 0x14) Master Configuration Register ebi1 -------- */
/* - -------- MATRIX_MCFG6 : (MATRIX Offset: 0x18) Master Configuration Register bridge -------- */
/* - -------- MATRIX_MCFG7 : (MATRIX Offset: 0x1c) Master Configuration Register gps -------- */
/* - -------- MATRIX_MCFG8 : (MATRIX Offset: 0x20) Master Configuration Register gps -------- */
/* - -------- MATRIX_SCFG0 : (MATRIX Offset: 0x40) Slave Configuration Register 0 -------- */
AT91C_MATRIX_SLOT_CYCLE   EQU (0xFF <<  0) ;- (MATRIX) Maximum Number of Allowed Cycles for a Burst
AT91C_MATRIX_DEFMSTR_TYPE EQU (0x3 << 16) ;- (MATRIX) Default Master Type
AT91C_MATRIX_DEFMSTR_TYPE_NO_DEFMSTR EQU (0x0 << 16) ;- (MATRIX) No Default Master. At the end of current slave access, if no other master request is pending, the slave is deconnected from all masters. This results in having a one cycle latency for the first transfer of a burst.
AT91C_MATRIX_DEFMSTR_TYPE_LAST_DEFMSTR EQU (0x1 << 16) ;- (MATRIX) Last Default Master. At the end of current slave access, if no other master request is pending, the slave stay connected with the last master having accessed it. This results in not having the one cycle latency when the last master re-trying access on the slave.
AT91C_MATRIX_DEFMSTR_TYPE_FIXED_DEFMSTR EQU (0x2 << 16) ;- (MATRIX) Fixed Default Master. At the end of current slave access, if no other master request is pending, the slave connects with fixed which number is in FIXED_DEFMSTR field. This results in not having the one cycle latency when the fixed master re-trying access on the slave.
AT91C_MATRIX_FIXED_DEFMSTR0 EQU (0x7 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_LCDC EQU (0x3 << 18) ;- (MATRIX) LCDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_2DGC EQU (0x4 << 18) ;- (MATRIX) 2DGC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_ISI EQU (0x5 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_EMAC EQU (0x7 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR0_USB EQU (0x8 << 18) ;- (MATRIX) USB Master is Default Master
AT91C_MATRIX_ARBT         EQU (0x3 << 24) ;- (MATRIX) Arbitration Type
/* - -------- MATRIX_SCFG1 : (MATRIX Offset: 0x44) Slave Configuration Register 1 -------- */
AT91C_MATRIX_FIXED_DEFMSTR1 EQU (0x7 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_LCDC EQU (0x3 << 18) ;- (MATRIX) LCDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_2DGC EQU (0x4 << 18) ;- (MATRIX) 2DGC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_ISI EQU (0x5 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_EMAC EQU (0x7 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR1_USB EQU (0x8 << 18) ;- (MATRIX) USB Master is Default Master
/* - -------- MATRIX_SCFG2 : (MATRIX Offset: 0x48) Slave Configuration Register 2 -------- */
AT91C_MATRIX_FIXED_DEFMSTR2 EQU (0x1 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR2_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR2_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR2_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
/* - -------- MATRIX_SCFG3 : (MATRIX Offset: 0x4c) Slave Configuration Register 3 -------- */
AT91C_MATRIX_FIXED_DEFMSTR3 EQU (0x7 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_LCDC EQU (0x3 << 18) ;- (MATRIX) LCDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_2DGC EQU (0x4 << 18) ;- (MATRIX) 2DGC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_ISI EQU (0x5 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_EMAC EQU (0x7 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR3_USB EQU (0x8 << 18) ;- (MATRIX) USB Master is Default Master
/* - -------- MATRIX_SCFG4 : (MATRIX Offset: 0x50) Slave Configuration Register 4 -------- */
AT91C_MATRIX_FIXED_DEFMSTR4 EQU (0x3 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR4_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
/* - -------- MATRIX_SCFG5 : (MATRIX Offset: 0x54) Slave Configuration Register 5 -------- */
AT91C_MATRIX_FIXED_DEFMSTR5 EQU (0x3 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_LCDC EQU (0x3 << 18) ;- (MATRIX) LCDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_2DGC EQU (0x4 << 18) ;- (MATRIX) 2DGC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_ISI EQU (0x5 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_EMAC EQU (0x7 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR5_USB EQU (0x8 << 18) ;- (MATRIX) USB Master is Default Master
/* - -------- MATRIX_SCFG6 : (MATRIX Offset: 0x58) Slave Configuration Register 6 -------- */
AT91C_MATRIX_FIXED_DEFMSTR6 EQU (0x3 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_LCDC EQU (0x3 << 18) ;- (MATRIX) LCDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_2DGC EQU (0x4 << 18) ;- (MATRIX) 2DGC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_ISI EQU (0x5 << 18) ;- (MATRIX) ISI Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_EMAC EQU (0x7 << 18) ;- (MATRIX) EMAC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR6_USB EQU (0x8 << 18) ;- (MATRIX) USB Master is Default Master
/* - -------- MATRIX_SCFG7 : (MATRIX Offset: 0x5c) Slave Configuration Register 7 -------- */
AT91C_MATRIX_FIXED_DEFMSTR7 EQU (0x3 << 18) ;- (MATRIX) Fixed Index of Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_ARM926I EQU (0x0 << 18) ;- (MATRIX) ARM926EJ-S Instruction Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_ARM926D EQU (0x1 << 18) ;- (MATRIX) ARM926EJ-S Data Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_PDC EQU (0x2 << 18) ;- (MATRIX) PDC Master is Default Master
AT91C_MATRIX_FIXED_DEFMSTR7_DMA EQU (0x6 << 18) ;- (MATRIX) DMA Controller Master is Default Master
/* - -------- MATRIX_PRAS0 : (MATRIX Offset: 0x80) PRAS0 Register -------- */
AT91C_MATRIX_M0PR         EQU (0x3 <<  0) ;- (MATRIX) ARM926EJ-S Instruction priority
AT91C_MATRIX_M1PR         EQU (0x3 <<  4) ;- (MATRIX) ARM926EJ-S Data priority
AT91C_MATRIX_M2PR         EQU (0x3 <<  8) ;- (MATRIX) PDC priority
AT91C_MATRIX_M3PR         EQU (0x3 << 12) ;- (MATRIX) LCDC priority
AT91C_MATRIX_M4PR         EQU (0x3 << 16) ;- (MATRIX) 2DGC priority
AT91C_MATRIX_M5PR         EQU (0x3 << 20) ;- (MATRIX) ISI priority
AT91C_MATRIX_M6PR         EQU (0x3 << 24) ;- (MATRIX) DMA priority
AT91C_MATRIX_M7PR         EQU (0x3 << 28) ;- (MATRIX) EMAC priority
/* - -------- MATRIX_PRBS0 : (MATRIX Offset: 0x84) PRBS0 Register -------- */
AT91C_MATRIX_M8PR         EQU (0x3 <<  0) ;- (MATRIX) USB priority
/* - -------- MATRIX_PRAS1 : (MATRIX Offset: 0x88) PRAS1 Register -------- */
/* - -------- MATRIX_PRBS1 : (MATRIX Offset: 0x8c) PRBS1 Register -------- */
/* - -------- MATRIX_PRAS2 : (MATRIX Offset: 0x90) PRAS2 Register -------- */
/* - -------- MATRIX_PRBS2 : (MATRIX Offset: 0x94) PRBS2 Register -------- */
/* - -------- MATRIX_PRAS3 : (MATRIX Offset: 0x98) PRAS3 Register -------- */
/* - -------- MATRIX_PRBS3 : (MATRIX Offset: 0x9c) PRBS3 Register -------- */
/* - -------- MATRIX_PRAS4 : (MATRIX Offset: 0xa0) PRAS4 Register -------- */
/* - -------- MATRIX_PRBS4 : (MATRIX Offset: 0xa4) PRBS4 Register -------- */
/* - -------- MATRIX_PRAS5 : (MATRIX Offset: 0xa8) PRAS5 Register -------- */
/* - -------- MATRIX_PRBS5 : (MATRIX Offset: 0xac) PRBS5 Register -------- */
/* - -------- MATRIX_PRAS6 : (MATRIX Offset: 0xb0) PRAS6 Register -------- */
/* - -------- MATRIX_PRBS6 : (MATRIX Offset: 0xb4) PRBS6 Register -------- */
/* - -------- MATRIX_PRAS7 : (MATRIX Offset: 0xb8) PRAS7 Register -------- */
/* - -------- MATRIX_PRBS7 : (MATRIX Offset: 0xbc) PRBS7 Register -------- */
/* - -------- MATRIX_MRCR : (MATRIX Offset: 0x100) MRCR Register -------- */
AT91C_MATRIX_RCA926I      EQU (0x1 <<  0) ;- (MATRIX) Remap Command Bit for ARM926EJ-S Instruction
AT91C_MATRIX_RCA926D      EQU (0x1 <<  1) ;- (MATRIX) Remap Command Bit for ARM926EJ-S Data
AT91C_MATRIX_RCB2         EQU (0x1 <<  2) ;- (MATRIX) Remap Command Bit for PDC
AT91C_MATRIX_RCB3         EQU (0x1 <<  3) ;- (MATRIX) Remap Command Bit for LCD
AT91C_MATRIX_RCB4         EQU (0x1 <<  4) ;- (MATRIX) Remap Command Bit for 2DGC
AT91C_MATRIX_RCB5         EQU (0x1 <<  5) ;- (MATRIX) Remap Command Bit for ISI
AT91C_MATRIX_RCB6         EQU (0x1 <<  6) ;- (MATRIX) Remap Command Bit for DMA
AT91C_MATRIX_RCB7         EQU (0x1 <<  7) ;- (MATRIX) Remap Command Bit for EMAC
AT91C_MATRIX_RCB8         EQU (0x1 <<  8) ;- (MATRIX) Remap Command Bit for USB

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR AHB CCFG Interface*/
/* - ******************************************************************************/
/* - -------- CCFG_TCMR : (CCFG Offset: 0x4) TCM Configuration -------- */
AT91C_CCFG_ITCM_SIZE      EQU (0xF <<  0) ;- (CCFG) Size of ITCM enabled memory block
AT91C_CCFG_ITCM_SIZE_0KB  EQU (0x0) ;- (CCFG) 0 KB (No ITCM Memory)
AT91C_CCFG_ITCM_SIZE_16KB EQU (0x5) ;- (CCFG) 16 KB
AT91C_CCFG_ITCM_SIZE_32KB EQU (0x6) ;- (CCFG) 32 KB
AT91C_CCFG_DTCM_SIZE      EQU (0xF <<  4) ;- (CCFG) Size of DTCM enabled memory block
AT91C_CCFG_DTCM_SIZE_0KB  EQU (0x0 <<  4) ;- (CCFG) 0 KB (No DTCM Memory)
AT91C_CCFG_DTCM_SIZE_16KB EQU (0x5 <<  4) ;- (CCFG) 16 KB
AT91C_CCFG_DTCM_SIZE_32KB EQU (0x6 <<  4) ;- (CCFG) 32 KB
AT91C_CCFG_RM             EQU (0xF <<  8) ;- (CCFG) Read Margin registers
/* - -------- CCFG_UDPHS : (CCFG Offset: 0xc) USB Device HS configuration -------- */
AT91C_CCFG_DONT_USE_UTMI_LOCK EQU (0x1 <<  0) ;- (CCFG) 
AT91C_CCFG_DONT_USE_UTMI_LOCK_DONT_USE_LOCK EQU (0x0) ;- (CCFG) 
/* - -------- CCFG_EBICSA : (CCFG Offset: 0x10) EBI Chip Select Assignement Register -------- */
AT91C_EBI_CS1A            EQU (0x1 <<  1) ;- (CCFG) Chip Select 1 Assignment
AT91C_EBI_CS1A_SMC        EQU (0x0 <<  1) ;- (CCFG) Chip Select 1 is assigned to the Static Memory Controller.
AT91C_EBI_CS1A_SDRAMC     EQU (0x1 <<  1) ;- (CCFG) Chip Select 1 is assigned to the SDRAM Controller.
AT91C_EBI_CS3A            EQU (0x1 <<  3) ;- (CCFG) Chip Select 3 Assignment
AT91C_EBI_CS3A_SMC        EQU (0x0 <<  3) ;- (CCFG) Chip Select 3 is only assigned to the Static Memory Controller and NCS3 behaves as defined by the SMC.
AT91C_EBI_CS3A_SM         EQU (0x1 <<  3) ;- (CCFG) Chip Select 3 is assigned to the Static Memory Controller and the SmartMedia Logic is activated.
AT91C_EBI_CS4A            EQU (0x1 <<  4) ;- (CCFG) Chip Select 4 Assignment
AT91C_EBI_CS4A_SMC        EQU (0x0 <<  4) ;- (CCFG) Chip Select 4 is only assigned to the Static Memory Controller and NCS4 behaves as defined by the SMC.
AT91C_EBI_CS4A_CF         EQU (0x1 <<  4) ;- (CCFG) Chip Select 4 is assigned to the Static Memory Controller and the CompactFlash Logic (first slot) is activated.
AT91C_EBI_CS5A            EQU (0x1 <<  5) ;- (CCFG) Chip Select 5 Assignment
AT91C_EBI_CS5A_SMC        EQU (0x0 <<  5) ;- (CCFG) Chip Select 5 is only assigned to the Static Memory Controller and NCS5 behaves as defined by the SMC
AT91C_EBI_CS5A_CF         EQU (0x1 <<  5) ;- (CCFG) Chip Select 5 is assigned to the Static Memory Controller and the CompactFlash Logic (second slot) is activated.
AT91C_EBI_DBPUC           EQU (0x1 <<  8) ;- (CCFG) Data Bus Pull-up Configuration
AT91C_EBI_SUPPLY          EQU (0x1 << 16) ;- (CCFG) EBI supply set to 1.8

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
/* -              SOFTWARE API DEFINITION  FOR Power Management Controler*/
/* - ******************************************************************************/
/* - -------- PMC_SCER : (PMC Offset: 0x0) System Clock Enable Register -------- */
AT91C_PMC_PCK             EQU (0x1 <<  0) ;- (PMC) Processor Clock
AT91C_PMC_OTG             EQU (0x1 <<  5) ;- (PMC) USB OTG Clock
AT91C_PMC_UHP             EQU (0x1 <<  6) ;- (PMC) USB Host Port Clock
AT91C_PMC_UDP             EQU (0x1 <<  7) ;- (PMC) USB Device Port Clock
AT91C_PMC_PCK0            EQU (0x1 <<  8) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK1            EQU (0x1 <<  9) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK2            EQU (0x1 << 10) ;- (PMC) Programmable Clock Output
AT91C_PMC_PCK3            EQU (0x1 << 11) ;- (PMC) Programmable Clock Output
/* - -------- PMC_SCDR : (PMC Offset: 0x4) System Clock Disable Register -------- */
/* - -------- PMC_SCSR : (PMC Offset: 0x8) System Clock Status Register -------- */
/* - -------- CKGR_UCKR : (PMC Offset: 0x1c) UTMI Clock Configuration Register -------- */
AT91C_CKGR_UPLLEN         EQU (0x1 << 16) ;- (PMC) UTMI PLL Enable
AT91C_CKGR_UPLLEN_DISABLED EQU (0x0 << 16) ;- (PMC) The UTMI PLL is disabled
AT91C_CKGR_UPLLEN_ENABLED EQU (0x1 << 16) ;- (PMC) The UTMI PLL is enabled
AT91C_CKGR_PLLCOUNT       EQU (0xF << 20) ;- (PMC) UTMI Oscillator Start-up Time
AT91C_CKGR_BIASEN         EQU (0x1 << 24) ;- (PMC) UTMI BIAS Enable
AT91C_CKGR_BIASEN_DISABLED EQU (0x0 << 24) ;- (PMC) The UTMI BIAS is disabled
AT91C_CKGR_BIASEN_ENABLED EQU (0x1 << 24) ;- (PMC) The UTMI BIAS is enabled
AT91C_CKGR_BIASCOUNT      EQU (0xF << 28) ;- (PMC) UTMI BIAS Start-up Time
/* - -------- CKGR_MOR : (PMC Offset: 0x20) Main Oscillator Register -------- */
AT91C_CKGR_MOSCEN         EQU (0x1 <<  0) ;- (PMC) Main Oscillator Enable
AT91C_CKGR_OSCBYPASS      EQU (0x1 <<  1) ;- (PMC) Main Oscillator Bypass
AT91C_CKGR_OSCOUNT        EQU (0xFF <<  8) ;- (PMC) Main Oscillator Start-up Time
/* - -------- CKGR_MCFR : (PMC Offset: 0x24) Main Clock Frequency Register -------- */
AT91C_CKGR_MAINF          EQU (0xFFFF <<  0) ;- (PMC) Main Clock Frequency
AT91C_CKGR_MAINRDY        EQU (0x1 << 16) ;- (PMC) Main Clock Ready
/* - -------- CKGR_PLLAR : (PMC Offset: 0x28) PLL A Register -------- */
AT91C_CKGR_DIVA           EQU (0xFF <<  0) ;- (PMC) Divider A Selected
AT91C_CKGR_DIVA_0         EQU (0x0) ;- (PMC) Divider A output is 0
AT91C_CKGR_DIVA_BYPASS    EQU (0x1) ;- (PMC) Divider A is bypassed
AT91C_CKGR_PLLACOUNT      EQU (0x3F <<  8) ;- (PMC) PLL A Counter
AT91C_CKGR_OUTA           EQU (0x3 << 14) ;- (PMC) PLL A Output Frequency Range
AT91C_CKGR_OUTA_0         EQU (0x0 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_1         EQU (0x1 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_2         EQU (0x2 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_OUTA_3         EQU (0x3 << 14) ;- (PMC) Please refer to the PLLA datasheet
AT91C_CKGR_MULA           EQU (0x7FF << 16) ;- (PMC) PLL A Multiplier
AT91C_CKGR_SRCA           EQU (0x1 << 29) ;- (PMC) 
/* - -------- CKGR_PLLBR : (PMC Offset: 0x2c) PLL B Register -------- */
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
AT91C_CKGR_USBDIV         EQU (0x3 << 28) ;- (PMC) Divider for USB Clocks
AT91C_CKGR_USBDIV_0       EQU (0x0 << 28) ;- (PMC) Divider output is PLL clock output
AT91C_CKGR_USBDIV_1       EQU (0x1 << 28) ;- (PMC) Divider output is PLL clock output divided by 2
AT91C_CKGR_USBDIV_2       EQU (0x2 << 28) ;- (PMC) Divider output is PLL clock output divided by 4
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
AT91C_PMC_MDIV_4          EQU (0x2 <<  8) ;- (PMC) The processor clock is four times faster than the master clock
/* - -------- PMC_PCKR : (PMC Offset: 0x40) Programmable Clock Register -------- */
/* - -------- PMC_IER : (PMC Offset: 0x60) PMC Interrupt Enable Register -------- */
AT91C_PMC_MOSCS           EQU (0x1 <<  0) ;- (PMC) MOSC Status/Enable/Disable/Mask
AT91C_PMC_LOCKA           EQU (0x1 <<  1) ;- (PMC) PLL A Status/Enable/Disable/Mask
AT91C_PMC_LOCKB           EQU (0x1 <<  2) ;- (PMC) PLL B Status/Enable/Disable/Mask
AT91C_PMC_MCKRDY          EQU (0x1 <<  3) ;- (PMC) Master Clock Status/Enable/Disable/Mask
AT91C_PMC_LOCKU           EQU (0x1 <<  6) ;- (PMC) PLL UTMI Status/Enable/Disable/Mask
AT91C_PMC_PCK0RDY         EQU (0x1 <<  8) ;- (PMC) PCK0_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK1RDY         EQU (0x1 <<  9) ;- (PMC) PCK1_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK2RDY         EQU (0x1 << 10) ;- (PMC) PCK2_RDY Status/Enable/Disable/Mask
AT91C_PMC_PCK3RDY         EQU (0x1 << 11) ;- (PMC) PCK3_RDY Status/Enable/Disable/Mask
/* - -------- PMC_IDR : (PMC Offset: 0x64) PMC Interrupt Disable Register -------- */
/* - -------- PMC_SR : (PMC Offset: 0x68) PMC Status Register -------- */
/* - -------- PMC_IMR : (PMC Offset: 0x6c) PMC Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Clock Generator Controler*/
/* - ******************************************************************************/
/* - -------- CKGR_UCKR : (CKGR Offset: 0x0) UTMI Clock Configuration Register -------- */
/* - -------- CKGR_MOR : (CKGR Offset: 0x4) Main Oscillator Register -------- */
/* - -------- CKGR_MCFR : (CKGR Offset: 0x8) Main Clock Frequency Register -------- */
/* - -------- CKGR_PLLAR : (CKGR Offset: 0xc) PLL A Register -------- */
/* - -------- CKGR_PLLBR : (CKGR Offset: 0x10) PLL B Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Reset Controller Interface*/
/* - ******************************************************************************/
/* - -------- RSTC_RCR : (RSTC Offset: 0x0) Reset Control Register -------- */
AT91C_RSTC_PROCRST        EQU (0x1 <<  0) ;- (RSTC) Processor Reset
AT91C_RSTC_ICERST         EQU (0x1 <<  1) ;- (RSTC) ICE Interface Reset
AT91C_RSTC_PERRST         EQU (0x1 <<  2) ;- (RSTC) Peripheral Reset
AT91C_RSTC_EXTRST         EQU (0x1 <<  3) ;- (RSTC) External Reset
AT91C_RSTC_KEY            EQU (0xFF << 24) ;- (RSTC) Password
/* - -------- RSTC_RSR : (RSTC Offset: 0x4) Reset Status Register -------- */
AT91C_RSTC_URSTS          EQU (0x1 <<  0) ;- (RSTC) User Reset Status
AT91C_RSTC_RSTTYP         EQU (0x7 <<  8) ;- (RSTC) Reset Type
AT91C_RSTC_RSTTYP_GENERAL EQU (0x0 <<  8) ;- (RSTC) General reset. Both VDDCORE and VDDBU rising.
AT91C_RSTC_RSTTYP_WAKEUP  EQU (0x1 <<  8) ;- (RSTC) WakeUp Reset. VDDCORE rising.
AT91C_RSTC_RSTTYP_WATCHDOG EQU (0x2 <<  8) ;- (RSTC) Watchdog Reset. Watchdog overflow occured.
AT91C_RSTC_RSTTYP_SOFTWARE EQU (0x3 <<  8) ;- (RSTC) Software Reset. Processor reset required by the software.
AT91C_RSTC_RSTTYP_USER    EQU (0x4 <<  8) ;- (RSTC) User Reset. NRST pin detected low.
AT91C_RSTC_NRSTL          EQU (0x1 << 16) ;- (RSTC) NRST pin level
AT91C_RSTC_SRCMP          EQU (0x1 << 17) ;- (RSTC) Software Reset Command in Progress.
/* - -------- RSTC_RMR : (RSTC Offset: 0x8) Reset Mode Register -------- */
AT91C_RSTC_URSTEN         EQU (0x1 <<  0) ;- (RSTC) User Reset Enable
AT91C_RSTC_URSTIEN        EQU (0x1 <<  4) ;- (RSTC) User Reset Interrupt Enable
AT91C_RSTC_ERSTL          EQU (0xF <<  8) ;- (RSTC) User Reset Enable

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Shut Down Controller Interface*/
/* - ******************************************************************************/
/* - -------- SHDWC_SHCR : (SHDWC Offset: 0x0) Shut Down Control Register -------- */
AT91C_SHDWC_SHDW          EQU (0x1 <<  0) ;- (SHDWC) Processor Reset
AT91C_SHDWC_KEY           EQU (0xFF << 24) ;- (SHDWC) Shut down KEY Password
/* - -------- SHDWC_SHMR : (SHDWC Offset: 0x4) Shut Down Mode Register -------- */
AT91C_SHDWC_WKMODE0       EQU (0x3 <<  0) ;- (SHDWC) Wake Up 0 Mode Selection
AT91C_SHDWC_WKMODE0_NONE  EQU (0x0) ;- (SHDWC) None. No detection is performed on the wake up input.
AT91C_SHDWC_WKMODE0_HIGH  EQU (0x1) ;- (SHDWC) High Level.
AT91C_SHDWC_WKMODE0_LOW   EQU (0x2) ;- (SHDWC) Low Level.
AT91C_SHDWC_WKMODE0_ANYLEVEL EQU (0x3) ;- (SHDWC) Any level change.
AT91C_SHDWC_CPTWK0        EQU (0xF <<  4) ;- (SHDWC) Counter On Wake Up 0
AT91C_SHDWC_WKMODE1       EQU (0x3 <<  8) ;- (SHDWC) Wake Up 1 Mode Selection
AT91C_SHDWC_WKMODE1_NONE  EQU (0x0 <<  8) ;- (SHDWC) None. No detection is performed on the wake up input.
AT91C_SHDWC_WKMODE1_HIGH  EQU (0x1 <<  8) ;- (SHDWC) High Level.
AT91C_SHDWC_WKMODE1_LOW   EQU (0x2 <<  8) ;- (SHDWC) Low Level.
AT91C_SHDWC_WKMODE1_ANYLEVEL EQU (0x3 <<  8) ;- (SHDWC) Any level change.
AT91C_SHDWC_CPTWK1        EQU (0xF << 12) ;- (SHDWC) Counter On Wake Up 1
AT91C_SHDWC_RTTWKEN       EQU (0x1 << 16) ;- (SHDWC) Real Time Timer Wake Up Enable
AT91C_SHDWC_RTCWKEN       EQU (0x1 << 17) ;- (SHDWC) Real Time Clock Wake Up Enable
/* - -------- SHDWC_SHSR : (SHDWC Offset: 0x8) Shut Down Status Register -------- */
AT91C_SHDWC_WAKEUP0       EQU (0x1 <<  0) ;- (SHDWC) Wake Up 0 Status
AT91C_SHDWC_WAKEUP1       EQU (0x1 <<  1) ;- (SHDWC) Wake Up 1 Status
AT91C_SHDWC_FWKUP         EQU (0x1 <<  2) ;- (SHDWC) Force Wake Up Status
AT91C_SHDWC_RTTWK         EQU (0x1 << 16) ;- (SHDWC) Real Time Timer wake Up
AT91C_SHDWC_RTCWK         EQU (0x1 << 17) ;- (SHDWC) Real Time Clock wake Up

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Real Time Timer Controller Interface*/
/* - ******************************************************************************/
/* - -------- RTTC_RTMR : (RTTC Offset: 0x0) Real-time Mode Register -------- */
AT91C_RTTC_RTPRES         EQU (0xFFFF <<  0) ;- (RTTC) Real-time Timer Prescaler Value
AT91C_RTTC_ALMIEN         EQU (0x1 << 16) ;- (RTTC) Alarm Interrupt Enable
AT91C_RTTC_RTTINCIEN      EQU (0x1 << 17) ;- (RTTC) Real Time Timer Increment Interrupt Enable
AT91C_RTTC_RTTRST         EQU (0x1 << 18) ;- (RTTC) Real Time Timer Restart
/* - -------- RTTC_RTAR : (RTTC Offset: 0x4) Real-time Alarm Register -------- */
AT91C_RTTC_ALMV           EQU (0x0 <<  0) ;- (RTTC) Alarm Value
/* - -------- RTTC_RTVR : (RTTC Offset: 0x8) Current Real-time Value Register -------- */
AT91C_RTTC_CRTV           EQU (0x0 <<  0) ;- (RTTC) Current Real-time Value
/* - -------- RTTC_RTSR : (RTTC Offset: 0xc) Real-time Status Register -------- */
AT91C_RTTC_ALMS           EQU (0x1 <<  0) ;- (RTTC) Real-time Alarm Status
AT91C_RTTC_RTTINC         EQU (0x1 <<  1) ;- (RTTC) Real-time Timer Increment

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
AT91C_MCI_RDPROOF         EQU (0x1 << 11) ;- (MCI) Read Proof Enable
AT91C_MCI_WRPROOF         EQU (0x1 << 12) ;- (MCI) Write Proof Enable
AT91C_MCI_PDCFBYTE        EQU (0x1 << 13) ;- (MCI) PDC Force Byte Transfer
AT91C_MCI_PDCPADV         EQU (0x1 << 14) ;- (MCI) PDC Padding Value
AT91C_MCI_PDCMODE         EQU (0x1 << 15) ;- (MCI) PDC Oriented Mode
AT91C_MCI_BLKLEN          EQU (0xFFFF << 16) ;- (MCI) Data Block Length
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
AT91C_MCI_SCDSEL          EQU (0x3 <<  0) ;- (MCI) SD Card Selector
AT91C_MCI_SCDBUS          EQU (0x1 <<  7) ;- (MCI) SDCard/SDIO Bus Width
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
/* - -------- MCI_BLKR : (MCI Offset: 0x18) MCI Block Register -------- */
AT91C_MCI_BCNT            EQU (0xFFFF <<  0) ;- (MCI) MMC/SDIO Block Count / SDIO Byte Count
/* - -------- MCI_SR : (MCI Offset: 0x40) MCI Status Register -------- */
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
AT91C_US_VAR_SYNC         EQU (0x1 << 22) ;- (USART) Variable synchronization of command/data sync Start Frame Delimiter
AT91C_US_MAX_ITER         EQU (0x1 << 24) ;- (USART) Number of Repetitions
AT91C_US_FILTER           EQU (0x1 << 28) ;- (USART) Receive Line Filter
AT91C_US_MANMODE          EQU (0x1 << 29) ;- (USART) Manchester Encoder/Decoder Enable
AT91C_US_MODSYNC          EQU (0x1 << 30) ;- (USART) Manchester Synchronization mode
AT91C_US_ONEBIT           EQU (0x1 << 31) ;- (USART) Start Frame Delimiter selector
/* - -------- US_IER : (USART Offset: 0x8) Debug Unit Interrupt Enable Register -------- */
AT91C_US_RXBRK            EQU (0x1 <<  2) ;- (USART) Break Received/End of Break
AT91C_US_TIMEOUT          EQU (0x1 <<  8) ;- (USART) Receiver Time-out
AT91C_US_ITERATION        EQU (0x1 << 10) ;- (USART) Max number of Repetitions Reached
AT91C_US_NACK             EQU (0x1 << 13) ;- (USART) Non Acknowledge
AT91C_US_RIIC             EQU (0x1 << 16) ;- (USART) Ring INdicator Input Change Flag
AT91C_US_DSRIC            EQU (0x1 << 17) ;- (USART) Data Set Ready Input Change Flag
AT91C_US_DCDIC            EQU (0x1 << 18) ;- (USART) Data Carrier Flag
AT91C_US_CTSIC            EQU (0x1 << 19) ;- (USART) Clear To Send Input Change Flag
AT91C_US_MANE             EQU (0x1 << 20) ;- (USART) Manchester Error Interrupt
/* - -------- US_IDR : (USART Offset: 0xc) Debug Unit Interrupt Disable Register -------- */
/* - -------- US_IMR : (USART Offset: 0x10) Debug Unit Interrupt Mask Register -------- */
/* - -------- US_CSR : (USART Offset: 0x14) Debug Unit Channel Status Register -------- */
AT91C_US_RI               EQU (0x1 << 20) ;- (USART) Image of RI Input
AT91C_US_DSR              EQU (0x1 << 21) ;- (USART) Image of DSR Input
AT91C_US_DCD              EQU (0x1 << 22) ;- (USART) Image of DCD Input
AT91C_US_CTS              EQU (0x1 << 23) ;- (USART) Image of CTS Input
AT91C_US_MANERR           EQU (0x1 << 24) ;- (USART) Manchester Error
/* - -------- US_MAN : (USART Offset: 0x50) Manchester Encoder Decoder Register -------- */
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
AT91C_PWMC_CHID4          EQU (0x1 <<  4) ;- (PWMC) Channel ID 4
AT91C_PWMC_CHID5          EQU (0x1 <<  5) ;- (PWMC) Channel ID 5
AT91C_PWMC_CHID6          EQU (0x1 <<  6) ;- (PWMC) Channel ID 6
AT91C_PWMC_CHID7          EQU (0x1 <<  7) ;- (PWMC) Channel ID 7
/* - -------- PWMC_DIS : (PWMC Offset: 0x8) PWMC Disable Register -------- */
/* - -------- PWMC_SR : (PWMC Offset: 0xc) PWMC Status Register -------- */
/* - -------- PWMC_IER : (PWMC Offset: 0x10) PWMC Interrupt Enable Register -------- */
/* - -------- PWMC_IDR : (PWMC Offset: 0x14) PWMC Interrupt Disable Register -------- */
/* - -------- PWMC_IMR : (PWMC Offset: 0x18) PWMC Interrupt Mask Register -------- */
/* - -------- PWMC_ISR : (PWMC Offset: 0x1c) PWMC Interrupt Status Register -------- */

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
/* -              SOFTWARE API DEFINITION  FOR TSADC*/
/* - ******************************************************************************/
/* - -------- TSADC_CR : (TSADC Offset: 0x0) Control Register -------- */
AT91C_TSADC_SWRST         EQU (0x1 <<  0) ;- (TSADC) Software Reset
AT91C_TSADC_START         EQU (0x1 <<  1) ;- (TSADC) Start Conversion
/* - -------- TSADC_MR : (TSADC Offset: 0x4) Mode Register -------- */
AT91C_TSADC_TSAMOD        EQU (0x3 <<  0) ;- (TSADC) Touch Screen ADC Mode
AT91C_TSADC_TSAMOD_ADC_ONLY_MODE EQU (0x0) ;- (TSADC) ADC Mode
AT91C_TSADC_TSAMOD_TS_ONLY_MODE EQU (0x1) ;- (TSADC) Touch Screen Only Mode
AT91C_TSADC_LOWRES        EQU (0x1 <<  4) ;- (TSADC) ADC Resolution
AT91C_TSADC_SLEEP         EQU (0x1 <<  5) ;- (TSADC) Sleep Mode
AT91C_TSADC_PENDET        EQU (0x1 <<  6) ;- (TSADC) Pen Detect Selection
AT91C_TSADC_PRESCAL       EQU (0x3F <<  8) ;- (TSADC) Prescaler Rate Selection
AT91C_TSADC_STARTUP       EQU (0x7F << 16) ;- (TSADC) Startup Time
AT91C_TSADC_SHTIM         EQU (0xF << 24) ;- (TSADC) Sample and Hold Time for ADC Channels
AT91C_TSADC_PENDBC        EQU (0xF << 28) ;- (TSADC) Pen Detect Debouncing Period
/* - -------- TSADC_TRGR : (TSADC Offset: 0x8) Trigger Register -------- */
AT91C_TSADC_TRGMOD        EQU (0x7 <<  0) ;- (TSADC) Trigger Mode
AT91C_TSADC_TRGMOD_NO_TRIGGER EQU (0x0) ;- (TSADC) No Trigger
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_RE EQU (0x1) ;- (TSADC) External Trigger Rising Edge
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_FE EQU (0x2) ;- (TSADC) External Trigger Falling Edge
AT91C_TSADC_TRGMOD_EXTERNAL_TRIGGER_AE EQU (0x3) ;- (TSADC) External Trigger Any Edge
AT91C_TSADC_TRGMOD_PENDET_TRIGGER EQU (0x4) ;- (TSADC) Pen Detect Trigger (only if PENDET is set and in Touch Screen mode only)
AT91C_TSADC_TRGMOD_PERIODIC_TRIGGER EQU (0x5) ;- (TSADC) Periodic Trigger (wrt TRGPER)
AT91C_TSADC_TRGMOD_CONT_TRIGGER EQU (0x6) ;- (TSADC) Continuous Trigger
AT91C_TSADC_TRGPER        EQU (0xFFFF << 16) ;- (TSADC) Trigger Period
/* - -------- TSADC_TSR : (TSADC Offset: 0xc) Touch Screen Register -------- */
AT91C_TSADC_TSSHTIM       EQU (0xF << 24) ;- (TSADC) Sample and Hold Time for Touch Screen Channels
/* - -------- TSADC_CHER : (TSADC Offset: 0x10) Channel Enable Register -------- */
AT91C_TSADC_CHENA0        EQU (0x1 <<  0) ;- (TSADC) Channel 0 Enable
AT91C_TSADC_CHENA1        EQU (0x1 <<  1) ;- (TSADC) Channel 1 Enable
AT91C_TSADC_CHENA2        EQU (0x1 <<  2) ;- (TSADC) Channel 2 Enable
AT91C_TSADC_CHENA3        EQU (0x1 <<  3) ;- (TSADC) Channel 3 Enable
AT91C_TSADC_CHENA4        EQU (0x1 <<  4) ;- (TSADC) Channel 4 Enable
AT91C_TSADC_CHENA5        EQU (0x1 <<  5) ;- (TSADC) Channel 5 Enable
/* - -------- TSADC_CHDR : (TSADC Offset: 0x14) Channel Disable Register -------- */
AT91C_TSADC_CHDIS0        EQU (0x1 <<  0) ;- (TSADC) Channel 0 Disable
AT91C_TSADC_CHDIS1        EQU (0x1 <<  1) ;- (TSADC) Channel 1 Disable
AT91C_TSADC_CHDIS2        EQU (0x1 <<  2) ;- (TSADC) Channel 2 Disable
AT91C_TSADC_CHDIS3        EQU (0x1 <<  3) ;- (TSADC) Channel 3 Disable
AT91C_TSADC_CHDIS4        EQU (0x1 <<  4) ;- (TSADC) Channel 4 Disable
AT91C_TSADC_CHDIS5        EQU (0x1 <<  5) ;- (TSADC) Channel 5 Disable
/* - -------- TSADC_CHSR : (TSADC Offset: 0x18) Channel Status Register -------- */
AT91C_TSADC_CHS0          EQU (0x1 <<  0) ;- (TSADC) Channel 0 Status
AT91C_TSADC_CHS1          EQU (0x1 <<  1) ;- (TSADC) Channel 1 Status
AT91C_TSADC_CHS2          EQU (0x1 <<  2) ;- (TSADC) Channel 2 Status
AT91C_TSADC_CHS3          EQU (0x1 <<  3) ;- (TSADC) Channel 3 Status
AT91C_TSADC_CHS4          EQU (0x1 <<  4) ;- (TSADC) Channel 4 Status
AT91C_TSADC_CHS5          EQU (0x1 <<  5) ;- (TSADC) Channel 5 Status
/* - -------- TSADC_SR : (TSADC Offset: 0x1c) Status Register -------- */
AT91C_TSADC_EOC0          EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion
AT91C_TSADC_EOC1          EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion
AT91C_TSADC_EOC2          EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion
AT91C_TSADC_EOC3          EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion
AT91C_TSADC_EOC4          EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion
AT91C_TSADC_EOC5          EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion
AT91C_TSADC_OVRE0         EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error
AT91C_TSADC_OVRE1         EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error
AT91C_TSADC_OVRE2         EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error
AT91C_TSADC_OVRE3         EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error
AT91C_TSADC_OVRE4         EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error
AT91C_TSADC_OVRE5         EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error
AT91C_TSADC_DRDY          EQU (0x1 << 16) ;- (TSADC) Data Ready
AT91C_TSADC_GOVRE         EQU (0x1 << 17) ;- (TSADC) General Overrun Error
AT91C_TSADC_ENDRX         EQU (0x1 << 18) ;- (TSADC) End of RX Buffer
AT91C_TSADC_RXBUFF        EQU (0x1 << 19) ;- (TSADC) RX Buffer Full
AT91C_TSADC_PENCNT        EQU (0x1 << 20) ;- (TSADC) Pen Contact
AT91C_TSADC_NOCNT         EQU (0x1 << 21) ;- (TSADC) No Contact
/* - -------- TSADC_LCDR : (TSADC Offset: 0x20) Last Converted Data Register -------- */
AT91C_TSADC_LDATA         EQU (0x3FF <<  0) ;- (TSADC) Last Converted Data
/* - -------- TSADC_IER : (TSADC Offset: 0x24) Interrupt Enable Register -------- */
AT91C_TSADC_IENAEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Enable
AT91C_TSADC_IENAOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Enable
AT91C_TSADC_IENAOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Enable
AT91C_TSADC_IENADRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Enable
AT91C_TSADC_IENAGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Enable
AT91C_TSADC_IENAENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Enable
AT91C_TSADC_IENARXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Enable
AT91C_TSADC_IENAPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Enable
AT91C_TSADC_IENANOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Enable
/* - -------- TSADC_IDR : (TSADC Offset: 0x28) Interrupt Disable Register -------- */
AT91C_TSADC_IDISEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Disable
AT91C_TSADC_IDISOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Disable
AT91C_TSADC_IDISOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Disable
AT91C_TSADC_IDISDRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Disable
AT91C_TSADC_IDISGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Disable
AT91C_TSADC_IDISENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Disable
AT91C_TSADC_IDISRXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Disable
AT91C_TSADC_IDISPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Disable
AT91C_TSADC_IDISNOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Disable
/* - -------- TSADC_IMR : (TSADC Offset: 0x2c) Interrupt Mask Register -------- */
AT91C_TSADC_IMSKEOC0      EQU (0x1 <<  0) ;- (TSADC) Channel 0 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC1      EQU (0x1 <<  1) ;- (TSADC) Channel 1 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC2      EQU (0x1 <<  2) ;- (TSADC) Channel 2 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC3      EQU (0x1 <<  3) ;- (TSADC) Channel 3 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC4      EQU (0x1 <<  4) ;- (TSADC) Channel 4 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKEOC5      EQU (0x1 <<  5) ;- (TSADC) Channel 5 End Of Conversion Interrupt Mask
AT91C_TSADC_IMSKOVRE0     EQU (0x1 <<  8) ;- (TSADC) Channel 0 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE1     EQU (0x1 <<  9) ;- (TSADC) Channel 1 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE2     EQU (0x1 << 10) ;- (TSADC) Channel 2 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE3     EQU (0x1 << 11) ;- (TSADC) Channel 3 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE4     EQU (0x1 << 12) ;- (TSADC) Channel 4 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKOVRE5     EQU (0x1 << 13) ;- (TSADC) Channel 5 Overrun Error Interrupt Mask
AT91C_TSADC_IMSKDRDY      EQU (0x1 << 16) ;- (TSADC) Data Ready Interrupt Mask
AT91C_TSADC_IMSKGOVRE     EQU (0x1 << 17) ;- (TSADC) General Overrun Error Interrupt Mask
AT91C_TSADC_IMSKENDRX     EQU (0x1 << 18) ;- (TSADC) End of RX Buffer Interrupt Mask
AT91C_TSADC_IMSKRXBUFF    EQU (0x1 << 19) ;- (TSADC) RX Buffer Full Interrupt Mask
AT91C_TSADC_IMSKPENCNT    EQU (0x1 << 20) ;- (TSADC) Pen Contact Interrupt Mask
AT91C_TSADC_IMSKNOCNT     EQU (0x1 << 21) ;- (TSADC) No Contact Interrupt Mask
/* - -------- TSADC_CDR0 : (TSADC Offset: 0x30) Channel 0 Data Register -------- */
AT91C_TSADC_DATA0         EQU (0x3FF <<  0) ;- (TSADC) Channel 0 Data
/* - -------- TSADC_CDR1 : (TSADC Offset: 0x34) Channel 1 Data Register -------- */
AT91C_TSADC_DATA1         EQU (0x3FF <<  0) ;- (TSADC) Channel 1 Data
/* - -------- TSADC_CDR2 : (TSADC Offset: 0x38) Channel 2 Data Register -------- */
AT91C_TSADC_DATA2         EQU (0x3FF <<  0) ;- (TSADC) Channel 2 Data
/* - -------- TSADC_CDR3 : (TSADC Offset: 0x3c) Channel 3 Data Register -------- */
AT91C_TSADC_DATA3         EQU (0x3FF <<  0) ;- (TSADC) Channel 3 Data
/* - -------- TSADC_CDR4 : (TSADC Offset: 0x40) Channel 4 Data Register -------- */
AT91C_TSADC_DATA4         EQU (0x3FF <<  0) ;- (TSADC) Channel 4 Data
/* - -------- TSADC_CDR5 : (TSADC Offset: 0x44) Channel 5 Data Register -------- */
AT91C_TSADC_DATA5         EQU (0x3FF <<  0) ;- (TSADC) Channel 5 Data

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR UDPHS Enpoint FIFO data register*/
/* - ******************************************************************************/

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR UDPHS Endpoint struct*/
/* - ******************************************************************************/
/* - -------- UDPHS_EPTCFG : (UDPHS_EPT Offset: 0x0) UDPHS Endpoint Config Register -------- */
AT91C_UDPHS_EPT_SIZE      EQU (0x7 <<  0) ;- (UDPHS_EPT) Endpoint Size
AT91C_UDPHS_EPT_SIZE_8    EQU (0x0) ;- (UDPHS_EPT)    8 bytes
AT91C_UDPHS_EPT_SIZE_16   EQU (0x1) ;- (UDPHS_EPT)   16 bytes
AT91C_UDPHS_EPT_SIZE_32   EQU (0x2) ;- (UDPHS_EPT)   32 bytes
AT91C_UDPHS_EPT_SIZE_64   EQU (0x3) ;- (UDPHS_EPT)   64 bytes
AT91C_UDPHS_EPT_SIZE_128  EQU (0x4) ;- (UDPHS_EPT)  128 bytes
AT91C_UDPHS_EPT_SIZE_256  EQU (0x5) ;- (UDPHS_EPT)  256 bytes
AT91C_UDPHS_EPT_SIZE_512  EQU (0x6) ;- (UDPHS_EPT)  512 bytes
AT91C_UDPHS_EPT_SIZE_1024 EQU (0x7) ;- (UDPHS_EPT) 1024 bytes
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
AT91C_UDPHS_BK_NUMBER_3   EQU (0x3 <<  6) ;- (UDPHS_EPT) Triple Bank (Bank0 / Bank1 / Bank2)
AT91C_UDPHS_NB_TRANS      EQU (0x3 <<  8) ;- (UDPHS_EPT) Number Of Transaction per Micro-Frame (High-Bandwidth iso only)
AT91C_UDPHS_EPT_MAPD      EQU (0x1 << 31) ;- (UDPHS_EPT) Endpoint Mapped (read only
/* - -------- UDPHS_EPTCTLENB : (UDPHS_EPT Offset: 0x4) UDPHS Endpoint Control Enable Register -------- */
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
/* - -------- UDPHS_EPTCTLDIS : (UDPHS_EPT Offset: 0x8) UDPHS Endpoint Control Disable Register -------- */
AT91C_UDPHS_EPT_DISABL    EQU (0x1 <<  0) ;- (UDPHS_EPT) Endpoint Disable
/* - -------- UDPHS_EPTCTL : (UDPHS_EPT Offset: 0xc) UDPHS Endpoint Control Register -------- */
/* - -------- UDPHS_EPTSETSTA : (UDPHS_EPT Offset: 0x14) UDPHS Endpoint Set Status Register -------- */
AT91C_UDPHS_FRCESTALL     EQU (0x1 <<  5) ;- (UDPHS_EPT) Stall Handshake Request Set/Clear/Status
AT91C_UDPHS_KILL_BANK     EQU (0x1 <<  9) ;- (UDPHS_EPT) KILL Bank
/* - -------- UDPHS_EPTCLRSTA : (UDPHS_EPT Offset: 0x18) UDPHS Endpoint Clear Status Register -------- */
AT91C_UDPHS_TOGGLESQ      EQU (0x1 <<  6) ;- (UDPHS_EPT) Data Toggle Clear
/* - -------- UDPHS_EPTSTA : (UDPHS_EPT Offset: 0x1c) UDPHS Endpoint Status Register -------- */
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
AT91C_UDPHS_BUSY_BANK_STA_11 EQU (0x3 << 18) ;- (UDPHS_EPT) 3 busy banks
AT91C_UDPHS_BYTE_COUNT    EQU (0x7FF << 20) ;- (UDPHS_EPT) UDPHS Byte Count

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR UDPHS DMA struct*/
/* - ******************************************************************************/
/* - -------- UDPHS_DMANXTDSC : (UDPHS_DMA Offset: 0x0) UDPHS DMA Next Descriptor Address Register -------- */
AT91C_UDPHS_NXT_DSC_ADD   EQU (0xFFFFFFF <<  4) ;- (UDPHS_DMA) next Channel Descriptor
/* - -------- UDPHS_DMAADDRESS : (UDPHS_DMA Offset: 0x4) UDPHS DMA Channel Address Register -------- */
AT91C_UDPHS_BUFF_ADD      EQU (0x0 <<  0) ;- (UDPHS_DMA) starting address of a DMA Channel transfer
/* - -------- UDPHS_DMACONTROL : (UDPHS_DMA Offset: 0x8) UDPHS DMA Channel Control Register -------- */
AT91C_UDPHS_CHANN_ENB     EQU (0x1 <<  0) ;- (UDPHS_DMA) Channel Enabled
AT91C_UDPHS_LDNXT_DSC     EQU (0x1 <<  1) ;- (UDPHS_DMA) Load Next Channel Transfer Descriptor Enable
AT91C_UDPHS_END_TR_EN     EQU (0x1 <<  2) ;- (UDPHS_DMA) Buffer Close Input Enable
AT91C_UDPHS_END_B_EN      EQU (0x1 <<  3) ;- (UDPHS_DMA) End of DMA Buffer Packet Validation
AT91C_UDPHS_END_TR_IT     EQU (0x1 <<  4) ;- (UDPHS_DMA) End Of Transfer Interrupt Enable
AT91C_UDPHS_END_BUFFIT    EQU (0x1 <<  5) ;- (UDPHS_DMA) End Of Channel Buffer Interrupt Enable
AT91C_UDPHS_DESC_LD_IT    EQU (0x1 <<  6) ;- (UDPHS_DMA) Descriptor Loaded Interrupt Enable
AT91C_UDPHS_BURST_LCK     EQU (0x1 <<  7) ;- (UDPHS_DMA) Burst Lock Enable
AT91C_UDPHS_BUFF_LENGTH   EQU (0xFFFF << 16) ;- (UDPHS_DMA) Buffer Byte Length (write only)
/* - -------- UDPHS_DMASTATUS : (UDPHS_DMA Offset: 0xc) UDPHS DMA Channelx Status Register -------- */
AT91C_UDPHS_CHANN_ACT     EQU (0x1 <<  1) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_TR_ST     EQU (0x1 <<  4) ;- (UDPHS_DMA) 
AT91C_UDPHS_END_BF_ST     EQU (0x1 <<  5) ;- (UDPHS_DMA) 
AT91C_UDPHS_DESC_LDST     EQU (0x1 <<  6) ;- (UDPHS_DMA) 
AT91C_UDPHS_BUFF_COUNT    EQU (0xFFFF << 16) ;- (UDPHS_DMA) 

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR UDPHS High Speed Device Interface*/
/* - ******************************************************************************/
/* - -------- UDPHS_CTRL : (UDPHS Offset: 0x0) UDPHS Control Register -------- */
AT91C_UDPHS_DEV_ADDR      EQU (0x7F <<  0) ;- (UDPHS) UDPHS Address
AT91C_UDPHS_FADDR_EN      EQU (0x1 <<  7) ;- (UDPHS) Function Address Enable
AT91C_UDPHS_EN_UDPHS      EQU (0x1 <<  8) ;- (UDPHS) UDPHS Enable
AT91C_UDPHS_DETACH        EQU (0x1 <<  9) ;- (UDPHS) Detach Command
AT91C_UDPHS_REWAKEUP      EQU (0x1 << 10) ;- (UDPHS) Send Remote Wake Up
AT91C_UDPHS_PULLD_DIS     EQU (0x1 << 11) ;- (UDPHS) PullDown Disable
/* - -------- UDPHS_FNUM : (UDPHS Offset: 0x4) UDPHS Frame Number Register -------- */
AT91C_UDPHS_MICRO_FRAME_NUM EQU (0x7 <<  0) ;- (UDPHS) Micro Frame Number
AT91C_UDPHS_FRAME_NUMBER  EQU (0x7FF <<  3) ;- (UDPHS) Frame Number as defined in the Packet Field Formats
AT91C_UDPHS_FNUM_ERR      EQU (0x1 << 31) ;- (UDPHS) Frame Number CRC Error
/* - -------- UDPHS_IEN : (UDPHS Offset: 0x10) UDPHS Interrupt Enable Register -------- */
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
AT91C_UDPHS_EPT_INT_7     EQU (0x1 << 15) ;- (UDPHS) Endpoint 7 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_8     EQU (0x1 << 16) ;- (UDPHS) Endpoint 8 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_9     EQU (0x1 << 17) ;- (UDPHS) Endpoint 9 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_10    EQU (0x1 << 18) ;- (UDPHS) Endpoint 10 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_11    EQU (0x1 << 19) ;- (UDPHS) Endpoint 11 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_12    EQU (0x1 << 20) ;- (UDPHS) Endpoint 12 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_13    EQU (0x1 << 21) ;- (UDPHS) Endpoint 13 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_14    EQU (0x1 << 22) ;- (UDPHS) Endpoint 14 Interrupt Enable/Status
AT91C_UDPHS_EPT_INT_15    EQU (0x1 << 23) ;- (UDPHS) Endpoint 15 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_1     EQU (0x1 << 25) ;- (UDPHS) DMA Channel 1 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_2     EQU (0x1 << 26) ;- (UDPHS) DMA Channel 2 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_3     EQU (0x1 << 27) ;- (UDPHS) DMA Channel 3 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_4     EQU (0x1 << 28) ;- (UDPHS) DMA Channel 4 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_5     EQU (0x1 << 29) ;- (UDPHS) DMA Channel 5 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_6     EQU (0x1 << 30) ;- (UDPHS) DMA Channel 6 Interrupt Enable/Status
AT91C_UDPHS_DMA_INT_7     EQU (0x1 << 31) ;- (UDPHS) DMA Channel 7 Interrupt Enable/Status
/* - -------- UDPHS_INTSTA : (UDPHS Offset: 0x14) UDPHS Interrupt Status Register -------- */
AT91C_UDPHS_SPEED         EQU (0x1 <<  0) ;- (UDPHS) Speed Status
/* - -------- UDPHS_CLRINT : (UDPHS Offset: 0x18) UDPHS Clear Interrupt Register -------- */
/* - -------- UDPHS_EPTRST : (UDPHS Offset: 0x1c) UDPHS Endpoints Reset Register -------- */
AT91C_UDPHS_RST_EPT_0     EQU (0x1 <<  0) ;- (UDPHS) Endpoint Reset 0
AT91C_UDPHS_RST_EPT_1     EQU (0x1 <<  1) ;- (UDPHS) Endpoint Reset 1
AT91C_UDPHS_RST_EPT_2     EQU (0x1 <<  2) ;- (UDPHS) Endpoint Reset 2
AT91C_UDPHS_RST_EPT_3     EQU (0x1 <<  3) ;- (UDPHS) Endpoint Reset 3
AT91C_UDPHS_RST_EPT_4     EQU (0x1 <<  4) ;- (UDPHS) Endpoint Reset 4
AT91C_UDPHS_RST_EPT_5     EQU (0x1 <<  5) ;- (UDPHS) Endpoint Reset 5
AT91C_UDPHS_RST_EPT_6     EQU (0x1 <<  6) ;- (UDPHS) Endpoint Reset 6
AT91C_UDPHS_RST_EPT_7     EQU (0x1 <<  7) ;- (UDPHS) Endpoint Reset 7
AT91C_UDPHS_RST_EPT_8     EQU (0x1 <<  8) ;- (UDPHS) Endpoint Reset 8
AT91C_UDPHS_RST_EPT_9     EQU (0x1 <<  9) ;- (UDPHS) Endpoint Reset 9
AT91C_UDPHS_RST_EPT_10    EQU (0x1 << 10) ;- (UDPHS) Endpoint Reset 10
AT91C_UDPHS_RST_EPT_11    EQU (0x1 << 11) ;- (UDPHS) Endpoint Reset 11
AT91C_UDPHS_RST_EPT_12    EQU (0x1 << 12) ;- (UDPHS) Endpoint Reset 12
AT91C_UDPHS_RST_EPT_13    EQU (0x1 << 13) ;- (UDPHS) Endpoint Reset 13
AT91C_UDPHS_RST_EPT_14    EQU (0x1 << 14) ;- (UDPHS) Endpoint Reset 14
AT91C_UDPHS_RST_EPT_15    EQU (0x1 << 15) ;- (UDPHS) Endpoint Reset 15
/* - -------- UDPHS_TSTSOFCNT : (UDPHS Offset: 0xd0) UDPHS Test SOF Counter Register -------- */
AT91C_UDPHS_SOFCNTMAX     EQU (0x3 <<  0) ;- (UDPHS) SOF Counter Max Value
AT91C_UDPHS_SOFCTLOAD     EQU (0x1 <<  7) ;- (UDPHS) SOF Counter Load
/* - -------- UDPHS_TSTCNTA : (UDPHS Offset: 0xd4) UDPHS Test A Counter Register -------- */
AT91C_UDPHS_CNTAMAX       EQU (0x7FFF <<  0) ;- (UDPHS) A Counter Max Value
AT91C_UDPHS_CNTALOAD      EQU (0x1 << 15) ;- (UDPHS) A Counter Load
/* - -------- UDPHS_TSTCNTB : (UDPHS Offset: 0xd8) UDPHS Test B Counter Register -------- */
AT91C_UDPHS_CNTBMAX       EQU (0x7FFF <<  0) ;- (UDPHS) B Counter Max Value
AT91C_UDPHS_CNTBLOAD      EQU (0x1 << 15) ;- (UDPHS) B Counter Load
/* - -------- UDPHS_TSTMODREG : (UDPHS Offset: 0xdc) UDPHS Test Mode Register -------- */
AT91C_UDPHS_TSTMODE       EQU (0x1F <<  1) ;- (UDPHS) UDPHS Core TestModeReg
/* - -------- UDPHS_TST : (UDPHS Offset: 0xe0) UDPHS Test Register -------- */
AT91C_UDPHS_SPEED_CFG     EQU (0x3 <<  0) ;- (UDPHS) Speed Configuration
AT91C_UDPHS_SPEED_CFG_NM  EQU (0x0) ;- (UDPHS) Normal Mode
AT91C_UDPHS_SPEED_CFG_RS  EQU (0x1) ;- (UDPHS) Reserved
AT91C_UDPHS_SPEED_CFG_HS  EQU (0x2) ;- (UDPHS) Force High Speed
AT91C_UDPHS_SPEED_CFG_FS  EQU (0x3) ;- (UDPHS) Force Full-Speed
AT91C_UDPHS_TST_J         EQU (0x1 <<  2) ;- (UDPHS) TestJMode
AT91C_UDPHS_TST_K         EQU (0x1 <<  3) ;- (UDPHS) TestKMode
AT91C_UDPHS_TST_PKT       EQU (0x1 <<  4) ;- (UDPHS) TestPacketMode
AT91C_UDPHS_OPMODE2       EQU (0x1 <<  5) ;- (UDPHS) OpMode2
/* - -------- UDPHS_RIPPADDRSIZE : (UDPHS Offset: 0xec) UDPHS PADDRSIZE Register -------- */
AT91C_UDPHS_IPPADDRSIZE   EQU (0x0 <<  0) ;- (UDPHS) 2^UDPHSDEV_PADDR_SIZE
/* - -------- UDPHS_RIPNAME1 : (UDPHS Offset: 0xf0) UDPHS Name Register -------- */
AT91C_UDPHS_IPNAME1       EQU (0x0 <<  0) ;- (UDPHS) ASCII string HUSB
/* - -------- UDPHS_RIPNAME2 : (UDPHS Offset: 0xf4) UDPHS Name Register -------- */
AT91C_UDPHS_IPNAME2       EQU (0x0 <<  0) ;- (UDPHS) ASCII string 2DEV
/* - -------- UDPHS_IPFEATURES : (UDPHS Offset: 0xf8) UDPHS Features Register -------- */
AT91C_UDPHS_EPT_NBR_MAX   EQU (0xF <<  0) ;- (UDPHS) Max Number of Endpoints
AT91C_UDPHS_DMA_CHANNEL_NBR EQU (0x7 <<  4) ;- (UDPHS) Number of DMA Channels
AT91C_UDPHS_DMA_B_SIZ     EQU (0x1 <<  7) ;- (UDPHS) DMA Buffer Size
AT91C_UDPHS_DMA_FIFO_WORD_DEPTH EQU (0xF <<  8) ;- (UDPHS) DMA FIFO Depth in words
AT91C_UDPHS_FIFO_MAX_SIZE EQU (0x7 << 12) ;- (UDPHS) DPRAM size
AT91C_UDPHS_BW_DPRAM      EQU (0x1 << 15) ;- (UDPHS) DPRAM byte write capability
AT91C_UDPHS_DATAB16_8     EQU (0x1 << 16) ;- (UDPHS) UTMI DataBus16_8
AT91C_UDPHS_ISO_EPT_1     EQU (0x1 << 17) ;- (UDPHS) Endpoint 1 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_2     EQU (0x1 << 18) ;- (UDPHS) Endpoint 2 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_3     EQU (0x1 << 19) ;- (UDPHS) Endpoint 3 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_4     EQU (0x1 << 20) ;- (UDPHS) Endpoint 4 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_5     EQU (0x1 << 21) ;- (UDPHS) Endpoint 5 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_6     EQU (0x1 << 22) ;- (UDPHS) Endpoint 6 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_7     EQU (0x1 << 23) ;- (UDPHS) Endpoint 7 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_8     EQU (0x1 << 24) ;- (UDPHS) Endpoint 8 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_9     EQU (0x1 << 25) ;- (UDPHS) Endpoint 9 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_10    EQU (0x1 << 26) ;- (UDPHS) Endpoint 10 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_11    EQU (0x1 << 27) ;- (UDPHS) Endpoint 11 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_12    EQU (0x1 << 28) ;- (UDPHS) Endpoint 12 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_13    EQU (0x1 << 29) ;- (UDPHS) Endpoint 13 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_14    EQU (0x1 << 30) ;- (UDPHS) Endpoint 14 High Bandwidth Isochronous Capability
AT91C_UDPHS_ISO_EPT_15    EQU (0x1 << 31) ;- (UDPHS) Endpoint 15 High Bandwidth Isochronous Capability
/* - -------- UDPHS_IPVERSION : (UDPHS Offset: 0xfc) UDPHS Version Register -------- */
AT91C_UDPHS_VERSION_NUM   EQU (0xFFFF <<  0) ;- (UDPHS) Give the IP version
AT91C_UDPHS_METAL_FIX_NUM EQU (0x7 << 16) ;- (UDPHS) Give the number of metal fixes

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR AC97 Controller Interface*/
/* - ******************************************************************************/
/* - -------- AC97C_MR : (AC97C Offset: 0x8) AC97C Mode Register -------- */
AT91C_AC97C_ENA           EQU (0x1 <<  0) ;- (AC97C) AC97 Controller Global Enable
AT91C_AC97C_WRST          EQU (0x1 <<  1) ;- (AC97C) Warm Reset
AT91C_AC97C_VRA           EQU (0x1 <<  2) ;- (AC97C) Variable RAte (for Data Slots)
/* - -------- AC97C_ICA : (AC97C Offset: 0x10) AC97C Input Channel Assignement Register -------- */
AT91C_AC97C_CHID3         EQU (0x7 <<  0) ;- (AC97C) Channel Id for the input slot 3
AT91C_AC97C_CHID3_NONE    EQU (0x0) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID3_CA      EQU (0x1) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID3_CB      EQU (0x2) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID3_CC      EQU (0x3) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID4         EQU (0x7 <<  3) ;- (AC97C) Channel Id for the input slot 4
AT91C_AC97C_CHID4_NONE    EQU (0x0 <<  3) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID4_CA      EQU (0x1 <<  3) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID4_CB      EQU (0x2 <<  3) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID4_CC      EQU (0x3 <<  3) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID5         EQU (0x7 <<  6) ;- (AC97C) Channel Id for the input slot 5
AT91C_AC97C_CHID5_NONE    EQU (0x0 <<  6) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID5_CA      EQU (0x1 <<  6) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID5_CB      EQU (0x2 <<  6) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID5_CC      EQU (0x3 <<  6) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID6         EQU (0x7 <<  9) ;- (AC97C) Channel Id for the input slot 6
AT91C_AC97C_CHID6_NONE    EQU (0x0 <<  9) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID6_CA      EQU (0x1 <<  9) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID6_CB      EQU (0x2 <<  9) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID6_CC      EQU (0x3 <<  9) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID7         EQU (0x7 << 12) ;- (AC97C) Channel Id for the input slot 7
AT91C_AC97C_CHID7_NONE    EQU (0x0 << 12) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID7_CA      EQU (0x1 << 12) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID7_CB      EQU (0x2 << 12) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID7_CC      EQU (0x3 << 12) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID8         EQU (0x7 << 15) ;- (AC97C) Channel Id for the input slot 8
AT91C_AC97C_CHID8_NONE    EQU (0x0 << 15) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID8_CA      EQU (0x1 << 15) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID8_CB      EQU (0x2 << 15) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID8_CC      EQU (0x3 << 15) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID9         EQU (0x7 << 18) ;- (AC97C) Channel Id for the input slot 9
AT91C_AC97C_CHID9_NONE    EQU (0x0 << 18) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID9_CA      EQU (0x1 << 18) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID9_CB      EQU (0x2 << 18) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID9_CC      EQU (0x3 << 18) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID10        EQU (0x7 << 21) ;- (AC97C) Channel Id for the input slot 10
AT91C_AC97C_CHID10_NONE   EQU (0x0 << 21) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID10_CA     EQU (0x1 << 21) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID10_CB     EQU (0x2 << 21) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID10_CC     EQU (0x3 << 21) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID11        EQU (0x7 << 24) ;- (AC97C) Channel Id for the input slot 11
AT91C_AC97C_CHID11_NONE   EQU (0x0 << 24) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID11_CA     EQU (0x1 << 24) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID11_CB     EQU (0x2 << 24) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID11_CC     EQU (0x3 << 24) ;- (AC97C) Channel C data will be transmitted during this slot
AT91C_AC97C_CHID12        EQU (0x7 << 27) ;- (AC97C) Channel Id for the input slot 12
AT91C_AC97C_CHID12_NONE   EQU (0x0 << 27) ;- (AC97C) No data will be transmitted during this slot
AT91C_AC97C_CHID12_CA     EQU (0x1 << 27) ;- (AC97C) Channel A data will be transmitted during this slot
AT91C_AC97C_CHID12_CB     EQU (0x2 << 27) ;- (AC97C) Channel B data will be transmitted during this slot
AT91C_AC97C_CHID12_CC     EQU (0x3 << 27) ;- (AC97C) Channel C data will be transmitted during this slot
/* - -------- AC97C_OCA : (AC97C Offset: 0x14) AC97C Output Channel Assignement Register -------- */
/* - -------- AC97C_CARHR : (AC97C Offset: 0x20) AC97C Channel A Receive Holding Register -------- */
AT91C_AC97C_RDATA         EQU (0xFFFFF <<  0) ;- (AC97C) Receive data
/* - -------- AC97C_CATHR : (AC97C Offset: 0x24) AC97C Channel A Transmit Holding Register -------- */
AT91C_AC97C_TDATA         EQU (0xFFFFF <<  0) ;- (AC97C) Transmit data
/* - -------- AC97C_CASR : (AC97C Offset: 0x28) AC97C Channel A Status Register -------- */
AT91C_AC97C_TXRDY         EQU (0x1 <<  0) ;- (AC97C) 
AT91C_AC97C_TXEMPTY       EQU (0x1 <<  1) ;- (AC97C) 
AT91C_AC97C_UNRUN         EQU (0x1 <<  2) ;- (AC97C) 
AT91C_AC97C_RXRDY         EQU (0x1 <<  4) ;- (AC97C) 
AT91C_AC97C_OVRUN         EQU (0x1 <<  5) ;- (AC97C) 
AT91C_AC97C_ENDTX         EQU (0x1 << 10) ;- (AC97C) 
AT91C_AC97C_TXBUFE        EQU (0x1 << 11) ;- (AC97C) 
AT91C_AC97C_ENDRX         EQU (0x1 << 14) ;- (AC97C) 
AT91C_AC97C_RXBUFF        EQU (0x1 << 15) ;- (AC97C) 
/* - -------- AC97C_CAMR : (AC97C Offset: 0x2c) AC97C Channel A Mode Register -------- */
AT91C_AC97C_SIZE          EQU (0x3 << 16) ;- (AC97C) 
AT91C_AC97C_SIZE_20_BITS  EQU (0x0 << 16) ;- (AC97C) Data size is 20 bits
AT91C_AC97C_SIZE_18_BITS  EQU (0x1 << 16) ;- (AC97C) Data size is 18 bits
AT91C_AC97C_SIZE_16_BITS  EQU (0x2 << 16) ;- (AC97C) Data size is 16 bits
AT91C_AC97C_SIZE_10_BITS  EQU (0x3 << 16) ;- (AC97C) Data size is 10 bits
AT91C_AC97C_CEM           EQU (0x1 << 18) ;- (AC97C) 
AT91C_AC97C_CEN           EQU (0x1 << 21) ;- (AC97C) 
AT91C_AC97C_PDCEN         EQU (0x1 << 22) ;- (AC97C) 
/* - -------- AC97C_CBRHR : (AC97C Offset: 0x30) AC97C Channel B Receive Holding Register -------- */
/* - -------- AC97C_CBTHR : (AC97C Offset: 0x34) AC97C Channel B Transmit Holding Register -------- */
/* - -------- AC97C_CBSR : (AC97C Offset: 0x38) AC97C Channel B Status Register -------- */
/* - -------- AC97C_CBMR : (AC97C Offset: 0x3c) AC97C Channel B Mode Register -------- */
/* - -------- AC97C_CORHR : (AC97C Offset: 0x40) AC97C Codec Channel Receive Holding Register -------- */
AT91C_AC97C_SDATA         EQU (0xFFFF <<  0) ;- (AC97C) Status Data
/* - -------- AC97C_COTHR : (AC97C Offset: 0x44) AC97C Codec Channel Transmit Holding Register -------- */
AT91C_AC97C_CDATA         EQU (0xFFFF <<  0) ;- (AC97C) Command Data
AT91C_AC97C_CADDR         EQU (0x7F << 16) ;- (AC97C) COdec control register index
AT91C_AC97C_READ          EQU (0x1 << 23) ;- (AC97C) Read/Write command
/* - -------- AC97C_COSR : (AC97C Offset: 0x48) AC97C CODEC Status Register -------- */
/* - -------- AC97C_COMR : (AC97C Offset: 0x4c) AC97C CODEC Mode Register -------- */
/* - -------- AC97C_SR : (AC97C Offset: 0x50) AC97C Status Register -------- */
AT91C_AC97C_SOF           EQU (0x1 <<  0) ;- (AC97C) 
AT91C_AC97C_WKUP          EQU (0x1 <<  1) ;- (AC97C) 
AT91C_AC97C_COEVT         EQU (0x1 <<  2) ;- (AC97C) 
AT91C_AC97C_CAEVT         EQU (0x1 <<  3) ;- (AC97C) 
AT91C_AC97C_CBEVT         EQU (0x1 <<  4) ;- (AC97C) 
/* - -------- AC97C_IER : (AC97C Offset: 0x54) AC97C Interrupt Enable Register -------- */
/* - -------- AC97C_IDR : (AC97C Offset: 0x58) AC97C Interrupt Disable Register -------- */
/* - -------- AC97C_IMR : (AC97C Offset: 0x5c) AC97C Interrupt Mask Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR LCD Controller*/
/* - ******************************************************************************/
/* - -------- LCDC_FRMP1 : (LCDC Offset: 0x8) DMA Frame Pointer 1 Register -------- */
AT91C_LCDC_FRMPT1         EQU (0x3FFFFF <<  0) ;- (LCDC) Frame Pointer Address 1
/* - -------- LCDC_FRMP2 : (LCDC Offset: 0xc) DMA Frame Pointer 2 Register -------- */
AT91C_LCDC_FRMPT2         EQU (0x1FFFFF <<  0) ;- (LCDC) Frame Pointer Address 2
/* - -------- LCDC_FRMCFG : (LCDC Offset: 0x18) DMA Frame Config Register -------- */
AT91C_LCDC_FRSIZE         EQU (0x3FFFFF <<  0) ;- (LCDC) FRAME SIZE
AT91C_LCDC_BLENGTH        EQU (0xF << 24) ;- (LCDC) BURST LENGTH
/* - -------- LCDC_DMACON : (LCDC Offset: 0x1c) DMA Control Register -------- */
AT91C_LCDC_DMAEN          EQU (0x1 <<  0) ;- (LCDC) DAM Enable
AT91C_LCDC_DMARST         EQU (0x1 <<  1) ;- (LCDC) DMA Reset (WO)
AT91C_LCDC_DMABUSY        EQU (0x1 <<  2) ;- (LCDC) DMA Reset (WO)
AT91C_LCDC_DMAUPDT        EQU (0x1 <<  3) ;- (LCDC) DMA Configuration Update
AT91C_LCDC_DMA2DEN        EQU (0x1 <<  4) ;- (LCDC) 2D Addressing Enable
/* - -------- LCDC_DMA2DCFG : (LCDC Offset: 0x20) DMA 2D addressing configuration Register -------- */
AT91C_LCDC_ADDRINC        EQU (0xFFFF <<  0) ;- (LCDC) Number of 32b words that the DMA must jump when going to the next line
AT91C_LCDC_PIXELOFF       EQU (0x1F << 24) ;- (LCDC) Offset (in bits) of the first pixel of the screen in the memory word which contain it
/* - -------- LCDC_LCDCON1 : (LCDC Offset: 0x800) LCD Control 1 Register -------- */
AT91C_LCDC_BYPASS         EQU (0x1 <<  0) ;- (LCDC) Bypass lcd_pccklk divider
AT91C_LCDC_CLKVAL         EQU (0x1FF << 12) ;- (LCDC) 9-bit Divider for pixel clock frequency
AT91C_LCDC_LINCNT         EQU (0x7FF << 21) ;- (LCDC) Line Counter (RO)
/* - -------- LCDC_LCDCON2 : (LCDC Offset: 0x804) LCD Control 2 Register -------- */
AT91C_LCDC_DISTYPE        EQU (0x3 <<  0) ;- (LCDC) Display Type
AT91C_LCDC_DISTYPE_STNMONO EQU (0x0) ;- (LCDC) STN Mono
AT91C_LCDC_DISTYPE_STNCOLOR EQU (0x1) ;- (LCDC) STN Color
AT91C_LCDC_DISTYPE_TFT    EQU (0x2) ;- (LCDC) TFT
AT91C_LCDC_SCANMOD        EQU (0x1 <<  2) ;- (LCDC) Scan Mode
AT91C_LCDC_SCANMOD_SINGLESCAN EQU (0x0 <<  2) ;- (LCDC) Single Scan
AT91C_LCDC_SCANMOD_DUALSCAN EQU (0x1 <<  2) ;- (LCDC) Dual Scan
AT91C_LCDC_IFWIDTH        EQU (0x3 <<  3) ;- (LCDC) Interface Width
AT91C_LCDC_IFWIDTH_FOURBITSWIDTH EQU (0x0 <<  3) ;- (LCDC) 4 Bits
AT91C_LCDC_IFWIDTH_EIGTHBITSWIDTH EQU (0x1 <<  3) ;- (LCDC) 8 Bits
AT91C_LCDC_IFWIDTH_SIXTEENBITSWIDTH EQU (0x2 <<  3) ;- (LCDC) 16 Bits
AT91C_LCDC_PIXELSIZE      EQU (0x7 <<  5) ;- (LCDC) Bits per pixel
AT91C_LCDC_PIXELSIZE_ONEBITSPERPIXEL EQU (0x0 <<  5) ;- (LCDC) 1 Bits
AT91C_LCDC_PIXELSIZE_TWOBITSPERPIXEL EQU (0x1 <<  5) ;- (LCDC) 2 Bits
AT91C_LCDC_PIXELSIZE_FOURBITSPERPIXEL EQU (0x2 <<  5) ;- (LCDC) 4 Bits
AT91C_LCDC_PIXELSIZE_EIGTHBITSPERPIXEL EQU (0x3 <<  5) ;- (LCDC) 8 Bits
AT91C_LCDC_PIXELSIZE_SIXTEENBITSPERPIXEL EQU (0x4 <<  5) ;- (LCDC) 16 Bits
AT91C_LCDC_PIXELSIZE_TWENTYFOURBITSPERPIXEL EQU (0x5 <<  5) ;- (LCDC) 24 Bits
AT91C_LCDC_INVVD          EQU (0x1 <<  8) ;- (LCDC) lcd datas polarity
AT91C_LCDC_INVVD_NORMALPOL EQU (0x0 <<  8) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVVD_INVERTEDPOL EQU (0x1 <<  8) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVFRAME       EQU (0x1 <<  9) ;- (LCDC) lcd vsync polarity
AT91C_LCDC_INVFRAME_NORMALPOL EQU (0x0 <<  9) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVFRAME_INVERTEDPOL EQU (0x1 <<  9) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVLINE        EQU (0x1 << 10) ;- (LCDC) lcd hsync polarity
AT91C_LCDC_INVLINE_NORMALPOL EQU (0x0 << 10) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVLINE_INVERTEDPOL EQU (0x1 << 10) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVCLK         EQU (0x1 << 11) ;- (LCDC) lcd pclk polarity
AT91C_LCDC_INVCLK_NORMALPOL EQU (0x0 << 11) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVCLK_INVERTEDPOL EQU (0x1 << 11) ;- (LCDC) Inverted Polarity
AT91C_LCDC_INVDVAL        EQU (0x1 << 12) ;- (LCDC) lcd dval polarity
AT91C_LCDC_INVDVAL_NORMALPOL EQU (0x0 << 12) ;- (LCDC) Normal Polarity
AT91C_LCDC_INVDVAL_INVERTEDPOL EQU (0x1 << 12) ;- (LCDC) Inverted Polarity
AT91C_LCDC_CLKMOD         EQU (0x1 << 15) ;- (LCDC) lcd pclk Mode
AT91C_LCDC_CLKMOD_ACTIVEONLYDISP EQU (0x0 << 15) ;- (LCDC) Active during display period
AT91C_LCDC_CLKMOD_ALWAYSACTIVE EQU (0x1 << 15) ;- (LCDC) Always Active
AT91C_LCDC_MEMOR          EQU (0x1 << 31) ;- (LCDC) lcd pclk Mode
AT91C_LCDC_MEMOR_BIGIND   EQU (0x0 << 31) ;- (LCDC) Big Endian
AT91C_LCDC_MEMOR_LITTLEIND EQU (0x1 << 31) ;- (LCDC) Little Endian
/* - -------- LCDC_TIM1 : (LCDC Offset: 0x808) LCDC Timing Config 1 Register -------- */
AT91C_LCDC_VFP            EQU (0xFF <<  0) ;- (LCDC) Vertical Front Porch
AT91C_LCDC_VBP            EQU (0xFF <<  8) ;- (LCDC) Vertical Back Porch
AT91C_LCDC_VPW            EQU (0x3F << 16) ;- (LCDC) Vertical Synchronization Pulse Width
AT91C_LCDC_VHDLY          EQU (0xF << 24) ;- (LCDC) Vertical to Horizontal Delay
/* - -------- LCDC_TIM2 : (LCDC Offset: 0x80c) LCDC Timing Config 2 Register -------- */
AT91C_LCDC_HBP            EQU (0xFF <<  0) ;- (LCDC) Horizontal Back Porch
AT91C_LCDC_HPW            EQU (0x3F <<  8) ;- (LCDC) Horizontal Synchronization Pulse Width
AT91C_LCDC_HFP            EQU (0x3FF << 22) ;- (LCDC) Horizontal Front Porch
/* - -------- LCDC_LCDFRCFG : (LCDC Offset: 0x810) LCD Frame Config Register -------- */
AT91C_LCDC_LINEVAL        EQU (0x7FF <<  0) ;- (LCDC) Vertical Size of LCD Module
AT91C_LCDC_HOZVAL         EQU (0x7FF << 21) ;- (LCDC) Horizontal Size of LCD Module
/* - -------- LCDC_FIFO : (LCDC Offset: 0x814) LCD FIFO Register -------- */
AT91C_LCDC_FIFOTH         EQU (0xFFFF <<  0) ;- (LCDC) FIFO Threshold
/* - -------- LCDC_MVAL : (LCDC Offset: 0x818) LCD Mode Toggle Rate Value Register -------- */
AT91C_LCDC_MVALUE         EQU (0xFF <<  0) ;- (LCDC) Toggle Rate Value
AT91C_LCDC_MMODE          EQU (0x1 << 31) ;- (LCDC) Toggle Rate Sel
AT91C_LCDC_MMODE_EACHFRAME EQU (0x0 << 31) ;- (LCDC) Each Frame
AT91C_LCDC_MMODE_MVALDEFINED EQU (0x1 << 31) ;- (LCDC) Defined by MVAL
/* - -------- LCDC_DP1_2 : (LCDC Offset: 0x81c) Dithering Pattern 1/2 -------- */
AT91C_LCDC_DP1_2_FIELD    EQU (0xFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP4_7 : (LCDC Offset: 0x820) Dithering Pattern 4/7 -------- */
AT91C_LCDC_DP4_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP3_5 : (LCDC Offset: 0x824) Dithering Pattern 3/5 -------- */
AT91C_LCDC_DP3_5_FIELD    EQU (0xFFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP2_3 : (LCDC Offset: 0x828) Dithering Pattern 2/3 -------- */
AT91C_LCDC_DP2_3_FIELD    EQU (0xFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP5_7 : (LCDC Offset: 0x82c) Dithering Pattern 5/7 -------- */
AT91C_LCDC_DP5_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP3_4 : (LCDC Offset: 0x830) Dithering Pattern 3/4 -------- */
AT91C_LCDC_DP3_4_FIELD    EQU (0xFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP4_5 : (LCDC Offset: 0x834) Dithering Pattern 4/5 -------- */
AT91C_LCDC_DP4_5_FIELD    EQU (0xFFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_DP6_7 : (LCDC Offset: 0x838) Dithering Pattern 6/7 -------- */
AT91C_LCDC_DP6_7_FIELD    EQU (0xFFFFFFF <<  0) ;- (LCDC) Ratio
/* - -------- LCDC_PWRCON : (LCDC Offset: 0x83c) LCDC Power Control Register -------- */
AT91C_LCDC_PWR            EQU (0x1 <<  0) ;- (LCDC) LCD Module Power Control
AT91C_LCDC_GUARDT         EQU (0x7F <<  1) ;- (LCDC) Delay in Frame Period
AT91C_LCDC_BUSY           EQU (0x1 << 31) ;- (LCDC) Read Only : 1 indicates that LCDC is busy
AT91C_LCDC_BUSY_LCDNOTBUSY EQU (0x0 << 31) ;- (LCDC) LCD is Not Busy
AT91C_LCDC_BUSY_LCDBUSY   EQU (0x1 << 31) ;- (LCDC) LCD is Busy
/* - -------- LCDC_CTRSTCON : (LCDC Offset: 0x840) LCDC Contrast Control Register -------- */
AT91C_LCDC_PS             EQU (0x3 <<  0) ;- (LCDC) LCD Contrast Counter Prescaler
AT91C_LCDC_PS_NOTDIVIDED  EQU (0x0) ;- (LCDC) Counter Freq is System Freq.
AT91C_LCDC_PS_DIVIDEDBYTWO EQU (0x1) ;- (LCDC) Counter Freq is System Freq divided by 2.
AT91C_LCDC_PS_DIVIDEDBYFOUR EQU (0x2) ;- (LCDC) Counter Freq is System Freq divided by 4.
AT91C_LCDC_PS_DIVIDEDBYEIGHT EQU (0x3) ;- (LCDC) Counter Freq is System Freq divided by 8.
AT91C_LCDC_POL            EQU (0x1 <<  2) ;- (LCDC) Polarity of output Pulse
AT91C_LCDC_POL_NEGATIVEPULSE EQU (0x0 <<  2) ;- (LCDC) Negative Pulse
AT91C_LCDC_POL_POSITIVEPULSE EQU (0x1 <<  2) ;- (LCDC) Positive Pulse
AT91C_LCDC_ENA            EQU (0x1 <<  3) ;- (LCDC) PWM generator Control
AT91C_LCDC_ENA_PWMGEMDISABLED EQU (0x0 <<  3) ;- (LCDC) PWM Generator Disabled
AT91C_LCDC_ENA_PWMGEMENABLED EQU (0x1 <<  3) ;- (LCDC) PWM Generator Disabled
/* - -------- LCDC_CTRSTVAL : (LCDC Offset: 0x844) Contrast Value Register -------- */
AT91C_LCDC_CVAL           EQU (0xFF <<  0) ;- (LCDC) PWM Compare Value
/* - -------- LCDC_IER : (LCDC Offset: 0x848) LCDC Interrupt Enable Register -------- */
AT91C_LCDC_LNI            EQU (0x1 <<  0) ;- (LCDC) Line Interrupt
AT91C_LCDC_LSTLNI         EQU (0x1 <<  1) ;- (LCDC) Last Line Interrupt
AT91C_LCDC_EOFI           EQU (0x1 <<  2) ;- (LCDC) End Of Frame Interrupt
AT91C_LCDC_UFLWI          EQU (0x1 <<  4) ;- (LCDC) FIFO Underflow Interrupt
AT91C_LCDC_OWRI           EQU (0x1 <<  5) ;- (LCDC) Over Write Interrupt
AT91C_LCDC_MERI           EQU (0x1 <<  6) ;- (LCDC) Memory Error  Interrupt
/* - -------- LCDC_IDR : (LCDC Offset: 0x84c) LCDC Interrupt Disable Register -------- */
/* - -------- LCDC_IMR : (LCDC Offset: 0x850) LCDC Interrupt Mask Register -------- */
/* - -------- LCDC_ISR : (LCDC Offset: 0x854) LCDC Interrupt Status Register -------- */
/* - -------- LCDC_ICR : (LCDC Offset: 0x858) LCDC Interrupt Clear Register -------- */
/* - -------- LCDC_GPR : (LCDC Offset: 0x85c) LCDC General Purpose Register -------- */
AT91C_LCDC_GPRBUS         EQU (0xFF <<  0) ;- (LCDC) 8 bits available
/* - -------- LCDC_ITR : (LCDC Offset: 0x860) Interrupts Test Register -------- */
/* - -------- LCDC_IRR : (LCDC Offset: 0x864) Interrupts Raw Status Register -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR HDMA Channel structure*/
/* - ******************************************************************************/
/* - -------- HDMA_SADDR : (HDMA_CH Offset: 0x0)  -------- */
AT91C_SADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
/* - -------- HDMA_DADDR : (HDMA_CH Offset: 0x4)  -------- */
AT91C_DADDR               EQU (0x0 <<  0) ;- (HDMA_CH) 
/* - -------- HDMA_DSCR : (HDMA_CH Offset: 0x8)  -------- */
AT91C_DSCR_IF             EQU (0x3 <<  0) ;- (HDMA_CH) 
AT91C_DSCR                EQU (0x3FFFFFFF <<  2) ;- (HDMA_CH) 
/* - -------- HDMA_CTRLA : (HDMA_CH Offset: 0xc)  -------- */
AT91C_BTSIZE              EQU (0xFFF <<  0) ;- (HDMA_CH) 
AT91C_FC                  EQU (0x7 << 12) ;- (HDMA_CH) 
AT91C_AUTO                EQU (0x1 << 15) ;- (HDMA_CH) 
AT91C_SCSIZE              EQU (0x7 << 16) ;- (HDMA_CH) 
AT91C_DCSIZE              EQU (0x7 << 20) ;- (HDMA_CH) 
AT91C_SRC_WIDTH           EQU (0x3 << 24) ;- (HDMA_CH) 
AT91C_DST_WIDTH           EQU (0x3 << 28) ;- (HDMA_CH) 
/* - -------- HDMA_CTRLB : (HDMA_CH Offset: 0x10)  -------- */
AT91C_SIF                 EQU (0x3 <<  0) ;- (HDMA_CH) 
AT91C_DIF                 EQU (0x3 <<  4) ;- (HDMA_CH) 
AT91C_SRC_PIP             EQU (0x1 <<  8) ;- (HDMA_CH) 
AT91C_DST_PIP             EQU (0x1 << 12) ;- (HDMA_CH) 
AT91C_SRC_DSCR            EQU (0x1 << 16) ;- (HDMA_CH) 
AT91C_DST_DSCR            EQU (0x1 << 20) ;- (HDMA_CH) 
AT91C_SRC_INCR            EQU (0x3 << 24) ;- (HDMA_CH) 
AT91C_DST_INCR            EQU (0x3 << 28) ;- (HDMA_CH) 
/* - -------- HDMA_CFG : (HDMA_CH Offset: 0x14)  -------- */
AT91C_SRC_PER             EQU (0xF <<  0) ;- (HDMA_CH) 
AT91C_DST_PER             EQU (0xF <<  4) ;- (HDMA_CH) 
AT91C_SRC_REP             EQU (0x1 <<  8) ;- (HDMA_CH) 
AT91C_SRC_H2SEL           EQU (0x1 <<  9) ;- (HDMA_CH) 
AT91C_DST_REP             EQU (0x1 << 12) ;- (HDMA_CH) 
AT91C_DST_H2SEL           EQU (0x1 << 13) ;- (HDMA_CH) 
AT91C_LOCK_IF             EQU (0x1 << 20) ;- (HDMA_CH) 
AT91C_LOCK_B              EQU (0x1 << 21) ;- (HDMA_CH) 
AT91C_LOCK_IF_L           EQU (0x1 << 22) ;- (HDMA_CH) 
AT91C_AHB_PROT            EQU (0x7 << 24) ;- (HDMA_CH) 
/* - -------- HDMA_SPIP : (HDMA_CH Offset: 0x18)  -------- */
AT91C_SPIP_HOLE           EQU (0xFFFF <<  0) ;- (HDMA_CH) 
AT91C_SPIP_BOUNDARY       EQU (0x3FF << 16) ;- (HDMA_CH) 
/* - -------- HDMA_DPIP : (HDMA_CH Offset: 0x1c)  -------- */
AT91C_DPIP_HOLE           EQU (0xFFFF <<  0) ;- (HDMA_CH) 
AT91C_DPIP_BOUNDARY       EQU (0x3FF << 16) ;- (HDMA_CH) 
/* - -------- HDMA_BDSCR : (HDMA_CH Offset: 0x20)  -------- */
/* - -------- HDMA_CADDR : (HDMA_CH Offset: 0x24)  -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR HDMA controller*/
/* - ******************************************************************************/
/* - -------- HDMA_GCFG : (HDMA Offset: 0x0)  -------- */
AT91C_IF0_BIGEND          EQU (0x1 <<  0) ;- (HDMA) 
AT91C_IF1_BIGEND          EQU (0x1 <<  1) ;- (HDMA) 
AT91C_IF2_BIGEND          EQU (0x1 <<  2) ;- (HDMA) 
AT91C_IF3_BIGEND          EQU (0x1 <<  3) ;- (HDMA) 
AT91C_ARB_CFG             EQU (0x1 <<  4) ;- (HDMA) 
/* - -------- HDMA_EN : (HDMA Offset: 0x4)  -------- */
AT91C_HDMA_ENABLE         EQU (0x1 <<  0) ;- (HDMA) 
/* - -------- HDMA_SREQ : (HDMA Offset: 0x8)  -------- */
AT91C_SOFT_SREQ           EQU (0xFFFF <<  0) ;- (HDMA) 
/* - -------- HDMA_BREQ : (HDMA Offset: 0xc)  -------- */
AT91C_SOFT_BREQ           EQU (0xFFFF <<  0) ;- (HDMA) 
/* - -------- HDMA_LAST : (HDMA Offset: 0x10)  -------- */
AT91C_SOFT_LAST           EQU (0xFFFF <<  0) ;- (HDMA) 
/* - -------- HDMA_SYNC : (HDMA Offset: 0x14)  -------- */
AT91C_SYNC_REQ            EQU (0xFFFF <<  0) ;- (HDMA) 
/* - -------- HDMA_EBCIER : (HDMA Offset: 0x18)  -------- */
AT91C_BTC                 EQU (0xFF <<  0) ;- (HDMA) 
AT91C_CBTC                EQU (0xFF <<  8) ;- (HDMA) 
AT91C_ERR                 EQU (0xFF << 16) ;- (HDMA) 
/* - -------- HDMA_EBCIDR : (HDMA Offset: 0x1c)  -------- */
/* - -------- HDMA_EBCIMR : (HDMA Offset: 0x20)  -------- */
/* - -------- HDMA_EBCISR : (HDMA Offset: 0x24)  -------- */
/* - -------- HDMA_CHER : (HDMA Offset: 0x28)  -------- */
AT91C_ENABLE              EQU (0xFF <<  0) ;- (HDMA) 
AT91C_SUSPEND             EQU (0xFF <<  8) ;- (HDMA) 
AT91C_KEEPON              EQU (0xFF << 24) ;- (HDMA) 
/* - -------- HDMA_CHDR : (HDMA Offset: 0x2c)  -------- */
AT91C_RESUME              EQU (0xFF <<  8) ;- (HDMA) 
/* - -------- HDMA_CHSR : (HDMA Offset: 0x30)  -------- */
AT91C_STALLED             EQU (0xFF << 14) ;- (HDMA) 
AT91C_EMPTY               EQU (0xFF << 16) ;- (HDMA) 
/* - -------- HDMA_RSVD : (HDMA Offset: 0x34)  -------- */
/* - -------- HDMA_RSVD : (HDMA Offset: 0x38)  -------- */

/* - ******************************************************************************/
/* -              SOFTWARE API DEFINITION  FOR Error Correction Code controller*/
/* - ******************************************************************************/
/* - -------- ECC_CR : (ECC Offset: 0x0) ECC reset register -------- */
AT91C_ECC_RST             EQU (0x1 <<  0) ;- (ECC) ECC reset parity
/* - -------- ECC_MR : (ECC Offset: 0x4) ECC page size register -------- */
AT91C_ECC_PAGE_SIZE       EQU (0x3 <<  0) ;- (ECC) Nand Flash page size
/* - -------- ECC_SR : (ECC Offset: 0x8) ECC status register -------- */
AT91C_ECC_RECERR          EQU (0x1 <<  0) ;- (ECC) ECC error
AT91C_ECC_ECCERR          EQU (0x1 <<  1) ;- (ECC) ECC single error
AT91C_ECC_MULERR          EQU (0x1 <<  2) ;- (ECC) ECC_MULERR
/* - -------- ECC_PR : (ECC Offset: 0xc) ECC parity register -------- */
AT91C_ECC_BITADDR         EQU (0xF <<  0) ;- (ECC) Bit address error
AT91C_ECC_WORDADDR        EQU (0xFFF <<  4) ;- (ECC) address of the failing bit
/* - -------- ECC_NPR : (ECC Offset: 0x10) ECC N parity register -------- */
AT91C_ECC_NPARITY         EQU (0xFFFF <<  0) ;- (ECC) ECC parity N 
/* - -------- ECC_VR : (ECC Offset: 0xfc) ECC version register -------- */
AT91C_ECC_VR              EQU (0xF <<  0) ;- (ECC) ECC version register

/* - ******************************************************************************/
/* -               REGISTER ADDRESS DEFINITION FOR AT91SAM9RL64*/
/* - ******************************************************************************/
/* - ========== Register definition for SYS peripheral ========== */
AT91C_SYS_SLCKSEL         EQU (0xFFFFFD50) ;- (SYS) Slow Clock Selection Register
AT91C_SYS_GPBR            EQU (0xFFFFFD60) ;- (SYS) General Purpose Register
/* - ========== Register definition for EBI peripheral ========== */
AT91C_EBI_DUMMY           EQU (0xFFFFE800) ;- (EBI) Dummy register - Do not use
/* - ========== Register definition for SDRAMC peripheral ========== */
AT91C_SDRAMC_MR           EQU (0xFFFFEA00) ;- (SDRAMC) SDRAM Controller Mode Register
AT91C_SDRAMC_IMR          EQU (0xFFFFEA1C) ;- (SDRAMC) SDRAM Controller Interrupt Mask Register
AT91C_SDRAMC_LPR          EQU (0xFFFFEA10) ;- (SDRAMC) SDRAM Controller Low Power Register
AT91C_SDRAMC_ISR          EQU (0xFFFFEA20) ;- (SDRAMC) SDRAM Controller Interrupt Mask Register
AT91C_SDRAMC_IDR          EQU (0xFFFFEA18) ;- (SDRAMC) SDRAM Controller Interrupt Disable Register
AT91C_SDRAMC_CR           EQU (0xFFFFEA08) ;- (SDRAMC) SDRAM Controller Configuration Register
AT91C_SDRAMC_TR           EQU (0xFFFFEA04) ;- (SDRAMC) SDRAM Controller Refresh Timer Register
AT91C_SDRAMC_MDR          EQU (0xFFFFEA24) ;- (SDRAMC) SDRAM Memory Device Register
AT91C_SDRAMC_HSR          EQU (0xFFFFEA0C) ;- (SDRAMC) SDRAM Controller High Speed Register
AT91C_SDRAMC_IER          EQU (0xFFFFEA14) ;- (SDRAMC) SDRAM Controller Interrupt Enable Register
/* - ========== Register definition for SMC peripheral ========== */
AT91C_SMC_CTRL1           EQU (0xFFFFEC1C) ;- (SMC)  Control Register for CS 1
AT91C_SMC_PULSE7          EQU (0xFFFFEC74) ;- (SMC)  Pulse Register for CS 7
AT91C_SMC_PULSE6          EQU (0xFFFFEC64) ;- (SMC)  Pulse Register for CS 6
AT91C_SMC_SETUP4          EQU (0xFFFFEC40) ;- (SMC)  Setup Register for CS 4
AT91C_SMC_PULSE3          EQU (0xFFFFEC34) ;- (SMC)  Pulse Register for CS 3
AT91C_SMC_CYCLE5          EQU (0xFFFFEC58) ;- (SMC)  Cycle Register for CS 5
AT91C_SMC_CYCLE2          EQU (0xFFFFEC28) ;- (SMC)  Cycle Register for CS 2
AT91C_SMC_CTRL2           EQU (0xFFFFEC2C) ;- (SMC)  Control Register for CS 2
AT91C_SMC_CTRL0           EQU (0xFFFFEC0C) ;- (SMC)  Control Register for CS 0
AT91C_SMC_PULSE5          EQU (0xFFFFEC54) ;- (SMC)  Pulse Register for CS 5
AT91C_SMC_PULSE1          EQU (0xFFFFEC14) ;- (SMC)  Pulse Register for CS 1
AT91C_SMC_PULSE0          EQU (0xFFFFEC04) ;- (SMC)  Pulse Register for CS 0
AT91C_SMC_CYCLE7          EQU (0xFFFFEC78) ;- (SMC)  Cycle Register for CS 7
AT91C_SMC_CTRL4           EQU (0xFFFFEC4C) ;- (SMC)  Control Register for CS 4
AT91C_SMC_CTRL3           EQU (0xFFFFEC3C) ;- (SMC)  Control Register for CS 3
AT91C_SMC_SETUP7          EQU (0xFFFFEC70) ;- (SMC)  Setup Register for CS 7
AT91C_SMC_CTRL7           EQU (0xFFFFEC7C) ;- (SMC)  Control Register for CS 7
AT91C_SMC_SETUP1          EQU (0xFFFFEC10) ;- (SMC)  Setup Register for CS 1
AT91C_SMC_CYCLE0          EQU (0xFFFFEC08) ;- (SMC)  Cycle Register for CS 0
AT91C_SMC_CTRL5           EQU (0xFFFFEC5C) ;- (SMC)  Control Register for CS 5
AT91C_SMC_CYCLE1          EQU (0xFFFFEC18) ;- (SMC)  Cycle Register for CS 1
AT91C_SMC_CTRL6           EQU (0xFFFFEC6C) ;- (SMC)  Control Register for CS 6
AT91C_SMC_SETUP0          EQU (0xFFFFEC00) ;- (SMC)  Setup Register for CS 0
AT91C_SMC_PULSE4          EQU (0xFFFFEC44) ;- (SMC)  Pulse Register for CS 4
AT91C_SMC_SETUP5          EQU (0xFFFFEC50) ;- (SMC)  Setup Register for CS 5
AT91C_SMC_SETUP2          EQU (0xFFFFEC20) ;- (SMC)  Setup Register for CS 2
AT91C_SMC_CYCLE3          EQU (0xFFFFEC38) ;- (SMC)  Cycle Register for CS 3
AT91C_SMC_CYCLE6          EQU (0xFFFFEC68) ;- (SMC)  Cycle Register for CS 6
AT91C_SMC_SETUP6          EQU (0xFFFFEC60) ;- (SMC)  Setup Register for CS 6
AT91C_SMC_CYCLE4          EQU (0xFFFFEC48) ;- (SMC)  Cycle Register for CS 4
AT91C_SMC_PULSE2          EQU (0xFFFFEC24) ;- (SMC)  Pulse Register for CS 2
AT91C_SMC_SETUP3          EQU (0xFFFFEC30) ;- (SMC)  Setup Register for CS 3
/* - ========== Register definition for MATRIX peripheral ========== */
AT91C_MATRIX_PRBS4        EQU (0xFFFFEEA4) ;- (MATRIX)  PRBS4 : ebi
AT91C_MATRIX_SCFG3        EQU (0xFFFFEE4C) ;- (MATRIX)  Slave Configuration Register 3 : usb_dev_hs
AT91C_MATRIX_MCFG6        EQU (0xFFFFEE18) ;- (MATRIX)  Master Configuration Register 6 
AT91C_MATRIX_PRAS3        EQU (0xFFFFEE98) ;- (MATRIX)  PRAS3 : usb_dev_hs
AT91C_MATRIX_PRBS7        EQU (0xFFFFEEBC) ;- (MATRIX)  PRBS7
AT91C_MATRIX_PRAS5        EQU (0xFFFFEEA8) ;- (MATRIX)  PRAS5 : bridge
AT91C_MATRIX_PRAS0        EQU (0xFFFFEE80) ;- (MATRIX)  PRAS0 : rom
AT91C_MATRIX_PRAS2        EQU (0xFFFFEE90) ;- (MATRIX)  PRAS2 : lcdc
AT91C_MATRIX_MCFG5        EQU (0xFFFFEE14) ;- (MATRIX)  Master Configuration Register 5 : bridge
AT91C_MATRIX_MCFG1        EQU (0xFFFFEE04) ;- (MATRIX)  Master Configuration Register 1 ; htcm
AT91C_MATRIX_MRCR         EQU (0xFFFFEF00) ;- (MATRIX)  Master Remp Control Register 
AT91C_MATRIX_PRBS2        EQU (0xFFFFEE94) ;- (MATRIX)  PRBS2 : lcdc
AT91C_MATRIX_SCFG4        EQU (0xFFFFEE50) ;- (MATRIX)  Slave Configuration Register 4 ; ebi
AT91C_MATRIX_SCFG6        EQU (0xFFFFEE58) ;- (MATRIX)  Slave Configuration Register 6
AT91C_MATRIX_MCFG0        EQU (0xFFFFEE00) ;- (MATRIX)  Master Configuration Register 0 : rom
AT91C_MATRIX_MCFG8        EQU (0xFFFFEE20) ;- (MATRIX)  Master Configuration Register 8 
AT91C_MATRIX_MCFG7        EQU (0xFFFFEE1C) ;- (MATRIX)  Master Configuration Register 7 
AT91C_MATRIX_MCFG4        EQU (0xFFFFEE10) ;- (MATRIX)  Master Configuration Register 4 : ebi
AT91C_MATRIX_SCFG1        EQU (0xFFFFEE44) ;- (MATRIX)  Slave Configuration Register 1 : htcm
AT91C_MATRIX_SCFG7        EQU (0xFFFFEE5C) ;- (MATRIX)  Slave Configuration Register 7
AT91C_MATRIX_MCFG2        EQU (0xFFFFEE08) ;- (MATRIX)  Master Configuration Register 2 : lcdc
AT91C_MATRIX_PRBS0        EQU (0xFFFFEE84) ;- (MATRIX)  PRBS0 : rom
AT91C_MATRIX_PRAS7        EQU (0xFFFFEEB8) ;- (MATRIX)  PRAS7
AT91C_MATRIX_SCFG0        EQU (0xFFFFEE40) ;- (MATRIX)  Slave Configuration Register 0 : rom
AT91C_MATRIX_PRBS5        EQU (0xFFFFEEAC) ;- (MATRIX)  PRBS5 : bridge
AT91C_MATRIX_PRBS3        EQU (0xFFFFEE9C) ;- (MATRIX)  PRBS3 : usb_dev_hs
AT91C_MATRIX_PRAS6        EQU (0xFFFFEEB0) ;- (MATRIX)  PRAS6
AT91C_MATRIX_MCFG3        EQU (0xFFFFEE0C) ;- (MATRIX)  Master Configuration Register 3 : usb_dev_hs
AT91C_MATRIX_PRAS1        EQU (0xFFFFEE88) ;- (MATRIX)  PRAS1 : htcm
AT91C_MATRIX_SCFG2        EQU (0xFFFFEE48) ;- (MATRIX)  Slave Configuration Register 2 : lcdc
AT91C_MATRIX_PRBS6        EQU (0xFFFFEEB4) ;- (MATRIX)  PRBS6
AT91C_MATRIX_PRAS4        EQU (0xFFFFEEA0) ;- (MATRIX)  PRAS4 : ebi
AT91C_MATRIX_SCFG5        EQU (0xFFFFEE54) ;- (MATRIX)  Slave Configuration Register 5 : bridge
AT91C_MATRIX_PRBS1        EQU (0xFFFFEE8C) ;- (MATRIX)  PRBS1 : htcm
/* - ========== Register definition for CCFG peripheral ========== */
AT91C_CCFG_MATRIXVERSION  EQU (0xFFFFEFFC) ;- (CCFG)  Version Register
AT91C_CCFG_TCMR           EQU (0xFFFFEF14) ;- (CCFG)  TCM configuration
AT91C_CCFG_EBICSA         EQU (0xFFFFEF20) ;- (CCFG)  EBI Chip Select Assignement Register
AT91C_CCFG_UDPHS          EQU (0xFFFFEF1C) ;- (CCFG)  USB Device HS configuration
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
/* - ========== Register definition for PIOD peripheral ========== */
AT91C_PIOD_ODSR           EQU (0xFFFFFA38) ;- (PIOD) Output Data Status Register
AT91C_PIOD_ABSR           EQU (0xFFFFFA78) ;- (PIOD) AB Select Status Register
AT91C_PIOD_PSR            EQU (0xFFFFFA08) ;- (PIOD) PIO Status Register
AT91C_PIOD_PPUDR          EQU (0xFFFFFA60) ;- (PIOD) Pull-up Disable Register
AT91C_PIOD_OER            EQU (0xFFFFFA10) ;- (PIOD) Output Enable Register
AT91C_PIOD_OWDR           EQU (0xFFFFFAA4) ;- (PIOD) Output Write Disable Register
AT91C_PIOD_PER            EQU (0xFFFFFA00) ;- (PIOD) PIO Enable Register
AT91C_PIOD_IFSR           EQU (0xFFFFFA28) ;- (PIOD) Input Filter Status Register
AT91C_PIOD_IFER           EQU (0xFFFFFA20) ;- (PIOD) Input Filter Enable Register
AT91C_PIOD_ODR            EQU (0xFFFFFA14) ;- (PIOD) Output Disable Registerr
AT91C_PIOD_PPUSR          EQU (0xFFFFFA68) ;- (PIOD) Pull-up Status Register
AT91C_PIOD_IFDR           EQU (0xFFFFFA24) ;- (PIOD) Input Filter Disable Register
AT91C_PIOD_PDSR           EQU (0xFFFFFA3C) ;- (PIOD) Pin Data Status Register
AT91C_PIOD_PPUER          EQU (0xFFFFFA64) ;- (PIOD) Pull-up Enable Register
AT91C_PIOD_IDR            EQU (0xFFFFFA44) ;- (PIOD) Interrupt Disable Register
AT91C_PIOD_MDDR           EQU (0xFFFFFA54) ;- (PIOD) Multi-driver Disable Register
AT91C_PIOD_ISR            EQU (0xFFFFFA4C) ;- (PIOD) Interrupt Status Register
AT91C_PIOD_OSR            EQU (0xFFFFFA18) ;- (PIOD) Output Status Register
AT91C_PIOD_CODR           EQU (0xFFFFFA34) ;- (PIOD) Clear Output Data Register
AT91C_PIOD_MDSR           EQU (0xFFFFFA58) ;- (PIOD) Multi-driver Status Register
AT91C_PIOD_PDR            EQU (0xFFFFFA04) ;- (PIOD) PIO Disable Register
AT91C_PIOD_IER            EQU (0xFFFFFA40) ;- (PIOD) Interrupt Enable Register
AT91C_PIOD_OWSR           EQU (0xFFFFFAA8) ;- (PIOD) Output Write Status Register
AT91C_PIOD_BSR            EQU (0xFFFFFA74) ;- (PIOD) Select B Register
AT91C_PIOD_ASR            EQU (0xFFFFFA70) ;- (PIOD) Select A Register
AT91C_PIOD_SODR           EQU (0xFFFFFA30) ;- (PIOD) Set Output Data Register
AT91C_PIOD_IMR            EQU (0xFFFFFA48) ;- (PIOD) Interrupt Mask Register
AT91C_PIOD_OWER           EQU (0xFFFFFAA0) ;- (PIOD) Output Write Enable Register
AT91C_PIOD_MDER           EQU (0xFFFFFA50) ;- (PIOD) Multi-driver Enable Register
/* - ========== Register definition for PMC peripheral ========== */
AT91C_PMC_PCER            EQU (0xFFFFFC10) ;- (PMC) Peripheral Clock Enable Register
AT91C_PMC_PCKR            EQU (0xFFFFFC40) ;- (PMC) Programmable Clock Register
AT91C_PMC_MCKR            EQU (0xFFFFFC30) ;- (PMC) Master Clock Register
AT91C_PMC_PLLAR           EQU (0xFFFFFC28) ;- (PMC) PLL A Register
AT91C_PMC_PCDR            EQU (0xFFFFFC14) ;- (PMC) Peripheral Clock Disable Register
AT91C_PMC_SCSR            EQU (0xFFFFFC08) ;- (PMC) System Clock Status Register
AT91C_PMC_MCFR            EQU (0xFFFFFC24) ;- (PMC) Main Clock  Frequency Register
AT91C_PMC_IMR             EQU (0xFFFFFC6C) ;- (PMC) Interrupt Mask Register
AT91C_PMC_IER             EQU (0xFFFFFC60) ;- (PMC) Interrupt Enable Register
AT91C_PMC_UCKR            EQU (0xFFFFFC1C) ;- (PMC) UTMI Clock Configuration Register
AT91C_PMC_MOR             EQU (0xFFFFFC20) ;- (PMC) Main Oscillator Register
AT91C_PMC_IDR             EQU (0xFFFFFC64) ;- (PMC) Interrupt Disable Register
AT91C_PMC_PLLBR           EQU (0xFFFFFC2C) ;- (PMC) PLL B Register
AT91C_PMC_SCDR            EQU (0xFFFFFC04) ;- (PMC) System Clock Disable Register
AT91C_PMC_PCSR            EQU (0xFFFFFC18) ;- (PMC) Peripheral Clock Status Register
AT91C_PMC_SCER            EQU (0xFFFFFC00) ;- (PMC) System Clock Enable Register
AT91C_PMC_SR              EQU (0xFFFFFC68) ;- (PMC) Status Register
/* - ========== Register definition for CKGR peripheral ========== */
AT91C_CKGR_MOR            EQU (0xFFFFFC20) ;- (CKGR) Main Oscillator Register
AT91C_CKGR_PLLBR          EQU (0xFFFFFC2C) ;- (CKGR) PLL B Register
AT91C_CKGR_MCFR           EQU (0xFFFFFC24) ;- (CKGR) Main Clock  Frequency Register
AT91C_CKGR_PLLAR          EQU (0xFFFFFC28) ;- (CKGR) PLL A Register
AT91C_CKGR_UCKR           EQU (0xFFFFFC1C) ;- (CKGR) UTMI Clock Configuration Register
/* - ========== Register definition for RSTC peripheral ========== */
AT91C_RSTC_RCR            EQU (0xFFFFFD00) ;- (RSTC) Reset Control Register
AT91C_RSTC_RMR            EQU (0xFFFFFD08) ;- (RSTC) Reset Mode Register
AT91C_RSTC_RSR            EQU (0xFFFFFD04) ;- (RSTC) Reset Status Register
/* - ========== Register definition for SHDWC peripheral ========== */
AT91C_SHDWC_SHSR          EQU (0xFFFFFD18) ;- (SHDWC) Shut Down Status Register
AT91C_SHDWC_SHMR          EQU (0xFFFFFD14) ;- (SHDWC) Shut Down Mode Register
AT91C_SHDWC_SHCR          EQU (0xFFFFFD10) ;- (SHDWC) Shut Down Control Register
/* - ========== Register definition for RTTC peripheral ========== */
AT91C_RTTC_RTSR           EQU (0xFFFFFD2C) ;- (RTTC) Real-time Status Register
AT91C_RTTC_RTMR           EQU (0xFFFFFD20) ;- (RTTC) Real-time Mode Register
AT91C_RTTC_RTVR           EQU (0xFFFFFD28) ;- (RTTC) Real-time Value Register
AT91C_RTTC_RTAR           EQU (0xFFFFFD24) ;- (RTTC) Real-time Alarm Register
/* - ========== Register definition for PITC peripheral ========== */
AT91C_PITC_PIVR           EQU (0xFFFFFD38) ;- (PITC) Period Interval Value Register
AT91C_PITC_PISR           EQU (0xFFFFFD34) ;- (PITC) Period Interval Status Register
AT91C_PITC_PIIR           EQU (0xFFFFFD3C) ;- (PITC) Period Interval Image Register
AT91C_PITC_PIMR           EQU (0xFFFFFD30) ;- (PITC) Period Interval Mode Register
/* - ========== Register definition for WDTC peripheral ========== */
AT91C_WDTC_WDCR           EQU (0xFFFFFD40) ;- (WDTC) Watchdog Control Register
AT91C_WDTC_WDSR           EQU (0xFFFFFD48) ;- (WDTC) Watchdog Status Register
AT91C_WDTC_WDMR           EQU (0xFFFFFD44) ;- (WDTC) Watchdog Mode Register
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
/* - ========== Register definition for TCB0 peripheral ========== */
AT91C_TCB0_BMR            EQU (0xFFFA00C4) ;- (TCB0) TC Block Mode Register
AT91C_TCB0_BCR            EQU (0xFFFA00C0) ;- (TCB0) TC Block Control Register
/* - ========== Register definition for TCB1 peripheral ========== */
AT91C_TCB1_BMR            EQU (0xFFFA0104) ;- (TCB1) TC Block Mode Register
AT91C_TCB1_BCR            EQU (0xFFFA0100) ;- (TCB1) TC Block Control Register
/* - ========== Register definition for TCB2 peripheral ========== */
AT91C_TCB2_BCR            EQU (0xFFFA0140) ;- (TCB2) TC Block Control Register
AT91C_TCB2_BMR            EQU (0xFFFA0144) ;- (TCB2) TC Block Mode Register
/* - ========== Register definition for PDC_MCI peripheral ========== */
AT91C_MCI_TPR             EQU (0xFFFA4108) ;- (PDC_MCI) Transmit Pointer Register
AT91C_MCI_PTCR            EQU (0xFFFA4120) ;- (PDC_MCI) PDC Transfer Control Register
AT91C_MCI_RNPR            EQU (0xFFFA4110) ;- (PDC_MCI) Receive Next Pointer Register
AT91C_MCI_TNCR            EQU (0xFFFA411C) ;- (PDC_MCI) Transmit Next Counter Register
AT91C_MCI_TCR             EQU (0xFFFA410C) ;- (PDC_MCI) Transmit Counter Register
AT91C_MCI_RCR             EQU (0xFFFA4104) ;- (PDC_MCI) Receive Counter Register
AT91C_MCI_RNCR            EQU (0xFFFA4114) ;- (PDC_MCI) Receive Next Counter Register
AT91C_MCI_TNPR            EQU (0xFFFA4118) ;- (PDC_MCI) Transmit Next Pointer Register
AT91C_MCI_RPR             EQU (0xFFFA4100) ;- (PDC_MCI) Receive Pointer Register
AT91C_MCI_PTSR            EQU (0xFFFA4124) ;- (PDC_MCI) PDC Transfer Status Register
/* - ========== Register definition for MCI peripheral ========== */
AT91C_MCI_IDR             EQU (0xFFFA4048) ;- (MCI) MCI Interrupt Disable Register
AT91C_MCI_MR              EQU (0xFFFA4004) ;- (MCI) MCI Mode Register
AT91C_MCI_VR              EQU (0xFFFA40FC) ;- (MCI) MCI Version Register
AT91C_MCI_IER             EQU (0xFFFA4044) ;- (MCI) MCI Interrupt Enable Register
AT91C_MCI_IMR             EQU (0xFFFA404C) ;- (MCI) MCI Interrupt Mask Register
AT91C_MCI_SR              EQU (0xFFFA4040) ;- (MCI) MCI Status Register
AT91C_MCI_DTOR            EQU (0xFFFA4008) ;- (MCI) MCI Data Timeout Register
AT91C_MCI_CR              EQU (0xFFFA4000) ;- (MCI) MCI Control Register
AT91C_MCI_CMDR            EQU (0xFFFA4014) ;- (MCI) MCI Command Register
AT91C_MCI_SDCR            EQU (0xFFFA400C) ;- (MCI) MCI SD Card Register
AT91C_MCI_BLKR            EQU (0xFFFA4018) ;- (MCI) MCI Block Register
AT91C_MCI_RDR             EQU (0xFFFA4030) ;- (MCI) MCI Receive Data Register
AT91C_MCI_ARGR            EQU (0xFFFA4010) ;- (MCI) MCI Argument Register
AT91C_MCI_TDR             EQU (0xFFFA4034) ;- (MCI) MCI Transmit Data Register
AT91C_MCI_RSPR            EQU (0xFFFA4020) ;- (MCI) MCI Response Register
/* - ========== Register definition for PDC_TWI0 peripheral ========== */
AT91C_TWI0_RNCR           EQU (0xFFFA8114) ;- (PDC_TWI0) Receive Next Counter Register
AT91C_TWI0_TCR            EQU (0xFFFA810C) ;- (PDC_TWI0) Transmit Counter Register
AT91C_TWI0_RCR            EQU (0xFFFA8104) ;- (PDC_TWI0) Receive Counter Register
AT91C_TWI0_TNPR           EQU (0xFFFA8118) ;- (PDC_TWI0) Transmit Next Pointer Register
AT91C_TWI0_RNPR           EQU (0xFFFA8110) ;- (PDC_TWI0) Receive Next Pointer Register
AT91C_TWI0_RPR            EQU (0xFFFA8100) ;- (PDC_TWI0) Receive Pointer Register
AT91C_TWI0_TNCR           EQU (0xFFFA811C) ;- (PDC_TWI0) Transmit Next Counter Register
AT91C_TWI0_TPR            EQU (0xFFFA8108) ;- (PDC_TWI0) Transmit Pointer Register
AT91C_TWI0_PTSR           EQU (0xFFFA8124) ;- (PDC_TWI0) PDC Transfer Status Register
AT91C_TWI0_PTCR           EQU (0xFFFA8120) ;- (PDC_TWI0) PDC Transfer Control Register
/* - ========== Register definition for TWI0 peripheral ========== */
AT91C_TWI0_IDR            EQU (0xFFFA8028) ;- (TWI0) Interrupt Disable Register
AT91C_TWI0_RHR            EQU (0xFFFA8030) ;- (TWI0) Receive Holding Register
AT91C_TWI0_SMR            EQU (0xFFFA8008) ;- (TWI0) Slave Mode Register
AT91C_TWI0_IER            EQU (0xFFFA8024) ;- (TWI0) Interrupt Enable Register
AT91C_TWI0_THR            EQU (0xFFFA8034) ;- (TWI0) Transmit Holding Register
AT91C_TWI0_MMR            EQU (0xFFFA8004) ;- (TWI0) Master Mode Register
AT91C_TWI0_CR             EQU (0xFFFA8000) ;- (TWI0) Control Register
AT91C_TWI0_CWGR           EQU (0xFFFA8010) ;- (TWI0) Clock Waveform Generator Register
AT91C_TWI0_IADR           EQU (0xFFFA800C) ;- (TWI0) Internal Address Register
AT91C_TWI0_IMR            EQU (0xFFFA802C) ;- (TWI0) Interrupt Mask Register
AT91C_TWI0_SR             EQU (0xFFFA8020) ;- (TWI0) Status Register
/* - ========== Register definition for TWI1 peripheral ========== */
AT91C_TWI1_THR            EQU (0xFFFAC034) ;- (TWI1) Transmit Holding Register
AT91C_TWI1_IDR            EQU (0xFFFAC028) ;- (TWI1) Interrupt Disable Register
AT91C_TWI1_SMR            EQU (0xFFFAC008) ;- (TWI1) Slave Mode Register
AT91C_TWI1_CWGR           EQU (0xFFFAC010) ;- (TWI1) Clock Waveform Generator Register
AT91C_TWI1_IADR           EQU (0xFFFAC00C) ;- (TWI1) Internal Address Register
AT91C_TWI1_RHR            EQU (0xFFFAC030) ;- (TWI1) Receive Holding Register
AT91C_TWI1_IER            EQU (0xFFFAC024) ;- (TWI1) Interrupt Enable Register
AT91C_TWI1_MMR            EQU (0xFFFAC004) ;- (TWI1) Master Mode Register
AT91C_TWI1_SR             EQU (0xFFFAC020) ;- (TWI1) Status Register
AT91C_TWI1_IMR            EQU (0xFFFAC02C) ;- (TWI1) Interrupt Mask Register
AT91C_TWI1_CR             EQU (0xFFFAC000) ;- (TWI1) Control Register
/* - ========== Register definition for PDC_US0 peripheral ========== */
AT91C_US0_TCR             EQU (0xFFFB010C) ;- (PDC_US0) Transmit Counter Register
AT91C_US0_PTCR            EQU (0xFFFB0120) ;- (PDC_US0) PDC Transfer Control Register
AT91C_US0_RNCR            EQU (0xFFFB0114) ;- (PDC_US0) Receive Next Counter Register
AT91C_US0_PTSR            EQU (0xFFFB0124) ;- (PDC_US0) PDC Transfer Status Register
AT91C_US0_TNCR            EQU (0xFFFB011C) ;- (PDC_US0) Transmit Next Counter Register
AT91C_US0_RNPR            EQU (0xFFFB0110) ;- (PDC_US0) Receive Next Pointer Register
AT91C_US0_RCR             EQU (0xFFFB0104) ;- (PDC_US0) Receive Counter Register
AT91C_US0_TPR             EQU (0xFFFB0108) ;- (PDC_US0) Transmit Pointer Register
AT91C_US0_TNPR            EQU (0xFFFB0118) ;- (PDC_US0) Transmit Next Pointer Register
AT91C_US0_RPR             EQU (0xFFFB0100) ;- (PDC_US0) Receive Pointer Register
/* - ========== Register definition for US0 peripheral ========== */
AT91C_US0_RHR             EQU (0xFFFB0018) ;- (US0) Receiver Holding Register
AT91C_US0_NER             EQU (0xFFFB0044) ;- (US0) Nb Errors Register
AT91C_US0_IER             EQU (0xFFFB0008) ;- (US0) Interrupt Enable Register
AT91C_US0_CR              EQU (0xFFFB0000) ;- (US0) Control Register
AT91C_US0_MAN             EQU (0xFFFB0050) ;- (US0) Manchester Encoder Decoder Register
AT91C_US0_THR             EQU (0xFFFB001C) ;- (US0) Transmitter Holding Register
AT91C_US0_CSR             EQU (0xFFFB0014) ;- (US0) Channel Status Register
AT91C_US0_BRGR            EQU (0xFFFB0020) ;- (US0) Baud Rate Generator Register
AT91C_US0_RTOR            EQU (0xFFFB0024) ;- (US0) Receiver Time-out Register
AT91C_US0_TTGR            EQU (0xFFFB0028) ;- (US0) Transmitter Time-guard Register
AT91C_US0_IDR             EQU (0xFFFB000C) ;- (US0) Interrupt Disable Register
AT91C_US0_MR              EQU (0xFFFB0004) ;- (US0) Mode Register
AT91C_US0_IF              EQU (0xFFFB004C) ;- (US0) IRDA_FILTER Register
AT91C_US0_FIDI            EQU (0xFFFB0040) ;- (US0) FI_DI_Ratio Register
AT91C_US0_IMR             EQU (0xFFFB0010) ;- (US0) Interrupt Mask Register
/* - ========== Register definition for PDC_US1 peripheral ========== */
AT91C_US1_PTCR            EQU (0xFFFB4120) ;- (PDC_US1) PDC Transfer Control Register
AT91C_US1_RCR             EQU (0xFFFB4104) ;- (PDC_US1) Receive Counter Register
AT91C_US1_RPR             EQU (0xFFFB4100) ;- (PDC_US1) Receive Pointer Register
AT91C_US1_PTSR            EQU (0xFFFB4124) ;- (PDC_US1) PDC Transfer Status Register
AT91C_US1_TPR             EQU (0xFFFB4108) ;- (PDC_US1) Transmit Pointer Register
AT91C_US1_TCR             EQU (0xFFFB410C) ;- (PDC_US1) Transmit Counter Register
AT91C_US1_RNPR            EQU (0xFFFB4110) ;- (PDC_US1) Receive Next Pointer Register
AT91C_US1_TNCR            EQU (0xFFFB411C) ;- (PDC_US1) Transmit Next Counter Register
AT91C_US1_RNCR            EQU (0xFFFB4114) ;- (PDC_US1) Receive Next Counter Register
AT91C_US1_TNPR            EQU (0xFFFB4118) ;- (PDC_US1) Transmit Next Pointer Register
/* - ========== Register definition for US1 peripheral ========== */
AT91C_US1_THR             EQU (0xFFFB401C) ;- (US1) Transmitter Holding Register
AT91C_US1_TTGR            EQU (0xFFFB4028) ;- (US1) Transmitter Time-guard Register
AT91C_US1_BRGR            EQU (0xFFFB4020) ;- (US1) Baud Rate Generator Register
AT91C_US1_IDR             EQU (0xFFFB400C) ;- (US1) Interrupt Disable Register
AT91C_US1_MR              EQU (0xFFFB4004) ;- (US1) Mode Register
AT91C_US1_RTOR            EQU (0xFFFB4024) ;- (US1) Receiver Time-out Register
AT91C_US1_MAN             EQU (0xFFFB4050) ;- (US1) Manchester Encoder Decoder Register
AT91C_US1_CR              EQU (0xFFFB4000) ;- (US1) Control Register
AT91C_US1_IMR             EQU (0xFFFB4010) ;- (US1) Interrupt Mask Register
AT91C_US1_FIDI            EQU (0xFFFB4040) ;- (US1) FI_DI_Ratio Register
AT91C_US1_RHR             EQU (0xFFFB4018) ;- (US1) Receiver Holding Register
AT91C_US1_IER             EQU (0xFFFB4008) ;- (US1) Interrupt Enable Register
AT91C_US1_CSR             EQU (0xFFFB4014) ;- (US1) Channel Status Register
AT91C_US1_IF              EQU (0xFFFB404C) ;- (US1) IRDA_FILTER Register
AT91C_US1_NER             EQU (0xFFFB4044) ;- (US1) Nb Errors Register
/* - ========== Register definition for PDC_US2 peripheral ========== */
AT91C_US2_TNCR            EQU (0xFFFB811C) ;- (PDC_US2) Transmit Next Counter Register
AT91C_US2_RNCR            EQU (0xFFFB8114) ;- (PDC_US2) Receive Next Counter Register
AT91C_US2_TNPR            EQU (0xFFFB8118) ;- (PDC_US2) Transmit Next Pointer Register
AT91C_US2_PTCR            EQU (0xFFFB8120) ;- (PDC_US2) PDC Transfer Control Register
AT91C_US2_TCR             EQU (0xFFFB810C) ;- (PDC_US2) Transmit Counter Register
AT91C_US2_RPR             EQU (0xFFFB8100) ;- (PDC_US2) Receive Pointer Register
AT91C_US2_TPR             EQU (0xFFFB8108) ;- (PDC_US2) Transmit Pointer Register
AT91C_US2_RCR             EQU (0xFFFB8104) ;- (PDC_US2) Receive Counter Register
AT91C_US2_PTSR            EQU (0xFFFB8124) ;- (PDC_US2) PDC Transfer Status Register
AT91C_US2_RNPR            EQU (0xFFFB8110) ;- (PDC_US2) Receive Next Pointer Register
/* - ========== Register definition for US2 peripheral ========== */
AT91C_US2_RTOR            EQU (0xFFFB8024) ;- (US2) Receiver Time-out Register
AT91C_US2_CSR             EQU (0xFFFB8014) ;- (US2) Channel Status Register
AT91C_US2_CR              EQU (0xFFFB8000) ;- (US2) Control Register
AT91C_US2_BRGR            EQU (0xFFFB8020) ;- (US2) Baud Rate Generator Register
AT91C_US2_NER             EQU (0xFFFB8044) ;- (US2) Nb Errors Register
AT91C_US2_MAN             EQU (0xFFFB8050) ;- (US2) Manchester Encoder Decoder Register
AT91C_US2_FIDI            EQU (0xFFFB8040) ;- (US2) FI_DI_Ratio Register
AT91C_US2_TTGR            EQU (0xFFFB8028) ;- (US2) Transmitter Time-guard Register
AT91C_US2_RHR             EQU (0xFFFB8018) ;- (US2) Receiver Holding Register
AT91C_US2_IDR             EQU (0xFFFB800C) ;- (US2) Interrupt Disable Register
AT91C_US2_THR             EQU (0xFFFB801C) ;- (US2) Transmitter Holding Register
AT91C_US2_MR              EQU (0xFFFB8004) ;- (US2) Mode Register
AT91C_US2_IMR             EQU (0xFFFB8010) ;- (US2) Interrupt Mask Register
AT91C_US2_IF              EQU (0xFFFB804C) ;- (US2) IRDA_FILTER Register
AT91C_US2_IER             EQU (0xFFFB8008) ;- (US2) Interrupt Enable Register
/* - ========== Register definition for PDC_US3 peripheral ========== */
AT91C_US3_TNPR            EQU (0xFFFBC118) ;- (PDC_US3) Transmit Next Pointer Register
AT91C_US3_TCR             EQU (0xFFFBC10C) ;- (PDC_US3) Transmit Counter Register
AT91C_US3_RNCR            EQU (0xFFFBC114) ;- (PDC_US3) Receive Next Counter Register
AT91C_US3_RPR             EQU (0xFFFBC100) ;- (PDC_US3) Receive Pointer Register
AT91C_US3_TPR             EQU (0xFFFBC108) ;- (PDC_US3) Transmit Pointer Register
AT91C_US3_RCR             EQU (0xFFFBC104) ;- (PDC_US3) Receive Counter Register
AT91C_US3_RNPR            EQU (0xFFFBC110) ;- (PDC_US3) Receive Next Pointer Register
AT91C_US3_PTCR            EQU (0xFFFBC120) ;- (PDC_US3) PDC Transfer Control Register
AT91C_US3_TNCR            EQU (0xFFFBC11C) ;- (PDC_US3) Transmit Next Counter Register
AT91C_US3_PTSR            EQU (0xFFFBC124) ;- (PDC_US3) PDC Transfer Status Register
/* - ========== Register definition for US3 peripheral ========== */
AT91C_US3_IF              EQU (0xFFFBC04C) ;- (US3) IRDA_FILTER Register
AT91C_US3_CSR             EQU (0xFFFBC014) ;- (US3) Channel Status Register
AT91C_US3_TTGR            EQU (0xFFFBC028) ;- (US3) Transmitter Time-guard Register
AT91C_US3_MAN             EQU (0xFFFBC050) ;- (US3) Manchester Encoder Decoder Register
AT91C_US3_CR              EQU (0xFFFBC000) ;- (US3) Control Register
AT91C_US3_THR             EQU (0xFFFBC01C) ;- (US3) Transmitter Holding Register
AT91C_US3_MR              EQU (0xFFFBC004) ;- (US3) Mode Register
AT91C_US3_NER             EQU (0xFFFBC044) ;- (US3) Nb Errors Register
AT91C_US3_IDR             EQU (0xFFFBC00C) ;- (US3) Interrupt Disable Register
AT91C_US3_BRGR            EQU (0xFFFBC020) ;- (US3) Baud Rate Generator Register
AT91C_US3_IMR             EQU (0xFFFBC010) ;- (US3) Interrupt Mask Register
AT91C_US3_FIDI            EQU (0xFFFBC040) ;- (US3) FI_DI_Ratio Register
AT91C_US3_RTOR            EQU (0xFFFBC024) ;- (US3) Receiver Time-out Register
AT91C_US3_IER             EQU (0xFFFBC008) ;- (US3) Interrupt Enable Register
AT91C_US3_RHR             EQU (0xFFFBC018) ;- (US3) Receiver Holding Register
/* - ========== Register definition for PDC_SSC0 peripheral ========== */
AT91C_SSC0_TNPR           EQU (0xFFFC0118) ;- (PDC_SSC0) Transmit Next Pointer Register
AT91C_SSC0_RNPR           EQU (0xFFFC0110) ;- (PDC_SSC0) Receive Next Pointer Register
AT91C_SSC0_TCR            EQU (0xFFFC010C) ;- (PDC_SSC0) Transmit Counter Register
AT91C_SSC0_PTCR           EQU (0xFFFC0120) ;- (PDC_SSC0) PDC Transfer Control Register
AT91C_SSC0_PTSR           EQU (0xFFFC0124) ;- (PDC_SSC0) PDC Transfer Status Register
AT91C_SSC0_TNCR           EQU (0xFFFC011C) ;- (PDC_SSC0) Transmit Next Counter Register
AT91C_SSC0_TPR            EQU (0xFFFC0108) ;- (PDC_SSC0) Transmit Pointer Register
AT91C_SSC0_RCR            EQU (0xFFFC0104) ;- (PDC_SSC0) Receive Counter Register
AT91C_SSC0_RPR            EQU (0xFFFC0100) ;- (PDC_SSC0) Receive Pointer Register
AT91C_SSC0_RNCR           EQU (0xFFFC0114) ;- (PDC_SSC0) Receive Next Counter Register
/* - ========== Register definition for SSC0 peripheral ========== */
AT91C_SSC0_IDR            EQU (0xFFFC0048) ;- (SSC0) Interrupt Disable Register
AT91C_SSC0_RHR            EQU (0xFFFC0020) ;- (SSC0) Receive Holding Register
AT91C_SSC0_IER            EQU (0xFFFC0044) ;- (SSC0) Interrupt Enable Register
AT91C_SSC0_CR             EQU (0xFFFC0000) ;- (SSC0) Control Register
AT91C_SSC0_RCMR           EQU (0xFFFC0010) ;- (SSC0) Receive Clock ModeRegister
AT91C_SSC0_SR             EQU (0xFFFC0040) ;- (SSC0) Status Register
AT91C_SSC0_TSHR           EQU (0xFFFC0034) ;- (SSC0) Transmit Sync Holding Register
AT91C_SSC0_CMR            EQU (0xFFFC0004) ;- (SSC0) Clock Mode Register
AT91C_SSC0_RSHR           EQU (0xFFFC0030) ;- (SSC0) Receive Sync Holding Register
AT91C_SSC0_THR            EQU (0xFFFC0024) ;- (SSC0) Transmit Holding Register
AT91C_SSC0_RFMR           EQU (0xFFFC0014) ;- (SSC0) Receive Frame Mode Register
AT91C_SSC0_TCMR           EQU (0xFFFC0018) ;- (SSC0) Transmit Clock Mode Register
AT91C_SSC0_TFMR           EQU (0xFFFC001C) ;- (SSC0) Transmit Frame Mode Register
AT91C_SSC0_IMR            EQU (0xFFFC004C) ;- (SSC0) Interrupt Mask Register
/* - ========== Register definition for PDC_SSC1 peripheral ========== */
AT91C_SSC1_RNCR           EQU (0xFFFC4114) ;- (PDC_SSC1) Receive Next Counter Register
AT91C_SSC1_PTCR           EQU (0xFFFC4120) ;- (PDC_SSC1) PDC Transfer Control Register
AT91C_SSC1_TCR            EQU (0xFFFC410C) ;- (PDC_SSC1) Transmit Counter Register
AT91C_SSC1_PTSR           EQU (0xFFFC4124) ;- (PDC_SSC1) PDC Transfer Status Register
AT91C_SSC1_TNPR           EQU (0xFFFC4118) ;- (PDC_SSC1) Transmit Next Pointer Register
AT91C_SSC1_RCR            EQU (0xFFFC4104) ;- (PDC_SSC1) Receive Counter Register
AT91C_SSC1_RNPR           EQU (0xFFFC4110) ;- (PDC_SSC1) Receive Next Pointer Register
AT91C_SSC1_RPR            EQU (0xFFFC4100) ;- (PDC_SSC1) Receive Pointer Register
AT91C_SSC1_TNCR           EQU (0xFFFC411C) ;- (PDC_SSC1) Transmit Next Counter Register
AT91C_SSC1_TPR            EQU (0xFFFC4108) ;- (PDC_SSC1) Transmit Pointer Register
/* - ========== Register definition for SSC1 peripheral ========== */
AT91C_SSC1_IMR            EQU (0xFFFC404C) ;- (SSC1) Interrupt Mask Register
AT91C_SSC1_IER            EQU (0xFFFC4044) ;- (SSC1) Interrupt Enable Register
AT91C_SSC1_THR            EQU (0xFFFC4024) ;- (SSC1) Transmit Holding Register
AT91C_SSC1_RFMR           EQU (0xFFFC4014) ;- (SSC1) Receive Frame Mode Register
AT91C_SSC1_TFMR           EQU (0xFFFC401C) ;- (SSC1) Transmit Frame Mode Register
AT91C_SSC1_IDR            EQU (0xFFFC4048) ;- (SSC1) Interrupt Disable Register
AT91C_SSC1_RSHR           EQU (0xFFFC4030) ;- (SSC1) Receive Sync Holding Register
AT91C_SSC1_TCMR           EQU (0xFFFC4018) ;- (SSC1) Transmit Clock Mode Register
AT91C_SSC1_RHR            EQU (0xFFFC4020) ;- (SSC1) Receive Holding Register
AT91C_SSC1_RCMR           EQU (0xFFFC4010) ;- (SSC1) Receive Clock ModeRegister
AT91C_SSC1_CR             EQU (0xFFFC4000) ;- (SSC1) Control Register
AT91C_SSC1_SR             EQU (0xFFFC4040) ;- (SSC1) Status Register
AT91C_SSC1_CMR            EQU (0xFFFC4004) ;- (SSC1) Clock Mode Register
AT91C_SSC1_TSHR           EQU (0xFFFC4034) ;- (SSC1) Transmit Sync Holding Register
/* - ========== Register definition for PWMC_CH0 peripheral ========== */
AT91C_PWMC_CH0_Reserved   EQU (0xFFFC8214) ;- (PWMC_CH0) Reserved
AT91C_PWMC_CH0_CUPDR      EQU (0xFFFC8210) ;- (PWMC_CH0) Channel Update Register
AT91C_PWMC_CH0_CCNTR      EQU (0xFFFC820C) ;- (PWMC_CH0) Channel Counter Register
AT91C_PWMC_CH0_CDTYR      EQU (0xFFFC8204) ;- (PWMC_CH0) Channel Duty Cycle Register
AT91C_PWMC_CH0_CPRDR      EQU (0xFFFC8208) ;- (PWMC_CH0) Channel Period Register
AT91C_PWMC_CH0_CMR        EQU (0xFFFC8200) ;- (PWMC_CH0) Channel Mode Register
/* - ========== Register definition for PWMC_CH1 peripheral ========== */
AT91C_PWMC_CH1_CPRDR      EQU (0xFFFC8228) ;- (PWMC_CH1) Channel Period Register
AT91C_PWMC_CH1_CMR        EQU (0xFFFC8220) ;- (PWMC_CH1) Channel Mode Register
AT91C_PWMC_CH1_Reserved   EQU (0xFFFC8234) ;- (PWMC_CH1) Reserved
AT91C_PWMC_CH1_CUPDR      EQU (0xFFFC8230) ;- (PWMC_CH1) Channel Update Register
AT91C_PWMC_CH1_CDTYR      EQU (0xFFFC8224) ;- (PWMC_CH1) Channel Duty Cycle Register
AT91C_PWMC_CH1_CCNTR      EQU (0xFFFC822C) ;- (PWMC_CH1) Channel Counter Register
/* - ========== Register definition for PWMC_CH2 peripheral ========== */
AT91C_PWMC_CH2_CCNTR      EQU (0xFFFC824C) ;- (PWMC_CH2) Channel Counter Register
AT91C_PWMC_CH2_CUPDR      EQU (0xFFFC8250) ;- (PWMC_CH2) Channel Update Register
AT91C_PWMC_CH2_Reserved   EQU (0xFFFC8254) ;- (PWMC_CH2) Reserved
AT91C_PWMC_CH2_CDTYR      EQU (0xFFFC8244) ;- (PWMC_CH2) Channel Duty Cycle Register
AT91C_PWMC_CH2_CMR        EQU (0xFFFC8240) ;- (PWMC_CH2) Channel Mode Register
AT91C_PWMC_CH2_CPRDR      EQU (0xFFFC8248) ;- (PWMC_CH2) Channel Period Register
/* - ========== Register definition for PWMC_CH3 peripheral ========== */
AT91C_PWMC_CH3_CPRDR      EQU (0xFFFC8268) ;- (PWMC_CH3) Channel Period Register
AT91C_PWMC_CH3_CCNTR      EQU (0xFFFC826C) ;- (PWMC_CH3) Channel Counter Register
AT91C_PWMC_CH3_CDTYR      EQU (0xFFFC8264) ;- (PWMC_CH3) Channel Duty Cycle Register
AT91C_PWMC_CH3_CUPDR      EQU (0xFFFC8270) ;- (PWMC_CH3) Channel Update Register
AT91C_PWMC_CH3_Reserved   EQU (0xFFFC8274) ;- (PWMC_CH3) Reserved
AT91C_PWMC_CH3_CMR        EQU (0xFFFC8260) ;- (PWMC_CH3) Channel Mode Register
/* - ========== Register definition for PWMC peripheral ========== */
AT91C_PWMC_IMR            EQU (0xFFFC8018) ;- (PWMC) PWMC Interrupt Mask Register
AT91C_PWMC_SR             EQU (0xFFFC800C) ;- (PWMC) PWMC Status Register
AT91C_PWMC_IER            EQU (0xFFFC8010) ;- (PWMC) PWMC Interrupt Enable Register
AT91C_PWMC_VR             EQU (0xFFFC80FC) ;- (PWMC) PWMC Version Register
AT91C_PWMC_MR             EQU (0xFFFC8000) ;- (PWMC) PWMC Mode Register
AT91C_PWMC_DIS            EQU (0xFFFC8008) ;- (PWMC) PWMC Disable Register
AT91C_PWMC_ENA            EQU (0xFFFC8004) ;- (PWMC) PWMC Enable Register
AT91C_PWMC_IDR            EQU (0xFFFC8014) ;- (PWMC) PWMC Interrupt Disable Register
AT91C_PWMC_ISR            EQU (0xFFFC801C) ;- (PWMC) PWMC Interrupt Status Register
/* - ========== Register definition for PDC_SPI peripheral ========== */
AT91C_SPI_PTCR            EQU (0xFFFCC120) ;- (PDC_SPI) PDC Transfer Control Register
AT91C_SPI_RNPR            EQU (0xFFFCC110) ;- (PDC_SPI) Receive Next Pointer Register
AT91C_SPI_RCR             EQU (0xFFFCC104) ;- (PDC_SPI) Receive Counter Register
AT91C_SPI_TPR             EQU (0xFFFCC108) ;- (PDC_SPI) Transmit Pointer Register
AT91C_SPI_PTSR            EQU (0xFFFCC124) ;- (PDC_SPI) PDC Transfer Status Register
AT91C_SPI_TNCR            EQU (0xFFFCC11C) ;- (PDC_SPI) Transmit Next Counter Register
AT91C_SPI_RPR             EQU (0xFFFCC100) ;- (PDC_SPI) Receive Pointer Register
AT91C_SPI_TCR             EQU (0xFFFCC10C) ;- (PDC_SPI) Transmit Counter Register
AT91C_SPI_RNCR            EQU (0xFFFCC114) ;- (PDC_SPI) Receive Next Counter Register
AT91C_SPI_TNPR            EQU (0xFFFCC118) ;- (PDC_SPI) Transmit Next Pointer Register
/* - ========== Register definition for SPI peripheral ========== */
AT91C_SPI_IER             EQU (0xFFFCC014) ;- (SPI) Interrupt Enable Register
AT91C_SPI_RDR             EQU (0xFFFCC008) ;- (SPI) Receive Data Register
AT91C_SPI_SR              EQU (0xFFFCC010) ;- (SPI) Status Register
AT91C_SPI_IMR             EQU (0xFFFCC01C) ;- (SPI) Interrupt Mask Register
AT91C_SPI_TDR             EQU (0xFFFCC00C) ;- (SPI) Transmit Data Register
AT91C_SPI_IDR             EQU (0xFFFCC018) ;- (SPI) Interrupt Disable Register
AT91C_SPI_CSR             EQU (0xFFFCC030) ;- (SPI) Chip Select Register
AT91C_SPI_CR              EQU (0xFFFCC000) ;- (SPI) Control Register
AT91C_SPI_MR              EQU (0xFFFCC004) ;- (SPI) Mode Register
/* - ========== Register definition for PDC_TSADC peripheral ========== */
AT91C_TSADC_RNPR          EQU (0xFFFD0110) ;- (PDC_TSADC) Receive Next Pointer Register
AT91C_TSADC_RNCR          EQU (0xFFFD0114) ;- (PDC_TSADC) Receive Next Counter Register
AT91C_TSADC_PTSR          EQU (0xFFFD0124) ;- (PDC_TSADC) PDC Transfer Status Register
AT91C_TSADC_PTCR          EQU (0xFFFD0120) ;- (PDC_TSADC) PDC Transfer Control Register
AT91C_TSADC_TCR           EQU (0xFFFD010C) ;- (PDC_TSADC) Transmit Counter Register
AT91C_TSADC_TNPR          EQU (0xFFFD0118) ;- (PDC_TSADC) Transmit Next Pointer Register
AT91C_TSADC_RCR           EQU (0xFFFD0104) ;- (PDC_TSADC) Receive Counter Register
AT91C_TSADC_TPR           EQU (0xFFFD0108) ;- (PDC_TSADC) Transmit Pointer Register
AT91C_TSADC_TNCR          EQU (0xFFFD011C) ;- (PDC_TSADC) Transmit Next Counter Register
AT91C_TSADC_RPR           EQU (0xFFFD0100) ;- (PDC_TSADC) Receive Pointer Register
/* - ========== Register definition for TSADC peripheral ========== */
AT91C_TSADC_IER           EQU (0xFFFD0024) ;- (TSADC) Interrupt Enable Register
AT91C_TSADC_MR            EQU (0xFFFD0004) ;- (TSADC) Mode Register
AT91C_TSADC_CDR4          EQU (0xFFFD0040) ;- (TSADC) Channel Data Register 4
AT91C_TSADC_CDR2          EQU (0xFFFD0038) ;- (TSADC) Channel Data Register 2
AT91C_TSADC_TRGR          EQU (0xFFFD0008) ;- (TSADC) Trigger Register
AT91C_TSADC_IDR           EQU (0xFFFD0028) ;- (TSADC) Interrupt Disable Register
AT91C_TSADC_CHER          EQU (0xFFFD0010) ;- (TSADC) Channel Enable Register
AT91C_TSADC_CDR5          EQU (0xFFFD0044) ;- (TSADC) Channel Data Register 5
AT91C_TSADC_CDR3          EQU (0xFFFD003C) ;- (TSADC) Channel Data Register 3
AT91C_TSADC_TSR           EQU (0xFFFD000C) ;- (TSADC) Touch Screen Register
AT91C_TSADC_IMR           EQU (0xFFFD002C) ;- (TSADC) Interrupt Mask Register
AT91C_TSADC_CR            EQU (0xFFFD0000) ;- (TSADC) Control Register
AT91C_TSADC_SR            EQU (0xFFFD001C) ;- (TSADC) Status Register
AT91C_TSADC_LCDR          EQU (0xFFFD0020) ;- (TSADC) Last Converted Register
AT91C_TSADC_CHSR          EQU (0xFFFD0018) ;- (TSADC) Channel Status Register
AT91C_TSADC_CDR0          EQU (0xFFFD0030) ;- (TSADC) Channel Data Register 0
AT91C_TSADC_CHDR          EQU (0xFFFD0014) ;- (TSADC) Channel Disable Register
AT91C_TSADC_CDR1          EQU (0xFFFD0034) ;- (TSADC) Channel Data Register 1
/* - ========== Register definition for UDPHS_EPTFIFO peripheral ========== */
AT91C_UDPHS_EPTFIFO_READEPTF EQU (0x006F0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 15
AT91C_UDPHS_EPTFIFO_READEPT5 EQU (0x00650000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 5
AT91C_UDPHS_EPTFIFO_READEPT1 EQU (0x00610000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 1
AT91C_UDPHS_EPTFIFO_READEPTE EQU (0x006E0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 14
AT91C_UDPHS_EPTFIFO_READEPT4 EQU (0x00640000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 4
AT91C_UDPHS_EPTFIFO_READEPTD EQU (0x006D0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 13
AT91C_UDPHS_EPTFIFO_READEPT2 EQU (0x00620000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 2
AT91C_UDPHS_EPTFIFO_READEPT6 EQU (0x00660000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 6
AT91C_UDPHS_EPTFIFO_READEPT9 EQU (0x00690000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 9
AT91C_UDPHS_EPTFIFO_READEPT0 EQU (0x00600000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 0
AT91C_UDPHS_EPTFIFO_READEPTA EQU (0x006A0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 10
AT91C_UDPHS_EPTFIFO_READEPT3 EQU (0x00630000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 3
AT91C_UDPHS_EPTFIFO_READEPTC EQU (0x006C0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 12
AT91C_UDPHS_EPTFIFO_READEPTB EQU (0x006B0000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 11
AT91C_UDPHS_EPTFIFO_READEPT8 EQU (0x00680000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 8
AT91C_UDPHS_EPTFIFO_READEPT7 EQU (0x00670000) ;- (UDPHS_EPTFIFO) FIFO Endpoint Data Register 7
/* - ========== Register definition for UDPHS_EPT_0 peripheral ========== */
AT91C_UDPHS_EPT_0_EPTSTA  EQU (0xFFFD411C) ;- (UDPHS_EPT_0) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_0_EPTCTLENB EQU (0xFFFD4104) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_0_EPTCFG  EQU (0xFFFD4100) ;- (UDPHS_EPT_0) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_0_EPTSETSTA EQU (0xFFFD4114) ;- (UDPHS_EPT_0) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_0_EPTCTLDIS EQU (0xFFFD4108) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_0_EPTCLRSTA EQU (0xFFFD4118) ;- (UDPHS_EPT_0) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_0_EPTCTL  EQU (0xFFFD410C) ;- (UDPHS_EPT_0) UDPHS Endpoint Control Register
/* - ========== Register definition for UDPHS_EPT_1 peripheral ========== */
AT91C_UDPHS_EPT_1_EPTCTL  EQU (0xFFFD412C) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_1_EPTSTA  EQU (0xFFFD413C) ;- (UDPHS_EPT_1) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_1_EPTCTLDIS EQU (0xFFFD4128) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_1_EPTCFG  EQU (0xFFFD4120) ;- (UDPHS_EPT_1) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_1_EPTSETSTA EQU (0xFFFD4134) ;- (UDPHS_EPT_1) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_1_EPTCLRSTA EQU (0xFFFD4138) ;- (UDPHS_EPT_1) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_1_EPTCTLENB EQU (0xFFFD4124) ;- (UDPHS_EPT_1) UDPHS Endpoint Control Enable Register
/* - ========== Register definition for UDPHS_EPT_2 peripheral ========== */
AT91C_UDPHS_EPT_2_EPTCLRSTA EQU (0xFFFD4158) ;- (UDPHS_EPT_2) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_2_EPTCTLDIS EQU (0xFFFD4148) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_2_EPTSETSTA EQU (0xFFFD4154) ;- (UDPHS_EPT_2) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_2_EPTCFG  EQU (0xFFFD4140) ;- (UDPHS_EPT_2) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_2_EPTCTL  EQU (0xFFFD414C) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_2_EPTSTA  EQU (0xFFFD415C) ;- (UDPHS_EPT_2) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_2_EPTCTLENB EQU (0xFFFD4144) ;- (UDPHS_EPT_2) UDPHS Endpoint Control Enable Register
/* - ========== Register definition for UDPHS_EPT_3 peripheral ========== */
AT91C_UDPHS_EPT_3_EPTSTA  EQU (0xFFFD417C) ;- (UDPHS_EPT_3) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_3_EPTSETSTA EQU (0xFFFD4174) ;- (UDPHS_EPT_3) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_3_EPTCTL  EQU (0xFFFD416C) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_3_EPTCTLENB EQU (0xFFFD4164) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_3_EPTCLRSTA EQU (0xFFFD4178) ;- (UDPHS_EPT_3) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_3_EPTCFG  EQU (0xFFFD4160) ;- (UDPHS_EPT_3) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_3_EPTCTLDIS EQU (0xFFFD4168) ;- (UDPHS_EPT_3) UDPHS Endpoint Control Disable Register
/* - ========== Register definition for UDPHS_EPT_4 peripheral ========== */
AT91C_UDPHS_EPT_4_EPTCLRSTA EQU (0xFFFD4198) ;- (UDPHS_EPT_4) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_4_EPTCTL  EQU (0xFFFD418C) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_4_EPTSTA  EQU (0xFFFD419C) ;- (UDPHS_EPT_4) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_4_EPTCTLENB EQU (0xFFFD4184) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_4_EPTCFG  EQU (0xFFFD4180) ;- (UDPHS_EPT_4) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_4_EPTSETSTA EQU (0xFFFD4194) ;- (UDPHS_EPT_4) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_4_EPTCTLDIS EQU (0xFFFD4188) ;- (UDPHS_EPT_4) UDPHS Endpoint Control Disable Register
/* - ========== Register definition for UDPHS_EPT_5 peripheral ========== */
AT91C_UDPHS_EPT_5_EPTSETSTA EQU (0xFFFD41B4) ;- (UDPHS_EPT_5) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_5_EPTCTLDIS EQU (0xFFFD41A8) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_5_EPTCTL  EQU (0xFFFD41AC) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_5_EPTCTLENB EQU (0xFFFD41A4) ;- (UDPHS_EPT_5) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_5_EPTCFG  EQU (0xFFFD41A0) ;- (UDPHS_EPT_5) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_5_EPTCLRSTA EQU (0xFFFD41B8) ;- (UDPHS_EPT_5) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_5_EPTSTA  EQU (0xFFFD41BC) ;- (UDPHS_EPT_5) UDPHS Endpoint Status Register
/* - ========== Register definition for UDPHS_EPT_6 peripheral ========== */
AT91C_UDPHS_EPT_6_EPTCFG  EQU (0xFFFD41C0) ;- (UDPHS_EPT_6) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_6_EPTCLRSTA EQU (0xFFFD41D8) ;- (UDPHS_EPT_6) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_6_EPTCTL  EQU (0xFFFD41CC) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_6_EPTCTLDIS EQU (0xFFFD41C8) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_6_EPTSTA  EQU (0xFFFD41DC) ;- (UDPHS_EPT_6) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_6_EPTSETSTA EQU (0xFFFD41D4) ;- (UDPHS_EPT_6) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_6_EPTCTLENB EQU (0xFFFD41C4) ;- (UDPHS_EPT_6) UDPHS Endpoint Control Enable Register
/* - ========== Register definition for UDPHS_EPT_7 peripheral ========== */
AT91C_UDPHS_EPT_7_EPTSETSTA EQU (0xFFFD41F4) ;- (UDPHS_EPT_7) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_7_EPTCTLDIS EQU (0xFFFD41E8) ;- (UDPHS_EPT_7) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_7_EPTCTLENB EQU (0xFFFD41E4) ;- (UDPHS_EPT_7) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_7_EPTSTA  EQU (0xFFFD41FC) ;- (UDPHS_EPT_7) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_7_EPTCLRSTA EQU (0xFFFD41F8) ;- (UDPHS_EPT_7) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_7_EPTCTL  EQU (0xFFFD41EC) ;- (UDPHS_EPT_7) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_7_EPTCFG  EQU (0xFFFD41E0) ;- (UDPHS_EPT_7) UDPHS Endpoint Config Register
/* - ========== Register definition for UDPHS_EPT_8 peripheral ========== */
AT91C_UDPHS_EPT_8_EPTCLRSTA EQU (0xFFFD4218) ;- (UDPHS_EPT_8) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_8_EPTCTLDIS EQU (0xFFFD4208) ;- (UDPHS_EPT_8) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_8_EPTSTA  EQU (0xFFFD421C) ;- (UDPHS_EPT_8) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_8_EPTCFG  EQU (0xFFFD4200) ;- (UDPHS_EPT_8) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_8_EPTCTL  EQU (0xFFFD420C) ;- (UDPHS_EPT_8) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_8_EPTSETSTA EQU (0xFFFD4214) ;- (UDPHS_EPT_8) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_8_EPTCTLENB EQU (0xFFFD4204) ;- (UDPHS_EPT_8) UDPHS Endpoint Control Enable Register
/* - ========== Register definition for UDPHS_EPT_9 peripheral ========== */
AT91C_UDPHS_EPT_9_EPTCLRSTA EQU (0xFFFD4238) ;- (UDPHS_EPT_9) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_9_EPTCTLENB EQU (0xFFFD4224) ;- (UDPHS_EPT_9) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_9_EPTSTA  EQU (0xFFFD423C) ;- (UDPHS_EPT_9) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_9_EPTSETSTA EQU (0xFFFD4234) ;- (UDPHS_EPT_9) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_9_EPTCTL  EQU (0xFFFD422C) ;- (UDPHS_EPT_9) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_9_EPTCFG  EQU (0xFFFD4220) ;- (UDPHS_EPT_9) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_9_EPTCTLDIS EQU (0xFFFD4228) ;- (UDPHS_EPT_9) UDPHS Endpoint Control Disable Register
/* - ========== Register definition for UDPHS_EPT_10 peripheral ========== */
AT91C_UDPHS_EPT_10_EPTCTL EQU (0xFFFD424C) ;- (UDPHS_EPT_10) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_10_EPTSETSTA EQU (0xFFFD4254) ;- (UDPHS_EPT_10) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_10_EPTCFG EQU (0xFFFD4240) ;- (UDPHS_EPT_10) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_10_EPTCLRSTA EQU (0xFFFD4258) ;- (UDPHS_EPT_10) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_10_EPTSTA EQU (0xFFFD425C) ;- (UDPHS_EPT_10) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_10_EPTCTLDIS EQU (0xFFFD4248) ;- (UDPHS_EPT_10) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_10_EPTCTLENB EQU (0xFFFD4244) ;- (UDPHS_EPT_10) UDPHS Endpoint Control Enable Register
/* - ========== Register definition for UDPHS_EPT_11 peripheral ========== */
AT91C_UDPHS_EPT_11_EPTCTL EQU (0xFFFD426C) ;- (UDPHS_EPT_11) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_11_EPTCFG EQU (0xFFFD4260) ;- (UDPHS_EPT_11) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_11_EPTSTA EQU (0xFFFD427C) ;- (UDPHS_EPT_11) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_11_EPTCTLENB EQU (0xFFFD4264) ;- (UDPHS_EPT_11) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_11_EPTCLRSTA EQU (0xFFFD4278) ;- (UDPHS_EPT_11) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_11_EPTSETSTA EQU (0xFFFD4274) ;- (UDPHS_EPT_11) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_11_EPTCTLDIS EQU (0xFFFD4268) ;- (UDPHS_EPT_11) UDPHS Endpoint Control Disable Register
/* - ========== Register definition for UDPHS_EPT_12 peripheral ========== */
AT91C_UDPHS_EPT_12_EPTCTLENB EQU (0xFFFD4284) ;- (UDPHS_EPT_12) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_12_EPTSTA EQU (0xFFFD429C) ;- (UDPHS_EPT_12) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_12_EPTCTLDIS EQU (0xFFFD4288) ;- (UDPHS_EPT_12) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_12_EPTSETSTA EQU (0xFFFD4294) ;- (UDPHS_EPT_12) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_12_EPTCLRSTA EQU (0xFFFD4298) ;- (UDPHS_EPT_12) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_12_EPTCTL EQU (0xFFFD428C) ;- (UDPHS_EPT_12) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_12_EPTCFG EQU (0xFFFD4280) ;- (UDPHS_EPT_12) UDPHS Endpoint Config Register
/* - ========== Register definition for UDPHS_EPT_13 peripheral ========== */
AT91C_UDPHS_EPT_13_EPTSETSTA EQU (0xFFFD42B4) ;- (UDPHS_EPT_13) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_13_EPTCTLENB EQU (0xFFFD42A4) ;- (UDPHS_EPT_13) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_13_EPTCFG EQU (0xFFFD42A0) ;- (UDPHS_EPT_13) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_13_EPTSTA EQU (0xFFFD42BC) ;- (UDPHS_EPT_13) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_13_EPTCLRSTA EQU (0xFFFD42B8) ;- (UDPHS_EPT_13) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_13_EPTCTLDIS EQU (0xFFFD42A8) ;- (UDPHS_EPT_13) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_13_EPTCTL EQU (0xFFFD42AC) ;- (UDPHS_EPT_13) UDPHS Endpoint Control Register
/* - ========== Register definition for UDPHS_EPT_14 peripheral ========== */
AT91C_UDPHS_EPT_14_EPTCFG EQU (0xFFFD42C0) ;- (UDPHS_EPT_14) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_14_EPTCLRSTA EQU (0xFFFD42D8) ;- (UDPHS_EPT_14) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_14_EPTCTLENB EQU (0xFFFD42C4) ;- (UDPHS_EPT_14) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_14_EPTCTL EQU (0xFFFD42CC) ;- (UDPHS_EPT_14) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_14_EPTSTA EQU (0xFFFD42DC) ;- (UDPHS_EPT_14) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_14_EPTSETSTA EQU (0xFFFD42D4) ;- (UDPHS_EPT_14) UDPHS Endpoint Set Status Register
AT91C_UDPHS_EPT_14_EPTCTLDIS EQU (0xFFFD42C8) ;- (UDPHS_EPT_14) UDPHS Endpoint Control Disable Register
/* - ========== Register definition for UDPHS_EPT_15 peripheral ========== */
AT91C_UDPHS_EPT_15_EPTCLRSTA EQU (0xFFFD42F8) ;- (UDPHS_EPT_15) UDPHS Endpoint Clear Status Register
AT91C_UDPHS_EPT_15_EPTCTLDIS EQU (0xFFFD42E8) ;- (UDPHS_EPT_15) UDPHS Endpoint Control Disable Register
AT91C_UDPHS_EPT_15_EPTSTA EQU (0xFFFD42FC) ;- (UDPHS_EPT_15) UDPHS Endpoint Status Register
AT91C_UDPHS_EPT_15_EPTCFG EQU (0xFFFD42E0) ;- (UDPHS_EPT_15) UDPHS Endpoint Config Register
AT91C_UDPHS_EPT_15_EPTCTLENB EQU (0xFFFD42E4) ;- (UDPHS_EPT_15) UDPHS Endpoint Control Enable Register
AT91C_UDPHS_EPT_15_EPTCTL EQU (0xFFFD42EC) ;- (UDPHS_EPT_15) UDPHS Endpoint Control Register
AT91C_UDPHS_EPT_15_EPTSETSTA EQU (0xFFFD42F4) ;- (UDPHS_EPT_15) UDPHS Endpoint Set Status Register
/* - ========== Register definition for UDPHS_DMA_1 peripheral ========== */
AT91C_UDPHS_DMA_1_DMACONTROL EQU (0xFFFD4318) ;- (UDPHS_DMA_1) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_1_DMAADDRESS EQU (0xFFFD4314) ;- (UDPHS_DMA_1) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_1_DMASTATUS EQU (0xFFFD431C) ;- (UDPHS_DMA_1) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_1_DMANXTDSC EQU (0xFFFD4310) ;- (UDPHS_DMA_1) UDPHS DMA Channel Next Descriptor Address
/* - ========== Register definition for UDPHS_DMA_2 peripheral ========== */
AT91C_UDPHS_DMA_2_DMACONTROL EQU (0xFFFD4328) ;- (UDPHS_DMA_2) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_2_DMASTATUS EQU (0xFFFD432C) ;- (UDPHS_DMA_2) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_2_DMANXTDSC EQU (0xFFFD4320) ;- (UDPHS_DMA_2) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_2_DMAADDRESS EQU (0xFFFD4324) ;- (UDPHS_DMA_2) UDPHS DMA Channel Address Register
/* - ========== Register definition for UDPHS_DMA_3 peripheral ========== */
AT91C_UDPHS_DMA_3_DMACONTROL EQU (0xFFFD4338) ;- (UDPHS_DMA_3) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_3_DMAADDRESS EQU (0xFFFD4334) ;- (UDPHS_DMA_3) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_3_DMANXTDSC EQU (0xFFFD4330) ;- (UDPHS_DMA_3) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_3_DMASTATUS EQU (0xFFFD433C) ;- (UDPHS_DMA_3) UDPHS DMA Channel Status Register
/* - ========== Register definition for UDPHS_DMA_4 peripheral ========== */
AT91C_UDPHS_DMA_4_DMASTATUS EQU (0xFFFD434C) ;- (UDPHS_DMA_4) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_4_DMANXTDSC EQU (0xFFFD4340) ;- (UDPHS_DMA_4) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_4_DMAADDRESS EQU (0xFFFD4344) ;- (UDPHS_DMA_4) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_4_DMACONTROL EQU (0xFFFD4348) ;- (UDPHS_DMA_4) UDPHS DMA Channel Control Register
/* - ========== Register definition for UDPHS_DMA_5 peripheral ========== */
AT91C_UDPHS_DMA_5_DMASTATUS EQU (0xFFFD435C) ;- (UDPHS_DMA_5) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_5_DMACONTROL EQU (0xFFFD4358) ;- (UDPHS_DMA_5) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_5_DMAADDRESS EQU (0xFFFD4354) ;- (UDPHS_DMA_5) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_5_DMANXTDSC EQU (0xFFFD4350) ;- (UDPHS_DMA_5) UDPHS DMA Channel Next Descriptor Address
/* - ========== Register definition for UDPHS_DMA_6 peripheral ========== */
AT91C_UDPHS_DMA_6_DMAADDRESS EQU (0xFFFD4364) ;- (UDPHS_DMA_6) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_6_DMACONTROL EQU (0xFFFD4368) ;- (UDPHS_DMA_6) UDPHS DMA Channel Control Register
AT91C_UDPHS_DMA_6_DMASTATUS EQU (0xFFFD436C) ;- (UDPHS_DMA_6) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_6_DMANXTDSC EQU (0xFFFD4360) ;- (UDPHS_DMA_6) UDPHS DMA Channel Next Descriptor Address
/* - ========== Register definition for UDPHS_DMA_7 peripheral ========== */
AT91C_UDPHS_DMA_7_DMANXTDSC EQU (0xFFFD4370) ;- (UDPHS_DMA_7) UDPHS DMA Channel Next Descriptor Address
AT91C_UDPHS_DMA_7_DMAADDRESS EQU (0xFFFD4374) ;- (UDPHS_DMA_7) UDPHS DMA Channel Address Register
AT91C_UDPHS_DMA_7_DMASTATUS EQU (0xFFFD437C) ;- (UDPHS_DMA_7) UDPHS DMA Channel Status Register
AT91C_UDPHS_DMA_7_DMACONTROL EQU (0xFFFD4378) ;- (UDPHS_DMA_7) UDPHS DMA Channel Control Register
/* - ========== Register definition for UDPHS peripheral ========== */
AT91C_UDPHS_TSTMODREG     EQU (0xFFFD40DC) ;- (UDPHS) UDPHS Test Mode Register
AT91C_UDPHS_RIPNAME2      EQU (0xFFFD40F4) ;- (UDPHS) UDPHS Name2 Register
AT91C_UDPHS_TSTSOFCNT     EQU (0xFFFD40D0) ;- (UDPHS) UDPHS Test SOF Counter Register
AT91C_UDPHS_EPTRST        EQU (0xFFFD401C) ;- (UDPHS) UDPHS Endpoints Reset Register
AT91C_UDPHS_TSTCNTA       EQU (0xFFFD40D4) ;- (UDPHS) UDPHS Test A Counter Register
AT91C_UDPHS_IEN           EQU (0xFFFD4010) ;- (UDPHS) UDPHS Interrupt Enable Register
AT91C_UDPHS_TSTCNTB       EQU (0xFFFD40D8) ;- (UDPHS) UDPHS Test B Counter Register
AT91C_UDPHS_TST           EQU (0xFFFD40E0) ;- (UDPHS) UDPHS Test Register
AT91C_UDPHS_IPFEATURES    EQU (0xFFFD40F8) ;- (UDPHS) UDPHS Features Register
AT91C_UDPHS_RIPNAME1      EQU (0xFFFD40F0) ;- (UDPHS) UDPHS Name1 Register
AT91C_UDPHS_FNUM          EQU (0xFFFD4004) ;- (UDPHS) UDPHS Frame Number Register
AT91C_UDPHS_CLRINT        EQU (0xFFFD4018) ;- (UDPHS) UDPHS Clear Interrupt Register
AT91C_UDPHS_IPVERSION     EQU (0xFFFD40FC) ;- (UDPHS) UDPHS Version Register
AT91C_UDPHS_RIPPADDRSIZE  EQU (0xFFFD40EC) ;- (UDPHS) UDPHS PADDRSIZE Register
AT91C_UDPHS_CTRL          EQU (0xFFFD4000) ;- (UDPHS) UDPHS Control Register
AT91C_UDPHS_INTSTA        EQU (0xFFFD4014) ;- (UDPHS) UDPHS Interrupt Status Register
/* - ========== Register definition for PDC_AC97C peripheral ========== */
AT91C_AC97C_PTSR          EQU (0xFFFD8124) ;- (PDC_AC97C) PDC Transfer Status Register
AT91C_AC97C_PTCR          EQU (0xFFFD8120) ;- (PDC_AC97C) PDC Transfer Control Register
AT91C_AC97C_TNPR          EQU (0xFFFD8118) ;- (PDC_AC97C) Transmit Next Pointer Register
AT91C_AC97C_TNCR          EQU (0xFFFD811C) ;- (PDC_AC97C) Transmit Next Counter Register
AT91C_AC97C_RNPR          EQU (0xFFFD8110) ;- (PDC_AC97C) Receive Next Pointer Register
AT91C_AC97C_RNCR          EQU (0xFFFD8114) ;- (PDC_AC97C) Receive Next Counter Register
AT91C_AC97C_RPR           EQU (0xFFFD8100) ;- (PDC_AC97C) Receive Pointer Register
AT91C_AC97C_TCR           EQU (0xFFFD810C) ;- (PDC_AC97C) Transmit Counter Register
AT91C_AC97C_TPR           EQU (0xFFFD8108) ;- (PDC_AC97C) Transmit Pointer Register
AT91C_AC97C_RCR           EQU (0xFFFD8104) ;- (PDC_AC97C) Receive Counter Register
/* - ========== Register definition for AC97C peripheral ========== */
AT91C_AC97C_CBSR          EQU (0xFFFD8038) ;- (AC97C) Channel B Status Register
AT91C_AC97C_CBMR          EQU (0xFFFD803C) ;- (AC97C) Channel B Mode Register
AT91C_AC97C_CBRHR         EQU (0xFFFD8030) ;- (AC97C) Channel B Receive Holding Register (optional)
AT91C_AC97C_COTHR         EQU (0xFFFD8044) ;- (AC97C) COdec Transmit Holding Register
AT91C_AC97C_OCA           EQU (0xFFFD8014) ;- (AC97C) Output Channel Assignement Register
AT91C_AC97C_IMR           EQU (0xFFFD805C) ;- (AC97C) Interrupt Mask Register
AT91C_AC97C_CORHR         EQU (0xFFFD8040) ;- (AC97C) COdec Transmit Holding Register
AT91C_AC97C_CBTHR         EQU (0xFFFD8034) ;- (AC97C) Channel B Transmit Holding Register (optional)
AT91C_AC97C_CARHR         EQU (0xFFFD8020) ;- (AC97C) Channel A Receive Holding Register
AT91C_AC97C_CASR          EQU (0xFFFD8028) ;- (AC97C) Channel A Status Register
AT91C_AC97C_IER           EQU (0xFFFD8054) ;- (AC97C) Interrupt Enable Register
AT91C_AC97C_MR            EQU (0xFFFD8008) ;- (AC97C) Mode Register
AT91C_AC97C_COSR          EQU (0xFFFD8048) ;- (AC97C) CODEC Status Register
AT91C_AC97C_COMR          EQU (0xFFFD804C) ;- (AC97C) CODEC Mask Status Register
AT91C_AC97C_CATHR         EQU (0xFFFD8024) ;- (AC97C) Channel A Transmit Holding Register
AT91C_AC97C_ICA           EQU (0xFFFD8010) ;- (AC97C) Input Channel AssignementRegister
AT91C_AC97C_IDR           EQU (0xFFFD8058) ;- (AC97C) Interrupt Disable Register
AT91C_AC97C_CAMR          EQU (0xFFFD802C) ;- (AC97C) Channel A Mode Register
AT91C_AC97C_VERSION       EQU (0xFFFD80FC) ;- (AC97C) Version Register
AT91C_AC97C_SR            EQU (0xFFFD8050) ;- (AC97C) Status Register
/* - ========== Register definition for LCDC peripheral ========== */
AT91C_LCDC_MVAL           EQU (0x00500818) ;- (LCDC) LCD Mode Toggle Rate Value Register
AT91C_LCDC_PWRCON         EQU (0x0050083C) ;- (LCDC) Power Control Register
AT91C_LCDC_ISR            EQU (0x00500854) ;- (LCDC) Interrupt Enable Register
AT91C_LCDC_FRMP1          EQU (0x00500008) ;- (LCDC) DMA Frame Pointer Register 1
AT91C_LCDC_CTRSTVAL       EQU (0x00500844) ;- (LCDC) Contrast Value Register
AT91C_LCDC_ICR            EQU (0x00500858) ;- (LCDC) Interrupt Clear Register
AT91C_LCDC_TIM1           EQU (0x00500808) ;- (LCDC) LCD Timing Config 1 Register
AT91C_LCDC_DMACON         EQU (0x0050001C) ;- (LCDC) DMA Control Register
AT91C_LCDC_ITR            EQU (0x00500860) ;- (LCDC) Interrupts Test Register
AT91C_LCDC_IDR            EQU (0x0050084C) ;- (LCDC) Interrupt Disable Register
AT91C_LCDC_DP4_7          EQU (0x00500820) ;- (LCDC) Dithering Pattern DP4_7 Register
AT91C_LCDC_DP5_7          EQU (0x0050082C) ;- (LCDC) Dithering Pattern DP5_7 Register
AT91C_LCDC_IRR            EQU (0x00500864) ;- (LCDC) Interrupts Raw Status Register
AT91C_LCDC_DP3_4          EQU (0x00500830) ;- (LCDC) Dithering Pattern DP3_4 Register
AT91C_LCDC_IMR            EQU (0x00500850) ;- (LCDC) Interrupt Mask Register
AT91C_LCDC_LCDFRCFG       EQU (0x00500810) ;- (LCDC) LCD Frame Config Register
AT91C_LCDC_CTRSTCON       EQU (0x00500840) ;- (LCDC) Contrast Control Register
AT91C_LCDC_DP1_2          EQU (0x0050081C) ;- (LCDC) Dithering Pattern DP1_2 Register
AT91C_LCDC_FRMP2          EQU (0x0050000C) ;- (LCDC) DMA Frame Pointer Register 2
AT91C_LCDC_LCDCON1        EQU (0x00500800) ;- (LCDC) LCD Control 1 Register
AT91C_LCDC_DP4_5          EQU (0x00500834) ;- (LCDC) Dithering Pattern DP4_5 Register
AT91C_LCDC_FRMA2          EQU (0x00500014) ;- (LCDC) DMA Frame Address Register 2
AT91C_LCDC_BA1            EQU (0x00500000) ;- (LCDC) DMA Base Address Register 1
AT91C_LCDC_DMA2DCFG       EQU (0x00500020) ;- (LCDC) DMA 2D addressing configuration
AT91C_LCDC_LUT_ENTRY      EQU (0x00500C00) ;- (LCDC) LUT Entries Register
AT91C_LCDC_DP6_7          EQU (0x00500838) ;- (LCDC) Dithering Pattern DP6_7 Register
AT91C_LCDC_FRMCFG         EQU (0x00500018) ;- (LCDC) DMA Frame Configuration Register
AT91C_LCDC_TIM2           EQU (0x0050080C) ;- (LCDC) LCD Timing Config 2 Register
AT91C_LCDC_DP3_5          EQU (0x00500824) ;- (LCDC) Dithering Pattern DP3_5 Register
AT91C_LCDC_FRMA1          EQU (0x00500010) ;- (LCDC) DMA Frame Address Register 1
AT91C_LCDC_IER            EQU (0x00500848) ;- (LCDC) Interrupt Enable Register
AT91C_LCDC_DP2_3          EQU (0x00500828) ;- (LCDC) Dithering Pattern DP2_3 Register
AT91C_LCDC_FIFO           EQU (0x00500814) ;- (LCDC) LCD FIFO Register
AT91C_LCDC_BA2            EQU (0x00500004) ;- (LCDC) DMA Base Address Register 2
AT91C_LCDC_LCDCON2        EQU (0x00500804) ;- (LCDC) LCD Control 2 Register
AT91C_LCDC_GPR            EQU (0x0050085C) ;- (LCDC) General Purpose Register
/* - ========== Register definition for LCDC_16B_TFT peripheral ========== */
AT91C_TFT_MVAL            EQU (0x00500818) ;- (LCDC_16B_TFT) LCD Mode Toggle Rate Value Register
AT91C_TFT_PWRCON          EQU (0x0050083C) ;- (LCDC_16B_TFT) Power Control Register
AT91C_TFT_ISR             EQU (0x00500854) ;- (LCDC_16B_TFT) Interrupt Enable Register
AT91C_TFT_FRMP1           EQU (0x00500008) ;- (LCDC_16B_TFT) DMA Frame Pointer Register 1
AT91C_TFT_CTRSTVAL        EQU (0x00500844) ;- (LCDC_16B_TFT) Contrast Value Register
AT91C_TFT_ICR             EQU (0x00500858) ;- (LCDC_16B_TFT) Interrupt Clear Register
AT91C_TFT_TIM1            EQU (0x00500808) ;- (LCDC_16B_TFT) LCD Timing Config 1 Register
AT91C_TFT_DMACON          EQU (0x0050001C) ;- (LCDC_16B_TFT) DMA Control Register
AT91C_TFT_ITR             EQU (0x00500860) ;- (LCDC_16B_TFT) Interrupts Test Register
AT91C_TFT_IDR             EQU (0x0050084C) ;- (LCDC_16B_TFT) Interrupt Disable Register
AT91C_TFT_DP4_7           EQU (0x00500820) ;- (LCDC_16B_TFT) Dithering Pattern DP4_7 Register
AT91C_TFT_DP5_7           EQU (0x0050082C) ;- (LCDC_16B_TFT) Dithering Pattern DP5_7 Register
AT91C_TFT_IRR             EQU (0x00500864) ;- (LCDC_16B_TFT) Interrupts Raw Status Register
AT91C_TFT_DP3_4           EQU (0x00500830) ;- (LCDC_16B_TFT) Dithering Pattern DP3_4 Register
AT91C_TFT_IMR             EQU (0x00500850) ;- (LCDC_16B_TFT) Interrupt Mask Register
AT91C_TFT_LCDFRCFG        EQU (0x00500810) ;- (LCDC_16B_TFT) LCD Frame Config Register
AT91C_TFT_CTRSTCON        EQU (0x00500840) ;- (LCDC_16B_TFT) Contrast Control Register
AT91C_TFT_DP1_2           EQU (0x0050081C) ;- (LCDC_16B_TFT) Dithering Pattern DP1_2 Register
AT91C_TFT_FRMP2           EQU (0x0050000C) ;- (LCDC_16B_TFT) DMA Frame Pointer Register 2
AT91C_TFT_LCDCON1         EQU (0x00500800) ;- (LCDC_16B_TFT) LCD Control 1 Register
AT91C_TFT_DP4_5           EQU (0x00500834) ;- (LCDC_16B_TFT) Dithering Pattern DP4_5 Register
AT91C_TFT_FRMA2           EQU (0x00500014) ;- (LCDC_16B_TFT) DMA Frame Address Register 2
AT91C_TFT_BA1             EQU (0x00500000) ;- (LCDC_16B_TFT) DMA Base Address Register 1
AT91C_TFT_DMA2DCFG        EQU (0x00500020) ;- (LCDC_16B_TFT) DMA 2D addressing configuration
AT91C_TFT_LUT_ENTRY       EQU (0x00500C00) ;- (LCDC_16B_TFT) LUT Entries Register
AT91C_TFT_DP6_7           EQU (0x00500838) ;- (LCDC_16B_TFT) Dithering Pattern DP6_7 Register
AT91C_TFT_FRMCFG          EQU (0x00500018) ;- (LCDC_16B_TFT) DMA Frame Configuration Register
AT91C_TFT_TIM2            EQU (0x0050080C) ;- (LCDC_16B_TFT) LCD Timing Config 2 Register
AT91C_TFT_DP3_5           EQU (0x00500824) ;- (LCDC_16B_TFT) Dithering Pattern DP3_5 Register
AT91C_TFT_FRMA1           EQU (0x00500010) ;- (LCDC_16B_TFT) DMA Frame Address Register 1
AT91C_TFT_IER             EQU (0x00500848) ;- (LCDC_16B_TFT) Interrupt Enable Register
AT91C_TFT_DP2_3           EQU (0x00500828) ;- (LCDC_16B_TFT) Dithering Pattern DP2_3 Register
AT91C_TFT_FIFO            EQU (0x00500814) ;- (LCDC_16B_TFT) LCD FIFO Register
AT91C_TFT_BA2             EQU (0x00500004) ;- (LCDC_16B_TFT) DMA Base Address Register 2
AT91C_TFT_LCDCON2         EQU (0x00500804) ;- (LCDC_16B_TFT) LCD Control 2 Register
AT91C_TFT_GPR             EQU (0x0050085C) ;- (LCDC_16B_TFT) General Purpose Register
/* - ========== Register definition for HDMA_CH_0 peripheral ========== */
AT91C_HDMA_CH_0_CFG       EQU (0xFFFFE650) ;- (HDMA_CH_0) HDMA Channel Configuration Register
AT91C_HDMA_CH_0_CTRLB     EQU (0xFFFFE64C) ;- (HDMA_CH_0) HDMA Channel Control B Register
AT91C_HDMA_CH_0_DADDR     EQU (0xFFFFE640) ;- (HDMA_CH_0) HDMA Channel Destination Address Register
AT91C_HDMA_CH_0_DPIP      EQU (0xFFFFE658) ;- (HDMA_CH_0) HDMA Channel Destination Picture in Picture Configuration Register
AT91C_HDMA_CH_0_SPIP      EQU (0xFFFFE654) ;- (HDMA_CH_0) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_0_DSCR      EQU (0xFFFFE644) ;- (HDMA_CH_0) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_0_BDSCR     EQU (0xFFFFE65C) ;- (HDMA_CH_0) HDMA Reserved
AT91C_HDMA_CH_0_SADDR     EQU (0xFFFFE63C) ;- (HDMA_CH_0) HDMA Channel Source Address Register
AT91C_HDMA_CH_0_CTRLA     EQU (0xFFFFE648) ;- (HDMA_CH_0) HDMA Channel Control A Register
AT91C_HDMA_CH_0_CADDR     EQU (0xFFFFE660) ;- (HDMA_CH_0) HDMA Reserved
/* - ========== Register definition for HDMA_CH_1 peripheral ========== */
AT91C_HDMA_CH_1_DADDR     EQU (0xFFFFE668) ;- (HDMA_CH_1) HDMA Channel Destination Address Register
AT91C_HDMA_CH_1_BDSCR     EQU (0xFFFFE684) ;- (HDMA_CH_1) HDMA Reserved
AT91C_HDMA_CH_1_CFG       EQU (0xFFFFE678) ;- (HDMA_CH_1) HDMA Channel Configuration Register
AT91C_HDMA_CH_1_CTRLB     EQU (0xFFFFE674) ;- (HDMA_CH_1) HDMA Channel Control B Register
AT91C_HDMA_CH_1_SADDR     EQU (0xFFFFE664) ;- (HDMA_CH_1) HDMA Channel Source Address Register
AT91C_HDMA_CH_1_CTRLA     EQU (0xFFFFE670) ;- (HDMA_CH_1) HDMA Channel Control A Register
AT91C_HDMA_CH_1_CADDR     EQU (0xFFFFE688) ;- (HDMA_CH_1) HDMA Reserved
AT91C_HDMA_CH_1_SPIP      EQU (0xFFFFE67C) ;- (HDMA_CH_1) HDMA Channel Source Picture in Picture Configuration Register
AT91C_HDMA_CH_1_DSCR      EQU (0xFFFFE66C) ;- (HDMA_CH_1) HDMA Channel Descriptor Address Register
AT91C_HDMA_CH_1_DPIP      EQU (0xFFFFE680) ;- (HDMA_CH_1) HDMA Channel Destination Picture in Picture Configuration Register
/* - ========== Register definition for HDMA peripheral ========== */
AT91C_HDMA_CHER           EQU (0xFFFFE628) ;- (HDMA) HDMA Channel Handler Enable Register
AT91C_HDMA_EN             EQU (0xFFFFE604) ;- (HDMA) HDMA Controller Enable Register
AT91C_HDMA_EBCIMR         EQU (0xFFFFE620) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Mask Register
AT91C_HDMA_BREQ           EQU (0xFFFFE60C) ;- (HDMA) HDMA Software Chunk Transfer Request Register
AT91C_HDMA_EBCIDR         EQU (0xFFFFE61C) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Disable register
AT91C_HDMA_CHDR           EQU (0xFFFFE62C) ;- (HDMA) HDMA Channel Handler Disable Register
AT91C_HDMA_RSVD0          EQU (0xFFFFE634) ;- (HDMA) HDMA Reserved
AT91C_HDMA_SYNC           EQU (0xFFFFE614) ;- (HDMA) HDMA Request Synchronization Register
AT91C_HDMA_GCFG           EQU (0xFFFFE600) ;- (HDMA) HDMA Global Configuration Register
AT91C_HDMA_EBCISR         EQU (0xFFFFE624) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Status Register
AT91C_HDMA_CHSR           EQU (0xFFFFE630) ;- (HDMA) HDMA Channel Handler Status Register
AT91C_HDMA_LAST           EQU (0xFFFFE610) ;- (HDMA) HDMA Software Last Transfer Flag Register
AT91C_HDMA_RSVD1          EQU (0xFFFFE638) ;- (HDMA) HDMA Reserved
AT91C_HDMA_SREQ           EQU (0xFFFFE608) ;- (HDMA) HDMA Software Single Request Register
AT91C_HDMA_EBCIER         EQU (0xFFFFE618) ;- (HDMA) HDMA Error, Chained Buffer transfer completed and Buffer transfer completed Interrupt Enable register
/* - ========== Register definition for HECC peripheral ========== */
AT91C_HECC_VR             EQU (0xFFFFE8FC) ;- (HECC)  ECC Version register
AT91C_HECC_NPR            EQU (0xFFFFE810) ;- (HECC)  ECC Parity N register
AT91C_HECC_SR             EQU (0xFFFFE808) ;- (HECC)  ECC Status register
AT91C_HECC_PR             EQU (0xFFFFE80C) ;- (HECC)  ECC Parity register
AT91C_HECC_MR             EQU (0xFFFFE804) ;- (HECC)  ECC Page size register
AT91C_HECC_CR             EQU (0xFFFFE800) ;- (HECC)  ECC reset register

/* - ******************************************************************************/
/* -               PIO DEFINITIONS FOR AT91SAM9RL64*/
/* - ******************************************************************************/
AT91C_PIO_PA0             EQU (1 <<  0) ;- Pin Controlled by PA0
AT91C_PA0_MC_DA0          EQU (AT91C_PIO_PA0) ;-  
AT91C_PIO_PA1             EQU (1 <<  1) ;- Pin Controlled by PA1
AT91C_PA1_MC_CDA          EQU (AT91C_PIO_PA1) ;-  
AT91C_PIO_PA10            EQU (1 << 10) ;- Pin Controlled by PA10
AT91C_PA10_CTS0           EQU (AT91C_PIO_PA10) ;-  
AT91C_PA10_RK0            EQU (AT91C_PIO_PA10) ;-  
AT91C_PIO_PA11            EQU (1 << 11) ;- Pin Controlled by PA11
AT91C_PA11_TXD1           EQU (AT91C_PIO_PA11) ;-  
AT91C_PIO_PA12            EQU (1 << 12) ;- Pin Controlled by PA12
AT91C_PA12_RXD1           EQU (AT91C_PIO_PA12) ;-  
AT91C_PIO_PA13            EQU (1 << 13) ;- Pin Controlled by PA13
AT91C_PA13_TXD2           EQU (AT91C_PIO_PA13) ;-  
AT91C_PA13_TD1            EQU (AT91C_PIO_PA13) ;-  
AT91C_PIO_PA14            EQU (1 << 14) ;- Pin Controlled by PA14
AT91C_PA14_RXD2           EQU (AT91C_PIO_PA14) ;-  
AT91C_PA14_RD1            EQU (AT91C_PIO_PA14) ;-  
AT91C_PIO_PA15            EQU (1 << 15) ;- Pin Controlled by PA15
AT91C_PA15_TD0            EQU (AT91C_PIO_PA15) ;-  
AT91C_PIO_PA16            EQU (1 << 16) ;- Pin Controlled by PA16
AT91C_PA16_RD0            EQU (AT91C_PIO_PA16) ;-  
AT91C_PIO_PA17            EQU (1 << 17) ;- Pin Controlled by PA17
AT91C_PA17_AD0            EQU (AT91C_PIO_PA17) ;-  
AT91C_PIO_PA18            EQU (1 << 18) ;- Pin Controlled by PA18
AT91C_PA18_AD1            EQU (AT91C_PIO_PA18) ;-  
AT91C_PA18_RTS1           EQU (AT91C_PIO_PA18) ;-  
AT91C_PIO_PA19            EQU (1 << 19) ;- Pin Controlled by PA19
AT91C_PA19_AD2            EQU (AT91C_PIO_PA19) ;-  
AT91C_PA19_CTS1           EQU (AT91C_PIO_PA19) ;-  
AT91C_PIO_PA2             EQU (1 <<  2) ;- Pin Controlled by PA2
AT91C_PA2_MC_CK           EQU (AT91C_PIO_PA2) ;-  
AT91C_PIO_PA20            EQU (1 << 20) ;- Pin Controlled by PA20
AT91C_PA20_AD3            EQU (AT91C_PIO_PA20) ;-  
AT91C_PA20_SCK3           EQU (AT91C_PIO_PA20) ;-  
AT91C_PIO_PA21            EQU (1 << 21) ;- Pin Controlled by PA21
AT91C_PA21_DRXD           EQU (AT91C_PIO_PA21) ;-  
AT91C_PIO_PA22            EQU (1 << 22) ;- Pin Controlled by PA22
AT91C_PA22_DTXD           EQU (AT91C_PIO_PA22) ;-  
AT91C_PA22_RF0            EQU (AT91C_PIO_PA22) ;-  
AT91C_PIO_PA23            EQU (1 << 23) ;- Pin Controlled by PA23
AT91C_PA23_TWD0           EQU (AT91C_PIO_PA23) ;-  
AT91C_PIO_PA24            EQU (1 << 24) ;- Pin Controlled by PA24
AT91C_PA24_TWCK0          EQU (AT91C_PIO_PA24) ;-  
AT91C_PIO_PA25            EQU (1 << 25) ;- Pin Controlled by PA25
AT91C_PA25_MISO           EQU (AT91C_PIO_PA25) ;-  
AT91C_PIO_PA26            EQU (1 << 26) ;- Pin Controlled by PA26
AT91C_PA26_MOSI           EQU (AT91C_PIO_PA26) ;-  
AT91C_PIO_PA27            EQU (1 << 27) ;- Pin Controlled by PA27
AT91C_PA27_SPCK           EQU (AT91C_PIO_PA27) ;-  
AT91C_PIO_PA28            EQU (1 << 28) ;- Pin Controlled by PA28
AT91C_PA28_NPCS0          EQU (AT91C_PIO_PA28) ;-  
AT91C_PIO_PA29            EQU (1 << 29) ;- Pin Controlled by PA29
AT91C_PA29_RTS2           EQU (AT91C_PIO_PA29) ;-  
AT91C_PA29_TF1            EQU (AT91C_PIO_PA29) ;-  
AT91C_PIO_PA3             EQU (1 <<  3) ;- Pin Controlled by PA3
AT91C_PA3_MC_DA1          EQU (AT91C_PIO_PA3) ;-  
AT91C_PA3_TCLK0           EQU (AT91C_PIO_PA3) ;-  
AT91C_PIO_PA30            EQU (1 << 30) ;- Pin Controlled by PA30
AT91C_PA30_CTS2           EQU (AT91C_PIO_PA30) ;-  
AT91C_PA30_TK1            EQU (AT91C_PIO_PA30) ;-  
AT91C_PIO_PA31            EQU (1 << 31) ;- Pin Controlled by PA31
AT91C_PA31_NWAIT          EQU (AT91C_PIO_PA31) ;-  
AT91C_PA31_IRQ            EQU (AT91C_PIO_PA31) ;-  
AT91C_PIO_PA4             EQU (1 <<  4) ;- Pin Controlled by PA4
AT91C_PA4_MC_DA2          EQU (AT91C_PIO_PA4) ;-  
AT91C_PA4_TIOA0           EQU (AT91C_PIO_PA4) ;-  
AT91C_PIO_PA5             EQU (1 <<  5) ;- Pin Controlled by PA5
AT91C_PA5_MC_DA3          EQU (AT91C_PIO_PA5) ;-  
AT91C_PA5_TIOB0           EQU (AT91C_PIO_PA5) ;-  
AT91C_PIO_PA6             EQU (1 <<  6) ;- Pin Controlled by PA6
AT91C_PA6_TXD0            EQU (AT91C_PIO_PA6) ;-  
AT91C_PIO_PA7             EQU (1 <<  7) ;- Pin Controlled by PA7
AT91C_PA7_RXD0            EQU (AT91C_PIO_PA7) ;-  
AT91C_PIO_PA8             EQU (1 <<  8) ;- Pin Controlled by PA8
AT91C_PA8_SCK0            EQU (AT91C_PIO_PA8) ;-  
AT91C_PA8_RF1             EQU (AT91C_PIO_PA8) ;-  
AT91C_PIO_PA9             EQU (1 <<  9) ;- Pin Controlled by PA9
AT91C_PA9_RTS0            EQU (AT91C_PIO_PA9) ;-  
AT91C_PA9_RK1             EQU (AT91C_PIO_PA9) ;-  
AT91C_PIO_PB0             EQU (1 <<  0) ;- Pin Controlled by PB0
AT91C_PB0_TXD3            EQU (AT91C_PIO_PB0) ;-  
AT91C_PIO_PB1             EQU (1 <<  1) ;- Pin Controlled by PB1
AT91C_PB1_RXD3            EQU (AT91C_PIO_PB1) ;-  
AT91C_PIO_PB10            EQU (1 << 10) ;- Pin Controlled by PB10
AT91C_PB10_A25_CFRNW      EQU (AT91C_PIO_PB10) ;-  
AT91C_PB10_FIQ            EQU (AT91C_PIO_PB10) ;-  
AT91C_PIO_PB11            EQU (1 << 11) ;- Pin Controlled by PB11
AT91C_PB11_A18            EQU (AT91C_PIO_PB11) ;-  
AT91C_PIO_PB12            EQU (1 << 12) ;- Pin Controlled by PB12
AT91C_PB12_A19            EQU (AT91C_PIO_PB12) ;-  
AT91C_PIO_PB13            EQU (1 << 13) ;- Pin Controlled by PB13
AT91C_PB13_A20            EQU (AT91C_PIO_PB13) ;-  
AT91C_PIO_PB14            EQU (1 << 14) ;- Pin Controlled by PB14
AT91C_PB14_A23            EQU (AT91C_PIO_PB14) ;-  
AT91C_PB14_PCK0           EQU (AT91C_PIO_PB14) ;-  
AT91C_PIO_PB15            EQU (1 << 15) ;- Pin Controlled by PB15
AT91C_PB15_A24            EQU (AT91C_PIO_PB15) ;-  
AT91C_PB15_ADTRG          EQU (AT91C_PIO_PB15) ;-  
AT91C_PIO_PB16            EQU (1 << 16) ;- Pin Controlled by PB16
AT91C_PB16_D16            EQU (AT91C_PIO_PB16) ;-  
AT91C_PIO_PB17            EQU (1 << 17) ;- Pin Controlled by PB17
AT91C_PB17_D17            EQU (AT91C_PIO_PB17) ;-  
AT91C_PIO_PB18            EQU (1 << 18) ;- Pin Controlled by PB18
AT91C_PB18_D18            EQU (AT91C_PIO_PB18) ;-  
AT91C_PIO_PB19            EQU (1 << 19) ;- Pin Controlled by PB19
AT91C_PB19_D19            EQU (AT91C_PIO_PB19) ;-  
AT91C_PIO_PB2             EQU (1 <<  2) ;- Pin Controlled by PB2
AT91C_PB2_A21_NANDALE     EQU (AT91C_PIO_PB2) ;-  
AT91C_PIO_PB20            EQU (1 << 20) ;- Pin Controlled by PB20
AT91C_PB20_D20            EQU (AT91C_PIO_PB20) ;-  
AT91C_PIO_PB21            EQU (1 << 21) ;- Pin Controlled by PB21
AT91C_PB21_D21            EQU (AT91C_PIO_PB21) ;-  
AT91C_PIO_PB22            EQU (1 << 22) ;- Pin Controlled by PB22
AT91C_PB22_D22            EQU (AT91C_PIO_PB22) ;-  
AT91C_PIO_PB23            EQU (1 << 23) ;- Pin Controlled by PB23
AT91C_PB23_D23            EQU (AT91C_PIO_PB23) ;-  
AT91C_PIO_PB24            EQU (1 << 24) ;- Pin Controlled by PB24
AT91C_PB24_D24            EQU (AT91C_PIO_PB24) ;-  
AT91C_PIO_PB25            EQU (1 << 25) ;- Pin Controlled by PB25
AT91C_PB25_D25            EQU (AT91C_PIO_PB25) ;-  
AT91C_PIO_PB26            EQU (1 << 26) ;- Pin Controlled by PB26
AT91C_PB26_D26            EQU (AT91C_PIO_PB26) ;-  
AT91C_PIO_PB27            EQU (1 << 27) ;- Pin Controlled by PB27
AT91C_PB27_D27            EQU (AT91C_PIO_PB27) ;-  
AT91C_PIO_PB28            EQU (1 << 28) ;- Pin Controlled by PB28
AT91C_PB28_D28            EQU (AT91C_PIO_PB28) ;-  
AT91C_PIO_PB29            EQU (1 << 29) ;- Pin Controlled by PB29
AT91C_PB29_D29            EQU (AT91C_PIO_PB29) ;-  
AT91C_PIO_PB3             EQU (1 <<  3) ;- Pin Controlled by PB3
AT91C_PB3_A22_NANDCLE     EQU (AT91C_PIO_PB3) ;-  
AT91C_PIO_PB30            EQU (1 << 30) ;- Pin Controlled by PB30
AT91C_PB30_D30            EQU (AT91C_PIO_PB30) ;-  
AT91C_PIO_PB31            EQU (1 << 31) ;- Pin Controlled by PB31
AT91C_PB31_D31            EQU (AT91C_PIO_PB31) ;-  
AT91C_PIO_PB4             EQU (1 <<  4) ;- Pin Controlled by PB4
AT91C_PB4_NANDOE          EQU (AT91C_PIO_PB4) ;-  
AT91C_PIO_PB5             EQU (1 <<  5) ;- Pin Controlled by PB5
AT91C_PB5_NANDWE          EQU (AT91C_PIO_PB5) ;-  
AT91C_PIO_PB6             EQU (1 <<  6) ;- Pin Controlled by PB6
AT91C_PB6_NCS3_NANDCS     EQU (AT91C_PIO_PB6) ;-  
AT91C_PIO_PB7             EQU (1 <<  7) ;- Pin Controlled by PB7
AT91C_PB7_NCS4_CFCS0      EQU (AT91C_PIO_PB7) ;-  
AT91C_PB7_NPCS1           EQU (AT91C_PIO_PB7) ;-  
AT91C_PIO_PB8             EQU (1 <<  8) ;- Pin Controlled by PB8
AT91C_PB8_CFE1            EQU (AT91C_PIO_PB8) ;-  
AT91C_PB8_PWM0            EQU (AT91C_PIO_PB8) ;-  
AT91C_PIO_PB9             EQU (1 <<  9) ;- Pin Controlled by PB9
AT91C_PB9_CFE2            EQU (AT91C_PIO_PB9) ;-  
AT91C_PB9_PWM1            EQU (AT91C_PIO_PB9) ;-  
AT91C_PIO_PC0             EQU (1 <<  0) ;- Pin Controlled by PC0
AT91C_PC0_TF0             EQU (AT91C_PIO_PC0) ;-  
AT91C_PIO_PC1             EQU (1 <<  1) ;- Pin Controlled by PC1
AT91C_PC1_TK0             EQU (AT91C_PIO_PC1) ;-  
AT91C_PC1_LCDPWR          EQU (AT91C_PIO_PC1) ;-  
AT91C_PIO_PC10            EQU (1 << 10) ;- Pin Controlled by PC10
AT91C_PC10_LCDD2          EQU (AT91C_PIO_PC10) ;-  
AT91C_PC10_LCDD4          EQU (AT91C_PIO_PC10) ;-  
AT91C_PIO_PC11            EQU (1 << 11) ;- Pin Controlled by PC11
AT91C_PC11_LCDD3          EQU (AT91C_PIO_PC11) ;-  
AT91C_PC11_LCDD5          EQU (AT91C_PIO_PC11) ;-  
AT91C_PIO_PC12            EQU (1 << 12) ;- Pin Controlled by PC12
AT91C_PC12_LCDD4          EQU (AT91C_PIO_PC12) ;-  
AT91C_PC12_LCDD6          EQU (AT91C_PIO_PC12) ;-  
AT91C_PIO_PC13            EQU (1 << 13) ;- Pin Controlled by PC13
AT91C_PC13_LCDD5          EQU (AT91C_PIO_PC13) ;-  
AT91C_PC13_LCDD7          EQU (AT91C_PIO_PC13) ;-  
AT91C_PIO_PC14            EQU (1 << 14) ;- Pin Controlled by PC14
AT91C_PC14_LCDD6          EQU (AT91C_PIO_PC14) ;-  
AT91C_PC14_LCDD10         EQU (AT91C_PIO_PC14) ;-  
AT91C_PIO_PC15            EQU (1 << 15) ;- Pin Controlled by PC15
AT91C_PC15_LCDD7          EQU (AT91C_PIO_PC15) ;-  
AT91C_PC15_LCDD11         EQU (AT91C_PIO_PC15) ;-  
AT91C_PIO_PC16            EQU (1 << 16) ;- Pin Controlled by PC16
AT91C_PC16_LCDD8          EQU (AT91C_PIO_PC16) ;-  
AT91C_PC16_LCDD12         EQU (AT91C_PIO_PC16) ;-  
AT91C_PIO_PC17            EQU (1 << 17) ;- Pin Controlled by PC17
AT91C_PC17_LCDD9          EQU (AT91C_PIO_PC17) ;-  
AT91C_PC17_LCDD13         EQU (AT91C_PIO_PC17) ;-  
AT91C_PIO_PC18            EQU (1 << 18) ;- Pin Controlled by PC18
AT91C_PC18_LCDD10         EQU (AT91C_PIO_PC18) ;-  
AT91C_PC18_LCDD14         EQU (AT91C_PIO_PC18) ;-  
AT91C_PIO_PC19            EQU (1 << 19) ;- Pin Controlled by PC19
AT91C_PC19_LCDD11         EQU (AT91C_PIO_PC19) ;-  
AT91C_PC19_LCDD15         EQU (AT91C_PIO_PC19) ;-  
AT91C_PIO_PC2             EQU (1 <<  2) ;- Pin Controlled by PC2
AT91C_PC2_LCDMOD          EQU (AT91C_PIO_PC2) ;-  
AT91C_PC2_PWM0            EQU (AT91C_PIO_PC2) ;-  
AT91C_PIO_PC20            EQU (1 << 20) ;- Pin Controlled by PC20
AT91C_PC20_LCDD12         EQU (AT91C_PIO_PC20) ;-  
AT91C_PC20_LCDD18         EQU (AT91C_PIO_PC20) ;-  
AT91C_PIO_PC21            EQU (1 << 21) ;- Pin Controlled by PC21
AT91C_PC21_LCDD13         EQU (AT91C_PIO_PC21) ;-  
AT91C_PC21_LCDD19         EQU (AT91C_PIO_PC21) ;-  
AT91C_PIO_PC22            EQU (1 << 22) ;- Pin Controlled by PC22
AT91C_PC22_LCDD14         EQU (AT91C_PIO_PC22) ;-  
AT91C_PC22_LCDD20         EQU (AT91C_PIO_PC22) ;-  
AT91C_PIO_PC23            EQU (1 << 23) ;- Pin Controlled by PC23
AT91C_PC23_LCDD15         EQU (AT91C_PIO_PC23) ;-  
AT91C_PC23_LCDD21         EQU (AT91C_PIO_PC23) ;-  
AT91C_PIO_PC24            EQU (1 << 24) ;- Pin Controlled by PC24
AT91C_PC24_LCDD16         EQU (AT91C_PIO_PC24) ;-  
AT91C_PC24_LCDD22         EQU (AT91C_PIO_PC24) ;-  
AT91C_PIO_PC25            EQU (1 << 25) ;- Pin Controlled by PC25
AT91C_PC25_LCDD17         EQU (AT91C_PIO_PC25) ;-  
AT91C_PC25_LCDD23         EQU (AT91C_PIO_PC25) ;-  
AT91C_PIO_PC26            EQU (1 << 26) ;- Pin Controlled by PC26
AT91C_PC26_LCDD18         EQU (AT91C_PIO_PC26) ;-  
AT91C_PIO_PC27            EQU (1 << 27) ;- Pin Controlled by PC27
AT91C_PC27_LCDD19         EQU (AT91C_PIO_PC27) ;-  
AT91C_PIO_PC28            EQU (1 << 28) ;- Pin Controlled by PC28
AT91C_PC28_LCDD20         EQU (AT91C_PIO_PC28) ;-  
AT91C_PIO_PC29            EQU (1 << 29) ;- Pin Controlled by PC29
AT91C_PC29_LCDD21         EQU (AT91C_PIO_PC29) ;-  
AT91C_PC29_TIOA1          EQU (AT91C_PIO_PC29) ;-  
AT91C_PIO_PC3             EQU (1 <<  3) ;- Pin Controlled by PC3
AT91C_PC3_LCDCC           EQU (AT91C_PIO_PC3) ;-  
AT91C_PC3_PWM1            EQU (AT91C_PIO_PC3) ;-  
AT91C_PIO_PC30            EQU (1 << 30) ;- Pin Controlled by PC30
AT91C_PC30_LCDD22         EQU (AT91C_PIO_PC30) ;-  
AT91C_PC30_TIOB1          EQU (AT91C_PIO_PC30) ;-  
AT91C_PIO_PC31            EQU (1 << 31) ;- Pin Controlled by PC31
AT91C_PC31_LCDD23         EQU (AT91C_PIO_PC31) ;-  
AT91C_PC31_TCLK1          EQU (AT91C_PIO_PC31) ;-  
AT91C_PIO_PC4             EQU (1 <<  4) ;- Pin Controlled by PC4
AT91C_PC4_LCDVSYNC        EQU (AT91C_PIO_PC4) ;-  
AT91C_PIO_PC5             EQU (1 <<  5) ;- Pin Controlled by PC5
AT91C_PC5_LCDHSYNC        EQU (AT91C_PIO_PC5) ;-  
AT91C_PIO_PC6             EQU (1 <<  6) ;- Pin Controlled by PC6
AT91C_PC6_LCDDOTCK        EQU (AT91C_PIO_PC6) ;-  
AT91C_PIO_PC7             EQU (1 <<  7) ;- Pin Controlled by PC7
AT91C_PC7_LCDDEN          EQU (AT91C_PIO_PC7) ;-  
AT91C_PIO_PC8             EQU (1 <<  8) ;- Pin Controlled by PC8
AT91C_PC8_LCDD0           EQU (AT91C_PIO_PC8) ;-  
AT91C_PC8_LCDD2           EQU (AT91C_PIO_PC8) ;-  
AT91C_PIO_PC9             EQU (1 <<  9) ;- Pin Controlled by PC9
AT91C_PC9_LCDD1           EQU (AT91C_PIO_PC9) ;-  
AT91C_PC9_LCDD3           EQU (AT91C_PIO_PC9) ;-  
AT91C_PIO_PD0             EQU (1 <<  0) ;- Pin Controlled by PD0
AT91C_PD0_NCS2            EQU (AT91C_PIO_PD0) ;-  
AT91C_PIO_PD1             EQU (1 <<  1) ;- Pin Controlled by PD1
AT91C_PD1_AC97_FS         EQU (AT91C_PIO_PD1) ;-  
AT91C_PIO_PD10            EQU (1 << 10) ;- Pin Controlled by PD10
AT91C_PD10_TWD1           EQU (AT91C_PIO_PD10) ;-  
AT91C_PD10_TIOA2          EQU (AT91C_PIO_PD10) ;-  
AT91C_PIO_PD11            EQU (1 << 11) ;- Pin Controlled by PD11
AT91C_PD11_TWCK1          EQU (AT91C_PIO_PD11) ;-  
AT91C_PD11_TIOB2          EQU (AT91C_PIO_PD11) ;-  
AT91C_PIO_PD12            EQU (1 << 12) ;- Pin Controlled by PD12
AT91C_PD12_PWM2           EQU (AT91C_PIO_PD12) ;-  
AT91C_PD12_PCK1           EQU (AT91C_PIO_PD12) ;-  
AT91C_PIO_PD13            EQU (1 << 13) ;- Pin Controlled by PD13
AT91C_PD13_NCS5_CFCS1     EQU (AT91C_PIO_PD13) ;-  
AT91C_PD13_NPCS3          EQU (AT91C_PIO_PD13) ;-  
AT91C_PIO_PD14            EQU (1 << 14) ;- Pin Controlled by PD14
AT91C_PD14_DSR0           EQU (AT91C_PIO_PD14) ;-  
AT91C_PD14_PWM0           EQU (AT91C_PIO_PD14) ;-  
AT91C_PIO_PD15            EQU (1 << 15) ;- Pin Controlled by PD15
AT91C_PD15_DTR0           EQU (AT91C_PIO_PD15) ;-  
AT91C_PD15_PWM1           EQU (AT91C_PIO_PD15) ;-  
AT91C_PIO_PD16            EQU (1 << 16) ;- Pin Controlled by PD16
AT91C_PD16_DCD0           EQU (AT91C_PIO_PD16) ;-  
AT91C_PD16_PWM2           EQU (AT91C_PIO_PD16) ;-  
AT91C_PIO_PD17            EQU (1 << 17) ;- Pin Controlled by PD17
AT91C_PD17_RI0            EQU (AT91C_PIO_PD17) ;-  
AT91C_PIO_PD18            EQU (1 << 18) ;- Pin Controlled by PD18
AT91C_PD18_PWM3           EQU (AT91C_PIO_PD18) ;-  
AT91C_PIO_PD19            EQU (1 << 19) ;- Pin Controlled by PD19
AT91C_PD19_PCK0           EQU (AT91C_PIO_PD19) ;-  
AT91C_PIO_PD2             EQU (1 <<  2) ;- Pin Controlled by PD2
AT91C_PD2_AC97_CK         EQU (AT91C_PIO_PD2) ;-  
AT91C_PD2_SCK1            EQU (AT91C_PIO_PD2) ;-  
AT91C_PIO_PD20            EQU (1 << 20) ;- Pin Controlled by PD20
AT91C_PD20_PCK1           EQU (AT91C_PIO_PD20) ;-  
AT91C_PIO_PD21            EQU (1 << 21) ;- Pin Controlled by PD21
AT91C_PD21_TCLK2          EQU (AT91C_PIO_PD21) ;-  
AT91C_PIO_PD3             EQU (1 <<  3) ;- Pin Controlled by PD3
AT91C_PD3_AC97_TX         EQU (AT91C_PIO_PD3) ;-  
AT91C_PD3_CTS3            EQU (AT91C_PIO_PD3) ;-  
AT91C_PIO_PD4             EQU (1 <<  4) ;- Pin Controlled by PD4
AT91C_PD4_AC97_RX         EQU (AT91C_PIO_PD4) ;-  
AT91C_PD4_RTS3            EQU (AT91C_PIO_PD4) ;-  
AT91C_PIO_PD5             EQU (1 <<  5) ;- Pin Controlled by PD5
AT91C_PD5_DTXD            EQU (AT91C_PIO_PD5) ;-  
AT91C_PD5_PWM2            EQU (AT91C_PIO_PD5) ;-  
AT91C_PIO_PD6             EQU (1 <<  6) ;- Pin Controlled by PD6
AT91C_PD6_AD4             EQU (AT91C_PIO_PD6) ;-  
AT91C_PIO_PD7             EQU (1 <<  7) ;- Pin Controlled by PD7
AT91C_PD7_AD5             EQU (AT91C_PIO_PD7) ;-  
AT91C_PIO_PD8             EQU (1 <<  8) ;- Pin Controlled by PD8
AT91C_PD8_NPCS2           EQU (AT91C_PIO_PD8) ;-  
AT91C_PD8_PWM3            EQU (AT91C_PIO_PD8) ;-  
AT91C_PIO_PD9             EQU (1 <<  9) ;- Pin Controlled by PD9
AT91C_PD9_SCK2            EQU (AT91C_PIO_PD9) ;-  
AT91C_PD9_NPCS3           EQU (AT91C_PIO_PD9) ;-  

/* - ******************************************************************************/
/* -               PERIPHERAL ID DEFINITIONS FOR AT91SAM9RL64*/
/* - ******************************************************************************/
AT91C_ID_FIQ              EQU ( 0) ;- Advanced Interrupt Controller (FIQ)
AT91C_ID_SYS              EQU ( 1) ;- System Controller
AT91C_ID_PIOA             EQU ( 2) ;- Parallel IO Controller A
AT91C_ID_PIOB             EQU ( 3) ;- Parallel IO Controller B
AT91C_ID_PIOC             EQU ( 4) ;- Parallel IO Controller C
AT91C_ID_PIOD             EQU ( 5) ;- Parallel IO Controller D
AT91C_ID_US0              EQU ( 6) ;- USART 0
AT91C_ID_US1              EQU ( 7) ;- USART 1
AT91C_ID_US2              EQU ( 8) ;- USART 2
AT91C_ID_US3              EQU ( 9) ;- USART 2
AT91C_ID_MCI              EQU (10) ;- Multimedia Card Interface
AT91C_ID_TWI0             EQU (11) ;- TWI 0
AT91C_ID_TWI1             EQU (12) ;- TWI 1
AT91C_ID_SPI              EQU (13) ;- Serial Peripheral Interface
AT91C_ID_SSC0             EQU (14) ;- Serial Synchronous Controller 0
AT91C_ID_SSC1             EQU (15) ;- Serial Synchronous Controller 1
AT91C_ID_TC0              EQU (16) ;- Timer Counter 0
AT91C_ID_TC1              EQU (17) ;- Timer Counter 1
AT91C_ID_TC2              EQU (18) ;- Timer Counter 2
AT91C_ID_PWMC             EQU (19) ;- Pulse Width Modulation Controller
AT91C_ID_TSADC            EQU (20) ;- Touch Screen Controller
AT91C_ID_HDMA             EQU (21) ;- HDMA
AT91C_ID_UDPHS            EQU (22) ;- USB Device HS
AT91C_ID_LCDC             EQU (23) ;- LCD Controller
AT91C_ID_AC97C            EQU (24) ;- AC97 Controller
AT91C_ID_IRQ0             EQU (31) ;- Advanced Interrupt Controller (IRQ0)
AT91C_ALL_INT             EQU (0x81FFFFFF) ;- ALL VALID INTERRUPTS

/* - ******************************************************************************/
/* -               BASE ADDRESS DEFINITIONS FOR AT91SAM9RL64*/
/* - ******************************************************************************/
AT91C_BASE_SYS            EQU (0xFFFFC000) ;- (SYS) Base Address
AT91C_BASE_EBI            EQU (0xFFFFE800) ;- (EBI) Base Address
AT91C_BASE_SDRAMC         EQU (0xFFFFEA00) ;- (SDRAMC) Base Address
AT91C_BASE_SMC            EQU (0xFFFFEC00) ;- (SMC) Base Address
AT91C_BASE_MATRIX         EQU (0xFFFFEE00) ;- (MATRIX) Base Address
AT91C_BASE_CCFG           EQU (0xFFFFEF10) ;- (CCFG) Base Address
AT91C_BASE_AIC            EQU (0xFFFFF000) ;- (AIC) Base Address
AT91C_BASE_PDC_DBGU       EQU (0xFFFFF300) ;- (PDC_DBGU) Base Address
AT91C_BASE_DBGU           EQU (0xFFFFF200) ;- (DBGU) Base Address
AT91C_BASE_PIOA           EQU (0xFFFFF400) ;- (PIOA) Base Address
AT91C_BASE_PIOB           EQU (0xFFFFF600) ;- (PIOB) Base Address
AT91C_BASE_PIOC           EQU (0xFFFFF800) ;- (PIOC) Base Address
AT91C_BASE_PIOD           EQU (0xFFFFFA00) ;- (PIOD) Base Address
AT91C_BASE_PMC            EQU (0xFFFFFC00) ;- (PMC) Base Address
AT91C_BASE_CKGR           EQU (0xFFFFFC1C) ;- (CKGR) Base Address
AT91C_BASE_RSTC           EQU (0xFFFFFD00) ;- (RSTC) Base Address
AT91C_BASE_SHDWC          EQU (0xFFFFFD10) ;- (SHDWC) Base Address
AT91C_BASE_RTTC           EQU (0xFFFFFD20) ;- (RTTC) Base Address
AT91C_BASE_PITC           EQU (0xFFFFFD30) ;- (PITC) Base Address
AT91C_BASE_WDTC           EQU (0xFFFFFD40) ;- (WDTC) Base Address
AT91C_BASE_RTC            EQU (0xFFFFFE00) ;- (RTC) Base Address
AT91C_BASE_TC0            EQU (0xFFFA0000) ;- (TC0) Base Address
AT91C_BASE_TC1            EQU (0xFFFA0040) ;- (TC1) Base Address
AT91C_BASE_TC2            EQU (0xFFFA0080) ;- (TC2) Base Address
AT91C_BASE_TCB0           EQU (0xFFFA0000) ;- (TCB0) Base Address
AT91C_BASE_TCB1           EQU (0xFFFA0040) ;- (TCB1) Base Address
AT91C_BASE_TCB2           EQU (0xFFFA0080) ;- (TCB2) Base Address
AT91C_BASE_PDC_MCI        EQU (0xFFFA4100) ;- (PDC_MCI) Base Address
AT91C_BASE_MCI            EQU (0xFFFA4000) ;- (MCI) Base Address
AT91C_BASE_PDC_TWI0       EQU (0xFFFA8100) ;- (PDC_TWI0) Base Address
AT91C_BASE_TWI0           EQU (0xFFFA8000) ;- (TWI0) Base Address
AT91C_BASE_TWI1           EQU (0xFFFAC000) ;- (TWI1) Base Address
AT91C_BASE_PDC_US0        EQU (0xFFFB0100) ;- (PDC_US0) Base Address
AT91C_BASE_US0            EQU (0xFFFB0000) ;- (US0) Base Address
AT91C_BASE_PDC_US1        EQU (0xFFFB4100) ;- (PDC_US1) Base Address
AT91C_BASE_US1            EQU (0xFFFB4000) ;- (US1) Base Address
AT91C_BASE_PDC_US2        EQU (0xFFFB8100) ;- (PDC_US2) Base Address
AT91C_BASE_US2            EQU (0xFFFB8000) ;- (US2) Base Address
AT91C_BASE_PDC_US3        EQU (0xFFFBC100) ;- (PDC_US3) Base Address
AT91C_BASE_US3            EQU (0xFFFBC000) ;- (US3) Base Address
AT91C_BASE_PDC_SSC0       EQU (0xFFFC0100) ;- (PDC_SSC0) Base Address
AT91C_BASE_SSC0           EQU (0xFFFC0000) ;- (SSC0) Base Address
AT91C_BASE_PDC_SSC1       EQU (0xFFFC4100) ;- (PDC_SSC1) Base Address
AT91C_BASE_SSC1           EQU (0xFFFC4000) ;- (SSC1) Base Address
AT91C_BASE_PWMC_CH0       EQU (0xFFFC8200) ;- (PWMC_CH0) Base Address
AT91C_BASE_PWMC_CH1       EQU (0xFFFC8220) ;- (PWMC_CH1) Base Address
AT91C_BASE_PWMC_CH2       EQU (0xFFFC8240) ;- (PWMC_CH2) Base Address
AT91C_BASE_PWMC_CH3       EQU (0xFFFC8260) ;- (PWMC_CH3) Base Address
AT91C_BASE_PWMC           EQU (0xFFFC8000) ;- (PWMC) Base Address
AT91C_BASE_PDC_SPI        EQU (0xFFFCC100) ;- (PDC_SPI) Base Address
AT91C_BASE_SPI            EQU (0xFFFCC000) ;- (SPI) Base Address
AT91C_BASE_PDC_TSADC      EQU (0xFFFD0100) ;- (PDC_TSADC) Base Address
AT91C_BASE_TSADC          EQU (0xFFFD0000) ;- (TSADC) Base Address
AT91C_BASE_UDPHS_EPTFIFO  EQU (0x00600000) ;- (UDPHS_EPTFIFO) Base Address
AT91C_BASE_UDPHS_EPT_0    EQU (0xFFFD4100) ;- (UDPHS_EPT_0) Base Address
AT91C_BASE_UDPHS_EPT_1    EQU (0xFFFD4120) ;- (UDPHS_EPT_1) Base Address
AT91C_BASE_UDPHS_EPT_2    EQU (0xFFFD4140) ;- (UDPHS_EPT_2) Base Address
AT91C_BASE_UDPHS_EPT_3    EQU (0xFFFD4160) ;- (UDPHS_EPT_3) Base Address
AT91C_BASE_UDPHS_EPT_4    EQU (0xFFFD4180) ;- (UDPHS_EPT_4) Base Address
AT91C_BASE_UDPHS_EPT_5    EQU (0xFFFD41A0) ;- (UDPHS_EPT_5) Base Address
AT91C_BASE_UDPHS_EPT_6    EQU (0xFFFD41C0) ;- (UDPHS_EPT_6) Base Address
AT91C_BASE_UDPHS_EPT_7    EQU (0xFFFD41E0) ;- (UDPHS_EPT_7) Base Address
AT91C_BASE_UDPHS_EPT_8    EQU (0xFFFD4200) ;- (UDPHS_EPT_8) Base Address
AT91C_BASE_UDPHS_EPT_9    EQU (0xFFFD4220) ;- (UDPHS_EPT_9) Base Address
AT91C_BASE_UDPHS_EPT_10   EQU (0xFFFD4240) ;- (UDPHS_EPT_10) Base Address
AT91C_BASE_UDPHS_EPT_11   EQU (0xFFFD4260) ;- (UDPHS_EPT_11) Base Address
AT91C_BASE_UDPHS_EPT_12   EQU (0xFFFD4280) ;- (UDPHS_EPT_12) Base Address
AT91C_BASE_UDPHS_EPT_13   EQU (0xFFFD42A0) ;- (UDPHS_EPT_13) Base Address
AT91C_BASE_UDPHS_EPT_14   EQU (0xFFFD42C0) ;- (UDPHS_EPT_14) Base Address
AT91C_BASE_UDPHS_EPT_15   EQU (0xFFFD42E0) ;- (UDPHS_EPT_15) Base Address
AT91C_BASE_UDPHS_DMA_1    EQU (0xFFFD4310) ;- (UDPHS_DMA_1) Base Address
AT91C_BASE_UDPHS_DMA_2    EQU (0xFFFD4320) ;- (UDPHS_DMA_2) Base Address
AT91C_BASE_UDPHS_DMA_3    EQU (0xFFFD4330) ;- (UDPHS_DMA_3) Base Address
AT91C_BASE_UDPHS_DMA_4    EQU (0xFFFD4340) ;- (UDPHS_DMA_4) Base Address
AT91C_BASE_UDPHS_DMA_5    EQU (0xFFFD4350) ;- (UDPHS_DMA_5) Base Address
AT91C_BASE_UDPHS_DMA_6    EQU (0xFFFD4360) ;- (UDPHS_DMA_6) Base Address
AT91C_BASE_UDPHS_DMA_7    EQU (0xFFFD4370) ;- (UDPHS_DMA_7) Base Address
AT91C_BASE_UDPHS          EQU (0xFFFD4000) ;- (UDPHS) Base Address
AT91C_BASE_PDC_AC97C      EQU (0xFFFD8100) ;- (PDC_AC97C) Base Address
AT91C_BASE_AC97C          EQU (0xFFFD8000) ;- (AC97C) Base Address
AT91C_BASE_LCDC           EQU (0x00500000) ;- (LCDC) Base Address
AT91C_BASE_LCDC_16B_TFT   EQU (0x00500000) ;- (LCDC_16B_TFT) Base Address
AT91C_BASE_HDMA_CH_0      EQU (0xFFFFE63C) ;- (HDMA_CH_0) Base Address
AT91C_BASE_HDMA_CH_1      EQU (0xFFFFE664) ;- (HDMA_CH_1) Base Address
AT91C_BASE_HDMA           EQU (0xFFFFE600) ;- (HDMA) Base Address
AT91C_BASE_HECC           EQU (0xFFFFE800) ;- (HECC) Base Address

/* - ******************************************************************************/
/* -               MEMORY MAPPING DEFINITIONS FOR AT91SAM9RL64*/
/* - ******************************************************************************/
/* - ITCM*/
AT91C_ITCM                EQU (0x00100000) ;- Maximum ITCM Area base address
AT91C_ITCM_SIZE           EQU (0x00010000) ;- Maximum ITCM Area size in byte (64 Kbytes)
/* - DTCM*/
AT91C_DTCM                EQU (0x00200000) ;- Maximum DTCM Area base address
AT91C_DTCM_SIZE           EQU (0x00010000) ;- Maximum DTCM Area size in byte (64 Kbytes)
/* - IRAM*/
AT91C_IRAM                EQU (0x00300000) ;- Maximum Internal SRAM base address
AT91C_IRAM_SIZE           EQU (0x00010000) ;- Maximum Internal SRAM size in byte (64 Kbytes)
/* - IRAM_MIN*/
AT91C_IRAM_MIN            EQU (0x00300000) ;- Minimum Internal RAM base address
AT91C_IRAM_MIN_SIZE       EQU (0x00004000) ;- Minimum Internal RAM size in byte (16 Kbytes)
/* - IROM*/
AT91C_IROM                EQU (0x00400000) ;- Internal ROM base address
AT91C_IROM_SIZE           EQU (0x00008000) ;- Internal ROM size in byte (32 Kbytes)
/* - EBI_CS0*/
AT91C_EBI_CS0             EQU (0x10000000) ;- EBI Chip Select 0 base address
AT91C_EBI_CS0_SIZE        EQU (0x10000000) ;- EBI Chip Select 0 size in byte (262144 Kbytes)
/* - EBI_CS1*/
AT91C_EBI_CS1             EQU (0x20000000) ;- EBI Chip Select 1 base address
AT91C_EBI_CS1_SIZE        EQU (0x10000000) ;- EBI Chip Select 1 size in byte (262144 Kbytes)
/* - EBI_SDRAM*/
AT91C_EBI_SDRAM           EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_SIZE      EQU (0x10000000) ;- SDRAM on EBI Chip Select 1 size in byte (262144 Kbytes)
/* - EBI_SDRAM_16BIT*/
AT91C_EBI_SDRAM_16BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_16BIT_SIZE EQU (0x02000000) ;- SDRAM on EBI Chip Select 1 size in byte (32768 Kbytes)
/* - EBI_SDRAM_32BIT*/
AT91C_EBI_SDRAM_32BIT     EQU (0x20000000) ;- SDRAM on EBI Chip Select 1 base address
AT91C_EBI_SDRAM_32BIT_SIZE EQU (0x04000000) ;- SDRAM on EBI Chip Select 1 size in byte (65536 Kbytes)
/* - EBI_CS2*/
AT91C_EBI_CS2             EQU (0x30000000) ;- EBI Chip Select 2 base address
AT91C_EBI_CS2_SIZE        EQU (0x10000000) ;- EBI Chip Select 2 size in byte (262144 Kbytes)
/* - EBI_CS3*/
AT91C_EBI_CS3             EQU (0x40000000) ;- EBI Chip Select 3 base address
AT91C_EBI_CS3_SIZE        EQU (0x10000000) ;- EBI Chip Select 3 size in byte (262144 Kbytes)
/* - EBI_SM*/
AT91C_EBI_SM              EQU (0x40000000) ;- NANDFLASH on EBI Chip Select 3 base address
AT91C_EBI_SM_SIZE         EQU (0x10000000) ;- NANDFLASH on EBI Chip Select 3 size in byte (262144 Kbytes)
/* - EBI_CS4*/
AT91C_EBI_CS4             EQU (0x50000000) ;- EBI Chip Select 4 base address
AT91C_EBI_CS4_SIZE        EQU (0x10000000) ;- EBI Chip Select 4 size in byte (262144 Kbytes)
/* - EBI_CF0*/
AT91C_EBI_CF0             EQU (0x50000000) ;- CompactFlash 0 on EBI Chip Select 4 base address
AT91C_EBI_CF0_SIZE        EQU (0x10000000) ;- CompactFlash 0 on EBI Chip Select 4 size in byte (262144 Kbytes)
/* - EBI_CS5*/
AT91C_EBI_CS5             EQU (0x60000000) ;- EBI Chip Select 5 base address
AT91C_EBI_CS5_SIZE        EQU (0x10000000) ;- EBI Chip Select 5 size in byte (262144 Kbytes)
/* - EBI_CF1*/
AT91C_EBI_CF1             EQU (0x60000000) ;- CompactFlash 1 on EBIChip Select 5 base address
AT91C_EBI_CF1_SIZE        EQU (0x10000000) ;- CompactFlash 1 on EBIChip Select 5 size in byte (262144 Kbytes)
#endif /* __IAR_SYSTEMS_ASM__ */


#endif /* AT91SAM9RL64_H */
