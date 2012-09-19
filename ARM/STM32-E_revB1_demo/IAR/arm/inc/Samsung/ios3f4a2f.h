/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Samsung S3F4A2F
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2005
 **
 **    $Revision: 30256 $
 **
 ***************************************************************************/

#ifndef __S3F4A2F_H
#define __S3F4A2F_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S3F4A2F SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

/* ADC Enable Clock Register 
   ADC Disable Clock Register */
typedef struct {
  __REG32           : 1;
  __REG32 ADC       : 1;
  __REG32           :29;
  __REG32 DBGEN     : 1;
} __adc_ecr_bits;

/* ADC Power Management Status Register */
typedef struct {
  __REG32           : 1;
  __REG32 ADC       : 1;
  __REG32           : 2;
  __REG32 IPICODE   :26;
  __REG32           : 1;
  __REG32 DBGEN     : 1;
} __adc_pmsr_bits;

/* ADC Control Register */
typedef struct {
  __REG32 SWRST     : 1;
  __REG32 ADCEN     : 1;
  __REG32 ADCDIS    : 1;
  __REG32 START     : 1;
  __REG32 STOP      : 1;
  __REG32           :27;
} __adc_cr_bits;

/* ADC Mode Register */
typedef struct {
  __REG32 PRLVAL    : 5;
  __REG32 IES       : 1;
  __REG32           :10;
  __REG32 NBRCH     : 4;
  __REG32 CONTCV    : 1;
  __REG32           :11;
} __adc_mr_bits;

/* ADC Clear Status Register */
typedef struct {
  __REG32           : 2;
  __REG32 OVR       : 1;
  __REG32           :29;
} __adc_csr_bits;

/* ADC Status Register */
typedef struct {
  __REG32 EOC       : 1;
  __REG32 READY     : 1;
  __REG32 OVR       : 1;
  __REG32           : 5;
  __REG32 ADCENS    : 1;
  __REG32 CTCVS     : 1;
  __REG32           :22;
} __adc_sr_bits;

/* ADC Interrupt Enable Register 
   ADC Interrupt Disable Register 
   ADC Interrupt Mask Register */
typedef struct {
  __REG32 EOC       : 1;
  __REG32 READY     : 1;
  __REG32 OVR       : 1;
  __REG32           :29;
} __adc_ier_bits;

/* ADC Conversion Mode Register 0 */
typedef struct {
  __REG32 CV1       : 4;
  __REG32 CV2       : 4;
  __REG32 CV3       : 4;
  __REG32 CV4       : 4;
  __REG32 CV5       : 4;
  __REG32 CV6       : 4;
  __REG32 CV7       : 4;
  __REG32 CV8       : 4;
} __adc_cmr0_bits;

/* ADC Conversion Mode Register 1 */
typedef struct {
  __REG32 CV9       : 4;
  __REG32 CV10      : 4;
  __REG32 CV11      : 4;
  __REG32 CV12      : 4;
  __REG32 CV13      : 4;
  __REG32 CV14      : 4;
  __REG32 CV15      : 4;
  __REG32 CV16      : 4;
} __adc_cmr1_bits;

/* ADC Convert Data Register */
typedef struct {
  __REG32 DATA      :10;
  __REG32           :22;
} __adc_dr_bits;

/* CLKMNGR Oscillator and PLL Status Register */
typedef struct {
  __REG32 PLLST     : 1;
  __REG32 OSCST     : 1;
  __REG32 LFOST     : 1;
  __REG32 LFUSED    : 1;
  __REG32           :28;
} __cm_str_bits;

/* CLKMNGR Wait for Interrupt */
typedef struct {
  __REG32           : 5;
  __REG32 PCLK1     : 1;
  __REG32           :10;
  __REG32 WFIKEY    :16;
} __cm_wfir_bits;

/* CLKMNGR PLL Stabilization Time */
typedef struct {
  __REG32 PST       :11;
  __REG32           : 5;
  __REG32 PLLKEY    :16;
} __cm_pstr_bits;

/* CLKMNGR PLL Divider Parameters */
typedef struct {
  __REG32 PMUL      : 8;
  __REG32 PLL_POST  : 2;
  __REG32 PLL_PRE   : 6;
  __REG32 PDPKEY    :16;
} __cm_pdpr_bits;

/* CLKMNGR Oscillator Stabilization Time */
typedef struct {
  __REG32 OST       :16;
  __REG32 OSTKEY    :16;
} __cm_ostr_bits;

/* CLKMNGR Master Clock Divider */
typedef struct {
  __REG32 PCLK1DIV  : 3;
  __REG32           :29;
} __cm_divbr_bits;

/* CLKMNGR System Clock Selection */
typedef struct {
  __REG32 CMCLK_SEL : 2;
  __REG32           :14;
  __REG32 SELKEY    :16;
} __cm_selr_bits;

/* CLKMNGR Reset Status */
typedef struct {
  __REG32 WD        : 1;
  __REG32 CM        : 1;
  __REG32 LVD       : 1;
  __REG32           :29;
} __cm_rsr_bits;

/* CLKMNGR Master Oscillator Clock Divider */
typedef struct {
  __REG32 MDIV      : 9;
  __REG32           : 1;
  __REG32 CDIV      : 3;
  __REG32 LDIV      : 3;
  __REG32 MDIVKEY   :16;
} __cm_mdivr_bits;

/* CLKMNGR Low Frequency Oscillator Control */
typedef struct {
  __REG32 LFOSCEN   : 1;
  __REG32 LFSEL     : 1;
  __REG32           : 6;
  __REG32 LF_ST     : 8;
  __REG32 LFOSCKEY  :16;
} __cm_lfoscr_bits;

/* CLKMNGR Control Register */
typedef struct {
  __REG32 HALTMODE  : 1;
  __REG32           : 3;
  __REG32 STOPMODE  : 1;
  __REG32 IDLEMODE  : 1;
  __REG32           :10;
  __REG32 CRKEY     :16;
} __cm_cr_bits;

/* CLKMNGR Mode Register */
typedef struct {
  __REG32 CM_EN     : 1;
  __REG32           :15;
  __REG32 MRKEY     :16;
} __cm_mr_bits;

/* CM Clear Status Register
   CM Status Register 
   CM Interrupt Enable Register
   CM Interrupt Disable Register
   CM Interrupt Mask Register */
typedef struct {
  __REG32 STABLE    : 1;
  __REG32           :31;
} __cm_csr_bits;

/* CAN Enable Clock Register 
   CAN Disable Clock Register
   CAN Power Management Status Register */
typedef struct {
  __REG32           : 1;
  __REG32 CAN       : 1;
  __REG32           :29;
  __REG32 DBGEN     : 1;
} __can_ecr_bits;

/* CAN Control Register */
typedef struct {
  __REG32 SWRST     : 1;
  __REG32 CANEN     : 1;
  __REG32 CANDIS    : 1;
  __REG32 CCEN      : 1;
  __REG32 CCDIS     : 1;
  __REG32           : 3;
  __REG32 RQBTX     : 1;
  __REG32 ABBTX     : 1;
  __REG32 STSR      : 1;
  __REG32           :21;
} __can_cr_bits;

/* CAN Mode Register */
typedef struct {
  __REG32 BD        :10;
  __REG32           : 2;
  __REG32 SJW       : 2;
  __REG32 AR        : 1;
  __REG32           : 1;
  __REG32 PHSEG1    : 4;
  __REG32 RHSEG2    : 3;
  __REG32           : 9;
} __can_mr_bits;

/* CAN Clear Status Register */
typedef struct {
  __REG32           : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32           :16;
} __can_csr_bits;

/* CAN Status Register */
typedef struct {
  __REG32 ISS       : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32 CANENS    : 1;
  __REG32 ERWARN    : 1;
  __REG32 ERPASS    : 1;
  __REG32 BUSOFF    : 1;
  __REG32 BUSY0     : 1;
  __REG32 BUSY1     : 1;
  __REG32 RS        : 1;
  __REG32 TS        : 1;
  __REG32 CCENS     : 1;
  __REG32 BTXPD     : 1;
  __REG32           : 6;
} __can_sr_bits;

/* CAN Interrupt Enable Register
   CAN Interrupt Disable Register
   CAN Interrupt Mask Register */
typedef struct {
  __REG32           : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32           :16;
} __can_ier_bits;

/* CAN Interrupt Source Status Register
   CAN Source Interrupt Enable Register
   CAN Source Interrupt Disable Register
   CAN Source Interrupt Mask Register
   CAN Transmission Request Register
   CAN New Data Register
   CAN Message Valid Register */
typedef struct {
  __REG32 CH1       : 1;
  __REG32 CH2       : 1;
  __REG32 CH3       : 1;
  __REG32 CH4       : 1;
  __REG32 CH5       : 1;
  __REG32 CH6       : 1;
  __REG32 CH7       : 1;
  __REG32 CH8       : 1;
  __REG32 CH9       : 1;
  __REG32 CH10      : 1;
  __REG32 CH11      : 1;
  __REG32 CH12      : 1;
  __REG32 CH13      : 1;
  __REG32 CH14      : 1;
  __REG32 CH15      : 1;
  __REG32 CH16      : 1;
  __REG32 CH17      : 1;
  __REG32 CH18      : 1;
  __REG32 CH19      : 1;
  __REG32 CH20      : 1;
  __REG32 CH21      : 1;
  __REG32 CH22      : 1;
  __REG32 CH23      : 1;
  __REG32 CH24      : 1;
  __REG32 CH25      : 1;
  __REG32 CH26      : 1;
  __REG32 CH27      : 1;
  __REG32 CH28      : 1;
  __REG32 CH29      : 1;
  __REG32 CH30      : 1;
  __REG32 CH31      : 1;
  __REG32 CH32      : 1;
} __can_issr_bits;

/* CAN Highest Priority Interrupt Register */
typedef struct {
  __REG32 INTID     :16;
  __REG32           :16;
} __can_hpir_bits;

/* CAN Error Counter Register */
typedef struct {
  __REG32 REC       : 7;
  __REG32 REP       : 1;
  __REG32 TEC       : 8;
  __REG32           :16;
} __can_ercr_bits;

/* CAN Interface X Transfer Management Register */
typedef struct {
  __REG32 NUMBER    : 6;
  __REG32           : 1;
  __REG32 WR        : 1;
  __REG32 ADAR      : 1;
  __REG32 ADBR      : 1;
  __REG32 AMSKR     : 1;
  __REG32 AIR       : 1;
  __REG32 AMCR      : 1;
  __REG32           : 1;
  __REG32 TRND      : 1;
  __REG32 CLRIT     : 1;
  __REG32           :16;
} __can_tmr_bits;

/* CAN Interface X Data A/B Register */
typedef struct {
  __REG32 DATA0     : 8;
  __REG32 DATA1     : 8;
  __REG32 DATA2     : 8;
  __REG32 DATA3     : 8;
} __can_dar_bits;

/* CAN Interface X Mask Register */
typedef struct {
  __REG32 EXTMASK   :18;
  __REG32 BASEMASK  :11;
  __REG32           : 1;
  __REG32 MMDIR     : 1;
  __REG32 MXTD      : 1;
} __can_mskr_bits;

/* CAN Interface X Identifier Register */
typedef struct {
  __REG32 EXTID     :18;
  __REG32 BASEID    :11;
  __REG32 MDIR      : 1;
  __REG32 XTD       : 1;
  __REG32 MSGVAL    : 1;
} __can_ir_bits;

/* CAN Interface X Message Control Register */
typedef struct {
  __REG32 DLC       : 4;
  __REG32           : 3;
  __REG32 OVERWRITE : 1;
  __REG32 TXRQST    : 1;
  __REG32 RMTEN     : 1;
  __REG32 RXIE      : 1;
  __REG32 TXIE      : 1;
  __REG32 UMASK     : 1;
  __REG32 ITPND     : 1;
  __REG32 MSGLST    : 1;
  __REG32 NEWDAT    : 1;
  __REG32           :16;
} __can_mcr_bits;

/* CAN Test Register */
typedef struct {
  __REG32 BASIC     : 1;
  __REG32 SILENT    : 1;
  __REG32 LBACK     : 1;
  __REG32 TX        : 2;
  __REG32 TXOPD     : 1;
  __REG32 RX        : 1;
  __REG32           : 9;
  __REG32 TSTKEY    :16;
} __can_tstr_bits;

/* PIO Pin Enable Register
   PIO Pin Disable Register
   PIO Pin Status Register
   PIO Output Enable Register
   PIO Output Disable Register
   PIO Output Status Register
   PIO Write Output Data Register
   PIO Set Output Data Register
   PIO Clear Output Data Register
   PIO Output Data Status Register
   PIO Pin Data Status Register
   PIO Multi-Driver Enable Register
   PIO Multi-Driver Disable Register
   PIO Multi-Driver Status Register
   PIO Clear Status Register
   PIO Status Register
   PIO Interrupt Enable Register
   PIO Interrupt Disable Register
   PIO Interrupt Mask Register */
typedef struct {
  __REG32 P0            : 1;
  __REG32 P1            : 1;
  __REG32 P2            : 1;
  __REG32 P3            : 1;
  __REG32 P4            : 1;
  __REG32 P5            : 1;
  __REG32 P6            : 1;
  __REG32 P7            : 1;
  __REG32 P8            : 1;
  __REG32 P9            : 1;
  __REG32 P10           : 1;
  __REG32 P11           : 1;
  __REG32 P12           : 1;
  __REG32 P13           : 1;
  __REG32 P14           : 1;
  __REG32 P15           : 1;
  __REG32 P16           : 1;
  __REG32 P17           : 1;
  __REG32 P18           : 1;
  __REG32 P19           : 1;
  __REG32 P20           : 1;
  __REG32 P21           : 1;
  __REG32 P22           : 1;
  __REG32 P23           : 1;
  __REG32 P24           : 1;
  __REG32 P25           : 1;
  __REG32 P26           : 1;
  __REG32 P27           : 1;
  __REG32 P28           : 1;
  __REG32 P29           : 1;
  __REG32 P30           : 1;
  __REG32 P31           : 1;
} __pio_per_bits;

/* PIO Enable Clock Register
   PIO Disable Clock Register */
typedef struct {
  __REG32 PIO           : 1;
  __REG32               :31;
} __pio_ecr_bits;

/* PIO Power Management Status Register */
typedef struct {
  __REG32 PIO           : 1;
  __REG32               : 3;
  __REG32 IPIDCODE      :26;
  __REG32               : 2;
} __pio_pmsr_bits;

/* PIO Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32               :31;
} __pio_cr_bits;

/* GPT Enable Clock Register
   GPT Disable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 GPT           : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __gpt_ecr_bits;

/* GPT Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 GPT           : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __gpt_pmsr_bits;

/* GPT Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 CLKEN         : 1;
  __REG32 CLKDIS        : 1;
  __REG32 SWTRG         : 1;
  __REG32               :28;
} __gpt_cr_bits;

/* GPT Mode Register */
typedef union {
  /* GPT_MR_CMx*/
  struct {
  __REG32 CCLKS         : 3;
  __REG32 CCLKI         : 1;
  __REG32 CBURST        : 2;
  __REG32 LDBSTOP       : 1;
  __REG32 LDBDIS        : 1;
  __REG32 CETRGEDG      : 2;
  __REG32 ABETRG        : 1;
  __REG32               : 3;
  __REG32 CCPCTRG       : 1;
  __REG32 CWAVE         : 1;
  __REG32 LDRA          : 2;
  __REG32 LDRB          : 2;
  __REG32               :12;
  };
  /* GPT_MR_WMx*/
  struct {
  __REG32 WCLKS         : 3;
  __REG32 WCLKI         : 1;
  __REG32 WBURST        : 2;
  __REG32 CPCSTOP       : 1;
  __REG32 CPCDIS        : 1;
  __REG32 WEEVTEDG      : 2;
  __REG32 EEVT          : 2;
  __REG32 ENETRG        : 1;
  __REG32               : 1;
  __REG32 WCPCTRG       : 1;
  __REG32 WWAVE         : 1;
  __REG32 ACPA          : 2;
  __REG32 ACPC          : 2;
  __REG32 AEEVT         : 2;
  __REG32 ASWTRG        : 2;
  __REG32 BCPB          : 2;
  __REG32 BCPC          : 2;
  __REG32 BEEVT         : 2;
  __REG32 BSWTRG        : 2;
  };
} __gpt_mr_bits;

/* GPT Clear Status Register */
typedef struct {
  __REG32 COVFS         : 1;
  __REG32 LOVRS         : 1;
  __REG32 CPAS          : 1;
  __REG32 CPBS          : 1;
  __REG32 CPCS          : 1;
  __REG32 LDRAS         : 1;
  __REG32 LDRBS         : 1;
  __REG32 ETRGS         : 1;
  __REG32               :24;
} __gpt_csr_bits;

/* GPT Status Register */
typedef struct {
  __REG32 COVFS         : 1;
  __REG32 LOVRS         : 1;
  __REG32 CPAS          : 1;
  __REG32 CPBS          : 1;
  __REG32 CPCS          : 1;
  __REG32 LDRAS         : 1;
  __REG32 LDRBS         : 1;
  __REG32 ETRGS         : 1;
  __REG32 CLKSTA        : 1;
  __REG32 MTIOA         : 1;
  __REG32 MTIOB         : 1;
  __REG32               : 5;
  __REG32 TIOBS         : 1;
  __REG32 TIOAS         : 1;
  __REG32 TCLKS         : 1;
  __REG32               :13;
} __gpt_sr_bits;

/* GPT Interrupt Enable Register
   GPT Interrupt Disable Register
   GPT Interrupt Mask Register */
typedef struct {
  __REG32 COVFS         : 1;
  __REG32 LOVRS         : 1;
  __REG32 CPAS          : 1;
  __REG32 CPBS          : 1;
  __REG32 CPCS          : 1;
  __REG32 LDRAS         : 1;
  __REG32 LDRBS         : 1;
  __REG32 ETRGS         : 1;
  __REG32               : 8;
  __REG32 TIOBS         : 1;
  __REG32 TIOAS         : 1;
  __REG32 TCLKS         : 1;
  __REG32               :13;
} __gpt_ier_bits;

/* GPT Counter Value */
typedef struct {
  __REG32 CV            :16;
  __REG32               :16;
} __gpt_cv_bits;

/* GPT Register A */
typedef struct {
  __REG32 RA            :16;
  __REG32               :16;
} __gpt_ra_bits;

/* GPT Register B */
typedef struct {
  __REG32 RB            :16;
  __REG32               :16;
} __gpt_rb_bits;

/* GPT Register C */
typedef struct {
  __REG32 RC            :16;
  __REG32               :16;
} __gpt_rc_bits;

/* GPT Block Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 TCSYNC        : 1;
  __REG32               :30;
} __gpt_bcr_bits;

/* GPT Block Mode Register */
typedef struct {
  __REG32 TC0XC0S       : 2;
  __REG32 TC1XC1S       : 2;
  __REG32 TC2XC2S       : 2;
  __REG32               :26;
} __gpt_bmr_bits;

/* GIC Source Mode */
typedef struct {
  __REG32 PRIOR         : 3;
  __REG32 SDI           : 1;
  __REG32 SRCTYP        : 2;
  __REG32               :26;
} __gic_smr_bits;

/* GIC Interrupt Status */
typedef struct {
  __REG32 IRQID         : 6;
  __REG32               :26;
} __gic_isr_bits;

/* GIC Interrupt Pending 0
   GIC Interrupt Mask 0
   GIC Interrupt Enable Command 0
   GIC Interrupt Disable Command 0
   GIC Interrupt Clear Command 0
   GIC Interrupt Set Command 0 */
typedef struct {
  __REG32               : 1;
  __REG32 STOP_MODE     : 1;
  __REG32 LVD           : 1;
  __REG32 IFC           : 1;
  __REG32 PWM0          : 1;
  __REG32 ADC0          : 1;
  __REG32 SPI0          : 1;
  __REG32 WD            : 1;
  __REG32 CAN0          : 1;
  __REG32 GPT0CH0       : 1;
  __REG32 ST0           : 1;
  __REG32 ST1           : 1;
  __REG32 UART0         : 1;
  __REG32 I2C0          : 1;
  __REG32               : 6;
  __REG32 USART0        : 1;
  __REG32               : 1;
  __REG32 SWIRQ1        : 1;
  __REG32               : 1;
  __REG32 SPI1          : 1;
  __REG32 STT           : 1;
  __REG32 GPT0CH1       : 1;
  __REG32 GPT0CH2       : 1;
  __REG32 LDMA          : 1;
  __REG32 GPIO0         : 1;
  __REG32 GPIO1         : 1;
  __REG32               : 1;
} __gic_ipr0_bits;

/* GIC Interrupt Pending 1
   GIC Interrupt Mask 1
   GIC Interrupt Enable Command 1
   GIC Interrupt Disable Command 1
   GIC Interrupt Clear Command 1
   GIC Interrupt Set Command 1 */
typedef struct {
  __REG32               : 1;
  __REG32 PWM1          : 1;
  __REG32 PWM2          : 1;
  __REG32 PWM3          : 1;
  __REG32               : 3;
  __REG32 SWIRQ2        : 1;
  __REG32 IRQ0          : 1;
  __REG32 IRQ1          : 1;
  __REG32 IRQ2          : 1;
  __REG32 IRQ3          : 1;
  __REG32 IRQ4          : 1;
  __REG32 IRQ5          : 1;
  __REG32 IRQ6          : 1;
  __REG32 IRQ7          : 1;
  __REG32 IRQ8          : 1;
  __REG32 IRQ9          : 1;
  __REG32               : 9;
  __REG32 STABLE        : 1;
  __REG32 SWIRQ3        : 1;
  __REG32 SWIRQ4        : 1;
  __REG32 SWIRQ5        : 1;
  __REG32 SWIRQ6        : 1;
} __gic_ipr1_bits;

/* GIC Core Interrupt Status */
typedef struct {
  __REG32 NFIQ          : 1;
  __REG32 NIRQ          : 1;
  __REG32               :30;
} __gic_cisr_bits;

/* IO Configuration Mode Register 0 */
typedef struct {
  __REG32 PIO0_0_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO0_1_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO0_2_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO0_2_F3EN   : 1;
  __REG32 PIO0_3_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO0_3_F3EN   : 1;
  __REG32 PIO0_4_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO0_5_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO0_6_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO0_7_PUENB  : 1;
  __REG32               : 3;
} __ioconf_mr0_bits;

/* IO Configuration Mode Register 1 */
typedef struct {
  __REG32 PIO0_8_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO0_8_F3EN   : 1;
  __REG32 PIO0_9_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO0_9_F3EN   : 1;
  __REG32 PIO0_10_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_10_F3EN  : 1;
  __REG32 PIO0_11_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_11_F3EN  : 1;
  __REG32 PIO0_12_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_12_F3EN  : 1;
  __REG32 PIO0_13_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_13_F3EN  : 1;
  __REG32 PIO0_14_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_14_F3EN  : 1;
  __REG32 PIO0_15_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_15_F3EN  : 1;
} __ioconf_mr1_bits;

/* IO Configuration Mode Register 2 */
typedef struct {
  __REG32 PIO0_16_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_16_F3EN  : 1;
  __REG32 PIO0_17_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_17_F3EN  : 1;
  __REG32 PIO0_18_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_18_F3EN  : 1;
  __REG32 PIO0_19_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_19_F3EN  : 1;
  __REG32 PIO0_20_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_20_F3EN  : 1;
  __REG32 PIO0_21_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_21_F3EN  : 1;
  __REG32 PIO0_22_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO0_22_F3EN  : 1;
  __REG32               : 4;
} __ioconf_mr2_bits;

/* IO Configuration Mode Register 3 */
typedef struct {
  __REG32 PIO1_0_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_0_F3EN   : 1;
  __REG32 PIO1_1_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_1_F3EN   : 1;
  __REG32 PIO1_2_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_2_F3EN   : 1;
  __REG32 PIO1_3_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_3_F3EN   : 1;
  __REG32 PIO1_4_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_4_F3EN   : 1;
  __REG32 PIO1_5_PUENB  : 1;
  __REG32               : 2;
  __REG32 PIO1_5_F3EN   : 1;
  __REG32 PIO1_6_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO1_7_PUENB  : 1;
  __REG32               : 3;
} __ioconf_mr3_bits;

/* IO Configuration Mode Register 4 */
typedef struct {
  __REG32 PIO1_8_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO1_9_PUENB  : 1;
  __REG32               : 3;
  __REG32 PIO1_10_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO1_10_F3EN  : 1;
  __REG32 PIO1_11_PUENB : 1;
  __REG32               : 2;
  __REG32 PIO1_11_F3EN  : 1;
  __REG32               :16;
} __ioconf_mr4_bits;

/* IO Configuration Mode Register 5 */
typedef struct {
  __REG32               : 3;
  __REG32 PIO1_12_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_13_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_14_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_15_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_16_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_17_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_18_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_19_F3EN  : 1;
} __ioconf_mr5_bits;

/* IO Configuration Mode Register 6 */
typedef struct {
  __REG32               : 3;
  __REG32 PIO1_20_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_21_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_22_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_23_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_24_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_25_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_26_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_27_F3EN  : 1;
} __ioconf_mr6_bits;

/* IO Configuration Mode Register 7 */
typedef struct {
  __REG32               : 3;
  __REG32 PIO1_28_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_29_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_30_F3EN  : 1;
  __REG32               : 3;
  __REG32 PIO1_31_F3EN  : 1;
  __REG32               :16;
} __ioconf_mr7_bits;

/* IO Configuration Mode Register 15 */
typedef struct {
  __REG32 STOP_WU0      : 5;
  __REG32               : 3;
  __REG32 STOP_WU1      : 5;
  __REG32               : 3;
  __REG32 STOP_WU2      : 5;
  __REG32               : 3;
  __REG32 STOP_WU3      : 5;
  __REG32               : 3;
} __ioconf_mr15_bits;

/* IO Configuration Mode Register 16 */
typedef struct {
  __REG32 LVD_RST_EN    : 1;
  __REG32 LVD_INT_EN    : 1;
  __REG32               :30;
} __ioconf_mr16_bits;

/* I2C Enable Clock Register
   I2C Disable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 I2C           : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __i2c_ecr_bits;

/* I2C Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 I2C           : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __i2c_pmsr_bits;

/* I2C Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 AA            : 1;
  __REG32 STO           : 1;
  __REG32 STA           : 1;
  __REG32 SI            : 1;
  __REG32               : 3;
  __REG32 ENA           : 1;
  __REG32               :23;
} __i2c_cr_bits;

/* I2C Mode Register */
typedef struct {
  __REG32 PRV           :12;
  __REG32 FAST          : 1;
  __REG32               :19;
} __i2c_mr_bits;

/* I2C Status Register */
typedef struct {
  __REG32               : 3;
  __REG32 SR            : 5;
  __REG32               :24;
} __i2c_sr_bits;

/* I2C Status Register
   I2C Interrupt Disable Register
   I2C Interrupt Mask Register */
typedef struct {
  __REG32               : 4;
  __REG32 SI            : 1;
  __REG32               :27;
} __i2c_ier_bits;

/* I2C Serial Data Register */
typedef struct {
  __REG32 DAT           : 8;
  __REG32               :24;
} __i2c_dat_bits;

/* I2C Serial Slave Address Register */
typedef struct {
  __REG32 GC            : 1;
  __REG32 ADR           : 7;
  __REG32               :24;
} __i2c_adr_bits;

/* I2C Hold/Setup Delay Register */
typedef struct {
  __REG32 DL            : 8;
  __REG32               :24;
} __i2c_thold_bits;

/* IFC Power Management Status Register */
typedef struct {
  __REG32               : 4;
  __REG32 IPICODE       :26;
  __REG32               : 2;
} __ifc_pmsr_bits;

/* IFC Control Register */
typedef struct {
  __REG32               : 1;
  __REG32 SE            : 1;
  __REG32 CE            : 1;
  __REG32               : 5;
  __REG32 CRKEY         : 8;
  __REG32               :10;
  __REG32 SECTOR        : 5;
  __REG32               : 1;
} __ifc_cr_bits;

/* IFC Mode Register */
typedef struct {
  __REG32               : 4;
  __REG32 STANDEN       : 1;
  __REG32               : 2;
  __REG32 WPR           : 1;
  __REG32 MRKEY         : 8;
  __REG32               : 8;
  __REG32 BA            : 8;
} __ifc_mr_bits;

/* IFC Clear Status
   IFC Interrupt Enable
   IFC Interrupt Disable
   IFC Interrupt Mask */
typedef struct {
  __REG32 ENDWR         : 1;
  __REG32 ENDERASE      : 1;
  __REG32 DACCESS       : 1;
  __REG32               :29;
} __ifc_csr_bits;

/* IFC Status Register */
typedef struct {
  __REG32 ENDWR         : 1;
  __REG32 ENDERASE      : 1;
  __REG32 DACCESS       : 1;
  __REG32               : 5;
  __REG32 BUSY          : 1;
  __REG32               :23;
} __ifc_sr_bits;

/* IRC Mode Register */
typedef struct {
  __REG32               :20;
  __REG32 BA            :12;
} __irc_mr_bits;


/* LDMA Enable Clock Register
   LDMA Disable Clock Register */
typedef struct {
  __REG32               :31;
  __REG32 DBGEN         : 1;
} __ldma_ecr_bits;

/* LDMA Power Management Status Register */
typedef struct {
  __REG32               : 4;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __ldma_pmsr_bits;

/* LDMA Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32               :31;
} __ldma_cr_bits;

/* LDMA Status Register
   LDMA Interrupt Enable Register
   LDMA Interrupt Disable Register
   LDMA Interrupt Mask Register */
typedef struct {
  __REG32 CH0_IT        : 1;
  __REG32 CH1_IT        : 1;
  __REG32 CH2_IT        : 1;
  __REG32 CH3_IT        : 1;
  __REG32 CH4_IT        : 1;
  __REG32 CH5_IT        : 1;
  __REG32 CH6_IT        : 1;
  __REG32 CH7_IT        : 1;
  __REG32               :24;
} __ldma_sr_bits;

/* LDMA Channel x Control Register */
typedef struct {
  __REG32               : 1;
  __REG32 LCHEN         : 1;
  __REG32 LCHDIS        : 1;
  __REG32               :29;
} __ldma_crx_bits;

/* LDMA Channel x Mode Register */
typedef struct {
  __REG32 SRC           : 1;
  __REG32 DEST          : 1;
  __REG32 SRC_INCR      : 1;
  __REG32 DEST_INCR     : 1;
  __REG32 TRIG          : 1;
  __REG32 LDMA_SIZE     : 2;
  __REG32               : 1;
  __REG32 CHREADY       : 5;
  __REG32               :19;
} __ldma_mrx_bits;

/* LDMA Channel x Interrupt Clear Status Register
   LDMA Channel x Interrupt Enable Register
   LDMA Channel x Interrupt Disable Register
   LDMA Channel x Interrupt Mask Register */
typedef struct {
  __REG32 LDMA_END      : 1;
  __REG32 SRC_ERROR     : 1;
  __REG32 DEST_ERROR    : 1;
  __REG32               :29;
} __ldma_csrx_bits;

/* LDMA Channel x Status Register */
typedef struct {
  __REG32 LDMA_END      : 1;
  __REG32 SRC_ERROR     : 1;
  __REG32 DEST_ERROR    : 1;
  __REG32               : 5;
  __REG32 CHEN          : 1;
  __REG32               :23;
} __ldma_srx_bits;

/* LDMA Channel x Transfer Counter */
typedef struct {
  __REG32 DATA_CNT      :16;
  __REG32               :16;
} __ldma_cntrx_bits;

/* PWM Enable Clock Register
   PWM Disable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 PWM           : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __pwm_ecr_bits;

/* PWM Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 PWM           : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __pwm_pmsr_bits;

/* PWM Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 PWMEN0        : 1;
  __REG32 PWMDIS0       : 1;
  __REG32               : 6;
  __REG32 PWMEN1        : 1;
  __REG32 PWMDIS1       : 1;
  __REG32               :21;
} __pwm_cr_bits;

/* PWM Mode Register */
typedef struct {
  __REG32 PRESCAL0      : 5;
  __REG32 PL0           : 1;
  __REG32               : 2;
  __REG32 PRESCAL1      : 5;
  __REG32 PL1           : 1;
  __REG32               :18;
} __pwm_mr_bits;

/* PWM Clear Status Register
   PWM Interrupt Enable Register
   PWM Interrupt Disable Register
   PWM Interrupt Mask Register */
typedef struct {
  __REG32 PSTA0         : 1;
  __REG32 PEND0         : 1;
  __REG32               : 6;
  __REG32 PSTA1         : 1;
  __REG32 PEND1         : 1;
  __REG32               :22;
} __pwm_csr_bits;

/* PWM Status Register */
typedef struct {
  __REG32 PSTA0         : 1;
  __REG32 PEND0         : 1;
  __REG32 PWENS0        : 1;
  __REG32               : 5;
  __REG32 PSTA1         : 1;
  __REG32 PEND1         : 1;
  __REG32 PWENS1        : 1;
  __REG32               :21;
} __pwm_sr_bits;

/* PWM Delay Register */
typedef struct {
  __REG32 DELAY         :16;
  __REG32               :16;
} __pwm_dly_bits;

/* PWM Pulse Register */
typedef struct {
  __REG32 PULSE         :16;
  __REG32               :16;
} __pwm_pul_bits;

/* SPI Enable Clock Register
   SPI Disable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 SPI           : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __spi_ecr_bits;

/* SPI Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 SPI           : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __spi_pmsr_bits;

/* SPI Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 SPIEN         : 1;
  __REG32 SPIDIS        : 1;
  __REG32               :29;
} __spi_cr_bits;

/* SPI Mode Register */
typedef struct {
  __REG32 MSTR          : 1;
  __REG32 PS            : 1;
  __REG32 PCSDEC        : 1;
  __REG32 DIV32         : 1;
  __REG32 MODFEN        : 1;
  __REG32               : 2;
  __REG32 LLB           : 1;
  __REG32               : 8;
  __REG32 PCS           : 4;
  __REG32               : 4;
  __REG32 DLYBCS        : 8;
} __spi_mr_bits;

/* SPI Clear Status Register */
typedef struct {
  __REG32               : 2;
  __REG32 MODF          : 1;
  __REG32 OVRE          : 1;
  __REG32               : 2;
  __REG32 ENDTRANS      : 1;
  __REG32               :25;
} __spi_csr_bits;

/* SPI Status Register */
typedef struct {
  __REG32 RDRF          : 1;
  __REG32 TDRE          : 1;
  __REG32 MODF          : 1;
  __REG32 OVRE          : 1;
  __REG32               : 2;
  __REG32 ENDTRANS      : 1;
  __REG32               : 1;
  __REG32 ENS           : 1;
  __REG32 BUSY          : 1;
  __REG32               :22;
} __spi_sr_bits;

/* SPI Interrupt Enable Register
   SPI Interrupt Disable Register
   SPI Interrupt Mask Register */
typedef struct {
  __REG32 RDRF          : 1;
  __REG32 TDRE          : 1;
  __REG32 MODF          : 1;
  __REG32 OVRE          : 1;
  __REG32               : 2;
  __REG32 ENDTRANS      : 1;
  __REG32               :25;
} __spi_ier_bits;

/* SPI Receive Data Register */
typedef struct {
  __REG32 RD            :16;
  __REG32 PCS           : 4;
  __REG32               :12;
} __spi_rdr_bits;

/* SPI Transmit Data Register */
typedef struct {
  __REG32 TD            :16;
  __REG32 PCS           : 4;
  __REG32               :12;
} __spi_tdr_bits;

/* SPI Slave Select Register x */
typedef struct {
  __REG32 CPOL          : 1;
  __REG32 NCPHA         : 1;
  __REG32               : 2;
  __REG32 BITS          : 4;
  __REG32 SCBR          : 8;
  __REG32 DLYBS         : 8;
  __REG32 DLYBCT        : 8;
} __spi_ssr_bits;

/* SPI1 Mode Register */
typedef struct {
  __REG32 MSTR          : 1;
  __REG32               : 2;
  __REG32 DIV32         : 1;
  __REG32 MODFEN        : 1;
  __REG32               : 2;
  __REG32 LLB           : 1;
  __REG32               :24;
} __spi8_mr_bits;

/* SPI1 Receive Data Register */
typedef struct {
  __REG32 RD            : 8;
  __REG32               :24;
} __spi8_rdr_bits;

/* SPI1 Transmit Data Register */
typedef struct {
  __REG32 TD            : 8;
  __REG32               :24;
} __spi8_tdr_bits;

/* SPI1 Slave Select Register x */
typedef struct {
  __REG32 CPOL          : 1;
  __REG32 NCPHA         : 1;
  __REG32               : 2;
  __REG32 BITS          : 3;
  __REG32               : 1;
  __REG32 SCBR          : 8;
  __REG32 DLYBS         : 2;
  __REG32               : 6;
  __REG32 DLYBCT        : 2;
  __REG32               : 6;
} __spi8_ssr_bits;

/* ST Enable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 ST            : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __st_ecr_bits;

/* ST Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 ST            : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __st_pmsr_bits;

/* ST Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32               : 7;
  __REG32 CHEN0         : 1;
  __REG32 CHEN1         : 1;
  __REG32               : 6;
  __REG32 CHDIS0        : 1;
  __REG32 CHDIS1        : 1;
  __REG32               :14;
} __st_cr_bits;

/* ST Clear Status Register
   ST Interrupt Enable Register
   ST Interrupt Disable Register
   ST Interrupt Mask Register */
typedef struct {
  __REG32 CHEND0        : 1;
  __REG32 CHEND1        : 1;
  __REG32               :14;
  __REG32 CHDIS0        : 1;
  __REG32 CHDIS1        : 1;
  __REG32               : 6;
  __REG32 CHLD0         : 1;
  __REG32 CHLD1         : 1;
  __REG32               : 6;
} __st_csr_bits;

/* ST Status Register */
typedef struct {
  __REG32 CHEND0        : 1;
  __REG32 CHEND1        : 1;
  __REG32               : 6;
  __REG32 CHENS0        : 1;
  __REG32 CHENS1        : 1;
  __REG32               : 6;
  __REG32 CHDIS0        : 1;
  __REG32 CHDIS1        : 1;
  __REG32               : 6;
  __REG32 CHLD0         : 1;
  __REG32 CHLD1         : 1;
  __REG32               : 6;
} __st_sr_bits;

/* ST Channel x Prescaler Register */
typedef struct {
  __REG32 PRESCAL       : 4;
  __REG32 AUTOREL       : 1;
  __REG32               : 3;
  __REG32 SYSCAL        :11;
  __REG32               :13;
} __st_pr_bits;

/* ST Channel x Counter Register */
typedef struct {
  __REG32 LOAD          :16;
  __REG32               :16;
} __st_ct_bits;

/* ST Current Counter x Value Register */
typedef struct {
  __REG32 COUNT         :16;
  __REG32               :16;
} __st_ccv_bits;

/* SFM Chip Identifier Register */
typedef struct {
  __REG32               : 1;
  __REG32 MC            :11;
  __REG32 PN            :16;
  __REG32 VER           : 4;
} __sfm_cidr_bits;

/* SFM Architecture Register */
typedef struct {
  __REG32 ARC           : 2;
  __REG32               : 6;
  __REG32 NVPT          : 3;
  __REG32 NVPE          : 1;
  __REG32 NVDT          : 3;
  __REG32 NVDE          : 1;
  __REG32 BOOT          : 1;
  __REG32 IRAME         : 1;
  __REG32               :14;
} __sfm_arcr_bits;

/* SFM Memory Size Register */
typedef struct {
  __REG32 NVPMS         : 8;
  __REG32 NVDMS         : 8;
  __REG32 IRAMS         : 8;
  __REG32               : 8;
} __sfm_msr_bits;

/* STT Enable Clock Register
   STT Disable Clock Register
   STT Power Management Status Register */
typedef struct {
  __REG32               :31;
  __REG32 DBGEN         : 1;
} __stt_ecr_bits;

/* STT Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32 CNTEN         : 1;
  __REG32 CNTDIS        : 1;
  __REG32 ALARMEN       : 1;
  __REG32 ALARMDIS      : 1;
  __REG32               :27;
} __stt_cr_bits;

/* STT Mode Register */
typedef struct {
  __REG32 CNTRST        : 1;
  __REG32               :31;
} __stt_mr_bits;

/* STT Clear Status Register
   STT Interrupt Enable Register
   STT Interrupt Disable Register
   STT Interrupt Mask Register */
typedef struct {
  __REG32 ALARM         : 1;
  __REG32 CNTEN         : 1;
  __REG32 CNTDIS        : 1;
  __REG32 ALARMEN       : 1;
  __REG32 ALARMDIS      : 1;
  __REG32               :27;
} __stt_csr_bits;

/* STT Status Register */
typedef struct {
  __REG32 ALARM         : 1;
  __REG32 CNTEN         : 1;
  __REG32 CNTDIS        : 1;
  __REG32 ALARMEN       : 1;
  __REG32 ALARMDIS      : 1;
  __REG32 WSEC          : 1;
  __REG32               : 2;
  __REG32 CNTENS        : 1;
  __REG32 ALARMENS      : 1;
  __REG32               :22;
} __stt_sr_bits;

/* USART Enable Clock Register
   USART Disable Clock Register */
typedef struct {
  __REG32               : 1;
  __REG32 USART         : 1;
  __REG32               :29;
  __REG32 DBGEN         : 1;
} __us_ecr_bits;

/* USART Power Management Status Register */
typedef struct {
  __REG32               : 1;
  __REG32 USART         : 1;
  __REG32               : 2;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __us_pmsr_bits;

/* USART Control Register */
typedef struct {
  __REG32 SWRST         : 1;
  __REG32               : 1;
  __REG32 RSTRX         : 1;
  __REG32 RSTTX         : 1;
  __REG32 RXEN          : 1;
  __REG32 RXDIS         : 1;
  __REG32 TXEN          : 1;
  __REG32 TXDIS         : 1;
  __REG32               : 1;
  __REG32 STTBRK        : 1;
  __REG32 STPBRK        : 1;
  __REG32 STTTO         : 1;
  __REG32 SENDA         : 1;
  __REG32               : 3;
  __REG32 STHEADER      : 1;
  __REG32 STREPS        : 1;
  __REG32               :14;
} __us_cr_bits;

/* USART Mode Register */
typedef struct {
  __REG32 LIN           : 1;
  __REG32 SENDTIME      : 3;
  __REG32 CLKS          : 2;
  __REG32 CHRL          : 2;
  __REG32 SYNC          : 1;
  __REG32 PAR           : 3;
  __REG32 NBSTOP        : 2;
  __REG32 CHMODE        : 2;
  __REG32 SMCARDPT      : 1;
  __REG32 MODE9         : 1;
  __REG32 CLKO          : 1;
  __REG32 LIN2_0        : 1;
  __REG32               :12;
} __us_mr_bits;

/* USART Clear Status Register */
typedef struct {
  __REG32               : 2;
  __REG32 RXBRK         : 1;
  __REG32 ENDRX         : 1;
  __REG32 ENDTX         : 1;    
  __REG32 OVRE          : 1;
  __REG32 FRAME         : 1;
  __REG32 PARE          : 1;
  __REG32               : 2;
  __REG32 IDLE          : 1;
  __REG32               :13;
  __REG32 ENDHEADER     : 1;
  __REG32 ENDMESS       : 1;
  __REG32 NOTRESP       : 1;
  __REG32 BITERROR      : 1;
  __REG32 IPERROR       : 1;
  __REG32 CHECKSUM      : 1;
  __REG32 WAKEUP        : 1;
  __REG32               : 1;
} __us_csr_bits;

/* USART Status Register */
typedef struct {
  __REG32 RXRDY         : 1;
  __REG32 TXRDY         : 1;
  __REG32 RXBRK         : 1;
  __REG32               : 2;    
  __REG32 OVRE          : 1;
  __REG32 FRAME         : 1;
  __REG32 PARE          : 1;
  __REG32 TIMEOUT       : 1;
  __REG32 TXEMPTY       : 1;
  __REG32 IDLE          : 1;
  __REG32 IDLEFLAG      : 1;
  __REG32               :12;
  __REG32 ENDHEADER     : 1;
  __REG32 ENDMESS       : 1;
  __REG32 NOTRESP       : 1;
  __REG32 BITERROR      : 1;
  __REG32 IPERROR       : 1;
  __REG32 CHECKSUM      : 1;
  __REG32 WAKEUP        : 1;
  __REG32               : 1;
} __us_sr_bits;

/* USART Interrupt Enable Register
   USART Interrupt Disable Register
   USART Interrupt Mask Register */
typedef struct {
  __REG32 RXRDY         : 1;
  __REG32 TXRDY         : 1;
  __REG32 RXBRK         : 1;
  __REG32               : 2;    
  __REG32 OVRE          : 1;
  __REG32 FRAME         : 1;
  __REG32 PARE          : 1;
  __REG32 TIMEOUT       : 1;
  __REG32 TXEMPTY       : 1;
  __REG32 IDLE          : 1;
  __REG32               :13;
  __REG32 ENDHEADER     : 1;
  __REG32 ENDMESS       : 1;
  __REG32 NOTRESP       : 1;
  __REG32 BITERROR      : 1;
  __REG32 IPERROR       : 1;
  __REG32 CHECKSUM      : 1;
  __REG32 WAKEUP        : 1;
  __REG32               : 1;
} __us_ier_bits;

/* USART Receiver Holding Register */
typedef struct {
  __REG32 RXCHR         : 9;
  __REG32               :23;
} __us_rhr_bits;

/* USART Receiver Holding Register */
typedef struct {
  __REG32 TXCHR         : 9;
  __REG32               :23;
} __us_thr_bits;

/* USART Baud Rate Generator Register */
typedef struct {
  __REG32 CD            :16;
  __REG32               :16;
} __us_brgr_bits;

/* USART Receiver Time-Out Register */
typedef struct {
  __REG32 TO            :16;
  __REG32               :16;
} __us_rtor_bits;

/* USART Transmit Time-Guard Register */
typedef struct {
  __REG32 TG            : 8;
  __REG32               :24;
} __us_ttgr_bits;

/* USART LIN Identifier Register */
typedef struct {
  __REG32 IDENTIFIER    : 6;
  __REG32 NDATA         : 3;
  __REG32 CHK_SEL       : 1;
  __REG32               : 6;
  __REG32 WAKE_UP_TIME  :14;
  __REG32               : 2;
} __us_lir_bits;

/* USART Data Field Write 0 Register
   USART Data Field Read 0 Register */
typedef struct {
  __REG32 DATA0         : 8;
  __REG32 DATA1         : 8;
  __REG32 DATA2         : 8;
  __REG32 DATA3         : 8;
} __us_dfwr0_bits;

/* USART Data Field Write 1 Register
   USART Data Field Read 1 Register */
typedef struct {
  __REG32 DATA4         : 8;
  __REG32 DATA5         : 8;
  __REG32 DATA6         : 8;
  __REG32 DATA7         : 8;
} __us_dfwr1_bits;

/* USART Synchronous Break Length Register */
typedef struct {
  __REG32 SYNC_BRK      : 5;
  __REG32               :27;
} __us_sblr_bits;

/* WD Enable Clock Register
   WD Disable Clock Register */
typedef struct {
  __REG32               :31;
  __REG32 DBGEN         : 1;
} __wd_ecr_bits;

/* WD Power Management Status Register */
typedef struct {
  __REG32               : 4;
  __REG32 IPIDCODE      :26;
  __REG32               : 1;
  __REG32 DBGEN         : 1;
} __wd_pmsr_bits;

/* WD Control Register */
typedef struct {
  __REG32 RSTKEY        :16;
  __REG32               :16;
} __wd_cr_bits;

/* WD Mode Register */
typedef struct {
  __REG32 WDPDIV        : 3;
  __REG32               : 5;
  __REG32 PCV           :16;
  __REG32 CKEY          : 8;
} __wd_mr_bits;

/* WD Overflow Mode Register */
typedef struct {
  __REG32 WDEN          : 1;
  __REG32 RSTEN         : 1;
  __REG32               : 2;
  __REG32 OKEY          :12;
  __REG32               :16;
} __wd_omr_bits;

/* WD Clear Status Register
   WD Interrupt Enable Register
   WD Interrupt Disable Register
   WD Interrupt Mask Register */
typedef struct {
  __REG32 WDPEND        : 1;
  __REG32 WDOVF         : 1;
  __REG32               :30;
} __wd_csr_bits;

/* WD Clear Status Register */
typedef struct {
  __REG32 WDPEND        : 1;
  __REG32 WDOVF         : 1;
  __REG32               : 6;
  __REG32 PENDING       : 1;
  __REG32 CLEAR_STATUS  : 1;
  __REG32               :22;
} __wd_sr_bits;

/* WD Clear Status Register */
typedef struct {
  __REG32 RSTALW        : 1;
  __REG32               : 7;
  __REG32 PWL           :16;
  __REG32 PWKEY         : 8;
} __wd_pwr_bits;

/* WD Clear Status Register */
typedef struct {
  __REG32 COUNT         :16;
  __REG32 RESET         : 1;
  __REG32               :15;
} __wd_ctr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ECR,         0xFFE0C050,__WRITE      ,__adc_ecr_bits);
__IO_REG32_BIT(ADC_DCR,         0xFFE0C054,__WRITE      ,__adc_ecr_bits);
__IO_REG32_BIT(ADC_PMSR,        0xFFE0C058,__READ       ,__adc_pmsr_bits);
__IO_REG32_BIT(ADC_CR,          0xFFE0C060,__WRITE      ,__adc_cr_bits);
__IO_REG32_BIT(ADC_MR,          0xFFE0C064,__READ_WRITE ,__adc_mr_bits);
__IO_REG32_BIT(ADC_CSR,         0xFFE0C06C,__WRITE      ,__adc_csr_bits);
__IO_REG32_BIT(ADC_SR,          0xFFE0C070,__READ       ,__adc_sr_bits);
__IO_REG32_BIT(ADC_IER,         0xFFE0C074,__WRITE      ,__adc_ier_bits);
__IO_REG32_BIT(ADC_IDR,         0xFFE0C078,__WRITE      ,__adc_ier_bits);
__IO_REG32_BIT(ADC_IMR,         0xFFE0C07C,__READ       ,__adc_ier_bits);
__IO_REG32_BIT(ADC_CMR0,        0xFFE0C080,__READ_WRITE ,__adc_cmr0_bits);
__IO_REG32_BIT(ADC_CMR1,        0xFFE0C084,__READ_WRITE ,__adc_cmr1_bits);
__IO_REG32_BIT(ADC_DR,          0xFFE0C088,__READ       ,__adc_dr_bits);

/***************************************************************************
 **
 **  CM
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_STR,          0xFFFE8000,__READ       ,__cm_str_bits);
__IO_REG32_BIT(CM_WFIR,         0xFFFE8008,__READ_WRITE ,__cm_wfir_bits);
__IO_REG32_BIT(CM_PSTR,         0xFFFE800C,__READ_WRITE ,__cm_pstr_bits);
__IO_REG32_BIT(CM_PDPR,         0xFFFE8010,__READ_WRITE ,__cm_pdpr_bits);
__IO_REG32_BIT(CM_OSTR,         0xFFFE8014,__READ_WRITE ,__cm_ostr_bits);
__IO_REG32_BIT(CM_DIVBR,        0xFFFE801C,__READ_WRITE ,__cm_divbr_bits);
__IO_REG32_BIT(CM_SELR,         0xFFFE8020,__READ_WRITE ,__cm_selr_bits);
__IO_REG32_BIT(CM_RSR,          0xFFFE8024,__READ       ,__cm_rsr_bits);
__IO_REG32_BIT(CM_MDIVR,        0xFFFE8028,__READ_WRITE ,__cm_mdivr_bits);
__IO_REG32_BIT(CM_LFOSCR,       0xFFFE802C,__READ_WRITE ,__cm_lfoscr_bits);
__IO_REG32_BIT(CM_CR,           0xFFFE8030,__WRITE      ,__cm_cr_bits);
__IO_REG32_BIT(CM_MR,           0xFFFE8064,__READ_WRITE ,__cm_mr_bits);
__IO_REG32_BIT(CM_CSR,          0xFFFE806C,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_SR,           0xFFFE8070,__READ       ,__cm_csr_bits);
__IO_REG32_BIT(CM_IER,          0xFFFE8074,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_IDR,          0xFFFE8078,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_IMR,          0xFFFE807C,__READ       ,__cm_csr_bits);

/***************************************************************************
 **
 **  CANB0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0_ECR,        0xFFE18050,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_DCR,        0xFFE18054,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_PMSR,       0xFFE18058,__READ       ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_CR,         0xFFE18060,__WRITE      ,__can_cr_bits);
__IO_REG32_BIT(CAN0_MR,         0xFFE18064,__READ_WRITE ,__can_mr_bits);
__IO_REG32_BIT(CAN0_CSR,        0xFFE1806C,__WRITE      ,__can_csr_bits);
__IO_REG32_BIT(CAN0_SR,         0xFFE18070,__READ       ,__can_sr_bits);
__IO_REG32_BIT(CAN0_IER,        0xFFE18074,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN0_IDR,        0xFFE18078,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN0_IMR,        0xFFE1807C,__READ       ,__can_ier_bits);
__IO_REG32_BIT(CAN0_ISSR,       0xFFE18084,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIER,       0xFFE18088,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIDR,       0xFFE1808C,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIMR,       0xFFE18090,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_HPIR,       0xFFE18094,__READ       ,__can_hpir_bits);
__IO_REG32_BIT(CAN0_ERCR,       0xFFE18098,__READ       ,__can_ercr_bits);
__IO_REG32_BIT(CAN0_TMR0,       0xFFE18100,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN0_DAR0,       0xFFE18104,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_DBR0,       0xFFE18108,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_MSKR0,      0xFFE1810C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN0_IR0,        0xFFE18110,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN0_MCR0,       0xFFE18114,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN0_STPR0,      0xFFE18118,__READ       );
__IO_REG32_BIT(CAN0_TMR1,       0xFFE18120,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN0_DAR1,       0xFFE18124,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_DBR1,       0xFFE18128,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_MSKR1,      0xFFE1812C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN0_IR1,        0xFFE18130,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN0_MCR1,       0xFFE18134,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN0_STPR1,      0xFFE18138,__READ       );
__IO_REG32_BIT(CAN0_TRR,        0xFFE18140,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_NDR,        0xFFE18144,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_MVR,        0xFFE18148,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_TSTR,       0xFFE1814C,__READ_WRITE ,__can_tstr_bits);

/***************************************************************************
 **
 **  PIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(PIO0_PER,        0xFFE64000,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_PDR,        0xFFE64004,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_PSR,        0xFFE64008,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_OER,        0xFFE64010,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_ODR,        0xFFE64014,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_OSR,        0xFFE64018,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_WODR,       0xFFE6402C,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_SODR,       0xFFE64030,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_CODR,       0xFFE64034,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_ODSR,       0xFFE64038,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_PDSR,       0xFFE6403C,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_MDER,       0xFFE64040,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_MDDR,       0xFFE64044,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_MDSR,       0xFFE64048,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_ECR,        0xFFE64050,__WRITE      ,__pio_ecr_bits);
__IO_REG32_BIT(PIO0_DCR,        0xFFE64054,__WRITE      ,__pio_ecr_bits);
__IO_REG32_BIT(PIO0_PMSR,       0xFFE64058,__READ       ,__pio_pmsr_bits);
__IO_REG32_BIT(PIO0_CR,         0xFFE64060,__WRITE      ,__pio_cr_bits);
__IO_REG32_BIT(PIO0_CSR,        0xFFE6406C,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_SR,         0xFFE64070,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO0_IER,        0xFFE64074,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_IDR,        0xFFE64078,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO0_IMR,        0xFFE6407C,__READ       ,__pio_per_bits);

/***************************************************************************
 **
 **  PIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(PIO1_PER,        0xFFE68000,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_PDR,        0xFFE68004,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_PSR,        0xFFE68008,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_OER,        0xFFE68010,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_ODR,        0xFFE68014,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_OSR,        0xFFE68018,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_WODR,       0xFFE6802C,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_SODR,       0xFFE68030,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_CODR,       0xFFE68034,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_ODSR,       0xFFE68038,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_PDSR,       0xFFE6803C,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_MDER,       0xFFE68040,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_MDDR,       0xFFE68044,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_MDSR,       0xFFE68048,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_ECR,        0xFFE68050,__WRITE      ,__pio_ecr_bits);
__IO_REG32_BIT(PIO1_DCR,        0xFFE68054,__WRITE      ,__pio_ecr_bits);
__IO_REG32_BIT(PIO1_PMSR,       0xFFE68058,__READ       ,__pio_pmsr_bits);
__IO_REG32_BIT(PIO1_CR,         0xFFE68060,__WRITE      ,__pio_cr_bits);
__IO_REG32_BIT(PIO1_CSR,        0xFFE6806C,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_SR,         0xFFE68070,__READ       ,__pio_per_bits);
__IO_REG32_BIT(PIO1_IER,        0xFFE68074,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_IDR,        0xFFE68078,__WRITE      ,__pio_per_bits);
__IO_REG32_BIT(PIO1_IMR,        0xFFE6807C,__READ       ,__pio_per_bits);

/***************************************************************************
 **
 **  GPT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT_ECR0,        0xFFE1C050,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_DCR0,        0xFFE1C054,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_PMSR0,       0xFFE1C058,__READ       ,__gpt_pmsr_bits);
__IO_REG32_BIT(GPT_CR0,         0xFFE1C060,__WRITE      ,__gpt_cr_bits);
__IO_REG32_BIT(GPT_MR_CM0,      0xFFE1C064,__READ_WRITE ,__gpt_mr_bits);
#define GPT_MR_WM0      GPT_MR_CM0
#define GPT_MR_WM0_bit  GPT_MR_CM0_bit
__IO_REG32_BIT(GPT_CSR0,        0xFFE1C06C,__WRITE      ,__gpt_csr_bits);
__IO_REG32_BIT(GPT_SR0,         0xFFE1C070,__READ       ,__gpt_sr_bits);
__IO_REG32_BIT(GPT_IER0,        0xFFE1C074,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IDR0,        0xFFE1C078,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IMR0,        0xFFE1C07C,__READ       ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_CV0,         0xFFE1C080,__READ       ,__gpt_cv_bits);
__IO_REG32_BIT(GPT_RA0,         0xFFE1C084,__READ_WRITE ,__gpt_ra_bits);
__IO_REG32_BIT(GPT_RB0,         0xFFE1C088,__READ_WRITE ,__gpt_rb_bits);
__IO_REG32_BIT(GPT_RC0,         0xFFE1C08C,__READ_WRITE ,__gpt_rc_bits);
__IO_REG32_BIT(GPT_ECR1,        0xFFE1C150,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_DCR1,        0xFFE1C154,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_PMSR1,       0xFFE1C158,__READ       ,__gpt_pmsr_bits);
__IO_REG32_BIT(GPT_CR1,         0xFFE1C160,__WRITE      ,__gpt_cr_bits);
__IO_REG32_BIT(GPT_MR_CM1,      0xFFE1C164,__READ_WRITE ,__gpt_mr_bits);
#define GPT_MR_WM1      GPT_MR_CM1     
#define GPT_MR_WM1_bit  GPT_MR_CM1_bit 
__IO_REG32_BIT(GPT_CSR1,        0xFFE1C16C,__WRITE      ,__gpt_csr_bits);
__IO_REG32_BIT(GPT_SR1,         0xFFE1C170,__READ       ,__gpt_sr_bits);
__IO_REG32_BIT(GPT_IER1,        0xFFE1C174,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IDR1,        0xFFE1C178,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IMR1,        0xFFE1C17C,__READ       ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_CV1,         0xFFE1C180,__READ       ,__gpt_cv_bits);
__IO_REG32_BIT(GPT_RA1,         0xFFE1C184,__READ_WRITE ,__gpt_ra_bits);
__IO_REG32_BIT(GPT_RB1,         0xFFE1C188,__READ_WRITE ,__gpt_rb_bits);
__IO_REG32_BIT(GPT_RC1,         0xFFE1C18C,__READ_WRITE ,__gpt_rc_bits);
__IO_REG32_BIT(GPT_ECR2,        0xFFE1C250,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_DCR2,        0xFFE1C254,__WRITE      ,__gpt_ecr_bits);
__IO_REG32_BIT(GPT_PMSR2,       0xFFE1C258,__READ       ,__gpt_pmsr_bits);
__IO_REG32_BIT(GPT_CR2,         0xFFE1C260,__WRITE      ,__gpt_cr_bits);
__IO_REG32_BIT(GPT_MR_CM2,      0xFFE1C264,__READ_WRITE ,__gpt_mr_bits);
#define GPT_MR_WM2      GPT_MR_CM2     
#define GPT_MR_WM2_bit  GPT_MR_CM2_bit 
__IO_REG32_BIT(GPT_CSR2,        0xFFE1C26C,__WRITE      ,__gpt_csr_bits);
__IO_REG32_BIT(GPT_SR2,         0xFFE1C270,__READ       ,__gpt_sr_bits);
__IO_REG32_BIT(GPT_IER2,        0xFFE1C274,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IDR2,        0xFFE1C278,__WRITE      ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_IMR2,        0xFFE1C27C,__READ       ,__gpt_ier_bits);
__IO_REG32_BIT(GPT_CV2,         0xFFE1C280,__READ       ,__gpt_cv_bits);
__IO_REG32_BIT(GPT_RA2,         0xFFE1C284,__READ_WRITE ,__gpt_ra_bits);
__IO_REG32_BIT(GPT_RB2,         0xFFE1C288,__READ_WRITE ,__gpt_rb_bits);
__IO_REG32_BIT(GPT_RC2,         0xFFE1C28C,__READ_WRITE ,__gpt_rc_bits);
__IO_REG32_BIT(GPT_BCR,         0xFFE1C300,__WRITE      ,__gpt_bcr_bits);
__IO_REG32_BIT(GPT_BMR,         0xFFE1C304,__READ_WRITE ,__gpt_bmr_bits);

/***************************************************************************
 **
 **  GIC
 **
 ***************************************************************************/
__IO_REG32_BIT(GIC_SMR0,        0xFFFFEF00,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR1,        0xFFFFEF04,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR2,        0xFFFFEF08,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR3,        0xFFFFEF0C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR4,        0xFFFFEF10,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR5,        0xFFFFEF14,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR6,        0xFFFFEF18,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR7,        0xFFFFEF1C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR8,        0xFFFFEF20,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR9,        0xFFFFEF24,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR10,       0xFFFFEF28,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR11,       0xFFFFEF2C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR12,       0xFFFFEF30,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR13,       0xFFFFEF34,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR14,       0xFFFFEF38,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR15,       0xFFFFEF3C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR16,       0xFFFFEF40,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR17,       0xFFFFEF44,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR18,       0xFFFFEF48,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR19,       0xFFFFEF4C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR20,       0xFFFFEF50,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR21,       0xFFFFEF54,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR22,       0xFFFFEF58,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR23,       0xFFFFEF5C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR24,       0xFFFFEF60,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR25,       0xFFFFEF64,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR26,       0xFFFFEF68,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR27,       0xFFFFEF6C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR28,       0xFFFFEF70,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR29,       0xFFFFEF74,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR30,       0xFFFFEF78,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR31,       0xFFFFEF7C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR32,       0xFFFFEF80,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR33,       0xFFFFEF84,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR34,       0xFFFFEF88,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR35,       0xFFFFEF8C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR36,       0xFFFFEF90,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR37,       0xFFFFEF94,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR38,       0xFFFFEF98,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR39,       0xFFFFEF9C,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR40,       0xFFFFEFA0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR41,       0xFFFFEFA4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR42,       0xFFFFEFA8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR43,       0xFFFFEFAC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR44,       0xFFFFEFB0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR45,       0xFFFFEFB4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR46,       0xFFFFEFB8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR47,       0xFFFFEFBC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR48,       0xFFFFEFC0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR49,       0xFFFFEFC4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR50,       0xFFFFEFC8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR51,       0xFFFFEFCC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR52,       0xFFFFEFD0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR53,       0xFFFFEFD4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR54,       0xFFFFEFD8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR55,       0xFFFFEFDC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR56,       0xFFFFEFE0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR57,       0xFFFFEFE4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR58,       0xFFFFEFE8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR59,       0xFFFFEFEC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR60,       0xFFFFEFF0,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR61,       0xFFFFEFF4,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR62,       0xFFFFEFF8,__READ_WRITE ,__gic_smr_bits);
__IO_REG32_BIT(GIC_SMR63,       0xFFFFEFFC,__READ_WRITE ,__gic_smr_bits);
__IO_REG32(    GIC_SVR0,        0xFFFFF000,__READ_WRITE );
__IO_REG32(    GIC_SVR1,        0xFFFFF004,__READ_WRITE );
__IO_REG32(    GIC_SVR2,        0xFFFFF008,__READ_WRITE );
__IO_REG32(    GIC_SVR3,        0xFFFFF00C,__READ_WRITE );
__IO_REG32(    GIC_SVR4,        0xFFFFF010,__READ_WRITE );
__IO_REG32(    GIC_SVR5,        0xFFFFF014,__READ_WRITE );
__IO_REG32(    GIC_SVR6,        0xFFFFF018,__READ_WRITE );
__IO_REG32(    GIC_SVR7,        0xFFFFF01C,__READ_WRITE );
__IO_REG32(    GIC_SVR8,        0xFFFFF020,__READ_WRITE );
__IO_REG32(    GIC_SVR9,        0xFFFFF024,__READ_WRITE );
__IO_REG32(    GIC_SVR10,       0xFFFFF028,__READ_WRITE );
__IO_REG32(    GIC_SVR11,       0xFFFFF02C,__READ_WRITE );
__IO_REG32(    GIC_SVR12,       0xFFFFF030,__READ_WRITE );
__IO_REG32(    GIC_SVR13,       0xFFFFF034,__READ_WRITE );
__IO_REG32(    GIC_SVR14,       0xFFFFF038,__READ_WRITE );
__IO_REG32(    GIC_SVR15,       0xFFFFF03C,__READ_WRITE );
__IO_REG32(    GIC_SVR16,       0xFFFFF040,__READ_WRITE );
__IO_REG32(    GIC_SVR17,       0xFFFFF044,__READ_WRITE );
__IO_REG32(    GIC_SVR18,       0xFFFFF048,__READ_WRITE );
__IO_REG32(    GIC_SVR19,       0xFFFFF04C,__READ_WRITE );
__IO_REG32(    GIC_SVR20,       0xFFFFF050,__READ_WRITE );
__IO_REG32(    GIC_SVR21,       0xFFFFF054,__READ_WRITE );
__IO_REG32(    GIC_SVR22,       0xFFFFF058,__READ_WRITE );
__IO_REG32(    GIC_SVR23,       0xFFFFF05C,__READ_WRITE );
__IO_REG32(    GIC_SVR24,       0xFFFFF060,__READ_WRITE );
__IO_REG32(    GIC_SVR25,       0xFFFFF064,__READ_WRITE );
__IO_REG32(    GIC_SVR26,       0xFFFFF068,__READ_WRITE );
__IO_REG32(    GIC_SVR27,       0xFFFFF06C,__READ_WRITE );
__IO_REG32(    GIC_SVR28,       0xFFFFF070,__READ_WRITE );
__IO_REG32(    GIC_SVR29,       0xFFFFF074,__READ_WRITE );
__IO_REG32(    GIC_SVR30,       0xFFFFF078,__READ_WRITE );
__IO_REG32(    GIC_SVR31,       0xFFFFF07C,__READ_WRITE );
__IO_REG32(    GIC_SVR32,       0xFFFFF080,__READ_WRITE );
__IO_REG32(    GIC_SVR33,       0xFFFFF084,__READ_WRITE );
__IO_REG32(    GIC_SVR34,       0xFFFFF088,__READ_WRITE );
__IO_REG32(    GIC_SVR35,       0xFFFFF08C,__READ_WRITE );
__IO_REG32(    GIC_SVR36,       0xFFFFF090,__READ_WRITE );
__IO_REG32(    GIC_SVR37,       0xFFFFF094,__READ_WRITE );
__IO_REG32(    GIC_SVR38,       0xFFFFF098,__READ_WRITE );
__IO_REG32(    GIC_SVR39,       0xFFFFF09C,__READ_WRITE );
__IO_REG32(    GIC_SVR40,       0xFFFFF0A0,__READ_WRITE );
__IO_REG32(    GIC_SVR41,       0xFFFFF0A4,__READ_WRITE );
__IO_REG32(    GIC_SVR42,       0xFFFFF0A8,__READ_WRITE );
__IO_REG32(    GIC_SVR43,       0xFFFFF0AC,__READ_WRITE );
__IO_REG32(    GIC_SVR44,       0xFFFFF0B0,__READ_WRITE );
__IO_REG32(    GIC_SVR45,       0xFFFFF0B4,__READ_WRITE );
__IO_REG32(    GIC_SVR46,       0xFFFFF0B8,__READ_WRITE );
__IO_REG32(    GIC_SVR47,       0xFFFFF0BC,__READ_WRITE );
__IO_REG32(    GIC_SVR48,       0xFFFFF0C0,__READ_WRITE );
__IO_REG32(    GIC_SVR49,       0xFFFFF0C4,__READ_WRITE );
__IO_REG32(    GIC_SVR50,       0xFFFFF0C8,__READ_WRITE );
__IO_REG32(    GIC_SVR51,       0xFFFFF0CC,__READ_WRITE );
__IO_REG32(    GIC_SVR52,       0xFFFFF0D0,__READ_WRITE );
__IO_REG32(    GIC_SVR53,       0xFFFFF0D4,__READ_WRITE );
__IO_REG32(    GIC_SVR54,       0xFFFFF0D8,__READ_WRITE );
__IO_REG32(    GIC_SVR55,       0xFFFFF0DC,__READ_WRITE );
__IO_REG32(    GIC_SVR56,       0xFFFFF0E0,__READ_WRITE );
__IO_REG32(    GIC_SVR57,       0xFFFFF0E4,__READ_WRITE );
__IO_REG32(    GIC_SVR58,       0xFFFFF0E8,__READ_WRITE );
__IO_REG32(    GIC_SVR59,       0xFFFFF0EC,__READ_WRITE );
__IO_REG32(    GIC_SVR60,       0xFFFFF0F0,__READ_WRITE );
__IO_REG32(    GIC_SVR61,       0xFFFFF0F4,__READ_WRITE );
__IO_REG32(    GIC_SVR62,       0xFFFFF0F8,__READ_WRITE );
__IO_REG32(    GIC_SVR63,       0xFFFFF0FC,__READ_WRITE );
__IO_REG32(    GIC_IVR,         0xFFFFF100,__READ       );
__IO_REG32(    GIC_FVR,         0xFFFFF104,__READ       );
__IO_REG32_BIT(GIC_ISR,         0xFFFFF108,__READ       ,__gic_isr_bits);
__IO_REG32_BIT(GIC_IPR0,        0xFFFFF10C,__READ       ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_IPR1,        0xFFFFF110,__READ       ,__gic_ipr1_bits);
__IO_REG32_BIT(GIC_IMR0,        0xFFFFF114,__READ       ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_IMR1,        0xFFFFF118,__READ       ,__gic_ipr1_bits);
__IO_REG32_BIT(GIC_CISR,        0xFFFFF11C,__READ       ,__gic_cisr_bits);
__IO_REG32_BIT(GIC_IECR0,       0xFFFFF120,__WRITE      ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_IECR1,       0xFFFFF124,__WRITE      ,__gic_ipr1_bits);
__IO_REG32_BIT(GIC_IDCR0,       0xFFFFF128,__WRITE      ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_IDCR1,       0xFFFFF12C,__WRITE      ,__gic_ipr1_bits);
__IO_REG32_BIT(GIC_ICCR0,       0xFFFFF130,__WRITE      ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_ICCR1,       0xFFFFF134,__WRITE      ,__gic_ipr1_bits);
__IO_REG32_BIT(GIC_ISCR0,       0xFFFFF138,__WRITE      ,__gic_ipr0_bits);
__IO_REG32_BIT(GIC_ISCR1,       0xFFFFF13C,__WRITE      ,__gic_ipr1_bits);
__IO_REG32(    GIC_EOICR,       0xFFFFF140,__WRITE      );
__IO_REG32(    GIC_SPU,         0xFFFFF144,__READ_WRITE );

/***************************************************************************
 **
 **  IOCONF
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCONF_MR0,      0xFFE2C000,__READ_WRITE ,__ioconf_mr0_bits);
__IO_REG32_BIT(IOCONF_MR1,      0xFFE2C004,__READ_WRITE ,__ioconf_mr1_bits);
__IO_REG32_BIT(IOCONF_MR2,      0xFFE2C008,__READ_WRITE ,__ioconf_mr2_bits);
__IO_REG32_BIT(IOCONF_MR3,      0xFFE2C00C,__READ_WRITE ,__ioconf_mr3_bits);
__IO_REG32_BIT(IOCONF_MR4,      0xFFE2C010,__READ_WRITE ,__ioconf_mr4_bits);
__IO_REG32_BIT(IOCONF_MR5,      0xFFE2C014,__READ_WRITE ,__ioconf_mr5_bits);
__IO_REG32_BIT(IOCONF_MR6,      0xFFE2C018,__READ_WRITE ,__ioconf_mr6_bits);
__IO_REG32_BIT(IOCONF_MR7,      0xFFE2C01C,__READ_WRITE ,__ioconf_mr7_bits);
__IO_REG32_BIT(IOCONF_MR15,     0xFFE2C03C,__READ_WRITE ,__ioconf_mr15_bits);
__IO_REG32_BIT(IOCONF_MR16,     0xFFE2C040,__READ_WRITE ,__ioconf_mr16_bits);

/***************************************************************************
 **
 **  I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_ECR,        0xFFE50050,__WRITE      ,__i2c_ecr_bits);
__IO_REG32_BIT(I2C0_DCR,        0xFFE50054,__WRITE      ,__i2c_ecr_bits);
__IO_REG32_BIT(I2C0_PMSR,       0xFFE50058,__READ       ,__i2c_pmsr_bits);
__IO_REG32_BIT(I2C0_CR,         0xFFE50060,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C0_MR,         0xFFE50064,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C0_SR,         0xFFE50070,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C0_IER,        0xFFE50074,__WRITE      ,__i2c_ier_bits);
__IO_REG32_BIT(I2C0_IDR,        0xFFE50078,__WRITE      ,__i2c_ier_bits);
__IO_REG32_BIT(I2C0_IMR,        0xFFE5007C,__READ       ,__i2c_ier_bits);
__IO_REG32_BIT(I2C0_DAT,        0xFFE50080,__READ_WRITE ,__i2c_dat_bits);
__IO_REG32_BIT(I2C0_ADR,        0xFFE50084,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_THOLD,      0xFFE50088,__READ_WRITE ,__i2c_thold_bits);

/***************************************************************************
 **
 **  IFC
 **
 ***************************************************************************/
__IO_REG32_BIT(IFC_PMSR,        0xFFE04058,__READ       ,__ifc_pmsr_bits);
__IO_REG32_BIT(IFC_CR,          0xFFE04060,__WRITE      ,__ifc_cr_bits);
__IO_REG32_BIT(IFC_MR,          0xFFE04064,__READ_WRITE ,__ifc_mr_bits);
__IO_REG32_BIT(IFC_CSR,         0xFFE0406C,__WRITE      ,__ifc_csr_bits);
__IO_REG32_BIT(IFC_SR,          0xFFE04070,__READ       ,__ifc_sr_bits);
__IO_REG32_BIT(IFC_IER,         0xFFE04074,__WRITE      ,__ifc_csr_bits);
__IO_REG32_BIT(IFC_IDR,         0xFFE04078,__WRITE      ,__ifc_csr_bits);
__IO_REG32_BIT(IFC_IMR,         0xFFE0407C,__READ       ,__ifc_csr_bits);

/***************************************************************************
 **
 **  IRC
 **
 ***************************************************************************/
__IO_REG32_BIT(IRC_MR,          0xFFFF0000,__READ_WRITE ,__irc_mr_bits);

/***************************************************************************
 **
 **  LDMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(LDMA_ECR,        0xFFFF8050,__WRITE      ,__ldma_ecr_bits);
__IO_REG32_BIT(LDMA_DCR,        0xFFFF8054,__WRITE      ,__ldma_ecr_bits);
__IO_REG32_BIT(LDMA_PMSR,       0xFFFF8058,__READ       ,__ldma_pmsr_bits);
__IO_REG32_BIT(LDMA_CR,         0xFFFF8060,__WRITE      ,__ldma_cr_bits);
__IO_REG32_BIT(LDMA_SR,         0xFFFF8070,__READ       ,__ldma_sr_bits);
__IO_REG32_BIT(LDMA_IER,        0xFFFF8074,__WRITE      ,__ldma_sr_bits);
__IO_REG32_BIT(LDMA_IDR,        0xFFFF8078,__WRITE      ,__ldma_sr_bits);
__IO_REG32_BIT(LDMA_IMR,        0xFFFF807C,__READ       ,__ldma_sr_bits);
__IO_REG32_BIT(LDMA_CR0,        0xFFFF8100,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR0,        0xFFFF8104,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR0,       0xFFFF8108,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR0,        0xFFFF810C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER0,       0xFFFF8110,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR0,       0xFFFF8114,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR0,       0xFFFF8118,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR0,     0xFFFF811C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR0,     0xFFFF8120,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR0,      0xFFFF8124,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR1,        0xFFFF8180,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR1,        0xFFFF8184,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR1,       0xFFFF8188,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR1,        0xFFFF818C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER1,       0xFFFF8190,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR1,       0xFFFF8194,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR1,       0xFFFF8198,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR1,     0xFFFF819C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR1,     0xFFFF81A0,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR1,      0xFFFF81A4,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR2,        0xFFFF8200,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR2,        0xFFFF8204,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR2,       0xFFFF8208,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR2,        0xFFFF820C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER2,       0xFFFF8210,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR2,       0xFFFF8214,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR2,       0xFFFF8218,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR2,     0xFFFF821C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR2,     0xFFFF8220,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR2,      0xFFFF8224,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR3,        0xFFFF8280,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR3,        0xFFFF8284,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR3,       0xFFFF8288,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR3,        0xFFFF828C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER3,       0xFFFF8290,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR3,       0xFFFF8294,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR3,       0xFFFF8298,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR3,     0xFFFF829C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR3,     0xFFFF82A0,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR3,      0xFFFF82A4,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR4,        0xFFFF8300,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR4,        0xFFFF8304,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR4,       0xFFFF8308,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR4,        0xFFFF830C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER4,       0xFFFF8310,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR4,       0xFFFF8314,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR4,       0xFFFF8318,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR4,     0xFFFF831C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR4,     0xFFFF8320,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR4,      0xFFFF8324,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR5,        0xFFFF8380,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR5,        0xFFFF8384,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR5,       0xFFFF8388,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR5,        0xFFFF838C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER5,       0xFFFF8390,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR5,       0xFFFF8394,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR5,       0xFFFF8398,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR5,     0xFFFF839C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR5,     0xFFFF83A0,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR5,      0xFFFF83A4,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR6,        0xFFFF8400,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR6,        0xFFFF8404,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR6,       0xFFFF8408,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR6,        0xFFFF840C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER6,       0xFFFF8410,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR6,       0xFFFF8414,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR6,       0xFFFF8418,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR6,     0xFFFF841C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR6,     0xFFFF8420,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR6,      0xFFFF8424,__READ_WRITE ,__ldma_cntrx_bits);
__IO_REG32_BIT(LDMA_CR7,        0xFFFF8480,__WRITE      ,__ldma_crx_bits);
__IO_REG32_BIT(LDMA_MR7,        0xFFFF8484,__READ_WRITE ,__ldma_mrx_bits);
__IO_REG32_BIT(LDMA_CSR7,       0xFFFF8488,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_SR7,        0xFFFF848C,__READ       ,__ldma_srx_bits);
__IO_REG32_BIT(LDMA_IER7,       0xFFFF8490,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IDR7,       0xFFFF8494,__WRITE      ,__ldma_csrx_bits);
__IO_REG32_BIT(LDMA_IMR7,       0xFFFF8498,__READ       ,__ldma_csrx_bits);
__IO_REG32(    LDMA_ASRCR7,     0xFFFF849C,__READ_WRITE );
__IO_REG32(    LDMA_ADSTR7,     0xFFFF84A0,__READ_WRITE );
__IO_REG32_BIT(LDMA_CNTR7,      0xFFFF84A4,__READ_WRITE ,__ldma_cntrx_bits);

/***************************************************************************
 **
 **  PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0_ECR,        0xFFE08050,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM0_DCR,        0xFFE08054,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM0_PMSR,       0xFFE08058,__READ       ,__pwm_pmsr_bits);
__IO_REG32_BIT(PWM0_CR,         0xFFE08060,__WRITE      ,__pwm_cr_bits);
__IO_REG32_BIT(PWM0_MR,         0xFFE08064,__READ_WRITE ,__pwm_mr_bits);
__IO_REG32_BIT(PWM0_CSR,        0xFFE0806C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_SR,         0xFFE08070,__READ       ,__pwm_sr_bits);
__IO_REG32_BIT(PWM0_IER,        0xFFE08074,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_IDR,        0xFFE08078,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_IMR,        0xFFE0807C,__READ       ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_DLY0,       0xFFE08080,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM0_PUL0,       0xFFE08084,__READ_WRITE ,__pwm_pul_bits);
__IO_REG32_BIT(PWM0_DLY1,       0xFFE08088,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM0_PUL1,       0xFFE0808C,__READ_WRITE ,__pwm_pul_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_ECR,        0xFFE8C050,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM1_DCR,        0xFFE8C054,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM1_PMSR,       0xFFE8C058,__READ       ,__pwm_pmsr_bits);
__IO_REG32_BIT(PWM1_CR,         0xFFE8C060,__WRITE      ,__pwm_cr_bits);
__IO_REG32_BIT(PWM1_MR,         0xFFE8C064,__READ_WRITE ,__pwm_mr_bits);
__IO_REG32_BIT(PWM1_CSR,        0xFFE8C06C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_SR,         0xFFE8C070,__READ       ,__pwm_sr_bits);
__IO_REG32_BIT(PWM1_IER,        0xFFE8C074,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_IDR,        0xFFE8C078,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_IMR,        0xFFE8C07C,__READ       ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_DLY0,       0xFFE8C080,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM1_PUL0,       0xFFE8C084,__READ_WRITE ,__pwm_pul_bits);
__IO_REG32_BIT(PWM1_DLY1,       0xFFE8C088,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM1_PUL1,       0xFFE8C08C,__READ_WRITE ,__pwm_pul_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_ECR,        0xFFE90050,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM2_DCR,        0xFFE90054,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM2_PMSR,       0xFFE90058,__READ       ,__pwm_pmsr_bits);
__IO_REG32_BIT(PWM2_CR,         0xFFE90060,__WRITE      ,__pwm_cr_bits);
__IO_REG32_BIT(PWM2_MR,         0xFFE90064,__READ_WRITE ,__pwm_mr_bits);
__IO_REG32_BIT(PWM2_CSR,        0xFFE9006C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_SR,         0xFFE90070,__READ       ,__pwm_sr_bits);
__IO_REG32_BIT(PWM2_IER,        0xFFE90074,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_IDR,        0xFFE90078,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_IMR,        0xFFE9007C,__READ       ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_DLY0,       0xFFE90080,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM2_PUL0,       0xFFE90084,__READ_WRITE ,__pwm_pul_bits);
__IO_REG32_BIT(PWM2_DLY1,       0xFFE90088,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM2_PUL1,       0xFFE9008C,__READ_WRITE ,__pwm_pul_bits);

/***************************************************************************
 **
 **  PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM3_ECR,        0xFFE94050,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM3_DCR,        0xFFE94054,__WRITE      ,__pwm_ecr_bits);
__IO_REG32_BIT(PWM3_PMSR,       0xFFE94058,__READ       ,__pwm_pmsr_bits);
__IO_REG32_BIT(PWM3_CR,         0xFFE94060,__WRITE      ,__pwm_cr_bits);
__IO_REG32_BIT(PWM3_MR,         0xFFE94064,__READ_WRITE ,__pwm_mr_bits);
__IO_REG32_BIT(PWM3_CSR,        0xFFE9406C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_SR,         0xFFE94070,__READ       ,__pwm_sr_bits);
__IO_REG32_BIT(PWM3_IER,        0xFFE94074,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_IDR,        0xFFE94078,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_IMR,        0xFFE9407C,__READ       ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_DLY0,       0xFFE94080,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM3_PUL0,       0xFFE94084,__READ_WRITE ,__pwm_pul_bits);
__IO_REG32_BIT(PWM3_DLY1,       0xFFE94088,__READ_WRITE ,__pwm_dly_bits);
__IO_REG32_BIT(PWM3_PUL1,       0xFFE9408C,__READ_WRITE ,__pwm_pul_bits);

/***************************************************************************
 **
 **  SPI1 - 16bits
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_ECR,        0xFFE60050,__WRITE      ,__spi_ecr_bits);
__IO_REG32_BIT(SPI1_DCR,        0xFFE60054,__WRITE      ,__spi_ecr_bits);
__IO_REG32_BIT(SPI1_PMSR,       0xFFE60058,__READ       ,__spi_pmsr_bits);
__IO_REG32_BIT(SPI1_CR,         0xFFE60060,__WRITE      ,__spi_cr_bits);
__IO_REG32_BIT(SPI1_MR,         0xFFE60064,__READ_WRITE ,__spi_mr_bits);
__IO_REG32_BIT(SPI1_CSR,        0xFFE6006C,__WRITE      ,__spi_csr_bits);
__IO_REG32_BIT(SPI1_SR,         0xFFE60070,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI1_IER,        0xFFE60074,__WRITE      ,__spi_ier_bits);
__IO_REG32_BIT(SPI1_IDR,        0xFFE60078,__WRITE      ,__spi_ier_bits);
__IO_REG32_BIT(SPI1_IMR,        0xFFE6007C,__READ       ,__spi_ier_bits);
__IO_REG32_BIT(SPI1_RDR,        0xFFE60080,__READ       ,__spi_rdr_bits);
__IO_REG32_BIT(SPI1_TDR,        0xFFE60084,__WRITE      ,__spi_tdr_bits);
__IO_REG32_BIT(SPI1_SSR0,       0xFFE60090,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32_BIT(SPI1_SSR1,       0xFFE60094,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32_BIT(SPI1_SSR2,       0xFFE60098,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32_BIT(SPI1_SSR3,       0xFFE6009C,__READ_WRITE ,__spi_ssr_bits);

/***************************************************************************
 **
 **  SPI0 - 8bits
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_ECR,        0xFFE10050,__WRITE      ,__spi_ecr_bits);
__IO_REG32_BIT(SPI0_DCR,        0xFFE10054,__WRITE      ,__spi_ecr_bits);
__IO_REG32_BIT(SPI0_PMSR,       0xFFE10058,__READ       ,__spi_pmsr_bits);
__IO_REG32_BIT(SPI0_CR,         0xFFE10060,__WRITE      ,__spi_cr_bits);
__IO_REG32_BIT(SPI0_MR,         0xFFE10064,__READ_WRITE ,__spi8_mr_bits);
__IO_REG32_BIT(SPI0_CSR,        0xFFE1006C,__WRITE      ,__spi_csr_bits);
__IO_REG32_BIT(SPI0_SR,         0xFFE10070,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI0_IER,        0xFFE10074,__WRITE      ,__spi_ier_bits);
__IO_REG32_BIT(SPI0_IDR,        0xFFE10078,__WRITE      ,__spi_ier_bits);
__IO_REG32_BIT(SPI0_IMR,        0xFFE1007C,__READ       ,__spi_ier_bits);
__IO_REG32_BIT(SPI0_RDR,        0xFFE10080,__READ       ,__spi8_rdr_bits);
__IO_REG32_BIT(SPI0_TDR,        0xFFE10084,__WRITE      ,__spi8_tdr_bits);
__IO_REG32_BIT(SPI0_SSR,        0xFFE10090,__READ_WRITE ,__spi8_ssr_bits);

/***************************************************************************
 **
 **  ST0
 **
 ***************************************************************************/
__IO_REG32_BIT(ST0_ECR,         0xFFE20050,__WRITE      ,__st_ecr_bits);
__IO_REG32_BIT(ST0_DCR,         0xFFE20054,__WRITE      ,__st_ecr_bits);
__IO_REG32_BIT(ST0_PMSR,        0xFFE20058,__READ       ,__st_pmsr_bits);
__IO_REG32_BIT(ST0_CR,          0xFFE20060,__WRITE      ,__st_cr_bits);
__IO_REG32_BIT(ST0_CSR,         0xFFE2006C,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST0_SR,          0xFFE20070,__READ       ,__st_sr_bits);
__IO_REG32_BIT(ST0_IER,         0xFFE20074,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST0_IDR,         0xFFE20078,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST0_IMR,         0xFFE2007C,__READ       ,__st_csr_bits);
__IO_REG32_BIT(ST0_PR0,         0xFFE20080,__READ_WRITE ,__st_pr_bits);
__IO_REG32_BIT(ST0_CT0,         0xFFE20084,__READ_WRITE ,__st_ct_bits);
__IO_REG32_BIT(ST0_PR1,         0xFFE20088,__READ_WRITE ,__st_pr_bits);
__IO_REG32_BIT(ST0_CT1,         0xFFE2008C,__READ_WRITE ,__st_ct_bits);
__IO_REG32_BIT(ST0_CCV0,        0xFFE20200,__READ       ,__st_ccv_bits);
__IO_REG32_BIT(ST0_CCV1,        0xFFE20204,__READ       ,__st_ccv_bits);

/***************************************************************************
 **
 **  ST1
 **
 ***************************************************************************/
__IO_REG32_BIT(ST1_ECR,         0xFFE24050,__WRITE      ,__st_ecr_bits);
__IO_REG32_BIT(ST1_DCR,         0xFFE24054,__WRITE      ,__st_ecr_bits);
__IO_REG32_BIT(ST1_PMSR,        0xFFE24058,__READ       ,__st_pmsr_bits);
__IO_REG32_BIT(ST1_CR,          0xFFE24060,__WRITE      ,__st_cr_bits);
__IO_REG32_BIT(ST1_CSR,         0xFFE2406C,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST1_SR,          0xFFE24070,__READ       ,__st_sr_bits);
__IO_REG32_BIT(ST1_IER,         0xFFE24074,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST1_IDR,         0xFFE24078,__WRITE      ,__st_csr_bits);
__IO_REG32_BIT(ST1_IMR,         0xFFE2407C,__READ       ,__st_csr_bits);
__IO_REG32_BIT(ST1_PR0,         0xFFE24080,__READ_WRITE ,__st_pr_bits);
__IO_REG32_BIT(ST1_CT0,         0xFFE24084,__READ_WRITE ,__st_ct_bits);
__IO_REG32_BIT(ST1_PR1,         0xFFE24088,__READ_WRITE ,__st_pr_bits);
__IO_REG32_BIT(ST1_CT1,         0xFFE2408C,__READ_WRITE ,__st_ct_bits);
__IO_REG32_BIT(ST1_CCV0,        0xFFE24200,__READ       ,__st_ccv_bits);
__IO_REG32_BIT(ST1_CCV1,        0xFFE24204,__READ       ,__st_ccv_bits);

/***************************************************************************
 **
 **  SFM
 **
 ***************************************************************************/
__IO_REG32_BIT(SFM_CIDR,        0xFFFE4000,__READ       ,__sfm_cidr_bits);
__IO_REG32_BIT(SFM_ARCR,        0xFFFE4004,__READ       ,__sfm_arcr_bits);
__IO_REG32_BIT(SFM_MSR,         0xFFFE4008,__READ       ,__sfm_msr_bits);

/***************************************************************************
 **
 **  STT
 **
 ***************************************************************************/
__IO_REG32_BIT(STT_ECR,         0xFFE30050,__WRITE      ,__stt_ecr_bits);
__IO_REG32_BIT(STT_DCR,         0xFFE30054,__WRITE      ,__stt_ecr_bits);
__IO_REG32_BIT(STT_PMSR,        0xFFE30058,__READ       ,__stt_ecr_bits);
__IO_REG32_BIT(STT_CR,          0xFFE30060,__WRITE      ,__stt_cr_bits);
__IO_REG32_BIT(STT_MR,          0xFFE30064,__READ_WRITE ,__stt_mr_bits);
__IO_REG32_BIT(STT_CSR,         0xFFE3006C,__WRITE      ,__stt_csr_bits);
__IO_REG32_BIT(STT_SR,          0xFFE30070,__READ       ,__stt_sr_bits);
__IO_REG32_BIT(STT_IER,         0xFFE30074,__WRITE      ,__stt_csr_bits);
__IO_REG32_BIT(STT_IDR,         0xFFE30078,__WRITE      ,__stt_csr_bits);
__IO_REG32_BIT(STT_IMR,         0xFFE3007C,__READ       ,__stt_csr_bits);
__IO_REG32(    STT_CNT,         0xFFE30080,__READ_WRITE );
__IO_REG32(    STT_ALR,         0xFFE30084,__READ_WRITE );

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(UA0_ECR,         0xFFE28050,__WRITE      ,__us_ecr_bits);
__IO_REG32_BIT(UA0_DCR,         0xFFE28054,__WRITE      ,__us_ecr_bits);
__IO_REG32_BIT(UA0_PMSR,        0xFFE28058,__READ       ,__us_pmsr_bits);
__IO_REG32_BIT(UA0_CR,          0xFFE28060,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(UA0_MR,          0xFFE28064,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(UA0_CSR,         0xFFE2806C,__WRITE      ,__us_csr_bits);
__IO_REG32_BIT(UA0_SR,          0xFFE28070,__READ       ,__us_sr_bits);
__IO_REG32_BIT(UA0_IER,         0xFFE28074,__WRITE      ,__us_ier_bits);
__IO_REG32_BIT(UA0_IDR,         0xFFE28078,__WRITE      ,__us_ier_bits);
__IO_REG32_BIT(UA0_IMR,         0xFFE2807C,__READ       ,__us_ier_bits);
__IO_REG32_BIT(UA0_RHR,         0xFFE28080,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(UA0_THR,         0xFFE28084,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(UA0_BRGR,        0xFFE28088,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(UA0_RTOR,        0xFFE2808C,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(UA0_TTGR,        0xFFE28090,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(UA0_LIR,         0xFFE28094,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(UA0_DFWR0,       0xFFE28098,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(UA0_DFWR1,       0xFFE2809C,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(UA0_DFRR0,       0xFFE280A0,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(UA0_DFRR1,       0xFFE280A4,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(UA0_SBLR,        0xFFE280A8,__READ_WRITE ,__us_sblr_bits);

/***************************************************************************
 **
 **  USART0
 **
 ***************************************************************************/
__IO_REG32_BIT(US0_ECR,         0xFFE38050,__WRITE      ,__us_ecr_bits);
__IO_REG32_BIT(US0_DCR,         0xFFE38054,__WRITE      ,__us_ecr_bits);
__IO_REG32_BIT(US0_PMSR,        0xFFE38058,__READ       ,__us_pmsr_bits);
__IO_REG32_BIT(US0_CR,          0xFFE38060,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US0_MR,          0xFFE38064,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US0_CSR,         0xFFE3806C,__WRITE      ,__us_csr_bits);
__IO_REG32_BIT(US0_SR,          0xFFE38070,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US0_IER,         0xFFE38074,__WRITE      ,__us_ier_bits);
__IO_REG32_BIT(US0_IDR,         0xFFE38078,__WRITE      ,__us_ier_bits);
__IO_REG32_BIT(US0_IMR,         0xFFE3807C,__READ       ,__us_ier_bits);
__IO_REG32_BIT(US0_RHR,         0xFFE38080,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US0_THR,         0xFFE38084,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US0_BRGR,        0xFFE38088,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US0_RTOR,        0xFFE3808C,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US0_TTGR,        0xFFE38090,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US0_LIR,         0xFFE38094,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US0_DFWR0,       0xFFE38098,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFWR1,       0xFFE3809C,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_DFRR0,       0xFFE380A0,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFRR1,       0xFFE380A4,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_SBLR,        0xFFE380A8,__READ_WRITE ,__us_sblr_bits);

/***************************************************************************
 **
 **  WD
 **
 ***************************************************************************/
__IO_REG32_BIT(WD_ECR,          0xFFE14050,__WRITE      ,__wd_ecr_bits);
__IO_REG32_BIT(WD_DCR,          0xFFE14054,__WRITE      ,__wd_ecr_bits);
__IO_REG32_BIT(WD_PMSR,         0xFFE14058,__READ       ,__wd_pmsr_bits);
__IO_REG32_BIT(WD_CR,           0xFFE14060,__WRITE      ,__wd_cr_bits);
__IO_REG32_BIT(WD_MR,           0xFFE14064,__READ_WRITE ,__wd_mr_bits);
__IO_REG32_BIT(WD_OMR,          0xFFE14068,__READ_WRITE ,__wd_omr_bits);
__IO_REG32_BIT(WD_CSR,          0xFFE1406C,__WRITE      ,__wd_csr_bits);
__IO_REG32_BIT(WD_SR,           0xFFE14070,__READ       ,__wd_sr_bits);
__IO_REG32_BIT(WD_IER,          0xFFE14074,__WRITE      ,__wd_csr_bits);
__IO_REG32_BIT(WD_IDR,          0xFFE14078,__WRITE      ,__wd_csr_bits);
__IO_REG32_BIT(WD_IMR,          0xFFE1407C,__READ       ,__wd_csr_bits);
__IO_REG32_BIT(WD_PWR,          0xFFE14080,__READ_WRITE ,__wd_pwr_bits);
__IO_REG32_BIT(WD_CTR,          0xFFE14084,__READ       ,__wd_ctr_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  S3F4A2F interrupt source number
 **
 ***************************************************************************/
#define INO_INT_STOP_MODE 0x01    /* Stop mode controller                  */
#define INO_INT_LVD       0x02    /* Low voltage detector                  */
#define INO_INT_IFC       0x03    /* Interleave flash controller           */
#define INO_INT_PWM0      0x04    /* Pulse width modulation                */
#define INO_INT_ADC0      0x05    /* Analog to digital converter           */
#define INO_INT_SPI0      0x06    /* Serial peripheral Interface 8 bits    */
#define INO_INT_WD        0x07    /* Watchdog                              */
#define INO_INT_CAN0      0x08    /* Controller area network               */
#define INO_INT_GPT0CH0   0x09    /* General purpose timer channel  0      */
#define INO_INT_ST0       0x0a    /* Simple timer 0                        */
#define INO_INT_ST1       0x0b    /* Simple timer 1                        */
#define INO_INT_UART0     0x0c    /* UART 0                                */
#define INO_INT_I2C0      0x0d    /* Inter integrated circuit 0            */
#define INO_INT_USART0    0x14    /* USART 0                               */
#define INO_SWIRQ1        0x16    /* Software interrupt                    */
#define INO_INT_SPI1      0x18    /* Serial peripheral Interface16 bits    */
#define INO_INT_STT       0x19    /* Stamp timer                           */
#define INO_INT_GPT0CH1   0x1a    /* General purpose timer channel 1       */
#define INO_INT_GPT0CH2   0x1b    /* General purpose timer channel 2       */
#define INO_INT_LDMA      0x1c    /* Lite direct memory access             */
#define INO_INT_GPIO0     0x1d    /* General purpose I/O 0                 */
#define INO_INT_GPIO1     0x1e    /* General purpose I/O 1                 */
#define INO_INT_PWM1      0x21    /* Pulse width modulation 1              */
#define INO_INT_PWM2      0x22    /* Pulse width modulation 2              */
#define INO_INT_PWM3      0x23    /* Pulse width modulation 3              */
#define INO_SWIRQ2        0x27    /* Software interrupt 2                  */
#define INO_IRQ0          0x28    /* External interrupt 0                  */
#define INO_IRQ1          0x29    /* External interrupt 1                  */
#define INO_IRQ2          0x2A    /* External interrupt 2                  */
#define INO_IRQ3          0x2B    /* External interrupt 3                  */
#define INO_IRQ4          0x2C    /* External interrupt 4                  */
#define INO_IRQ5          0x2D    /* External interrupt 5                  */
#define INO_IRQ6          0x2E    /* External interrupt 6                  */
#define INO_IRQ7          0x2F    /* External interrupt 7                  */
#define INO_IRQ8          0x30    /* External interrupt 8                  */
#define INO_IRQ9          0x31    /* External interrupt 9                  */
#define INO_IRQ10         0x32    /* External interrupt 10                 */
#define INO_IRQ11         0x33    /* External interrupt 11                 */
#define INO_INT_STABLE    0x3B    /* Stable interrupt                      */
#define INO_SWIRQ3        0x3C    /* Software interrupt 3                  */
#define INO_SWIRQ4        0x3D    /* Software interrupt 4                  */
#define INO_SWIRQ5        0x3E    /* Software interrupt 5                  */
#define INO_SWIRQ6        0x3F    /* Software interrupt 6                  */

#endif    /* __S3F4A2F_H */
