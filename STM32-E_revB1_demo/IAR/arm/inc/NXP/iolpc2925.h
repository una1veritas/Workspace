/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC2925
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2008
 **
 **    $Revision: 36510 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC2925_H
#define __IOLPC2925_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC2925 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/
#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

/* FCTR register */
typedef struct{
__REG32 FS_CS       : 1;
__REG32 FS_WRE      : 1;
__REG32 FS_WEB      : 1;
__REG32             : 1;
__REG32 FS_DCR      : 1;
__REG32 FS_RLD      : 1;
__REG32 FS_ISS      : 1;
__REG32 FS_WPB      : 1;
__REG32             : 1;
__REG32 FS_PD       : 1;
__REG32 FS_PDL      : 1;
__REG32 FS_RLS      : 1;
__REG32 FS_PROGREQ  : 1;
__REG32 FS_CACHEBYP : 1;
__REG32 FS_CACHECLR : 1;
__REG32 FS_LOADREQ  : 1;
__REG32             :16;
} __fctr_bits;

/* FPTR register */
typedef struct{
__REG32 TR          :15;
__REG32 EN_T        : 1;
__REG32             :16;
} __fptr_bits;

/* FTCTR register */
typedef struct{
__REG32 FS_ALL1     : 1;
__REG32 FS_ALL0     : 1;
__REG32 FS_CKB      : 1;
__REG32 FS_INVCKB   : 1;
__REG32 F_MSS       : 1;
__REG32 F_DCT       : 1;
__REG32 F_SEW       : 1;
__REG32 F_SOW       : 1;
__REG32 F_DAW       : 1;
__REG32 F_FEG       : 1;
__REG32 F_FOG       : 1;
__REG32 F_FAS       : 1;
__REG32 FS_HVSS0    : 1;
__REG32 FS_HVSS1    : 1;
__REG32 F_FSS       : 1;
__REG32 F_RRS       : 1;
__REG32 F_IVP       : 1;
__REG32 F_EVP       : 1;
__REG32 F_MARW      : 1;
__REG32 F_EVN       : 1;
__REG32 F_MARI      : 1;
__REG32 F_TRIP1     : 1;
__REG32 F_TRIP5     : 1;
__REG32 F_TRIP20    : 1;
__REG32 FS_HVSS2    : 1;
__REG32 F_ECCTST    : 1;
__REG32 FS_FFR      : 1;
__REG32 FS_CHECK_EN : 1;
__REG32 FS_BYPASS_W : 1;
__REG32 FS_BYPASS_R : 1;
__REG32 FS_TDC      : 1;
__REG32 FS_PARCEL   : 1;
} __ftctr_bits;

/* FBWST register */
typedef struct{
__REG32 WST         : 8;
__REG32             : 6;
__REG32 SPECALWAYS  : 1;
__REG32 CACHE2EN    : 1;
__REG32             :16;
} __fbwst_bits;

/* FCRA register */
typedef struct{
__REG32 FCRA        :12;
__REG32             :20;
} __fcra_bits;

/* FMSSTART */
typedef struct{
__REG32 FMSSTART    :17;
__REG32             :15;
} __fmsstart_bits;

/* FMSSTOP register */
typedef struct{
__REG32 FMSSTOP     :17;
__REG32 MISR_START  : 1;
__REG32             :14;
} __fmsstop_bits;

/* FMC_INT_CLR_ENABLE */
/* FMC_INT_SET_ENABLE */
/* FMC_INT_ENABLE */
/* FMC_INT_STATUS */
/* FMC_INT_CLR_STATUS */
/* FMC_INT_SET_STATUS */
typedef struct{
__REG32 END_OF_ERASE : 1;
__REG32 END_OF_BURN  : 1;
__REG32 END_OF_MISR  : 1;
__REG32              :29;
} __fmc_int_bits;

/* EEPROM command register */
typedef struct{
__REG32 CMD          : 3;
__REG32 RDPREFETCH   : 1;
__REG32 PAR_ACCESS   : 1;
__REG32              :27;
} __eecmd_bits;

/* EEPROM address register */
typedef struct{
__REG32 ADDR         :14;
__REG32              :18;
} __eeaddr_bits;

/* EEPROM wait state register */
typedef struct{
__REG32 PHASE3       : 8;
__REG32 PHASE2       : 8;
__REG32 PHASE1       : 8;
__REG32              :8;
} __eewstate_bits;

/* EEPROM clock divider register */
typedef struct{
__REG32 CLKDIV       :16;
__REG32              :16;
} __eeclkdiv_bits;

/* EEPROM power down/DCM register */
typedef struct{
__REG32 PWRDWN       : 1;
__REG32              :31;
} __eepwrdwn_bits;

/*EEPROM BIST start address register*/
typedef struct{
__REG32 STARTA      :14;
__REG32             :18;
} __eemsstart_bits;

/* EEPROM BIST stop address register */
typedef struct{
__REG32 STOPA       :14;
__REG32             :16;
__REG32 DEVSEL      : 1;
__REG32 STRTBIST    : 1;
} __eemsstop_bits;

/* EEPROM signature register */
typedef struct{
__REG32 DATA_SIG    :16;
__REG32 PARITY_SIG  :16;
} __eemssig_bits;

/* SMBIDCYRn register */
typedef struct{
__REG32 IDCY         : 4;
__REG32              :28;
} __smbidcyrx_bits;

/* SMBWST1Rn register */
typedef struct{
__REG32 WST1         : 5;
__REG32              :27;
} __smbwst1rx_bits;

/* SMBWST2Rn register */
typedef struct{
__REG32 WST2         : 5;
__REG32              :27;
} __smbwst2rx_bits;

/* SMBWSTOENRn register */
typedef struct{
__REG32 WSTOEN       : 4;
__REG32              :28;
} __smbwstoenrx_bits;

/* SMBWSTWENRn register */
typedef struct{
__REG32 WSTWEN       : 4;
__REG32              :28;
} __smbwstwenrx_bits;

/* SMBCRn register */
typedef struct{
__REG32 RBLE         : 1;
__REG32              : 2;
__REG32 CSPOL        : 1;
__REG32 WP           : 1;
__REG32 BM           : 1;
__REG32 MW           : 2;
__REG32              :24;
} __smbcrx_bits;

/* SMBSRn register */
typedef struct{
__REG32              : 1;
__REG32 WRITEPROTERR : 1;
__REG32              :30;
} __smbsrx_bits;


/* FREQ_MON register */
typedef struct{
__REG32 RCNT         : 9;
__REG32 FCNT         :14;
__REG32 MEAS         : 1;
__REG32 CLK_SEL      : 8;
} __freq_mon_bits;

/* CGU0 RDET register */
typedef struct{
__REG32 LP_OSC_PRESENT  : 1;
__REG32 XTAL_PRESENT    : 1;
__REG32 PLL_PRESENT     : 1;
__REG32 PLL120_PRESENT  : 1;
__REG32 PLL240_PRESENT  : 1;
__REG32 FDIV0_PRESENT   : 1;
__REG32 FDIV1_PRESENT   : 1;
__REG32 FDIV2_PRESENT   : 1;
__REG32 FDIV3_PRESENT   : 1;
__REG32 FDIV4_PRESENT   : 1;
__REG32 FDIV5_PRESENT   : 1;
__REG32 FDIV6_PRESENT   : 1;
__REG32                 :20;
} __cgu0_rdet_bits;

/* CGU1 RDET register */
typedef struct{
__REG32 BASE_ICLK1_CLK_PRESENT  : 1;
__REG32 BASE_ICLK0_CLK_PRESENT  : 1;
__REG32 PLL_PRESENT             : 1;
__REG32 PLL120_PRESENT          : 1;
__REG32 PLL240_PRESENT          : 1;
__REG32 FDIV0_PRESENT           : 1;
__REG32                         :26;
} __cgu1_rdet_bits;

/* XTAL_OSC_STATUS register */
/* XTAL_OSC_CONTROL */
typedef struct{
__REG32 ENABLE        : 1;
__REG32 BYPASS        : 1;
__REG32 HF            : 1;
__REG32               :29;
} __xtal_osc_status_bits;

/* PLL status register */
typedef struct{
__REG32 LOCK          : 1;
__REG32               :31;
} __pll_status_bits;

/* PLL control register */
typedef struct{
__REG32 PD              : 1;
__REG32 BYPASS          : 1;
__REG32 P23EN           : 1;
__REG32                 : 4;
__REG32 DIRECT          : 1;
__REG32 PSEL            : 2;
__REG32                 : 1;
__REG32 AUTOBLOK        : 1;
__REG32                 : 4;
__REG32 MSEL            : 5;
__REG32                 : 3;
__REG32 CLK_SEL         : 8;
} __pll_control_bits;

/* FDIV_STATUS_n */
/* FDIV_CONTROL_n */
typedef struct{
__REG32 DENOMINATOR     :12;
__REG32 LOAD            :12;
__REG32 CLK_SEL         : 8;
} __fdiv_status_x_bits;

/* SAFE_CLK_STATUS, PCR_CLK_STATUS register */
typedef struct{
__REG32                 : 2;
__REG32 IDIV            : 3;
__REG32                 :27;
} __safe_clk_status_bits;

/* SAFE_CLK_CONF, PCR_CLK_CONF register */
typedef struct{
__REG32                 : 2;
__REG32 IDIV            : 3;
__REG32                 :19;
__REG32 CLK_SEL         : 8;
} __safe_clk_conf_bits;

/* XX_CLK_STATUS register*/
typedef struct{
__REG32 PD              : 1;
__REG32 RTX             : 1;
__REG32 IDIV            : 3;
__REG32                 :27;
} __xx_clk_status_bits;

/* XX_CLK_CONF register */
typedef struct{
__REG32 PD              : 1;
__REG32                 : 1;
__REG32 IDIV            : 3;
__REG32                 : 6;
__REG32 AUTOBLOK        : 1;
__REG32                 :12;
__REG32 CLK_SEL         : 8;
} __xx_clk_conf_bits;

/* CGU_INT_CLR_ENABLE */
/* CGU_INT_SET_ENABLE */
/* CGU_INT_ENABLE */
/* CGU_INT_STATUS */
/* CGU_INT_CLR_STATUS */
/* CGU_INT_SET_STATUS */
typedef struct{
__REG32 LP_OSC          : 1;
__REG32 CRYSTAL         : 1;
__REG32 PL160M          : 1;
__REG32 PL160M120       : 1;
__REG32 PL160M240       : 1;
__REG32 FDIV0           : 1;
__REG32 FDIV1           : 1;
__REG32 FDIV2           : 1;
__REG32 FDIV3           : 1;
__REG32 FDIV4           : 1;
__REG32 FDIV5           : 1;
__REG32 FDIV6           : 1;
__REG32                 :20;
} __cgu_int_bits;

/* BUS_DISABLE register */
typedef struct{
__REG32 RRBUS           : 1;
__REG32                 :31;
} __bus_disable_bits;

/* RESET_CONTROL0 register */
typedef struct{
__REG32                 : 1;
__REG32 RGU_RST_CTRL    : 1;
__REG32 PCR_RST_CTRL    : 1;
__REG32 COLD_RST_CTRL   : 1;
__REG32 WARM_RST_CTRL   : 1;
__REG32                 :27;
} __reset_ctrl0_bits;

/* RESET_CONTROL1 register */
typedef struct{
__REG32 SCU_RST_CTRL        : 1;
__REG32 CFID_RST_CTRL       : 1;
__REG32                     : 2;
__REG32 FMC_RST_CTRL        : 1;
__REG32 EMC_RST_CTRL        : 1;
__REG32                     : 1;
__REG32                     : 1;
__REG32 GESS_A2V_RST_CTRL   : 1;
__REG32 PESS_A2V_RST_CTRL   : 1;
__REG32 GPIO_RST_CTRL       : 1;
__REG32 UART_RST_CTRL       : 1;
__REG32 TMR_RST_CTRL        : 1;
__REG32 SPI_RST_CTRL        : 1;
__REG32 IVNSS_A2V_RST_CTRL  : 1;
__REG32 IVNSS_CAN_RST_CTRL  : 1;
__REG32 IVNSS_LIN_RST_CTRL  : 1;
__REG32 MSCSS_A2V_RST_CTRL  : 1;
__REG32 MSCSS_PWM_RST_CTRL  : 1;
__REG32 MSCSS_ADC_RST_CTRL  : 1;
__REG32 MSCSS_TMR_RST_CTRL  : 1;
__REG32 IVNSS_I2C_RST_CTRL  : 1;
__REG32 MSCSS_QEI_RST_CTRL  : 1;
__REG32 DMA_RST_CTRL        : 1;
__REG32 USB_RST_CTRL        : 1;
__REG32                     : 3;
__REG32 VIC_RST_CTRL        : 1;
__REG32 AHB_RST_CTRL        : 1;
__REG32                     : 2;
} __reset_ctrl1_bits;

/* RESET_STATUS0 register */
typedef struct{
__REG32 POR_RST_STAT    : 2;
__REG32 RGU_RST_STAT    : 2;
__REG32 PCR_RST_STAT    : 2;
__REG32 COLD_RST_STAT   : 2;
__REG32 WARM_RST_STAT   : 2;
__REG32                 :22;
} __reset_status0_bits;

/* RESET_STATUS2 register bit description */
typedef struct{
__REG32 SCU_RST_STAT        : 2;
__REG32 CFID_RST_STAT       : 2;
__REG32                     : 4;
__REG32 FMC_RST_STAT        : 2;
__REG32 EMC_RST_STAT        : 2;
__REG32                     : 2;
__REG32                     : 2;
__REG32 GESS_A2V_RST_STAT   : 2;
__REG32 PESS_A2V_RST_STAT   : 2;
__REG32 GPIO_RST_STAT       : 2;
__REG32 UART_RST_STAT       : 2;
__REG32 TMR_RST_STAT        : 2;
__REG32 SPI_RST_STAT        : 2;
__REG32 IVNSS_A2V_RST_STAT  : 2;
__REG32 IVNSS_CAN_RST_STAT  : 2;
} __reset_status2_bits;

/* RESET_STATUS3 register */
typedef struct{
__REG32 IVNSS_LIN_RST_STAT  : 2;
__REG32 MSCSS_A2V_RST_STAT  : 2;
__REG32 MSCSS_PWM_RST_STAT  : 2;
__REG32 MSCSS_ADC_RST_STAT  : 2;
__REG32 MSCSS_TMR_RST_STAT  : 2;
__REG32 IVNSCC_I2C_STAT     : 2;
__REG32 MSCSS_QEI_STAT      : 2;
__REG32 DMA_STAT            : 2;
__REG32 USB_STAT            : 2;
__REG32                     : 6;
__REG32 VIC_RST_STAT        : 2;
__REG32 AHB_RST_STAT        : 2;
__REG32                     : 4;
} __reset_status3_bits;

/* RST_ACTIVE_STATUS0 register */
typedef struct{
__REG32 POR_RST_STAT      : 1;
__REG32 RGU_RST_STAT      : 1;
__REG32 PCR_RST_STAT      : 1;
__REG32 COLD_RST_STAT     : 1;
__REG32 WARM_RST_STAT     : 1;
__REG32                   :27;
} __rst_active_status0_bits;

/* RST_ACTIVE_STATUS1 register */
typedef struct{
__REG32 SCU_RST_STAT          : 1;
__REG32 CFID_RST_STAT         : 1;
__REG32                       : 2;
__REG32 FMC_RST_STAT          : 1;
__REG32 EMC_RST_STAT          : 1;
__REG32                       : 1;
__REG32                       : 1;
__REG32 GESS_A2V_RST_STAT     : 1;
__REG32 PESS_A2V_RST_STAT     : 1;
__REG32 GPIO_RST_STAT         : 1;
__REG32 UART_RST_STAT         : 1;
__REG32 TMR_RST_STAT          : 1;
__REG32 SPI_RST_STAT          : 1;
__REG32 IVNSS_A2V_RST_STAT    : 1;
__REG32 IVNSS_CAN_RST_STAT    : 1;
__REG32 IVNSS_LIN_RST_STAT    : 1;
__REG32 MSCSS_A2V_RST_STAT    : 1;
__REG32 MSCSS_PWM_RST_STAT    : 1;
__REG32 MSCSS_ADC_RST_STAT    : 1;
__REG32 MSCSS_TMR_RST_STAT    : 1;
__REG32 IVNSS_I2C_RST_STAT    : 1;
__REG32 MSCSS_QEI_RST_STAT    : 1;
__REG32 DMA_RST_STAT          : 1;
__REG32 USB_RST_STAT          : 1;
__REG32                       : 3;
__REG32 VIC_RST_STAT          : 1;
__REG32 AHB_RST_STAT          : 1;
__REG32                       : 2;
} __rst_active_status1_bits;

/* RGU_RST_SRC register */
typedef struct{
__REG32 POR               : 1;
__REG32 RSTN_PIN          : 1;
__REG32                   :30;
} __rgu_rst_src_bits;

/* PCR_RST_SRC register */
typedef struct{
__REG32                   : 2;
__REG32 RGU               : 1;
__REG32 WDT_TMR           : 1;
__REG32                   :28;
} __pcr_rst_src_bits;

/* COLD_RST_SRC register */
typedef struct{
__REG32                   : 4;
__REG32 PCR               : 1;
__REG32                   :27;
} __cold_rst_src_bits;

/*  
WARM_RST_SRC,
SCU_RST_SRC, 
CFID_RST_SRC,
FMC_RST_SRC, 
EMC_RST_SRC, 
SMC_RST_SRC, 
*/
typedef struct{
__REG32                   : 5;
__REG32 COLD              : 1;
__REG32                   :26;
} __warm_rst_src_bits;

/*  
GESS_A2V_RST_SRC, 
PESS_A2V_RST_SRC, 
GPIO_RST_SRC,     
UART_RST_SRC,     
TMR_RST_SRC,      
SPI_RST_SRC,      
IVNSS_A2V_RST_SRC,
IVNSS_CAN_RST_SRC,
IVNSS_LIN_RST_SRC,
MSCSS_A2V_RST_SRC,
MSCSS_PWM_RST_SRC,
MSCSS_ADC_RST_SRC,
MSCSS_TMR_RST_SRC,
VIC_RST_SRC,      
AHB_RST_SRC,      
*/
typedef struct{
__REG32                   : 6;
__REG32 WARM              : 1;
__REG32                   :25;
} __gess_a2v_rst_src_bits;

/* PM register */
typedef struct{
__REG32 PD                : 1;
__REG32                   :31;
} __PM_bits;

/* Base-clock status */
typedef struct{
__REG32 BASE0_STAT        : 1;
__REG32 BASE1_STAT        : 1;
__REG32 BASE2_STAT        : 1;
__REG32 BASE3_STAT        : 1;
__REG32 BASE4_STAT        : 1;
__REG32 BASE5_STAT        : 1;
__REG32 BASE6_STAT        : 1;
__REG32 BASE7_STAT        : 1;
__REG32 BASE8_STAT        : 1;
__REG32 BASE9_STAT        : 1;
__REG32 BASE10_STAT       : 1;
__REG32 BASE11_INT0       : 1;
__REG32 BASE12_INT1       : 1;
__REG32                   :19;
} __BASE_STAT_bits;

/* CLK_CFG_*** register */
typedef struct{
__REG32 RUN               : 1;
__REG32 AUTO              : 1;
__REG32 WAKEUP            : 1;
__REG32                   :29;
} __CLK_CFG_xx_bits;

/* CLK_STAT_*** register */
typedef struct{
__REG32 RS                : 1;
__REG32 AS                : 1;
__REG32 WS                : 1;
__REG32                   : 5;
__REG32 SM                : 2;
__REG32                   :22;
} __CLK_STAT_xx_bits;

/* SFSPn_m register */
typedef struct{
__REG32 FUNC_SEL          : 2;
__REG32 PAD_TYPE          : 3;
__REG32                   :27;
} __sfspx_y_bits;

/* SFSP5_18 register */
typedef struct{
__REG32 FUNC_SEL          : 2;
__REG32                   : 2;
__REG32 VBUS              : 1;
__REG32                   :27;
} __sfsp5_18_bits;

/* SEC_DIS register*/
/* SEC_STA register*/
typedef struct{
__REG32                   : 1;
__REG32 DIS               : 1;
__REG32                   :30;
} __sec_dis_bits;

/*Shadow memory registers*/
typedef struct{
__REG32                   :10;
__REG32 SMMSA             :22;
} __ssmmx_bits;

/*AHB master priority registers*/
typedef struct{
__REG32 PRIO              : 3;
__REG32                   :29;
} __smpx_bits;

/* CHIPID register */
typedef struct{
__REG32                   : 1;
__REG32 MANUFACTURER_ID   :11;
__REG32 PART_NR           :16;
__REG32 VERSION           : 4;
} __chipid_bits;

/* FEAT0 register */
typedef struct{
__REG32 PACKAGE_ID        : 4;
__REG32                   :28;
} __feat0_bits;

/* FEAT3 register */
typedef struct{
__REG32                   :31;
__REG32 JTAGSEC           : 1;
} __feat3_bits;

/* PEND register */
typedef struct{
__REG32 PEND0         : 1;
__REG32 PEND1         : 1;
__REG32 PEND2         : 1;
__REG32 PEND3         : 1;
__REG32 PEND4         : 1;
__REG32 PEND5         : 1;
__REG32 PEND6         : 1;
__REG32 PEND7         : 1;
__REG32 PEND8         : 1;
__REG32 PEND9         : 1;
__REG32 PEND10        : 1;
__REG32 PEND11        : 1;
__REG32 PEND12        : 1;
__REG32 PEND13        : 1;
__REG32 PEND14        : 1;
__REG32 PEND15        : 1;
__REG32 PEND16        : 1;
__REG32 PEND17        : 1;
__REG32 PEND18        : 1;
__REG32 PEND19        : 1;
__REG32 PEND20        : 1;
__REG32 PEND21        : 1;
__REG32 PEND22        : 1;
__REG32 PEND23        : 1;
__REG32 PEND24        : 1;
__REG32 PEND25        : 1;
__REG32 PEND26        : 1;
__REG32               : 5;
} __pend_bits;

/* INT_CLR register */
typedef struct{
__REG32 INT_CLR0         : 1;
__REG32 INT_CLR1         : 1;
__REG32 INT_CLR2         : 1;
__REG32 INT_CLR3         : 1;
__REG32 INT_CLR4         : 1;
__REG32 INT_CLR5         : 1;
__REG32 INT_CLR6         : 1;
__REG32 INT_CLR7         : 1;
__REG32 INT_CLR8         : 1;
__REG32 INT_CLR9         : 1;
__REG32 INT_CLR10        : 1;
__REG32 INT_CLR11        : 1;
__REG32 INT_CLR12        : 1;
__REG32 INT_CLR13        : 1;
__REG32 INT_CLR14        : 1;
__REG32 INT_CLR15        : 1;
__REG32 INT_CLR16        : 1;
__REG32 INT_CLR17        : 1;
__REG32 INT_CLR18        : 1;
__REG32 INT_CLR19        : 1;
__REG32 INT_CLR20        : 1;
__REG32 INT_CLR21        : 1;
__REG32 INT_CLR22        : 1;
__REG32 INT_CLR23        : 1;
__REG32 INT_CLR24        : 1;
__REG32 INT_CLR25        : 1;
__REG32 INT_CLR26        : 1;
__REG32                  : 5;
} __int_clr_bits;

/* INT_SET register */
typedef struct{
__REG32 INT_SET0         : 1;
__REG32 INT_SET1         : 1;
__REG32 INT_SET2         : 1;
__REG32 INT_SET3         : 1;
__REG32 INT_SET4         : 1;
__REG32 INT_SET5         : 1;
__REG32 INT_SET6         : 1;
__REG32 INT_SET7         : 1;
__REG32 INT_SET8         : 1;
__REG32 INT_SET9         : 1;
__REG32 INT_SET10        : 1;
__REG32 INT_SET11        : 1;
__REG32 INT_SET12        : 1;
__REG32 INT_SET13        : 1;
__REG32 INT_SET14        : 1;
__REG32 INT_SET15        : 1;
__REG32 INT_SET16        : 1;
__REG32 INT_SET17        : 1;
__REG32 INT_SET18        : 1;
__REG32 INT_SET19        : 1;
__REG32 INT_SET20        : 1;
__REG32 INT_SET21        : 1;
__REG32 INT_SET22        : 1;
__REG32 INT_SET23        : 1;
__REG32 INT_SET24        : 1;
__REG32 INT_SET25        : 1;
__REG32 INT_SET26        : 1;
__REG32                  : 5;
} __int_set_bits;

/* MASK register */
typedef struct{
__REG32 MASK0         : 1;
__REG32 MASK1         : 1;
__REG32 MASK2         : 1;
__REG32 MASK3         : 1;
__REG32 MASK4         : 1;
__REG32 MASK5         : 1;
__REG32 MASK6         : 1;
__REG32 MASK7         : 1;
__REG32 MASK8         : 1;
__REG32 MASK9         : 1;
__REG32 MASK10        : 1;
__REG32 MASK11        : 1;
__REG32 MASK12        : 1;
__REG32 MASK13        : 1;
__REG32 MASK14        : 1;
__REG32 MASK15        : 1;
__REG32 MASK16        : 1;
__REG32 MASK17        : 1;
__REG32 MASK18        : 1;
__REG32 MASK19        : 1;
__REG32 MASK20        : 1;
__REG32 MASK21        : 1;
__REG32 MASK22        : 1;
__REG32 MASK23        : 1;
__REG32 MASK24        : 1;
__REG32 MASK25        : 1;
__REG32 MASK26        : 1;
__REG32               : 5;
} __mask_bits;

/* MASK_CLR register */
typedef struct{
__REG32 MASK_CLR0         : 1;
__REG32 MASK_CLR1         : 1;
__REG32 MASK_CLR2         : 1;
__REG32 MASK_CLR3         : 1;
__REG32 MASK_CLR4         : 1;
__REG32 MASK_CLR5         : 1;
__REG32 MASK_CLR6         : 1;
__REG32 MASK_CLR7         : 1;
__REG32 MASK_CLR8         : 1;
__REG32 MASK_CLR9         : 1;
__REG32 MASK_CLR10        : 1;
__REG32 MASK_CLR11        : 1;
__REG32 MASK_CLR12        : 1;
__REG32 MASK_CLR13        : 1;
__REG32 MASK_CLR14        : 1;
__REG32 MASK_CLR15        : 1;
__REG32 MASK_CLR16        : 1;
__REG32 MASK_CLR17        : 1;
__REG32 MASK_CLR18        : 1;
__REG32 MASK_CLR19        : 1;
__REG32 MASK_CLR20        : 1;
__REG32 MASK_CLR21        : 1;
__REG32 MASK_CLR22        : 1;
__REG32 MASK_CLR23        : 1;
__REG32 MASK_CLR24        : 1;
__REG32 MASK_CLR25        : 1;
__REG32 MASK_CLR26        : 1;
__REG32                   : 5;
} __mask_clr_bits;

/* MASK_SET register */
typedef struct{
__REG32 MASK_SET0         : 1;
__REG32 MASK_SET1         : 1;
__REG32 MASK_SET2         : 1;
__REG32 MASK_SET3         : 1;
__REG32 MASK_SET4         : 1;
__REG32 MASK_SET5         : 1;
__REG32 MASK_SET6         : 1;
__REG32 MASK_SET7         : 1;
__REG32 MASK_SET8         : 1;
__REG32 MASK_SET9         : 1;
__REG32 MASK_SET10        : 1;
__REG32 MASK_SET11        : 1;
__REG32 MASK_SET12        : 1;
__REG32 MASK_SET13        : 1;
__REG32 MASK_SET14        : 1;
__REG32 MASK_SET15        : 1;
__REG32 MASK_SET16        : 1;
__REG32 MASK_SET17        : 1;
__REG32 MASK_SET18        : 1;
__REG32 MASK_SET19        : 1;
__REG32 MASK_SET20        : 1;
__REG32 MASK_SET21        : 1;
__REG32 MASK_SET22        : 1;
__REG32 MASK_SET23        : 1;
__REG32 MASK_SET24        : 1;
__REG32 MASK_SET25        : 1;
__REG32 MASK_SET26        : 1;
__REG32                   : 5;
} __mask_set_bits;

/* APR register */
typedef struct{
__REG32 APR0         : 1;
__REG32 APR1         : 1;
__REG32 APR2         : 1;
__REG32 APR3         : 1;
__REG32 APR4         : 1;
__REG32 APR5         : 1;
__REG32 APR6         : 1;
__REG32 APR7         : 1;
__REG32 APR8         : 1;
__REG32 APR9         : 1;
__REG32 APR10        : 1;
__REG32 APR11        : 1;
__REG32 APR12        : 1;
__REG32 APR13        : 1;
__REG32 APR14        : 1;
__REG32 APR15        : 1;
__REG32 APR16        : 1;
__REG32 APR17        : 1;
__REG32 APR18        : 1;
__REG32 APR19        : 1;
__REG32 APR20        : 1;
__REG32 APR21        : 1;
__REG32 APR22        : 1;
__REG32 APR23        : 1;
__REG32 APR24        : 1;
__REG32 APR25        : 1;
__REG32 APR26        : 1;
__REG32              : 5;
} __apr_bits;

/* ATR Register */
typedef struct{
__REG32 ATR0         : 1;
__REG32 ATR1         : 1;
__REG32 ATR2         : 1;
__REG32 ATR3         : 1;
__REG32 ATR4         : 1;
__REG32 ATR5         : 1;
__REG32 ATR6         : 1;
__REG32 ATR7         : 1;
__REG32 ATR8         : 1;
__REG32 ATR9         : 1;
__REG32 ATR10        : 1;
__REG32 ATR11        : 1;
__REG32 ATR12        : 1;
__REG32 ATR13        : 1;
__REG32 ATR14        : 1;
__REG32 ATR15        : 1;
__REG32 ATR16        : 1;
__REG32 ATR17        : 1;
__REG32 ATR18        : 1;
__REG32 ATR19        : 1;
__REG32 ATR20        : 1;
__REG32 ATR21        : 1;
__REG32 ATR22        : 1;
__REG32 ATR23        : 1;
__REG32 ATR24        : 1;
__REG32 ATR25        : 1;
__REG32 ATR26        : 1;
__REG32              : 5;
} __atr_bits;

/* RSR Register */
typedef struct{
__REG32 RSR0         : 1;
__REG32 RSR1         : 1;
__REG32 RSR2         : 1;
__REG32 RSR3         : 1;
__REG32 RSR4         : 1;
__REG32 RSR5         : 1;
__REG32 RSR6         : 1;
__REG32 RSR7         : 1;
__REG32 RSR8         : 1;
__REG32 RSR9         : 1;
__REG32 RSR10        : 1;
__REG32 RSR11        : 1;
__REG32 RSR12        : 1;
__REG32 RSR13        : 1;
__REG32 RSR14        : 1;
__REG32 RSR15        : 1;
__REG32 RSR16        : 1;
__REG32 RSR17        : 1;
__REG32 RSR18        : 1;
__REG32 RSR19        : 1;
__REG32 RSR20        : 1;
__REG32 RSR21        : 1;
__REG32 RSR22        : 1;
__REG32 RSR23        : 1;
__REG32 RSR24        : 1;
__REG32 RSR25        : 1;
__REG32 RSR26        : 1;
__REG32              : 5;
} __rsr_bits;

/* SPI configuration register */
typedef struct{
__REG32 SPI_ENABLE      : 1;
__REG32 MS_MODE         : 1;
__REG32 LOOPBACK_MODE   : 1;
__REG32 TRANSMIT_MODE   : 1;
__REG32 SLAVE_DISABLE   : 1;
__REG32 TIMER_TRIGGER   : 1;
__REG32 SOFTWARE_RESET  : 1;
__REG32 UPDATE_ENABLE   : 1;
__REG32                 : 8;
__REG32 INTER_SLAVE_DLY :16;
} __spix_config_bits;

/* SPI slave-enable register */
typedef struct{
__REG32 SLV_ENABLE_0    : 2;
__REG32 SLV_ENABLE_1    : 2;
__REG32 SLV_ENABLE_2    : 2;
__REG32 SLV_ENABLE_3    : 2;
__REG32                 :24;
} __spix_slv_enable_bits;

/* SPI transmit-FIFO flush register */
typedef struct{
__REG32 TX_FIFO_FLUSH   : 1;
__REG32                 :31;
} __spix_tx_fifo_flush_bits;

/* SPI FIFO data register */
typedef struct{
__REG32 FIFO_DATA       :16;
__REG32                 :16;
} __spix_fifo_data_bits;

/* SPI receive FIFO POP register */
typedef struct{
__REG32 RX_FIFO_POP     : 1;
__REG32                 :31;
} __spix_rx_fifo_pop_bits;

/* SPI receive-FIFO read-mode register */
typedef struct{
__REG32 RX_FIFO_PROTECT : 1;
__REG32                 :31;
} __spix_rx_fifo_readmode_bits;

/* SPI DMA settings register */
typedef struct{
__REG32 RX_DMA_ENABLE   : 1;
__REG32 TX_DMA_ENABLE   : 1;
__REG32 RX_DMA_BURST    : 3;
__REG32 TX_DMA_BURST    : 3;
__REG32                 :24;
} __spix_dma_settings_bits;

/* SPI status register (Status) */
typedef struct{
__REG32 TX_FIFO_EMPTY   : 1;
__REG32 TX_FIFO_FULL    : 1;
__REG32 RX_FIFO_EMPTY   : 1;
__REG32 RX_FIFO_FULL    : 1;
__REG32 SPI_BUSY        : 1;
__REG32 SMS_MODE_BUSY   : 1;
__REG32                 :26;
} __spix_status_bits;

/* SPI slave-settings 1 register */
typedef struct{
__REG32 CLK_DIVISOR1         : 8;
__REG32 CLK_DIVISOR2         : 8;
__REG32 NUMBER_WORDS         : 8;
__REG32 INTER_TRANSFER_DLY   : 8;
} __spix_slvy_settings1_bits;

/* SPI slave-settings 2 register */
typedef struct{
__REG32 WORDSIZE        : 5;
__REG32 SPH             : 1;
__REG32 SPO             : 1;
__REG32 TRANSFER_FORMAT : 1;
__REG32 CS_VALUE        : 1;
__REG32 PRE_POST_CS_DLY : 8;
__REG32                 :15;
} __spix_slvy_settings2_bits;

/* SPI FIFO interrupt threshold register */
typedef struct{
__REG32 RX_THRESHOLD    : 8;
__REG32 TX_THRESHOLD    : 8;
__REG32                 :16;
} __spix_int_threshold_bits;

/* SPI_INT_CLR_ENABLE */
/* SPI_INT_SET_ENABLE */
/* SPI_INT_ENABLE */
/* SPI_INT_STATUS */
/* SPI_INT_CLR_STATUS */
/* SPI_INT_SET_STATUS */
typedef struct{
__REG32 OV              : 1;
__REG32 TO              : 1;
__REG32 RX              : 1;
__REG32 TX              : 1;
__REG32 SMS             : 1;
__REG32                 :27;
} __spix_int_bits;

/* Watchdog timer-control register */
typedef struct{
__REG32 COUNTER_ENABLE  : 1;
__REG32 COUNTER_RESET   : 1;
__REG32 PAUSE_ENABLE    : 1;
__REG32 WD_KEY          :29;
} __wd_tcr_bits;

/* Watchdog debug register */
typedef struct{
__REG32 WD_RST_DIS      : 1;
__REG32 WD_KEY          :31;
} __wd_debug_bits;

/* WDT_INT_CLR_ENABLE */
/* WDT_INT_SET_ENABLE */
/* WDT_INT_ENABLE */
/* WDT_INT_STATUS */
/* WDT_INT_CLR_STATUS */
/* WDT_INT_SET_STATUS */
typedef struct{
__REG32                 : 8;
__REG32 WD              : 1;
__REG32                 :23;
} __wdt_int_bits;

/* UART Receive Buffer Register (RBR) */
/* UART Transmit Holding Register (THR) */
/* UART Divisor Latch Register  Low (DLL) */
typedef union {
  /*UxRBR*/
  struct {
    __REG32 RBR           : 8;
    __REG32               :24;
  } ;
  /*UxTHR*/
  struct {
    __REG32 THR           : 8;
    __REG32               :24;
  } ;
  /*UxDLL*/
  struct {
    __REG32 DLL           : 8;
    __REG32               :24;
  } ;
} __uartrbr_bits;

/* UART Interrupt Enable Register (IER) */
/* UART Divisor Latch MSB Register (DLM) */
typedef union {
  /*UxIER*/
  struct {
    __REG32 RDAIE         : 1;
    __REG32 THREIE        : 1;
    __REG32 RXLSIE        : 1;
    __REG32               : 5;
    __REG32 ABTOINTEN     : 1;
    __REG32 ABEOINTEN     : 1;
    __REG32               :22;
  } ;
  /*UxDLM*/
  struct {
    __REG32 DLM           : 8;
    __REG32               :24;
  } ;
} __uartier_bits;

/* UART Interrupt Identification Register (IIR) */
/* UART FIFO Control Register (FCR) */
typedef union {
  /*UxIIR*/
  struct {
    __REG32 IP            : 1;
    __REG32 IID           : 3;
    __REG32               : 2;
    __REG32 IIRFE         : 2;
    __REG32 ABEOINT       : 1;
    __REG32 ABTOINT       : 1;
    __REG32               :22;
  };
  /*UxFCR*/
  struct {
    __REG32 FCRFE         : 1;
    __REG32 RFR           : 1;
    __REG32 TFR           : 1;
    __REG32 DMAMODE       : 1;
    __REG32               : 2;
    __REG32 RTLS          : 2;
    __REG32               :24;
  };
} __uartiir_bits;

/* UART Line Control Register (LCR) */
typedef struct {
  __REG32 WLS             : 2;
  __REG32 SBS             : 1;
  __REG32 PE              : 1;
  __REG32 PS              : 2;
  __REG32 BC              : 1;
  __REG32 DLAB            : 1;
  __REG32                 :24;
} __uartlcr_bits;

/* UART Line Status Register (LSR) */
typedef struct {
  __REG32 DR              : 1;
  __REG32 OE              : 1;
  __REG32 PE              : 1;
  __REG32 FE              : 1;
  __REG32 BI              : 1;
  __REG32 THRE            : 1;
  __REG32 TEMT            : 1;
  __REG32 RXFE            : 1;
  __REG32                 :24;
} __uartlsr_bits;

/* UART scratch register */
typedef struct {
  __REG32 SCR             : 8;
  __REG32                 :24;
} __uartscr_bits;

/* UART Auto-baud Control Register */
typedef struct{
  __REG32 START           : 1;
  __REG32 MODE            : 1;
  __REG32 AUTORESTART     : 1;
  __REG32                 : 5;
  __REG32 ABEOINTCLR      : 1;
  __REG32 ABTOINTCLR      : 1;
  __REG32                 :22;
} __uartacr_bits;

/* UART Fractional Divider Register */
typedef struct{
  __REG32 DIVADDVAL       : 4;
  __REG32 MULVAL          : 4;
  __REG32                 :24;
} __uartfdr_bits;

/* Transmit Enable Register */
typedef struct{
  __REG32                 : 7;
  __REG32 TXEN            : 1;
  __REG32                 :24;
} __uartter_bits;

/* RS485 Control register */
typedef struct{
  __REG32 NMMEN           : 1;
  __REG32 RXDIS           : 1;
  __REG32 AADEN           : 1;
  __REG32                 :29;
} __uartrs485ctrl_bits;

/* RS485 Address Match register */
typedef struct{
  __REG32 ADRMATCH        : 8;
  __REG32                 :24;
} __uartadrmatch_bits;

/*  GPIO0 port input register */
typedef struct {
  __REG32 PINS0            : 1;
  __REG32 PINS1            : 1;
  __REG32 PINS2            : 1;
  __REG32 PINS3            : 1;
  __REG32 PINS4            : 1;
  __REG32 PINS5            : 1;
  __REG32 PINS6            : 1;
  __REG32 PINS7            : 1;
  __REG32 PINS8            : 1;
  __REG32 PINS9            : 1;
  __REG32 PINS10           : 1;
  __REG32 PINS11           : 1;
  __REG32 PINS12           : 1;
  __REG32 PINS13           : 1;
  __REG32 PINS14           : 1;
  __REG32 PINS15           : 1;
  __REG32 PINS16           : 1;
  __REG32 PINS17           : 1;
  __REG32 PINS18           : 1;
  __REG32 PINS19           : 1;
  __REG32 PINS20           : 1;
  __REG32 PINS21           : 1;
  __REG32 PINS22           : 1;
  __REG32 PINS23           : 1;
  __REG32 PINS24           : 1;
  __REG32 PINS25           : 1;
  __REG32 PINS26           : 1;
  __REG32 PINS27           : 1;
  __REG32 PINS28           : 1;
  __REG32 PINS29           : 1;
  __REG32 PINS30           : 1;
  __REG32 PINS31           : 1;
} __gpio0_pins_bits;

/*  GPIO0 port output register */
typedef struct {
  __REG32 OR0            : 1;
  __REG32 OR1            : 1;
  __REG32 OR2            : 1;
  __REG32 OR3            : 1;
  __REG32 OR4            : 1;
  __REG32 OR5            : 1;
  __REG32 OR6            : 1;
  __REG32 OR7            : 1;
  __REG32 OR8            : 1;
  __REG32 OR9            : 1;
  __REG32 OR10           : 1;
  __REG32 OR11           : 1;
  __REG32 OR12           : 1;
  __REG32 OR13           : 1;
  __REG32 OR14           : 1;
  __REG32 OR15           : 1;
  __REG32 OR16           : 1;
  __REG32 OR17           : 1;
  __REG32 OR18           : 1;
  __REG32 OR19           : 1;
  __REG32 OR20           : 1;
  __REG32 OR21           : 1;
  __REG32 OR22           : 1;
  __REG32 OR23           : 1;
  __REG32 OR24           : 1;
  __REG32 OR25           : 1;
  __REG32 OR26           : 1;
  __REG32 OR27           : 1;
  __REG32 OR28           : 1;
  __REG32 OR29           : 1;
  __REG32 OR30           : 1;
  __REG32 OR31           : 1;
} __gpio0_or_bits;

/*  GPIO0 port direction register */
typedef struct {
  __REG32 DR0            : 1;
  __REG32 DR1            : 1;
  __REG32 DR2            : 1;
  __REG32 DR3            : 1;
  __REG32 DR4            : 1;
  __REG32 DR5            : 1;
  __REG32 DR6            : 1;
  __REG32 DR7            : 1;
  __REG32 DR8            : 1;
  __REG32 DR9            : 1;
  __REG32 DR10           : 1;
  __REG32 DR11           : 1;
  __REG32 DR12           : 1;
  __REG32 DR13           : 1;
  __REG32 DR14           : 1;
  __REG32 DR15           : 1;
  __REG32 DR16           : 1;
  __REG32 DR17           : 1;
  __REG32 DR18           : 1;
  __REG32 DR19           : 1;
  __REG32 DR20           : 1;
  __REG32 DR21           : 1;
  __REG32 DR22           : 1;
  __REG32 DR23           : 1;
  __REG32 DR24           : 1;
  __REG32 DR25           : 1;
  __REG32 DR26           : 1;
  __REG32 DR27           : 1;
  __REG32 DR28           : 1;
  __REG32 DR29           : 1;
  __REG32 DR30           : 1;
  __REG32 DR31           : 1;
} __gpio0_dr_bits;

/* GPIO1 port input register */
typedef struct {
  __REG32 PINS0            : 1;
  __REG32 PINS1            : 1;
  __REG32 PINS2            : 1;
  __REG32 PINS3            : 1;
  __REG32 PINS4            : 1;
  __REG32 PINS5            : 1;
  __REG32 PINS6            : 1;
  __REG32 PINS7            : 1;
  __REG32 PINS8            : 1;
  __REG32 PINS9            : 1;
  __REG32 PINS10           : 1;
  __REG32 PINS11           : 1;
  __REG32 PINS12           : 1;
  __REG32 PINS13           : 1;
  __REG32 PINS14           : 1;
  __REG32 PINS15           : 1;
  __REG32 PINS16           : 1;
  __REG32 PINS17           : 1;
  __REG32 PINS18           : 1;
  __REG32 PINS19           : 1;
  __REG32 PINS20           : 1;
  __REG32 PINS21           : 1;
  __REG32 PINS22           : 1;
  __REG32 PINS23           : 1;
  __REG32 PINS24           : 1;
  __REG32 PINS25           : 1;
  __REG32 PINS26           : 1;
  __REG32 PINS27           : 1;
  __REG32                  : 4;
} __gpio1_pins_bits;

/*GPIO1 port output register */
typedef struct {
  __REG32 OR0            : 1;
  __REG32 OR1            : 1;
  __REG32 OR2            : 1;
  __REG32 OR3            : 1;
  __REG32 OR4            : 1;
  __REG32 OR5            : 1;
  __REG32 OR6            : 1;
  __REG32 OR7            : 1;
  __REG32 OR8            : 1;
  __REG32 OR9            : 1;
  __REG32 OR10           : 1;
  __REG32 OR11           : 1;
  __REG32 OR12           : 1;
  __REG32 OR13           : 1;
  __REG32 OR14           : 1;
  __REG32 OR15           : 1;
  __REG32 OR16           : 1;
  __REG32 OR17           : 1;
  __REG32 OR18           : 1;
  __REG32 OR19           : 1;
  __REG32 OR20           : 1;
  __REG32 OR21           : 1;
  __REG32 OR22           : 1;
  __REG32 OR23           : 1;
  __REG32 OR24           : 1;
  __REG32 OR25           : 1;
  __REG32 OR26           : 1;
  __REG32 OR27           : 1;
  __REG32                : 4;
} __gpio1_or_bits;

/*GPIO1 port direction register */
typedef struct {
  __REG32 DR0            : 1;
  __REG32 DR1            : 1;
  __REG32 DR2            : 1;
  __REG32 DR3            : 1;
  __REG32 DR4            : 1;
  __REG32 DR5            : 1;
  __REG32 DR6            : 1;
  __REG32 DR7            : 1;
  __REG32 DR8            : 1;
  __REG32 DR9            : 1;
  __REG32 DR10           : 1;
  __REG32 DR11           : 1;
  __REG32 DR12           : 1;
  __REG32 DR13           : 1;
  __REG32 DR14           : 1;
  __REG32 DR15           : 1;
  __REG32 DR16           : 1;
  __REG32 DR17           : 1;
  __REG32 DR18           : 1;
  __REG32 DR19           : 1;
  __REG32 DR20           : 1;
  __REG32 DR21           : 1;
  __REG32 DR22           : 1;
  __REG32 DR23           : 1;
  __REG32 DR24           : 1;
  __REG32 DR25           : 1;
  __REG32 DR26           : 1;
  __REG32 DR27           : 1;
  __REG32                : 4;
} __gpio1_dr_bits;

/* GPIO5 port input register */
typedef struct {
  __REG32                  :18;
  __REG32 PINS18           : 1;
  __REG32 PINS19           : 1;
  __REG32                  :12;
} __gpio5_pins_bits;

/* GPIO5 port output register */
typedef struct {
  __REG32                  :18;
  __REG32 OR18             : 1;
  __REG32 OR19             : 1;
  __REG32                  :12;
} __gpio5_or_bits;

/* GPIO5 port direction register */
typedef struct {
  __REG32                  :18;
  __REG32 DR18             : 1;
  __REG32 DR19             : 1;
  __REG32                  :12;
} __gpio5_dr_bits;

/* CAN acceptance-filter mode register */
typedef struct {
  __REG32 ACCOFF          :1;
  __REG32 ACCBP           :1;
  __REG32 EFCAN           :1;
  __REG32                 :29;
} __camode_bits;

/* CAN acceptance-filter standard-frame explicit start-address register */
typedef struct {
  __REG32                 : 2;
  __REG32 SFESA           :10;
  __REG32                 :20;
} __casfesa_bits;

/* CAN acceptance-filter standard-frame group start-address register */
typedef struct {
  __REG32                 : 2;
  __REG32 SFGSA           :10;
  __REG32                 :20;
} __casfgsa_bits;

/* CAN acceptance-filter extended-frame explicit start-address register */
typedef struct {
  __REG32                 : 2;
  __REG32 EFESA           :10;
  __REG32                 :20;
} __caefesa_bits;

/* CAN acceptance-filter extended-frame group start-address register */
typedef struct {
  __REG32                 : 2;
  __REG32 EFGSA           :10;
  __REG32                 :20;
} __caefgsa_bits;

/* CAN acceptance-filter end of look-up table address register */
typedef struct {
  __REG32                 : 2;
  __REG32 EOTA            :10;
  __REG32                 :20;
} __caeota_bits;

/* CAN acceptance filter look-up table error address register */
typedef struct {
  __REG32                 : 2;
  __REG32 LUTEA           : 9;
  __REG32                 :21;
} __calutea_bits;

/* CAN acceptance-filter look-up table error register */
typedef struct {
  __REG32 LUTE            : 1;
  __REG32                 :31;
} __calute_bits;

/* CAN controller central transmit-status register */
typedef struct {
  __REG32 TS0             : 1;
  __REG32 TS1             : 1;
  __REG32 TS2             : 1;
  __REG32 TS3             : 1;
  __REG32 TS4             : 1;
  __REG32 TS5             : 1;
  __REG32                 : 2;
  __REG32 TBS0            : 1;
  __REG32 TBS1            : 1;
  __REG32 TBS2            : 1;
  __REG32 TBS3            : 1;
  __REG32 TBS4            : 1;
  __REG32 TBS5            : 1;
  __REG32                 : 2;
  __REG32 TCS0            : 1;
  __REG32 TCS1            : 1;
  __REG32 TCS2            : 1;
  __REG32 TCS3            : 1;
  __REG32 TCS4            : 1;
  __REG32 TCS5            : 1;
  __REG32                 :10;
} __cccts_bits;

/* CAN controller central receive-status register */
typedef struct {
  __REG32 RS0             : 1;
  __REG32 RS1             : 1;
  __REG32 RS2             : 1;
  __REG32 RS3             : 1;
  __REG32 RS4             : 1;
  __REG32 RS5             : 1;
  __REG32                 : 2;
  __REG32 RBS0            : 1;
  __REG32 RBS1            : 1;
  __REG32 RBS2            : 1;
  __REG32 RBS3            : 1;
  __REG32 RBS4            : 1;
  __REG32 RBS5            : 1;
  __REG32                 : 2;
  __REG32 DOS0            : 1;
  __REG32 DOS1            : 1;
  __REG32 DOS2            : 1;
  __REG32 DOS3            : 1;
  __REG32 DOS4            : 1;
  __REG32 DOS5            : 1;
  __REG32                 :10;
} __cccrs_bits;

/* CAN controller central miscellaneous-status register */
typedef struct {
  __REG32 ES0             : 1;
  __REG32 ES1             : 1;
  __REG32 ES2             : 1;
  __REG32 ES3             : 1;
  __REG32 ES4             : 1;
  __REG32 ES5             : 1;
  __REG32                 : 2;
  __REG32 BS0             : 1;
  __REG32 BS1             : 1;
  __REG32 BS2             : 1;
  __REG32 BS3             : 1;
  __REG32 BS4             : 1;
  __REG32 BS5             : 1;
  __REG32                 :18;
} __cccms_bits;

/* CAN controller mode register */
typedef struct {
  __REG32 RM              :1;
  __REG32 LOM             :1;
  __REG32 STM             :1;
  __REG32 TPM             :1;
  __REG32                 :1;
  __REG32 RPM             :1;
  __REG32                 :26;
} __ccmode_bits;

/* CAN controller command register */
typedef struct {
  __REG32 TR              :1;
  __REG32 AT              :1;
  __REG32 RRB             :1;
  __REG32 CDO             :1;
  __REG32 SRR             :1;
  __REG32 STB1            :1;
  __REG32 STB2            :1;
  __REG32 STB3            :1;
  __REG32                 :24;
} __cccmd_bits;

/* CAN controller global status register */
typedef struct {
  __REG32 RBS              :1;
  __REG32 DOS              :1;
  __REG32 TBS              :1;
  __REG32 TCS              :1;
  __REG32 RS               :1;
  __REG32 TS               :1;
  __REG32 ES               :1;
  __REG32 BS               :1;
  __REG32                  :8;
  __REG32 RXERR            :8;
  __REG32 TXERR            :8;
} __ccgs_bits;

/* CAN controller interrupt and capture register */
typedef struct {
  __REG32 RI               :1;
  __REG32 TI1              :1;
  __REG32 EWI              :1;
  __REG32 DOI              :1;
  __REG32                  :1;
  __REG32 EPI              :1;
  __REG32 ALI              :1;
  __REG32 BEI              :1;
  __REG32 IDI              :1;
  __REG32 TI2              :1;
  __REG32 TI3              :1;
  __REG32                  :5;
  __REG32 ERRCC            :5;
  __REG32 ERRDIR           :1;
  __REG32 ERRT             :2;
  __REG32 ALCBIT           :5;
  __REG32                  :3;
} __ccic_bits;

/* CAN controller interrupt-enable register */
typedef struct {
  __REG32 RIE               :1;
  __REG32 TIE1              :1;
  __REG32 EWIE              :1;
  __REG32 DOIE              :1;
  __REG32                   :1;
  __REG32 EPIE              :1;
  __REG32 ALIE              :1;
  __REG32 BEIE              :1;
  __REG32 IDIE              :1;
  __REG32 TI2E              :1;
  __REG32 TI3E              :1;
  __REG32                   :21;
} __ccie_bits;

/* CAN controller bus timing register */
typedef struct {
  __REG32 BRP                :10;
  __REG32                    : 4;
  __REG32 SJW                : 2;
  __REG32 TSEG1              : 4;
  __REG32 TSEG2              : 3;
  __REG32 SAM                : 1;
  __REG32                    : 8;
} __ccbt_bits;

/* CAN controller error-warning limit register */
typedef struct {
  __REG32 EWL                : 8;
  __REG32                    :24;
} __ccewl_bits;

/* CAN controller status register */
typedef struct {
  __REG32 RBS                :1;
  __REG32 DOS                :1;
  __REG32 TBS1               :1;
  __REG32 TCS1               :1;
  __REG32 RS                 :1;
  __REG32 TS1                :1;
  __REG32 ES                 :1;
  __REG32 BS                 :1;
  __REG32 /*RBS*/            :1;
  __REG32 /*DOS*/            :1;
  __REG32 TBS2               :1;
  __REG32 TCS2               :1;
  __REG32 /*RS*/             :1;
  __REG32 TS2                :1;
  __REG32 /*ES*/             :1;
  __REG32 /*BS*/             :1;
  __REG32 /*RBS*/            :1;
  __REG32 /*DOS*/            :1;
  __REG32 TBS3               :1;
  __REG32 TCS3               :1;
  __REG32 /*RS*/             :1;
  __REG32 TS3                :1;
  __REG32 /*ES*/             :1;
  __REG32 /*BS*/             :1;
  __REG32                    :8;
} __ccstat_bits;

/* CAN controller receive-buffer message info register */
typedef struct {
  __REG32 IDI                :10;
  __REG32 BP                 :1;
  __REG32                    :5;
  __REG32 DLC                :4;
  __REG32                    :10;
  __REG32 RTR                :1;
  __REG32 FF                 :1;
} __ccrxbmi_bits;

/* CAN controller receive buffer identifier register */
typedef struct {
 __REG32 ID                 :29;
 __REG32                    : 3;
} __ccrxbid_bits;

/* CAN controller receive buffer data A register */
typedef struct {
  __REG32 DB1                 :8;
  __REG32 DB2                 :8;
  __REG32 DB3                 :8;
  __REG32 DB4                 :8;
} __ccrxbda_bits;

/* CAN rx data register B */
typedef struct {
  __REG32 DB5                 :8;
  __REG32 DB6                 :8;
  __REG32 DB7                 :8;
  __REG32 DB8                 :8;
} __ccrxbdb_bits;

/* CAN controller transmit-buffer message info registers */
typedef struct {
  __REG32 TXPRIO            :8;
  __REG32                   :8;
  __REG32 DLC               :4;
  __REG32                   :10;
  __REG32 RTR               :1;
  __REG32 FF                :1;
} __cctxbmi_bits;

/* LIN master-controller mode register */
typedef struct {
  __REG32 LRM                 : 1;
  __REG32                     : 6;
  __REG32 MODE                : 1;
  __REG32                     :24;
} __lmode_bits;

/* LIN master-controller configuration register */
typedef struct {
  __REG32 SBL                 : 3;
  __REG32 IBS                 : 2;
  __REG32                     : 1;
  __REG32 SWCS                : 1;
  __REG32 SWPA                : 1;
  __REG32                     :24;
} __lcfg_bits;

/* LIN master-controller command register */
typedef struct {
  __REG32 TR                  : 1;
  __REG32                     : 6;
  __REG32 SSB                 : 1;
  __REG32                     :24;
} __lcmd_bits;

/* LIN master-controller fractional baud rate generator register */
typedef struct {
  __REG32 INT                 :16;
  __REG32 FRAC                : 4;
  __REG32                     :12;
} __lfbrg_bits;

/* LIN master-controller status register */
typedef struct {
  __REG32 MR                  : 1;
  __REG32 MBA                 : 1;
  __REG32 HS                  : 1;
  __REG32 RS                  : 1;
  __REG32 TS                  : 1;
  __REG32 ES                  : 1;
  __REG32 IS                  : 1;
  __REG32                     : 1;
  __REG32 RLL                 : 1;
  __REG32 TTL                 : 1;
  __REG32                     :22;
} __lstat_bits;

/* LIN master-controller interrupt and capture register */
typedef struct {
  __REG32 RI                  : 1;
  __REG32 TI                  : 1;
  __REG32 BEI                 : 1;
  __REG32 CSI                 : 1;
  __REG32 NRI                 : 1;
  __REG32 RTLCEI              : 1;
  __REG32 WPI                 : 1;
  __REG32                     : 1;
  __REG32 EC                  : 4;
  __REG32                     :20;
} __lic_bits;

/* LIN master-controller interrupt enable register */
typedef struct {
  __REG32 RIE                 : 1;
  __REG32 TIE                 : 1;
  __REG32 BEIE                : 1;
  __REG32 CSIE                : 1;
  __REG32 NRIE                : 1;
  __REG32 RTLCEIE             : 1;
  __REG32 WPIE                : 1;
  __REG32                     :25;
} __lie_bits;

/* LIN master-controller checksum register */
typedef struct {
  __REG32 CS                  : 8;
  __REG32                     :24;
} __lcs_bits;

/* LIN master-controller time-out register */
typedef struct {
  __REG32 TO                  : 8;
  __REG32                     :24;
} __lto_bits;

/* LIN master-controller message buffer registers */
typedef struct {
  __REG32 ID                  : 6;
  __REG32 P0                  : 1;
  __REG32 P1                  : 1;
  __REG32                     : 8;
  __REG32 DLC                 : 5;
  __REG32                     : 3;
  __REG32 DD                  : 1;
  __REG32 CSID                : 1;
  __REG32                     : 6;
} __lid_bits;

/* LDATA register bit description */
typedef struct {
  __REG32 DF1                 : 8;
  __REG32 DF2                 : 8;
  __REG32 DF3                 : 8;
  __REG32 DF4                 : 8;
} __ldata_bits;

/* LDATB register bit description */
typedef struct {
  __REG32 DF5                 : 8;
  __REG32 DF6                 : 8;
  __REG32 DF7                 : 8;
  __REG32 DF8                 : 8;
} __ldatb_bits;

/* LDATC register bit description */
typedef struct {
  __REG32 DF9                 : 8;
  __REG32 DF10                : 8;
  __REG32 DF11                : 8;
  __REG32 DF12                : 8;
} __ldatc_bits;

/* LDATD register bit description */
typedef struct {
  __REG32 DF13                : 8;
  __REG32 DF14                : 8;
  __REG32 DF15                : 8;
  __REG32 DF16                : 8;
} __ldatd_bits;

/* I2C Control Set Register */
typedef struct{
__REG32       : 2;
__REG32 AA    : 1;
__REG32 SI    : 1;
__REG32 STO   : 1;
__REG32 STA   : 1;
__REG32 I2EN  : 1;
__REG32       :25;
} __i2conset_bits;

/* I2C Control Clear Register */
typedef struct{
__REG32        : 2;
__REG32 AAC    : 1;
__REG32 SIC    : 1;
__REG32        : 1;
__REG32 STAC   : 1;
__REG32 I2ENC  : 1;
__REG32        :25;
} __i2conclr_bits;

/* I2C Status Register */
typedef struct{
__REG32 STATUS  : 8;
__REG32         :24;
} __i2stat_bits;

/* I2C Data Register */
typedef struct{
__REG32 DATA  : 8;
__REG32       :24;
} __i2dat_bits;

/* I2C Slave Address Register */
typedef struct{
__REG32 GC    : 1;
__REG32 ADDR  : 7;
__REG32       :24;
} __i2adr_bits;

/* I2C SCL Duty Cycle Register */
typedef struct{
__REG32 COUNT  :16;
__REG32        :16;
} __i2scl_bits;

/*I2C Monitor mode control register*/
typedef struct{
__REG32 MM_ENA    : 1;
__REG32 ENA_SCL   : 1;
__REG32           : 1;
__REG32 MATCH_ALL : 1;
__REG32           :28;
} __i2mmctrl_bits;

/* I2C Mask registers */
typedef struct{
__REG32       : 1;
__REG32 MASK  : 7;
__REG32       :24;
} __i2mask_bits;

/* Timer control register (TCR) */
typedef struct {
  __REG32 COUNTER_ENABLE      : 1;
  __REG32 COUNTER_RESET       : 1;
  __REG32                     :30;
} __tmr_tcr_bits;

/* MSCSS Timer control register (TCR) */
typedef struct {
  __REG32 COUNTER_ENABLE      : 1;
  __REG32 COUNTER_RESET       : 1;
  __REG32 PAUSE_ENABLE        : 1;
  __REG32                     :29;
} __mscss_tcr_bits;

/* Timer match-control register */
typedef struct {
  __REG32 RESET_0             : 1;
  __REG32 STOP_0              : 1;
  __REG32 RESET_1             : 1;
  __REG32 STOP_1              : 1;
  __REG32 RESET_2             : 1;
  __REG32 STOP_2              : 1;
  __REG32 RESET_3             : 1;
  __REG32 STOP_3              : 1;
  __REG32                     :24;
} __tmr_mcr_bits;

/* Timer external-match register */
typedef struct {
  __REG32 EMR_0               : 1;
  __REG32 EMR_1               : 1;
  __REG32 EMR_2               : 1;
  __REG32 EMR_3               : 1;
  __REG32 CTRL_0              : 2;
  __REG32 CTRL_1              : 2;
  __REG32 CTRL_2              : 2;
  __REG32 CTRL_3              : 2;
  __REG32                     :20;
} __tmr_emr_bits;

/* Timer capture-control register */
typedef struct {
  __REG32 RISE_0              : 1;
  __REG32 FALL_0              : 1;
  __REG32 RISE_1              : 1;
  __REG32 FALL_1              : 1;
  __REG32 RISE_2              : 1;
  __REG32 FALL_2              : 1;
  __REG32 RISE_3              : 1;
  __REG32 FALL_3              : 1;
  __REG32                     :24;
} __tmr_ccr_bits;

/* Timer interrupt bit description */
typedef struct {
  __REG32 M0                  : 1;
  __REG32 M1                  : 1;
  __REG32 M2                  : 1;
  __REG32 M3                  : 1;
  __REG32 C0                  : 1;
  __REG32 C1                  : 1;
  __REG32 C2                  : 1;
  __REG32 C3                  : 1;
  __REG32                     :24;
} __tmr_int_bits;

/* ADC channel configuration register */
typedef struct {
  __REG32 ACC                 : 4;
  __REG32                     :28;
} __adc_acc_bits;

/* ADC channel-compare register */
typedef struct {
  __REG32 COMP_R              :10;
  __REG32                     : 6;
  __REG32 MATCH               : 2;
  __REG32                     :14;
} __adc_comp_bits;

/* ADC channel conversion data register */
typedef struct {
  __REG32 ACD                 :10;
  __REG32                     :22;
} __adc_acd_bits;

/* ADC Compare status register */
typedef struct {
  __REG32 COMP_STATUS_0       : 1;
  __REG32 COMP_STATUS_1       : 1;
  __REG32 COMP_STATUS_2       : 1;
  __REG32 COMP_STATUS_3       : 1;
  __REG32 COMP_STATUS_4       : 1;
  __REG32 COMP_STATUS_5       : 1;
  __REG32 COMP_STATUS_6       : 1;
  __REG32 COMP_STATUS_7       : 1;
  __REG32 COMP_STATUS_8       : 1;
  __REG32 COMP_STATUS_9       : 1;
  __REG32 COMP_STATUS_10      : 1;
  __REG32 COMP_STATUS_11      : 1;
  __REG32 COMP_STATUS_12      : 1;
  __REG32 COMP_STATUS_13      : 1;
  __REG32 COMP_STATUS_14      : 1;
  __REG32 COMP_STATUS_15      : 1;
  __REG32                     :16;
} __adc_comp_status_bits;

/* ADC Compare-status clear register */
typedef struct {
  __REG32 COMP_STATUS_CLR_0   : 1;
  __REG32 COMP_STATUS_CLR_1   : 1;
  __REG32 COMP_STATUS_CLR_2   : 1;
  __REG32 COMP_STATUS_CLR_3   : 1;
  __REG32 COMP_STATUS_CLR_4   : 1;
  __REG32 COMP_STATUS_CLR_5   : 1;
  __REG32 COMP_STATUS_CLR_6   : 1;
  __REG32 COMP_STATUS_CLR_7   : 1;
  __REG32 COMP_STATUS_CLR_8   : 1;
  __REG32 COMP_STATUS_CLR_9   : 1;
  __REG32 COMP_STATUS_CLR_10  : 1;
  __REG32 COMP_STATUS_CLR_11  : 1;
  __REG32 COMP_STATUS_CLR_12  : 1;
  __REG32 COMP_STATUS_CLR_13  : 1;
  __REG32 COMP_STATUS_CLR_14  : 1;
  __REG32 COMP_STATUS_CLR_15  : 1;
  __REG32                     :16;
} __adc_comp_status_clr_bits;

/* ADC configuration register */
typedef struct {
  __REG32 ADC_CSCAN           : 1;
  __REG32 ADC_PD              : 1;
  __REG32                     : 6;
  __REG32 POSEDGE_START_0     : 1;
  __REG32 NEGEDGE_START_0     : 1;
  __REG32 POSEDGE_START_1     : 1;
  __REG32 NEGEDGE_START_1     : 1;
  __REG32 POSEDGE_START_2     : 1;
  __REG32 NEGEDGE_START_2     : 1;
  __REG32 POSEDGE_START_3     : 1;
  __REG32 NEGEDGE_START_3     : 1;
  __REG32                     :16;
} __adc_config_bits;

/* ADC control register */
typedef struct {
  __REG32 START               : 1;
  __REG32 STOP                : 1;
  __REG32 UPDATE              : 1;
  __REG32                     :29;
} __adc_control_bits;

/* ADC status register */
typedef struct {
  __REG32 ADC_STATUS          : 1;
  __REG32 ADC_CONFIG          : 1;
  __REG32                     :30;
} __adc_status_bits;

/* ADC interrupt bit description */
typedef struct {
  __REG32 SCAN                : 1;
  __REG32 COMPARE             : 1;
  __REG32                     :30;
} __adc_int_bits;

/* PWM mode control register */
typedef struct {
  __REG32 CNT_ENA             : 1;
  __REG32 CNT_RESET           : 1;
  __REG32 RUN_ONCE            : 1;
  __REG32 SYNC_OUT_ENA        : 1;
  __REG32 SYNC_SEL            : 1;
  __REG32 TRANS_ENA_SEL       : 1;
  __REG32 TRANS_ENA           : 1;
  __REG32 UPD_ENA             : 1;
  __REG32                     :24;
} __pwm_modectl_bits;

/* PWM trap control register */
typedef struct {
  __REG32 TRAP_ENA0           : 1;
  __REG32 TRAP_ENA1           : 1;
  __REG32 TRAP_ENA2           : 1;
  __REG32 TRAP_ENA3           : 1;
  __REG32 TRAP_ENA4           : 1;
  __REG32 TRAP_ENA5           : 1;
  __REG32                     :10;
  __REG32 TRAP_POL            : 1;
  __REG32                     :15;
} __pwm_trpctl_bits;

/* PWM capture control register */
typedef struct {
  __REG32 CAPT_EDGE0          : 2;
  __REG32 CAPT_EDGE1          : 2;
  __REG32 CAPT_EDGE2          : 2;
  __REG32 CAPT_EDGE3          : 2;
  __REG32                     :24;
} __pwm_captctl_bits;

/* PWM capture source register */
typedef struct {
  __REG32 CAPT_SRC0           : 2;
  __REG32 CAPT_SRC1           : 2;
  __REG32 CAPT_SRC2           : 2;
  __REG32 CAPT_SRC3           : 2;
  __REG32                     :24;
} __pwm_captsrc_bits;

/* PWM control register */
typedef struct {
  __REG32 ACT_LVL0            : 1;
  __REG32 ACT_LVL1            : 1;
  __REG32 ACT_LVL2            : 1;
  __REG32 ACT_LVL3            : 1;
  __REG32 ACT_LVL4            : 1;
  __REG32 ACT_LVL5            : 1;
  __REG32                     :10;
  __REG32 BURST_ENA0          : 1;
  __REG32 BURST_ENA1          : 1;
  __REG32 BURST_ENA2          : 1;
  __REG32 BURST_ENA3          : 1;
  __REG32 BURST_ENA4          : 1;
  __REG32 BURST_ENA5          : 1;
  __REG32                     :10;
} __pwm_ctrl_bits;

/* PWM period register */
typedef struct {
  __REG32 PRD                 :16;
  __REG32                     :16;
} __pwm_prd_bits;

/* PWM prescale register */
typedef struct {
  __REG32 PRSC                :16;
  __REG32                     :16;
} __pwm_prsc_bits;

/* PWM synchronization delay register */
typedef struct {
  __REG32 DLY                 :16;
  __REG32                     :16;
} __pwm_syndel_bits;

/* PWM count register */
typedef struct {
  __REG32 CNT                 :16;
  __REG32                     :16;
} __pwm_cnt_bits;

/* PWM match active registers */
typedef struct {
  __REG32 MTCHACT             :16;
  __REG32                     :16;
} __pwm_mtchact_bits;

/* PWM match deactive registers */
typedef struct {
  __REG32 MTDECHACT           :16;
  __REG32                     :16;
} __pwm_mtchdeact_bits;

/* PWM capture registers */
typedef struct {
  __REG32 CAPT                :16;
  __REG32                     :16;
} __pwm_cap_bits;

/* PWM mode control shadow register */
typedef struct {
  __REG32 CNT_ENA_SYNC        : 1;
  __REG32 CNT_RESET_SYNC      : 1;
  __REG32 RUN_ONCE_SYNC       : 1;
  __REG32 SYNC_SEL_SYNC       : 1;
  __REG32 TRANS_ENA_SEL_SYNC  : 1;
  __REG32                     :27;
} __pwm_modectls_bits;

/* PWM trap control shadow register */
typedef struct {
  __REG32 TRAP_ENA_SYNC0      : 1;
  __REG32 TRAP_ENA_SYNC1      : 1;
  __REG32 TRAP_ENA_SYNC2      : 1;
  __REG32 TRAP_ENA_SYNC3      : 1;
  __REG32 TRAP_ENA_SYNC4      : 1;
  __REG32 TRAP_ENA_SYNC5      : 1;
  __REG32                     :10;
  __REG32 TRAP_POL_SYNC       : 1;
  __REG32                     :15;
} __pwm_trpctls_bits;

/* PWM capture control shadow register */
typedef struct {
  __REG32 CAPT_EDGE_SYNC0     : 2;
  __REG32 CAPT_EDGE_SYNC1     : 2;
  __REG32 CAPT_EDGE_SYNC2     : 2;
  __REG32 CAPT_EDGE_SYNC3     : 2;
  __REG32                     :24;
} __pwm_captctls_bits;

/* PWM capture source shadow register */
typedef struct {
  __REG32 CAPT_SRC_SYNC0      : 2;
  __REG32 CAPT_SRC_SYNC1      : 2;
  __REG32 CAPT_SRC_SYNC2      : 2;
  __REG32 CAPT_SRC_SYNC3      : 2;
  __REG32                     :24;
} __pwm_captsrcs_bits;

/* PWM control shadow register */
typedef struct {
  __REG32 ACT_LVL_SHAD0       : 1;
  __REG32 ACT_LVL_SHAD1       : 1;
  __REG32 ACT_LVL_SHAD2       : 1;
  __REG32 ACT_LVL_SHAD3       : 1;
  __REG32 ACT_LVL_SHAD4       : 1;
  __REG32 ACT_LVL_SHAD5       : 1;
  __REG32                     :10;
  __REG32 BURST_ENA_SHAD0     : 1;
  __REG32 BURST_ENA_SHAD1     : 1;
  __REG32 BURST_ENA_SHAD2     : 1;
  __REG32 BURST_ENA_SHAD3     : 1;
  __REG32 BURST_ENA_SHAD4     : 1;
  __REG32 BURST_ENA_SHAD5     : 1;
  __REG32                     :10;
} __pwm_ctrls_bits;

/* PWM period shadow register */
typedef struct {
  __REG32 PRD_SHAD            :16;
  __REG32                     :16;
} __pwm_prds_bits;

/* PWM prescale shadow register */
typedef struct {
  __REG32 PRSC_SHAD           :16;
  __REG32                     :16;
} __pwm_prscs_bits;

/* PWM synchronization delay shadow register */
typedef struct {
  __REG32 DLY_SHAD            :16;
  __REG32                     :16;
} __pwm_syndels_bits;

/* PWM match active shadow registers */
typedef struct {
  __REG32 MTCHACT_SHAD        :16;
  __REG32                     :16;
} __pwm_mtchacts_bits;

/* PWM match deactive shadow registers */
typedef struct {
  __REG32 MTDECHACT_SHAD      :16;
  __REG32                     :16;
} __pwm_mtchdeacts_bits;

/* PWM interrupt sources */
typedef struct {
  __REG32 CO                  : 1;
  __REG32 TD                  : 1;
  __REG32 UD                  : 1;
  __REG32 EMGY                : 1;
  __REG32                     :28;
} __pwm_int_bits;

/* PWM Match interrupt sources */
typedef struct {
  __REG32 MTCHACT0            : 1;
  __REG32 MTCHACT1            : 1;
  __REG32 MTCHACT2            : 1;
  __REG32 MTCHACT3            : 1;
  __REG32 MTCHACT4            : 1;
  __REG32 MTCHACT5            : 1;
  __REG32 MTCHDEACT0          : 1;
  __REG32 MTCHDEACT1          : 1;
  __REG32 MTCHDEACT2          : 1;
  __REG32 MTCHDEACT3          : 1;
  __REG32 MTCHDEACT4          : 1;
  __REG32 MTCHDEACT5          : 1;
  __REG32                     :20;
} __pwm_int_mtch_bits;

/* PWM Capture interrupt sources */
typedef struct {
  __REG32 CAPT0               : 1;
  __REG32 CAPT1               : 1;
  __REG32 CAPT2               : 1;
  __REG32 CAPT3               : 1;
  __REG32                     :28;
} __pwm_int_capt_bits;

/* QEI Control register */
typedef struct{
__REG32 RESP        : 1;
__REG32 RESPI       : 1;
__REG32 RESV        : 1;
__REG32 RESI        : 1;
__REG32             :28;
} __qeicon_bits;

/* QEI Configuration register */
typedef struct{
__REG32 DIRINV      : 1;
__REG32 SIGMODE     : 1;
__REG32 CAPMODE     : 1;
__REG32 INVINX      : 1;
__REG32             :28;
} __qeiconf_bits;

/* QEI Status register */
typedef struct{
__REG32 DIR         : 1;
__REG32             :31;
} __qeistat_bits;

/* QEI Interrupt Status register */
/* QEI Interrupt Set register */
/* QEI Interrupt Clear register */
typedef struct{
__REG32 INX_INT     : 1;
__REG32 TIM_INT     : 1;
__REG32 VELC_INT    : 1;
__REG32 DIR_INT     : 1;
__REG32 ERR_INT     : 1;
__REG32 ENCLK_INT   : 1;
__REG32 POS0_INT    : 1;
__REG32 POS1_INT    : 1;
__REG32 POS2_INT    : 1;
__REG32 REV_INT     : 1;
__REG32 POS0REV_INT : 1;
__REG32 POS1REV_INT : 1;
__REG32 POS2REV_INT : 1;
__REG32             :19;
} __qeiintstat_bits;

/* Interrupt priority mask register */
typedef struct {
  __REG32 PRIORITY_LIMITER    : 4;
  __REG32                     :28;
} __int_prioritymask_bits;

/* Interrupt vector register */
typedef struct {
  __REG32                     : 3;
  __REG32 INDEX               : 6;
  __REG32                     : 2;
  __REG32 TABLE_ADDR          :21;
} __int_vector_bits;

/* Interrupt-pending register 0 */
typedef struct {
  __REG32                     : 1;
  __REG32 PENDING1            : 1;
  __REG32 PENDING2            : 1;
  __REG32 PENDING3            : 1;
  __REG32 PENDING4            : 1;
  __REG32 PENDING5            : 1;
  __REG32 PENDING6            : 1;
  __REG32 PENDING7            : 1;
  __REG32 PENDING8            : 1;
  __REG32 PENDING9            : 1;
  __REG32 PENDING10           : 1;
  __REG32 PENDING11           : 1;
  __REG32 PENDING12           : 1;
  __REG32 PENDING13           : 1;
  __REG32 PENDING14           : 1;
  __REG32 PENDING15           : 1;
  __REG32 PENDING16           : 1;
  __REG32 PENDING17           : 1;
  __REG32 PENDING18           : 1;
  __REG32 PENDING19           : 1;
  __REG32 PENDING20           : 1;
  __REG32 PENDING21           : 1;
  __REG32 PENDING22           : 1;
  __REG32 PENDING23           : 1;
  __REG32 PENDING24           : 1;
  __REG32 PENDING25           : 1;
  __REG32 PENDING26           : 1;
  __REG32 PENDING27           : 1;
  __REG32 PENDING28           : 1;
  __REG32 PENDING29           : 1;
  __REG32 PENDING30           : 1;
  __REG32 PENDING31           : 1;
} __int_pending_1_31_bits;

/* Interrupt-pending register 1 */
typedef struct {
  __REG32 PENDING32           : 1;
  __REG32 PENDING33           : 1;
  __REG32 PENDING34           : 1;
  __REG32 PENDING35           : 1;
  __REG32 PENDING36           : 1;
  __REG32 PENDING37           : 1;
  __REG32 PENDING38           : 1;
  __REG32 PENDING39           : 1;
  __REG32 PENDING40           : 1;
  __REG32 PENDING41           : 1;
  __REG32 PENDING42           : 1;
  __REG32 PENDING43           : 1;
  __REG32 PENDING44           : 1;
  __REG32 PENDING45           : 1;
  __REG32 PENDING46           : 1;
  __REG32 PENDING47           : 1;
  __REG32 PENDING48           : 1;
  __REG32 PENDING49           : 1;
  __REG32 PENDING50           : 1;
  __REG32 PENDING51           : 1;
  __REG32 PENDING52           : 1;
  __REG32 PENDING53           : 1;
  __REG32 PENDING54           : 1;
  __REG32 PENDING55           : 1;
  __REG32 PENDING56           : 1;
  __REG32 PENDING57           : 1;
  __REG32 PENDING58           : 1;
  __REG32 PENDING59           : 1;
  __REG32 PENDING60           : 1;
  __REG32 PENDING61           : 1;
  __REG32 PENDING62           : 1;
  __REG32 PENDING63           : 1;
} __int_pending_32_63_bits;

/* Interrupt controller features register */
typedef struct {
  __REG32 N                   : 8;
  __REG32 P                   : 8;
  __REG32 T                   : 6;
  __REG32                     :10;
} __int_features_bits;

/* INT_REQUEST register bit description */
typedef struct {
  __REG32 PRIORITY_LEVEL      : 4;
  __REG32                     : 4;
  __REG32 TARGET              : 1;
  __REG32                     : 7;
  __REG32 ENABLE              : 1;
  __REG32 ACTIVE_LOW          : 1;
  __REG32                     : 7;
  __REG32 WE_ACTIVE_LOW       : 1;
  __REG32 WE_ENABLE           : 1;
  __REG32 WE_TARGET           : 1;
  __REG32 WE_PRIORITY_LEVEL   : 1;
  __REG32 CLR_SWINT           : 1;
  __REG32 SET_SWINT           : 1;
  __REG32 PENDING             : 1;
} __int_request_bits;

/* USB Clock Control register */
typedef struct{
__REG32                 : 1;
__REG32 DEV_CLK_EN      : 1;
__REG32                 : 1;
__REG32 PORTSEL_CLK_EN  : 1;
__REG32 AHB_CLK_EN      : 1;
__REG32                 :27;
} __usbclkctrl_bits;

/* USB Clock Status register */
typedef struct{
__REG32                 : 1;
__REG32 DEV_CLK_ON      : 1;
__REG32                 : 1;
__REG32 PORTSEL_CLK_ON  : 1;
__REG32 AHB_CLK_ON      : 1;
__REG32                 :27;
} __usbclkst_bits;

/* USB - Device Interrupt Status Register */
/* USB - Device Interrupt Enable Register */
/* USB - Device Interrupt Clear Register */
/* USB - Device Interrupt Set Register */
typedef struct {
  __REG32 FRAME             : 1;
  __REG32 EP_FAST           : 1;
  __REG32 EP_SLOW           : 1;
  __REG32 DEV_STAT          : 1;
  __REG32 CCEMTY            : 1;
  __REG32 CDFULL            : 1;
  __REG32 RXENDPKT          : 1;
  __REG32 TXENDPKT          : 1;
  __REG32 EP_RLZED          : 1;
  __REG32 ERR_INT           : 1;
  __REG32                   :22;
} __usbdevintst_bits;

/* USB - Device Interrupt Priority Register */
typedef struct {
  __REG8  FRAME             : 1;
  __REG8  EP_FAST           : 1;
  __REG8                    : 6;
} __usbdevintpri_bits;

/* USB - Endpoint Interrupt Status Register */
/* USB - Endpoint Interrupt Enable Register */
/* USB - Endpoint Interrupt Clear Register */
/* USB - Endpoint Interrupt Set Register */
/* USB - Endpoint Interrupt Priority Register */
typedef struct {
  __REG32 EP_0RX            : 1;
  __REG32 EP_0TX            : 1;
  __REG32 EP_1RX            : 1;
  __REG32 EP_1TX            : 1;
  __REG32 EP_2RX            : 1;
  __REG32 EP_2TX            : 1;
  __REG32 EP_3RX            : 1;
  __REG32 EP_3TX            : 1;
  __REG32 EP_4RX            : 1;
  __REG32 EP_4TX            : 1;
  __REG32 EP_5RX            : 1;
  __REG32 EP_5TX            : 1;
  __REG32 EP_6RX            : 1;
  __REG32 EP_6TX            : 1;
  __REG32 EP_7RX            : 1;
  __REG32 EP_7TX            : 1;
  __REG32 EP_8RX            : 1;
  __REG32 EP_8TX            : 1;
  __REG32 EP_9RX            : 1;
  __REG32 EP_9TX            : 1;
  __REG32 EP_10RX           : 1;
  __REG32 EP_10TX           : 1;
  __REG32 EP_11RX           : 1;
  __REG32 EP_11TX           : 1;
  __REG32 EP_12RX           : 1;
  __REG32 EP_12TX           : 1;
  __REG32 EP_13RX           : 1;
  __REG32 EP_13TX           : 1;
  __REG32 EP_14RX           : 1;
  __REG32 EP_14TX           : 1;
  __REG32 EP_15RX           : 1;
  __REG32 EP_15TX           : 1;
} __usbepintst_bits;

/* USB - Realize Enpoint Register */
/* USB - DMA Request Status Register */
/* USB - DMA Request Clear Register */
/* USB - DMA Request Set Regiser */
/* USB - EP DMA Status Register */
/* USB - EP DMA Enable Register */
/* USB - EP DMA Disable Register */
/* USB - New DD Request Interrupt Status Register */
/* USB - New DD Request Interrupt Clear Register */
/* USB - New DD Request Interrupt Set Register */
/* USB - End Of Transfer Interrupt Status Register */
/* USB - End Of Transfer Interrupt Clear Register */
/* USB - End Of Transfer Interrupt Set Register */
/* USB - System Error Interrupt Status Register */
/* USB - System Error Interrupt Clear Register */
/* USB - System Error Interrupt Set Register */
typedef struct {
  __REG32 EP0               : 1;
  __REG32 EP1               : 1;
  __REG32 EP2               : 1;
  __REG32 EP3               : 1;
  __REG32 EP4               : 1;
  __REG32 EP5               : 1;
  __REG32 EP6               : 1;
  __REG32 EP7               : 1;
  __REG32 EP8               : 1;
  __REG32 EP9               : 1;
  __REG32 EP10              : 1;
  __REG32 EP11              : 1;
  __REG32 EP12              : 1;
  __REG32 EP13              : 1;
  __REG32 EP14              : 1;
  __REG32 EP15              : 1;
  __REG32 EP16              : 1;
  __REG32 EP17              : 1;
  __REG32 EP18              : 1;
  __REG32 EP19              : 1;
  __REG32 EP20              : 1;
  __REG32 EP21              : 1;
  __REG32 EP22              : 1;
  __REG32 EP23              : 1;
  __REG32 EP24              : 1;
  __REG32 EP25              : 1;
  __REG32 EP26              : 1;
  __REG32 EP27              : 1;
  __REG32 EP28              : 1;
  __REG32 EP29              : 1;
  __REG32 EP30              : 1;
  __REG32 EP31              : 1;
} __usbreep_bits;

/* USB - Endpoint Index Register */
typedef struct {
  __REG32 PHY_ENDP          : 5;
  __REG32                   :27;
} __usbepin_bits;

/* USB - MaxPacketSize Register */
typedef struct {
  __REG32 MPS               :10;
  __REG32                   :22;
} __usbmaxpsize_bits;

/* USB - Receive Packet Length Register */
typedef struct {
  __REG32 PKT_LNGTH         :10;
  __REG32 DV                : 1;
  __REG32 PKT_RDY           : 1;
  __REG32                   :20;
} __usbrxplen_bits;

/* USB - Transmit Packet Length Register */
typedef struct {
  __REG32 PKT_LNGHT         :10;
  __REG32                   :22;
} __usbtxplen_bits;

/* USB - Control Register */
typedef struct {
  __REG32 RD_EN             : 1;
  __REG32 WR_EN             : 1;
  __REG32 LOG_ENDPOINT      : 4;
  __REG32                   :26;
} __usbctrl_bits;

/* USB - Command Code Register */
typedef struct {
  __REG32                   : 8;
  __REG32 CMD_PHASE         : 8;
  __REG32 CMD_CODE          : 8;
  __REG32                   : 8;
} __usbcmdcode_bits;

/* USB - Command Data Register */
typedef struct {
  __REG32 CMD_DATA          : 8;
  __REG32                   :24;
} __usbcmddata_bits;

/* USB - DMA Interrupt Status Register */
/* USB - DMA Interrupt Enable Register */
typedef struct {
  __REG32 EOT       : 1;
  __REG32 NDDR      : 1;
  __REG32 ERR       : 1;
  __REG32           :29;
} __usbdmaintst_bits;

/* DMA Interrupt Status Register */
typedef struct{
__REG32 INTSTATUS0  : 1;
__REG32 INTSTATUS1  : 1;
__REG32 INTSTATUS2  : 1;
__REG32 INTSTATUS3  : 1;
__REG32 INTSTATUS4  : 1;
__REG32 INTSTATUS5  : 1;
__REG32 INTSTATUS6  : 1;
__REG32 INTSTATUS7  : 1;
__REG32             :24;
} __dmacintstatus_bits;

/* DMA Interrupt Terminal Count Request Status Register */
typedef struct{
__REG32 INTTCSTATUS0  : 1;
__REG32 INTTCSTATUS1  : 1;
__REG32 INTTCSTATUS2  : 1;
__REG32 INTTCSTATUS3  : 1;
__REG32 INTTCSTATUS4  : 1;
__REG32 INTTCSTATUS5  : 1;
__REG32 INTTCSTATUS6  : 1;
__REG32 INTTCSTATUS7  : 1;
__REG32               :24;
} __dmacinttcstatus_bits;

/* DMA Interrupt Terminal Count Request Clear Register */
typedef struct{
__REG32 INTTCCLEAR0   : 1;
__REG32 INTTCCLEAR1   : 1;
__REG32 INTTCCLEAR2   : 1;
__REG32 INTTCCLEAR3   : 1;
__REG32 INTTCCLEAR4   : 1;
__REG32 INTTCCLEAR5   : 1;
__REG32 INTTCCLEAR6   : 1;
__REG32 INTTCCLEAR7   : 1;
__REG32               :24;
} __dmacinttcclear_bits;

/* DMA Interrupt Error Status Register */
typedef struct{
__REG32 INTERRORSTATUS0 : 1;
__REG32 INTERRORSTATUS1 : 1;
__REG32 INTERRORSTATUS2 : 1;
__REG32 INTERRORSTATUS3 : 1;
__REG32 INTERRORSTATUS4 : 1;
__REG32 INTERRORSTATUS5 : 1;
__REG32 INTERRORSTATUS6 : 1;
__REG32 INTERRORSTATUS7 : 1;
__REG32                 :24;
} __dmacinterrstat_bits;

/* DMA Interrupt Error Clear Register */
typedef struct{
__REG32 INTERRCLR0      : 1;
__REG32 INTERRCLR1      : 1;
__REG32 INTERRCLR2      : 1;
__REG32 INTERRCLR3      : 1;
__REG32 INTERRCLR4      : 1;
__REG32 INTERRCLR5      : 1;
__REG32 INTERRCLR6      : 1;
__REG32 INTERRCLR7      : 1;
__REG32                 :24;
} __dmacinterrclr_bits;

/* DMA Raw Interrupt Terminal Count Status Register */
typedef struct{
__REG32 RAWINTTCSTATUS0 : 1;
__REG32 RAWINTTCSTATUS1 : 1;
__REG32 RAWINTTCSTATUS2 : 1;
__REG32 RAWINTTCSTATUS3 : 1;
__REG32 RAWINTTCSTATUS4 : 1;
__REG32 RAWINTTCSTATUS5 : 1;
__REG32 RAWINTTCSTATUS6 : 1;
__REG32 RAWINTTCSTATUS7 : 1;
__REG32                 :24;
} __dmacrawinttcstatus_bits;

/* DMA Raw Error Interrupt Status Register */
typedef struct{
__REG32 RAWINTERRORSTATUS0  : 1;
__REG32 RAWINTERRORSTATUS1  : 1;
__REG32 RAWINTERRORSTATUS2  : 1;
__REG32 RAWINTERRORSTATUS3  : 1;
__REG32 RAWINTERRORSTATUS4  : 1;
__REG32 RAWINTERRORSTATUS5  : 1;
__REG32 RAWINTERRORSTATUS6  : 1;
__REG32 RAWINTERRORSTATUS7  : 1;
__REG32                     :24;
} __dmacrawinterrorstatus_bits;

/* DMA Enabled Channel Register */
typedef struct{
__REG32 ENABLEDCHANNELS0  : 1;
__REG32 ENABLEDCHANNELS1  : 1;
__REG32 ENABLEDCHANNELS2  : 1;
__REG32 ENABLEDCHANNELS3  : 1;
__REG32 ENABLEDCHANNELS4  : 1;
__REG32 ENABLEDCHANNELS5  : 1;
__REG32 ENABLEDCHANNELS6  : 1;
__REG32 ENABLEDCHANNELS7  : 1;
__REG32                   :24;
} __dmacenbldchns_bits;

/* DMA Software Burst Request Register */
typedef struct{
__REG32 SOFTBREQSPI0TX    : 1;
__REG32 SOFTBREQSPI0RX    : 1;
__REG32 SOFTBREQSPI1TX    : 1;
__REG32 SOFTBREQSPI1RX    : 1;
__REG32 SOFTBREQSPI2TX    : 1;
__REG32 SOFTBREQSPI2RX    : 1;
__REG32 SOFTBREQU0TX      : 1;
__REG32 SOFTBREQU0RX      : 1;
__REG32 SOFTBREQU1TX      : 1;
__REG32 SOFTBREQU1RX      : 1;
__REG32                   :22;
} __dmacsoftbreq_bits;

/* DMA Software Single Request Register */
typedef struct{
__REG32 SOFTSREQSPI0TX    : 1;
__REG32 SOFTSREQSPI0RX    : 1;
__REG32 SOFTSREQSPI1TX    : 1;
__REG32 SOFTSREQSPI1RX    : 1;
__REG32 SOFTSREQSPI2TX    : 1;
__REG32 SOFTSREQSPI2RX    : 1;
__REG32                   :26;
} __dmacsoftsreq_bits;

/* DMA Software Last Burst Request Register */
typedef struct{
__REG32 SOFTLBREQSPI0TX   : 1;
__REG32 SOFTLBREQSPI0RX   : 1;
__REG32 SOFTLBREQSPI1TX   : 1;
__REG32 SOFTLBREQSPI1RX   : 1;
__REG32 SOFTLBREQSPI2TX   : 1;
__REG32 SOFTLBREQSPI2RX   : 1;
__REG32 SOFTLBREQU0TX     : 1;
__REG32 SOFTLBREQU0RX     : 1;
__REG32 SOFTLBREQU1TX     : 1;
__REG32 SOFTLBREQU1RX     : 1;
__REG32                   :22;
} __dmacsoftlbreq_bits;

/* DMA Software Last Single Request Register */
typedef struct{
__REG32 SOFTLSREQSPI0TX   : 1;
__REG32 SOFTLSREQSPI0RX   : 1;
__REG32 SOFTLSREQSPI1TX   : 1;
__REG32 SOFTLSREQSPI1RX   : 1;
__REG32 SOFTLSREQSPI2TX   : 1;
__REG32 SOFTLSREQSPI2RX   : 1;
__REG32                   :26;
} __dmacsoftlsreq_bits;

/* DMA Synchronization Register */
typedef struct{
__REG32 DMACSYNC0   : 1;
__REG32 DMACSYNC1   : 1;
__REG32 DMACSYNC2   : 1;
__REG32 DMACSYNC3   : 1;
__REG32 DMACSYNC4   : 1;
__REG32 DMACSYNC5   : 1;
__REG32 DMACSYNC6   : 1;
__REG32 DMACSYNC7   : 1;
__REG32 DMACSYNC8   : 1;
__REG32 DMACSYNC9   : 1;
__REG32 DMACSYNC10  : 1;
__REG32 DMACSYNC11  : 1;
__REG32 DMACSYNC12  : 1;
__REG32 DMACSYNC13  : 1;
__REG32 DMACSYNC14  : 1;
__REG32 DMACSYNC15  : 1;
__REG32             :16;
} __dmacsync_bits;

/* DMA Configuration Register */
typedef struct{
__REG32 E           : 1;
__REG32 M0          : 1;
__REG32 M1          : 1;
__REG32             :29;
} __dmacconfig_bits;

/* DMA Channel Linked List Item registers */
typedef struct{
__REG32 LM          : 1;
__REG32             : 1;
__REG32 LLI         :30;
} __dma_lli_bits;

/* DMA Channel Control Registers */
typedef struct{
__REG32 TRANSFERSIZE  :12;
__REG32 SBSIZE        : 3;
__REG32 DBSIZE        : 3;
__REG32 SWIDTH        : 3;
__REG32 DWIDTH        : 3;
__REG32 S             : 1;
__REG32 D             : 1;
__REG32 SI            : 1;
__REG32 DI            : 1;
__REG32 PROT1         : 1;
__REG32 PROT2         : 1;
__REG32 PROT3         : 1;
__REG32 I             : 1;
} __dma_ctrl_bits;

/* DMA Channel Configuration Registers */
typedef struct{
__REG32 E               : 1;
__REG32 SRCPERIPHERAL   : 5;
__REG32 DESTPERIPHERAL  : 5;
__REG32 FLOWCNTRL       : 3;
__REG32 IE              : 1;
__REG32 ITC             : 1;
__REG32 L               : 1;
__REG32 A               : 1;
__REG32 H               : 1;
__REG32                 :13;
} __dma_cfg_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** Flash & EEPROM
 **
 ***************************************************************************/
__IO_REG32_BIT(FCTR,                  0x20200000,__READ_WRITE ,__fctr_bits    );
__IO_REG32_BIT(FPTR,                  0x20200008,__READ_WRITE ,__fptr_bits    );
__IO_REG32_BIT(FTCTR,                 0x2020000C,__READ_WRITE ,__ftctr_bits    );
__IO_REG32_BIT(FBWST,                 0x20200010,__READ_WRITE ,__fbwst_bits   );
__IO_REG32_BIT(FCRA,                  0x2020001C,__READ_WRITE ,__fcra_bits    );
__IO_REG32_BIT(FMSSTART,              0x20200020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,               0x20200024,__READ_WRITE ,__fmsstop_bits );
__IO_REG32(    FMSW0,                 0x2020002C,__READ                       );
__IO_REG32(    FMSW1,                 0x20200030,__READ                       );
__IO_REG32(    FMSW2,                 0x20200034,__READ                       );
__IO_REG32(    FMSW3,                 0x20200038,__READ                       );
__IO_REG32_BIT(EECMD,                 0x20200080,__READ_WRITE ,__eecmd_bits        );
__IO_REG32_BIT(EEADDR,                0x20200084,__READ_WRITE ,__eeaddr_bits       );
__IO_REG32(    EEWDATA,               0x20200088,__WRITE      );
__IO_REG32(    EERDATA,               0x2020008C,__READ       );
__IO_REG32_BIT(EEWSTATE,              0x20200090,__READ_WRITE ,__eewstate_bits     );
__IO_REG32_BIT(EECLKDIV,              0x20200094,__READ_WRITE ,__eeclkdiv_bits     );
__IO_REG32_BIT(EEPWRDWN,              0x20200098,__READ_WRITE ,__eepwrdwn_bits     );
__IO_REG32_BIT(EEMSSTART,             0x2020009C,__READ_WRITE ,__eemsstart_bits    );
__IO_REG32_BIT(EEMSSTOP,              0x202000A0,__READ_WRITE ,__eemsstop_bits     );
__IO_REG32_BIT(EEMSSIG,               0x202000A4,__READ       ,__eemssig_bits      );
__IO_REG32_BIT(FMC_INT_CLR_ENABLE,    0x20200FD8,__WRITE      ,__fmc_int_bits );
__IO_REG32_BIT(FMC_INT_SET_ENABLE,    0x20200FDC,__WRITE      ,__fmc_int_bits );
__IO_REG32_BIT(FMC_INT_STATUS,        0x20200FE0,__READ       ,__fmc_int_bits );
__IO_REG32_BIT(FMC_INT_ENABLE,        0x20200FE4,__READ       ,__fmc_int_bits );
__IO_REG32_BIT(FMC_INT_CLR_STATUS,    0x20200FE8,__WRITE      ,__fmc_int_bits );
__IO_REG32_BIT(FMC_INT_SET_STATUS,    0x20200FEC,__WRITE      ,__fmc_int_bits );

/***************************************************************************
 **
 ** CGU0 ( Clock Generation Unit 0)
 **
 ***************************************************************************/
__IO_REG32_BIT(CGU0_FREQ_MON,              0xFFFF8014,__READ_WRITE ,__freq_mon_bits        );
__IO_REG32_BIT(CGU0_RDET,                  0xFFFF8018,__READ       ,__cgu0_rdet_bits       );
__IO_REG32_BIT(CGU0_XTAL_OSC_STATUS,       0xFFFF801C,__READ       ,__xtal_osc_status_bits );
__IO_REG32_BIT(CGU0_XTAL_OSC_CONTROL,      0xFFFF8020,__READ_WRITE ,__xtal_osc_status_bits );
__IO_REG32_BIT(CGU0_PLL_STATUS,            0xFFFF8024,__READ       ,__pll_status_bits      );
__IO_REG32_BIT(CGU0_PLL_CONTROL,           0xFFFF8028,__READ_WRITE ,__pll_control_bits     );
__IO_REG32_BIT(CGU0_FDIV_STATUS_0,         0xFFFF802C,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_0,        0xFFFF8030,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_1,         0xFFFF8034,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_1,        0xFFFF8038,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_2,         0xFFFF803C,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_2,        0xFFFF8040,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_3,         0xFFFF8044,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_3,        0xFFFF8048,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_4,         0xFFFF804C,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_4,        0xFFFF8050,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_5,         0xFFFF8054,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_5,        0xFFFF8058,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_STATUS_6,         0xFFFF805C,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_FDIV_CONTROL_6,        0xFFFF8060,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU0_SAFE_CLK_STATUS,       0xFFFF8064,__READ       ,__safe_clk_status_bits );
__IO_REG32_BIT(CGU0_SAFE_CLK_CONF,         0xFFFF8068,__READ_WRITE ,__safe_clk_conf_bits   );
__IO_REG32_BIT(CGU0_SYS_CLK_STATUS,        0xFFFF806C,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_SYS_CLK_CONF,          0xFFFF8070,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_PCR_CLK_STATUS,        0xFFFF8074,__READ       ,__safe_clk_status_bits );
__IO_REG32_BIT(CGU0_PCR_CLK_CONF,          0xFFFF8078,__READ_WRITE ,__safe_clk_conf_bits   );
__IO_REG32_BIT(CGU0_IVNSS_CLK_STATUS,      0xFFFF807C,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_IVNSS_CLK_CONF,        0xFFFF8080,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_MSCSS_CLK_STATUS,      0xFFFF8084,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_MSCSS_CLK_CONF,        0xFFFF8088,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_ICLK0_CLK_CONF,        0xFFFF808C,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_ICLK0_CLK_STATUS,      0xFFFF8090,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_UART_CLK_STATUS,       0xFFFF8094,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_UART_CLK_CONF,         0xFFFF8098,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_SPI_CLK_STATUS,        0xFFFF809C,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_SPI_CLK_CONF,          0xFFFF80A0,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_TMR_CLK_STATUS,        0xFFFF80A4,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_TMR_CLK_CONF,          0xFFFF80A8,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_ADC_CLK_STATUS,        0xFFFF80AC,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU0_ADC_CLK_CONF,          0xFFFF80B0,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_ICLK1_CLK_CONF,        0xFFFF80BC,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU0_ICLK1_CLK_STATUS,      0xFFFF80C0,__READ       ,__xx_clk_status_bits   );

__IO_REG32_BIT(CGU0_INT_CLR_ENABLE,        0xFFFF8FD8,__WRITE      ,__cgu_int_bits );
__IO_REG32_BIT(CGU0_INT_SET_ENABLE,        0xFFFF8FDC,__WRITE      ,__cgu_int_bits );
__IO_REG32_BIT(CGU0_INT_STATUS,            0xFFFF8FE0,__READ       ,__cgu_int_bits );
__IO_REG32_BIT(CGU0_INT_ENABLE,            0xFFFF8FE4,__READ       ,__cgu_int_bits );
__IO_REG32_BIT(CGU0_INT_CLR_STATUS,        0xFFFF8FE8,__WRITE      ,__cgu_int_bits );
__IO_REG32_BIT(CGU0_INT_SET_STATUS,        0xFFFF8FEC,__WRITE      ,__cgu_int_bits );

__IO_REG32_BIT(CGU0_BUS_DISABLE,           0xFFFF8FF4,__READ_WRITE ,__bus_disable_bits );

/***************************************************************************
 **
 ** CGU1 ( Clock Generation Unit 1)
 **
 ***************************************************************************/
__IO_REG32_BIT(CGU1_FREQ_MON,              0xFFFFB014,__READ_WRITE ,__freq_mon_bits        );
__IO_REG32_BIT(CGU1_RDET,                  0xFFFFB018,__READ       ,__cgu1_rdet_bits       );
__IO_REG32_BIT(CGU1_PLL_STATUS,            0xFFFFB01C,__READ       ,__pll_status_bits      );
__IO_REG32_BIT(CGU1_PLL_CONTROL,           0xFFFFB020,__READ_WRITE ,__pll_control_bits     );
__IO_REG32_BIT(CGU1_FDIV_STATUS_0,         0xFFFFB024,__READ       ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU1_FDIV_CONTROL_0,        0xFFFFB028,__READ_WRITE ,__fdiv_status_x_bits   );
__IO_REG32_BIT(CGU1_USB_CLK_STATUS,        0xFFFFB02C,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU1_USB_CLK_CONF,          0xFFFFB030,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU1_USB_I2C_CLK_STATUS,    0xFFFFB034,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU1_USB_I2C_CLK_CONF,      0xFFFFB038,__READ_WRITE ,__xx_clk_conf_bits     );
__IO_REG32_BIT(CGU1_OUT_CLK_STATUS,        0xFFFFB03C,__READ       ,__xx_clk_status_bits   );
__IO_REG32_BIT(CGU1_OUT_CLK_CONF,          0xFFFFB040,__READ_WRITE ,__xx_clk_conf_bits     );

__IO_REG32_BIT(CGU1_BUS_DISABLE,           0xFFFFBFF4,__READ_WRITE ,__bus_disable_bits );

/***************************************************************************
 **
 ** RGU ( Reset Generation Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(RESET_CTRL0,           0xFFFF9100,__WRITE      ,__reset_ctrl0_bits       );
__IO_REG32_BIT(RESET_CTRL1,           0xFFFF9104,__WRITE      ,__reset_ctrl1_bits       );
__IO_REG32_BIT(RESET_STATUS0,         0xFFFF9110,__READ_WRITE ,__reset_status0_bits     );
__IO_REG32(    RESET_STATUS1,         0xFFFF9114,__READ                                 );
__IO_REG32_BIT(RESET_STATUS2,         0xFFFF9118,__READ_WRITE ,__reset_status2_bits     );
__IO_REG32_BIT(RESET_STATUS3,         0xFFFF911C,__READ_WRITE ,__reset_status3_bits     );
__IO_REG32_BIT(RST_ACTIVE_STATUS0,    0xFFFF9150,__READ       ,__rst_active_status0_bits);
__IO_REG32_BIT(RST_ACTIVE_STATUS1,    0xFFFF9154,__READ       ,__rst_active_status1_bits);
__IO_REG32_BIT(RGU_RST_SRC,           0xFFFF9404,__READ_WRITE ,__rgu_rst_src_bits       );
__IO_REG32_BIT(PCR_RST_SRC,           0xFFFF9408,__READ_WRITE ,__pcr_rst_src_bits       );
__IO_REG32_BIT(COLD_RST_SRC,          0xFFFF940C,__READ_WRITE ,__cold_rst_src_bits      );
__IO_REG32_BIT(WARM_RST_SRC,          0xFFFF9410,__READ_WRITE ,__warm_rst_src_bits      );
__IO_REG32_BIT(SCU_RST_SRC,           0xFFFF9480,__READ_WRITE ,__warm_rst_src_bits      );
__IO_REG32_BIT(CFID_RST_SRC,          0xFFFF9484,__READ_WRITE ,__warm_rst_src_bits      );
__IO_REG32_BIT(FMC_RST_SRC,           0xFFFF9490,__READ_WRITE ,__warm_rst_src_bits      );
__IO_REG32_BIT(EMC_RST_SRC,           0xFFFF9494,__READ_WRITE ,__warm_rst_src_bits      );
__IO_REG32_BIT(GESS_A2V_RST_SRC,      0xFFFF94A0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(PESS_A2V_RST_SRC,      0xFFFF94A4,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(GPIO_RST_SRC,          0xFFFF94A8,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(UART_RST_SRC,          0xFFFF94AC,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(TMR_RST_SRC,           0xFFFF94B0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(SPI_RST_SRC,           0xFFFF94B4,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(IVNSS_A2V_RST_SRC,     0xFFFF94B8,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(IVNSS_CAN_RST_SRC,     0xFFFF94BC,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(IVNSS_LIN_RST_SRC,     0xFFFF94C0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(MSCSS_A2V_RST_SRC,     0xFFFF94C4,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(MSCSS_PWM_RST_SRC,     0xFFFF94C8,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(MSCSS_ADC_RST_SRC,     0xFFFF94CC,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(MSCSS_TMR_RST_SRC,     0xFFFF94D0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(I2C_RST_SRC,           0xFFFF94D4,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(QEI_RST_SRC,           0xFFFF94D8,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(DMA_RST_SRC,           0xFFFF94DC,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(USB_RST_SRC,           0xFFFF94E0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(VIC_RST_SRC,           0xFFFF94F0,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(AHB_RST_SRC,           0xFFFF94F4,__READ_WRITE ,__gess_a2v_rst_src_bits  );
__IO_REG32_BIT(RGU_BUS_DISABLE,       0xFFFF9FF4,__READ_WRITE ,__bus_disable_bits       );

/***************************************************************************
 **
 ** PMU ( Power Management Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(PM,                    0xFFFFA000,__READ_WRITE ,__PM_bits         );
__IO_REG32_BIT(BASE_STAT,             0xFFFFA004,__READ       ,__BASE_STAT_bits  );
__IO_REG32_BIT(CLK_CFG_SAFE,          0xFFFFA100,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SAFE,         0xFFFFA104,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_CPU,           0xFFFFA200,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_CPU,          0xFFFFA204,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_SYS,           0xFFFFA208,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SYS,          0xFFFFA20C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PCR,           0xFFFFA210,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PCR,          0xFFFFA214,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_FMC,           0xFFFFA218,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_FMC,          0xFFFFA21C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_RAM0,          0xFFFFA220,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_RAM0,         0xFFFFA224,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_RAM1,          0xFFFFA228,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_RAM1,         0xFFFFA22C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_SMC,           0xFFFFA230,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SMC,          0xFFFFA234,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_GESS,          0xFFFFA238,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_GESS,         0xFFFFA23C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_VIC,           0xFFFFA240,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_VIC,          0xFFFFA244,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PESS,          0xFFFFA248,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PESS,         0xFFFFA24C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_GPIO0,         0xFFFFA250,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_GPIO0,        0xFFFFA254,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_GPIO1,         0xFFFFA258,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_GPIO1,        0xFFFFA25C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_IVNSS_A,       0xFFFFA270,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_IVNSS_A,      0xFFFFA274,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_MSCSS_A,       0xFFFFA278,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_MSCSS_A,      0xFFFFA27C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_GPIO5,         0xFFFFA288,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_GPIO5,        0xFFFFA28C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_DMA,           0xFFFFA290,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_DMA,          0xFFFFA294,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_USB,           0xFFFFA298,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_USB,          0xFFFFA29C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PCR_IP,        0xFFFFA300,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PCR_IP,       0xFFFFA304,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_IVNSS_APB,     0xFFFFA400,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_IVNSS_APB,    0xFFFFA404,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_CANCA,         0xFFFFA408,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_CANCA,        0xFFFFA40C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_CANC0,         0xFFFFA410,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_CANC0,        0xFFFFA414,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_CANC1,         0xFFFFA418,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_CANC1,        0xFFFFA41C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_I2C0,          0xFFFFA420,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_I2C0,         0xFFFFA424,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_I2C1,          0xFFFFA428,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_I2C1,         0xFFFFA42C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_LIN0,          0xFFFFA440,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_LIN0,         0xFFFFA444,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_LIN1,          0xFFFFA448,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_LIN1,         0xFFFFA44C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_MSCSS_APB,     0xFFFFA500,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_MSCSS_APB,    0xFFFFA504,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_MTMR0,         0xFFFFA508,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_MTMR0,        0xFFFFA50C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_MTMR1,         0xFFFFA510,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_MTMR1,        0xFFFFA514,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PWM0,          0xFFFFA518,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PWM0,         0xFFFFA51C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PWM1,          0xFFFFA520,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PWM1,         0xFFFFA524,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PWM2,          0xFFFFA528,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PWM2,         0xFFFFA52C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_PWM3,          0xFFFFA530,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_PWM3,         0xFFFFA534,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_ADC1_APB,      0xFFFFA540,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_ADC1_APB,     0xFFFFA544,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_ADC2_APB,      0xFFFFA548,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_ADC2_APB,     0xFFFFA54C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_QEI_APB,       0xFFFFA550,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_QEI_APB,      0xFFFFA554,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_OUT_CLK,       0xFFFFA600,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_OUT_CLK,      0xFFFFA604,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_UART0,         0xFFFFA700,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_UART0,        0xFFFFA704,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_UART1,         0xFFFFA708,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_UART1,        0xFFFFA70C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_SPI0,          0xFFFFA800,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SPI0,         0xFFFFA804,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_SPI1,          0xFFFFA808,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SPI1,         0xFFFFA80C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_SPI2,          0xFFFFA810,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_SPI2,         0xFFFFA814,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_TMR0,          0xFFFFA900,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_TMR0,         0xFFFFA904,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_TMR1,          0xFFFFA908,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_TMR1,         0xFFFFA90C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_TMR2,          0xFFFFA910,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_TMR2,         0xFFFFA914,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_TMR3,          0xFFFFA918,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_TMR3,         0xFFFFA91C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_ADC1,          0xFFFFAA08,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_ADC1,         0xFFFFAA0C,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_ADC2,          0xFFFFAA10,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_ADC2,         0xFFFFAA14,__READ       ,__CLK_STAT_xx_bits);
__IO_REG32_BIT(CLK_CFG_USB_CLK,       0xFFFFAC08,__READ_WRITE ,__CLK_CFG_xx_bits );
__IO_REG32_BIT(CLK_STAT_USB_CLK,      0xFFFFAC0C,__READ       ,__CLK_STAT_xx_bits);

/***************************************************************************
 **
 ** SFSP0 (SCU Function Select Port 0)
 **
 ***************************************************************************/
__IO_REG32_BIT(SFSP0_0,               0xE0001000,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_1,               0xE0001004,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_2,               0xE0001008,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_3,               0xE000100C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_4,               0xE0001010,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_5,               0xE0001014,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_6,               0xE0001018,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_7,               0xE000101C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_8,               0xE0001020,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_9,               0xE0001024,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_10,              0xE0001028,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_11,              0xE000102C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_12,              0xE0001030,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_13,              0xE0001034,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_14,              0xE0001038,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_15,              0xE000103C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_16,              0xE0001040,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_17,              0xE0001044,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_18,              0xE0001048,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_19,              0xE000104C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_20,              0xE0001050,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_21,              0xE0001054,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_22,              0xE0001058,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_23,              0xE000105C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_24,              0xE0001060,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_25,              0xE0001064,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_26,              0xE0001068,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_27,              0xE000106C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_28,              0xE0001070,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_29,              0xE0001074,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_30,              0xE0001078,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP0_31,              0xE000107C,__READ_WRITE ,__sfspx_y_bits);

/***************************************************************************
 **
 ** SFSP1 (SCU Function Select Port 1)
 **
 ***************************************************************************/
__IO_REG32_BIT(SFSP1_0,               0xE0001100,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_1,               0xE0001104,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_2,               0xE0001108,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_3,               0xE000110C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_4,               0xE0001110,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_5,               0xE0001114,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_6,               0xE0001118,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_7,               0xE000111C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_8,               0xE0001120,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_9,               0xE0001124,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_10,              0xE0001128,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_11,              0xE000112C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_12,              0xE0001130,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_13,              0xE0001134,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_14,              0xE0001138,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_15,              0xE000113C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_16,              0xE0001140,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_17,              0xE0001144,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_18,              0xE0001148,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_19,              0xE000114C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_20,              0xE0001150,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_21,              0xE0001154,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_22,              0xE0001158,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_23,              0xE000115C,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_24,              0xE0001160,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_25,              0xE0001164,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_26,              0xE0001168,__READ_WRITE ,__sfspx_y_bits);
__IO_REG32_BIT(SFSP1_27,              0xE000116C,__READ_WRITE ,__sfspx_y_bits);

/***************************************************************************
 **
 ** SFSP5 (SCU Function Select Port 5)
 **
 ***************************************************************************/
__IO_REG32_BIT(SFSP5_18,              0xE0001548,__READ_WRITE ,__sfsp5_18_bits);

/***************************************************************************
 **
 ** SCU
 **
 ***************************************************************************/
__IO_REG32_BIT(SEC_DIS,               0xE0001B00,__READ_WRITE ,__sec_dis_bits);
__IO_REG32_BIT(SEC_STA,               0xE0001B04,__READ_WRITE ,__sec_dis_bits); 
__IO_REG32_BIT(SSMM0,                 0xE0001C00,__READ_WRITE ,__ssmmx_bits);
__IO_REG32_BIT(SSMM1,                 0xE0001C04,__READ_WRITE ,__ssmmx_bits);
__IO_REG32_BIT(SSMM2,                 0xE0001C08,__READ_WRITE ,__ssmmx_bits);
__IO_REG32_BIT(SSMM3,                 0xE0001C0C,__READ_WRITE ,__ssmmx_bits);
__IO_REG32_BIT(SMP0,                  0xE0001D00,__READ_WRITE ,__smpx_bits);
__IO_REG32_BIT(SMP1,                  0xE0001D04,__READ_WRITE ,__smpx_bits);
__IO_REG32_BIT(SMP2,                  0xE0001D08,__READ_WRITE ,__smpx_bits);
__IO_REG32_BIT(SMP3,                  0xE0001D0C,__READ_WRITE ,__smpx_bits);

/***************************************************************************
 **
 ** CFID (Chip and Feature identification)
 **
 ***************************************************************************/
__IO_REG32_BIT(CHIPID,                0xE0000000,__READ       ,__chipid_bits);
__IO_REG32_BIT(FEAT0,                 0xE0000100,__READ       ,__feat0_bits );
__IO_REG32(    FEAT1,                 0xE0000104,__READ);
__IO_REG32(    FEAT2,                 0xE0000108,__READ);
__IO_REG32_BIT(FEAT3,                 0xE000010C,__READ       ,__feat3_bits );

/***************************************************************************
 **
 ** ER ( Event Router )
 **
 ***************************************************************************/
__IO_REG32_BIT(PEND,                  0xE0002C00,__READ       ,__pend_bits    );
__IO_REG32_BIT(INT_CLR,               0xE0002C20,__WRITE      ,__int_clr_bits );
__IO_REG32_BIT(INT_SET,               0xE0002C40,__WRITE      ,__int_set_bits );
__IO_REG32_BIT(MASK,                  0xE0002C60,__READ       ,__mask_bits    );
__IO_REG32_BIT(MASK_CLR,              0xE0002C80,__WRITE      ,__mask_clr_bits);
__IO_REG32_BIT(MASK_SET,              0xE0002CA0,__WRITE      ,__mask_set_bits);
__IO_REG32_BIT(APR,                   0xE0002CC0,__READ_WRITE ,__apr_bits     );
__IO_REG32_BIT(ATR,                   0xE0002CE0,__READ_WRITE ,__atr_bits     );
__IO_REG32_BIT(RSR,                   0xE0002D20,__READ_WRITE ,__rsr_bits     );

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CONFIG,           0xE0047000,__READ_WRITE ,__spix_config_bits    );
__IO_REG32_BIT(SPI0_SLV_ENABLE,       0xE0047004,__READ_WRITE ,__spix_slv_enable_bits);
__IO_REG32_BIT(SPI0_TX_FIFO_FLUSH,    0xE0047008,__WRITE      ,__spix_tx_fifo_flush_bits);
__IO_REG32_BIT(SPI0_FIFO_DATA,        0xE004700C,__READ_WRITE ,__spix_fifo_data_bits);
__IO_REG32_BIT(SPI0_RX_FIFO_POP,      0xE0047010,__WRITE      ,__spix_rx_fifo_pop_bits);
__IO_REG32_BIT(SPI0_RX_FIFO_READMODE, 0xE0047014,__READ_WRITE ,__spix_rx_fifo_readmode_bits);
__IO_REG32_BIT(SPI0_DMA_SETTINGS,     0xE0047018,__READ_WRITE ,__spix_dma_settings_bits);
__IO_REG32_BIT(SPI0_STATUS,           0xE004701C,__READ       ,__spix_status_bits);
__IO_REG32_BIT(SPI0_SLV0_SETTINGS1,   0xE0047024,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI0_SLV0_SETTINGS2,   0xE0047028,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI0_SLV1_SETTINGS1,   0xE004702C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI0_SLV1_SETTINGS2,   0xE0047030,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI0_SLV2_SETTINGS1,   0xE0047034,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI0_SLV2_SETTINGS2,   0xE0047038,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI0_SLV3_SETTINGS1,   0xE004703C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI0_SLV3_SETTINGS2,   0xE0047040,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI0_INT_THRESHOLD,    0xE0047FD4,__READ_WRITE ,__spix_int_threshold_bits);
__IO_REG32_BIT(SPI0_INT_CLR_ENABLE,   0xE0047FD8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI0_INT_SET_ENABLE,   0xE0047FDC,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI0_INT_STATUS,       0xE0047FE0,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI0_INT_ENABLE,       0xE0047FE4,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI0_INT_CLR_STATUS,   0xE0047FE8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI0_INT_SET_STATUS,   0xE0047FEC,__WRITE      ,__spix_int_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CONFIG,           0xE0048000,__READ_WRITE ,__spix_config_bits    );
__IO_REG32_BIT(SPI1_SLV_ENABLE,       0xE0048004,__READ_WRITE ,__spix_slv_enable_bits);
__IO_REG32_BIT(SPI1_TX_FIFO_FLUSH,    0xE0048008,__WRITE      ,__spix_tx_fifo_flush_bits);
__IO_REG32_BIT(SPI1_FIFO_DATA,        0xE004800C,__READ_WRITE ,__spix_fifo_data_bits);
__IO_REG32_BIT(SPI1_RX_FIFO_POP,      0xE0048010,__WRITE      ,__spix_rx_fifo_pop_bits);
__IO_REG32_BIT(SPI1_RX_FIFO_READMODE, 0xE0048014,__READ_WRITE ,__spix_rx_fifo_readmode_bits);
__IO_REG32_BIT(SPI1_DMA_SETTINGS,     0xE0048018,__READ_WRITE ,__spix_dma_settings_bits);
__IO_REG32_BIT(SPI1_STATUS,           0xE004801C,__READ       ,__spix_status_bits);
__IO_REG32_BIT(SPI1_SLV0_SETTINGS1,   0xE0048024,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI1_SLV0_SETTINGS2,   0xE0048028,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI1_SLV1_SETTINGS1,   0xE004802C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI1_SLV1_SETTINGS2,   0xE0048030,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI1_SLV2_SETTINGS1,   0xE0048034,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI1_SLV2_SETTINGS2,   0xE0048038,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI1_SLV3_SETTINGS1,   0xE004803C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI1_SLV3_SETTINGS2,   0xE0048040,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI1_INT_THRESHOLD,    0xE0048FD4,__READ_WRITE ,__spix_int_threshold_bits);
__IO_REG32_BIT(SPI1_INT_CLR_ENABLE,   0xE0048FD8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI1_INT_SET_ENABLE,   0xE0048FDC,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI1_INT_STATUS,       0xE0048FE0,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI1_INT_ENABLE,       0xE0048FE4,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI1_INT_CLR_STATUS,   0xE0048FE8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI1_INT_SET_STATUS,   0xE0048FEC,__WRITE      ,__spix_int_bits);

/***************************************************************************
 **
 ** SPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI2_CONFIG,           0xE0049000,__READ_WRITE ,__spix_config_bits    );
__IO_REG32_BIT(SPI2_SLV_ENABLE,       0xE0049004,__READ_WRITE ,__spix_slv_enable_bits);
__IO_REG32_BIT(SPI2_TX_FIFO_FLUSH,    0xE0049008,__WRITE      ,__spix_tx_fifo_flush_bits);
__IO_REG32_BIT(SPI2_FIFO_DATA,        0xE004900C,__READ_WRITE ,__spix_fifo_data_bits);
__IO_REG32_BIT(SPI2_RX_FIFO_POP,      0xE0049010,__WRITE      ,__spix_rx_fifo_pop_bits);
__IO_REG32_BIT(SPI2_RX_FIFO_READMODE, 0xE0049014,__READ_WRITE ,__spix_rx_fifo_readmode_bits);
__IO_REG32_BIT(SPI2_DMA_SETTINGS,     0xE0049018,__READ_WRITE ,__spix_dma_settings_bits);
__IO_REG32_BIT(SPI2_STATUS,           0xE004901C,__READ       ,__spix_status_bits);
__IO_REG32_BIT(SPI2_SLV0_SETTINGS1,   0xE0049024,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI2_SLV0_SETTINGS2,   0xE0049028,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI2_SLV1_SETTINGS1,   0xE004902C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI2_SLV1_SETTINGS2,   0xE0049030,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI2_SLV2_SETTINGS1,   0xE0049034,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI2_SLV2_SETTINGS2,   0xE0049038,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI2_SLV3_SETTINGS1,   0xE004903C,__READ_WRITE ,__spix_slvy_settings1_bits);
__IO_REG32_BIT(SPI2_SLV3_SETTINGS2,   0xE0049040,__READ_WRITE ,__spix_slvy_settings2_bits);
__IO_REG32_BIT(SPI2_INT_THRESHOLD,    0xE0049FD4,__READ_WRITE ,__spix_int_threshold_bits);
__IO_REG32_BIT(SPI2_INT_CLR_ENABLE,   0xE0049FD8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI2_INT_SET_ENABLE,   0xE0049FDC,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI2_INT_STATUS,       0xE0049FE0,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI2_INT_ENABLE,       0xE0049FE4,__READ       ,__spix_int_bits);
__IO_REG32_BIT(SPI2_INT_CLR_STATUS,   0xE0049FE8,__WRITE      ,__spix_int_bits);
__IO_REG32_BIT(SPI2_INT_SET_STATUS,   0xE0049FEC,__WRITE      ,__spix_int_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WD_TCR,                0xE0040000,__READ_WRITE ,__wd_tcr_bits    );
__IO_REG32(    WD_TC,                 0xE0040004,__READ                         );
__IO_REG32(    WD_PR,                 0xE0040008,__READ_WRITE                   );
__IO_REG32(    WD_KEY,                0xE0040038,__READ_WRITE                   );
__IO_REG32(    WD_TIMEOUT,            0xE004003C,__READ_WRITE                   );
__IO_REG32_BIT(WD_DEBUG,              0xE0040040,__READ_WRITE ,__wd_debug_bits  );
__IO_REG32_BIT(WDT_INT_CLR_ENABLE,    0xE0040FD8,__WRITE      ,__wdt_int_bits   );
__IO_REG32_BIT(WDT_INT_SET_ENABLE,    0xE0040FDC,__WRITE      ,__wdt_int_bits   );
__IO_REG32_BIT(WDT_INT_STATUS,        0xE0040FE0,__READ       ,__wdt_int_bits   );
__IO_REG32_BIT(WDT_INT_ENABLE,        0xE0040FE4,__READ       ,__wdt_int_bits   );
__IO_REG32_BIT(WDT_INT_CLR_STATUS,    0xE0040FE8,__WRITE      ,__wdt_int_bits   );
__IO_REG32_BIT(WDT_INT_SET_STATUS,    0xE0040FEC,__WRITE      ,__wdt_int_bits   );

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG32_BIT(U0RBR,              0xE0045000,__READ_WRITE  ,__uartrbr_bits);
#define U0THR U0RBR
#define U0THR_bit U0RBR_bit
#define U0DLL U0RBR
#define U0DLL_bit U0RBR_bit

/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,              0xE0045004,__READ_WRITE ,__uartier_bits);
#define U0DLM      UART0IER
#define U0DLM_bit      UART0IER_bit
/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0IIR,              0xE0045008,__READ_WRITE ,__uartiir_bits);
#define U0FCR U0IIR
#define U0FCR_bit  U0IIR_bit

__IO_REG32_BIT(U0LCR,              0xE004500C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U0LSR,              0xE0045014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(U0SCR,              0xE004501C,__READ_WRITE ,__uartscr_bits);
__IO_REG32_BIT(U0ACR,              0xE0045020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U0FDR,              0xE0045028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U0TER,              0xE0045030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U0RS485CTRL,        0xE004504C,__READ_WRITE ,__uartrs485ctrl_bits);
__IO_REG32_BIT(U0ADRMATCH,         0xE0045050,__READ_WRITE ,__uartadrmatch_bits);
 
/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
/* U1DLL, U1RBR and U1THR share the same address */
__IO_REG32_BIT(U1RBR,              0xE0046000,__READ_WRITE  ,__uartrbr_bits);
#define U1THR U1RBR
#define U1THR_bit U1RBR_bit
#define U1DLL U1RBR
#define U1DLL_bit U1RBR_bit

/* U1DLM and U1IER share the same address */
__IO_REG32_BIT(U1IER,              0xE0046004,__READ_WRITE ,__uartier_bits);
#define U1DLM      U1IER
#define U1DLM_bit      U1IER_bit
/* U1FCR and U1IIR share the same address */
__IO_REG32_BIT(U1IIR,              0xE0046008,__READ_WRITE ,__uartiir_bits);
#define U1FCR U1IIR
#define U1FCR_bit  U1IIR_bit

__IO_REG32_BIT(U1LCR,              0xE004600C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U1LSR,              0xE0046014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(U1SCR,              0xE004601C,__READ_WRITE ,__uartscr_bits);
__IO_REG32_BIT(U1ACR,              0xE0046020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U1FDR,              0xE0046028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U1TER,              0xE0046030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U1RS485CTRL,        0xE004604C,__READ_WRITE ,__uartrs485ctrl_bits);
__IO_REG32_BIT(U1ADRMATCH,         0xE0046050,__READ_WRITE ,__uartadrmatch_bits);
 
/***************************************************************************
 **
 ** GPIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0_PINS,            0xE004A000,__READ       ,__gpio0_pins_bits);
__IO_REG32_BIT(GPIO0_OR,              0xE004A004,__READ_WRITE ,__gpio0_or_bits  );
__IO_REG32_BIT(GPIO0_DR,              0xE004A008,__READ_WRITE ,__gpio0_dr_bits  );

/***************************************************************************
 **
 ** GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_PINS,            0xE004B000,__READ       ,__gpio1_pins_bits);
__IO_REG32_BIT(GPIO1_OR,              0xE004B004,__READ_WRITE ,__gpio1_or_bits  );
__IO_REG32_BIT(GPIO1_DR,              0xE004B008,__READ_WRITE ,__gpio1_dr_bits  );

/***************************************************************************
 **
 ** GPIO5
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO5_PINS,            0xE004F000,__READ       ,__gpio5_pins_bits);
__IO_REG32_BIT(GPIO5_OR,              0xE004F004,__READ_WRITE ,__gpio5_or_bits  );
__IO_REG32_BIT(GPIO5_DR,              0xE004F008,__READ_WRITE ,__gpio5_dr_bits  );

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CAMODE,                0xE0087000,__READ_WRITE ,__camode_bits);
__IO_REG32_BIT(CASFESA,               0xE0087004,__READ_WRITE ,__casfesa_bits);
__IO_REG32_BIT(CASFGSA,               0xE0087008,__READ_WRITE ,__casfgsa_bits);
__IO_REG32_BIT(CAEFESA,               0xE008700C,__READ_WRITE ,__caefesa_bits);
__IO_REG32_BIT(CAEFGSA,               0xE0087010,__READ_WRITE ,__caefgsa_bits);
__IO_REG32_BIT(CAEOTA,                0xE0087014,__READ_WRITE ,__caeota_bits);
__IO_REG32_BIT(CALUTEA,               0xE0087018,__READ       ,__calutea_bits);
__IO_REG32_BIT(CALUTE,                0xE008701C,__READ       ,__calute_bits);
__IO_REG32_BIT(CCCTS,                 0xE0088000,__READ       ,__cccts_bits);
__IO_REG32_BIT(CCCRS,                 0xE0088004,__READ       ,__cccrs_bits);
__IO_REG32_BIT(CCCMS,                 0xE0088008,__READ       ,__cccms_bits);
__IO_REG32_BIT(C0CMODE,               0xE0080000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C0CCMD,                0xE0080004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C0CGS,                 0xE0080008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C0CIC,                 0xE008000C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C0CIE,                 0xE0080010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C0CBT,                 0xE0080014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C0CEWL,                0xE0080018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C0CSTAT,               0xE008001C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C0CRXBMI,              0xE0080020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C0CRXBID,              0xE0080024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C0CRXBDA,              0xE0080028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C0CRXBDB,              0xE008002C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C0CTXB1MI,             0xE0080030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C0CTXB1ID,             0xE0080034,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C0CTXB1DA,             0xE0080038,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C0CTXB1DB,             0xE008003C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C0CTXB2MI,             0xE0080040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C0CTXB2ID,             0xE0080044,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C0CTXB2DA,             0xE0080048,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C0CTXB2DB,             0xE008004C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C0CTXB3MI,             0xE0080050,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C0CTXB3ID,             0xE0080054,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C0CTXB3DA,             0xE0080058,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C0CTXB3DB,             0xE008005C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C1CMODE,               0xE0081000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C1CCMD,                0xE0081004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C1CGS,                 0xE0081008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C1CIC,                 0xE008100C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C1CIE,                 0xE0081010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C1CBT,                 0xE0081014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C1CEWL,                0xE0081018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C1CSTAT,               0xE008101C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C1CRXBMI,              0xE0081020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C1CRXBID,              0xE0081024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C1CRXBDA,              0xE0081028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C1CRXBDB,              0xE008102C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C1CTXB1MI,             0xE0081030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C1CTXB1ID,             0xE0081034,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C1CTXB1DA,             0xE0081038,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C1CTXB1DB,             0xE008103C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C1CTXB2MI,             0xE0081040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C1CTXB2ID,             0xE0081044,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C1CTXB2DA,             0xE0081048,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C1CTXB2DB,             0xE008104C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C1CTXB3MI,             0xE0081050,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C1CTXB3ID,             0xE0081054,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C1CTXB3DA,             0xE0081058,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C1CTXB3DB,             0xE008105C,__READ_WRITE ,__ccrxbdb_bits);

/***************************************************************************
 **
 ** LIN0
 **
 ***************************************************************************/
__IO_REG32_BIT(L0MODE,                  0xE0089000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L0CFG,                   0xE0089004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L0CMD,                   0xE0089008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L0FBRG,                  0xE008900C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L0STAT,                  0xE0089010,__READ       ,__lstat_bits);
__IO_REG32_BIT(L0IC,                    0xE0089014,__READ       ,__lic_bits);
__IO_REG32_BIT(L0IE,                    0xE0089018,__READ_WRITE ,__lie_bits);
__IO_REG32_BIT(L0CS,                    0xE0089020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L0TO,                    0xE0089024,__READ_WRITE ,__lto_bits);
__IO_REG32_BIT(L0ID,                    0xE0089028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L0DATA,                  0xE008902C,__READ_WRITE ,__ldata_bits);
__IO_REG32_BIT(L0DATB,                  0xE0089030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L0DATC,                  0xE0089034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L0DATD,                  0xE0089038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** LIN1
 **
 ***************************************************************************/
__IO_REG32_BIT(L1MODE,                  0xE008A000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L1CFG,                   0xE008A004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L1CMD,                   0xE008A008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L1FBRG,                  0xE008A00C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L1STAT,                  0xE008A010,__READ       ,__lstat_bits);
__IO_REG32_BIT(L1IC,                    0xE008A014,__READ       ,__lic_bits);
__IO_REG32_BIT(L1IE,                    0xE008A018,__READ_WRITE ,__lie_bits);
__IO_REG32_BIT(L1CS,                    0xE008A020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L1TO,                    0xE008A024,__READ_WRITE ,__lto_bits);
__IO_REG32_BIT(L1ID,                    0xE008A028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L1DATA,                  0xE008A02C,__READ_WRITE ,__ldata_bits);
__IO_REG32_BIT(L1DATB,                  0xE008A030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L1DATC,                  0xE008A034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L1DATD,                  0xE008A038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CONSET,      0xE0082000,__READ_WRITE,__i2conset_bits);
__IO_REG32_BIT(I2C0STAT,        0xE0082004,__READ      ,__i2stat_bits);
__IO_REG32_BIT(I2C0DAT,         0xE0082008,__READ_WRITE,__i2dat_bits);
__IO_REG32_BIT(I2C0ADR,         0xE008200C,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C0SCLH,        0xE0082010,__READ_WRITE,__i2scl_bits);
__IO_REG32_BIT(I2C0SCLL,        0xE0082014,__READ_WRITE,__i2scl_bits);
__IO_REG32_BIT(I2C0CONCLR,      0xE0082018,__WRITE     ,__i2conclr_bits);
__IO_REG32_BIT(I2C0MMCTRL,      0xE008201C,__READ_WRITE,__i2mmctrl_bits);
__IO_REG32_BIT(I2C0ADR1,        0xE0082020,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR2,        0xE0082024,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR3,        0xE0082028,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C0DATA_BUFFER, 0xE008202C,__READ      ,__i2dat_bits);
__IO_REG32_BIT(I2C0MASK0,       0xE0082030,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C0MASK1,       0xE0082034,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C0MASK2,       0xE0082038,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C0MASK3,       0xE008203C,__READ_WRITE,__i2mask_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CONSET,      0xE0083000,__READ_WRITE,__i2conset_bits);
__IO_REG32_BIT(I2C1STAT,        0xE0083004,__READ      ,__i2stat_bits);
__IO_REG32_BIT(I2C1DAT,         0xE0083008,__READ_WRITE,__i2dat_bits);
__IO_REG32_BIT(I2C1ADR,         0xE008300C,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C1SCLH,        0xE0083010,__READ_WRITE,__i2scl_bits);
__IO_REG32_BIT(I2C1SCLL,        0xE0083014,__READ_WRITE,__i2scl_bits);
__IO_REG32_BIT(I2C1CONCLR,      0xE0083018,__WRITE     ,__i2conclr_bits);
__IO_REG32_BIT(I2C1MMCTRL,      0xE008301C,__READ_WRITE,__i2mmctrl_bits);
__IO_REG32_BIT(I2C1ADR1,        0xE0083020,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR2,        0xE0083024,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR3,        0xE0083028,__READ_WRITE,__i2adr_bits);
__IO_REG32_BIT(I2C1DATA_BUFFER, 0xE008302C,__READ      ,__i2dat_bits);
__IO_REG32_BIT(I2C1MASK0,       0xE0083030,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C1MASK1,       0xE0083034,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C1MASK2,       0xE0083038,__READ_WRITE,__i2mask_bits);
__IO_REG32_BIT(I2C1MASK3,       0xE008303C,__READ_WRITE,__i2mask_bits);

/***************************************************************************
 **
 ** TMR0
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR0_TCR,                0xE0041000,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    TMR0_TC,                 0xE0041004,__READ_WRITE );
__IO_REG32(    TMR0_PR,                 0xE0041008,__READ_WRITE );
__IO_REG32_BIT(TMR0_MCR,                0xE004100C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(TMR0_EMR,                0xE0041010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    TMR0_MR0,                0xE0041014,__READ_WRITE );
__IO_REG32(    TMR0_MR1,                0xE0041018,__READ_WRITE );
__IO_REG32(    TMR0_MR2,                0xE004101C,__READ_WRITE );
__IO_REG32(    TMR0_MR3,                0xE0041020,__READ_WRITE );
__IO_REG32_BIT(TMR0_CCR,                0xE0041024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    TMR0_CR0,                0xE0041028,__READ       );
__IO_REG32(    TMR0_CR1,                0xE004102C,__READ       );
__IO_REG32(    TMR0_CR2,                0xE0041030,__READ       );
__IO_REG32(    TMR0_CR3,                0xE0041034,__READ       );
__IO_REG32_BIT(TMR0_INT_CLR_ENABLE,     0xE0041FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR0_INT_SET_ENABLE,     0xE0041FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR0_INT_STATUS,         0xE0041FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR0_INT_ENABLE,         0xE0041FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR0_INT_CLR_STATUS,     0xE0041FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR0_INT_SET_STATUS,     0xE0041FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** TMR1
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR1_TCR,                0xE0042000,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    TMR1_TC,                 0xE0042004,__READ_WRITE );
__IO_REG32(    TMR1_PR,                 0xE0042008,__READ_WRITE );
__IO_REG32_BIT(TMR1_MCR,                0xE004200C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(TMR1_EMR,                0xE0042010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    TMR1_MR0,                0xE0042014,__READ_WRITE );
__IO_REG32(    TMR1_MR1,                0xE0042018,__READ_WRITE );
__IO_REG32(    TMR1_MR2,                0xE004201C,__READ_WRITE );
__IO_REG32(    TMR1_MR3,                0xE0042020,__READ_WRITE );
__IO_REG32_BIT(TMR1_CCR,                0xE0042024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    TMR1_CR0,                0xE0042028,__READ       );
__IO_REG32(    TMR1_CR1,                0xE004202C,__READ       );
__IO_REG32(    TMR1_CR2,                0xE0042030,__READ       );
__IO_REG32(    TMR1_CR3,                0xE0042034,__READ       );
__IO_REG32_BIT(TMR1_INT_CLR_ENABLE,     0xE0042FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR1_INT_SET_ENABLE,     0xE0042FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR1_INT_STATUS,         0xE0042FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR1_INT_ENABLE,         0xE0042FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR1_INT_CLR_STATUS,     0xE0042FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR1_INT_SET_STATUS,     0xE0042FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** TMR2
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR2_TCR,                0xE0043000,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    TMR2_TC,                 0xE0043004,__READ_WRITE );
__IO_REG32(    TMR2_PR,                 0xE0043008,__READ_WRITE );
__IO_REG32_BIT(TMR2_MCR,                0xE004300C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(TMR2_EMR,                0xE0043010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    TMR2_MR0,                0xE0043014,__READ_WRITE );
__IO_REG32(    TMR2_MR1,                0xE0043018,__READ_WRITE );
__IO_REG32(    TMR2_MR2,                0xE004301C,__READ_WRITE );
__IO_REG32(    TMR2_MR3,                0xE0043020,__READ_WRITE );
__IO_REG32_BIT(TMR2_CCR,                0xE0043024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    TMR2_CR0,                0xE0043028,__READ       );
__IO_REG32(    TMR2_CR1,                0xE004302C,__READ       );
__IO_REG32(    TMR2_CR2,                0xE0043030,__READ       );
__IO_REG32(    TMR2_CR3,                0xE0043034,__READ       );
__IO_REG32_BIT(TMR2_INT_CLR_ENABLE,     0xE0043FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR2_INT_SET_ENABLE,     0xE0043FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR2_INT_STATUS,         0xE0043FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR2_INT_ENABLE,         0xE0043FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR2_INT_CLR_STATUS,     0xE0043FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR2_INT_SET_STATUS,     0xE0043FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** TMR3
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR3_TCR,                0xE0044000,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    TMR3_TC,                 0xE0044004,__READ_WRITE );
__IO_REG32(    TMR3_PR,                 0xE0044008,__READ_WRITE );
__IO_REG32_BIT(TMR3_MCR,                0xE004400C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(TMR3_EMR,                0xE0044010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    TMR3_MR0,                0xE0044014,__READ_WRITE );
__IO_REG32(    TMR3_MR1,                0xE0044018,__READ_WRITE );
__IO_REG32(    TMR3_MR2,                0xE004401C,__READ_WRITE );
__IO_REG32(    TMR3_MR3,                0xE0044020,__READ_WRITE );
__IO_REG32_BIT(TMR3_CCR,                0xE0044024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    TMR3_CR0,                0xE0044028,__READ       );
__IO_REG32(    TMR3_CR1,                0xE004402C,__READ       );
__IO_REG32(    TMR3_CR2,                0xE0044030,__READ       );
__IO_REG32(    TMR3_CR3,                0xE0044034,__READ       );
__IO_REG32_BIT(TMR3_INT_CLR_ENABLE,     0xE0044FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR3_INT_SET_ENABLE,     0xE0044FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR3_INT_STATUS,         0xE0044FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR3_INT_ENABLE,         0xE0044FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(TMR3_INT_CLR_STATUS,     0xE0044FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(TMR3_INT_SET_STATUS,     0xE0044FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** MSCSS_TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(MSCSS0_TCR,              0xE00C0000,__READ_WRITE ,__mscss_tcr_bits);
__IO_REG32(    MSCSS0_TC,               0xE00C0004,__READ_WRITE );
__IO_REG32(    MSCSS0_PR,               0xE00C0008,__READ_WRITE );
__IO_REG32_BIT(MSCSS0_MCR,              0xE00C000C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(MSCSS0_EMR,              0xE00C0010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    MSCSS0_MR0,              0xE00C0014,__READ_WRITE );
__IO_REG32(    MSCSS0_MR1,              0xE00C0018,__READ_WRITE );
__IO_REG32(    MSCSS0_MR2,              0xE00C001C,__READ_WRITE );
__IO_REG32(    MSCSS0_MR3,              0xE00C0020,__READ_WRITE );
__IO_REG32_BIT(MSCSS0_CCR,              0xE00C0024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    MSCSS0_CR0,              0xE00C0028,__READ       );
__IO_REG32(    MSCSS0_CR1,              0xE00C002C,__READ       );
__IO_REG32(    MSCSS0_CR2,              0xE00C0030,__READ       );
__IO_REG32(    MSCSS0_CR3,              0xE00C0034,__READ       );
__IO_REG32_BIT(MSCSS0_INT_CLR_ENABLE,   0xE00C0FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS0_INT_SET_ENABLE,   0xE00C0FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS0_INT_STATUS,       0xE00C0FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS0_INT_ENABLE,       0xE00C0FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS0_INT_CLR_STATUS,   0xE00C0FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS0_INT_SET_STATUS,   0xE00C0FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** MSCSS_TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(MSCSS1_TCR,              0xE00C1000,__READ_WRITE ,__mscss_tcr_bits);
__IO_REG32(    MSCSS1_TC,               0xE00C1004,__READ_WRITE );
__IO_REG32(    MSCSS1_PR,               0xE00C1008,__READ_WRITE );
__IO_REG32_BIT(MSCSS1_MCR,              0xE00C100C,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32_BIT(MSCSS1_EMR,              0xE00C1010,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32(    MSCSS1_MR0,              0xE00C1014,__READ_WRITE );
__IO_REG32(    MSCSS1_MR1,              0xE00C1018,__READ_WRITE );
__IO_REG32(    MSCSS1_MR2,              0xE00C101C,__READ_WRITE );
__IO_REG32(    MSCSS1_MR3,              0xE00C1020,__READ_WRITE );
__IO_REG32_BIT(MSCSS1_CCR,              0xE00C1024,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    MSCSS1_CR0,              0xE00C1028,__READ       );
__IO_REG32(    MSCSS1_CR1,              0xE00C102C,__READ       );
__IO_REG32(    MSCSS1_CR2,              0xE00C1030,__READ       );
__IO_REG32(    MSCSS1_CR3,              0xE00C1034,__READ       );
__IO_REG32_BIT(MSCSS1_INT_CLR_ENABLE,   0xE00C1FD8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS1_INT_SET_ENABLE,   0xE00C1FDC,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS1_INT_STATUS,       0xE00C1FE0,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS1_INT_ENABLE,       0xE00C1FE4,__READ       ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS1_INT_CLR_STATUS,   0xE00C1FE8,__WRITE      ,__tmr_int_bits);
__IO_REG32_BIT(MSCSS1_INT_SET_STATUS,   0xE00C1FEC,__WRITE      ,__tmr_int_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1_ACC0,               0xE00C3000,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC1,               0xE00C3004,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC2,               0xE00C3008,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC3,               0xE00C300C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC4,               0xE00C3010,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC5,               0xE00C3014,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC6,               0xE00C3018,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC7,               0xE00C301C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC8,               0xE00C3020,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC9,               0xE00C3024,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC10,              0xE00C3028,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC11,              0xE00C302C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC12,              0xE00C3030,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC13,              0xE00C3034,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC14,              0xE00C3038,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_ACC15,              0xE00C303C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC1_COMP0,              0xE00C3100,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP1,              0xE00C3104,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP2,              0xE00C3108,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP3,              0xE00C310C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP4,              0xE00C3110,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP5,              0xE00C3114,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP6,              0xE00C3118,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP7,              0xE00C311C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP8,              0xE00C3120,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP9,              0xE00C3124,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP10,             0xE00C3128,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP11,             0xE00C312C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP12,             0xE00C3130,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP13,             0xE00C3134,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP14,             0xE00C3138,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_COMP15,             0xE00C313C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC1_ACD0,               0xE00C3200,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD1,               0xE00C3204,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD2,               0xE00C3208,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD3,               0xE00C320C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD4,               0xE00C3210,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD5,               0xE00C3214,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD6,               0xE00C3218,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD7,               0xE00C321C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD8,               0xE00C3220,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD9,               0xE00C3224,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD10,              0xE00C3228,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD11,              0xE00C322C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD12,              0xE00C3230,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD13,              0xE00C3234,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD14,              0xE00C3238,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_ACD15,              0xE00C323C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC1_COMP_STATUS,        0xE00C3300,__READ       ,__adc_comp_status_bits);
__IO_REG32_BIT(ADC1_COMP_STATUS_CLR,    0xE00C3304,__WRITE      ,__adc_comp_status_clr_bits);
__IO_REG32_BIT(ADC1_CONFIG,             0xE00C3400,__READ_WRITE ,__adc_config_bits);
__IO_REG32_BIT(ADC1_CONTROL,            0xE00C3404,__READ_WRITE ,__adc_control_bits);
__IO_REG32_BIT(ADC1_STATUS,             0xE00C3408,__READ       ,__adc_status_bits);
__IO_REG32_BIT(ADC1_INT_CLR_ENABLE,     0xE00C3FD8,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC1_INT_SET_ENABLE,     0xE00C3FDC,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC1_INT_STATUS,         0xE00C3FE0,__READ       ,__adc_int_bits);
__IO_REG32_BIT(ADC1_INT_ENABLE,         0xE00C3FE4,__READ       ,__adc_int_bits);
__IO_REG32_BIT(ADC1_INT_CLR_STATUS,     0xE00C3FE8,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC1_INT_SET_STATUS,     0xE00C3FEC,__WRITE      ,__adc_int_bits);

/***************************************************************************
 **
 ** ADC2
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC2_ACC0,               0xE00C4000,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC1,               0xE00C4004,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC2,               0xE00C4008,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC3,               0xE00C400C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC4,               0xE00C4010,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC5,               0xE00C4014,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC6,               0xE00C4018,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC7,               0xE00C401C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC8,               0xE00C4020,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC9,               0xE00C4024,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC10,              0xE00C4028,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC11,              0xE00C402C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC12,              0xE00C4030,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC13,              0xE00C4034,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC14,              0xE00C4038,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_ACC15,              0xE00C403C,__READ_WRITE ,__adc_acc_bits);
__IO_REG32_BIT(ADC2_COMP0,              0xE00C4100,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP1,              0xE00C4104,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP2,              0xE00C4108,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP3,              0xE00C410C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP4,              0xE00C4110,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP5,              0xE00C4114,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP6,              0xE00C4118,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP7,              0xE00C411C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP8,              0xE00C4120,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP9,              0xE00C4124,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP10,             0xE00C4128,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP11,             0xE00C412C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP12,             0xE00C4130,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP13,             0xE00C4134,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP14,             0xE00C4138,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_COMP15,             0xE00C413C,__READ_WRITE ,__adc_comp_bits);
__IO_REG32_BIT(ADC2_ACD0,               0xE00C4200,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD1,               0xE00C4204,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD2,               0xE00C4208,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD3,               0xE00C420C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD4,               0xE00C4210,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD5,               0xE00C4214,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD6,               0xE00C4218,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD7,               0xE00C421C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD8,               0xE00C4220,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD9,               0xE00C4224,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD10,              0xE00C4228,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD11,              0xE00C422C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD12,              0xE00C4230,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD13,              0xE00C4234,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD14,              0xE00C4238,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_ACD15,              0xE00C423C,__READ       ,__adc_acd_bits);
__IO_REG32_BIT(ADC2_COMP_STATUS,        0xE00C4300,__READ       ,__adc_comp_status_bits);
__IO_REG32_BIT(ADC2_COMP_STATUS_CLR,    0xE00C4304,__WRITE      ,__adc_comp_status_clr_bits);
__IO_REG32_BIT(ADC2_CONFIG,             0xE00C4400,__READ_WRITE ,__adc_config_bits);
__IO_REG32_BIT(ADC2_CONTROL,            0xE00C4404,__READ_WRITE ,__adc_control_bits);
__IO_REG32_BIT(ADC2_STATUS,             0xE00C4408,__READ       ,__adc_status_bits);
__IO_REG32_BIT(ADC2_INT_CLR_ENABLE,     0xE00C4FD8,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC2_INT_SET_ENABLE,     0xE00C4FDC,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC2_INT_STATUS,         0xE00C4FE0,__READ       ,__adc_int_bits);
__IO_REG32_BIT(ADC2_INT_ENABLE,         0xE00C4FE4,__READ       ,__adc_int_bits);
__IO_REG32_BIT(ADC2_INT_CLR_STATUS,     0xE00C4FE8,__WRITE      ,__adc_int_bits);
__IO_REG32_BIT(ADC2_INT_SET_STATUS,     0xE00C4FEC,__WRITE      ,__adc_int_bits);

/***************************************************************************
 **
 ** PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0_MODECTL,            0xE00C5000,__READ_WRITE ,__pwm_modectl_bits);
__IO_REG32_BIT(PWM0_TRPCTL,             0xE00C5004,__READ_WRITE ,__pwm_trpctl_bits);
__IO_REG32_BIT(PWM0_CAPTCTL,            0xE00C5008,__READ_WRITE ,__pwm_captctl_bits);
__IO_REG32_BIT(PWM0_CAPTSRC,            0xE00C500C,__READ_WRITE ,__pwm_captsrc_bits);
__IO_REG32_BIT(PWM0_CTRL,               0xE00C5010,__READ_WRITE ,__pwm_ctrl_bits);
__IO_REG32_BIT(PWM0_PRD,                0xE00C5014,__READ_WRITE ,__pwm_prd_bits);
__IO_REG32_BIT(PWM0_PRSC,               0xE00C5018,__READ_WRITE ,__pwm_prsc_bits);
__IO_REG32_BIT(PWM0_SYNDEL,             0xE00C501C,__READ_WRITE ,__pwm_syndel_bits);
__IO_REG32_BIT(PWM0_CNT,                0xE00C5020,__READ       ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM0_MTCHACT0,           0xE00C5100,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHACT1,           0xE00C5104,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHACT2,           0xE00C5108,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHACT3,           0xE00C510C,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHACT4,           0xE00C5110,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHACT5,           0xE00C5114,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT0,         0xE00C5200,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT1,         0xE00C5204,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT2,         0xE00C5208,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT3,         0xE00C520C,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT4,         0xE00C5210,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_MTCHDEACT5,         0xE00C5214,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM0_CAP0,               0xE00C5300,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM0_CAP1,               0xE00C5304,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM0_CAP2,               0xE00C5308,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM0_CAP3,               0xE00C530C,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM0_MODECTLS,           0xE00C5800,__READ       ,__pwm_modectls_bits);
__IO_REG32_BIT(PWM0_TRPCTLS,            0xE00C5804,__READ       ,__pwm_trpctls_bits);
__IO_REG32_BIT(PWM0_CAPTCTLS,           0xE00C5808,__READ       ,__pwm_captctls_bits);
__IO_REG32_BIT(PWM0_CAPTSRCS,           0xE00C580C,__READ       ,__pwm_captsrcs_bits);
__IO_REG32_BIT(PWM0_CTRLS,              0xE00C5810,__READ       ,__pwm_ctrls_bits);
__IO_REG32_BIT(PWM0_PRDS,               0xE00C5814,__READ       ,__pwm_prds_bits);
__IO_REG32_BIT(PWM0_PRSCS,              0xE00C5818,__READ       ,__pwm_prscs_bits);
__IO_REG32_BIT(PWM0_SYNDELS,            0xE00C581C,__READ       ,__pwm_syndels_bits);
__IO_REG32_BIT(PWM0_MTCHACTS0,          0xE00C5900,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHACTS1,          0xE00C5904,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHACTS2,          0xE00C5908,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHACTS3,          0xE00C590C,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHACTS4,          0xE00C5910,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHACTS5,          0xE00C5914,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS0,        0xE00C5A00,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS1,        0xE00C5A04,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS2,        0xE00C5A08,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS3,        0xE00C5A0C,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS4,        0xE00C5A10,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_MTCHDEACTS5,        0xE00C5A14,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM0_INT_CLR_ENABLE,     0xE00C5F90,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_SET_ENABLE,     0xE00C5F94,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_STATUS,         0xE00C5F98,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_ENABLE,         0xE00C5F9C,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_CLR_STATUS,     0xE00C5FA0,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_SET_STATUS,     0xE00C5FA4,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_CLR_ENABLE,0xE00C5FA8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_SET_ENABLE,0xE00C5FAC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_STATUS,    0xE00C5FB0,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_ENABLE,    0xE00C5FB4,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_CLR_STATUS,0xE00C5FB8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_MTCH_SET_STATUS,0xE00C5FBC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_CLR_ENABLE,0xE00C5FC0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_SET_ENABLE,0xE00C5FC4,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_STATUS,    0xE00C5FC8,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_ENABLE,    0xE00C5FCC,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_CLR_STATUS,0xE00C5FD0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM0_INT_CAPT_SET_STATUS,0xE00C5FD4,__WRITE      ,__pwm_int_capt_bits);

/***************************************************************************
 **
 ** PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_MODECTL,            0xE00C6000,__READ_WRITE ,__pwm_modectl_bits);
__IO_REG32_BIT(PWM1_TRPCTL,             0xE00C6004,__READ_WRITE ,__pwm_trpctl_bits);
__IO_REG32_BIT(PWM1_CAPTCTL,            0xE00C6008,__READ_WRITE ,__pwm_captctl_bits);
__IO_REG32_BIT(PWM1_CAPTSRC,            0xE00C600C,__READ_WRITE ,__pwm_captsrc_bits);
__IO_REG32_BIT(PWM1_CTRL,               0xE00C6010,__READ_WRITE ,__pwm_ctrl_bits);
__IO_REG32_BIT(PWM1_PRD,                0xE00C6014,__READ_WRITE ,__pwm_prd_bits);
__IO_REG32_BIT(PWM1_PRSC,               0xE00C6018,__READ_WRITE ,__pwm_prsc_bits);
__IO_REG32_BIT(PWM1_SYNDEL,             0xE00C601C,__READ_WRITE ,__pwm_syndel_bits);
__IO_REG32_BIT(PWM1_CNT,                0xE00C6020,__READ       ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM1_MTCHACT0,           0xE00C6100,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHACT1,           0xE00C6104,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHACT2,           0xE00C6108,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHACT3,           0xE00C610C,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHACT4,           0xE00C6110,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHACT5,           0xE00C6114,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT0,         0xE00C6200,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT1,         0xE00C6204,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT2,         0xE00C6208,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT3,         0xE00C620C,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT4,         0xE00C6210,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_MTCHDEACT5,         0xE00C6214,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM1_CAP0,               0xE00C6300,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM1_CAP1,               0xE00C6304,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM1_CAP2,               0xE00C6308,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM1_CAP3,               0xE00C630C,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM1_MODECTLS,           0xE00C6800,__READ       ,__pwm_modectls_bits);
__IO_REG32_BIT(PWM1_TRPCTLS,            0xE00C6804,__READ       ,__pwm_trpctls_bits);
__IO_REG32_BIT(PWM1_CAPTCTLS,           0xE00C6808,__READ       ,__pwm_captctls_bits);
__IO_REG32_BIT(PWM1_CAPTSRCS,           0xE00C680C,__READ       ,__pwm_captsrcs_bits);
__IO_REG32_BIT(PWM1_CTRLS,              0xE00C6810,__READ       ,__pwm_ctrls_bits);
__IO_REG32_BIT(PWM1_PRDS,               0xE00C6814,__READ       ,__pwm_prds_bits);
__IO_REG32_BIT(PWM1_PRSCS,              0xE00C6818,__READ       ,__pwm_prscs_bits);
__IO_REG32_BIT(PWM1_SYNDELS,            0xE00C681C,__READ       ,__pwm_syndels_bits);
__IO_REG32_BIT(PWM1_MTCHACTS0,          0xE00C6900,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHACTS1,          0xE00C6904,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHACTS2,          0xE00C6908,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHACTS3,          0xE00C690C,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHACTS4,          0xE00C6910,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHACTS5,          0xE00C6914,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS0,        0xE00C6A00,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS1,        0xE00C6A04,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS2,        0xE00C6A08,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS3,        0xE00C6A0C,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS4,        0xE00C6A10,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_MTCHDEACTS5,        0xE00C6A14,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM1_INT_CLR_ENABLE,     0xE00C6F90,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_SET_ENABLE,     0xE00C6F94,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_STATUS,         0xE00C6F98,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_ENABLE,         0xE00C6F9C,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_CLR_STATUS,     0xE00C6FA0,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_SET_STATUS,     0xE00C6FA4,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_CLR_ENABLE,0xE00C6FA8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_SET_ENABLE,0xE00C6FAC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_STATUS,    0xE00C6FB0,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_ENABLE,    0xE00C6FB4,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_CLR_STATUS,0xE00C6FB8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_MTCH_SET_STATUS,0xE00C6FBC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_CLR_ENABLE,0xE00C6FC0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_SET_ENABLE,0xE00C6FC4,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_STATUS,    0xE00C6FC8,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_ENABLE,    0xE00C6FCC,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_CLR_STATUS,0xE00C6FD0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM1_INT_CAPT_SET_STATUS,0xE00C6FD4,__WRITE      ,__pwm_int_capt_bits);

/***************************************************************************
 **
 ** PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_MODECTL,            0xE00C7000,__READ_WRITE ,__pwm_modectl_bits);
__IO_REG32_BIT(PWM2_TRPCTL,             0xE00C7004,__READ_WRITE ,__pwm_trpctl_bits);
__IO_REG32_BIT(PWM2_CAPTCTL,            0xE00C7008,__READ_WRITE ,__pwm_captctl_bits);
__IO_REG32_BIT(PWM2_CAPTSRC,            0xE00C700C,__READ_WRITE ,__pwm_captsrc_bits);
__IO_REG32_BIT(PWM2_CTRL,               0xE00C7010,__READ_WRITE ,__pwm_ctrl_bits);
__IO_REG32_BIT(PWM2_PRD,                0xE00C7014,__READ_WRITE ,__pwm_prd_bits);
__IO_REG32_BIT(PWM2_PRSC,               0xE00C7018,__READ_WRITE ,__pwm_prsc_bits);
__IO_REG32_BIT(PWM2_SYNDEL,             0xE00C701C,__READ_WRITE ,__pwm_syndel_bits);
__IO_REG32_BIT(PWM2_CNT,                0xE00C7020,__READ       ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM2_MTCHACT0,           0xE00C7100,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHACT1,           0xE00C7104,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHACT2,           0xE00C7108,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHACT3,           0xE00C710C,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHACT4,           0xE00C7110,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHACT5,           0xE00C7114,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT0,         0xE00C7200,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT1,         0xE00C7204,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT2,         0xE00C7208,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT3,         0xE00C720C,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT4,         0xE00C7210,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_MTCHDEACT5,         0xE00C7214,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM2_CAP0,               0xE00C7300,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM2_CAP1,               0xE00C7304,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM2_CAP2,               0xE00C7308,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM2_CAP3,               0xE00C730C,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM2_MODECTLS,           0xE00C7800,__READ       ,__pwm_modectls_bits);
__IO_REG32_BIT(PWM2_TRPCTLS,            0xE00C7804,__READ       ,__pwm_trpctls_bits);
__IO_REG32_BIT(PWM2_CAPTCTLS,           0xE00C7808,__READ       ,__pwm_captctls_bits);
__IO_REG32_BIT(PWM2_CAPTSRCS,           0xE00C780C,__READ       ,__pwm_captsrcs_bits);
__IO_REG32_BIT(PWM2_CTRLS,              0xE00C7810,__READ       ,__pwm_ctrls_bits);
__IO_REG32_BIT(PWM2_PRDS,               0xE00C7814,__READ       ,__pwm_prds_bits);
__IO_REG32_BIT(PWM2_PRSCS,              0xE00C7818,__READ       ,__pwm_prscs_bits);
__IO_REG32_BIT(PWM2_SYNDELS,            0xE00C781C,__READ       ,__pwm_syndels_bits);
__IO_REG32_BIT(PWM2_MTCHACTS0,          0xE00C7900,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHACTS1,          0xE00C7904,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHACTS2,          0xE00C7908,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHACTS3,          0xE00C790C,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHACTS4,          0xE00C7910,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHACTS5,          0xE00C7914,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS0,        0xE00C7A00,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS1,        0xE00C7A04,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS2,        0xE00C7A08,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS3,        0xE00C7A0C,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS4,        0xE00C7A10,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_MTCHDEACTS5,        0xE00C7A14,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM2_INT_CLR_ENABLE,     0xE00C7F90,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_SET_ENABLE,     0xE00C7F94,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_STATUS,         0xE00C7F98,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_ENABLE,         0xE00C7F9C,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_CLR_STATUS,     0xE00C7FA0,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_SET_STATUS,     0xE00C7FA4,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_CLR_ENABLE,0xE00C7FA8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_SET_ENABLE,0xE00C7FAC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_STATUS,    0xE00C7FB0,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_ENABLE,    0xE00C7FB4,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_CLR_STATUS,0xE00C7FB8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_MTCH_SET_STATUS,0xE00C7FBC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_CLR_ENABLE,0xE00C7FC0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_SET_ENABLE,0xE00C7FC4,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_STATUS,    0xE00C7FC8,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_ENABLE,    0xE00C7FCC,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_CLR_STATUS,0xE00C7FD0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM2_INT_CAPT_SET_STATUS,0xE00C7FD4,__WRITE      ,__pwm_int_capt_bits);

/***************************************************************************
 **
 ** PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM3_MODECTL,            0xE00C8000,__READ_WRITE ,__pwm_modectl_bits);
__IO_REG32_BIT(PWM3_TRPCTL,             0xE00C8004,__READ_WRITE ,__pwm_trpctl_bits);
__IO_REG32_BIT(PWM3_CAPTCTL,            0xE00C8008,__READ_WRITE ,__pwm_captctl_bits);
__IO_REG32_BIT(PWM3_CAPTSRC,            0xE00C800C,__READ_WRITE ,__pwm_captsrc_bits);
__IO_REG32_BIT(PWM3_CTRL,               0xE00C8010,__READ_WRITE ,__pwm_ctrl_bits);
__IO_REG32_BIT(PWM3_PRD,                0xE00C8014,__READ_WRITE ,__pwm_prd_bits);
__IO_REG32_BIT(PWM3_PRSC,               0xE00C8018,__READ_WRITE ,__pwm_prsc_bits);
__IO_REG32_BIT(PWM3_SYNDEL,             0xE00C801C,__READ_WRITE ,__pwm_syndel_bits);
__IO_REG32_BIT(PWM3_CNT,                0xE00C8020,__READ       ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM3_MTCHACT0,           0xE00C8100,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHACT1,           0xE00C8104,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHACT2,           0xE00C8108,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHACT3,           0xE00C810C,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHACT4,           0xE00C8110,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHACT5,           0xE00C8114,__READ_WRITE ,__pwm_mtchact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT0,         0xE00C8200,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT1,         0xE00C8204,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT2,         0xE00C8208,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT3,         0xE00C820C,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT4,         0xE00C8210,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_MTCHDEACT5,         0xE00C8214,__READ_WRITE ,__pwm_mtchdeact_bits);
__IO_REG32_BIT(PWM3_CAP0,               0xE00C8300,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM3_CAP1,               0xE00C8304,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM3_CAP2,               0xE00C8308,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM3_CAP3,               0xE00C830C,__READ       ,__pwm_cap_bits);
__IO_REG32_BIT(PWM3_MODECTLS,           0xE00C8800,__READ       ,__pwm_modectls_bits);
__IO_REG32_BIT(PWM3_TRPCTLS,            0xE00C8804,__READ       ,__pwm_trpctls_bits);
__IO_REG32_BIT(PWM3_CAPTCTLS,           0xE00C8808,__READ       ,__pwm_captctls_bits);
__IO_REG32_BIT(PWM3_CAPTSRCS,           0xE00C880C,__READ       ,__pwm_captsrcs_bits);
__IO_REG32_BIT(PWM3_CTRLS,              0xE00C8810,__READ       ,__pwm_ctrls_bits);
__IO_REG32_BIT(PWM3_PRDS,               0xE00C8814,__READ       ,__pwm_prds_bits);
__IO_REG32_BIT(PWM3_PRSCS,              0xE00C8818,__READ       ,__pwm_prscs_bits);
__IO_REG32_BIT(PWM3_SYNDELS,            0xE00C881C,__READ       ,__pwm_syndels_bits);
__IO_REG32_BIT(PWM3_MTCHACTS0,          0xE00C8900,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHACTS1,          0xE00C8904,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHACTS2,          0xE00C8908,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHACTS3,          0xE00C890C,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHACTS4,          0xE00C8910,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHACTS5,          0xE00C8914,__READ       ,__pwm_mtchacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS0,        0xE00C8A00,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS1,        0xE00C8A04,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS2,        0xE00C8A08,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS3,        0xE00C8A0C,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS4,        0xE00C8A10,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_MTCHDEACTS5,        0xE00C8A14,__READ       ,__pwm_mtchdeacts_bits);
__IO_REG32_BIT(PWM3_INT_CLR_ENABLE,     0xE00C8F90,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_SET_ENABLE,     0xE00C8F94,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_STATUS,         0xE00C8F98,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_ENABLE,         0xE00C8F9C,__READ       ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_CLR_STATUS,     0xE00C8FA0,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_SET_STATUS,     0xE00C8FA4,__WRITE      ,__pwm_int_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_CLR_ENABLE,0xE00C8FA8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_SET_ENABLE,0xE00C8FAC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_STATUS,    0xE00C8FB0,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_ENABLE,    0xE00C8FB4,__READ       ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_CLR_STATUS,0xE00C8FB8,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_MTCH_SET_STATUS,0xE00C8FBC,__WRITE      ,__pwm_int_mtch_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_CLR_ENABLE,0xE00C8FC0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_SET_ENABLE,0xE00C8FC4,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_STATUS,    0xE00C8FC8,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_ENABLE,    0xE00C8FCC,__READ       ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_CLR_STATUS,0xE00C8FD0,__WRITE      ,__pwm_int_capt_bits);
__IO_REG32_BIT(PWM3_INT_CAPT_SET_STATUS,0xE00C8FD4,__WRITE      ,__pwm_int_capt_bits);

/***************************************************************************
 **
 ** Quadrature Encoder Interface
 **
 ***************************************************************************/
__IO_REG32_BIT(QEICON,                0xE00C9000,__WRITE      ,__qeicon_bits);
__IO_REG32_BIT(QEISTAT,               0xE00C9004,__READ       ,__qeistat_bits);
__IO_REG32_BIT(QEICONF,               0xE00C9008,__READ_WRITE ,__qeiconf_bits);
__IO_REG32(    QEIPOS,                0xE00C900C,__READ       );
__IO_REG32(    QEIMAXPSOS,            0xE00C9010,__READ_WRITE );
__IO_REG32(    CMPOS0,                0xE00C9014,__READ_WRITE );
__IO_REG32(    CMPOS1,                0xE00C9018,__READ_WRITE );
__IO_REG32(    CMPOS2,                0xE00C901C,__READ_WRITE );
__IO_REG32(    INXCNT,                0xE00C9020,__READ       );
__IO_REG32(    INXCMP,                0xE00C9024,__READ_WRITE );
__IO_REG32(    QEILOAD,               0xE00C9028,__READ_WRITE );
__IO_REG32(    QEITIME,               0xE00C902C,__READ       );
__IO_REG32(    QEIVEL,                0xE00C9030,__READ       );
__IO_REG32(    QEICAP,                0xE00C9034,__READ       );
__IO_REG32(    VELCOMP,               0xE00C9038,__READ_WRITE );
__IO_REG32(    FILTER,                0xE00C903C,__READ_WRITE );
__IO_REG32_BIT(QEIIES,                0xE00C9FDC,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEIIEC,                0xE00C9FD8,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEIINTSTAT,            0xE00C9FE0,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEIIE,                 0xE00C9FE4,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEICLR,                0xE00C9FE8,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEISET,                0xE00C9FEC,__WRITE      ,__qeiintstat_bits);

/***************************************************************************
 **
 ** VIC
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_PRIORITYMASK_0,    0xFFFFF000,__READ_WRITE ,__int_prioritymask_bits);
__IO_REG32_BIT(INT_PRIORITYMASK_1,    0xFFFFF004,__READ_WRITE ,__int_prioritymask_bits);
__IO_REG32_BIT(INT_VECTOR_0,          0xFFFFF100,__READ_WRITE ,__int_vector_bits);
__IO_REG32_BIT(INT_VECTOR_1,          0xFFFFF104,__READ_WRITE ,__int_vector_bits);
__IO_REG32_BIT(INT_PENDING_1_31,      0xFFFFF200,__READ       ,__int_pending_1_31_bits);
__IO_REG32_BIT(INT_PENDING_32_63,     0xFFFFF204,__READ       ,__int_pending_32_63_bits);
__IO_REG32_BIT(INT_FEATURES,          0xFFFFF300,__READ       ,__int_features_bits);
__IO_REG32_BIT(INT_REQUEST_1,         0xFFFFF404,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_2,         0xFFFFF408,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_3,         0xFFFFF40C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_4,         0xFFFFF410,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_5,         0xFFFFF414,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_6,         0xFFFFF418,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_7,         0xFFFFF41C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_8,         0xFFFFF420,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_9,         0xFFFFF424,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_10,        0xFFFFF428,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_11,        0xFFFFF42C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_12,        0xFFFFF430,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_13,        0xFFFFF434,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_14,        0xFFFFF438,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_15,        0xFFFFF43C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_16,        0xFFFFF440,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_17,        0xFFFFF444,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_18,        0xFFFFF448,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_19,        0xFFFFF44C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_20,        0xFFFFF450,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_21,        0xFFFFF454,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_22,        0xFFFFF458,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_23,        0xFFFFF45C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_24,        0xFFFFF460,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_25,        0xFFFFF464,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_26,        0xFFFFF468,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_27,        0xFFFFF46C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_28,        0xFFFFF470,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_29,        0xFFFFF474,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_30,        0xFFFFF478,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_31,        0xFFFFF47C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_32,        0xFFFFF480,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_33,        0xFFFFF484,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_34,        0xFFFFF488,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_35,        0xFFFFF48C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_36,        0xFFFFF490,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_37,        0xFFFFF494,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_38,        0xFFFFF498,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_39,        0xFFFFF49C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_40,        0xFFFFF4A0,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_41,        0xFFFFF4A4,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_42,        0xFFFFF4A8,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_43,        0xFFFFF4AC,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_44,        0xFFFFF4B0,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_45,        0xFFFFF4B4,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_46,        0xFFFFF4B8,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_47,        0xFFFFF4BC,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_48,        0xFFFFF4C0,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_49,        0xFFFFF4C4,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_50,        0xFFFFF4C8,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_51,        0xFFFFF4CC,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_52,        0xFFFFF4D0,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_53,        0xFFFFF4D4,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_54,        0xFFFFF4D8,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_55,        0xFFFFF4DC,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_56,        0xFFFFF4E0,__READ_WRITE ,__int_request_bits);

/***************************************************************************
 **
 ** USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USBCLKCTRL,            0xE0100FF4,__READ_WRITE ,__usbclkctrl_bits);
__IO_REG32_BIT(USBCLKST,              0xE0100FF8,__READ       ,__usbclkst_bits);
__IO_REG32_BIT(USBDEVINTST,           0xE0100200,__READ       ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTEN,           0xE0100204,__READ_WRITE ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTCLR,          0xE0100208,__WRITE      ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTSET,          0xE010020C,__WRITE      ,__usbdevintst_bits);
__IO_REG8_BIT( USBDEVINTPRI,          0xE010022C,__WRITE      ,__usbdevintpri_bits);
__IO_REG32_BIT(USBEPINTST,            0xE0100230,__READ       ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTEN,            0xE0100234,__READ_WRITE ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTCLR,           0xE0100238,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTSET,           0xE010023C,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTPRI,           0xE0100240,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBREEP,               0xE0100244,__READ_WRITE ,__usbreep_bits);
__IO_REG32_BIT(USBEPIN,               0xE0100248,__WRITE      ,__usbepin_bits);
__IO_REG32_BIT(USBMAXPSIZE,           0xE010024C,__READ_WRITE ,__usbmaxpsize_bits);
__IO_REG32(    USBRXDATA,             0xE0100218,__READ);
__IO_REG32_BIT(USBRXPLEN,             0xE0100220,__READ       ,__usbrxplen_bits);
__IO_REG32(    USBTXDATA,             0xE010021C,__WRITE);
__IO_REG32_BIT(USBTXPLEN,             0xE0100224,__WRITE      ,__usbtxplen_bits);
__IO_REG32_BIT(USBCTRL,               0xE0100228,__READ_WRITE ,__usbctrl_bits);
__IO_REG32_BIT(USBCMDCODE,            0xE0100210,__WRITE      ,__usbcmdcode_bits);
__IO_REG32_BIT(USBCMDDATA,            0xE0100214,__READ       ,__usbcmddata_bits);
__IO_REG32_BIT(USBDMARST,             0xE0100250,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBDMARCLR,            0xE0100254,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBDMARSET,            0xE0100258,__WRITE      ,__usbreep_bits);
__IO_REG32(    USBUDCAH,              0xE0100280,__READ_WRITE );
__IO_REG32_BIT(USBEPDMAST,            0xE0100284,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBEPDMAEN,            0xE0100288,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBEPDMADIS,           0xE010028C,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBDMAINTST,           0xE0100290,__READ       ,__usbdmaintst_bits);
__IO_REG32_BIT(USBDMAINTEN,           0xE0100294,__READ_WRITE ,__usbdmaintst_bits);
__IO_REG32_BIT(USBEOTINTST,           0xE01002A0,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBEOTINTCLR,          0xE01002A4,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBEOTINTSET,          0xE01002A8,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTST,          0xE01002AC,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTCLR,         0xE01002B0,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTSET,         0xE01002B4,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTST,        0xE01002B8,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTCLR,       0xE01002BC,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTSET,       0xE01002C0,__WRITE      ,__usbreep_bits);

/***************************************************************************
 **
 ** GPDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACINTSTATUS,         0xE0140000,__READ      ,__dmacintstatus_bits);
__IO_REG32_BIT(DMACINTTCSTATUS,       0xE0140004,__READ      ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACINTTCCLEAR,        0xE0140008,__WRITE     ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACINTERRSTAT,        0xE014000C,__READ      ,__dmacinterrstat_bits);
__IO_REG32_BIT(DMACINTERRCLR,         0xE0140010,__WRITE     ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRAWINTTCSTATUS,    0xE0140014,__READ      ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRAWINTERRORSTATUS, 0xE0140018,__READ      ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACENBLDCHNS,         0xE014001C,__READ      ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSOFTBREQ,          0xE0140020,__READ_WRITE,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSOFTSREQ,          0xE0140024,__READ_WRITE,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACSOFTLBREQ,         0xE0140028,__READ_WRITE,__dmacsoftlbreq_bits);
__IO_REG32_BIT(DMACSOFTLSREQ,         0xE014002C,__READ_WRITE,__dmacsoftlsreq_bits);
__IO_REG32_BIT(DMACCONFIGURATION,     0xE0140030,__READ_WRITE,__dmacconfig_bits);
__IO_REG32_BIT(DMACSYNC,              0xE0140034,__READ_WRITE,__dmacsync_bits);
__IO_REG32(    DMACC0SRCADDR,         0xE0140100,__READ_WRITE);
__IO_REG32(    DMACC0DESTADDR,        0xE0140104,__READ_WRITE);
__IO_REG32_BIT(DMACC0LLI,             0xE0140108,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC0CONTROL,         0xE014010C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC0CONFIGURATION,   0xE0140110,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC1SRCADDR,         0xE0140120,__READ_WRITE);
__IO_REG32(    DMACC1DESTADDR,        0xE0140124,__READ_WRITE);
__IO_REG32_BIT(DMACC1LLI,             0xE0140128,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC1CONTROL,         0xE014012C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC1CONFIGURATION,   0xE0140130,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC2SRCADDR,         0xE0140140,__READ_WRITE);
__IO_REG32(    DMACC2DESTADDR,        0xE0140144,__READ_WRITE);
__IO_REG32_BIT(DMACC2LLI,             0xE0140148,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC2CONTROL,         0xE014014C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC2CONFIGURATION,   0xE0140150,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC3SRCADDR,         0xE0140160,__READ_WRITE);
__IO_REG32(    DMACC3DESTADDR,        0xE0140164,__READ_WRITE);
__IO_REG32_BIT(DMACC3LLI,             0xE0140168,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC3CONTROL,         0xE014016C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC3CONFIGURATION,   0xE0140170,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC4SRCADDR,         0xE0140180,__READ_WRITE);
__IO_REG32(    DMACC4DESTADDR,        0xE0140184,__READ_WRITE);
__IO_REG32_BIT(DMACC4LLI,             0xE0140188,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC4CONTROL,         0xE014018C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC4CONFIGURATION,   0xE0140190,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC5SRCADDR,         0xE01401A0,__READ_WRITE);
__IO_REG32(    DMACC5DESTADDR,        0xE01401A4,__READ_WRITE);
__IO_REG32_BIT(DMACC5LLI,             0xE01401A8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC5CONTROL,         0xE01401AC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC5CONFIGURATION,   0xE01401B0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC6SRCADDR,         0xE01401C0,__READ_WRITE);
__IO_REG32(    DMACC6DESTADDR,        0xE01401C4,__READ_WRITE);
__IO_REG32_BIT(DMACC6LLI,             0xE01401C8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC6CONTROL,         0xE01401CC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC6CONFIGURATION,   0xE01401D0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC7SRCADDR,         0xE01401E0,__READ_WRITE);
__IO_REG32(    DMACC7DESTADDR,        0xE01401E4,__READ_WRITE);
__IO_REG32_BIT(DMACC7LLI,             0xE01401E8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC7CONTROL,         0xE01401EC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC7CONFIGURATION,   0xE01401F0,__READ_WRITE,__dma_cfg_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV        0x00  /* Reset                              */
#define UNDEFV        0x04  /* Undefined instruction              */
#define SWIV          0x08  /* Software interrupt                 */
#define PABORTV       0x0C  /* Prefetch abort                     */
#define DABORTV       0x10  /* Data abort                         */
#define IRQV          0x18  /* Normal interrupt                   */
#define FIQV          0x1C  /* Fast interrupt                     */

/***************************************************************************
 **
 **  DMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define DMA_SPI0TX       0  /* SPI0 Tx                            */
#define DMA_SPI0RX       1  /* SPI0 Rx                            */
#define DMA_SPI1TX       2  /* SPI1 Tx                            */
#define DMA_SPI1RX       3  /* SPI1 Rx                            */
#define DMA_SPI2TX       4  /* SPI2 Tx                            */
#define DMA_SPI2RX       5  /* SPI2 Rx                            */
#define DMA_UART0TX      6  /* UART0 transmit                     */
#define DMA_UART0RX      7  /* UART0 receive                      */
#define DMA_UART1TX      8  /* UART1 transmit                     */
#define DMA_UART1RX      9  /* UART1 receive                      */

/***************************************************************************
 **
 **  VIC Interrupt channels
 **
 ***************************************************************************/
#define VIC_WDT           1   /* Interrupt from Watchdog timer                                               */
#define VIC_TIMER0        2   /* Capture or match interrupt from timer 0                                     */
#define VIC_TIMER1        3   /* Capture or match interrupt from timer 1                                     */
#define VIC_TIMER2        4   /* Capture or match interrupt from timer 2                                     */
#define VIC_TIMER3        5   /* Capture or match interrupt from timer 3                                     */
#define VIC_UART0         6   /* General interrupt from 16C550 UART 0                                        */
#define VIC_UART1         7   /* General interrupt from 16C550 UART 1                                        */
#define VIC_SPI0          8   /* General interrupt from SPI 0                                                */
#define VIC_SPI1          9   /* General interrupt from SPI 1                                                */
#define VIC_SPI2          10  /* General interrupt from SPI 2                                                */
#define VIC_FLASH         11  /* Signature, burn or erase finished interrupt from flash                      */
#define VIC_DEBUGRX       12  /* Comms Rx for ARM debug mode                                                 */
#define VIC_DEBUGTX       13  /* Comms Tx for ARM debug mode                                                 */
#define VIC_MSCSS_TIMER0  14  /* Capture or match interrupt from MSCSS timer 0                               */
#define VIC_MSCSS_TIMER1  15  /* Capture or match interrupt from MSCSS timer 1                               */
#define VIC_ADC1          17  /* ADC interrupt from ADC 1                                                    */
#define VIC_ADC2          18  /* ADC interrupt from ADC 2                                                    */
#define VIC_PWM0          19  /* PWM interrupt from PWM 0                                                    */
#define VIC_PWM0_CAP_MAT  20  /* PWM capture/match interrupt from PWM 0                                      */
#define VIC_PWM1          21  /* PWM interrupt from PWM 1                                                    */
#define VIC_PWM1_CAP_MAT  22  /* PWM capture/match interrupt from PWM 1                                      */
#define VIC_PWM2          23  /* PWM interrupt from PWM 2                                                    */
#define VIC_PWM2_CAP_MAT  24  /* PWM capture/match interrupt from PWM 2                                      */
#define VIC_PWM3          25  /* PWM interrupt from PWM 3                                                    */
#define VIC_PWM3_CAP_MAT  26  /* PWM capture/match interrupt from PWM 3                                      */
#define VIC_ENVN_RTR      27  /* Event, wake up tick interrupt from Event Router                             */
#define VIC_LIN0          28  /* General interrupt from LIN master controller 0                              */
#define VIC_LIN1          29  /* General interrupt from LIN master controller 1                              */
#define VIC_I2C0          30  /* I2C interrupt from I2C0 (SI state change)                                   */
#define VIC_I2C1          31  /* I2C interrupt from I2C1 (SI state change)                                   */
#define VIC_GPDMA         32  /* DMA                                                                         */
#define VIC_GPDMA_ERR     33  /* DMA err                                                                     */
#define VIC_GPDMA_TC      34  /* DMA tc                                                                      */
#define VIC_CAN_FULL      35  /* FullCan                                                                     */
#define VIC_CAN_ALL       36  /* Combined general interrupt of all CAN controllers and the CAN look-up table */
#define VIC_CAN0_RX       37  /* Message-received interrupt from CAN controller 0                            */
#define VIC_CAN1_RX       38  /* Message-received interrupt from CAN controller 1                            */
#define VIC_CAN0_TX       43  /* Message-transmitted interrupt from CAN controller 0                         */
#define VIC_CAN0_RX       44  /* Message-transmitted interrupt from CAN controller 1                         */
#define VIC_USB_I2C       45  /* USB I2C                                                                     */
#define VIC_USB_DEV_H     46  /* USB device, high-priority                                                   */
#define VIC_USB_DEV_L     47  /* USB device, low-priority                                                    */
#define VIC_USB_DEV_DMA   48  /* USB device DMA                                                              */
#define VIC_QEI           52  /* quadrature encoder interrupt                                                */
#define VIC_CGU0          55  /* CGU0                                                                        */
#define VIC_CGU1          56  /* CGU1                                                                        */

#endif    /* __IOLPC2925_H */
