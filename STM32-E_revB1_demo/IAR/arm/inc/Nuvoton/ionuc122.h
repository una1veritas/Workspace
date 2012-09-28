/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton NUC122 Devices
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46177 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IONUC122_H
#define __IONUC122_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   NUC122 SPECIAL FUNCTION REGISTERS
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

/* System Reset Source Register (GCR_RSTSRC) */
typedef struct {
  __REG32 RSTS_POR        : 1;
  __REG32 RSTS_RESET      : 1;
  __REG32 RSTS_WDT        : 1;
  __REG32 RSTS_LVR        : 1;
  __REG32 RSTS_BOD        : 1;
  __REG32 RSTS_SYS        : 1;
  __REG32                 : 1;
  __REG32 RSTS_CPU        : 1;
  __REG32                 : 24;
} __gcr_rstsrc_bits;

/* Peripheral Reset Control Resister 1 (GCR_IPRSTC1) */
typedef struct {
  __REG32 CHIP_RST        : 1;
  __REG32 CPU_RST         : 1;  
  __REG32                 : 30;
} __gcr_iprstc1_bits;

/* Peripheral Reset Control Resister 2 (GCR_IPRSTC2) */
typedef struct {
  __REG32                 : 1;
  __REG32 GPIO_RST        : 1;
  __REG32 TMR0_RST        : 1;
  __REG32 TMR1_RST        : 1;
  __REG32 TMR2_RST        : 1;
  __REG32 TMR3_RST        : 1;
  __REG32                 : 3;
  __REG32 I2C1_RST        : 1;
  __REG32                 : 2;
  __REG32 SPI0_RST        : 1;
  __REG32 SPI1_RST        : 1;
  __REG32                 : 2;
  __REG32 UART0_RST       : 1;
  __REG32 UART1_RST       : 1;
  __REG32                 : 2;
  __REG32 PWM03_RST       : 1;
  __REG32                 : 2;
  __REG32 PS2_RST         : 1;
  __REG32                 : 3;
  __REG32 USBD_RST        : 1;
  __REG32                 : 4;
} __gcr_iprstc2_bits;

/* Brown-Out Detector Control Register (GCR_BODCR) */
typedef struct {
  __REG32 BOD_EN          : 1;
  __REG32 BOD_VL          : 2;
  __REG32 BOD_RSTEN       : 1;
  __REG32 BOD_INTF        : 1;
  __REG32 BOD_LPM         : 1;
  __REG32 BOD_OUT         : 1;
  __REG32 LVR_EN          : 1;
  __REG32                 : 24;
} __gcr_bodcr_bits;

/* Power-On-Reset Control Register (GCR_PORCR */
typedef struct {
  __REG32 POR_DIS_CODE    : 16;
  __REG32                 : 16;
} __gcr_porcr_bits;

/* Multiple function Pin GPIOA Control register (GCR_GPA_MFP) */
typedef struct {
  __REG32                  :10;
  __REG32 GPA_MFP10        : 1;
  __REG32 GPA_MFP11        : 1;
  __REG32 GPA_MFP12        : 1;
  __REG32 GPA_MFP13        : 1;
  __REG32 GPA_MFP14        : 1;
  __REG32 GPA_MFP15        : 1;
  __REG32 GPA_TYPE0        : 1;
  __REG32 GPA_TYPE1        : 1;
  __REG32 GPA_TYPE2        : 1;
  __REG32 GPA_TYPE3        : 1;
  __REG32 GPA_TYPE4        : 1;
  __REG32 GPA_TYPE5        : 1;
  __REG32 GPA_TYPE6        : 1;
  __REG32 GPA_TYPE7        : 1;
  __REG32 GPA_TYPE8        : 1;
  __REG32 GPA_TYPE9        : 1;
  __REG32 GPA_TYPE10       : 1;
  __REG32 GPA_TYPE11       : 1;
  __REG32 GPA_TYPE12       : 1;
  __REG32 GPA_TYPE13       : 1;
  __REG32 GPA_TYPE14       : 1;
  __REG32 GPA_TYPE15       : 1;
} __gcr_gpa_mfp_bits;

/* Multiple function Pin GPIOB Control register (GCR_GPB_MFP) */
typedef struct {
  __REG32 GPB_MFP0         : 1;
  __REG32 GPB_MFP1         : 1;
  __REG32 GPB_MFP2         : 1;
  __REG32 GPB_MFP3         : 1;
  __REG32 GPB_MFP4         : 1;
  __REG32 GPB_MFP5         : 1;
  __REG32 GPB_MFP6         : 1;
  __REG32 GPB_MFP7         : 1;
  __REG32 GPB_MFP8         : 1;
  __REG32 GPB_MFP9         : 1;
  __REG32 GPB_MFP10        : 1;
  __REG32                  : 3;  
  __REG32 GPB_MFP14        : 1;
  __REG32 GPB_MFP15        : 1;
  __REG32 GPB_TYPE0        : 1;
  __REG32 GPB_TYPE1        : 1;
  __REG32 GPB_TYPE2        : 1;
  __REG32 GPB_TYPE3        : 1;
  __REG32 GPB_TYPE4        : 1;
  __REG32 GPB_TYPE5        : 1;
  __REG32 GPB_TYPE6        : 1;
  __REG32 GPB_TYPE7        : 1;
  __REG32 GPB_TYPE8        : 1;
  __REG32 GPB_TYPE9        : 1;
  __REG32 GPB_TYPE10       : 1;
  __REG32 GPB_TYPE11       : 1;
  __REG32 GPB_TYPE12       : 1;
  __REG32 GPB_TYPE13       : 1;
  __REG32 GPB_TYPE14       : 1;
  __REG32 GPB_TYPE15       : 1;
} __gcr_gpb_mfp_bits;

/* Multiple function Pin GPIOC Control register (GCR_GPC_MFP) */
typedef struct {
  __REG32 GPC_MFP0         : 1;
  __REG32 GPC_MFP1         : 1;
  __REG32 GPC_MFP2         : 1;
  __REG32 GPC_MFP3         : 1;
  __REG32 GPC_MFP4         : 1;
  __REG32 GPC_MFP5         : 1;
  __REG32                  : 2;
  __REG32 GPC_MFP8         : 1;
  __REG32 GPC_MFP9         : 1;
  __REG32 GPC_MFP10        : 1;
  __REG32 GPC_MFP11        : 1;
  __REG32 GPC_MFP12        : 1;
  __REG32 GPC_MFP13        : 1;
  __REG32                  : 2;
  __REG32 GPC_TYPE0        : 1;
  __REG32 GPC_TYPE1        : 1;
  __REG32 GPC_TYPE2        : 1;
  __REG32 GPC_TYPE3        : 1;
  __REG32 GPC_TYPE4        : 1;
  __REG32 GPC_TYPE5        : 1;
  __REG32 GPC_TYPE6        : 1;
  __REG32 GPC_TYPE7        : 1;
  __REG32 GPC_TYPE8        : 1;
  __REG32 GPC_TYPE9        : 1;
  __REG32 GPC_TYPE10       : 1;
  __REG32 GPC_TYPE11       : 1;
  __REG32 GPC_TYPE12       : 1;
  __REG32 GPC_TYPE13       : 1;
  __REG32 GPC_TYPE14       : 1;
  __REG32 GPC_TYPE15       : 1;
} __gcr_gpc_mfp_bits;

/* Multiple function Pin GPIOB Control register (GCR_GPD_MFP) */
typedef struct {
  __REG32 GPD_MFP0         : 1;
  __REG32 GPD_MFP1         : 1;
  __REG32 GPD_MFP2         : 1;
  __REG32 GPD_MFP3         : 1;
  __REG32 GPD_MFP4         : 1;
  __REG32 GPD_MFP5         : 1;
  __REG32                  : 2;
  __REG32 GPD_MFP8         : 1;
  __REG32 GPD_MFP9         : 1;
  __REG32 GPD_MFP10        : 1;
  __REG32 GPD_MFP11        : 1;
  __REG32                  : 4;  
  __REG32 GPD_TYPE0        : 1;
  __REG32 GPD_TYPE1        : 1;
  __REG32 GPD_TYPE2        : 1;
  __REG32 GPD_TYPE3        : 1;
  __REG32 GPD_TYPE4        : 1;
  __REG32 GPD_TYPE5        : 1;
  __REG32 GPD_TYPE6        : 1;
  __REG32 GPD_TYPE7        : 1;
  __REG32 GPD_TYPE8        : 1;
  __REG32 GPD_TYPE9        : 1;
  __REG32 GPD_TYPE10       : 1;
  __REG32 GPD_TYPE11       : 1;
  __REG32 GPD_TYPE12       : 1;
  __REG32 GPD_TYPE13       : 1;
  __REG32 GPD_TYPE14       : 1;
  __REG32 GPD_TYPE15       : 1;
} __gcr_gpd_mfp_bits;

/* Alternative Multiple function Pin Control register (GCR_ALT_MFP) */
typedef struct {
  __REG32 PB10_S01        : 1;
  __REG32 PB9_S11         : 1;
  __REG32 ALT_MFP2        : 1;
  __REG32 ALT_MFP3        : 1;
  __REG32 ALT_MFP4        : 1;
  __REG32 ALT_MFP5        : 1;
  __REG32 ALT_MFP6        : 1;
  __REG32 ALT_MFP7        : 1;
  __REG32 ALT_MFP8        : 1;
  __REG32 ALT_MFP9        : 1;
  __REG32 ALT_MFP10       : 1;
  __REG32 ALT_MFP11       : 1;
  __REG32 ALT_MFP12       : 1;
  __REG32 ALT_MFP13       : 1;
  __REG32 ALT_MFP14       : 1;
  __REG32 ALT_MFP15       : 1;
  __REG32 ALT_MFP16       : 1;
  __REG32 ALT_MFP17       : 1;
  __REG32 ALT_MFP18       : 1;
  __REG32 ALT_MFP19       : 1;
  __REG32 ALT_MFP20       : 1;
  __REG32 ALT_MFP21       : 1;
  __REG32                 :10;
} __gcr_alt_mfp_bits;

/* Register Write-Protection Control Register (GCR_REGWRPROT) */
typedef union {
  /* GCR_REGWRPROT */
  struct
  {
    __REG32 REGWRPROT     : 8;
    __REG32               : 24;
    
  };
  /* GCR_REGPROTDIS */
  struct
  {
    __REG32 REGPROTDIS    : 1;
    __REG32               : 31;
  };
  
} __gcr_regwrprot_bits;

/* SysTick Control and Status Register (SCS_SYST_CSR) */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSRC         : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __scs_syst_csr_bits;

/* SysTick Reload Value Register (SCS_SYST_RVR) */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __scs_syst_rvr_bits;

/* SysTick Current Value Register (SCS_SYST_CVR) */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __scs_syst_cvr_bits;

/* Interrupt Set-Enable Registers 0-31 (SCS_NVIC_ISER) */
typedef struct {
  __REG32  SETENA0        : 1;
  __REG32  SETENA1        : 1;
  __REG32  SETENA2        : 1;
  __REG32  SETENA3        : 1;
  __REG32  SETENA4        : 1;
  __REG32  SETENA5        : 1;
  __REG32  SETENA6        : 1;
  __REG32  SETENA7        : 1;
  __REG32  SETENA8        : 1;
  __REG32  SETENA9        : 1;
  __REG32  SETENA10       : 1;
  __REG32  SETENA11       : 1;
  __REG32  SETENA12       : 1;
  __REG32  SETENA13       : 1;
  __REG32  SETENA14       : 1;
  __REG32  SETENA15       : 1;
  __REG32  SETENA16       : 1;
  __REG32  SETENA17       : 1;
  __REG32  SETENA18       : 1;
  __REG32  SETENA19       : 1;
  __REG32  SETENA20       : 1;
  __REG32  SETENA21       : 1;
  __REG32  SETENA22       : 1;
  __REG32  SETENA23       : 1;
  __REG32  SETENA24       : 1;
  __REG32  SETENA25       : 1;
  __REG32  SETENA26       : 1;
  __REG32  SETENA27       : 1;
  __REG32  SETENA28       : 1;
  __REG32  SETENA29       : 1;
  __REG32  SETENA30       : 1;
  __REG32  SETENA31       : 1;
} __scs_nvic_iser_bits;

/* Interrupt Clear-Enable Registers 0-31 (SCS_NVIC_ICER) */
typedef struct {
  __REG32  CLRENA0        : 1;
  __REG32  CLRENA1        : 1;
  __REG32  CLRENA2        : 1;
  __REG32  CLRENA3        : 1;
  __REG32  CLRENA4        : 1;
  __REG32  CLRENA5        : 1;
  __REG32  CLRENA6        : 1;
  __REG32  CLRENA7        : 1;
  __REG32  CLRENA8        : 1;
  __REG32  CLRENA9        : 1;
  __REG32  CLRENA10       : 1;
  __REG32  CLRENA11       : 1;
  __REG32  CLRENA12       : 1;
  __REG32  CLRENA13       : 1;
  __REG32  CLRENA14       : 1;
  __REG32  CLRENA15       : 1;
  __REG32  CLRENA16       : 1;
  __REG32  CLRENA17       : 1;
  __REG32  CLRENA18       : 1;
  __REG32  CLRENA19       : 1;
  __REG32  CLRENA20       : 1;
  __REG32  CLRENA21       : 1;
  __REG32  CLRENA22       : 1;
  __REG32  CLRENA23       : 1;
  __REG32  CLRENA24       : 1;
  __REG32  CLRENA25       : 1;
  __REG32  CLRENA26       : 1;
  __REG32  CLRENA27       : 1;
  __REG32  CLRENA28       : 1;
  __REG32  CLRENA29       : 1;
  __REG32  CLRENA30       : 1;
  __REG32  CLRENA31       : 1;
} __scs_nvic_icer_bits;

/* Interrupt Set-Pending Register 0-31 (SCS_NVIC_ISPR) */
typedef struct {
  __REG32  SETPEND0       : 1;
  __REG32  SETPEND1       : 1;
  __REG32  SETPEND2       : 1;
  __REG32  SETPEND3       : 1;
  __REG32  SETPEND4       : 1;
  __REG32  SETPEND5       : 1;
  __REG32  SETPEND6       : 1;
  __REG32  SETPEND7       : 1;
  __REG32  SETPEND8       : 1;
  __REG32  SETPEND9       : 1;
  __REG32  SETPEND10      : 1;
  __REG32  SETPEND11      : 1;
  __REG32  SETPEND12      : 1;
  __REG32  SETPEND13      : 1;
  __REG32  SETPEND14      : 1;
  __REG32  SETPEND15      : 1;
  __REG32  SETPEND16      : 1;
  __REG32  SETPEND17      : 1;
  __REG32  SETPEND18      : 1;
  __REG32  SETPEND19      : 1;
  __REG32  SETPEND20      : 1;
  __REG32  SETPEND21      : 1;
  __REG32  SETPEND22      : 1;
  __REG32  SETPEND23      : 1;
  __REG32  SETPEND24      : 1;
  __REG32  SETPEND25      : 1;
  __REG32  SETPEND26      : 1;
  __REG32  SETPEND27      : 1;
  __REG32  SETPEND28      : 1;
  __REG32  SETPEND29      : 1;
  __REG32  SETPEND30      : 1;
  __REG32  SETPEND31      : 1;
} __scs_nvic_ispr_bits;

/* Interrupt Clear-Pending Register 0-31 (SCS_NVIC_ICPR) */
typedef struct {
  __REG32  CLRPEND0       : 1;
  __REG32  CLRPEND1       : 1;
  __REG32  CLRPEND2       : 1;
  __REG32  CLRPEND3       : 1;
  __REG32  CLRPEND4       : 1;
  __REG32  CLRPEND5       : 1;
  __REG32  CLRPEND6       : 1;
  __REG32  CLRPEND7       : 1;
  __REG32  CLRPEND8       : 1;
  __REG32  CLRPEND9       : 1;
  __REG32  CLRPEND10      : 1;
  __REG32  CLRPEND11      : 1;
  __REG32  CLRPEND12      : 1;
  __REG32  CLRPEND13      : 1;
  __REG32  CLRPEND14      : 1;
  __REG32  CLRPEND15      : 1;
  __REG32  CLRPEND16      : 1;
  __REG32  CLRPEND17      : 1;
  __REG32  CLRPEND18      : 1;
  __REG32  CLRPEND19      : 1;
  __REG32  CLRPEND20      : 1;
  __REG32  CLRPEND21      : 1;
  __REG32  CLRPEND22      : 1;
  __REG32  CLRPEND23      : 1;
  __REG32  CLRPEND24      : 1;
  __REG32  CLRPEND25      : 1;
  __REG32  CLRPEND26      : 1;
  __REG32  CLRPEND27      : 1;
  __REG32  CLRPEND28      : 1;
  __REG32  CLRPEND29      : 1;
  __REG32  CLRPEND30      : 1;
  __REG32  CLRPEND31      : 1;
} __scs_nvic_icpr_bits;

/* Interrupt Priority Registers 0-3 (SCS_NVIC_IPR0) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_0          : 2;
  __REG32                 : 6;
  __REG32  PRI_1          : 2;
  __REG32                 : 6;
  __REG32  PRI_2          : 2;
  __REG32                 : 6;
  __REG32  PRI_3          : 2;
} __scs_nvic_ipr0_bits;

/* Interrupt Priority Registers 4-7 (SCS_NVIC_IPR1) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_4          : 2;
  __REG32                 : 6;
  __REG32  PRI_5          : 2;
  __REG32                 : 6;
  __REG32  PRI_6          : 2;
  __REG32                 : 6;
  __REG32  PRI_7          : 2;
} __scs_nvic_ipr1_bits;

/* Interrupt Priority Registers 8-11 (SCS_NVIC_IPR2) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_8          : 2;
  __REG32                 : 6;
  __REG32  PRI_9          : 2;
  __REG32                 : 6;
  __REG32  PRI_10         : 2;
  __REG32                 : 6;
  __REG32  PRI_11         : 2;
} __scs_nvic_ipr2_bits;

/* Interrupt Priority Registers 12-15 (SCS_NVIC_IPR3) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_12         : 2;
  __REG32                 : 6;
  __REG32  PRI_13         : 2;
  __REG32                 : 6;
  __REG32  PRI_14         : 2;
  __REG32                 : 6;
  __REG32  PRI_15         : 2;
} __scs_nvic_ipr3_bits;

/* Interrupt Priority Registers 16-19 (SCS_NVIC_IPR4) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_16         : 2;
  __REG32                 : 6;
  __REG32  PRI_17         : 2;
  __REG32                 : 6;
  __REG32  PRI_18         : 2;
  __REG32                 : 6;
  __REG32  PRI_19         : 2;
} __scs_nvic_ipr4_bits;

/* Interrupt Priority Registers 20-23 (SCS_NVIC_IPR5) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_20         : 2;
  __REG32                 : 6;
  __REG32  PRI_21         : 2;
  __REG32                 : 6;
  __REG32  PRI_22         : 2;
  __REG32                 : 6;
  __REG32  PRI_23         : 2;
} __scs_nvic_ipr5_bits;

/* Interrupt Priority Registers 24-27 (SCS_NVIC_IPR6) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_24         : 2;
  __REG32                 : 6;
  __REG32  PRI_25         : 2;
  __REG32                 : 6;
  __REG32  PRI_26         : 2;
  __REG32                 : 6;
  __REG32  PRI_27         : 2;
} __scs_nvic_ipr6_bits;

/* Interrupt Priority Registers 28-31 (SCS_NVIC_IPR7) */
typedef struct {
  __REG32                 : 6;
  __REG32  PRI_28         : 2;
  __REG32                 : 6;
  __REG32  PRI_29         : 2;
  __REG32                 : 6;
  __REG32  PRI_30         : 2;
  __REG32                 : 6;
  __REG32  PRI_31         : 2;
} __scs_nvic_ipr7_bits;

/* CPUID Register (SCS_CPIUD) */
typedef struct {
  __REG32  REVISION       : 4;
  __REG32  PARTNO         :12;
  __REG32  PART           : 4;
  __REG32                 : 4;
  __REG32  IMPLEMENTER    : 8;
} __scs_cpuid_bits;

/* Interrupt Control State Register (SCS_ICSR) */
typedef struct {
  __REG32  VECTACTIVE     : 6;
  __REG32                 : 6;
  __REG32  VECTPENDING    : 6;
  __REG32                 : 4;
  __REG32  ISRPENDING     : 1;
  __REG32  ISRPREEMPT     : 1;
  __REG32                 : 1;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __scs_icsr_bits;

/* Application Interrupt and Reset Control Register (SCS_AIRCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 VECTCLRACTIVE   : 1;
  __REG32 SYSRESETREQ     : 1;
  __REG32                 : 13;
  __REG32 VECTORKEY       : 16;
} __scs_aircr_bits;

/* System Control Register (SCS_SCR) */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scs_scr_bits;

/* System Handler Priority Register 2 (SCS_SHPR2) */
typedef struct {
  __REG32                 :24;
  __REG32  PRI_11         : 8;
} __scs_shpr2_bits;

/* System Handler Priority Register 3 (SCS_SHPR3) */
typedef struct {
  __REG32                 :16;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __scs_shpr3_bits;

/* Interrupt Source Identify Register (INT_IRQn_SRC) */
typedef struct{
  __REG32 INT_SRC0        : 1;
  __REG32 INT_SRC1        : 1;
  __REG32 INT_SRC2        : 1;
  __REG32                 :29;
} __int_irqx_src_bits;

/* Interrupt Source Identify Register 6-7 (INT_IRQn_SRC) */
typedef struct{
  __REG32 INT_SRC0        : 1;
  __REG32 INT_SRC1        : 1;
  __REG32 INT_SRC2        : 1;
  __REG32 INT_SRC3        : 1;
  __REG32                 :28;
} __int_irq67_src_bits;

/* NMI Interrupt Source Select Control Register (INT_NMI_SEL) */
typedef struct{
  __REG32 NMI_SEL         : 5;
  __REG32                 : 2;
  __REG32 INT_TEST        : 1;
  __REG32                 :24;
} __int_nmi_sel_bits;

/* MCU Interrupt Request Source Register (INT_MCU_IRQ) */
typedef struct{
  __REG32 MCU_IRQ0        : 1;
  __REG32 MCU_IRQ1        : 1;
  __REG32 MCU_IRQ2        : 1;
  __REG32 MCU_IRQ3        : 1;
  __REG32 MCU_IRQ4        : 1;
  __REG32 MCU_IRQ5        : 1;
  __REG32 MCU_IRQ6        : 1;
  __REG32 MCU_IRQ7        : 1;
  __REG32 MCU_IRQ8        : 1;
  __REG32 MCU_IRQ9        : 1;
  __REG32 MCU_IRQ10       : 1;
  __REG32 MCU_IRQ11       : 1;
  __REG32 MCU_IRQ12       : 1;
  __REG32 MCU_IRQ13       : 1;
  __REG32 MCU_IRQ14       : 1;
  __REG32 MCU_IRQ15       : 1;
  __REG32 MCU_IRQ16       : 1;
  __REG32 MCU_IRQ17       : 1;
  __REG32 MCU_IRQ18       : 1;
  __REG32 MCU_IRQ19       : 1;
  __REG32 MCU_IRQ20       : 1;
  __REG32 MCU_IRQ21       : 1;
  __REG32 MCU_IRQ22       : 1;
  __REG32 MCU_IRQ23       : 1;
  __REG32 MCU_IRQ24       : 1;
  __REG32 MCU_IRQ25       : 1;
  __REG32 MCU_IRQ26       : 1;
  __REG32 MCU_IRQ27       : 1;
  __REG32 MCU_IRQ28       : 1;
  __REG32 MCU_IRQ29       : 1;
  __REG32 MCU_IRQ30       : 1;
  __REG32 MCU_IRQ31       : 1;
} __int_mcu_irq_bits;

/* Power Down Control Register (CLK_PWRCON) */
typedef struct{
__REG32 XTL12M_EN       : 1;
__REG32 XTL32K_EN       : 1;
__REG32 OSC22M_EN       : 1;
__REG32 OSC10K_EN       : 1;
__REG32 PD_WU_DLY       : 1;
__REG32 PD_WU_INT_EN    : 1;
__REG32 PD_WU_STS       : 1;
__REG32 PWR_DOWN_EN     : 1;
__REG32 PD_WAIT_CPU     : 1;
__REG32                 :23;
} __clk_pwrcon_bits;

/* AHB Devices Clock Enable Control Register (CLK_AHBCLK) */
typedef struct{
__REG32                 : 2;
__REG32 ISP_EN          : 1;
__REG32                 :29;
} __clk_ahbclk_bits;

/* APB Devices Clock Enable Control Register (CLK_APBCLK) */
typedef struct{
__REG32 WDT_EN          : 1;
__REG32 RTC_EN          : 1;
__REG32 TMR0_EN         : 1;
__REG32 TMR1_EN         : 1;
__REG32 TMR2_EN         : 1;
__REG32 TMR3_EN         : 1;
__REG32                 : 3;
__REG32 I2C1_EN         : 1;
__REG32                 : 2;
__REG32 SPI0_EN         : 1;
__REG32 SPI1_EN         : 1;
__REG32                 : 2;
__REG32 UART0_EN        : 1;
__REG32 UART1_EN        : 1;
__REG32                 : 2;
__REG32 PWM01_EN        : 1;
__REG32 PWM23_EN        : 1;
__REG32                 : 5;
__REG32 USBD_EN         : 1;
__REG32                 : 3;
__REG32 PS2_EN          : 1;
} __clk_apbclk_bits;

/* Clock status monitor Register (CLK_CLKSTATUS) */
typedef struct {
  __REG32 XTL12M_STB      : 1;
  __REG32 XTL32K_STB      : 1;
  __REG32 PLL_STB         : 1;
  __REG32 OSC10K_STB      : 1;
  __REG32 OSC22M_STB      : 1;
  __REG32                 : 2;
  __REG32 CLK_SW_FAIL     : 1;
  __REG32                 : 24;
} __clk_clkstatus_bits;

/* Clock Source Select Control Register 0 (CLK_CLKSEL0) */
typedef struct{
__REG32 HCLK_S          : 3;
__REG32 STCLK_S         : 3;
__REG32                 :26;
} __clk_clksel0_bits;

/* Clock Source Select Control Register  1 (CLK_CLKSEL1) */
typedef struct{
__REG32 WDT_S           : 2;
__REG32                 : 6;
__REG32 TMR0_S          : 3;
__REG32                 : 1;
__REG32 TMR1_S          : 3;
__REG32                 : 1;
__REG32 TMR2_S          : 3;
__REG32                 : 1;
__REG32 TMR3_S          : 3;
__REG32                 : 1;
__REG32 UART_S          : 2;
__REG32                 : 2;
__REG32 PWM01_S         : 2;
__REG32 PWM23_S         : 2;
} __clk_clksel1_bits;

/* Clock Divider Register (CLK_CLKDIV) */
typedef struct{
__REG32 HCLK_N          : 4;
__REG32 USB_N           : 4;
__REG32 UART_N          : 4;
__REG32                 :20;
} __clk_clkdiv_bits;

/* PLL Control Register (CLK_PLLCON) */
typedef struct{
__REG32 FB_DV           : 9;
__REG32 IN_DV           : 5;
__REG32 OUT_DV          : 2;
__REG32 PD              : 1;
__REG32 BP              : 1;
__REG32 OE              : 1;
__REG32 PLL_SRC         : 1;
__REG32                 :12;
} __clk_pllcon_bits;

/* USB Interrupt Enable Register (USB_INTEN) */
typedef struct{
__REG32 BUS_IE          : 1;
__REG32 USB_IE          : 1;
__REG32 FLDET_IE        : 1;
__REG32 WAKEUP_IE       : 1;
__REG32                 : 4;
__REG32 WAKEUP_EN       : 1;
__REG32                 : 6;
__REG32 INNAK_EN        : 1;
__REG32                 :16;
} __usb_inten_bits;

/* USB Interrupt Event Status Register (USB_INTSTS) */
typedef struct{
__REG32 BUS_STS         : 1;
__REG32 USB_STS         : 1;
__REG32 FLDET_STS       : 1;
__REG32 WAKEUP_STS      : 1;
__REG32                 :12;
__REG32 EPEVT0          : 1;
__REG32 EPEVT1          : 1;
__REG32 EPEVT2          : 1;
__REG32 EPEVT3          : 1;
__REG32 EPEVT4          : 1;
__REG32 EPEVT5          : 1;
__REG32                 : 9;
__REG32 SETUP           : 1;
} __usb_intsts_bits;

/* USB Function Address Register (USB_FADDR) */
typedef struct{
__REG32 FADDR           : 7;
__REG32                 :25;
} __usb_faddr_bits;

/* USB Endpoint Status Register (USB_EPSTS) */
typedef struct{
__REG32                 : 7;
__REG32 OVERRUN         : 1;
__REG32 EPSTS0          : 3;
__REG32 EPSTS1          : 3;
__REG32 EPSTS2          : 3;
__REG32 EPSTS3          : 3;
__REG32 EPSTS4          : 3;
__REG32 EPSTS5          : 3;
__REG32                 : 6;
} __usb_epsts_bits;

/* USB Bus States & Attribution Register (USB_ATTR) */
typedef struct{
__REG32 USBRST          : 1;
__REG32 SUSPEND         : 1;
__REG32 RESUME          : 1;
__REG32 TIMEOUT         : 1;
__REG32 PHY_EN          : 1;
__REG32 RWAKEUP         : 1;
__REG32                 : 1;
__REG32 USB_EN          : 1;
__REG32 DPPU_EN         : 1;
__REG32 PWRDN           : 1;
__REG32 BYTEM           : 1;
__REG32                 :21;
} __usb_attr_bits;

/* USB Floating detection Register (USB_FLDET) */
typedef struct{
__REG32 FLDET           : 1;
__REG32                 :31;
} __usb_fldet_bits;

/* USB Buffer Segmentation Register (USB_BUFSEGx) x = 0~5 */
typedef struct{
__REG32                 : 3;
__REG32 BUFSEG          : 6;
__REG32                 :23;
} __usb_bufseg_bits;

/* USB Maximal Payload Register (USB_MXPLDx) x = 0~5 */
typedef struct{
__REG32 MXPLD           : 9;
__REG32                 :23;
} __usb_mxpld_bits;

/* USB Configuration Register (USB_CFGx) x = 0~5 */
typedef struct{
__REG32 EPT             : 4;
__REG32 ISOCH           : 1;
__REG32 STATE           : 2;
__REG32 DSQ_SYNC        : 1;
__REG32                 : 1;
__REG32 STALL           : 1;
__REG32                 :22;
} __usb_cfg_bits;

/* USB Extra Configuration Register (USB_CFGPx) x = 0~5 */
typedef struct{
__REG32 CLRRDY          : 1;
__REG32 SSTALL          : 1;
__REG32                 :30;
} __usb_cfgp_bits;

/* Drive SE0 Register (USB_DRVSE0) */
typedef struct{
__REG32 DRVSE0          : 1;
__REG32                 :31;
} __usb_drvse0_bits;

/* GPIO Port [A/B/C/D/E] Bit Mode Control (GP_GPIOx_PMD) */
typedef struct{
__REG32 PMD0            : 2;
__REG32 PMD1            : 2;
__REG32 PMD2            : 2;
__REG32 PMD3            : 2;
__REG32 PMD4            : 2;
__REG32 PMD5            : 2;
__REG32 PMD6            : 2;
__REG32 PMD7            : 2;
__REG32 PMD8            : 2;
__REG32 PMD9            : 2;
__REG32 PMD10           : 2;
__REG32 PMD11           : 2;
__REG32 PMD12           : 2;
__REG32 PMD13           : 2;
__REG32 PMD14           : 2;
__REG32 PMD15           : 2;
} __gp_gpiox_pmd_bits;

/* GPIO Port [A/B/C/D/E] Bit OFF Digital Resistor Enable (GP_GPIOx_OFFD) */
typedef struct{
__REG32                 :16;
__REG32 OFFD0           : 1;
__REG32 OFFD1           : 1;
__REG32 OFFD2           : 1;
__REG32 OFFD3           : 1;
__REG32 OFFD4           : 1;
__REG32 OFFD5           : 1;
__REG32 OFFD6           : 1;
__REG32 OFFD7           : 1;
__REG32 OFFD8           : 1;
__REG32 OFFD9           : 1;
__REG32 OFFD10          : 1;
__REG32 OFFD11          : 1;
__REG32 OFFD12          : 1;
__REG32 OFFD13          : 1;
__REG32 OFFD14          : 1;
__REG32 OFFD15          : 1;
} __gp_gpiox_offd_bits;

/* GPIO Port [A/B/C/D/E] Data Output Value (GP_GPIOx_DOUT) */
typedef struct{
__REG32 DOUT0           : 1;
__REG32 DOUT1           : 1;
__REG32 DOUT2           : 1;
__REG32 DOUT3           : 1;
__REG32 DOUT4           : 1;
__REG32 DOUT5           : 1;
__REG32 DOUT6           : 1;
__REG32 DOUT7           : 1;
__REG32 DOUT8           : 1;
__REG32 DOUT9           : 1;
__REG32 DOUT10          : 1;
__REG32 DOUT11          : 1;
__REG32 DOUT12          : 1;
__REG32 DOUT13          : 1;
__REG32 DOUT14          : 1;
__REG32 DOUT15          : 1;
__REG32                 :16;
} __gp_gpiox_dout_bits;

/* GPIO Port [A/B/C/D/E] Data Output Write Mask (GP_GPIOx_DMASK) */
typedef struct{
__REG32 DMASK0          : 1;
__REG32 DMASK1          : 1;
__REG32 DMASK2          : 1;
__REG32 DMASK3          : 1;
__REG32 DMASK4          : 1;
__REG32 DMASK5          : 1;
__REG32 DMASK6          : 1;
__REG32 DMASK7          : 1;
__REG32 DMASK8          : 1;
__REG32 DMASK9          : 1;
__REG32 DMASK10         : 1;
__REG32 DMASK11         : 1;
__REG32 DMASK12         : 1;
__REG32 DMASK13         : 1;
__REG32 DMASK14         : 1;
__REG32 DMASK15         : 1;
__REG32                 :16;
} __gp_gpiox_dmask_bits;

/* GPIO Port [A/B/C/D/E] Pin Value (GP_GPIOx_PIN) */
typedef struct{
__REG32 PIN0            : 1;
__REG32 PIN1            : 1;
__REG32 PIN2            : 1;
__REG32 PIN3            : 1;
__REG32 PIN4            : 1;
__REG32 PIN5            : 1;
__REG32 PIN6            : 1;
__REG32 PIN7            : 1;
__REG32 PIN8            : 1;
__REG32 PIN9            : 1;
__REG32 PIN10           : 1;
__REG32 PIN11           : 1;
__REG32 PIN12           : 1;
__REG32 PIN13           : 1;
__REG32 PIN14           : 1;
__REG32 PIN15           : 1;
__REG32                 :16;
} __gp_gpiox_pin_bits;

/* GPIO Port [A/B/C/D/E] De-bounce Enable (GP_GPIOx_DBEN) */
typedef struct{
__REG32 DBEN0           : 1;
__REG32 DBEN1           : 1;
__REG32 DBEN2           : 1;
__REG32 DBEN3           : 1;
__REG32 DBEN4           : 1;
__REG32 DBEN5           : 1;
__REG32 DBEN6           : 1;
__REG32 DBEN7           : 1;
__REG32 DBEN8           : 1;
__REG32 DBEN9           : 1;
__REG32 DBEN10          : 1;
__REG32 DBEN11          : 1;
__REG32 DBEN12          : 1;
__REG32 DBEN13          : 1;
__REG32 DBEN14          : 1;
__REG32 DBEN15          : 1;
__REG32                 :16;
} __gp_gpiox_dben_bits;

/* GPIO Port [A/B/C/D/E] Interrupt Mode Control (GP_GPIOx_IMD) */
typedef struct{
__REG32 IMD0            : 1;
__REG32 IMD1            : 1;
__REG32 IMD2            : 1;
__REG32 IMD3            : 1;
__REG32 IMD4            : 1;
__REG32 IMD5            : 1;
__REG32 IMD6            : 1;
__REG32 IMD7            : 1;
__REG32 IMD8            : 1;
__REG32 IMD9            : 1;
__REG32 IMD10           : 1;
__REG32 IMD11           : 1;
__REG32 IMD12           : 1;
__REG32 IMD13           : 1;
__REG32 IMD14           : 1;
__REG32 IMD15           : 1;
__REG32                 :16;
} __gp_gpiox_imd_bits;

/* GPIO Port [A/B/C/D] Interrupt Enable Control (GP_GPIOx_IEN) */
typedef struct{
__REG32 IF_EN0          : 1;
__REG32 IF_EN1          : 1;
__REG32 IF_EN2          : 1;
__REG32 IF_EN3          : 1;
__REG32 IF_EN4          : 1;
__REG32 IF_EN5          : 1;
__REG32 IF_EN6          : 1;
__REG32 IF_EN7          : 1;
__REG32 IF_EN8          : 1;
__REG32 IF_EN9          : 1;
__REG32 IF_EN10         : 1;
__REG32 IF_EN11         : 1;
__REG32 IF_EN12         : 1;
__REG32 IF_EN13         : 1;
__REG32 IF_EN14         : 1;
__REG32 IF_EN15         : 1;
__REG32 IR_EN0          : 1;
__REG32 IR_EN1          : 1;
__REG32 IR_EN2          : 1;
__REG32 IR_EN3          : 1;
__REG32 IR_EN4          : 1;
__REG32 IR_EN5          : 1;
__REG32 IR_EN6          : 1;
__REG32 IR_EN7          : 1;
__REG32 IR_EN8          : 1;
__REG32 IR_EN9          : 1;
__REG32 IR_EN10         : 1;
__REG32 IR_EN11         : 1;
__REG32 IR_EN12         : 1;
__REG32 IR_EN13         : 1;
__REG32 IR_EN14         : 1;
__REG32 IR_EN15         : 1;
} __gp_gpiox_ien_bits;

/* GPIO Port [A/B/C/D/E] Interrupt Trigger Source (GP_GPIOx_ISRC) */
typedef struct{
__REG32 ISRC0           : 1;
__REG32 ISRC1           : 1;
__REG32 ISRC2           : 1;
__REG32 ISRC3           : 1;
__REG32 ISRC4           : 1;
__REG32 ISRC5           : 1;
__REG32 ISRC6           : 1;
__REG32 ISRC7           : 1;
__REG32 ISRC8           : 1;
__REG32 ISRC9           : 1;
__REG32 ISRC10          : 1;
__REG32 ISRC11          : 1;
__REG32 ISRC12          : 1;
__REG32 ISRC13          : 1;
__REG32 ISRC14          : 1;
__REG32 ISRC15          : 1;
__REG32                 :16;
} __gp_gpiox_isrc_bits;

/* Interrupt De-bounce Cycle Control (GP_DBNCECON) */
typedef struct{
__REG32 DBCLKSEL        : 4;
__REG32 DBCLKSRC        : 1;
__REG32 ICLK_ON         : 1;
__REG32                 :26;
} __gp_dbncecon_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA0_DOUT) */
typedef struct{
__REG32 GPIOA0_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa0_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA1_DOUT) */
typedef struct{
__REG32 GPIOA1_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa1_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA2_DOUT) */
typedef struct{
__REG32 GPIOA2_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa2_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA3_DOUT) */
typedef struct{
__REG32 GPIOA3_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa3_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA4_DOUT) */
typedef struct{
__REG32 GPIOA4_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa4_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA5_DOUT) */
typedef struct{
__REG32 GPIOA5_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa5_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA6_DOUT) */
typedef struct{
__REG32 GPIOA6_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa6_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA7_DOUT) */
typedef struct{
__REG32 GPIOA7_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa7_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA8_DOUT) */
typedef struct{
__REG32 GPIOA8_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa8_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA9_DOUT) */
typedef struct{
__REG32 GPIOA9_DOUT     : 1;
__REG32                 :31;
} __gp_gpioa9_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA10_DOUT) */
typedef struct{
__REG32 GPIOA10_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa10_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA11_DOUT) */
typedef struct{
__REG32 GPIOA11_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa11_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA12_DOUT) */
typedef struct{
__REG32 GPIOA12_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa12_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA13_DOUT) */
typedef struct{
__REG32 GPIOA13_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa13_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA14_DOUT) */
typedef struct{
__REG32 GPIOA14_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa14_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOA15_DOUT) */
typedef struct{
__REG32 GPIOA15_DOUT    : 1;
__REG32                 :31;
} __gp_gpioa15_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB0_DOUT) */
typedef struct{
__REG32 GPIOB0_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob0_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB1_DOUT) */
typedef struct{
__REG32 GPIOB1_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob1_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB2_DOUT) */
typedef struct{
__REG32 GPIOB2_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob2_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB3_DOUT) */
typedef struct{
__REG32 GPIOB3_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob3_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB4_DOUT) */
typedef struct{
__REG32 GPIOB4_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob4_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB5_DOUT) */
typedef struct{
__REG32 GPIOB5_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob5_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB6_DOUT) */
typedef struct{
__REG32 GPIOB6_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob6_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB7_DOUT) */
typedef struct{
__REG32 GPIOB7_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob7_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB8_DOUT) */
typedef struct{
__REG32 GPIOB8_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob8_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB9_DOUT) */
typedef struct{
__REG32 GPIOB9_DOUT     : 1;
__REG32                 :31;
} __gp_gpiob9_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GPIOB10_DOUT) */
typedef struct{
__REG32 GPIOB10_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob10_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB11_DOUT) */
typedef struct{
__REG32 GPIOB11_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob11_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB12_DOUT) */
typedef struct{
__REG32 GPIOB12_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob12_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB13_DOUT) */
typedef struct{
__REG32 GPIOB13_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob13_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB14_DOUT) */
typedef struct{
__REG32 GPIOB14_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob14_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOB15_DOUT) */
typedef struct{
__REG32 GPIOB15_DOUT    : 1;
__REG32                 :31;
} __gp_gpiob15_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC0_DOUT) */
typedef struct{
__REG32 GPIOC0_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc0_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC1_DOUT) */
typedef struct{
__REG32 GPIOC1_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc1_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC2_DOUT) */
typedef struct{
__REG32 GPIOC2_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc2_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC3_DOUT) */
typedef struct{
__REG32 GPIOC3_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc3_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC4_DOUT) */
typedef struct{
__REG32 GPIOC4_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc4_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC5_DOUT) */
typedef struct{
__REG32 GPIOC5_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc5_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC6_DOUT) */
typedef struct{
__REG32 GPIOC6_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc6_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC7_DOUT) */
typedef struct{
__REG32 GPIOC7_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc7_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC8_DOUT) */
typedef struct{
__REG32 GPIOC8_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc8_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC9_DOUT) */
typedef struct{
__REG32 GPIOC9_DOUT     : 1;
__REG32                 :31;
} __gp_gpioc9_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC10_DOUT) */
typedef struct{
__REG32 GPIOC10_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc10_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC11_DOUT) */
typedef struct{
__REG32 GPIOC11_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc11_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC12_DOUT) */
typedef struct{
__REG32 GPIOC12_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc12_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC13_DOUT) */
typedef struct{
__REG32 GPIOC13_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc13_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC14_DOUT) */
typedef struct{
__REG32 GPIOC14_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc14_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOC15_DOUT) */
typedef struct{
__REG32 GPIOC15_DOUT    : 1;
__REG32                 :31;
} __gp_gpioc15_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD0_DOUT) */
typedef struct{
__REG32 GPIOD0_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod0_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD1_DOUT) */
typedef struct{
__REG32 GPIOD1_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod1_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD2_DOUT) */
typedef struct{
__REG32 GPIOD2_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod2_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD3_DOUT) */
typedef struct{
__REG32 GPIOD3_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod3_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD4_DOUT) */
typedef struct{
__REG32 GPIOD4_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod4_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD5_DOUT) */
typedef struct{
__REG32 GPIOD5_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod5_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD6_DOUT) */
typedef struct{
__REG32 GPIOD6_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod6_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD7_DOUT) */
typedef struct{
__REG32 GPIOD7_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod7_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD8_DOUT) */
typedef struct{
__REG32 GPIOD8_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod8_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD9_DOUT) */
typedef struct{
__REG32 GPIOD9_DOUT     : 1;
__REG32                 :31;
} __gp_gpiod9_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD10_DOUT) */
typedef struct{
__REG32 GPIOD10_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod10_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD11_DOUT) */
typedef struct{
__REG32 GPIOD11_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod11_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD12_DOUT) */
typedef struct{
__REG32 GPIOD12_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod12_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD13_DOUT) */
typedef struct{
__REG32 GPIOD13_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod13_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD14_DOUT) */
typedef struct{
__REG32 GPIOD14_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod14_dout_bits;

/* GPIO Port [A/B/C/D] I/O Bit Output/Input Control (GP_GPIOD15_DOUT) */
typedef struct{
__REG32 GPIOD15_DOUT    : 1;
__REG32                 :31;
} __gp_gpiod15_dout_bits;

/* I2C CONTROL REGISTER (I2Cx_I2CON) */
typedef struct{
__REG32                 : 2;
__REG32 AA              : 1;
__REG32 SI              : 1;
__REG32 STO             : 1;
__REG32 STA             : 1;
__REG32 ENS1            : 1;
__REG32 EI              : 1;
__REG32                 :24;
} __i2cx_i2con_bits;

/* I2C DATA REGISTE (I2Cx_I2CDAT) */
typedef struct{
__REG32 I2DAT           : 8;
__REG32                 :24;
} __i2cx_i2cdat_bits;

/* I2C STATUS REGISTER (I2Cx_I2CSTATUS) */
typedef struct{
__REG32 I2STATUS        : 8;
__REG32                 :24;
} __i2cx_i2cstatus_bits;

/* I2C BAUD RATE CONTROL REGISTER (I2Cx_I2CLK) */
typedef struct{
__REG32 I2CLK           : 8;
__REG32                 :24;
} __i2cx_i2clk_bits;

/* I2C TIME-OUT COUNTER REGISTER (I2C_I2CTOC) */
typedef struct{
__REG32 TIF             : 1;
__REG32 DIV4            : 1;
__REG32 ENTI            : 1;
__REG32                 :29;
} __i2cx_i2ctoc_bits;

/* I2C SLAVE ADDRESS REGISTER (I2Cx_I2CADDRy) */
typedef struct{
__REG32 GC              : 1;
__REG32 I2ADDR          : 7;
__REG32                 :24;
} __i2cx_i2caddry_bits;

/* I2C SLAVE ADDRESS MASK REGISTER (I2Cx_I2CADMy) */
typedef struct{
__REG32 GC              : 1;
__REG32 I2ADM           : 7;
__REG32                 :24;
} __i2cx_i2cadmy_bits;

/* PWM Pre-scalar Register (PWMx_PPR) */
typedef struct {
  __REG32 CP01            : 8;
  __REG32 CP23            : 8;
  __REG32 DZI01           : 8;
  __REG32 DZI23           : 8;
} __pwmx_ppr_bits;

/* PWM Clock Select Register (PWMx_CSR) */
typedef struct {
  __REG32 CSR0            : 3;
  __REG32                 : 1;
  __REG32 CSR1            : 3;
  __REG32                 : 1;
  __REG32 CSR2            : 3;
  __REG32                 : 1;
  __REG32 CSR3            : 3;
  __REG32                 : 17;
} __pwmx_csr_bits;

/* PWM Control Register (PWMx_PCR) */
typedef struct {
  __REG32 CH0EN           : 1;
  __REG32                 : 1;
  __REG32 CH0INV          : 1;
  __REG32 CH0MOD          : 1;
  __REG32 DZEN01          : 1;
  __REG32 DZEN23          : 1;
  __REG32                 : 2;
  __REG32 CH1EN           : 1;
  __REG32                 : 1;
  __REG32 CH1INV          : 1;
  __REG32 CH1MOD          : 1;
  __REG32                 : 4;
  __REG32 CH2EN           : 1;
  __REG32                 : 1;
  __REG32 CH2INV          : 1;
  __REG32 CH2MOD          : 1;
  __REG32                 : 4;
  __REG32 CH3EN           : 1;
  __REG32                 : 1;
  __REG32 CH3INV          : 1;
  __REG32 CH3MOD          : 1;
  __REG32                 : 4;
} __pwmx_pcr_bits;

/* PWM Counter Register (PWMx_CNR0,PWMx_CNR1,PWMx_CNR2,PWMx_CNR3) */
typedef struct {
  __REG32 CNR             : 16;
  __REG32                 : 16;
} __pwmx_cnry_bits;

/* PWM Comparator Register (PWMx_CMR0,PWMx_CMR1,PWMx_CMR2,PWMx_CMR3) */
typedef struct {
  __REG32 CMR             : 16;
  __REG32                 : 16;
} __pwmx_cmry_bits;

/* PWM Data Register (PWMx_PDR0,PWMx_PDR1,PWMx_PDR2,PWMx_PDR3) */
typedef struct {
  __REG32 PDR             : 16;
  __REG32                 : 16;
} __pwmx_pdry_bits;

/* PWM Interrupt Enable Register (PWMx_PIER) */
typedef struct {
  __REG32 PWMIE0          : 1;
  __REG32 PWMIE1          : 1;
  __REG32 PWMIE2          : 1;
  __REG32 PWMIE3          : 1;
  __REG32                 : 28;
} __pwmx_pier_bits;

/* PWM Interrupt Indication Register (PWMx_PIIR) */
typedef struct {
  __REG32 PWMIF0          : 1;
  __REG32 PWMIF1          : 1;
  __REG32 PWMIF2          : 1;
  __REG32 PWMIF3          : 1;
  __REG32                 : 28;
} __pwmx_piir_bits;

/* Capture Control Register (PWMx_CCR0) */
typedef struct {
  __REG32 INV0            : 1;
  __REG32 CRL_IE0         : 1;
  __REG32 CFL_IE0         : 1;
  __REG32 CAPCH0EN        : 1;
  __REG32 CAPIF0          : 1;
  __REG32                 : 1;
  __REG32 CRLRI0          : 1;
  __REG32 CFLRI0          : 1;
  __REG32                 : 8;
  __REG32 INV1            : 1;
  __REG32 CRL_IE1         : 1;
  __REG32 CFL_IE1         : 1;
  __REG32 CAPCH1EN        : 1;
  __REG32 CAPIF1          : 1;
  __REG32                 : 1;
  __REG32 CRLRI1          : 1;
  __REG32 CFLRI1          : 1;
  __REG32                 : 8;
} __pwmx_ccr0_bits;

/* Capture Control Register (PWMx_CCR2) */
typedef struct {
  __REG32 INV2            : 1;
  __REG32 CRL_IE2         : 1;
  __REG32 CFL_IE2         : 1;
  __REG32 CAPCH2EN        : 1;
  __REG32 CAPIF2          : 1;
  __REG32                 : 1;
  __REG32 CRLRI2          : 1;
  __REG32 CFLRI2          : 1;
  __REG32                 : 8;
  __REG32 INV3            : 1;
  __REG32 CRL_IE3         : 1;
  __REG32 CFL_IE3         : 1;
  __REG32 CAPCH3EN        : 1;
  __REG32 CAPIF3          : 1;
  __REG32                 : 1;
  __REG32 CRLRI3          : 1;
  __REG32 CFLRI3          : 1;
  __REG32                 : 8;
} __pwmx_ccr2_bits;

/* Capture Rising Latch Register (PWMx_CRLR0,PWMx_CRLR1,PWMx_CRLR2,PWMx_CRLR3) */
typedef struct {
  __REG32 CRLR            : 16;
  __REG32                 : 16;
} __pwmx_crlry_bits;

/* Capture Falling Latch Register (PWMx_CFLR0,PWMx_CFLR1,PWMx_CFLR2,PWMx_CFLR3) */
typedef struct {
  __REG32 CFLR            : 16;
  __REG32                 : 16;
} __pwmx_cflry_bits;

/* Capture Input Enable Register (PWMx_CAPENR) */
typedef struct {
  __REG32 CAPENR          : 4;
  __REG32                 : 28;
} __pwmx_capenr_bits;

/* PWM Output Enable (PWMx_POE) */
typedef struct {
  __REG32 PWM0            : 1;
  __REG32 PWM1            : 1;
  __REG32 PWM2            : 1;
  __REG32 PWM3            : 1;
  __REG32                 : 28;
} __pwmx_poe_bits;

/* RTC Initiation Register (RTC_INIR) */
typedef struct{
__REG32 Active          : 1;
__REG32                 :31;
} __rtc_inir_bits;

/* RTC Access Enable Register (RTC_AER) */
typedef struct{
__REG32 AER             :16;
__REG32 ENF             : 1;
__REG32                 :15;
} __rtc_aer_bits;

/* RTC Frequency Compensation Register (RTC_FCR) */
typedef struct{
__REG32 FRACTION        : 6;
__REG32                 : 2;
__REG32 INTEGER         : 4;
__REG32                 :20;
} __rtc_fcr_bits;

/* RTC Time Loading Register (RTC_TLR) */
typedef struct{
__REG32 _1SEC           : 4;
__REG32 _10SEC          : 3;
__REG32                 : 1;
__REG32 _1MIN           : 4;
__REG32 _10MIN          : 3;
__REG32                 : 1;
__REG32 _1HR            : 4;
__REG32 _10HR           : 2;
__REG32                 :10;
} __rtc_tlr_bits;

/* RTC Calendar Loading Register (RTC_CLR) */
typedef struct{
__REG32 _1DAY           : 4;
__REG32 _10DAY          : 2;
__REG32                 : 2;
__REG32 _1MON           : 4;
__REG32 _10MON          : 1;
__REG32                 : 3;
__REG32 _1YEAR          : 4;
__REG32 _10YEAR         : 4;
__REG32                 : 8;
} __rtc_clr_bits;

/* RTC Time Scale Selection Register (RTC_TSSR) */
typedef struct{
__REG32 _24hr_12hr      : 1;
__REG32                 :31;
} __rtc_tssr_bits;

/* RTC Day of the Week Register (RTC_DWR) */
typedef struct{
__REG32 DWR             : 3;
__REG32                 :29;
} __rtc_dwr_bits;

/* RTC Leap year Indication Register (RTC_LIR) */
typedef struct{
__REG32 LIR             : 1;
__REG32                 :31;
} __rtc_lir_bits;

/* RTC Interrupt Enable Register (RTC_RIER) */
typedef struct{
__REG32 AIER            : 1;
__REG32 TIER            : 1;
__REG32                 :30;
} __rtc_rier_bits;

/* RTC Interrupt Indication Register (RTC_RIIR) */
typedef struct{
__REG32 AIF             : 1;
__REG32 TIF             : 1;
__REG32                 :30;
} __rtc_riir_bits;

/* RTC Time Tick Register (RTC_TTR) */
typedef struct{
__REG32 TTR             : 3;
__REG32 TWKE            : 1;
__REG32                 :28;
} __rtc_ttr_bits;

/* Control and Status Register (SPIx_SPI_CNTRL) */
typedef struct {
  __REG32 GO_BUSY         : 1;
  __REG32 RX_NEG          : 1;
  __REG32 TX_NEG          : 1;
  __REG32 TX_BIT_LEN      : 5;
  __REG32 TX_NUM          : 2;
  __REG32 LSB             : 1;
  __REG32 CLKP            : 1;
  __REG32 SP_CYCLE        : 4;
  __REG32 IF              : 1;
  __REG32 IE              : 1;
  __REG32 SLAVE           : 1;
  __REG32 REORDER         : 2;
  __REG32 FIFO            : 1;
  __REG32                 : 1;
  __REG32 VARCLK_EN       : 1;
  __REG32 RX_EMPTY        : 1;
  __REG32 RX_FULL         : 1;
  __REG32 TX_EMPTY        : 1;
  __REG32 TX_FULL         : 1;
  __REG32                 : 4;
} __spix_spi_cntrl_bits;

/* Clock Divider Register (SPIx_SPI_DIVIDER) */
typedef struct {
  __REG32 DIVIDER         : 16;
  __REG32 DIVIDER2        : 16;
} __spix_spi_divider_bits;

/* Slave Select Register (SPIx_SPI_SSR) */
typedef struct {
  __REG32 SSR             : 2;
  __REG32 SS_LVL          : 1;
  __REG32 AUTOSS          : 1;
  __REG32 SS_LTRIG        : 1;
  __REG32 LTRIG_FLAG      : 1;
  __REG32                 : 26;
} __spix_spi_ssr_bits;

/* Timer x Control and Status Register (TMRx_TCSRy) */
typedef struct{
__REG32 PRESCALE        : 8;
__REG32                 : 8;
__REG32 TDR_EN          : 1;
__REG32                 : 7;
__REG32 CTB             : 1;
__REG32 CACT            : 1;
__REG32 CRST            : 1;
__REG32 MODE            : 2;
__REG32 IE              : 1;
__REG32 CEN             : 1;
__REG32 DBGACK_TMR      : 1;
} __tmrx_tcsry_bits;

/* Timer x Compare Register (TMRx_TCMPRy) */
typedef struct{
__REG32 TCMP            :24;
__REG32                 : 8;
} __tmrx_tcmpry_bits;

/* Timer x Interrupt Status Register (TMRx_TISRy) */
typedef struct{
__REG32 TIF             : 1;
__REG32                 :31;
} __tmrx_tisry_bits;

/* Timer x Data Register (TMRx_TDRy) */
typedef struct{
__REG32 TDR             :24;
__REG32                 : 8;
} __tmrx_tdry_bits;

/* Watchdog Timer Control Register (WDT_WTCR) */
typedef struct{
__REG32 WTR             : 1;
__REG32 WTRE            : 1;
__REG32 WTRF            : 1;
__REG32 WTIF            : 1;
__REG32 WTWKE           : 1;
__REG32 WTWKF           : 1;
__REG32 WTIE            : 1;
__REG32 WTE             : 1;
__REG32 WTIS            : 3;
__REG32                 :20;
__REG32 DBGACK_WDT      : 1;
} __wdt_wtcr_bits;

/* Receive Buffer Register (UARTx_UA_RBR) */
/* Transmit Holding Register (UARTx_UA_THR) */
typedef union {
  /* UART0_UA_RBR */
  /* UART1_UA_RBR */
  struct {
    __REG32 RBR               : 8;
    __REG32                   :24;
  };
  /* UART0_UA_THR */
  /* UART1_UA_THR */
  struct {
    __REG32 THR               : 8;
    __REG32                   :24;  
  };
  
  struct {
    __REG32 DATA              : 8;
    __REG32                   :24;  
  };
} __uartx_ua_rbr_thr_bits;

/* Interrupt Enable Register (UARTx_UA_IER) */
typedef struct {
  __REG32 RDA_IEN           : 1;
  __REG32 THRE_IEN          : 1;
  __REG32 RLS_IEN           : 1;
  __REG32 MODEM_IEN         : 1;
  __REG32 RTO_IEN           : 1;
  __REG32 BUF_ERR_IEN       : 1;
  __REG32 WAKE_EN           : 1;
  __REG32                   : 1;
  __REG32                   : 3;
  __REG32 TIME_OUT_EN       : 1;
  __REG32 AUTO_RTS_EN       : 1;
  __REG32 AUTO_CTS_EN       : 1;  
  __REG32                   :18;
} __uartx_ua_ier_bits;

/* FIFO Control Register (UARTx_UA_FCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 RFR               : 1;
  __REG32 TFR               : 1;
  __REG32                   : 1;
  __REG32 RFITL             : 4;
  __REG32                   : 8;
  __REG32 RTS_TRI_LEV       : 4;
  __REG32                   :12;
} __uartx_ua_fcr_bits;

/* UART Line Control Register (UARTx_UA_LCR) */
typedef struct {
  __REG32 WLS               : 2;
  __REG32 NSB               : 1;
  __REG32 PBE               : 1;
  __REG32 EPE               : 1;
  __REG32 SPE               : 1;
  __REG32 BCB               : 1;
  __REG32                   :25;
} __uartx_ua_lcr_bits;

/* UART MODEM Control Register (UARTx_UA_MCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 RTS               : 1;
  __REG32                   : 7;
  __REG32 LEV_RTS           : 1;
  __REG32                   : 3;
  __REG32 RTS_ST            : 1;
  __REG32                   :18;
} __uartx_ua_mcr_bits;

/* UART Modem Status Register (UARTx_UA_MSR) */
typedef struct {
  __REG32 DCTSF             : 1;
  __REG32                   : 3;
  __REG32 CTS_ST            : 1;
  __REG32                   : 3;
  __REG32 LEV_CTS           : 1;
  __REG32                   :23;
} __uartx_ua_msr_bits;

/* UART FIFO Status Register (UARTx_UA_FSR) */
typedef struct {
  __REG32 RX_OVER_IF        : 1;
  __REG32                   : 2;
  __REG32 RS_485_ADD_DETF   : 1;
  __REG32 PEF               : 1;
  __REG32 FEF               : 1;
  __REG32 BIF               : 1;
  __REG32                   : 1;
  __REG32 RX_POINTER        : 6;
  __REG32 RX_EMPTY          : 1;
  __REG32 RX_FULL           : 1;
  __REG32 TX_POINTER        : 6;
  __REG32 TX_EMPTY          : 1;
  __REG32 TX_FULL           : 1;
  __REG32 TX_OVER_IF        : 1;
  __REG32                   : 3;
  __REG32 TE_FLAG           : 1;
  __REG32                   : 3;
} __uartx_ua_fsr_bits;

/* UART Interrupt Status Control Register (UARTx_UA_ISR) */
typedef struct {
  __REG32 RDA_IF              : 1;
  __REG32 THRE_IF             : 1;
  __REG32 RLS_IF              : 1;
  __REG32 MODEM_IF            : 1;
  __REG32 TOUT_IF             : 1;
  __REG32 BUF_ERR_IF          : 1;
  __REG32                     : 2;
  __REG32 RDA_INT             : 1;
  __REG32 THRE_INT            : 1;
  __REG32 RLS_INT             : 1;
  __REG32 MODEM_INT           : 1;
  __REG32 TOUT_INT            : 1;
  __REG32 BUF_ERR_INT         : 1;
  __REG32                     : 18;
} __uartx_ua_isr_bits;

/* UART Time out Register (UARTx_UA_TOR) */
typedef struct {
  __REG32 TOIC              : 8;
  __REG32 DLY               : 8;
  __REG32                   :16;
} __uartx_ua_tor_bits;

/* Baud Rate Divider Register (UARTx_UA_BAUD) */
typedef struct {
  __REG32 BRD               :16;
  __REG32                   : 8;
  __REG32 DIVIDER_X         : 4;
  __REG32 DIV_X_ONE         : 1;
  __REG32 DIV_X_EN          : 1;
  __REG32                   : 2;
} __uartx_ua_baud_bits;

/* IrDA Control Register (UARTx_UA_IRCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 TX_SELECT         : 1;
  __REG32                   : 3;
  __REG32 INV_TX            : 1;
  __REG32 INV_RX            : 1;
  __REG32                   :25;
} __uartx_ua_ircr_bits;

/* UART Alternate Control/Status Register (UARTx_UA_ALT_CSR) */
typedef struct {
  __REG32                   : 8;
  __REG32 RS_485_NMM        : 1;
  __REG32 RS_485_ADD        : 1;
  __REG32 RS_485_AUD        : 1;
  __REG32                   : 4;
  __REG32 RS_485_ADD_EN     : 1;
  __REG32                   : 8;
  __REG32 ADDR_MATCH        : 8;
} __uartx_ua_alt_csr_bits;

/* UART Function Select Register (UARTx_UA_FUN_SEL) */
typedef struct {
  __REG32 FUN_SEL           : 2;
  __REG32                   :30;
} __uartx_ua_fun_sel_bits;

/* PS2 Control Register (PS2_PS2CON) */
typedef struct {
  __REG32 PS2EN             : 1;
  __REG32 TXINTEN           : 1;
  __REG32 RXINTEN           : 1;
  __REG32 TXFIFO_DEPTH      : 4;
  __REG32 ACK               : 1;
  __REG32 CLRFIFO           : 1;
  __REG32 OVERRIDE          : 1;
  __REG32 FPS2CLK           : 1;
  __REG32 FPS2DAT           : 1;
  __REG32                   :20;
} __ps2_ps2con_bits;

/* PS2 Receiver DATA Register (PS2_PS2RXDATA ) */
typedef struct {
  __REG32 RXDATA            : 8;
  __REG32                   :24;
} __ps2_ps2rxdata_bits;

/* PS2 Status Register (PS2_PS2STATUS) */
typedef struct {
  __REG32 PS2CLK            : 1;
  __REG32 PS2DATA           : 1;
  __REG32 FRAMERR           : 1;
  __REG32 RXPARITY          : 1;
  __REG32 RXBUSY            : 1;
  __REG32 TXBUSY            : 1;
  __REG32 RXOVF             : 1;
  __REG32 TXEMPTY           : 1;
  __REG32 BYTEIDX           : 4;
  __REG32                   :20;
} __ps2_ps2status_bits;

/* PS2 Interrupt Identification Register (PS2_PS2INTID) */
typedef struct {
  __REG32 RXINT             : 1;
  __REG32 TXINT             : 1;
  __REG32                   :30;
} __ps2_ps2intid_bits;

/* ISP Control Register (FMC_ISPCON) */
typedef struct {
  __REG32 ISPEN             : 1;
  __REG32 BS                : 1;
  __REG32                   : 2;
  __REG32 CFGUEN            : 1;
  __REG32 LDUEN             : 1;
  __REG32 ISPFF             : 1;
  __REG32                   : 1;
  __REG32 PT                : 3;
  __REG32                   : 1;
  __REG32 ET                : 3;
  __REG32                   :17;
} __fmc_ispcon_bits;

/* ISP Command (FMc_ISPCMD) */
typedef struct {
  __REG32 FCTRL0            : 1;
  __REG32 FCTRL1            : 1;
  __REG32 FCTRL2            : 1;
  __REG32 FCTRL3            : 1;
  __REG32 FCEN              : 1;
  __REG32 FOEN              : 1;
  __REG32                   :26;
} __fmc_ispcmd_bits;

/* ISP Trigger Control Register (FMc_ISPTRG) */
typedef struct {
  __REG32 ISPGO             : 1;
  __REG32                   :31;
} __fmc_isptrg_bits;

/* Flash Access Time Control Register (FMc_FATCON) */
typedef struct {
  __REG32                   : 4;
  __REG32 LFOM              : 1;
  __REG32                   : 1;
  __REG32 MFOM              : 1;
  __REG32                   :25;
} __fmc_fatcon_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** SCS (System Controllers Space)
 **
 ***************************************************************************/
 
__IO_REG32_BIT(SCS_SYST_CSR,          0xE000E010,__READ_WRITE ,__scs_syst_csr_bits);
__IO_REG32_BIT(SCS_SYST_RVR,          0xE000E014,__READ_WRITE ,__scs_syst_rvr_bits);
__IO_REG32_BIT(SCS_SYST_CVR,          0xE000E018,__READ_WRITE ,__scs_syst_cvr_bits);
__IO_REG32_BIT(SCS_NVIC_ISER,         0xE000E100,__READ_WRITE ,__scs_nvic_iser_bits);
__IO_REG32_BIT(SCS_NVIC_ICER,         0xE000E180,__READ_WRITE ,__scs_nvic_icer_bits);
__IO_REG32_BIT(SCS_NVIC_ISPR,         0xE000E200,__READ_WRITE ,__scs_nvic_ispr_bits);
__IO_REG32_BIT(SCS_NVIC_ICPR,         0xE000E280,__READ_WRITE ,__scs_nvic_icpr_bits);
__IO_REG32_BIT(SCS_NVIC_IPR0,         0xE000E400,__READ_WRITE ,__scs_nvic_ipr0_bits);
__IO_REG32_BIT(SCS_NVIC_IPR1,         0xE000E404,__READ_WRITE ,__scs_nvic_ipr1_bits);
__IO_REG32_BIT(SCS_NVIC_IPR2,         0xE000E408,__READ_WRITE ,__scs_nvic_ipr2_bits);
__IO_REG32_BIT(SCS_NVIC_IPR3,         0xE000E40C,__READ_WRITE ,__scs_nvic_ipr3_bits);
__IO_REG32_BIT(SCS_NVIC_IPR4,         0xE000E410,__READ_WRITE ,__scs_nvic_ipr4_bits);
__IO_REG32_BIT(SCS_NVIC_IPR5,         0xE000E414,__READ_WRITE ,__scs_nvic_ipr5_bits);
__IO_REG32_BIT(SCS_NVIC_IPR6,         0xE000E418,__READ_WRITE ,__scs_nvic_ipr6_bits);
__IO_REG32_BIT(SCS_NVIC_IPR7,         0xE000E41C,__READ_WRITE ,__scs_nvic_ipr7_bits);
__IO_REG32_BIT(SCS_CPUID,             0xE000ED00,__READ       ,__scs_cpuid_bits);
__IO_REG32_BIT(SCS_ICSR,              0xE000ED04,__READ_WRITE ,__scs_icsr_bits);
__IO_REG32_BIT(SCS_AIRCR,             0xE000ED0C,__READ_WRITE ,__scs_aircr_bits);
__IO_REG32_BIT(SCS_SCR,               0xE000ED10,__READ_WRITE ,__scs_scr_bits);
__IO_REG32_BIT(SCS_SHPR2,             0xE000ED1C,__READ_WRITE ,__scs_shpr2_bits);
__IO_REG32_BIT(SCS_SHPR3,             0xE000ED20,__READ_WRITE ,__scs_shpr3_bits);

/***************************************************************************
 **
 ** GCR (Global Control Registers)
 **
 ***************************************************************************/
 
__IO_REG32(    GCR_PDID,              0x50000000,__READ       ); 
__IO_REG32_BIT(GCR_RSTSRC,            0x50000004,__READ_WRITE ,__gcr_rstsrc_bits);
__IO_REG32_BIT(GCR_IPRSTC1,           0x50000008,__READ_WRITE ,__gcr_iprstc1_bits);
__IO_REG32_BIT(GCR_IPRSTC2,           0x5000000C,__READ_WRITE ,__gcr_iprstc2_bits);
__IO_REG32_BIT(GCR_BODCR,             0x50000018,__READ_WRITE ,__gcr_bodcr_bits);
__IO_REG32_BIT(GCR_PORCR,             0x50000024,__READ_WRITE ,__gcr_porcr_bits);
__IO_REG32_BIT(GCR_GPA_MFP,           0x50000030,__READ_WRITE ,__gcr_gpa_mfp_bits);
__IO_REG32_BIT(GCR_GPB_MFP,           0x50000034,__READ_WRITE ,__gcr_gpb_mfp_bits);
__IO_REG32_BIT(GCR_GPC_MFP,           0x50000038,__READ_WRITE ,__gcr_gpc_mfp_bits);
__IO_REG32_BIT(GCR_GPD_MFP,           0x5000003C,__READ_WRITE ,__gcr_gpd_mfp_bits);
__IO_REG32_BIT(GCR_ALT_MFP,           0x50000050,__READ_WRITE ,__gcr_alt_mfp_bits);
__IO_REG32_BIT(GCR_REGWRPROT,         0x50000100,__READ_WRITE ,__gcr_regwrprot_bits);
#define GCR_REGPROTDIS        GCR_REGWRPROT
#define GCR_REGPROTDIS_bit    GCR_REGWRPROT_bit

/***************************************************************************
 **
 ** INT (Interrupt Multiplexer)
 **
 ***************************************************************************/
 
__IO_REG32_BIT(INT_IRQ0_SRC,           0x50000300,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ1_SRC,           0x50000304,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ2_SRC,           0x50000308,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ3_SRC,           0x5000030C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ4_SRC,           0x50000310,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ5_SRC,           0x50000314,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ6_SRC,           0x50000318,__READ       ,__int_irq67_src_bits);
__IO_REG32_BIT(INT_IRQ7_SRC,           0x5000031C,__READ       ,__int_irq67_src_bits);
__IO_REG32_BIT(INT_IRQ8_SRC,           0x50000320,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ9_SRC,           0x50000324,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ10_SRC,          0x50000328,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ11_SRC,          0x5000032C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ12_SRC,          0x50000330,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ13_SRC,          0x50000334,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ14_SRC,          0x50000338,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ15_SRC,          0x5000033C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ16_SRC,          0x50000340,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ17_SRC,          0x50000344,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ18_SRC,          0x50000348,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ19_SRC,          0x5000034C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ20_SRC,          0x50000350,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ21_SRC,          0x50000354,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ22_SRC,          0x50000358,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ23_SRC,          0x5000035C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ24_SRC,          0x50000360,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ25_SRC,          0x50000364,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ26_SRC,          0x50000368,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ27_SRC,          0x5000036C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ28_SRC,          0x50000370,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ29_SRC,          0x50000374,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ30_SRC,          0x50000378,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ31_SRC,          0x5000037C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_NMI_SEL,            0x50000380,__READ_WRITE ,__int_nmi_sel_bits);
__IO_REG32_BIT(INT_MCU_IRQ,            0x50000384,__READ_WRITE ,__int_mcu_irq_bits);

/***************************************************************************
 **
 ** CLK (Clock Controller)
 **
 ***************************************************************************/

__IO_REG32_BIT(CLK_PWRCON,             0x50000200,__READ_WRITE ,__clk_pwrcon_bits);
__IO_REG32_BIT(CLK_AHBCLK,             0x50000204,__READ_WRITE ,__clk_ahbclk_bits);
__IO_REG32_BIT(CLK_APBCLK,             0x50000208,__READ_WRITE ,__clk_apbclk_bits);
__IO_REG32_BIT(CLK_CLKSTATUS,          0x5000020C,__READ_WRITE ,__clk_clkstatus_bits);
__IO_REG32_BIT(CLK_CLKSEL0,            0x50000210,__READ_WRITE ,__clk_clksel0_bits);
__IO_REG32_BIT(CLK_CLKSEL1,            0x50000214,__READ_WRITE ,__clk_clksel1_bits);
__IO_REG32_BIT(CLK_CLKDIV,             0x50000218,__READ_WRITE ,__clk_clkdiv_bits);
__IO_REG32_BIT(CLK_PLLCON,             0x50000220,__READ_WRITE ,__clk_pllcon_bits);

/***************************************************************************
 **
 ** USB (USB Device Controller)
 **
 ***************************************************************************/
 
__IO_REG32_BIT(USB_INTEN,                 0x40060000,__READ_WRITE ,__usb_inten_bits);
__IO_REG32_BIT(USB_INTSTS,                0x40060004,__READ_WRITE ,__usb_intsts_bits);
__IO_REG32_BIT(USB_FADDR,                 0x40060008,__READ_WRITE ,__usb_faddr_bits);
__IO_REG32_BIT(USB_EPSTS,                 0x4006000C,__READ       ,__usb_epsts_bits);
__IO_REG32_BIT(USB_ATTR,                  0x40060010,__READ_WRITE ,__usb_attr_bits);
__IO_REG32_BIT(USB_FLDET,                 0x40060014,__READ       ,__usb_fldet_bits);
__IO_REG32_BIT(USB_BUFSEG,                0x40060018,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_BUFSEG0,               0x40060020,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD0,                0x40060024,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG0,                  0x40060028,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP0,                 0x4006002C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_BUFSEG1,               0x40060030,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD1,                0x40060034,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG1,                  0x40060038,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP1,                 0x4006003C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_BUFSEG2,               0x40060040,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD2,                0x40060044,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG2,                  0x40060048,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP2,                 0x4006004C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_BUFSEG3,               0x40060050,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD3,                0x40060054,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG3,                  0x40060058,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP3,                 0x4006005C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_BUFSEG4,               0x40060060,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD4,                0x40060064,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG4,                  0x40060068,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP4,                 0x4006006C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_BUFSEG5,               0x40060070,__READ_WRITE ,__usb_bufseg_bits);
__IO_REG32_BIT(USB_MXPLD5,                0x40060074,__READ_WRITE ,__usb_mxpld_bits);
__IO_REG32_BIT(USB_CFG5,                  0x40060078,__READ_WRITE ,__usb_cfg_bits);
__IO_REG32_BIT(USB_CFGP5,                 0x4006007C,__READ_WRITE ,__usb_cfgp_bits);
__IO_REG32_BIT(USB_DRVSE0,                0x40060090,__READ_WRITE ,__usb_drvse0_bits);

/***************************************************************************
 **
 ** *GP (General Purpose I/O)
 **
 ***************************************************************************/
__IO_REG32_BIT(GP_GPIOA_PMD,             0x50004000,__READ_WRITE ,__gp_gpiox_pmd_bits);
__IO_REG32_BIT(GP_GPIOA_OFFD,            0x50004004,__READ_WRITE ,__gp_gpiox_offd_bits);
__IO_REG32_BIT(GP_GPIOA_DOUT,            0x50004008,__READ_WRITE ,__gp_gpiox_dout_bits);
__IO_REG32_BIT(GP_GPIOA_DMASK,           0x5000400C,__READ_WRITE ,__gp_gpiox_dmask_bits);
__IO_REG32_BIT(GP_GPIOA_PIN,             0x50004010,__READ       ,__gp_gpiox_pin_bits);
__IO_REG32_BIT(GP_GPIOA_DBEN,            0x50004014,__READ_WRITE ,__gp_gpiox_dben_bits);
__IO_REG32_BIT(GP_GPIOA_IMD,             0x50004018,__READ_WRITE ,__gp_gpiox_imd_bits);
__IO_REG32_BIT(GP_GPIOA_IEN,             0x5000401C,__READ_WRITE ,__gp_gpiox_ien_bits);
__IO_REG32_BIT(GP_GPIOA_ISRC,            0x50004020,__READ_WRITE ,__gp_gpiox_isrc_bits);
__IO_REG32_BIT(GP_GPIOB_PMD,             0x50004040,__READ_WRITE ,__gp_gpiox_pmd_bits);
__IO_REG32_BIT(GP_GPIOB_OFFD,            0x50004044,__READ_WRITE ,__gp_gpiox_offd_bits);
__IO_REG32_BIT(GP_GPIOB_DOUT,            0x50004048,__READ_WRITE ,__gp_gpiox_dout_bits);
__IO_REG32_BIT(GP_GPIOB_DMASK,           0x5000404C,__READ_WRITE ,__gp_gpiox_dmask_bits);
__IO_REG32_BIT(GP_GPIOB_PIN,             0x50004050,__READ       ,__gp_gpiox_pin_bits);
__IO_REG32_BIT(GP_GPIOB_DBEN,            0x50004054,__READ_WRITE ,__gp_gpiox_dben_bits);
__IO_REG32_BIT(GP_GPIOB_IMD,             0x50004058,__READ_WRITE ,__gp_gpiox_imd_bits);
__IO_REG32_BIT(GP_GPIOB_IEN,             0x5000405C,__READ_WRITE ,__gp_gpiox_ien_bits);
__IO_REG32_BIT(GP_GPIOB_ISRC,            0x50004060,__READ_WRITE ,__gp_gpiox_isrc_bits);
__IO_REG32_BIT(GP_GPIOC_PMD,             0x50004080,__READ_WRITE ,__gp_gpiox_pmd_bits);
__IO_REG32_BIT(GP_GPIOC_OFFD,            0x50004084,__READ_WRITE ,__gp_gpiox_offd_bits);
__IO_REG32_BIT(GP_GPIOC_DOUT,            0x50004088,__READ_WRITE ,__gp_gpiox_dout_bits);
__IO_REG32_BIT(GP_GPIOC_DMASK,           0x5000408C,__READ_WRITE ,__gp_gpiox_dmask_bits);
__IO_REG32_BIT(GP_GPIOC_PIN,             0x50004090,__READ       ,__gp_gpiox_pin_bits);
__IO_REG32_BIT(GP_GPIOC_DBEN,            0x50004094,__READ_WRITE ,__gp_gpiox_dben_bits);
__IO_REG32_BIT(GP_GPIOC_IMD,             0x50004098,__READ_WRITE ,__gp_gpiox_imd_bits);
__IO_REG32_BIT(GP_GPIOC_IEN,             0x5000409C,__READ_WRITE ,__gp_gpiox_ien_bits);
__IO_REG32_BIT(GP_GPIOC_ISRC,            0x500040A0,__READ_WRITE ,__gp_gpiox_isrc_bits);
__IO_REG32_BIT(GP_GPIOD_PMD,             0x500040C0,__READ_WRITE ,__gp_gpiox_pmd_bits);
__IO_REG32_BIT(GP_GPIOD_OFFD,            0x500040C4,__READ_WRITE ,__gp_gpiox_offd_bits);
__IO_REG32_BIT(GP_GPIOD_DOUT,            0x500040C8,__READ_WRITE ,__gp_gpiox_dout_bits);
__IO_REG32_BIT(GP_GPIOD_DMASK,           0x500040CC,__READ_WRITE ,__gp_gpiox_dmask_bits);
__IO_REG32_BIT(GP_GPIOD_PIN,             0x500040D0,__READ       ,__gp_gpiox_pin_bits);
__IO_REG32_BIT(GP_GPIOD_DBEN,            0x500040D4,__READ_WRITE ,__gp_gpiox_dben_bits);
__IO_REG32_BIT(GP_GPIOD_IMD,             0x500040D8,__READ_WRITE ,__gp_gpiox_imd_bits);
__IO_REG32_BIT(GP_GPIOD_IEN,             0x500040DC,__READ_WRITE ,__gp_gpiox_ien_bits);
__IO_REG32_BIT(GP_GPIOD_ISRC,            0x500040E0,__READ_WRITE ,__gp_gpiox_isrc_bits);
__IO_REG32_BIT(GP_DBNCECON,              0x50004180,__READ_WRITE ,__gp_dbncecon_bits);
__IO_REG32_BIT(GP_GPIOA0_DOUT,           0x50004200,__READ_WRITE ,__gp_gpioa0_dout_bits);
__IO_REG32_BIT(GP_GPIOA1_DOUT,           0x50004204,__READ_WRITE ,__gp_gpioa1_dout_bits);
__IO_REG32_BIT(GP_GPIOA2_DOUT,           0x50004208,__READ_WRITE ,__gp_gpioa2_dout_bits);
__IO_REG32_BIT(GP_GPIOA3_DOUT,           0x5000420C,__READ_WRITE ,__gp_gpioa3_dout_bits);
__IO_REG32_BIT(GP_GPIOA4_DOUT,           0x50004210,__READ_WRITE ,__gp_gpioa4_dout_bits);
__IO_REG32_BIT(GP_GPIOA5_DOUT,           0x50004214,__READ_WRITE ,__gp_gpioa5_dout_bits);
__IO_REG32_BIT(GP_GPIOA6_DOUT,           0x50004218,__READ_WRITE ,__gp_gpioa6_dout_bits);
__IO_REG32_BIT(GP_GPIOA7_DOUT,           0x5000421C,__READ_WRITE ,__gp_gpioa7_dout_bits);
__IO_REG32_BIT(GP_GPIOA8_DOUT,           0x50004220,__READ_WRITE ,__gp_gpioa8_dout_bits);
__IO_REG32_BIT(GP_GPIOA9_DOUT,           0x50004224,__READ_WRITE ,__gp_gpioa9_dout_bits);
__IO_REG32_BIT(GP_GPIOA10_DOUT,          0x50004228,__READ_WRITE ,__gp_gpioa10_dout_bits);
__IO_REG32_BIT(GP_GPIOA11_DOUT,          0x5000422C,__READ_WRITE ,__gp_gpioa11_dout_bits);
__IO_REG32_BIT(GP_GPIOA12_DOUT,          0x50004230,__READ_WRITE ,__gp_gpioa12_dout_bits);
__IO_REG32_BIT(GP_GPIOA13_DOUT,          0x50004234,__READ_WRITE ,__gp_gpioa13_dout_bits);
__IO_REG32_BIT(GP_GPIOA14_DOUT,          0x50004238,__READ_WRITE ,__gp_gpioa14_dout_bits);
__IO_REG32_BIT(GP_GPIOA15_DOUT,          0x5000423C,__READ_WRITE ,__gp_gpioa15_dout_bits);
__IO_REG32_BIT(GP_GPIOB0_DOUT,           0x50004240,__READ_WRITE ,__gp_gpiob0_dout_bits);
__IO_REG32_BIT(GP_GPIOB1_DOUT,           0x50004244,__READ_WRITE ,__gp_gpiob1_dout_bits);
__IO_REG32_BIT(GP_GPIOB2_DOUT,           0x50004248,__READ_WRITE ,__gp_gpiob2_dout_bits);
__IO_REG32_BIT(GP_GPIOB3_DOUT,           0x5000424C,__READ_WRITE ,__gp_gpiob3_dout_bits);
__IO_REG32_BIT(GP_GPIOB4_DOUT,           0x50004250,__READ_WRITE ,__gp_gpiob4_dout_bits);
__IO_REG32_BIT(GP_GPIOB5_DOUT,           0x50004254,__READ_WRITE ,__gp_gpiob5_dout_bits);
__IO_REG32_BIT(GP_GPIOB6_DOUT,           0x50004258,__READ_WRITE ,__gp_gpiob6_dout_bits);
__IO_REG32_BIT(GP_GPIOB7_DOUT,           0x5000425C,__READ_WRITE ,__gp_gpiob7_dout_bits);
__IO_REG32_BIT(GP_GPIOB8_DOUT,           0x50004260,__READ_WRITE ,__gp_gpiob8_dout_bits);
__IO_REG32_BIT(GP_GPIOB9_DOUT,           0x50004264,__READ_WRITE ,__gp_gpiob9_dout_bits);
__IO_REG32_BIT(GP_GPIOB10_DOUT,          0x50004268,__READ_WRITE ,__gp_gpiob10_dout_bits);
__IO_REG32_BIT(GP_GPIOB11_DOUT,          0x5000426C,__READ_WRITE ,__gp_gpiob11_dout_bits);
__IO_REG32_BIT(GP_GPIOB12_DOUT,          0x50004270,__READ_WRITE ,__gp_gpiob12_dout_bits);
__IO_REG32_BIT(GP_GPIOB13_DOUT,          0x50004274,__READ_WRITE ,__gp_gpiob13_dout_bits);
__IO_REG32_BIT(GP_GPIOB14_DOUT,          0x50004278,__READ_WRITE ,__gp_gpiob14_dout_bits);
__IO_REG32_BIT(GP_GPIOB15_DOUT,          0x5000427C,__READ_WRITE ,__gp_gpiob15_dout_bits);
__IO_REG32_BIT(GP_GPIOC0_DOUT,           0x50004280,__READ_WRITE ,__gp_gpioc0_dout_bits);
__IO_REG32_BIT(GP_GPIOC1_DOUT,           0x50004284,__READ_WRITE ,__gp_gpioc1_dout_bits);
__IO_REG32_BIT(GP_GPIOC2_DOUT,           0x50004288,__READ_WRITE ,__gp_gpioc2_dout_bits);
__IO_REG32_BIT(GP_GPIOC3_DOUT,           0x5000428C,__READ_WRITE ,__gp_gpioc3_dout_bits);
__IO_REG32_BIT(GP_GPIOC4_DOUT,           0x50004290,__READ_WRITE ,__gp_gpioc4_dout_bits);
__IO_REG32_BIT(GP_GPIOC5_DOUT,           0x50004294,__READ_WRITE ,__gp_gpioc5_dout_bits);
__IO_REG32_BIT(GP_GPIOC6_DOUT,           0x50004298,__READ_WRITE ,__gp_gpioc6_dout_bits);
__IO_REG32_BIT(GP_GPIOC7_DOUT,           0x5000429C,__READ_WRITE ,__gp_gpioc7_dout_bits);
__IO_REG32_BIT(GP_GPIOC8_DOUT,           0x500042A0,__READ_WRITE ,__gp_gpioc8_dout_bits);
__IO_REG32_BIT(GP_GPIOC9_DOUT,           0x500042A4,__READ_WRITE ,__gp_gpioc9_dout_bits);
__IO_REG32_BIT(GP_GPIOC10_DOUT,          0x500042A8,__READ_WRITE ,__gp_gpioc10_dout_bits);
__IO_REG32_BIT(GP_GPIOC11_DOUT,          0x500042AC,__READ_WRITE ,__gp_gpioc11_dout_bits);
__IO_REG32_BIT(GP_GPIOC12_DOUT,          0x500042B0,__READ_WRITE ,__gp_gpioc12_dout_bits);
__IO_REG32_BIT(GP_GPIOC13_DOUT,          0x500042B4,__READ_WRITE ,__gp_gpioc13_dout_bits);
__IO_REG32_BIT(GP_GPIOC14_DOUT,          0x500042B8,__READ_WRITE ,__gp_gpioc14_dout_bits);
__IO_REG32_BIT(GP_GPIOC15_DOUT,          0x500042BC,__READ_WRITE ,__gp_gpioc15_dout_bits);
__IO_REG32_BIT(GP_GPIOD0_DOUT,           0x500042C0,__READ_WRITE ,__gp_gpiod0_dout_bits);
__IO_REG32_BIT(GP_GPIOD1_DOUT,           0x500042C4,__READ_WRITE ,__gp_gpiod1_dout_bits);
__IO_REG32_BIT(GP_GPIOD2_DOUT,           0x500042C8,__READ_WRITE ,__gp_gpiod2_dout_bits);
__IO_REG32_BIT(GP_GPIOD3_DOUT,           0x500042CC,__READ_WRITE ,__gp_gpiod3_dout_bits);
__IO_REG32_BIT(GP_GPIOD4_DOUT,           0x500042D0,__READ_WRITE ,__gp_gpiod4_dout_bits);
__IO_REG32_BIT(GP_GPIOD5_DOUT,           0x500042D4,__READ_WRITE ,__gp_gpiod5_dout_bits);
__IO_REG32_BIT(GP_GPIOD6_DOUT,           0x500042D8,__READ_WRITE ,__gp_gpiod6_dout_bits);
__IO_REG32_BIT(GP_GPIOD7_DOUT,           0x500042DC,__READ_WRITE ,__gp_gpiod7_dout_bits);
__IO_REG32_BIT(GP_GPIOD8_DOUT,           0x500042E0,__READ_WRITE ,__gp_gpiod8_dout_bits);
__IO_REG32_BIT(GP_GPIOD9_DOUT,           0x500042E4,__READ_WRITE ,__gp_gpiod9_dout_bits);
__IO_REG32_BIT(GP_GPIOD10_DOUT,          0x500042E8,__READ_WRITE ,__gp_gpiod10_dout_bits);
__IO_REG32_BIT(GP_GPIOD11_DOUT,          0x500042EC,__READ_WRITE ,__gp_gpiod11_dout_bits);
__IO_REG32_BIT(GP_GPIOD12_DOUT,          0x500042F0,__READ_WRITE ,__gp_gpiod12_dout_bits);
__IO_REG32_BIT(GP_GPIOD13_DOUT,          0x500042F4,__READ_WRITE ,__gp_gpiod13_dout_bits);
__IO_REG32_BIT(GP_GPIOD14_DOUT,          0x500042F8,__READ_WRITE ,__gp_gpiod14_dout_bits);
__IO_REG32_BIT(GP_GPIOD15_DOUT,          0x500042FC,__READ_WRITE ,__gp_gpiod15_dout_bits);

/***************************************************************************
 **
 ** I2C1 (I2C1 Serial Interface Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_I2CON,               0x40120000,__READ_WRITE ,__i2cx_i2con_bits);
__IO_REG32_BIT(I2C1_I2CADDR0,            0x40120004,__READ_WRITE ,__i2cx_i2caddry_bits);
__IO_REG32_BIT(I2C1_I2CDAT,              0x40120008,__READ_WRITE ,__i2cx_i2cdat_bits);
__IO_REG32_BIT(I2C1_I2CSTATUS,           0x4012000C,__READ       ,__i2cx_i2cstatus_bits);
__IO_REG32_BIT(I2C1_I2CLK,               0x40120010,__READ_WRITE ,__i2cx_i2clk_bits);
__IO_REG32_BIT(I2C1_I2CTOC,              0x40120014,__READ_WRITE ,__i2cx_i2ctoc_bits);
__IO_REG32_BIT(I2C1_I2CADDR1,            0x40120018,__READ_WRITE ,__i2cx_i2caddry_bits);
__IO_REG32_BIT(I2C1_I2CADDR2,            0x4012001C,__READ_WRITE ,__i2cx_i2caddry_bits);
__IO_REG32_BIT(I2C1_I2CADDR3,            0x40120020,__READ_WRITE ,__i2cx_i2caddry_bits);
__IO_REG32_BIT(I2C1_I2CADM0,             0x40120024,__READ_WRITE ,__i2cx_i2cadmy_bits);
__IO_REG32_BIT(I2C1_I2CADM1,             0x40120028,__READ_WRITE ,__i2cx_i2cadmy_bits);
__IO_REG32_BIT(I2C1_I2CADM2,             0x4012002C,__READ_WRITE ,__i2cx_i2cadmy_bits);
__IO_REG32_BIT(I2C1_I2CADM3,             0x40120030,__READ_WRITE ,__i2cx_i2cadmy_bits);

/***************************************************************************
 **
 ** PWMA (PWM Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMA_PPR,               0x40040000,__READ_WRITE ,__pwmx_ppr_bits);
__IO_REG32_BIT(PWMA_CSR,               0x40040004,__READ_WRITE ,__pwmx_csr_bits);
__IO_REG32_BIT(PWMA_PCR,               0x40040008,__READ_WRITE ,__pwmx_pcr_bits);
__IO_REG32_BIT(PWMA_CNR0,              0x4004000C,__READ_WRITE ,__pwmx_cnry_bits);
__IO_REG32_BIT(PWMA_CMR0,              0x40040010,__READ_WRITE ,__pwmx_cmry_bits);
__IO_REG32_BIT(PWMA_PDR0,              0x40040014,__READ       ,__pwmx_pdry_bits);
__IO_REG32_BIT(PWMA_CNR1,              0x40040018,__READ_WRITE ,__pwmx_cnry_bits);
__IO_REG32_BIT(PWMA_CMR1,              0x4004001C,__READ_WRITE ,__pwmx_cmry_bits);
__IO_REG32_BIT(PWMA_PDR1,              0x40040020,__READ       ,__pwmx_pdry_bits);
__IO_REG32_BIT(PWMA_CNR2,              0x40040024,__READ_WRITE ,__pwmx_cnry_bits);
__IO_REG32_BIT(PWMA_CMR2,              0x40040028,__READ_WRITE ,__pwmx_cmry_bits);
__IO_REG32_BIT(PWMA_PDR2,              0x4004002C,__READ       ,__pwmx_pdry_bits);
__IO_REG32_BIT(PWMA_CNR3,              0x40040030,__READ_WRITE ,__pwmx_cnry_bits);
__IO_REG32_BIT(PWMA_CMR3,              0x40040034,__READ_WRITE ,__pwmx_cmry_bits);
__IO_REG32_BIT(PWMA_PDR3,              0x40040038,__READ       ,__pwmx_pdry_bits);
__IO_REG32_BIT(PWMA_PIER,              0x40040040,__READ_WRITE ,__pwmx_pier_bits);
__IO_REG32_BIT(PWMA_PIIR,              0x40040044,__READ_WRITE ,__pwmx_piir_bits);
__IO_REG32_BIT(PWMA_CCR0,              0x40040050,__READ_WRITE ,__pwmx_ccr0_bits);
__IO_REG32_BIT(PWMA_CCR2,              0x40040054,__READ_WRITE ,__pwmx_ccr2_bits);
__IO_REG32_BIT(PWMA_CRLR0,             0x40040058,__READ_WRITE ,__pwmx_crlry_bits);
__IO_REG32_BIT(PWMA_CFLR0,             0x4004005C,__READ_WRITE ,__pwmx_cflry_bits);
__IO_REG32_BIT(PWMA_CRLR1,             0x40040060,__READ_WRITE ,__pwmx_crlry_bits);
__IO_REG32_BIT(PWMA_CFLR1,             0x40040064,__READ_WRITE ,__pwmx_cflry_bits);
__IO_REG32_BIT(PWMA_CRLR2,             0x40040068,__READ_WRITE ,__pwmx_crlry_bits);
__IO_REG32_BIT(PWMA_CFLR2,             0x4004006C,__READ_WRITE ,__pwmx_cflry_bits);
__IO_REG32_BIT(PWMA_CRLR3,             0x40040070,__READ_WRITE ,__pwmx_crlry_bits);
__IO_REG32_BIT(PWMA_CFLR3,             0x40040074,__READ_WRITE ,__pwmx_cflry_bits);
__IO_REG32_BIT(PWMA_CAPENR,            0x40040078,__READ_WRITE ,__pwmx_capenr_bits);
__IO_REG32_BIT(PWMA_POE,               0x4004007C,__READ_WRITE ,__pwmx_poe_bits);

/***************************************************************************
 **
 ** RTC (Real Time Clock)
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_INIR,              0x40008000,__READ_WRITE ,__rtc_inir_bits);
__IO_REG32_BIT(RTC_AER,               0x40008004,__READ_WRITE ,__rtc_aer_bits);
__IO_REG32_BIT(RTC_FCR,               0x40008008,__READ_WRITE ,__rtc_fcr_bits);
__IO_REG32_BIT(RTC_TLR,               0x4000800C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CLR,               0x40008010,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_TSSR,              0x40008014,__READ_WRITE ,__rtc_tssr_bits);
__IO_REG32_BIT(RTC_DWR,               0x40008018,__READ_WRITE ,__rtc_dwr_bits);
__IO_REG32_BIT(RTC_TAR,               0x4000801C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CAR,               0x40008020,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_LIR,               0x40008024,__READ       ,__rtc_lir_bits);
__IO_REG32_BIT(RTC_RIER,              0x40008028,__READ_WRITE ,__rtc_rier_bits);
__IO_REG32_BIT(RTC_RIIR,              0x4000802C,__READ_WRITE ,__rtc_riir_bits);
__IO_REG32_BIT(RTC_TTR,               0x40008030,__READ_WRITE ,__rtc_ttr_bits);

/***************************************************************************
 **
 ** SPI0 (SPI0 Controller)
 **
 ***************************************************************************/

__IO_REG32_BIT(SPI0_SPI_CNTRL,            0x40030000,__READ_WRITE ,__spix_spi_cntrl_bits);
__IO_REG32_BIT(SPI0_SPI_DIVIDER,          0x40030004,__READ_WRITE ,__spix_spi_divider_bits);
__IO_REG32_BIT(SPI0_SPI_SSR,              0x40030008,__READ_WRITE ,__spix_spi_ssr_bits);
__IO_REG32(    SPI0_SPI_RX0,              0x40030010,__READ       );
__IO_REG32(    SPI0_SPI_RX1,              0x40030014,__READ       );
__IO_REG32(    SPI0_SPI_TX0,              0x40030020,__WRITE      );
__IO_REG32(    SPI0_SPI_TX1,              0x40030024,__WRITE      );
__IO_REG32(    SPI0_SPI_VARCLK,           0x40030034,__READ_WRITE );

/***************************************************************************
 **
 ** SPI1 (SPI1 Controller)
 **
 ***************************************************************************/

__IO_REG32_BIT(SPI1_SPI_CNTRL,            0x40034000,__READ_WRITE ,__spix_spi_cntrl_bits);
__IO_REG32_BIT(SPI1_SPI_DIVIDER,          0x40034004,__READ_WRITE ,__spix_spi_divider_bits);
__IO_REG32_BIT(SPI1_SPI_SSR,              0x40034008,__READ_WRITE ,__spix_spi_ssr_bits);
__IO_REG32(    SPI1_SPI_RX0,              0x40034010,__READ       );
__IO_REG32(    SPI1_SPI_RX1,              0x40034014,__READ       );
__IO_REG32(    SPI1_SPI_TX0,              0x40034020,__WRITE      );
__IO_REG32(    SPI1_SPI_TX1,              0x40034024,__WRITE      );
__IO_REG32(    SPI1_SPI_VARCLK,           0x40034034,__READ_WRITE );

/***************************************************************************
 **
 ** TMR01 (Timers 0 & 1)
 **
 ***************************************************************************/

__IO_REG32_BIT(TMR01_TCSR0,                 0x40010000,__READ_WRITE ,__tmrx_tcsry_bits);
__IO_REG32_BIT(TMR01_TCMPR0,                0x40010004,__READ_WRITE ,__tmrx_tcmpry_bits);
__IO_REG32_BIT(TMR01_TISR0,                 0x40010008,__READ_WRITE ,__tmrx_tisry_bits);
__IO_REG32_BIT(TMR01_TDR0,                  0x4001000C,__READ       ,__tmrx_tdry_bits);
__IO_REG32_BIT(TMR01_TCSR1,                 0x40010020,__READ_WRITE ,__tmrx_tcsry_bits);
__IO_REG32_BIT(TMR01_TCMPR1,                0x40010024,__READ_WRITE ,__tmrx_tcmpry_bits);
__IO_REG32_BIT(TMR01_TISR1,                 0x40010028,__READ_WRITE ,__tmrx_tisry_bits);
__IO_REG32_BIT(TMR01_TDR1,                  0x4001002C,__READ       ,__tmrx_tdry_bits);

/***************************************************************************
 **
 ** TMR23 (Timers 2 & 3)
 **
 ***************************************************************************/

__IO_REG32_BIT(TMR23_TCSR2,                 0x40110000,__READ_WRITE ,__tmrx_tcsry_bits);
__IO_REG32_BIT(TMR23_TCMPR2,                0x40110004,__READ_WRITE ,__tmrx_tcmpry_bits);
__IO_REG32_BIT(TMR23_TISR2,                 0x40110008,__READ_WRITE ,__tmrx_tisry_bits);
__IO_REG32_BIT(TMR23_TDR2,                  0x4011000C,__READ       ,__tmrx_tdry_bits);
__IO_REG32_BIT(TMR23_TCSR3,                 0x40110020,__READ_WRITE ,__tmrx_tcsry_bits);
__IO_REG32_BIT(TMR23_TCMPR3,                0x40110024,__READ_WRITE ,__tmrx_tcmpry_bits);
__IO_REG32_BIT(TMR23_TISR3,                 0x40110028,__READ_WRITE ,__tmrx_tisry_bits);
__IO_REG32_BIT(TMR23_TDR3,                  0x4011002C,__READ       ,__tmrx_tdry_bits);

/***************************************************************************
 **
 ** WDT (Watchdog Timer)
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_WTCR,                  0x40004000,__READ_WRITE ,__wdt_wtcr_bits);

/***************************************************************************
 **
 ** UART0 (UART0 Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0_UA_RBR,               0x40050000,__READ_WRITE ,__uartx_ua_rbr_thr_bits);
#define UART0_UA_THR     UART0_UA_RBR
#define UART0_UA_THR_bit UART0_UA_RBR_bit
__IO_REG32_BIT(UART0_UA_IER,               0x40050004,__READ_WRITE ,__uartx_ua_ier_bits);
__IO_REG32_BIT(UART0_UA_FCR,               0x40050008,__READ_WRITE ,__uartx_ua_fcr_bits);
__IO_REG32_BIT(UART0_UA_LCR,               0x4005000C,__READ_WRITE ,__uartx_ua_lcr_bits);
__IO_REG32_BIT(UART0_UA_MCR,               0x40050010,__READ_WRITE ,__uartx_ua_mcr_bits);
__IO_REG32_BIT(UART0_UA_MSR,               0x40050014,__READ_WRITE ,__uartx_ua_msr_bits);
__IO_REG32_BIT(UART0_UA_FSR,               0x40050018,__READ_WRITE ,__uartx_ua_fsr_bits);
__IO_REG32_BIT(UART0_UA_ISR,               0x4005001C,__READ_WRITE ,__uartx_ua_isr_bits);
__IO_REG32_BIT(UART0_UA_TOR,               0x40050020,__READ_WRITE ,__uartx_ua_tor_bits);
__IO_REG32_BIT(UART0_UA_BAUD,              0x40050024,__READ_WRITE ,__uartx_ua_baud_bits);
__IO_REG32_BIT(UART0_UA_IRCR,              0x40050028,__READ_WRITE ,__uartx_ua_ircr_bits);
__IO_REG32_BIT(UART0_UA_ALT_CSR,           0x4005002C,__READ_WRITE ,__uartx_ua_alt_csr_bits);
__IO_REG32_BIT(UART0_UA_FUN_SEL,           0x40050030,__READ_WRITE ,__uartx_ua_fun_sel_bits);

/***************************************************************************
 **
 ** UART1 (UART1 Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1_UA_RBR,               0x40150000,__READ_WRITE ,__uartx_ua_rbr_thr_bits);
#define UART1_UA_THR     UART1_UA_RBR
#define UART1_UA_THR_bit UART1_UA_RBR_bit
__IO_REG32_BIT(UART1_UA_IER,               0x40150004,__READ_WRITE ,__uartx_ua_ier_bits);
__IO_REG32_BIT(UART1_UA_FCR,               0x40150008,__READ_WRITE ,__uartx_ua_fcr_bits);
__IO_REG32_BIT(UART1_UA_LCR,               0x4015000C,__READ_WRITE ,__uartx_ua_lcr_bits);
__IO_REG32_BIT(UART1_UA_MCR,               0x40150010,__READ_WRITE ,__uartx_ua_mcr_bits);
__IO_REG32_BIT(UART1_UA_MSR,               0x40150014,__READ_WRITE ,__uartx_ua_msr_bits);
__IO_REG32_BIT(UART1_UA_FSR,               0x40150018,__READ_WRITE ,__uartx_ua_fsr_bits);
__IO_REG32_BIT(UART1_UA_ISR,               0x4015001C,__READ_WRITE ,__uartx_ua_isr_bits);
__IO_REG32_BIT(UART1_UA_TOR,               0x40150020,__READ_WRITE ,__uartx_ua_tor_bits);
__IO_REG32_BIT(UART1_UA_BAUD,              0x40150024,__READ_WRITE ,__uartx_ua_baud_bits);
__IO_REG32_BIT(UART1_UA_IRCR,              0x40150028,__READ_WRITE ,__uartx_ua_ircr_bits);
__IO_REG32_BIT(UART1_UA_ALT_CSR,           0x4015002C,__READ_WRITE ,__uartx_ua_alt_csr_bits);
__IO_REG32_BIT(UART1_UA_FUN_SEL,           0x40150030,__READ_WRITE ,__uartx_ua_fun_sel_bits);

/***************************************************************************
 **
 ** PS2 (PS/2 Interface Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PS2_PS2CON,                0x40100000,__READ_WRITE ,__ps2_ps2con_bits);
__IO_REG32(    PS2_PS2TXDATA0,            0x40100004,__READ_WRITE);
__IO_REG32(    PS2_PS2TXDATA1,            0x40100008,__READ_WRITE);
__IO_REG32(    PS2_PS2TXDATA2,            0x4010000C,__READ_WRITE);
__IO_REG32(    PS2_PS2TXDATA3,            0x40100010,__READ_WRITE);
__IO_REG32_BIT(PS2_PS2RXDATA,             0x40100014,__READ       ,__ps2_ps2rxdata_bits);
__IO_REG32_BIT(PS2_PS2STATUS,             0x40100018,__READ_WRITE ,__ps2_ps2status_bits);
__IO_REG32_BIT(PS2_PS2INTID,              0x4010001C,__READ_WRITE ,__ps2_ps2intid_bits);

/***************************************************************************
 **
 ** FMC (Flash Memory controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(FMC_ISPCON,              0x5000C000,__READ_WRITE ,__fmc_ispcon_bits);
__IO_REG32(    FMC_ISPADR,              0x5000C004,__READ_WRITE );
__IO_REG32(    FMC_ISPDAT,              0x5000C008,__READ_WRITE );
__IO_REG32_BIT(FMC_ISPCMD,              0x5000C00C,__READ_WRITE ,__fmc_ispcmd_bits);
__IO_REG32_BIT(FMC_ISPTRG,              0x5000C010,__READ_WRITE ,__fmc_isptrg_bits);
__IO_REG32_BIT(FMC_FATCON,              0x5000C018,__READ_WRITE ,__fmc_fatcon_bits);

/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  NVIC Interrupt channels
 **
 ***************************************************************************/
#define MAIN_STACK             0  /* Main Stack                                             */
#define RESETI                 1  /* Reset                                                  */
#define NMII                   2  /* Non-maskable Interrupt                                 */
#define HFI                    3  /* Hard Fault                                             */
#define SVCI                  11  /* SVCall                                                 */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_BOD_OUT          16  /* Brownout low voltage detected interrupt                */
#define NVIC_WDT_INT          17  /* Watch Dog Timer interrupt                              */
#define NVIC_EINT0            18  /* External signal interrupt from PB.14 pin               */
#define NVIC_EINT1            19  /* External signal interrupt from PB.15 pin               */
#define NVIC_GPAB_INT         20  /* External signal interrupt from PA[15:0] / PB[13:0]     */                                    */
#define NVIC_GPCD_INT         21  /* External signal interrupt from PC[15:0] / PD[15:0]     */
#define NVIC_PWMA_INT         22  /* PWM0, PWM1, PWM2 and PWM3 interrupt                    */
#define NVIC_TMR0_INT         24  /* Timer 0 interrupt                                      */
#define NVIC_TMR1_INT         25  /* Timer 1 interrupt                                      */
#define NVIC_TMR2_INT         26  /* Timer 2 interrupt                                      */
#define NVIC_TMR3_INT         27  /* Timer 3 interrupt                                      */
#define NVIC_UART0_INT        28  /* UART1 interrupt                                        */
#define NVIC_UART1_INT        29  /* UART1 interrupt                                        */
#define NVIC_SPI0_INT         30  /* SPI0 interrupt                                         */
#define NVIC_SPI1_INT         31  /* SPI1 interrupt                                         */
#define NVIC_I2C1_INT         35  /* I2C1 interrupt                                         */
#define NVIC_USB_INT          39  /* USB FS Device interrupt                                */
#define NVIC_PS2_INT          40  /* PS2 interrupt                                          */
#define NVIC_PWRWU_INT        44  /* Clock controller interrupt for chip wake up from power-down state*/
#define NVIC_RTC_INT          47  /* Real time clock interrupt                              */

#endif    /* __NUC122_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = SVC            0x2C
Interrupt3   = PendSV         0x38
Interrupt4   = SysTick        0x3C
Interrupt5   = BOD_OUT        0x40
Interrupt6   = WDT_INT        0x44
Interrupt7   = EINT0          0x48
Interrupt8   = EINT1          0x4C
Interrupt9   = GPAB_INT       0x50
Interrupt10  = GPCD_INT       0x54
Interrupt11  = PWMA_INT       0x58
Interrupt12  = TMR0_INT       0x60
Interrupt13  = TMR1_INT       0x64
Interrupt14  = TMR2_INT       0x68
Interrupt15  = TMR3_INT       0x6C
Interrupt16  = UART0_INT      0x70
Interrupt17  = UART1_INT      0x74
Interrupt18  = SPI0_INT       0x78
Interrupt19  = SPI1_INT       0x7C
Interrupt20  = I2C1_INT       0x8C
Interrupt21  = USB_INT        0x9C
Interrupt22  = PS2_INT        0xA0
Interrupt23  = PWRWU_INT      0xB0
Interrupt24  = RTC_INT        0xBC
###DDF-INTERRUPT-END###*/

