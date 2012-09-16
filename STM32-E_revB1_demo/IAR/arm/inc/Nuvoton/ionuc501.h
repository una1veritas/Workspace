/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton NUC510
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 31908 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IONUC510_H
#define __IONUC510_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   NUC510 SPECIAL FUNCTION REGISTERS
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

/* Product Identifier Register (PDID) */
typedef struct{
__REG32 CID             :24;
__REG32 CVI             : 4;
__REG32                 : 4;
} __pdid_bits;

/* System Power On Configuration Register (SPOCR) */
typedef struct{
__REG32 SYS_CFG         : 7;
__REG32                 :13;
__REG32 IBR_remap       : 1;
__REG32                 :11;
} __spocr_bits;

/* CPU Control Register (CPUCR) */
typedef struct{
__REG32 CPURST          : 1;
__REG32                 :31;
} __cpucr_bits;

/* MISC Control Register (MISCR) */
typedef struct{
__REG32 LVD_EN          : 1;
__REG32 LVR_WARM        : 1;
__REG32                 :30;
} __miscr_bits;

/* IP Reset Control Register (IPRST) */
typedef struct{
__REG32 UR0_RST         : 1;
__REG32 UR1_RST         : 1;
__REG32                 : 3;
__REG32 TMR_RST         : 1;
__REG32                 : 2;
__REG32 PWM_RST         : 1;
__REG32 I2C_RST         : 1;
__REG32 SPIM_RST        : 1;
__REG32 UDC_RST         : 1;
__REG32                 : 4;
__REG32 APU_RST         : 1;
__REG32                 : 8;
__REG32 SRAM_RST        : 1;
__REG32                 : 1;
__REG32 GPIO_RST        : 1;
__REG32 ADC_RST         : 1;
__REG32                 : 1;
__REG32 SPIMS_RST       : 1;
__REG32                 : 1;
} __iprst_bits;

/* AHB Control Register (AHB_CTRL) */
typedef struct{
__REG32 PRTMOD0         : 1;
__REG32                 : 3;
__REG32 IPEN            : 1;
__REG32 IPACT           : 1;
__REG32                 :26;
} __ahb_ctrl_bits;

/* PAD Control Register (PAD_REG0) */
typedef struct{
__REG32 PWM_TMR0_O      : 5;
__REG32 PWM_TMR0_I      : 3;
__REG32 PWM_TMR1_O      : 5;
__REG32 PWM_TMR1_I      : 3;
__REG32 PWM_TMR2_O      : 5;
__REG32 PWM_TMR2_I      : 3;
__REG32 PWM_TMR3_O      : 5;
__REG32 PWM_TMR3_I      : 3;
} __pad_reg0_bits;

/* PAD Control Register (PAD_REG1) */
typedef struct{
__REG32 ICE_EN          : 1;
__REG32 I2CP_EN         : 2;
__REG32 SPIM1_EN        : 2;
__REG32 SPIMS_EN        : 1;
__REG32 SPIM0_EN        : 1;
__REG32                 : 1;
__REG32 UART0_EN        : 1;
__REG32 UART1_EN        : 1;
__REG32                 : 1;
__REG32 UART0_MEN       : 1;
__REG32 UART1_MEN       : 1;
__REG32                 : 3;
__REG32 ADCP_EN0        : 1;
__REG32 ADCP_EN1        : 1;
__REG32 ADCP_EN2        : 1;
__REG32 ADCP_EN3        : 1;
__REG32 ADCP_EN4        : 1;
__REG32 ADCP_EN5        : 1;
__REG32 ADCP_EN6        : 1;
__REG32 ADCP_EN7        : 1;
__REG32                 : 8;
} __pad_reg1_bits;

/* PAD Control Register (PAD_REG2) */
typedef struct{
__REG32 USBDET_SEL      : 4;
__REG32                 :28;
} __pad_reg2_bits;

/* GPIOA driving strength (GPA_DS) */
typedef struct{
__REG32 GPA_DS0         : 1;
__REG32 GPA_DS1         : 1;
__REG32 GPA_DS2         : 1;
__REG32 GPA_DS3         : 1;
__REG32 GPA_DS4         : 1;
__REG32 GPA_DS5         : 1;
__REG32 GPA_DS6         : 1;
__REG32 GPA_DS7         : 1;
__REG32 GPA_DS8         : 1;
__REG32 GPA_DS9         : 1;
__REG32 GPA_DS10        : 1;
__REG32 GPA_DS11        : 1;
__REG32 GPA_DS12        : 1;
__REG32 GPA_DS13        : 1;
__REG32 GPA_DS14        : 1;
__REG32 GPA_DS15        : 1;
__REG32                 :16;
} __gpa_ds_bits;

/* GPIOB driving strength (GPB_DS) */
typedef struct{
__REG32 GPB_DS0         : 1;
__REG32 GPB_DS1         : 1;
__REG32 GPB_DS2         : 1;
__REG32 GPB_DS3         : 1;
__REG32 GPB_DS4         : 1;
__REG32 GPB_DS5         : 1;
__REG32 GPB_DS6         : 1;
__REG32 GPB_DS7         : 1;
__REG32 GPB_DS8         : 1;
__REG32 GPB_DS9         : 1;
__REG32                 :22;
} __gpb_ds_bits;

/* GPIOC driving strength (GPC_DS) */
typedef struct{
__REG32 GPC_DS0         : 1;
__REG32 GPC_DS1         : 1;
__REG32 GPC_DS2         : 1;
__REG32 GPC_DS3         : 1;
__REG32 GPC_DS4         : 1;
__REG32 GPC_DS5         : 1;
__REG32 GPC_DS6         : 1;
__REG32 GPC_DS7         : 1;
__REG32 GPC_DS8         : 1;
__REG32 GPC_DS9         : 1;
__REG32 GPC_DS10        : 1;
__REG32                 :21;
} __gpc_ds_bits;

/* Power Down Control Register (PWRCON) */
typedef struct{
__REG32 XTAL_EN         : 1;
__REG32 XIN_CTL         : 1;
__REG32 INTSTS          : 1;
__REG32 INT_EN          : 1;
__REG32                 : 4;
__REG32 PRESCALER       :16;
__REG32                 : 8;
} __pwrcon_bits;

/* AHB Devices Clock Enable Control Register (AHBCLK) */
typedef struct{
__REG32 CPU_CK_EN       : 1;
__REG32 APB_CK_EN       : 1;
__REG32                 : 4;
__REG32 USBD_CK_EN      : 1;
__REG32 SPIM_CK_EN      : 1;
__REG32 APU_CK_EN       : 1;
__REG32                 :23;
} __ahbclk_bits;

/* APB Devices Clock Enable Control Register (APBCLK) */
typedef struct{
__REG32 TIMER_CK_EN     : 1;
__REG32 WD_CK_EN        : 1;
__REG32 RTC_CK_EN       : 1;
__REG32 UART0_CK_EN     : 1;
__REG32 UART1_CK_EN     : 1;
__REG32 PWM_CK_EN       : 1;
__REG32 I2C_CK_EN       : 1;
__REG32                 : 1;
__REG32 SPIMS_CK_EN     : 1;
__REG32 ADC_CK_EN       : 1;
__REG32                 :22;
} __apbclk_bits;

/* Clock Source Select Control Register (CLKSEL) */
typedef struct{
__REG32 HCLK_S          : 2;
__REG32 USB_S           : 2;
__REG32 APU_S           : 2;
__REG32 UART_S          : 2;
__REG32                 : 6;
__REG32 ADC_S           : 2;
__REG32                 :16;
} __clksel_bits;

/* Clock Divider Register0 (CLKDIV0) */
typedef struct{
__REG32 HCLK_N          : 4;
__REG32                 : 2;
__REG32 APB_N           : 2;
__REG32 APU_N           : 8;
__REG32 UART_N          : 4;
__REG32 USB_N           : 4;
__REG32                 : 8;
} __clkdiv0_bits;

/* Clock Divider Register1 (CLKDIV1) */
typedef struct{
__REG32                 :16;
__REG32 ADC_N           : 8;
__REG32                 : 8;
} __clkdiv1_bits;

/* MPLL Control Register (MPLLCON) */
typedef struct{
__REG32 FB_DV           : 9;
__REG32 IN_DV           : 5;
__REG32 OUT_DV          : 2;
__REG32 PD              : 1;
__REG32 BP              : 1;
__REG32 OE              : 1;
__REG32                 :13;
} __mpllcon_bits;

/* Control and Status Register (CNTRL) */
typedef struct{
__REG32 GO_BUSY         : 1;
__REG32 Rx_NEG          : 1;
__REG32 Tx_NEG          : 1;
__REG32 Tx_BIT_LEN      : 5;
__REG32 Tx_NUM          : 2;
__REG32 LSB             : 1;
__REG32 INS_DUMMY       : 1;
__REG32 SLEEP           : 4;
__REG32 IF              : 1;
__REG32 IE              : 1;
__REG32 F_TYPE          : 1;
__REG32 F_DRD           : 1;
__REG32 BOOT_SPI        : 1;
__REG32 DIS_M           : 1;
__REG32 COMMAND         : 1;
__REG32 OEN             : 1;
__REG32 SPI_MODE        : 8;
} __spim_cntrl_bits;

/* Divider Register (DIVIDER) */
typedef struct{
__REG32 DIVIDER         :16;
__REG32 IDLE_CNT        : 4;
__REG32 SCLK_IN_DLY     : 3;
__REG32                 : 9;
} __spim_divider_bits;

/* Slave Select Register (SSR) */
typedef struct{
__REG32 SSR             : 1;
__REG32                 : 1;
__REG32 SS_LVL          : 1;
__REG32 ASS             : 1;
__REG32                 :28;
} __spim_ssr_bits;

/* Code Length Register (CODE_LEN) */
typedef struct{
__REG32 CODE_LEN        :24;
__REG32                 : 8;
} __spim_code_len_bits;

/* SPI Flash Start Address Register (SPIM_ADDR) */
typedef struct{
__REG32 SPIM_ADDR       :24;
__REG32                 : 8;
} __spim_addr_bits;

/* APU Control Registers */
typedef struct{
__REG32 APURUN          : 1;
__REG32                 :15;
__REG32 APURST          : 1;
__REG32                 :15;
} __apucon_bits;

/* Parameter Control Register */
typedef struct{
__REG32                 :16;
__REG32 SWAP            : 1;
__REG32                 : 8;
__REG32 ZERO_EN         : 1;
__REG32                 : 6;
} __parcon_bits;

/* APU Power Down Control Register */
typedef struct{
__REG32                 :16;
__REG32 ANA_PD          : 1;
__REG32                 :15;
} __pdcon_bits;

/* APU Interrupt Register */
typedef struct{
__REG32 T1INTS          : 1;
__REG32 T2INTS          : 1;
__REG32                 :14;
__REG32 T1INTEN         : 1;
__REG32 T2INTEN         : 1;
__REG32                 :14;
} __apuint_bits;

/* SRAM Control Register 0 (SCTRL0 ~ SCTRL15) */
typedef struct{
__REG32 VALID           : 1;
__REG32                 :10;
__REG32 TAG             :18;
__REG32                 : 3;
} __sctrl_bits;

/* USBD Interrupt Enable Register (IEF) */
typedef struct{
__REG32 BUSEN           : 1;
__REG32 USBEN           : 1;
__REG32 FLDEN           : 1;
__REG32 WAKEUPEN        : 1;
__REG32                 : 4;
__REG32 WAKEFUEN        : 1;
__REG32                 : 6;
__REG32 INNAKEN         : 1;
__REG32                 :16;
} __ief_bits;

/* USBD Interrupt Event Flag Register (EVF) */
typedef struct{
__REG32 BUS             : 1;
__REG32 USB             : 1;
__REG32 FLD             : 1;
__REG32 WAKEUP          : 1;
__REG32                 :12;
__REG32 EPTF0           : 1;
__REG32 EPTF1           : 1;
__REG32 EPTF2           : 1;
__REG32 EPTF3           : 1;
__REG32 EPTF4           : 1;
__REG32 EPTF5           : 1;
__REG32                 : 9;
__REG32 Setup           : 1;
} __evf_bits;

/* USBD Function Address Register (FADDR) */
typedef struct{
__REG32 FADDR           : 7;
__REG32                 :25;
} __faddr_bits;

/* USBD System States Register (STS) */
typedef struct{
__REG32 EPT             : 4;
__REG32 STS             : 3;
__REG32 Overrun         : 1;
__REG32 STS0            : 3;
__REG32 STS1            : 3;
__REG32 STS2            : 3;
__REG32 STS3            : 3;
__REG32 STS4            : 3;
__REG32 STS5            : 3;
__REG32                 : 6;
} __sts_bits;

/* USBD Bus States & Attribution Register (ATTR) */
typedef struct{
__REG32 usbRST          : 1;
__REG32 Suspend         : 1;
__REG32 Resume          : 1;
__REG32 Timeout         : 1;
__REG32 enPHY           : 1;
__REG32 RWakeup         : 1;
__REG32                 : 1;
__REG32 enUSB           : 1;
__REG32                 :24;
} __attr_bits;

/* USBD Floating detection Register (FLODET) */
typedef struct{
__REG32 FLODET          : 1;
__REG32                 :31;
} __flodet_bits;

/* USBD Buffer Segmentation Register (BUFSEGx) x = 0~5 */
typedef struct{
__REG32 BUFSEG          : 9;
__REG32                 :23;
} __bufseg_bits;

/* USBD Maximal Payload Register (MXPLDx) x = 0~5 */
typedef struct{
__REG32 MXPLD           : 8;
__REG32                 :24;
} __mxpld_bits;

/* USBD Configuration Register (CFGx) x = 0~5 */
typedef struct{
__REG32 EPT             : 4;
__REG32 ISOCH           : 1;
__REG32 state           : 2;
__REG32 DSQ             : 1;
__REG32                 : 1;
__REG32 stall_ctl       : 1;
__REG32                 :22;
} __cfg_bits;

/* USBD Extra Configuration Register (CFGPx) x = 0~5 */
typedef struct{
__REG32 CFGP            : 1;
__REG32 stall           : 1;
__REG32                 :30;
} __cfgp_bits;

/* AIC Source Control Registers AIC_SCR1 */
typedef struct{
__REG32 PRIORITY0       : 3;
__REG32                 : 3;
__REG32 TYPE0           : 2;
__REG32 PRIORITY1       : 3;
__REG32                 : 3;
__REG32 TYPE1           : 2;
__REG32 PRIORITY2       : 3;
__REG32                 : 3;
__REG32 TYPE2           : 2;
__REG32 PRIORITY3       : 3;
__REG32                 : 3;
__REG32 TYPE3           : 2;
} __aic_scr1_bits;

/* AIC Source Control Registers AIC_SCR2 */
typedef struct{
__REG32 PRIORITY4       : 3;
__REG32                 : 3;
__REG32 TYPE4           : 2;
__REG32 PRIORITY5       : 3;
__REG32                 : 3;
__REG32 TYPE5           : 2;
__REG32 PRIORITY6       : 3;
__REG32                 : 3;
__REG32 TYPE6           : 2;
__REG32 PRIORITY7       : 3;
__REG32                 : 3;
__REG32 TYPE7           : 2;
} __aic_scr2_bits;

/* AIC Source Control Registers AIC_SCR3 */
typedef struct{
__REG32 PRIORITY8       : 3;
__REG32                 : 3;
__REG32 TYPE8           : 2;
__REG32 PRIORITY9       : 3;
__REG32                 : 3;
__REG32 TYPE9           : 2;
__REG32 PRIORITY10      : 3;
__REG32                 : 3;
__REG32 TYPE10          : 2;
__REG32 PRIORITY11      : 3;
__REG32                 : 3;
__REG32 TYPE11          : 2;
} __aic_scr3_bits;

/* AIC Source Control Registers AIC_SCR4 */
typedef struct{
__REG32 PRIORITY12      : 3;
__REG32                 : 3;
__REG32 TYPE12          : 2;
__REG32 PRIORITY13      : 3;
__REG32                 : 3;
__REG32 TYPE13          : 2;
__REG32 PRIORITY14      : 3;
__REG32                 : 3;
__REG32 TYPE14          : 2;
__REG32 PRIORITY15      : 3;
__REG32                 : 3;
__REG32 TYPE15          : 2;
} __aic_scr4_bits;

/* AIC Source Control Registers AIC_SCR5 */
typedef struct{
__REG32 PRIORITY16      : 3;
__REG32                 : 3;
__REG32 TYPE16          : 2;
__REG32 PRIORITY17      : 3;
__REG32                 : 3;
__REG32 TYPE17          : 2;
__REG32 PRIORITY18      : 3;
__REG32                 : 3;
__REG32 TYPE18          : 2;
__REG32 PRIORITY19      : 3;
__REG32                 : 3;
__REG32 TYPE19          : 2;
} __aic_scr5_bits;

/* AIC Source Control Registers AIC_SCR6 */
typedef struct{
__REG32 PRIORITY20      : 3;
__REG32                 : 3;
__REG32 TYPE20          : 2;
__REG32 PRIORITY21      : 3;
__REG32                 : 3;
__REG32 TYPE21          : 2;
__REG32 PRIORITY22      : 3;
__REG32                 : 3;
__REG32 TYPE22          : 2;
__REG32 PRIORITY23      : 3;
__REG32                 : 3;
__REG32 TYPE23          : 2;
} __aic_scr6_bits;

/* AIC Source Control Registers AIC_SCR7 */
typedef struct{
__REG32 PRIORITY24      : 3;
__REG32                 : 3;
__REG32 TYPE24          : 2;
__REG32 PRIORITY25      : 3;
__REG32                 : 3;
__REG32 TYPE25          : 2;
__REG32 PRIORITY26      : 3;
__REG32                 : 3;
__REG32 TYPE26          : 2;
__REG32 PRIORITY27      : 3;
__REG32                 : 3;
__REG32 TYPE27          : 2;
} __aic_scr7_bits;

/* AIC Source Control Registers AIC_SCR8 */
typedef struct{
__REG32 PRIORITY28      : 3;
__REG32                 : 3;
__REG32 TYPE28          : 2;
__REG32 PRIORITY29      : 3;
__REG32                 : 3;
__REG32 TYPE29          : 2;
__REG32 PRIORITY30      : 3;
__REG32                 : 3;
__REG32 TYPE30          : 2;
__REG32 PRIORITY31      : 3;
__REG32                 : 3;
__REG32 TYPE31          : 2;
} __aic_scr8_bits;

/* AIC Interrupt Raw Status Register (AIC_IRSR) */
typedef struct{
__REG32                 : 1;
__REG32 INT_WDT         : 1;
__REG32                 : 1;
__REG32 INT_GPIO0       : 1;
__REG32 INT_GPIO1       : 1;
__REG32 INT_GPIO2       : 1;
__REG32 INT_GPIO3       : 1;
__REG32 INT_APU         : 1;
__REG32                 : 2;
__REG32 INT_ADC         : 1;
__REG32 INT_RTC         : 1;
__REG32 INT_UART0       : 1;
__REG32 INT_UART1       : 1;
__REG32 INT_TMR1        : 1;
__REG32 INT_TMR0        : 1;
__REG32                 : 3;
__REG32 INT_USB         : 1;
__REG32                 : 2;
__REG32 INT_PWM0        : 1;
__REG32 INT_PWM1        : 1;
__REG32 INT_PWM2        : 1;
__REG32 INT_PWM3        : 1;
__REG32 INT_I2C         : 1;
__REG32 INT_SPIMS       : 1;
__REG32                 : 1;
__REG32 INT_PWR         : 1;
__REG32 INT_SPIM        : 1;
__REG32                 : 1;
} __aic_irsr_bits;

/* AIC IRQ Priority Encoding Register (AIC_IPER) */
typedef struct{
__REG32                 : 2;
__REG32 VECTOR          : 5;
__REG32                 :25;
} __aic_iper_bits;

/* AIC Interrupt Source Number Register (AIC_ISNR) */
typedef struct{
__REG32 IRQID           : 5;
__REG32                 :27;
} __aic_isnr_bits;

/* AIC Output Interrupt Status Register (AIC_OISR) */
typedef struct{
__REG32 FIQ             : 1;
__REG32 IRQ             : 1;
__REG32                 :30;
} __aic_oisr_bits;

/* GPIO Port [A] Bit Output Mode Enable (GPIOA_OMD) */
typedef struct{
__REG32 OMD0            : 1;
__REG32 OMD1            : 1;
__REG32 OMD2            : 1;
__REG32 OMD3            : 1;
__REG32 OMD4            : 1;
__REG32 OMD5            : 1;
__REG32 OMD6            : 1;
__REG32 OMD7            : 1;
__REG32 OMD8            : 1;
__REG32 OMD9            : 1;
__REG32 OMD10           : 1;
__REG32 OMD11           : 1;
__REG32 OMD12           : 1;
__REG32 OMD13           : 1;
__REG32 OMD14           : 1;
__REG32 OMD15           : 1;
__REG32                 :16;
} __gpioa_omd_bits;

/* GPIO Port [B] Bit Output Mode Enable (GPIOB_OMD) */
typedef struct{
__REG32 OMD0            : 1;
__REG32 OMD1            : 1;
__REG32 OMD2            : 1;
__REG32 OMD3            : 1;
__REG32 OMD4            : 1;
__REG32 OMD5            : 1;
__REG32 OMD6            : 1;
__REG32 OMD7            : 1;
__REG32 OMD8            : 1;
__REG32 OMD9            : 1;
__REG32                 :22;
} __gpiob_omd_bits;

/* GPIO Port [C] Bit Output Mode Enable (GPIOC_OMD) */
typedef struct{
__REG32 OMD0            : 1;
__REG32 OMD1            : 1;
__REG32 OMD2            : 1;
__REG32 OMD3            : 1;
__REG32 OMD4            : 1;
__REG32 OMD5            : 1;
__REG32 OMD6            : 1;
__REG32 OMD7            : 1;
__REG32 OMD8            : 1;
__REG32 OMD9            : 1;
__REG32 OMD10           : 1;
__REG32                 :21;
} __gpioc_omd_bits;

/* GPIO Port [A] Bit Pull-up Resistor Enable (GPIOA_PUEN) */
typedef struct{
__REG32 PUEN0           : 1;
__REG32 PUEN1           : 1;
__REG32 PUEN2           : 1;
__REG32 PUEN3           : 1;
__REG32 PUEN4           : 1;
__REG32 PUEN5           : 1;
__REG32 PUEN6           : 1;
__REG32 PUEN7           : 1;
__REG32 PUEN8           : 1;
__REG32 PUEN9           : 1;
__REG32 PUEN10          : 1;
__REG32 PUEN11          : 1;
__REG32 PUEN12          : 1;
__REG32 PUEN13          : 1;
__REG32 PUEN14          : 1;
__REG32 PUEN15          : 1;
__REG32                 :16;
} __gpioa_puen_bits;

/* GPIO Port [B] Bit Pull-up Resistor Enable (GPIOB_PUEN) */
typedef struct{
__REG32 PUEN0           : 1;
__REG32 PUEN1           : 1;
__REG32 PUEN2           : 1;
__REG32 PUEN3           : 1;
__REG32 PUEN4           : 1;
__REG32 PUEN5           : 1;
__REG32 PUEN6           : 1;
__REG32 PUEN7           : 1;
__REG32 PUEN8           : 1;
__REG32 PUEN9           : 1;
__REG32                 :22;
} __gpiob_puen_bits;

/* GPIO Port [C] Bit Pull-up Resistor Enable (GPIOC_PUEN) */
typedef struct{
__REG32 PUEN0           : 1;
__REG32 PUEN1           : 1;
__REG32 PUEN2           : 1;
__REG32 PUEN3           : 1;
__REG32 PUEN4           : 1;
__REG32 PUEN5           : 1;
__REG32 PUEN6           : 1;
__REG32 PUEN7           : 1;
__REG32 PUEN8           : 1;
__REG32 PUEN9           : 1;
__REG32 PUEN10          : 1;
__REG32                 :21;
} __gpioc_puen_bits;

/* GPIO Port [A] Data Output Value (GPIOA_DOUT) */
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
} __gpioa_dout_bits;

/* GPIO Port [B] Data Output Value (GPIOB_DOUT) */
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
__REG32                 :22;
} __gpiob_dout_bits;

/* GPIO Port [C] Data Output Value (GPIOC_DOUT) */
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
__REG32                 :21;
} __gpioc_dout_bits;

/* GPIO Port [A] Pin Value (GPIOA _PIN) */
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
} __gpioa_pin_bits;

/* GPIO Port [B] Pin Value (GPIOB_PIN) */
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
__REG32                 :22;
} __gpiob_pin_bits;

/* GPIO Port [C] Pin Value (GPIOC_PIN) */
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
__REG32                 :21;
} __gpioc_pin_bits;

/* Interrupt Debounce Control (DBNCECON) */
typedef struct{
__REG32 DBEN            : 4;
__REG32 DBCLKSEL        : 4;
__REG32                 :24;
} __dbncecon_bits;

/* GPIO Port A IRQ Source Grouping */
typedef struct{
__REG32 GPA0SEL         : 2;
__REG32 GPA1SEL         : 2;
__REG32 GPA2SEL         : 2;
__REG32 GPA3SEL         : 2;
__REG32 GPA4SEL         : 2;
__REG32 GPA5SEL         : 2;
__REG32 GPA6SEL         : 2;
__REG32 GPA7SEL         : 2;
__REG32 GPA8SEL         : 2;
__REG32 GPA9SEL         : 2;
__REG32 GPA10SEL        : 2;
__REG32 GPA11SEL        : 2;
__REG32 GPA12SEL        : 2;
__REG32 GPA13SEL        : 2;
__REG32 GPA14SEL        : 2;
__REG32 GPA15SEL        : 2;
} __irqsrcgpa_bits;

/* GPIO Port B IRQ Source Grouping */
typedef struct{
__REG32 GPB0SEL         : 2;
__REG32 GPB1SEL         : 2;
__REG32 GPB2SEL         : 2;
__REG32 GPB3SEL         : 2;
__REG32 GPB4SEL         : 2;
__REG32 GPB5SEL         : 2;
__REG32 GPB6SEL         : 2;
__REG32 GPB7SEL         : 2;
__REG32 GPB8SEL         : 2;
__REG32 GPB9SEL         : 2;
__REG32                 :12;
} __irqsrcgpb_bits;

/* GPIO Port C IRQ Source Grouping */
typedef struct{
__REG32 GPC0SEL         : 2;
__REG32 GPC1SEL         : 2;
__REG32 GPC2SEL         : 2;
__REG32 GPC3SEL         : 2;
__REG32 GPC4SEL         : 2;
__REG32 GPC5SEL         : 2;
__REG32 GPC6SEL         : 2;
__REG32 GPC7SEL         : 2;
__REG32 GPC8SEL         : 2;
__REG32 GPC9SEL         : 2;
__REG32 GPC10SEL        : 2;
__REG32                 :10;
} __irqsrcgpc_bits;

/* GPIO A Interrupt Enable (IRQENGPA) */
typedef struct{
__REG32 PA0ENF          : 1;
__REG32 PA1ENF          : 1;
__REG32 PA2ENF          : 1;
__REG32 PA3ENF          : 1;
__REG32 PA4ENF          : 1;
__REG32 PA5ENF          : 1;
__REG32 PA6ENF          : 1;
__REG32 PA7ENF          : 1;
__REG32 PA8ENF          : 1;
__REG32 PA9ENF          : 1;
__REG32 PA10ENF         : 1;
__REG32 PA11ENF         : 1;
__REG32 PA12ENF         : 1;
__REG32 PA13ENF         : 1;
__REG32 PA14ENF         : 1;
__REG32 PA15ENF         : 1;
__REG32 PA0ENR          : 1;
__REG32 PA1ENR          : 1;
__REG32 PA2ENR          : 1;
__REG32 PA3ENR          : 1;
__REG32 PA4ENR          : 1;
__REG32 PA5ENR          : 1;
__REG32 PA6ENR          : 1;
__REG32 PA7ENR          : 1;
__REG32 PA8ENR          : 1;
__REG32 PA9ENR          : 1;
__REG32 PA10ENR         : 1;
__REG32 PA11ENR         : 1;
__REG32 PA12ENR         : 1;
__REG32 PA13ENR         : 1;
__REG32 PA14ENR         : 1;
__REG32 PA15ENR         : 1;
} __irqengpa_bits;

/* GPIO B Interrupt Enable (IRQENGPB) */
typedef struct{
__REG32 PB0ENF          : 1;
__REG32 PB1ENF          : 1;
__REG32 PB2ENF          : 1;
__REG32 PB3ENF          : 1;
__REG32 PB4ENF          : 1;
__REG32 PB5ENF          : 1;
__REG32 PB6ENF          : 1;
__REG32 PB7ENF          : 1;
__REG32 PB8ENF          : 1;
__REG32 PB9ENF          : 1;
__REG32                 : 6;
__REG32 PB0ENR          : 1;
__REG32 PB1ENR          : 1;
__REG32 PB2ENR          : 1;
__REG32 PB3ENR          : 1;
__REG32 PB4ENR          : 1;
__REG32 PB5ENR          : 1;
__REG32 PB6ENR          : 1;
__REG32 PB7ENR          : 1;
__REG32 PB8ENR          : 1;
__REG32 PB9ENR          : 1;
__REG32                 : 6;
} __irqengpb_bits;

/* GPIO C Interrupt Enable (IRQENGPC) */
typedef struct{
__REG32 PC0ENF          : 1;
__REG32 PC1ENF          : 1;
__REG32 PC2ENF          : 1;
__REG32 PC3ENF          : 1;
__REG32 PC4ENF          : 1;
__REG32 PC5ENF          : 1;
__REG32 PC6ENF          : 1;
__REG32 PC7ENF          : 1;
__REG32 PC8ENF          : 1;
__REG32 PC9ENF          : 1;
__REG32 PC10ENF         : 1;
__REG32                 : 5;
__REG32 PC0ENR          : 1;
__REG32 PC1ENR          : 1;
__REG32 PC2ENR          : 1;
__REG32 PC3ENR          : 1;
__REG32 PC4ENR          : 1;
__REG32 PC5ENR          : 1;
__REG32 PC6ENR          : 1;
__REG32 PC7ENR          : 1;
__REG32 PC8ENR          : 1;
__REG32 PC9ENR          : 1;
__REG32 PC10ENR         : 1;
__REG32                 : 5;
} __irqengpc_bits;

/* Interrupt Latch Trigger Selection (IRQLHSEL) */
typedef struct{
__REG32 IRQ0LHE         : 1;
__REG32 IRQ1LHE         : 1;
__REG32 IRQ2LHE         : 1;
__REG32 IRQ3LHE         : 1;
__REG32 IRQ0Wake        : 1;
__REG32 IRQ1Wake        : 1;
__REG32 IRQ2Wake        : 1;
__REG32 IRQ3Wake        : 1;
__REG32 IRQ_SRCC        : 1;
__REG32                 :23;
} __irqlhsel_bits;

/* GPIO A Interrupt Latch (IRQLHGPA) */
typedef struct{
__REG32 PA0LHV          : 1;
__REG32 PA1LHV          : 1;
__REG32 PA2LHV          : 1;
__REG32 PA3LHV          : 1;
__REG32 PA4LHV          : 1;
__REG32 PA5LHV          : 1;
__REG32 PA6LHV          : 1;
__REG32 PA7LHV          : 1;
__REG32 PA8LHV          : 1;
__REG32 PA9LHV          : 1;
__REG32 PA10LHV         : 1;
__REG32 PA11LHV         : 1;
__REG32 PA12LHV         : 1;
__REG32 PA13LHV         : 1;
__REG32 PA14LHV         : 1;
__REG32 PA15LHV         : 1;
__REG32                 :16;
} __irqlhgpa_bits;

/* GPIO B Interrupt Latch (IRQLHGPB) */
typedef struct{
__REG32 PB0LHV          : 1;
__REG32 PB1LHV          : 1;
__REG32 PB2LHV          : 1;
__REG32 PB3LHV          : 1;
__REG32 PB4LHV          : 1;
__REG32 PB5LHV          : 1;
__REG32 PB6LHV          : 1;
__REG32 PB7LHV          : 1;
__REG32 PB8LHV          : 1;
__REG32 PB9LHV          : 1;
__REG32                 :22;
} __irqlhgpb_bits;

/* GPIO C Interrupt Latch (IRQLHGPC) */
typedef struct{
__REG32 PC0LHV          : 1;
__REG32 PC1LHV          : 1;
__REG32 PC2LHV          : 1;
__REG32 PC3LHV          : 1;
__REG32 PC4LHV          : 1;
__REG32 PC5LHV          : 1;
__REG32 PC6LHV          : 1;
__REG32 PC7LHV          : 1;
__REG32 PC8LHV          : 1;
__REG32 PC9LHV          : 1;
__REG32 PC10LHV         : 1;
__REG32                 :21;
} __irqlhgpc_bits;

/* IRQ Interrupt Trigger Source 0 (IRQTGSRC0) */
typedef struct{
__REG32 PA0TG           : 1;
__REG32 PA1TG           : 1;
__REG32 PA2TG           : 1;
__REG32 PA3TG           : 1;
__REG32 PA4TG           : 1;
__REG32 PA5TG           : 1;
__REG32 PA6TG           : 1;
__REG32 PA7TG           : 1;
__REG32 PA8TG           : 1;
__REG32 PA9TG           : 1;
__REG32 PA10TG          : 1;
__REG32 PA11TG          : 1;
__REG32 PA12TG          : 1;
__REG32 PA13TG          : 1;
__REG32 PA14TG          : 1;
__REG32 PA15TG          : 1;
__REG32 PB0TG           : 1;
__REG32 PB1TG           : 1;
__REG32 PB2TG           : 1;
__REG32 PB3TG           : 1;
__REG32 PB4TG           : 1;
__REG32 PB5TG           : 1;
__REG32 PB6TG           : 1;
__REG32 PB7TG           : 1;
__REG32 PB8TG           : 1;
__REG32 PB9TG           : 1;
__REG32                 : 6;
} __irqtgsrc0_bits;

/* IRQ Interrupt Trigger Source 1 (IRQTGSRC1) */
typedef struct{
__REG32 PC0TG           : 1;
__REG32 PC1TG           : 1;
__REG32 PC2TG           : 1;
__REG32 PC3TG           : 1;
__REG32 PC4TG           : 1;
__REG32 PC5TG           : 1;
__REG32 PC6TG           : 1;
__REG32 PC7TG           : 1;
__REG32 PC8TG           : 1;
__REG32 PC9TG           : 1;
__REG32 PC10TG          : 1;
__REG32                 :21;
} __irqtgsrc1_bits;

/* I2C Control and Status Register (CSR) */
typedef struct{
__REG32 I2C_EN          : 1;
__REG32 IE              : 1;
__REG32 IF              : 1;
__REG32                 : 1;
__REG32 Tx_NUM          : 2;
__REG32                 : 2;
__REG32 I2C_TIP         : 1;
__REG32 I2C_AL          : 1;
__REG32 I2C_BUSY        : 1;
__REG32 I2C_RxACK       : 1;
__REG32                 :20;
} __i2c_csr_bits;

/* I2C Pre-scale Register (DIVIDER) */
typedef struct{
__REG32 DIVIDER         :16;
__REG32                 :16;
} __i2c_divider_bits;

/* I2C Command Register (CMDR) */
typedef struct{
__REG32 ACK             : 1;
__REG32 WRITE           : 1;
__REG32 READ            : 1;
__REG32 STOP            : 1;
__REG32 START           : 1;
__REG32                 :27;
} __i2c_cmdr_bits;

/* I2C Software Mode Register (SWR) */
typedef struct{
__REG32 SCW             : 1;
__REG32 SDW             : 1;
__REG32 SEW             : 1;
__REG32 SCR             : 1;
__REG32 SDR             : 1;
__REG32 SER             : 1;
__REG32                 :26;
} __i2c_swr_bits;

/* I2C Data Receive Register (RxR) */
typedef struct{
__REG32 RX              : 8;
__REG32                 :24;
} __i2c_rxd_bits;

/* PWM Pre-Scale Register (PPR) */
typedef struct{
__REG32 CP0             : 8;
__REG32 CP1             : 8;
__REG32 DZI0            : 8;
__REG32 DZI1            : 8;
} __pwm_ppr_bits;

/* PWM Clock Selector Register (CSR) */
typedef struct{
__REG32 CSR0            : 3;
__REG32                 : 1;
__REG32 CSR1            : 3;
__REG32                 : 1;
__REG32 CSR2            : 3;
__REG32                 : 1;
__REG32 CSR3            : 3;
__REG32                 :17;
} __pwm_csr_bits;

/* PWM Control Register (PCR) */
typedef struct{
__REG32 CH0EN           : 1;
__REG32                 : 1;
__REG32 CH0INV          : 1;
__REG32 CH0MOD          : 1;
__REG32 DZEN0           : 1;
__REG32 DZEN1           : 1;
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
} __pwm_pcr_bits;

/* PWM Counter Register 3-0 (CNR3-0) */
typedef struct{
__REG32 CNR             :16;
__REG32                 :16;
} __pwm_cnr_bits;

/* PWM Comparator Register 3-0 (CMR3-0) */
typedef struct{
__REG32 CMR             :16;
__REG32                 :16;
} __pwm_cmr_bits;

/* PWM Data Register 3-0 (PDR 3-0) */
typedef struct{
__REG32 PDR             :16;
__REG32                 :16;
} __pwm_pdr_bits;

/* PWM Interrupt Enable Register (PIER) */
typedef struct{
__REG32 PIER0           : 1;
__REG32 PIER1           : 1;
__REG32 PIER2           : 1;
__REG32 PIER3           : 1;
__REG32                 :28;
} __pwm_pier_bits;

/* PWM Interrupt Indication Register (PIIR) */
typedef struct{
__REG32 PIIR0           : 1;
__REG32 PIIR1           : 1;
__REG32 PIIR2           : 1;
__REG32 PIIR3           : 1;
__REG32                 :28;
} __pwm_piir_bits;

/* PWM Capture Control Register (CCR0) */
typedef struct{
__REG32 INV0            : 1;
__REG32 RL_IE0          : 1;
__REG32 FL_IE0          : 1;
__REG32 CAPCH0EN        : 1;
__REG32 CIIR0           : 1;
__REG32                 : 1;
__REG32 CRLRD0          : 1;
__REG32 CFLRD0          : 1;
__REG32                 : 8;
__REG32 INV1            : 1;
__REG32 RL_IE1          : 1;
__REG32 FL_IE1          : 1;
__REG32 CAPCH1EN        : 1;
__REG32 CIIR1           : 1;
__REG32                 : 1;
__REG32 CRLRD1          : 1;
__REG32 CFLRD1          : 1;
__REG32                 : 8;
} __pwm_ccr0_bits;

/* PWM Capture Control Register (CCR1) */
typedef struct{
__REG32 INV2            : 1;
__REG32 RL_IE2          : 1;
__REG32 FL_IE2          : 1;
__REG32 CAPCH2EN        : 1;
__REG32 CIIR2           : 1;
__REG32                 : 1;
__REG32 CRLRD2          : 1;
__REG32 CFLRD2          : 1;
__REG32                 : 8;
__REG32 INV3            : 1;
__REG32 RL_IE3          : 1;
__REG32 FL_IE3          : 1;
__REG32 CAPCH3EN        : 1;
__REG32 CIIR3           : 1;
__REG32                 : 1;
__REG32 CRLRD3          : 1;
__REG32                 : 9;
} __pwm_ccr1_bits;

/* PWM Capture Rising Latch Register3-0 (CRLR3-0) */
typedef struct{
__REG32 CRLR            :16;
__REG32                 :16;
} __pwm_crlr_bits;

/* PWM Capture Falling Latch Register3-0 (CFLR3-0) */
typedef struct{
__REG32 CFLR0           :16;
__REG32                 :16;
} __pwm_cflr_bits;

/* PWM Capture Input Enable Register (CAPENR) */
typedef struct{
__REG32 CAPENR          : 4;
__REG32                 :28;
} __pwm_capenr_bits;

/* PWM Output Enable Register (POE) */
typedef struct{
__REG32 PWM0            : 1;
__REG32 PWM1            : 1;
__REG32 PWM2            : 1;
__REG32 PWM3            : 1;
__REG32                 :28;
} __pwm_poe_bits;

/* RTC Initiation Register (INIR) */
typedef struct{
__REG32 Active          : 1;
__REG32 INIR            :31;
} __rtc_inir_bits;

/* RTC Access Enable Register (AER) */
typedef struct{
__REG32 AER             :16;
__REG32 ENF             : 1;
__REG32                 :15;
} __rtc_aer_bits;

/* RTC Frequency Compensation Register (FCR) */
typedef struct{
__REG32 FRACTION        : 6;
__REG32                 : 2;
__REG32 INTEGER         : 4;
__REG32                 :20;
} __rtc_fcr_bits;

/* RTC Time Loading Register (TLR) */
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

/* RTC Calendar Loading Register (CLR) */
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

/* RTC Time Scale Selection Register (TSSR) */
typedef struct{
__REG32 _24HOUR         : 1;
__REG32                 :31;
} __rtc_tssr_bits;

/* RTC Day of the Week Register (DWR) */
typedef struct{
__REG32 DWR             : 3;
__REG32                 :29;
} __rtc_dwr_bits;

/* RTC Leap year Indication Register (LIR) */
typedef struct{
__REG32 LIR             : 1;
__REG32                 :31;
} __rtc_lir_bits;

/* RTC Interrupt Enable Register (RIER) */
typedef struct{
__REG32 AIER            : 1;
__REG32 TIER            : 1;
__REG32                 :30;
} __rtc_rier_bits;

/* RTC Interrupt Indication Register (RIIR) */
typedef struct{
__REG32 AI              : 1;
__REG32 TI              : 1;
__REG32                 :30;
} __rtc_riir_bits;

/* RTC Time Tick Register (TTR) */
typedef struct{
__REG32 TTR             : 3;
__REG32                 :29;
} __rtc_ttr_bits;

/* SPI Control and Status Register (CNTRL) */
typedef struct{
__REG32 GO_BUSY         : 1;
__REG32 Rx_NEG          : 1;
__REG32 Tx_NEG          : 1;
__REG32 Tx_BIT_LEN      : 5;
__REG32 Tx_NUM          : 2;
__REG32 LSB             : 1;
__REG32 CLKP            : 1;
__REG32 SLEEP           : 4;
__REG32 IF              : 1;
__REG32 IE              : 1;
__REG32 SLAVE           : 1;
__REG32                 :13;
} __spi_cntrl_bits;

/* SPI Divider Register (DIVIDER) */
typedef struct{
__REG32 DIVIDER         :16;
__REG32                 :16;
} __spi_divider_bits;

/* SPI Slave Select Register (SSR) */
typedef struct{
__REG32 SSR             : 1;
__REG32                 : 1;
__REG32 SS_LVL          : 1;
__REG32 ASS             : 1;
__REG32                 :28;
} __spi_ssr_bits;

/* Timer Control Register 0~1 (TCSR0~TCSR1) */
typedef struct{
__REG32 PRESCALE        : 8;
__REG32                 :17;
__REG32 CACT            : 1;
__REG32 CRST            : 1;
__REG32 MODE            : 2;
__REG32 EI              : 1;
__REG32 CEN             : 1;
__REG32 nDBGACK_EN      : 1;
} __tcsr_bits;

/* Timer Interrupt Status Register (TISR) */
typedef struct{
__REG32 TIF0            : 1;
__REG32 TIF1            : 1;
__REG32                 :30;
} __tisr_bits;

/* Watchdog Timer Control Register (WTCR) */
typedef struct{
__REG32 WTR             : 1;
__REG32 WTRE            : 1;
__REG32 WTRF            : 1;
__REG32 WTIF            : 1;
__REG32 WTIS            : 2;
__REG32 WTIE            : 1;
__REG32 WTE             : 1;
__REG32 WTTME           : 1;
__REG32 nDBGACK_EN      : 1;
__REG32 WTCLK           : 1;
__REG32                 :21;
} __wtcr_bits;

/* UART Interrupt Enable Register (UA_IER)
   UART Divisor Latch (High Byte) Register (UA_DLM) */
typedef union {
  /*UAx_IER*/
  struct {
  __REG32 RDAIE             : 1;
  __REG32 THREIE            : 1;
  __REG32 RLSIE             : 1;
  __REG32 MSIE              : 1;
  __REG32 RTOIE             : 1;
  __REG32 nDBGACK_EN        : 1;
  __REG32 WakeIE            : 1;
  __REG32 Wake_o_IE         : 1;
  __REG32                   :24;
  };
  /* UAx_DLM*/
  struct {
  __REG32 BRD_HI            : 8;
  __REG32                   :24;
  };
} __uart_ier_bits;

/* UART Interrupt Identification Register (UA_IIR)
   UART FIFO Control Register (UA_FCR) */
typedef union {
  /*UAx_IIR*/
  struct {
  __REG32 IID               : 3;
  __REG32 IID_RX            : 1;
  __REG32                   : 1;
  __REG32 RFTLS             : 2;
  __REG32 FMES              : 1;
  __REG32                   :24;
  };
/* UAx_FCR*/
  struct {
  __REG32 FME               : 1;
  __REG32 RFR               : 1;
  __REG32 TFR               : 1;
  __REG32                   : 1;
  __REG32 RFITL             : 4;
  __REG32                   :24;
  };
} __uart_iir_bits;

/* UART Line Control Register (UA_LCR) */
typedef struct {
  __REG32 WLS               : 2;
  __REG32 NSB               : 1;
  __REG32 PBE               : 1;
  __REG32 EPE               : 1;
  __REG32 SPE               : 1;
  __REG32 BCB               : 1;
  __REG32 DLAB              : 1;
  __REG32                   :24;
} __uart_lcr_bits;

/* UART MODEM Control Register (UA_MCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 RTS               : 1;
  __REG32                   : 2;
  __REG32 LBME              : 1;
  __REG32                   :27;
} __uart_mcr_bits;

/* UART Line Status Control Register (UA_LSR) */
typedef struct {
  __REG32 RFDR              : 1;
  __REG32 OEI               : 1;
  __REG32 PEI               : 1;
  __REG32 FEI               : 1;
  __REG32 BII               : 1;
  __REG32 THRE              : 1;
  __REG32 TE                : 1;
  __REG32 ERR_RX            : 1;
  __REG32                   :24;
} __uart_lsr_bits;

/* UART Modem Status Register (UA_MSR) */
typedef struct {
  __REG32 DCTS              : 1;
  __REG32                   : 3;
  __REG32 CTS               : 1;
  __REG32                   :27;
} __uart_msr_bits;

/* UART Time out Register (UA_TOR) */
typedef struct {
  __REG32 TOIC              : 7;
  __REG32                   :25;
} __uart_tor_bits;

/* ADC Control Register (ADC_CON) */
typedef struct {
  __REG32 ADC_FINISH        : 1;
  __REG32 ADC_DIV           : 8;
  __REG32 ADC_MUX           : 3;
  __REG32 ADC_READ_CONV     : 1;
  __REG32 ADC_CONV          : 1;
  __REG32 ADC_MODE          : 2;
  __REG32 ADC_RST           : 1;
  __REG32 ADC_EN            : 1;
  __REG32 ADC_INT           : 1;
  __REG32 LVD_INT           : 1;
  __REG32                   : 1;
  __REG32 ADC_INT_EN        : 1;
  __REG32 LVD_INT_EN        : 1;
  __REG32                   : 9;
} __adc_con_bits;

/* ADC X data buffer (ADC_XDATA) */
typedef struct {
  __REG32 ADC_XDATA         :10;
  __REG32                   :22;
} __adc_xdata_bits;

/* Low Voltage Detector Control Register (LV_CON) */
typedef struct {
  __REG32 SW_CON            : 3;
  __REG32 LV_EN             : 1;
  __REG32                   :28;
} __lv_con_bits;

/* Low Voltage Detector Status Register (LV_STS) */
typedef struct {
  __REG32 LV_status         : 1;
  __REG32                   :31;
} __lv_sts_bits;

/* Audio control register (AUDIO_CON) */
typedef struct {
  __REG32 AUDIO_RESET       : 1;
  __REG32 AUDIO_EN          : 1;
  __REG32 AUDIO_HPEN        : 1;
  __REG32 AUDIO_VOL         : 5;
  __REG32 VOL_EN            : 1;
  __REG32 AUD_INT           : 1;
  __REG32                   :16;
  __REG32 OP_OFFSET         : 4;
  __REG32 AUD_INT_MODE      : 2;
} __audio_con_bits;

/* Audio control register (AUDIO_BUF0) */
typedef struct {
  __REG32 AUDIO_DATA0       :16;
  __REG32 AUDIO_DATA1       :16;
} __audio_buf0_bits;

/* Audio control register (AUDIO_BUF1) */
typedef struct {
  __REG32 AUDIO_DATA2       :16;
  __REG32 AUDIO_DATA3       :16;
} __audio_buf1_bits;

/* Audio control register (AUDIO_BUF2) */
typedef struct {
  __REG32 AUDIO_DATA4       :16;
  __REG32 AUDIO_DATA5       :16;
} __audio_buf2_bits;

/* Audio control register (AUDIO_BUF3) */
typedef struct {
  __REG32 AUDIO_DATA6       :16;
  __REG32 AUDIO_DATA7       :16;
} __audio_buf3_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** GCR (System Manager Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PDID,                  0xB1000000,__READ       ,__pdid_bits);
__IO_REG32_BIT(SPOCR,                 0xB1000004,__READ_WRITE ,__spocr_bits);
__IO_REG32_BIT(CPUCR,                 0xB1000008,__READ_WRITE ,__cpucr_bits);
__IO_REG32_BIT(MISCR,                 0xB100000C,__READ_WRITE ,__miscr_bits);
__IO_REG32_BIT(IPRST,                 0xB1000014,__READ_WRITE ,__iprst_bits);
__IO_REG32_BIT(AHB_CTRL,              0xB1000020,__READ_WRITE ,__ahb_ctrl_bits);
__IO_REG32_BIT(PAD_REG0,              0xB1000030,__READ_WRITE ,__pad_reg0_bits);
__IO_REG32_BIT(PAD_REG1,              0xB1000034,__READ_WRITE ,__pad_reg1_bits);
__IO_REG32_BIT(PAD_REG2,              0xB1000038,__READ_WRITE ,__pad_reg2_bits);
__IO_REG32_BIT(GPA_DS,                0xB1000074,__READ_WRITE ,__gpa_ds_bits);
__IO_REG32_BIT(GPB_DS,                0xB1000078,__READ_WRITE ,__gpb_ds_bits);
__IO_REG32_BIT(GPC_DS,                0xB100007C,__READ_WRITE ,__gpc_ds_bits);

/***************************************************************************
 **
 ** CLK (Clock Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PWRCON,                0xB1000200,__READ_WRITE ,__pwrcon_bits);
__IO_REG32_BIT(AHBCLK,                0xB1000204,__READ_WRITE ,__ahbclk_bits);
__IO_REG32_BIT(APBCLK,                0xB1000208,__READ_WRITE ,__apbclk_bits);
__IO_REG32_BIT(CLKSEL,                0xB1000210,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(CLKDIV0,               0xB1000214,__READ_WRITE ,__clkdiv0_bits);
__IO_REG32_BIT(CLKDIV1,               0xB1000218,__READ_WRITE ,__clkdiv1_bits);
__IO_REG32_BIT(MPLLCON,               0xB1000220,__READ_WRITE ,__mpllcon_bits);

/***************************************************************************
 **
 ** SPIM (SPIM Serial Interface Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(SPIM_CNTRL,            0xB1007000,__READ_WRITE ,__spim_cntrl_bits);
__IO_REG32_BIT(SPIM_DIVIDER,          0xB1007004,__READ_WRITE ,__spim_divider_bits);
__IO_REG32_BIT(SPIM_SSR,              0xB1007008,__READ_WRITE ,__spim_ssr_bits);
__IO_REG32(    SPIM_Rx0,              0xB1007010,__READ       );
__IO_REG32(    SPIM_Rx1,              0xB1007014,__READ       );
__IO_REG32(    SPIM_Rx2,              0xB1007018,__READ       );
__IO_REG32(    SPIM_Rx3,              0xB100701C,__READ       );
__IO_REG32(    SPIM_Tx0,              0xB1007020,__READ_WRITE );
__IO_REG32(    SPIM_Tx1,              0xB1007024,__READ_WRITE );
__IO_REG32(    SPIM_Tx2,              0xB1007028,__READ_WRITE );
__IO_REG32(    SPIM_Tx3,              0xB100702C,__READ_WRITE );
__IO_REG32(    SPIM_AHB_ADDR,         0xB1007030,__READ_WRITE );
__IO_REG32_BIT(SPIM_CODE_LEN,         0xB1007034,__READ_WRITE ,__spim_code_len_bits);
__IO_REG32_BIT(SPIM_ADDR,             0xB1007040,__READ_WRITE ,__spim_addr_bits);

/***************************************************************************
 **
 ** APU (Audio Processing Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(APUCON,                0xB1008000,__READ_WRITE ,__apucon_bits);
__IO_REG32_BIT(PARCON,                0xB1008004,__READ_WRITE ,__parcon_bits);
__IO_REG32_BIT(PDCON,                 0xB1008008,__READ_WRITE ,__pdcon_bits);
__IO_REG32_BIT(APUINT,                0xB100800C,__READ_WRITE ,__apuint_bits);
__IO_REG32(    RAMBSAD,               0xB1008010,__READ_WRITE );
__IO_REG32(    THAD1,                 0xB1008014,__READ_WRITE );
__IO_REG32(    THAD2,                 0xB1008018,__READ_WRITE );
__IO_REG32(    CURAD,                 0xB100801C,__READ       );

/***************************************************************************
 **
 ** SRAM Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(SCTRL0,                0xB1004000,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL1,                0xB1004004,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL2,                0xB1004008,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL3,                0xB100400C,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL4,                0xB1004010,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL5,                0xB1004014,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL6,                0xB1004018,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL7,                0xB100401C,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL8,                0xB1004020,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL9,                0xB1004024,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL10,               0xB1004028,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL11,               0xB100402C,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL12,               0xB1004030,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL13,               0xB1004034,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL14,               0xB1004038,__READ_WRITE ,__sctrl_bits);
__IO_REG32_BIT(SCTRL15,               0xB100403C,__READ_WRITE ,__sctrl_bits);

/***************************************************************************
 **
 ** USBD (USB Device Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(IEF,                   0xB1009000,__READ_WRITE ,__ief_bits);
__IO_REG32_BIT(EVF,                   0xB1009004,__READ       ,__evf_bits);
__IO_REG32_BIT(FADDR,                 0xB1009008,__READ_WRITE ,__faddr_bits);
__IO_REG32_BIT(STS,                   0xB100900C,__READ_WRITE ,__sts_bits);
__IO_REG32_BIT(ATTR,                  0xB1009010,__READ_WRITE ,__attr_bits);
__IO_REG32_BIT(FLODET,                0xB1009014,__READ       ,__flodet_bits);
__IO_REG32_BIT(BUFSEG,                0xB1009018,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(BUFSEG0,               0xB1009020,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD0,                0xB1009024,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG0,                  0xB1009028,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP0,                 0xB100902C,__READ_WRITE ,__cfgp_bits);
__IO_REG32_BIT(BUFSEG1,               0xB1009030,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD1,                0xB1009034,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG1,                  0xB1009038,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP1,                 0xB100903C,__READ_WRITE ,__cfgp_bits);
__IO_REG32_BIT(BUFSEG2,               0xB1009040,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD2,                0xB1009044,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG2,                  0xB1009048,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP2,                 0xB100904C,__READ_WRITE ,__cfgp_bits);
__IO_REG32_BIT(BUFSEG3,               0xB1009050,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD3,                0xB1009054,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG3,                  0xB1009058,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP3,                 0xB100905C,__READ_WRITE ,__cfgp_bits);
__IO_REG32_BIT(BUFSEG4,               0xB1009060,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD4,                0xB1009064,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG4,                  0xB1009068,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP4,                 0xB100906C,__READ_WRITE ,__cfgp_bits);
__IO_REG32_BIT(BUFSEG5,               0xB1009070,__READ_WRITE ,__bufseg_bits);
__IO_REG32_BIT(MXPLD5,                0xB1009074,__READ_WRITE ,__mxpld_bits);
__IO_REG32_BIT(CFG5,                  0xB1009078,__READ_WRITE ,__cfg_bits);
__IO_REG32_BIT(CFGP5,                 0xB100907C,__READ_WRITE ,__cfgp_bits);

/***************************************************************************
 **
 ** AIC (Advanced Interrupt Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(AIC_SCR1,              0xB8002000,__READ_WRITE ,__aic_scr1_bits);
__IO_REG32_BIT(AIC_SCR2,              0xB8002004,__READ_WRITE ,__aic_scr2_bits);
__IO_REG32_BIT(AIC_SCR3,              0xB8002008,__READ_WRITE ,__aic_scr3_bits);
__IO_REG32_BIT(AIC_SCR4,              0xB800200C,__READ_WRITE ,__aic_scr4_bits);
__IO_REG32_BIT(AIC_SCR5,              0xB8002010,__READ_WRITE ,__aic_scr5_bits);
__IO_REG32_BIT(AIC_SCR6,              0xB8002014,__READ_WRITE ,__aic_scr6_bits);
__IO_REG32_BIT(AIC_SCR7,              0xB8002018,__READ_WRITE ,__aic_scr7_bits);
__IO_REG32_BIT(AIC_SCR8,              0xB800201C,__READ_WRITE ,__aic_scr8_bits);
__IO_REG32_BIT(AIC_IRSR,              0xB8002100,__READ       ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_IASR,              0xB8002104,__READ       ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_ISR,               0xB8002108,__READ       ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_IPER,              0xB800210C,__READ       ,__aic_iper_bits);
__IO_REG32_BIT(AIC_ISNR,              0xB8002110,__READ       ,__aic_isnr_bits);
__IO_REG32_BIT(AIC_IMR,               0xB8002114,__READ       ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_OISR,              0xB8002118,__READ       ,__aic_oisr_bits);
__IO_REG32_BIT(AIC_MECR,              0xB8002120,__WRITE      ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_MDCR,              0xB8002124,__WRITE      ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_SSCR,              0xB8002128,__WRITE      ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_SCCR,              0xB800212C,__WRITE      ,__aic_irsr_bits);
__IO_REG32(    AIC_EOSCR,             0xB8002130,__WRITE      );
__IO_REG32(    AIC_TEST,              0xB8002134,__READ_WRITE );

/***************************************************************************
 **
 ** GPIO (General Purpose I/O)
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_OMD,             0xB8003000,__READ_WRITE ,__gpioa_omd_bits);
__IO_REG32_BIT(GPIOA_PUEN,            0xB8003004,__READ_WRITE ,__gpioa_puen_bits);
__IO_REG32_BIT(GPIOA_DOUT,            0xB8003008,__READ_WRITE ,__gpioa_dout_bits);
__IO_REG32_BIT(GPIOA_PIN,             0xB800300C,__READ       ,__gpioa_pin_bits);
__IO_REG32_BIT(GPIOB_OMD,             0xB8003010,__READ_WRITE ,__gpiob_omd_bits);
__IO_REG32_BIT(GPIOB_PUEN,            0xB8003014,__READ_WRITE ,__gpiob_puen_bits);
__IO_REG32_BIT(GPIOB_DOUT,            0xB8003018,__READ_WRITE ,__gpiob_dout_bits);
__IO_REG32_BIT(GPIOB_PIN,             0xB800301C,__READ       ,__gpiob_pin_bits);
__IO_REG32_BIT(GPIOC_OMD,             0xB8003020,__READ_WRITE ,__gpioc_omd_bits);
__IO_REG32_BIT(GPIOC_PUEN,            0xB8003024,__READ_WRITE ,__gpioc_puen_bits);
__IO_REG32_BIT(GPIOC_DOUT,            0xB8003028,__READ_WRITE ,__gpioc_dout_bits);
__IO_REG32_BIT(GPIOC_PIN,             0xB800302C,__READ       ,__gpioc_pin_bits);
__IO_REG32_BIT(DBNCECON,              0xB8003070,__READ_WRITE ,__dbncecon_bits);
__IO_REG32_BIT(IRQSRCGPA,             0xB8003080,__READ_WRITE ,__irqsrcgpa_bits);
__IO_REG32_BIT(IRQSRCGPB,             0xB8003084,__READ_WRITE ,__irqsrcgpb_bits);
__IO_REG32_BIT(IRQSRCGPC,             0xB8003088,__READ_WRITE ,__irqsrcgpc_bits);
__IO_REG32_BIT(IRQENGPA,              0xB8003090,__READ_WRITE ,__irqengpa_bits);
__IO_REG32_BIT(IRQENGPB,              0xB8003094,__READ_WRITE ,__irqengpb_bits);
__IO_REG32_BIT(IRQENGPC,              0xB8003098,__READ_WRITE ,__irqengpc_bits);
__IO_REG32_BIT(IRQLHSEL,              0xB80030A0,__READ_WRITE ,__irqlhsel_bits);
__IO_REG32_BIT(IRQLHGPA,              0xB80030A4,__READ       ,__irqlhgpa_bits);
__IO_REG32_BIT(IRQLHGPB,              0xB80030A8,__READ       ,__irqlhgpb_bits);
__IO_REG32_BIT(IRQLHGPC,              0xB80030AC,__READ       ,__irqlhgpc_bits);
__IO_REG32_BIT(IRQTGSRC0,             0xB80030B4,__READ_WRITE ,__irqtgsrc0_bits);
__IO_REG32_BIT(IRQTGSRC1,             0xB80030B8,__READ_WRITE ,__irqtgsrc1_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_CSR,               0xB8004000,__READ_WRITE ,__i2c_csr_bits);
__IO_REG32_BIT(I2C_DIVIDER,           0xB8004004,__READ_WRITE ,__i2c_divider_bits);
__IO_REG32_BIT(I2C_CMDR,              0xB8004008,__READ_WRITE ,__i2c_cmdr_bits);
__IO_REG32_BIT(I2C_SWR,               0xB800400C,__READ_WRITE ,__i2c_swr_bits);
__IO_REG32_BIT(I2C_RxD,               0xB8004010,__READ       ,__i2c_rxd_bits);
__IO_REG32(    I2C_TxD,               0xB8004014,__READ_WRITE );

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_PPR,               0xB8007000,__READ_WRITE ,__pwm_ppr_bits);
__IO_REG32_BIT(PWM_CSR,               0xB8007004,__READ_WRITE ,__pwm_csr_bits);
__IO_REG32_BIT(PWM_PCR,               0xB8007008,__READ_WRITE ,__pwm_pcr_bits);
__IO_REG32_BIT(PWM_CNR0,              0xB800700C,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR0,              0xB8007010,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR0,              0xB8007014,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR1,              0xB8007018,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR1,              0xB800701C,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR1,              0xB8007020,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR2,              0xB8007024,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR2,              0xB8007028,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR2,              0xB800702C,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR3,              0xB8007030,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR3,              0xB8007034,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR3,              0xB8007038,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_PIER,              0xB8007040,__READ_WRITE ,__pwm_pier_bits);
__IO_REG32_BIT(PWM_PIIR,              0xB8007044,__READ_WRITE ,__pwm_piir_bits);
__IO_REG32_BIT(PWM_CCR0,              0xB8007050,__READ_WRITE ,__pwm_ccr0_bits);
__IO_REG32_BIT(PWM_CCR1,              0xB8007054,__READ_WRITE ,__pwm_ccr1_bits);
__IO_REG32_BIT(PWM_CRLR0,             0xB8007058,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR0,             0xB800705C,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR1,             0xB8007060,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR1,             0xB8007064,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR2,             0xB8007068,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR2,             0xB800706C,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR3,             0xB8007070,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR3,             0xB8007074,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CAPENR,            0xB8007078,__READ_WRITE ,__pwm_capenr_bits);
__IO_REG32_BIT(PWM_POE,               0xB800707C,__READ_WRITE ,__pwm_poe_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_INIR,              0xB8008000,__READ_WRITE ,__rtc_inir_bits);
__IO_REG32_BIT(RTC_AER,               0xB8008004,__READ_WRITE ,__rtc_aer_bits);
__IO_REG32_BIT(RTC_FCR,               0xB8008008,__READ_WRITE ,__rtc_fcr_bits);
__IO_REG32_BIT(RTC_TLR,               0xB800800C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CLR,               0xB8008010,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_TSSR,              0xB8008014,__READ_WRITE ,__rtc_tssr_bits);
__IO_REG32_BIT(RTC_DWR,               0xB8008018,__READ_WRITE ,__rtc_dwr_bits);
__IO_REG32_BIT(RTC_TAR,               0xB800801C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CAR,               0xB8008020,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_LIR,               0xB8008024,__READ       ,__rtc_lir_bits);
__IO_REG32_BIT(RTC_RIER,              0xB8008028,__READ_WRITE ,__rtc_rier_bits);
__IO_REG32_BIT(RTC_RIIR,              0xB800802C,__READ_WRITE ,__rtc_riir_bits);
__IO_REG32_BIT(RTC_TTR,               0xB8008030,__READ_WRITE ,__rtc_ttr_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI_CNTRL,             0xB800A000,__READ_WRITE ,__spi_cntrl_bits);
__IO_REG32_BIT(SPI_DIVIDER,           0xB800A004,__READ_WRITE ,__spi_divider_bits);
__IO_REG32_BIT(SPI_SSR,               0xB800A008,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32(    SPI_Rx0,               0xB800A010,__READ_WRITE );
#define SPI_Tx0 SPI_Rx0
__IO_REG32(    SPI_Rx1,               0xB800A014,__READ_WRITE );
#define SPI_Tx1 SPI_Rx1
__IO_REG32(    SPI_Rx2,               0xB800A018,__READ_WRITE );
#define SPI_Tx2 SPI_Rx2
__IO_REG32(    SPI_Rx3,               0xB800A01C,__READ_WRITE );
#define SPI_Tx3 SPI_Rx3

/***************************************************************************
 **
 ** TIMER
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR0,                 0xB800B000,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCSR1,                 0xB800B004,__READ_WRITE ,__tcsr_bits);
__IO_REG32(    TICR0,                 0xB800B008,__READ_WRITE );
__IO_REG32(    TICR1,                 0xB800B00C,__READ_WRITE );
__IO_REG32(    TDR0,                  0xB800B010,__READ       );
__IO_REG32(    TDR1,                  0xB800B014,__READ       );
__IO_REG32_BIT(TISR,                  0xB800B018,__READ_WRITE ,__tisr_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WTCR,                  0xB800B01C,__READ_WRITE ,__wtcr_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG8(     UA0_RBR,               0xB800C000,__READ_WRITE );
#define UA0_THR     UA0_RBR
#define UA0_DLL     UA0_RBR
__IO_REG32_BIT(UA0_IER,               0xB800C004,__READ_WRITE ,__uart_ier_bits);
#define UA0_DLM     UA0_IER
#define UA0_DLM_bit UA0_IER_bit
__IO_REG32_BIT(UA0_IIR,               0xB800C008,__READ_WRITE ,__uart_iir_bits);
#define UA0_FCR     UA0_IIR
#define UA0_FCR_bit UA0_IIR_bit
__IO_REG32_BIT(UA0_LCR,               0xB800C00C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UA0_MCR,               0xB800C010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UA0_LSR,               0xB800C014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UA0_MSR,               0xB800C018,__READ       ,__uart_msr_bits);
__IO_REG32_BIT(UA0_TOR,               0xB800C01C,__READ_WRITE ,__uart_tor_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG8(     UA1_RBR,               0xB800C100,__READ_WRITE );
#define UA1_THR     UA1_RBR
#define UA1_DLL     UA1_RBR
__IO_REG32_BIT(UA1_IER,               0xB800C104,__READ_WRITE ,__uart_ier_bits);
#define UA1_DLM     UA1_IER
#define UA1_DLM_bit UA1_IER_bit
__IO_REG32_BIT(UA1_IIR,               0xB800C108,__READ_WRITE ,__uart_iir_bits);
#define UA1_FCR     UA1_IIR
#define UA1_FCR_bit UA1_IIR_bit
__IO_REG32_BIT(UA1_LCR,               0xB800C10C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UA1_MCR,               0xB800C110,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UA1_LSR,               0xB800C114,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UA1_MSR,               0xB800C118,__READ       ,__uart_msr_bits);
__IO_REG32_BIT(UA1_TOR,               0xB800C11C,__READ_WRITE ,__uart_tor_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CON,               0xB8001000,__READ_WRITE ,__adc_con_bits);
__IO_REG32_BIT(ADC_XDATA,             0xB800100C,__READ       ,__adc_xdata_bits);
__IO_REG32_BIT(LV_CON,                0xB8001014,__READ_WRITE ,__lv_con_bits);
__IO_REG32_BIT(LV_STS,                0xB8001018,__READ_WRITE ,__lv_sts_bits);
__IO_REG32_BIT(AUDIO_CON,             0xB800101C,__READ_WRITE ,__audio_con_bits);
__IO_REG32_BIT(AUDIO_BUF0,            0xB8001020,__READ_WRITE ,__audio_buf0_bits);
__IO_REG32_BIT(AUDIO_BUF1,            0xB8001024,__READ_WRITE ,__audio_buf1_bits);
__IO_REG32_BIT(AUDIO_BUF2,            0xB8001028,__READ_WRITE ,__audio_buf2_bits);
__IO_REG32_BIT(AUDIO_BUF3,            0xB800102C,__READ_WRITE ,__audio_buf3_bits);

/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV        0x00  /* Reset                                       */
#define UNDEFV        0x04  /* Undefined instruction                       */
#define SWIV          0x08  /* Software interrupt                          */
#define PABORTV       0x0c  /* Prefetch abort                              */
#define DABORTV       0x10  /* Data abort                                  */
#define IRQV          0x18  /* Normal interrupt                            */
#define FIQV          0x1c  /* Fast interrupt                              */

/***************************************************************************
 **
 **  VIC Interrupt channels
 **
 ***************************************************************************/
#define AIC_WDT          1  /* Watchdog                                    */
#define AIC_GPIO0        3  /* External Interrupt 0                        */
#define AIC_GPIO1        4  /* External Interrupt 1                        */
#define AIC_GPIO2        5  /* External Interrupt 2                        */
#define AIC_GPIO3        6  /* External Interrupt 3                        */
#define AIC_APU          7  /* Audio Processing Unit Interrupt             */
#define AIC_ADC         10  /* ADC Interrupt                               */
#define AIC_RTC         11  /* RTC Interrupt                               */
#define AIC_UART0       12  /* UART Interrupt0                             */
#define AIC_UART1       13  /* UART Interrupt1                             */
#define AIC_TIMER1      14  /* Timer Interrupt 1                           */
#define AIC_TIMER0      15  /* Timer Interrupt 0                           */
#define AIC_USB         19  /* USB Device Interrupt                        */
#define AIC_PWM0        22  /* PWM Interrupt 0                             */
#define AIC_PWM1        23  /* PWM Interrupt 1                             */
#define AIC_PWM2        24  /* PWM Interrupt 2                             */
#define AIC_PWM3        25  /* PWM Interrupt 3                             */
#define AIC_I2C         26  /* I2C Interrupt                               */
#define AIC_SPIMS       27  /* SPI(Master/Slave) Serial Interface Interrupt*/
#define AIC_PWR         29  /* System Wake-Up Interrupt                    */
#define AIC_SPIM        30  /* SPIM0/1 Interrupt                           */

#endif    /* __NUC510_H */
