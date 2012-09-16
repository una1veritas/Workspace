/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC3250
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 41683 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOLPC3250_H
#define __IOLPC3250_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC3250 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* Boot Map Control Register */
typedef struct{
__REG32 MAP  : 1;
__REG32      :31;
} __boot_map_bits;

/* Power Control Register Register */
typedef struct{
__REG32 STOP_MODE           : 1;
__REG32 HIGHCORE_SEL        : 1;
__REG32 RUN_MODE            : 1;
__REG32 SYSCLKEN_SEL        : 1;
__REG32 SYSCLKEN_OUT        : 1;
__REG32 HIGHCORE_OUT        : 1;
__REG32                     : 1;
__REG32 SDRAM_AESR          : 1;
__REG32 EMCSREFREQ_UPDATE   : 1;
__REG32 EMCSREFREQ_VALUE    : 1;
__REG32 HCKL_FORCE          : 1;
__REG32                     :21;
} __pwr_ctrl_bits;

/* Main Oscillator Control Register */
typedef struct{
__REG32 ENABLE   : 1;
__REG32 TEST     : 1;
__REG32 CAP_SEL  : 7;
__REG32          :23;
} __osc_ctrl_bits;

/* SYSCLK Control Register */
typedef struct{
__REG32 SYSCLK_MUX         : 1;
__REG32 SYSCLK_SEL         : 1;
__REG32 SYSCLK_SWITCH_DLY  :10;
__REG32                    :20;
} __sysclk_ctrl_bits;

/* PLL397 Control Register */
typedef struct{
__REG32 PLL_LOCK       : 1;
__REG32 PLL397_STATUS  : 1;
__REG32                : 4;
__REG32 PLL397_BIAS    : 3;
__REG32 PLL397_BYPASS  : 1;
__REG32 PLL_MSLOCK     : 1;
__REG32                :21;
} __pll397_ctrl_bits;

/* HCLK PLL Control Register */
typedef struct{
__REG32 PLL_LOCK    : 1;
__REG32 M           : 8;
__REG32 N           : 2;
__REG32 P           : 2;
__REG32 FEEDBACK    : 1;
__REG32 DIRECT      : 1;
__REG32 BYPASS      : 1;
__REG32 POWER_DOWN  : 1;
__REG32             :15;
} __hclkpll_ctrl_bits;

/* HCLK Divider Control Register */
typedef struct{
__REG32 HCLK        : 2;
__REG32 PERIPH_CLK  : 5;
__REG32 DDRAM_CLK   : 2;
__REG32             :23;
} __hclkdiv_ctrl_bits;

/* Test Clock Selection Register */
typedef struct{
__REG32 TST_CLK2      : 1;
__REG32 TST_CLK2_SEL  : 3;
__REG32 TST_CLK1      : 1;
__REG32 TST_CLK1_SEL  : 2;
__REG32               :25;
} __test_clk_bits;

/* Autoclock Control Register */
typedef struct{
__REG32 IROM       : 1;
__REG32 IRAM       : 1;
__REG32            : 4;
__REG32 USB_SLAVE  : 1;
__REG32            :25;
} __autoclk_ctrl_bits;

/* Start Enable Register for Pin Sources */
typedef struct{
__REG32               : 3;
__REG32 GPI_08        : 1;
__REG32 GPI_09        : 1;
__REG32 GPI_10        : 1;
__REG32 SPI2_DATIN    : 1;
__REG32 GPI_07        : 1;
__REG32 SPI1_DATIN    : 1;
__REG32 SYSCLKEN_PIN  : 1;
__REG32 GPI_00        : 1;
__REG32 GPI_01        : 1;
__REG32 GPI_02        : 1;
__REG32 GPI_03        : 1;
__REG32 GPI_04        : 1;
__REG32 GPI_05        : 1;
__REG32 GPI_06        : 1;
__REG32 MSDIO_START   : 1;
__REG32 SDIO_INT_N    : 1;
__REG32               : 2;
__REG32 U1_RX         : 1;
__REG32 U2_RX         : 1;
__REG32 U2_HCTS       : 1;
__REG32 U3_RX         : 1;
__REG32 GPI_11        : 1;
__REG32 U5_RX         : 1;
__REG32               : 1;
__REG32 U6_IRRX       : 1;
__REG32               : 1;
__REG32 U7_HCTS       : 1;
__REG32 U7_RX         : 1;
} __start_er_pin_bits;

/* Start Enable Register for Internal Sources */
typedef struct{
__REG32 GPIO_00            : 1;
__REG32 GPIO_01            : 1;
__REG32 GPIO_02            : 1;
__REG32 GPIO_03            : 1;
__REG32 GPIO_04            : 1;
__REG32 GPIO_05            : 1;
__REG32 P0_P1              : 1;
__REG32 EMAC               : 1;
__REG32                    : 8;
__REG32 KEY_IRQ            : 1;
__REG32                    : 2;
__REG32 USB_OTG_ATX_INT_N  : 1;
__REG32 USB_OTG_TIMER_INT  : 1;
__REG32 USB_I2C_INT        : 1;
__REG32 USB_INT            : 1;
__REG32 USB_NEED_CLK       : 1;
__REG32 RTC_INT            : 1;
__REG32 MSTIMER_INT        : 1;
__REG32 USB_AHB_NEED_CLK   : 1;
__REG32                    : 2;
__REG32 TS_AUX             : 1;
__REG32 TS_P               : 1;
__REG32 TS_INT             : 1;
} __start_er_int_bits;

/*Port 0 and Port 1 start and interrupt enable register */
typedef struct{
__REG32 P0_0               : 1;
__REG32 P0_1               : 1;
__REG32 P0_2               : 1;
__REG32 P0_3               : 1;
__REG32 P0_4               : 1;
__REG32 P0_5               : 1;
__REG32 P0_6               : 1;
__REG32 P0_7               : 1;
__REG32 P1_0               : 1;
__REG32 P1_2               : 1;
__REG32 P1_3               : 1;
__REG32 P1_23_4            : 1;
__REG32                    :20;
} __p0_intr_er_bits;

/* USB Control Register */
typedef struct{
__REG32 PLL_LOCK           : 1;
__REG32 M                  : 8;
__REG32 N                  : 2;
__REG32 P                  : 2;
__REG32 FEEDBACK           : 1;
__REG32 DIRECT             : 1;
__REG32 BYPASS             : 1;
__REG32 POWER_DOWN         : 1;
__REG32 USB_Clken1         : 1;
__REG32 USB_Clken2         : 1;
__REG32 PAD_CTRL           : 2;
__REG32 HOST_NEED_CLK_ENA  : 1;
__REG32 DEV_NEED_CLK_ENA   : 1;
__REG32 I2C_ENA            : 1;
__REG32 SLAVE_HCLK_CTRL    : 1;
__REG32                    : 7;
} __usb_ctrl_bits;

/* USB clock pre-divide register */
typedef struct{
__REG32 USB_RATE           : 4;
__REG32                    :28;
} __usbdiv_ctrl_bits;

/* Memory Card Control Register */
typedef struct{
__REG32 MSSDCLK_SEL         : 4;
__REG32                     : 1;
__REG32 CLK_ENA             : 1;
__REG32 MSSDIO0_PAD_CTRL    : 1;
__REG32 MSSDIO1_PAD_CTRL    : 1;
__REG32 MSSDIO2_3_PAD_CTRL  : 1;
__REG32 PULL_UP_ENA         : 1;
__REG32 SD_PINS_DIS         : 1;
__REG32                     :21;
} __ms_ctrl_bits;

/* DMA Clock Control Register */
typedef struct{
__REG32 DMA_CLK_ENA  : 1;
__REG32              :31;
} __dmaclk_ctrl_bits;

/* NAND Flash Clock Control Register */
typedef struct{
__REG32 SLC_NAND_ENA      : 1;
__REG32 MLC_NAND_ENA      : 1;
__REG32 SLC_MLC_SEL       : 1;
__REG32 DMA_NAND_INT_ENA  : 1;
__REG32 DMA_NAND_RnB_ENA  : 1;
__REG32 INT_SEL           : 1;
__REG32                   :26;
} __flashclk_ctrl_bits;

/* Ethernet MAC Clock Control register */
typedef struct{
__REG32 REG_CLK           : 1;
__REG32 SLAVE_CLK         : 1;
__REG32 MASTER_CLK        : 1;
__REG32 HDW_INF_CTRL      : 2;
__REG32                   :27;
} __macclk_ctrl_bits;

/* LCD Clock Control register */
typedef struct{
__REG32 CLKDIV            : 5;
__REG32 HCLK_ENABLE       : 1;
__REG32 MODE_SELECT       : 2;
__REG32 DISPLAY_TYPE      : 1;
__REG32                   :23;
} __lcdclk_ctrl_bits;

/* I2S Clock Control register */
typedef struct{
__REG32 I2S0_CLK_ENA      : 1;
__REG32 I2S1_CLK_ENA      : 1;
__REG32 I2S0_CLK_RX_MODE  : 1;
__REG32 I2S0_CLK_TX_MODE  : 1;
__REG32 I2S1_DMA1         : 1;
__REG32 I2S1_CLK_RX_MODE  : 1;
__REG32 I2S1_CLK_TX_MODE  : 1;
__REG32                   :25;
} __i2s_ctrl_bits;

/* SSP Control register */
typedef struct{
__REG32 SSP0_CLK_ENA      : 1;
__REG32 SSP1_CLK_ENA      : 1;
__REG32 SSP0_TX_DMA       : 1;
__REG32 SSP0_RX_DMA       : 1;
__REG32 SSP1_TX_DMA       : 1;
__REG32 SSP1_RX_DMA       : 1;
__REG32                   :26;
} __ssp_ctrl_bits;

/* SPI Block Control Register */
typedef struct{
__REG32 SPI1_CLK_ENA  : 1;
__REG32 SPI1_PIN_SEL  : 1;
__REG32 SPI1_CLK_OUT  : 1;
__REG32 SPI1_DATIO    : 1;
__REG32 SPI2_CLK_ENA  : 1;
__REG32 SPI2_PIN_SEL  : 1;
__REG32 SPI2_CLK_OUT  : 1;
__REG32 SPI2_DATIO    : 1;
__REG32               :24;
} __spi_ctrl_bits;

/* I2C Clock Control Register */
typedef struct{
__REG32 I2C1_CLK_ENA      : 1;
__REG32 I2C2_CLK_ENA      : 1;
__REG32 I2C1_DRV_MODE     : 1;
__REG32 I2C2_DRV_MODE     : 1;
__REG32 USB_I2C_DRV_MODE  : 1;
__REG32                   :27;
} __i2cclk_ctrl_bits;

/* Timer & PWM Clock Control1 register */
typedef struct{
__REG32 TIMER4_CLK_EN : 1;
__REG32 TIMER5_CLK_EN : 1;
__REG32 TIMER0_CLK_EN : 1;
__REG32 TIMER1_CLK_EN : 1;
__REG32 TIMER2_CLK_EN : 1;
__REG32 TIMER3_CLK_EN : 1;
__REG32               :26;
} __timclk_ctrl1_bits;

/* Timer Clock Control Regis */
typedef struct{
__REG32 WDT_CLK_ENA  : 1;
__REG32 HST_CLK_ENA  : 1;
__REG32              :30;
} __timclk_ctrl_bits;

/* ADC Clock Control Register */
typedef struct{
__REG32 _32K_CLK_ENA : 1;
__REG32              :31;
} __adclk_ctrl_bits;

/* ADC Clock Control1 register */
typedef struct{
__REG32 ADC_FREQ       : 8;
__REG32 ADCCLK_SEL     : 1;
__REG32                :23;
} __adclk_ctrl1_bits;

/* Keyboard Scan Clock Control Register */
typedef struct{
__REG32 CLK_ENA  : 1;
__REG32          :31;
} __keyclk_ctrl_bits;

/* PWM Clock Control Register */
typedef struct{
__REG32 PWM1_CLK_ENA  : 1;
__REG32 PWM1_CLK_SEL  : 1;
__REG32 PWM2_CLK_ENA  : 1;
__REG32 PWM2_CLK_SEL  : 1;
__REG32 PWM1_FREQ     : 4;
__REG32 PWM2_FREQ     : 4;
__REG32               :20;
} __pwmclk_ctrl_bits;

/* UART Clock Control Register */
typedef struct{
__REG32 UART3_CLK_ENA  : 1;
__REG32 UART4_CLK_ENA  : 1;
__REG32 UART5_CLK_ENA  : 1;
__REG32 UART6_CLK_ENA  : 1;
__REG32                :28;
} __uartclk_ctrl_bits;

/* Interrupt Enable Register for the Main Interrupt Controller */
typedef struct{
__REG32 Sub1IRQn     : 1;
__REG32 Sub2IRQn     : 1;
__REG32              : 1;
__REG32 TIMER4_MCPWM : 1;
__REG32 TIMER5       : 1;
__REG32 HSTIMER_INT  : 1;
__REG32 WATCH_INT    : 1;
__REG32 IIR3         : 1;
__REG32 IIR4         : 1;
__REG32 IIR5         : 1;
__REG32 IIR6         : 1;
__REG32 FLASH_INT    : 1;
__REG32              : 1;
__REG32 SD1_INT      : 1;
__REG32 LCD_INT      : 1;
__REG32 SD0_INT      : 1;
__REG32 TIMER0       : 1;
__REG32 TIMER1       : 1;
__REG32 TIMER2       : 1;
__REG32 TIMER3       : 1;
__REG32 SSP0         : 1;
__REG32 SSP1         : 1;
__REG32 I2S0         : 1;
__REG32 I2S1         : 1;
__REG32 IIR7         : 1;
__REG32 IIR2         : 1;
__REG32 IIR1         : 1;
__REG32 MSTIMER_INT  : 1;
__REG32 DMAINT       : 1;
__REG32 Ethernet     : 1;
__REG32 Sub1FIQn     : 1;
__REG32 Sub2FIQn     : 1;
} __mic_bits;

/* Interrupt Enable Register for Sub Interrupt Controller 1 */
typedef struct{
__REG32                    : 1;
__REG32 JTAG_COMM_TX       : 1;
__REG32 JTAG_COMM_RX       : 1;
__REG32                    : 1;
__REG32 GPI_28             : 1;
__REG32                    : 1;
__REG32 TS_P               : 1;
__REG32 TS_IRQ             : 1;
__REG32 TS_AUX             : 1;
__REG32                    : 3;
__REG32 SPI2_INT           : 1;
__REG32 PLLUSB_INT         : 1;
__REG32 PLLHCLK_INT        : 1;
__REG32                    : 2;
__REG32 PLL397_INT         : 1;
__REG32 I2C_2_INT          : 1;
__REG32 I2C_1_INT          : 1;
__REG32 RTC_INT            : 1;
__REG32                    : 1;
__REG32 KEY_IRQ            : 1;
__REG32 SPI1_INT           : 1;
__REG32 SW_INT             : 1;
__REG32 USB_otg_timer_int  : 1;
__REG32 USB_otg_atx_int_n  : 1;
__REG32 USB_host_int       : 1;
__REG32 USB_dev_dma_in     : 1;
__REG32 USB_dev_lp_int     : 1;
__REG32 USB_dev_hp_int     : 1;
__REG32 USB_i2c_int        : 1;
} __sic1_bits;

/* Interrupt Enable Register for Sub Interrupt Controller 2 */
typedef struct{
__REG32 GPIO_00     : 1;
__REG32 GPIO_01     : 1;
__REG32 GPIO_02     : 1;
__REG32 GPIO_03     : 1;
__REG32 GPIO_04     : 1;
__REG32 GPIO_05     : 1;
__REG32 SPI2_DATIN  : 1;
__REG32 U2_HCTS     : 1;
__REG32 Pn_GPIO     : 1;
__REG32 GPI_08      : 1;
__REG32 GPI_09      : 1;
__REG32 GPI_10      : 1;
__REG32 U7_HCTS     : 1;
__REG32             : 2;
__REG32 GPI_07      : 1;
__REG32             : 2;
__REG32 SDIO_INT_N  : 1;
__REG32 U5_RX       : 1;
__REG32 SPI1_DATIN  : 1;
__REG32             : 1;
__REG32 GPI_00      : 1;
__REG32 GPI_01      : 1;
__REG32 GPI_02      : 1;
__REG32 GPI_03      : 1;
__REG32 GPI_04      : 1;
__REG32 GPI_05      : 1;
__REG32 GPI_06      : 1;
__REG32             : 2;
__REG32 SYSCLK_MUX  : 1;
} __sic2_bits;

/* Software Interrupt Register */
typedef struct{
__REG32 SW_INT  : 1;
__REG32 DATA    : 7;
__REG32         :24;
} __sw_int_bits;

/* DMA Interrupt Status Register
   DMA Interrupt Terminal Count Request Status Register
   DMA Interrupt Terminal Count Request Clear Register
   DMA Interrupt Error Status Register
   DMA Interrupt Error Clear Register
   DMA Raw Interrupt Terminal Count Status Register
   DMA Raw Error Interrupt Status Register
   DMA Enabled Channel Register */
typedef struct{
__REG32 DMA_CH0  : 1;
__REG32 DMA_CH1  : 1;
__REG32 DMA_CH2  : 1;
__REG32 DMA_CH3  : 1;
__REG32 DMA_CH4  : 1;
__REG32 DMA_CH5  : 1;
__REG32 DMA_CH6  : 1;
__REG32 DMA_CH7  : 1;
__REG32          :24;
} __DMACIntStat_bits;

/* DMA Software Burst Request Register
   DMA Software Single Request Register
   DMA Software Last Burst Request Register
   DMA Software Last Single Request Register
   DMA Synchronization Register */
typedef struct{
__REG32 DMA_LINE0   : 1;
__REG32 DMA_LINE1   : 1;
__REG32 DMA_LINE2   : 1;
__REG32 DMA_LINE3   : 1;
__REG32 DMA_LINE4   : 1;
__REG32 DMA_LINE5   : 1;
__REG32 DMA_LINE6   : 1;
__REG32 DMA_LINE7   : 1;
__REG32 DMA_LINE8   : 1;
__REG32 DMA_LINE9   : 1;
__REG32 DMA_LINE10  : 1;
__REG32 DMA_LINE11  : 1;
__REG32 DMA_LINE12  : 1;
__REG32 DMA_LINE13  : 1;
__REG32 DMA_LINE14  : 1;
__REG32 DMA_LINE15  : 1;
__REG32             :16;
} __DMACSoftBReq_bits;

/* DMA Configuration Register */
typedef struct{
__REG32 E   : 1;
__REG32 M0  : 1;
__REG32 M1  : 1;
__REG32     :29;
} __DMACConfig_bits;

/* DMA Channel Linked List Item registers */
typedef struct{
__REG32 LM   : 1;
__REG32      : 1;
__REG32 LLI  :30;
} __dma_lli_bits;

/*DMA channel control registers */
typedef struct{
__REG32 TransferSize  :12;
__REG32 SBSize        : 3;
__REG32 DBSize        : 3;
__REG32 SWidth        : 3;
__REG32 DWidth        : 3;
__REG32 S             : 1;
__REG32 D             : 1;
__REG32 SI            : 1;
__REG32 DI            : 1;
__REG32 Prot1         : 1;
__REG32 Prot2         : 1;
__REG32 Prot3         : 1;
__REG32 I             : 1;
} __dma_ctrl_bits;

/* DMA Channel Configuration registers */
typedef struct{
__REG32 E               : 1;
__REG32 SrcPeripheral   : 5;
__REG32 DestPeripheral  : 5;
__REG32 FlowCntrl       : 3;
__REG32 IE              : 1;
__REG32 ITC             : 1;
__REG32 L               : 1;
__REG32 A               : 1;
__REG32 H               : 1;
__REG32                 :13;
} __dma_cfg_bits;

/* SDRAM Clock Control Register */
typedef struct{
__REG32 CLK_DIS           : 1;
__REG32 DDR_SEL           : 1;
__REG32 DDR_DQSIN_DELAY   : 4;
__REG32 RTC_TICK_EN       : 1;
__REG32 SW_DDR_CAL        : 1;
__REG32 CAL_DELAY         : 1;
__REG32 SENS_FACTOR       : 3;
__REG32 DLY_ADD_STATUS    : 1;
__REG32 HCLKDELAY_DELAY   : 5;
__REG32 SW_DDR_RESET      : 1;
__REG32 SDRAM_PIN_SPEED1  : 1;
__REG32 SDRAM_PIN_SPEED2  : 1;
__REG32 SDRAM_PIN_SPEED3  : 1;
__REG32                   :10;
} __sdramclk_ctrl_bits;

/* SDRAM Controller Control Register */
typedef struct{
__REG32 E  : 1;
__REG32    : 1;
__REG32 L  : 1;
__REG32    :29;
} __emc_ctrl_bits;

/* SDRAM Controller Status Register */
typedef struct{
__REG32 B   : 1;
__REG32     : 1;
__REG32 SA  : 1;
__REG32     :29;
} __emc_status_bits;

/* SDRAM Controller Configuration Register */
typedef struct{
__REG32 N  : 1;
__REG32    :31;
} __emc_cfg_bits;

/* Dynamic Memory Control Register */
typedef struct{
__REG32 CE     : 1;
__REG32 CS     : 1;
__REG32 SR     : 1;
__REG32 SRMCC  : 1;
__REG32 IMCC   : 1;
__REG32 MMC    : 1;
__REG32        : 1;
__REG32 I      : 2;
__REG32        : 4;
__REG32 DP     : 1;
__REG32        :18;
} __emcd_ctrl_bits;

/* Dynamic Memory Refresh Timer Register */
typedef struct{
__REG32 REFRESH  :11;
__REG32          :21;
} __emcd_refresh_bits;

/* Dynamic Memory Read Configuration Register */
typedef struct{
__REG32 SRD  : 2;
__REG32      : 2;
__REG32 SRP  : 1;
__REG32      : 3;
__REG32 DRD  : 2;
__REG32      : 2;
__REG32 DRP  : 1;
__REG32      :19;
} __emcd_read_cfg_bits;

/* Dynamic Memory Precharge Command Period Register */
typedef struct{
__REG32 tRP  : 4;
__REG32      :28;
} __emcd_trp_bits;

/* Dynamic Memory Active to Precharge Command Period Register */
typedef struct{
__REG32 tRAS  : 4;
__REG32       :28;
} __emcd_tras_bits;

/* Dynamic Memory Self-refresh Exit Time Register */
typedef struct{
__REG32 tSREX  : 7;
__REG32        :25;
} __emcd_tsrex_bits;

/* Dynamic Memory Write Recovery Time Register */
typedef struct{
__REG32 tWR  : 4;
__REG32      :28;
} __emcd_twr_bits;

/* Dynamic Memory Active To Active Command Period Register */
typedef struct{
__REG32 tRC  : 5;
__REG32      :27;
} __emcd_trc_bits;

/* Dynamic Memory Auto-refresh Period Register */
typedef struct{
__REG32 tRFC  : 5;
__REG32       :27;
} __emcd_trfc_bits;

/* Dynamic Memory Exit Self-refresh Register */
typedef struct{
__REG32 tXSR  : 8;
__REG32       :24;
} __emcd_txsr_bits;

/* Dynamic Memory Active Bank A to Active Bank B Time Register */
typedef struct{
__REG32 tRRD  : 4;
__REG32       :28;
} __emcd_trrd_bits;

/* Dynamic Memory Load Mode Register To Active Command Time */
typedef struct{
__REG32 tMRD  : 4;
__REG32       :28;
} __emcd_tmrd_bits;

/* Dynamic Memory Last Data In to Read Command Time */
typedef struct{
__REG32 tCDLR  : 4;
__REG32        :28;
} __emcd_tcdlr_bits;

/* Static Memory Extended Wait register */
typedef struct{
__REG32 EXTENDEDWAIT  :10;
__REG32               :22;
} __emcs_extendedwait_bits;

/* Dynamic Memory Configuration Register */
typedef struct{
__REG32 MD  : 3;
__REG32     : 4;
__REG32 AM  : 8;
__REG32     : 5;
__REG32 P   : 1;
__REG32     :11;
} __emcd_cfg_bits;

/* Dynamic Memory RAS & CAS Delay Register */
typedef struct{
__REG32 RAS  : 4;
__REG32      : 3;
__REG32 CAS  : 4;
__REG32      :21;
} __emcd_ras_cas_bits;

/* Static Memory Configuration registers */
typedef struct{
__REG32 MW  : 2;
__REG32     : 1;
__REG32 PM  : 1;
__REG32     : 2;
__REG32 PC  : 1;
__REG32 PB  : 1;
__REG32 EW  : 1;
__REG32     :11;
__REG32 P   : 1;
__REG32     :11;
} __emcs_config_bits;

/* Static Memory Write Enable Delay registers */
typedef struct{
__REG32 WAITWEN   : 4;
__REG32           :28;
} __emcs_waitwen_bits;

/* Static Memory Output Enable Delay registers */
typedef struct{
__REG32 WAITOEN   : 4;
__REG32           :28;
} __emcs_waitoen_bits;

/* SStatic Memory Read Delay registers */
typedef struct{
__REG32 WAITRD    : 5;
__REG32           :27;
} __emcs_waitrd_bits;

/* Static Memory Page Mode Read Delay registers */
typedef struct{
__REG32 WAITPAGE  : 5;
__REG32           :27;
} __emcs_waitpage_bits;

/* Static Memory Write Delay registers */
typedef struct{
__REG32 WAITWR    : 5;
__REG32           :27;
} __emcs_waitwr_bits;

/* Static Memory Turn Round Delay registers */
typedef struct{
__REG32 WAITTURN  : 4;
__REG32           :28;
} __emcs_waitturn_bits;


/* SDRAM Controller AHB Control Registers */
typedef struct{
__REG32 E  : 1;
__REG32    :31;
} __emcahb_ctrl_bits;

/* SDRAM Controller AHB Status Registers */
typedef struct{
__REG32    : 1;
__REG32 S  : 1;
__REG32    :30;
} __emcahb_status_bits;

/* SDRAM Controller AHB Timeout Registers */
typedef struct{
__REG32 AHBTIMEOUT  :10;
__REG32             :22;
} __emcahb_time_bits;

/* DDR Calibration Delay Value */
typedef struct{
__REG32 CAL_DLY  : 5;
__REG32          :27;
} __ddr_cal_dly_bits;

/* MLC NAND ECC Auto Encode Register */
typedef struct{
__REG32 AUTO_PRG_CMD      : 8;
__REG32 AUTO_PRG_CMD_ENA  : 1;
__REG32                   :23;
} __mlc_ecc_auto_enc_reg_bits;

/* MLC NAND Reset Overhead Buffer Pointer Register */
typedef struct{
__REG32 ROBP  : 1;
__REG32       :31;
} __mlc_robp_bits;

/* MLC NAND Software Write Protection Address Low Register */
typedef struct{
__REG32 LOWER  :24;
__REG32        : 8;
} __mlc_sw_wp_add_low_bits;

/* MLC NAND Software Write Protection Address High Register */
typedef struct{
__REG32 UPPER  :24;
__REG32        : 8;
} __mlc_sw_wp_add_hi_bits;

/* MLC NAND Controller Configuration Register */
typedef struct{
__REG32 IO_BUS_16BIT  : 1;
__REG32 ADD_WORD_4    : 1;
__REG32 LARGE_BLOCK   : 1;
__REG32 WR_PROTECT    : 1;
__REG32               :28;
} __mlc_icr_bits;

/* MLC NAND Timing Register */
typedef struct{
__REG32 WR_LOW      : 4;
__REG32 WR_HIGH     : 4;
__REG32 RD_LOW      : 4;
__REG32 RD_HIGH     : 4;
__REG32 NAND_TA     : 4;
__REG32 BUSY_DELAY  : 4;
__REG32 TCEA_DELAY  : 2;
__REG32             : 6;
} __mlc_time_bits;

/* MLC NAND Interrupt Mask Register */
typedef struct{
__REG32 SW_WR_PROT_FAULT  : 1;
__REG32 ECC_READY         : 1;
__REG32 ECC_ERROR         : 1;
__REG32 ECC_FAULT         : 1;
__REG32 CONT_READY        : 1;
__REG32 NAND_READY        : 1;
__REG32                   :26;
} __mlc_irq_bits;

/* MLC NAND Status Register */
typedef struct{
__REG32 NAND_READY         : 1;
__REG32 CONT_READY         : 1;
__REG32 ECC_READY          : 1;
__REG32 ERRORS_DETECTED    : 1;
__REG32 ERROR_SYMB_NUMBER  : 2;
__REG32 DECODER_FAULT      : 1;
__REG32                    :25;
} __mlc_isr_bits;

/* MLC NAND Chip-Enable Host Control Register */
typedef struct{
__REG32 nCE  : 1;
__REG32      :31;
} __mlc_ceh_bits;

/* SLC NAND Flash Data Register */
typedef struct{
__REG32 DATA  : 8;
__REG32       :24;
} __slc_data_bits;

/* SLC NAND Flash Control Register */
typedef struct{
__REG32 DMA_START  : 1;
__REG32 ECC_CLEAR  : 1;
__REG32 SW_RESET   : 1;
__REG32            :29;
} __slc_ctrl_bits;

/* SLC NAND Flash Configuration Register */
typedef struct{
__REG32 WIDTH      : 1;
__REG32 DMA_DIR    : 1;
__REG32 DMA_BURST  : 1;
__REG32 ECC_EN     : 1;
__REG32 DMA_ECC    : 1;
__REG32 CE_LOW     : 1;
__REG32            :26;
} __slc_cfg_bits;

/* SLC NAND Flash Status Register */
typedef struct{
__REG32 READY       : 1;
__REG32 SLC_ACTIVE  : 1;
__REG32 DMA_ACTIVE  : 1;
__REG32             :29;
} __slc_status_bits;

/* SLC NAND Flash Interrupt Status Register */
typedef struct{
__REG32 INT_RDY_STAT  : 1;
__REG32 INT_TC_STAT   : 1;
__REG32               :30;
} __slc_int_stat_bits;

/* SLC NAND Flash Interrupt Enable Register */
typedef struct{
__REG32 INT_RDY_EN  : 1;
__REG32 INT_TC_EN   : 1;
__REG32             :30;
} __slc_int_ena_bits;

/* SLC NAND Flash Interrupt Set Register */
typedef struct{
__REG32 INT_RDY_SET  : 1;
__REG32 INT_TC_SET   : 1;
__REG32              :30;
} __slc_isr_bits;

/* SLC NAND Flash Interrupt Clear Register */
typedef struct{
__REG32 INT_RDY_CLR  : 1;
__REG32 INT_TC_CLR   : 1;
__REG32              :30;
} __slc_icr_bits;

/* SLC NAND Flash Timing Arcs configuration Register */
typedef struct{
__REG32 R_SETUP  : 4;
__REG32 R_HOLD   : 4;
__REG32 R_WIDTH  : 4;
__REG32 R_RDY    : 4;
__REG32 W_SETUP  : 4;
__REG32 W_HOLD   : 4;
__REG32 W_WIDTH  : 4;
__REG32 W_RDY    : 4;
} __slc_tac_bits;

/* SLC NAND Flash Error Correction Code Register */
typedef struct{
__REG32 CP  : 6;
__REG32 LP  :16;
__REG32     :10;
} __slc_ecc_bits;

/* Horizontal Timing register */
typedef struct{
__REG32                 : 2;
__REG32 PPL             : 6;
__REG32 HSW             : 8;
__REG32 HFP             : 8;
__REG32 HBP             : 8;
}__lcd_timh_bits;

/* Vertical Timing register */
typedef struct{
__REG32 LPP             :10;
__REG32 VSW             : 6;
__REG32 VFP             : 8;
__REG32 VBP             : 8;
}__lcd_timv_bits;

/* Clock and Signal Polarity register */
typedef struct{
__REG32 PCD_LO          : 5;
__REG32 CLKSEL          : 1;
__REG32 ACB             : 5;
__REG32 IVS             : 1;
__REG32 IHS             : 1;
__REG32 IPC             : 1;
__REG32 IOE             : 1;
__REG32                 : 1;
__REG32 CPL             :10;
__REG32 BCD             : 1;
__REG32 PCD_HI          : 5;
}__lcd_pol_bits;

/* Line End Control register */
typedef struct{
__REG32 LED             : 7;
__REG32                 : 9;
__REG32 LEE             : 1;
__REG32                 :15;
}__lcd_le_bits;

/* LCD Control register */
typedef struct{
__REG32 LcdEn           : 1;
__REG32 LcdBpp          : 3;
__REG32 LcdBW           : 1;
__REG32 LcdTFT          : 1;
__REG32 LcdMono8        : 1;
__REG32 LcdDual         : 1;
__REG32 BGR             : 1;
__REG32 BEBO            : 1;
__REG32 BEPO            : 1;
__REG32 LcdPwr          : 1;
__REG32 LcdVComp        : 2;
__REG32                 : 2;
__REG32 WATERMARK       : 1;
__REG32                 :15;
}__lcd_ctrl_bits;

/* Interrupt Mask register */
typedef struct{
__REG32                 : 1;
__REG32 FUFIM           : 1;
__REG32 LNBUIM          : 1;
__REG32 VCompIM         : 1;
__REG32 BERIM           : 1;
__REG32                 :27;
}__lcd_intmsk_bits;

/* Raw Interrupt Status register */
typedef struct{
__REG32                 : 1;
__REG32 FUFRIS          : 1;
__REG32 LNBURIS         : 1;
__REG32 VCompRIS        : 1;
__REG32 BERRAW          : 1;
__REG32                 :27;
}__lcd_intraw_bits;

/* Masked Interrupt Status register */
typedef struct{
__REG32                 : 1;
__REG32 FUFMIS          : 1;
__REG32 LNBUMIS         : 1;
__REG32 VCompMIS        : 1;
__REG32 BERMIS          : 1;
__REG32                 :27;
}__lcd_intstat_bits;

/* Interrupt Clear register */
typedef struct{
__REG32                 : 1;
__REG32 FUFIC           : 1;
__REG32 LNBUIC          : 1;
__REG32 VCompIC         : 1;
__REG32 BERIC           : 1;
__REG32                 :27;
}__lcd_intclr_bits;

/* Cursor Control register */
typedef struct{
__REG32 CrsrOn          : 1;
__REG32                 : 3;
__REG32 CrsrNum         : 2;
__REG32                 :26;
}__crsr_ctrl_bits;

/* Cursor Configuration register */
typedef struct{
__REG32 CrsrSize        : 1;
__REG32 FrameSync       : 1;
__REG32                 :30;
}__crsr_cfg_bits;

/* Cursor Palette register 0 */
/* Cursor Palette register 1 */
typedef struct{
__REG32 Red             : 8;
__REG32 Green           : 8;
__REG32 Blue            : 8;
__REG32                 : 8;
}__crsr_pal_bits;

/* Cursor XY Position register */
typedef struct{
__REG32 CrsrX           :10;
__REG32                 : 6;
__REG32 CrsrY           :10;
__REG32                 : 6;
}__crsr_xy_bits;

/* Cursor Clip Position register */
typedef struct{
__REG32 CrsrClipX       : 6;
__REG32                 : 2;
__REG32 CrsrClipY       : 6;
__REG32                 :18;
}__crsr_clip_bits;

/* Cursor Interrupt Mask register */
typedef struct{
__REG32 CrsrIM          : 1;
__REG32                 :31;
}__crsr_intmsk_bits;

/* Cursor Interrupt Clear register */
typedef struct{
__REG32 CrsrIC          : 1;
__REG32                 :31;
}__crsr_intclr_bits;

/* Cursor Raw Interrupt Status register */
typedef struct{
__REG32 CrsrRIS         : 1;
__REG32                 :31;
}__crsr_intraw_bits;

/* Cursor Masked Interrupt Status register */
typedef struct{
__REG32 CrsrMIS         : 1;
__REG32                 :31;
}__crsr_intstat_bits;

/* A/D Status Register */
typedef struct{
__REG32                 : 7;
__REG32 TS_FIFO_EMPTY   : 1;
__REG32 TS_FIFO_OVERRUN : 1;
__REG32                 :23;
} __adc_stat_bits;

/* A/D Status Register */
typedef struct{
__REG32 TS_XPC          : 1;
__REG32 TS_XMC          : 1;
__REG32 TS_YPC          : 1;
__REG32 TS_YMC          : 1;
__REG32 IN_SEL          : 2;
__REG32 P_Ref           : 2;
__REG32 N_Ref           : 2;
__REG32                 :22;
} __adc_select_bits;

/* A/D Control register */
typedef struct{
__REG32 TS_AUTO_EN      : 1;
__REG32 STROBE          : 1;
__REG32 PDN_CTRL        : 1;
__REG32 TS_POS_DET      : 1;
__REG32 TS_Y_ACC        : 3;
__REG32 TS_X_ACC        : 3;
__REG32 TS_AUX_EN       : 1;
__REG32 TS_FIFO_CTRL    : 2;
__REG32                 :19;
} __adc_ctrl_bits;

/* Touchscreen controller sample FIFO register*/
typedef struct{
__REG32 TS_Y_VALUE      :10;
__REG32                 : 6;
__REG32 TS_X_VALUE      :10;
__REG32                 : 3;
__REG32 FIFO_OVERRUN    : 1;
__REG32 FIFO_EMPTY      : 1;
__REG32 TSC_P_LEVEL     : 1;
} __tsc_sample_fifo_bits;

/* Touchscreen controller Delay Time register*/
typedef struct{
__REG32 TSC_DTR         :20;
__REG32                 :12;
} __tsc_dtr_bits;

/* Touchscreen controller Rise Time register*/
typedef struct{
__REG32 TSC_RTR         :20;
__REG32                 :12;
} __tsc_rtr_bits;

/* Touchscreen controller Update Time register*/
typedef struct{
__REG32 TSC_UTR         :20;
__REG32                 :12;
} __tsc_utr_bits;

/* Touchscreen controller Touch Time register */
typedef struct{
__REG32 TSC_TTR         :20;
__REG32                 :12;
} __tsc_ttr_bits;

/* Touchscreen controller Drain X Plate Time Register */
typedef struct{
__REG32 TSC_DXP         :20;
__REG32                 :12;
} __tsc_dxp_bits;

/* Touchscreen controller Minimum X value Register */
typedef struct{
__REG32 TSC_MIN_X       :10;
__REG32                 :22;
} __tsc_min_x_bits;

/* Touchscreen controller Maximum X value Register */
typedef struct{
__REG32 TSC_MAX_X       :10;
__REG32                 :22;
} __tsc_max_x_bits;

/* Touchscreen controller Minimum Y value Register */
typedef struct{
__REG32 TSC_MIN_Y       :10;
__REG32                 :22;
} __tsc_min_y_bits;

/* Touchscreen controller Maximum Y value Register */
typedef struct{
__REG32 TSC_MAX_Y       :10;
__REG32                 :22;
} __tsc_max_y_bits;

/* Touchscreen controller AUX Minimum value Register */
typedef struct{
__REG32 TSC_AUX_MIN     :10;
__REG32                 :22;
} __tsc_aux_min_bits;

/* Touchscreen controller AUX Maximum value Register */
typedef struct{
__REG32 TSC_AUX_MAX     :10;
__REG32                 :22;
} __tsc_aux_max_bits;

/* Touchscreen controller AUX Value Register */
typedef struct{
__REG32 TSC_AUX_VALUE   :10;
__REG32                 :22;
} __tsc_aux_value_bits;

/* Touchscreen controller ADC Value Register */
typedef struct{
__REG32 ADC_VALUE       :10;
__REG32 TSC_P_LEVEL     : 1;
__REG32                 :21;
} __adc_value_bits;

/* Keypad State Machine Current State Register */
typedef struct{
__REG32 STATE  : 2;
__REG32        : 6;
__REG32        :24;
} __ks_state_cond_bits;

/* Keypad State Machine Current State Register */
typedef struct{
__REG32 KIRQN  : 1;
__REG32        :31;
} __ks_irq_bits;

/* Keypad Scan Clock Control Register */
typedef struct{
__REG32 ScanOnce  : 1;
__REG32 Clk32     : 1;
__REG32           :30;
} __ks_fast_tst_bits;

/* Keypad Matrix Dimension Select Register */
typedef struct{
__REG32 MX_DIM  : 4;
__REG32         :28;
} __ks_matrix_dim_bits;

/* MAC Configuration Register 1 */
typedef struct{
__REG32 RE        : 1;
__REG32 PARF      : 1;
__REG32 RXFC      : 1;
__REG32 TXFC      : 1;
__REG32 LB        : 1;
__REG32           : 3;
__REG32 RSTTX     : 1;
__REG32 RSTMCSTX  : 1;
__REG32 RSTRX     : 1;
__REG32 RSTMCSRX  : 1;
__REG32           : 2;
__REG32 SIMRST    : 1;
__REG32 SOFTRST   : 1;
__REG32           :16;
} __mac1_bits;

/* MAC Configuration Register 2 */
typedef struct{
__REG32 FD        : 1;
__REG32 FLC       : 1;
__REG32 HFE       : 1;
__REG32 DLYCRC    : 1;
__REG32 CRCEN     : 1;
__REG32 PADCRCEN  : 1;
__REG32 VLANCRCEN : 1;
__REG32 ADPE      : 1;
__REG32 PPE       : 1;
__REG32 LPE       : 1;
__REG32           : 2;
__REG32 NB        : 1;
__REG32 BP        : 1;
__REG32 ED        : 1;
__REG32           :17;
} __mac2_bits;

/* Back-to-Back Inter-Packet-Gap Register */
typedef struct{
__REG32 IPG       : 7;
__REG32           :25;
} __ipgt_bits;

/* Non Back-to-Back Inter-Packet-Gap Register */
typedef struct{
__REG32 IPGR2     : 7;
__REG32           : 1;
__REG32 IPGR1     : 7;
__REG32           :17;
} __ipgr_bits;

/*Collision Window / Retry Register */
typedef struct{
__REG32 RM        : 4;
__REG32           : 4;
__REG32 CW        : 6;
__REG32           :18;
} __clrt_bits;

/* Maximum Frame Register */
typedef struct{
__REG32 MAXF      :16;
__REG32           :16;
} __maxf_bits;

/* PHY Support Register */
typedef struct{
__REG32             : 8;
__REG32 SPEED       : 1;
__REG32             :23;
} __supp_bits;

/* Test Register */
typedef struct{
__REG32 SPQ         : 1;
__REG32 TP          : 1;
__REG32 TB          : 1;
__REG32             :29;
} __test_bits;

/* MII Mgmt Configuration Register */
typedef struct{
__REG32 SI          : 1;
__REG32 SP          : 1;
__REG32 CS          : 3;
__REG32             :10;
__REG32 RSTMIIMGMT  : 1;
__REG32             :16;
} __mcfg_bits;

/* MII Mgmt Command Register */
typedef struct{
__REG32 READ        : 1;
__REG32 SCAN        : 1;
__REG32             :30;
} __mcmd_bits;

/* MII Mgmt Address Register */
typedef struct{
__REG32 REGADDR     : 5;
__REG32             : 3;
__REG32 PHY_ADDR    : 5;
__REG32             :19;
} __madr_bits;

/* MII Mgmt Write Data Register */
typedef struct{
__REG32 WRITEDATA   :16;
__REG32             :16;
} __mwtd_bits;

/* MII Mgmt Read Data Register */
typedef struct{
__REG32 READDATA    :16;
__REG32             :16;
} __mrdd_bits;

/* MII Mgmt Indicators Register */
typedef struct{
__REG32 BUSY          : 1;
__REG32 SCANNING      : 1;
__REG32 NOT_VALID     : 1;
__REG32 MII_LINK_FAIL : 1;
__REG32               :28;
} __mind_bits;

/* Station Address 0 Register */
typedef struct{
__REG32 STATION_ADDR_2  : 8;
__REG32 STATION_ADDR_1  : 8;
__REG32                 :16;
} __sa0_bits;

/* Station Address 1 Register */
typedef struct{
__REG32 STATION_ADDR_4  : 8;
__REG32 STATION_ADDR_3  : 8;
__REG32                 :16;
} __sa1_bits;

/* Station Address 2 Register */
typedef struct{
__REG32 STATION_ADDR_6  : 8;
__REG32 STATION_ADDR_5  : 8;
__REG32                 :16;
} __sa2_bits;

/* Command Register */
typedef struct{
__REG32 RXENABLE        : 1;
__REG32 TXENABLE        : 1;
__REG32                 : 1;
__REG32 REGRESET        : 1;
__REG32 TXRESET         : 1;
__REG32 RXRESET         : 1;
__REG32 PASSRUNTFRAME   : 1;
__REG32 PASSRXFILTER    : 1;
__REG32 TXFLOWCONTROL   : 1;
__REG32 RMII            : 1;
__REG32 FULLDUPLEX      : 1;
__REG32                 :21;
} __command_bits;

/* Status Register */
typedef struct{
__REG32 RXSTATUS        : 1;
__REG32 TXSTATUS        : 1;
__REG32                 :30;
} __status_bits;

/* Receive Number of Descriptors Register */
typedef struct{
__REG32 RXDESCRIPTORNUMBER  :16;
__REG32                     :16;
} __rxdescrn_bits;

/* Receive Produce Index Register */
typedef struct{
__REG32 RXPRODUCDINDEX  :16;
__REG32                 :16;
} __rxprodind_bits;

/* Receive Consume Index Register */
typedef struct{
__REG32 RXCONSUMEINDEX  :16;
__REG32                 :16;
} __rxcomind_bits;

/* Transmit Number of Descriptors Register */
typedef struct{
__REG32 TXDESCRIPTORNUMBER  :16;
__REG32                     :16;
} __txdescrn_bits;

/* Transmit Produce Index Register */
typedef struct{
__REG32 TXPRODUCDINDEX  :16;
__REG32                 :16;
} __txprodind_bits;

/* Transmit Consume Index Register */
typedef struct{
__REG32 TXCONSUMEINDEX  :16;
__REG32                 :16;
} __txcomind_bits;

/* Transmit Status Vector 0 Register */
typedef struct{
__REG32 CCR_ERR         : 1;
__REG32 LCERR           : 1;
__REG32 LOOR            : 1;
__REG32 DONE            : 1;
__REG32 MULTICAST       : 1;
__REG32 BROADCAST       : 1;
__REG32 PD              : 1;
__REG32 ED              : 1;
__REG32 EC              : 1;
__REG32 LC              : 1;
__REG32 GIANT           : 1;
__REG32 UNDERRUN        : 1;
__REG32 TB              :16;
__REG32 CF              : 1;
__REG32 PAUSE           : 1;
__REG32 BACKPRESSURE    : 1;
__REG32 VLAN            : 1;
} __tsv0_bits;

/* Transmit Status Vector 1 Register */
typedef struct{
__REG32 TBC             :16;
__REG32 TCC             : 4;
__REG32                 :12;
} __tsv1_bits;

/* Receive Status Vector Register */
typedef struct{
__REG32 RBC             :16;
__REG32 PPI             : 1;
__REG32 RXDVEPS         : 1;
__REG32 CEPS            : 1;
__REG32 RCV             : 1;
__REG32 CRC_ERR         : 1;
__REG32 LCE             : 1;
__REG32 LOOR            : 1;
__REG32 R_OK            : 1;
__REG32 MULTICAST       : 1;
__REG32 BROADCAST       : 1;
__REG32 DN              : 1;
__REG32 CF              : 1;
__REG32 PAUSE           : 1;
__REG32 UO              : 1;
__REG32 VLAN            : 1;
__REG32                 : 1;
} __rsv_bits;

/* Flow Control Counter Register */
typedef struct{
__REG32 MC              :16;
__REG32 PT              :16;
} __fwctrlcnt_bits;

/* Flow Control Status Register */
typedef struct{
__REG32 MCC             :16;
__REG32                 :16;
} __fwctrlstat_bits;

/* Receive Filter Control Register */
typedef struct{
__REG32 AUE             : 1;
__REG32 ABE             : 1;
__REG32 AME             : 1;
__REG32 AUHE            : 1;
__REG32 AMHE            : 1;
__REG32 APE             : 1;
__REG32                 : 6;
__REG32 MPEWOL          : 1;
__REG32 RXFEWOL         : 1;
__REG32                 :18;
} __rxflctrl_bits;

/* Receive Filter WoL Status Register */
typedef struct{
__REG32 AUWOL           : 1;
__REG32 ABWOL           : 1;
__REG32 AMWOL           : 1;
__REG32 AUHWOL          : 1;
__REG32 AMHWOL          : 1;
__REG32 APWOL           : 1;
__REG32                 : 1;
__REG32 RXFWOL          : 1;
__REG32 MPWOL           : 1;
__REG32                 :23;
} __rxflwolstat_bits;

/* Receive Filter WoL Clear Register */
typedef struct{
__REG32 AUWOLC          : 1;
__REG32 ABWOLC          : 1;
__REG32 AMWOLC          : 1;
__REG32 AUHWOLC         : 1;
__REG32 AMHWOLC         : 1;
__REG32 APWOLC          : 1;
__REG32                 : 1;
__REG32 RXFWOLC         : 1;
__REG32 MPWOLC          : 1;
__REG32                 :23;
} __rxflwolclr_bits;

/* Interrupt Status Register */
typedef struct{
__REG32 RXOVERRUNINT    : 1;
__REG32 RXERRORINT      : 1;
__REG32 RXFINISHEDINT   : 1;
__REG32 RXDONEINT       : 1;
__REG32 TXUNDERRUNINT   : 1;
__REG32 TXERRORINT      : 1;
__REG32 TXFINISHEDINT   : 1;
__REG32 TXDONEINT       : 1;
__REG32                 : 4;
__REG32 SOFTINT         : 1;
__REG32 WAKEUPINT       : 1;
__REG32                 :18;
} __intstat_bits;

/* Interrupt Enable Register */
typedef struct{
__REG32 RXOVERRUNINTEN  : 1;
__REG32 RXERRORINTEN    : 1;
__REG32 RXFINISHEDINTEN : 1;
__REG32 RXDONEINTEN     : 1;
__REG32 TXUNDERRUNINTEN : 1;
__REG32 TXERRORINTEN    : 1;
__REG32 TXFINISHEDINTEN : 1;
__REG32 TXDONEINTEN     : 1;
__REG32                 : 4;
__REG32 SOFTINTEN       : 1;
__REG32 WAKEUPINTEN     : 1;
__REG32                 :18;
} __intena_bits;

/* Interrupt Clear Register */
typedef struct{
__REG32 RXOVERRUNINTCLR : 1;
__REG32 RXERRORINTCLR   : 1;
__REG32 RXFINISHEDINTCLR: 1;
__REG32 RXDONEINTCLR    : 1;
__REG32 TXUNDERRUNINTCLR: 1;
__REG32 TXERRORINTCLR   : 1;
__REG32 TXFINISHEDINTCLR: 1;
__REG32 TXDONEINTCLR    : 1;
__REG32                 : 4;
__REG32 SOFTINTCLR      : 1;
__REG32 WAKEUPINTCLR    : 1;
__REG32                 :18;
} __intclr_bits;

/* Interrupt Set Register */
typedef struct{
__REG32 RXOVERRUNINTSET : 1;
__REG32 RXERRORINTSET   : 1;
__REG32 RXFINISHEDINTSET: 1;
__REG32 RXDONEINTSET    : 1;
__REG32 TXUNDERRUNINTSET: 1;
__REG32 TXERRORINTSET   : 1;
__REG32 TXFINISHEDINTSET: 1;
__REG32 TXDONEINTSET    : 1;
__REG32                 : 4;
__REG32 SOFTINTSET      : 1;
__REG32 WAKEUPINTSET    : 1;
__REG32                 :18;
} __intset_bits;

/* Power Down Register */
typedef struct{
__REG32                 :31;
__REG32 POWERDOWN       : 1;
} __pwrdn_bits;

/* HcRevision Register */
typedef struct{
__REG32 REV  : 8;
__REG32      :24;
} __HcRevision_bits;

/* HcControl Register */
typedef struct{
__REG32 CBSR  : 2;
__REG32 PLE   : 1;
__REG32 IE    : 1;
__REG32 CLE   : 1;
__REG32 BLE   : 1;
__REG32 HCFS  : 2;
__REG32 IR    : 1;
__REG32 RWC   : 1;
__REG32 RWE   : 1;
__REG32       :21;
} __HcControl_bits;

/* HcCommandStatus Register */
typedef struct{
__REG32 HCR  : 1;
__REG32 CLF  : 1;
__REG32 BLF  : 1;
__REG32 OCR  : 1;
__REG32      :12;
__REG32 SOC  : 2;
__REG32      :14;
} __HcCommandStatus_bits;

/* HcInterruptStatus Register */
typedef struct{
__REG32 SO    : 1;
__REG32 WDH   : 1;
__REG32 SF    : 1;
__REG32 RD    : 1;
__REG32 UE    : 1;
__REG32 FNO   : 1;
__REG32 RHSC  : 1;
__REG32       :23;
__REG32 OC    : 1;
__REG32       : 1;
} __HcInterruptStatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct{
__REG32 SO    : 1;
__REG32 WDH   : 1;
__REG32 SF    : 1;
__REG32 RD    : 1;
__REG32 UE    : 1;
__REG32 FNO   : 1;
__REG32 RHSC  : 1;
__REG32       :23;
__REG32 OC    : 1;
__REG32 MIE   : 1;
} __HcInterruptEnable_bits;

/* HcHCCA Register */
typedef struct{
__REG32       : 8;
__REG32 HCCA  :24;
} __HcHCCA_bits;

/* HcPeriodCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 PCED  :28;
} __HcPeriodCurrentED_bits;

/* HcControlHeadED Registerr */
typedef struct{
__REG32       : 4;
__REG32 CHED  :28;
} __HcControlHeadED_bits;

/* HcControlCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 CCED  :28;
} __HcControlCurrentED_bits;

/* HcBulkHeadED Register */
typedef struct{
__REG32       : 4;
__REG32 BHED  :28;
} __HcBulkHeadED_bits;

/* HcBulkCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 BCED  :28;
} __HcBulkCurrentED_bits;

/* HcDoneHead Register */
typedef struct{
__REG32     : 4;
__REG32 DH  :28;
} __HcDoneHead_bits;

/* HcFmInterval Register */
typedef struct{
__REG32 FI     :14;
__REG32        : 2;
__REG32 FSMPS  :15;
__REG32 FIT    : 1;
} __HcFmInterval_bits;

/* HcFmRemaining Register */
typedef struct{
__REG32 FR   :14;
__REG32      :17;
__REG32 FRT  : 1;
} __HcFmRemaining_bits;

/* HcFmNumber Register */
typedef struct{
__REG32 FN  :16;
__REG32     :16;
} __HcFmNumber_bits;

/* HcPeriodicStart Register */
typedef struct{
__REG32 PS  :14;
__REG32     :18;
} __HcPeriodicStart_bits;

/* HcLSThreshold Register */
typedef struct{
__REG32 LST  :12;
__REG32      :20;
} __HcLSThreshold_bits;

/* HcRhDescriptorA Register */
typedef struct{
__REG32 NDP     : 8;
__REG32 PSM     : 1;     /* ??*/
__REG32 NPS     : 1;     /* ??*/
__REG32 DT      : 1;
__REG32 OCPM    : 1;
__REG32 NOCP    : 1;
__REG32         :11;
__REG32 POTPGT  : 8;
} __HcRhDescriptorA_bits;

/* HcRhDescriptorB Register */
typedef struct{
__REG32 DR    :16;
__REG32 PPCM  :16;
} __HcRhDescriptorB_bits;

/* HcRhStatus Register */
typedef struct{
__REG32 LPS   : 1;
__REG32 OCI   : 1;
__REG32       :13;
__REG32 DRWE  : 1;
__REG32 LPSC  : 1;
__REG32 CCIC  : 1;
__REG32       :13;
__REG32 CRWE  : 1;
} __HcRhStatus_bits;

/* HcRhPortStatus[1:2] Register */
typedef struct{
__REG32 CCS   : 1;
__REG32 PES   : 1;
__REG32 PSS   : 1;
__REG32 POCI  : 1;
__REG32 PRS   : 1;
__REG32       : 3;
__REG32 PPS   : 1;
__REG32 LSDA  : 1;
__REG32       : 6;
__REG32 CSC   : 1;
__REG32 PESC  : 1;
__REG32 PSSC  : 1;
__REG32 OCIC  : 1;
__REG32 PRSC  : 1;
__REG32       :11;
} __HcRhPortStatus_bits;

/* USB - Device Interrupt Status Register */
/* USB - Device Interrupt Enable Register */
/* USB - Device Interrupt Clear Register */
/* USB - Device Interrupt Set Register */
typedef struct{
__REG32 FRAME     : 1;
__REG32 EP_FAST   : 1;
__REG32 EP_SLOW   : 1;
__REG32 DEV_STAT  : 1;
__REG32 CCEMTY    : 1;
__REG32 CDFULL    : 1;
__REG32 RxENDPKT  : 1;
__REG32 TxENDPKT  : 1;
__REG32 EP_RLZED  : 1;
__REG32 ERR_INT   : 1;
__REG32           :22;
} __devints_bits;

/* USB - Device Interrupt Priority Register */
typedef struct{
__REG32 FRAME    : 1;
__REG32 EP_FAST  : 1;
__REG32          :30;
} __devintpri_bits;

/* USB - Endpoint Interrupt Status Register */
/* USB - Endpoint Interrupt Enable Register */
/* USB - Endpoint Interrupt Clear Register */
/* USB - Endpoint Interrupt Set Register */
/* USB - Endpoint Interrupt Priority Register */
typedef struct{
__REG32 EP_0RX   : 1;
__REG32 EP_0TX   : 1;
__REG32 EP_1RX   : 1;
__REG32 EP_1TX   : 1;
__REG32 EP_2RX   : 1;
__REG32 EP_2TX   : 1;
__REG32 EP_3RX   : 1;
__REG32 EP_3TX   : 1;
__REG32 EP_4RX   : 1;
__REG32 EP_4TX   : 1;
__REG32 EP_5RX   : 1;
__REG32 EP_5TX   : 1;
__REG32 EP_6RX   : 1;
__REG32 EP_6TX   : 1;
__REG32 EP_7RX   : 1;
__REG32 EP_7TX   : 1;
__REG32 EP_8RX   : 1;
__REG32 EP_8TX   : 1;
__REG32 EP_9RX   : 1;
__REG32 EP_9TX   : 1;
__REG32 EP_10RX  : 1;
__REG32 EP_10TX  : 1;
__REG32 EP_11RX  : 1;
__REG32 EP_11TX  : 1;
__REG32 EP_12RX  : 1;
__REG32 EP_12TX  : 1;
__REG32 EP_13RX  : 1;
__REG32 EP_13TX  : 1;
__REG32 EP_14RX  : 1;
__REG32 EP_14TX  : 1;
__REG32 EP_15RX  : 1;
__REG32 EP_15TX  : 1;
} __endpints_bits;

/* USB - Realize Enpoint Register */
typedef struct{
__REG32 EP0   : 1;
__REG32 EP1   : 1;
__REG32 EP2   : 1;
__REG32 EP3   : 1;
__REG32 EP4   : 1;
__REG32 EP5   : 1;
__REG32 EP6   : 1;
__REG32 EP7   : 1;
__REG32 EP8   : 1;
__REG32 EP9   : 1;
__REG32 EP10  : 1;
__REG32 EP11  : 1;
__REG32 EP12  : 1;
__REG32 EP13  : 1;
__REG32 EP14  : 1;
__REG32 EP15  : 1;
__REG32 EP16  : 1;
__REG32 EP17  : 1;
__REG32 EP18  : 1;
__REG32 EP19  : 1;
__REG32 EP20  : 1;
__REG32 EP21  : 1;
__REG32 EP22  : 1;
__REG32 EP23  : 1;
__REG32 EP24  : 1;
__REG32 EP25  : 1;
__REG32 EP26  : 1;
__REG32 EP27  : 1;
__REG32 EP28  : 1;
__REG32 EP29  : 1;
__REG32 EP30  : 1;
__REG32 EP31  : 1;
} __realizeendp_bits;

/* USB - Endpoint Index Register */
typedef struct{
__REG32 PHY_ENDP  : 5;
__REG32           :27;
} __endpind_bits;

/* USB - MaxPacketSize Register */
typedef struct{
__REG32 MaxPacketSize  :10;
__REG32                :22;
} __maxpacksize_bits;

/* USB - Receive Packet Length Register */
typedef struct{
__REG32 PKT_LNGTH  :10;
__REG32 DV         : 1;
__REG32 PKT_RDY    : 1;
__REG32            :20;
} __rcvepktlen_bits;

/* USB - Transmit Packet Length Register */
typedef struct{
__REG32 PKT_LNGHT  :10;
__REG32            :22;
} __transmitpktlen_bits;

/* USB - Control Register */
typedef struct{
__REG32 RD_EN         : 1;
__REG32 WR_EN         : 1;
__REG32 LOG_ENDPOINT  : 4;
__REG32               :26;
} __usbctrl_bits;

/* USB - Command Code Register */
typedef struct{
__REG32            : 8;
__REG32 CMD_PHASE  : 8;
__REG32 CMD_CODE   : 8;
__REG32            : 8;
} __cmdcode_bits;

/* USB - Command Data Register */
typedef struct{
__REG32 CMD_DATA  : 8;
__REG32           :24;
} __cmddata_bits;

/* USB - DMA Request Status Register */
/* USB - DMA Request Clear Register */
/* USB - DMA Request Set Regiser */
typedef struct{
__REG32 EP0   : 1;
__REG32 EP1   : 1;
__REG32 EP2   : 1;
__REG32 EP3   : 1;
__REG32 EP4   : 1;
__REG32 EP5   : 1;
__REG32 EP6   : 1;
__REG32 EP7   : 1;
__REG32 EP8   : 1;
__REG32 EP9   : 1;
__REG32 EP10  : 1;
__REG32 EP11  : 1;
__REG32 EP12  : 1;
__REG32 EP13  : 1;
__REG32 EP14  : 1;
__REG32 EP15  : 1;
__REG32 EP16  : 1;
__REG32 EP17  : 1;
__REG32 EP18  : 1;
__REG32 EP19  : 1;
__REG32 EP20  : 1;
__REG32 EP21  : 1;
__REG32 EP22  : 1;
__REG32 EP23  : 1;
__REG32 EP24  : 1;
__REG32 EP25  : 1;
__REG32 EP26  : 1;
__REG32 EP27  : 1;
__REG32 EP28  : 1;
__REG32 EP29  : 1;
__REG32 EP30  : 1;
__REG32 EP31  : 1;
} __dmarqstdiv_bits;

/* USB - UDCA Head Register */
typedef struct{
__REG32              : 7;
__REG32 UDCA_Header  :25;
} __udcahead_bits;

/* USB - EP DMA Status Register */
/* USB - EP DMA Enable Register */
/* USB - EP DMA Disable Register */
typedef struct{
__REG32 EP0   : 1;
__REG32 EP1   : 1;
__REG32 EP2   : 1;
__REG32 EP3   : 1;
__REG32 EP4   : 1;
__REG32 EP5   : 1;
__REG32 EP6   : 1;
__REG32 EP7   : 1;
__REG32 EP8   : 1;
__REG32 EP9   : 1;
__REG32 EP10  : 1;
__REG32 EP11  : 1;
__REG32 EP12  : 1;
__REG32 EP13  : 1;
__REG32 EP14  : 1;
__REG32 EP15  : 1;
__REG32 EP16  : 1;
__REG32 EP17  : 1;
__REG32 EP18  : 1;
__REG32 EP19  : 1;
__REG32 EP20  : 1;
__REG32 EP21  : 1;
__REG32 EP22  : 1;
__REG32 EP23  : 1;
__REG32 EP24  : 1;
__REG32 EP25  : 1;
__REG32 EP26  : 1;
__REG32 EP27  : 1;
__REG32 EP28  : 1;
__REG32 EP29  : 1;
__REG32 EP30  : 1;
__REG32 EP31  : 1;
} __epdmadiv_bits;

/* USB - DMA Interrupt Status Register */
/* USB - DMA Interrupt Enable Register */
typedef struct{
__REG32 End_of_Transfer_Interrupt  : 1;
__REG32 New_DD_Request_Interrupt   : 1;
__REG32 System_Error_Interrupt     : 1;
__REG32                            :29;
} __dmaintstat_bits;

/* USB - New DD Request Interrupt Status Register */
/* USB - New DD Request Interrupt Clear Register */
/* USB - New DD Request Interrupt Set Register */
/* USB - End Of Transfer Interrupt Status Register */
/* USB - End Of Transfer Interrupt Clear Register */
/* USB - End Of Transfer Interrupt Set Register */
/* USB - System Error Interrupt Status Register */
/* USB - System Error Interrupt Clear Register */
/* USB - System Error Interrupt Set Register */
typedef struct{
__REG32 EP0   : 1;
__REG32 EP1   : 1;
__REG32 EP2   : 1;
__REG32 EP3   : 1;
__REG32 EP4   : 1;
__REG32 EP5   : 1;
__REG32 EP6   : 1;
__REG32 EP7   : 1;
__REG32 EP8   : 1;
__REG32 EP9   : 1;
__REG32 EP10  : 1;
__REG32 EP11  : 1;
__REG32 EP12  : 1;
__REG32 EP13  : 1;
__REG32 EP14  : 1;
__REG32 EP15  : 1;
__REG32 EP16  : 1;
__REG32 EP17  : 1;
__REG32 EP18  : 1;
__REG32 EP19  : 1;
__REG32 EP20  : 1;
__REG32 EP21  : 1;
__REG32 EP22  : 1;
__REG32 EP23  : 1;
__REG32 EP24  : 1;
__REG32 EP25  : 1;
__REG32 EP26  : 1;
__REG32 EP27  : 1;
__REG32 EP28  : 1;
__REG32 EP29  : 1;
__REG32 EP30  : 1;
__REG32 EP31  : 1;
} __newdddiv_bits;

/* OTG_int_status Register */
typedef struct{
__REG32 timer_interrupt_status  : 1;
__REG32 remove_pullup           : 1;
__REG32 hnp_failure             : 1;
__REG32 hnp_success             : 1;
__REG32                         :28;
} __otg_int_status_bits;

/* OTG_int_enable Register */
typedef struct{
__REG32 timer_interrupt_en  : 1;
__REG32 remove_pullup_en    : 1;
__REG32 hnp_failure_en      : 1;
__REG32 hnp_success_en      : 1;
__REG32                     :28;
} __otg_int_enable_bits;

/* OTG_int_set Register */
typedef struct{
__REG32 timer_interrupt_set  : 1;
__REG32 remove_pullup_set    : 1;
__REG32 hnp_failure_set      : 1;
__REG32 hnp_success_set      : 1;
__REG32                      :28;
} __otg_int_set_bits;

/* OTG_int_clr Register */
typedef struct{
__REG32 timer_interrupt_clear  : 1;
__REG32 remove_pullup_clear    : 1;
__REG32 hnp_failure_clear      : 1;
__REG32 hnp_success_clear      : 1;
__REG32                        :28;
} __otg_int_clr_bits;

/* OTG_status and control Register */
typedef struct{
__REG32 Host_En             : 1;
__REG32                     : 1;
__REG32 Timer_scale         : 2;
__REG32 Timer_mode          : 1;
__REG32 Timer_enable        : 1;
__REG32 Timer_reset         : 1;
__REG32 Transparent_I2C_en  : 1;
__REG32 b_to_a_hnp_track    : 1;
__REG32 a_to_b_hnp_track    : 1;
__REG32 Pullup_removed      : 1;
__REG32                     : 5;
__REG32 Timer_count         :16;
} __otg_stat_ctrl_bits;

/* OTG_clock Registers */
typedef struct{
__REG32 HOST_CLK_ON  : 1;
__REG32 DEV_CLK_ON   : 1;
__REG32 I2C_CLK_ON   : 1;
__REG32 OTG_CLK_ON   : 1;
__REG32 AHB_CLK_ON   : 1;
__REG32              :27;
} __otg_clock_bits;

/* OTG I2C_TX/I2C_RX Register */
typedef union{
  /*I2C_RX*/
  struct {
__REG32 RX_Data  : 8;
__REG32          :24;
  };
  /*I2C_TX*/
  struct {
__REG32 TX_Data  : 8;
__REG32 START    : 1;
__REG32 STOP     : 1;
__REG32          :22;
  };
} __otg_i2c_rx_tx_bits;

/* OTG I2C_STS Register */
typedef struct{
__REG32 TDI     : 1;
__REG32 AFI     : 1;
__REG32 NAI     : 1;
__REG32 DRMI    : 1;
__REG32 DRSI    : 1;
__REG32 Active  : 1;
__REG32 SCL     : 1;
__REG32 SDA     : 1;
__REG32 RFF     : 1;
__REG32 RFE     : 1;
__REG32 TFF     : 1;
__REG32 TFE     : 1;
__REG32         :20;
} __otg_i2c_sts_bits;

/* OTG I2C_CTL Register */
typedef struct{
__REG32 TDIE    : 1;
__REG32 AFIE    : 1;
__REG32 NAIE    : 1;
__REG32 DRMIE   : 1;
__REG32 DRSIE   : 1;
__REG32 RFFIE   : 1;
__REG32 RFDAIE  : 1;
__REG32 TFFIE   : 1;
__REG32 SRST    : 1;
__REG32         :23;
} __otg_i2c_ctl_bits;

/* Power control register */
typedef struct{
__REG32 Ctrl       : 2;
__REG32            : 4;
__REG32 OpenDrain  : 1;
__REG32            :25;
} __sd_power_bits;

/* Clock control register */
typedef struct{
__REG32 ClkDiv   : 8;
__REG32 Enable   : 1;
__REG32 PwrSave  : 1;
__REG32 Bypass   : 1;
__REG32 WideBus  : 1;
__REG32          :20;
} __sd_clock_bits;

/* Command register */
typedef struct{
__REG32 CmdIndex   : 6;
__REG32 Response   : 1;
__REG32 LongRsp    : 1;
__REG32 Interrupt  : 1;
__REG32 Pending    : 1;
__REG32 Enable     : 1;
__REG32            :21;
} __sd_command_bits;

/* Command response register */
typedef struct{
__REG32 RespCmd  : 6;
__REG32          :26;
} __sd_respcmd_bits;

/* Data control register */
typedef struct{
__REG32 Enable     : 1;
__REG32 Direction  : 1;
__REG32 Mode       : 1;
__REG32 DMAEnable  : 1;
__REG32 BlockSize  : 4;
__REG32            :24;
} __sd_datactrl_bits;

/* Status register */
typedef struct{
__REG32 CmdCrcFail       : 1;
__REG32 DataCrcFail      : 1;
__REG32 CmdTimeOut       : 1;
__REG32 DataTimeOut      : 1;
__REG32 TxUnderrun       : 1;
__REG32 RxOverrun        : 1;
__REG32 CmdRespEnd       : 1;
__REG32 CmdSent          : 1;
__REG32 DataEnd          : 1;
__REG32 StartBitErr      : 1;
__REG32 DataBlockEnd     : 1;
__REG32 CmdActive        : 1;
__REG32 TxActive         : 1;
__REG32 RxActive         : 1;
__REG32 TxFifoHalfEmpty  : 1;
__REG32 RxFifoHalfFull   : 1;
__REG32 TxFifoFull       : 1;
__REG32 RxFifoFull       : 1;
__REG32 TxFifoEmpty      : 1;
__REG32 RxFifoEmpty      : 1;
__REG32 TxDataAvlbl      : 1;
__REG32 RxDataAvlbl      : 1;
__REG32                  :10;
} __sd_status_bits;

/* Clear register */
typedef struct{
__REG32 CmdCrcFailClr    : 1;
__REG32 DataCrcFailClr   : 1;
__REG32 CmdTimeOutClr    : 1;
__REG32 DataTimeOutClr   : 1;
__REG32 TxUnderrunClr    : 1;
__REG32 RxOverrunClr     : 1;
__REG32 CmdRespEndClr    : 1;
__REG32 CmdSentClr       : 1;
__REG32 DataEndClr       : 1;
__REG32 StartBitErrClr   : 1;
__REG32 DataBlockEndClr  : 1;
__REG32                  :21;
} __sd_clear_bits;

/* FIFO counter register */
typedef struct{
__REG32 DataCount  :15;
__REG32            :17;
} __sd_fifocnt_bits;

/* UART interupt enable register */
typedef struct{
__REG32 RDAIE   : 1;
__REG32 THREIE  : 1;
__REG32 RXLSIE  : 1;
__REG32         : 5;
__REG32         :24;
} __uartier_bits;

/* UART interupt identification register and fifo control register */
typedef union{
  /*UxIIR*/
  struct {
__REG32 IP     : 1;
__REG32 IID    : 3;
__REG32        : 2;
__REG32 IIRFE  : 2;
__REG32        :24;
  };
  /*UxFCR*/
  struct {
__REG32 FCRFE  : 1;
__REG32 RFR    : 1;
__REG32 TFR    : 1;
__REG32 FCTRL  : 1;
__REG32 TTLS   : 2;
__REG32 RTLS   : 2;
__REG32        :24;
  };
} __uartfcriir_bits;

/* UART line control register */
typedef struct{
__REG32 WLS   : 2;
__REG32 SBS   : 1;
__REG32 PE    : 1;
__REG32 PS    : 2;
__REG32 BC    : 1;
__REG32 DLAB  : 1;
__REG32       :24;
} __uartlcr_bits;

/* UART line status register */
typedef struct{
__REG32 DR    : 1;
__REG32 OE    : 1;
__REG32 PE    : 1;
__REG32 FE    : 1;
__REG32 BI    : 1;
__REG32 THRE  : 1;
__REG32 TEMT  : 1;
__REG32 RXFE  : 1;
__REG32       :24;
} __uartlsr_bits;

/* UART Rx FIFO Level Register */
typedef struct{
__REG32 RXLEV  : 7;
__REG32        :25;
} __uartrexlev_bits;

/* UART Clock Select Register */
typedef struct{
__REG32 Y        : 8;
__REG32 X        : 8;
__REG32 CLK_SEL  : 1;
__REG32          :15;
} __uartclk_bits;

/* IrDA Clock Control Register */
typedef struct{
__REG32 Y  : 8;
__REG32 X  : 8;
__REG32    :16;
} __irdaclk_bits;

/* UART Control Register */
typedef struct{
__REG32 UART5_MODE    : 1;
__REG32 IR_TxLength   : 1;
__REG32 IR_RxLength   : 1;
__REG32 IRRX6_INV     : 1;
__REG32 IRTX6_INV     : 1;
__REG32 UART6_IRDA    : 1;
__REG32               : 3;
__REG32 HDPX_EN       : 1;
__REG32 HDPX_INV      : 1;
__REG32 UART3_MD_CTRL : 1;
__REG32               :20;
} __uart_ctrl_bits;

/* UART Clock Mode Register */
typedef struct{
__REG32            : 4;
__REG32 UART3_CLK  : 2;
__REG32 UART4_CLK  : 2;
__REG32 UART5_CLK  : 2;
__REG32 UART6_CLK  : 2;
__REG32            : 2;
__REG32 CLK_STAT   : 1;
__REG32            : 1;
__REG32 CLK_STATX  : 7;
__REG32            : 9;
} __uart_clkmode_bits;

/* UART Loopback Control Register */
typedef struct{
__REG32 LOOPBACK1  : 1;
__REG32 LOOPBACK2  : 1;
__REG32 LOOPBACK3  : 1;
__REG32 LOOPBACK4  : 1;
__REG32 LOOPBACK5  : 1;
__REG32 LOOPBACK6  : 1;
__REG32 LOOPBACK7  : 1;
__REG32            :25;
} __uart_loop_bits;

/* High Speed UARTn Receiver/Transmitter FIFO Register */
typedef union{
  /*HSUx_RX*/
  struct {
__REG32 HSU_RX_DATA   : 8;
__REG32 HSU_RX_EMPTY  : 1;
__REG32 HSU_ERROR     : 1;
__REG32 HSU_BREAK     : 1;
__REG32               :21;
  };
  /*HSUx_TX*/
  struct {
__REG32 HSU_TX_DATA   : 8;
__REG32               :24;
  };
} __hsu_rx_tx_bits;

/* High Speed UARTn Level Register */
typedef struct{
__REG32 HSU_RX_LEV  : 8;
__REG32 HSU_TX_LEV  : 8;
__REG32             :16;
} __hsu_level_bits;

/* High Speed UARTn Interrupt Identification Register */
typedef struct{
__REG32 HSU_TX          : 1;
__REG32 HSU_RX_TRIG     : 1;
__REG32 HSU_RX_TIMEOUT  : 1;
__REG32 HSU_FE          : 1;
__REG32 HSU_BRK         : 1;
__REG32 HSU_RX_OE       : 1;
__REG32 HSU_TX_INT_SET  : 1;
__REG32                 :25;
} __hsu_iir_bits;

/* High Speed UARTn Control Register */
typedef struct{
__REG32 HSU_TX_TRIG     : 2;
__REG32 HSU_RX_TRIG     : 3;
__REG32 HSU_TX_INT_EN   : 1;
__REG32 HSU_RX_INT_EN   : 1;
__REG32 HSU_ERR_INT_EN  : 1;
__REG32 HSU_BREAK       : 1;
__REG32 HSU_OFFSET      : 5;
__REG32 HCTS_EN         : 1;
__REG32 HCTS_INV        : 1;
__REG32 TMO_CONFIG      : 2;
__REG32 HRTS_EN         : 1;
__REG32 HRTS_TRIG       : 2;
__REG32 HRTS_INV        : 1;
__REG32                 :10;
} __hsu_ctrl_bits;

/* SPIn Global Control Register */
typedef struct{
__REG32 enable  : 1;
__REG32 rst     : 1;
__REG32         :30;
} __spi_global_bits;

/* SPIn Control Register */
typedef struct{
__REG32 rate       : 7;
__REG32 ms         : 1;
__REG32            : 1;
__REG32 bitnum     : 4;
__REG32 shift_off  : 1;
__REG32 thr        : 1;
__REG32 rxtx       : 1;
__REG32 mode       : 2;
__REG32            : 1;
__REG32 msb        : 1;
__REG32            : 1;
__REG32 bpol       : 1;
__REG32 bhalt      : 1;
__REG32 unidir     : 1;
__REG32            : 8;
} __spi_con_bits;

/* SPIn Interrupt Enable Register */
typedef struct{
__REG32 intthr  : 1;
__REG32 inteot  : 1;
__REG32         :30;
} __spi_ier_bits;

/* SPIn Status Register */
typedef struct{
__REG32 be        : 1;
__REG32 thr       : 1;
__REG32 bf        : 1;
__REG32 shiftact  : 1;
__REG32           : 2;
__REG32 busylev   : 1;
__REG32 eot       : 1;
__REG32 intclr    : 1;
__REG32           :23;
} __spi_stat_bits;

/* SPIn Timer Control Register */
typedef struct{
__REG32 mode   : 1;
__REG32 pirqe  : 1;
__REG32 tirqe  : 1;
__REG32        :29;
} __spi_tim_ctrl_bits;

/* SPIn Timer Status Register */
typedef struct{
__REG32           :15;
__REG32 tirqstat  : 1;
__REG32           :16;
} __spi_tim_stat_bits;

/* SSP Control Register 0 */
typedef struct{
__REG32 DSS  : 4;
__REG32 FRF  : 2;
__REG32 CPOL : 1;
__REG32 CPHA : 1;
__REG32 SCR  : 8;
__REG32      :16;
} __sspcr0_bits;

/* SSP Control Register 1 */
typedef struct{
__REG32 LBM  : 1;
__REG32 SSE  : 1;
__REG32 MS   : 1;
__REG32 SOD  : 1;
__REG32      :28;
} __sspcr1_bits;

/* SSP Data Register */
typedef struct{
__REG32 DATA :16;
__REG32      :16;
} __sspdr_bits;

/* SSP Status Register */
typedef struct{
__REG32 TFE  : 1;
__REG32 TNF  : 1;
__REG32 RNE  : 1;
__REG32 RFF  : 1;
__REG32 BSY  : 1;
__REG32      :27;
} __sspsr_bits;

/* SSP Clock Prescale Register */
typedef struct{
__REG32 CPSDVSR : 8;
__REG32         :24;
} __sspcpsr_bits;

/* SSP Interrupt Mask Set/Clear Register */
typedef struct{
__REG32 RORIM  : 1;
__REG32 RTIM   : 1;
__REG32 RXIM   : 1;
__REG32 TXIM   : 1;
__REG32        :28;
} __sspimsc_bits;

/* SSP Raw Interrupt Status Register */
typedef struct{
__REG32 RORRIS  : 1;
__REG32 RTRIS   : 1;
__REG32 RXRIS   : 1;
__REG32 TXRIS   : 1;
__REG32         :28;
} __sspris_bits;

/* SSP Masked Interrupt Status Register */
typedef struct{
__REG32 RORMIS  : 1;
__REG32 RTMIS   : 1;
__REG32 RXMIS   : 1;
__REG32 TXMIS   : 1;
__REG32         :28;
} __sspmis_bits;

/* SSP Interrupt Clear Register */
typedef struct{
__REG32 RORIC  : 1;
__REG32 RTIC   : 1;
__REG32        :30;
} __sspicr_bits;

/* SSP DMA Control Register */
typedef struct{
__REG32 RXDMAE : 1;
__REG32 TXDMAE : 1;
__REG32        :30;
} __sspdmacr_bits;

/* I2C RX/TX Data FIFO */
typedef union{
  /*I2Cx_RX*/
  struct {
__REG32 RxData  : 8;
__REG32         :24;
  };
  /*I2Cx_TX*/
  struct {
__REG32 TxData  : 8;
__REG32 START   : 1;
__REG32 STOP    : 1;
__REG32         :22;
  };
} __i2c_rx_tx_bits;

/* I2C Status Register */
typedef struct{
__REG32 TDI     : 1;
__REG32 AFI     : 1;
__REG32 NAI     : 1;
__REG32 DRMI    : 1;
__REG32 DRSI    : 1;
__REG32 ACTIVE  : 1;
__REG32 SCL     : 1;
__REG32 SDA     : 1;
__REG32 RFF     : 1;
__REG32 RFE     : 1;
__REG32 TFF     : 1;
__REG32 TFE     : 1;
__REG32 TFFS    : 1;
__REG32 TFES    : 1;
__REG32         :18;
} __i2c_stat_bits;

/* I2C Control Register */
typedef struct{
__REG32 TDIE    : 1;
__REG32 AFIE    : 1;
__REG32 NAIE    : 1;
__REG32 DRMIE   : 1;
__REG32 DRSIE   : 1;
__REG32 DAIE    : 1;
__REG32 RFDAIE  : 1;
__REG32 TFFIE   : 1;
__REG32 RESET   : 1;
__REG32 SEVEN   : 1;
__REG32 TFFSIE  : 1;
__REG32         :21;
} __i2c_ctrl_bits;

/* I2C Clock Divider High */
typedef struct{
__REG32 CLK_DIV_HI  :10;
__REG32             :22;
} __i2c_clk_hi_bits;

/* I2C Clock Divider Low */
typedef struct{
__REG32 CLK_DIV_LO  :10;
__REG32             :22;
} __i2c_clk_lo_bits;

/* I2Cn Slave Address */
typedef struct{
__REG32 ADR         :10;
__REG32             :22;
} __i2c_adr_bits;

/* I2Cn Receive FIFO level */
typedef struct{
__REG32 RxFL        : 2;
__REG32             :30;
} __i2c_rxfl_bits;

/* I2Cn Transmit FIFO level */
typedef struct{
__REG32 TxFL        : 2;
__REG32             :30;
} __i2c_txfl_bits;

/* I2Cn Slave Transmit FIFO */
typedef struct{
__REG32 TXS         : 8;
__REG32             :24;
} __i2c_s_tx_bits;

/* I2Cn Slave Transmit FIFO level */
typedef struct{
__REG32 TxFL        : 2;
__REG32             :30;
} __i2c_s_txfl_bits;

/* I2S Digital Audio Output Registes */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32 MUTE          : 1;
__REG32               :16;
} __i2sdao_bits;

/* I2S Digital Audio Input Register */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32               :17;
} __i2sdai_bits;

/* I2S Status Feedback Register */
typedef struct{
__REG32 IRQ           : 1;
__REG32 DMAREQ1       : 1;
__REG32 DMAREQ2       : 1;
__REG32               : 5;
__REG32 RX_LEVEL      : 4;
__REG32               : 4;
__REG32 TX_LEVEL      : 4;
__REG32               :12;
} __i2sstate_bits;

/* I2S DMA Configuration Register */
typedef struct{
__REG32 RX_DMA_EN     : 1;
__REG32 TX_DMA_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_DMA  : 4;
__REG32               : 4;
__REG32 TX_DEPTH_DMA  : 4;
__REG32               :12;
} __i2sdma_bits;

/* I2S Interrupt Request Control register */
typedef struct{
__REG32 RX_IRQ_EN     : 1;
__REG32 TX_IRQ_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_IRQ  : 8;
__REG32 TX_DEPTH_IRQ  : 8;
__REG32               : 8;
} __i2sirq_bits;

/* I2S Transmit Clock Rate Register */
typedef struct{
__REG32 X             : 8;
__REG32 Y             : 8;
__REG32               :16;
} __i2stxrate_bits;

/* I2S Receive Clock Rate Register */
typedef struct{
__REG32 X             : 8;
__REG32 Y             : 8;
__REG32               :16;
} __i2srxrate_bits;

/* TIMER interrupt register */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32 CR1INT  : 1;
__REG32 CR2INT  : 1;
__REG32 CR3INT  : 1;
__REG32         :24;
} __tmr_ir_bits;

/* TIMER control register */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tmr_cr_bits;

/* TIMER count control register */
typedef struct{
__REG32 CTM : 2;     /*Counter/Timer Mode*/
__REG32 CIS : 2;     /*Count Input Select*/
__REG32     :28;
} __tmr_ctcr_bits;

/* TIMER match control register */
typedef struct{
__REG32 MR0I     : 1;
__REG32 MR0R     : 1;
__REG32 MR0S     : 1;
__REG32 MR1I     : 1;
__REG32 MR1R     : 1;
__REG32 MR1S     : 1;
__REG32 MR2I     : 1;
__REG32 MR2R     : 1;
__REG32 MR2S     : 1;
__REG32 MR3I     : 1;
__REG32 MR3R     : 1;
__REG32 MR3S     : 1;
__REG32          :20;
} __tmr_mcr_bits;

/* TIMER capture control register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1I    : 1;
__REG32 CAP2RE   : 1;
__REG32 CAP2FE   : 1;
__REG32 CAP2I    : 1;
__REG32 CAP3RE   : 1;
__REG32 CAP3FE   : 1;
__REG32 CAP3I    : 1;
__REG32          :20;
} __tmr_ccr_bits;

/* TIMER external match register */
typedef struct{
__REG32 EM0   : 1;
__REG32 EM1   : 1;
__REG32 EM2   : 1;
__REG32 EM3   : 1;
__REG32 EMC0  : 2;
__REG32 EMC1  : 2;
__REG32 EMC2  : 2;
__REG32 EMC3  : 2;
__REG32       :20;
} __tmr_emr_bits;

/* High Speed Timer Interrupt Status Register */
typedef struct{
__REG32 MATCH0_INT  : 1;
__REG32 MATCH1_INT  : 1;
__REG32 MATCH2_INT  : 1;
__REG32             : 1;
__REG32 GPI_06      : 1;
__REG32 RTC_TICK    : 1;
__REG32             :26;
} __hstim_int_bits;

/* High Speed Timer Control Register */
typedef struct{
__REG32 COUNT_ENAB   : 1;
__REG32 RESET_COUNT  : 1;
__REG32 PAUSE_EN     : 1;
__REG32              :29;
} __hstim_ctrl_bits;

/* High Speed Timer Match Control Register */
typedef struct{
__REG32 MR0_INT       : 1;
__REG32 RESET_COUNT0  : 1;
__REG32 STOP_COUNT0   : 1;
__REG32 MR1_INT       : 1;
__REG32 RESET_COUNT1  : 1;
__REG32 STOP_COUNT1   : 1;
__REG32 MR2_INT       : 1;
__REG32 RESET_COUNT2  : 1;
__REG32 STOP_COUNT2   : 1;
__REG32               :23;
} __hstim_mctrl_bits;

/* High Speed Timer Capture Control Register */
typedef struct{
__REG32 RISING_EDGE     : 1;
__REG32 FALLING_EDGE    : 1;
__REG32 HSTIM_INT       : 1;
__REG32 RTC_TICK_RISE   : 1;
__REG32 RTC_TICK_FALL   : 1;
__REG32 RTC_TICK_EVENT  : 1;
__REG32                 :26;
} __hstim_ccr_bits;

/* MilliSecond Timer Interrupt Status Register */
typedef struct{
__REG32 MATCH0_INT  : 1;
__REG32 MATCH1_INT  : 1;
__REG32             :30;
} __mstim_int_bits;

/* MilliSecond Timer Control Register */
typedef struct{
__REG32 COUNT_ENAB   : 1;
__REG32 RESET_COUNT  : 1;
__REG32 PAUSE_EN     : 1;
__REG32              :29;
} __mstim_ctrl_bits;

/* MilliSecond Timer Match Control Register */
typedef struct{
__REG32 MR0_INT       : 1;
__REG32 RESET_COUNT0  : 1;
__REG32 STOP_COUNT0   : 1;
__REG32 MR1_INT       : 1;
__REG32 RESET_COUNT1  : 1;
__REG32 STOP_COUNT1   : 1;
__REG32               :26;
} __mstim_mctrl_bits;

/* RTC Control Register */
typedef struct{
__REG32 MATCH0_INTE  : 1;
__REG32 MATCH1_INTE  : 1;
__REG32 MATCH0_ONSW  : 1;
__REG32 MATCH1_ONSW  : 1;
__REG32 RESET        : 1;
__REG32              : 1;
__REG32 CLK_DIS      : 1;
__REG32 FORCE_ONSW   : 1;
__REG32              : 2;
__REG32 RTC_CLK      : 1;
__REG32              :21;
} __rtc_ctrl_bits;

/* RTC Interrupt Status Register */
typedef struct{
__REG32 MATCH0_INT   : 1;
__REG32 MATCH1_INT   : 1;
__REG32 ONSW_STATUS  : 1;
__REG32              :29;
} __rtc_intstat_bits;

/* Watchdog Timer Interrupt Status Register */
typedef struct{
__REG32 MATCH_INT  : 1;
__REG32            :31;
} __wdtim_int_bits;

/* Watchdog Timer Control Register */
typedef struct{
__REG32 COUNT_ENAB   : 1;
__REG32 RESET_COUNT  : 1;
__REG32 PAUSE_EN     : 1;
__REG32              :29;
} __wdtim_ctrl_bits;

/* Watchdog Timer Match Control Register */
typedef struct{
__REG32 MR0_INT       : 1;
__REG32 RESET_COUNT0  : 1;
__REG32 STOP_COUNT0   : 1;
__REG32 M_RES1        : 1;
__REG32 M_RES2        : 1;
__REG32 RESFRC1       : 1;
__REG32 RESFRC2       : 1;
__REG32               :25;
} __wdtim_mctrl_bits;

/* Watchdog Timer External Match Control Register */
typedef struct{
__REG32 EXT_MATCH0  : 1;
__REG32             : 3;
__REG32 MATCH_CTRL  : 2;
__REG32             :26;
} __wdtim_emr_bits;

/* Watchdog Timer Reset Source Register */
typedef struct{
__REG32 INTERNAL    : 1;
__REG32             :31;
} __wdtim_res_bits;

/* PWM1 Control Register */
typedef struct{
__REG32 PWM1_DUTY       : 8;
__REG32 PWM1_RELOADV    : 8;
__REG32                 :14;
__REG32 PWM1_PIN_LEVEL  : 1;
__REG32 PWM1_EN         : 1;
} __pwm1_ctrl_bits;

/* PWM2 Control Register */
typedef struct{
__REG32 PWM2_DUTY       : 8;
__REG32 PWM2_RELOADV    : 8;
__REG32                 :13;
__REG32 PWM2_INT        : 1;
__REG32 PWM2_PIN_LEVEL  : 1;
__REG32 PWM2_EN         : 1;
} __pwm2_ctrl_bits;

/* P0 Input Pin State register */
/* P0 Output Pin Set register */
/* P0 Output Pin Clear register */
/* P0 Output Pin State Register */
/* P0 Direction Set Register */
/* P0 Direction Clear Register */
/* P0 Direction State Register */
typedef struct{
__REG32 P0_0  : 1;
__REG32 P0_1  : 1;
__REG32 P0_2  : 1;
__REG32 P0_3  : 1;
__REG32 P0_4  : 1;
__REG32 P0_5  : 1;
__REG32 P0_6  : 1;
__REG32 P0_7  : 1;
__REG32       :24;
} __p0_bits;

/* P1 Input Pin State register */
/* P1 Output Pin Set register */
/* P1 Output Pin Clear register */
/* P1 Output Pin State Register */
/* P1 Direction Set Register */
/* P1 Direction Clear Register */
/* P1 Direction State Register */
typedef struct{
__REG32 P1_0  : 1;
__REG32 P1_1  : 1;
__REG32 P1_2  : 1;
__REG32 P1_3  : 1;
__REG32 P1_4  : 1;
__REG32 P1_5  : 1;
__REG32 P1_6  : 1;
__REG32 P1_7  : 1;
__REG32 P1_8  : 1;
__REG32 P1_9  : 1;
__REG32 P1_10 : 1;
__REG32 P1_11 : 1;
__REG32 P1_12 : 1;
__REG32 P1_13 : 1;
__REG32 P1_14 : 1;
__REG32 P1_15 : 1;
__REG32 P1_16 : 1;
__REG32 P1_17 : 1;
__REG32 P1_18 : 1;
__REG32 P1_19 : 1;
__REG32 P1_20 : 1;
__REG32 P1_21 : 1;
__REG32 P1_22 : 1;
__REG32 P1_23 : 1;
__REG32       : 8;
} __p1_bits;

/* P2 Input Pin State register */
/* P2 Output Pin Set register */
/* P2 Output Pin Clear register */
typedef struct{
__REG32 P2_0  : 1;
__REG32 P2_1  : 1;
__REG32 P2_2  : 1;
__REG32 P2_3  : 1;
__REG32 P2_4  : 1;
__REG32 P2_5  : 1;
__REG32 P2_6  : 1;
__REG32 P2_7  : 1;
__REG32 P2_8  : 1;
__REG32 P2_9  : 1;
__REG32 P2_10 : 1;
__REG32 P2_11 : 1;
__REG32 P2_12 : 1;
__REG32       :19;
} __p2_bits;

/* Port2 and Port3 Direction Set Register */
/* P2 Direction Clear Register */
/* P2 Direction State Register */
typedef struct{
__REG32 P2_0    : 1;
__REG32 P2_1    : 1;
__REG32 P2_2    : 1;
__REG32 P2_3    : 1;
__REG32 P2_4    : 1;
__REG32 P2_5    : 1;
__REG32 P2_6    : 1;
__REG32 P2_7    : 1;
__REG32 P2_8    : 1;
__REG32 P2_9    : 1;
__REG32 P2_10   : 1;
__REG32 P2_11   : 1;
__REG32 P2_12   : 1;
__REG32         :12;
__REG32 GPIO_00 : 1;
__REG32 GPIO_01 : 1;
__REG32 GPIO_02 : 1;
__REG32 GPIO_03 : 1;
__REG32 GPIO_04 : 1;
__REG32 GPIO_05 : 1;
__REG32         : 1;
} __p2_p3_bits;

/* P3 Input Pin State Register */
typedef struct{
__REG32 GPI_00  : 1;
__REG32 GPI_01  : 1;
__REG32 GPI_02  : 1;
__REG32 GPI_03  : 1;
__REG32 GPI_04  : 1;
__REG32 GPI_05  : 1;
__REG32 GPI_06  : 1;
__REG32 GPI_07  : 1;
__REG32 GPI_08  : 1;
__REG32 GPI_09  : 1;
__REG32 GPIO_00 : 1;
__REG32 GPIO_01 : 1;
__REG32 GPIO_02 : 1;
__REG32 GPIO_03 : 1;
__REG32 GPIO_04 : 1;
__REG32 GPI_15  : 1;
__REG32 GPI_16  : 1;
__REG32 GPI_17  : 1;
__REG32 GPI_18  : 1;
__REG32 GPI_19  : 1;
__REG32 GPI_20  : 1;
__REG32 GPI_21  : 1;
__REG32 GPI_22  : 1;
__REG32 GPI_23  : 1;
__REG32 GPIO_05 : 1;
__REG32 GPI_25  : 1;
__REG32         : 1;
__REG32 GPI_27  : 1;
__REG32 GPI_28  : 1;
__REG32         : 3;
} __p3_in_bits;

/* P3 Output Pin Set Register */
/* P3 Output Pin Clear Register */
/* P3 Output Pin State Register */
typedef struct{
__REG32 GPO_00  : 1;
__REG32 GPO_01  : 1;
__REG32 GPO_02  : 1;
__REG32 GPO_03  : 1;
__REG32 GPO_04  : 1;
__REG32 GPO_05  : 1;
__REG32 GPO_06  : 1;
__REG32 GPO_07  : 1;
__REG32 GPO_08  : 1;
__REG32 GPO_09  : 1;
__REG32 GPO_10  : 1;
__REG32 GPO_11  : 1;
__REG32 GPO_12  : 1;
__REG32 GPO_13  : 1;
__REG32 GPO_14  : 1;
__REG32 GPO_15  : 1;
__REG32 GPO_16  : 1;
__REG32 GPO_17  : 1;
__REG32 GPO_18  : 1;
__REG32 GPO_19  : 1;
__REG32 GPO_20  : 1;
__REG32 GPO_21  : 1;
__REG32 GPO_22  : 1;
__REG32 GPO_23  : 1;
__REG32         : 1;
__REG32 GPIO_00 : 1;
__REG32 GPIO_01 : 1;
__REG32 GPIO_02 : 1;
__REG32 GPIO_03 : 1;
__REG32 GPIO_04 : 1;
__REG32 GPIO_05 : 1;
__REG32         : 1;
} __p3_out_bits;

/* Peripheral multiplexer Set register */
/* Peripheral multiplexer Clear register */
/* Peripheral multiplexer State register */
typedef struct{
__REG32         : 2;
__REG32 MAT3_1  : 1;
__REG32 MAT3_0  : 1;
__REG32 CAP3_0  : 1;
__REG32 MOSI1   : 1;
__REG32 MISO1   : 1;
__REG32         : 1;
__REG32 SCK1    : 1;
__REG32 MOSI0   : 1;
__REG32 MISO0   : 1;
__REG32         : 1;
__REG32 SCK0    : 1;
__REG32 PWM3_6  : 1;
__REG32 PWM3_5  : 1;
__REG32 MAT1_1  : 1;
__REG32         : 1;
__REG32 PWM3_4  : 1;
__REG32 PWM3_3  : 1;
__REG32 PWM3_2  : 1;
__REG32 PWM3_1  : 1;
__REG32         :11;
} __p_mux_bits;

/* Port 0 multiplexer Set register */
/* Port 0 multiplexer Clear register */
/* Port 0 multiplexer State register */
typedef struct{
__REG32 I2S1RX_CLK  : 1;
__REG32 I2S1RX_WS   : 1;
__REG32 I2S0RX_SDA  : 1;
__REG32 I2S0RX_CLK  : 1;
__REG32 I2S0RX_WS   : 1;
__REG32 I2S0TX_SDA  : 1;
__REG32 I2S0TX_CLK  : 1;
__REG32 I2S0TX_WS   : 1;
__REG32             :24;
} __p0_mux_bits;

/* Port 1 multiplexer Set register */
/* Port 1 multiplexer Clear register */
/* Port 1 multiplexer State register */
typedef struct{
__REG32 P1_0  : 1;
__REG32 P1_1  : 1;
__REG32 P1_2  : 1;
__REG32 P1_3  : 1;
__REG32 P1_4  : 1;
__REG32 P1_5  : 1;
__REG32 P1_6  : 1;
__REG32 P1_7  : 1;
__REG32 P1_8  : 1;
__REG32 P1_9  : 1;
__REG32 P1_10 : 1;
__REG32 P1_11 : 1;
__REG32 P1_12 : 1;
__REG32 P1_13 : 1;
__REG32 P1_14 : 1;
__REG32 P1_15 : 1;
__REG32 P1_16 : 1;
__REG32 P1_17 : 1;
__REG32 P1_18 : 1;
__REG32 P1_19 : 1;
__REG32 P1_20 : 1;
__REG32 P1_21 : 1;
__REG32 P1_22 : 1;
__REG32 P1_23 : 1;
__REG32       : 8;
} __p1_mux_bits;

/* Port 2 multiplexer Set register */
/* Port 2 multiplexer Clear register */
/* Port 2 multiplexer State register */
typedef struct{
__REG32 KEY_ROW6  : 1;
__REG32 KEY_ROW7  : 1;
__REG32 U4_TX     : 1;
__REG32 EMC_D_SEL : 1;
__REG32 SSEL1     : 1;
__REG32 SSEL0     : 1;
__REG32           :26;
} __p2_mux_bits;

/* Port 3 multiplexer Set register */
/* Port 3 multiplexer Clear register */
/* Port 3 multiplexer State register */
typedef struct{
__REG32         : 2;
__REG32 MAT1_0  : 1;
__REG32         : 3;
__REG32 PWM4_3  : 1;
__REG32         : 1;
__REG32 PWM4_2  : 1;
__REG32 PWM4_1  : 1;
__REG32 PWM3_6  : 1;
__REG32         : 1;
__REG32 PWM3_5  : 1;
__REG32 PWM3_4  : 1;
__REG32         : 1;
__REG32 PWM3_3  : 1;
__REG32 PWM3_2  : 1;
__REG32         : 1;
__REG32 PWM3_1  : 1;
__REG32         :13;
} __p3_mux_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** System control block
 **
 ***************************************************************************/
__IO_REG32_BIT(BOOT_MAP,              0x40004014,__READ_WRITE,__boot_map_bits);
__IO_REG32_BIT(PWR_CTRL,              0x40004044,__READ_WRITE,__pwr_ctrl_bits);
__IO_REG32_BIT(OSC_CTRL,              0x4000404C,__READ_WRITE,__osc_ctrl_bits);
__IO_REG32_BIT(SYSCLK_CTRL,           0x40004050,__READ_WRITE,__sysclk_ctrl_bits);
__IO_REG32_BIT(PLL397_CTRL,           0x40004048,__READ_WRITE,__pll397_ctrl_bits);
__IO_REG32_BIT(HCLKPLL_CTRL,          0x40004058,__READ_WRITE,__hclkpll_ctrl_bits);
__IO_REG32_BIT(HCLKDIV_CTRL,          0x40004040,__READ_WRITE,__hclkdiv_ctrl_bits);
__IO_REG32_BIT(TEST_CLK,              0x400040A4,__READ_WRITE,__test_clk_bits);
__IO_REG32_BIT(AUTOCLK_CTRL,          0x400040EC,__READ_WRITE,__autoclk_ctrl_bits);
__IO_REG32_BIT(START_ER_PIN,          0x40004030,__READ_WRITE,__start_er_pin_bits);
__IO_REG32_BIT(START_ER_INT,          0x40004020,__READ_WRITE,__start_er_int_bits);
__IO_REG32_BIT(P0_INTR_ER,            0x40004018,__READ_WRITE,__p0_intr_er_bits);
__IO_REG32_BIT(START_SR_PIN,          0x40004038,__READ      ,__start_er_pin_bits);
__IO_REG32_BIT(START_SR_INT,          0x40004028,__READ      ,__start_er_int_bits);
__IO_REG32_BIT(START_RSR_PIN,         0x40004034,__READ_WRITE,__start_er_pin_bits);
__IO_REG32_BIT(START_RSR_INT,         0x40004024,__READ_WRITE,__start_er_int_bits);
__IO_REG32_BIT(START_APR_PIN,         0x4000403C,__READ_WRITE,__start_er_pin_bits);
__IO_REG32_BIT(START_APR_INT,         0x4000402C,__READ_WRITE,__start_er_int_bits);
__IO_REG32_BIT(USB_CTRL,              0x40004064,__READ_WRITE,__usb_ctrl_bits);
__IO_REG32_BIT(USBDIV_CTRL,           0x4000401C,__READ_WRITE,__usbdiv_ctrl_bits);
__IO_REG32_BIT(MS_CTRL,               0x40004080,__READ_WRITE,__ms_ctrl_bits);
__IO_REG32_BIT(DMACLK_CTRL,           0x400040E8,__READ_WRITE,__dmaclk_ctrl_bits);
__IO_REG32_BIT(FLASHCLK_CTRL,         0x400040C8,__READ_WRITE,__flashclk_ctrl_bits);
__IO_REG32_BIT(MACCLK_CTRL,           0x40004090,__READ_WRITE,__macclk_ctrl_bits);
__IO_REG32_BIT(LCDCLK_CTRL,           0x40004054,__READ_WRITE,__lcdclk_ctrl_bits);
__IO_REG32_BIT(I2S_CTRL,              0x4000407C,__READ_WRITE,__i2s_ctrl_bits);
__IO_REG32_BIT(SSP_CTRL,              0x40004078,__READ_WRITE,__ssp_ctrl_bits);
__IO_REG32_BIT(SPI_CTRL,              0x400040C4,__READ_WRITE,__spi_ctrl_bits);
__IO_REG32_BIT(I2CCLK_CTRL,           0x400040AC,__READ_WRITE,__i2cclk_ctrl_bits);
__IO_REG32_BIT(TIMCLK_CTRL1,          0x400040C0,__READ_WRITE,__timclk_ctrl1_bits);
__IO_REG32_BIT(TIMCLK_CTRL,           0x400040BC,__READ_WRITE,__timclk_ctrl_bits);
__IO_REG32_BIT(ADCLK_CTRL,            0x400040B4,__READ_WRITE,__adclk_ctrl_bits);
__IO_REG32_BIT(ADCLK_CTRL1,           0x40004060,__READ_WRITE,__adclk_ctrl1_bits);
__IO_REG32_BIT(KEYCLK_CTRL,           0x400040B0,__READ_WRITE,__keyclk_ctrl_bits);
__IO_REG32_BIT(PWMCLK_CTRL,           0x400040B8,__READ_WRITE,__pwmclk_ctrl_bits);
__IO_REG32_BIT(UARTCLK_CTRL,          0x400040E4,__READ_WRITE,__uartclk_ctrl_bits);

/***************************************************************************
 **
 ** Interrupt Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(MIC_ER,                0x40008000,__READ_WRITE,__mic_bits);
__IO_REG32_BIT(MIC_RSR,               0x40008004,__READ_WRITE,__mic_bits);
__IO_REG32_BIT(MIC_SR,                0x40008008,__READ      ,__mic_bits);
__IO_REG32_BIT(MIC_APR,               0x4000800C,__READ_WRITE,__mic_bits);
__IO_REG32_BIT(MIC_ATR,               0x40008010,__READ_WRITE,__mic_bits);
__IO_REG32_BIT(MIC_ITR,               0x40008014,__READ_WRITE,__mic_bits);
__IO_REG32_BIT(SIC1_ER,               0x4000C000,__READ_WRITE,__sic1_bits);
__IO_REG32_BIT(SIC1_RSR,              0x4000C004,__READ_WRITE,__sic1_bits);
__IO_REG32_BIT(SIC1_SR,               0x4000C008,__READ      ,__sic1_bits);
__IO_REG32_BIT(SIC1_APR,              0x4000C00C,__READ_WRITE,__sic1_bits);
__IO_REG32_BIT(SIC1_ATR,              0x4000C010,__READ_WRITE,__sic1_bits);
__IO_REG32_BIT(SIC1_ITR,              0x4000C014,__READ_WRITE,__sic1_bits);
__IO_REG32_BIT(SIC2_ER,               0x40010000,__READ_WRITE,__sic2_bits);
__IO_REG32_BIT(SIC2_RSR,              0x40010004,__READ_WRITE,__sic2_bits);
__IO_REG32_BIT(SIC2_SR,               0x40010008,__READ      ,__sic2_bits);
__IO_REG32_BIT(SIC2_APR,              0x4001000C,__READ_WRITE,__sic2_bits);
__IO_REG32_BIT(SIC2_ATR,              0x40010010,__READ_WRITE,__sic2_bits);
__IO_REG32_BIT(SIC2_ITR,              0x40010014,__READ_WRITE,__sic2_bits);
__IO_REG32_BIT(SW_INT,                0x400040A8,__READ_WRITE,__sw_int_bits);

/***************************************************************************
 **
 ** GPDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStat,           0x31000000,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACIntTCStat,         0x31000004,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACIntTCClear,        0x31000008,__WRITE     ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACIntErrStat,        0x3100000C,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACIntErrClr,         0x31000010,__WRITE     ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACRawIntTCStat,      0x31000014,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACRawIntErrStat,     0x31000018,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACEnbldChns,         0x3100001C,__READ      ,__DMACIntStat_bits);
__IO_REG32_BIT(DMACSoftBReq,          0x31000020,__READ_WRITE,__DMACSoftBReq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0x31000024,__READ_WRITE,__DMACSoftBReq_bits);
__IO_REG32_BIT(DMACSoftLBReq,         0x31000028,__READ_WRITE,__DMACSoftBReq_bits);
__IO_REG32_BIT(DMACSoftLSReq,         0x3100002C,__READ_WRITE,__DMACSoftBReq_bits);
__IO_REG32_BIT(DMACConfig,            0x31000030,__READ_WRITE,__DMACConfig_bits);
__IO_REG32_BIT(DMACSync,              0x31000034,__READ_WRITE,__DMACSoftBReq_bits);
__IO_REG32(    DMACC0SrcAddr,         0x31000100,__READ_WRITE);
__IO_REG32(    DMACC0DestAddr,        0x31000104,__READ_WRITE);
__IO_REG32_BIT(DMACC0LLI,             0x31000108,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC0Control,         0x3100010C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC0Config,          0x31000110,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC1SrcAddr,         0x31000120,__READ_WRITE);
__IO_REG32(    DMACC1DestAddr,        0x31000124,__READ_WRITE);
__IO_REG32_BIT(DMACC1LLI,             0x31000128,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC1Control,         0x3100012C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC1Config,          0x31000130,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC2SrcAddr,         0x31000140,__READ_WRITE);
__IO_REG32(    DMACC2DestAddr,        0x31000144,__READ_WRITE);
__IO_REG32_BIT(DMACC2LLI,             0x31000148,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC2Control,         0x3100014C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC2Config,          0x31000150,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC3SrcAddr,         0x31000160,__READ_WRITE);
__IO_REG32(    DMACC3DestAddr,        0x31000164,__READ_WRITE);
__IO_REG32_BIT(DMACC3LLI,             0x31000168,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC3Control,         0x3100016C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC3Config,          0x31000170,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC4SrcAddr,         0x31000180,__READ_WRITE);
__IO_REG32(    DMACC4DestAddr,        0x31000184,__READ_WRITE);
__IO_REG32_BIT(DMACC4LLI,             0x31000188,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC4Control,         0x3100018C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC4Config,          0x31000190,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC5SrcAddr,         0x310001A0,__READ_WRITE);
__IO_REG32(    DMACC5DestAddr,        0x310001A4,__READ_WRITE);
__IO_REG32_BIT(DMACC5LLI,             0x310001A8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC5Control,         0x310001AC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC5Config,          0x310001B0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC6SrcAddr,         0x310001C0,__READ_WRITE);
__IO_REG32(    DMACC6DestAddr,        0x310001C4,__READ_WRITE);
__IO_REG32_BIT(DMACC6LLI,             0x310001C8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC6Control,         0x310001CC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC6Config,          0x310001D0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC7SrcAddr,         0x310001E0,__READ_WRITE);
__IO_REG32(    DMACC7DestAddr,        0x310001E4,__READ_WRITE);
__IO_REG32_BIT(DMACC7LLI,             0x310001E8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC7Control,         0x310001EC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC7Config,          0x310001F0,__READ_WRITE,__dma_cfg_bits);

/***************************************************************************
 **
 ** EMC
 **
 ***************************************************************************/
__IO_REG32_BIT(SDRAMCLK_CTRL,         0x40004068,__READ_WRITE,__sdramclk_ctrl_bits);
__IO_REG32_BIT(EMCControl,            0x31080000,__READ_WRITE,__emc_ctrl_bits);
__IO_REG32_BIT(EMCStatus,             0x31080004,__READ      ,__emc_status_bits);
__IO_REG32_BIT(EMCConfig,             0x31080008,__READ_WRITE,__emc_cfg_bits);
__IO_REG32_BIT(EMCDynamicControl,     0x31080020,__READ_WRITE,__emcd_ctrl_bits);
__IO_REG32_BIT(EMCDynamicRefresh,     0x31080024,__READ_WRITE,__emcd_refresh_bits);
__IO_REG32_BIT(EMCDynamicReadConfig,  0x31080028,__READ_WRITE,__emcd_read_cfg_bits);
__IO_REG32_BIT(EMCDynamictRP,         0x31080030,__READ_WRITE,__emcd_trp_bits);
__IO_REG32_BIT(EMCDynamictRAS,        0x31080034,__READ_WRITE,__emcd_tras_bits);
__IO_REG32_BIT(EMCDynamictSREX,       0x31080038,__READ_WRITE,__emcd_tsrex_bits);
__IO_REG32_BIT(EMCDynamictWR,         0x31080044,__READ_WRITE,__emcd_twr_bits);
__IO_REG32_BIT(EMCDynamictRC,         0x31080048,__READ_WRITE,__emcd_trc_bits);
__IO_REG32_BIT(EMCDynamictRFC,        0x3108004C,__READ_WRITE,__emcd_trfc_bits);
__IO_REG32_BIT(EMCDynamictXSR,        0x31080050,__READ_WRITE,__emcd_txsr_bits);
__IO_REG32_BIT(EMCDynamictRRD,        0x31080054,__READ_WRITE,__emcd_trrd_bits);
__IO_REG32_BIT(EMCDynamictMRD,        0x31080058,__READ_WRITE,__emcd_tmrd_bits);
__IO_REG32_BIT(EMCDynamictCDLR,       0x3108005C,__READ_WRITE,__emcd_tcdlr_bits);
__IO_REG32_BIT(EMCStaticExtendedWait, 0x31080080,__READ_WRITE,__emcs_extendedwait_bits);
__IO_REG32_BIT(EMCDynamicConfig0,     0x31080100,__READ_WRITE,__emcd_cfg_bits);
__IO_REG32_BIT(EMCDynamicRasCas0,     0x31080104,__READ_WRITE,__emcd_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig1,     0x31080120,__READ_WRITE,__emcd_cfg_bits);
__IO_REG32_BIT(EMCDynamicRasCas1,     0x31080124,__READ_WRITE,__emcd_ras_cas_bits);
__IO_REG32_BIT(EMCStaticConfig0,      0x31080200,__READ_WRITE,__emcs_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen0,     0x31080204,__READ_WRITE,__emcs_waitwen_bits);
__IO_REG32_BIT(EMCStaticWaitOen0,     0x31080208,__READ_WRITE,__emcs_waitoen_bits);
__IO_REG32_BIT(EMCStaticWaitRd0,      0x3108020C,__READ_WRITE,__emcs_waitrd_bits);
__IO_REG32_BIT(EMCStaticWaitPage0,    0x31080210,__READ_WRITE,__emcs_waitpage_bits);
__IO_REG32_BIT(EMCStaticWaitWr0,      0x31080214,__READ_WRITE,__emcs_waitwr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn0,    0x31080218,__READ_WRITE,__emcs_waitturn_bits);
__IO_REG32_BIT(EMCStaticConfig1,      0x31080220,__READ_WRITE,__emcs_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen1,     0x31080224,__READ_WRITE,__emcs_waitwen_bits);
__IO_REG32_BIT(EMCStaticWaitOen1,     0x31080228,__READ_WRITE,__emcs_waitoen_bits);
__IO_REG32_BIT(EMCStaticWaitRd1,      0x3108022C,__READ_WRITE,__emcs_waitrd_bits);
__IO_REG32_BIT(EMCStaticWaitPage1,    0x31080230,__READ_WRITE,__emcs_waitpage_bits);
__IO_REG32_BIT(EMCStaticWaitWr1,      0x31080234,__READ_WRITE,__emcs_waitwr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn1,    0x31080238,__READ_WRITE,__emcs_waitturn_bits);
__IO_REG32_BIT(EMCStaticConfig2,      0x31080240,__READ_WRITE,__emcs_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen2,     0x31080244,__READ_WRITE,__emcs_waitwen_bits);
__IO_REG32_BIT(EMCStaticWaitOen2,     0x31080248,__READ_WRITE,__emcs_waitoen_bits);
__IO_REG32_BIT(EMCStaticWaitRd2,      0x3108024C,__READ_WRITE,__emcs_waitrd_bits);
__IO_REG32_BIT(EMCStaticWaitPage2,    0x31080250,__READ_WRITE,__emcs_waitpage_bits);
__IO_REG32_BIT(EMCStaticWaitWr2,      0x31080254,__READ_WRITE,__emcs_waitwr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn2,    0x31080258,__READ_WRITE,__emcs_waitturn_bits);
__IO_REG32_BIT(EMCStaticConfig3,      0x31080260,__READ_WRITE,__emcs_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen3,     0x31080264,__READ_WRITE,__emcs_waitwen_bits);
__IO_REG32_BIT(EMCStaticWaitOen3,     0x31080268,__READ_WRITE,__emcs_waitoen_bits);
__IO_REG32_BIT(EMCStaticWaitRd3,      0x3108026C,__READ_WRITE,__emcs_waitrd_bits);
__IO_REG32_BIT(EMCStaticWaitPage3,    0x31080270,__READ_WRITE,__emcs_waitpage_bits);
__IO_REG32_BIT(EMCStaticWaitWr3,      0x31080274,__READ_WRITE,__emcs_waitwr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn3,    0x31080278,__READ_WRITE,__emcs_waitturn_bits);
__IO_REG32_BIT(EMCAHBControl0,        0x31080400,__READ_WRITE,__emcahb_ctrl_bits);
__IO_REG32_BIT(EMCAHBStatus0,         0x31080404,__READ      ,__emcahb_status_bits);
__IO_REG32_BIT(EMCAHBTimeOut0,        0x31080408,__READ_WRITE,__emcahb_time_bits);
__IO_REG32_BIT(EMCAHBControl2,        0x31080440,__READ_WRITE,__emcahb_ctrl_bits);
__IO_REG32_BIT(EMCAHBStatus2,         0x31080444,__READ      ,__emcahb_status_bits);
__IO_REG32_BIT(EMCAHBTimeOut2,        0x31080448,__READ_WRITE,__emcahb_time_bits);
__IO_REG32_BIT(EMCAHBControl3,        0x31080460,__READ_WRITE,__emcahb_ctrl_bits);
__IO_REG32_BIT(EMCAHBStatus3,         0x31080464,__READ      ,__emcahb_status_bits);
__IO_REG32_BIT(EMCAHBTimeOut3,        0x31080468,__READ_WRITE,__emcahb_time_bits);
__IO_REG32_BIT(EMCAHBControl4,        0x31080480,__READ_WRITE,__emcahb_ctrl_bits);
__IO_REG32_BIT(EMCAHBStatus4,         0x31080484,__READ      ,__emcahb_status_bits);
__IO_REG32_BIT(EMCAHBTimeOut4,        0x31080488,__READ_WRITE,__emcahb_time_bits);
__IO_REG32(    DDR_LAP_NOM,           0x4000406C,__READ_WRITE);
__IO_REG32(    DDR_LAP_COUNT,         0x40004070,__READ      );
__IO_REG32_BIT(DDR_CAL_DELAY,         0x40004074,__READ      ,__ddr_cal_dly_bits);

/***************************************************************************
 **
 ** Multi-level NAND Controller
 **
 ***************************************************************************/
__IO_REG8(     MLC_CMD,               0x200B8000,__WRITE     );
__IO_REG8(     MLC_ADDR,              0x200B8004,__WRITE     );
__IO_REG8(     MLC_ECC_ENC_REG,       0x200B8008,__WRITE     );
__IO_REG8(     MLC_ECC_DEC_REG,       0x200B800C,__WRITE     );
__IO_REG32_BIT(MLC_ECC_AUTO_ENC_REG,  0x200B8010,__WRITE     ,__mlc_ecc_auto_enc_reg_bits);
__IO_REG8(     MLC_ECC_AUTO_DEC_REG,  0x200B8014,__WRITE     );
__IO_REG8(     MLC_RPR,               0x200B8018,__WRITE     );
__IO_REG8(     MLC_WPR,               0x200B801C,__WRITE     );
__IO_REG8(     MLC_RUBP,              0x200B8020,__WRITE     );
__IO_REG32_BIT(MLC_ROBP,              0x200B8024,__WRITE     ,__mlc_robp_bits);
__IO_REG32_BIT(MLC_SW_WP_ADD_LOW,     0x200B8028,__WRITE     ,__mlc_sw_wp_add_low_bits);
__IO_REG32_BIT(MLC_SW_WP_ADD_HIG,     0x200B802C,__WRITE     ,__mlc_sw_wp_add_hi_bits);
__IO_REG32_BIT(MLC_ICR,               0x200B8030,__WRITE     ,__mlc_icr_bits);
__IO_REG32_BIT(MLC_TIME_REG,          0x200B8034,__WRITE     ,__mlc_time_bits);
__IO_REG32_BIT(MLC_IRQ_MR,            0x200B8038,__WRITE     ,__mlc_irq_bits);
__IO_REG32_BIT(MLC_IRQ_SR,            0x200B803C,__READ      ,__mlc_irq_bits);
__IO_REG16(    MLC_LOCK_PR,           0x200B8044,__WRITE     );
__IO_REG32_BIT(MLC_ISR,               0x200B8048,__READ      ,__mlc_isr_bits);
__IO_REG32_BIT(MLC_CEH,               0x200B804C,__WRITE     ,__mlc_ceh_bits);

/***************************************************************************
 **
 ** Single-level NAND Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(SLC_DATA,              0x20020000,__READ_WRITE,__slc_data_bits);
__IO_REG32(    SLC_ADDR,              0x20020004,__WRITE     );
__IO_REG32(    SLC_CMD,               0x20020008,__WRITE     );
__IO_REG32(    SLC_STOP,              0x2002000C,__WRITE     );
__IO_REG32_BIT(SLC_CTRL,              0x20020010,__READ_WRITE,__slc_ctrl_bits);
__IO_REG32_BIT(SLC_CFG,               0x20020014,__READ_WRITE,__slc_cfg_bits);
__IO_REG32_BIT(SLC_STAT,              0x20020018,__READ      ,__slc_status_bits);
__IO_REG32_BIT(SLC_INT_STAT,          0x2002001C,__READ      ,__slc_int_stat_bits);
__IO_REG32_BIT(SLC_IEN,               0x20020020,__READ_WRITE,__slc_int_ena_bits);
__IO_REG32_BIT(SLC_ISR,               0x20020024,__WRITE     ,__slc_isr_bits);
__IO_REG32_BIT(SLC_ICR,               0x20020028,__WRITE     ,__slc_icr_bits);
__IO_REG32_BIT(SLC_TAC,               0x2002002C,__READ_WRITE,__slc_tac_bits);
__IO_REG32(    SLC_TC,                0x20020030,__READ_WRITE);
__IO_REG32_BIT(SLC_ECC,               0x20020034,__READ      ,__slc_ecc_bits);
__IO_REG32(    SLC_DMA_DATA,          0x20020038,__READ_WRITE);

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_TIMH,              0x31040000,__READ_WRITE ,__lcd_timh_bits);
__IO_REG32_BIT(LCD_TIMV,              0x31040004,__READ_WRITE ,__lcd_timv_bits);
__IO_REG32_BIT(LCD_POL,               0x31040008,__READ_WRITE ,__lcd_pol_bits);
__IO_REG32_BIT(LCD_LE,                0x3104000C,__READ_WRITE ,__lcd_le_bits);
__IO_REG32(    LCD_UPBASE,            0x31040010,__READ_WRITE );
__IO_REG32(    LCD_LPBASE,            0x31040014,__READ_WRITE );
__IO_REG32_BIT(LCD_CTRL,              0x31040018,__READ_WRITE ,__lcd_ctrl_bits);
__IO_REG32_BIT(LCD_INTMSK,            0x3104001C,__READ_WRITE ,__lcd_intmsk_bits);
__IO_REG32_BIT(LCD_INTRAW,            0x31040020,__READ       ,__lcd_intraw_bits);
__IO_REG32_BIT(LCD_INTSTAT,           0x31040024,__READ       ,__lcd_intstat_bits);
__IO_REG32_BIT(LCD_INTCLR,            0x31040028,__WRITE      ,__lcd_intclr_bits);
__IO_REG32(    LCD_UPCURR,            0x3104002C,__READ       );
__IO_REG32(    LCD_LPCURR,            0x31040030,__READ       );
__IO_REG32(    LCD_PAL_BASE,          0x31040200,__READ_WRITE );
__IO_REG32(    CRSR_IMG_BASE,         0x31040800,__READ_WRITE );
__IO_REG32_BIT(CRSR_CTRL,             0x31040C00,__READ_WRITE ,__crsr_ctrl_bits);
__IO_REG32_BIT(CRSR_CFG,              0x31040C04,__READ_WRITE ,__crsr_cfg_bits);
__IO_REG32_BIT(CRSR_PAL0,             0x31040C08,__READ_WRITE ,__crsr_pal_bits);
__IO_REG32_BIT(CRSR_PAL1,             0x31040C0C,__READ_WRITE ,__crsr_pal_bits);
__IO_REG32_BIT(CRSR_XY,               0x31040C10,__READ_WRITE ,__crsr_xy_bits);
__IO_REG32_BIT(CRSR_CLIP,             0x31040C14,__READ_WRITE ,__crsr_clip_bits);
__IO_REG32_BIT(CRSR_INTMSK,           0x31040C20,__READ_WRITE ,__crsr_intmsk_bits);
__IO_REG32_BIT(CRSR_INTCLR,           0x31040C24,__WRITE      ,__crsr_intclr_bits);
__IO_REG32_BIT(CRSR_INTRAW,           0x31040C28,__READ       ,__crsr_intraw_bits);
__IO_REG32_BIT(CRSR_INTSTAT,          0x31040C2C,__READ       ,__crsr_intstat_bits);

/***************************************************************************
 **
 ** ADC & Touch screen controller
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_STAT,              0x40048000,__READ      ,__adc_stat_bits);
__IO_REG32_BIT(ADC_SELECT,            0x40048004,__READ_WRITE,__adc_select_bits);
__IO_REG32_BIT(ADC_CTRL,              0x40048008,__READ_WRITE,__adc_ctrl_bits);
__IO_REG32_BIT(TSC_SAMPLE_FIFO,       0x4004800C,__READ      ,__tsc_sample_fifo_bits);
__IO_REG32_BIT(TSC_DTR,               0x40048010,__READ_WRITE,__tsc_dtr_bits);
__IO_REG32_BIT(TSC_RTR,               0x40048014,__READ_WRITE,__tsc_rtr_bits);
__IO_REG32_BIT(TSC_UTR,               0x40048018,__READ_WRITE,__tsc_utr_bits);
__IO_REG32_BIT(TSC_TTR,               0x4004801C,__READ_WRITE,__tsc_ttr_bits);
__IO_REG32_BIT(TSC_DXP,               0x40048020,__READ_WRITE,__tsc_dxp_bits);
__IO_REG32_BIT(TSC_MIN_X,             0x40048024,__READ_WRITE,__tsc_min_x_bits);
__IO_REG32_BIT(TSC_MAX_X,             0x40048028,__READ_WRITE,__tsc_max_x_bits);
__IO_REG32_BIT(TSC_MIN_Y,             0x4004802C,__READ_WRITE,__tsc_min_y_bits);
__IO_REG32_BIT(TSC_MAX_Y,             0x40048030,__READ_WRITE,__tsc_max_y_bits);
__IO_REG32(    TSC_AUX_UTR,           0x40048034,__READ_WRITE);
__IO_REG32_BIT(TSC_AUX_MIN,           0x40048038,__READ_WRITE,__tsc_aux_min_bits);
__IO_REG32_BIT(TSC_AUX_MAX,           0x4004803C,__READ_WRITE,__tsc_aux_max_bits);
__IO_REG32_BIT(TSC_AUX_VALUE,         0x40048044,__READ      ,__tsc_aux_value_bits);
__IO_REG32_BIT(ADC_VALUE,             0x40048048,__READ      ,__adc_value_bits);       

/***************************************************************************
 **
 ** Keyboard Scan
 **
 ***************************************************************************/
__IO_REG32(    KS_DEB,                0x40050000,__READ_WRITE);
__IO_REG32_BIT(KS_STATE_COND,         0x40050004,__READ      ,__ks_state_cond_bits);
__IO_REG32_BIT(KS_IRQ,                0x40050008,__READ_WRITE,__ks_irq_bits);
__IO_REG32(    KS_SCAN_CTL,           0x4005000C,__READ_WRITE);
__IO_REG32_BIT(KS_FAST_TST,           0x40050010,__READ_WRITE,__ks_fast_tst_bits);
__IO_REG32_BIT(KS_MATRIX_DIM,         0x40050014,__READ_WRITE,__ks_matrix_dim_bits);
__IO_REG32(    KS_DATA0,              0x40050040,__READ      );
__IO_REG32(    KS_DATA1,              0x40050044,__READ      );
__IO_REG32(    KS_DATA2,              0x40050048,__READ      );
__IO_REG32(    KS_DATA3,              0x4005004C,__READ      );
__IO_REG32(    KS_DATA4,              0x40050050,__READ      );
__IO_REG32(    KS_DATA5,              0x40050054,__READ      );
__IO_REG32(    KS_DATA6,              0x40050058,__READ      );
__IO_REG32(    KS_DATA7,              0x4005005C,__READ      );

/***************************************************************************
 **
 **  ETHERNET
 **
 ***************************************************************************/
__IO_REG32_BIT(MAC1,                  0x31060000,__READ_WRITE ,__mac1_bits);
__IO_REG32_BIT(MAC2,                  0x31060004,__READ_WRITE ,__mac2_bits);
__IO_REG32_BIT(IPGT,                  0x31060008,__READ_WRITE ,__ipgt_bits);
__IO_REG32_BIT(IPGR,                  0x3106000C,__READ_WRITE ,__ipgr_bits);
__IO_REG32_BIT(CLRT,                  0x31060010,__READ_WRITE ,__clrt_bits);
__IO_REG32_BIT(MAXF,                  0x31060014,__READ_WRITE ,__maxf_bits);
__IO_REG32_BIT(SUPP,                  0x31060018,__READ_WRITE ,__supp_bits);
__IO_REG32_BIT(TEST,                  0x3106001C,__READ_WRITE ,__test_bits);
__IO_REG32_BIT(MCFG,                  0x31060020,__READ_WRITE ,__mcfg_bits);
__IO_REG32_BIT(MCMD,                  0x31060024,__READ_WRITE ,__mcmd_bits);
__IO_REG32_BIT(MADR,                  0x31060028,__READ_WRITE ,__madr_bits);
__IO_REG32_BIT(MWTD,                  0x3106002C,__WRITE      ,__mwtd_bits);
__IO_REG32_BIT(MRDD,                  0x31060030,__READ       ,__mrdd_bits);
__IO_REG32_BIT(MIND,                  0x31060034,__READ       ,__mind_bits);
__IO_REG32_BIT(SA0,                   0x31060040,__READ_WRITE ,__sa0_bits);
__IO_REG32_BIT(SA1,                   0x31060044,__READ_WRITE ,__sa1_bits);
__IO_REG32_BIT(SA2,                   0x31060048,__READ_WRITE ,__sa2_bits);
__IO_REG32_BIT(COMMAND,               0x31060100,__READ_WRITE ,__command_bits);
__IO_REG32_BIT(STATUS,                0x31060104,__READ       ,__status_bits);
__IO_REG32(    RXDESCRIPTOR,          0x31060108,__READ_WRITE );
__IO_REG32(    RXSTATUS,              0x3106010C,__READ_WRITE );
__IO_REG32_BIT(RXDESCRIPTORNUMBER,    0x31060110,__READ_WRITE ,__rxdescrn_bits);
__IO_REG32_BIT(RXPRODUCEINDEX,        0x31060114,__READ       ,__rxprodind_bits);
__IO_REG32_BIT(RXCONSUMEINDEX,        0x31060118,__READ_WRITE ,__rxcomind_bits);
__IO_REG32(    TXDESCRIPTOR,          0x3106011C,__READ_WRITE );
__IO_REG32(    TXSTATUS,              0x31060120,__READ_WRITE );
__IO_REG32_BIT(TXDESCRIPTORNUMBER,    0x31060124,__READ_WRITE ,__txdescrn_bits);
__IO_REG32_BIT(TXPRODUCEINDEX,        0x31060128,__READ_WRITE ,__txprodind_bits);
__IO_REG32_BIT(TXCONSUMEINDEX,        0x3106012C,__READ       ,__txcomind_bits);
__IO_REG32_BIT(TSV0,                  0x31060158,__READ       ,__tsv0_bits);
__IO_REG32_BIT(TSV1,                  0x3106015C,__READ       ,__tsv1_bits);
__IO_REG32_BIT(RSV,                   0x31060160,__READ       ,__rsv_bits);
__IO_REG32_BIT(FLOWCONTROLCOUNTER,    0x31060170,__READ_WRITE ,__fwctrlcnt_bits);
__IO_REG32_BIT(FLOWCONTROLSTATUS,     0x31060174,__READ       ,__fwctrlstat_bits);
__IO_REG32_BIT(RXFILTERCTRL,          0x31060200,__READ_WRITE ,__rxflctrl_bits);
__IO_REG32_BIT(RXFILTERWOLSTATUS,     0x31060204,__READ_WRITE ,__rxflwolstat_bits);
__IO_REG32_BIT(RXFILTERWOLCLEAR,      0x31060208,__READ_WRITE ,__rxflwolclr_bits);
__IO_REG32(    HASHFILTERL,           0x31060210,__READ_WRITE );
__IO_REG32(    HASHFILTERH,           0x31060214,__READ_WRITE );
__IO_REG32_BIT(INTSTATUS,             0x31060FE0,__READ       ,__intstat_bits);
__IO_REG32_BIT(INTENABLE,             0x31060FE4,__READ_WRITE ,__intena_bits);
__IO_REG32_BIT(INTCLEAR,              0x31060FE8,__WRITE      ,__intclr_bits);
__IO_REG32_BIT(INTSET,                0x31060FEC,__WRITE      ,__intset_bits);
__IO_REG32_BIT(POWERDOWN,             0x31060FF4,__READ_WRITE ,__pwrdn_bits);

/***************************************************************************
 **
 ** USB Host (OHCI) Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(HcRevision,            0x31020000,__READ      ,__HcRevision_bits);
__IO_REG32_BIT(HcControl,             0x31020004,__READ_WRITE,__HcControl_bits);
__IO_REG32_BIT(HcCommandStatus,       0x31020008,__READ_WRITE,__HcCommandStatus_bits);
__IO_REG32_BIT(HcInterruptStatus,     0x3102000C,__READ_WRITE,__HcInterruptStatus_bits);
__IO_REG32_BIT(HcInterruptEnable,     0x31020010,__READ_WRITE,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcInterruptDisable,    0x31020014,__READ_WRITE,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcHCCA,                0x31020018,__READ_WRITE,__HcHCCA_bits);
__IO_REG32_BIT(HcPeriodCurrentED,     0x3102001C,__READ      ,__HcPeriodCurrentED_bits);
__IO_REG32_BIT(HcControlHeadED,       0x31020020,__READ_WRITE,__HcControlHeadED_bits);
__IO_REG32_BIT(HcControlCurrentED,    0x31020024,__READ_WRITE,__HcControlCurrentED_bits);
__IO_REG32_BIT(HcBulkHeadED,          0x31020028,__READ_WRITE,__HcBulkHeadED_bits);
__IO_REG32_BIT(HcBulkCurrentED,       0x3102002C,__READ_WRITE,__HcBulkCurrentED_bits);
__IO_REG32_BIT(HcDoneHead,            0x31020030,__READ      ,__HcDoneHead_bits);
__IO_REG32_BIT(HcFmInterval,          0x31020034,__READ_WRITE,__HcFmInterval_bits);
__IO_REG32_BIT(HcFmRemaining,         0x31020038,__READ      ,__HcFmRemaining_bits);
__IO_REG32_BIT(HcFmNumber,            0x3102003C,__READ      ,__HcFmNumber_bits);
__IO_REG32_BIT(HcPeriodicStart,       0x31020040,__READ_WRITE,__HcPeriodicStart_bits);
__IO_REG32_BIT(HcLSThreshold,         0x31020044,__READ_WRITE,__HcLSThreshold_bits);
__IO_REG32_BIT(HcRhDescriptorA,       0x31020048,__READ_WRITE,__HcRhDescriptorA_bits);
__IO_REG32_BIT(HcRhDescriptorB,       0x3102004C,__READ_WRITE,__HcRhDescriptorB_bits);
__IO_REG32_BIT(HcRhStatus,            0x31020050,__READ_WRITE,__HcRhStatus_bits);
__IO_REG32_BIT(HcRhPortStatus1,       0x31020054,__READ_WRITE,__HcRhPortStatus_bits);
__IO_REG32_BIT(HcRhPortStatus2,       0x31020058,__READ_WRITE,__HcRhPortStatus_bits);

/***************************************************************************
 **
 ** USB Device Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(USBDevIntSt,           0x31020200,__READ      ,__devints_bits);
__IO_REG32_BIT(USBDevIntEn,           0x31020204,__READ_WRITE,__devints_bits);
__IO_REG32_BIT(USBDevIntClr,          0x31020208,__WRITE     ,__devints_bits);
__IO_REG32_BIT(USBDevIntSet,          0x3102020C,__WRITE     ,__devints_bits);
__IO_REG32_BIT(USBDevIntPri,          0x3102022C,__WRITE     ,__devintpri_bits);
__IO_REG32_BIT(USBEpIntSt,            0x31020230,__READ      ,__endpints_bits);
__IO_REG32_BIT(USBEpIntEn,            0x31020234,__READ_WRITE,__endpints_bits);
__IO_REG32_BIT(USBEpIntClr,           0x31020238,__WRITE     ,__endpints_bits);
__IO_REG32_BIT(USBEpIntSet,           0x3102023C,__WRITE     ,__endpints_bits);
__IO_REG32_BIT(USBEpIntPri,           0x31020240,__WRITE     ,__endpints_bits);
__IO_REG32_BIT(USBReEp,               0x31020244,__READ_WRITE,__realizeendp_bits);
__IO_REG32_BIT(USBEpInd,              0x31020248,__WRITE     ,__endpind_bits);
__IO_REG32_BIT(USBEpMaxPSize,         0x3102024C,__READ_WRITE,__maxpacksize_bits);
__IO_REG32(    USBRxData,             0x31020218,__READ      );
__IO_REG32_BIT(USBRxPLen,             0x31020220,__READ      ,__rcvepktlen_bits);
__IO_REG32(    USBTxData,             0x3102021C,__WRITE     );
__IO_REG32_BIT(USBTxPLen,             0x31020224,__WRITE     ,__transmitpktlen_bits);
__IO_REG32_BIT(USBCtrl,               0x31020228,__READ_WRITE,__usbctrl_bits);
__IO_REG32_BIT(USBCmdCode,            0x31020210,__WRITE     ,__cmdcode_bits);
__IO_REG32_BIT(USBCmdData,            0x31020214,__READ      ,__cmddata_bits);
__IO_REG32_BIT(USBDMARSt,             0x31020250,__READ      ,__dmarqstdiv_bits);
__IO_REG32_BIT(USBDMARClr,            0x31020254,__WRITE     ,__dmarqstdiv_bits);
__IO_REG32_BIT(USBDMARSet,            0x31020258,__WRITE     ,__dmarqstdiv_bits);
__IO_REG32_BIT(USBUDCAH,              0x31020280,__READ_WRITE,__udcahead_bits);
__IO_REG32_BIT(USBEpDMASt,            0x31020284,__READ      ,__epdmadiv_bits);
__IO_REG32_BIT(USBEpDMAEn,            0x31020288,__WRITE     ,__epdmadiv_bits);
__IO_REG32_BIT(USBEpDMADis,           0x3102028C,__WRITE     ,__epdmadiv_bits);
__IO_REG32_BIT(USBDMAIntSt,           0x31020290,__READ      ,__dmaintstat_bits);
__IO_REG32_BIT(USBDMAIntEn,           0x31020294,__READ_WRITE,__dmaintstat_bits);
__IO_REG32_BIT(USBEoTIntSt,           0x310202A0,__READ      ,__newdddiv_bits);
__IO_REG32_BIT(USBEoTIntClr,          0x310202A4,__WRITE     ,__newdddiv_bits);
__IO_REG32_BIT(USBEoTIntSet,          0x310202A8,__WRITE     ,__newdddiv_bits);
__IO_REG32_BIT(USBNDDRIntSt,          0x310202AC,__READ      ,__newdddiv_bits);
__IO_REG32_BIT(USBNDDRIntClr,         0x310202B0,__WRITE     ,__newdddiv_bits);
__IO_REG32_BIT(USBNDDRIntSet,         0x310202B4,__WRITE     ,__newdddiv_bits);
__IO_REG32_BIT(USBSysErrIntSt,        0x310202B8,__READ      ,__newdddiv_bits);
__IO_REG32_BIT(USBSysErrIntClr,       0x310202BC,__WRITE     ,__newdddiv_bits);
__IO_REG32_BIT(USBSysErrIntSet,       0x310202C0,__WRITE     ,__newdddiv_bits);

/***************************************************************************
 **
 ** USB OTG Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(OTG_int_status,        0x31020100,__READ      ,__otg_int_status_bits);
__IO_REG32_BIT(OTG_int_enable,        0x31020104,__READ_WRITE,__otg_int_enable_bits);
__IO_REG32_BIT(OTG_int_set,           0x31020108,__WRITE     ,__otg_int_set_bits);
__IO_REG32_BIT(OTG_int_clear,         0x3102010C,__WRITE     ,__otg_int_clr_bits);
__IO_REG32_BIT(OTG_stat_ctrl,         0x31020110,__READ_WRITE,__otg_stat_ctrl_bits);
__IO_REG32(    OTG_timer,             0x31020114,__READ_WRITE);
__IO_REG32_BIT(I2C_RX,                0x31020300,__READ_WRITE,__otg_i2c_rx_tx_bits);
#define I2C_TX      I2C_RX
#define I2C_TX_bit  I2C_RX_bit
__IO_REG32_BIT(I2C_STS,               0x31020304,__READ_WRITE,__otg_i2c_sts_bits);
__IO_REG32_BIT(I2C_CTL,               0x31020308,__READ_WRITE,__otg_i2c_ctl_bits);
__IO_REG32(    I2C_CLKHI,             0x3102030C,__READ_WRITE);
__IO_REG32(    I2C_CLKLO,             0x31020310,__READ_WRITE);
__IO_REG32_BIT(OTG_clock_control,     0x31020FF4,__READ_WRITE,__otg_clock_bits);
__IO_REG32_BIT(OTG_clock_status,      0x31020FF8,__READ      ,__otg_clock_bits);

/***************************************************************************
 **
 ** SD-Card
 **
 ***************************************************************************/
__IO_REG32_BIT(SD_Power,              0x20098000,__READ_WRITE,__sd_power_bits);
__IO_REG32_BIT(SD_Clock,              0x20098004,__READ_WRITE,__sd_clock_bits);
__IO_REG32(    SD_Argument,           0x20098008,__READ_WRITE);
__IO_REG32_BIT(SD_Command,            0x2009800C,__READ_WRITE,__sd_command_bits);
__IO_REG32_BIT(SD_Respcmd,            0x20098010,__READ      ,__sd_respcmd_bits);
__IO_REG32(    SD_Response0,          0x20098014,__READ      );
__IO_REG32(    SD_Response1,          0x20098018,__READ      );
__IO_REG32(    SD_Response2,          0x2009801C,__READ      );
__IO_REG32(    SD_Response3,          0x20098020,__READ      );
__IO_REG32(    SD_DataTimer,          0x20098024,__READ_WRITE);
__IO_REG32(    SD_DataLength,         0x20098028,__READ_WRITE);
__IO_REG32_BIT(SD_DataCtrl,           0x2009802C,__READ_WRITE,__sd_datactrl_bits);
__IO_REG32(    SD_DataCnt,            0x20098030,__READ      );
__IO_REG32_BIT(SD_Status,             0x20098034,__READ      ,__sd_status_bits);
__IO_REG32_BIT(SD_Clear,              0x20098038,__WRITE     ,__sd_clear_bits);
__IO_REG32_BIT(SD_Mask0,              0x2009803C,__READ_WRITE,__sd_status_bits);
__IO_REG32_BIT(SD_Mask1,              0x20098040,__READ_WRITE,__sd_status_bits);
__IO_REG32_BIT(SD_FIFOCnt,            0x20098048,__READ      ,__sd_fifocnt_bits);
__IO_REG32(    SD_FIFO0,              0x20098080,__READ_WRITE);
__IO_REG32(    SD_FIFO1,              0x20098084,__READ_WRITE);
__IO_REG32(    SD_FIFO2,              0x20098088,__READ_WRITE);
__IO_REG32(    SD_FIFO3,              0x2009808C,__READ_WRITE);
__IO_REG32(    SD_FIFO4,              0x20098090,__READ_WRITE);
__IO_REG32(    SD_FIFO5,              0x20098094,__READ_WRITE);
__IO_REG32(    SD_FIFO6,              0x20098098,__READ_WRITE);
__IO_REG32(    SD_FIFO7,              0x2009809C,__READ_WRITE);
__IO_REG32(    SD_FIFO8,              0x200980A0,__READ_WRITE);
__IO_REG32(    SD_FIFO9,              0x200980A4,__READ_WRITE);
__IO_REG32(    SD_FIFO10,             0x200980A8,__READ_WRITE);
__IO_REG32(    SD_FIFO11,             0x200980AC,__READ_WRITE);
__IO_REG32(    SD_FIFO12,             0x200980B0,__READ_WRITE);
__IO_REG32(    SD_FIFO13,             0x200980B4,__READ_WRITE);
__IO_REG32(    SD_FIFO14,             0x200980B8,__READ_WRITE);
__IO_REG32(    SD_FIFO15,             0x200980BC,__READ_WRITE);

/***************************************************************************
 **
 **  UARTs Common registers
 **
 ***************************************************************************/
__IO_REG32_BIT(UART_CTRL,             0x40054000,__READ_WRITE,__uart_ctrl_bits);
__IO_REG32_BIT(UART_CLKMODE,          0x40054004,__READ_WRITE,__uart_clkmode_bits);
__IO_REG32_BIT(UART_LOOP,             0x40054008,__READ_WRITE,__uart_loop_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
/* U3DLL, U3RBR and U3THR share the same address */
__IO_REG32(    U3RBRTHR,              0x40080000,__READ_WRITE);
#define U3DLL U3RBRTHR
#define U3RBR U3RBRTHR
#define U3THR U3RBRTHR

/* U3DLM and U3IER share the same address */
__IO_REG32_BIT(U3IER,                 0x40080004,__READ_WRITE,__uartier_bits);
#define U3DLM U3IER

/* U3FCR and U3IIR share the same address */
__IO_REG32_BIT(U3FCR,                 0x40080008,__READ_WRITE,__uartfcriir_bits);
#define U3IIR      U3FCR
#define U3IIR_bit  U3FCR_bit

__IO_REG32_BIT(U3LCR,                 0x4008000C,__READ_WRITE,__uartlcr_bits);
__IO_REG32_BIT(U3LSR,                 0x40080014,__READ      ,__uartlsr_bits);
__IO_REG32_BIT(U3RXLEV,               0x4008001C,__READ      ,__uartrexlev_bits);
__IO_REG32_BIT(U3CLK,                 0x400040D0,__READ_WRITE,__uartclk_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
/* U4DLL, U4RBR and U4THR share the same address */
__IO_REG32(    U4RBRTHR,              0x40088000,__READ_WRITE);
#define U4DLL U4RBRTHR
#define U4RBR U4RBRTHR
#define U4THR U4RBRTHR

/* U4DLM and U4IER share the same address */
__IO_REG32_BIT(U4IER,                 0x40088004,__READ_WRITE,__uartier_bits);
#define U4DLM U4IER

/* U4FCR and U4IIR share the same address */
__IO_REG32_BIT(U4FCR,                 0x40088008,__READ_WRITE,__uartfcriir_bits);
#define U4IIR      U4FCR
#define U4IIR_bit  U4FCR_bit

__IO_REG32_BIT(U4LCR,                 0x4008800C,__READ_WRITE,__uartlcr_bits);
__IO_REG32_BIT(U4LSR,                 0x40088014,__READ      ,__uartlsr_bits);
__IO_REG32_BIT(U4RXLEV,               0x4008801C,__READ      ,__uartrexlev_bits);
__IO_REG32_BIT(U4CLK,                 0x400040D4,__READ_WRITE,__uartclk_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
/* U5DLL, U5RBR and U5THR share the same address */
__IO_REG32(    U5RBRTHR,              0x40090000,__READ_WRITE);
#define U5DLL U5RBRTHR
#define U5RBR U5RBRTHR
#define U5THR U5RBRTHR

/* U5DLM and U5IER share the same address */
__IO_REG32_BIT(U5IER,                 0x40090004,__READ_WRITE,__uartier_bits);
#define U5DLM U5IER

/* U5FCR and U5IIR share the same address */
__IO_REG32_BIT(U5FCR,                 0x40090008,__READ_WRITE,__uartfcriir_bits);
#define U5IIR      U5FCR
#define U5IIR_bit  U5FCR_bit

__IO_REG32_BIT(U5LCR,                 0x4009000C,__READ_WRITE,__uartlcr_bits);
__IO_REG32_BIT(U5LSR,                 0x40090014,__READ      ,__uartlsr_bits);
__IO_REG32_BIT(U5RXLEV,               0x4009001C,__READ      ,__uartrexlev_bits);
__IO_REG32_BIT(U5CLK,                 0x400040D8,__READ_WRITE,__uartclk_bits);

/***************************************************************************
 **
 **  UART6
 **
 ***************************************************************************/
/* U6DLL, U6RBR and U6THR share the same address */
__IO_REG32(    U6RBRTHR,              0x40098000,__READ_WRITE);
#define U6DLL U6RBRTHR
#define U6RBR U6RBRTHR
#define U6THR U6RBRTHR

/* U6DLM and U6IER share the same address */
__IO_REG32_BIT(U6IER,                 0x40098004,__READ_WRITE,__uartier_bits);
#define U6DLM U5IER

/* U5FCR and U5IIR share the same address */
__IO_REG32_BIT(U6FCR,                 0x40098008,__READ_WRITE,__uartfcriir_bits);
#define U6IIR      U6FCR
#define U6IIR_bit  U6FCR_bit

__IO_REG32_BIT(U6LCR,                 0x4009800C,__READ_WRITE,__uartlcr_bits);
__IO_REG32_BIT(U6LSR,                 0x40098014,__READ      ,__uartlsr_bits);
__IO_REG32_BIT(U6RXLEV,               0x4009801C,__READ      ,__uartrexlev_bits);
__IO_REG32_BIT(U6CLK,                 0x400040DC,__READ_WRITE,__uartclk_bits);
__IO_REG32_BIT(IRDACLK,               0x400040E0,__READ_WRITE,__irdaclk_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(HSU1_RX,               0x40014000,__READ_WRITE,__hsu_rx_tx_bits);
#define HSU1_TX     HSU1_RX
#define HSU1_TX_bit HSU1_RX_bit
__IO_REG32_BIT(HSU1_LEVEL,            0x40014004,__READ      ,__hsu_level_bits);
__IO_REG32_BIT(HSU1_IIR,              0x40014008,__READ_WRITE,__hsu_iir_bits);
__IO_REG32_BIT(HSU1_CTRL,             0x4001400C,__READ_WRITE,__hsu_ctrl_bits);
__IO_REG32(    HSU1_RATE,             0x40014010,__READ_WRITE);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(HSU2_RX,               0x40018000,__READ_WRITE,__hsu_rx_tx_bits);
#define HSU2_TX     HSU2_RX
#define HSU2_TX_bit HSU2_RX_bit
__IO_REG32_BIT(HSU2_LEVEL,            0x40018004,__READ      ,__hsu_level_bits);
__IO_REG32_BIT(HSU2_IIR,              0x40018008,__READ_WRITE,__hsu_iir_bits);
__IO_REG32_BIT(HSU2_CTRL,             0x4001800C,__READ_WRITE,__hsu_ctrl_bits);
__IO_REG32(    HSU2_RATE,             0x40018010,__READ_WRITE);

/***************************************************************************
 **
 **  UART7
 **
 ***************************************************************************/
__IO_REG32_BIT(HSU7_RX,               0x4001C000,__READ_WRITE,__hsu_rx_tx_bits);
#define HSU7_TX     HSU7_RX
#define HSU7_TX_bit HSU7_RX_bit
__IO_REG32_BIT(HSU7_LEVEL,            0x4001C004,__READ      ,__hsu_level_bits);
__IO_REG32_BIT(HSU7_IIR,              0x4001C008,__READ_WRITE,__hsu_iir_bits);
__IO_REG32_BIT(HSU7_CTRL,             0x4001C00C,__READ_WRITE,__hsu_ctrl_bits);
__IO_REG32(    HSU7_RATE,             0x4001C010,__READ_WRITE);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_GLOBAL,           0x20088000,__READ_WRITE,__spi_global_bits);
__IO_REG32_BIT(SPI1_CON,              0x20088004,__READ_WRITE,__spi_con_bits);
__IO_REG32(    SPI1_FRM,              0x20088008,__READ_WRITE);
__IO_REG32_BIT(SPI1_IER,              0x2008800C,__READ_WRITE,__spi_ier_bits);
__IO_REG32_BIT(SPI1_STAT,             0x20088010,__READ_WRITE,__spi_stat_bits);
__IO_REG32(    SPI1_DAT,              0x20088014,__READ_WRITE);
__IO_REG32_BIT(SPI1_TIM_CTRL,         0x20088400,__READ_WRITE,__spi_tim_ctrl_bits);
__IO_REG32(    SPI1_TIM_COUNT,        0x20088404,__READ_WRITE);
__IO_REG32_BIT(SPI1_TIM_STAT,         0x20088408,__READ_WRITE,__spi_tim_stat_bits);

/***************************************************************************
 **
 ** SPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI2_GLOBAL,           0x20090000,__READ_WRITE,__spi_global_bits);
__IO_REG32_BIT(SPI2_CON,              0x20090004,__READ_WRITE,__spi_con_bits);
__IO_REG32(    SPI2_FRM,              0x20090008,__READ_WRITE);
__IO_REG32_BIT(SPI2_IER,              0x2009000C,__READ_WRITE,__spi_ier_bits);
__IO_REG32_BIT(SPI2_STAT,             0x20090010,__READ_WRITE,__spi_stat_bits);
__IO_REG32(    SPI2_DAT,              0x20090014,__READ_WRITE);
__IO_REG32_BIT(SPI2_TIM_CTRL,         0x20090400,__READ_WRITE,__spi_tim_ctrl_bits);
__IO_REG32(    SPI2_TIM_COUNT,        0x20090404,__READ_WRITE);
__IO_REG32_BIT(SPI2_TIM_STAT,         0x20090408,__READ_WRITE,__spi_tim_stat_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,               0x20084000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,               0x20084004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,                0x20084008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,                0x2008400C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,              0x20084010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,              0x20084014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,               0x20084018,__READ_WRITE ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,               0x2008401C,__READ_WRITE ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,               0x20084020,__READ_WRITE ,__sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,             0x20084024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,               0x2008C000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,               0x2008C004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,                0x2008C008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,                0x2008C00C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,              0x2008C010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,              0x2008C014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,               0x2008C018,__READ_WRITE ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,               0x2008C01C,__READ_WRITE ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,               0x2008C020,__READ_WRITE ,__sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,             0x2008C024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_TX,               0x400A0000,__READ_WRITE,__i2c_rx_tx_bits);
#define I2C1_RX     I2C1_TX
#define I2C1_RX_bit I2C1_TX_bit
__IO_REG32_BIT(I2C1_STAT,             0x400A0004,__READ_WRITE,__i2c_stat_bits);
__IO_REG32_BIT(I2C1_CTRL,             0x400A0008,__READ_WRITE,__i2c_ctrl_bits);
__IO_REG32_BIT(I2C1_CLK_HI,           0x400A000C,__READ_WRITE,__i2c_clk_hi_bits);
__IO_REG32_BIT(I2C1_CLK_LO,           0x400A0010,__READ_WRITE,__i2c_clk_lo_bits);
__IO_REG32_BIT(I2C1_ADR,              0x400A0014,__READ_WRITE,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_RXFL,             0x400A0018,__READ      ,__i2c_rxfl_bits);
__IO_REG32_BIT(I2C1_TXFL,             0x400A001C,__READ      ,__i2c_txfl_bits);
__IO_REG32(    I2C1_RXB,              0x400A0020,__READ);
__IO_REG32(    I2C1_TXB,              0x400A0024,__READ);
__IO_REG32_BIT(I2C1_S_TX,             0x400A0028,__WRITE     ,__i2c_s_tx_bits);
__IO_REG32_BIT(I2C1_S_TXFL,           0x400A002C,__READ      ,__i2c_s_txfl_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C2_RX,               0x400A8000,__READ_WRITE,__i2c_rx_tx_bits);
#define I2C2_TX     I2C2_RX
#define I2C2_TX_bit I2C2_RX_bit
__IO_REG32_BIT(I2C2_STAT,             0x400A8004,__READ_WRITE,__i2c_stat_bits);
__IO_REG32_BIT(I2C2_CTRL,             0x400A8008,__READ_WRITE,__i2c_ctrl_bits);
__IO_REG32_BIT(I2C2_CLK_HI,           0x400A800C,__READ_WRITE,__i2c_clk_hi_bits);
__IO_REG32_BIT(I2C2_CLK_LO,           0x400A8010,__READ_WRITE,__i2c_clk_lo_bits);
__IO_REG32_BIT(I2C2_ADR,              0x400A8014,__READ_WRITE,__i2c_adr_bits);
__IO_REG32_BIT(I2C2_RXFL,             0x400A8018,__READ      ,__i2c_rxfl_bits);
__IO_REG32_BIT(I2C2_TXFL,             0x400A801C,__READ      ,__i2c_txfl_bits);
__IO_REG32(    I2C2_RXB,              0x400A8020,__READ);
__IO_REG32(    I2C2_TXB,              0x400A8024,__READ);
__IO_REG32_BIT(I2C2_S_TX,             0x400A8028,__WRITE     ,__i2c_s_tx_bits);
__IO_REG32_BIT(I2C2_S_TXFL,           0x400A802C,__READ      ,__i2c_s_txfl_bits);

/***************************************************************************
 **
 ** I2S0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S0DAO,                0x20094000,__READ_WRITE ,__i2sdao_bits);
__IO_REG32_BIT(I2S0DAI,                0x20094004,__READ_WRITE ,__i2sdai_bits);
__IO_REG32(    I2S0TXFIFO,             0x20094008,__WRITE);
__IO_REG32(    I2S0RXFIFO,             0x2009400C,__READ);
__IO_REG32_BIT(I2S0STATE,              0x20094010,__READ       ,__i2sstate_bits);
__IO_REG32_BIT(I2S0DMA1,               0x20094014,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2S0DMA2,               0x20094018,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2S0IRQ,                0x2009401C,__READ_WRITE ,__i2sirq_bits);
__IO_REG32_BIT(I2S0TXRATE,             0x20094020,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2S0RXRATE,             0x20094024,__READ_WRITE ,__i2srxrate_bits);

/***************************************************************************
 **
 ** I2S1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S1DAO,                0x2009C000,__READ_WRITE ,__i2sdao_bits);
__IO_REG32_BIT(I2S1DAI,                0x2009C004,__READ_WRITE ,__i2sdai_bits);
__IO_REG32(    I2S1TXFIFO,             0x2009C008,__WRITE);
__IO_REG32(    I2S1RXFIFO,             0x2009C00C,__READ);
__IO_REG32_BIT(I2S1STATE,              0x2009C010,__READ       ,__i2sstate_bits);
__IO_REG32_BIT(I2S1DMA1,               0x2009C014,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2S1DMA2,               0x2009C018,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2S1IRQ,                0x2009C01C,__READ_WRITE ,__i2sirq_bits);
__IO_REG32_BIT(I2S1TXRATE,             0x2009C020,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2S1RXRATE,             0x2009C024,__READ_WRITE ,__i2srxrate_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(T0IR,                  0x40044000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T0TCR,                 0x40044004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T0TC,                  0x40044008,__READ_WRITE);
__IO_REG32(    T0PR,                  0x4004400C,__READ_WRITE);
__IO_REG32(    T0PC,                  0x40044010,__READ_WRITE);
__IO_REG32_BIT(T0MCR,                 0x40044014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T0MR0,                 0x40044018,__READ_WRITE);
__IO_REG32(    T0MR1,                 0x4004401C,__READ_WRITE);
__IO_REG32(    T0MR2,                 0x40044020,__READ_WRITE);
__IO_REG32(    T0MR3,                 0x40044024,__READ_WRITE);
__IO_REG32_BIT(T0CCR,                 0x40044028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T0CR0,                 0x4004402C,__READ);
__IO_REG32(    T0CR1,                 0x40044030,__READ);
__IO_REG32(    T0CR2,                 0x40044034,__READ);
__IO_REG32(    T0CR3,                 0x40044038,__READ);
__IO_REG32_BIT(T0EMR,                 0x4004403C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T0CTCR,                0x40044070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(T1IR,                  0x4004C000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T1TCR,                 0x4004C004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T1TC,                  0x4004C008,__READ_WRITE);
__IO_REG32(    T1PR,                  0x4004C00C,__READ_WRITE);
__IO_REG32(    T1PC,                  0x4004C010,__READ_WRITE);
__IO_REG32_BIT(T1MCR,                 0x4004C014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T1MR0,                 0x4004C018,__READ_WRITE);
__IO_REG32(    T1MR1,                 0x4004C01C,__READ_WRITE);
__IO_REG32(    T1MR2,                 0x4004C020,__READ_WRITE);
__IO_REG32(    T1MR3,                 0x4004C024,__READ_WRITE);
__IO_REG32_BIT(T1CCR,                 0x4004C028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T1CR0,                 0x4004C02C,__READ);
__IO_REG32(    T1CR1,                 0x4004C030,__READ);
__IO_REG32(    T1CR2,                 0x4004C034,__READ);
__IO_REG32(    T1CR3,                 0x4004C038,__READ);
__IO_REG32_BIT(T1EMR,                 0x4004C03C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T1CTCR,                0x4004C070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(T2IR,                  0x40058000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T2TCR,                 0x40058004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T2TC,                  0x40058008,__READ_WRITE);
__IO_REG32(    T2PR,                  0x4005800C,__READ_WRITE);
__IO_REG32(    T2PC,                  0x40058010,__READ_WRITE);
__IO_REG32_BIT(T2MCR,                 0x40058014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T2MR0,                 0x40058018,__READ_WRITE);
__IO_REG32(    T2MR1,                 0x4005801C,__READ_WRITE);
__IO_REG32(    T2MR2,                 0x40058020,__READ_WRITE);
__IO_REG32(    T2MR3,                 0x40058024,__READ_WRITE);
__IO_REG32_BIT(T2CCR,                 0x40058028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T2CR0,                 0x4005802C,__READ);
__IO_REG32(    T2CR1,                 0x40058030,__READ);
__IO_REG32(    T2CR2,                 0x40058034,__READ);
__IO_REG32(    T2CR3,                 0x40058038,__READ);
__IO_REG32_BIT(T2EMR,                 0x4005803C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T2CTCR,                0x40058070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(T3IR,                  0x40060000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T3TCR,                 0x40060004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T3TC,                  0x40060008,__READ_WRITE);
__IO_REG32(    T3PR,                  0x4006000C,__READ_WRITE);
__IO_REG32(    T3PC,                  0x40060010,__READ_WRITE);
__IO_REG32_BIT(T3MCR,                 0x40060014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T3MR0,                 0x40060018,__READ_WRITE);
__IO_REG32(    T3MR1,                 0x4006001C,__READ_WRITE);
__IO_REG32(    T3MR2,                 0x40060020,__READ_WRITE);
__IO_REG32(    T3MR3,                 0x40060024,__READ_WRITE);
__IO_REG32_BIT(T3CCR,                 0x40060028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T3CR0,                 0x4006002C,__READ);
__IO_REG32(    T3CR1,                 0x40060030,__READ);
__IO_REG32(    T3CR2,                 0x40060034,__READ);
__IO_REG32(    T3CR3,                 0x40060038,__READ);
__IO_REG32_BIT(T3EMR,                 0x4006003C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T3CTCR,                0x40060070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER4
 **
 ***************************************************************************/
__IO_REG32_BIT(T4IR,                  0x4002C000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T4TCR,                 0x4002C004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T4TC,                  0x4002C008,__READ_WRITE);
__IO_REG32(    T4PR,                  0x4002C00C,__READ_WRITE);
__IO_REG32(    T4PC,                  0x4002C010,__READ_WRITE);
__IO_REG32_BIT(T4MCR,                 0x4002C014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T4MR0,                 0x4002C018,__READ_WRITE);
__IO_REG32(    T4MR1,                 0x4002C01C,__READ_WRITE);
__IO_REG32(    T4MR2,                 0x4002C020,__READ_WRITE);
__IO_REG32(    T4MR3,                 0x4002C024,__READ_WRITE);
__IO_REG32_BIT(T4CCR,                 0x4002C028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T4CR0,                 0x4002C02C,__READ);
__IO_REG32(    T4CR1,                 0x4002C030,__READ);
__IO_REG32(    T4CR2,                 0x4002C034,__READ);
__IO_REG32(    T4CR3,                 0x4002C038,__READ);
__IO_REG32_BIT(T4EMR,                 0x4002C03C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T4CTCR,                0x4002C070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER5
 **
 ***************************************************************************/
__IO_REG32_BIT(T5IR,                  0x40030000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T5TCR,                 0x40030004,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    T5TC,                  0x40030008,__READ_WRITE);
__IO_REG32(    T5PR,                  0x4003000C,__READ_WRITE);
__IO_REG32(    T5PC,                  0x40030010,__READ_WRITE);
__IO_REG32_BIT(T5MCR,                 0x40030014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T5MR0,                 0x40030018,__READ_WRITE);
__IO_REG32(    T5MR1,                 0x4003001C,__READ_WRITE);
__IO_REG32(    T5MR2,                 0x40030020,__READ_WRITE);
__IO_REG32(    T5MR3,                 0x40030024,__READ_WRITE);
__IO_REG32_BIT(T5CCR,                 0x40030028,__READ_WRITE ,__tmr_ccr_bits);
__IO_REG32(    T5CR0,                 0x4003002C,__READ);
__IO_REG32(    T5CR1,                 0x40030030,__READ);
__IO_REG32(    T5CR2,                 0x40030034,__READ);
__IO_REG32(    T5CR3,                 0x40030038,__READ);
__IO_REG32_BIT(T5EMR,                 0x4003003C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T5CTCR,                0x40030070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** High Speed Timer
 **
 ***************************************************************************/
__IO_REG32_BIT(HSTIM_INT,             0x40038000,__READ_WRITE,__hstim_int_bits);
__IO_REG32_BIT(HSTIM_CTRL,            0x40038004,__READ_WRITE,__hstim_ctrl_bits);
__IO_REG32(    HSTIM_COUNTER,         0x40038008,__READ_WRITE);
__IO_REG32(    HSTIM_PMATCH,          0x4003800C,__READ_WRITE);
__IO_REG32(    HSTIM_PCOUNT,          0x40038010,__READ_WRITE);
__IO_REG32_BIT(HSTIM_MCTRL,           0x40038014,__READ_WRITE,__hstim_mctrl_bits);
__IO_REG32(    HSTIM_MATCH0,          0x40038018,__READ_WRITE);
__IO_REG32(    HSTIM_MATCH1,          0x4003801C,__READ_WRITE);
__IO_REG32(    HSTIM_MATCH2,          0x40038020,__READ_WRITE);
__IO_REG32_BIT(HSTIM_CCR,             0x40038028,__READ_WRITE,__hstim_ccr_bits);
__IO_REG32(    HSTIM_CR0,             0x4003802C,__READ      );
__IO_REG32(    HSTIM_CR1,             0x40038030,__READ      );

/***************************************************************************
 **
 ** Millisecond Timer
 **
 ***************************************************************************/
__IO_REG32_BIT(MSTIM_INT,             0x40034000,__READ_WRITE,__mstim_int_bits);
__IO_REG32_BIT(MSTIM_CTRL,            0x40034004,__READ_WRITE,__mstim_ctrl_bits);
__IO_REG32(    MSTIM_COUNTER,         0x40034008,__READ_WRITE);
__IO_REG32_BIT(MSTIM_MCTRL,           0x40034014,__READ_WRITE,__mstim_mctrl_bits);
__IO_REG32(    MSTIM_MATCH0,          0x40034018,__READ_WRITE);
__IO_REG32(    MSTIM_MATCH1,          0x4003401C,__READ_WRITE);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTC_UCOUNT,            0x40024000,__READ_WRITE);
__IO_REG32(    RTC_DCOUNT,            0x40024004,__READ_WRITE);
__IO_REG32(    RTC_MATCH0,            0x40024008,__READ_WRITE);
__IO_REG32(    RTC_MATCH1,            0x4002400C,__READ_WRITE);
__IO_REG32_BIT(RTC_CTRL,              0x40024010,__READ_WRITE,__rtc_ctrl_bits);
__IO_REG32_BIT(RTC_INTSTAT,           0x40024014,__READ_WRITE,__rtc_intstat_bits);
__IO_REG32(    RTC_KEY,               0x40024018,__READ_WRITE);
__IO_REG32(    RTC_SRAM0,             0x40024080,__READ_WRITE);
__IO_REG32(    RTC_SRAM1,             0x40024084,__READ_WRITE);
__IO_REG32(    RTC_SRAM2,             0x40024088,__READ_WRITE);
__IO_REG32(    RTC_SRAM3,             0x4002408C,__READ_WRITE);
__IO_REG32(    RTC_SRAM4,             0x40024090,__READ_WRITE);
__IO_REG32(    RTC_SRAM5,             0x40024094,__READ_WRITE);
__IO_REG32(    RTC_SRAM6,             0x40024098,__READ_WRITE);
__IO_REG32(    RTC_SRAM7,             0x4002409C,__READ_WRITE);
__IO_REG32(    RTC_SRAM8,             0x400240A0,__READ_WRITE);
__IO_REG32(    RTC_SRAM9,             0x400240A4,__READ_WRITE);
__IO_REG32(    RTC_SRAM10,            0x400240A8,__READ_WRITE);
__IO_REG32(    RTC_SRAM11,            0x400240AC,__READ_WRITE);
__IO_REG32(    RTC_SRAM12,            0x400240B0,__READ_WRITE);
__IO_REG32(    RTC_SRAM13,            0x400240B4,__READ_WRITE);
__IO_REG32(    RTC_SRAM14,            0x400240B8,__READ_WRITE);
__IO_REG32(    RTC_SRAM15,            0x400240BC,__READ_WRITE);
__IO_REG32(    RTC_SRAM16,            0x400240C0,__READ_WRITE);
__IO_REG32(    RTC_SRAM17,            0x400240C4,__READ_WRITE);
__IO_REG32(    RTC_SRAM18,            0x400240C8,__READ_WRITE);
__IO_REG32(    RTC_SRAM19,            0x400240CC,__READ_WRITE);
__IO_REG32(    RTC_SRAM20,            0x400240D0,__READ_WRITE);
__IO_REG32(    RTC_SRAM21,            0x400240D4,__READ_WRITE);
__IO_REG32(    RTC_SRAM22,            0x400240D8,__READ_WRITE);
__IO_REG32(    RTC_SRAM23,            0x400240DC,__READ_WRITE);
__IO_REG32(    RTC_SRAM24,            0x400240E0,__READ_WRITE);
__IO_REG32(    RTC_SRAM25,            0x400240E4,__READ_WRITE);
__IO_REG32(    RTC_SRAM26,            0x400240E8,__READ_WRITE);
__IO_REG32(    RTC_SRAM27,            0x400240EC,__READ_WRITE);
__IO_REG32(    RTC_SRAM28,            0x400240F0,__READ_WRITE);
__IO_REG32(    RTC_SRAM29,            0x400240F4,__READ_WRITE);
__IO_REG32(    RTC_SRAM30,            0x400240F8,__READ_WRITE);
__IO_REG32(    RTC_SRAM31,            0x400240FC,__READ_WRITE);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDTIM_INT,             0x4003C000,__READ_WRITE,__wdtim_int_bits);
__IO_REG32_BIT(WDTIM_CTRL,            0x4003C004,__READ_WRITE,__wdtim_ctrl_bits);
__IO_REG32(    WDTIM_COUNTER,         0x4003C008,__READ_WRITE);
__IO_REG32_BIT(WDTIM_MCTRL,           0x4003C00C,__READ_WRITE,__wdtim_mctrl_bits);
__IO_REG32(    WDTIM_MATCH0,          0x4003C010,__READ_WRITE);
__IO_REG32_BIT(WDTIM_EMR,             0x4003C014,__READ_WRITE,__wdtim_emr_bits);
__IO_REG32(    WDTIM_PULSE,           0x4003C018,__READ_WRITE);
__IO_REG32_BIT(WDTIM_RES,             0x4003C01C,__READ      ,__wdtim_res_bits);

/***************************************************************************
 **
 **Simple PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_CTRL,             0x4005C000,__READ_WRITE,__pwm1_ctrl_bits);
__IO_REG32_BIT(PWM2_CTRL,             0x4005C004,__READ_WRITE,__pwm2_ctrl_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(P0_INP_STATE,          0x40028040,__READ       ,__p0_bits);
__IO_REG32_BIT(P0_OUTP_SET,           0x40028044,__WRITE      ,__p0_bits);
__IO_REG32_BIT(P0_OUTP_CLR,           0x40028048,__WRITE      ,__p0_bits);
__IO_REG32_BIT(P0_OUTP_STATE,         0x4002804C,__READ       ,__p0_bits);
__IO_REG32_BIT(P0_DIR_SET,            0x40028050,__WRITE      ,__p0_bits);
__IO_REG32_BIT(P0_DIR_CLR,            0x40028054,__WRITE      ,__p0_bits);
__IO_REG32_BIT(P0_DIR_STATE,          0x40028058,__READ       ,__p0_bits);
__IO_REG32_BIT(P1_INP_STATE,          0x40028060,__READ       ,__p1_bits);
__IO_REG32_BIT(P1_OUTP_SET,           0x40028064,__WRITE      ,__p1_bits);
__IO_REG32_BIT(P1_OUTP_CLR,           0x40028068,__WRITE      ,__p1_bits);
__IO_REG32_BIT(P1_OUTP_STATE,         0x4002806C,__READ       ,__p1_bits);
__IO_REG32_BIT(P1_DIR_SET,            0x40028070,__WRITE      ,__p1_bits);
__IO_REG32_BIT(P1_DIR_CLR,            0x40028074,__WRITE      ,__p1_bits);
__IO_REG32_BIT(P1_DIR_STATE,          0x40028078,__READ       ,__p1_bits);
__IO_REG32_BIT(P2_INP_STATE,          0x4002801C,__READ       ,__p2_bits);
__IO_REG32_BIT(P2_OUTP_SET,           0x40028020,__WRITE      ,__p2_bits);
__IO_REG32_BIT(P2_OUTP_CLR,           0x40028024,__WRITE      ,__p2_bits);
__IO_REG32_BIT(P2_DIR_SET,            0x40028010,__WRITE      ,__p2_p3_bits);
__IO_REG32_BIT(P2_DIR_CLR,            0x40028014,__WRITE      ,__p2_p3_bits);
__IO_REG32_BIT(P2_DIR_STATE,          0x40028018,__READ       ,__p2_p3_bits);
__IO_REG32_BIT(P3_INP_STATE,          0x40028000,__READ       ,__p3_in_bits);
__IO_REG32_BIT(P3_OUTP_SET,           0x40028004,__WRITE      ,__p3_out_bits);
__IO_REG32_BIT(P3_OUTP_CLR,           0x40028008,__WRITE      ,__p3_out_bits);
__IO_REG32_BIT(P3_OUTP_STATE,         0x4002800C,__READ       ,__p3_out_bits);

/***************************************************************************
 **
 ** Pin multiplexing
 **
 ***************************************************************************/
__IO_REG32_BIT(P_MUX_SET,             0x40028100,__WRITE      ,__p_mux_bits);
__IO_REG32_BIT(P_MUX_CLR,             0x40028104,__WRITE      ,__p_mux_bits);
__IO_REG32_BIT(P_MUX_STATE,           0x40028108,__READ       ,__p_mux_bits);
__IO_REG32_BIT(P0_MUX_SET,            0x40028120,__WRITE      ,__p0_mux_bits);
__IO_REG32_BIT(P0_MUX_CLR,            0x40028124,__WRITE      ,__p0_mux_bits);
__IO_REG32_BIT(P0_MUX_STATE,          0x40028128,__READ       ,__p0_mux_bits);
__IO_REG32_BIT(P1_MUX_SET,            0x40028130,__WRITE      ,__p1_mux_bits);
__IO_REG32_BIT(P1_MUX_CLR,            0x40028134,__WRITE      ,__p1_mux_bits);
__IO_REG32_BIT(P1_MUX_STATE,          0x40028138,__READ       ,__p1_mux_bits);
__IO_REG32_BIT(P2_MUX_SET,            0x40028028,__WRITE      ,__p2_mux_bits);
__IO_REG32_BIT(P2_MUX_CLR,            0x4002802C,__WRITE      ,__p2_mux_bits);
__IO_REG32_BIT(P2_MUX_STATE,          0x40028030,__READ       ,__p2_mux_bits);
__IO_REG32_BIT(P3_MUX_SET,            0x40028110,__WRITE      ,__p3_mux_bits);
__IO_REG32_BIT(P3_MUX_CLR,            0x40028114,__WRITE      ,__p3_mux_bits);
__IO_REG32_BIT(P3_MUX_STATE,          0x40028118,__READ       ,__p3_mux_bits);

/***************************************************************************
 **
 ** Unique Serial ID 
 **
 ***************************************************************************/
__IO_REG32(    SERIAL_ID0,            0x40004130,__READ);
__IO_REG32(    SERIAL_ID1,            0x40004134,__READ);
__IO_REG32(    SERIAL_ID2,            0x40004138,__READ);
__IO_REG32(    SERIAL_ID3,            0x4000413C,__READ);

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
#define RESETV  0x00  /* Reset                           */
#define UNDEFV  0x04  /* Undefined instruction           */
#define SWIV    0x08  /* Software interrupt              */
#define PABORTV 0x0c  /* Prefetch abort                  */
#define DABORTV 0x10  /* Data abort                      */
#define IRQV    0x18  /* Normal interrupt                */
#define FIQV    0x1c  /* Fast interrupt                  */

/***************************************************************************
 **
 **  Main Interrupt channels
 **
 ***************************************************************************/
#define MainSub1IRQn         0  /* Low priority (FIQ) interrupts from SIC1     */
#define MainSub2IRQn         1  /* Low priority (FIQ) interrupts from SIC2     */
#define MainTimer4_MCPWM     3  /* Timer4/MCPWM interrupt                      */
#define MainTimer5           4  /* Timer5 interrupt                            */
#define MainHSTIMER_INT      5  /* Match interrupt from the High Speed timer   */
#define MainWATCH_INT        6  /* Watchdog Timer interrupt                    */
#define MainIIR3             7  /* UART3 interrupt                             */
#define MainIIR4             8  /* UART4 interrupt                             */
#define MainIIR5             9  /* UART5 interrupt                             */
#define MainIIR6            10  /* UART6 interrupt                             */
#define MainFLASH_INT       11  /* Interrupt from the NAND Flash controller    */
#define MainSD1_INT         13  /* Interrupt 1 from the SD Card interface      */
#define MainLCD_INT         14  /* Interrupt from the LCD controller           */
#define MainSD0_INT         15  /* Interrupt 0 from the SD Card interface      */
#define MainTimer0          16  /* Timer0 interrupt                            */
#define MainTimer1          17  /* Timer1 interrupt                            */
#define MainTimer2          18  /* Timer2 interrupt                            */
#define MainTimer3          19  /* Timer3 interrupt                            */
#define MainSSP0            20  /* SSP0 interrupt                              */
#define MainSSP1            21  /* SSP1 interrupt                              */
#define MainI2S0            22  /* I2S0 interrupt                              */
#define MainI2S1            23  /* I2S1 interrupt                              */
#define MainIIR7            24  /* UART7 interrupt                             */
#define MainIIR2            25  /* UART2 interrupt                             */
#define MainIIR1            26  /* UART1 interrupt                             */
#define MainMSTIMER_INT     27  /* Match interrupt 0/1 from Millisecond Timer  */
#define MainDMAINT          28  /* General Purpose DMA Controller interrupt    */
#define MainEthernet        29  /* Ethernet interrupt                          */
#define MainSub1FIQn        30  /* High priority (FIQ) interrupts from SIC1    */
#define MainSub0FIQn        31  /* High priority (FIQ) interrupts from SIC0    */

/***************************************************************************
 **
 **  Sub1 Interrupt channels
 **
 ***************************************************************************/
#define Sub1JTAG_COMM_TX       1 /* Transmitter empty interrupt from the JTAG  */
#define Sub1JTAG_COMM_RX       2 /* Receiver full interrupt from the JTAG      */
#define Sub1GPI_28             4 /* Interrupt from the GPI_28 pin              */
#define Sub1TS_P               6 /* Touch screen pen down interrupt            */
#define Sub1ADC_INT            7 /* Touch screen irq interrupt (A/D Converter interrupt)*/
#define Sub1TS_AUX             8 /* Touch screen aux interrupt                 */
#define Sub1SPI2_INT          12 /* Interrupt from the SPI2 interface          */
#define Sub1PLLUSB_INT        13 /* Lock interrupt from the USB PLL            */
#define Sub1PLLHCLK_INT       14 /* Lock interrupt from the HCLK PLL           */
#define Sub1PLL397_INT        17 /* Lock interrupt from the 397x PLL           */
#define Sub1I2C_2_INT         18 /* Interrupt from the I2C2 interface          */
#define Sub1I2C_1_INT         19 /* Interrupt from the I2C1 interface          */
#define Sub1RTC_INT           20 /* Match interrupt 0 or 1 from the RTC        */
#define Sub1KEY_IRQ           22 /* Keyboard scanner interrupt                 */
#define Sub1SPI1_INT          23 /* Interrupt from the SPI1 interface          */
#define Sub1SW_INT            24 /* Software interrupt (caused by bit 0 of the SW_INT register)*/
#define Sub1USB_otg_timer_int 25 /* USB timer interru                          */
#define Sub1USB_otg_atx_int_n 26 /* External USB transceiver interrupt         */
#define Sub1USB_host_int      27 /* USB host interrupt                         */
#define Sub1USB_dev_dma_int   28 /* USB DMA interrupt                          */
#define Sub1USB_dev_lp_int    29 /* USB low priority interrupt                 */
#define Sub1USB_dev_hp_int    30 /* USB high priority interrupt                */
#define Sub1USB_i2c_int       31 /* Interrupt from the USB I2C interface       */

/***************************************************************************
 **
 **  Sub2 Interrupt channels
 **
 ***************************************************************************/
#define Sub2GPIO_00          0  /* Interrupt from the GPI_00 pin               */
#define Sub2GPIO_01          1  /* Interrupt from the GPI_01 pin               */
#define Sub2GPIO_02          2  /* Interrupt from the GPI_02 pin               */
#define Sub2GPIO_03          3  /* Interrupt from the GPI_03 pin               */
#define Sub2GPIO_04          4  /* Interrupt from the GPI_04 pin               */
#define Sub2GPIO_05          5  /* Interrupt from the GPI_05 pin               */
#define Sub2SPI2_DATIN       6  /* Interrupt from the SPI1_DATIN) pin          */
#define Sub2U2_HCTS          7  /* Interrupt from the UART2 HCTS pin           */
#define Sub2Pn_GPIO          8  /* ALL Port 0 and Port 1 GPIO pins OR’ed       */
#define Sub2GPI_08           9  /* Interrupt from the GPI_08 pin               */
#define Sub2GPI_09          10  /* Interrupt from the GPI_08 pin               */
#define Sub2GPI_10          11  /* Interrupt from the GPI_08 pin               */
#define Sub2U7_HCTS         12  /* Interrupt from the UART7 HCTS pin           */
#define Sub2GPI_07          15  /* Interrupt from the GPI_07 pin               */
#define Sub2SDIO_INT_N      18  /* Interrupt from the MS_DIO1 pin              */
#define Sub2U5_RX           19  /* Interrupt from the UART5 RX pin             */
#define Sub2SPI1_DATIN      20  /* Interrupt from the SPI1_DATIN) pin          */
#define Sub2GPI_00          22  /* Interrupt from the GPI_00 pin               */
#define Sub2GPI_01          23  /* Interrupt from the GPI_01 pin               */
#define Sub2GPI_02          24  /* Interrupt from the GPI_02 pin               */
#define Sub2GPI_03          25  /* Interrupt from the GPI_03 pin               */
#define Sub2GPI_04          26  /* Interrupt from the GPI_04 pin               */
#define Sub2GPI_05          27  /* Interrupt from the GPI_05 pin               */
#define Sub2GPI_06          28  /* Interrupt from the GPI_06 pin               */
#define Sub2SYSCLK_mux      31  /* Status of the SYSCLK Mux                    */

/***************************************************************************
 **
 **  DMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define DMA_I2S0_DMA0        0  /* I2S0 DMA Request 0                       */   
#define DMA_NAND_Flash0      1  /* NAND Flash                               */
#define DMA_SPI2             3  /* SPI2 receive & transmit                  */
#define DMA_SSP1_RX          3  /* SSP1 receive                             */
#define DMA_SD_Card          4  /* SD Card interface receive & transmit     */
#define DMA_HS_Uart1_TX      5  /* HS-Uart1 transmit                        */
#define DMA_HS_Uart1_RX      6  /* HS-Uart1 receive                         */
#define DMA_HS_Uart2_TX      7  /* HS-Uart2 transmit                        */
#define DMA_HS_Uart2_RX      8  /* HS-Uart2 receive                         */
#define DMA_HS_Uart7_TX      9  /* HS-Uart7 transmit                        */
#define DMA_HS_Uart7_RX     10  /* HS-Uart7 receive                         */
#define DMA_I2S1_DMA1       10  /* I2S1 DMA Request 1                       */   
#define DMA_SPI1            11  /* SPI1 receive & transmit                  */
#define DMA_SSP1_TX         11  /* SSP1 transmit                            */
#define DMA_NAND_Flash1     12  /* NAND Flash                               */
#define DMA_I2S0_DMA1       13  /* I2S0 DMA Request 1                       */   
#define DMA_SSP0_RX         14  /* SSP0 receive                             */
#define DMA_SSP0_TX         15  /* SSP0 transmit                            */

#endif    /* __IOLPC3250_H */
