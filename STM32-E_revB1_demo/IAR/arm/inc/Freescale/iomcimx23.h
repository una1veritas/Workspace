/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MCIMX23
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIMX23_H
#define __MCIMX23_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX23 SPECIAL FUNCTION REGISTERS
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

/* PLL Control Register 0 */
typedef struct {
__REG32                :16;
__REG32 POWER          : 1;
__REG32                : 1;
__REG32 EN_USB_CLKS    : 1;
__REG32                : 1;
__REG32 DIV_SEL        : 2;
__REG32                : 2;
__REG32 CP_SEL         : 2;
__REG32                : 2;
__REG32 LFR_SEL        : 2;
__REG32                : 2;
} __hw_clkctrl_pllctrl0_bits;

/* PLL Control Register 1 */
typedef struct {
__REG32 LOCK_COUNT     :16;
__REG32                :14;
__REG32 FORCE_LOCK     : 1;
__REG32 LOCK           : 1;
} __hw_clkctrl_pllctrl1_bits;

/* CPU Clock Control Registe */
typedef struct {
__REG32 DIV_CPU           : 6;
__REG32                   : 6;
__REG32 INTERRUPT_WAIT    : 1;
__REG32                   : 3;
__REG32 DIV_XTAL          :10;
__REG32 DIV_XTAL_FRAC_EN  : 1;
__REG32                   : 1;
__REG32 BUSY_REF_CPU      : 1;
__REG32 BUSY_REF_XTAL     : 1;
__REG32                   : 2;
} __hw_clkctrl_cpu_bits;

/* AHB, APBH Bus Clock Control Register */
typedef struct {
__REG32 DIV                   : 5;
__REG32 DIV_FRAC_EN           : 1;
__REG32                       :10;
__REG32 SLOW_DIV              : 3;
__REG32                       : 1;
__REG32 AUTO_SLOW_MODE        : 1;
__REG32 CPU_INSTR_AS_ENABLE   : 1;
__REG32 CPU_DATA_AS_ENABLE    : 1;
__REG32 TRAFFIC_AS_ENABLE     : 1;
__REG32 TRAFFIC_JAM_AS_ENABLE : 1;
__REG32 APBXDMA_AS_ENABLE     : 1;
__REG32 APBHDMA_AS_ENABLE     : 1;
__REG32 PXP_AS_ENABLE         : 1;
__REG32 DCP_AS_ENABLE         : 1;
__REG32 BUSY                  : 1;
__REG32                       : 2;
} __hw_clkctrl_hbus_bits;

/* APBX Clock Control Register */
typedef struct {
__REG32 DIV                   :10;
__REG32                       :21;
__REG32 BUSY                  : 1;
} __hw_clkctrl_xbus_bits;

/* XTAL Clock Control Register */
typedef struct {
__REG32 DIV_UART              : 2;
__REG32                       :24;
__REG32 TIMROT_CLK32K_GATE    : 1;
__REG32 DIGCTRL_CLK1M_GATE    : 1;
__REG32 DRI_CLK24M_GATE       : 1;
__REG32 PWM_CLK24M_GATE       : 1;
__REG32 FILT_CLK24M_GATE      : 1;
__REG32 UART_CLK_GATE         : 1;
} __hw_clkctrl_xtal_bits;

/* PIX (LCDIF) Clock Control Register */
typedef struct {
__REG32 DIV                   :12;
__REG32 DIV_FRAC_EN           : 1;
__REG32                       :16;
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_pix_bits;

/* Synchronous Serial Port Clock Control Register */
typedef struct {
__REG32 DIV                   : 9;
__REG32                       :20;
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_ssp_bits;

/* General-Purpose Media Interface Clock Control Register */
typedef struct {
__REG32 DIV                   :10;
__REG32                       :19;
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_gpmi_bits;

/* SPDIF Clock Control Register */
typedef struct {
__REG32                       :31;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_spdif_bits;

/* EMI Clock Control Register */
typedef struct {
__REG32 DIV_EMI               : 6;
__REG32                       : 2;
__REG32 DIV_XTAL              : 4;
__REG32                       :14;
__REG32 BUSY_SYNC_MODE        : 1;
__REG32 BUSY_REF_CPU          : 1;
__REG32 BUSY_REF_EMI          : 1;
__REG32 BUSY_REF_XTAL         : 1;
__REG32 SYNC_MODE_EN          : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_emi_bits;

/* SAIF Clock Control Register */
typedef struct {
__REG32 DIV                   :16;
__REG32 DIV_FRAC_EN           : 1;
__REG32                       :12;
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_saif_bits;

/* TV Encode Clock Control Register */
typedef struct {
__REG32                       :30;
__REG32 CLK_TV_GATE           : 1;
__REG32 CLK_TV108M_GATE       : 1;
} __hw_clkctrl_tv_bits;

/* ETM Clock Control Register */
typedef struct {
__REG32 DIV                   : 6;
__REG32 DIV_FRAC_EN           : 1;
__REG32                       :22;
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
} __hw_clkctrl_etm_bits;

/* Fractional Clock Control Register */
typedef union {
  /*HW_CLKCTRL_FRAC*/
  struct {
  __REG32 CPUFRAC             : 6;
  __REG32 CPU_STABLE          : 1;
  __REG32 CLKGATECPU          : 1;
  __REG32 EMIFRAC             : 6;
  __REG32 EMI_STABLE          : 1;
  __REG32 CLKGATEEMI          : 1;
  __REG32 PIXFRAC             : 6;
  __REG32 PIX_STABLE          : 1;
  __REG32 CLKGATEPIX          : 1;
  __REG32 IOFRAC              : 6;
  __REG32 IO_STABLE           : 1;
  __REG32 CLKGATEIO           : 1;
  };
  
  struct {
    union {
      /*HW_CLKCTRL_CPUFRAC*/
      struct {
      __REG8  CPUFRAC         : 6;
      __REG8  CPU_STABLE      : 1;
      __REG8  CLKGATECPU      : 1;
      } __cpufrac_bit;
      __REG8  __cpufrac;
    };
    union {
      /*HW_CLKCTRL_EMIFRAC*/
      struct {
      __REG8  EMIFRAC         : 6;
      __REG8  EMI_STABLE      : 1;
      __REG8  CLKGATEEMI      : 1;
      } __emifrac_bit;
      __REG8  __emifrac;
    };
    union {
      /*HW_CLKCTRL_PIXFRAC*/
      struct {
      __REG8  PIXFRAC         : 6;
      __REG8  PIX_STABLE      : 1;
      __REG8  CLKGATEPIX      : 1;
      } __pixfrac_bit;
      __REG8  __pixfrac;
    };
    union {
      /*HW_CLKCTRL_IOFRAC*/
      struct {
      __REG8  IOFRAC          : 6;
      __REG8  IO_STABLE       : 1;
      __REG8  CLKGATEIO       : 1;
      } __iofrac_bit;
      __REG8  __iofrac;
    };
  };
} __hw_clkctrl_frac_bits;

/* Fractional Clock Control Register 1 */
typedef struct {
__REG32                       :30;
__REG32 VID_STABLE            : 1;
__REG32 CLKGATEVID            : 1;
} __hw_clkctrl_frac1_bits;

/* Clock Frequency Sequence Control Register */
typedef struct {
__REG32 BYPASS_SAIF           : 1;
__REG32 BYPASS_PIX            : 1;
__REG32                       : 1;
__REG32 BYPASS_IR             : 1;
__REG32 BYPASS_GPMI           : 1;
__REG32 BYPASS_SSP            : 1;
__REG32 BYPASS_EMI            : 1;
__REG32 BYPASS_CPU            : 1;
__REG32 BYPASS_ETM            : 1;
__REG32                       :23;
} __hw_clkctrl_clkseq_bits;

/* System Software Reset Register */
typedef struct {
__REG32 DIG                   : 1;
__REG32 CHIP                  : 1;
__REG32                       :30;
} __hw_clkctrl_reset_bits;

/* ClkCtrl Status Register */
typedef struct {
__REG32                       :30;
__REG32 CPU_LIMIT             : 2;
} __hw_clkctrl_status_bits;

/* ClkCtrl Version Register */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_clkctrl_version_bits;

/* Interrupt Collector Level Acknowledge Register (HW_ICOLL_LEVELACK) */
typedef struct {        
__REG32 IRQLEVELACK    : 4;
__REG32                :28;
} __hw_icoll_levelack_bits;

/* Interrupt Collector Control Register (HW_ICOLL_CTRL) */
typedef struct {        
__REG32                		:16;
__REG32 IRQ_FINAL_ENABLE	: 1;
__REG32 FIQ_FINAL_ENABLE	: 1;
__REG32 ARM_RSE_MODE			: 1;
__REG32 NO_NESTING				: 1;
__REG32 BYPASS_FSM				: 1;
__REG32 VECTOR_PITCH			: 3;
__REG32                		: 6;
__REG32 CLKGATE						: 1;
__REG32 SFTRST						: 1;
} __hw_icoll_ctrl_bits;

/* Interrupt Collector Status Register (HW_ICOLL_STAT) */
typedef struct {        
__REG32 VECTOR_NUMBER			: 7;
__REG32                		:25;
} __hw_icoll_stat_bits;

/* Interrupt Collector Raw Interrupt Input Register 0 (HW_ICOLL_RAW0) */
typedef struct {        
__REG32 RAW_IRQ0					: 1;
__REG32 RAW_IRQ1					: 1;
__REG32 RAW_IRQ2					: 1;
__REG32 RAW_IRQ3					: 1;
__REG32 RAW_IRQ4					: 1;
__REG32 RAW_IRQ5					: 1;
__REG32 RAW_IRQ6					: 1;
__REG32 RAW_IRQ7					: 1;
__REG32 RAW_IRQ8					: 1;
__REG32 RAW_IRQ9					: 1;
__REG32 RAW_IRQ10					: 1;
__REG32 RAW_IRQ11					: 1;
__REG32 RAW_IRQ12					: 1;
__REG32 RAW_IRQ13					: 1;
__REG32 RAW_IRQ14					: 1;
__REG32 RAW_IRQ15					: 1;
__REG32 RAW_IRQ16					: 1;
__REG32 RAW_IRQ17					: 1;
__REG32 RAW_IRQ18					: 1;
__REG32 RAW_IRQ19					: 1;
__REG32 RAW_IRQ20					: 1;
__REG32 RAW_IRQ21					: 1;
__REG32 RAW_IRQ22					: 1;
__REG32 RAW_IRQ23					: 1;
__REG32 RAW_IRQ24					: 1;
__REG32 RAW_IRQ25					: 1;
__REG32 RAW_IRQ26					: 1;
__REG32 RAW_IRQ27					: 1;
__REG32 RAW_IRQ28					: 1;
__REG32 RAW_IRQ29					: 1;
__REG32 RAW_IRQ30					: 1;
__REG32 RAW_IRQ31					: 1;
} __hw_icoll_raw0_bits;

/* Interrupt Collector Raw Interrupt Input Register 1 (HW_ICOLL_RAW1) */
typedef struct {        
__REG32 RAW_IRQ32					: 1;
__REG32 RAW_IRQ33					: 1;
__REG32 RAW_IRQ34					: 1;
__REG32 RAW_IRQ35					: 1;
__REG32 RAW_IRQ36					: 1;
__REG32 RAW_IRQ37					: 1;
__REG32 RAW_IRQ38					: 1;
__REG32 RAW_IRQ39					: 1;
__REG32 RAW_IRQ40					: 1;
__REG32 RAW_IRQ41					: 1;
__REG32 RAW_IRQ42					: 1;
__REG32 RAW_IRQ43					: 1;
__REG32 RAW_IRQ44					: 1;
__REG32 RAW_IRQ45					: 1;
__REG32 RAW_IRQ46					: 1;
__REG32 RAW_IRQ47					: 1;
__REG32 RAW_IRQ48					: 1;
__REG32 RAW_IRQ49					: 1;
__REG32 RAW_IRQ50					: 1;
__REG32 RAW_IRQ51					: 1;
__REG32 RAW_IRQ52					: 1;
__REG32 RAW_IRQ53					: 1;
__REG32 RAW_IRQ54					: 1;
__REG32 RAW_IRQ55					: 1;
__REG32 RAW_IRQ56					: 1;
__REG32 RAW_IRQ57					: 1;
__REG32 RAW_IRQ58					: 1;
__REG32 RAW_IRQ59					: 1;
__REG32 RAW_IRQ60					: 1;
__REG32 RAW_IRQ61					: 1;
__REG32 RAW_IRQ62					: 1;
__REG32 RAW_IRQ63					: 1;
} __hw_icoll_raw1_bits;

/* Interrupt Collector Raw Interrupt Input Register 2 (HW_ICOLL_RAW2) */
typedef struct {        
__REG32 RAW_IRQ64					: 1;
__REG32 RAW_IRQ65					: 1;
__REG32 RAW_IRQ66					: 1;
__REG32 RAW_IRQ67					: 1;
__REG32 RAW_IRQ68					: 1;
__REG32 RAW_IRQ69					: 1;
__REG32 RAW_IRQ70					: 1;
__REG32 RAW_IRQ71					: 1;
__REG32 RAW_IRQ72					: 1;
__REG32 RAW_IRQ73					: 1;
__REG32 RAW_IRQ74					: 1;
__REG32 RAW_IRQ75					: 1;
__REG32 RAW_IRQ76					: 1;
__REG32 RAW_IRQ77					: 1;
__REG32 RAW_IRQ78					: 1;
__REG32 RAW_IRQ79					: 1;
__REG32 RAW_IRQ80					: 1;
__REG32 RAW_IRQ81					: 1;
__REG32 RAW_IRQ82					: 1;
__REG32 RAW_IRQ83					: 1;
__REG32 RAW_IRQ84					: 1;
__REG32 RAW_IRQ85					: 1;
__REG32 RAW_IRQ86					: 1;
__REG32 RAW_IRQ87					: 1;
__REG32 RAW_IRQ88					: 1;
__REG32 RAW_IRQ89					: 1;
__REG32 RAW_IRQ90					: 1;
__REG32 RAW_IRQ91					: 1;
__REG32 RAW_IRQ92					: 1;
__REG32 RAW_IRQ93					: 1;
__REG32 RAW_IRQ94					: 1;
__REG32 RAW_IRQ95					: 1;
} __hw_icoll_raw2_bits;

/* Interrupt Collector Raw Interrupt Input Register 3 (HW_ICOLL_RAW3) */
typedef struct {        
__REG32 RAW_IRQ96					: 1;
__REG32 RAW_IRQ97					: 1;
__REG32 RAW_IRQ98					: 1;
__REG32 RAW_IRQ99					: 1;
__REG32 RAW_IRQ100				: 1;
__REG32 RAW_IRQ101				: 1;
__REG32 RAW_IRQ102				: 1;
__REG32 RAW_IRQ103				: 1;
__REG32 RAW_IRQ104				: 1;
__REG32 RAW_IRQ105				: 1;
__REG32 RAW_IRQ106				: 1;
__REG32 RAW_IRQ107				: 1;
__REG32 RAW_IRQ108				: 1;
__REG32 RAW_IRQ109				: 1;
__REG32 RAW_IRQ110				: 1;
__REG32 RAW_IRQ111				: 1;
__REG32 RAW_IRQ112				: 1;
__REG32 RAW_IRQ113				: 1;
__REG32 RAW_IRQ114				: 1;
__REG32 RAW_IRQ115				: 1;
__REG32 RAW_IRQ116				: 1;
__REG32 RAW_IRQ117				: 1;
__REG32 RAW_IRQ118				: 1;
__REG32 RAW_IRQ119				: 1;
__REG32 RAW_IRQ120				: 1;
__REG32 RAW_IRQ121				: 1;
__REG32 RAW_IRQ122				: 1;
__REG32 RAW_IRQ123				: 1;
__REG32 RAW_IRQ124				: 1;
__REG32 RAW_IRQ125				: 1;
__REG32 RAW_IRQ126				: 1;
__REG32 RAW_IRQ127				: 1;
} __hw_icoll_raw3_bits;

/* Interrupt Collector Interrupt Register 0-127 (HW_ICOLL_INTERRUPT0-127) */
typedef struct {        
__REG32 PRIORITY					: 2;
__REG32 ENABLE						: 1;
__REG32 SOFTIRQ						: 1;
__REG32 ENFIQ							: 1;
__REG32 									:27;
} __hw_icoll_interrupt_bits;

/* Interrupt Collector Debug Register 0 (HW_ICOLL_DEBUG) */
typedef struct {        
__REG32 VECTOR_FSM 				:10;
__REG32 									: 6;
__REG32 IRQ								: 1;
__REG32 FIQ								: 1;
__REG32 									: 2;
__REG32 REQUESTS_BY_LEVEL	: 4;
__REG32 LEVEL_REQUESTS		: 4;
__REG32 INSERVICE					: 4;
} __hw_icoll_debug_bits;

/* Interrupt Collector Debug Flag Register (HW_ICOLL_DBGFLAG) */
typedef struct {        
__REG32 FLAG			 				:16;
__REG32 									:16;
} __hw_icoll_dbgflag_bits;

/* Interrupt Collector Debug Read Request Register 0 (HW_ICOLL_DBGREQUEST0) */
typedef struct {        
__REG32 BITS0						: 1;
__REG32 BITS1						: 1;
__REG32 BITS2						: 1;
__REG32 BITS3						: 1;
__REG32 BITS4						: 1;
__REG32 BITS5						: 1;
__REG32 BITS6						: 1;
__REG32 BITS7						: 1;
__REG32 BITS8						: 1;
__REG32 BITS9						: 1;
__REG32 BITS10					: 1;
__REG32 BITS11					: 1;
__REG32 BITS12					: 1;
__REG32 BITS13					: 1;
__REG32 BITS14					: 1;
__REG32 BITS15					: 1;
__REG32 BITS16					: 1;
__REG32 BITS17					: 1;
__REG32 BITS18					: 1;
__REG32 BITS19					: 1;
__REG32 BITS20					: 1;
__REG32 BITS21					: 1;
__REG32 BITS22					: 1;
__REG32 BITS23					: 1;
__REG32 BITS24					: 1;
__REG32 BITS25					: 1;
__REG32 BITS26					: 1;
__REG32 BITS27					: 1;
__REG32 BITS28					: 1;
__REG32 BITS29					: 1;
__REG32 BITS30					: 1;
__REG32 BITS31					: 1;
} __hw_icoll_dbgrequest1_bits;

/* Interrupt Collector Debug Read Request Register 1 (HW_ICOLL_DBGREQUEST1) */
typedef struct {        
__REG32 BITS32					: 1;
__REG32 BITS33					: 1;
__REG32 BITS34					: 1;
__REG32 BITS35					: 1;
__REG32 BITS36					: 1;
__REG32 BITS37					: 1;
__REG32 BITS38					: 1;
__REG32 BITS39					: 1;
__REG32 BITS40					: 1;
__REG32 BITS41					: 1;
__REG32 BITS42					: 1;
__REG32 BITS43					: 1;
__REG32 BITS44					: 1;
__REG32 BITS45					: 1;
__REG32 BITS46					: 1;
__REG32 BITS47					: 1;
__REG32 BITS48					: 1;
__REG32 BITS49					: 1;
__REG32 BITS50					: 1;
__REG32 BITS51					: 1;
__REG32 BITS52					: 1;
__REG32 BITS53					: 1;
__REG32 BITS54					: 1;
__REG32 BITS55					: 1;
__REG32 BITS56					: 1;
__REG32 BITS57					: 1;
__REG32 BITS58					: 1;
__REG32 BITS59					: 1;
__REG32 BITS60					: 1;
__REG32 BITS61					: 1;
__REG32 BITS62					: 1;
__REG32 BITS63					: 1;
} __hw_icoll_dbgrequest0_bits;

/* Interrupt Collector Debug Read Request Register 2 (HW_ICOLL_DBGREQUEST2) */

typedef struct {        
__REG32 BITS64					: 1;
__REG32 BITS65					: 1;
__REG32 BITS66					: 1;
__REG32 BITS67					: 1;
__REG32 BITS68					: 1;
__REG32 BITS69					: 1;
__REG32 BITS70					: 1;
__REG32 BITS71					: 1;
__REG32 BITS72					: 1;
__REG32 BITS73					: 1;
__REG32 BITS74					: 1;
__REG32 BITS75					: 1;
__REG32 BITS76					: 1;
__REG32 BITS77					: 1;
__REG32 BITS78					: 1;
__REG32 BITS79					: 1;
__REG32 BITS80					: 1;
__REG32 BITS81					: 1;
__REG32 BITS82					: 1;
__REG32 BITS83					: 1;
__REG32 BITS84					: 1;
__REG32 BITS85					: 1;
__REG32 BITS86					: 1;
__REG32 BITS87					: 1;
__REG32 BITS88					: 1;
__REG32 BITS89					: 1;
__REG32 BITS90					: 1;
__REG32 BITS91					: 1;
__REG32 BITS92					: 1;
__REG32 BITS93					: 1;
__REG32 BITS94					: 1;
__REG32 BITS95					: 1;
} __hw_icoll_dbgrequest2_bits;

/* Interrupt Collector Debug Read Request Register 3 (HW_ICOLL_DBGREQUEST3) */
typedef struct {        
__REG32 BITS96					: 1;
__REG32 BITS97					: 1;
__REG32 BITS98					: 1;
__REG32 BITS99					: 1;
__REG32 BITS100					: 1;
__REG32 BITS101					: 1;
__REG32 BITS102					: 1;
__REG32 BITS103					: 1;
__REG32 BITS104					: 1;
__REG32 BITS105					: 1;
__REG32 BITS106					: 1;
__REG32 BITS107					: 1;
__REG32 BITS108					: 1;
__REG32 BITS109					: 1;
__REG32 BITS110					: 1;
__REG32 BITS111					: 1;
__REG32 BITS112					: 1;
__REG32 BITS113					: 1;
__REG32 BITS114					: 1;
__REG32 BITS115					: 1;
__REG32 BITS116					: 1;
__REG32 BITS117					: 1;
__REG32 BITS118					: 1;
__REG32 BITS119					: 1;
__REG32 BITS120					: 1;
__REG32 BITS121					: 1;
__REG32 BITS122					: 1;
__REG32 BITS123					: 1;
__REG32 BITS124					: 1;
__REG32 BITS125					: 1;
__REG32 BITS126					: 1;
__REG32 BITS127					: 1;
} __hw_icoll_dbgrequest3_bits;

/* Interrupt Collector Version Register */
typedef struct {        
__REG32 STEP						:16;
__REG32 MINOR						: 8;
__REG32 MAJOR						: 8;
} __hw_icoll_version_bits;

/* DIGCTL Control Register */
typedef struct {        
__REG32 LATCH_ENTROPY       : 1;
__REG32 JTAG_SHIELD 		    : 1;
__REG32 USB_CLKGATE			    : 1;
__REG32 DEBUG_DISABLE 	    : 1;
__REG32 TRAP_ENABLE   	    : 1;
__REG32 TRAP_IN_RANGE  	    : 1;
__REG32 USE_SERIAL_JTAG	    : 1;
__REG32 SY_CLKGATE          : 1;
__REG32 SY_SFTRST           : 1;
__REG32 SY_ENDIAN           : 1;
__REG32                     : 1;
__REG32 SAIF_ALT_BITCLK_SEL : 1;
__REG32 SAIF_CLKMST_SEL     : 1;
__REG32 SAIF_CLKMUX_SEL     : 2;
__REG32 SAIF_LOOPBACK       : 1;
__REG32 UART_LOOPBACK       : 1;
__REG32 ARM_BIST_START      : 1;
__REG32 DIGITAL_TESTMODE    : 1;
__REG32 ANALOG_TESTMODE     : 1;
__REG32 USB_TESTMODE        : 1;
__REG32 ARM_BIST_CLKEN      : 1;
__REG32 DCP_BIST_START      : 1;
__REG32 DCP_BIST_CLKEN      : 1;
__REG32 LCD_BIST_START      : 1;
__REG32 LCD_BIST_CLKEN      : 1;
__REG32 CACHE_BIST_TMODE    : 1;
__REG32                     : 2;
__REG32 TRAP_IRQ            : 1;
__REG32 XTAL24M_GATE        : 1;
__REG32                     : 1;
} __hw_digctl_ctrl_bits;

/* DIGCTL Status Register */
typedef struct {        
__REG32 WRITTEN             : 1;
__REG32 PACKAGE_TYPE  	    : 3;
__REG32 JTAG_IN_USE			    : 1;
__REG32 LCD_BIST_DONE 	    : 1;
__REG32 LCD_BIST_PASS 	    : 1;
__REG32 LCD_BIST_FAIL  	    : 1;
__REG32 DCP_BIST_DONE 	    : 1;
__REG32 DCP_BIST_PASS       : 1;
__REG32 DCP_BIST_FAIL       : 1;
__REG32                     :17;
__REG32 USB_DEVICE_PRESENT  : 1;
__REG32 USB_HOST_PRESENT    : 1;
__REG32 USB_OTG_PRESENT     : 1;
__REG32 USB_HS_PRESENT      : 1;
} __hw_digctl_status_bits;

/* On-Chip RAM Control Register */
typedef struct {        
__REG32 RAM_REPAIR_EN       : 1;
__REG32               	    : 7;
__REG32 SPEED_SELECT  	    : 4;
__REG32                     :20;
} __hw_digctl_ramctrl_bits;

/* On-Chip RAM Repair Address Register */
typedef struct {        
__REG32 ADDR                :16;
__REG32                     :16;
} __hw_digctl_ramrepair_bits;

/* On-Chip ROM Control Register */
typedef struct {        
__REG32 RD_MARGIN           : 4;
__REG32                     :28;
} __hw_digctl_romctrl_bits;

/* SJTAG Debug Register */
typedef struct {        
__REG32 SJTAG_DEBUG_OE      : 1;
__REG32 SJTAG_DEBUG_DATA    : 1;
__REG32 SJTAG_PIN_STATE     : 1;
__REG32 ACTIVE              : 1;
__REG32 DELAYED_ACTIVE      : 4;
__REG32 SJTAG_MODE    	    : 1;
__REG32 SJTAG_TDI     	    : 1;
__REG32 SJTAG_TDO     	    : 1;
__REG32               	    : 5;
__REG32 SJTAG_STATE   	    :11;
__REG32                     : 5;
} __hw_digctl_sjtagdbg_bits;

/* SRAM BIST Control and Status */
typedef struct {        
__REG32 START               : 1;
__REG32 DONE                : 1;
__REG32 PASS                : 1;
__REG32 FAIL                : 1;
__REG32                     : 4;
__REG32 BIST_CLKEN    	    : 1;
__REG32 BIST_DATA_CHANGE    : 1;
__REG32 BIST_DEBUG_MODE	    : 1;
__REG32                     :21;
} __hw_digctl_ocram_bist_csr_bits;

/* SRAM Status Register 8 */
typedef struct {        
__REG32 FAILADDR00          :13;
__REG32                     : 3;
__REG32 FAILADDR01          :13;
__REG32                     : 3;
} __hw_digctl_ocram_status8_bits;

/* SRAM Status Register 9 */
typedef struct {        
__REG32 FAILADDR10          :13;
__REG32                     : 3;
__REG32 FAILADDR11          :13;
__REG32                     : 3;
} __hw_digctl_ocram_status9_bits;

/* SRAM Status Register 10 */
typedef struct {        
__REG32 FAILADDR20          :13;
__REG32                     : 3;
__REG32 FAILADDR21          :13;
__REG32                     : 3;
} __hw_digctl_ocram_status10_bits;

/* SRAM Status Register 11 */
typedef struct {        
__REG32 FAILADDR30          :13;
__REG32                     : 3;
__REG32 FAILADDR31          :13;
__REG32                     : 3;
} __hw_digctl_ocram_status11_bits;

/* SRAM Status Register 12 */
typedef struct {        
__REG32 FAILSTATE00         : 4;
__REG32                     : 4;
__REG32 FAILSTATE01         : 4;
__REG32                     : 4;
__REG32 FAILSTATE10         : 4;
__REG32                     : 4;
__REG32 FAILSTATE11         : 4;
__REG32                     : 4;
} __hw_digctl_ocram_status12_bits;

/* SRAM Status Register 13 */
typedef struct {        
__REG32 FAILSTATE20         : 4;
__REG32                     : 4;
__REG32 FAILSTATE21         : 4;
__REG32                     : 4;
__REG32 FAILSTATE30         : 4;
__REG32                     : 4;
__REG32 FAILSTATE31         : 4;
__REG32                     : 4;
} __hw_digctl_ocram_status13_bits;

/* Digital Control ARM Cache Register Description */
typedef struct {        
__REG32 ITAG_SS             : 2;
__REG32                     : 2;
__REG32 DTAG_SS             : 2;
__REG32                     : 2;
__REG32 CACHE_SS            : 2;
__REG32                     : 2;
__REG32 DRTY_SS             : 2;
__REG32                     : 2;
__REG32 VALID_SS            : 2;
__REG32                     :14;
} __hw_digctl_armcache_bits;

/* Digital Control Chip Revision Register */
typedef struct {        
__REG32 REVISION            : 8;
__REG32                     : 8;
__REG32 PRODUCT_CODE        :16;
} __hw_digctl_chipid_bits;

/* AHB Statistics Control Register */
typedef struct {        
__REG32 L0_MASTER_SELECT    : 4;
__REG32                     : 4;
__REG32 L1_MASTER_SELECT    : 4;
__REG32                     : 4;
__REG32 L2_MASTER_SELECT    : 4;
__REG32                     : 4;
__REG32 L3_MASTER_SELECT    : 4;
__REG32                     : 4;
} __hw_digctl_ahb_stats_select_bits;

/* EMI CLK/CLKN Delay Adjustment Register */
typedef struct {        
__REG32 NUM_TAPS            : 5;
__REG32                     :27;
} __hw_digctl_emiclk_delay_bits;

/* OTP Controller Control Register */
typedef struct {        
__REG32 ADDR                : 5;
__REG32                     : 3;
__REG32 BUSY                : 1;
__REG32 ERROR               : 1;
__REG32                     : 2;
__REG32 RD_BANK_OPEN        : 1;
__REG32 RELOAD_SHADOWS      : 1;
__REG32                     : 2;
__REG32 WR_UNLOCK           :16;
} __hw_ocotp_ctrl_bits;

/* Customer Capability Shadow Register */
typedef struct {        
__REG32                         : 1;
__REG32 RTC_XTAL_32000_PRESENT  : 1;
__REG32 TC_XTAL_32768_PRESENT   : 1;
__REG32 USE_PARALLEL_JTAG       : 1;
__REG32 ENABLE_SJTAG_12MA_DRIVE : 1;
__REG32                         :25;
__REG32 CUST_DISABLE_JANUSDRM10 : 1;
__REG32 CUST_DISABLE_WMADRM9    : 1;
} __hw_ocotp_custcap_bits;

/* LOCK Shadow Register OTP */
typedef struct {        
__REG32 CUST0                   : 1;
__REG32 CUST1                   : 1;
__REG32 CUST2                   : 1;
__REG32 CUST3                   : 1;
__REG32 CRYPTOKEY               : 1;
__REG32 CRYPTODCP               : 1;
__REG32 HWSW_SHADOW             : 1;
__REG32 CUSTCAP_SHADOW          : 1;
__REG32 HWSW                    : 1;
__REG32 CUSTCAP                 : 1;
__REG32 ROM_SHADOW              : 1;
__REG32 UNALLOCATED             : 5;
__REG32 UN0                     : 1;
__REG32 UN1                     : 1;
__REG32 UN2                     : 1;
__REG32 OPS                     : 1;
__REG32 PIN                     : 1;
__REG32 CRYPTOKEY_ALT           : 1;
__REG32 CRYPTODCP_ALT           : 1;
__REG32 HWSW_SHADOW_ALT         : 1;
__REG32 ROM0                    : 1;
__REG32 ROM1                    : 1;
__REG32 ROM2                    : 1;
__REG32 ROM3                    : 1;
__REG32 ROM4                    : 1;
__REG32 ROM5                    : 1;
__REG32 ROM6                    : 1;
__REG32 ROM7                    : 1;
} __hw_ocotp_lock_bits;

/* Shadow Register for OTP Bank3 Word0 (ROM Use 0) */
typedef struct {        
__REG32                             : 3;
__REG32 SD_MBR_BOOT                 : 1;
__REG32 ENABLE_UNENCRYPTED_BOOT     : 1;
__REG32 ENABLE_USB_BOOT_SERIAL_NUM  : 1;
__REG32 DISABLE_SPI_NOR_FAST_READ   : 1;
__REG32                             : 1;
__REG32 SSP_SCK_INDEX               : 4;
__REG32 SD_BUS_WIDTH                : 2;
__REG32 SD_POWER_UP_DELAY           : 6;
__REG32 SD_POWER_GATE_GPIO          : 2;
__REG32 USE_PARALLEL_JTAG           : 1;
__REG32 ENABLE_PJTAG_12MA_DRIVE     : 1;
__REG32 BOOT_MODE                   : 8;
} __hw_ocotp_rom0_bits;

/* Shadow Register for OTP Bank3 Word1 (ROM Use 1) */
typedef struct {        
__REG32 NUMBER_OF_NANDS             : 3;
__REG32                             : 5;
__REG32 BOOT_SEARCH_COUNT           : 4;
__REG32 USE_ALT_SSP1_DATA4_7        : 1;
__REG32 SD_INIT_SEQ_1_DISABLE       : 1;
__REG32 SD_CMD0_DISABLE             : 1;
__REG32 SD_INIT_SEQ_2_ENABLE        : 1;
__REG32 SD_INCREASE_INIT_SEQ_TIME   : 1;
__REG32 SSP1_EXT_PULLUP             : 1;
__REG32 SSP2_EXT_PULLUP             : 1;
__REG32 UNTOUCH_INTERNAL_SSP_PULLUP : 1;
__REG32 ENABLE_NAND0_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND1_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND2_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND3_CE_RDY_PULLUP  : 1;
__REG32 USE_ALT_GPMI_CE2            : 1;
__REG32 USE_ALT_GPMI_RDY2           : 1;
__REG32 USE_ALT_GPMI_CE3            : 2;
__REG32 USE_ALT_GPMI_RDY3           : 2;
__REG32                             : 2;
} __hw_ocotp_rom1_bits;

/* Shadow Register for OTP Bank3 Word2 (ROM Use 2) */
typedef struct {        
__REG32 USB_PID                     :16;
__REG32 USB_VID                     :16;
} __hw_ocotp_rom2_bits;

/* Shadow Register for OTP Bank3 Word7 (ROM Use 7) */
typedef struct {        
__REG32 ENABLE_PIN_BOOT_CHECK       : 1;
__REG32                             : 1;
__REG32 ENABLE_ARM_ICACHE           : 1;
__REG32 I2C_USE_400KHZ              : 1;
__REG32                             : 4;
__REG32 ENABLE_SSP_12MA_DRIVE       : 1;
__REG32                             :23;
} __hw_ocotp_rom7_bits;

/* OTP Controller Version Register */
typedef struct {        
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __hw_ocotp_version_bits;

/* ID Register */
typedef struct{
__REG32 ID        : 6;
__REG32           : 2;
__REG32 NID       : 6;
__REG32           : 2;
__REG32 TAG       : 5;
__REG32 REVISION  : 4;
__REG32 VERSION   : 4;
__REG32 CIVERSION : 3;
} __hw_usbctrl_id_bits;

/* HWGENERAL—General Hardware Parameters */
typedef struct{
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 2;
__REG32           :21;
} __hw_usbctrl_hwgeneral_bits;

/* HWHOST—Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __hw_usbctrl_hwhost_bits;

/* HWDEVICE—Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __hw_usbctrl_hwdevice_bits;

/* HWTXBUF—TX Buffer Hardware Parameters */
typedef struct{
__REG32 TXBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __hw_usbctrl_hwtxbuf_bits;

/* HWRXBUF—RX Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __hw_usbctrl_hwrxbuf_bits;

/* HCSPARAMS—Host Control Structural Parameters */
typedef struct{
__REG32 N_PORTS   : 4;
__REG32 PPC       : 1;
__REG32           : 3;
__REG32 N_PCC     : 4;
__REG32 N_CC      : 4;
__REG32 PI        : 1;
__REG32           : 3;
__REG32 N_PTT     : 4;
__REG32 N_TT      : 4;
__REG32           : 4;
} __hw_usbctrl_hcsparams_bits;

/* HCCPARAMS—Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __hw_usbctrl_hccparams_bits;

/* USB Device Interface Version Number */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __hw_usbctrl_dciversion_bits;

/* HCCPARAMS—Host Control Capability Parameters */
typedef struct{
__REG32 DEN       : 5;
__REG32           : 2;
__REG32 DC        : 1;
__REG32 HC        : 1;
__REG32           :23;
} __hw_usbctrl_dccparams_bits;

/* SBUSFG - control for the system bus interface */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __hw_usbctrl_sbuscfg_bits;

/* USB General Purpose Timer #0 Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __hw_usbctrl_gptimerxld_bits;

/* USB General Purpose Timer #0 Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __hw_usbctrl_gptimerxctrl_bits;

/* USB Command Register (USBCMD) */
typedef struct{
__REG32 RS        : 1;
__REG32 RST       : 1;
__REG32 FS0       : 1;
__REG32 FS1       : 1;
__REG32 PSE       : 1;
__REG32 ASE       : 1;
__REG32 IAA       : 1;
__REG32 LR        : 1;
__REG32 ASP0      : 1;
__REG32 ASP1      : 1;
__REG32           : 1;
__REG32 ASPE      : 1;
__REG32           : 1;
__REG32 SUTW      : 1;
__REG32 ADTW      : 1;
__REG32 FS2       : 1;
__REG32 ITC       : 8;
__REG32           : 8;
} __hw_usbctrl_usbcmd_bits;

/* USBSTS—USB Status */
typedef struct{
__REG32 UI        : 1;
__REG32 UEI       : 1;
__REG32 PCI       : 1;
__REG32 FRI       : 1;
__REG32 SEI       : 1;
__REG32 AAI       : 1;
__REG32 URI       : 1;
__REG32 SRI       : 1;
__REG32 SLI       : 1;
__REG32           : 1;
__REG32 ULPII     : 1;
__REG32           : 1;
__REG32 HCH       : 1;
__REG32 RCL       : 1;
__REG32 PS        : 1;
__REG32 AS        : 1;
__REG32 NAKI      : 1;
__REG32           : 1;
__REG32 UAI       : 1;
__REG32 UPI       : 1;
__REG32           : 4;
__REG32 TI0       : 1;
__REG32 TI1       : 1;
__REG32           : 6;
} __hw_usbctrl_usbsts_bits;

/* USBINTR—USB Interrupt Enable */
typedef struct{
__REG32 UE        : 1;
__REG32 UEE       : 1;
__REG32 PCE       : 1;
__REG32 FRE       : 1;
__REG32 SEE       : 1;
__REG32 AAE       : 1;
__REG32 URE       : 1;
__REG32 SRE       : 1;
__REG32 SLE       : 1;
__REG32           : 1;
__REG32 ULPIE     : 1;
__REG32           : 5;
__REG32 NAKE      : 1;
__REG32           : 1;
__REG32 UAIE      : 1;
__REG32 UPIE      : 1;
__REG32           : 4;
__REG32 TIE0      : 1;
__REG32 TIE1      : 1;
__REG32           : 6;
} __hw_usbctrl_usbintr_bits;

/* FRINDEX—USB Frame Index */
typedef struct{
__REG32 FRINDEX   :14;
__REG32           :18;
} __hw_usbctrl_frindex_bits;

/* PERIODICLISTBASE—Host Controller Frame List Base Address */
/* DEVICEADDR—Device Controller USB Device Address */
typedef union {
  /* HW_USBCTRL_PERIODICLISTBASE */
  struct{
    __REG32           :12;
    __REG32 PERBASE   :20;
  };
  /* HW_USBCTRL_DEVICEADDR */
  struct{
    __REG32           :24;
    __REG32 USBADRA   : 1;
    __REG32 USBADR    : 7;
  };
} __hw_usbctrl_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
  /* HW_USBCTRL_ASYNCLISTADDR */
  struct{
    __REG32           : 5;
    __REG32 ASYBASE   :27;
  };
  /* HW_USBCTRL_ENDPOINTLISTADDR */
  struct{
    __REG32           :11;
    __REG32 EPBASE    :21;
  };
} __hw_usbctrl_asynclistaddr_bits;

/* Embedded TT Asynchronous Buffer Status and Control Register */
typedef struct{
__REG32           :24;
__REG32 TTHA      : 7;
__REG32           : 1;
} __hw_usbctrl_ttctrl_bits;

/* Programmable Burst Size Register */
typedef struct{
__REG32 RXPBURST  : 8;
__REG32 TXPBURST  : 8;
__REG32           :16;
} __hw_usbctrl_burstsize_bits;

/* TXFILLTUNING Register */
typedef struct{
__REG32 TXSCHOH     : 7;
__REG32             : 1;
__REG32 TXSCHHEALTH : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __hw_usbctrl_txfilltuning_bits;

/* Inter-Chip Control Register */
typedef struct{
__REG32 IC_VDD      : 3;
__REG32 IC_ENABLE   : 1;
__REG32             :28;
} __hw_usbctrl_ic_usb_bits;

/* ULPI VIEWPORT */
typedef struct{
__REG32 ULPIDATWR : 8;
__REG32 ULPIDATRD : 8;
__REG32 ULPIADDR  : 8;
__REG32 ULPIPORT  : 3;
__REG32 ULPISS    : 1;
__REG32           : 1;
__REG32 ULPIRW    : 1;
__REG32 ULPIRUN   : 1;
__REG32 ULPIWU    : 1;
} __hw_usbctrl_ulpi_bits;

/* Endpoint NAK register */
typedef struct{
__REG32 EPRN0     : 1;
__REG32 EPRN1     : 1;
__REG32 EPRN2     : 1;
__REG32 EPRN3     : 1;
__REG32 EPRN4     : 1;
__REG32           :11;
__REG32 EPTN0     : 1;
__REG32 EPTN1     : 1;
__REG32 EPTN2     : 1;
__REG32 EPTN3     : 1;
__REG32 EPTN4     : 1;
__REG32           :11;
} __hw_usbctrl_endptnak_bits;

/*Endpoint NAK Enable Register */
typedef struct{
__REG32 EPRNE0      : 1;
__REG32 EPRNE1      : 1;
__REG32 EPRNE2      : 1;
__REG32 EPRNE3      : 1;
__REG32 EPRNE4      : 1;
__REG32             :11;
__REG32 EPTNE0      : 1;
__REG32 EPTNE1      : 1;
__REG32 EPTNE2      : 1;
__REG32 EPTNE3      : 1;
__REG32 EPTNE4      : 1;
__REG32             :11;
} __hw_usbctrl_endptnaken_bits;

/* PORTSCx—Port Status Control 1 */
typedef struct{
__REG32 CCS       : 1;
__REG32 CSC       : 1;
__REG32 PE        : 1;
__REG32 PEC       : 1;
__REG32 OCA       : 1;
__REG32 OCC       : 1;
__REG32 FPR       : 1;
__REG32 SUSP      : 1;
__REG32 PR        : 1;
__REG32 HSP       : 1;
__REG32 LS        : 2;
__REG32 PP        : 1;
__REG32 PO        : 1;
__REG32 PIC       : 2;
__REG32 PTC       : 4;
__REG32 WKCN      : 1;
__REG32 WKDS      : 1;
__REG32 WKOC      : 1;
__REG32 PHCD      : 1;
__REG32 PFSC      : 1;
__REG32           : 1;
__REG32 PSPD      : 2;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __hw_usbctrl_portsc_bits;

/* OTGSC—OTG Status Control */
typedef struct{
__REG32 VD        : 1;
__REG32 VC        : 1;
__REG32 HAAR      : 1;
__REG32 OT        : 1;
__REG32 DP        : 1;
__REG32 IDPU      : 1;
__REG32 HADP      : 1;
__REG32 HABA      : 1;
__REG32 ID        : 1;
__REG32 AVV       : 1;
__REG32 ASV       : 1;
__REG32 BSV       : 1;
__REG32 BSE       : 1;
__REG32 ONEMST    : 1;
__REG32 DPS       : 1;
__REG32           : 1;
__REG32 IDIS      : 1;
__REG32 AVVIS     : 1;
__REG32 ASVIS     : 1;
__REG32 BSVIS     : 1;
__REG32 BSEIS     : 1;
__REG32 ONEMSS    : 1;
__REG32 DPIS      : 1;
__REG32           : 1;
__REG32 IDIE      : 1;
__REG32 AVVIE     : 1;
__REG32 ASVIE     : 1;
__REG32 BSVIE     : 1;
__REG32 BSEIE     : 1;
__REG32 ONEMSE    : 1;
__REG32 DPIE      : 1;
__REG32           : 1;
} __hw_usbctrl_otgsc_bits;

/* USBMODE—USB Device Mode */
typedef struct{
__REG32 CM        : 2;
__REG32 ES        : 1;
__REG32 SLOM      : 1;
__REG32 SDIS      : 1;
__REG32 VBPS      : 1;
__REG32           :26;
} __hw_usbctrl_usbmode_bits;

/* ENDPTSETUPSTAT—Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT0   : 1;
__REG32 ENDPTSETUPSTAT1   : 1;
__REG32 ENDPTSETUPSTAT2   : 1;
__REG32 ENDPTSETUPSTAT3   : 1;
__REG32 ENDPTSETUPSTAT4   : 1;
__REG32                   :27;
} __hw_usbctrl_endptsetupstat_bits;

/* ENDPTPRIME—Endpoint Initialization */
typedef struct{
__REG32 PERB0       : 1;
__REG32 PERB1       : 1;
__REG32 PERB2       : 1;
__REG32 PERB3       : 1;
__REG32 PERB4       : 1;
__REG32             :11;
__REG32 PETB0       : 1;
__REG32 PETB1       : 1;
__REG32 PETB2       : 1;
__REG32 PETB3       : 1;
__REG32 PETB4       : 1;
__REG32             :11;
} __hw_usbctrl_endptprime_bits;

/* ENDPTFLUSH—Endpoint De-Initialize */
typedef struct{
__REG32 FERB0       : 1;
__REG32 FERB1       : 1;
__REG32 FERB2       : 1;
__REG32 FERB3       : 1;
__REG32 FERB4       : 1;
__REG32             :11;
__REG32 FETB0       : 1;
__REG32 FETB1       : 1;
__REG32 FETB2       : 1;
__REG32 FETB3       : 1;
__REG32 FETB4       : 1;
__REG32             :11;
} __hw_usbctrl_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR0       : 1;
__REG32 ERBR1       : 1;
__REG32 ERBR2       : 1;
__REG32 ERBR3       : 1;
__REG32 ERBR4       : 1;
__REG32             :11;
__REG32 ETBR0       : 1;
__REG32 ETBR1       : 1;
__REG32 ETBR2       : 1;
__REG32 ETBR3       : 1;
__REG32 ETBR4       : 1;
__REG32             :11;
} __hw_usbctrl_endptstat_bits;

/* ENDPTCOMPLETE—Endpoint Compete */
typedef struct{
__REG32 ERCE0       : 1;
__REG32 ERCE1       : 1;
__REG32 ERCE2       : 1;
__REG32 ERCE3       : 1;
__REG32 ERCE4       : 1;
__REG32             :11;
__REG32 ETCE0       : 1;
__REG32 ETCE1       : 1;
__REG32 ETCE2       : 1;
__REG32 ETCE3       : 1;
__REG32 ETCE4       : 1;
__REG32             :11;
} __hw_usbctrl_endptcomplete_bits;

/* Endpoint Control 0 Register (ENDPTCTRL0) */
typedef struct{
__REG32 RXS             : 1;
__REG32                 : 1;
__REG32 RXT             : 2;
__REG32                 : 3;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32                 : 1;
__REG32 TXT             : 2;
__REG32                 : 3;
__REG32 TXE             : 1;
__REG32                 : 8;
} __hw_usbctrl_endptctrl0_bits;

/* Endpoint Control x Registers (ENDPTCTRLx, x = 1…4) */
typedef struct{
__REG32 RXS             : 1;
__REG32 RXD             : 1;
__REG32 RXT             : 2;
__REG32                 : 1;
__REG32 RXI             : 1;
__REG32 RXR             : 1;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32 TXD             : 1;
__REG32 TXT             : 2;
__REG32                 : 1;
__REG32 TXI             : 1;
__REG32 TXR             : 1;
__REG32 TXE             : 1;
__REG32                 : 8;
} __hw_usbctrl_endptctrl_bits;

/* USB PHY Power-Down Register */
typedef struct{
__REG32                 :10;
__REG32 TXPWDFS         : 1;
__REG32 TXPWDIBIAS      : 1;
__REG32 TXPWDV2I        : 1;
__REG32                 : 4;
__REG32 RXPWDENV        : 1;
__REG32 RXPWD1PT1       : 1;
__REG32 RXPWDDIFF       : 1;
__REG32 RXPWDRX         : 1;
__REG32                 :11;
} __hw_usbphy_pwd_bits;

/* USB PHY Transmitter Control Register */
typedef struct{
__REG32 D_CAL                 : 4;
__REG32                       : 4;
__REG32 TXCAL45DN             : 4;
__REG32                       : 1;
__REG32 TXENCAL45DN           : 1;
__REG32                       : 2;
__REG32 TXCAL45DP             : 4;
__REG32                       : 1;
__REG32 TXENCAL45DP           : 1;
__REG32                       : 2;
__REG32 USBPHY_TX_SYNC_MUX    : 1;
__REG32 USBPHY_TX_SYNC_INVERT : 1;
__REG32 USBPHY_TX_EDGECTRL    : 3;
__REG32                       : 3;
} __hw_usbphy_tx_bits;

/* USB PHY Receiver Control Register */
typedef struct{
__REG32 ENVADJ                : 3;
__REG32                       : 1;
__REG32 DISCONADJ             : 3;
__REG32                       :15;
__REG32 RXDBYPASS             : 1;
__REG32                       : 9;
} __hw_usbphy_rx_bits;

/* USB PHY General Control Register */
typedef struct{
__REG32                       : 1;
__REG32 ENHOSTDISCONDETECT    : 1;
__REG32 ENIRQHOSTDISCON       : 1;
__REG32 HOSTDISCONDETECT_IRQ  : 1;
__REG32 ENDEVPLUGINDETECT     : 1;
__REG32 DEVPLUGIN_POLARITY    : 1;
__REG32                       : 1;
__REG32 ENOTGIDDETECT         : 1;
__REG32                       : 1;
__REG32 ENIRQRESUMEDETECT     : 1;
__REG32 RESUME_IRQ            : 1;
__REG32 ENIRQDEVPLUGIN        : 1;
__REG32 DEVPLUGIN_IRQ         : 1;
__REG32 DATA_ON_LRADC         : 1;
__REG32                       :14;
__REG32 HOST_FORCE_LS_SE0     : 1;
__REG32 UTMI_SUSPENDM         : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_usbphy_ctrl_bits;

/* USB PHY Status Register */
typedef struct{
__REG32                         : 3;
__REG32 HOSTDISCONDETECT_STATUS : 1;
__REG32                         : 2;
__REG32 DEVPLUGIN_STATUS        : 1;
__REG32                         : 1;
__REG32 OTGID_STATUS            : 1;
__REG32                         : 1;
__REG32 RESUME_STATUS           : 1;
__REG32                         :21;
} __hw_usbphy_status_bits;

/* USB PHY Debug Register */
typedef struct{
__REG32 OTGIDPIOLOCK            : 1;
__REG32 DEBUG_INTERFACE_HOLD    : 1;
__REG32 HSTPULLDOWN             : 2;
__REG32 ENHSTPULLDOWN           : 2;
__REG32                         : 2;
__REG32 TX2RXCOUNT              : 4;
__REG32 ENTX2RXCOUNT            : 1;
__REG32                         : 3;
__REG32 SQUELCHRESETCOUNT       : 5;
__REG32                         : 3;
__REG32 ENSQUELCHRESET          : 1;
__REG32 SQUELCHRESETLENGTH      : 4;
__REG32 HOST_RESUME_DEBUG       : 1;
__REG32 CLKGATE                 : 1;
__REG32                         : 1;
} __hw_usbphy_debug_bits;

/* UTMI Debug Status Register 0 */
typedef struct{
__REG32 LOOP_BACK_FAIL_COUNT    :16;
__REG32 UTMI_RXERROR_FAIL_COUNT :10;
__REG32 SQUELCH_COUNT           : 6;
} __hw_usbphy_debug0_status_bits;

/* UTMI Debug Status Register 1 */
typedef struct{
__REG32 DBG_ADDRESS             : 4;
__REG32                         : 8;
__REG32 ENTX2TX                 : 1;
__REG32 ENTAILADJVD             : 2;
__REG32                         :17;
} __hw_usbphy_debug1_bits;

/* UTMI RTL Version Description */
typedef struct{
__REG32 STEP                    :16;
__REG32 MINOR                   : 8;
__REG32 MAJOR                   : 8;
} __hw_usbphy_version_bits;

/* AHB to APBH Bridge Control and Status Register 0 */
typedef struct{
__REG32 CH0_FREEZE_CHANNEL	  : 1;
__REG32 CH1_FREEZE_CHANNEL	  : 1;
__REG32 CH2_FREEZE_CHANNEL	  : 1;
__REG32 CH3_FREEZE_CHANNEL	  : 1;
__REG32 CH4_FREEZE_CHANNEL	  : 1;
__REG32 CH5_FREEZE_CHANNEL	  : 1;
__REG32 CH6_FREEZE_CHANNEL	  : 1;
__REG32 CH7_FREEZE_CHANNEL	  : 1;
__REG32 CH0_CLKGATE_CHANNEL	  : 1;
__REG32 CH1_CLKGATE_CHANNEL	  : 1;
__REG32 CH2_CLKGATE_CHANNEL	  : 1;
__REG32 CH3_CLKGATE_CHANNEL	  : 1;
__REG32 CH4_CLKGATE_CHANNEL	  : 1;
__REG32 CH5_CLKGATE_CHANNEL	  : 1;
__REG32 CH6_CLKGATE_CHANNEL	  : 1;
__REG32 CH7_CLKGATE_CHANNEL	  : 1;
__REG32 CH0_RESET_CHANNEL		  : 1;
__REG32 CH1_RESET_CHANNEL		  : 1;
__REG32 CH2_RESET_CHANNEL		  : 1;
__REG32 CH3_RESET_CHANNEL		  : 1;
__REG32 CH4_RESET_CHANNEL		  : 1;
__REG32 CH5_RESET_CHANNEL		  : 1;
__REG32 CH6_RESET_CHANNEL		  : 1;
__REG32 CH7_RESET_CHANNEL		  : 1;
__REG32                       : 4;
__REG32 APB_BURST4_EN         : 1;
__REG32 AHB_BURST8_EN         : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_apbh_ctrl0_bits;

/* AHB to APBH Bridge Control and Status Register 1 */
typedef struct{
__REG32 CH0_CMDCMPLT_IRQ      : 1;
__REG32 CH1_CMDCMPLT_IRQ      : 1;
__REG32 CH2_CMDCMPLT_IRQ      : 1;
__REG32 CH3_CMDCMPLT_IRQ      : 1;
__REG32 CH4_CMDCMPLT_IRQ      : 1;
__REG32 CH5_CMDCMPLT_IRQ      : 1;
__REG32 CH6_CMDCMPLT_IRQ      : 1;
__REG32 CH7_CMDCMPLT_IRQ      : 1;
__REG32                       : 8;
__REG32 CH0_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH1_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH2_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH3_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH4_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH5_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH6_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH7_CMDCMPLT_IRQ_EN   : 1;
__REG32                       : 8;
} __hw_apbh_ctrl1_bits;

/* AHB to APBH Bridge Control and Status Register 2 */
typedef struct{
__REG32 CH0_CMDCMPLT_IRQ      : 1;
__REG32 CH1_CMDCMPLT_IRQ      : 1;
__REG32 CH2_CMDCMPLT_IRQ      : 1;
__REG32 CH3_CMDCMPLT_IRQ      : 1;
__REG32 CH4_CMDCMPLT_IRQ      : 1;
__REG32 CH5_CMDCMPLT_IRQ      : 1;
__REG32 CH6_CMDCMPLT_IRQ      : 1;
__REG32 CH7_CMDCMPLT_IRQ      : 1;
__REG32                       : 8;
__REG32 CH0_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH1_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH2_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH3_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH4_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH5_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH6_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH7_CMDCMPLT_IRQ_EN   : 1;
__REG32                       : 8;
} __hw_apbh_ctrl2_bits;

/* APBH DMA Channel 0..7 Command Register*/
typedef struct{
__REG32 COMMAND        				: 2;
__REG32 CHAIN	        				: 1;
__REG32 IRQONCMPLT     				: 1;
__REG32 NANDLOCK       				: 1;
__REG32 NANDWAIT4READY 				: 1;
__REG32 SEMAPHORE      				: 1;
__REG32 WAIT4ENDCMD    				: 1;
__REG32 HALTONTERMINATE				: 1;
__REG32 			        				: 3;
__REG32 CMDWORDS       				: 4;
__REG32 XFER_COUNT     				:16;
} __hw_apbh_ch_cmd_bits;

/* APBH DMA Channel 0..7 Semaphore Register */
typedef struct {        
__REG32 INCREMENT_SEMA 				: 8;
__REG32 			        				: 8;
__REG32 PHORE			     				: 8;
__REG32 			        				: 8;
} __hw_apbh_ch_sema_bits;

/* AHB to APBH DMA Channel 0..7 Debug Information */
typedef struct {        
__REG32 STATEMACHINE	 				: 5;
__REG32 			        				:15;
__REG32 WR_FIFO_FULL   				: 1;
__REG32 WR_FIFO_EMPTY  				: 1;
__REG32 RD_FIFO_FULL   				: 1;
__REG32 RD_FIFO_EMPTY  				: 1;
__REG32 NEXTCMDADDRVALID			: 1;
__REG32 LOCK			  	 				: 1;
__REG32 READY				   				: 1;
__REG32 SENSE				   				: 1;
__REG32 END					   				: 1;
__REG32 KICK				   				: 1;
__REG32 BURST				   				: 1;
__REG32 REQ					   				: 1;
} __hw_apbh_ch_debug1_bits;

/* AHB to APBH DMA Channel 0..7 Debug Information */
typedef struct {        
__REG32 AHB_BYTES   	 				:16;
__REG32 APB_BYTES   	 				:16;
} __hw_apbh_ch_debug2_bits;

/* APBH Bridge Version Register */
typedef struct {        
__REG32 STEP					 				:16;
__REG32 MINOR	        				: 8;
__REG32 MAJOR			     				: 8;
} __hw_apbh_version_bits;

/*AHB to APBX Bridge Control Register 0 */
typedef struct{
__REG32                       :30;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_apbx_ctrl0_bits;

/* AHB to APBX Bridge Control Register 1 */
typedef struct{
__REG32 CH0_CMDCMPLT_IRQ      : 1;
__REG32 CH1_CMDCMPLT_IRQ      : 1;
__REG32 CH2_CMDCMPLT_IRQ      : 1;
__REG32 CH3_CMDCMPLT_IRQ      : 1;
__REG32 CH4_CMDCMPLT_IRQ      : 1;
__REG32 CH5_CMDCMPLT_IRQ      : 1;
__REG32 CH6_CMDCMPLT_IRQ      : 1;
__REG32 CH7_CMDCMPLT_IRQ      : 1;
__REG32 CH9_CMDCMPLT_IRQ      : 1;
__REG32 CH8_CMDCMPLT_IRQ      : 1;
__REG32 CH10_CMDCMPLT_IRQ     : 1;
__REG32 CH11_CMDCMPLT_IRQ     : 1;
__REG32 CH12_CMDCMPLT_IRQ     : 1;
__REG32 CH13_CMDCMPLT_IRQ     : 1;
__REG32 CH14_CMDCMPLT_IRQ     : 1;
__REG32 CH15_CMDCMPLT_IRQ     : 1;
__REG32 CH0_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH1_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH2_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH3_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH4_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH5_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH6_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH7_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH8_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH9_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH10_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH11_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH12_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH13_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH14_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH15_CMDCMPLT_IRQ_EN  : 1;
} __hw_apbx_ctrl1_bits;

/* AHB to APBX Bridge Control and Status Register 2 */
typedef struct{
__REG32 CH0_CMDCMPLT_IRQ      : 1;
__REG32 CH1_CMDCMPLT_IRQ      : 1;
__REG32 CH2_CMDCMPLT_IRQ      : 1;
__REG32 CH3_CMDCMPLT_IRQ      : 1;
__REG32 CH4_CMDCMPLT_IRQ      : 1;
__REG32 CH5_CMDCMPLT_IRQ      : 1;
__REG32 CH6_CMDCMPLT_IRQ      : 1;
__REG32 CH7_CMDCMPLT_IRQ      : 1;
__REG32 CH8_CMDCMPLT_IRQ      : 1;
__REG32 CH9_CMDCMPLT_IRQ      : 1;
__REG32 CH10_CMDCMPLT_IRQ     : 1;
__REG32 CH11_CMDCMPLT_IRQ     : 1;
__REG32 CH12_CMDCMPLT_IRQ     : 1;
__REG32 CH13_CMDCMPLT_IRQ     : 1;
__REG32 CH14_CMDCMPLT_IRQ     : 1;
__REG32 CH15_CMDCMPLT_IRQ     : 1;
__REG32 CH0_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH1_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH2_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH3_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH4_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH5_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH6_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH7_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH8_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH9_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH10_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH11_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH12_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH13_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH14_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH15_CMDCMPLT_IRQ_EN  : 1;
} __hw_apbx_ctrl2_bits;

/* AHB to APBX Bridge Channel Register */
typedef struct{
__REG32 CH0_FREEZE_CHANNEL	  : 1;
__REG32 CH1_FREEZE_CHANNEL	  : 1;
__REG32 CH2_FREEZE_CHANNEL	  : 1;
__REG32 CH3_FREEZE_CHANNEL	  : 1;
__REG32 CH4_FREEZE_CHANNEL	  : 1;
__REG32 CH5_FREEZE_CHANNEL	  : 1;
__REG32 CH6_FREEZE_CHANNEL	  : 1;
__REG32 CH7_FREEZE_CHANNEL	  : 1;
__REG32 CH8_FREEZE_CHANNEL	  : 1;
__REG32 CH9_FREEZE_CHANNEL	  : 1;
__REG32 CH10_FREEZE_CHANNEL	  : 1;
__REG32 CH11_FREEZE_CHANNEL	  : 1;
__REG32 CH12_FREEZE_CHANNEL	  : 1;
__REG32 CH13_FREEZE_CHANNEL	  : 1;
__REG32 CH14_FREEZE_CHANNEL	  : 1;
__REG32 CH15_FREEZE_CHANNEL	  : 1;
__REG32 CH0_RESET_CHANNEL		  : 1;
__REG32 CH1_RESET_CHANNEL		  : 1;
__REG32 CH2_RESET_CHANNEL		  : 1;
__REG32 CH3_RESET_CHANNEL		  : 1;
__REG32 CH4_RESET_CHANNEL		  : 1;
__REG32 CH5_RESET_CHANNEL		  : 1;
__REG32 CH6_RESET_CHANNEL		  : 1;
__REG32 CH7_RESET_CHANNEL		  : 1;
__REG32 CH8_RESET_CHANNEL		  : 1;
__REG32 CH9_RESET_CHANNEL		  : 1;
__REG32 CH10_RESET_CHANNEL 	  : 1;
__REG32 CH11_RESET_CHANNEL	  : 1;
__REG32 CH12_RESET_CHANNEL	  : 1;
__REG32 CH13_RESET_CHANNEL	  : 1;
__REG32 CH14_RESET_CHANNEL	  : 1;
__REG32 CH15_RESET_CHANNEL	  : 1;
} __hw_apbx_channel_ctrl_bits;

/* APBH DMA Channel 0..7 Command Register*/
typedef struct{
__REG32 COMMAND        				: 2;
__REG32 CHAIN	        				: 1;
__REG32 IRQONCMPLT     				: 1;
__REG32                				: 2;
__REG32 SEMAPHORE      				: 1;
__REG32 WAIT4ENDCMD    				: 1;
__REG32 			        				: 4;
__REG32 CMDWORDS       				: 4;
__REG32 XFER_COUNT     				:16;
} __hw_apbx_ch_cmd_bits;

/* APBH DMA Channel 0..7 Semaphore Register */
typedef struct {        
__REG32 INCREMENT_SEMA 				: 8;
__REG32 			        				: 8;
__REG32 PHORE			     				: 8;
__REG32 			        				: 8;
} __hw_apbx_ch_sema_bits;

/* AHB to APBH DMA Channel 0..7 Debug Information */
typedef struct {        
__REG32 STATEMACHINE	 				: 5;
__REG32 			        				:15;
__REG32 WR_FIFO_FULL   				: 1;
__REG32 WR_FIFO_EMPTY  				: 1;
__REG32 RD_FIFO_FULL   				: 1;
__REG32 RD_FIFO_EMPTY  				: 1;
__REG32 NEXTCMDADDRVALID			: 1;
__REG32                       : 3;
__REG32 END					   				: 1;
__REG32 KICK				   				: 1;
__REG32 BURST				   				: 1;
__REG32 REQ					   				: 1;
} __hw_apbx_ch_debug1_bits;

/* AHB to APBH DMA Channel 0..7 Debug Information */
typedef struct {        
__REG32 AHB_BYTES   	 				:16;
__REG32 APB_BYTES   	 				:16;
} __hw_apbx_ch_debug2_bits;

/* APBH Bridge Version Register */
typedef struct {        
__REG32 STEP					 				:16;
__REG32 MINOR	        				: 8;
__REG32 MAJOR			     				: 8;
} __hw_apbx_version_bits;

/* EMI Control Register */
typedef struct {        
__REG32         			 				: 4;
__REG32 RESET_OUT     				: 1;
__REG32                				: 1;
__REG32 MEM_WIDTH     				: 1;
__REG32               				: 1;
__REG32 HIGH_PRIORITY_WRITE 	: 3;
__REG32               				: 1;
__REG32 PRIORITY_WRITE_ITER 	: 3;
__REG32                     	: 1;
__REG32 PORT_PRIORITY_ORDER 	: 5;
__REG32                      	: 1;
__REG32 ARB_MODE             	: 2;
__REG32 DLL_RESET            	: 1;
__REG32 DLL_SHIFT_RESET      	: 1;
__REG32 AXI_DEPTH           	: 2;
__REG32 TRAP_INIT            	: 1;
__REG32 TRAP_SR             	: 1;
__REG32                       : 1;
__REG32 SFTRST		     				: 1;
} __hw_emi_ctrl_bits;

/* EMI Version Register */
typedef struct {        
__REG32 STEP					 				:16;
__REG32 MINOR	        				: 8;
__REG32 MAJOR			     				: 8;
} __hw_emi_version_bits;

/* DRAM Control Register 00 */
typedef struct {        
__REG32 ADDR_CMP_EN    				: 1;
__REG32                				: 7;
__REG32 AHB0_FIFO_TYPE_REG  	: 1;
__REG32               				: 7;
__REG32 AHB0_R_PRIORITY      	: 1;
__REG32               				: 7;
__REG32 AHB0_W_PRIORITY      	: 1;
__REG32                     	: 7;
} __hw_dram_ctl00_bits;

/* DRAM Control Register 01 */
typedef struct {        
__REG32 AHB1_FIFO_TYPE_REG  	: 1;
__REG32                				: 7;
__REG32 AHB1_R_PRIORITY     	: 1;
__REG32               				: 7;
__REG32 AHB1_W_PRIORITY      	: 1;
__REG32               				: 7;
__REG32 AHB2_FIFO_TYPE_REG   	: 1;
__REG32                     	: 7;
} __hw_dram_ctl01_bits;

/* DRAM Control Register 02 */
typedef struct {        
__REG32 AHB2_R_PRIORITY     	: 1;
__REG32                				: 7;
__REG32 AHB2_W_PRIORITY     	: 1;
__REG32               				: 7;
__REG32 AHB3_FIFO_TYPE_REG   	: 1;
__REG32               				: 7;
__REG32 AHB3_R_PRIORITY     	: 1;
__REG32                     	: 7;
} __hw_dram_ctl02_bits;

/* DRAM Control Register 03 */
typedef struct {        
__REG32 AHB3_W_PRIORITY     	: 1;
__REG32                				: 7;
__REG32 AP                   	: 1;
__REG32               				: 7;
__REG32 AREFRESH             	: 1;
__REG32               				: 7;
__REG32 AUTO_REFRESH_MODE   	: 1;
__REG32                     	: 7;
} __hw_dram_ctl03_bits;

/* DRAM Control Register 04 */
typedef struct {        
__REG32 BANK_SPLIT_EN       	: 1;
__REG32                				: 7;
__REG32 CONCURRENTAP          : 1;
__REG32               				: 7;
__REG32 DLLLOCKREG           	: 1;
__REG32               				: 7;
__REG32 DLL_BYPASS_MODE     	: 1;
__REG32                     	: 7;
} __hw_dram_ctl04_bits;

/* DRAM Control Register 05 */
typedef struct {        
__REG32 EN_LOWPOWER_MODE     	: 1;
__REG32                				: 7;
__REG32 FAST_WRITE            : 1;
__REG32               				: 7;
__REG32 INTRPTAPBURST       	: 1;
__REG32               				: 7;
__REG32 INTRPTREADA         	: 1;
__REG32                     	: 7;
} __hw_dram_ctl05_bits;

/* DRAM Control Register 06 */
typedef struct {        
__REG32 INTRPTWRITEA         	: 1;
__REG32                				: 7;
__REG32 NO_CMD_INIT           : 1;
__REG32               				: 7;
__REG32 PLACEMENT_EN         	: 1;
__REG32               				: 7;
__REG32 POWER_DOWN          	: 1;
__REG32                     	: 7;
} __hw_dram_ctl06_bits;

/* DRAM Control Register 07 */
typedef struct {        
__REG32 PRIORITY_EN         	: 1;
__REG32                				: 7;
__REG32 RD2RD_TURN            : 1;
__REG32               				: 7;
__REG32 REG_DIMM_ENABLE      	: 1;
__REG32               				: 7;
__REG32 RW_SAME_EN          	: 1;
__REG32                     	: 7;
} __hw_dram_ctl07_bits;

/* DRAM Control Register 08 */
typedef struct {        
__REG32 SDR_MODE             	: 1;
__REG32                				: 7;
__REG32 SREFRESH              : 1;
__REG32               				: 7;
__REG32 START               	: 1;
__REG32               				: 7;
__REG32 TRAS_LOCKOUT         	: 1;
__REG32                     	: 7;
} __hw_dram_ctl08_bits;

/* DRAM Control Register 09 */
typedef struct {        
__REG32 WRITEINTERP          	  : 1;
__REG32                				  : 7;
__REG32 WRITE_MODEREG           : 1;
__REG32               				  : 7;
__REG32 OUT_OF_RANGE_SOURCE_ID  : 2;
__REG32               				  : 6;
__REG32 OUT_OF_RANGE_TYPE    	  : 2;
__REG32                     	  : 6;
} __hw_dram_ctl09_bits;

/* DRAM Control Register 10 */
typedef struct {        
__REG32 Q_FULLNESS           	: 2;
__REG32                				: 6;
__REG32 TEMRS                 : 2;
__REG32               				: 6;
__REG32 ADDR_PINS            	: 3;
__REG32               				: 5;
__REG32 AGE_COUNT            	: 3;
__REG32                     	: 5;
} __hw_dram_ctl10_bits;

/* DRAM Control Register 11 */
typedef struct {        
__REG32 CASLAT               	: 3;
__REG32                				: 5;
__REG32 COLUMN_SIZE           : 3;
__REG32               				: 5;
__REG32 COMMAND_AGE_COUNT     : 3;
__REG32               				: 5;
__REG32 MAX_CS_REG          	: 3;
__REG32                     	: 5;
} __hw_dram_ctl11_bits;

/* DRAM Control Register 12 */
typedef struct {        
__REG32 TCKE                  : 3;
__REG32                				:13;
__REG32 TRRD                  : 3;
__REG32               				: 5;
__REG32 TWR_INT             	: 3;
__REG32                     	: 5;
} __hw_dram_ctl12_bits;

/* DRAM Control Register 13 */
typedef struct {        
__REG32 TWTR                  : 3;
__REG32                				: 5;
__REG32 APREBIT               : 4;
__REG32                       : 4;
__REG32 CASLAT_LIN            : 4;
__REG32               				: 4;
__REG32 CASLAT_LIN_GATE      	: 4;
__REG32                     	: 4;
} __hw_dram_ctl13_bits;

/* DRAM Control Register 14 */
typedef struct {        
__REG32 CS_MAP                  : 4;
__REG32                		      : 4;
__REG32 INITAREF                : 4;
__REG32                         : 4;
__REG32 LOWPOWER_REFRESH_ENABLE : 4;
__REG32               				  : 4;
__REG32 MAX_COL_REG         	  : 4;
__REG32                     	  : 4;
} __hw_dram_ctl14_bits;

/* DRAM Control Register 15 */
typedef struct {        
__REG32 MAX_ROW_REG     : 4;
__REG32                	: 4;
__REG32 PORT_BUSY       : 4;
__REG32                 : 4;
__REG32 TDAL            : 4;
__REG32         			  : 4;
__REG32 TRP         	  : 4;
__REG32                 : 4;
} __hw_dram_ctl15_bits;

/* DRAM Control Register 16 */
typedef struct {        
__REG32 INT_ACK               : 4;
__REG32                      	: 4;
__REG32 LOWPOWER_AUTO_ENABLE  : 5;
__REG32                       : 3;
__REG32 LOWPOWER_CONTROL      : 5;
__REG32         			        : 3;
__REG32 TMRD         	        : 5;
__REG32                       : 3;
} __hw_dram_ctl16_bits;

/* DRAM Control Register 17 */
typedef struct {        
__REG32 TRC               : 5;
__REG32                   : 3;
__REG32 DLL_INCREMENT     : 8;
__REG32 DLL_LOCK          : 8;
__REG32 DLL_START_POIN    : 8;
} __hw_dram_ctl17_bits;

/* DRAM Control Register 18 */
typedef struct {        
__REG32 INT_MASK          : 5;
__REG32                   : 3;
__REG32 INT_STATUS        : 5;
__REG32                   : 3;
__REG32 DLL_DQS_DELAY_0   : 7;
__REG32                   : 1;
__REG32 DLL_DQS_DELAY_1   : 7;
__REG32                   : 1;
} __hw_dram_ctl18_bits;

/* DRAM Control Register 19 */
typedef struct {        
__REG32 DLL_DQS_DELAY_BYPASS_0  : 8;
__REG32 DLL_DQS_DELAY_BYPASS_1  : 8;
__REG32 DQS_OUT_SHIFT           : 7;
__REG32                         : 1;
__REG32 DQS_OUT_SHIFT_BYPASS    : 8;
} __hw_dram_ctl19_bits;

/* DRAM Control Register 20 */
typedef struct {        
__REG32 WR_DQS_SHIFT            : 7;
__REG32                         : 1;
__REG32 WR_DQS_SHIFT_BYPASS     : 8;
__REG32 TRAS_MIN                : 8;
__REG32 TRCD_INT                : 8;
} __hw_dram_ctl20_bits;

/* DRAM Control Register 21 */
typedef struct {        
__REG32 TRFC                    : 8;
__REG32 OUT_OF_RANGE_LENGTH     :10;
__REG32                         :14;
} __hw_dram_ctl21_bits;

/* DRAM Control Register 22 */
typedef struct {        
__REG32 AHB0_RDCNT              :11;
__REG32                         : 5;
__REG32 AHB0_WRCNT              :11;
__REG32                         : 5;
} __hw_dram_ctl22_bits;

/* DRAM Control Register 23 */
typedef struct {        
__REG32 AHB1_RDCNT              :11;
__REG32                         : 5;
__REG32 AHB1_WRCNT              :11;
__REG32                         : 5;
} __hw_dram_ctl23_bits;

/* DRAM Control Register 24 */
typedef struct {        
__REG32 AHB2_RDCNT              :11;
__REG32                         : 5;
__REG32 AHB2_WRCNT              :11;
__REG32                         : 5;
} __hw_dram_ctl24_bits;

/* DRAM Control Register 25 */
typedef struct {        
__REG32 AHB3_RDCNT              :11;
__REG32                         : 5;
__REG32 AHB3_WRCNT              :11;
__REG32                         : 5;
} __hw_dram_ctl25_bits;

/* DRAM Control Register 26 */
typedef struct {        
__REG32 TREF                    :12;
__REG32                         :20;
} __hw_dram_ctl26_bits;

/* DRAM Control Register 29 */
typedef struct {        
__REG32 LOWPOWER_EXTERNAL_CNT   :16;
__REG32 LOWPOWER_INTERNAL_CNT   :16;
} __hw_dram_ctl29_bits;

/* DRAM Control Register 30 */
typedef struct {        
__REG32 LOWPOWER_POWER_DOWN_CNT :16;
__REG32 LOWPOWER_REFRESH_HOLD   :16;
} __hw_dram_ctl30_bits;

/* DRAM Control Register 31 */
typedef struct {        
__REG32 LOWPOWER_SELF_REFRESH_CNT :16;
__REG32 TDLL                      :16;
} __hw_dram_ctl31_bits;

/* DRAM Control Register 32 */
typedef struct {        
__REG32 TRAS_MAX        :16;
__REG32 TXSNR           :16;
} __hw_dram_ctl32_bits;

/* DRAM Control Register 33 */
typedef struct {        
__REG32 TXSR            :16;
__REG32 VERSION         :16;
} __hw_dram_ctl33_bits;

/* DRAM Control Register 34 */
typedef struct {        
__REG32 TINIT           :24;
__REG32                 : 8;
} __hw_dram_ctl34_bits;

/* DRAM Control Register 35 */
typedef struct {        
__REG32 OUT_OF_RANGE_ADDR :31;
__REG32                   : 1;
} __hw_dram_ctl35_bits;

/* DRAM Control Register 36 */
typedef struct {        
__REG32 ACTIVE_AGING            : 1;
__REG32                         :15;
__REG32 ENABLE_QUICK_SREFRESH   : 1;
__REG32                         : 7;
__REG32 PWRUP_SREFRESH_EXIT     : 1;
__REG32                         : 7;
} __hw_dram_ctl36_bits;

/* DRAM Control Register 37 */
typedef struct {        
__REG32 TREF_ENABLE             : 1;
__REG32                         : 7;
__REG32 BUS_SHARE_TIMEOUT       :10;
__REG32                         :14;
} __hw_dram_ctl37_bits;

/* DRAM Control Register 38 */
typedef struct {        
__REG32 EMRS1_DATA              :13;
__REG32                         : 3;
__REG32 EMRS2_DATA_0            :13;
__REG32                         : 3;
} __hw_dram_ctl38_bits;

/* DRAM Control Register 39 */
typedef struct {        
__REG32 EMRS2_DATA_1            :13;
__REG32                         : 3;
__REG32 EMRS2_DATA_2            :13;
__REG32                         : 3;
} __hw_dram_ctl39_bits;

/* DRAM Control Register 40 */
typedef struct {        
__REG32 EMRS2_DATA_3            :13;
__REG32                         : 3;
__REG32 TPDEX                   :16;
} __hw_dram_ctl40_bits;

/* GPMI Control Register 0 */
typedef struct {        
__REG32 XFER_COUNT          :16;
__REG32 ADDRESS_INCREMENT   : 1;
__REG32                     : 3;
__REG32 CS                  : 2;
__REG32 LOCK_CS             : 1;
__REG32 WORD_LENGTH         : 1;
__REG32 COMMAND_MODE        : 2;
__REG32                     : 1;
__REG32 TIMEOUT_IRQ_EN      : 1;
__REG32                     : 1;
__REG32 RUN                 : 1;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_gpmi_ctrl0_bits;

/* GPMI Compare Register */
typedef struct {        
__REG32 REFERENCE           :16;
__REG32 MASK                :16;
} __hw_gpmi_compare_bits;

/* GPMI Integrated ECC Control Register */
typedef struct {        
__REG32 BUFFER_MASK         : 9;
__REG32                     : 3;
__REG32 ENABLE_ECC          : 1;
__REG32 ECC_CMD             : 2;
__REG32                     : 1;
__REG32 HANDLE              :16;
} __hw_gpmi_eccctrl_bits;

/* GPMI Integrated ECC Transfer Count Register */
typedef struct {        
__REG32 COUNT               :16;
__REG32                     :16;
} __hw_gpmi_ecccount_bits;

/* GPMI Control Register 1 */
typedef struct {        
__REG32 GPMI_MODE             : 1;
__REG32                       : 1;
__REG32 ATA_IRQRDY_POLARITY   : 1;
__REG32 DEV_RESET             : 1;
__REG32 ABORT_WAIT_FOR_READY0 : 1;
__REG32 ABORT_WAIT_FOR_READY1 : 1;
__REG32 ABORT_WAIT_FOR_READY2 : 1;
__REG32 ABORT_WAIT_FOR_READY3 : 1;
__REG32 BURST_EN              : 1;
__REG32 TIMEOUT_IRQ           : 1;
__REG32                       : 1;
__REG32 DMA2ECC_MODE          : 1;
__REG32 RDN_DELAY             : 4;
__REG32 HALF_PERIOD           : 1;
__REG32 DLL_ENABLE            : 1;
__REG32 BCH_MODE              : 1;
__REG32 GANGED_RDYBUSY        : 1;
__REG32 CE0_SEL               : 1;
__REG32 CE1_SEL               : 1;
__REG32 CE2_SEL               : 1;
__REG32 CE3_SEL               : 1;
__REG32                       : 8;
} __hw_gpmi_ctrl1_bits;

/* GPMI Timing Register 0 */
typedef struct {        
__REG32 DATA_SETUP            : 8;
__REG32 DATA_HOLD             : 8;
__REG32 ADDRESS_SETUP         : 8;
__REG32                       : 8;
} __hw_gpmi_timing0_bits;

/* GPMI Timing Register 1 */
typedef struct {        
__REG32                       :16;
__REG32 DEVICE_BUSY_TIMEOUT   :16;
} __hw_gpmi_timing1_bits;

/* GPMI Status Register */
typedef struct {        
__REG32 DEV0_ERROR            : 1;
__REG32 DEV1_ERROR            : 1;
__REG32 DEV2_ERROR            : 1;
__REG32 DEV3_ERROR            : 1;
__REG32 FIFO_FULL             : 1;
__REG32 FIFO_EMPTY            : 1;
__REG32 INVALID_BUFFER_MASK   : 1;
__REG32                       : 1;
__REG32 RDY_TIMEOUT           : 4;
__REG32                       :19;
__REG32 PRESENT               : 1;
} __hw_gpmi_stat_bits;

/* GPMI Debug Information Register */
typedef struct {        
__REG32 MAIN_STATE            : 4;
__REG32 PIN_STATE             : 3;
__REG32 BUSY                  : 1;
__REG32 UDMA_STATE            : 4;
__REG32 CMD_END               : 4;
__REG32 DMAREQ0               : 1;
__REG32 DMAREQ1               : 1;
__REG32 DMAREQ2               : 1;
__REG32 DMAREQ3               : 1;
__REG32 SENSE0                : 1;
__REG32 SENSE1                : 1;
__REG32 SENSE2                : 1;
__REG32 SENSE3                : 1;
__REG32 WAIT_FOR_READY_END0   : 1;
__REG32 WAIT_FOR_READY_END1   : 1;
__REG32 WAIT_FOR_READY_END2   : 1;
__REG32 WAIT_FOR_READY_END3   : 1;
__REG32 READY0                : 1;
__REG32 READY1                : 1;
__REG32 READY2                : 1;
__REG32 READY3                : 1;
} __hw_gpmi_debug_bits;

/* GPMI Version Register */
typedef struct {        
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_gpmi_version_bits;

/* GPMI Debug2 Information Register */
typedef struct {        
__REG32 RDN_TAP               : 6;
__REG32 UPDATE_WINDOW         : 1;
__REG32 VIEW_DELAYED_RDN      : 1;
__REG32 SYND2GPMI_READY       : 1;
__REG32 SYND2GPMI_VALID       : 1;
__REG32 GPMI2SYND_READY       : 1;
__REG32 GPMI2SYND_VALID       : 1;
__REG32 SYND2GPMI_BE          : 4;
__REG32                       :16;
} __hw_gpmi_debug2_bits;

/* GPMI Debug3 Information Register */
typedef struct {        
__REG32 DEV_WORD_CNTR         :16;
__REG32 APB_WORD_CNTR         :16;
} __hw_gpmi_debug3_bits;

/* Hardware ECC Accelerator Control Register */
typedef struct {        
__REG32 COMPLETE_IRQ          : 1;
__REG32 DEBUG_WRITE_IRQ       : 1;
__REG32 DEBUG_STALL_IRQ       : 1;
__REG32 BM_ERROR_IRQ          : 1;
__REG32                       : 4;
__REG32 COMPLETE_IRQ_EN       : 1;
__REG32 DEBUG_WRITE_IRQ_EN    : 1;
__REG32 DEBUG_STALL_IRQ_EN    : 1;
__REG32                       :13;
__REG32 THROTTLE              : 4;
__REG32                       : 1;
__REG32 AHBM_SFTRST           : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_ecc8_ctrl_bits;

/* Hardware ECC Accelerator Status Register 0 */
typedef struct {        
__REG32                       : 2;
__REG32 UNCORRECTABLE         : 1;
__REG32 CORRECTED             : 1;
__REG32 ALLONES               : 1;
__REG32                       : 3;
__REG32 STATUS_AUX            : 4;
__REG32 RS4ECC_DEC_PRESENT    : 1;
__REG32 RS4ECC_ENC_PRESENT    : 1;
__REG32 RS8ECC_DEC_PRESENT    : 1;
__REG32 RS8ECC_ENC_PRESENT    : 1;
__REG32 COMPLETED_CE          : 4;
__REG32 HANDLE                :12;
} __hw_ecc8_status0_bits;

/* Hardware ECC Accelerator Status Register 1 */
typedef struct {        
__REG32 STATUS_PAYLOAD0       : 4;
__REG32 STATUS_PAYLOAD1       : 4;
__REG32 STATUS_PAYLOAD2       : 4;
__REG32 STATUS_PAYLOAD3       : 4;
__REG32 STATUS_PAYLOAD4       : 4;
__REG32 STATUS_PAYLOAD5       : 4;
__REG32 STATUS_PAYLOAD6       : 4;
__REG32 STATUS_PAYLOAD7       : 4;
} __hw_ecc8_status1_bits;

/* Hardware ECC Accelerator Debug Register 0 */
typedef struct {        
__REG32 DEBUG_REG_SELECT          : 6;
__REG32                           : 2;
__REG32 BM_KES_TEST_BYPASS        : 1;
__REG32 KES_DEBUG_STALL           : 1;
__REG32 KES_DEBUG_STEP            : 1;
__REG32 KES_STANDALONE            : 1;
__REG32 KES_DEBUG_KICK            : 1;
__REG32 KES_DEBUG_MODE4K          : 1;
__REG32 KES_DEBUG_PAYLOAD_FLAG    : 1;
__REG32 KES_DEBUG_SHIFT_SYND      : 1;
__REG32 KES_DEBUG_SYNDROME_SYMBOL : 9;
__REG32                           : 7;
} __hw_ecc8_debug0_bits;

/* ECC8 Version Register */
typedef struct {        
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_ecc8_version_bits;

/* Hardware BCH ECC Accelerator Control Register */
typedef struct {        
__REG32 COMPLETE_IRQ          : 1;
__REG32                       : 1;
__REG32 DEBUG_STALL_IRQ       : 1;
__REG32 BM_ERROR_IRQ          : 1;
__REG32                       : 4;
__REG32 COMPLETE_IRQ_EN       : 1;
__REG32                       : 1;
__REG32 DEBUG_STALL_IRQ_EN    : 1;
__REG32                       : 5;
__REG32 M2M_ENABLE            : 1;
__REG32 M2M_ENCODE            : 1;
__REG32 M2M_LAYOUT            : 2;
__REG32                       : 2;
__REG32 DEBUGSYNDROME         : 1;
__REG32                       : 7;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_bch_ctrl_bits;

/* Hardware BCH ECC Accelerator Status Register 0 */
typedef struct {        
__REG32                       : 2;
__REG32 UNCORRECTABLE         : 1;
__REG32 CORRECTED             : 1;
__REG32 ALLONES               : 1;
__REG32                       : 3;
__REG32 STATUS_BLK0           : 8;
__REG32 COMPLETED_CE          : 4;
__REG32 HANDLE                :12;
} __hw_bch_status0_bits;

/* Hardware BCH ECC Accelerator Mode Register */
typedef struct {        
__REG32 ERASE_THRESHOLD       : 8;
__REG32                       :24;
} __hw_bch_mode_bits;

/* Hardware BCH ECC Accelerator Layout Select Register */
typedef struct {        
__REG32 CS0_SELEC             : 2;
__REG32 CS1_SELEC             : 2;
__REG32 CS2_SELEC             : 2;
__REG32 CS3_SELEC             : 2;
__REG32 CS4_SELEC             : 2;
__REG32 CS5_SELEC             : 2;
__REG32 CS6_SELEC             : 2;
__REG32 CS7_SELEC             : 2;
__REG32 CS8_SELEC             : 2;
__REG32 CS9_SELEC             : 2;
__REG32 CS10_SELEC            : 2;
__REG32 CS11_SELEC            : 2;
__REG32 CS12_SELEC            : 2;
__REG32 CS13_SELEC            : 2;
__REG32 CS14_SELEC            : 2;
__REG32 CS15_SELEC            : 2;
} __hw_bch_layoutselect_bits;

/* Hardware BCH ECC Flash 0 Layout 0 Register */
typedef struct {        
__REG32 DATA0_SIZE            :12;
__REG32 ECC0                  : 4;
__REG32 META_SIZE             : 8;
__REG32 NBLOCKS               : 8;
} __hw_bch_flashxlayout0_bits;

/* Hardware BCH ECC Flash 0 Layout 1 Register */
typedef struct {        
__REG32 DATAN_SIZE            :12;
__REG32 ECCN                  : 4;
__REG32 PAGE_SIZE             :16;
} __hw_bch_flashxlayout1_bits;

/* Hardware BCH ECC Debug Register 0 */
typedef struct {        
__REG32 DEBUG_REG_SELECT          : 6;
__REG32                           : 2;
__REG32 BM_KES_TEST_BYPASS        : 1;
__REG32 KES_DEBUG_STALL           : 1;
__REG32 KES_DEBUG_STEP            : 1;
__REG32 KES_STANDALONE            : 1;
__REG32 KES_DEBUG_KICK            : 1;
__REG32 KES_DEBUG_MODE4K          : 1;
__REG32 KES_DEBUG_PAYLOAD_FLAG    : 1;
__REG32 KES_DEBUG_SHIFT_SYND      : 1;
__REG32 KES_DEBUG_SYNDROME_SYMBOL : 9;
__REG32 ROM_BIST_COMPLETE         : 1;
__REG32 ROM_BIST_ENABLE           : 1;
__REG32                           : 5;
} __hw_bch_debug0_bits;

/* Hardware BCH ECC Version Register */
typedef struct {        
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_bch_version_bits;

/* DCP Control Register 0 */
typedef struct {        
__REG32 CHANNEL0_INTERRUPT_ENABLE : 1;
__REG32 CHANNEL1_INTERRUPT_ENABLE : 1;
__REG32 CHANNEL2_INTERRUPT_ENABLE : 1;
__REG32 CHANNEL3_INTERRUPT_ENABLE : 1;
__REG32                           :17;
__REG32 ENABLE_CONTEXT_SWITCHING  : 1;
__REG32 ENABLE_CONTEXT_CACHING    : 1;
__REG32 GATHER_RESIDUAL_WRITES    : 1;
__REG32                           : 5;
__REG32 PRESENT_CRYPTO            : 1;
__REG32 CLKGATE                   : 1;
__REG32 SFTRST                    : 1;
} __hw_dcp_ctrl_bits;

/* DCP Status Register */
typedef struct {        
__REG32 IRQ_CH0           : 1;
__REG32 IRQ_CH1           : 1;
__REG32 IRQ_CH2           : 1;
__REG32 IRQ_CH3           : 1;
__REG32                   :12;
__REG32 READY_CHANNEL0    : 1;
__REG32 READY_CHANNEL1    : 1;
__REG32 READY_CHANNEL2    : 1;
__REG32 READY_CHANNEL3    : 1;
__REG32                   : 4;
__REG32 CUR_CHANNEL       : 4;
__REG32 OTP_KEY_READY     : 1;
__REG32                   : 3;
} __hw_dcp_stat_bits;

/* DCP Channel Control Register */
typedef struct {        
__REG32 ENABLE_CHANNEL0         : 1;
__REG32 ENABLE_CHANNEL1         : 1;
__REG32 ENABLE_CHANNEL2         : 1;
__REG32 ENABLE_CHANNEL3         : 1;
__REG32                         : 4;
__REG32 HIGH_PRIORITY_CHANNEL0  : 1;
__REG32 HIGH_PRIORITY_CHANNEL1  : 1;
__REG32 HIGH_PRIORITY_CHANNEL2  : 1;
__REG32 HIGH_PRIORITY_CHANNEL3  : 1;
__REG32                         : 4;
__REG32 CH0_IRQ_MERGED          : 1;
__REG32                         :15;
} __hw_dcp_channelctrl_bits;

/* DCP Capability 0 Register */
typedef struct {        
__REG32 NUM_KEYS              : 8;
__REG32 NUM_CHANNELS          : 4;
__REG32                       :18;
__REG32 ENABLE_TZONE          : 1;
__REG32 DISABLE_DECRYPT       : 1;
} __hw_dcp_capability0_bits;

/* DCP Capability 1 Register */
typedef struct {        
__REG32 CIPHER_ALGORITHMS     :16;
__REG32 HASH_ALGORITHMS       :16;
} __hw_dcp_capability1_bits;

/* DCP Key Index */
typedef struct {        
__REG32 SUBWORD           : 2;
__REG32                   : 2;
__REG32 INDEX             : 2;
__REG32                   :26;
} __hw_dcp_key_bits;

/* DCP Work Packet 1 Status Register */
typedef struct {        
__REG32 INTERRUPT         : 1;
__REG32 DECR_SEMAPHORE    : 1;
__REG32 CHAIN             : 1;
__REG32 CHAIN_CONTIGUOUS  : 1;
__REG32 ENABLE_MEMCOPY    : 1;
__REG32 ENABLE_CIPHER     : 1;
__REG32 ENABLE_HASH       : 1;
__REG32 ENABLE_BLIT       : 1;
__REG32 CIPHER_ENCRYPT    : 1;
__REG32 CIPHER_INIT       : 1;
__REG32 OTP_KEY           : 1;
__REG32 PAYLOAD_KEY       : 1;
__REG32 HASH_INIT         : 1;
__REG32 HASH_TERM         : 1;
__REG32 CHECK_HASH        : 1;
__REG32 HASH_OUTPUT       : 1;
__REG32 CONSTANT_FILL     : 1;
__REG32 TEST_SEMA_IRQ     : 1;
__REG32 KEY_BYTESWAP      : 1;
__REG32 KEY_WORDSWAP      : 1;
__REG32 INPUT_BYTESWAP    : 1;
__REG32 INPUT_WORDSWAP    : 1;
__REG32 OUTPUT_BYTESWAP   : 1;
__REG32 OUTPUT_WORDSWAP   : 1;
__REG32 TAG               : 8;
} __hw_dcp_packet1_bits;

/* DCP Work Packet 2 Status Register */
typedef struct {        
__REG32 CIPHER_SELECT     : 4;
__REG32 CIPHER_MODE       : 4;
__REG32 KEY_SELECT        : 8;
__REG32 HASH_SELECT       : 4;
__REG32                   : 4;
__REG32 CIPHER_CFG        : 8;
} __hw_dcp_packet2_bits;

/* DCP Channel 0..3 Semaphore Register */
typedef struct {        
__REG32 INCREMENT         : 8;
__REG32                   : 8;
__REG32 VALUE             : 8;
__REG32                   : 8;
} __hw_dcp_chxsema_bits;

/* DCP Channel 0..3 Status Register */
typedef struct {        
__REG32                   : 1;
__REG32 HASH_MISMATCH     : 1;
__REG32 ERROR_SETUP       : 1;
__REG32 ERROR_PACKET      : 1;
__REG32 ERROR_SRC         : 1;
__REG32 ERROR_DST         : 1;
__REG32                   :10;
__REG32 ERROR_CODE        : 8;
__REG32 TAG               : 8;
} __hw_dcp_chxstat_bits;

/* DCP Channel 0..3 Options Register */
typedef struct {        
__REG32 RECOVERY_TIMER    :16;
__REG32                   :16;
} __hw_dcp_chxopts_bits;

/* DCP Debug Select Register */
typedef struct {        
__REG32 INDEX             : 8;
__REG32                   :24;
} __hw_dcp_dbgselect_bits;

/* DCP Version Register */
typedef struct {        
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_dcp_version_bits;

/* PXP Control Register 0 */
typedef struct {        
__REG32 ENABLE            : 1;
__REG32 IRQ_ENABLE        : 1;
__REG32                   : 2;
__REG32 OUTPUT_RGB_FORMAT : 4;
__REG32 ROTATE            : 2;
__REG32 HFLIP             : 1;
__REG32 VFLIP             : 1;
__REG32 S0_FORMAT         : 4;
__REG32 SUBSAMPLE         : 1;
__REG32 UPSAMPLE          : 1;
__REG32 SCALE             : 1;
__REG32 CROP              : 1;
__REG32 DELTA             : 1;
__REG32 IN_PLACE          : 1;
__REG32 ALPHA_OUTPUT      : 1;
__REG32                   : 1;
__REG32 INTERLACED_INPUT  : 2;
__REG32 INTERLACED_OUTPUT : 2;
__REG32                   : 2;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __hw_pxp_ctrl_bits;

/* PXP Status Register */
typedef struct {        
__REG32 IRQ               : 1;
__REG32 AXI_WRITE_ERROR   : 1;
__REG32 AXI_READ_ERROR    : 1;
__REG32                   : 1;
__REG32 AXI_ERROR_ID      : 4;
__REG32                   : 8;
__REG32 BLOCKY            : 8;
__REG32 BLOCKX            : 8;
} __hw_pxp_stat_bits;

/* PXP Output Buffer Size */
typedef struct {        
__REG32 HEIGHT            :12;
__REG32 WIDTH             :12;
__REG32 ALPHA             : 8;
} __hw_pxp_rgbsize_bits;

/* PXP Source 0 (video) Buffer Parameters */
typedef struct {        
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_s0param_bits;

/* Source 0 Cropping Register */
typedef struct {        
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_s0crop_bits;

/* Source 0 Scale Factor Register */
typedef struct {        
__REG32 XSCALE            :14;
__REG32                   : 2;
__REG32 YSCALE            :14;
__REG32                   : 2;
} __hw_pxp_s0scale_bits;

/* Source 0 Scale Offset Register */
typedef struct {        
__REG32 XOFFSET           :12;
__REG32                   : 4;
__REG32 YOFFSET           :12;
__REG32                   : 4;
} __hw_pxp_s0offset_bits;

/* Color Space Conversion Coefficient Register 0 */
typedef struct {        
__REG32 Y_OFFSET          : 9;
__REG32 UV_OFFSET         : 9;
__REG32 C0                :11;
__REG32                   : 2;
__REG32 YCBCR_MODE        : 1;
} __hw_pxp_csccoeff0_bits;

/* Color Space Conversion Coefficient Register 1 */
typedef struct {        
__REG32 C4                :11;
__REG32                   : 5;
__REG32 C1                :11;
__REG32                   : 5;
} __hw_pxp_csccoeff1_bits;

/* Color Space Conversion Coefficient Register 2 */
typedef struct {        
__REG32 C3                :11;
__REG32                   : 5;
__REG32 C2                :11;
__REG32                   : 5;
} __hw_pxp_csccoeff2_bits;

/* PXP Next Frame Pointer */
typedef struct {        
__REG32 ENABLED           : 1;
__REG32                   : 1;
__REG32 POINTER           :30;
} __hw_pxp_next_bits;

/* PXP S0 Color Key Low */
typedef struct {        
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_s0colorkeylow_bits;

/* PXP S0 Color Key High */
typedef struct {        
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_s0colorkeyhigh_bits;

/* PXP Overlay Color Key Low */
typedef struct {        
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_olcolorkeylow_bits;

/* PXP Overlay Color Key High */
typedef struct {        
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_olcolorkeyhigh_bits;

/* PXP Debug Control Register */
typedef struct {        
__REG32 SELECT            : 8;
__REG32                   :24;
} __hw_pxp_debugctrl_bits;

/* DCP Version Register */
typedef struct {        
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_pxp_version_bits;

/* PXP Overlay 0 Size */
typedef struct {        
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_olxsize_bits;

/* PXP Overlay 0 Size */
typedef struct {        
__REG32 ENABLE            : 1;
__REG32 ALPHA_CNTL        : 2;
__REG32 ENABLE_COLORKEY   : 1;
__REG32 FORMAT            : 4;
__REG32 ALPHA             : 8;
__REG32 ROP               : 4;
__REG32                   :12;
} __hw_pxp_olxparam_bits;

/* LCDIF General Control Register */
typedef struct {        
__REG32 RUN                 : 1;
__REG32 DATA_FORMAT_24_BIT  : 1;
__REG32 DATA_FORMAT_18_BIT  : 1;
__REG32 DATA_FORMAT_16_BIT  : 1;
__REG32                     : 1;
__REG32 LCDIF_MASTER        : 1;
__REG32                     : 1;
__REG32 RGB_TO_YCBCR422_CSC : 1;
__REG32 WORD_LENGTH         : 2;
__REG32 LCD_DATABUS_WIDTH   : 2;
__REG32 CSC_DATA_SWIZZLE    : 2;
__REG32 INPUT_DATA_SWIZZLE  : 2;
__REG32 DATA_SELECT         : 1;
__REG32 DOTCLK_MODE         : 1;
__REG32 VSYNC_MODE          : 1;
__REG32 BYPASS_COUNT        : 1;
__REG32 DVI_MODE            : 1;
__REG32 SHIFT_NUM_BITS      : 5;
__REG32 DATA_SHIFT_DIR      : 1;
__REG32 WAIT_FOR_VSYNC_EDGE : 1;
__REG32                     : 1;
__REG32 YCBCR422_INPUT      : 1;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_lcdif_ctrl_bits;

/* LCDIF General Control1 Register */
typedef struct {        
__REG32 RESET                 : 1;
__REG32 MODE86                : 1;
__REG32 BUSY_ENABLE           : 1;
__REG32 LCD_CS_CTRL           : 1;
__REG32                       : 4;
__REG32 VSYNC_EDGE_IRQ        : 1;
__REG32 CUR_FRAME_DONE_IRQ    : 1;
__REG32 UNDERFLOW_IRQ         : 1;
__REG32 OVERFLOW_IRQ          : 1;
__REG32 VSYNC_EDGE_IRQ_EN     : 1;
__REG32 CUR_FRAME_DONE_IRQ_EN : 1;
__REG32 UNDERFLOW_IRQ_EN      : 1;
__REG32 OVERFLOW_IRQ_EN       : 1;
__REG32 BYTE_PACKING_FORMAT   : 4;
__REG32 IRQ_ON_ALT_FIELDS     : 1;
__REG32 FIFO_CLEAR            : 1;
__REG32 START_ITRLC_F_SF      : 1;
__REG32 INTERLACE_FIELDS      : 1;
__REG32 RECOVER_ON_UNDERFLOW  : 1;
__REG32 BM_ERROR_IRQ          : 1;
__REG32 BM_ERROR_IRQ_EN       : 1;
__REG32                       : 5;
} __hw_lcdif_ctrl1_bits;

/* LCDIF Horizontal and Vertical Valid Data Count Register */
typedef struct {        
__REG32 H_COUNT       :16;
__REG32 V_COUNT       :16;
} __hw_lcdif_transfer_count_bits;

/* LCD Interface Timing Register */
typedef struct {        
__REG32 DATA_SETUP    : 8;
__REG32 DATA_HOLD     : 8;
__REG32 CMD_SETUP     : 8;
__REG32 CMD_HOLD      : 8;
} __hw_lcdif_timing_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register0 */
typedef struct {        
__REG32 VSYNC_PULSE_WIDTH :18;
__REG32 HALF_LINE_MODE    : 1;
__REG32 HALF_LINE         : 1;
__REG32 VSYNC_PW_UNIT     : 1;
__REG32 VSYNC_PRD_UNIT    : 1;
__REG32                   : 2;
__REG32 ENABLE_POL        : 1;
__REG32 DOTCLK_POL        : 1;
__REG32 HSYNC_POL         : 1;
__REG32 VSYNC_POL         : 1;
__REG32 ENABLE_PRESENT    : 1;
__REG32 VSYNC_OEB         : 1;
__REG32                   : 2;
} __hw_lcdif_vdctrl0_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register1 */
typedef struct {        
__REG32 VSYNC_PERIOD      :32;
} __hw_lcdif_vdctrl1_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register2 */
typedef struct {        
__REG32 HSYNC_PERIOD      :18;
__REG32                   : 6;
__REG32 HSYNC_PULSE_WIDTH : 8;
} __hw_lcdif_vdctrl2_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register3 */
typedef struct {        
__REG32 VWAIT_CNT         :16;
__REG32 HWAIT_CNT         :12;
__REG32 VSYNC_ONLY        : 1;
__REG32 MUX_SYNC_SIGNALS  : 1;
__REG32                   : 2;
} __hw_lcdif_vdctrl3_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register4 */
typedef struct {        
__REG32 DOTCLK_HVLD_D_CNT :18;
__REG32 SYNC_SIGNALS_ON   : 1;
__REG32                   :13;
} __hw_lcdif_vdctrl4_bits;

/* Digital Video Interface Control0 Register */
typedef struct {        
__REG32 V_LINES_CNT       :10;
__REG32 H_BLANKING_CNT    :10;
__REG32 H_ACTIVE_CNT      :11;
__REG32                   : 1;
} __hw_lcdif_dvictrl0_bits;

/* Digital Video Interface Control1 */
typedef struct {        
__REG32 F2_START_LINE     :10;
__REG32 F1_END_LINE       :10;
__REG32 F1_START_LINE     :10;
__REG32                   : 2;
} __hw_lcdif_dvictrl1_bits;

/* Digital Video Interface Control2 Register */
typedef struct {        
__REG32 V1_BLANK_END_LINE   :10;
__REG32 V1_BLANK_START_LINE :10;
__REG32 F2_END_LINE         :10;
__REG32                     : 2;
} __hw_lcdif_dvictrl2_bits;

/* Digital Video Interface Control3 Register */
typedef struct {        
__REG32 V2_BLANK_END_LINE   :10;
__REG32                     : 6;
__REG32 V2_BLANK_START_LINE :10;
__REG32                     : 6;
} __hw_lcdif_dvictrl3_bits;

/* Digital Video Interface Control4 Register */
typedef struct {        
__REG32 H_FILL_CNT        : 8;
__REG32 CR_FILL_VALUE     : 8;
__REG32 CB_FILL_VALUE     : 8;
__REG32 Y_FILL_VALUE      : 8;
} __hw_lcdif_dvictrl4_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient0 Register */
typedef struct {        
__REG32 CSC_SUBSAMPLE_FILTER  : 2;
__REG32                       :14;
__REG32 C0                    :10;
__REG32                       : 6;
} __hw_lcdif_csc_coeff0_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient1 Register */
typedef struct {        
__REG32 C1            :10;
__REG32               : 6;
__REG32 C2            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff1_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient2 Register */
typedef struct {        
__REG32 C3            :10;
__REG32               : 6;
__REG32 C4            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff2_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficent3 Register */
typedef struct {        
__REG32 C5            :10;
__REG32               : 6;
__REG32 C6            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff3_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficent4 Register */
typedef struct {        
__REG32 C7            :10;
__REG32               : 6;
__REG32 C8            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff4_bits;

/* RGB to YCbCr 4:2:2 CSC Offset Register */
typedef struct {        
__REG32 Y_OFFSET      : 9;
__REG32               : 7;
__REG32 CBCR_OFFSET   : 9;
__REG32               : 7;
} __hw_lcdif_csc_offset_bits;

/* RGB to YCbCr 4:2:2 CSC Limit Register */
typedef struct {        
__REG32 Y_MAX         : 8;
__REG32 Y_MIN         : 8;
__REG32 CBCR_MAX      : 8;
__REG32 CBCR_MIN      : 8;
} __hw_lcdif_csc_limit_bits;

/* LCD Interface Data Register */
typedef struct {        
__REG32 DATA_ZERO     : 8;
__REG32 DATA_ONE      : 8;
__REG32 DATA_TWO      : 8;
__REG32 DATA_THREE    : 8;
} __hw_lcdif_data_bits;

/* LCD Interface Status Register */
typedef struct {        
__REG32                   :24;
__REG32 DVI_CURRENT_FIELD : 1;
__REG32 BUSY              : 1;
__REG32 TXFIFO_EMPTY      : 1;
__REG32 TXFIFO_FULL       : 1;
__REG32 LFIFO_EMPTY       : 1;
__REG32 LFIFO_FULL        : 1;
__REG32 DMA_REQ           : 1;
__REG32 PRESENT           : 1;
} __hw_lcdif_stat_bits;

/* LCD Interface Version Register */
typedef struct {        
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_lcdif_version_bits;

/* LCD Interface Debug0 Register */
typedef struct {        
__REG32                     :16;
__REG32 CUR_STATE           : 7;
__REG32 EMPTY_WORD          : 1;
__REG32 CUR_FRAME_TX        : 1;
__REG32 VSYNC               : 1;
__REG32 HSYNC               : 1;
__REG32 ENABLE              : 1;
__REG32                     : 1;
__REG32 SYNC_SIGNALS_ON_REG : 1;
__REG32 W_FOR_VSYNC_EDG_OUT : 1;
__REG32 STRMNG_END_DTC      : 1;
} __hw_lcdif_debug0_bits;

/* LCD Interface Debug1 Register */
typedef struct {        
__REG32 V_DATA_COUNT        :16;
__REG32 H_DATA_COUNT        :16;
} __hw_lcdif_debug1_bits;

/* TV Encoder Control Register */
typedef struct {        
__REG32 DAC_MUX_MODE            : 1;
__REG32                         : 2;
__REG32 DAC_DATA_FIFO_RST       : 1;
__REG32 DAC_FIFO_NO_READ        : 1;
__REG32 DAC_FIFO_NO_WRITE       : 1;
__REG32                         :20;
__REG32 TVENC_COMPONENT_PRESENT : 1;
__REG32 TVENC_SVIDEO_PRESENT    : 1;
__REG32 TVENC_COMPOSITE_PRESENT : 1;
__REG32 TVENC_MVISION_PRESENT   : 1;
__REG32 CLKGATE                 : 1;
__REG32 SFTRST                  : 1;
} __hw_tvenc_ctrl_bits;

/* TV Encoder Configuration Register */
typedef struct {        
__REG32 ENCD_MODE         : 3;
__REG32                   : 1;
__REG32 SYNC_MODE         : 3;
__REG32 VSYNC_PHS         : 1;
__REG32 HSYNC_PHS         : 1;
__REG32 FSYNC_PHS         : 1;
__REG32 FSYNC_ENBL        : 1;
__REG32                   : 1;
__REG32 CLK_PHS           : 2;
__REG32 CGAIN             : 2;
__REG32 YGAIN_SEL         : 2;
__REG32 COLOR_BAR_EN      : 1;
__REG32 NO_PED            : 1;
__REG32 PAL_SHAPE         : 1;
__REG32 ADD_YPBPR_PED     : 1;
__REG32                   : 2;
__REG32 YDEL_ADJ          : 3;
__REG32 DEFAULT_PICFORM   : 1;
__REG32                   : 4;
} __hw_tvenc_config_bits;

/* TV Encoder Filter Control Register */
typedef struct {        
__REG32                   :10;
__REG32 YS_GAINSEL        : 2;
__REG32 YS_GAINSGN        : 1;
__REG32 COEFSEL_CLPF      : 1;
__REG32 YLPF_COEFSEL      : 1;
__REG32 SEL_YSHARP        : 1;
__REG32 SEL_CLPF          : 1;
__REG32 SEL_YLPF          : 1;
__REG32 YD_OFFSETSEL      : 1;
__REG32 YSHARP_BW         : 1;
__REG32                   :12;
} __hw_tvenc_filtctrl_bits;

/* TV Sync Offset Register */
typedef struct {        
__REG32 HLC               :10;
__REG32 VSO               :10;
__REG32 HSO               :11;
__REG32                   : 1;
} __hw_tvenc_syncoffset_bits;

/* TV Encoder Horizontal Timing Sync Register 0 */
typedef struct {        
__REG32 SYNC_STRT         :10;
__REG32                   : 6;
__REG32 SYNC_END          :10;
__REG32                   : 6;
} __hw_tvenc_htimingsync0_bits;

/* TV Encoder Horizontal Timing Sync Register 1 */
typedef struct {        
__REG32 SYNC_SREND        :10;
__REG32                   : 6;
__REG32 SYNC_EQEND        :10;
__REG32                   : 6;
} __hw_tvenc_htimingsync1_bits;

/* TV Encoder Horizontal Timing Active Register */
typedef struct {        
__REG32 ACTV_STRT         :10;
__REG32                   : 6;
__REG32 ACTV_END          :10;
__REG32                   : 6;
} __hw_tvenc_htimingactive_bits;

/* TV Encoder Horizontal Timing Color Burst Register 0 */
typedef struct {        
__REG32 NBRST_STRT        :10;
__REG32                   : 6;
__REG32 WBRST_STRT        :10;
__REG32                   : 6;
} __hw_tvenc_htimingburst0_bits;

/* TV Encoder Horizontal Timing Color Burst Register 1 */
typedef struct {        
__REG32 BRST_END          :10;
__REG32                   :22;
} __hw_tvenc_htimingburst1_bits;

/* TV Encoder Vertical Timing Register 0 */
typedef struct {        
__REG32 VSTRT_SUBPH       : 6;
__REG32                   : 2;
__REG32 VSTRT_ACTV        : 6;
__REG32                   : 2;
__REG32 VSTRT_PREEQ       :10;
__REG32                   : 6;
} __hw_tvenc_vtiming0_bits;

/* TV Encoder Vertical Timing Register 1 */
typedef struct {        
__REG32 LAST_FLD_LN       :10;
__REG32                   : 6;
__REG32 VSTRT_SERRA       : 6;
__REG32                   : 2;
__REG32 VSTRT_POSTEQ      : 6;
__REG32                   : 2;
} __hw_tvenc_vtiming1_bits;

/* TV Encoder Miscellaneous Line Control Register */
typedef struct {        
__REG32 Y_BLANK_CTRL      : 2;
__REG32 CS_INVERT_CTRL    : 1;
__REG32                   : 1;
__REG32 AGC_LVL_CTRL      : 2;
__REG32 BRUCHB            : 2;
__REG32 FSC_PHASE_RST     : 2;
__REG32 PAL_FSC_PHASE_ALT : 1;
__REG32 NTSC_LN_CNT       : 1;
__REG32                   : 4;
__REG32 LPF_RST_OFF       : 9;
__REG32                   : 7;
} __hw_tvenc_misc_bits;

/* TV Encoder Copy Protect Register */
typedef struct {        
__REG32 WSS_CGMS_DATA     :14;
__REG32 CGMS_ENBL         : 1;
__REG32 WSS_ENBL          : 1;
__REG32                   :16;
} __hw_tvenc_copyprotect_bits;

/* TV Encoder Closed Caption Register */
typedef struct {        
__REG32 CC_DATA           :16;
__REG32 CC_FILL           : 2;
__REG32 CC_ENBL           : 2;
__REG32                   :12;
} __hw_tvenc_closedcaption_bits;

/* TV Encoder Color Burst Register */
typedef struct {        
__REG32                   :16;
__REG32 PBA               : 8;
__REG32 NBA               : 8;
} __hw_tvenc_colorburst_bits;

/* TV Encoder DAC Control Register */
typedef struct {        
__REG32 CASC_ADJ            : 2;
__REG32 HALF_CURRENT        : 1;
__REG32 NO_INTERNAL_TERM    : 1;
__REG32 RVAL                : 3;
__REG32 LOWER_SIGNAL        : 1;
__REG32 DUMP_TOVDD1         : 1;
__REG32                     : 2;
__REG32 WELL_TOVDD          : 1;
__REG32 PWRUP1              : 1;
__REG32                     : 2;
__REG32 BYPASS_ACT_CASCODE  : 1;
__REG32 SELECT_CLK          : 1;
__REG32 INVERT_CLK          : 1;
__REG32 GAINUP              : 1;
__REG32 GAINDN              : 1;
__REG32 JACK_DIS_ADJ        : 2;
__REG32 DISABLE_GND_DETECT  : 1;
__REG32 TEST1               : 1;
__REG32 JACK1_DET_EN        : 1;
__REG32                     : 2;
__REG32 TEST2               : 1;
__REG32 JACK1_DIS_DET_EN    : 1;
__REG32                     : 2;
__REG32 TEST3               : 1;
} __hw_tvenc_dacctrl_bits;

/* TV Encoder DAC Status Register */
typedef struct {        
__REG32 ENIRQ_JACK          : 1;
__REG32 JACK1_DET_IRQ       : 1;
__REG32                     : 2;
__REG32 JACK1_DIS_DET_IRQ   : 1;
__REG32                     : 2;
__REG32 JACK1_GROUNDED      : 1;
__REG32                     : 2;
__REG32 JACK1_DET_STATUS    : 1;
__REG32                     :21;
} __hw_tvenc_dacstatus_bits;

/* TV Encoder vDAC Test Register */
typedef struct {        
__REG32 DATA                  :10;
__REG32 TEST_FIFO_FULL        : 1;
__REG32 BYPASS_PIX_INT_DROOP  : 1;
__REG32 BYPASS_PIX_INT        : 1;
__REG32 ENABLE_PIX_INT_GAIN   : 1;
__REG32                       :18;
} __hw_tvenc_vdactest_bits;

/* TV Encoder Version Register */
typedef struct {        
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_tvenc_version_bits;

/* SSP Control Register 0 */
typedef struct {        
__REG32 XFER_COUNT        :16;
__REG32 ENABLE            : 1;
__REG32 GET_RESP          : 1;
__REG32 CHECK_RESP        : 1;
__REG32 LONG_RESP         : 1;
__REG32 WAIT_FOR_CMD      : 1;
__REG32 WAIT_FOR_IRQ      : 1;
__REG32 BUS_WIDTH         : 2;
__REG32 DATA_XFER         : 1;
__REG32 READ              : 1;
__REG32 IGNORE_CRC        : 1;
__REG32 LOCK_CS           : 1;
__REG32 SDIO_IRQ_CHECK    : 1;
__REG32 RUN               : 1;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __hw_ssp_ctrl0_bits;

/* SD/MMC Command Register 0 */
typedef struct {        
__REG32 CMD               : 8;
__REG32 BLOCK_COUNT       : 8;
__REG32 BLOCK_SIZE        : 4;
__REG32 APPEND_8CYC       : 1;
__REG32 CONT_CLKING_EN    : 1;
__REG32 SLOW_CLKING_EN    : 1;
__REG32                   : 9;
} __hw_ssp_cmd0_bits;

/* SSP Timing Register */
typedef struct {        
__REG32 CLOCK_RATE        : 8;
__REG32 CLOCK_DIVIDE      : 8;
__REG32 TIMEOUT           :16;
} __hw_ssp_timing_bits;

/* SSP Control Register 1 */
typedef struct {        
__REG32 SSP_MODE            : 4;
__REG32 WORD_LENGTH         : 4;
__REG32 SLAVE_MODE          : 1;
__REG32 POLARITY            : 1;
__REG32 PHASE               : 1;
__REG32 SLAVE_OUT_DISABLE   : 1;
__REG32                     : 1;
__REG32 DMA_ENABLE          : 1;
__REG32 FIFO_OVERRUN_IRQ_EN : 1;
__REG32 FIFO_OVERRUN_IRQ    : 1;
__REG32 RECV_TIMEOUT_IRQ_EN : 1;
__REG32 RECV_TIMEOUT_IRQ    : 1;
__REG32                     : 2;
__REG32 FIFO_UNDERRUN_EN    : 1;
__REG32 FIFO_UNDERRUN_IRQ   : 1;
__REG32 DATA_CRC_IRQ_EN     : 1;
__REG32 DATA_CRC_IRQ        : 1;
__REG32 DATA_TIMEOUT_IRQ_EN : 1;
__REG32 DATA_TIMEOUT_IRQ    : 1;
__REG32 RESP_TIMEOUT_IRQ_EN : 1;
__REG32 RESP_TIMEOUT_IRQ    : 1;
__REG32 RESP_ERR_IRQ_EN     : 1;
__REG32 RESP_ERR_IRQ        : 1;
__REG32 SDIO_IRQ_EN         : 1;
__REG32 SDIO_IRQ            : 1;
} __hw_ssp_ctrl1_bits;

/* SSP Status Register */
typedef struct {        
__REG32 BUSY                : 1;
__REG32                     : 1;
__REG32 DATA_BUSY           : 1;
__REG32 CMD_BUSY            : 1;
__REG32 FIFO_UNDRFLW        : 1;
__REG32 FIFO_EMPTY          : 1;
__REG32                     : 2;
__REG32 FIFO_FULL           : 1;
__REG32 FIFO_OVRFLW         : 1;
__REG32                     : 1;
__REG32 RECV_TIMEOUT_STAT   : 1;
__REG32 TIMEOUT             : 1;
__REG32 DATA_CRC_ERR        : 1;
__REG32 RESP_TIMEOUT        : 1;
__REG32 RESP_ERR            : 1;
__REG32 RESP_CRC_ERR        : 1;
__REG32 SDIO_IRQ            : 1;
__REG32 DMAEND              : 1;
__REG32 DMAREQ              : 1;
__REG32 DMATERM             : 1;
__REG32 DMASENSE            : 1;
__REG32                     : 6;
__REG32 CARD_DETECT         : 1;
__REG32 SD_PRESENT          : 1;
__REG32                     : 1;
__REG32 PRESENT             : 1;
} __hw_ssp_status_bits;

/* SSP Debug Register */
typedef struct {        
__REG32 SSP_RXD         : 8;
__REG32 SSP_RESP        : 1;
__REG32 SSP_CMD         : 1;
__REG32 CMD_SM          : 2;
__REG32 MMC_SM          : 4;
__REG32 DMA_SM          : 3;
__REG32 CMD_OE          : 1;
__REG32                 : 4;
__REG32 DAT_SM          : 3;
__REG32 DATA_STALL      : 1;
__REG32 DATACRC_ERR     : 4;
} __hw_ssp_debug_bits;

/* SSP Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_ssp_version_bits;

/* Rotary Decoder Control Register */
typedef struct {        
__REG32 SELECT_A        : 3;
__REG32                 : 1;
__REG32 SELECT_B        : 3;
__REG32                 : 1;
__REG32 POLARITY_A      : 1;
__REG32 POLARITY_B      : 1;
__REG32 OVERSAMPLE      : 2;
__REG32 RELATIVE        : 1;
__REG32                 : 3;
__REG32 DIVIDER         : 6;
__REG32 STATE           : 3;
__REG32 TIM0_PRESENT    : 1;
__REG32 TIM1_PRESENT    : 1;
__REG32 TIM2_PRESENT    : 1;
__REG32 TIM3_PRESENT    : 1;
__REG32 ROTARY_PRESENT  : 1;
__REG32 CLKGATE         : 1;
__REG32 SFTRST          : 1;
} __hw_timrot_rotctrl_bits;

/* Rotary Decoder Up/Down Counter Register */
typedef struct {        
__REG32 UPDOWN          :16;
__REG32                 :16;
} __hw_timrot_rotcount_bits;

/* Timer 0 Control and Status Register */
/* Timer 1 Control and Status Register */
/* Timer 2 Control and Status Register */
typedef struct {        
__REG32 SELECT          : 4;
__REG32 PRESCALE        : 2;
__REG32 RELOAD          : 1;
__REG32 UPDATE          : 1;
__REG32 POLARITY        : 1;
__REG32                 : 5;
__REG32 IRQ_EN          : 1;
__REG32 IRQ             : 1;
__REG32                 :16;
} __hw_timrot_timctrlx_bits;

/* Timer 0 Count Register */
/* Timer 1 Count Register */
/* Timer 2 Count Register */
typedef struct {        
__REG32 FIXED_COUNT     :16;
__REG32 RUNNING_COUNT   :16;
} __hw_timrot_timcountx_bits;

/* Timer 3 Control and Status Register */
typedef struct {        
__REG32 SELECT          : 4;
__REG32 PRESCALE        : 2;
__REG32 RELOAD          : 1;
__REG32 UPDATE          : 1;
__REG32 POLARITY        : 1;
__REG32 DUTY_CYCLE      : 1;
__REG32 DUTY_VALID      : 1;
__REG32                 : 3;
__REG32 IRQ_EN          : 1;
__REG32 IRQ             : 1;
__REG32 TEST_SIGNAL     : 4;
__REG32                 :12;
} __hw_timrot_timctrl3_bits;

/* Timer 3 Count Register */
typedef struct {        
__REG32 HIGH_FIXED_COUNT  :16;
__REG32 LOW_RUNNING_COUNT :16;
} __hw_timrot_timcount3_bits;

/* TIMROT Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_timrot_version_bits;

/* Real-Time Clock Control Register */
typedef struct {        
__REG32 ALARM_IRQ_EN          : 1;
__REG32 ONEMSEC_IRQ_EN        : 1;
__REG32 ALARM_IRQ             : 1;
__REG32 ONEMSEC_IRQ           : 1;
__REG32 WATCHDOGEN            : 1;
__REG32 FORCE_UPDATE          : 1;
__REG32 SUPPRESS_COPY2ANALOG  : 1;
__REG32                       :23;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_rtc_ctrl_bits;

/* Real-Time Clock Status Register */
typedef struct {        
__REG32                       : 8;
__REG32 NEW_REGS0             : 1;
__REG32 NEW_REGS1             : 1;
__REG32 NEW_REGS2             : 1;
__REG32 NEW_REGS3             : 1;
__REG32 NEW_REGS4             : 1;
__REG32 NEW_REGS5             : 1;
__REG32 NEW_REGS6             : 1;
__REG32 NEW_REGS7             : 1;
__REG32 STALE_REGS0           : 1;
__REG32 STALE_REGS1           : 1;
__REG32 STALE_REGS2           : 1;
__REG32 STALE_REGS3           : 1;
__REG32 STALE_REGS4           : 1;
__REG32 STALE_REGS5           : 1;
__REG32 STALE_REGS6           : 1;
__REG32 STALE_REGS7           : 1;
__REG32                       : 3;
__REG32 XTAL32768_PRESENT     : 1;
__REG32 XTAL32000_PRESENT     : 1;
__REG32 WATCHDOG_PRESENT      : 1;
__REG32 ALARM_PRESENT         : 1;
__REG32 RTC_PRESENT           : 1;
} __hw_rtc_stat_bits;

/* Persistent State Register 0 */
typedef struct {        
__REG32 CLOCKSOURCE           : 1;
__REG32 ALARM_WAKE_EN         : 1;
__REG32 ALARM_EN              : 1;
__REG32 LCK_SECS              : 1;
__REG32 XTAL24MHZ_PWRUP       : 1;
__REG32 XTAL32KHZ_PWRUP       : 1;
__REG32 XTAL32_FREQ           : 1;
__REG32 ALARM_WAKE            : 1;
__REG32 MSEC_RES              : 5;
__REG32 DISABLE_XTALOK        : 1;
__REG32 LOWERBIAS             : 2;
__REG32 DISABLE_PSWITCH       : 1;
__REG32 AUTO_RESTART          : 1;
__REG32 SPARE_ANALOG          :14;
} __hw_rtc_persistent0_bits;

/* Real-Time Clock Debug Register */
typedef struct {        
__REG32 WATCHDOG_RESET        : 1;
__REG32 WATCHDOG_RESET_MASK   : 1;
__REG32                       :30;
} __hw_rtc_debug_bits;

/* Real-Time Clock Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_rtc_version_bits;

/* PWM Control and Status Register Description */
typedef struct {        
__REG32 PWM0_ENABLE           : 1;
__REG32 PWM1_ENABLE           : 1;
__REG32 PWM2_ENABLE           : 1;
__REG32 PWM3_ENABLE           : 1;
__REG32 PWM4_ENABLE           : 1;
__REG32 PWM2_ANA_CTRL_ENABLE  : 1;
__REG32 OUTPUT_CUTOFF_EN      : 1;
__REG32                       :18;
__REG32 PWM0_PRESENT          : 1;
__REG32 PWM1_PRESENT          : 1;
__REG32 PWM2_PRESENT          : 1;
__REG32 PWM3_PRESENT          : 1;
__REG32 PWM4_PRESENT          : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_pwm_ctrl_bits;

/* PWM Channel 0 Active Register */
/* PWM Channel 1 Active Register */
/* PWM Channel 2 Active Register */
/* PWM Channel 3 Active Register */
/* PWM Channel 4 Active Register */
typedef struct {        
__REG32 ACTIVE          :16;
__REG32 INACTIVE        :16;
} __hw_pwm_activex_bits;

/* PWM Channel 0 Period Register */
/* PWM Channel 1 Period Register */
/* PWM Channel 2 Period Register */
/* PWM Channel 3 Period Register */
/* PWM Channel 4 Period Register */
typedef struct {        
__REG32 PERIOD          :16;
__REG32 ACTIVE_STATE    : 2;
__REG32 INACTIVE_STATE  : 2;
__REG32 CDIV            : 3;
__REG32 MATT            : 1;
__REG32 MATT_SEL        : 1;
__REG32                 : 7;
} __hw_pwm_periodx_bits;

/* PWM Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_pwm_version_bits;

/* I2C Control Register 0 */
typedef struct {        
__REG32 XFER_COUNT            :16;
__REG32 DIRECTION             : 1;
__REG32 MASTER_MODE           : 1;
__REG32 SLAVE_ADDRESS_ENABLE  : 1;
__REG32 PRE_SEND_START        : 1;
__REG32 POST_SEND_STOP        : 1;
__REG32 RETAIN_CLOCK          : 1;
__REG32 CLOCK_HELD            : 1;
__REG32                       : 1;
__REG32 PIO_MODE              : 1;
__REG32 SEND_NAK_ON_LAST      : 1;
__REG32 ACKNOWLEDGE           : 1;
__REG32 PRE_ACK               : 1;
__REG32                       : 1;
__REG32 RUN                   : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_i2c_ctrl0_bits;

/* I2C Timing Register 0 */
typedef struct {        
__REG32 RCV_COUNT           :10;
__REG32                     : 6;
__REG32 HIGH_COUNT          :10;
__REG32                     : 6;
} __hw_i2c_timing0_bits;

/* I2C Timing Register 1 */
typedef struct {        
__REG32 XMIT_COUNT          :10;
__REG32                     : 6;
__REG32 LOW_COUNT           :10;
__REG32                     : 6;
} __hw_i2c_timing1_bits;

/* I2C Timing Register 2 */
typedef struct {        
__REG32 LEADIN_COUNT        :10;
__REG32                     : 6;
__REG32 BUS_FREE            :10;
__REG32                     : 6;
} __hw_i2c_timing2_bits;

/* I2C Control Register 1 */
typedef struct {        
__REG32 SLAVE_IRQ               : 1;
__REG32 SLAVE_STOP_IRQ          : 1;
__REG32 MASTER_LOSS_IRQ         : 1;
__REG32 EARLY_TERM_IRQ          : 1;
__REG32 OSIZE_XFER_TERM_IRQ     : 1;
__REG32 NO_SLAVE_ACK_IRQ        : 1;
__REG32 D_ENGINE_CMPLT_IRQ      : 1;
__REG32 BUS_FREE_IRQ            : 1;
__REG32 SLAVE_IRQ_EN            : 1;
__REG32 SLAVE_STOP_IRQ_EN       : 1;
__REG32 MASTER_LOSS_IRQ_EN      : 1;
__REG32 EARLY_TERM_IRQ_EN       : 1;
__REG32 OSIZE_XFER_TERM_IRQ_EN  : 1;
__REG32 NO_SLAVE_ACK_IRQ_EN     : 1;
__REG32 D_ENGINE_CMPLT_IRQ_EN   : 1;
__REG32 BUS_FREE_IRQ_EN         : 1;
__REG32                         : 8;
__REG32 BCAST_SLAVE_EN          : 1;
__REG32 FORCE_CLK_IDLE          : 1;
__REG32 FORCE_DATA_IDLE         : 1;
__REG32 ACK_MODE                : 1;
__REG32 CLR_GOT_A_NAK           : 1;
__REG32                         : 3;
} __hw_i2c_ctrl1_bits;

/* I2C Status Register */
typedef struct {        
__REG32 SLAVE_IRQ_SMR           : 1;
__REG32 SLAVE_STOP_IRQ_SMR      : 1;
__REG32 MASTER_LOSS_IRQ_SMR     : 1;
__REG32 EARLY_TERM_IRQ_SMR      : 1;
__REG32 OSIZE_XFER_TERM_IRQ_SMR : 1;
__REG32 NO_SLAVE_ACK_IRQ_SMR    : 1;
__REG32 D_ENGINE_CMPLT_IRQ_SMR  : 1;
__REG32 BUS_FREE_IRQ_SMR        : 1;
__REG32 SLAVE_BUSY              : 1;
__REG32 DATA_ENGINE_BUSY        : 1;
__REG32 CLK_GEN_BUSY            : 1;
__REG32 BUS_BUSY                : 1;
__REG32 DATA_ENGINE_DMA_WAIT    : 1;
__REG32 SLAVE_SEARCHING         : 1;
__REG32 SLAVE_FOUND             : 1;
__REG32 SLAVE_ADDR_EQ_ZERO      : 1;
__REG32 RCVD_SLAVE_ADDR         : 8;
__REG32                         : 4;
__REG32 GOT_A_NAK               : 1;
__REG32 ANY_ENABLED_IRQ         : 1;
__REG32                         : 1;
__REG32 MASTER_PRESENT          : 1;
} __hw_i2c_stat_bits;

/* I2C Device Debug Register 0 */
typedef struct {        
__REG32 SLAVE_STATE           :10;
__REG32 SLAVE_HOLD_CLK        : 1;
__REG32 TESTMODE              : 1;
__REG32 CHANGE_TOGGLE         : 1;
__REG32 GRAB_TOGGLE           : 1;
__REG32 STOP_TOGGLE           : 1;
__REG32 START_TOGGLE          : 1;
__REG32 DMA_STATE             :10;
__REG32                       : 2;
__REG32 DMATERMINATE          : 1;
__REG32 DMAKICK               : 1;
__REG32 DMAENDCMD             : 1;
__REG32 DMAREQ                : 1;
} __hw_i2c_debug0_bits;

/* I2C Device Debug Register 1 */
typedef struct {        
__REG32 FORCE_I2C_CLK_OE      : 1;
__REG32 FORCE_I2C_DATA_OE     : 1;
__REG32 FORCE_RCV_ACK         : 1;
__REG32 FORCE_ARB_LOSS        : 1;
__REG32 FORCE_CLK_ON          : 1;
__REG32                       : 3;
__REG32 LOCAL_SLAVE_TEST      : 1;
__REG32 LST_MODE              : 2;
__REG32                       : 5;
__REG32 CLK_GEN_STATE         : 8;
__REG32 DMA_BYTE_ENABLES      : 4;
__REG32                       : 2;
__REG32 I2C_DATA_IN           : 1;
__REG32 I2C_CLK_IN            : 1;
} __hw_i2c_debug1_bits;

/* I2C Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_i2c_version_bits;

/* UART Receive DMA Control Register */
typedef struct {        
__REG32 XFER_COUNT      :16;
__REG32 RXTIMEOUT       :11;
__REG32 RXTO_ENABLE     : 1;
__REG32 RX_SOURCE       : 1;
__REG32 RUN             : 1;
__REG32 CLKGATE         : 1;
__REG32 SFTRST          : 1;
} __hw_uartapp_ctrl0_bits;

/* UART Transmit DMA Control Register */
typedef struct {        
__REG32 XFER_COUNT      :16;
__REG32                 :12;
__REG32 RUN             : 1;
__REG32                 : 3;
} __hw_uartapp_ctrl1_bits;

/* UART Control Register */
typedef struct {        
__REG32 UARTEN          : 1;
__REG32                 : 5;
__REG32 USE_LCR2        : 1;
__REG32 LBE             : 1;
__REG32 TXE             : 1;
__REG32 RXE             : 1;
__REG32 DTR             : 1;
__REG32 RTS             : 1;
__REG32 OUT1            : 1;
__REG32 OUT2            : 1;
__REG32 RTSEN           : 1;
__REG32 CTSEN           : 1;
__REG32 TXIFLSEL        : 3;
__REG32                 : 1;
__REG32 RXIFLSEL        : 3;
__REG32                 : 1;
__REG32 RXDMAE          : 1;
__REG32 TXDMAE          : 1;
__REG32 DMAONERR        : 1;
__REG32 RTS_SEMAPHORE   : 1;
__REG32 INVERT_RX       : 1;
__REG32 INVERT_TX       : 1;
__REG32 INVERT_CTS      : 1;
__REG32 INVERT_RTS      : 1;
} __hw_uartapp_ctrl2_bits;

/* UART Line Control Register */
typedef struct {        
__REG32 BRK             : 1;
__REG32 PEN             : 1;
__REG32 EPS             : 1;
__REG32 STP2            : 1;
__REG32 FEN             : 1;
__REG32 WLEN            : 2;
__REG32 SPS             : 1;
__REG32 BAUD_DIVFRAC    : 6;
__REG32                 : 2;
__REG32 BAUD_DIVINT     :16;
} __hw_uartapp_linectrl_bits;

/* UART Line Control 2 Register */
typedef struct {        
__REG32                 : 1;
__REG32 PEN             : 1;
__REG32 EPS             : 1;
__REG32 STP2            : 1;
__REG32 FEN             : 1;
__REG32 WLEN            : 2;
__REG32 SPS             : 1;
__REG32 BAUD_DIVFRAC    : 6;
__REG32                 : 2;
__REG32 BAUD_DIVINT     :16;
} __hw_uartapp_linectrl2_bits;

/* UART Interrupt Register */
typedef struct {        
__REG32 RIMIS           : 1;
__REG32 CTSMIS          : 1;
__REG32 DCDMIS          : 1;
__REG32 DSRMIS          : 1;
__REG32 RXIS            : 1;
__REG32 TXIS            : 1;
__REG32 RTIS            : 1;
__REG32 FEIS            : 1;
__REG32 PEIS            : 1;
__REG32 BEIS            : 1;
__REG32 OEIS            : 1;
__REG32                 : 5;
__REG32 RIMIEN          : 1;
__REG32 CTSMIEN         : 1;
__REG32 DCDMIEN         : 1;
__REG32 DSRMIEN         : 1;
__REG32 RXIEN           : 1;
__REG32 TXIEN           : 1;
__REG32 RTIEN           : 1;
__REG32 FEIEN           : 1;
__REG32 PEIEN           : 1;
__REG32 BEIEN           : 1;
__REG32 OEIEN           : 1;
__REG32                 : 5;
} __hw_uartapp_intr_bits;

/* UART Status Register */
typedef struct {        
__REG32 RXCOUNT         :16;
__REG32 FERR            : 1;
__REG32 PERR            : 1;
__REG32 BERR            : 1;
__REG32 OERR            : 1;
__REG32 RXBYTE_INVALID  : 4;
__REG32 RXFE            : 1;
__REG32 TXFF            : 1;
__REG32 RXFF            : 1;
__REG32 TXFE            : 1;
__REG32 CTS             : 1;
__REG32 BUSY            : 1;
__REG32 HISPEED         : 1;
__REG32 PRESENT         : 1;
} __hw_uartapp_stat_bits;

/* UART Debug Register */
typedef struct {        
__REG32 RXDMARQ         : 1;
__REG32 TXDMARQ         : 1;
__REG32 RXCMDEND        : 1;
__REG32 TXCMDEND        : 1;
__REG32 RXDMARUN        : 1;
__REG32 TXDMARUN        : 1;
__REG32                 : 4;
__REG32 RXFBAUD_DIV     : 6;
__REG32 RXIBAUD_DIV     :16;
} __hw_uartapp_debug_bits;

/* UART Version Register */
typedef struct {        
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_uartapp_version_bits;

/* UART AutoBaud Register */
typedef struct {        
__REG32 BAUD_DETECT_ENABLE  : 1;
__REG32 START_BAUD_DETECT   : 1;
__REG32 START_WITH_RUNBIT   : 1;
__REG32 TWO_REF_CHARS       : 1;
__REG32 UPDATE_TX           : 1;
__REG32                     :11;
__REG32 REFCHAR0            : 8;
__REG32 REFCHAR1            : 8;
} __hw_uartapp_autobaud_bits;

/* DBG UART Data Register */
typedef struct {        
__REG32 DATA                : 8;
__REG32 FE                  : 1;
__REG32 PE                  : 1;
__REG32 BE                  : 1;
__REG32 OE                  : 1;
__REG32                     :20;
} __hw_uartdbgdr_bits;

/* DBG UART Receive Status Register (Read) / Error Clear Register (Write) */
typedef struct {        
__REG32 FE                  : 1;
__REG32 PE                  : 1;
__REG32 BE                  : 1;
__REG32 OE                  : 1;
__REG32 EC                  : 4;
__REG32                     :24;
} __hw_uartdbgrsr_ecr_bits;

/* DBG UART Flag Register */
typedef struct {        
__REG32 CTS                 : 1;
__REG32 DSR                 : 1;
__REG32 DCD                 : 1;
__REG32 BUSY                : 1;
__REG32 RXFE                : 1;
__REG32 TXFF                : 1;
__REG32 RXFF                : 1;
__REG32 TXFE                : 1;
__REG32 RI                  : 1;
__REG32                     :23;
} __hw_uartdbgfr_bits;

/* DBG UART IrDA Low-Power Counter Register */
typedef struct {        
__REG32 ILPDVSR             : 8;
__REG32                     :24;
} __hw_uartdbgilpr_bits;

/* DBG UART Integer Baud Rate Divisor Register */
typedef struct {        
__REG32 BAUD_DIVINT         :16;
__REG32                     :16;
} __hw_uartdbgibrd_bits;

/* DBG UART Fractional Baud Rate Divisor Register */
typedef struct {        
__REG32 BAUD_DIVFRAC        : 6;
__REG32                     :26;
} __hw_uartdbgfbrd_bits;

/* DBG UART Line Control Register, HIGH Byte */
typedef struct {        
__REG32 BRK                 : 1;
__REG32 PEN                 : 1;
__REG32 EPS                 : 1;
__REG32 STP2                : 1;
__REG32 FEN                 : 1;
__REG32 WLEN                : 2;
__REG32 SPS                 : 1;
__REG32                     :24;
} __hw_uartdbglcr_h_bits;

/* DBG UART Control Register */
typedef struct {        
__REG32 UARTEN              : 1;
__REG32 SIREN               : 1;
__REG32 SIRLP               : 1;
__REG32                     : 4;
__REG32 LBE                 : 1;
__REG32 TXE                 : 1;
__REG32 RXE                 : 1;
__REG32 DTR                 : 1;
__REG32 RTS                 : 1;
__REG32 OUT1                : 1;
__REG32 OUT2                : 1;
__REG32 RTSEN               : 1;
__REG32 CTSEN               : 1;
__REG32                     :16;
} __hw_uartdbgcr_bits;

/* DBG UART Interrupt FIFO Level Select Register */
typedef struct {        
__REG32 TXIFLSEL            : 3;
__REG32 RXIFLSEL            : 3;
__REG32                     :26;
} __hw_uartdbgifls_bits;

/* DBG UART Interrupt Mask Set/Clear Register */
typedef struct {        
__REG32 RIMIM               : 1;
__REG32 CTSMIM              : 1;
__REG32 DCDMIM              : 1;
__REG32 DSRMIM              : 1;
__REG32 RXIM                : 1;
__REG32 TXIM                : 1;
__REG32 RTIM                : 1;
__REG32 FEIM                : 1;
__REG32 PEIM                : 1;
__REG32 BEIM                : 1;
__REG32 OEIM                : 1;
__REG32                     :21;
} __hw_uartdbgimsc_bits;

/* DBG UART Raw Interrupt Status Register Register */
typedef struct {        
__REG32 RIRMIS              : 1;
__REG32 CTSRMIS             : 1;
__REG32 DCDRMIS             : 1;
__REG32 DSRRMIS             : 1;
__REG32 RXRIS               : 1;
__REG32 TXRIS               : 1;
__REG32 RTRIS               : 1;
__REG32 FERIS               : 1;
__REG32 PERIS               : 1;
__REG32 BERIS               : 1;
__REG32 OERIS               : 1;
__REG32                     :21;
} __hw_uartdbgris_bits;

/* DBG UART Masked Interrupt Status Register */
typedef struct {        
__REG32 RIMMIS              : 1;
__REG32 CTSMMIS             : 1;
__REG32 DCDMMIS             : 1;
__REG32 DSRMMIS             : 1;
__REG32 RXMIS               : 1;
__REG32 TXMIS               : 1;
__REG32 RTMIS               : 1;
__REG32 FEMIS               : 1;
__REG32 PEMIS               : 1;
__REG32 BEMIS               : 1;
__REG32 OEMIS               : 1;
__REG32                     :21;
} __hw_uartdbgmis_bits;

/* DBG UART Interrupt Clear Register */
typedef struct {        
__REG32 RIMIC               : 1;
__REG32 CTSMIC              : 1;
__REG32 DCDMIC              : 1;
__REG32 DSRMIC              : 1;
__REG32 RXIC                : 1;
__REG32 TXIC                : 1;
__REG32 RTIC                : 1;
__REG32 FEIC                : 1;
__REG32 PEIC                : 1;
__REG32 BEIC                : 1;
__REG32 OEIC                : 1;
__REG32                     :21;
} __hw_uartdbgicr_bits;

/* AUDIOIN Control Register */
typedef struct {        
__REG32 RUN                 : 1;
__REG32 FIFO_ERROR_IRQ_EN   : 1;
__REG32 FIFO_OVERFLOW_IRQ   : 1;
__REG32 FIFO_UNDERFLOW_IRQ  : 1;
__REG32 LOOPBACK            : 1;
__REG32 WORD_LENGTH         : 1;
__REG32 HPF_ENABLE          : 1;
__REG32 OFFSET_ENABLE       : 1;
__REG32 INVERT_1BIT         : 1;
__REG32 EDGE_SYNC           : 1;
__REG32 LR_SWAP             : 1;
__REG32                     : 5;
__REG32 DMAWAIT_COUNT       : 5;
__REG32                     : 9;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_audioin_ctrl_bits;

/* AUDIOIN Status Register */
typedef struct {        
__REG32                     :31;
__REG32 ADC_PRESENT         : 1;
} __hw_audioin_stat_bits;

/* AUDIOIN Sample Rate Register */
typedef struct {        
__REG32 SRC_FRAC            :13;
__REG32                     : 3;
__REG32 SRC_INT             : 5;
__REG32                     : 3;
__REG32 SRC_HOLD            : 3;
__REG32                     : 1;
__REG32 BASEMULT            : 3;
__REG32 OSR                 : 1;
} __hw_audioin_adcsrr_bits;

/* AUDIOIN Volume Register */
typedef struct {        
__REG32 VOLUME_RIGHT        : 8;
__REG32                     : 4;
__REG32 VOLUME_UPDATE_RIGHT : 1;
__REG32                     : 3;
__REG32 VOLUME_LEFT         : 8;
__REG32                     : 1;
__REG32 EN_ZCD              : 1;
__REG32                     : 2;
__REG32 VOLUME_UPDATE_LEFT  : 1;
__REG32                     : 3;
} __hw_audioin_adcvolume_bits;

/* AUDIOIN Debug Register */
typedef struct {        
__REG32 FIFO_STATUS         : 1;
__REG32 DMA_PREQ            : 1;
__REG32 SET_INTR3_HND_SHAKE : 1;
__REG32 ADC_DMAR_HND_SHAKE  : 1;
__REG32                     :27;
__REG32 ENABLE_ADCDMA       : 1;
} __hw_audioin_adcdebug_bits;

/* ADC Mux Volume and Select Control Register */
typedef struct {        
__REG32 GAIN_RIGHT            : 4;
__REG32 SELECT_RIGHT          : 2;
__REG32                       : 2;
__REG32 GAIN_LEFT             : 4;
__REG32 SELECT_LEFT           : 2;
__REG32                       :10;
__REG32 MUTE                  : 1;
__REG32 EN_ADC_ZCD            : 1;
__REG32                       : 2;
__REG32 VOLUME_UPDATE_PENDING : 1;
__REG32                       : 3;
} __hw_audioin_adcvol_bits;

/* Microphone and Line Control Register */
typedef struct {        
__REG32 MIC_GAIN            : 2;
__REG32                     : 2;
__REG32 MIC_CHOPCLK         : 2;
__REG32                     :10;
__REG32 MIC_BIAS            : 3;
__REG32                     : 1;
__REG32 MIC_RESISTOR        : 2;
__REG32                     : 2;
__REG32 MIC_SELECT          : 1;
__REG32                     : 3;
__REG32 DIVIDE_LINE2        : 1;
__REG32 DIVIDE_LINE1        : 1;
__REG32                     : 2;
} __hw_audioin_micline_bits;

/* Analog Clock Control Register */
typedef struct {        
__REG32 ADCDIV              : 3;
__REG32                     : 1;
__REG32 ADCCLK_SHIFT        : 2;
__REG32                     : 2;
__REG32 INVERT_ADCCLK       : 1;
__REG32 SLOW_DITHER         : 1;
__REG32 DITHER_OFF          : 1;
__REG32                     :20;
__REG32 CLKGATE             : 1;
} __hw_audioin_anaclkctrl_bits;

/* AUDIOIN Read Data Register */
typedef struct {        
__REG32 LOW                 :16;
__REG32 HIGH                :16;
} __hw_audioin_data_bits;

/* AUDIOOUT Control Register */
typedef struct {        
__REG32 RUN                 : 1;
__REG32 FIFO_ERROR_IRQ_EN   : 1;
__REG32 FIFO_OVERFLOW_IRQ   : 1;
__REG32 FIFO_UNDERFLOW_IRQ  : 1;
__REG32 LOOPBACK            : 1;
__REG32 DAC_ZERO_ENABLE     : 1;
__REG32 WORD_LENGTH         : 1;
__REG32                     : 1;
__REG32 SS3D_EFFECT         : 2;
__REG32                     : 2;
__REG32 INVERT_1BIT         : 1;
__REG32 EDGE_SYNC           : 1;
__REG32 LR_SWAP             : 1;
__REG32                     : 1;
__REG32 DMAWAIT_COUNT       : 5;
__REG32                     : 9;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_audioout_ctrl_bits;

/* AUDIOOUT Status Register */
typedef struct {        
__REG32                     :31;
__REG32 DAC_PRESENT         : 1;
} __hw_audioout_stat_bits;

/* AUDIOOUT Sample Rate Register */
typedef struct {        
__REG32 SRC_FRAC            :13;
__REG32                     : 3;
__REG32 SRC_INT             : 5;
__REG32                     : 3;
__REG32 SRC_HOLD            : 3;
__REG32                     : 1;
__REG32 BASEMULT            : 3;
__REG32                     : 1;
} __hw_audioout_dacsrr_bits;

/* AUDIOOUT Volume Register */
typedef struct {        
__REG32 VOLUME_RIGHT        : 8;
__REG32 MUTE_RIGHT          : 1;
__REG32                     : 3;
__REG32 VOLUME_UPDATE_RIGHT : 1;
__REG32                     : 3;
__REG32 VOLUME_LEFT         : 8;
__REG32 MUTE_LEFT           : 1;
__REG32 EN_ZCD              : 1;
__REG32                     : 2;
__REG32 VOLUME_UPDATE_LEFT  : 1;
__REG32                     : 3;
} __hw_audioout_dacvolume_bits;

/* AUDIOOUT Debug Register */
typedef struct {        
__REG32 FIFO_STATUS         : 1;
__REG32 DMA_PREQ            : 1;
__REG32 SET_INTR0_HND_SHAKE : 1;
__REG32 SET_INTR1_HND_SHAKE : 1;
__REG32 SET_INTR0_CLK_CROSS : 1;
__REG32 SET_INTR1_CLK_CROSS : 1;
__REG32                     : 2;
__REG32 RAM_SS              : 4;
__REG32                     :19;
__REG32 ENABLE_DACDMA       : 1;
} __hw_audioout_dacdebug_bits;

/* Headphone Volume and Select Control */
typedef struct {        
__REG32 VOL_RIGHT             : 7;
__REG32                       : 1;
__REG32 VOL_LEFT              : 7;
__REG32                       : 1;
__REG32 SELECT                : 1;
__REG32                       : 7;
__REG32 MUTE                  : 1;
__REG32 EN_MSTR_ZCD           : 1;
__REG32                       : 2;
__REG32 VOLUME_UPDATE_PENDING : 1;
__REG32                       : 3;
} __hw_audioout_hpvol_bits;

/* Audio Power-Down Control Register */
typedef struct {        
__REG32 HEADPHONE           : 1;
__REG32                     : 3;
__REG32 CAPLESS             : 1;
__REG32                     : 3;
__REG32 ADC                 : 1;
__REG32                     : 3;
__REG32 DAC                 : 1;
__REG32                     : 3;
__REG32 RIGHT_ADC           : 1;
__REG32                     : 3;
__REG32 SELFBIAS            : 1;
__REG32                     : 3;
__REG32 SPEAKER             : 1;
__REG32                     : 7;
} __hw_audioout_pwrdn_bits;

/* AUDIOOUT Reference Control Register */
typedef struct {        
__REG32 DAC_ADJ             : 3;
__REG32                     : 1;
__REG32 VAG_VAL             : 4;
__REG32 ADC_REFVAL          : 4;
__REG32 ADJ_VAG             : 1;
__REG32 ADJ_ADC             : 1;
__REG32 VDDXTAL_TO_VDDD     : 1;
__REG32                     : 1;
__REG32 BIAS_CTRL           : 2;
__REG32 LW_REF              : 1;
__REG32 LOW_PWR             : 1;
__REG32 VBG_ADJ             : 3;
__REG32                     : 1;
__REG32 XTAL_BGR_BIAS       : 1;
__REG32 RAISE_REF           : 1;
__REG32 FASTSETTLING        : 1;
__REG32                     : 5;
} __hw_audioout_refctrl_bits;

/* Miscellaneous Audio Controls Register */
typedef struct {        
__REG32                     : 4;
__REG32 HP_CLASSAB          : 1;
__REG32 HP_HOLD_GND         : 1;
__REG32                     : 2;
__REG32 SHORT_LVLADJR       : 3;
__REG32                     : 1;
__REG32 SHORT_LVLADJL       : 3;
__REG32                     : 2;
__REG32 SHORTMODE_LR        : 2;
__REG32                     : 1;
__REG32 SHORTMODE_CM        : 2;
__REG32                     : 2;
__REG32 SHORT_LR_STS        : 1;
__REG32                     : 3;
__REG32 SHORT_CM_STS        : 1;
__REG32                     : 3;
} __hw_audioout_anactrl_bits;

/* Miscellaneous Test Audio Controls Register */
typedef struct {        
__REG32 DAC_DIS_RTZ         : 1;
__REG32 DAC_DOUBLE_I        : 1;
__REG32 DAC_CLASSA          : 1;
__REG32 ADCTODAC_LOOP       : 1;
__REG32                     : 8;
__REG32 VAG_DOUBLE_I        : 1;
__REG32 VAG_CLASSA          : 1;
__REG32                     : 6;
__REG32 HP_IALL_ADJ         : 2;
__REG32 HP_I1_ADJ           : 2;
__REG32 TM_HPCOMMON         : 1;
__REG32 TM_LOOP             : 1;
__REG32 TM_ADCIN_TOHP       : 1;
__REG32                     : 1;
__REG32 HP_ANTIPOP          : 3;
__REG32                     : 1;
} __hw_audioout_test_bits;

/* BIST Control and Status Register */
typedef struct {        
__REG32 START               : 1;
__REG32 DONE                : 1;
__REG32 PASS                : 1;
__REG32 FAIL                : 1;
__REG32                     :28;
} __hw_audioout_bistctrl_bits;

/* Hardware BIST Status 0 Register */
typedef struct {        
__REG32 DATA                :24;
__REG32                     : 8;
} __hw_audioout_biststat0_bits;

/* Hardware AUDIOUT BIST Status 1 Register */
typedef struct {        
__REG32 ADDR                : 8;
__REG32                     :16;
__REG32 STATE               : 5;
__REG32                     : 3;
} __hw_audioout_biststat1_bits;

/* Analog Clock Control Register */
typedef struct {        
__REG32 DACDIV              : 3;
__REG32                     : 1;
__REG32 INVERT_DACCLK       : 1;
__REG32                     :26;
__REG32 CLKGATE             : 1;
} __hw_audioout_anaclkctrl_bits;

/* AUDIOOUT Write Data Register */
typedef struct {        
__REG32 LOW                 :16;
__REG32 HIGH                :16;
} __hw_audioout_data_bits;

/* AUDIOOUT Speaker Control Register */
typedef struct {        
__REG32                     :12;
__REG32 NEGDRIVER           : 2;
__REG32 POSDRIVER           : 2;
__REG32                     : 4;
__REG32 IALL_ADJ            : 2;
__REG32 I1_ADJ              : 2;
__REG32 MUTE                : 1;
__REG32                     : 7;
} __hw_audioout_speakerctrl_bits;

/* AUDIOOUT Write Data Register */
typedef struct {        
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_audioout_version_bits;

/* SPDIF Control Register */
typedef struct {        
__REG32 RUN                 : 1;
__REG32 FIFO_ERROR_IRQ_EN   : 1;
__REG32 FIFO_OVERFLOW_IRQ   : 1;
__REG32 FIFO_UNDERFLOW_IRQ  : 1;
__REG32 WORD_LENGTH         : 1;
__REG32 WAIT_END_XFER       : 1;
__REG32                     :10;
__REG32 DMAWAIT_COUNT       : 5;
__REG32                     : 9;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_spdif_ctrl_bits;

/* SPDIF Status Register */
typedef struct {        
__REG32 END_XFER            : 1;
__REG32                     :30;
__REG32 PRESENT             : 1;
} __hw_spdif_stat_bits;

/* SPDIF Frame Control Register */
typedef struct {        
__REG32 PRO                 : 1;
__REG32 AUDIO               : 1;
__REG32 COPY                : 1;
__REG32 PRE                 : 1;
__REG32 CC                  : 7;
__REG32                     : 1;
__REG32 L                   : 1;
__REG32 V                   : 1;
__REG32 USER_DATA           : 1;
__REG32                     : 1;
__REG32 AUTO_MUTE           : 1;
__REG32 V_CONFIG            : 1;
__REG32                     :14;
} __hw_spdif_framectrl_bits;

/* SPDIF Sample Rate Register */
typedef struct {        
__REG32 RATE                :20;
__REG32                     : 8;
__REG32 BASEMULT            : 3;
__REG32                     : 1;
} __hw_spdif_srr_bits;

/* SPDIF Debug Register */
typedef struct {        
__REG32 FIFO_STATUS         : 1;
__REG32 DMA_PREQ            : 1;
__REG32                     :30;
} __hw_spdif_debug_bits;

/* SPDIF Write Data Register */
typedef struct {        
__REG32 LOW                 :16;
__REG32 HIGH                :16;
} __hw_spdif_data_bits;

/* SPDIF Version Register */
typedef struct {        
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_spdif_version_bits;

/* SAIF Control Register */
typedef struct {        
__REG32 RUN                 : 1;
__REG32 READ_MODE           : 1;
__REG32 SLAVE_MODE          : 1;
__REG32 BITCLK_48XFS_ENABLE : 1;
__REG32 WORD_LENGTH         : 4;
__REG32 BITCLK_EDGE         : 1;
__REG32 LRCLK_POLARITY      : 1;
__REG32 JUSTIFY             : 1;
__REG32 DELAY               : 1;
__REG32 BIT_ORDER           : 1;
__REG32                     : 1;
__REG32 CHANNEL_NUM_SELECT  : 2;
__REG32 DMAWAIT_COUNT       : 5;
__REG32                     : 3;
__REG32 FIFO_SERVICE_IRQ_EN : 1;
__REG32 FIFO_ERROR_IRQ_EN   : 1;
__REG32 BITCLK_BASE_RATE    : 1;
__REG32 BITCLK_MULT_RATE    : 3;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_saif_ctrl_bits;

/* SAIF Status Register */
typedef struct {        
__REG32 BUSY                : 1;
__REG32                     : 3;
__REG32 FIFO_SERVICE_IRQ    : 1;
__REG32 FIFO_OVERFLOW_IRQ   : 1;
__REG32 FIFO_UNDERFLOW_IRQ  : 1;
__REG32                     : 9;
__REG32 DMA_PREQ            : 1;
__REG32                     :14;
__REG32 PRESENT             : 1;
} __hw_saif_stat_bits;

/* SAIF Data Register */
typedef struct {        
__REG32 PCM_LEFT            :16;
__REG32 PCM_RIGHT           :16;
} __hw_saif_data_bits;

/* SAIF Version Register */
typedef struct {        
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_saif_version_bits;

/* Power Control Register */
typedef struct {        
__REG32 ENIRQ_VDD5V_GT_VDDIO  : 1;
__REG32 VDD5V_GT_VDDIO_IRQ    : 1;
__REG32 POL_VDD5V_GT_VDDIO    : 1;
__REG32 ENIRQ_VBUS_VALID      : 1;
__REG32 VBUSVALID_IRQ         : 1;
__REG32 POL_VBUSVALID         : 1;
__REG32 ENIRQ_VDDD_BO         : 1;
__REG32 VDDD_BO_IRQ           : 1;
__REG32 ENIRQ_VDDA_BO         : 1;
__REG32 VDDA_BO_IRQ           : 1;
__REG32 ENIRQ_VDDIO_BO        : 1;
__REG32 VDDIO_BO_IRQ          : 1;
__REG32 ENIRQBATT_BO          : 1;
__REG32 BATT_BO_IRQ           : 1;
__REG32 ENIRQ_DC_OK           : 1;
__REG32 DC_OK_IRQ             : 1;
__REG32 POL_DC_OK             : 1;
__REG32 ENIRQ_PSWITCH         : 1;
__REG32 POL_PSWITCH           : 1;
__REG32 PSWITCH_IRQ_SRC       : 1;
__REG32 PSWITCH_IRQ           : 1;
__REG32 ENIRQ_VDD5V_DROOP     : 1;
__REG32 VDD5V_DROOP_IRQ       : 1;
__REG32 ENIRQ_DCDC4P2_BO      : 1;
__REG32 DCDC4P2_BO_IRQ        : 1;
__REG32                       : 2;
__REG32 PSWITCH_MID_TRAN      : 1;
__REG32                       : 2;
__REG32 CLKGATE               : 1;
__REG32                       : 1;
} __hw_power_ctrl_bits;

/* DC-DC 5V Control Register */
typedef struct {        
__REG32 ENABLE_DCDC           : 1;
__REG32 PWRUP_VBUS_CMPS       : 1;
__REG32 ILIMIT_EQ_ZERO        : 1;
__REG32 VBUSVALID_TO_B        : 1;
__REG32 VBUSVALID_5VDETECT    : 1;
__REG32 DCDC_XFER             : 1;
__REG32 ENABLE_LINREG_ILIMIT  : 1;
__REG32 PWDN_5VBRNOUT         : 1;
__REG32 VBUSVALID_TRSH        : 3;
__REG32                       : 1;
__REG32 CHARGE_4P2_ILIMIT     : 6;
__REG32                       : 2;
__REG32 PWD_CHARGE_4P2        : 1;
__REG32                       : 3;
__REG32 HEADROOM_ADJ          : 3;
__REG32                       : 1;
__REG32 VBUSDROOP_TRSH        : 2;
__REG32                       : 2;
} __hw_power_5vctrl_bits;

/* DC-DC Minimum Power and Miscellaneous Control Register */
typedef struct {        
__REG32 DC_HALFCLK            : 1;
__REG32 EN_DC_PFM             : 1;
__REG32 DC_STOPCLK            : 1;
__REG32 PWD_XTAL24            : 1;
__REG32 LESSANA_I             : 1;
__REG32 HALF_FETS             : 1;
__REG32 DOUBLE_FETS           : 1;
__REG32 VBG_OFF               : 1;
__REG32 SELECT_OSC            : 1;
__REG32 ENABLE_OSC            : 1;
__REG32 PWD_ANA_CMPS          : 1;
__REG32 USE_VDDXTAL_VBG       : 1;
__REG32 PWD_BO                : 1;
__REG32 VDAC_DUMP_CTRL        : 1;
__REG32 LOWPWR_4P2            : 1;
__REG32                       :17;
} __hw_power_minpwr_bits;

/* Battery Charge Control Register */
typedef struct {        
__REG32 BATTCHRG_I            : 6;
__REG32                       : 2;
__REG32 STOP_ILIMIT           : 4;
__REG32                       : 4;
__REG32 PWD_BATTCHRG          : 1;
__REG32                       : 2;
__REG32 CHRG_STS_OFF          : 1;
__REG32 ENABLE_FAULT_DETECT   : 1;
__REG32 ENABLE_CHARGER_RES    : 1;
__REG32 ENABLE_LOAD           : 1;
__REG32                       : 1;
__REG32 ADJ_VOLT              : 3;
__REG32                       : 5;
} __hw_power_charge_bits;

/* VDDD Supply Targets and Brownouts Control Register */
typedef struct {        
__REG32 TRG                   : 5;
__REG32                       : 3;
__REG32 BO_OFFSET             : 3;
__REG32                       : 5;
__REG32 LINREG_OFFSET         : 2;
__REG32                       : 2;
__REG32 DISABLE_FET           : 1;
__REG32 ENABLE_LINREG         : 1;
__REG32 DISABLE_STEPPING      : 1;
__REG32 PWDN_BRNOUT           : 1;
__REG32                       : 4;
__REG32 ADJTN                 : 4;
} __hw_power_vdddctrl_bits;

/* VDDA Supply Targets and Brownouts Control Register */
typedef struct {        
__REG32 TRG                 : 5;
__REG32                     : 3;
__REG32 BO_OFFSET           : 3;
__REG32                     : 1;
__REG32 LINREG_OFFSET       : 2;
__REG32                     : 2;
__REG32 DISABLE_FET         : 1;
__REG32 ENABLE_LINREG       : 1;
__REG32 DISABLE_STEPPING    : 1;
__REG32 PWDN_BRNOUT         : 1;
__REG32                     :12;
} __hw_power_vddactrl_bits;

/* VDDIO Supply Targets and Brownouts Control Register */
typedef struct {        
__REG32 TRG                 : 5;
__REG32                     : 3;
__REG32 BO_OFFSET           : 3;
__REG32                     : 1;
__REG32 LINREG_OFFSET       : 2;
__REG32                     : 2;
__REG32 DISABLE_FET         : 1;
__REG32 DISABLE_STEPPING    : 1;
__REG32 PWDN_BRNOUT         : 1;
__REG32                     : 1;
__REG32 ADJTN               : 4;
__REG32                     : 8;
} __hw_power_vddioctrl_bits;

/* VDDMEM Supply Targets Control Register */
typedef struct {        
__REG32 TRG                 : 5;
__REG32                     : 3;
__REG32 ENABLE_LINREG       : 1;
__REG32 ENABLE_ILIMIT       : 1;
__REG32 PULLDOWN_ACTIVE     : 1;
__REG32                     :21;
} __hw_power_vddmemctrl_bits;

/* DC-DC Converter 4.2V Control Register */
typedef struct {        
__REG32 CMPTRIP             : 5;
__REG32                     : 3;
__REG32 BO                  : 5;
__REG32                     : 3;
__REG32 TRG                 : 3;
__REG32                     : 1;
__REG32 HYST_THRESH         : 1;
__REG32 HYST_DIR            : 1;
__REG32 ENABLE_DCDC         : 1;
__REG32 ENABLE_4P2          : 1;
__REG32 ISTEAL_THRESH       : 2;
__REG32                     : 2;
__REG32 DROPOUT_CTRL        : 4;
} __hw_power_dcdc4p2_bits;

/* DC-DC Miscellaneous Register */
typedef struct {        
__REG32 SEL_PLLCLK          : 1;
__REG32 TEST                : 1;
__REG32 DELAY_TIMING        : 1;
__REG32                     : 1;
__REG32 FREQSEL             : 3;
__REG32                     :25;
} __hw_power_misc_bits;

/* DC-DC Duty Cycle Limits Control Register */
typedef struct {        
__REG32 NEGLIMIT            : 7;
__REG32                     : 1;
__REG32 POSLIMIT_BUCK       : 7;
__REG32                     :17;
} __hw_power_dclimits_bits;

/* Converter Loop Behavior Control Register */
typedef struct {        
__REG32 DC_C                : 2;
__REG32                     : 2;
__REG32 DC_R                : 4;
__REG32 DC_FF               : 3;
__REG32                     : 1;
__REG32 EN_RCSCALE          : 2;
__REG32 RCSCALE_THRESH      : 1;
__REG32 DF_HYST_THRESH      : 1;
__REG32 CM_HYST_THRESH      : 1;
__REG32 EN_DF_HYST          : 1;
__REG32 EN_CM_HYST          : 1;
__REG32 HYST_SIGN           : 1;
__REG32 TOGGLE_DIF          : 1;
__REG32                     :11;
} __hw_power_loopctrl_bits;

/* Power Subsystem Status Register */
typedef struct {        
__REG32 SESSEND             : 1;
__REG32 VBUSVALID           : 1;
__REG32 BVALID              : 1;
__REG32 AVALID              : 1;
__REG32 VDD5V_DROOP         : 1;
__REG32 VDD5V_GT_VDDIO      : 1;
__REG32 VDDD_BO             : 1;
__REG32 VDDA_BO             : 1;
__REG32 VDDIO_BO            : 1;
__REG32                     : 1;
__REG32 DCDC_4P2_BO         : 1;
__REG32 CHRGSTS             : 1;
__REG32 VDD5V_FAULT         : 1;
__REG32 BATT_BO             : 1;
__REG32 SESSEND_STATUS      : 1;
__REG32 VBUSVALID_STATUS    : 1;
__REG32 BVALID_STATUS       : 1;
__REG32 AVALID_STATUS       : 1;
__REG32                     : 2;
__REG32 PSWITCH             : 2;
__REG32                     : 2;
__REG32 PWRUP_SOURCE        : 6;
__REG32                     : 2;
} __hw_power_sts_bits;

/* Transistor Speed Control and Status Register */
typedef struct {        
__REG32 CTRL                : 2;
__REG32                     :14;
__REG32 STATUS              : 8;
__REG32                     : 8;
} __hw_power_speed_bits;

/* Battery Level Monitor Register */
typedef struct {        
__REG32 BRWNOUT_LVL         : 5;
__REG32                     : 3;
__REG32 BRWNOUT_PWD         : 1;
__REG32 PWDN_BATTBRNOUT     : 1;
__REG32 EN_BATADJ           : 1;
__REG32                     : 5;
__REG32 BATT_VAL            :10;
__REG32                     : 6;
} __hw_power_battmonitor_bits;

/* Power Module Reset Register */
typedef struct {        
__REG32 PWD                 : 1;
__REG32 PWD_OFF             : 1;
__REG32                     :14;
__REG32 UNLOCK              :16;
} __hw_power_reset_bits;

/* Power Module Debug Register */
typedef struct {        
__REG32 SESSENDPIOLOCK      : 1;
__REG32 BVALIDPIOLOCK       : 1;
__REG32 AVALIDPIOLOCK       : 1;
__REG32 VBUSVALIDPIOLOCK    : 1;
__REG32                     :28;
} __hw_power_debug_bits;

/* Power Module Version Register */
typedef struct {        
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_power_version_bits;

/* LRADC Control Register 0 */
typedef struct {        
__REG32 SCHEDULE_CH0        : 1;
__REG32 SCHEDULE_CH1        : 1;
__REG32 SCHEDULE_CH2        : 1;
__REG32 SCHEDULE_CH3        : 1;
__REG32 SCHEDULE_CH4        : 1;
__REG32 SCHEDULE_CH5        : 1;
__REG32 SCHEDULE_CH6        : 1;
__REG32 SCHEDULE_CH7        : 1;
__REG32                     : 8;
__REG32 XPLUS_ENABLE        : 1;
__REG32 YPLUS_ENABLE        : 1;
__REG32 XMINUS_ENABLE       : 1;
__REG32 YMINUS_ENABLE       : 1;
__REG32 TOUCH_DETECT_ENABLE : 1;
__REG32 ONCHIP_GROUNDREF    : 1;
__REG32                     : 8;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_lradc_ctrl0_bits;

/* LRADC Control Register 1 */
typedef struct {        
__REG32 LRADC0_IRQ          : 1;
__REG32 LRADC1_IRQ          : 1;
__REG32 LRADC2_IRQ          : 1;
__REG32 LRADC3_IRQ          : 1;
__REG32 LRADC4_IRQ          : 1;
__REG32 LRADC5_IRQ          : 1;
__REG32 LRADC6_IRQ          : 1;
__REG32 LRADC7_IRQ          : 1;
__REG32 TOUCH_DETECT_IRQ    : 1;
__REG32                     : 7;
__REG32 LRADC0_IRQ_EN       : 1;
__REG32 LRADC1_IRQ_EN       : 1;
__REG32 LRADC2_IRQ_EN       : 1;
__REG32 LRADC3_IRQ_EN       : 1;
__REG32 LRADC4_IRQ_EN       : 1;
__REG32 LRADC5_IRQ_EN       : 1;
__REG32 LRADC6_IRQ_EN       : 1;
__REG32 LRADC7_IRQ_EN       : 1;
__REG32 TOUCH_DETECT_IRQ_EN : 1;
__REG32                     : 7;
} __hw_lradc_ctrl1_bits;

/* LRADC Control Register 2 */
typedef struct {        
__REG32 TEMP_ISRC0            : 4;
__REG32 TEMP_ISRC1            : 4;
__REG32 TEMP_SENSOR_IENABLE0  : 1;
__REG32 TEMP_SENSOR_IENABLE1  : 1;
__REG32                       : 2;
__REG32 EXT_EN0               : 1;
__REG32 EXT_EN1               : 1;
__REG32                       : 1;
__REG32 TEMPSENSE_PWD         : 1;
__REG32 BL_BRIGHTNESS         : 5;
__REG32 BL_MUX_SELECT         : 1;
__REG32 BL_ENABLE             : 1;
__REG32 BL_AMP_BYPASS         : 1;
__REG32 DIVIDE_BY_TWO_CH0     : 1;
__REG32 DIVIDE_BY_TWO_CH1     : 1;
__REG32 DIVIDE_BY_TWO_CH2     : 1;
__REG32 DIVIDE_BY_TWO_CH3     : 1;
__REG32 DIVIDE_BY_TWO_CH4     : 1;
__REG32 DIVIDE_BY_TWO_CH5     : 1;
__REG32 DIVIDE_BY_TWO_CH6     : 1;
__REG32 DIVIDE_BY_TWO_CH7     : 1;
} __hw_lradc_ctrl2_bits;

/* LRADC Control Register 3 */
typedef struct {        
__REG32 INVERT_CLOCK          : 1;
__REG32 DELAY_CLOCK           : 1;
__REG32                       : 2;
__REG32 HIGH_TIME             : 2;
__REG32                       : 2;
__REG32 CYCLE_TIME            : 2;
__REG32                       :12;
__REG32 FORCE_ANALOG_PWDN     : 1;
__REG32 FORCE_ANALOG_PWUP     : 1;
__REG32 DISCARD               : 2;
__REG32                       : 6;
} __hw_lradc_ctrl3_bits;

/* LRADC Status Register */
typedef struct {        
__REG32 TOUCH_DETECT_RAW      : 1;
__REG32                       :15;
__REG32 CHANNEL0_PRESENT      : 1;
__REG32 CHANNEL1_PRESENT      : 1;
__REG32 CHANNEL2_PRESENT      : 1;
__REG32 CHANNEL3_PRESENT      : 1;
__REG32 CHANNEL4_PRESENT      : 1;
__REG32 CHANNEL5_PRESENT      : 1;
__REG32 CHANNEL6_PRESENT      : 1;
__REG32 CHANNEL7_PRESENT      : 1;
__REG32 TOUCH_PANEL_PRESENT   : 1;
__REG32 TEMP0_PRESENT         : 1;
__REG32 TEMP1_PRESENT         : 1;
__REG32                       : 5;
} __hw_lradc_status_bits;

/* LRADC Result Register 0 - 6 */
typedef struct {        
__REG32 VALUE                 :18;
__REG32                       : 6;
__REG32 NUM_SAMPLES           : 5;
__REG32 ACCUMULATE            : 1;
__REG32                       : 1;
__REG32 TOGGLE                : 1;
} __hw_lradc_chx_bits;

/* LRADC Result Register 0 - 6 */
typedef struct {        
__REG32 VALUE                 :18;
__REG32                       : 6;
__REG32 NUM_SAMPLES           : 5;
__REG32 ACCUMULATE            : 1;
__REG32 TESTMODE_TOGGLE       : 1;
__REG32 TOGGLE                : 1;
} __hw_lradc_ch7_bits;

/* LRADC Scheduling Delay 0 - 3 */
typedef struct {        
__REG32 DELAY                 :11;
__REG32 LOOP_COUNT            : 5;
__REG32 TRIGGER_DELAYS        : 4;
__REG32 KICK                  : 1;
__REG32                       : 3;
__REG32 TRIGGER_LRADCS_CH0    : 1;
__REG32 TRIGGER_LRADCS_CH1    : 1;
__REG32 TRIGGER_LRADCS_CH2    : 1;
__REG32 TRIGGER_LRADCS_CH3    : 1;
__REG32 TRIGGER_LRADCS_CH4    : 1;
__REG32 TRIGGER_LRADCS_CH5    : 1;
__REG32 TRIGGER_LRADCS_CH6    : 1;
__REG32 TRIGGER_LRADCS_CH7    : 1;
} __hw_lradc_delayx_bits;

/* LRADC Debug Register 0 */
typedef struct {        
__REG32 STATE                 :12;
__REG32                       : 4;
__REG32 READONLY              :16;
} __hw_lradc_debug0_bits;

/* LRADC Debug Register 1 */
typedef struct {        
__REG32 TESTMODE              : 1;
__REG32 TESTMODE5             : 1;
__REG32 TESTMODE6             : 1;
__REG32                       : 5;
__REG32 TESTMODE_COUNT        : 5;
__REG32                       : 3;
__REG32 REQUEST               : 8;
__REG32                       : 8;
} __hw_lradc_debug1_bits;

/* LRADC Battery Conversion Register */
typedef struct {        
__REG32 SCALED_BATT_VOLTAGE   :10;
__REG32                       : 6;
__REG32 SCALE_FACTOR          : 2;
__REG32                       : 2;
__REG32 AUTOMATIC             : 1;
__REG32                       :11;
} __hw_lradc_conversion_bits;

/* LRADC Control Register 4 */
typedef struct {        
__REG32 LRADC0SELECT          : 4;
__REG32 LRADC1SELECT          : 4;
__REG32 LRADC2SELECT          : 4;
__REG32 LRADC3SELECT          : 4;
__REG32 LRADC4SELECT          : 4;
__REG32 LRADC5SELECT          : 4;
__REG32 LRADC6SELECT          : 4;
__REG32 LRADC7SELECT          : 4;
} __hw_lradc_ctrl4_bits;

/* LRADC Version Register */
typedef struct {        
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_lradc_version_bits;

/* PINCTRL Block Control Register */
typedef struct {        
__REG32 IRQOUT0             : 1;
__REG32 IRQOUT1             : 1;
__REG32 IRQOUT2             : 1;
__REG32                     :21;
__REG32 PRESENT0            : 1;
__REG32 PRESENT1            : 1;
__REG32 PRESENT2            : 1;
__REG32 PRESENT3            : 1;
__REG32                     : 2;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_pinctrl_ctrl_bits;

/* PINCTRL Pin Mux Select Register 0 */
typedef struct {        
__REG32 BANK0_PIN00         : 2;
__REG32 BANK0_PIN01         : 2;
__REG32 BANK0_PIN02         : 2;
__REG32 BANK0_PIN03         : 2;
__REG32 BANK0_PIN04         : 2;
__REG32 BANK0_PIN05         : 2;
__REG32 BANK0_PIN06         : 2;
__REG32 BANK0_PIN07         : 2;
__REG32 BANK0_PIN08         : 2;
__REG32 BANK0_PIN09         : 2;
__REG32 BANK0_PIN10         : 2;
__REG32 BANK0_PIN11         : 2;
__REG32 BANK0_PIN12         : 2;
__REG32 BANK0_PIN13         : 2;
__REG32 BANK0_PIN14         : 2;
__REG32 BANK0_PIN15         : 2;
} __hw_pinctrl_muxsel0_bits;

/* PINCTRL Pin Mux Select Register 1 */
typedef struct {        
__REG32 BANK0_PIN16         : 2;
__REG32 BANK0_PIN17         : 2;
__REG32 BANK0_PIN18         : 2;
__REG32 BANK0_PIN19         : 2;
__REG32 BANK0_PIN20         : 2;
__REG32 BANK0_PIN21         : 2;
__REG32 BANK0_PIN22         : 2;
__REG32 BANK0_PIN23         : 2;
__REG32 BANK0_PIN24         : 2;
__REG32 BANK0_PIN25         : 2;
__REG32 BANK0_PIN26         : 2;
__REG32 BANK0_PIN27         : 2;
__REG32 BANK0_PIN28         : 2;
__REG32 BANK0_PIN29         : 2;
__REG32 BANK0_PIN30         : 2;
__REG32 BANK0_PIN31         : 2;
} __hw_pinctrl_muxsel1_bits;

/* PINCTRL Pin Mux Select Register 2 */
typedef struct {        
__REG32 BANK1_PIN00         : 2;
__REG32 BANK1_PIN01         : 2;
__REG32 BANK1_PIN02         : 2;
__REG32 BANK1_PIN03         : 2;
__REG32 BANK1_PIN04         : 2;
__REG32 BANK1_PIN05         : 2;
__REG32 BANK1_PIN06         : 2;
__REG32 BANK1_PIN07         : 2;
__REG32 BANK1_PIN08         : 2;
__REG32 BANK1_PIN09         : 2;
__REG32 BANK1_PIN10         : 2;
__REG32 BANK1_PIN11         : 2;
__REG32 BANK1_PIN12         : 2;
__REG32 BANK1_PIN13         : 2;
__REG32 BANK1_PIN14         : 2;
__REG32 BANK1_PIN15         : 2;
} __hw_pinctrl_muxsel2_bits;

/* PINCTRL Pin Mux Select Register 1 */
typedef struct {        
__REG32 BANK1_PIN16         : 2;
__REG32 BANK1_PIN17         : 2;
__REG32 BANK1_PIN18         : 2;
__REG32 BANK1_PIN19         : 2;
__REG32 BANK1_PIN20         : 2;
__REG32 BANK1_PIN21         : 2;
__REG32 BANK1_PIN22         : 2;
__REG32 BANK1_PIN23         : 2;
__REG32 BANK1_PIN24         : 2;
__REG32 BANK1_PIN25         : 2;
__REG32 BANK1_PIN26         : 2;
__REG32 BANK1_PIN27         : 2;
__REG32 BANK1_PIN28         : 2;
__REG32 BANK1_PIN29         : 2;
__REG32 BANK1_PIN30         : 2;
__REG32                     : 2;
} __hw_pinctrl_muxsel3_bits;

/* PINCTRL Pin Mux Select Register 4 */
typedef struct {        
__REG32 BANK2_PIN00         : 2;
__REG32 BANK2_PIN01         : 2;
__REG32 BANK2_PIN02         : 2;
__REG32 BANK2_PIN03         : 2;
__REG32 BANK2_PIN04         : 2;
__REG32 BANK2_PIN05         : 2;
__REG32 BANK2_PIN06         : 2;
__REG32 BANK2_PIN07         : 2;
__REG32 BANK2_PIN08         : 2;
__REG32 BANK2_PIN09         : 2;
__REG32 BANK2_PIN10         : 2;
__REG32 BANK2_PIN11         : 2;
__REG32 BANK2_PIN12         : 2;
__REG32 BANK2_PIN13         : 2;
__REG32 BANK2_PIN14         : 2;
__REG32 BANK2_PIN15         : 2;
} __hw_pinctrl_muxsel4_bits;

/* PINCTRL Pin Mux Select Register 5 */
typedef struct {        
__REG32 BANK2_PIN16         : 2;
__REG32 BANK2_PIN17         : 2;
__REG32 BANK2_PIN18         : 2;
__REG32 BANK2_PIN19         : 2;
__REG32 BANK2_PIN20         : 2;
__REG32 BANK2_PIN21         : 2;
__REG32 BANK2_PIN22         : 2;
__REG32 BANK2_PIN23         : 2;
__REG32 BANK2_PIN24         : 2;
__REG32 BANK2_PIN25         : 2;
__REG32 BANK2_PIN26         : 2;
__REG32 BANK2_PIN27         : 2;
__REG32 BANK2_PIN28         : 2;
__REG32 BANK2_PIN29         : 2;
__REG32 BANK2_PIN30         : 2;
__REG32 BANK2_PIN31         : 2;
} __hw_pinctrl_muxsel5_bits;

/* PINCTRL Pin Mux Select Register 6 */
typedef struct {        
__REG32 BANK3_PIN00         : 2;
__REG32 BANK3_PIN01         : 2;
__REG32 BANK3_PIN02         : 2;
__REG32 BANK3_PIN03         : 2;
__REG32 BANK3_PIN04         : 2;
__REG32 BANK3_PIN05         : 2;
__REG32 BANK3_PIN06         : 2;
__REG32 BANK3_PIN07         : 2;
__REG32 BANK3_PIN08         : 2;
__REG32 BANK3_PIN09         : 2;
__REG32 BANK3_PIN10         : 2;
__REG32 BANK3_PIN11         : 2;
__REG32 BANK3_PIN12         : 2;
__REG32 BANK3_PIN13         : 2;
__REG32 BANK3_PIN14         : 2;
__REG32 BANK3_PIN15         : 2;
} __hw_pinctrl_muxsel6_bits;

/* PINCTRL Pin Mux Select Register 7 */
typedef struct {        
__REG32 BANK3_PIN16         : 2;
__REG32 BANK3_PIN17         : 2;
__REG32 BANK3_PIN18         : 2;
__REG32 BANK3_PIN19         : 2;
__REG32 BANK3_PIN20         : 2;
__REG32 BANK3_PIN21         : 2;
__REG32                     :20;
} __hw_pinctrl_muxsel7_bits;

/* PINCTRL Drive Strength and Voltage Register 0 */
typedef struct {        
__REG32 BANK0_PIN00_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN01_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN02_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN03_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN04_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN05_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN06_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN07_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive0_bits;

/* PINCTRL Drive Strength and Voltage Register 1 */
typedef struct {        
__REG32 BANK0_PIN08_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN09_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN10_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN11_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN12_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN13_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN14_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN15_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive1_bits;

/* PINCTRL Drive Strength and Voltage Register 2 */
typedef struct {        
__REG32 BANK0_PIN16_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN17_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN18_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN19_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN20_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN21_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN22_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN23_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive2_bits;

/* PINCTRL Drive Strength and Voltage Register 3 */
typedef struct {        
__REG32 BANK0_PIN24_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN25_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN26_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN27_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN28_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN29_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN30_MA      : 2;
__REG32                     : 2;
__REG32 BANK0_PIN31_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive3_bits;

/* PINCTRL Drive Strength and Voltage Register 4 */
typedef struct {        
__REG32 BANK1_PIN00_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN01_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN02_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN03_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN04_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN05_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN06_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN07_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive4_bits;

/* PINCTRL Drive Strength and Voltage Register 5 */
typedef struct {        
__REG32 BANK1_PIN08_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN09_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN10_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN11_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN12_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN13_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN14_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN15_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive5_bits;

/* PINCTRL Drive Strength and Voltage Register 6 */
typedef struct {        
__REG32 BANK1_PIN16_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN17_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN18_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN19_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN20_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN21_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN22_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN23_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive6_bits;

/* PINCTRL Drive Strength and Voltage Register 7 */
typedef struct {        
__REG32 BANK1_PIN24_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN25_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN26_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN27_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN28_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN29_MA      : 2;
__REG32                     : 2;
__REG32 BANK1_PIN30_MA      : 2;
__REG32                     : 6;
} __hw_pinctrl_drive7_bits;

/* PINCTRL Drive Strength and Voltage Register 8 */
typedef struct {        
__REG32 BANK2_PIN00_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN01_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN02_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN03_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN04_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN05_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN06_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN07_MA      : 2;
__REG32                     : 2;
} __hw_pinctrl_drive8_bits;

/* PINCTRL Drive Strength and Voltage Register 9 */
typedef struct {        
__REG32 BANK2_PIN08_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN09_MA      : 2;
__REG32 BANK2_PIN09_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN10_MA      : 2;
__REG32 BANK2_PIN10_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN11_MA      : 2;
__REG32 BANK2_PIN11_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN12_MA      : 2;
__REG32 BANK2_PIN12_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN13_MA      : 2;
__REG32 BANK2_PIN13_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN14_MA      : 2;
__REG32 BANK2_PIN14_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN15_MA      : 2;
__REG32 BANK2_PIN15_V       : 1;
__REG32                     : 1;
} __hw_pinctrl_drive9_bits;

/* PINCTRL Drive Strength and Voltage Register 10 */
typedef struct {        
__REG32 BANK2_PIN16_MA      : 2;
__REG32 BANK2_PIN16_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN17_MA      : 2;
__REG32 BANK2_PIN17_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN18_MA      : 2;
__REG32 BANK2_PIN18_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN19_MA      : 2;
__REG32 BANK2_PIN19_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN20_MA      : 2;
__REG32 BANK2_PIN20_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN21_MA      : 2;
__REG32 BANK2_PIN21_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN22_MA      : 2;
__REG32 BANK2_PIN22_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN23_MA      : 2;
__REG32 BANK2_PIN23_V       : 1;
__REG32                     : 1;
} __hw_pinctrl_drive10_bits;

/* PINCTRL Drive Strength and Voltage Register 11 */
typedef struct {        
__REG32 BANK2_PIN24_MA      : 2;
__REG32 BANK2_PIN24_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN25_MA      : 2;
__REG32 BANK2_PIN25_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN26_MA      : 2;
__REG32 BANK2_PIN26_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN27_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN28_MA      : 2;
__REG32                     : 2;
__REG32 BANK2_PIN29_MA      : 2;
__REG32 BANK2_PIN29_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN30_MA      : 2;
__REG32 BANK2_PIN30_V       : 1;
__REG32                     : 1;
__REG32 BANK2_PIN31_MA      : 2;
__REG32 BANK2_PIN31_V       : 1;
__REG32                     : 1;
} __hw_pinctrl_drive11_bits;

/* PINCTRL Drive Strength and Voltage Register 12 */
typedef struct {        
__REG32 BANK3_PIN00_MA      : 2;
__REG32 BANK3_PIN00_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN01_MA      : 2;
__REG32 BANK3_PIN01_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN02_MA      : 2;
__REG32 BANK3_PIN02_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN03_MA      : 2;
__REG32 BANK3_PIN03_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN04_MA      : 2;
__REG32 BANK3_PIN04_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN05_MA      : 2;
__REG32 BANK3_PIN05_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN06_MA      : 2;
__REG32 BANK3_PIN06_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN07_MA      : 2;
__REG32 BANK3_PIN07_V       : 1;
__REG32                     : 1;
} __hw_pinctrl_drive12_bits;

/* PINCTRL Drive Strength and Voltage Register 13 */
typedef struct {        
__REG32 BANK3_PIN08_MA      : 2;
__REG32 BANK3_PIN08_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN09_MA      : 2;
__REG32 BANK3_PIN09_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN10_MA      : 2;
__REG32 BANK3_PIN10_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN11_MA      : 2;
__REG32 BANK3_PIN11_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN12_MA      : 2;
__REG32 BANK3_PIN12_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN13_MA      : 2;
__REG32 BANK3_PIN13_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN14_MA      : 2;
__REG32 BANK3_PIN14_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN15_MA      : 2;
__REG32 BANK3_PIN15_V       : 1;
__REG32                     : 1;
} __hw_pinctrl_drive13_bits;

/* PINCTRL Drive Strength and Voltage Register 10 */
typedef struct {        
__REG32 BANK3_PIN16_MA      : 2;
__REG32 BANK3_PIN16_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN17_MA      : 2;
__REG32 BANK3_PIN17_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN18_MA      : 2;
__REG32 BANK3_PIN18_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN19_MA      : 2;
__REG32 BANK3_PIN19_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN20_MA      : 2;
__REG32 BANK3_PIN20_V       : 1;
__REG32                     : 1;
__REG32 BANK3_PIN21_MA      : 2;
__REG32 BANK3_PIN21_V       : 1;
__REG32                     : 9;
} __hw_pinctrl_drive14_bits;

/* PINCTRL Bank 0 Pull Up Resistor Enable Register */
typedef struct {        
__REG32 BANK0_PIN00         : 1;
__REG32 BANK0_PIN01         : 1;
__REG32 BANK0_PIN02         : 1;
__REG32 BANK0_PIN03         : 1;
__REG32 BANK0_PIN04         : 1;
__REG32 BANK0_PIN05         : 1;
__REG32 BANK0_PIN06         : 1;
__REG32 BANK0_PIN07         : 1;
__REG32 BANK0_PIN08         : 1;
__REG32 BANK0_PIN09         : 1;
__REG32 BANK0_PIN10         : 1;
__REG32 BANK0_PIN11         : 1;
__REG32                     : 3;
__REG32 BANK0_PIN15         : 1;
__REG32                     : 2;
__REG32 BANK0_PIN18         : 1;
__REG32 BANK0_PIN19         : 1;
__REG32 BANK0_PIN20         : 1;
__REG32 BANK0_PIN21         : 1;
__REG32 BANK0_PIN22         : 1;
__REG32                     : 3;
__REG32 BANK0_PIN26         : 1;
__REG32 BANK0_PIN27         : 1;
__REG32 BANK0_PIN28         : 1;
__REG32 BANK0_PIN29         : 1;
__REG32 BANK0_PIN30         : 1;
__REG32 BANK0_PIN31         : 1;
} __hw_pinctrl_pull0_bits;

/* PINCTRL Bank 1 Pull Up Resistor Enable Register */
typedef struct {        
__REG32                     :18;
__REG32 BANK1_PIN18         : 1;
__REG32                     : 3;
__REG32 BANK1_PIN22         : 1;
__REG32                     : 5;
__REG32 BANK1_PIN28         : 1;
__REG32                     : 3;
} __hw_pinctrl_pull1_bits;

/* PINCTRL Bank 2 Pull Up Resistor Enable Register */
typedef struct {        
__REG32 BANK2_PIN00         : 1;
__REG32 BANK2_PIN01         : 1;
__REG32 BANK2_PIN02         : 1;
__REG32 BANK2_PIN03         : 1;
__REG32 BANK2_PIN04         : 1;
__REG32 BANK2_PIN05         : 1;
__REG32                     : 2;
__REG32 BANK2_PIN08         : 1;
__REG32                     :18;
__REG32 BANK2_PIN27         : 1;
__REG32 BANK2_PIN28         : 1;
__REG32                     : 3;
} __hw_pinctrl_pull2_bits;

/* PINCTRL Bank 3 Pad Keeper Disable Register */
typedef struct {        
__REG32 BANK3_PIN00         : 1;
__REG32 BANK3_PIN01         : 1;
__REG32 BANK3_PIN02         : 1;
__REG32 BANK3_PIN03         : 1;
__REG32 BANK3_PIN04         : 1;
__REG32 BANK3_PIN05         : 1;
__REG32 BANK3_PIN06         : 1;
__REG32 BANK3_PIN07         : 1;
__REG32 BANK3_PIN08         : 1;
__REG32 BANK3_PIN09         : 1;
__REG32 BANK3_PIN10         : 1;
__REG32 BANK3_PIN11         : 1;
__REG32 BANK3_PIN12         : 1;
__REG32 BANK3_PIN13         : 1;
__REG32 BANK3_PIN14         : 1;
__REG32 BANK3_PIN15         : 1;
__REG32 BANK3_PIN16         : 1;
__REG32 BANK3_PIN17         : 1;
__REG32                     :14;
} __hw_pinctrl_pull3_bits;

/* PINCTRL Bank 0 Data Output Register */
typedef struct {        
__REG32 DOUT0               : 1;
__REG32 DOUT1               : 1;
__REG32 DOUT2               : 1;
__REG32 DOUT3               : 1;
__REG32 DOUT4               : 1;
__REG32 DOUT5               : 1;
__REG32 DOUT6               : 1;
__REG32 DOUT7               : 1;
__REG32 DOUT8               : 1;
__REG32 DOUT9               : 1;
__REG32 DOUT10              : 1;
__REG32 DOUT11              : 1;
__REG32 DOUT12              : 1;
__REG32 DOUT13              : 1;
__REG32 DOUT14              : 1;
__REG32 DOUT15              : 1;
__REG32 DOUT16              : 1;
__REG32 DOUT17              : 1;
__REG32 DOUT18              : 1;
__REG32 DOUT19              : 1;
__REG32 DOUT20              : 1;
__REG32 DOUT21              : 1;
__REG32 DOUT22              : 1;
__REG32 DOUT23              : 1;
__REG32 DOUT24              : 1;
__REG32 DOUT25              : 1;
__REG32 DOUT26              : 1;
__REG32 DOUT27              : 1;
__REG32 DOUT28              : 1;
__REG32 DOUT29              : 1;
__REG32 DOUT30              : 1;
__REG32 DOUT31              : 1;
} __hw_pinctrl_dout0_bits;

/* PINCTRL Bank 1 Data Output Register */
typedef struct {        
__REG32 DOUT0               : 1;
__REG32 DOUT1               : 1;
__REG32 DOUT2               : 1;
__REG32 DOUT3               : 1;
__REG32 DOUT4               : 1;
__REG32 DOUT5               : 1;
__REG32 DOUT6               : 1;
__REG32 DOUT7               : 1;
__REG32 DOUT8               : 1;
__REG32 DOUT9               : 1;
__REG32 DOUT10              : 1;
__REG32 DOUT11              : 1;
__REG32 DOUT12              : 1;
__REG32 DOUT13              : 1;
__REG32 DOUT14              : 1;
__REG32 DOUT15              : 1;
__REG32 DOUT16              : 1;
__REG32 DOUT17              : 1;
__REG32 DOUT18              : 1;
__REG32 DOUT19              : 1;
__REG32 DOUT20              : 1;
__REG32 DOUT21              : 1;
__REG32 DOUT22              : 1;
__REG32 DOUT23              : 1;
__REG32 DOUT24              : 1;
__REG32 DOUT25              : 1;
__REG32 DOUT26              : 1;
__REG32 DOUT27              : 1;
__REG32 DOUT28              : 1;
__REG32 DOUT29              : 1;
__REG32 DOUT30              : 1;
__REG32                     : 1;
} __hw_pinctrl_dout1_bits;

/* PINCTRL Bank 2 Data Output Register */
typedef struct {        
__REG32 DOUT0               : 1;
__REG32 DOUT1               : 1;
__REG32 DOUT2               : 1;
__REG32 DOUT3               : 1;
__REG32 DOUT4               : 1;
__REG32 DOUT5               : 1;
__REG32 DOUT6               : 1;
__REG32 DOUT7               : 1;
__REG32 DOUT8               : 1;
__REG32 DOUT9               : 1;
__REG32 DOUT10              : 1;
__REG32 DOUT11              : 1;
__REG32 DOUT12              : 1;
__REG32 DOUT13              : 1;
__REG32 DOUT14              : 1;
__REG32 DOUT15              : 1;
__REG32 DOUT16              : 1;
__REG32 DOUT17              : 1;
__REG32 DOUT18              : 1;
__REG32 DOUT19              : 1;
__REG32 DOUT20              : 1;
__REG32 DOUT21              : 1;
__REG32 DOUT22              : 1;
__REG32 DOUT23              : 1;
__REG32 DOUT24              : 1;
__REG32 DOUT25              : 1;
__REG32 DOUT26              : 1;
__REG32 DOUT27              : 1;
__REG32 DOUT28              : 1;
__REG32 DOUT29              : 1;
__REG32 DOUT30              : 1;
__REG32 DOUT31              : 1;
} __hw_pinctrl_dout2_bits;

/* PINCTRL Bank 0 Data Input Register */
typedef struct {        
__REG32 DIN0                : 1;
__REG32 DIN1                : 1;
__REG32 DIN2                : 1;
__REG32 DIN3                : 1;
__REG32 DIN4                : 1;
__REG32 DIN5                : 1;
__REG32 DIN6                : 1;
__REG32 DIN7                : 1;
__REG32 DIN8                : 1;
__REG32 DIN9                : 1;
__REG32 DIN10               : 1;
__REG32 DIN11               : 1;
__REG32 DIN12               : 1;
__REG32 DIN13               : 1;
__REG32 DIN14               : 1;
__REG32 DIN15               : 1;
__REG32 DIN16               : 1;
__REG32 DIN17               : 1;
__REG32 DIN18               : 1;
__REG32 DIN19               : 1;
__REG32 DIN20               : 1;
__REG32 DIN21               : 1;
__REG32 DIN22               : 1;
__REG32 DIN23               : 1;
__REG32 DIN24               : 1;
__REG32 DIN25               : 1;
__REG32 DIN26               : 1;
__REG32 DIN27               : 1;
__REG32 DIN28               : 1;
__REG32 DIN29               : 1;
__REG32 DIN30               : 1;
__REG32 DIN31               : 1;
} __hw_pinctrl_din0_bits;

/* PINCTRL Bank 1 Data Input Register */
typedef struct {        
__REG32 DIN0                : 1;
__REG32 DIN1                : 1;
__REG32 DIN2                : 1;
__REG32 DIN3                : 1;
__REG32 DIN4                : 1;
__REG32 DIN5                : 1;
__REG32 DIN6                : 1;
__REG32 DIN7                : 1;
__REG32 DIN8                : 1;
__REG32 DIN9                : 1;
__REG32 DIN10               : 1;
__REG32 DIN11               : 1;
__REG32 DIN12               : 1;
__REG32 DIN13               : 1;
__REG32 DIN14               : 1;
__REG32 DIN15               : 1;
__REG32 DIN16               : 1;
__REG32 DIN17               : 1;
__REG32 DIN18               : 1;
__REG32 DIN19               : 1;
__REG32 DIN20               : 1;
__REG32 DIN21               : 1;
__REG32 DIN22               : 1;
__REG32 DIN23               : 1;
__REG32 DIN24               : 1;
__REG32 DIN25               : 1;
__REG32 DIN26               : 1;
__REG32 DIN27               : 1;
__REG32 DIN28               : 1;
__REG32 DIN29               : 1;
__REG32 DIN30               : 1;
__REG32                     : 1;
} __hw_pinctrl_din1_bits;

/* PINCTRL Bank 2 Data Input Register */
typedef struct {        
__REG32 DIN0                : 1;
__REG32 DIN1                : 1;
__REG32 DIN2                : 1;
__REG32 DIN3                : 1;
__REG32 DIN4                : 1;
__REG32 DIN5                : 1;
__REG32 DIN6                : 1;
__REG32 DIN7                : 1;
__REG32 DIN8                : 1;
__REG32 DIN9                : 1;
__REG32 DIN10               : 1;
__REG32 DIN11               : 1;
__REG32 DIN12               : 1;
__REG32 DIN13               : 1;
__REG32 DIN14               : 1;
__REG32 DIN15               : 1;
__REG32 DIN16               : 1;
__REG32 DIN17               : 1;
__REG32 DIN18               : 1;
__REG32 DIN19               : 1;
__REG32 DIN20               : 1;
__REG32 DIN21               : 1;
__REG32 DIN22               : 1;
__REG32 DIN23               : 1;
__REG32 DIN24               : 1;
__REG32 DIN25               : 1;
__REG32 DIN26               : 1;
__REG32 DIN27               : 1;
__REG32 DIN28               : 1;
__REG32 DIN29               : 1;
__REG32 DIN30               : 1;
__REG32 DIN31               : 1;
} __hw_pinctrl_din2_bits;

/* PINCTRL Bank 0 Data Output Enable Register */
typedef struct {        
__REG32 DOE0                : 1;
__REG32 DOE1                : 1;
__REG32 DOE2                : 1;
__REG32 DOE3                : 1;
__REG32 DOE4                : 1;
__REG32 DOE5                : 1;
__REG32 DOE6                : 1;
__REG32 DOE7                : 1;
__REG32 DOE8                : 1;
__REG32 DOE9                : 1;
__REG32 DOE10               : 1;
__REG32 DOE11               : 1;
__REG32 DOE12               : 1;
__REG32 DOE13               : 1;
__REG32 DOE14               : 1;
__REG32 DOE15               : 1;
__REG32 DOE16               : 1;
__REG32 DOE17               : 1;
__REG32 DOE18               : 1;
__REG32 DOE19               : 1;
__REG32 DOE20               : 1;
__REG32 DOE21               : 1;
__REG32 DOE22               : 1;
__REG32 DOE23               : 1;
__REG32 DOE24               : 1;
__REG32 DOE25               : 1;
__REG32 DOE26               : 1;
__REG32 DOE27               : 1;
__REG32 DOE28               : 1;
__REG32 DOE29               : 1;
__REG32 DOE30               : 1;
__REG32 DOE31               : 1;
} __hw_pinctrl_doe0_bits;

/* PINCTRL Bank 1 Data Output Enable Register */
typedef struct {        
__REG32 DOE0                : 1;
__REG32 DOE1                : 1;
__REG32 DOE2                : 1;
__REG32 DOE3                : 1;
__REG32 DOE4                : 1;
__REG32 DOE5                : 1;
__REG32 DOE6                : 1;
__REG32 DOE7                : 1;
__REG32 DOE8                : 1;
__REG32 DOE9                : 1;
__REG32 DOE10               : 1;
__REG32 DOE11               : 1;
__REG32 DOE12               : 1;
__REG32 DOE13               : 1;
__REG32 DOE14               : 1;
__REG32 DOE15               : 1;
__REG32 DOE16               : 1;
__REG32 DOE17               : 1;
__REG32 DOE18               : 1;
__REG32 DOE19               : 1;
__REG32 DOE20               : 1;
__REG32 DOE21               : 1;
__REG32 DOE22               : 1;
__REG32 DOE23               : 1;
__REG32 DOE24               : 1;
__REG32 DOE25               : 1;
__REG32 DOE26               : 1;
__REG32 DOE27               : 1;
__REG32 DOE28               : 1;
__REG32 DOE29               : 1;
__REG32 DOE30               : 1;
__REG32                     : 1;
} __hw_pinctrl_doe1_bits;

/* PINCTRL Bank 2 Data Output Enable Register */
typedef struct {        
__REG32 DOE0                : 1;
__REG32 DOE1                : 1;
__REG32 DOE2                : 1;
__REG32 DOE3                : 1;
__REG32 DOE4                : 1;
__REG32 DOE5                : 1;
__REG32 DOE6                : 1;
__REG32 DOE7                : 1;
__REG32 DOE8                : 1;
__REG32 DOE9                : 1;
__REG32 DOE10               : 1;
__REG32 DOE11               : 1;
__REG32 DOE12               : 1;
__REG32 DOE13               : 1;
__REG32 DOE14               : 1;
__REG32 DOE15               : 1;
__REG32 DOE16               : 1;
__REG32 DOE17               : 1;
__REG32 DOE18               : 1;
__REG32 DOE19               : 1;
__REG32 DOE20               : 1;
__REG32 DOE21               : 1;
__REG32 DOE22               : 1;
__REG32 DOE23               : 1;
__REG32 DOE24               : 1;
__REG32 DOE25               : 1;
__REG32 DOE26               : 1;
__REG32 DOE27               : 1;
__REG32 DOE28               : 1;
__REG32 DOE29               : 1;
__REG32 DOE30               : 1;
__REG32 DOE31               : 1;
} __hw_pinctrl_doe2_bits;

/* PINCTRL Bank 0 Interrupt Select Register */
typedef struct {        
__REG32 PIN2IRQ0            : 1;
__REG32 PIN2IRQ1            : 1;
__REG32 PIN2IRQ2            : 1;
__REG32 PIN2IRQ3            : 1;
__REG32 PIN2IRQ4            : 1;
__REG32 PIN2IRQ5            : 1;
__REG32 PIN2IRQ6            : 1;
__REG32 PIN2IRQ7            : 1;
__REG32 PIN2IRQ8            : 1;
__REG32 PIN2IRQ9            : 1;
__REG32 PIN2IRQ10           : 1;
__REG32 PIN2IRQ11           : 1;
__REG32 PIN2IRQ12           : 1;
__REG32 PIN2IRQ13           : 1;
__REG32 PIN2IRQ14           : 1;
__REG32 PIN2IRQ15           : 1;
__REG32 PIN2IRQ16           : 1;
__REG32 PIN2IRQ17           : 1;
__REG32 PIN2IRQ18           : 1;
__REG32 PIN2IRQ19           : 1;
__REG32 PIN2IRQ20           : 1;
__REG32 PIN2IRQ21           : 1;
__REG32 PIN2IRQ22           : 1;
__REG32 PIN2IRQ23           : 1;
__REG32 PIN2IRQ24           : 1;
__REG32 PIN2IRQ25           : 1;
__REG32 PIN2IRQ26           : 1;
__REG32 PIN2IRQ27           : 1;
__REG32 PIN2IRQ28           : 1;
__REG32 PIN2IRQ29           : 1;
__REG32 PIN2IRQ30           : 1;
__REG32 PIN2IRQ31           : 1;
} __hw_pinctrl_pin2irq0_bits;

/* PINCTRL Bank 1 Interrupt Select Register */
typedef struct {        
__REG32 PIN2IRQ0            : 1;
__REG32 PIN2IRQ1            : 1;
__REG32 PIN2IRQ2            : 1;
__REG32 PIN2IRQ3            : 1;
__REG32 PIN2IRQ4            : 1;
__REG32 PIN2IRQ5            : 1;
__REG32 PIN2IRQ6            : 1;
__REG32 PIN2IRQ7            : 1;
__REG32 PIN2IRQ8            : 1;
__REG32 PIN2IRQ9            : 1;
__REG32 PIN2IRQ10           : 1;
__REG32 PIN2IRQ11           : 1;
__REG32 PIN2IRQ12           : 1;
__REG32 PIN2IRQ13           : 1;
__REG32 PIN2IRQ14           : 1;
__REG32 PIN2IRQ15           : 1;
__REG32 PIN2IRQ16           : 1;
__REG32 PIN2IRQ17           : 1;
__REG32 PIN2IRQ18           : 1;
__REG32 PIN2IRQ19           : 1;
__REG32 PIN2IRQ20           : 1;
__REG32 PIN2IRQ21           : 1;
__REG32 PIN2IRQ22           : 1;
__REG32 PIN2IRQ23           : 1;
__REG32 PIN2IRQ24           : 1;
__REG32 PIN2IRQ25           : 1;
__REG32 PIN2IRQ26           : 1;
__REG32 PIN2IRQ27           : 1;
__REG32 PIN2IRQ28           : 1;
__REG32 PIN2IRQ29           : 1;
__REG32 PIN2IRQ30           : 1;
__REG32                     : 1;
} __hw_pinctrl_pin2irq1_bits;

/* PINCTRL Bank 2 Interrupt Select Register */
typedef struct {        
__REG32 PIN2IRQ0            : 1;
__REG32 PIN2IRQ1            : 1;
__REG32 PIN2IRQ2            : 1;
__REG32 PIN2IRQ3            : 1;
__REG32 PIN2IRQ4            : 1;
__REG32 PIN2IRQ5            : 1;
__REG32 PIN2IRQ6            : 1;
__REG32 PIN2IRQ7            : 1;
__REG32 PIN2IRQ8            : 1;
__REG32 PIN2IRQ9            : 1;
__REG32 PIN2IRQ10           : 1;
__REG32 PIN2IRQ11           : 1;
__REG32 PIN2IRQ12           : 1;
__REG32 PIN2IRQ13           : 1;
__REG32 PIN2IRQ14           : 1;
__REG32 PIN2IRQ15           : 1;
__REG32 PIN2IRQ16           : 1;
__REG32 PIN2IRQ17           : 1;
__REG32 PIN2IRQ18           : 1;
__REG32 PIN2IRQ19           : 1;
__REG32 PIN2IRQ20           : 1;
__REG32 PIN2IRQ21           : 1;
__REG32 PIN2IRQ22           : 1;
__REG32 PIN2IRQ23           : 1;
__REG32 PIN2IRQ24           : 1;
__REG32 PIN2IRQ25           : 1;
__REG32 PIN2IRQ26           : 1;
__REG32 PIN2IRQ27           : 1;
__REG32 PIN2IRQ28           : 1;
__REG32 PIN2IRQ29           : 1;
__REG32 PIN2IRQ30           : 1;
__REG32 PIN2IRQ31           : 1;
} __hw_pinctrl_pin2irq2_bits;

/* PINCTRL Bank 0 Interrupt Mask Register */
typedef struct {        
__REG32 IRQEN0              : 1;
__REG32 IRQEN1              : 1;
__REG32 IRQEN2              : 1;
__REG32 IRQEN3              : 1;
__REG32 IRQEN4              : 1;
__REG32 IRQEN5              : 1;
__REG32 IRQEN6              : 1;
__REG32 IRQEN7              : 1;
__REG32 IRQEN8              : 1;
__REG32 IRQEN9              : 1;
__REG32 IRQEN10             : 1;
__REG32 IRQEN11             : 1;
__REG32 IRQEN12             : 1;
__REG32 IRQEN13             : 1;
__REG32 IRQEN14             : 1;
__REG32 IRQEN15             : 1;
__REG32 IRQEN16             : 1;
__REG32 IRQEN17             : 1;
__REG32 IRQEN18             : 1;
__REG32 IRQEN19             : 1;
__REG32 IRQEN20             : 1;
__REG32 IRQEN21             : 1;
__REG32 IRQEN22             : 1;
__REG32 IRQEN23             : 1;
__REG32 IRQEN24             : 1;
__REG32 IRQEN25             : 1;
__REG32 IRQEN26             : 1;
__REG32 IRQEN27             : 1;
__REG32 IRQEN28             : 1;
__REG32 IRQEN29             : 1;
__REG32 IRQEN30             : 1;
__REG32 IRQEN31             : 1;
} __hw_pinctrl_irqen0_bits;

/* PINCTRL Bank 1 Interrupt Mask Register */
typedef struct {        
__REG32 IRQEN0              : 1;
__REG32 IRQEN1              : 1;
__REG32 IRQEN2              : 1;
__REG32 IRQEN3              : 1;
__REG32 IRQEN4              : 1;
__REG32 IRQEN5              : 1;
__REG32 IRQEN6              : 1;
__REG32 IRQEN7              : 1;
__REG32 IRQEN8              : 1;
__REG32 IRQEN9              : 1;
__REG32 IRQEN10             : 1;
__REG32 IRQEN11             : 1;
__REG32 IRQEN12             : 1;
__REG32 IRQEN13             : 1;
__REG32 IRQEN14             : 1;
__REG32 IRQEN15             : 1;
__REG32 IRQEN16             : 1;
__REG32 IRQEN17             : 1;
__REG32 IRQEN18             : 1;
__REG32 IRQEN19             : 1;
__REG32 IRQEN20             : 1;
__REG32 IRQEN21             : 1;
__REG32 IRQEN22             : 1;
__REG32 IRQEN23             : 1;
__REG32 IRQEN24             : 1;
__REG32 IRQEN25             : 1;
__REG32 IRQEN26             : 1;
__REG32 IRQEN27             : 1;
__REG32 IRQEN28             : 1;
__REG32 IRQEN29             : 1;
__REG32 IRQEN30             : 1;
__REG32                     : 1;
} __hw_pinctrl_irqen1_bits;

/* PINCTRL Bank 2 Interrupt Mask Register */
typedef struct {        
__REG32 IRQEN0              : 1;
__REG32 IRQEN1              : 1;
__REG32 IRQEN2              : 1;
__REG32 IRQEN3              : 1;
__REG32 IRQEN4              : 1;
__REG32 IRQEN5              : 1;
__REG32 IRQEN6              : 1;
__REG32 IRQEN7              : 1;
__REG32 IRQEN8              : 1;
__REG32 IRQEN9              : 1;
__REG32 IRQEN10             : 1;
__REG32 IRQEN11             : 1;
__REG32 IRQEN12             : 1;
__REG32 IRQEN13             : 1;
__REG32 IRQEN14             : 1;
__REG32 IRQEN15             : 1;
__REG32 IRQEN16             : 1;
__REG32 IRQEN17             : 1;
__REG32 IRQEN18             : 1;
__REG32 IRQEN19             : 1;
__REG32 IRQEN20             : 1;
__REG32 IRQEN21             : 1;
__REG32 IRQEN22             : 1;
__REG32 IRQEN23             : 1;
__REG32 IRQEN24             : 1;
__REG32 IRQEN25             : 1;
__REG32 IRQEN26             : 1;
__REG32 IRQEN27             : 1;
__REG32 IRQEN28             : 1;
__REG32 IRQEN29             : 1;
__REG32 IRQEN30             : 1;
__REG32 IRQEN31             : 1;
} __hw_pinctrl_irqen2_bits;

/* PINCTRL Bank 0 Interrupt Level/Edge Register */
typedef struct {        
__REG32 IRQLEVEL0           : 1;
__REG32 IRQLEVEL1           : 1;
__REG32 IRQLEVEL2           : 1;
__REG32 IRQLEVEL3           : 1;
__REG32 IRQLEVEL4           : 1;
__REG32 IRQLEVEL5           : 1;
__REG32 IRQLEVEL6           : 1;
__REG32 IRQLEVEL7           : 1;
__REG32 IRQLEVEL8           : 1;
__REG32 IRQLEVEL9           : 1;
__REG32 IRQLEVEL10          : 1;
__REG32 IRQLEVEL11          : 1;
__REG32 IRQLEVEL12          : 1;
__REG32 IRQLEVEL13          : 1;
__REG32 IRQLEVEL14          : 1;
__REG32 IRQLEVEL15          : 1;
__REG32 IRQLEVEL16          : 1;
__REG32 IRQLEVEL17          : 1;
__REG32 IRQLEVEL18          : 1;
__REG32 IRQLEVEL19          : 1;
__REG32 IRQLEVEL20          : 1;
__REG32 IRQLEVEL21          : 1;
__REG32 IRQLEVEL22          : 1;
__REG32 IRQLEVEL23          : 1;
__REG32 IRQLEVEL24          : 1;
__REG32 IRQLEVEL25          : 1;
__REG32 IRQLEVEL26          : 1;
__REG32 IRQLEVEL27          : 1;
__REG32 IRQLEVEL28          : 1;
__REG32 IRQLEVEL29          : 1;
__REG32 IRQLEVEL30          : 1;
__REG32 IRQLEVEL31          : 1;
} __hw_pinctrl_irqlevel0_bits;

/* PINCTRL Bank 1 Interrupt Level/Edge Register */
typedef struct {        
__REG32 IRQLEVEL0           : 1;
__REG32 IRQLEVEL1           : 1;
__REG32 IRQLEVEL2           : 1;
__REG32 IRQLEVEL3           : 1;
__REG32 IRQLEVEL4           : 1;
__REG32 IRQLEVEL5           : 1;
__REG32 IRQLEVEL6           : 1;
__REG32 IRQLEVEL7           : 1;
__REG32 IRQLEVEL8           : 1;
__REG32 IRQLEVEL9           : 1;
__REG32 IRQLEVEL10          : 1;
__REG32 IRQLEVEL11          : 1;
__REG32 IRQLEVEL12          : 1;
__REG32 IRQLEVEL13          : 1;
__REG32 IRQLEVEL14          : 1;
__REG32 IRQLEVEL15          : 1;
__REG32 IRQLEVEL16          : 1;
__REG32 IRQLEVEL17          : 1;
__REG32 IRQLEVEL18          : 1;
__REG32 IRQLEVEL19          : 1;
__REG32 IRQLEVEL20          : 1;
__REG32 IRQLEVEL21          : 1;
__REG32 IRQLEVEL22          : 1;
__REG32 IRQLEVEL23          : 1;
__REG32 IRQLEVEL24          : 1;
__REG32 IRQLEVEL25          : 1;
__REG32 IRQLEVEL26          : 1;
__REG32 IRQLEVEL27          : 1;
__REG32 IRQLEVEL28          : 1;
__REG32 IRQLEVEL29          : 1;
__REG32 IRQLEVEL30          : 1;
__REG32                     : 1;
} __hw_pinctrl_irqlevel1_bits;

/* PINCTRL Bank 2 Interrupt Level/Edge Register */
typedef struct {        
__REG32 IRQLEVEL0           : 1;
__REG32 IRQLEVEL1           : 1;
__REG32 IRQLEVEL2           : 1;
__REG32 IRQLEVEL3           : 1;
__REG32 IRQLEVEL4           : 1;
__REG32 IRQLEVEL5           : 1;
__REG32 IRQLEVEL6           : 1;
__REG32 IRQLEVEL7           : 1;
__REG32 IRQLEVEL8           : 1;
__REG32 IRQLEVEL9           : 1;
__REG32 IRQLEVEL10          : 1;
__REG32 IRQLEVEL11          : 1;
__REG32 IRQLEVEL12          : 1;
__REG32 IRQLEVEL13          : 1;
__REG32 IRQLEVEL14          : 1;
__REG32 IRQLEVEL15          : 1;
__REG32 IRQLEVEL16          : 1;
__REG32 IRQLEVEL17          : 1;
__REG32 IRQLEVEL18          : 1;
__REG32 IRQLEVEL19          : 1;
__REG32 IRQLEVEL20          : 1;
__REG32 IRQLEVEL21          : 1;
__REG32 IRQLEVEL22          : 1;
__REG32 IRQLEVEL23          : 1;
__REG32 IRQLEVEL24          : 1;
__REG32 IRQLEVEL25          : 1;
__REG32 IRQLEVEL26          : 1;
__REG32 IRQLEVEL27          : 1;
__REG32 IRQLEVEL28          : 1;
__REG32 IRQLEVEL29          : 1;
__REG32 IRQLEVEL30          : 1;
__REG32 IRQLEVEL31          : 1;
} __hw_pinctrl_irqlevel2_bits;

/* PINCTRL Bank 0 Interrupt Polarity Register */
typedef struct {        
__REG32 IRQPOL0             : 1;
__REG32 IRQPOL1             : 1;
__REG32 IRQPOL2             : 1;
__REG32 IRQPOL3             : 1;
__REG32 IRQPOL4             : 1;
__REG32 IRQPOL5             : 1;
__REG32 IRQPOL6             : 1;
__REG32 IRQPOL7             : 1;
__REG32 IRQPOL8             : 1;
__REG32 IRQPOL9             : 1;
__REG32 IRQPOL10            : 1;
__REG32 IRQPOL11            : 1;
__REG32 IRQPOL12            : 1;
__REG32 IRQPOL13            : 1;
__REG32 IRQPOL14            : 1;
__REG32 IRQPOL15            : 1;
__REG32 IRQPOL16            : 1;
__REG32 IRQPOL17            : 1;
__REG32 IRQPOL18            : 1;
__REG32 IRQPOL19            : 1;
__REG32 IRQPOL20            : 1;
__REG32 IRQPOL21            : 1;
__REG32 IRQPOL22            : 1;
__REG32 IRQPOL23            : 1;
__REG32 IRQPOL24            : 1;
__REG32 IRQPOL25            : 1;
__REG32 IRQPOL26            : 1;
__REG32 IRQPOL27            : 1;
__REG32 IRQPOL28            : 1;
__REG32 IRQPOL29            : 1;
__REG32 IRQPOL30            : 1;
__REG32 IRQPOL31            : 1;
} __hw_pinctrl_irqpol0_bits;

/* PINCTRL Bank 1 Interrupt Polarity Register */
typedef struct {        
__REG32 IRQPOL0             : 1;
__REG32 IRQPOL1             : 1;
__REG32 IRQPOL2             : 1;
__REG32 IRQPOL3             : 1;
__REG32 IRQPOL4             : 1;
__REG32 IRQPOL5             : 1;
__REG32 IRQPOL6             : 1;
__REG32 IRQPOL7             : 1;
__REG32 IRQPOL8             : 1;
__REG32 IRQPOL9             : 1;
__REG32 IRQPOL10            : 1;
__REG32 IRQPOL11            : 1;
__REG32 IRQPOL12            : 1;
__REG32 IRQPOL13            : 1;
__REG32 IRQPOL14            : 1;
__REG32 IRQPOL15            : 1;
__REG32 IRQPOL16            : 1;
__REG32 IRQPOL17            : 1;
__REG32 IRQPOL18            : 1;
__REG32 IRQPOL19            : 1;
__REG32 IRQPOL20            : 1;
__REG32 IRQPOL21            : 1;
__REG32 IRQPOL22            : 1;
__REG32 IRQPOL23            : 1;
__REG32 IRQPOL24            : 1;
__REG32 IRQPOL25            : 1;
__REG32 IRQPOL26            : 1;
__REG32 IRQPOL27            : 1;
__REG32 IRQPOL28            : 1;
__REG32 IRQPOL29            : 1;
__REG32 IRQPOL30            : 1;
__REG32                     : 1;
} __hw_pinctrl_irqpol1_bits;

/* PINCTRL Bank 2 Interrupt Polarity Register */
typedef struct {        
__REG32 IRQPOL0             : 1;
__REG32 IRQPOL1             : 1;
__REG32 IRQPOL2             : 1;
__REG32 IRQPOL3             : 1;
__REG32 IRQPOL4             : 1;
__REG32 IRQPOL5             : 1;
__REG32 IRQPOL6             : 1;
__REG32 IRQPOL7             : 1;
__REG32 IRQPOL8             : 1;
__REG32 IRQPOL9             : 1;
__REG32 IRQPOL10            : 1;
__REG32 IRQPOL11            : 1;
__REG32 IRQPOL12            : 1;
__REG32 IRQPOL13            : 1;
__REG32 IRQPOL14            : 1;
__REG32 IRQPOL15            : 1;
__REG32 IRQPOL16            : 1;
__REG32 IRQPOL17            : 1;
__REG32 IRQPOL18            : 1;
__REG32 IRQPOL19            : 1;
__REG32 IRQPOL20            : 1;
__REG32 IRQPOL21            : 1;
__REG32 IRQPOL22            : 1;
__REG32 IRQPOL23            : 1;
__REG32 IRQPOL24            : 1;
__REG32 IRQPOL25            : 1;
__REG32 IRQPOL26            : 1;
__REG32 IRQPOL27            : 1;
__REG32 IRQPOL28            : 1;
__REG32 IRQPOL29            : 1;
__REG32 IRQPOL30            : 1;
__REG32 IRQPOL31            : 1;
} __hw_pinctrl_irqpol2_bits;

/* PINCTRL Bank 0 Interrupt Status Register */
typedef struct {        
__REG32 IRQSTAT0            : 1;
__REG32 IRQSTAT1            : 1;
__REG32 IRQSTAT2            : 1;
__REG32 IRQSTAT3            : 1;
__REG32 IRQSTAT4            : 1;
__REG32 IRQSTAT5            : 1;
__REG32 IRQSTAT6            : 1;
__REG32 IRQSTAT7            : 1;
__REG32 IRQSTAT8            : 1;
__REG32 IRQSTAT9            : 1;
__REG32 IRQSTAT10           : 1;
__REG32 IRQSTAT11           : 1;
__REG32 IRQSTAT12           : 1;
__REG32 IRQSTAT13           : 1;
__REG32 IRQSTAT14           : 1;
__REG32 IRQSTAT15           : 1;
__REG32 IRQSTAT16           : 1;
__REG32 IRQSTAT17           : 1;
__REG32 IRQSTAT18           : 1;
__REG32 IRQSTAT19           : 1;
__REG32 IRQSTAT20           : 1;
__REG32 IRQSTAT21           : 1;
__REG32 IRQSTAT22           : 1;
__REG32 IRQSTAT23           : 1;
__REG32 IRQSTAT24           : 1;
__REG32 IRQSTAT25           : 1;
__REG32 IRQSTAT26           : 1;
__REG32 IRQSTAT27           : 1;
__REG32 IRQSTAT28           : 1;
__REG32 IRQSTAT29           : 1;
__REG32 IRQSTAT30           : 1;
__REG32 IRQSTAT31           : 1;
} __hw_pinctrl_irqstat0_bits;

/* PINCTRL Bank 1 Interrupt Status Register */
typedef struct {        
__REG32 IRQSTAT0            : 1;
__REG32 IRQSTAT1            : 1;
__REG32 IRQSTAT2            : 1;
__REG32 IRQSTAT3            : 1;
__REG32 IRQSTAT4            : 1;
__REG32 IRQSTAT5            : 1;
__REG32 IRQSTAT6            : 1;
__REG32 IRQSTAT7            : 1;
__REG32 IRQSTAT8            : 1;
__REG32 IRQSTAT9            : 1;
__REG32 IRQSTAT10           : 1;
__REG32 IRQSTAT11           : 1;
__REG32 IRQSTAT12           : 1;
__REG32 IRQSTAT13           : 1;
__REG32 IRQSTAT14           : 1;
__REG32 IRQSTAT15           : 1;
__REG32 IRQSTAT16           : 1;
__REG32 IRQSTAT17           : 1;
__REG32 IRQSTAT18           : 1;
__REG32 IRQSTAT19           : 1;
__REG32 IRQSTAT20           : 1;
__REG32 IRQSTAT21           : 1;
__REG32 IRQSTAT22           : 1;
__REG32 IRQSTAT23           : 1;
__REG32 IRQSTAT24           : 1;
__REG32 IRQSTAT25           : 1;
__REG32 IRQSTAT26           : 1;
__REG32 IRQSTAT27           : 1;
__REG32 IRQSTAT28           : 1;
__REG32 IRQSTAT29           : 1;
__REG32 IRQSTAT30           : 1;
__REG32                     : 1;
} __hw_pinctrl_irqstat1_bits;

/* PINCTRL Bank 2 Interrupt Status Register */
typedef struct {        
__REG32 IRQSTAT0            : 1;
__REG32 IRQSTAT1            : 1;
__REG32 IRQSTAT2            : 1;
__REG32 IRQSTAT3            : 1;
__REG32 IRQSTAT4            : 1;
__REG32 IRQSTAT5            : 1;
__REG32 IRQSTAT6            : 1;
__REG32 IRQSTAT7            : 1;
__REG32 IRQSTAT8            : 1;
__REG32 IRQSTAT9            : 1;
__REG32 IRQSTAT10           : 1;
__REG32 IRQSTAT11           : 1;
__REG32 IRQSTAT12           : 1;
__REG32 IRQSTAT13           : 1;
__REG32 IRQSTAT14           : 1;
__REG32 IRQSTAT15           : 1;
__REG32 IRQSTAT16           : 1;
__REG32 IRQSTAT17           : 1;
__REG32 IRQSTAT18           : 1;
__REG32 IRQSTAT19           : 1;
__REG32 IRQSTAT20           : 1;
__REG32 IRQSTAT21           : 1;
__REG32 IRQSTAT22           : 1;
__REG32 IRQSTAT23           : 1;
__REG32 IRQSTAT24           : 1;
__REG32 IRQSTAT25           : 1;
__REG32 IRQSTAT26           : 1;
__REG32 IRQSTAT27           : 1;
__REG32 IRQSTAT28           : 1;
__REG32 IRQSTAT29           : 1;
__REG32 IRQSTAT30           : 1;
__REG32 IRQSTAT31           : 1;
} __hw_pinctrl_irqstat2_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  CLKCTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_CLKCTRL_PLLCTRL0,               0x80040000,__READ_WRITE ,__hw_clkctrl_pllctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLLCTRL0_SET,           0x80040004,__WRITE      ,__hw_clkctrl_pllctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLLCTRL0_CLR,           0x80040008,__WRITE      ,__hw_clkctrl_pllctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLLCTRL0_TOG,           0x8004000C,__WRITE      ,__hw_clkctrl_pllctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLLCTRL1,               0x80040010,__READ_WRITE ,__hw_clkctrl_pllctrl1_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU,                    0x80040020,__READ_WRITE ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_SET,                0x80040024,__WRITE      ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_CLR,                0x80040028,__WRITE      ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_TOG,                0x8004002C,__WRITE      ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS,                   0x80040030,__READ_WRITE ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_SET,               0x80040034,__WRITE      ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_CLR,               0x80040038,__WRITE      ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_TOG,               0x8004003C,__WRITE      ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_XBUS,                   0x80040040,__READ_WRITE ,__hw_clkctrl_xbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL,                   0x80040050,__READ_WRITE ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_SET,               0x80040054,__WRITE      ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_CLR,               0x80040058,__WRITE      ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_TOG,               0x8004005C,__WRITE      ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_PIX,                    0x80040060,__READ_WRITE ,__hw_clkctrl_pix_bits);
__IO_REG32_BIT(HW_CLKCTRL_SSP,                    0x80040070,__READ_WRITE ,__hw_clkctrl_ssp_bits);
__IO_REG32_BIT(HW_CLKCTRL_GPMI,                   0x80040080,__READ_WRITE ,__hw_clkctrl_gpmi_bits);
__IO_REG32_BIT(HW_CLKCTRL_SPDIF,                  0x80040090,__READ_WRITE ,__hw_clkctrl_spdif_bits);
__IO_REG32_BIT(HW_CLKCTRL_EMI,                    0x800400A0,__READ_WRITE ,__hw_clkctrl_emi_bits);
__IO_REG32_BIT(HW_CLKCTRL_SAIF,                   0x800400C0,__READ_WRITE ,__hw_clkctrl_saif_bits);
__IO_REG32_BIT(HW_CLKCTRL_TV,                     0x800400D0,__READ_WRITE ,__hw_clkctrl_tv_bits);
__IO_REG32_BIT(HW_CLKCTRL_ETM,                    0x800400E0,__READ_WRITE ,__hw_clkctrl_etm_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC,                   0x800400F0,__READ_WRITE ,__hw_clkctrl_frac_bits);
#define HW_CLKCTRL_CPUFRAC      HW_CLKCTRL_FRAC_bit.__cpufrac
#define HW_CLKCTRL_CPUFRAC_bit  HW_CLKCTRL_FRAC_bit.__cpufrac_bit
#define HW_CLKCTRL_EMIFRAC      HW_CLKCTRL_FRAC_bit.__emifrac
#define HW_CLKCTRL_EMIFRAC_bit  HW_CLKCTRL_FRAC_bit.__emifrac_bit
#define HW_CLKCTRL_PIXFRAC      HW_CLKCTRL_FRAC_bit.__pixfrac
#define HW_CLKCTRL_PIXFRAC_bit  HW_CLKCTRL_FRAC_bit.__pixfrac_bit
#define HW_CLKCTRL_IOFRAC       HW_CLKCTRL_FRAC_bit.__iofrac
#define HW_CLKCTRL_IOFRAC_bit   HW_CLKCTRL_FRAC_bit.__iofrac_bit
__IO_REG32_BIT(HW_CLKCTRL_FRAC_SET,               0x800400F4,__WRITE      ,__hw_clkctrl_frac_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC_CLR,               0x800400F8,__WRITE      ,__hw_clkctrl_frac_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC_TOG,               0x800400FC,__WRITE      ,__hw_clkctrl_frac_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1,                  0x80040100,__READ_WRITE ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_SET,              0x80040104,__WRITE      ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_CLR,              0x80040108,__WRITE      ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_TOG,              0x8004010C,__WRITE      ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ,                 0x80040110,__READ_WRITE ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_SET,             0x80040114,__WRITE      ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_CLR,             0x80040118,__WRITE      ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_TOG,             0x8004011C,__WRITE      ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_RESET,                  0x80040120,__READ_WRITE ,__hw_clkctrl_reset_bits);
__IO_REG32_BIT(HW_CLKCTRL_STATUS,                 0x80040130,__READ       ,__hw_clkctrl_status_bits);
__IO_REG32_BIT(HW_CLKCTRL_VERSION,                0x80040140,__READ       ,__hw_clkctrl_version_bits);

/***************************************************************************
 **
 **  ICOLL
 **
 ***************************************************************************/
__IO_REG32(    HW_ICOLL_VECTOR,                   0x80000000,__READ_WRITE );
__IO_REG32(    HW_ICOLL_VECTOR_SET,               0x80000004,__WRITE      );
__IO_REG32(    HW_ICOLL_VECTOR_CLR,               0x80000008,__WRITE      );
__IO_REG32(    HW_ICOLL_VECTOR_TOG,               0x8000000C,__WRITE      );
__IO_REG32_BIT(HW_ICOLL_LEVELACK,                 0x80000010,__READ_WRITE ,__hw_icoll_levelack_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL,                     0x80000020,__READ_WRITE ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_SET,                 0x80000024,__WRITE      ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_CLR,                 0x80000028,__WRITE      ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_TOG,                 0x8000002C,__WRITE      ,__hw_icoll_ctrl_bits);
__IO_REG32(    HW_ICOLL_VBASE,                    0x80000040,__READ_WRITE );
__IO_REG32(    HW_ICOLL_VBASE_SET,                0x80000044,__WRITE      );
__IO_REG32(    HW_ICOLL_VBASE_CLR,                0x80000048,__WRITE      );
__IO_REG32(    HW_ICOLL_VBASE_TOG,                0x8000004C,__WRITE      );
__IO_REG32_BIT(HW_ICOLL_STAT,                     0x80000070,__READ       ,__hw_icoll_stat_bits);
__IO_REG32_BIT(HW_ICOLL_RAW0,                     0x800000A0,__READ       ,__hw_icoll_raw0_bits);
__IO_REG32_BIT(HW_ICOLL_RAW1,                     0x800000B0,__READ       ,__hw_icoll_raw1_bits);
__IO_REG32_BIT(HW_ICOLL_RAW2,                     0x800000C0,__READ       ,__hw_icoll_raw2_bits);
__IO_REG32_BIT(HW_ICOLL_RAW3,                     0x800000D0,__READ       ,__hw_icoll_raw3_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0,               0x80000120,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_SET,           0x80000124,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_CLR,           0x80000128,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_TOG,           0x8000012C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1,               0x80000130,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_SET,           0x80000134,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_CLR,           0x80000138,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_TOG,           0x8000013C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2,               0x80000140,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_SET,           0x80000144,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_CLR,           0x80000148,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_TOG,           0x8000014C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3,               0x80000150,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_SET,           0x80000154,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_CLR,           0x80000158,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_TOG,           0x8000015C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4,               0x80000160,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_SET,           0x80000164,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_CLR,           0x80000168,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_TOG,           0x8000016C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5,               0x80000170,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_SET,           0x80000174,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_CLR,           0x80000178,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_TOG,           0x8000017C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6,               0x80000180,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_SET,           0x80000184,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_CLR,           0x80000188,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_TOG,           0x8000018C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7,               0x80000190,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_SET,           0x80000194,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_CLR,           0x80000198,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_TOG,           0x8000019C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8,               0x800001A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_SET,           0x800001A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_CLR,           0x800001A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_TOG,           0x800001AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9,               0x800001B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_SET,           0x800001B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_CLR,           0x800001B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_TOG,           0x800001BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT10,              0x800001C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT10_SET,          0x800001C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT10_CLR,          0x800001C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT10_TOG,          0x800001CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT11,              0x800001D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT11_SET,          0x800001D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT11_CLR,          0x800001D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT11_TOG,          0x800001DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT12,              0x800001E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT12_SET,          0x800001E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT12_CLR,          0x800001E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT12_TOG,          0x800001EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT13,              0x800001F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT13_SET,          0x800001F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT13_CLR,          0x800001F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT13_TOG,          0x800001FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT14,              0x80000200,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT14_SET,          0x80000204,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT14_CLR,          0x80000208,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT14_TOG,          0x8000020C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT15,              0x80000210,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT15_SET,          0x80000214,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT15_CLR,          0x80000218,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT15_TOG,          0x8000021C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT16,              0x80000220,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT16_SET,          0x80000224,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT16_CLR,          0x80000228,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT16_TOG,          0x8000022C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT17,              0x80000230,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT17_SET,          0x80000234,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT17_CLR,          0x80000238,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT17_TOG,          0x8000023C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT18,              0x80000240,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT18_SET,          0x80000244,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT18_CLR,          0x80000248,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT18_TOG,          0x8000024C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT19,              0x80000250,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT19_SET,          0x80000254,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT19_CLR,          0x80000258,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT19_TOG,          0x8000025C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT20,              0x80000260,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT20_SET,          0x80000264,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT20_CLR,          0x80000268,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT20_TOG,          0x8000026C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT21,              0x80000270,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT21_SET,          0x80000274,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT21_CLR,          0x80000278,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT21_TOG,          0x8000027C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT22,              0x80000280,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT22_SET,          0x80000284,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT22_CLR,          0x80000288,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT22_TOG,          0x8000028C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT23,              0x80000290,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT23_SET,          0x80000294,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT23_CLR,          0x80000298,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT23_TOG,          0x8000029C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT24,              0x800002A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT24_SET,          0x800002A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT24_CLR,          0x800002A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT24_TOG,          0x800002AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT25,              0x800002B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT25_SET,          0x800002B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT25_CLR,          0x800002B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT25_TOG,          0x800002BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT26,              0x800002C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT26_SET,          0x800002C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT26_CLR,          0x800002C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT26_TOG,          0x800002CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT27,              0x800002D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT27_SET,          0x800002D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT27_CLR,          0x800002D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT27_TOG,          0x800002DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT28,              0x800002E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT28_SET,          0x800002E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT28_CLR,          0x800002E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT28_TOG,          0x800002EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT29,              0x800002F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT29_SET,          0x800002F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT29_CLR,          0x800002F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT29_TOG,          0x800002FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT30,              0x80000300,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT30_SET,          0x80000304,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT30_CLR,          0x80000308,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT30_TOG,          0x8000030C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT31,              0x80000310,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT31_SET,          0x80000314,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT31_CLR,          0x80000318,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT31_TOG,          0x8000031C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT32,              0x80000320,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT32_SET,          0x80000324,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT32_CLR,          0x80000328,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT32_TOG,          0x8000032C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT33,              0x80000330,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT33_SET,          0x80000334,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT33_CLR,          0x80000338,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT33_TOG,          0x8000033C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT34,              0x80000340,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT34_SET,          0x80000344,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT34_CLR,          0x80000348,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT34_TOG,          0x8000034C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT35,              0x80000350,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT35_SET,          0x80000354,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT35_CLR,          0x80000358,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT35_TOG,          0x8000035C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT36,              0x80000360,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT36_SET,          0x80000364,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT36_CLR,          0x80000368,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT36_TOG,          0x8000036C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT37,              0x80000370,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT37_SET,          0x80000374,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT37_CLR,          0x80000378,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT37_TOG,          0x8000037C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT38,              0x80000380,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT38_SET,          0x80000384,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT38_CLR,          0x80000388,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT38_TOG,          0x8000038C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT39,              0x80000390,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT39_SET,          0x80000394,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT39_CLR,          0x80000398,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT39_TOG,          0x8000039C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT40,              0x800003A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT40_SET,          0x800003A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT40_CLR,          0x800003A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT40_TOG,          0x800003AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT41,              0x800003B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT41_SET,          0x800003B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT41_CLR,          0x800003B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT41_TOG,          0x800003BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT42,              0x800003C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT42_SET,          0x800003C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT42_CLR,          0x800003C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT42_TOG,          0x800003CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT43,              0x800003D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT43_SET,          0x800003D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT43_CLR,          0x800003D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT43_TOG,          0x800003DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT44,              0x800003E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT44_SET,          0x800003E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT44_CLR,          0x800003E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT44_TOG,          0x800003EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT45,              0x800003F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT45_SET,          0x800003F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT45_CLR,          0x800003F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT45_TOG,          0x800003FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT46,              0x80000400,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT46_SET,          0x80000404,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT46_CLR,          0x80000408,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT46_TOG,          0x8000040C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT47,              0x80000410,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT47_SET,          0x80000414,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT47_CLR,          0x80000418,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT47_TOG,          0x8000041C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT48,              0x80000420,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT48_SET,          0x80000424,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT48_CLR,          0x80000428,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT48_TOG,          0x8000042C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT49,              0x80000430,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT49_SET,          0x80000434,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT49_CLR,          0x80000438,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT49_TOG,          0x8000043C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT50,              0x80000440,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT50_SET,          0x80000444,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT50_CLR,          0x80000448,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT50_TOG,          0x8000044C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT51,              0x80000450,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT51_SET,          0x80000454,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT51_CLR,          0x80000458,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT51_TOG,          0x8000045C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT52,              0x80000460,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT52_SET,          0x80000464,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT52_CLR,          0x80000468,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT52_TOG,          0x8000046C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT53,              0x80000470,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT53_SET,          0x80000474,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT53_CLR,          0x80000478,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT53_TOG,          0x8000047C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT54,              0x80000480,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT54_SET,          0x80000484,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT54_CLR,          0x80000488,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT54_TOG,          0x8000048C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT55,              0x80000490,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT55_SET,          0x80000494,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT55_CLR,          0x80000498,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT55_TOG,          0x8000049C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT56,              0x800004A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT56_SET,          0x800004A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT56_CLR,          0x800004A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT56_TOG,          0x800004AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT57,              0x800004B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT57_SET,          0x800004B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT57_CLR,          0x800004B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT57_TOG,          0x800004BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT58,              0x800004C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT58_SET,          0x800004C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT58_CLR,          0x800004C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT58_TOG,          0x800004CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT59,              0x800004D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT59_SET,          0x800004D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT59_CLR,          0x800004D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT59_TOG,          0x800004DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT60,              0x800004E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT60_SET,          0x800004E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT60_CLR,          0x800004E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT60_TOG,          0x800004EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT61,              0x800004F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT61_SET,          0x800004F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT61_CLR,          0x800004F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT61_TOG,          0x800004FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT62,              0x80000500,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT62_SET,          0x80000504,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT62_CLR,          0x80000508,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT62_TOG,          0x8000050C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT63,              0x80000510,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT63_SET,          0x80000514,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT63_CLR,          0x80000518,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT63_TOG,          0x8000051C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT64,              0x80000520,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT64_SET,          0x80000524,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT64_CLR,          0x80000528,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT64_TOG,          0x8000052C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT65,              0x80000530,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT65_SET,          0x80000534,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT65_CLR,          0x80000538,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT65_TOG,          0x8000053C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT66,              0x80000540,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT66_SET,          0x80000544,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT66_CLR,          0x80000548,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT66_TOG,          0x8000054C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT67,              0x80000550,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT67_SET,          0x80000554,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT67_CLR,          0x80000558,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT67_TOG,          0x8000055C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT68,              0x80000560,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT68_SET,          0x80000564,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT68_CLR,          0x80000568,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT68_TOG,          0x8000056C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT69,              0x80000570,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT69_SET,          0x80000574,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT69_CLR,          0x80000578,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT69_TOG,          0x8000057C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT70,              0x80000580,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT70_SET,          0x80000584,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT70_CLR,          0x80000588,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT70_TOG,          0x8000058C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT71,              0x80000590,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT71_SET,          0x80000594,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT71_CLR,          0x80000598,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT71_TOG,          0x8000059C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT72,              0x800005A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT72_SET,          0x800005A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT72_CLR,          0x800005A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT72_TOG,          0x800005AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT73,              0x800005B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT73_SET,          0x800005B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT73_CLR,          0x800005B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT73_TOG,          0x800005BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT74,              0x800005C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT74_SET,          0x800005C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT74_CLR,          0x800005C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT74_TOG,          0x800005CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT75,              0x800005D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT75_SET,          0x800005D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT75_CLR,          0x800005D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT75_TOG,          0x800005DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT76,              0x800005E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT76_SET,          0x800005E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT76_CLR,          0x800005E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT76_TOG,          0x800005EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT77,              0x800005F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT77_SET,          0x800005F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT77_CLR,          0x800005F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT77_TOG,          0x800005FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT78,              0x80000600,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT78_SET,          0x80000604,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT78_CLR,          0x80000608,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT78_TOG,          0x8000060C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT79,              0x80000610,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT79_SET,          0x80000614,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT79_CLR,          0x80000618,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT79_TOG,          0x8000061C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT80,              0x80000620,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT80_SET,          0x80000624,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT80_CLR,          0x80000628,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT80_TOG,          0x8000062C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT81,              0x80000630,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT81_SET,          0x80000634,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT81_CLR,          0x80000638,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT81_TOG,          0x8000063C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT82,              0x80000640,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT82_SET,          0x80000644,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT82_CLR,          0x80000648,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT82_TOG,          0x8000064C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT83,              0x80000650,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT83_SET,          0x80000654,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT83_CLR,          0x80000658,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT83_TOG,          0x8000065C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT84,              0x80000660,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT84_SET,          0x80000664,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT84_CLR,          0x80000668,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT84_TOG,          0x8000066C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT85,              0x80000670,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT85_SET,          0x80000674,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT85_CLR,          0x80000678,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT85_TOG,          0x8000067C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT86,              0x80000680,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT86_SET,          0x80000684,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT86_CLR,          0x80000688,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT86_TOG,          0x8000068C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT87,              0x80000690,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT87_SET,          0x80000694,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT87_CLR,          0x80000698,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT87_TOG,          0x8000069C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT88,              0x800006A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT88_SET,          0x800006A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT88_CLR,          0x800006A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT88_TOG,          0x800006AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT89,              0x800006B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT89_SET,          0x800006B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT89_CLR,          0x800006B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT89_TOG,          0x800006BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT90,              0x800006C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT90_SET,          0x800006C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT90_CLR,          0x800006C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT90_TOG,          0x800006CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT91,              0x800006D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT91_SET,          0x800006D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT91_CLR,          0x800006D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT91_TOG,          0x800006DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT92,              0x800006E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT92_SET,          0x800006E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT92_CLR,          0x800006E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT92_TOG,          0x800006EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT93,              0x800006F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT93_SET,          0x800006F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT93_CLR,          0x800006F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT93_TOG,          0x800006FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT94,              0x80000700,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT94_SET,          0x80000704,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT94_CLR,          0x80000708,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT94_TOG,          0x8000070C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT95,              0x80000710,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT95_SET,          0x80000714,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT95_CLR,          0x80000718,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT95_TOG,          0x8000071C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT96,              0x80000720,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT96_SET,          0x80000724,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT96_CLR,          0x80000728,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT96_TOG,          0x8000072C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT97,              0x80000730,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT97_SET,          0x80000734,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT97_CLR,          0x80000738,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT97_TOG,          0x8000073C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT98,              0x80000740,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT98_SET,          0x80000744,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT98_CLR,          0x80000748,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT98_TOG,          0x8000074C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT99,              0x80000750,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT99_SET,          0x80000754,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT99_CLR,          0x80000758,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT99_TOG,          0x8000075C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT100,             0x80000760,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT100_SET,         0x80000764,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT100_CLR,         0x80000768,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT100_TOG,         0x8000076C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT101,             0x80000770,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT101_SET,         0x80000774,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT101_CLR,         0x80000778,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT101_TOG,         0x8000077C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT102,             0x80000780,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT102_SET,         0x80000784,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT102_CLR,         0x80000788,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT102_TOG,         0x8000078C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT103,             0x80000790,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT103_SET,         0x80000794,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT103_CLR,         0x80000798,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT103_TOG,         0x8000079C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT104,             0x800007A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT104_SET,         0x800007A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT104_CLR,         0x800007A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT104_TOG,         0x800007AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT105,             0x800007B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT105_SET,         0x800007B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT105_CLR,         0x800007B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT105_TOG,         0x800007BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT106,             0x800007C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT106_SET,         0x800007C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT106_CLR,         0x800007C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT106_TOG,         0x800007CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT107,             0x800007D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT107_SET,         0x800007D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT107_CLR,         0x800007D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT107_TOG,         0x800007DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT108,             0x800007E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT108_SET,         0x800007E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT108_CLR,         0x800007E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT108_TOG,         0x800007EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT109,             0x800007F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT109_SET,         0x800007F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT109_CLR,         0x800007F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT109_TOG,         0x800007FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT110,             0x80000800,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT110_SET,         0x80000804,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT110_CLR,         0x80000808,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT110_TOG,         0x8000080C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT111,             0x80000810,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT111_SET,         0x80000814,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT111_CLR,         0x80000818,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT111_TOG,         0x8000081C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT112,             0x80000820,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT112_SET,         0x80000824,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT112_CLR,         0x80000828,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT112_TOG,         0x8000082C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT113,             0x80000830,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT113_SET,         0x80000834,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT113_CLR,         0x80000838,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT113_TOG,         0x8000083C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT114,             0x80000840,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT114_SET,         0x80000844,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT114_CLR,         0x80000848,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT114_TOG,         0x8000084C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT115,             0x80000850,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT115_SET,         0x80000854,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT115_CLR,         0x80000858,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT115_TOG,         0x8000085C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT116,             0x80000860,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT116_SET,         0x80000864,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT116_CLR,         0x80000868,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT116_TOG,         0x8000086C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT117,             0x80000870,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT117_SET,         0x80000874,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT117_CLR,         0x80000878,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT117_TOG,         0x8000087C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT118,             0x80000880,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT118_SET,         0x80000884,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT118_CLR,         0x80000888,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT118_TOG,         0x8000088C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT119,             0x80000890,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT119_SET,         0x80000894,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT119_CLR,         0x80000898,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT119_TOG,         0x8000089C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT120,             0x800008A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT120_SET,         0x800008A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT120_CLR,         0x800008A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT120_TOG,         0x800008AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT121,             0x800008B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT121_SET,         0x800008B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT121_CLR,         0x800008B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT121_TOG,         0x800008BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT122,             0x800008C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT122_SET,         0x800008C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT122_CLR,         0x800008C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT122_TOG,         0x800008CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT123,             0x800008D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT123_SET,         0x800008D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT123_CLR,         0x800008D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT123_TOG,         0x800008DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT124,             0x800008E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT124_SET,         0x800008E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT124_CLR,         0x800008E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT124_TOG,         0x800008EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT125,             0x800008F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT125_SET,         0x800008F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT125_CLR,         0x800008F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT125_TOG,         0x800008FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT126,             0x80000900,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT126_SET,         0x80000904,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT126_CLR,         0x80000908,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT126_TOG,         0x8000090C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT127,             0x80000910,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT127_SET,         0x80000914,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT127_CLR,         0x80000918,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT127_TOG,         0x8000091C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_DEBUG,       			        0x80001120,__READ				,__hw_icoll_debug_bits);
__IO_REG32(		 HW_ICOLL_DBGREAD0,                	0x80001130,__READ				);
__IO_REG32(		 HW_ICOLL_DBGREAD1,               	0x80001140,__READ				);
__IO_REG32_BIT(HW_ICOLL_DBGFLAG,       	        	0x80001150,__READ				,__hw_icoll_dbgflag_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST0,              0x80001160,__READ				,__hw_icoll_dbgrequest0_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST1,              0x80001170,__READ				,__hw_icoll_dbgrequest1_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST2,              0x80001180,__READ				,__hw_icoll_dbgrequest2_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST3,              0x80001190,__READ				,__hw_icoll_dbgrequest3_bits);
__IO_REG32_BIT(HW_ICOLL_VERSION,      	        	0x800011E0,__READ				,__hw_icoll_version_bits);

/***************************************************************************
 **
 **  DIGCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_DIGCTL_CTRL,                    0x8001C000,__READ_WRITE ,__hw_digctl_ctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_CTRL_SET,                0x8001C004,__WRITE      ,__hw_digctl_ctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_CTRL_CLR,                0x8001C008,__WRITE      ,__hw_digctl_ctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_CTRL_TOG,                0x8001C00C,__WRITE      ,__hw_digctl_ctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_STATUS,                  0x8001C010,__READ       ,__hw_digctl_status_bits);
__IO_REG32(    HW_DIGCTL_HCLKCOUNT,               0x8001C020,__READ       );
__IO_REG32_BIT(HW_DIGCTL_RAMCTRL,                 0x8001C030,__READ_WRITE ,__hw_digctl_ramctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMCTRL_SET,             0x8001C034,__WRITE      ,__hw_digctl_ramctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMCTRL_CLR,             0x8001C038,__WRITE      ,__hw_digctl_ramctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMCTRL_TOG,             0x8001C03C,__WRITE      ,__hw_digctl_ramctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMREPAIR,               0x8001C040,__READ_WRITE ,__hw_digctl_ramrepair_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMREPAIR_SET,           0x8001C044,__WRITE      ,__hw_digctl_ramrepair_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMREPAIR_CLR,           0x8001C048,__WRITE      ,__hw_digctl_ramrepair_bits);
__IO_REG32_BIT(HW_DIGCTL_RAMREPAIR_TOG,           0x8001C04C,__WRITE      ,__hw_digctl_ramrepair_bits);
__IO_REG32_BIT(HW_DIGCTL_ROMCTRL,                 0x8001C050,__READ_WRITE ,__hw_digctl_romctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_ROMCTRL_SET,             0x8001C054,__WRITE      ,__hw_digctl_romctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_ROMCTRL_CLR,             0x8001C058,__WRITE      ,__hw_digctl_romctrl_bits);
__IO_REG32_BIT(HW_DIGCTL_ROMCTRL_TOG,             0x8001C05C,__WRITE      ,__hw_digctl_romctrl_bits);
__IO_REG32(    HW_DIGCTL_WRITEONCE,               0x8001C060,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_ENTROPY,                 0x8001C070,__READ       );
__IO_REG32(    HW_DIGCTL_ENTROPY_LATCHED,         0x8001C0A0,__READ       );
__IO_REG32_BIT(HW_DIGCTL_SJTAGDBG,                0x8001C0B0,__READ_WRITE ,__hw_digctl_sjtagdbg_bits);
__IO_REG32_BIT(HW_DIGCTL_SJTAGDBG_SET,            0x8001C0B4,__WRITE      ,__hw_digctl_sjtagdbg_bits);
__IO_REG32_BIT(HW_DIGCTL_SJTAGDBG_CLR,            0x8001C0B8,__WRITE      ,__hw_digctl_sjtagdbg_bits);
__IO_REG32_BIT(HW_DIGCTL_SJTAGDBG_TOG,            0x8001C0BC,__WRITE      ,__hw_digctl_sjtagdbg_bits);
__IO_REG32(    HW_DIGCTL_MICROSECONDS,            0x8001C0C0,__READ       );
__IO_REG32(    HW_DIGCTL_DBGRD,                   0x8001C0D0,__READ       );
__IO_REG32(    HW_DIGCTL_DBG,                     0x8001C0E0,__READ       );
__IO_REG32_BIT(HW_DIGCTL_OCRAM_BIST_CSR,          0x8001C0F0,__READ_WRITE ,__hw_digctl_ocram_bist_csr_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_BIST_CSR_SET,      0x8001C0F4,__WRITE      ,__hw_digctl_ocram_bist_csr_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_BIST_CSR_CLR,      0x8001C0F8,__WRITE      ,__hw_digctl_ocram_bist_csr_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_BIST_CSR_TOG,      0x8001C0FC,__WRITE      ,__hw_digctl_ocram_bist_csr_bits);
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS0,           0x8001C110,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS1,           0x8001C120,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS2,           0x8001C130,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS3,           0x8001C140,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS4,           0x8001C150,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS5,           0x8001C160,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS6,           0x8001C170,__READ       );
__IO_REG32(    HW_DIGCTL_OCRAM_STATUS7,           0x8001C180,__READ       );
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS8,           0x8001C190,__READ       ,__hw_digctl_ocram_status8_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS9,           0x8001C1A0,__READ       ,__hw_digctl_ocram_status9_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS10,          0x8001C1B0,__READ       ,__hw_digctl_ocram_status10_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS11,          0x8001C1C0,__READ       ,__hw_digctl_ocram_status11_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS12,          0x8001C1D0,__READ       ,__hw_digctl_ocram_status12_bits);
__IO_REG32_BIT(HW_DIGCTL_OCRAM_STATUS13,          0x8001C1E0,__READ       ,__hw_digctl_ocram_status13_bits);
__IO_REG32(    HW_DIGCTL_SCRATCH0,                0x8001C290,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_SCRATCH1,                0x8001C2A0,__READ_WRITE );
__IO_REG32_BIT(HW_DIGCTL_ARMCACHE,                0x8001C2B0,__READ_WRITE ,__hw_digctl_armcache_bits);
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_ADDR_LOW,     0x8001C2C0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_ADDR_HIGH,    0x8001C2D0,__READ_WRITE );
__IO_REG8(     HW_DIGCTL_SGTL,                    0x8001C300,__READ       );
__IO_REG32_BIT(HW_DIGCTL_CHIPID,                  0x8001C310,__READ       ,__hw_digctl_chipid_bits);
__IO_REG32_BIT(HW_DIGCTL_AHB_STATS_SELECT,        0x8001C330,__READ_WRITE ,__hw_digctl_ahb_stats_select_bits);
__IO_REG32(    HW_DIGCTL_L0_AHB_ACTIVE_CYCLES,    0x8001C340,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L0_AHB_DATA_STALLED,     0x8001C350,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L0_AHB_DATA_CYCLES,      0x8001C360,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L1_AHB_ACTIVE_CYCLES,    0x8001C370,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L1_AHB_DATA_STALLED,     0x8001C380,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L1_AHB_DATA_CYCLES,      0x8001C390,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_ACTIVE_CYCLES,    0x8001C3A0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_DATA_STALLED,     0x8001C3B0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_DATA_CYCLES,      0x8001C3C0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_ACTIVE_CYCLES,    0x8001C3D0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_DATA_STALLED,     0x8001C3E0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_DATA_CYCLES,      0x8001C3F0,__READ_WRITE );
__IO_REG32_BIT(HW_DIGCTL_EMICLK_DELAY,            0x8001C500,__READ_WRITE ,__hw_digctl_emiclk_delay_bits);

/***************************************************************************
 **
 **  OCOTP
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_OCOTP_CTRL,                     0x8002C000,__READ_WRITE ,__hw_ocotp_ctrl_bits);
__IO_REG32_BIT(HW_OCOTP_CTRL_SET,                 0x8002C004,__WRITE      ,__hw_ocotp_ctrl_bits);
__IO_REG32_BIT(HW_OCOTP_CTRL_CLR,                 0x8002C008,__WRITE      ,__hw_ocotp_ctrl_bits);
__IO_REG32_BIT(HW_OCOTP_CTRL_TOG,                 0x8002C00C,__WRITE      ,__hw_ocotp_ctrl_bits);
__IO_REG32(    HW_OCOTP_DATA,                     0x8002C010,__READ_WRITE );
__IO_REG32(    HW_OCOTP_CUST0,                    0x8002C020,__READ       );
__IO_REG32(    HW_OCOTP_CUST1,                    0x8002C030,__READ       );
__IO_REG32(    HW_OCOTP_CUST2,                    0x8002C040,__READ       );
__IO_REG32(    HW_OCOTP_CUST3,                    0x8002C050,__READ       );
__IO_REG32(    HW_OCOTP_CRYPTO0,                  0x8002C060,__READ       );
__IO_REG32(    HW_OCOTP_CRYPTO1,                  0x8002C070,__READ       );
__IO_REG32(    HW_OCOTP_CRYPTO2,                  0x8002C080,__READ       );
__IO_REG32(    HW_OCOTP_CRYPTO3,                  0x8002C090,__READ       );
__IO_REG32(    HW_OCOTP_HWCAP0,                   0x8002C0A0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_HWCAP1,                   0x8002C0B0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_HWCAP2,                   0x8002C0C0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_HWCAP3,                   0x8002C0D0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_HWCAP4,                   0x8002C0E0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_HWCAP5,                   0x8002C0F0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SWCAP,                    0x8002C100,__READ_WRITE );
__IO_REG32_BIT(HW_OCOTP_CUSTCAP,                  0x8002C110,__READ_WRITE ,__hw_ocotp_custcap_bits);
__IO_REG32_BIT(HW_OCOTP_LOCK,                     0x8002C120,__READ       ,__hw_ocotp_lock_bits);
__IO_REG32(    HW_OCOTP_OPS0,                     0x8002C130,__READ       );
__IO_REG32(    HW_OCOTP_OPS1,                     0x8002C140,__READ       );
__IO_REG32(    HW_OCOTP_OPS2,                     0x8002C150,__READ       );
__IO_REG32(    HW_OCOTP_OPS3,                     0x8002C160,__READ       );
__IO_REG32(    HW_OCOTP_UN0,                      0x8002C170,__READ       );
__IO_REG32(    HW_OCOTP_UN1,                      0x8002C180,__READ       );
__IO_REG32(    HW_OCOTP_UN2,                      0x8002C190,__READ       );
__IO_REG32_BIT(HW_OCOTP_ROM0,                     0x8002C1A0,__READ_WRITE ,__hw_ocotp_rom0_bits);
__IO_REG32_BIT(HW_OCOTP_ROM1,                     0x8002C1B0,__READ_WRITE ,__hw_ocotp_rom1_bits);
__IO_REG32_BIT(HW_OCOTP_ROM2,                     0x8002C1C0,__READ_WRITE ,__hw_ocotp_rom2_bits);
__IO_REG32(    HW_OCOTP_ROM3,                     0x8002C1D0,__READ       );
__IO_REG32(    HW_OCOTP_ROM4,                     0x8002C1E0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_ROM5,                     0x8002C1F0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_ROM6,                     0x8002C200,__READ_WRITE );
__IO_REG32_BIT(HW_OCOTP_ROM7,                     0x8002C210,__READ_WRITE ,__hw_ocotp_rom7_bits);
__IO_REG32_BIT(HW_OCOTP_VERSION,                  0x8002C220,__READ_WRITE ,__hw_ocotp_version_bits);

/***************************************************************************
 **
 **  USB Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBCTRL_ID,                     0x80080000,__READ       ,__hw_usbctrl_id_bits);
__IO_REG32_BIT(HW_USBCTRL_HWGENERAL,              0x80080004,__READ       ,__hw_usbctrl_hwgeneral_bits);
__IO_REG32_BIT(HW_USBCTRL_HWHOST,                 0x80080008,__READ       ,__hw_usbctrl_hwhost_bits);
__IO_REG32_BIT(HW_USBCTRL_HWDEVICE,               0x8008000C,__READ       ,__hw_usbctrl_hwdevice_bits);
__IO_REG32_BIT(HW_USBCTRL_HWTXBUF,                0x80080010,__READ       ,__hw_usbctrl_hwtxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL_HWRXBUF,                0x80080014,__READ       ,__hw_usbctrl_hwrxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL_GPTIMER0LD,             0x80080080,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL_GPTIMER0CTRL,           0x80080084,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_GPTIMER1LD,             0x80080088,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL_GPTIMER1CTRL,           0x8008008C,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_SBUSCFG,                0x80080090,__READ_WRITE ,__hw_usbctrl_sbuscfg_bits);
__IO_REG8(     HW_USBCTRL_CAPLENGTH,              0x80080100,__READ       );
__IO_REG16(    HW_USBCTRL_HCIVERSION,             0x80080102,__READ       );
__IO_REG32_BIT(HW_USBCTRL_HCSPARAMS,              0x80080104,__READ       ,__hw_usbctrl_hcsparams_bits);
__IO_REG32_BIT(HW_USBCTRL_HCCPARAMS,              0x80080108,__READ       ,__hw_usbctrl_hccparams_bits);
__IO_REG32_BIT(HW_USBCTRL_DCIVERSION,             0x80080120,__READ       ,__hw_usbctrl_dciversion_bits);
__IO_REG32_BIT(HW_USBCTRL_DCCPARAMS,              0x80080124,__READ       ,__hw_usbctrl_dccparams_bits);
__IO_REG32_BIT(HW_USBCTRL_USBCMD,                 0x80080140,__READ_WRITE ,__hw_usbctrl_usbcmd_bits);
__IO_REG32_BIT(HW_USBCTRL_USBSTS,                 0x80080144,__READ_WRITE ,__hw_usbctrl_usbsts_bits);
__IO_REG32_BIT(HW_USBCTRL_USBINTR,                0x80080148,__READ_WRITE ,__hw_usbctrl_usbintr_bits);
__IO_REG32_BIT(HW_USBCTRL_FRINDEX,                0x8008014C,__READ_WRITE ,__hw_usbctrl_frindex_bits);
__IO_REG32_BIT(HW_USBCTRL_PERIODICLISTBASE,       0x80080154,__READ_WRITE ,__hw_usbctrl_periodiclistbase_bits);
#define HW_USBCTRL_DEVICEADDR      HW_USBCTRL_PERIODICLISTBASE
#define HW_USBCTRL_DEVICEADDR_bit  HW_USBCTRL_PERIODICLISTBASE_bit
__IO_REG32_BIT(HW_USBCTRL_ASYNCLISTADDR,          0x80080158,__READ_WRITE ,__hw_usbctrl_asynclistaddr_bits);
#define HW_USBCTRL_ENDPOINTLISTADDR      HW_USBCTRL_ASYNCLISTADDR
#define HW_USBCTRL_ENDPOINTLISTADDR_bit  HW_USBCTRL_ASYNCLISTADDR_bit
__IO_REG32_BIT(HW_USBCTRL_TTCTRL,                 0x8008015C,__READ_WRITE ,__hw_usbctrl_ttctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_BURSTSIZE,              0x80080160,__READ_WRITE ,__hw_usbctrl_burstsize_bits);
__IO_REG32_BIT(HW_USBCTRL_TXFILLTUNING,           0x80080164,__READ_WRITE ,__hw_usbctrl_txfilltuning_bits);
__IO_REG32_BIT(HW_USBCTRL_IC_USB,                 0x8008016C,__READ_WRITE ,__hw_usbctrl_ic_usb_bits);
__IO_REG32_BIT(HW_USBCTRL_ULPI,                   0x80080170,__READ_WRITE ,__hw_usbctrl_ulpi_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTNAK,               0x80080178,__READ_WRITE ,__hw_usbctrl_endptnak_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTNAKEN,             0x8008017C,__READ_WRITE ,__hw_usbctrl_endptnaken_bits);
__IO_REG32_BIT(HW_USBCTRL_PORTSC1,                0x80080184,__READ_WRITE ,__hw_usbctrl_portsc_bits);
__IO_REG32_BIT(HW_USBCTRL_OTGSC,                  0x800801A4,__READ_WRITE ,__hw_usbctrl_otgsc_bits);
__IO_REG32_BIT(HW_USBCTRL_USBMODE,                0x800801A8,__READ_WRITE ,__hw_usbctrl_usbmode_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTSETUPSTAT,         0x800801AC,__READ_WRITE ,__hw_usbctrl_endptsetupstat_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTPRIME,             0x800801B0,__READ_WRITE ,__hw_usbctrl_endptprime_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTFLUSH,             0x800801B4,__READ_WRITE ,__hw_usbctrl_endptflush_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTSTAT,              0x800801B8,__READ       ,__hw_usbctrl_endptstat_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCOMPLETE,          0x800801BC,__READ_WRITE ,__hw_usbctrl_endptcomplete_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCTRL0,             0x800801C0,__READ_WRITE ,__hw_usbctrl_endptctrl0_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCTRL1,             0x800801C4,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCTRL2,             0x800801C8,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCTRL3,             0x800801CC,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL_ENDPTCTRL4,             0x800801D0,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);

/***************************************************************************
 **
 **  USB 2.0 PHY
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBPHY_PWD,                    0x8007C000,__READ_WRITE ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY_PWD_SET,                0x8007C004,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY_PWD_CLR,                0x8007C008,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY_PWD_TOG,                0x8007C00C,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY_TX,                     0x8007C010,__READ_WRITE ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY_TX_SET,                 0x8007C014,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY_TX_CLR,                 0x8007C018,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY_TX_TOG,                 0x8007C01C,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY_RX,                     0x8007C020,__READ_WRITE ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY_RX_SET,                 0x8007C024,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY_RX_CLR,                 0x8007C028,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY_RX_TOG,                 0x8007C02C,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY_CTRL,                   0x8007C030,__READ_WRITE ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY_CTRL_SET,               0x8007C034,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY_CTRL_CLR,               0x8007C038,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY_CTRL_TOG,               0x8007C03C,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY_STATUS,                 0x8007C040,__READ_WRITE ,__hw_usbphy_status_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG,                  0x8007C050,__READ_WRITE ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG_SET,              0x8007C054,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG_CLR,              0x8007C058,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG_TOG,              0x8007C05C,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG0_STATUS,          0x8007C060,__READ       ,__hw_usbphy_debug0_status_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG1,                 0x8007C070,__READ_WRITE ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG1_SET,             0x8007C074,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG1_CLR,             0x8007C078,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY_DEBUG1_TOG,             0x8007C07C,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY_VERSION,                0x8007C080,__READ       ,__hw_usbphy_version_bits);

/***************************************************************************
 **
 **  APBH DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_APBH_CTRL0,                    0x80004000,__READ_WRITE ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_SET,                0x80004004,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_CLR,                0x80004008,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_TOG,                0x8000400C,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL1,                    0x80004010,__READ_WRITE ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_SET,                0x80004014,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_CLR,                0x80004018,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_TOG,                0x8000401C,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL2,                    0x80004020,__READ_WRITE ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_SET,                0x80004024,__WRITE      ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_CLR,                0x80004028,__WRITE      ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_TOG,                0x8000402C,__WRITE      ,__hw_apbh_ctrl2_bits);
/*__IO_REG32_BIT(HW_APBH_DEVSEL,                   0x80004030,__READ_WRITE ,__hw_apbh_devsel_bits);*/
__IO_REG32(    HW_APBH_CH0_CURCMDAR,             0x80004040,__READ       );
__IO_REG32(    HW_APBH_CH0_NXTCMDAR,             0x80004050,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH0_CMD,                  0x80004060,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH0_BAR,                  0x80004070,__READ       );
__IO_REG32_BIT(HW_APBH_CH0_SEMA,                 0x80004080,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH0_DEBUG1,               0x80004090,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH0_DEBUG2,               0x800040A0,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH1_CURCMDAR,             0x800040B0,__READ       );
__IO_REG32(    HW_APBH_CH1_NXTCMDAR,             0x800040C0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH1_CMD,                  0x800040D0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH1_BAR,                  0x800040E0,__READ       );
__IO_REG32_BIT(HW_APBH_CH1_SEMA,                 0x800040F0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH1_DEBUG1,               0x80004100,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH1_DEBUG2,               0x80004110,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH2_CURCMDAR,             0x80004120,__READ       );
__IO_REG32(    HW_APBH_CH2_NXTCMDAR,             0x80004130,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH2_CMD,                  0x80004140,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH2_BAR,                  0x80004150,__READ       );
__IO_REG32_BIT(HW_APBH_CH2_SEMA,                 0x80004160,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH2_DEBUG1,               0x80004170,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH2_DEBUG2,               0x80004180,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH3_CURCMDAR,             0x80004190,__READ       );
__IO_REG32(    HW_APBH_CH3_NXTCMDAR,             0x800041A0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH3_CMD,                  0x800041B0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH3_BAR,                  0x800041C0,__READ       );
__IO_REG32_BIT(HW_APBH_CH3_SEMA,                 0x800041D0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH3_DEBUG1,               0x800041E0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH3_DEBUG2,               0x800041F0,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH4_CURCMDAR,             0x80004200,__READ       );
__IO_REG32(    HW_APBH_CH4_NXTCMDAR,             0x80004210,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH4_CMD,                  0x80004220,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH4_BAR,                  0x80004230,__READ       );
__IO_REG32_BIT(HW_APBH_CH4_SEMA,                 0x80004240,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH4_DEBUG1,               0x80004250,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH4_DEBUG2,               0x80004260,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH5_CURCMDAR,             0x80004270,__READ       );
__IO_REG32(    HW_APBH_CH5_NXTCMDAR,             0x80004280,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH5_CMD,                  0x80004290,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH5_BAR,                  0x800042A0,__READ       );
__IO_REG32_BIT(HW_APBH_CH5_SEMA,                 0x800042B0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH5_DEBUG1,               0x800042C0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH5_DEBUG2,               0x800042D0,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH6_CURCMDAR,             0x800042E0,__READ       );
__IO_REG32(    HW_APBH_CH6_NXTCMDAR,             0x800042F0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH6_CMD,                  0x80004300,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH6_BAR,                  0x80004310,__READ       );
__IO_REG32_BIT(HW_APBH_CH6_SEMA,                 0x80004320,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH6_DEBUG1,               0x80004330,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH6_DEBUG2,               0x80004340,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32(    HW_APBH_CH7_CURCMDAR,             0x80004350,__READ       );
__IO_REG32(    HW_APBH_CH7_NXTCMDAR,             0x80004360,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH7_CMD,                  0x80004370,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH7_BAR,                  0x80004380,__READ       );
__IO_REG32_BIT(HW_APBH_CH7_SEMA,                 0x80004390,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH7_DEBUG1,               0x800043A0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32_BIT(HW_APBH_CH7_DEBUG2,               0x800043B0,__READ       ,__hw_apbh_ch_debug2_bits);
__IO_REG32_BIT(HW_APBH_VERSION,                  0x800043F0,__READ       ,__hw_apbh_version_bits);

/***************************************************************************
 **
 **  APBX DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_APBX_CTRL0,                    0x80024000,__READ_WRITE ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_SET,                0x80024004,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_CLR,                0x80024008,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_TOG,                0x8002400C,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL1,                    0x80024010,__READ_WRITE ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_SET,                0x80024014,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_CLR,                0x80024018,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_TOG,                0x8002401C,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL2,                    0x80024020,__READ_WRITE ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_SET,                0x80024024,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_CLR,                0x80024028,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_TOG,                0x8002402C,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL,             0x80024030,__READ_WRITE ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_SET,         0x80024034,__WRITE      ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_CLR,         0x80024038,__WRITE      ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_TOG,         0x8002403C,__WRITE      ,__hw_apbx_channel_ctrl_bits);
/*__IO_REG32_BIT(HW_APBX_DEVSEL,                   0x80024040,__READ_WRITE ,__HW_APBX_devsel_bits);*/
__IO_REG32(    HW_APBX_CH0_CURCMDAR,             0x80024100,__READ       );
__IO_REG32(    HW_APBX_CH0_NXTCMDAR,             0x80024110,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH0_CMD,                  0x80024120,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH0_BAR,                  0x80024130,__READ       );
__IO_REG32_BIT(HW_APBX_CH0_SEMA,                 0x80024140,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH0_DEBUG1,               0x80024150,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH0_DEBUG2,               0x80024160,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH1_CURCMDAR,             0x80024170,__READ       );
__IO_REG32(    HW_APBX_CH1_NXTCMDAR,             0x80024180,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH1_CMD,                  0x80024190,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH1_BAR,                  0x800241A0,__READ       );
__IO_REG32_BIT(HW_APBX_CH1_SEMA,                 0x800241B0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH1_DEBUG1,               0x800241C0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH1_DEBUG2,               0x800241D0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH2_CURCMDAR,             0x800241E0,__READ       );
__IO_REG32(    HW_APBX_CH2_NXTCMDAR,             0x800241F0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH2_CMD,                  0x80024200,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH2_BAR,                  0x80024210,__READ       );
__IO_REG32_BIT(HW_APBX_CH2_SEMA,                 0x80024220,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH2_DEBUG1,               0x80024230,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH2_DEBUG2,               0x80024240,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH3_CURCMDAR,             0x80024250,__READ       );
__IO_REG32(    HW_APBX_CH3_NXTCMDAR,             0x80024260,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH3_CMD,                  0x80024270,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH3_BAR,                  0x80024280,__READ       );
__IO_REG32_BIT(HW_APBX_CH3_SEMA,                 0x80024290,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH3_DEBUG1,               0x800242A0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH3_DEBUG2,               0x800242B0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH4_CURCMDAR,             0x800242C0,__READ       );
__IO_REG32(    HW_APBX_CH4_NXTCMDAR,             0x800242D0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH4_CMD,                  0x800242E0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH4_BAR,                  0x800242F0,__READ       );
__IO_REG32_BIT(HW_APBX_CH4_SEMA,                 0x80024300,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH4_DEBUG1,               0x80024310,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH4_DEBUG2,               0x80024320,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH5_CURCMDAR,             0x80024330,__READ       );
__IO_REG32(    HW_APBX_CH5_NXTCMDAR,             0x80024340,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH5_CMD,                  0x80024350,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH5_BAR,                  0x80024360,__READ       );
__IO_REG32_BIT(HW_APBX_CH5_SEMA,                 0x80024370,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH5_DEBUG1,               0x80024380,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH5_DEBUG2,               0x80024390,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH6_CURCMDAR,             0x800243A0,__READ       );
__IO_REG32(    HW_APBX_CH6_NXTCMDAR,             0x800243B0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH6_CMD,                  0x800243C0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH6_BAR,                  0x800243D0,__READ       );
__IO_REG32_BIT(HW_APBX_CH6_SEMA,                 0x800243E0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH6_DEBUG1,               0x800243F0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH6_DEBUG2,               0x80024400,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH7_CURCMDAR,             0x80024410,__READ       );
__IO_REG32(    HW_APBX_CH7_NXTCMDAR,             0x80024420,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH7_CMD,                  0x80024430,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH7_BAR,                  0x80024440,__READ       );
__IO_REG32_BIT(HW_APBX_CH7_SEMA,                 0x80024450,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH7_DEBUG1,               0x80024460,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH7_DEBUG2,               0x80024470,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH8_CURCMDAR,             0x80024480,__READ       );
__IO_REG32(    HW_APBX_CH8_NXTCMDAR,             0x80024490,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH8_CMD,                  0x800244A0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH8_BAR,                  0x800244B0,__READ       );
__IO_REG32_BIT(HW_APBX_CH8_SEMA,                 0x800244C0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH8_DEBUG1,               0x800244D0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH8_DEBUG2,               0x800244E0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH9_CURCMDAR,             0x800244F0,__READ       );
__IO_REG32(    HW_APBX_CH9_NXTCMDAR,             0x80024500,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH9_CMD,                  0x80024510,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH9_BAR,                  0x80024520,__READ       );
__IO_REG32_BIT(HW_APBX_CH9_SEMA,                 0x80024530,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH9_DEBUG1,               0x80024540,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH9_DEBUG2,               0x80024550,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH10_CURCMDAR,            0x80024560,__READ       );
__IO_REG32(    HW_APBX_CH10_NXTCMDAR,            0x80024570,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH10_CMD,                 0x80024580,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH10_BAR,                 0x80024590,__READ       );
__IO_REG32_BIT(HW_APBX_CH10_SEMA,                0x800245A0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH10_DEBUG1,              0x800245B0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH10_DEBUG2,              0x800245C0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH11_CURCMDAR,            0x800245D0,__READ       );
__IO_REG32(    HW_APBX_CH11_NXTCMDAR,            0x800245E0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH11_CMD,                 0x800245F0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH11_BAR,                 0x80024600,__READ       );
__IO_REG32_BIT(HW_APBX_CH11_SEMA,                0x80024610,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH11_DEBUG1,              0x80024620,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH11_DEBUG2,              0x80024630,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH12_CURCMDAR,            0x80024640,__READ       );
__IO_REG32(    HW_APBX_CH12_NXTCMDAR,            0x80024650,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH12_CMD,                 0x80024660,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH12_BAR,                 0x80024670,__READ       );
__IO_REG32_BIT(HW_APBX_CH12_SEMA,                0x80024680,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH12_DEBUG1,              0x80024690,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH12_DEBUG2,              0x800246A0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH13_CURCMDAR,            0x800246B0,__READ       );
__IO_REG32(    HW_APBX_CH13_NXTCMDAR,            0x800246C0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH13_CMD,                 0x800246D0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH13_BAR,                 0x800246E0,__READ       );
__IO_REG32_BIT(HW_APBX_CH13_SEMA,                0x800246F0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH13_DEBUG1,              0x80024700,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH13_DEBUG2,              0x80024710,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH14_CURCMDAR,            0x80024720,__READ       );
__IO_REG32(    HW_APBX_CH14_NXTCMDAR,            0x80024730,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH14_CMD,                 0x80024740,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH14_BAR,                 0x80024750,__READ       );
__IO_REG32_BIT(HW_APBX_CH14_SEMA,                0x80024760,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH14_DEBUG1,              0x80024770,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH14_DEBUG2,              0x80024780,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32(    HW_APBX_CH15_CURCMDAR,            0x80024790,__READ       );
__IO_REG32(    HW_APBX_CH15_NXTCMDAR,            0x800247A0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH15_CMD,                 0x800247B0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH15_BAR,                 0x800247C0,__READ       );
__IO_REG32_BIT(HW_APBX_CH15_SEMA,                0x800247D0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH15_DEBUG1,              0x800247E0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32_BIT(HW_APBX_CH15_DEBUG2,              0x800247F0,__READ       ,__hw_apbx_ch_debug2_bits);
__IO_REG32_BIT(HW_APBX_VERSION,                  0x80024800,__READ       ,__hw_apbx_version_bits);

/***************************************************************************
 **
 **  EMI
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_EMI_CTRL,                      0x80020000,__READ_WRITE ,__hw_emi_ctrl_bits);
__IO_REG32_BIT(HW_EMI_CTRL_SET,                  0x80020004,__WRITE      ,__hw_emi_ctrl_bits);
__IO_REG32_BIT(HW_EMI_CTRL_CLR,                  0x80020008,__WRITE      ,__hw_emi_ctrl_bits);
__IO_REG32_BIT(HW_EMI_CTRL_TOG,                  0x8002000C,__WRITE      ,__hw_emi_ctrl_bits);
__IO_REG32_BIT(HW_EMI_VERSION,                   0x800200F0,__READ_WRITE ,__hw_emi_version_bits);
__IO_REG32_BIT(HW_DRAM_CTL00,                    0x800E0000,__READ_WRITE ,__hw_dram_ctl00_bits);
__IO_REG32_BIT(HW_DRAM_CTL01,                    0x800E0004,__READ_WRITE ,__hw_dram_ctl01_bits);
__IO_REG32_BIT(HW_DRAM_CTL02,                    0x800E0008,__READ_WRITE ,__hw_dram_ctl02_bits);
__IO_REG32_BIT(HW_DRAM_CTL03,                    0x800E000C,__READ_WRITE ,__hw_dram_ctl03_bits);
__IO_REG32_BIT(HW_DRAM_CTL04,                    0x800E0010,__READ_WRITE ,__hw_dram_ctl04_bits);
__IO_REG32_BIT(HW_DRAM_CTL05,                    0x800E0014,__READ_WRITE ,__hw_dram_ctl05_bits);
__IO_REG32_BIT(HW_DRAM_CTL06,                    0x800E0018,__READ_WRITE ,__hw_dram_ctl06_bits);
__IO_REG32_BIT(HW_DRAM_CTL07,                    0x800E001C,__READ_WRITE ,__hw_dram_ctl07_bits);
__IO_REG32_BIT(HW_DRAM_CTL08,                    0x800E0020,__READ_WRITE ,__hw_dram_ctl08_bits);
__IO_REG32_BIT(HW_DRAM_CTL09,                    0x800E0024,__READ_WRITE ,__hw_dram_ctl09_bits);
__IO_REG32_BIT(HW_DRAM_CTL10,                    0x800E0028,__READ_WRITE ,__hw_dram_ctl10_bits);
__IO_REG32_BIT(HW_DRAM_CTL11,                    0x800E002C,__READ_WRITE ,__hw_dram_ctl11_bits);
__IO_REG32_BIT(HW_DRAM_CTL12,                    0x800E0030,__READ_WRITE ,__hw_dram_ctl12_bits);
__IO_REG32_BIT(HW_DRAM_CTL13,                    0x800E0034,__READ_WRITE ,__hw_dram_ctl13_bits);
__IO_REG32_BIT(HW_DRAM_CTL14,                    0x800E0038,__READ_WRITE ,__hw_dram_ctl14_bits);
__IO_REG32_BIT(HW_DRAM_CTL15,                    0x800E003C,__READ_WRITE ,__hw_dram_ctl15_bits);
__IO_REG32_BIT(HW_DRAM_CTL16,                    0x800E0040,__READ_WRITE ,__hw_dram_ctl16_bits);
__IO_REG32_BIT(HW_DRAM_CTL17,                    0x800E0044,__READ_WRITE ,__hw_dram_ctl17_bits);
__IO_REG32_BIT(HW_DRAM_CTL18,                    0x800E0048,__READ_WRITE ,__hw_dram_ctl18_bits);
__IO_REG32_BIT(HW_DRAM_CTL19,                    0x800E004C,__READ_WRITE ,__hw_dram_ctl19_bits);
__IO_REG32_BIT(HW_DRAM_CTL20,                    0x800E0050,__READ_WRITE ,__hw_dram_ctl20_bits);
__IO_REG32_BIT(HW_DRAM_CTL21,                    0x800E0054,__READ_WRITE ,__hw_dram_ctl21_bits);
__IO_REG32_BIT(HW_DRAM_CTL22,                    0x800E0058,__READ_WRITE ,__hw_dram_ctl22_bits);
__IO_REG32_BIT(HW_DRAM_CTL23,                    0x800E005C,__READ_WRITE ,__hw_dram_ctl23_bits);
__IO_REG32_BIT(HW_DRAM_CTL24,                    0x800E0060,__READ_WRITE ,__hw_dram_ctl24_bits);
__IO_REG32_BIT(HW_DRAM_CTL25,                    0x800E0064,__READ_WRITE ,__hw_dram_ctl25_bits);
__IO_REG32_BIT(HW_DRAM_CTL26,                    0x800E0068,__READ_WRITE ,__hw_dram_ctl26_bits);
/*__IO_REG32_BIT(HW_DRAM_CTL27,                    0x800E006C,__READ_WRITE ,__hw_dram_ctl27_bits);*/
/*__IO_REG32_BIT(HW_DRAM_CTL28,                    0x800E0070,__READ_WRITE ,__hw_dram_ctl28_bits);*/
__IO_REG32_BIT(HW_DRAM_CTL29,                    0x800E0074,__READ_WRITE ,__hw_dram_ctl29_bits);
__IO_REG32_BIT(HW_DRAM_CTL30,                    0x800E0078,__READ_WRITE ,__hw_dram_ctl30_bits);
__IO_REG32_BIT(HW_DRAM_CTL31,                    0x800E007C,__READ_WRITE ,__hw_dram_ctl31_bits);
__IO_REG32_BIT(HW_DRAM_CTL32,                    0x800E0080,__READ_WRITE ,__hw_dram_ctl32_bits);
__IO_REG32_BIT(HW_DRAM_CTL33,                    0x800E0084,__READ_WRITE ,__hw_dram_ctl33_bits);
__IO_REG32_BIT(HW_DRAM_CTL34,                    0x800E0088,__READ_WRITE ,__hw_dram_ctl34_bits);
__IO_REG32_BIT(HW_DRAM_CTL35,                    0x800E008C,__READ       ,__hw_dram_ctl35_bits);
__IO_REG32_BIT(HW_DRAM_CTL36,                    0x800E0090,__READ_WRITE ,__hw_dram_ctl36_bits);
__IO_REG32_BIT(HW_DRAM_CTL37,                    0x800E0094,__READ_WRITE ,__hw_dram_ctl37_bits);
__IO_REG32_BIT(HW_DRAM_CTL38,                    0x800E0098,__READ_WRITE ,__hw_dram_ctl38_bits);
__IO_REG32_BIT(HW_DRAM_CTL39,                    0x800E009C,__READ_WRITE ,__hw_dram_ctl39_bits);
__IO_REG32_BIT(HW_DRAM_CTL40,                    0x800E00A0,__READ_WRITE ,__hw_dram_ctl40_bits);

/***************************************************************************
 **
 **  GPMI
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_GPMI_CTRL0,                    0x8000C000,__READ_WRITE ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_SET,                0x8000C004,__WRITE      ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_CLR,                0x8000C008,__WRITE      ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_TOG,                0x8000C00C,__WRITE      ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_COMPARE,                  0x8000C010,__READ_WRITE ,__hw_gpmi_compare_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL,                  0x8000C020,__READ_WRITE ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_SET,              0x8000C024,__WRITE      ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_CLR,              0x8000C028,__WRITE      ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_TOG,              0x8000C02C,__WRITE      ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCOUNT,                 0x8000C030,__READ_WRITE ,__hw_gpmi_ecccount_bits);
__IO_REG32(    HW_GPMI_PAYLOAD,                  0x8000C040,__READ_WRITE );
__IO_REG32(    HW_GPMI_AUXILIARY,                0x8000C050,__READ_WRITE );
__IO_REG32_BIT(HW_GPMI_CTRL1,                    0x8000C060,__READ_WRITE ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_SET,                0x8000C064,__WRITE      ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_CLR,                0x8000C068,__WRITE      ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_TOG,                0x8000C06C,__WRITE      ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_TIMING0,                  0x8000C070,__READ_WRITE ,__hw_gpmi_timing0_bits);
__IO_REG32_BIT(HW_GPMI_TIMING1,                  0x8000C080,__READ_WRITE ,__hw_gpmi_timing1_bits);
__IO_REG32(    HW_GPMI_DATA,                     0x8000C0A0,__READ_WRITE );
__IO_REG32_BIT(HW_GPMI_STAT,                     0x8000C0B0,__READ       ,__hw_gpmi_stat_bits);
__IO_REG32_BIT(HW_GPMI_DEBUG,                    0x8000C0C0,__READ       ,__hw_gpmi_debug_bits);
__IO_REG32_BIT(HW_GPMI_VERSION,                  0x8000C0D0,__READ       ,__hw_gpmi_version_bits);
__IO_REG32_BIT(HW_GPMI_DEBUG2,                   0x8000C0E0,__READ_WRITE ,__hw_gpmi_debug2_bits);
__IO_REG32_BIT(HW_GPMI_DEBUG3,                   0x8000C0F0,__READ       ,__hw_gpmi_debug3_bits);

/***************************************************************************
 **
 **  ECC8
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_ECC8_CTRL,                     0x80008000,__READ_WRITE ,__hw_ecc8_ctrl_bits);
__IO_REG32_BIT(HW_ECC8_CTRL_SET,                 0x80008004,__WRITE      ,__hw_ecc8_ctrl_bits);
__IO_REG32_BIT(HW_ECC8_CTRL_CLR,                 0x80008008,__WRITE      ,__hw_ecc8_ctrl_bits);
__IO_REG32_BIT(HW_ECC8_CTRL_TOG,                 0x8000800C,__WRITE      ,__hw_ecc8_ctrl_bits);
__IO_REG32_BIT(HW_ECC8_STATUS0,                  0x80008010,__READ       ,__hw_ecc8_status0_bits);
__IO_REG32_BIT(HW_ECC8_STATUS1,                  0x80008020,__READ       ,__hw_ecc8_status1_bits);
__IO_REG32_BIT(HW_ECC8_DEBUG0,                   0x80008030,__READ_WRITE ,__hw_ecc8_debug0_bits);
__IO_REG32_BIT(HW_ECC8_DEBUG0_SET,               0x80008034,__WRITE      ,__hw_ecc8_debug0_bits);
__IO_REG32_BIT(HW_ECC8_DEBUG0_CLR,               0x80008038,__WRITE      ,__hw_ecc8_debug0_bits);
__IO_REG32_BIT(HW_ECC8_DEBUG0_TOG,               0x8000803C,__WRITE      ,__hw_ecc8_debug0_bits);
__IO_REG32(    HW_ECC8_DBGKESREAD,               0x80008040,__READ       );
__IO_REG32(    HW_ECC8_DBGCSFEREAD,              0x80008050,__READ       );
__IO_REG32(    HW_ECC8_DBGSYNDGENREAD,           0x80008060,__READ       );
__IO_REG32(    HW_ECC8_DBGAHBMREAD,              0x80008070,__READ       );
__IO_REG32(    HW_ECC8_BLOCKNAME,                0x80008080,__READ       );
__IO_REG32_BIT(HW_ECC8_VERSION,                  0x800080A0,__READ       ,__hw_ecc8_version_bits);

/***************************************************************************
 **
 **  BCH ECC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_BCH_CTRL,                      0x8000A000,__READ_WRITE ,__hw_bch_ctrl_bits);
__IO_REG32_BIT(HW_BCH_CTRL_SET,                  0x8000A004,__WRITE      ,__hw_bch_ctrl_bits);
__IO_REG32_BIT(HW_BCH_CTRL_CLR,                  0x8000A008,__WRITE      ,__hw_bch_ctrl_bits);
__IO_REG32_BIT(HW_BCH_CTRL_TOG,                  0x8000A00C,__WRITE      ,__hw_bch_ctrl_bits);
__IO_REG32_BIT(HW_BCH_STATUS0,                   0x8000A010,__READ       ,__hw_bch_status0_bits);
__IO_REG32_BIT(HW_BCH_MODE,                      0x8000A020,__READ_WRITE ,__hw_bch_mode_bits);
__IO_REG32(    HW_BCH_ENCODEPTR,                 0x8000A030,__READ_WRITE );
__IO_REG32(    HW_BCH_DATAPTR,                   0x8000A040,__READ_WRITE );
__IO_REG32(    HW_BCH_METAPTR,                   0x8000A050,__READ_WRITE );
__IO_REG32_BIT(HW_BCH_LAYOUTSELECT,              0x8000A070,__READ_WRITE ,__hw_bch_layoutselect_bits);
__IO_REG32_BIT(HW_BCH_FLASH0LAYOUT0,             0x8000A080,__READ_WRITE ,__hw_bch_flashxlayout0_bits);
__IO_REG32_BIT(HW_BCH_FLASH0LAYOUT1,             0x8000A090,__READ_WRITE ,__hw_bch_flashxlayout1_bits);
__IO_REG32_BIT(HW_BCH_FLASH1LAYOUT0,             0x8000A0A0,__READ_WRITE ,__hw_bch_flashxlayout0_bits);
__IO_REG32_BIT(HW_BCH_FLASH1LAYOUT1,             0x8000A0B0,__READ_WRITE ,__hw_bch_flashxlayout1_bits);
__IO_REG32_BIT(HW_BCH_FLASH2LAYOUT0,             0x8000A0C0,__READ_WRITE ,__hw_bch_flashxlayout0_bits);
__IO_REG32_BIT(HW_BCH_FLASH2LAYOUT1,             0x8000A0D0,__READ_WRITE ,__hw_bch_flashxlayout1_bits);
__IO_REG32_BIT(HW_BCH_FLASH3LAYOUT0,             0x8000A0E0,__READ_WRITE ,__hw_bch_flashxlayout0_bits);
__IO_REG32_BIT(HW_BCH_FLASH3LAYOUT1,             0x8000A0F0,__READ_WRITE ,__hw_bch_flashxlayout1_bits);
__IO_REG32_BIT(HW_BCH_DEBUG0,                    0x8000A100,__READ_WRITE ,__hw_bch_debug0_bits);
__IO_REG32_BIT(HW_BCH_DEBUG0_SET,                0x8000A104,__WRITE      ,__hw_bch_debug0_bits);
__IO_REG32_BIT(HW_BCH_DEBUG0_CLR,                0x8000A108,__WRITE      ,__hw_bch_debug0_bits);
__IO_REG32_BIT(HW_BCH_DEBUG0_TOG,                0x8000A10C,__WRITE      ,__hw_bch_debug0_bits);
__IO_REG32(    HW_BCH_DBGKESREAD,                0x8000A110,__READ       );
__IO_REG32(    HW_BCH_DBGCSFEREAD,               0x8000A120,__READ       );
__IO_REG32(    HW_BCH_DBGSYNDGENREAD,            0x8000A130,__READ       );
__IO_REG32(    HW_BCH_DBGAHBMREAD,               0x8000A140,__READ       );
__IO_REG32(    HW_BCH_BLOCKNAME,                 0x8000A150,__READ       );
__IO_REG32_BIT(HW_BCH_VERSION,                   0x8000A160,__READ       ,__hw_bch_version_bits);

/***************************************************************************
 **
 **  DCP
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_DCP_CTRL,                      0x80028000,__READ_WRITE ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_SET,                  0x80028004,__WRITE      ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_CLR,                  0x80028008,__WRITE      ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_TOG,                  0x8002800C,__WRITE      ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_STAT,                      0x80028010,__READ_WRITE ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_SET,                  0x80028014,__WRITE      ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_CLR,                  0x80028018,__WRITE      ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_TOG,                  0x8002801C,__WRITE      ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL,               0x80028020,__READ_WRITE ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_SET,           0x80028024,__WRITE      ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_CLR,           0x80028028,__WRITE      ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_TOG,           0x8002802C,__WRITE      ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CAPABILITY0,               0x80028030,__READ_WRITE ,__hw_dcp_capability0_bits);
__IO_REG32_BIT(HW_DCP_CAPABILITY1,               0x80028040,__READ       ,__hw_dcp_capability1_bits);
__IO_REG32(    HW_DCP_CONTEXT,                   0x80028050,__READ_WRITE );
__IO_REG32_BIT(HW_DCP_KEY,                       0x80028060,__READ_WRITE ,__hw_dcp_key_bits);
__IO_REG32(    HW_DCP_KEYDATA,                   0x80028070,__READ_WRITE );
__IO_REG32(    HW_DCP_PACKET0,                   0x80028080,__READ       );
__IO_REG32_BIT(HW_DCP_PACKET1,                   0x80028090,__READ       ,__hw_dcp_packet1_bits);
__IO_REG32_BIT(HW_DCP_PACKET2,                   0x800280A0,__READ       ,__hw_dcp_packet2_bits);
__IO_REG32(    HW_DCP_PACKET3,                   0x800280B0,__READ       );
__IO_REG32(    HW_DCP_PACKET4,                   0x800280C0,__READ       );
__IO_REG32(    HW_DCP_PACKET5,                   0x800280D0,__READ       );
__IO_REG32(    HW_DCP_PACKET6,                   0x800280E0,__READ       );
__IO_REG32(    HW_DCP_CH0CMDPTR,                 0x80028100,__READ_WRITE );
__IO_REG32_BIT(HW_DCP_CH0SEMA,                   0x80028110,__READ_WRITE ,__hw_dcp_chxsema_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT,                   0x80028120,__READ_WRITE ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_SET,               0x80028124,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_CLR,               0x80028128,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_TOG,               0x8002812C,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS,                   0x80028130,__READ_WRITE ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_SET,               0x80028134,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_CLR,               0x80028138,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_TOG,               0x8002813C,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32(    HW_DCP_CH1CMDPTR,                 0x80028140,__READ_WRITE );
__IO_REG32_BIT(HW_DCP_CH1SEMA,                   0x80028150,__READ_WRITE ,__hw_dcp_chxsema_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT,                   0x80028160,__READ_WRITE ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_SET,               0x80028164,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_CLR,               0x80028168,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_TOG,               0x8002816C,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS,                   0x80028170,__READ_WRITE ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_SET,               0x80028174,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_CLR,               0x80028178,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_TOG,               0x8002817C,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32(    HW_DCP_CH2CMDPTR,                 0x80028180,__READ_WRITE );
__IO_REG32_BIT(HW_DCP_CH2SEMA,                   0x80028190,__READ_WRITE ,__hw_dcp_chxsema_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT,                   0x800281A0,__READ_WRITE ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_SET,               0x800281A4,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_CLR,               0x800281A8,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_TOG,               0x800281AC,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS,                   0x800281B0,__READ_WRITE ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_SET,               0x800281B4,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_CLR,               0x800281B8,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_TOG,               0x800281BC,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32(    HW_DCP_CH3CMDPTR,                 0x800281C0,__READ_WRITE );
__IO_REG32_BIT(HW_DCP_CH3SEMA,                   0x800281D0,__READ_WRITE ,__hw_dcp_chxsema_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT,                   0x800281E0,__READ_WRITE ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_SET,               0x800281E4,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_CLR,               0x800281E8,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_TOG,               0x800281EC,__WRITE      ,__hw_dcp_chxstat_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS,                   0x800281F0,__READ_WRITE ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_SET,               0x800281F4,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_CLR,               0x800281F8,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_TOG,               0x800281FC,__WRITE      ,__hw_dcp_chxopts_bits);
__IO_REG32_BIT(HW_DCP_DBGSELECT,                 0x80028400,__READ_WRITE ,__hw_dcp_dbgselect_bits);
__IO_REG32(    HW_DCP_DBGDATA,                   0x80028410,__READ       );
__IO_REG32_BIT(HW_DCP_VERSION,                   0x80028430,__READ       ,__hw_dcp_version_bits);

/***************************************************************************
 **
 **  PXP
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_PXP_CTRL,                      0x8002A000,__READ_WRITE ,__hw_pxp_ctrl_bits);
__IO_REG32_BIT(HW_PXP_CTRL_SET,                  0x8002A004,__WRITE      ,__hw_pxp_ctrl_bits);
__IO_REG32_BIT(HW_PXP_CTRL_CLR,                  0x8002A008,__WRITE      ,__hw_pxp_ctrl_bits);
__IO_REG32_BIT(HW_PXP_CTRL_TOG,                  0x8002A00C,__WRITE      ,__hw_pxp_ctrl_bits);
__IO_REG32_BIT(HW_PXP_STAT,                      0x8002A010,__READ_WRITE ,__hw_pxp_stat_bits);
__IO_REG32_BIT(HW_PXP_STAT_SET,                  0x8002A014,__WRITE      ,__hw_pxp_stat_bits);
__IO_REG32_BIT(HW_PXP_STAT_CLR,                  0x8002A018,__WRITE      ,__hw_pxp_stat_bits);
__IO_REG32_BIT(HW_PXP_STAT_TOG,                  0x8002A01C,__WRITE      ,__hw_pxp_stat_bits);
__IO_REG32(    HW_PXP_RGBBUF,                    0x8002A020,__READ_WRITE );
__IO_REG32(    HW_PXP_RGBBUF2,                   0x8002A030,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_RGBSIZE,                   0x8002A040,__READ_WRITE ,__hw_pxp_rgbsize_bits);
__IO_REG32(    HW_PXP_S0BUF,                     0x8002A050,__READ_WRITE );
__IO_REG32(    HW_PXP_S0UBUF,                    0x8002A060,__READ_WRITE );
__IO_REG32(    HW_PXP_S0VBUF,                    0x8002A070,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_S0PARAM,                   0x8002A080,__READ_WRITE ,__hw_pxp_s0param_bits);
__IO_REG32(    HW_PXP_S0BACKGROUND,              0x8002A090,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_S0CROP ,                   0x8002A0A0,__READ_WRITE ,__hw_pxp_s0crop_bits);
__IO_REG32_BIT(HW_PXP_S0SCALE ,                  0x8002A0B0,__READ_WRITE ,__hw_pxp_s0scale_bits);
__IO_REG32_BIT(HW_PXP_S0OFFSET ,                 0x8002A0C0,__READ_WRITE ,__hw_pxp_s0offset_bits);
__IO_REG32_BIT(HW_PXP_CSCCOEFF0 ,                0x8002A0D0,__READ_WRITE ,__hw_pxp_csccoeff0_bits);
__IO_REG32_BIT(HW_PXP_CSCCOEFF1 ,                0x8002A0E0,__READ_WRITE ,__hw_pxp_csccoeff1_bits);
__IO_REG32_BIT(HW_PXP_CSCCOEFF2 ,                0x8002A0F0,__READ_WRITE ,__hw_pxp_csccoeff2_bits);
__IO_REG32_BIT(HW_PXP_NEXT,                      0x8002A100,__READ_WRITE ,__hw_pxp_next_bits);
__IO_REG32_BIT(HW_PXP_NEXT_SET,                  0x8002A104,__WRITE      ,__hw_pxp_next_bits);
__IO_REG32_BIT(HW_PXP_NEXT_CLR,                  0x8002A108,__WRITE      ,__hw_pxp_next_bits);
__IO_REG32_BIT(HW_PXP_NEXT_TOG,                  0x8002A10C,__WRITE      ,__hw_pxp_next_bits);
__IO_REG32_BIT(HW_PXP_S0COLORKEYLOW,             0x8002A180,__READ_WRITE ,__hw_pxp_s0colorkeylow_bits);
__IO_REG32_BIT(HW_PXP_S0COLORKEYHIGH,            0x8002A190,__READ_WRITE ,__hw_pxp_s0colorkeyhigh_bits);
__IO_REG32_BIT(HW_PXP_OLCOLORKEYLOW,             0x8002A1A0,__READ_WRITE ,__hw_pxp_olcolorkeylow_bits);
__IO_REG32_BIT(HW_PXP_OLCOLORKEYHIGH,            0x8002A1B0,__READ_WRITE ,__hw_pxp_olcolorkeyhigh_bits);
__IO_REG32_BIT(HW_PXP_DEBUGCTRL,                 0x8002A1D0,__READ_WRITE ,__hw_pxp_debugctrl_bits);
__IO_REG32(    HW_PXP_DEBUG,                     0x8002A1E0,__READ       );
__IO_REG32_BIT(HW_PXP_VERSION,                   0x8002A1F0,__READ       ,__hw_pxp_version_bits);
__IO_REG32(    HW_PXP_OL0,                       0x8002A200,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL0SIZE,                   0x8002A210,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL0PARAM,                  0x8002A220,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL1,                       0x8002A240,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL1SIZE,                   0x8002A250,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL1PARAM,                  0x8002A260,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL2,                       0x8002A280,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL2SIZE,                   0x8002A290,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL2PARAM,                  0x8002A2A0,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL3,                       0x8002A2C0,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL3SIZE,                   0x8002A2D0,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL3PARAM,                  0x8002A2E0,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL4,                       0x8002A300,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL4SIZE,                   0x8002A310,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL4PARAM,                  0x8002A320,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL5,                       0x8002A340,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL5SIZE,                   0x8002A350,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL5PARAM,                  0x8002A360,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL6,                       0x8002A380,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL6SIZE,                   0x8002A390,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL6PARAM,                  0x8002A3A0,__READ_WRITE ,__hw_pxp_olxparam_bits);
__IO_REG32(    HW_PXP_OL7,                       0x8002A3C0,__READ_WRITE );
__IO_REG32_BIT(HW_PXP_OL7SIZE,                   0x8002A3D0,__READ_WRITE ,__hw_pxp_olxsize_bits);
__IO_REG32_BIT(HW_PXP_OL7PARAM,                  0x8002A3E0,__READ_WRITE ,__hw_pxp_olxparam_bits);

/***************************************************************************
 **
 **  LCDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_LCDIF_CTRL,                    0x80030000,__READ_WRITE ,__hw_lcdif_ctrl_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL_SET,                0x80030004,__WRITE      ,__hw_lcdif_ctrl_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL_CLR,                0x80030008,__WRITE      ,__hw_lcdif_ctrl_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL_TOG,                0x8003000C,__WRITE      ,__hw_lcdif_ctrl_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL1,                   0x80030010,__READ_WRITE ,__hw_lcdif_ctrl1_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL1_SET,               0x80030014,__WRITE      ,__hw_lcdif_ctrl1_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL1_CLR,               0x80030018,__WRITE      ,__hw_lcdif_ctrl1_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL1_TOG,               0x8003001C,__WRITE      ,__hw_lcdif_ctrl1_bits);
__IO_REG32_BIT(HW_LCDIF_TRANSFER_COUNT,          0x80030020,__READ_WRITE ,__hw_lcdif_transfer_count_bits);
__IO_REG32(    HW_LCDIF_CUR_BUF,                 0x80030030,__READ_WRITE );
__IO_REG32(    HW_LCDIF_NEXT_BUF,                0x80030040,__READ_WRITE );
__IO_REG32_BIT(HW_LCDIF_TIMING,                  0x80030060,__READ_WRITE ,__hw_lcdif_timing_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL0,                 0x80030070,__READ_WRITE ,__hw_lcdif_vdctrl0_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL1,                 0x80030080,__READ_WRITE ,__hw_lcdif_vdctrl1_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL2,                 0x80030090,__READ_WRITE ,__hw_lcdif_vdctrl2_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL3,                 0x800300A0,__READ_WRITE ,__hw_lcdif_vdctrl3_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL4,                 0x800300B0,__READ_WRITE ,__hw_lcdif_vdctrl4_bits);
__IO_REG32_BIT(HW_LCDIF_DVICTRL0,                0x800300C0,__READ_WRITE ,__hw_lcdif_dvictrl0_bits);
__IO_REG32_BIT(HW_LCDIF_DVICTRL1,                0x800300D0,__READ_WRITE ,__hw_lcdif_dvictrl1_bits);
__IO_REG32_BIT(HW_LCDIF_DVICTRL2,                0x800300E0,__READ_WRITE ,__hw_lcdif_dvictrl2_bits);
__IO_REG32_BIT(HW_LCDIF_DVICTRL3,                0x800300F0,__READ_WRITE ,__hw_lcdif_dvictrl3_bits);
__IO_REG32_BIT(HW_LCDIF_DVICTRL4,                0x80030100,__READ_WRITE ,__hw_lcdif_dvictrl4_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_COEFF0,              0x80030110,__READ_WRITE ,__hw_lcdif_csc_coeff0_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_COEFF1,              0x80030120,__READ_WRITE ,__hw_lcdif_csc_coeff1_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_COEFF2,              0x80030130,__READ_WRITE ,__hw_lcdif_csc_coeff2_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_COEFF3,              0x80030140,__READ_WRITE ,__hw_lcdif_csc_coeff3_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_COEFF4,              0x80030150,__READ_WRITE ,__hw_lcdif_csc_coeff4_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_OFFSET,              0x80030160,__READ_WRITE ,__hw_lcdif_csc_offset_bits);
__IO_REG32_BIT(HW_LCDIF_CSC_LIMIT,               0x80030170,__READ_WRITE ,__hw_lcdif_csc_limit_bits);
__IO_REG32_BIT(HW_LCDIF_DATA,                    0x800301B0,__READ_WRITE ,__hw_lcdif_data_bits);
__IO_REG32(    HW_LCDIF_BM_ERROR_STAT,           0x800301C0,__READ_WRITE );
__IO_REG32_BIT(HW_LCDIF_STAT,                    0x800301D0,__READ       ,__hw_lcdif_stat_bits);
__IO_REG32_BIT(HW_LCDIF_VERSION,                 0x800301E0,__READ       ,__hw_lcdif_version_bits);
__IO_REG32_BIT(HW_LCDIF_DEBUG0,                  0x800301F0,__READ       ,__hw_lcdif_debug0_bits);
__IO_REG32_BIT(HW_LCDIF_DEBUG1,                  0x80030200,__READ       ,__hw_lcdif_debug1_bits);

/***************************************************************************
 **
 **  TVENC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_TVENC_CTRL,                    0x80038000,__READ_WRITE ,__hw_tvenc_ctrl_bits);
__IO_REG32_BIT(HW_TVENC_CTRL_SET,                0x80038004,__WRITE      ,__hw_tvenc_ctrl_bits);
__IO_REG32_BIT(HW_TVENC_CTRL_CLR,                0x80038008,__WRITE      ,__hw_tvenc_ctrl_bits);
__IO_REG32_BIT(HW_TVENC_CTRL_TOG,                0x8003800C,__WRITE      ,__hw_tvenc_ctrl_bits);
__IO_REG32_BIT(HW_TVENC_CONFIG,                  0x80038010,__READ_WRITE ,__hw_tvenc_config_bits);
__IO_REG32_BIT(HW_TVENC_CONFIG_SET,              0x80038014,__WRITE      ,__hw_tvenc_config_bits);
__IO_REG32_BIT(HW_TVENC_CONFIG_CLR,              0x80038018,__WRITE      ,__hw_tvenc_config_bits);
__IO_REG32_BIT(HW_TVENC_CONFIG_TOG,              0x8003801C,__WRITE      ,__hw_tvenc_config_bits);
__IO_REG32_BIT(HW_TVENC_FILTCTRL,                0x80038020,__READ_WRITE ,__hw_tvenc_filtctrl_bits);
__IO_REG32_BIT(HW_TVENC_FILTCTRL_SET,            0x80038024,__WRITE      ,__hw_tvenc_filtctrl_bits);
__IO_REG32_BIT(HW_TVENC_FILTCTRL_CLR,            0x80038028,__WRITE      ,__hw_tvenc_filtctrl_bits);
__IO_REG32_BIT(HW_TVENC_FILTCTRL_TOG,            0x8003802C,__WRITE      ,__hw_tvenc_filtctrl_bits);
__IO_REG32_BIT(HW_TVENC_SYNCOFFSET,              0x80038030,__READ_WRITE ,__hw_tvenc_syncoffset_bits);
__IO_REG32_BIT(HW_TVENC_SYNCOFFSET_SET,          0x80038034,__WRITE      ,__hw_tvenc_syncoffset_bits);
__IO_REG32_BIT(HW_TVENC_SYNCOFFSET_CLR,          0x80038038,__WRITE      ,__hw_tvenc_syncoffset_bits);
__IO_REG32_BIT(HW_TVENC_SYNCOFFSET_TOG,          0x8003803C,__WRITE      ,__hw_tvenc_syncoffset_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC0,            0x80038040,__READ_WRITE ,__hw_tvenc_htimingsync0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC0_SET,        0x80038044,__WRITE      ,__hw_tvenc_htimingsync0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC0_CLR,        0x80038048,__WRITE      ,__hw_tvenc_htimingsync0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC0_TOG,        0x8003804C,__WRITE      ,__hw_tvenc_htimingsync0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC1,            0x80038050,__READ_WRITE ,__hw_tvenc_htimingsync1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC1_SET,        0x80038054,__WRITE      ,__hw_tvenc_htimingsync1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC1_CLR,        0x80038058,__WRITE      ,__hw_tvenc_htimingsync1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGSYNC1_TOG,        0x8003805C,__WRITE      ,__hw_tvenc_htimingsync1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGACTIVE,           0x80038060,__READ_WRITE ,__hw_tvenc_htimingactive_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGACTIVE_SET,       0x80038064,__WRITE      ,__hw_tvenc_htimingactive_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGACTIVE_CLR,       0x80038068,__WRITE      ,__hw_tvenc_htimingactive_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGACTIVE_TOG,       0x8003806C,__WRITE      ,__hw_tvenc_htimingactive_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST0,           0x80038070,__READ_WRITE ,__hw_tvenc_htimingburst0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST0_SET,       0x80038074,__WRITE      ,__hw_tvenc_htimingburst0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST0_CLR,       0x80038078,__WRITE      ,__hw_tvenc_htimingburst0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST0_TOG,       0x8003807C,__WRITE      ,__hw_tvenc_htimingburst0_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST1,           0x80038080,__READ_WRITE ,__hw_tvenc_htimingburst1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST1_SET,       0x80038084,__WRITE      ,__hw_tvenc_htimingburst1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST1_CLR,       0x80038088,__WRITE      ,__hw_tvenc_htimingburst1_bits);
__IO_REG32_BIT(HW_TVENC_HTIMINGBURST1_TOG,       0x8003808C,__WRITE      ,__hw_tvenc_htimingburst1_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING0,                0x80038090,__READ_WRITE ,__hw_tvenc_vtiming0_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING0_SET,            0x80038094,__WRITE      ,__hw_tvenc_vtiming0_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING0_CLR,            0x80038098,__WRITE      ,__hw_tvenc_vtiming0_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING0_TOG,            0x8003809C,__WRITE      ,__hw_tvenc_vtiming0_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING1,                0x800380A0,__READ_WRITE ,__hw_tvenc_vtiming1_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING1_SET,            0x800380A4,__WRITE      ,__hw_tvenc_vtiming1_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING1_CLR,            0x800380A8,__WRITE      ,__hw_tvenc_vtiming1_bits);
__IO_REG32_BIT(HW_TVENC_VTIMING1_TOG,            0x800380AC,__WRITE      ,__hw_tvenc_vtiming1_bits);
__IO_REG32_BIT(HW_TVENC_MISC,                    0x800380B0,__READ_WRITE ,__hw_tvenc_misc_bits);
__IO_REG32_BIT(HW_TVENC_MISC_SET,                0x800380B4,__WRITE      ,__hw_tvenc_misc_bits);
__IO_REG32_BIT(HW_TVENC_MISC_CLR,                0x800380B8,__WRITE      ,__hw_tvenc_misc_bits);
__IO_REG32_BIT(HW_TVENC_MISC_TOG,                0x800380BC,__WRITE      ,__hw_tvenc_misc_bits);
__IO_REG32(    HW_TVENC_COLORSUB0,               0x800380C0,__READ_WRITE );
__IO_REG32(    HW_TVENC_COLORSUB0_SET,           0x800380C4,__WRITE      );
__IO_REG32(    HW_TVENC_COLORSUB0_CLR,           0x800380C8,__WRITE      );
__IO_REG32(    HW_TVENC_COLORSUB0_TOG,           0x800380CC,__WRITE      );
__IO_REG32(    HW_TVENC_COLORSUB1,               0x800380D0,__READ_WRITE );
__IO_REG32(    HW_TVENC_COLORSUB1_SET,           0x800380D4,__WRITE      );
__IO_REG32(    HW_TVENC_COLORSUB1_CLR,           0x800380D8,__WRITE      );
__IO_REG32(    HW_TVENC_COLORSUB1_TOG,           0x800380DC,__WRITE      );
__IO_REG32_BIT(HW_TVENC_COPYPROTECT,             0x800380E0,__READ_WRITE ,__hw_tvenc_copyprotect_bits);
__IO_REG32_BIT(HW_TVENC_COPYPROTECT_SET,         0x800380E4,__WRITE      ,__hw_tvenc_copyprotect_bits);
__IO_REG32_BIT(HW_TVENC_COPYPROTECT_CLR,         0x800380E8,__WRITE      ,__hw_tvenc_copyprotect_bits);
__IO_REG32_BIT(HW_TVENC_COPYPROTECT_TOG,         0x800380EC,__WRITE      ,__hw_tvenc_copyprotect_bits);
__IO_REG32_BIT(HW_TVENC_CLOSEDCAPTION,           0x800380F0,__READ_WRITE ,__hw_tvenc_closedcaption_bits);
__IO_REG32_BIT(HW_TVENC_CLOSEDCAPTION_SET,       0x800380F4,__WRITE      ,__hw_tvenc_closedcaption_bits);
__IO_REG32_BIT(HW_TVENC_CLOSEDCAPTION_CLR,       0x800380F8,__WRITE      ,__hw_tvenc_closedcaption_bits);
__IO_REG32_BIT(HW_TVENC_CLOSEDCAPTION_TOG,       0x800380FC,__WRITE      ,__hw_tvenc_closedcaption_bits);
__IO_REG32_BIT(HW_TVENC_COLORBURST,              0x80038140,__READ_WRITE ,__hw_tvenc_colorburst_bits);
__IO_REG32_BIT(HW_TVENC_COLORBURST_SET,          0x80038144,__WRITE      ,__hw_tvenc_colorburst_bits);
__IO_REG32_BIT(HW_TVENC_COLORBURST_CLR,          0x80038148,__WRITE      ,__hw_tvenc_colorburst_bits);
__IO_REG32_BIT(HW_TVENC_COLORBURST_TOG,          0x8003814C,__WRITE      ,__hw_tvenc_colorburst_bits);
__IO_REG32_BIT(HW_TVENC_DACCTRL,                 0x800381A0,__READ_WRITE ,__hw_tvenc_dacctrl_bits);
__IO_REG32_BIT(HW_TVENC_DACCTRL_SET,             0x800381A4,__WRITE      ,__hw_tvenc_dacctrl_bits);
__IO_REG32_BIT(HW_TVENC_DACCTRL_CLR,             0x800381A8,__WRITE      ,__hw_tvenc_dacctrl_bits);
__IO_REG32_BIT(HW_TVENC_DACCTRL_TOG,             0x800381AC,__WRITE      ,__hw_tvenc_dacctrl_bits);
__IO_REG32_BIT(HW_TVENC_DACSTATUS,               0x800381B0,__READ_WRITE ,__hw_tvenc_dacstatus_bits);
__IO_REG32_BIT(HW_TVENC_DACSTATUS_SET,           0x800381B4,__WRITE      ,__hw_tvenc_dacstatus_bits);
__IO_REG32_BIT(HW_TVENC_DACSTATUS_CLR,           0x800381B8,__WRITE      ,__hw_tvenc_dacstatus_bits);
__IO_REG32_BIT(HW_TVENC_DACSTATUS_TOG,           0x800381BC,__WRITE      ,__hw_tvenc_dacstatus_bits);
__IO_REG32_BIT(HW_TVENC_VDACTEST,                0x800381C0,__READ_WRITE ,__hw_tvenc_vdactest_bits);
__IO_REG32_BIT(HW_TVENC_VDACTEST_SET,            0x800381C4,__WRITE      ,__hw_tvenc_vdactest_bits);
__IO_REG32_BIT(HW_TVENC_VDACTEST_CLR,            0x800381C8,__WRITE      ,__hw_tvenc_vdactest_bits);
__IO_REG32_BIT(HW_TVENC_VDACTEST_TOG,            0x800381CC,__WRITE      ,__hw_tvenc_vdactest_bits);
__IO_REG32_BIT(HW_TVENC_VERSION,                 0x800381D0,__READ       ,__hw_tvenc_version_bits);

/***************************************************************************
 **
 **  SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP1_CTRL0,                   0x80010000,__READ_WRITE ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_SET,               0x80010004,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_CLR,               0x80010008,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_TOG,               0x8001000C,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0,                    0x80010010,__READ_WRITE ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_SET,                0x80010014,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_CLR,                0x80010018,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_TOG,                0x8001001C,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP1_CMD1,                    0x80010020,__READ_WRITE );
__IO_REG32(    HW_SSP1_COMPREF,                 0x80010030,__READ_WRITE );
__IO_REG32(    HW_SSP1_COMPMASK,                0x80010040,__READ_WRITE );
__IO_REG32_BIT(HW_SSP1_TIMING,                  0x80010050,__READ_WRITE ,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1,                   0x80010060,__READ_WRITE ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_SET,               0x80010064,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_CLR,               0x80010068,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_TOG,               0x8001006C,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP1_DATA,                    0x80010070,__READ_WRITE );
__IO_REG32(    HW_SSP1_SDRESP0,                 0x80010080,__READ       );
__IO_REG32(    HW_SSP1_SDRESP1,                 0x80010090,__READ       );
__IO_REG32(    HW_SSP1_SDRESP2,                 0x800100A0,__READ       );
__IO_REG32(    HW_SSP1_SDRESP3,                 0x800100B0,__READ       );
__IO_REG32_BIT(HW_SSP1_STATUS,                  0x800100C0,__READ       ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP1_DEBUG,                   0x80010100,__READ       ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP1_VERSION,                 0x80010110,__READ       ,__hw_ssp_version_bits);

/***************************************************************************
 **
 **  SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP2_CTRL0,                   0x80034000,__READ_WRITE ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_SET,               0x80034004,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_CLR,               0x80034008,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_TOG,               0x8003400C,__WRITE      ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0,                    0x80034010,__READ_WRITE ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_SET,                0x80034014,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_CLR,                0x80034018,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_TOG,                0x8003401C,__WRITE      ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP2_CMD1,                    0x80034020,__READ_WRITE );
__IO_REG32(    HW_SSP2_COMPREF,                 0x80034030,__READ_WRITE );
__IO_REG32(    HW_SSP2_COMPMASK,                0x80034040,__READ_WRITE );
__IO_REG32_BIT(HW_SSP2_TIMING,                  0x80034050,__READ_WRITE ,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1,                   0x80034060,__READ_WRITE ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_SET,               0x80034064,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_CLR,               0x80034068,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_TOG,               0x8003406C,__WRITE      ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP2_DATA,                    0x80034070,__READ_WRITE );
__IO_REG32(    HW_SSP2_SDRESP0,                 0x80034080,__READ       );
__IO_REG32(    HW_SSP2_SDRESP1,                 0x80034090,__READ       );
__IO_REG32(    HW_SSP2_SDRESP2,                 0x800340A0,__READ       );
__IO_REG32(    HW_SSP2_SDRESP3,                 0x800340B0,__READ       );
__IO_REG32_BIT(HW_SSP2_STATUS,                  0x800340C0,__READ       ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP2_DEBUG,                   0x80034100,__READ       ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP2_VERSION,                 0x80034110,__READ       ,__hw_ssp_version_bits);

/***************************************************************************
 **
 **  TIMROT
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_TIMROT_ROTCTRL,               0x80068000,__READ_WRITE ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_SET,           0x80068004,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_CLR,           0x80068008,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_TOG,           0x8006800C,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCOUNT,              0x80068010,__READ       ,__hw_timrot_rotcount_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0,              0x80068020,__READ_WRITE ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_SET,          0x80068024,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_CLR,          0x80068028,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_TOG,          0x8006802C,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCOUNT0,             0x80068030,__READ_WRITE ,__hw_timrot_timcountx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1,              0x80068040,__READ_WRITE ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_SET,          0x80068044,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_CLR,          0x80068048,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_TOG,          0x8006804C,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCOUNT1,             0x80068050,__READ_WRITE ,__hw_timrot_timcountx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2,              0x80068060,__READ_WRITE ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_SET,          0x80068064,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_CLR,          0x80068068,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_TOG,          0x8006806C,__WRITE      ,__hw_timrot_timctrlx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCOUNT2,             0x80068070,__READ_WRITE ,__hw_timrot_timcountx_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3,              0x80068080,__READ_WRITE ,__hw_timrot_timctrl3_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_SET,          0x80068084,__WRITE      ,__hw_timrot_timctrl3_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_CLR,          0x80068088,__WRITE      ,__hw_timrot_timctrl3_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_TOG,          0x8006808C,__WRITE      ,__hw_timrot_timctrl3_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCOUNT3,             0x80068090,__READ_WRITE ,__hw_timrot_timcount3_bits);
__IO_REG32_BIT(HW_TIMROT_VERSION,               0x800680A0,__READ       ,__hw_timrot_version_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_RTC_CTRL,                     0x8005C000,__READ_WRITE ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_SET,                 0x8005C004,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_CLR,                 0x8005C008,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_TOG,                 0x8005C00C,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_STAT,                     0x8005C010,__READ       ,__hw_rtc_stat_bits);
__IO_REG32(    HW_RTC_MILLISECONDS,             0x8005C020,__READ_WRITE );
__IO_REG32(    HW_RTC_MILLISECONDS_SET,         0x8005C024,__WRITE      );
__IO_REG32(    HW_RTC_MILLISECONDS_CLR,         0x8005C028,__WRITE      );
__IO_REG32(    HW_RTC_MILLISECONDS_TOG,         0x8005C02C,__WRITE      );
__IO_REG32(    HW_RTC_SECONDS,                  0x8005C030,__READ_WRITE );
__IO_REG32(    HW_RTC_SECONDS_SET,              0x8005C034,__WRITE      );
__IO_REG32(    HW_RTC_SECONDS_CLR,              0x8005C038,__WRITE      );
__IO_REG32(    HW_RTC_SECONDS_TOG,              0x8005C03C,__WRITE      );
__IO_REG32(    HW_RTC_ALARM,                    0x8005C040,__READ_WRITE );
__IO_REG32(    HW_RTC_ALARM_SET,                0x8005C044,__WRITE      );
__IO_REG32(    HW_RTC_ALARM_CLR,                0x8005C048,__WRITE      );
__IO_REG32(    HW_RTC_ALARM_TOG,                0x8005C04C,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG,                 0x8005C050,__READ_WRITE );
__IO_REG32(    HW_RTC_WATCHDOG_SET,             0x8005C054,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG_CLR,             0x8005C058,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG_TOG,             0x8005C05C,__WRITE      );
__IO_REG32_BIT(HW_RTC_PERSISTENT0,              0x8005C060,__READ_WRITE ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_SET,          0x8005C064,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_CLR,          0x8005C068,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_TOG,          0x8005C06C,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32(    HW_RTC_PERSISTENT1,              0x8005C070,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT1_SET,          0x8005C074,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT1_CLR,          0x8005C078,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT1_TOG,          0x8005C07C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2,              0x8005C080,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT2_SET,          0x8005C084,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2_CLR,          0x8005C088,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2_TOG,          0x8005C08C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3,              0x8005C090,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT3_SET,          0x8005C094,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3_CLR,          0x8005C098,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3_TOG,          0x8005C09C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4,              0x8005C0A0,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT4_SET,          0x8005C0A4,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4_CLR,          0x8005C0A8,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4_TOG,          0x8005C0AC,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5,              0x8005C0B0,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT5_SET,          0x8005C0B4,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5_CLR,          0x8005C0B8,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5_TOG,          0x8005C0BC,__WRITE      );
__IO_REG32_BIT(HW_RTC_DEBUG,                    0x8005C0C0,__READ_WRITE ,__hw_rtc_debug_bits);
__IO_REG32_BIT(HW_RTC_DEBUG_SET,                0x8005C0C4,__WRITE      ,__hw_rtc_debug_bits);
__IO_REG32_BIT(HW_RTC_DEBUG_CLR,                0x8005C0C8,__WRITE      ,__hw_rtc_debug_bits);
__IO_REG32_BIT(HW_RTC_DEBUG_TOG,                0x8005C0CC,__WRITE      ,__hw_rtc_debug_bits);
__IO_REG32_BIT(HW_RTC_VERSION,                  0x8005C0D0,__READ       ,__hw_rtc_version_bits);

/***************************************************************************
 **
 **  PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_PWM_CTRL,                     0x80064000,__READ_WRITE ,__hw_pwm_ctrl_bits);
__IO_REG32_BIT(HW_PWM_CTRL_SET,                 0x80064004,__WRITE      ,__hw_pwm_ctrl_bits);
__IO_REG32_BIT(HW_PWM_CTRL_CLR,                 0x80064008,__WRITE      ,__hw_pwm_ctrl_bits);
__IO_REG32_BIT(HW_PWM_CTRL_TOG,                 0x8006400C,__WRITE      ,__hw_pwm_ctrl_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE0,                  0x80064010,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE0_SET,              0x80064014,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE0_CLR,              0x80064018,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE0_TOG,              0x8006401C,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD0,                  0x80064020,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD0_SET,              0x80064024,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD0_CLR,              0x80064028,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD0_TOG,              0x8006402C,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE1,                  0x80064030,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE1_SET,              0x80064034,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE1_CLR,              0x80064038,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE1_TOG,              0x8006403C,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD1,                  0x80064040,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD1_SET,              0x80064044,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD1_CLR,              0x80064048,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD1_TOG,              0x8006404C,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE2,                  0x80064050,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE2_SET,              0x80064054,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE2_CLR,              0x80064058,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE2_TOG,              0x8006405C,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD2,                  0x80064060,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD2_SET,              0x80064064,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD2_CLR,              0x80064068,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD2_TOG,              0x8006406C,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE3,                  0x80064070,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE3_SET,              0x80064074,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE3_CLR,              0x80064078,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE3_TOG,              0x8006407C,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD3,                  0x80064080,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD3_SET,              0x80064084,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD3_CLR,              0x80064088,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD3_TOG,              0x8006408C,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE4,                  0x80064090,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE4_SET,              0x80064094,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE4_CLR,              0x80064098,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE4_TOG,              0x8006409C,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD4,                  0x800640A0,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD4_SET,              0x800640A4,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD4_CLR,              0x800640A8,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD4_TOG,              0x800640AC,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_VERSION,                  0x800640B0,__READ       ,__hw_pwm_version_bits);

/***************************************************************************
 **
 **  I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_I2C_CTRL0,                    0x80058000,__READ_WRITE ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C_CTRL0_SET,                0x80058004,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C_CTRL0_CLR,                0x80058008,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C_CTRL0_TOG,                0x8005800C,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C_TIMING0,                  0x80058010,__READ_WRITE ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C_TIMING0_SET,              0x80058014,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C_TIMING0_CLR,              0x80058018,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C_TIMING0_TOG,              0x8005801C,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C_TIMING1,                  0x80058020,__READ_WRITE ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C_TIMING1_SET,              0x80058024,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C_TIMING1_CLR,              0x80058028,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C_TIMING1_TOG,              0x8005802C,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C_TIMING2,                  0x80058030,__READ_WRITE ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C_TIMING2_SET,              0x80058034,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C_TIMING2_CLR,              0x80058038,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C_TIMING2_TOG,              0x8005803C,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C_CTRL1,                    0x80058040,__READ_WRITE ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C_CTRL1_SET,                0x80058044,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C_CTRL1_CLR,                0x80058048,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C_CTRL1_TOG,                0x8005804C,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C_STAT,                     0x80058050,__READ       ,__hw_i2c_stat_bits);
__IO_REG32(    HW_I2C_DATA,                     0x80058060,__READ_WRITE );
__IO_REG32_BIT(HW_I2C_DEBUG0,                   0x80058070,__READ_WRITE ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C_DEBUG0_SET,               0x80058074,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C_DEBUG0_CLR,               0x80058078,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C_DEBUG0_TOG,               0x8005807C,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C_DEBUG1,                   0x80058080,__READ_WRITE ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C_DEBUG1_SET,               0x80058084,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C_DEBUG1_CLR,               0x80058088,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C_DEBUG1_TOG,               0x8005808C,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C_VERSION,                  0x80058090,__READ       ,__hw_i2c_version_bits);

/***************************************************************************
 **
 **  APPUART1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTAPP1_CTRL0,               0x8006C000,__READ_WRITE ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL0_SET,           0x8006C004,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL0_CLR,           0x8006C008,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL0_TOG,           0x8006C00C,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL1,               0x8006C010,__READ_WRITE ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL1_SET,           0x8006C014,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL1_CLR,           0x8006C018,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL1_TOG,           0x8006C01C,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL2,               0x8006C020,__READ_WRITE ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL2_SET,           0x8006C024,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL2_CLR,           0x8006C028,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_CTRL2_TOG,           0x8006C02C,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL,            0x8006C030,__READ_WRITE ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL_SET,        0x8006C034,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL_CLR,        0x8006C038,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL_TOG,        0x8006C03C,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL2,           0x8006C040,__READ_WRITE ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL2_SET,       0x8006C044,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL2_CLR,       0x8006C048,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_LINECTRL2_TOG,       0x8006C04C,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP1_INTR,                0x8006C050,__READ_WRITE ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP1_INTR_SET,            0x8006C054,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP1_INTR_CLR,            0x8006C058,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP1_INTR_TOG,            0x8006C05C,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32(    HW_UARTAPP1_DATA,                0x8006C060,__READ_WRITE );
__IO_REG32_BIT(HW_UARTAPP1_STAT,                0x8006C070,__READ_WRITE ,__hw_uartapp_stat_bits);
__IO_REG32_BIT(HW_UARTAPP1_DEBUG,               0x8006C080,__READ       ,__hw_uartapp_debug_bits);
__IO_REG32_BIT(HW_UARTAPP1_VERSION,             0x8006C090,__READ       ,__hw_uartapp_version_bits);
__IO_REG32_BIT(HW_UARTAPP1_AUTOBAUD,            0x8006C0A0,__READ_WRITE ,__hw_uartapp_autobaud_bits);

/***************************************************************************
 **
 **  APPUART2
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTAPP2_CTRL0,               0x8006E000,__READ_WRITE ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL0_SET,           0x8006E004,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL0_CLR,           0x8006E008,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL0_TOG,           0x8006E00C,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL1,               0x8006E010,__READ_WRITE ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL1_SET,           0x8006E014,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL1_CLR,           0x8006E018,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL1_TOG,           0x8006E01C,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL2,               0x8006E020,__READ_WRITE ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL2_SET,           0x8006E024,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL2_CLR,           0x8006E028,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_CTRL2_TOG,           0x8006E02C,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL,            0x8006E030,__READ_WRITE ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL_SET,        0x8006E034,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL_CLR,        0x8006E038,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL_TOG,        0x8006E03C,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL2,           0x8006E040,__READ_WRITE ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL2_SET,       0x8006E044,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL2_CLR,       0x8006E048,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_LINECTRL2_TOG,       0x8006E04C,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP2_INTR,                0x8006E050,__READ_WRITE ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP2_INTR_SET,            0x8006E054,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP2_INTR_CLR,            0x8006E058,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP2_INTR_TOG,            0x8006E05C,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32(    HW_UARTAPP2_DATA,                0x8006E060,__READ_WRITE );
__IO_REG32_BIT(HW_UARTAPP2_STAT,                0x8006E070,__READ_WRITE ,__hw_uartapp_stat_bits);
__IO_REG32_BIT(HW_UARTAPP2_DEBUG,               0x8006E080,__READ       ,__hw_uartapp_debug_bits);
__IO_REG32_BIT(HW_UARTAPP2_VERSION,             0x8006E090,__READ       ,__hw_uartapp_version_bits);
__IO_REG32_BIT(HW_UARTAPP2_AUTOBAUD,            0x8006E0A0,__READ_WRITE ,__hw_uartapp_autobaud_bits);

/***************************************************************************
 **
 **  DBGUART
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTDBGDR,                    0x80070000,__READ_WRITE ,__hw_uartdbgdr_bits);
__IO_REG32_BIT(HW_UARTDBGRSR_ECR,               0x80070004,__READ_WRITE ,__hw_uartdbgrsr_ecr_bits);
__IO_REG32_BIT(HW_UARTDBGFR,                    0x80070018,__READ       ,__hw_uartdbgfr_bits);
__IO_REG32_BIT(HW_UARTDBGILPR,                  0x80070020,__READ_WRITE ,__hw_uartdbgilpr_bits);
__IO_REG32_BIT(HW_UARTDBGIBRD,                  0x80070024,__READ_WRITE ,__hw_uartdbgibrd_bits);
__IO_REG32_BIT(HW_UARTDBGFBRD,                  0x80070028,__READ_WRITE ,__hw_uartdbgfbrd_bits);
__IO_REG32_BIT(HW_UARTDBGLCR_H,                 0x8007002C,__READ_WRITE ,__hw_uartdbglcr_h_bits);
__IO_REG32_BIT(HW_UARTDBGCR,                    0x80070030,__READ_WRITE ,__hw_uartdbgcr_bits);
__IO_REG32_BIT(HW_UARTDBGIFLS,                  0x80070034,__READ_WRITE ,__hw_uartdbgifls_bits);
__IO_REG32_BIT(HW_UARTDBGIMSC,                  0x80070038,__READ_WRITE ,__hw_uartdbgimsc_bits);
__IO_REG32_BIT(HW_UARTDBGRIS,                   0x8007003C,__READ       ,__hw_uartdbgris_bits);
__IO_REG32_BIT(HW_UARTDBGMIS,                   0x80070040,__READ       ,__hw_uartdbgmis_bits);
__IO_REG32_BIT(HW_UARTDBGICR,                   0x80070044,__WRITE      ,__hw_uartdbgicr_bits);

/***************************************************************************
 **
 **  AUDIOIN
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_AUDIOIN_CTRL,                 0x8004C000,__READ_WRITE ,__hw_audioin_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_CTRL_SET,             0x8004C004,__WRITE      ,__hw_audioin_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_CTRL_CLR,             0x8004C008,__WRITE      ,__hw_audioin_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_CTRL_TOG,             0x8004C00C,__WRITE      ,__hw_audioin_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_STAT,                 0x8004C010,__READ       ,__hw_audioin_stat_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCSRR,               0x8004C020,__READ_WRITE ,__hw_audioin_adcsrr_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCSRR_SET,           0x8004C024,__WRITE      ,__hw_audioin_adcsrr_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCSRR_CLR,           0x8004C028,__WRITE      ,__hw_audioin_adcsrr_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCSRR_TOG,           0x8004C02C,__WRITE      ,__hw_audioin_adcsrr_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOLUME,            0x8004C030,__READ_WRITE ,__hw_audioin_adcvolume_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOLUME_SET,        0x8004C034,__WRITE      ,__hw_audioin_adcvolume_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOLUME_CLR,        0x8004C038,__WRITE      ,__hw_audioin_adcvolume_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOLUME_TOG,        0x8004C03C,__WRITE      ,__hw_audioin_adcvolume_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCDEBUG,             0x8004C040,__READ_WRITE ,__hw_audioin_adcdebug_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCDEBUG_SET,         0x8004C044,__WRITE      ,__hw_audioin_adcdebug_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCDEBUG_CLR,         0x8004C048,__WRITE      ,__hw_audioin_adcdebug_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCDEBUG_TOG,         0x8004C04C,__WRITE      ,__hw_audioin_adcdebug_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOL,               0x8004C050,__READ_WRITE ,__hw_audioin_adcvol_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOL_SET,           0x8004C054,__WRITE      ,__hw_audioin_adcvol_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOL_CLR,           0x8004C058,__WRITE      ,__hw_audioin_adcvol_bits);
__IO_REG32_BIT(HW_AUDIOIN_ADCVOL_TOG,           0x8004C05C,__WRITE      ,__hw_audioin_adcvol_bits);
__IO_REG32_BIT(HW_AUDIOIN_MICLINE,              0x8004C060,__READ_WRITE ,__hw_audioin_micline_bits);
__IO_REG32_BIT(HW_AUDIOIN_MICLINE_SET,          0x8004C064,__WRITE      ,__hw_audioin_micline_bits);
__IO_REG32_BIT(HW_AUDIOIN_MICLINE_CLR,          0x8004C068,__WRITE      ,__hw_audioin_micline_bits);
__IO_REG32_BIT(HW_AUDIOIN_MICLINE_TOG,          0x8004C06C,__WRITE      ,__hw_audioin_micline_bits);
__IO_REG32_BIT(HW_AUDIOIN_ANACLKCTRL,           0x8004C070,__READ_WRITE ,__hw_audioin_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_ANACLKCTRL_SET,       0x8004C074,__WRITE      ,__hw_audioin_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_ANACLKCTRL_CLR,       0x8004C078,__WRITE      ,__hw_audioin_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_ANACLKCTRL_TOG,       0x8004C07C,__WRITE      ,__hw_audioin_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOIN_DATA,                 0x8004C080,__READ       ,__hw_audioin_data_bits);

/***************************************************************************
 **
 **  AUDIOOUT
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_AUDIOOUT_CTRL,                0x80048000,__READ_WRITE ,__hw_audioout_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_CTRL_SET,            0x80048004,__WRITE      ,__hw_audioout_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_CTRL_CLR,            0x80048008,__WRITE      ,__hw_audioout_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_CTRL_TOG,            0x8004800C,__WRITE      ,__hw_audioout_ctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_STAT,                0x80048010,__READ       ,__hw_audioout_stat_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACSRR,              0x80048020,__READ_WRITE ,__hw_audioout_dacsrr_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACSRR_SET,          0x80048024,__WRITE      ,__hw_audioout_dacsrr_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACSRR_CLR,          0x80048028,__WRITE      ,__hw_audioout_dacsrr_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACSRR_TOG,          0x8004802C,__WRITE      ,__hw_audioout_dacsrr_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACVOLUME,           0x80048030,__READ_WRITE ,__hw_audioout_dacvolume_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACVOLUME_SET,       0x80048034,__WRITE      ,__hw_audioout_dacvolume_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACVOLUME_CLR,       0x80048038,__WRITE      ,__hw_audioout_dacvolume_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACVOLUME_TOG,       0x8004803C,__WRITE      ,__hw_audioout_dacvolume_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACDEBUG,            0x80048040,__READ_WRITE ,__hw_audioout_dacdebug_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACDEBUG_SET,        0x80048044,__WRITE      ,__hw_audioout_dacdebug_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACDEBUG_CLR,        0x80048048,__WRITE      ,__hw_audioout_dacdebug_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DACDEBUG_TOG,        0x8004804C,__WRITE      ,__hw_audioout_dacdebug_bits);
__IO_REG32_BIT(HW_AUDIOOUT_HPVOL,               0x80048050,__READ_WRITE ,__hw_audioout_hpvol_bits);
__IO_REG32_BIT(HW_AUDIOOUT_HPVOL_SET,           0x80048054,__WRITE      ,__hw_audioout_hpvol_bits);
__IO_REG32_BIT(HW_AUDIOOUT_HPVOL_CLR,           0x80048058,__WRITE      ,__hw_audioout_hpvol_bits);
__IO_REG32_BIT(HW_AUDIOOUT_HPVOL_TOG,           0x8004805C,__WRITE      ,__hw_audioout_hpvol_bits);
__IO_REG32_BIT(HW_AUDIOOUT_PWRDN,               0x80048070,__READ_WRITE ,__hw_audioout_pwrdn_bits);
__IO_REG32_BIT(HW_AUDIOOUT_PWRDN_SET,           0x80048074,__WRITE      ,__hw_audioout_pwrdn_bits);
__IO_REG32_BIT(HW_AUDIOOUT_PWRDN_CLR,           0x80048078,__WRITE      ,__hw_audioout_pwrdn_bits);
__IO_REG32_BIT(HW_AUDIOOUT_PWRDN_TOG,           0x8004807C,__WRITE      ,__hw_audioout_pwrdn_bits);
__IO_REG32_BIT(HW_AUDIOOUT_REFCTRL,             0x80048080,__READ_WRITE ,__hw_audioout_refctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_REFCTRL_SET,         0x80048084,__WRITE      ,__hw_audioout_refctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_REFCTRL_CLR,         0x80048088,__WRITE      ,__hw_audioout_refctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_REFCTRL_TOG,         0x8004808C,__WRITE      ,__hw_audioout_refctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACTRL,             0x80048090,__READ_WRITE ,__hw_audioout_anactrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACTRL_SET,         0x80048094,__WRITE      ,__hw_audioout_anactrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACTRL_CLR,         0x80048098,__WRITE      ,__hw_audioout_anactrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACTRL_TOG,         0x8004809C,__WRITE      ,__hw_audioout_anactrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_TEST,                0x800480A0,__READ_WRITE ,__hw_audioout_test_bits);
__IO_REG32_BIT(HW_AUDIOOUT_TEST_SET,            0x800480A4,__WRITE      ,__hw_audioout_test_bits);
__IO_REG32_BIT(HW_AUDIOOUT_TEST_CLR,            0x800480A8,__WRITE      ,__hw_audioout_test_bits);
__IO_REG32_BIT(HW_AUDIOOUT_TEST_TOG,            0x800480AC,__WRITE      ,__hw_audioout_test_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTCTRL,            0x800480B0,__READ_WRITE ,__hw_audioout_bistctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTCTRL_SET,        0x800480B4,__WRITE      ,__hw_audioout_bistctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTCTRL_CLR,        0x800480B8,__WRITE      ,__hw_audioout_bistctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTCTRL_TOG,        0x800480BC,__WRITE      ,__hw_audioout_bistctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTSTAT0,           0x800480C0,__READ       ,__hw_audioout_biststat0_bits);
__IO_REG32_BIT(HW_AUDIOOUT_BISTSTAT1,           0x800480D0,__READ       ,__hw_audioout_biststat1_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACLKCTRL,          0x800480E0,__READ_WRITE ,__hw_audioout_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACLKCTRL_SET,      0x800480E4,__WRITE      ,__hw_audioout_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACLKCTRL_CLR,      0x800480E8,__WRITE      ,__hw_audioout_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_ANACLKCTRL_TOG,      0x800480EC,__WRITE      ,__hw_audioout_anaclkctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DATA,                0x800480F0,__READ_WRITE ,__hw_audioout_data_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DATA_SET,            0x800480F4,__WRITE      ,__hw_audioout_data_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DATA_CLR,            0x800480F8,__WRITE      ,__hw_audioout_data_bits);
__IO_REG32_BIT(HW_AUDIOOUT_DATA_TOG,            0x800480FC,__WRITE      ,__hw_audioout_data_bits);
__IO_REG32_BIT(HW_AUDIOOUT_SPEAKERCTRL,         0x80048100,__READ_WRITE ,__hw_audioout_speakerctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_SPEAKERCTRL_SET,     0x80048104,__WRITE      ,__hw_audioout_speakerctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_SPEAKERCTRL_CLR,     0x80048108,__WRITE      ,__hw_audioout_speakerctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_SPEAKERCTRL_TOG,     0x8004810C,__WRITE      ,__hw_audioout_speakerctrl_bits);
__IO_REG32_BIT(HW_AUDIOOUT_VERSION,             0x80048200,__READ       ,__hw_audioout_version_bits);

/***************************************************************************
 **
 **  SPDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SPDIF_CTRL,                   0x80054000,__READ_WRITE ,__hw_spdif_ctrl_bits);
__IO_REG32_BIT(HW_SPDIF_CTRL_SET,               0x80054004,__WRITE      ,__hw_spdif_ctrl_bits);
__IO_REG32_BIT(HW_SPDIF_CTRL_CLR,               0x80054008,__WRITE      ,__hw_spdif_ctrl_bits);
__IO_REG32_BIT(HW_SPDIF_CTRL_TOG,               0x8005400C,__WRITE      ,__hw_spdif_ctrl_bits);
__IO_REG32_BIT(HW_SPDIF_STAT,                   0x80054010,__READ       ,__hw_spdif_stat_bits);
__IO_REG32_BIT(HW_SPDIF_FRAMECTRL,              0x80054020,__READ_WRITE ,__hw_spdif_framectrl_bits);
__IO_REG32_BIT(HW_SPDIF_FRAMECTRL_SET,          0x80054024,__WRITE      ,__hw_spdif_framectrl_bits);
__IO_REG32_BIT(HW_SPDIF_FRAMECTRL_CLR,          0x80054028,__WRITE      ,__hw_spdif_framectrl_bits);
__IO_REG32_BIT(HW_SPDIF_FRAMECTRL_TOG,          0x8005402C,__WRITE      ,__hw_spdif_framectrl_bits);
__IO_REG32_BIT(HW_SPDIF_SRR,                    0x80054030,__READ_WRITE ,__hw_spdif_srr_bits);
__IO_REG32_BIT(HW_SPDIF_SRR_SET,                0x80054034,__WRITE      ,__hw_spdif_srr_bits);
__IO_REG32_BIT(HW_SPDIF_SRR_CLR,                0x80054038,__WRITE      ,__hw_spdif_srr_bits);
__IO_REG32_BIT(HW_SPDIF_SRR_TOG,                0x8005403C,__WRITE      ,__hw_spdif_srr_bits);
__IO_REG32_BIT(HW_SPDIF_DEBUG,                  0x80054040,__READ       ,__hw_spdif_debug_bits);
__IO_REG32_BIT(HW_SPDIF_DATA,                   0x80054050,__READ_WRITE ,__hw_spdif_data_bits);
__IO_REG32_BIT(HW_SPDIF_DATA_SET,               0x80054054,__WRITE      ,__hw_spdif_data_bits);
__IO_REG32_BIT(HW_SPDIF_DATA_CLR,               0x80054058,__WRITE      ,__hw_spdif_data_bits);
__IO_REG32_BIT(HW_SPDIF_DATA_TOG,               0x8005405C,__WRITE      ,__hw_spdif_data_bits);
__IO_REG32_BIT(HW_SPDIF_VERSION,                0x80054060,__READ       ,__hw_spdif_version_bits);

/***************************************************************************
 **
 **  SAIF1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SAIF1_CTRL,                   0x80042000,__READ_WRITE ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_SET,               0x80042004,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_CLR,               0x80042008,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_TOG,               0x8004200C,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_STAT,                   0x80042010,__READ       ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF1_STAT_CLR,               0x80042018,__WRITE      ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF1_DATA,                   0x80042020,__READ_WRITE ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_SET,               0x80042024,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_CLR,               0x80042028,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_TOG,               0x8004202C,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_VERSION,                0x80042030,__READ       ,__hw_saif_version_bits);

/***************************************************************************
 **
 **  SAIF2
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SAIF2_CTRL,                   0x80046000,__READ_WRITE ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF2_CTRL_SET,               0x80046004,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF2_CTRL_CLR,               0x80046008,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF2_CTRL_TOG,               0x8004600C,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF2_STAT,                   0x80046010,__READ       ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF2_STAT_CLR,               0x80046018,__WRITE      ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF2_DATA,                   0x80046020,__READ_WRITE ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF2_DATA_SET,               0x80046024,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF2_DATA_CLR,               0x80046028,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF2_DATA_TOG,               0x8004602C,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF2_VERSION,                0x80046030,__READ       ,__hw_saif_version_bits);

/***************************************************************************
 **
 **  Power Control
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_POWER_CTRL,                   0x80044000,__READ_WRITE ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_SET,               0x80044004,__WRITE      ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_CLR,               0x80044008,__WRITE      ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_TOG,               0x8004400C,__WRITE      ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL,                 0x80044010,__READ_WRITE ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_SET,             0x80044014,__WRITE      ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_CLR,             0x80044018,__WRITE      ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_TOG,             0x8004401C,__WRITE      ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_MINPWR,                 0x80044020,__READ_WRITE ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_SET,             0x80044024,__WRITE      ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_CLR,             0x80044028,__WRITE      ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_TOG,             0x8004402C,__WRITE      ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_CHARGE,                 0x80044030,__READ_WRITE ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_SET,             0x80044034,__WRITE      ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_CLR,             0x80044038,__WRITE      ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_TOG,             0x8004403C,__WRITE      ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_VDDDCTRL,               0x80044040,__READ_WRITE ,__hw_power_vdddctrl_bits);
__IO_REG32_BIT(HW_POWER_VDDACTRL,               0x80044050,__READ_WRITE ,__hw_power_vddactrl_bits);
__IO_REG32_BIT(HW_POWER_VDDIOCTRL,              0x80044060,__READ_WRITE ,__hw_power_vddioctrl_bits);
__IO_REG32_BIT(HW_POWER_VDDMEMCTRL,             0x80044070,__READ_WRITE ,__hw_power_vddmemctrl_bits);
__IO_REG32_BIT(HW_POWER_DCDC4P2,                0x80044080,__READ_WRITE ,__hw_power_dcdc4p2_bits);
__IO_REG32_BIT(HW_POWER_MISC,                   0x80044090,__READ_WRITE ,__hw_power_misc_bits);
__IO_REG32_BIT(HW_POWER_DCLIMITS,               0x800440A0,__READ_WRITE ,__hw_power_dclimits_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL,               0x800440B0,__READ_WRITE ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_SET,           0x800440B4,__WRITE      ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_CLR,           0x800440B8,__WRITE      ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_TOG,           0x800440BC,__WRITE      ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_STS,                    0x800440C0,__READ_WRITE ,__hw_power_sts_bits);
__IO_REG32_BIT(HW_POWER_SPEED,                  0x800440D0,__READ_WRITE ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_SET,              0x800440D4,__WRITE      ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_CLR,              0x800440D8,__WRITE      ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_TOG,              0x800440DC,__WRITE      ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_BATTMONITOR,            0x800440E0,__READ_WRITE ,__hw_power_battmonitor_bits);
__IO_REG32_BIT(HW_POWER_RESET,                  0x80044100,__READ_WRITE ,__hw_power_reset_bits);
__IO_REG32_BIT(HW_POWER_RESET_SET,              0x80044104,__WRITE      ,__hw_power_reset_bits);
__IO_REG32_BIT(HW_POWER_RESET_CLR,              0x80044108,__WRITE      ,__hw_power_reset_bits);
__IO_REG32_BIT(HW_POWER_RESET_TOG,              0x8004410C,__WRITE      ,__hw_power_reset_bits);
__IO_REG32_BIT(HW_POWER_DEBUG,                  0x80044110,__READ_WRITE ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_SET,              0x80044114,__WRITE      ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_CLR,              0x80044118,__WRITE      ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_TOG,              0x8004411C,__WRITE      ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_VERSION,                0x80044130,__READ       ,__hw_power_version_bits);

/***************************************************************************
 **
 **  LRADC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_LRADC_CTRL0,                  0x80050000,__READ_WRITE ,__hw_lradc_ctrl0_bits);
__IO_REG32_BIT(HW_LRADC_CTRL0_SET,              0x80050004,__WRITE      ,__hw_lradc_ctrl0_bits);
__IO_REG32_BIT(HW_LRADC_CTRL0_CLR,              0x80050008,__WRITE      ,__hw_lradc_ctrl0_bits);
__IO_REG32_BIT(HW_LRADC_CTRL0_TOG,              0x8005000C,__WRITE      ,__hw_lradc_ctrl0_bits);
__IO_REG32_BIT(HW_LRADC_CTRL1,                  0x80050010,__READ_WRITE ,__hw_lradc_ctrl1_bits);
__IO_REG32_BIT(HW_LRADC_CTRL1_SET,              0x80050014,__WRITE      ,__hw_lradc_ctrl1_bits);
__IO_REG32_BIT(HW_LRADC_CTRL1_CLR,              0x80050018,__WRITE      ,__hw_lradc_ctrl1_bits);
__IO_REG32_BIT(HW_LRADC_CTRL1_TOG,              0x8005001C,__WRITE      ,__hw_lradc_ctrl1_bits);
__IO_REG32_BIT(HW_LRADC_CTRL2,                  0x80050020,__READ_WRITE ,__hw_lradc_ctrl2_bits);
__IO_REG32_BIT(HW_LRADC_CTRL2_SET,              0x80050024,__WRITE      ,__hw_lradc_ctrl2_bits);
__IO_REG32_BIT(HW_LRADC_CTRL2_CLR,              0x80050028,__WRITE      ,__hw_lradc_ctrl2_bits);
__IO_REG32_BIT(HW_LRADC_CTRL2_TOG,              0x8005002C,__WRITE      ,__hw_lradc_ctrl2_bits);
__IO_REG32_BIT(HW_LRADC_CTRL3,                  0x80050030,__READ_WRITE ,__hw_lradc_ctrl3_bits);
__IO_REG32_BIT(HW_LRADC_CTRL3_SET,              0x80050034,__WRITE      ,__hw_lradc_ctrl3_bits);
__IO_REG32_BIT(HW_LRADC_CTRL3_CLR,              0x80050038,__WRITE      ,__hw_lradc_ctrl3_bits);
__IO_REG32_BIT(HW_LRADC_CTRL3_TOG,              0x8005003C,__WRITE      ,__hw_lradc_ctrl3_bits);
__IO_REG32_BIT(HW_LRADC_STATUS,                 0x80050040,__READ       ,__hw_lradc_status_bits);
__IO_REG32_BIT(HW_LRADC_CH0,                    0x80050050,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH0_SET,                0x80050054,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH0_CLR,                0x80050058,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH0_TOG,                0x8005005C,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH1,                    0x80050060,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH1_SET,                0x80050064,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH1_CLR,                0x80050068,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH1_TOG,                0x8005006C,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH2,                    0x80050070,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH2_SET,                0x80050074,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH2_CLR,                0x80050078,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH2_TOG,                0x8005007C,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH3,                    0x80050080,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH3_SET,                0x80050084,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH3_CLR,                0x80050088,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH3_TOG,                0x8005008C,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH4,                    0x80050090,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH4_SET,                0x80050094,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH4_CLR,                0x80050098,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH4_TOG,                0x8005009C,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH5,                    0x800500A0,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH5_SET,                0x800500A4,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH5_CLR,                0x800500A8,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH5_TOG,                0x800500AC,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH6,                    0x800500B0,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH6_SET,                0x800500B4,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH6_CLR,                0x800500B8,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH6_TOG,                0x800500BC,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH7,                    0x800500C0,__READ_WRITE ,__hw_lradc_ch7_bits);
__IO_REG32_BIT(HW_LRADC_CH7_SET,                0x800500C4,__WRITE      ,__hw_lradc_ch7_bits);
__IO_REG32_BIT(HW_LRADC_CH7_CLR,                0x800500C8,__WRITE      ,__hw_lradc_ch7_bits);
__IO_REG32_BIT(HW_LRADC_CH7_TOG,                0x800500CC,__WRITE      ,__hw_lradc_ch7_bits);
__IO_REG32_BIT(HW_LRADC_DELAY0,                 0x800500D0,__READ_WRITE ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY0_SET,             0x800500D4,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY0_CLR,             0x800500D8,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY0_TOG,             0x800500DC,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY1,                 0x800500E0,__READ_WRITE ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY1_SET,             0x800500E4,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY1_CLR,             0x800500E8,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY1_TOG,             0x800500EC,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY2,                 0x800500F0,__READ_WRITE ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY2_SET,             0x800500F4,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY2_CLR,             0x800500F8,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY2_TOG,             0x800500FC,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY3,                 0x80050100,__READ_WRITE ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY3_SET,             0x80050104,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY3_CLR,             0x80050108,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DELAY3_TOG,             0x8005010C,__WRITE      ,__hw_lradc_delayx_bits);
__IO_REG32_BIT(HW_LRADC_DEBUG0,                 0x80050110,__READ       ,__hw_lradc_debug0_bits);
__IO_REG32_BIT(HW_LRADC_DEBUG1,                 0x80050120,__READ_WRITE ,__hw_lradc_debug1_bits);
__IO_REG32_BIT(HW_LRADC_DEBUG1_SET,             0x80050124,__WRITE      ,__hw_lradc_debug1_bits);
__IO_REG32_BIT(HW_LRADC_DEBUG1_CLR,             0x80050128,__WRITE      ,__hw_lradc_debug1_bits);
__IO_REG32_BIT(HW_LRADC_DEBUG1_TOG,             0x8005012C,__WRITE      ,__hw_lradc_debug1_bits);
__IO_REG32_BIT(HW_LRADC_CONVERSION,             0x80050130,__READ_WRITE ,__hw_lradc_conversion_bits);
__IO_REG32_BIT(HW_LRADC_CONVERSION_SET,         0x80050134,__WRITE      ,__hw_lradc_conversion_bits);
__IO_REG32_BIT(HW_LRADC_CONVERSION_CLR,         0x80050138,__WRITE      ,__hw_lradc_conversion_bits);
__IO_REG32_BIT(HW_LRADC_CONVERSION_TOG,         0x8005013C,__WRITE      ,__hw_lradc_conversion_bits);
__IO_REG32_BIT(HW_LRADC_CTRL4,                  0x80050140,__READ_WRITE ,__hw_lradc_ctrl4_bits);
__IO_REG32_BIT(HW_LRADC_CTRL4_SET,              0x80050144,__WRITE      ,__hw_lradc_ctrl4_bits);
__IO_REG32_BIT(HW_LRADC_CTRL4_CLR,              0x80050148,__WRITE      ,__hw_lradc_ctrl4_bits);
__IO_REG32_BIT(HW_LRADC_CTRL4_TOG,              0x8005014C,__WRITE      ,__hw_lradc_ctrl4_bits);
__IO_REG32_BIT(HW_LRADC_VERSION,                0x80050150,__READ       ,__hw_lradc_version_bits);

/***************************************************************************
 **
 **  PINCTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_PINCTRL_CTRL,                 0x80018000,__READ_WRITE ,__hw_pinctrl_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_CTRL_SET,             0x80018004,__WRITE      ,__hw_pinctrl_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_CTRL_CLR,             0x80018008,__WRITE      ,__hw_pinctrl_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_CTRL_TOG,             0x8001800C,__WRITE      ,__hw_pinctrl_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0,              0x80018100,__READ_WRITE ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_SET,          0x80018104,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_CLR,          0x80018108,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_TOG,          0x8001810C,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1,              0x80018110,__READ_WRITE ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_SET,          0x80018114,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_CLR,          0x80018118,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_TOG,          0x8001811C,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2,              0x80018120,__READ_WRITE ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_SET,          0x80018124,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_CLR,          0x80018128,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_TOG,          0x8001812C,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3,              0x80018130,__READ_WRITE ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_SET,          0x80018134,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_CLR,          0x80018138,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_TOG,          0x8001813C,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4,              0x80018140,__READ_WRITE ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_SET,          0x80018144,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_CLR,          0x80018148,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_TOG,          0x8001814C,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5,              0x80018150,__READ_WRITE ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_SET,          0x80018154,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_CLR,          0x80018158,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_TOG,          0x8001815C,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6,              0x80018160,__READ_WRITE ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_SET,          0x80018164,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_CLR,          0x80018168,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_TOG,          0x8001816C,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7,              0x80018170,__READ_WRITE ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_SET,          0x80018174,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_CLR,          0x80018178,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_TOG,          0x8001817C,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0,               0x80018200,__READ_WRITE ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_SET,           0x80018204,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_CLR,           0x80018208,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_TOG,           0x8001820C,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE1,               0x80018210,__READ_WRITE ,__hw_pinctrl_drive1_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE1_SET,           0x80018214,__WRITE      ,__hw_pinctrl_drive1_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE1_CLR,           0x80018218,__WRITE      ,__hw_pinctrl_drive1_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE1_TOG,           0x8001821C,__WRITE      ,__hw_pinctrl_drive1_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2,               0x80018220,__READ_WRITE ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_SET,           0x80018224,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_CLR,           0x80018228,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_TOG,           0x8001822C,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3,               0x80018230,__READ_WRITE ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_SET,           0x80018234,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_CLR,           0x80018238,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_TOG,           0x8001823C,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4,               0x80018240,__READ_WRITE ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_SET,           0x80018244,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_CLR,           0x80018248,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_TOG,           0x8001824C,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5,               0x80018250,__READ_WRITE ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_SET,           0x80018254,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_CLR,           0x80018258,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_TOG,           0x8001825C,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6,               0x80018260,__READ_WRITE ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_SET,           0x80018264,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_CLR,           0x80018268,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_TOG,           0x8001826C,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7,               0x80018270,__READ_WRITE ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_SET,           0x80018274,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_CLR,           0x80018278,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_TOG,           0x8001827C,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8,               0x80018280,__READ_WRITE ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_SET,           0x80018284,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_CLR,           0x80018288,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_TOG,           0x8001828C,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9,               0x80018290,__READ_WRITE ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_SET,           0x80018294,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_CLR,           0x80018298,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_TOG,           0x8001829C,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10,              0x800182A0,__READ_WRITE ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_SET,          0x800182A4,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_CLR,          0x800182A8,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_TOG,          0x800182AC,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11,              0x800182B0,__READ_WRITE ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_SET,          0x800182B4,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_CLR,          0x800182B8,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_TOG,          0x800182BC,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12,              0x800182C0,__READ_WRITE ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_SET,          0x800182C4,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_CLR,          0x800182C8,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_TOG,          0x800182CC,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13,              0x800182D0,__READ_WRITE ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_SET,          0x800182D4,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_CLR,          0x800182D8,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_TOG,          0x800182DC,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14,              0x800182E0,__READ_WRITE ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_SET,          0x800182E4,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_CLR,          0x800182E8,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_TOG,          0x800182EC,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0,                0x80018400,__READ_WRITE ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_SET,            0x80018404,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_CLR,            0x80018408,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_TOG,            0x8001840C,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1,                0x80018410,__READ_WRITE ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_SET,            0x80018414,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_CLR,            0x80018418,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_TOG,            0x8001841C,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2,                0x80018420,__READ_WRITE ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_SET,            0x80018424,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_CLR,            0x80018428,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_TOG,            0x8001842C,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3,                0x80018430,__READ_WRITE ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_SET,            0x80018434,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_CLR,            0x80018438,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_TOG,            0x8001843C,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0,                0x80018500,__READ_WRITE ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_SET,            0x80018504,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_CLR,            0x80018508,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_TOG,            0x8001850C,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1,                0x80018510,__READ_WRITE ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_SET,            0x80018514,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_CLR,            0x80018518,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_TOG,            0x8001851C,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2,                0x80018520,__READ_WRITE ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_SET,            0x80018524,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_CLR,            0x80018528,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_TOG,            0x8001852C,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN0,                 0x80018600,__READ       ,__hw_pinctrl_din0_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN1,                 0x80018610,__READ       ,__hw_pinctrl_din1_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN2,                 0x80018620,__READ       ,__hw_pinctrl_din2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0,                 0x80018700,__READ_WRITE ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_SET,             0x80018704,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_CLR,             0x80018708,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_TOG,             0x8001870C,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1,                 0x80018710,__READ_WRITE ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_SET,             0x80018714,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_CLR,             0x80018718,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_TOG,             0x8001871C,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2,                 0x80018720,__READ_WRITE ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_SET,             0x80018724,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_CLR,             0x80018728,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_TOG,             0x8001872C,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0,             0x80018800,__READ_WRITE ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_SET,         0x80018804,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_CLR,         0x80018808,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_TOG,         0x8001880C,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1,             0x80018810,__READ_WRITE ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_SET,         0x80018814,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_CLR,         0x80018818,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_TOG,         0x8001881C,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2,             0x80018820,__READ_WRITE ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_SET,         0x80018824,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_CLR,         0x80018828,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_TOG,         0x8001882C,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0,               0x80018900,__READ_WRITE ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_SET,           0x80018904,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_CLR,           0x80018908,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_TOG,           0x8001890C,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1,               0x80018910,__READ_WRITE ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_SET,           0x80018914,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_CLR,           0x80018918,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_TOG,           0x8001891C,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2,               0x80018920,__READ_WRITE ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_SET,           0x80018924,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_CLR,           0x80018928,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_TOG,           0x8001892C,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0,            0x80018A00,__READ_WRITE ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_SET,        0x80018A04,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_CLR,        0x80018A08,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_TOG,        0x80018A0C,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1,            0x80018A10,__READ_WRITE ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_SET,        0x80018A14,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_CLR,        0x80018A18,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_TOG,        0x80018A1C,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2,            0x80018A20,__READ_WRITE ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_SET,        0x80018A24,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_CLR,        0x80018A28,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_TOG,        0x80018A2C,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0,              0x80018B00,__READ_WRITE ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_SET,          0x80018B04,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_CLR,          0x80018B08,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_TOG,          0x80018B0C,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1,              0x80018B10,__READ_WRITE ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_SET,          0x80018B14,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_CLR,          0x80018B18,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_TOG,          0x80018B1C,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2,              0x80018B20,__READ_WRITE ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_SET,          0x80018B24,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_CLR,          0x80018B28,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_TOG,          0x80018B2C,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0,             0x80018C00,__READ       ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0_CLR,         0x80018C08,__WRITE      ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1,             0x80018C10,__READ       ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1_CLR,         0x80018C18,__WRITE      ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2,             0x80018C20,__READ       ,__hw_pinctrl_irqstat2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2_CLR,         0x80018C28,__WRITE      ,__hw_pinctrl_irqstat2_bits);

/* Assembler specific declarations  ****************************************/

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
 **   MCIMX23 APBH DMA channels
 **
 ***************************************************************************/
#define APBH_DMA_SSP1          1
#define APBH_DMA_SSP2          2
#define APBH_DMA_NAND0         4
#define APBH_DMA_NAND1         5
#define APBH_DMA_NAND2         6
#define APBH_DMA_NAND3         7

/***************************************************************************
 **
 **   MCIMX23 APXH DMA channels
 **
 ***************************************************************************/
#define APBX_DMA_AUDIOIN       0
#define APBX_DMA_AUDIOOUT      1
#define APBX_DMA_SPDIF_TX      2
#define APBX_DMA_I2C           3
#define APBX_DMA_SAIF1         4
#define APBX_DMA_UARTAPP1_RX   6
#define APBX_DMA_UARTAPP1_TX   7
#define APBX_DMA_UARTAPP2_RX   8
#define APBX_DMA_UARTAPP2_TX   9
#define APBX_DMA_SAIF2         10

/***************************************************************************
 **
 **   MCIMX23 interrupt sources
 **
 ***************************************************************************/
#define INT_UARTDBG            0              /* Non DMA on the debug UART */
#define INT_COMMS_RX_TX        1              /* JTAG debug communications port*/
#define INT_SSP2_ERROR         2              /* SSP2 device-level error and status*/
#define INT_VDD5V              3              /* IRQ on 5V connect or disconnect. Shared with DCDC status,
                                                 Linear Regulator status, PSWITCH, and Host 4.2V*/
#define INT_HEADPHONE_SHORT    4              /* HEADPHONE_SHORT*/
#define INT_AUDIOOUT_DMA       5              /* DAC_DMA*/
#define INT_AUDIOOUT_ERROR     6              /* DAC FIFO buffer underflow*/
#define INT_AUDIOIN_DMA        7              /* ADC DMA channel */
#define INT_AUDIOIN_ERROR      8              /* ADC FIFO buffer overflow*/
#define INT_SPDIF_SAIF2_DMA    9              /* SPDIF DMA channel, SAIF2 DMA channel*/
#define INT_SPDIF_SAIF1_SAIF2  10             /* SPDIF, SAIF1, SAIF2 FIFO underflow/overflow*/
#define INT_USB_CTRL           11             /* USB controller */
#define INT_USB_WAKEUP         12             /* USB wakeup. Also ARC core to remain suspended*/
#define INT_GPMI_DMA           13             /* From DMA channel for GPMI*/
#define INT_SSP1_DMA           14             /* From DMA channel for SSP1*/
#define INT_SSP_ERROR          15             /* SSP1 device-level error and status*/
#define INT_GPIO0              16             /* GPIO bank 0 interrupt*/
#define INT_GPIO1              17             /* GPIO bank 1 interrupt*/
#define INT_GPIO2              18             /* GPIO bank 2 interrupt*/
#define INT_SAIF1_DMA          19             /* SAIF1 DMA channel*/
#define INT_SSP2_DMA           20             /* From DMA channel for SSP2*/
#define INT_ECC8               21             /* ECC8 completion interrupt*/
#define INT_RTC_ALARM          22             /* RTC alarm event */
#define INT_UARTAPP1_TX_DMA    23             /* Application UART1 transmitter DMA*/
#define INT_UARTAPP1_INTERNAL  24             /* Application UART1 internal error*/
#define INT_UARTAPP1_RX_DMA    25             /* Application UART1 receiver DMA*/
#define INT_I2C_DMA            26             /* From DMA channel for I2C*/
#define INT_I2C_ERROR          27             /* From I2C device detected errors and line conditions*/
#define INT_TIMER0             28             /* TIMROT Timer0, recommend to set as FIQ*/
#define INT_TIMER1             29             /* TIMROT Timer1, recommend to set as FIQ*/
#define INT_TIMER2             30             /* TIMROT Timer2, recommend to set as FIQ*/
#define INT_TIMER3             31             /* TIMROT Timer3, recommend to set as FIQ*/
#define INT_BATT_BRNOUT        32             /* Power module battery brownout detect, recommend to set as FIQ*/
#define INT_VDDD_BRNOUT        33             /* Power module VDDD brownout detect, recommend to set as FIQ*/
#define INT_VDDIO_BRNOUT       34             /* Power module VDDIO brownout detect, recommend to set as FIQ*/
#define INT_VDD18_BRNOUT       35             /* Power module VDD18 brownout detect, recommend to set as FIQ*/
#define INT_TOUCH_DETECT       36             /* Touch detection*/
#define INT_LRADC_CH0          37             /* Low Resolution ADC Channel 0 complete*/
#define INT_LRADC_CH1          38             /* Low Resolution ADC Channel 1 complete*/
#define INT_LRADC_CH2          39             /* Low Resolution ADC Channel 2 complete*/
#define INT_LRADC_CH3          40             /* Low Resolution ADC Channel 3 complete*/
#define INT_LRADC_CH4          41             /* Low Resolution ADC Channel 4 complete*/
#define INT_LRADC_CH5          42             /* Low Resolution ADC Channel 5 complete*/
#define INT_LRADC_CH6          43             /* Low Resolution ADC Channel 6 complete*/
#define INT_LRADC_CH7          44             /* Low Resolution ADC Channel 7 complete*/
#define INT_LCDIF_DMA          45             /* From DMA channel for LCDIF*/
#define INT_LCDIF_ERROR        46             /* LCDIF error*/
#define INT_DIGCTL_DEBUG_TRAP  47             /* AHB arbiter debug trap*/
#define INT_RTC_1MSEC          48             /* RTC 1 ms tick interrupt*/
#define INT_GPMI               51             /* From GPMI internal error and status IRQ*/
#define INT_DCP_VMI            53             /* DCP Channel 0 virtual memory page copy*/
#define INT_DCP                54             /* DCP*/
#define INT_BCH                56             /* BCH consolidated Interrupt*/
#define INT_PXP                57             /* Pixel Pipeline consolidated Interrupt  */
#define INT_UARTAPP2_TX_DMA    58             /* Application UART2 transmitter DMA*/
#define INT_UARTAPP2_INTERNAL  59             /* Application UART2 internal error*/
#define INT_UARTAPP2_RX_DMA    60             /* Application UART2 receiver DMA*/
#define INT_VDAC_DETECT        61             /* Video dac, jack presence auto-detect*/
#define INT_VDD5V_DROOP        64             /* 5V Droop, recommend to be set as FIQ*/
#define INT_DCDC4P2_BO         65             /* 4.2V regulated supply brown-out, recommend to be set as FIQ*/

#endif    /* __MCIMX23_H */
